import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gguf import GGUFWriter,GGUFReader
from common import sha,validate_recipe
from structural_export import export
from allocate import optimize


class ExportTests(unittest.TestCase):
    def fixture(self,root):
        p=root/'base.gguf';w=GGUFWriter(str(p),'escha');w.add_string('test.unicode','Escha δοκιμή')
        w.add_array('test.tokens',['a','b','c']);w.add_uint32('escha.format_version',1)
        w.add_tensor('blk.0.ffn_gate.escha_code',np.arange(128,dtype=np.int16).reshape(2,2,32))
        w.add_tensor('blk.0.ffn_up.escha_code',np.arange(192,dtype=np.int16).reshape(2,2,48))
        w.add_tensor('blk.0.ffn_gate.escha_rin',np.ones(32,np.float16))
        w.add_tensor('token_embd.lowgpu_codes',np.arange(160,dtype=np.int8).reshape(4,40))
        w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close();return p

    def test_noop_exact_and_variable_widths_preserve_other_payloads(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=self.fixture(root);limit=source.stat().st_size
            export(source,root/'noop.gguf',{},sha(source),limit)
            self.assertEqual(sha(source),sha(root/'noop.gguf'))
            a=root/'a.npy';b=root/'b.npy';np.save(a,np.full((2,2,48),123,np.int16));np.save(b,np.full((2,2,32),456,np.int16))
            result=export(source,root/'swap.gguf',{'blk.0.ffn_gate.escha_code':str(a),'blk.0.ffn_up.escha_code':str(b)},sha(source),limit)
            self.assertEqual(result['bytes'],limit)
            new={x.name:x for x in GGUFReader(str(root/'swap.gguf')).tensors}
            old={x.name:x for x in GGUFReader(str(source)).tensors}
            self.assertEqual(new['blk.0.ffn_gate.escha_code'].data.shape,(2,2,48))
            self.assertEqual(new['blk.0.ffn_up.escha_code'].data.shape,(2,2,32))
            np.testing.assert_array_equal(new['token_embd.lowgpu_codes'].data,old['token_embd.lowgpu_codes'].data)
            with self.assertRaises(ValueError):export(source,root/'over.gguf',{'blk.0.ffn_gate.escha_code':str(a)},sha(source),limit)
            with self.assertRaises(ValueError):export(source,root/'swap.gguf',{},sha(source),limit)

    def test_protected_vocabulary_and_wrong_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=self.fixture(root);a=root/'v.npy';np.save(a,np.zeros((4,40),np.int8))
            with self.assertRaises(ValueError):export(source,root/'bad.gguf',{'token_embd.lowgpu_codes':str(a)},sha(source),source.stat().st_size)


class AllocationTests(unittest.TestCase):
    def test_measurement_targets_are_individual_and_head_is_excluded(self):
        from measure import individual_targets
        from types import SimpleNamespace
        class Target:
            def __init__(self,key,linears,qbits):self.key=key;self.linears=linears;self.qbits_key=qbits
        gate=SimpleNamespace(key='gate',qbits_key='bits');up=SimpleNamespace(key='up',qbits_key='bits');head=SimpleNamespace(key='lm_head',qbits_key='head_bits')
        original=lambda mod:[Target('gate_up',[gate,up],'bits'),Target('lm_head',[head],'head_bits')]
        collect=individual_targets(SimpleNamespace(Target=Target),original,{'gate','up'})
        result=collect(None);self.assertEqual([r.key for r in result],['gate','up']);self.assertTrue(all(len(r.linears)==1 for r in result))

    def test_upstream_changes_map_under_exact_budget_and_pins_unsupported_widths(self):
        rows=[dict(name=f'blk.{i}.x',source_key=f'model.language_model.layers.{i}.mlp.gate_proj',
                   ic=128,oc=128,k=k,allowed_k=ks) for i,(k,ks) in enumerate([(2,[2,3]),(3,[2,3]),(2,[2])])]
        policy=dict(projections=rows,body_bits=sum(r['ic']*r['oc']*r['k'] for r in rows))
        m=dict(model='synthetic',results=[]);ref=dict(results=[])
        for i,r in enumerate(rows):
            sensitivity=[100.,1.,1.][i]
            m['results'].append(dict(key=r['source_key'],qbits_key='bits',numel=16384,shaped=True,
                levels=[dict(kld=sensitivity*x*x,rfn_actual=x) for x in (.2,.1)]))
            ref['results'].append(dict(key=r['source_key'],K=r['k'],numel=16384,rfn=.2 if r['k']==2 else .1))
        with tempfile.TemporaryDirectory() as t:
            recipe,audit=optimize(policy,m,ref,t)
            self.assertEqual([recipe['tensors'][r['source_key']] for r in rows],[3,2,2])
            self.assertEqual(audit['body_bits'],policy['body_bits'])
            self.assertLess(audit['predicted_flexible_candidate'],audit['predicted_flexible_baseline'])
            bad=copy.deepcopy(recipe);bad['tensors'][rows[2]['source_key']]=3
            with self.assertRaises(ValueError):validate_recipe(policy,bad)
            m['results'][0]['shaped']=False
            with self.assertRaises(ValueError):optimize(policy,m,ref,t)


class GradientTests(unittest.TestCase):
    def test_cpu_backed_saved_tensors_preserve_forward_and_input_gradient(self):
        import torch
        from hf_stream import Streaming
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layers=torch.nn.ModuleList([torch.nn.Linear(8,8),torch.nn.Linear(8,8)])
                self.head=torch.nn.Linear(8,4);self.embedding=torch.nn.Identity()
            def get_input_embeddings(self):return self.embedding
            def get_output_embeddings(self):return self.head
            def forward(self,x):
                for layer in self.layers:x=x+torch.tanh(layer(x))
                return self.head(x)
        torch.manual_seed(17);model=Toy().double().eval()
        storage=torch.randn(81,dtype=torch.float64)
        model.layers[0].weight.data=storage[17:].view(8,8)
        for p in model.parameters():p.requires_grad_(False)
        x=torch.randn(3,8,dtype=torch.float64,requires_grad=True);baseline=model(x);baseline.square().sum().backward();grad=x.grad.clone()
        stream=Streaming(model,'cpu');y=x.detach().clone().requires_grad_(True)
        try:
            with stream.saved_tensors():result=model(y);result.square().sum().backward()
            torch.testing.assert_close(result,baseline,rtol=0,atol=0);torch.testing.assert_close(y.grad,grad,rtol=0,atol=0)
        finally:stream.close()


class WorkflowTests(unittest.TestCase):
    def test_stage_resumes_valid_outputs_and_recovers_corruption(self):
        import sys
        from run import Stages
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'plan.json').write_text(json.dumps({'factor_stores':[]}))
            source=root/'input';source.write_text('original');output=root/'output'
            cmd=[sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text())',str(source),str(output)]
            stages=Stages(root);stages.run('copy',cmd,[output]);stages.run('copy',cmd,[output])
            self.assertEqual(len(list((root/'stages/copy').glob('attempt-*.json'))),1)
            output.write_text('corrupt');stages.run('copy',cmd,[output]);self.assertEqual(output.read_text(),'original')
            self.assertEqual(len(list((root/'stages/copy').glob('attempt-*.json'))),2)
            source.write_text('different input')
            with self.assertRaises(ValueError):stages.run('copy',cmd,[output])

    def test_original_sensitivity_cli_accepts_the_planned_flags(self):
        import ast,argparse
        from common import TREE
        block=ast.parse((TREE/'sc_measure.py').read_text()).body[-1]
        namespace={'argparse':argparse};exec(compile(ast.Module(body=block.body[:-1],type_ignores=[]),'original_cli','exec'),namespace)
        args=namespace['parser'].parse_args(['-m','/source','--streaming','--max-sys','24','-tr','/trace',
                '-r','16','-l','1024','--shaped','--h_rows','64','--rfn_ref','/rfn','--rfn_scale','1.0,0.5','-o','/measurement'])
        self.assertTrue(args.shaped);self.assertEqual(args.load_mode,'streaming');self.assertEqual(args.h_rows,64)

    def test_v2_cannot_start_before_v1_measurements_complete(self):
        from run import v1_gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=dict(full_model_exported=True,layer_runtime_passed=True,projection_runtime_passed=True)
            (root/'compatibility-gates.json').write_text(json.dumps(g))
            with self.assertRaises(ValueError):v1_gate({'v1':str(root)})
            g['qualification_complete']=True;(root/'compatibility-gates.json').write_text(json.dumps(g))
            (root/'full').mkdir();(root/'full/conversion.complete.json').write_text('{}')
            (root/'full-escha.gguf.receipt.json').write_text('{}')
            self.assertTrue(v1_gate({'v1':str(root)})['qualification_complete'])

    def test_trace_unicode_and_public_prompt_exclusion(self):
        from trace_data import prepare
        from safetensors.numpy import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'ref';ref.mkdir()
            docs=[dict(prompt='held out',category='code'),dict(prompt='line\u2028inside string',category='code'),dict(prompt='third',category='tools')]
            (root/'calib.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in docs)+'\n')
            (root/'public.jsonl').write_text(json.dumps(docs[0])+'\n')
            (ref/'kld.json').write_text(json.dumps({'docs':[{'tokens':3} for _ in docs]}))
            np.arange(9,dtype='<i4').tofile(ref/'tokens.i32')
            out=root/'trace.safetensors';prepare(ref,root/'calib.jsonl',root/'public.jsonl',out,rows=2,cols=3)
            receipt=json.loads(Path(str(out)+'.receipt.json').read_text())
            self.assertEqual({x['original_doc'] for x in receipt['selection']},{1,2})
            self.assertEqual(load_file(str(out))['input_ids'].shape,(2,3))


if __name__=='__main__':unittest.main()

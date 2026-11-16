"""Projection/layer pilot using unmodified upstream calibration finalization and LDLQ."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from adapter import atomic_json, convert_tiles, inventory, require, sha, validate_scales


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',required=True)
    p.add_argument('--inputs',required=True,help='Retained body-block0 directory with sources.json')
    p.add_argument('--hessians',required=True)
    p.add_argument('--native-codec',required=True)
    p.add_argument('--lease-state',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--projection',default='blk.0.attn_gate')
    p.add_argument('--layer',type=int)
    a=p.parse_args()
    lease=json.loads(Path(a.lease_state).read_text())
    require(lease['state']=='running' and lease['child_pid']==os.getpid(), 'Must execute as the 4090 guardian child')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==lease['gpu'], 'Lease device mismatch')
    import torch
    from gguf import GGUFReader
    from safetensors import safe_open
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    reader=GGUFReader(a.template);tensors={t.name:t for t in reader.tensors}
    rows=inventory(reader)
    selected=[r for r in rows if r['layer']==a.layer] if a.layer is not None else [r for r in rows if r['name']==a.projection]
    require(bool(selected),'No selected projections')
    source={r['name']:r for r in json.loads((Path(a.inputs)/'sources.json').read_text())['projections']}
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    lib=ctypes.CDLL(a.native_codec)
    lib.native_lut.argtypes=[ctypes.c_void_p]
    lib.native_body.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
    lut=torch.empty(65536,device='cuda',dtype=torch.float16)
    require(lib.native_lut(lut.data_ptr())==0,'Native LUT error')
    require(np.array_equal(lut.cpu().numpy().view('u2'),tensors['escha_lut'].data.reshape(-1).view('u2')), 'Native LUT mismatch')
    results=[]
    for row in selected:
        name=row['name'];record=source[name]
        require(record['source_key']==row['source_key']+'.weight','BF16 source mapping differs')
        weightfile=Path(a.inputs)/record['file'];hfile=Path(a.hessians)/record['hessian_file']
        require(sha(weightfile)==record['sha256'],'BF16 source digest differs')
        identity=dict(projection=row,weight_sha256=record['sha256'],hessian_sha256=sha(hfile),
                      template_sha256=sha(a.template),pilot_sha256=sha(__file__),adapter_sha256=sha(Path(__file__).with_name('adapter.py')),
                      quantizer_sha256=sha(Path(__file__).resolve().parents[2]/'exllamav3/modules/quant/exl3_lib/quantize.py'),
                      sigma_reg=.025,seed=20261115,scaled_hessian=True,codebook='mcg')
        receipt=output/(name+'.json');artifact=output/(name+'.npz')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            require(old['identity']==identity and sha(artifact)==old['artifact_sha256'], 'Resume identity/artifact differs')
            results.append(old);continue
        require(not artifact.exists(),'Orphan artifact: preserve it and inspect before resuming')
        start=time.monotonic()
        with safe_open(hfile,framework='np') as f:
            meta=f.metadata();upper=f.get_tensor('hessian_upper')
        ic,oc=row['ic'],row['oc']
        require(int(meta['k'])==ic and row['source_key'] in meta['linears'].split(','), 'Hessian source/basis differs')
        require(meta['layout']=='upper triangle, row-major, packed' and int(meta['rows'])>0, 'Hessian layout/count differs')
        dense=np.zeros((ic,ic),np.float32);cursor=0
        for i in range(ic):
            dense[i,i:]=upper[cursor:cursor+ic-i];cursor+=ic-i
        require(cursor==len(upper) and np.isfinite(dense).all(),'Malformed Hessian')
        dense+=np.triu(dense,1).T
        # Upstream finalize_capture_H divides the accumulated moment by this sample count,
        # rotates it, regularizes it and computes the block LDL factorization.
        hd=dict(H=torch.from_numpy(dense).cuda(),count=int(meta['rows']),finalized=False,
                device=torch.device('cuda:0'),first_key=row['source_key'])
        weights=np.load(weightfile,allow_pickle=False)
        require(weights.dtype==np.uint16 and weights.shape==(oc,ic),'Expected BF16 bit-pattern [out,in] source')
        w=torch.from_numpy(weights.copy()).view(torch.bfloat16).float().T.contiguous().cuda()
        require(torch.isfinite(w).all().item(),'Nonfinite BF16 source')
        qa=dict(K=row['k'],devices=['cuda:0'],mcg=True,mul1=False,seed=20261115,sigma_reg=.025,
                scaled_hessian=True,buf_size_k=128,apply_out_scales=True)
        _,proxy,encoded=quantize_exl3(w,hd,qa,False)
        require(not qa['q_fallback'],'Uncalibrated fallback is forbidden')
        require(np.isfinite(proxy) and proxy>=0,'Invalid upstream weighted reconstruction error')
        code=convert_tiles(encoded['trellis'].cpu().numpy(),row['k'])
        ri,ro=validate_scales(encoded['suh'].cpu().numpy(),encoded['svh'].cpu().numpy(),ic,oc)
        require(code.shape==tensors[name+'.escha_code'].data.shape,'Export geometry mismatch')
        # Verify actual retained CUDA decoder against upstream output on tiles across the projection.
        ids=np.linspace(0,code.reshape(-1,16*row['k']).shape[0]-1,128,dtype=int)
        native_code=torch.from_numpy(code.reshape(-1,16*row['k'])[ids].copy()).cuda()
        dep=torch.from_numpy(tensors[f"escha_dep_k{row['k']}"].data.copy()).cuda()
        decoded=torch.empty((len(ids),256),device='cuda',dtype=torch.float16)
        require(lib.native_body(native_code.data_ptr(),dep.data_ptr(),decoded.data_ptr(),row['k'],len(ids))==0,'Native decode error')
        from adapter import escha_windows,mcg_lut
        expected=mcg_lut()[escha_windows(native_code.cpu().numpy(),row['k'])]
        require(np.array_equal(expected.view('u2'),decoded.cpu().numpy().view('u2')),'CUDA reconstruction differs')
        tmp=artifact.with_suffix('.partial.npz')
        np.savez(tmp,code=code,rin=ri,rout=ro)
        os.replace(tmp,artifact)
        result=dict(identity=identity,artifact_sha256=sha(artifact),proxy_error=float(proxy),seconds=time.monotonic()-start,
                    native_tiles=len(ids),native_decoder_bit_exact=True,projection_quantized=True,
                    runtime_compatible=None,corrections='Template correction/bias tensors preserved; upstream adds no corrections')
        atomic_json(receipt,result);results.append(result)
        print('PROJECTION_COMPLETE',name,json.dumps(result),flush=True)
        del w,hd,encoded,weights,dense
        torch.cuda.empty_cache()
    atomic_json(output/'summary.json',dict(results=results,complete=True,layer=a.layer,
                limitations='Native codec checked; whole-runtime projection/layer inference and KLD are separate gates'))


if __name__=='__main__':main()

"""Compare resident vs offloaded forward, gradients and upstream YaQA factors."""
import argparse
import copy
import importlib.util
import sys
import torch
from common import TREE,atomic_json,require,sha
from hf_stream import Streaming


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([torch.nn.Linear(128,128),torch.nn.Linear(128,128)])
        self.head=torch.nn.Linear(128,32);self.embedding=torch.nn.Identity()
    def get_input_embeddings(self):return self.embedding
    def get_output_embeddings(self):return self.head
    def forward(self,x):
        for layer in self.layers:x=x+torch.tanh(layer(x))
        return self.head(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    require(torch.cuda.is_available(),'GPU parity must actually run on CUDA')
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(37)
    source=TREE/'util/yaqa_hessians.py';spec=importlib.util.spec_from_file_location('original_yaqa_parity',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=Toy().eval()
    for param in base.parameters():param.requires_grad_(False)
    resident=copy.deepcopy(base).cuda();stream=Streaming(base)
    x=torch.randn(32,128,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_(True)
    collectors=[];handles=[]
    for model in (resident,base):
        layer=model.layers[0];c=module.Collector('test','test',layer,torch.device('cuda:0'),False,True,False);c.begin_pass('out')
        def hook(m,inputs,out,c=c):
            c.forward('out',inputs[0],None);out.register_hook(lambda d:c.backward('out',d))
        handles.append(layer.register_forward_hook(hook));collectors.append(c)
    try:
        expected=resident(x);expected.square().sum().backward()
        with stream.saved_tensors():actual=base(y);actual.square().sum().backward()
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(y.grad,x.grad,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(collectors[0].hout,collectors[1].hout,rtol=1e-5,atol=1e-5)
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        class MiniGDN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg=Qwen3_5TextConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=1,head_dim=64,
                    linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=128,linear_value_head_dim=128,
                    layer_types=['linear_attention'])
                self.layers=torch.nn.ModuleList([Qwen3_5GatedDeltaNet(cfg,0)])
                self.identity=torch.nn.Identity()
                with torch.no_grad():
                    for name,p in self.named_parameters():
                        p.copy_(torch.randn_like(p)*.02)
                        if name.endswith('A_log'):p.fill_(-.1)
                        if name.endswith('norm.weight'):p.fill_(1)
                for p in self.parameters():p.requires_grad_(False)
            def get_input_embeddings(self):return self.identity
            def get_output_embeddings(self):return self.identity
            def forward(self,x):return self.layers[0](x)
        gdn=MiniGDN().to(dtype=torch.bfloat16).eval();reference=copy.deepcopy(gdn).cuda();offload=Streaming(gdn)
        gx=(torch.randn(1,32,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_(True)
        gy=gx.detach().clone().requires_grad_(True)
        try:
            gr=reference(gx);gr.float().square().sum().backward()
            with offload.saved_tensors():go=gdn(gy);go.float().square().sum().backward()
            forward_rel=float((go.float()-gr.float()).norm()/gr.float().norm().clamp_min(1e-12))
            gradient_rel=float((gy.grad.float()-gx.grad.float()).norm()/gx.grad.float().norm().clamp_min(1e-12))
            require(forward_rel<1e-3 and gradient_rel<1e-3,'Actual Qwen GDN offloaded gradient differs')
        finally:offload.close()
        atomic_json(a.out,dict(passed=True,device=torch.cuda.get_device_name(),forward_max_delta=float((actual-expected).abs().max()),
            gradient_max_delta=float((x.grad-y.grad).abs().max()),upstream_Collector_sha256=sha(source),
            Qwen_GDN_forward_relative_delta=forward_rel,Qwen_GDN_gradient_relative_delta=gradient_rel,
            scope='CUDA MLP/full-Hout and actual tiny Qwen GDN BF16 forward/gradient parity; full checkpoint also gets a real-model pilot'))
    finally:
        for h in handles:h.remove()
        stream.close()


if __name__=='__main__':main()

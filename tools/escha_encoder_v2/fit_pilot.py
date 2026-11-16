"""Exercise actual MCG two-sided LDLQ and native reconstruction before full factors."""
import argparse
import ctypes
import time
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors import safe_open
from common import TREE,MCG,atomic_json,require,sha,convert_tiles,validate_scales


def square(upper,n):
    require(upper.size==n*(n+1)//2,'Invalid packed covariance geometry')
    result=np.zeros((n,n),np.float32);offset=0
    for i in range(n):result[i,i:]=upper[offset:offset+n-i];offset+=n-i
    result+=np.triu(result,1).T;return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    import json
    plan=json.loads(Path(a.plan).read_text());root=Path(plan['work']);assets=Path(plan['assets']);source=Path(plan['source'])
    sys.path.insert(0,str(TREE));from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    key='model.language_model.layers.0.linear_attn.out_proj';ic,oc=6144,5120
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(source/index[key+'.weight'],framework='pt',device='cpu') as f:w=f.get_tensor(key+'.weight')
    require(w.dtype==torch.bfloat16 and tuple(w.shape)==(oc,ic),'Pilot source differs')
    hfile=Path(plan['v1']).parent/'hessians/model.language_model.layers.0.linear_attn.out_proj.input.safetensors'
    with safe_open(hfile,framework='np') as f:
        metadata=f.metadata();hin=square(f.get_tensor('hessian_upper'),ic)
    outpath=root/'yaqa-pilot'/(key+'.safetensors')
    with safe_open(outpath,framework='np') as f:hout=square(f.get_tensor('hout'),oc)
    torch.backends.cuda.matmul.allow_tf32=False
    hd=dict(H=torch.from_numpy(hin).cuda(),count=int(metadata['rows']),finalized=False,
            device=torch.device('cuda:0'),first_key=key)
    out_factor=torch.from_numpy(hout).cuda()
    qa=dict(K=3,devices=['cuda:0'],mcg=True,mul1=False,seed=plan['seed'],sigma_reg=.025,
            scaled_hessian=True,buf_size_k=128,apply_out_scales=None,H_out=out_factor,
            out_hessian_diag=out_factor.diagonal().clone())
    start=time.monotonic()
    with torch.inference_mode():_,proxy,encoded=quantize_exl3(w.float().T.contiguous().cuda(),hd,qa,False)
    require(not qa['q_fallback'] and np.isfinite(proxy) and proxy>=0,'Two-sided fit numerical failure')
    require(qa.get('out_scales_hessian') is True,'YaQA output scaling did not activate')
    code=convert_tiles(encoded['trellis'].cpu().numpy(),3)
    validate_scales(encoded['suh'].cpu().numpy(),encoded['svh'].cpu().numpy(),ic,oc)
    require(tuple(code.shape)==(ic//16,oc//16,48),'MCG K3 geometry differs')
    from adapter import dependency,escha_windows,mcg_lut
    ids=np.linspace(0,code.reshape(-1,48).shape[0]-1,128,dtype=int)
    sample=code.reshape(-1,48)[ids].copy();gpu=torch.from_numpy(sample).cuda()
    dep=torch.from_numpy(dependency(3)).cuda();decoded=torch.empty((len(ids),256),device='cuda',dtype=torch.float16)
    library=ctypes.CDLL(str(assets/'tools/native_codec_dual.so'));library.native_body.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
    require(library.native_body(gpu.data_ptr(),dep.data_ptr(),decoded.data_ptr(),3,len(ids))==0,'Native pilot decoder error')
    expected=mcg_lut()[escha_windows(sample,3)]
    require(np.array_equal(decoded.cpu().numpy().view('u2'),expected.view('u2')),'Native two-sided MCG reconstruction differs')
    atomic_json(root/'two-sided-pilot.json',dict(passed=True,key=key,K=3,seconds=time.monotonic()-start,
         proxy_error=float(proxy),native_tiles=len(ids),native_bit_exact=True,Hout_sha256=sha(outpath),
         scope='Real two-sided LDLQ+MCG/native-decode capability; pilot factors are not used for final fitting'))


if __name__=='__main__':main()

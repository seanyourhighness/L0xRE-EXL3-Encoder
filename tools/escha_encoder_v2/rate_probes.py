"""Actual YaQA MCG K2/K3 rate diagnostics on representative original projections."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors import safe_open
from common import TREE,read,require,sha,atomic_json,convert_tiles,validate_scales
from fit_pilot import square


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=read(a.plan)
    root=Path(plan['work']);source=Path(plan['source']);assets=Path(plan['assets'])
    sys.path.insert(0,str(TREE));from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from exllamav3.ext import exllamav3_ext as ext
    from adapter import dependency,escha_windows,mcg_lut
    index=read(source/'model.safetensors.index.json')['weight_map'];rows={r['name']:r for r in plan['policy']['projections']}
    choices=[('blk.0.ssm_out','linear_attn.out_proj'),('blk.0.ffn_gate','post_attention'),('blk.0.ffn_up','post_attention')]
    out=root/'rate-probes';out.mkdir(exist_ok=True);results=[];torch.backends.cuda.matmul.allow_tf32=False
    lib=ctypes.CDLL(str(assets/'tools/native_codec_dual.so'));lib.native_body.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
    for name,hname in choices:
        row=rows[name];key=row['source_key'];ic,oc=row['ic'],row['oc']
        with safe_open(source/index[key+'.weight'],framework='pt',device='cpu') as f:original=f.get_tensor(key+'.weight').T.contiguous().cuda().float()
        hf=Path(plan['v1']).parent/'hessians'/('model.language_model.layers.0.'+hname+'.input.safetensors')
        with safe_open(hf,framework='np') as f:meta=f.metadata();hin=square(f.get_tensor('hessian_upper'),ic)
        factor=root/'yaqa-pilot'/(key+'.safetensors')
        with safe_open(factor,framework='np') as f:hout=square(f.get_tensor('hout'),oc)
        for k in (2,3):
            path=out/(name+f'.K{k}.json')
            identity=dict(source_revision=plan['source_revision'],Hout_sha256=sha(factor),Hin_sha256=sha(hf),K=k,key=key,seed=plan['seed'],sigma_reg=.025)
            if path.exists():
                old=read(path);require(old['identity']==identity,'Rate-probe identity changed');results.append(old);continue
            start=time.monotonic();hd=dict(H=torch.from_numpy(hin.copy()).cuda(),count=int(meta['rows']),finalized=False,
                device=torch.device('cuda:0'),first_key=key)
            ho=torch.from_numpy(hout.copy()).cuda();qa=dict(K=k,devices=['cuda:0'],mcg=True,mul1=False,seed=plan['seed'],sigma_reg=.025,
                scaled_hessian=True,buf_size_k=128,apply_out_scales=None,H_out=ho,out_hessian_diag=ho.diagonal().clone())
            with torch.inference_mode():_,proxy,encoded=quantize_exl3(original.clone(),hd,qa,False)
            require(not qa['q_fallback'] and qa.get('out_scales_hessian') is True and np.isfinite(proxy),'Invalid rate probe')
            require(encoded['trellis'].shape==(ic//16,oc//16,16*k),'Encoded rate geometry differs')
            reconstructed=torch.empty((ic,oc),device='cuda',dtype=torch.float16)
            ext.reconstruct_had_slice(reconstructed,encoded['trellis'],encoded['suh'],encoded['svh'],k,True,False,0)
            rfn=float(((reconstructed.float()-original).square().sum()/original.square().sum()).sqrt())
            # Repack a distributed sample; fitting diagnostics need not re-export
            # the whole projection to validate rate and native decoder capability.
            ring=encoded['trellis'].reshape(-1,16*k);ids=torch.linspace(0,len(ring)-1,128,device='cuda').long()
            code=convert_tiles(ring[ids].cpu().numpy(),k);gpu=torch.from_numpy(code).cuda();dep=torch.from_numpy(dependency(k)).cuda()
            decoded=torch.empty((len(code),256),device='cuda',dtype=torch.float16)
            require(lib.native_body(gpu.data_ptr(),dep.data_ptr(),decoded.data_ptr(),k,len(code))==0,'Native rate decode failed')
            require(np.array_equal(decoded.cpu().numpy().view('u2'),mcg_lut()[escha_windows(code,k)].view('u2')),'Native rate reconstruction differs')
            result=dict(identity=identity,rfn=rfn,proxy_error=float(proxy),seconds=time.monotonic()-start,native_bit_exact=True,native_tiles=len(code))
            atomic_json(path,result);results.append(result);print('RATE_PROBE',name,k,rfn,flush=True)
            del hd,ho,encoded,reconstructed;torch.cuda.empty_cache()
        del original;torch.cuda.empty_cache()
    ratios={name:next(r['rfn'] for r in results if r['identity']['key']==rows[name]['source_key'] and r['identity']['K']==2)/
                 next(r['rfn'] for r in results if r['identity']['key']==rows[name]['source_key'] and r['identity']['K']==3) for name,_ in choices}
    atomic_json(root/'rate-probes.json',dict(passed=True,results=results,K2_over_K3_RFN=ratios,
       recommended_max_noise_scale=max(2.5,max(ratios.values())*1.1),
       scope='Actual stored-half YaQA MCG projection RFNs with pilot factors; diagnostics, not final-model or full-corpus guarantees'))


if __name__=='__main__':main()

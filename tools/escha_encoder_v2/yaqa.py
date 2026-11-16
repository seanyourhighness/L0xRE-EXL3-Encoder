"""Run the pinned YaQA collector with a memory-bounded original BF16 model."""
import argparse
import importlib.util
from pathlib import Path
import sys
import torch
from common import TREE,atomic_json,require,sha
from hf_stream import Streaming,cpu_model,guard_collector_recompute


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',required=True);p.add_argument('--calibration',required=True);p.add_argument('--out',required=True)
    p.add_argument('--layer',type=int,required=True);p.add_argument('--rows',type=int,default=250)
    p.add_argument('--layer-end',type=int);p.add_argument('--skip-rows',type=int,default=0)
    p.add_argument('--cols',type=int,default=2048);p.add_argument('--samples',type=int,default=1)
    p.add_argument('--sides',choices=['out','both'],default='out');p.add_argument('--passes',type=int)
    p.add_argument('--seed',type=int,default=20261116)
    return p


def main():
    a=parser().parse_args()
    end=a.layer_end or a.layer+1
    require(0<=a.layer<end<=64 and a.rows>0 and a.cols>0,'Invalid YaQA range')
    source=TREE/'util/yaqa_hessians.py';spec=importlib.util.spec_from_file_location('upstream_yaqa',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    holders=[]
    def load(args,devices):
        model=cpu_model(args,devices);stream=Streaming(model,activation_checkpoint=True)
        holders.append(stream);return model
    module.load_model=load
    guard_collector_recompute(module.Collector,lambda:any(x.recomputing for x in holders))
    args=argparse.Namespace(model_dir=a.model_dir,cal_data=a.calibration,out_dir=a.out,device='0',hessian_device='0',
                            reserve_vram=None,layers=(a.layer,end),head=False,rows=a.rows,skip_rows=a.skip_rows,cols=a.cols,
                            sides=a.sides,passes=a.passes,unweighted=False,diag=False,samples=a.samples,temperature=1.,
                            seed=a.seed,tf32=False,pattern=module.default_pattern,key_prefix=None,dry_run=False,log_interval=10)
    # The context is entered inside the loader so upstream creates and uses its
    # original Collector objects and backward hooks without duplicate forwards.
    original_loader=module.load_model
    def load_with_context(args,devices):
        model=original_loader(args,devices);context=holders[-1].saved_tensors();context.__enter__();holders[-1]._context=context
        return model
    module.load_model=load_with_context
    try:module.main(args)
    finally:
        for stream in reversed(holders):
            if hasattr(stream,'_context'):stream._context.__exit__(None,None,None)
            stream.close()
    from safetensors import safe_open
    files=sorted(Path(a.out).glob('*.safetensors'));require(bool(files),'No YaQA factors produced')
    for file in files:
        with safe_open(file,framework='np') as f:
            keys=list(f.keys());require('hout' in keys and 'hout_diag' in keys,'Require full Hout for two-sided LDLQ')
            import numpy as np
            for key in keys:require(np.isfinite(f.get_tensor(key)).all(),'Nonfinite factor: '+str(file))
    atomic_json(Path(a.out)/'collector.json',dict(upstream_sha256=sha(source),sides=a.sides,layer=a.layer,layer_end=end,
                rows=a.rows,cols=a.cols,samples=a.samples,seed=a.seed,files={f.name:sha(f) for f in files},
                method='Original YaQA full-Hout Collector; CPU-backed weights and saved tensors; online Hin during EXL3 conversion'))


if __name__=='__main__':main()

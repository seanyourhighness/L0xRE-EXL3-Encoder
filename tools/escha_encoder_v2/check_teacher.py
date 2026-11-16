"""Prove original-source forward fidelity before collecting Fisher factors."""
import argparse
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,require,sha
from hf_stream import cpu_model,Streaming


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();assets=Path(a.assets);source=assets/'source/radiance/tools/score.py'
    spec=importlib.util.spec_from_file_location('retained_reference_scorer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ref=module.Reference(str(assets/'data/smoke8'));ids=ref.doc_tokens(0)[:9].astype(np.int64)
    model=cpu_model(SimpleNamespace(model_dir=a.model),[0]);stream=Streaming(model)
    try:
        with torch.inference_mode():logits=model(input_ids=torch.from_numpy(ids)[None],use_cache=False).logits[0,:8].float().cpu().numpy()
        mean=float(ref.score_rows(0,0,logits)[0].mean());require(np.isfinite(mean) and mean<.005,'Original BF16 teacher forward compatibility failed')
        atomic_json(a.out,dict(passed=True,mean_reference_KL=mean,limit=.005,rows=8,
                    method='Original HF BF16 checkpoint, CPU-backed CUDA execution, pinned native golden reference',scorer_sha256=sha(source)))
    finally:stream.close()


if __name__=='__main__':main()

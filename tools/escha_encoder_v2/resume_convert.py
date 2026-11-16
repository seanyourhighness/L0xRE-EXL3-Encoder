"""Original checked converter entry point, with automatic durable resume."""
from pathlib import Path
import argparse
from common import require,read,atomic_json
import convert_checked as checked
from durable_io import recover

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--v2-context',required=True);control,rest=p.parse_known_args()
    args=checked.parser.parse_args(rest);require(args.codebook=='mcg','V2 requires MCG')
    target=Path(args.work_dir)/'V2_CONTEXT.json';context=read(control.v2_context)
    if target.exists():require(read(target)==context,'Calibration/recipe/factors changed; cannot resume these weights')
    else:atomic_json(target,context)
    recover(args.work_dir)
    args.resume=(Path(args.work_dir)/'ckpt/job.json').exists()
    checked.install(checked.convert_model,args.work_dir,args.out_dir)
    inputs,job,ok,error=checked.prepare(args)
    require(ok,error);checked.main(inputs,job)

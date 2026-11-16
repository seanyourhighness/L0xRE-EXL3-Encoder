"""Unmodified EXL3 automatic converter with a fail-fast calibration guard."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exllamav3.modules.quant.exl3_lib import quantize

original_finalize = quantize.finalize_capture_H


def checked_finalize(*args, **kwargs):
    result = original_finalize(*args, **kwargs)
    if result[0]:
        raise ValueError('Escha V1 forbids uncalibrated fallback; preserve checkpoint and inspect capture')
    return result


quantize.finalize_capture_H = checked_finalize

from exllamav3.conversion.convert_model import parser, prepare, main
from exllamav3.conversion import convert_model
from durable_io import install

if __name__ == '__main__':
    arguments=parser.parse_args()
    if arguments.codebook != 'mcg':
        raise ValueError('Escha V1 requires MCG')
    install(convert_model,arguments.work_dir,arguments.out_dir)
    in_args,job_state,ok,error=prepare(arguments)
    if not ok:
        raise ValueError(error)
    main(in_args,job_state)

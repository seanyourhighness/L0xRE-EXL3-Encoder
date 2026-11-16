"""Original SC math with individual Escha targets, not EXL3 fused-width ties."""
import argparse
import ast
import importlib.util
import sys
from common import TREE,read,require


def individual_targets(module,original,keys):
    def collect(mod):
        result=[];seen=set()
        for target in original(mod):
            for linear in target.linears:
                if linear.key in keys and linear.key not in seen:
                    result.append(module.Target(linear.key,[linear],linear.qbits_key));seen.add(linear.key)
        return result
    return collect


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--contract',required=True);control,rest=p.parse_known_args()
    keys={r['source_key'] for r in read(control.contract)['projections']}
    sys.path.insert(0,str(TREE));source=TREE/'sc_measure.py'
    spec=importlib.util.spec_from_file_location('original_sc_measure',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.collect_targets=individual_targets(module,module.collect_targets,keys)
    # Reuse the pinned script's own parser declarations, excluding its final
    # main(parser.parse_args()) call. No measurement/perturbation code is changed.
    block=ast.parse(source.read_text()).body[-1]
    require(isinstance(block,ast.If) and isinstance(block.body[-1],ast.Expr) and
            isinstance(block.body[-1].value,ast.Call) and getattr(block.body[-1].value.func,'id',None)=='main',
            'Upstream CLI structure changed')
    cli=ast.Module(body=block.body[:-1],type_ignores=[])
    exec(compile(cli,str(source),'exec'),module.__dict__)
    args=module.parser.parse_args(rest);module.main(args)
    require({x['key'] for x in read(args.out)['results']}==keys,'SC did not measure all individual coded projections')


if __name__=='__main__':main()

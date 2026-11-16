"""Materialize native-checked pilot projections as a same-size candidate GGUF."""
import argparse
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from adapter import atomic_json, inventory, require, sha
from gguf_patch import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--template',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.pilot);summary=json.loads((root/'summary.json').read_text())
    require(summary['complete'] and summary['results'],'Incomplete pilot')
    reader=GGUFReader(a.template);inventory(reader);tensors={t.name:t for t in reader.tensors}
    digest=sha(a.template);replacements={}
    for result in summary['results']:
        identity=result['identity'];name=identity['projection']['name']
        require(identity['template_sha256']==digest and result['native_decoder_bit_exact'],'Unvalidated pilot identity')
        artifact=root/(name+'.npz');require(sha(artifact)==result['artifact_sha256'],'Pilot artifact changed')
        with np.load(artifact,allow_pickle=False) as data:
            for key,suffix in (('code','escha_code'),('rin','escha_rin'),('rout','escha_rout')):
                value=data[key];target=tensors[name+'.'+suffix]
                require(value.dtype==target.data.dtype and value.shape==target.data.shape,'Pilot contract differs')
                path=root/(name+'.'+suffix+'.npy');np.save(path,value)
                replacements[name+'.'+suffix]=str(path)
    result=export(a.template,a.output,replacements,digest)
    atomic_json(str(a.output)+'.receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'}),flush=True)


if __name__=='__main__':main()

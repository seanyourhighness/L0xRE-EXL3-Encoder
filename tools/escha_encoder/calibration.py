"""Pack retained calibration-only token prefixes for EXL3's existing data loader."""
import argparse
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from adapter import atomic_json, require, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True)
    p.add_argument('--splits',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.reference)
    splits=json.loads(Path(a.splits).read_text())
    meta=json.loads((root/'kld.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    ids=[r['original_doc'] for r in selection]
    require(ids==splits['calibration']['radiance_calib_document_ids'], 'Calibration document selection differs')
    lengths=np.array([r['tokens'] for r in meta['docs']],np.int64)
    tokens=np.fromfile(root/'tokens.i32',np.int32)
    require(len(lengths)==len(ids) and sum(lengths)==len(tokens),'Token/length metadata mismatch')
    packed=np.zeros((len(ids),int(lengths.max())),np.int64);offset=0
    for i,n in enumerate(lengths):
        packed[i,:n]=tokens[offset:offset+n];offset+=n
    target=Path(a.output);require(not target.exists(),'Calibration output already exists')
    save_file(dict(input_ids=packed,lengths=lengths),str(target),metadata={
        'source_model_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
        'split':'calibration-only','selection_sha256':sha(root/'selection.json'),
        'tokens_sha256':sha(root/'tokens.i32'),'note':'Original prefixes, no labels or golden distributions included'})
    atomic_json(str(target)+'.json',dict(sha256=sha(target),rows=len(ids),tokens=len(tokens),
                max_length=int(lengths.max()),document_ids=ids,splits_sha256=sha(a.splits),
                selection_sha256=sha(root/'selection.json'),tokens_sha256=sha(root/'tokens.i32')))


if __name__=='__main__':main()

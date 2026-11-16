"""Frozen allocation, upstream automatic conversion, resumable Escha export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from gguf import GGUFReader
from safetensors import safe_open
from adapter import (MCG, MODEL_BYTES, UPSTREAM_COMMIT, atomic_json, convert_tiles,
                     inventory, require, sha, validate_scales)
from gguf_patch import export


def prepare(a):
    root=Path(a.work).resolve();root.mkdir(parents=True,exist_ok=True)
    template=Path(a.template).resolve()
    require(template.stat().st_size==MODEL_BYTES,'Template size differs from V1 contract')
    reader=GGUFReader(str(template));rows=inventory(reader)
    # Small body projections outside the 400 coded tensors stay unquantized upstream.
    # Existing vocabulary and small tensors are retained byte-for-byte by Escha export.
    recipe={'target_bpw':sum(r['ic']*r['oc']*r['k'] for r in rows)/sum(r['ic']*r['oc'] for r in rows),
            'tensors':{'*':16,**{r['source_key']:r['k'] for r in rows}}}
    source=Path(__file__).resolve().parents[2]
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(source),'merge-base','--is-ancestor',UPSTREAM_COMMIT,head],check=True)
    subprocess.run(['git','-C',str(source),'diff','--exit-code',UPSTREAM_COMMIT,'--','exllamav3','convert.py'],check=True)
    source_dir=Path(a.source).resolve()
    download=json.loads((source_dir/'download-state.json').read_text())
    require(download['state']=='complete' and download['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'A complete pinned BF16 source cache is required')
    for file in download['files']:
        require(sha(source_dir/file['file'])==file['sha256'],'BF16 source file differs: '+file['file'])
    with safe_open(a.calibration,framework='np') as f:
        ids=f.get_tensor('input_ids');lengths=f.get_tensor('lengths')
    require(ids.ndim==2 and lengths.shape==(len(ids),) and ids.dtype.kind in 'iu','Invalid packed calibration')
    require((ids>=0).all() and (lengths>0).all() and (lengths<=ids.shape[1]).all(),'Invalid calibration lengths/tokens')
    manifest=dict(format='l0xre-exl3-escha-v1',upstream_commit=UPSTREAM_COMMIT,adapter_commit=head,template=str(template),
                  template_sha256=sha(template),bytes=MODEL_BYTES,projections=rows,
                  source_dir=str(source_dir),source_revision=download['revision'],source_files=download['files'],
                  calibration=str(Path(a.calibration).resolve()),cal_rows=len(ids),cal_cols=ids.shape[1],
                  calibration_sha256=sha(a.calibration),codebook='mcg',multiplier=MCG,
                  correction_policy='Preserve all template bias/correction tensors; reject upstream projection biases',
                  retained='Vocabulary, small tensors, tokenizer, metadata, dependency tables',
                  implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    if (root/'manifest.json').exists():
        require(json.loads((root/'manifest.json').read_text())==manifest,'Resume manifest changed')
    else:
        atomic_json(root/'manifest.json',manifest)
        # JSON is valid YAML and avoids a second recipe serializer.
        atomic_json(root/'recipe.yaml',recipe)
    command=[sys.executable,str(Path(__file__).with_name('convert_checked.py')),'-i',manifest['source_dir'],'-w',str(root/'conversion'),
             '-o',str(root/'converted'),'-rcp',str(root/'recipe.yaml'),'-cb','mcg','-hb','16','-eb','16',
             '-vb','16','-mb','16','-cd',manifest['calibration'],'-cr',str(len(ids)),'-cc',str(ids.shape[1]),
             '-d','0','-cpi','120']
    atomic_json(root/'convert-command.json',command)
    print(json.dumps(dict(manifest=str(root/'manifest.json'),command=command),indent=2))


def run_conversion(a):
    root=Path(a.work)
    manifest=json.loads((root/'manifest.json').read_text())
    require(sha(manifest['calibration'])==manifest['calibration_sha256'],'Calibration digest changed')
    gates=json.loads(Path(a.gates).read_text())
    require(gates.get('template_sha256')==manifest['template_sha256'] and
            gates.get('upstream_commit')==manifest['upstream_commit'], 'Compatibility gates have different identities')
    require(all(gates.get(k) is True for k in ('cpu_compatibility_passed','projection_runtime_passed','layer_runtime_passed')),
            'Full conversion requires successful projection and complete-layer runtime compatibility gates')
    # Caller executes this command under the machine's GPU guardian, as for the pilot.
    state=json.loads(Path(a.lease_state).read_text())
    require(state['state']=='running' and state['child_pid']==os.getpid(),'Must be the lease guardian child')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==state['gpu'],'Lease device mismatch')
    command=json.loads((root/'convert-command.json').read_text())
    if a.resume:command.append('--resume')
    subprocess.run(command,check=True)
    files=sorted((root/'converted').glob('*.safetensors'))
    keys=set()
    for file in files:
        with safe_open(file,framework='np') as f:keys.update(f.keys())
    require(all(r['source_key']+'.trellis' in keys for r in manifest['projections']),
            'Upstream conversion did not produce all 400 body projections')
    atomic_json(root/'conversion.complete.json',dict(manifest_sha256=sha(root/'manifest.json'),
                files={str(f.relative_to(root)):sha(f) for f in files}))


def export_model(a):
    root=Path(a.work);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['implementation_sha256'].items():
        require(sha(Path(__file__).with_name(name))==digest,'Exporter implementation changed: '+name)
    require(sha(manifest['template'])==manifest['template_sha256'],'Template digest changed')
    reader=GGUFReader(manifest['template']);tensors={t.name:t for t in reader.tensors}
    rows=inventory(reader);require(rows==manifest['projections'],'Allocation changed')
    checkpoint=Path(a.checkpoint)
    files=sorted(checkpoint.glob('*.safetensors'))
    require(bool(files),'No EXL3 checkpoint safetensors')
    keys={}
    for file in files:
        with safe_open(file,framework='np') as f:
            for key in f.keys():
                require(key not in keys,'Duplicate checkpoint tensor: '+key)
                keys[key]=file
    patches={};receipts=[]
    selected=[r for r in rows if r['name']==a.projection] if a.projection else (
        [r for r in rows if r['layer']==a.layer] if a.layer is not None else rows)
    require(bool(selected),'Empty export selection')
    for row in selected:
        name=row['name'];key=row['source_key']
        # Reject an unrepresentable correction instead of silently dropping it.
        require(key+'.bias' not in keys,'Upstream projection bias needs an explicit runtime compatibility proof')
        packed={}
        for suffix in ('trellis','suh','svh','mcg'):
            full=key+'.'+suffix
            require(full in keys,'Missing EXL3 MCG tensor: '+full)
            with safe_open(keys[full],framework='np') as f:packed[suffix]=f.get_tensor(full)
        marker=int(np.asarray(packed['mcg']).reshape(-1)[0]) & 0xffffffff
        require(marker==MCG,'Wrong codebook marker')
        require(packed['trellis'].shape==(row['ic']//16,row['oc']//16,16*row['k']), 'Wrong K or geometry: '+name)
        identity={suffix:sha(keys[key+'.'+suffix]) for suffix in packed}
        directory=root/'export-checkpoints'/name;directory.mkdir(parents=True,exist_ok=True)
        record=directory/'receipt.json'
        if record.exists():
            receipt=json.loads(record.read_text());require(receipt['source_files']==identity,'Checkpoint source changed')
            for suffix,digest in receipt['output_sha256'].items():
                require(sha(directory/(suffix+'.npy'))==digest,'Resume output damaged')
        else:
            # An interrupted uncommitted projection is recomputed from its immutable source.
            # Only a receipt commits a checkpoint; candidate .npy files are safe to replace.
            code=convert_tiles(packed['trellis'],row['k'])
            ri,ro=validate_scales(packed['suh'],packed['svh'],row['ic'],row['oc'])
            arrays=dict(escha_code=code,escha_rin=ri,escha_rout=ro)
            for suffix,array in arrays.items():
                target=tensors[name+'.'+suffix]
                require(array.shape==target.data.shape and array.dtype==target.data.dtype,'Output contract differs')
                temp=directory/(suffix+'.partial.npy');np.save(temp,array);os.replace(temp,directory/(suffix+'.npy'))
            receipt=dict(projection=row,source_files=identity,
                         output_sha256={s:sha(directory/(s+'.npy')) for s in arrays})
            atomic_json(record,receipt)
        receipts.append(receipt)
        for suffix in ('escha_code','escha_rin','escha_rout'):patches[name+'.'+suffix]=str(directory/(suffix+'.npy'))
        print('EXPORT_CHECKPOINT',name,flush=True)
    result=export(manifest['template'],a.output,patches,manifest['template_sha256'])
    require(result['bytes']==MODEL_BYTES,'Model budget changed')
    atomic_json(str(a.output)+'.receipt.json',dict(export=result,projections=receipts,
                status='exported-unqualified',full_model=len(selected)==400,
                runtime_qualified=False,kld=None,dflash_throughput=None))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--template',required=True);prep.add_argument('--source',required=True)
    prep.add_argument('--calibration',required=True);prep.add_argument('--work',required=True)
    run=sub.add_parser('convert');run.add_argument('--work',required=True)
    run.add_argument('--lease-state',required=True);run.add_argument('--resume',action='store_true')
    run.add_argument('--gates',required=True)
    exp=sub.add_parser('export');exp.add_argument('--work',required=True);exp.add_argument('--checkpoint',required=True)
    exp.add_argument('--output',required=True)
    sel=exp.add_mutually_exclusive_group();sel.add_argument('--projection');sel.add_argument('--layer',type=int)
    a=p.parse_args();{'prepare':prepare,'convert':run_conversion,'export':export_model}[a.action](a)


if __name__=='__main__':main()

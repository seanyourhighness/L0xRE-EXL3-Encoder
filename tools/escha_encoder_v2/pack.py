"""MCG export with a changed allocation; checkpoint independent CPU projections."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import numpy as np
from safetensors import safe_open
from common import read,require,sha,atomic_json,MCG,convert_tiles,validate_scales,validate_recipe
from structural_export import export


def pack_one(task):
    row,k,source,hashes,directory,recipe_sha=task;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=dict(row=row,k=k,source=hashes,recipe_sha256=recipe_sha,codebook=MCG)
    receipt=directory/'receipt.json'
    if receipt.exists():
        saved=read(receipt);require(saved['identity']==identity,'Packing checkpoint identity differs')
        for suffix,digest in saved['outputs'].items():require(sha(directory/(suffix+'.npy'))==digest,'Damaged packing checkpoint')
        return row['name']
    data={}
    for suffix,file in source.items():
        key=row['source_key']+'.'+suffix
        with safe_open(file,framework='np') as f:data[suffix]=f.get_tensor(key)
    require((int(data['mcg'].reshape(-1)[0])&0xffffffff)==MCG,'Checkpoint codebook is not MCG')
    require(data['trellis'].shape==(row['ic']//16,row['oc']//16,16*k),'Checkpoint allocation differs from recipe')
    code=convert_tiles(data['trellis'],k);ri,ro=validate_scales(data['suh'],data['svh'],row['ic'],row['oc'])
    arrays=dict(escha_code=code,escha_rin=ri,escha_rout=ro)
    for suffix,value in arrays.items():
        tmp=directory/(suffix+'.partial.npy')
        with tmp.open('wb') as f:np.save(f,value);f.flush();os.fsync(f.fileno())
        os.replace(tmp,directory/(suffix+'.npy'))
    atomic_json(receipt,dict(identity=identity,outputs={s:sha(directory/(s+'.npy')) for s in arrays}))
    return row['name']


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--recipe',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--work',required=True);p.add_argument('--output',required=True)
    p.add_argument('--template-sha256',required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    require(1<=a.workers<=4,'Packing worker bound1–4')
    policy=read(a.contract);recipe=read(a.recipe);audit=validate_recipe(policy,recipe)
    files=sorted(Path(a.checkpoint).glob('*.safetensors'));require(bool(files),'No converted tensors')
    locations={};hashes={}
    for file in files:
        hashes[str(file)]=sha(file)
        with safe_open(file,framework='np') as f:
            for key in f.keys():require(key not in locations,'Duplicate checkpoint key');locations[key]=str(file)
    tasks=[];root=Path(a.work)
    for row in policy['projections']:
        key=row['source_key'];require(key+'.bias' not in locations,'Unexpected correction/bias in EXL3 projection')
        require(all(key+'.'+s in locations for s in ('trellis','suh','svh','mcg')),'Missing MCG projection: '+key)
        src={s:locations[key+'.'+s] for s in ('trellis','suh','svh','mcg')}
        tasks.append((row,recipe['tensors'][key],src,{s:hashes[file] for s,file in src.items()},root/row['name'],sha(a.recipe)))
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(pack_one,tasks,chunksize=1):print('V2_PACKED',name,flush=True)
    patches={row['name']+'.'+s:str(root/row['name']/(s+'.npy')) for row in policy['projections'] for s in ('escha_code','escha_rin','escha_rout')}
    if Path(a.output).exists():
        import hashlib
        result=read(str(a.output)+'.receipt.json')
        require(result['source_sha256']==a.template_sha256 and sha(a.output)==result['sha256'],'Existing candidate differs')
        table={r['name']:r for r in result['tensors']}
        for name,file in patches.items():
            array=np.load(file,mmap_mode='r',allow_pickle=False)
            require(hashlib.sha256(memoryview(array).cast('B')).hexdigest()==table[name]['sha256'],'Candidate/checkpoint payload differs')
    else:result=export(policy['template'],a.output,patches,a.template_sha256,policy['byte_ceiling'])
    atomic_json(str(a.output)+'.allocation.json',audit);print('V2_EXPORTED',result['sha256'],result['bytes'],flush=True)


if __name__=='__main__':main()

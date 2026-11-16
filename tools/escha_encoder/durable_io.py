"""Crash-safe storage around upstream EXL3; fitting and file payloads are unchanged."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def sync_file(path):
    with open(path,'rb') as f:os.fsync(f.fileno())


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def validate_tensor_file(path):
    path=Path(path);size=path.stat().st_size
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Truncated safetensors prefix: '+str(path))
        n=int.from_bytes(raw,'little')
        if n>16<<20 or n>size-8:raise ValueError('Invalid safetensors header length: '+str(path))
        header=json.loads(f.read(n))
    for name,info in header.items():
        if name=='__metadata__':continue
        start,end=info['data_offsets']
        if not 0<=start<=end<=size-8-n:raise ValueError('Truncated tensor '+name+': '+str(path))
    return size


def validate_checkpoint(folder):
    folder=Path(folder);job=json.loads((folder/'job.json').read_text())
    idx=job['next_module_idx']
    if not isinstance(idx,int) or idx<0:raise ValueError('Invalid checkpoint index')
    if idx:
        for name in ('state.safetensors','original_input_ids.safetensors'):validate_tensor_file(folder/name)
    manifest=folder/'durability.json'
    if manifest.exists():
        for name,pin in json.loads(manifest.read_text())['files'].items():
            if (folder/name).stat().st_size!=pin['bytes'] or digest(folder/name)!=pin['sha256']:
                raise ValueError('Checkpoint integrity mismatch: '+name)
    return job


def recover(work):
    """Select an intact upstream checkpoint; quarantine incomplete writes without deleting them."""
    work=Path(work);current=work/'ckpt'
    if not current.exists() and not (work/'ckpt_old').exists():return None
    valid=[];errors={}
    for name in ('ckpt','ckpt_old','ckpt_new'):
        if not (work/name).exists():continue
        try:valid.append((validate_checkpoint(work/name)['next_module_idx'],name))
        except Exception as e:errors[name]=type(e).__name__+': '+str(e)
    if not valid:raise ValueError('No intact EXL3 checkpoint; preserve artifacts: '+json.dumps(errors))
    idx,name=max(valid,key=lambda x:(x[0],x[1]=='ckpt'))
    damaged=[]
    for file in sorted((work/'qtensors').glob('*.safetensors')):
        try:validate_tensor_file(file)
        except Exception as e:damaged.append((file,str(e)))
    if name=='ckpt' and not damaged and not (work/'ckpt_new').exists():
        return dict(selected=name,next_module_idx=idx,recovered=False)
    archive=work/('recovery-'+str(time.time_ns()));archive.mkdir()
    if name!='ckpt':
        staged=work/('ckpt_recover_'+uuid.uuid4().hex)
        shutil.copytree(work/name,staged)
        for file in staged.iterdir():
            if file.is_file():sync_file(file)
        validate_checkpoint(staged);sync_dir(staged)
        if current.exists():os.rename(current,archive/'ckpt_damaged')
        os.rename(staged,current);sync_dir(work)
    if (work/'ckpt_new').exists():
        os.rename(work/'ckpt_new',archive/'ckpt_new_previous');sync_dir(work)
    # Preserve valid but uncommitted layer payloads before upstream resume refits
    # from the older activation checkpoint; remove only structurally damaged files.
    import re
    for file in sorted((work/'qtensors').glob('*.safetensors')):
        match=re.search(r'\.layers\.(\d+)\.safetensors$',file.name)
        if match and int(match[1])+1>=idx:
            shutil.copy2(file,archive/file.name);sync_file(archive/file.name)
    for file,error in damaged:
        if not (archive/file.name).exists():
            shutil.copy2(file,archive/file.name);sync_file(archive/file.name)
        file.unlink()
    result=dict(selected=name,next_module_idx=idx,recovered=True,errors=errors,
                damaged_tensors=[dict(file=str(p),error=e) for p,e in damaged],archive=str(archive))
    (archive/'recovery.json').write_text(json.dumps(result,indent=2)+'\n');sync_file(archive/'recovery.json');sync_dir(archive)
    return result


def install(converter,work,out):
    """Use the original serializer, fsync its bytes, and publish atomically."""
    work=Path(work).resolve();roots=(work,Path(out).resolve())
    original=converter.save_file
    def save(tensors,filename,metadata=None):
        path=Path(filename).resolve()
        if not any(path.is_relative_to(root) for root in roots):return original(tensors,filename,metadata)
        temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            original(tensors,str(temp),metadata)
            validate_tensor_file(temp);sync_file(temp)
            os.replace(temp,path);sync_dir(path.parent)
        finally:
            if temp.exists():temp.unlink()
    # Modules imported before installation hold their own serializer aliases.
    for module in tuple(sys.modules.values()):
        if module and getattr(module,'__name__','').startswith('exllamav3.') and getattr(module,'save_file',None) is original:
            module.save_file=save
    original_dict=converter.save_dict
    def save_dict(filename,data,args):
        original_dict(filename,data,args);path=Path(args['work_dir'])/filename;sync_file(path);sync_dir(path.parent)
    converter.save_dict=save_dict
    original_preserves=converter.save_preserves
    def preserves(data,idx,filename,args):
        original_preserves(data,idx,filename,args)
        folder=Path(args['work_dir'])/Path(filename).parent
        if folder.name!='ckpt_new':return
        files={}
        for file in folder.iterdir():
            if not file.is_file() or file.name=='durability.json':continue
            if file.suffix=='.safetensors':validate_tensor_file(file)
            sync_file(file);files[file.name]=dict(bytes=file.stat().st_size,sha256=digest(file))
        marker=folder/'durability.json';marker.write_text(json.dumps(dict(files=files),indent=2)+'\n')
        sync_file(marker);sync_dir(folder)
    converter.save_preserves=preserves
    original_rename=os.rename
    def rename(src,dst,*args,**kwargs):
        result=original_rename(src,dst,*args,**kwargs)
        target=Path(dst)
        if target.parent.resolve()==work and target.name in ('ckpt','ckpt_old'):sync_dir(work)
        return result
    os.rename=rename

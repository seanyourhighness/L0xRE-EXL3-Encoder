import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import torch
import durable_io as io


def upstream_serializer():
    path=Path(__file__).resolve().parents[2]/'exllamav3/loader/safetensors_alt.py'
    spec=importlib.util.spec_from_file_location('original_safetensors_serializer',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


class StorageTests(unittest.TestCase):
    def test_serializer_payload_unchanged_and_checkpoint_manifest(self):
        source=upstream_serializer();rename=os.rename
        fake=types.ModuleType('exllamav3.storage_guard_test');fake.save_file=source.save_file
        def save_dict(filename,data,args):(Path(args['work_dir'])/filename).write_text(json.dumps(data))
        fake.save_dict=save_dict;fake.save_preserves=lambda *args:None
        sys.modules[fake.__name__]=fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);work=root/'conversion';work.mkdir();out=root/'out';out.mkdir()
                values={'float':torch.arange(16,dtype=torch.float32),'half':torch.ones(7,dtype=torch.float16),
                        'trellis':torch.arange(48,dtype=torch.int16),
                        'mcg':torch.tensor(0xCBAC1FED,dtype=torch.uint32).view(torch.int32)}
                source.save_file(values,str(root/'original.safetensors'))
                io.install(fake,work,out);fake.save_file(values,str(out/'guarded.safetensors'))
                self.assertEqual((root/'original.safetensors').read_bytes(),(out/'guarded.safetensors').read_bytes())
                folder=work/'ckpt_new';folder.mkdir();args={'work_dir':str(work)}
                fake.save_dict('ckpt_new/job.json',{'next_module_idx':2},args)
                for name in ('state.safetensors','original_input_ids.safetensors'):
                    fake.save_file(values,str(folder/name))
                fake.save_preserves([],1,'ckpt_new/preserve.pt',args)
                self.assertEqual(io.validate_checkpoint(folder)['next_module_idx'],2)
                self.assertTrue((folder/'durability.json').exists())
                with (folder/'state.safetensors').open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
                with self.assertRaises(ValueError):io.validate_checkpoint(folder)
        finally:
            os.rename=rename;sys.modules.pop(fake.__name__,None)

    def test_reboot_truncation_falls_back_and_preserves_artifacts(self):
        source=upstream_serializer()
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);values={'tensor.0':torch.ones(128,dtype=torch.float16)}
            for name,idx in [('ckpt',5),('ckpt_old',2)]:
                folder=work/name;folder.mkdir();(folder/'job.json').write_text(json.dumps({'next_module_idx':idx}))
                for filename in ('state.safetensors','original_input_ids.safetensors'):
                    source.save_file(values,str(folder/filename))
            expected=io.digest(work/'ckpt_old/state.safetensors')
            with (work/'ckpt/state.safetensors').open('r+b') as f:f.truncate(32)
            qt=work/'qtensors';qt.mkdir()
            source.save_file(values,str(qt/'model.language_model.layers.3.safetensors'))
            with (qt/'model.language_model.layers.3.safetensors').open('r+b') as f:f.truncate(40)
            result=io.recover(work)
            self.assertEqual(result['next_module_idx'],2)
            self.assertEqual(io.digest(work/'ckpt/state.safetensors'),expected)
            self.assertTrue((Path(result['archive'])/'ckpt_damaged/state.safetensors').exists())
            self.assertTrue((Path(result['archive'])/'model.language_model.layers.3.safetensors').exists())
            self.assertFalse((qt/'model.language_model.layers.3.safetensors').exists())


if __name__=='__main__':unittest.main()

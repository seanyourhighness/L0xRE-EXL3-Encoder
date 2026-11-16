"""Canonical frozen benchlocal 0.9.9 entry point, with an isolated catalog copy."""
import os
import hashlib
import json
from pathlib import Path
import sys

source=Path(os.environ['L0XRE_BENCHLOCAL_SOURCE']).resolve()
if not (source/'benchlocal_cli').is_dir():
    raise ValueError('Missing frozen benchlocal catalog')
manifest=json.loads((source/'MANIFEST.json').read_text())
for relative,expected in manifest['files'].items():
    digest=hashlib.sha256()
    with (source/relative).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError('Frozen task catalog changed: '+relative)
sys.path.insert(0,str(source))
import benchlocal_cli
if benchlocal_cli.__version__!='0.9.9':
    raise ValueError('Canonical qualification requires benchlocal 0.9.9')
from benchlocal_cli.cli import main

if __name__=='__main__':raise SystemExit(main())

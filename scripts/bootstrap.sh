#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
python - <<'BOOTSTRAP'
import json
import subprocess
from pathlib import Path
root = Path.cwd()
spec = json.loads((root/'third_party/versions.json').read_text())['slime']
target = root/'third_party/slime'
if not target.exists():
    subprocess.run(['git','clone','--no-checkout',spec['url'],str(target)],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',spec['commit']],check=True)
actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
if actual != spec['commit']:
    raise SystemExit('Slime revision differs from the pinned version')
patch = root/'third_party/patches/slime.patch'
check = subprocess.run(['git','-C',str(target),'apply','--reverse','--check',str(patch)],capture_output=True)
if check.returncode:
    subprocess.run(['git','-C',str(target),'apply','--check',str(patch)],check=True)
    subprocess.run(['git','-C',str(target),'apply',str(patch)],check=True)
BOOTSTRAP
npm --prefix runtime/dsh install
python -m pip install -e '.[test]'
python -m harness_s.cli check-pool

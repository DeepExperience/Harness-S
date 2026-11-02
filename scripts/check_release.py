#!/usr/bin/env python3
"""Check public assets, imports and accidental private infrastructure references."""
from pathlib import Path
import ast
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from harness_s.config import ROOT
from harness_s.pool import HarnessPool

errors = []
count = HarnessPool(ROOT).verify()
for base in ('src','scripts','configs','harness_pool','runtime/appworld','third_party'):
    for path in (ROOT/base).rglob('*'):
        if any(part in ('__pycache__','node_modules','.git','slime') for part in path.parts) or not path.is_file():
            continue
        if path.suffix not in ('.py','.sh','.json','.md','.mjs','.yml','.yaml','.patch'):
            continue
        text = path.read_text()
        if path.suffix=='.py':
            ast.parse(text,filename=str(path))
        # Construct patterns to avoid flagging the scanner's source itself.
        if re.search(r'/m' + r'nt/|xiaohong' + r'shu\.com|red' + r'note\.life|sk-' + r'or-v1-[a-zA-Z0-9]{20,}|sk-' + r'[a-zA-Z0-9]{24,}',text):
            errors.append(str(path.relative_to(ROOT)))
if errors:
    raise SystemExit('Private paths or credentials: '+', '.join(errors))
print(f'Release checks passed: {count} harnesses; Python syntax and private-path scan clean.')

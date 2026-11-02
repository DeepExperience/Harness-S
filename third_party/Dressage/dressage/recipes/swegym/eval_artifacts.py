"""Archive complete grading logs before a disposable evaluator is removed."""
import asyncio
import base64
import hashlib
import os
from pathlib import Path
import re
import shlex


def log_archive_command(instance_id):
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(instance_id))
    name = 'dressage_swegym_eval_' + slug + '.log'
    return ('cd /tmp && test -f ' + shlex.quote(name) + ' && tar -czf - -- '
            + shlex.quote(name) + '*')


async def save_eval_logs(paddock, state, metadata, session_id):
    root = os.environ.get('HAPT_JOINT_RUN_ROOT')
    if not root:
        return
    from dressage.paddock.blackbox.execute_hooks import maybe_await
    info = metadata.setdefault('swegym_eval', {})
    async def execute(cmd):
        return await maybe_await(paddock.execute_cmd(state, session_id=session_id, cmd=cmd, timeout=15))
    archive = '/tmp/hapt-official-eval-logs.tar.gz'
    try:
        async def collect():
            result = await execute(log_archive_command(metadata.get('instance_id', '')) + ' > ' + archive + ' && wc -c < ' + archive)
            if result.get('returncode') != 0:
                return
            size = int(result['stdout'].strip())
            if not 0 < size <= 32 * 1024 * 1024:
                raise ValueError('unexpected compressed grading log size')
            chunks = []
            for index in range((size + 8191) // 8192):
                result = await execute(f'dd if={archive} bs=8192 skip={index} count=1 2>/dev/null | base64 -w0')
                if result.get('returncode') != 0 or result.get('stdout_truncated'):
                    raise RuntimeError('grading log transport failed/truncated')
                chunks.append(base64.b64decode(result['stdout'], validate=True))
            blob = b''.join(chunks)
            if len(blob) != size:
                raise ValueError('incomplete grading log archive')
            target = Path(root)/'verifier-artifacts'/session_id/'official-eval-logs.tar.gz'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            info.update(log_archive_path=str(target), log_archive_sha256=hashlib.sha256(blob).hexdigest())
        await asyncio.wait_for(collect(), timeout=30)
    except Exception as exc:
        info['log_archive_warning'] = f'{type(exc).__name__}: {exc}'

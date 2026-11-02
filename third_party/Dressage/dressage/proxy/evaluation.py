"""Freeze resident weights through evaluation and bounded natural request drain."""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import time
import httpx

logger = logging.getLogger(__name__)

class EvaluationDrainTimeout(RuntimeError):
    """The supervisor must rebuild the owned service before any weight update."""


def _receipt(path, value):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)


@contextmanager
def evaluation_window(base_url, *, client=None, drain_timeout_seconds=None, receipt=None):
    owns_client = client is None
    client = client or httpx.Client(timeout=65.0, trust_env=False)
    base_url = base_url.rstrip('/')
    timeout = float(drain_timeout_seconds if drain_timeout_seconds is not None else
                    os.environ.get('HAPT_EVALUATION_DRAIN_TIMEOUT_SECONDS', '1800'))
    if not 0 < timeout <= 1800:
        raise ValueError('evaluation drain timeout must be in (0, 1800]')
    resumed = False
    state = {'evaluation_completed': False, 'service_quiesced': False, 'drain_checks': []}
    try:
        response = client.get(f'{base_url}/integration/capabilities')
        response.raise_for_status()
        capabilities = response.json()
        if capabilities.get('sampling_seed_mapping') != 'openai.seed->sglang.sampling_seed/v1':
            raise RuntimeError('resident proxy lacks the seed mapping fix')
        if not (capabilities.get('weight_version_authoritative') and capabilities.get('weight_versions_consistent')
                and capabilities.get('current_weight_version') is not None):
            raise RuntimeError('resident Policy weight version cannot be verified')
        resumed = True
        response = client.post(f'{base_url}/v1/rollout/resume', json={
            'reason': 'frozen_harness_evaluation', 'version': str(capabilities['current_weight_version'])})
        response.raise_for_status()
        if response.json().get('status') in {'backend_not_ready', 'shutting_down'}:
            raise RuntimeError('training proxy could not resume for HE')
        yield capabilities
        state['evaluation_completed'] = True
        _receipt(receipt, state)
    finally:
        try:
            if resumed:
                started = time.monotonic()
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        state['rebuild_required'] = True
                        _receipt(receipt, state)
                        raise EvaluationDrainTimeout('EVALUATION_DRAIN_REBUILD_REQUIRED: requests have not drained; Policy must remain frozen')
                    try:
                        response = client.post(f'{base_url}/v1/rollout/pause', json={
                            'reason': 'harness_evaluation_complete', 'mode': 'drain',
                            'timeout_seconds': min(30.0, remaining)}, timeout=min(60.0, remaining))
                        response.raise_for_status()
                        status = response.json()
                    except httpx.HTTPError as exc:
                        status = {'quiesced': False, 'error': str(exc)}
                    state['drain_checks'].append({'elapsed_seconds': time.monotonic() - started, **status})
                    state['service_quiesced'] = bool(status.get('quiesced'))
                    _receipt(receipt, state)
                    logger.info('evaluation drain: %s', state['drain_checks'][-1])
                    if state['service_quiesced']:
                        break
                    # Also bound a proxy that returns immediately instead of waiting.
                    time.sleep(min(1.0, max(0, timeout - (time.monotonic() - started))))
        finally:
            if owns_client:
                client.close()

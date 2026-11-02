"""Versioned grading command upgrade; original dataset and frozen runs stay intact."""
import base64
import importlib.util
from pathlib import Path
import re
import shlex
import zlib

BUDGET_MARKER = '# HAPT_PHASE_SOURCE_B64='


def unwrap_budget_script(script):
    lines = script.splitlines()
    if len(lines) > 1 and lines[1].startswith(BUDGET_MARKER):
        return zlib.decompress(base64.b64decode(lines[1][len(BUDGET_MARKER):])).decode()
    return script


def budget_eval_script(script, instance_id, budgets):
    """Keep official prepare/test/cleanup commands, give each its own clock."""
    script = instrument_eval_script(unwrap_budget_script(script), instance_id)
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(instance_id))
    phases = shlex.quote(f'/tmp/dressage_swegym_eval_{slug}.log.phases')
    times = shlex.quote(f'/tmp/dressage_swegym_eval_{slug}.log.timings')
    prefix = '#!/bin/bash\nset +e\n: > ' + phases + '\n(\n'
    prepare, separator, tail = script[len(prefix):].partition('\n)\nHAPT_PREP_RC=$?\n')
    if not script.startswith(prefix) or not separator:
        raise ValueError('unsupported phased evaluation script')
    test = next(line for line in tail.splitlines() if line.startswith('pytest '))
    cleanup = next(line for line in tail.splitlines() if line.startswith('git checkout '))
    source = base64.b64encode(zlib.compress(script.encode())).decode()
    result = '#!/bin/bash\n' + BUDGET_MARKER + source + '\nset +e\n: > ' + phases + '\n: > ' + times + '\n'
    activation = 'source /opt/miniconda3/bin/activate\nconda activate testbed\ncd /testbed\nexport PATH="/opt/hapt-swegym-bin:$PATH"\n'
    for stage, body, variable in [('prepare', prepare, 'HAPT_PREP_RC'), ('test', activation + test, 'HAPT_TEST_RC'), ('cleanup', activation + cleanup, 'HAPT_CLEAN_RC')]:
        marker = 'HAPT_OFFICIAL_' + stage.upper() + '_END'
        env = {'prepare': 'PREPARE', 'test': 'VERIFIER', 'cleanup': 'CLEANUP'}[stage]
        result += (f'HAPT_PHASE_START=$SECONDS\ntimeout --kill-after=5 "${{HAPT_SWEGYM_{env}_TIMEOUT_SECONDS:-{budgets[stage]}}}" bash -s <<\'{marker}\'\n'
                   + body + '\n' + marker + '\n' + variable + '=$?\n'
                   + f'printf "{stage}=%s\\n" "${variable}" >> ' + phases + '\n'
                   + f'printf "{stage}=%s\\n" "$((SECONDS-HAPT_PHASE_START))" >> ' + times + '\n')
        if stage == 'prepare':
            result += '[ "$HAPT_PREP_RC" -eq 0 ] || exit "$HAPT_PREP_RC"\n'
    return result + 'exit "$HAPT_TEST_RC"\n'

PYDANTIC_INSTALLED_RUNTIME_PROBE = r"""
import importlib.util, importlib.metadata, json
from pathlib import Path
from packaging.requirements import Requirement
from pip._vendor import tomli
project = tomli.loads(Path('/testbed/pyproject.toml').read_text())['project']
assert project['name'] == 'pydantic'
checked = {}
for text in project['dependencies']:
    requirement = Requirement(text)
    if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
        continue
    version = importlib.metadata.version(requirement.name)
    assert requirement.specifier.contains(version), (text, version)
    checked[requirement.name] = version
spec = importlib.util.find_spec('pydantic')
assert spec and str(Path(spec.origin).resolve()).startswith('/testbed/pydantic/')
import pytest
print(json.dumps({'installed_dependencies': checked, 'source': spec.origin,
                  'bootstrap': 'pydantic-installed-runtime-v1'}))
"""


def instrument_eval_script(script, instance_id):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_id))
    phases = shlex.quote(f"/tmp/dressage_swegym_eval_{slug}.log.phases")
    # Newly converted datasets already carry this wrapper. Verify its complete
    # structure before reusing it; a second wrapping would corrupt phase exits.
    prefix = "#!/bin/bash\nset +e\n: > " + phases + "\n(\n"
    if script.startswith(prefix):
        prepare, separator, tail = script[len(prefix):].partition('\n)\nHAPT_PREP_RC=$?\n')
        tests = [line for line in tail.splitlines() if line.startswith('pytest ')]
        cleanups = [line for line in tail.splitlines() if line.startswith('git checkout ')]
        if not separator or len(tests) != 1 or len(cleanups) != 1:
            raise ValueError('incomplete instrumented evaluation script')
        reconstructed = prepare + tests[0] + '\n' + cleanups[0] + '\n'
        if instrument_eval_script(reconstructed, instance_id) != script:
            raise ValueError('instrumented evaluation phase wrapper changed')
        return script
    # The pinned package emits exactly one pytest command after its test-patch
    # heredoc, followed by a checkout. Do not rewrite patch contents.
    split = script.rfind("\nEOF_114329324912\n")
    if split < 0:
        raise ValueError("unsupported official eval script: test-patch delimiter")
    start = split + len("\nEOF_114329324912\n")
    tail = script[start:].strip().splitlines()
    if len(tail) != 2 or not tail[0].startswith("pytest ") or not tail[1].startswith("git checkout "):
        raise ValueError("unsupported official eval script: test/cleanup structure")
    prepare = script[:start].replace("set -xo pipefail", "set -exo pipefail", 1)
    # Run setup in a subshell so errexit cannot be suppressed by `if`/`||`.
    return ("#!/bin/bash\nset +e\n: > " + phases + "\n(\n" + prepare + "\n)\n"
            + 'HAPT_PREP_RC=$?\nprintf "prepare=%s\\n" "$HAPT_PREP_RC" >> ' + phases + '\n'
            + '[ "$HAPT_PREP_RC" -eq 0 ] || exit "$HAPT_PREP_RC"\n'
            + 'source /opt/miniconda3/bin/activate\nconda activate testbed\ncd /testbed\n'
            + 'export PATH="/opt/hapt-swegym-bin:$PATH"\n'
            + tail[0] + '\nHAPT_TEST_RC=$?\nprintf "test=%s\\n" "$HAPT_TEST_RC" >> ' + phases + '\n'
            + tail[1] + '\nHAPT_CLEAN_RC=$?\nprintf "cleanup=%s\\n" "$HAPT_CLEAN_RC" >> ' + phases + '\n'
            + 'exit "$HAPT_TEST_RC"\n')


def upgrade_grading_command(metadata, *, phase_budgets=None):
    """Retain the exact pinned parser/script; replace only the wrapper/driver."""
    from .runtime import LEGACY_MYPY_TASKS, SETUPTOOLS_ASSET, MONAI_WSI_TASKS, MONAI_WSI_ASSET
    from .runtime import BUILD_LIMIT_SCRIPT, HYDRA_ISORT_ASSET
    before = metadata['blackbox_execute_cmds']['before_agent']
    if not any(hook.get('name') == 'swegym_build_limits' for hook in before):
        before.append({'name': 'swegym_build_limits', 'timeout': 30, 'required': True,
                       'cmd': '/opt/miniconda3/envs/testbed/bin/python -c ' + shlex.quote(BUILD_LIMIT_SCRIPT)})
    if metadata.get('instance_id') == 'facebookresearch__hydra-616' and not any(h.get('name') == 'swegym_hydra_isort' for h in before):
        # The commit retains the 4.3.21 version string; pip would otherwise
        # keep a same-version release already in the image instead of this wheel.
        before.append({'name': 'swegym_hydra_isort', 'timeout': 120, 'required': True,
                       'cmd': '/opt/miniconda3/envs/testbed/bin/python -m pip install --no-index --no-deps --force-reinstall '
                              + shlex.quote(HYDRA_ISORT_ASSET['destination'])})
    if (metadata.get('repo', '').lower() == 'project-monai/monai'
            and metadata['instance_id'].rsplit('-', 1)[-1] in MONAI_WSI_TASKS):
        before = metadata['blackbox_execute_cmds']['before_agent']
        if not any(hook.get('name') == 'swegym_wsi_test_asset' for hook in before):
            before.append({'name': 'swegym_wsi_test_asset', 'timeout': 120, 'required': True,
                           'cmd': 'cp ' + shlex.quote(MONAI_WSI_ASSET['destination']) + ' /testbed/tests/testing_data/temp_CMU-1.tiff.tiff'})
        metadata['public_test_data_recipe'] = 'monai-cmu1-sha256-v1'
    if (metadata.get('repo') == 'python/mypy'
            and metadata['instance_id'].rsplit('-', 1)[-1] in LEGACY_MYPY_TASKS):
        before = metadata['blackbox_execute_cmds']['before_agent']
        if not any(hook.get('name') == 'swegym_legacy_build_runtime' for hook in before):
            before.append({'name': 'swegym_legacy_build_runtime', 'timeout': 120, 'required': True,
                           'cmd': '/opt/miniconda3/envs/testbed/bin/python -m pip install --no-index --no-deps ' + shlex.quote(SETUPTOOLS_ASSET['destination'])})
        metadata['environment_install_recipe'] = 'mypy-legacy-setuptools-58.2.0-v1'
    # Upgrade already-generated copies of these two mandatory repair hooks.
    # Do not default arbitrary user hooks: malformed configuration must fail.
    for hook in metadata['blackbox_execute_cmds']['before_agent']:
        if hook.get('name') in {'swegym_wsi_test_asset', 'swegym_legacy_build_runtime'}:
            hook.setdefault('required', True)
    path = Path(__file__).resolve().parents[3] / "examples/data/swegym/prepare_swegym_data.py"
    spec = importlib.util.spec_from_file_location("_hapt_swegym_converter", path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    for hook in metadata["blackbox_execute_cmds"]["after_agent"]:
        if hook.get("name") != "swegym_harness":
            continue
        decoded = []
        for payload in re.findall(r"[A-Za-z0-9+/=]{100,}", hook["cmd"]):
            try:
                decoded.append(zlib.decompress(base64.b64decode(payload)).decode())
            except (ValueError, zlib.error, UnicodeDecodeError):
                pass
        scripts = [x for x in decoded if x.startswith("#!/bin/bash")]
        parsers = [x for x in decoded if "MAP_REPO_TO_PARSER =" in x]
        if len(scripts) != 1 or len(parsers) != 1:
            raise ValueError("cannot identify the frozen official evaluation script/parser")
        parser = parsers[0].split("\nimport base64,json,sys,zlib", 1)[0]
        if parser == parsers[0]:
            raise ValueError("cannot identify the frozen grading driver")
        parser = parser.rstrip()
        eval_script = unwrap_budget_script(scripts[0])
        if metadata.get('instance_id') == 'facebookresearch__hydra-616':
            # Resolve the exact Git dependency locally without altering the
            # tracked requirements or installing a different released isort.
            rewrite = "\n".join([
                'from pathlib import Path',
                "p=Path('/testbed/requirements/dev.txt')",
                "text=p.read_text()",
                "old='isort@git+https://github.com/timothycrosley/isort.git@c54b3dd#egg=isort'",
                "assert text.count(old)==1, 'Hydra isort recipe changed'",
                "text=text.replace(old, 'isort @ file://" + HYDRA_ISORT_ASSET['destination'] + "')",
                "text='\\n'.join('-r '+str(p.parent/line[3:].strip()) if line.startswith('-r ') else line for line in text.splitlines())+'\\n'",
                "Path('/tmp/hapt-hydra-dev.txt').write_text(text)",
            ])
            original = 'pip install -r requirements/dev.txt'
            if original in eval_script:
                eval_script = eval_script.replace(original,
                    'python -c ' + shlex.quote(rewrite) + '\npip install --no-index -r /tmp/hapt-hydra-dev.txt', 1)
            metadata['environment_install_recipe'] = 'hydra-isort-c54b3dd-offline-v1'
        if metadata.get("repo") == "iterative/dvc" and metadata.get('environment_install_recipe') != 'dvc-pinned-image-editable-v1':
            prefix, separator, patch_tail = eval_script.partition("\ngit apply -v - <<")
            lines = prefix.splitlines()
            matches = [i for i, line in enumerate(lines) if line.startswith("python -m pip install --upgrade pip wheel GitPython;")]
            if len(matches) != 1 or not separator:
                raise ValueError("unsupported DVC installation recipe")
            # Preserve the official image's installed runtime; the generic
            # historical bootstrap requests mutually incompatible dev extras.
            # Reinstall source so edits/build failures remain observable.
            lines[matches[0]] = "python -m pip install --no-build-isolation --no-deps -e ."
            eval_script = "\n".join(lines) + separator + patch_tail
            metadata["environment_install_recipe"] = "dvc-pinned-image-editable-v1"
        if metadata.get("repo") == "pydantic/pydantic" and "pdm add pre-commit; make install;" in eval_script:
            # The pinned image already has the task runtime. Development hook
            # installation resolves floating dependencies and requires network.
            # Verify every declared runtime dependency and the source binding.
            eval_script = eval_script.replace("pdm add pre-commit; make install;",
                "python -c " + shlex.quote(PYDANTIC_INSTALLED_RUNTIME_PROBE) + ";", 1)
            metadata["environment_install_recipe"] = "pydantic-installed-runtime-v1"
        if metadata.get('repo') == 'bokeh/bokeh' and metadata.get('environment_install_recipe') != 'bokeh-installed-runtime-v1':
            # The official image already binds Python imports to src/bokeh.
            # Rebuilding the JS release demands deleted history/tags and a
            # networked Node toolchain, unrelated to these Python test targets.
            prefix, separator, patch_tail = eval_script.partition('\ngit apply -v - <<')
            if prefix.count('python -m pip install -e .') != 1 or not separator:
                raise ValueError('unsupported Bokeh installation recipe')
            probe = PYDANTIC_INSTALLED_RUNTIME_PROBE.replace('pydantic', 'bokeh').replace('/testbed/bokeh/', '/testbed/src/bokeh/')
            prefix = prefix.replace('python -m pip install -e .', 'python -c ' + shlex.quote(probe), 1)
            eval_script = prefix + separator + patch_tail
            metadata['environment_install_recipe'] = 'bokeh-installed-runtime-v1'
        timeouts = re.findall(r"\btimeout ([0-9]+) bash ", hook["cmd"])
        if len(timeouts) != 1:
            raise ValueError("cannot identify frozen evaluator timeout")
        timeout = int(metadata.get('swegym_phase_budgets', {}).get('test', timeouts[0]))
        budgets = phase_budgets or {'prepare': 900, 'test': timeout, 'cleanup': 60}
        hook["cmd"] = converter.build_eval_command(
            eval_script=eval_script, instance=metadata, timeout=timeout,
            log_parser_source=parser+"\n"+converter.SANDBOX_GRADING_DRIVER,
            phase_budgets=budgets)
        hook['timeout'] = sum(budgets.values()) + 120
        metadata['swegym_phase_budgets'] = budgets
        metadata['evaluation_budget_protocol'] = 'separate-queue-prepare-test-v1'
        metadata["verifier_protocol"] = "phased-pytest-v2"
        metadata["evaluation_transport"] = "stdout-pipe-v1"

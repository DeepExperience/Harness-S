"""Shared task runtime for agent tools, fresh grading, and environment checks.

Control services still use their absolute Python executable. Public test assets
are mounted as individual read-only files; no repository or image is modified.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

RUNTIME_EPOCH = "swegym-offline-phase-budgets-v3"
TESTBED = "/opt/miniconda3/envs/testbed"
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MONAI_ASSET = {
    "name": "se_resnet50-ce0d4300.pth",
    "sha256": "ce0d430017d3f4aa6b5658c72209f3bfffb060207fd26a2ef0b203ce592eba01",
    "url": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/se_resnet50-ce0d4300.pth",
    "destination": "/root/.cache/torch/hub/checkpoints/se_resnet50-ce0d4300.pth",
}




VERSIONEER_ASSET = {'name': 'versioneer-0.29-py3-none-any.whl', 'sha256': '0f1a137bb5d6811e96a79bb0486798aeae9b9c6efc24b389659cebb0ee396cb9', 'url': 'https://files.pythonhosted.org/packages/b0/79/f0f1ca286b78f6f33c521a36b5cbd5bd697c0d66217d8856f443aeb9dd77/versioneer-0.29-py3-none-any.whl', 'destination': '/opt/hapt-build-deps/versioneer-0.29-py3-none-any.whl'}

PANDAS_HTTP_ASSET = {'name': 'pandas-50748-http.zip', 'sha256': 'f53a7e14749262558ce639af27561cbd2bace00b9fd4966a6c35b2b27b4f8d7d', 'destination': '/opt/hapt-pandas-http.zip', 'source': 'public HTTP snapshots; raw test data pinned to task base commit'}

# These pinned images combine importlib-metadata 0.20 with setuptools 75,
# whose metadata finder API is incompatible. Keep the old test dependency;
# use the last pre-PEP660 build frontend verified against these source trees.
LEGACY_MYPY_TASKS = frozenset(('9380', '9573', '10053', '10174', '10284', '10382', '10415', '10766'))
SETUPTOOLS_ASSET = {'name': 'setuptools-58.2.0-py3-none-any.whl', 'sha256': '2551203ae6955b9876741a26ab3e767bb3242dafe86a32a749ea0d78b6792f11', 'url': 'https://files.pythonhosted.org/packages/41/f4/a7ca4859317232b1efb64a826b8d2d7299bb77fb60bdb08e2bd1d61cf80d/setuptools-58.2.0-py3-none-any.whl', 'destination': '/opt/hapt-build-deps/setuptools-58.2.0-py3-none-any.whl'}
MONAI_MRI_ASSET = {'name': 'Prostate_T2W_AX_1.nii', 'sha256': 'a14231f539c0f365a5f83f2a046969a9b9870e56ffd126fd8e7242364d25938a', 'url': 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/Prostate_T2W_AX_1.nii', 'destination': '/opt/hapt-test-data/Prostate_T2W_AX_1.nii'}
MONAI_WSI_TASKS = frozenset(('3979', '4943'))
MONAI_WSI_ASSET = {'name': 'CMU-1.tiff', 'sha256': '73a7e89bc15576587c3d68e55d9bf92f09690280166240b48ff4b48230b13bcd', 'url': 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/CMU-1.tiff', 'destination': '/opt/hapt-test-data/CMU-1.tiff'}
HYDRA_ISORT_ASSET = {
    'name': 'isort-4.3.21-py3-none-any.whl',
    'sha256': 'e6e01069745d287cfc47cf594c24a8a61834670cd1da9a5e83471d03210b8fc1',
    'destination': '/opt/hapt-build-deps/isort-4.3.21-py3-none-any.whl',
    'source_commit': 'c54b3dd4620f9b3e7ac127c583ecb029fb90f1b7',
    'source_url': 'https://codeload.github.com/PyCQA/isort/tar.gz/c54b3dd4620f9b3e7ac127c583ecb029fb90f1b7',
    'source_sha256': '9b6c38a45b224cfd751878723d650078c7b4ef729c27f8cad87c7e577ee5c672',
    'build_backend': 'poetry==1.1.15 / poetry-core==1.0.8',
}

# Installed only inside disposable task containers, never into the pinned image.
BUILD_LIMIT_SCRIPT = r'''
import os
from pathlib import Path
p=Path('/opt/miniconda3/envs/testbed/bin/ninja')
if p.exists():
    original=p.with_name('ninja.hapt-original')
    if not original.exists():
        p.rename(original)
        p.write_text('#!/bin/sh\nexec '+str(original)+' "$@" -j"${HAPT_SWEGYM_BUILD_JOBS:-4}"\n')
        p.chmod(0o755)
print('SWE-Gym compiler jobs: '+os.environ.get('HAPT_SWEGYM_BUILD_JOBS','4'))
'''

def is_task_image(image: str) -> bool:
    return "/sweb.eval." in image or image.startswith("sweb.eval.")


def assets_for_image(image: str) -> list[dict]:
    if is_task_image(image) and 'facebookresearch_s_hydra-616:' in image:
        return [dict(HYDRA_ISORT_ASSET)]
    if is_task_image(image) and any('python_s_mypy-' + task + ':' in image for task in LEGACY_MYPY_TASKS):
        return [dict(SETUPTOOLS_ASSET)]
    if is_task_image(image) and "project-monai" in image.lower():
        assets = [dict(MONAI_ASSET)]
        if any('monai-' + task + ':' in image.lower() for task in MONAI_WSI_TASKS):
            assets.append(dict(MONAI_WSI_ASSET))
        if "monai-6034:" in image.lower():
            assets.append(dict(MONAI_MRI_ASSET))
        return assets
    if is_task_image(image) and "dask_s_dask-" in image:
        return [dict(VERSIONEER_ASSET)]
    if is_task_image(image) and "pandas-dev_s_pandas-50748:" in image:
        return [dict(PANDAS_HTTP_ASSET)]
    return []


def runtime_identity(image: str) -> dict:
    identity = {"epoch": RUNTIME_EPOCH, "python_prefix": TESTBED,
                "assets": assets_for_image(image)}
    if 'modin-project_s_modin-' in image:
        identity['modin_runtime'] = {'cpus': int(os.environ.get('HAPT_SWEGYM_VERIFIER_CPUS', '4')),
                                     'object_store_bytes': 2147483648, 'ray_address': 'local'}
    if compat_plugin(image):
        identity["pytest_compat_sha256"] = hashlib.sha256(
            (Path(__file__).parent / "compat" / (compat_plugin(image) + ".py")).read_bytes()).hexdigest()
    if is_task_image(image) and "pandas-dev_s_pandas-" in image:
        identity["moto_server_compat_sha256"] = hashlib.sha256(
            (Path(__file__).parent / "compat/moto_server").read_bytes()).hexdigest()
    return identity


def compat_plugin(image: str) -> str:
    if not is_task_image(image):
        return ""
    if "project-monai_s_monai-6034:" in image.lower():
        return "hapt_monai_http_cache"
    if "python_s_mypy-" in image:
        return "hapt_mypy_pytest"
    if "getmoto_s_moto-4799:" in image:
        return "hapt_moto_offline"
    if "pandas-dev_s_pandas-50748:" in image:
        return "hapt_pandas_http_cache"
    return ""


def task_python_path(image: str, existing: str = "") -> str:
    paths = ["/opt/hapt-swegym-compat"] if compat_plugin(image) else []
    if is_task_image(image) and "dask_s_dask-" in image:
        paths.append(VERSIONEER_ASSET["destination"])
    return ":".join([*paths, *([existing] if existing else [])])


def tool_path(image: str, node_bin: str | Path | None = None) -> str:
    paths = [f"{TESTBED}/bin", "/opt/miniconda3/bin"] if is_task_image(image) else []
    if is_task_image(image) and "pandas-dev_s_pandas-" in image:
        paths.insert(0, "/opt/hapt-swegym-bin")
    if node_bin:
        paths.append(str(node_bin))
    return ":".join([*paths, SYSTEM_PATH])


@lru_cache(maxsize=64)
def _check_asset(path: str, size: int, mtime_ns: int, ctime_ns: int, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"SWE-Gym public test asset checksum mismatch: {path}")


def docker_runtime_args(image: str, project_root: Path, node_bin=None, *, evaluator=False) -> list[str]:
    if not is_task_image(image):
        return []
    args = ["--env", "PATH=" + tool_path(image, node_bin),
            "--env", "CONDA_PREFIX=" + TESTBED, "--env", "CONDA_DEFAULT_ENV=testbed",
            # Images already contain pinned build dependencies. PIP's negative
            # boolean option uses 0 to disable isolated build environments.
            "--env", "PIP_NO_BUILD_ISOLATION=0", "--env", "PIP_NO_INDEX=1"]
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        args += ['--env', key + '=1']
    jobs = int(os.environ.get('HAPT_SWEGYM_BUILD_JOBS', '4'))
    if jobs < 1:
        raise ValueError('SWE-Gym build jobs must be positive')
    args += ['--env', f'HAPT_SWEGYM_BUILD_JOBS={jobs}', '--env', f'MAX_JOBS={jobs}',
             '--env', f'CMAKE_BUILD_PARALLEL_LEVEL={jobs}']
    if 'modin-project_s_modin-' in image:
        # Modin's CpuCount defaults to host os.cpu_count(), ignoring Docker's
        # CPU quota; this otherwise prestarts 180 Ray workers per task here.
        args += ['--env', 'MODIN_CPUS=' + os.environ.get('HAPT_SWEGYM_VERIFIER_CPUS', '4'),
                 '--env', 'MODIN_MEMORY=2147483648', '--env', 'RAY_ADDRESS=local']
    for key in ('PREPARE', 'VERIFIER', 'CLEANUP'):
        name = 'HAPT_SWEGYM_' + key + '_TIMEOUT_SECONDS'
        if name in os.environ:
            args += ['--env', name + '=' + os.environ[name]]
    if evaluator:
        cpus = int(os.environ.get('HAPT_SWEGYM_VERIFIER_CPUS', '4'))
        memory = int(os.environ.get('HAPT_SWEGYM_VERIFIER_MEMORY_GB', '16'))
        if min(cpus, memory) < 1:
            raise ValueError('SWE-Gym verifier resources must be positive')
        args += ['--cpus', str(cpus), '--memory', str(memory) + 'g']
    if compat_plugin(image):
        source_root = Path(os.environ.get("HAPT_JOINT_CODE_SNAPSHOT") or project_root)
        source = source_root / "third_party/Dressage/dressage/recipes/swegym/compat" / (compat_plugin(image) + ".py")
        if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != runtime_identity(image)["pytest_compat_sha256"]:
            raise ValueError("SWE-Gym pytest compatibility module differs from controller source")
        args += ["--mount", f"type=bind,src={source.resolve()},dst=/opt/hapt-swegym-compat/{compat_plugin(image)}.py,readonly",
                 "--env", "PYTEST_PLUGINS=" + compat_plugin(image)]
    if task_python_path(image):
        args += ["--env", "HAPT_SWEGYM_TASK_PYTHONPATH=" + task_python_path(image),
                 "--env", "PYTHONPATH=" + task_python_path(image)]
    if is_task_image(image) and "pandas-dev_s_pandas-" in image:
        source_root = Path(os.environ.get("HAPT_JOINT_CODE_SNAPSHOT") or project_root)
        source = source_root / "third_party/Dressage/dressage/recipes/swegym/compat/moto_server"
        if hashlib.sha256(source.read_bytes()).hexdigest() != runtime_identity(image)["moto_server_compat_sha256"]:
            raise ValueError("SWE-Gym Moto CLI compatibility differs from controller source")
        args += ["--mount", f"type=bind,src={source.resolve()},dst=/opt/hapt-swegym-bin/moto_server,readonly"]
    for asset in assets_for_image(image):
        path = project_root / "runtime/swegym-assets" / asset["sha256"] / asset["name"]
        if not path.is_file():
            raise FileNotFoundError(f"SWE-Gym public test asset missing: {path}; run scripts/prepare_swegym_test_assets.py")
        stat = path.stat()
        _check_asset(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, asset["sha256"])
        args += ["--mount", f"type=bind,src={path.resolve()},dst={asset['destination']},readonly"]
    return args


# Use the same non-login bash path as DSH's model-facing bash executor.
TOOL_ENV_PROBE = (
    "import json,sys,shutil,pytest; "
    f"assert sys.prefix == {TESTBED!r}, (sys.executable,sys.prefix); "
    f"assert all((shutil.which(x) or '').startswith({(TESTBED + '/bin/')!r}) for x in ('python','pip','pytest')); "
    "print(json.dumps({'python':sys.executable,'prefix':sys.prefix,"
    "'pip':shutil.which('pip'),'pytest':shutil.which('pytest')}))"
)

"""DressageDataSource — text-first data source extending slime's base.

Supports plain string prompts in JSONL without requiring a multimodal processor.
All fields beyond prompt and label are stored in Sample.metadata for downstream dispatch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CROSS_HARNESS_MODE = "mixed_baseline"
_CROSS_HARNESS_GROUP_LAYOUTS = ("distinct", "shared")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

try:
    from slime.rollout.data_source import RolloutDataSourceWithBuffer
    from slime.utils.types import Sample
except ImportError:
    logger.warning(
        "slime not importable — DressageDataSource will use standalone types. "
        "For production use, ensure slime is installed."
    )

    from dataclasses import dataclass, field
    from enum import Enum

    @dataclass
    class Sample:
        group_index: int | None = None
        index: int | None = None
        prompt: str | list[dict[str, str]] = ""
        tokens: list[int] = field(default_factory=list)
        response: str = ""
        response_length: int = 0
        label: str | None = None
        reward: float | dict[str, Any] | None = None
        loss_mask: list[int] | None = None
        rollout_log_probs: list[float] | None = None
        remove_sample: bool = False
        metadata: dict = field(default_factory=dict)
        generate_function_path: str | None = None

        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"
            FAILED = "failed"

        status: Status = Status.PENDING

    class RolloutDataSourceWithBuffer:
        def __init__(self, args):
            self.args = args
            self.buffer = []
            self.metadata = {}

        def get_samples(self, num_samples):
            raise NotImplementedError

        def add_samples(self, samples):
            if not samples:
                return
            for group in samples:
                self.buffer.append(group)

        def save(self, rollout_id):
            pass

        def load(self, rollout_id=None):
            pass

        def __len__(self):
            return 0


def _read_jsonl(path: str):
    """Read a JSONL file and yield dicts."""
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSON decode error at line %d: %s", line_num, e)
                continue


class DressageDataSource(RolloutDataSourceWithBuffer):
    """Text-first data source for Dressage.

    Extends slime's data source with:
    - Plain string prompts work without a multimodal processor.
    - Per-sample metadata passthrough: agent_mode, env_type, tool_set, reward_fn
      are preserved in Sample.metadata for downstream dispatch.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.buffer: list[list[Sample]] = []
        self.metadata: dict = {}
        # Joint Policy--Harness training may change only the mutable Harness
        # overlay at optimizer-step boundaries.  Keep the binding separate
        # from the frozen dataset so the dataset fingerprint remains stable.
        self._active_harness: dict[str, str] | None = None
        from dressage.rollout.harness_mix import validate_mode
        self._harness_mix_mode = validate_mode(getattr(args, "harness_mix", "active_only"),
                                              getattr(args, "n_samples_per_prompt", 1))
        self._harness_assignment = None
        self._h0_retry_on_zero = os.environ.get('HAPT_H0_RETRY_ON_ZERO') == '1'
        if self._h0_retry_on_zero and (self._harness_mix_mode != 'active_only'
                                      or getattr(args, 'n_samples_per_prompt', 1) != 8):
            raise ValueError('H0 retry requires active_only G8')

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0

        prompt_key = getattr(args, "input_key", None) or "prompt"
        label_key = getattr(args, "label_key", None)
        metadata_key = getattr(args, "metadata_key", "metadata")
        multimodal_keys = getattr(args, "multimodal_keys", None)
        cross_harness_mode = os.environ.get("DRESSAGE_CROSS_HARNESS_MODE", "").strip()
        group_layout = os.environ.get(
            "DRESSAGE_CROSS_HARNESS_GROUP_LAYOUT", "distinct"
        ).strip()
        if group_layout not in _CROSS_HARNESS_GROUP_LAYOUTS:
            raise ValueError(
                "DRESSAGE_CROSS_HARNESS_GROUP_LAYOUT must be one of "
                f"{_CROSS_HARNESS_GROUP_LAYOUTS}; got {group_layout!r}"
            )
        if group_layout == "shared" and cross_harness_mode != _CROSS_HARNESS_MODE:
            raise ValueError(
                "shared cross-Harness grouping requires "
                f"DRESSAGE_CROSS_HARNESS_MODE={_CROSS_HARNESS_MODE}"
            )
        self._cross_harness_group_layout = group_layout
        if self._harness_mix_mode != "active_only" and cross_harness_mode:
            raise ValueError("dynamic Harness mixture cannot use legacy cross-Harness reward/grouping mode")
        self._cross_harness_expected_harnesses = tuple(
            value.strip()
            for value in os.environ.get(
                "DRESSAGE_CROSS_HARNESS_EXPECTED_HARNESSES",
                "minimal,standard-core-native",
            ).split(",")
            if value.strip()
        )

        prompt_data = getattr(args, "prompt_data", None) or os.environ.get(
            "PROMPT_DATA"
        )
        self._use_text_first = multimodal_keys is None

        # A MOPD config may own a weighted collection of datasets.  Keep the
        # legacy single --prompt-data path unchanged when no datasets are
        # declared in that config.
        self._mixture_samples: list[list[Sample]] | None = None
        self._mixture_weights: list[float] = []
        self._mixture_current: list[float] = []
        self._mixture_offsets: list[int] = []
        self._mixture_epochs: list[int] = []
        self._mixture_base_samples: list[list[Sample]] | None = None
        self._base_samples: list[Sample] | None = None
        mopd_config_path = getattr(args, "mopd_teacher_config", None) or os.environ.get(
            "DRESSAGE_MOPD_TEACHER_CONFIG"
        )
        if self._use_text_first and mopd_config_path:
            from dressage.rollout.mopd import load_mopd_config

            mopd_config = load_mopd_config(str(mopd_config_path))
            if mopd_config.datasets:
                self._mixture_samples = []
                for dataset in mopd_config.datasets:
                    dataset_samples = self._load_text_first(
                        dataset.path,
                        prompt_key,
                        label_key,
                        metadata_key,
                        metadata_overrides=dataset.metadata,
                        generate_function_path=dataset.generate_function_path,
                    )
                    if not dataset_samples:
                        raise ValueError(
                            f"MOPD dataset {dataset.name!r} has no valid samples: {dataset.path}"
                        )
                    self._mixture_samples.append(dataset_samples)
                    self._mixture_weights.append(dataset.weight)
                self._mixture_current = [0.0] * len(self._mixture_samples)
                self._mixture_offsets = [0] * len(self._mixture_samples)
                self._mixture_epochs = [0] * len(self._mixture_samples)
                self._mixture_base_samples = copy.deepcopy(self._mixture_samples)
                self._samples = None
                logger.info(
                    "DressageDataSource: loaded MOPD mixture datasets=%s weights=%s",
                    [dataset.name for dataset in mopd_config.datasets],
                    self._mixture_weights,
                )

        if self._mixture_samples is not None:
            if self._cross_harness_group_layout == "shared":
                raise ValueError(
                    "shared cross-Harness grouping does not support MOPD mixture datasets"
                )
        elif self._use_text_first and prompt_data:
            self._samples = self._load_text_first(
                prompt_data, prompt_key, label_key, metadata_key
            )
            self._base_samples = copy.deepcopy(self._samples)
            if self._cross_harness_group_layout == "shared":
                self._validate_shared_cross_harness_rows()
            if getattr(args, "rollout_shuffle", False):
                if self._cross_harness_group_layout == "shared":
                    raise ValueError(
                        "shared cross-Harness grouping requires rollout_shuffle=False "
                        "so adjacent paired rows remain atomic"
                    )
                self._shuffle(self.epoch_id)
        elif prompt_data:
            try:
                super().__init__(args)
                self._samples = None
            except Exception:
                logger.warning("Failed slime parent init, falling back to text-first")
                self._samples = self._load_text_first(
                    prompt_data, prompt_key, label_key, metadata_key
                )
                self._base_samples = copy.deepcopy(self._samples)
        else:
            self._samples = []

    @staticmethod
    def _sample_task_and_harness(sample: Sample) -> tuple[str, str]:
        metadata = getattr(sample, "metadata", None) or {}
        task_id = str(metadata.get("task_id") or metadata.get("instance_id") or "")
        harness_id = str(metadata.get("harness_id") or "")
        return task_id, harness_id

    def _validate_shared_cross_harness_rows(self) -> None:
        if self._samples is None:
            raise ValueError("shared cross-Harness grouping requires text-first prompt data")
        if len(self._cross_harness_expected_harnesses) != 2:
            raise ValueError(
                "shared cross-Harness grouping currently requires exactly two expected Harnesses"
            )
        if not self._samples or len(self._samples) % 2:
            raise ValueError(
                "shared cross-Harness prompt data must contain a non-empty even number of rows"
            )
        expected = list(self._cross_harness_expected_harnesses)
        for offset in range(0, len(self._samples), 2):
            pair = self._samples[offset : offset + 2]
            identities = [self._sample_task_and_harness(sample) for sample in pair]
            task_ids = {task_id for task_id, _ in identities}
            harnesses = [harness_id for _, harness_id in identities]
            if "" in task_ids or len(task_ids) != 1 or harnesses != expected:
                raise ValueError(
                    "shared cross-Harness rows must be adjacent, share one task id, and "
                    f"follow Harness order {expected}; rows {offset}-{offset + 1} got {identities}"
                )

    def _load_text_first(
        self,
        path: str,
        prompt_key: str,
        label_key: str | None,
        metadata_key: str,
        metadata_overrides: dict[str, Any] | None = None,
        generate_function_path: str | None = None,
    ) -> list[Sample]:
        """Load JSONL as plain-text samples with metadata passthrough."""
        samples = []
        for data in _read_jsonl(path):
            prompt = data.get(prompt_key, "")
            label = data.get(label_key) if label_key else data.get("label")

            meta = dict(data.get(metadata_key) or {})
            for key in (
                "agent_mode",
                "env_type",
                "env_args",
                "tool_set",
                "agent_id",
                "reward_fn",
            ):
                if key in data and key not in meta:
                    meta[key] = data[key]

            for key, val in data.items():
                if (
                    key not in (prompt_key, "label", metadata_key, label_key)
                    and key not in meta
                ):
                    meta[key] = val

            # Dataset-level routing is authoritative. In particular, it must
            # not collide with task-local fields such as ALFWorld's task_type.
            if metadata_overrides:
                meta.update(copy.deepcopy(metadata_overrides))

            # Slime natively dispatches on this per-sample field. Dataset
            # routing is resolved once here; no MOPD generate wrapper is needed.
            sample_generate_path = (
                generate_function_path
                or data.get("generate_function_path")
                or meta.get("generate_function_path")
            )
            samples.append(
                Sample(
                    prompt=prompt,
                    label=str(label) if label is not None else None,
                    metadata=meta,
                    generate_function_path=(
                        str(sample_generate_path).strip()
                        if sample_generate_path
                        else None
                    ),
                )
            )

        logger.info("DressageDataSource: loaded %d samples from %s", len(samples), path)
        return samples

    def _shuffle(self, epoch_id: int) -> None:
        if self._samples is None:
            return
        seed = getattr(self.args, "rollout_seed", 42)
        if self._base_samples is not None:
            self._samples = copy.deepcopy(self._base_samples)
        random.Random(seed + epoch_id).shuffle(self._samples)
        self.epoch_id = epoch_id

    def _next_mixture_sample(self) -> Sample:
        """Select a source by smooth weighted round-robin and return one row."""
        assert self._mixture_samples is not None
        total_weight = sum(self._mixture_weights)
        for index, weight in enumerate(self._mixture_weights):
            self._mixture_current[index] += weight
        dataset_index = max(
            range(len(self._mixture_current)),
            key=lambda index: (self._mixture_current[index], -index),
        )
        self._mixture_current[dataset_index] -= total_weight

        dataset = self._mixture_samples[dataset_index]
        offset = self._mixture_offsets[dataset_index]
        if offset >= len(dataset):
            self._mixture_epochs[dataset_index] += 1
            if self._mixture_base_samples is not None:
                dataset = copy.deepcopy(self._mixture_base_samples[dataset_index])
                self._mixture_samples[dataset_index] = dataset
            if getattr(self.args, "rollout_shuffle", False):
                seed = int(getattr(self.args, "rollout_seed", 42))
                rng = random.Random(
                    seed + self._mixture_epochs[dataset_index] * 1009 + dataset_index
                )
                rng.shuffle(dataset)
            offset = 0
        sample = dataset[offset]
        self._mixture_offsets[dataset_index] = offset + 1
        return sample

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return num_samples prompt groups. Each group has n_samples_per_prompt clones."""
        buffer_samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(buffer_samples)

        if num_samples == 0:
            return buffer_samples

        if self._mixture_samples is not None:
            base_samples = [self._next_mixture_sample() for _ in range(num_samples)]
        elif self._samples is None:
            return buffer_samples + super().get_samples(num_samples)
        elif self._cross_harness_group_layout == "shared":
            base_samples = []
            for _ in range(num_samples):
                if self.sample_offset >= len(self._samples):
                    if _env_flag("DRESSAGE_ALLOW_PARTIAL_FINAL_BATCH"):
                        break
                    self.epoch_id += 1
                    self.sample_offset = 0
                if self.sample_offset + 1 >= len(self._samples):
                    break
                base_samples.append(self._samples[self.sample_offset : self.sample_offset + 2])
                self.sample_offset += 2
        else:
            base_samples = []
            for _ in range(num_samples):
                if self.sample_offset >= len(self._samples):
                    if _env_flag("DRESSAGE_ALLOW_PARTIAL_FINAL_BATCH"):
                        break
                    self.epoch_id += 1
                    if getattr(self.args, "rollout_shuffle", False):
                        self._shuffle(self.epoch_id)
                    self.sample_offset = 0

                if self.sample_offset >= len(self._samples):
                    break

                base_samples.append(self._samples[self.sample_offset])
                self.sample_offset += 1

        n_per = getattr(self.args, "n_samples_per_prompt", 1)
        clones_per_row = n_per
        if self._cross_harness_group_layout == "shared":
            harness_count = len(self._cross_harness_expected_harnesses)
            if n_per < harness_count or n_per % harness_count:
                raise ValueError(
                    "shared cross-Harness grouping interprets n_samples_per_prompt "
                    "as the total atomic group size; it must be evenly divisible by "
                    f"the {harness_count} expected Harnesses, got {n_per}"
                )
            clones_per_row = n_per // harness_count
        groups = []

        for base_sample_or_pair in base_samples:

            group = []
            logical_rows = (
                base_sample_or_pair
                if isinstance(base_sample_or_pair, list)
                else [base_sample_or_pair]
            )
            for base_sample in logical_rows:
                for clone_index in range(clones_per_row):
                    s = copy.deepcopy(base_sample)
                    self._apply_active_harness(s, clone_index)
                    self._apply_deterministic_clone_seed(
                        s, clone_index,
                        default_seed=int(getattr(self.args, "rollout_seed", 42) or 0),
                    )
                    s.group_index = self.sample_group_index
                    s.index = self.sample_index
                    self.sample_index += 1
                    group.append(s)

            self.sample_group_index += 1
            groups.append(group)

        return buffer_samples + groups

    @staticmethod
    def _apply_deterministic_clone_seed(
        sample: Sample, clone_index: int, *, default_seed: int = 42
    ) -> None:
        """Give each G-sample clone a stable DSH sampling seed.

        Any explicitly seeded blackbox row needs separate clone streams,
        including legacy rows without ``metadata.sampling_seed_mode``.
        Retries reuse the already-cloned group and therefore keep the exact
        same task/clone seed.
        """

        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            return
        backend = metadata.get("backend_options")
        proxy = backend.get("proxy") if isinstance(backend, dict) else None
        if not isinstance(proxy, dict) or proxy.get("sampling_seed_base") is None:
            if metadata.get("sampling_seed_mode") == "task-clone-v1":
                raise ValueError(
                    "task-clone-v1 requires metadata.backend_options.proxy.sampling_seed_base"
                )
            if metadata.get("blackbox_type") not in {
                "deepseek_harness", "opencode", "claude_code", "codex", "openclaw"
            }:
                return  # Do not invent options for arbitrary external agents.
            if backend is None:
                backend = metadata["backend_options"] = {}
            if proxy is None:
                proxy = backend["proxy"] = {}
            identity = metadata.get("task_id") or metadata.get("instance_id") or sample.prompt
            raw = json.dumps(["dressage-clone-v2", default_seed, identity], sort_keys=True)
            proxy["sampling_seed_base"] = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:4], "big")
        base = proxy["sampling_seed_base"]
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise ValueError("sampling_seed_base must be a nonnegative integer")
        proxy["sampling_seed_base"] = base + int(clone_index)
        if proxy.get("sampling_mode") is None:
            proxy["sampling_mode"] = "force"
        metadata["sampling_clone_index"] = int(clone_index)
        metadata["sampling_seed_derivation"] = "clone-offset+turn-stride-1000003/v2"

    def set_active_harness(self, binding: dict[str, str]) -> dict[str, str]:
        """Bind an immutable Harness overlay to all future sample clones.

        Switching while retry groups are buffered could mix two Harnesses in
        one logical optimizer step, so the operation intentionally fails
        closed until the buffer is empty.
        """
        if self._harness_mix_mode != "active_only":
            raise ValueError("mixed Harness mode requires atomic set_harness_assignment")
        required = {"version", "sha256", "path"}
        if not isinstance(binding, dict) or set(binding) != required:
            raise ValueError(
                "active Harness binding must contain exactly version, sha256, path"
            )
        normalized = {key: str(binding[key]).strip() for key in required}
        if any(not value for value in normalized.values()):
            raise ValueError("active Harness binding values must be non-empty")
        path = Path(normalized["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"active Harness overlay is missing: {path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != normalized["sha256"]:
            raise ValueError(
                "active Harness SHA256 mismatch: "
                f"expected={normalized['sha256']} actual={actual_sha256}"
            )
        if self.buffer:
            raise RuntimeError(
                "cannot switch active Harness while retry sample groups are buffered"
            )
        normalized["path"] = str(path)
        self._active_harness = normalized
        self.metadata["active_harness"] = copy.deepcopy(normalized)
        return copy.deepcopy(normalized)

    def get_active_harness(self) -> dict[str, str] | None:
        return copy.deepcopy(self._active_harness)

    def set_harness_assignment(self, plan):
        from dressage.rollout.harness_mix import normalize_assignment
        normalized = normalize_assignment(plan)
        if (normalized["mode"] != self._harness_mix_mode or
                normalized["group_size"] != getattr(self.args, "n_samples_per_prompt", 1)):
            raise ValueError("Harness assignment differs from configured sampling mode/G")
        if self.buffer:
            raise RuntimeError("cannot switch Harness assignment while retry groups are buffered")
        if self._harness_assignment and "h0" in normalized["bindings"]:
            if self._harness_assignment["bindings"]["h0"] != normalized["bindings"]["h0"]:
                raise ValueError("baseline H0 binding must remain immutable")
        self._harness_assignment = normalized
        from dressage.rollout.harness_mix import primary_binding
        self._active_harness = copy.deepcopy(primary_binding(normalized))
        self.metadata["harness_assignment"] = copy.deepcopy(normalized)
        self.metadata["active_harness"] = copy.deepcopy(self._active_harness)
        return copy.deepcopy(normalized)

    def get_harness_assignment(self):
        return copy.deepcopy(self._harness_assignment)

    def _apply_active_harness(self, sample: Sample, clone_index=0) -> None:
        binding = self._active_harness
        role = "active"
        metadata = copy.deepcopy(getattr(sample, "metadata", None) or {})
        if self._harness_mix_mode != "active_only":
            if self._harness_assignment is None:
                raise RuntimeError("mixed rollout requested before atomic Harness assignment")
            from dressage.rollout.harness_mix import clone_role
            role = clone_role(self._harness_assignment, metadata.get("task_id"), clone_index)
            binding = self._harness_assignment["bindings"][role]
            if self._harness_mix_mode == "router_topk":
                metadata["router_stage_version"] = self._harness_assignment["stage_version"]
                metadata["route_plan_version"] = self._harness_assignment["route_plan_version"]
                if self._harness_assignment['schema_version'] in (4, 6, 7):
                    metadata['h0_first_rescue'] = True
        if binding is None:
            return
        if self._harness_mix_mode == 'fixed_backend_equal':
            from dressage.rollout.harness_mix import apply_backend_binding
            sample.metadata = apply_backend_binding(metadata, self._harness_assignment, role)
            return
        backend = copy.deepcopy(metadata.get("backend_options") or {})
        runtime = copy.deepcopy(backend.get("runtime") or {})
        runtime["harness_overlay"] = binding["path"]
        backend["runtime"] = runtime
        # backend_options.harness_id selects the executable DSH composition
        # and is schema-constrained (for AppWorld it must stay
        # task-environment-native).  The evolving version is experiment
        # metadata, not a new backend kind.
        metadata["backend_options"] = backend
        metadata["harness_id"] = binding["version"]
        metadata["harness_version"] = binding["version"]
        metadata["harness_sha256"] = binding["sha256"]
        metadata["harness_role"] = role
        metadata["harness_mix"] = self._harness_mix_mode
        if self._h0_retry_on_zero:
            if binding['version'] != 'h0':
                raise ValueError('H0 retry requires the immutable H0 binding')
            metadata['h0_retry_on_zero'] = True
        sample.metadata = metadata

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if not self.buffer or num_samples == 0:
            return []
        num_to_pop = min(len(self.buffer), num_samples)
        samples = self.buffer[:num_to_pop]
        del self.buffer[:num_to_pop]
        return samples

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Add sample groups back to buffer (for retry on failure)."""
        if not samples:
            return
        for group in samples:
            self.buffer.append(group)

    def save(self, rollout_id: int) -> None:
        if not getattr(self.args, "rollout_global_dataset", True):
            return
        save_root = getattr(self.args, "save", None)
        if not save_root:
            logger.warning(
                "DressageDataSource.save(%s) skipped because args.save is unset",
                rollout_id,
            )
            return
        path = self._checkpoint_path(str(save_root), rollout_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 3,
            "dataset_fingerprint": self._dataset_fingerprint(),
            "cross_harness_group_layout": self._cross_harness_group_layout,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "sample_offset": self.sample_offset,
            "mixture_current": self._mixture_current,
            "mixture_offsets": self._mixture_offsets,
            "mixture_epochs": self._mixture_epochs,
            "buffer": self.buffer,
            "metadata": self.metadata,
            "active_harness": self._active_harness,
            "harness_mix_mode": self._harness_mix_mode,
            "harness_assignment": self._harness_assignment,
            **({'h0_retry_on_zero': True} if self._h0_retry_on_zero else {}),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        logger.info("saved Dressage data-source state to %s", path)

    def load(self, rollout_id: int | None = None) -> None:
        if not getattr(self.args, "rollout_global_dataset", True):
            return
        load_root = getattr(self.args, "load", None)
        if not load_root:
            return
        path = self._checkpoint_path(str(load_root), rollout_id)
        if not path.is_file():
            logger.info("Dressage data-source checkpoint does not exist: %s", path)
            return
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict) or state.get("schema_version") not in {1, 2, 3}:
            raise ValueError(f"unsupported Dressage data-source checkpoint: {path}")
        if state.get("harness_mix_mode", "active_only") != self._harness_mix_mode:
            raise ValueError("Dressage data-source checkpoint Harness mixture changed; explicit new run required")
        # Evaluation restores weights and the attested dataset cursor, but
        # deliberately disables training-only second-wave sampling. Keep
        # rejecting every recipe change when resuming training, and never
        # allow an evaluation to turn retry sampling on.
        evaluation_disables_retry = (
            os.environ.get('HAPT_JOINT_EVAL_POLICY') in {'best', 'final', 'checkpoint-eval'}
            and self._harness_mix_mode == 'active_only'
            and bool(state.get('h0_retry_on_zero')) and not self._h0_retry_on_zero)
        if bool(state.get('h0_retry_on_zero')) != self._h0_retry_on_zero and not evaluation_disables_retry:
            raise ValueError('Dressage data-source checkpoint H0 retry mode changed')
        from dressage.rollout.harness_mix import normalize_assignment, primary_binding
        restored_plan = state.get("harness_assignment")
        if restored_plan is not None:
            restored_plan = normalize_assignment(restored_plan)
            if (restored_plan["mode"] != self._harness_mix_mode or
                    restored_plan["group_size"] != getattr(self.args, "n_samples_per_prompt", 1) or
                    primary_binding(restored_plan) != state.get("active_harness")):
                raise ValueError("checkpoint Harness assignment is inconsistent")
        elif self._harness_mix_mode != "active_only":
            raise ValueError("mixed sampler checkpoint is missing its Harness assignment")
        expected_fingerprint = self._dataset_fingerprint()
        if state.get("dataset_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Dressage data-source checkpoint dataset drift: "
                f"checkpoint={state.get('dataset_fingerprint')} "
                f"current={expected_fingerprint}"
            )
        if state.get("cross_harness_group_layout") != self._cross_harness_group_layout:
            raise ValueError(
                "Dressage data-source checkpoint grouping mode changed: "
                f"checkpoint={state.get('cross_harness_group_layout')!r} "
                f"current={self._cross_harness_group_layout!r}"
            )

        self.epoch_id = int(state.get("epoch_id", 0))
        self.sample_group_index = int(state.get("sample_group_index", 0))
        self.sample_index = int(state.get("sample_index", 0))
        self.sample_offset = int(state.get("sample_offset", 0))
        self.metadata = dict(state.get("metadata") or {})
        self.buffer = list(state.get("buffer") or [])
        self._harness_assignment = restored_plan
        active_harness = state.get("active_harness")
        if active_harness is not None:
            # The checkpoint was written only after validation, so assigning
            # directly avoids the intentional non-empty-buffer switch guard.
            required = {"version", "sha256", "path"}
            if not isinstance(active_harness, dict) or set(active_harness) != required:
                raise ValueError("invalid active Harness binding in checkpoint")
            harness_path = Path(str(active_harness["path"])).resolve()
            if not harness_path.is_file() or hashlib.sha256(harness_path.read_bytes()).hexdigest() != str(
                active_harness["sha256"]
            ):
                raise ValueError("active Harness checkpoint binding is missing or changed")
            self._active_harness = {
                "version": str(active_harness["version"]),
                "sha256": str(active_harness["sha256"]),
                "path": str(harness_path),
            }

        if self._mixture_samples is not None:
            current = list(state.get("mixture_current") or [])
            offsets = [int(value) for value in state.get("mixture_offsets") or []]
            epochs = [int(value) for value in state.get("mixture_epochs") or []]
            expected = len(self._mixture_samples)
            if not all(len(values) == expected for values in (current, offsets, epochs)):
                raise ValueError("Dressage mixture checkpoint shape changed")
            self._mixture_current = [float(value) for value in current]
            self._mixture_offsets = offsets
            self._mixture_epochs = epochs
            if self._mixture_base_samples is not None:
                self._mixture_samples = copy.deepcopy(self._mixture_base_samples)
                if getattr(self.args, "rollout_shuffle", False):
                    seed = int(getattr(self.args, "rollout_seed", 42))
                    for index, epoch in enumerate(self._mixture_epochs):
                        if epoch <= 0:
                            continue
                        random.Random(seed + epoch * 1009 + index).shuffle(
                            self._mixture_samples[index]
                        )
            for offset, dataset in zip(self._mixture_offsets, self._mixture_samples):
                if offset < 0 or offset > len(dataset):
                    raise ValueError("Dressage mixture checkpoint offset is out of range")
        elif self._samples is not None:
            if getattr(self.args, "rollout_shuffle", False):
                self._shuffle(self.epoch_id)
            elif self._base_samples is not None:
                self._samples = copy.deepcopy(self._base_samples)
            if self.sample_offset < 0 or self.sample_offset > len(self._samples):
                raise ValueError("Dressage data-source checkpoint offset is out of range")
        logger.info("loaded Dressage data-source state from %s", path)

    @staticmethod
    def _checkpoint_path(root: str, rollout_id: int | None) -> Path:
        return Path(root) / "rollout" / f"dressage_data_source_state_{rollout_id}.pkl"

    def _dataset_fingerprint(self) -> str:
        def sample_payload(sample: Sample) -> dict[str, Any]:
            return {
                "prompt": getattr(sample, "prompt", None),
                "label": getattr(sample, "label", None),
                "metadata": getattr(sample, "metadata", None),
                "generate_function_path": getattr(
                    sample, "generate_function_path", None
                ),
            }

        if self._mixture_base_samples is not None:
            payload: Any = [
                [sample_payload(sample) for sample in dataset]
                for dataset in self._mixture_base_samples
            ]
        else:
            source = self._base_samples if self._base_samples is not None else []
            payload = [sample_payload(sample) for sample in source]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def __len__(self) -> int:
        if self._mixture_samples is not None:
            return sum(len(dataset) for dataset in self._mixture_samples)
        if self._samples is not None:
            divisor = 2 if self._cross_harness_group_layout == "shared" else 1
            return len(self._samples) // divisor
        return 0

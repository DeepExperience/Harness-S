"""Immutable, environment-neutral assignment of Harnesses to GRPO clones.

The role is provenance, NOT a reward, loss weight, or a second group identity.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ACTIVE_ONLY = "active_only"
H0_ACTIVE_EQUAL = "h0_active_equal"
POOL_EQUAL = "pool_equal"
POOL_WEIGHTED = "pool_weighted"
ROUTER_TOPK = "router_topk"
FIXED_BACKEND_EQUAL = 'fixed_backend_equal'
MIXED_MODES = {H0_ACTIVE_EQUAL, POOL_EQUAL, ROUTER_TOPK, POOL_WEIGHTED, FIXED_BACKEND_EQUAL}


def fixed_backend_assignment(h0, claude):
    compact = lambda h: {k:h[k] for k in ('version','path','sha256')}
    bindings = {'h0':normalize_binding(compact(h0)), 'active':normalize_binding(compact(claude))}
    descriptor = json.loads(Path(bindings['active']['path']).read_text())
    if (descriptor['blackbox_type'] != 'claude_code'
            or hashlib.sha256(Path(descriptor['options']['executable']).read_bytes()).hexdigest()
                != descriptor['executable_sha256']):
        raise ValueError('frozen Claude executable identity changed')
    return dict(schema_version=9, mode=FIXED_BACKEND_EQUAL, group_size=8,
                roles=['h0']*4+['active']*4, bindings=bindings, backend_templates={'active':descriptor})


def apply_backend_binding(metadata, plan, role):
    """Pure clone-time transformation: no file reads, no shared mutable options."""
    binding = plan['bindings'][role]
    backend = deepcopy(metadata['backend_options'])
    if role == 'active':
        template = plan['backend_templates'][role]
        backend = {**deepcopy(template['options']), 'task_environment':backend['task_environment'],
                   'proxy':backend['proxy']}
        metadata['blackbox_type'] = template['blackbox_type']
    else:
        backend['runtime']['harness_overlay'] = binding['path']
        metadata['blackbox_type'] = 'deepseek_harness'
    metadata.update(backend_options=backend, harness_id=binding['version'],
        harness_version=binding['version'], harness_sha256=binding['sha256'],
        harness_role=role, harness_mix=FIXED_BACKEND_EQUAL)
    return metadata


def validate_mode(mode, group_size):
    if mode not in {ACTIVE_ONLY, *MIXED_MODES}:
        raise ValueError(f"unsupported Harness mixture: {mode!r}")
    if mode in MIXED_MODES and group_size != 8:
        raise ValueError(f"{mode} v1 requires G=8")
    return mode


def normalize_binding(binding):
    required = {"version", "sha256", "path"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("Harness binding must contain exactly version, sha256, path")
    result = {key: str(binding[key]).strip() for key in required}
    if any(not value for value in result.values()):
        raise ValueError("Harness binding values must be non-empty")
    path = Path(result["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Harness overlay is missing: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != result["sha256"]:
        raise ValueError("Harness SHA256 mismatch")
    return {**result, "path": str(path)}


def assignment(mode, h0, active, group_size=8, *, pool=None):
    """Construct and validate both arms BEFORE publishing a new assignment."""
    validate_mode(mode, group_size)
    compact = lambda record: {key: record[key] for key in ("version", "sha256", "path")}
    if mode in {POOL_EQUAL, POOL_WEIGHTED}:
        if not isinstance(pool, dict) or len(pool.get("slots", [])) != 4:
            raise ValueError("pool_equal requires four frozen Harness slots")
        bindings = {f"slot{i}": normalize_binding(compact(slot["harness"])) for i, slot in enumerate(pool["slots"])}
        if [s.get("slot_id") for s in pool["slots"]] != list(bindings):
            raise ValueError("pool slots must be ordered slot0..slot3")
        return {"schema_version": 1, "mode": mode, "group_size": group_size,
                "roles": ([f"slot{i//2}" for i in range(8)] if mode == POOL_EQUAL else
                          ["slot0"] * 3 + ["slot1"] * 3 + ["slot2", "slot3"]), "bindings": bindings}
    bindings = {"active": normalize_binding(compact(active))}
    if mode == H0_ACTIVE_EQUAL:
        bindings["h0"] = normalize_binding(compact(h0))
    return {"schema_version": 1, "mode": mode, "group_size": group_size,
            "roles": (["h0"] * 4 + ["active"] * 4 if mode == H0_ACTIVE_EQUAL
                      else ["active"] * group_size), "bindings": bindings}


def normalize_assignment(plan):
    if isinstance(plan, dict) and plan.get('mode') == FIXED_BACKEND_EQUAL:
        expected = fixed_backend_assignment(plan['bindings']['h0'], plan['bindings']['active'])
        if plan != expected:
            raise ValueError('fixed backend identity/4+4 assignment changed')
        return expected
    if isinstance(plan, dict) and plan.get("mode") == ROUTER_TOPK:
        return normalize_router_assignment(plan)
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "mode", "group_size", "roles", "bindings"}:
        raise ValueError("invalid Harness assignment fields")
    if plan["schema_version"] != 1 or type(plan["group_size"]) is not int or plan["group_size"] < 1:
        raise ValueError("invalid Harness assignment schema/group size")
    if plan["mode"] in {POOL_EQUAL, POOL_WEIGHTED}:
        if set(plan["bindings"]) != {f"slot{i}" for i in range(4)}:
            raise ValueError("pool assignment requires four bindings")
        pool = {"slots": [{"slot_id": f"slot{i}", "harness": plan["bindings"][f"slot{i}"]} for i in range(4)]}
        expected = assignment(plan["mode"], None, None, plan["group_size"], pool=pool)
    else:
        expected = assignment(plan["mode"], plan["bindings"].get("h0"),
                              plan["bindings"]["active"], plan["group_size"])
    if plan["roles"] != expected["roles"] or set(plan["bindings"]) != set(expected["bindings"]):
        raise ValueError("Harness assignment role/clone layout changed")
    return deepcopy(expected)


def primary_binding(plan):
    """Compatibility representative; never used to assign all mixed clones."""
    role = "h0" if plan["mode"] == ROUTER_TOPK else "slot0" if plan["mode"] in {POOL_EQUAL, POOL_WEIGHTED} else "active"
    return plan["bindings"][role]


def clone_role(plan, task_id, clone_index):
    """Pure hot-path lookup. Hash/file verification happens once when binding."""
    if type(clone_index) is not int or not 0 <= clone_index < plan["group_size"]:
        raise ValueError("invalid Harness clone index")
    if plan.get('schema_version') in (4, 6, 7) and plan['mode'] == ROUTER_TOPK:
        return 'h0'
    if plan.get('schema_version') == 8 and plan['mode'] == ROUTER_TOPK:
        return plan['task_roles']['*'][clone_index]
    roles = plan["task_roles"][str(task_id)] if plan["mode"] == ROUTER_TOPK else plan["roles"]
    return roles[clone_index]


def router_assignment(h0, members, routes, *, stage_version, route_plan_version, layout=None):
    if layout == 'fixed-h0-top4-v1':
        if len(routes) != 1 or routes[0]['task_id'] != '*':
            raise ValueError('fixed mixture requires one global allocation')
        ranked = routes[0]['ranked_harnesses']
        if len(ranked) != 5 or ranked[0] != 'h0' or len(set(ranked)) != 5:
            raise ValueError('fixed mixture requires H0 plus four distinct Harnesses')
        compact = lambda h: {k: h[k] for k in ('version', 'sha256', 'path')}
        return normalize_router_assignment(dict(schema_version=8, mode=ROUTER_TOPK, group_size=8,
            bindings={h['version']:compact(h) for h in members},
            task_roles={'*':['h0']*4+ranked[1:]}, stage_version=stage_version, route_plan_version=route_plan_version))
    if layout not in (None, 'top4-3311-v1', 'h0-first-3311-v1', 'top4-2222-v1', 'h0-first-2222-v1', 'h0-first-11111111-v1'):
        raise ValueError('unknown Router allocation version')
    compact = lambda h: {k: h[k] for k in ("version", "sha256", "path")}
    bindings = {h["version"]: compact(h) for h in members}
    bindings["h0"] = compact(h0)
    contents = {h["version"]: h["content_sha256"] for h in members}
    task_roles = {}
    for route in routes:
        tid, ranked = str(route["task_id"]), route["ranked_harnesses"]
        size = 8 if layout == 'h0-first-11111111-v1' else 4
        if (tid in task_roles or (len(ranked) != size if layout else not 1 <= len(ranked) <= 3)
                or len(set(ranked)) != len(ranked)
                or any(h not in contents for h in ranked)
                or len({contents[h] for h in ranked}) != len(ranked)):
            raise ValueError("invalid per-task distinct Harness ranking")
        budget = ([1]*8 if layout == 'h0-first-11111111-v1' else
                  [2, 2, 2, 2] if layout in ('top4-2222-v1', 'h0-first-2222-v1') else
                  [3, 3, 1, 1] if layout else {1: [8], 2: [4, 4], 3: [4, 2, 2]}[len(ranked)])
        task_roles[tid] = [h for h, n in zip(ranked, budget) for _ in range(n)]
    schema = {None: 2, 'top4-3311-v1': 3, 'h0-first-3311-v1': 4,
              'top4-2222-v1': 5, 'h0-first-2222-v1': 6, 'h0-first-11111111-v1': 7}[layout]
    return normalize_router_assignment({"schema_version": schema, "mode": ROUTER_TOPK,
        "group_size": 8, "bindings": bindings, "task_roles": task_roles,
        "stage_version": stage_version, "route_plan_version": route_plan_version})


def normalize_router_assignment(plan):
    required = {"schema_version", "mode", "group_size", "bindings", "task_roles", "stage_version", "route_plan_version"}
    if set(plan) != required or plan["schema_version"] not in (2, 3, 4, 5, 6, 7, 8) or plan["group_size"] != 8 or plan["mode"] != ROUTER_TOPK:
        raise ValueError("invalid router assignment contract")
    if any(not isinstance(plan[k], str) or not plan[k] for k in ("stage_version", "route_plan_version")):
        raise ValueError("router assignment requires committed identities")
    if not isinstance(plan["bindings"], dict) or "h0" not in plan["bindings"]:
        raise ValueError("router assignment requires immutable H0 binding")
    if not isinstance(plan["task_roles"], dict):
        raise ValueError("router assignment requires a task map")
    bindings = {k: normalize_binding(v) for k, v in plan["bindings"].items()}
    if plan['schema_version'] == 8:
        roles = plan['task_roles'].get('*', [])
        if (set(plan['task_roles']) != {'*'} or len(bindings) != 5
                or len(roles) != 8 or roles[:4] != ['h0']*4
                or len(set(roles[4:])) != 4 or 'h0' in roles[4:]
                or set(roles) != set(bindings)
                or any(k != b['version'] for k,b in bindings.items())):
            raise ValueError('fixed mixture requires immutable 4/1/1/1/1 allocation')
        return {**deepcopy(plan), 'bindings': bindings}
    for tid, roles in plan["task_roles"].items():
        if not isinstance(tid, str) or not tid or not isinstance(roles, list) or len(roles) != 8:
            raise ValueError("every routed task requires eight clone slots")
        ordered = list(dict.fromkeys(roles))
        top4 = plan['schema_version'] in (3, 4, 5, 6)
        top8 = plan['schema_version'] == 7
        if any(h not in bindings for h in ordered) or (len(ordered) != 8 if top8 else len(ordered) != 4 if top4 else not 1 <= len(ordered) <= 3):
            raise ValueError("unknown routed Harness")
        budget = ([1]*8 if top8 else [2, 2, 2, 2] if plan['schema_version'] in (5, 6) else
                  [3, 3, 1, 1] if top4 else {1: [8], 2: [4, 4], 3: [4, 2, 2]}[len(ordered)])
        expected = [h for h, n in zip(ordered, budget) for _ in range(n)]
        if roles != expected:
            raise ValueError("router clone layout differs from its allocation version")
    return {**deepcopy(plan), "bindings": bindings}

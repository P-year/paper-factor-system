"""
agent/prompts/decision.py - DEPRECATED shim

Phase 1：原 DECISION_PROMPT / FEASIBILITY_PROMPT 已搬到
pipelines/paper_factor/config.yaml:prompts.decision / .feasibility。

这里改为从 yaml 读取并 re-export，打一行 deprecation warning。
"""
import sys
import warnings
from pathlib import Path


_warned = False


def _warn():
    global _warned
    if not _warned:
        warnings.warn(
            "agent.prompts.decision is deprecated. "
            "Use harness.registry.get('paper_factor').prompts['decision'|'feasibility'] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True


def _load(key: str) -> str:
    import yaml as _yaml
    yaml_path = Path(__file__).resolve().parent.parent.parent / "pipelines" / "paper_factor" / "config.yaml"
    with yaml_path.open(encoding="utf-8") as f:
        return _yaml.safe_load(f)["prompts"][key]


DECISION_PROMPT = _load("decision")
FEASIBILITY_PROMPT = _load("feasibility")
_warn()
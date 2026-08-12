from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptDefinition:
    name: str
    version: str
    system_prompt: str
    user_template: str

    def render(self, **values: str) -> str:
        return self.user_template.format(**values)


class PromptRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def extraction(self, pass_name: str) -> PromptDefinition:
        path = self.directory / "extraction.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            definition = data["prompts"][pass_name]
        except KeyError as error:
            available = ", ".join(sorted(data.get("prompts", {})))
            raise KeyError(
                f"No extraction prompt for {pass_name!r}; available: {available}"
            ) from error
        return PromptDefinition(**definition)


def load_relation_vocabulary(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    relations = data.get("relations", [])
    if not isinstance(relations, list) or not all(isinstance(item, str) for item in relations):
        raise ValueError(f"Invalid relation vocabulary in {path}")
    if "OTHER" not in relations:
        raise ValueError("Relation vocabulary must include OTHER")
    return relations

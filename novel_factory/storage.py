"""Filesystem-backed checkpoint and artifact storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from novel_factory.config import AppConfig
from novel_factory.schemas import RunLogEvent, SceneQaReport
from novel_factory.utils import (
    ensure_directory,
    format_chapter_number,
    format_scene_number,
    json_dumps,
    read_text,
    slugify,
    timestamp_utc,
    write_text,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class RunStorage:
    """Persists all project artifacts under runs/<project_slug>/."""

    def __init__(self, config: AppConfig, project: str) -> None:
        self.config = config
        self.project_slug = slugify(project)
        self.root = ensure_directory(config.run_root / self.project_slug)
        self.scenes_dir = ensure_directory(self.root / "scenes")
        self.qa_dir = ensure_directory(self.root / "qa")
        self.rewrites_dir = ensure_directory(self.root / "rewrites")
        self.chapters_dir = ensure_directory(self.root / "chapters")
        self.ensure_layout()

    def ensure_layout(self) -> None:
        """Creates the expected run directory layout."""

        for directory in [
            self.root,
            self.scenes_dir,
            self.qa_dir,
            self.rewrites_dir,
            self.chapters_dir,
        ]:
            ensure_directory(directory)

    @property
    def synopsis_path(self) -> Path:
        return self.root / "input_synopsis.md"

    @property
    def intake_markdown_path(self) -> Path:
        return self.root / "book_intake.md"

    @property
    def intake_json_path(self) -> Path:
        return self.root / "book_intake.json"

    @property
    def story_spec_path(self) -> Path:
        return self.root / "story_spec.json"

    @property
    def outline_path(self) -> Path:
        return self.root / "outline.json"

    @property
    def scene_cards_path(self) -> Path:
        return self.root / "scene_cards.json"

    @property
    def plan_qa_path(self) -> Path:
        return self.root / "plan_qa_report.json"

    @property
    def continuity_path(self) -> Path:
        return self.root / "continuity_state.json"

    @property
    def initial_continuity_path(self) -> Path:
        return self.root / "initial_continuity_state.json"

    @property
    def final_markdown_path(self) -> Path:
        return self.root / "final_manuscript.md"

    @property
    def final_text_path(self) -> Path:
        return self.root / "final_manuscript.txt"

    @property
    def global_qa_path(self) -> Path:
        return self.root / "global_qa_report.json"

    @property
    def run_log_path(self) -> Path:
        return self.root / "run_log.jsonl"

    def scene_path(self, scene_number: int) -> Path:
        return self.scenes_dir / f"scene_{format_scene_number(scene_number)}.md"

    def rewrite_path(
        self,
        scene_number: int,
        attempt_number: int,
        *,
        phase_label: str = "draft",
    ) -> Path:
        return self.rewrites_dir / (
            f"scene_{format_scene_number(scene_number)}_{phase_label}_attempt_{attempt_number:02d}.md"
        )

    def chapter_path(self, chapter_number: int) -> Path:
        return self.chapters_dir / f"Chapter_{format_chapter_number(chapter_number)}.md"

    def scene_qa_path(self, scene_number: int) -> Path:
        return self.qa_dir / f"scene_{format_scene_number(scene_number)}_qa.json"

    def chapter_qa_path(self, chapter_number: int) -> Path:
        return self.qa_dir / f"chapter_{format_chapter_number(chapter_number)}_qa.json"

    def arc_qa_path(self, arc_name: str) -> Path:
        return self.qa_dir / f"arc_{slugify(arc_name)}_qa.json"

    def save_text(self, path: Path, text: str) -> None:
        """Saves a plain-text file."""

        write_text(path, text, encoding=self.config.synopsis_encoding)

    def load_text(self, path: Path) -> str:
        """Loads a plain-text file."""

        return read_text(path, encoding=self.config.synopsis_encoding)

    def save_model(self, path: Path, model: BaseModel) -> None:
        """Serializes a Pydantic model to JSON."""

        self.save_text(path, json_dumps(model.model_dump()))

    def load_model(self, path: Path, model_type: type[ModelT]) -> ModelT:
        """Deserializes a Pydantic model from JSON."""

        payload = json.loads(self.load_text(path))
        return model_type.model_validate(payload)

    def append_log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """Appends a JSONL run event."""

        log_event = RunLogEvent(timestamp=timestamp_utc(), event=event, payload=payload or {})
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_log_path.open("a", encoding=self.config.synopsis_encoding) as handle:
            handle.write(log_event.model_dump_json() + "\n")

    def has_approved_scene(self, scene_number: int) -> bool:
        """Returns True when a scene and passing QA file both exist."""

        scene_path = self.scene_path(scene_number)
        qa_path = self.scene_qa_path(scene_number)
        if not scene_path.exists() or not qa_path.exists():
            return False
        report = self.load_model(qa_path, SceneQaReport)
        return report.pass_fail

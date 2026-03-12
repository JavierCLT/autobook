"""Checkpointed end-to-end pipeline orchestration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from novel_factory.config import AppConfig
from novel_factory.generators import NovelGenerator
from novel_factory.intake import parse_book_intake
from novel_factory.judges import GlobalJudge, SceneJudge
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.schemas import (
    ArcQaReport,
    BookIntake,
    ChapterQaReport,
    ContinuityState,
    DeterministicValidationReport,
    GlobalQaReport,
    Outline,
    RepairTarget,
    SceneCard,
    SceneCardCollection,
    SceneQaReport,
    StorySpec,
)
from novel_factory.storage import RunStorage
from novel_factory.utils import (
    chapter_scene_numbers,
    count_words,
    get_scene_card,
    plain_text_from_markdown,
    truncate_text,
)
from novel_factory.validators import PlanValidator, SceneValidator

logger = logging.getLogger(__name__)


class SceneApprovalError(RuntimeError):
    """Raised when a scene cannot pass QA after the rewrite budget is exhausted."""


@dataclass
class ProjectArtifacts:
    """Loaded project state required by pipeline phases."""

    synopsis: str
    book_intake: BookIntake | None
    story_spec: StorySpec
    outline: Outline
    scene_cards: list[SceneCard]
    continuity_state: ContinuityState


class NovelPipeline:
    """Runs the planning, drafting, QA, repair, and assembly pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = OpenAIResponsesClient(config)
        self.generators = NovelGenerator(self.llm, config)
        self.plan_validator = PlanValidator()
        self.validator = SceneValidator()
        self.scene_judge = SceneJudge(self.llm, config)
        self.global_judge = GlobalJudge(self.llm, config)

    def bootstrap(
        self,
        *,
        project: str,
        synopsis_file: Path | None = None,
        intake_file: Path | None = None,
    ) -> RunStorage:
        """Creates or resumes the planning artifacts for a project."""

        storage = RunStorage(self.config, project)
        if synopsis_file is None and intake_file is None:
            raise ValueError("Provide either synopsis_file or intake_file.")

        book_intake: BookIntake | None = None
        if intake_file is not None:
            intake_text = intake_file.read_text(encoding=self.config.synopsis_encoding).strip()
            if not intake_text:
                raise ValueError(f"Intake file is empty: {intake_file}")
            book_intake = parse_book_intake(intake_text)
            storage.save_text(storage.intake_markdown_path, intake_text + "\n")
            storage.save_model(storage.intake_json_path, book_intake)

        if synopsis_file is not None:
            synopsis_text = synopsis_file.read_text(encoding=self.config.synopsis_encoding).strip()
            if not synopsis_text:
                raise ValueError(f"Synopsis file is empty: {synopsis_file}")
        else:
            assert book_intake is not None
            synopsis_text = book_intake.fields.get("synopsis", "").strip()
            if not synopsis_text:
                raise ValueError("Intake file does not contain a synopsis field.")

        storage.save_text(storage.synopsis_path, synopsis_text + "\n")
        storage.append_log(
            "bootstrap_started",
            {
                "project": storage.project_slug,
                "synopsis_file": str(synopsis_file) if synopsis_file else None,
                "intake_file": str(intake_file) if intake_file else None,
            },
        )

        if storage.story_spec_path.exists():
            story_spec = storage.load_model(storage.story_spec_path, StorySpec)
            logger.info("Loaded existing story_spec.json")
        else:
            story_spec = self.generators.generate_story_spec(synopsis_text, book_intake=book_intake)
            storage.save_model(storage.story_spec_path, story_spec)
            storage.append_log("story_spec_generated", {"title": story_spec.title_working})

        if storage.outline_path.exists():
            outline = storage.load_model(storage.outline_path, Outline)
            logger.info("Loaded existing outline.json")
        else:
            outline = self.generators.generate_outline(
                synopsis_text,
                story_spec,
                book_intake=book_intake,
            )
            storage.save_model(storage.outline_path, outline)
            storage.append_log("outline_generated", {"chapters": len(outline.chapters)})

        if storage.scene_cards_path.exists():
            scene_cards = storage.load_model(storage.scene_cards_path, SceneCardCollection).scene_cards
            logger.info("Loaded existing scene_cards.json")
        else:
            scene_cards = self.generators.generate_scene_cards(
                synopsis_text,
                story_spec,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(
                storage.scene_cards_path,
                SceneCardCollection(scene_cards=scene_cards),
            )
            storage.append_log("scene_cards_generated", {"scenes": len(scene_cards)})

        if storage.initial_continuity_path.exists():
            initial_continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        else:
            initial_continuity = self.generators.generate_initial_continuity(
                story_spec,
                outline,
                book_intake=book_intake,
            )
            storage.save_model(storage.initial_continuity_path, initial_continuity)
            storage.append_log("initial_continuity_generated", {})

        plan_report = self.plan_validator.validate(
            story_spec=story_spec,
            outline=outline,
            scene_cards=scene_cards,
            initial_continuity=initial_continuity,
        )
        storage.save_model(storage.plan_qa_path, plan_report)
        storage.append_log("plan_validation_completed", {"pass_fail": plan_report.pass_fail})
        if not plan_report.pass_fail:
            raise RuntimeError(
                "Planning artifacts failed deterministic plan QA. Adjust the bootstrap prompts or start from a fresh project slug."
            )

        if not storage.continuity_path.exists():
            storage.save_model(storage.continuity_path, initial_continuity)

        storage.append_log(
            "bootstrap_completed",
            {
                "story_spec_path": str(storage.story_spec_path),
                "outline_path": str(storage.outline_path),
                "scene_cards_path": str(storage.scene_cards_path),
            },
        )
        return storage

    def draft_scene(self, *, project: str, scene_index: int, force: bool = False) -> SceneQaReport:
        """Drafts a single scene and runs the full approval loop."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if storage.has_approved_scene(scene_index) and not force:
            logger.info("Scene %s is already approved; skipping.", scene_index)
            return storage.load_model(storage.scene_qa_path(scene_index), SceneQaReport)

        scene_card = get_scene_card(artifacts.scene_cards, scene_index)
        continuity_before = self._continuity_before_scene(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
            scene_number=scene_index,
        )

        scene_text, qa_report = self._run_scene_loop(
            storage=storage,
            story_spec=artifacts.story_spec,
            outline=artifacts.outline,
            book_intake=artifacts.book_intake,
            scene_card=scene_card,
            continuity_before=continuity_before,
            phase_label="draft",
            attempt_generator=lambda rewrite_brief, current_draft: self.generators.draft_scene(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                scene_card=scene_card,
                continuity_state=continuity_before,
                recent_scene_summaries=self._recent_summaries(continuity_before),
                book_intake=artifacts.book_intake,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
            ),
        )

        self._approve_scene(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_card=scene_card,
            continuity_before=continuity_before,
            scene_text=scene_text,
            qa_report=qa_report,
        )

        # Keep continuity_state.json aligned with the full approved prefix after ad hoc rewrites.
        self._rebuild_continuity_from_approved_scenes(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
        )
        return qa_report

    def run_project(
        self,
        *,
        project: str,
        synopsis_file: Path | None = None,
        intake_file: Path | None = None,
    ) -> GlobalQaReport:
        """Runs the full project from planning through QA and repair."""

        storage = self.bootstrap(project=project, synopsis_file=synopsis_file, intake_file=intake_file)
        artifacts = self._load_project_artifacts(storage)
        self._rebuild_continuity_from_approved_scenes(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
        )

        approved_prefix = self._approved_prefix_length(storage, len(artifacts.scene_cards))
        for scene_number in range(approved_prefix + 1, len(artifacts.scene_cards) + 1):
            self.draft_scene(project=project, scene_index=scene_number)

        self.assemble_manuscript(project=project)
        self._run_editorial_qa(project=project)
        report = self.global_qa(project=project)
        if report.pass_fail:
            return report
        repaired_report = self.repair_project(project=project)
        if not repaired_report.pass_fail:
            raise RuntimeError("Targeted repairs completed, but global QA still fails.")
        return repaired_report

    def assemble_manuscript(self, *, project: str) -> Path:
        """Assembles approved scenes into chapter files and final manuscript outputs."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        for scene_card in artifacts.scene_cards:
            if not storage.has_approved_scene(scene_card.scene_number):
                raise RuntimeError(
                    f"Cannot assemble manuscript before scene {scene_card.scene_number} is approved."
                )

        chapter_texts: list[str] = []
        for chapter in artifacts.outline.chapters:
            scene_numbers = chapter_scene_numbers(artifacts.scene_cards, chapter.chapter_number)
            scene_texts = [storage.load_text(storage.scene_path(scene_number)) for scene_number in scene_numbers]
            chapter_markdown = (
                f"# Chapter {chapter.chapter_number:02d}: {chapter.title}\n\n"
                + "\n\n".join(scene_texts).strip()
                + "\n"
            )
            storage.save_text(storage.chapter_path(chapter.chapter_number), chapter_markdown)
            chapter_texts.append(chapter_markdown.strip())

        final_markdown = f"# {artifacts.story_spec.title_working}\n\n" + "\n\n".join(chapter_texts) + "\n"
        storage.save_text(storage.final_markdown_path, final_markdown)
        storage.save_text(storage.final_text_path, plain_text_from_markdown(final_markdown))
        storage.append_log("manuscript_assembled", {"path": str(storage.final_markdown_path)})
        return storage.final_markdown_path

    def global_qa(self, *, project: str) -> GlobalQaReport:
        """Runs manuscript-level QA on the assembled manuscript."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if not storage.final_markdown_path.exists():
            self.assemble_manuscript(project=project)

        manuscript_text = storage.load_text(storage.final_markdown_path)
        report = self.global_judge.judge(
            story_spec=artifacts.story_spec,
            outline=artifacts.outline,
            manuscript_text=manuscript_text,
            book_intake=artifacts.book_intake,
        )
        storage.save_model(storage.global_qa_path, report)
        storage.append_log("global_qa_completed", {"pass_fail": report.pass_fail})
        return report

    def _run_editorial_qa(self, *, project: str) -> None:
        """Runs chapter and arc QA before the manuscript-level judge."""

        storage = RunStorage(self.config, project)
        max_editorial_rounds = 3
        for round_number in range(0, max_editorial_rounds):
            artifacts = self._load_project_artifacts(storage)
            chapter_reports, arc_reports = self._judge_editorial_layers(
                storage=storage,
                artifacts=artifacts,
            )
            repair_targets = self._collect_editorial_repair_targets(chapter_reports, arc_reports)
            if not repair_targets:
                return
            if round_number >= max_editorial_rounds - 1:
                raise RuntimeError(
                    "Editorial QA still failed after targeted scene repairs. Inspect chapter and arc QA reports."
                )
            self._apply_editorial_repair_targets(
                storage=storage,
                artifacts=artifacts,
                repair_targets=repair_targets,
            )
            self._rebuild_continuity_from_approved_scenes(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_cards=artifacts.scene_cards,
            )
            self.assemble_manuscript(project=project)

    def repair_project(self, *, project: str) -> GlobalQaReport:
        """Applies targeted scene repairs from a failed global QA report."""

        storage = RunStorage(self.config, project)
        artifacts = self._load_project_artifacts(storage)
        if storage.global_qa_path.exists():
            global_report = storage.load_model(storage.global_qa_path, GlobalQaReport)
        else:
            global_report = self.global_qa(project=project)

        if global_report.pass_fail:
            return global_report
        if not global_report.repair_targets:
            raise RuntimeError("Global QA failed without repair targets.")

        seen_scene_numbers: set[int] = set()
        for target in sorted(global_report.repair_targets, key=lambda item: item.scene_number):
            if target.scene_number in seen_scene_numbers:
                continue
            seen_scene_numbers.add(target.scene_number)
            scene_card = get_scene_card(artifacts.scene_cards, target.scene_number)
            continuity_before = self._continuity_before_scene(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_cards=artifacts.scene_cards,
                scene_number=target.scene_number,
            )
            current_scene = storage.load_text(storage.scene_path(target.scene_number))
            repaired_scene, repaired_qa = self._run_scene_loop(
                storage=storage,
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                book_intake=artifacts.book_intake,
                scene_card=scene_card,
                continuity_before=continuity_before,
                phase_label="repair",
                attempt_generator=lambda rewrite_brief, current_draft, scene_card=scene_card, continuity_before=continuity_before, current_scene=current_scene: self.generators.repair_scene(
                    story_spec=artifacts.story_spec,
                    outline=artifacts.outline,
                    scene_card=scene_card,
                    continuity_state=continuity_before,
                    current_scene=current_draft or current_scene,
                    global_qa_report=global_report,
                    rewrite_brief=rewrite_brief or target.rewrite_brief,
                    book_intake=artifacts.book_intake,
                ),
                initial_rewrite_brief=target.rewrite_brief,
                initial_current_draft=current_scene,
            )
            self._approve_scene(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_card=scene_card,
                continuity_before=continuity_before,
                scene_text=repaired_scene,
                qa_report=repaired_qa,
            )
            storage.append_log(
                "scene_repaired",
                {"scene_number": target.scene_number, "reason": target.reason},
            )

        self._rebuild_continuity_from_approved_scenes(
            storage=storage,
            story_spec=artifacts.story_spec,
            scene_cards=artifacts.scene_cards,
        )
        self.assemble_manuscript(project=project)
        self._run_editorial_qa(project=project)
        return self.global_qa(project=project)

    def _judge_editorial_layers(
        self,
        *,
        storage: RunStorage,
        artifacts: ProjectArtifacts,
    ) -> tuple[list[ChapterQaReport], list[ArcQaReport]]:
        """Runs chapter-level and targeted arc-level QA."""

        chapter_reports: list[ChapterQaReport] = []
        for chapter in artifacts.outline.chapters:
            chapter_text = storage.load_text(storage.chapter_path(chapter.chapter_number))
            report = self.global_judge.judge_chapter(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                chapter_number=chapter.chapter_number,
                chapter_text=chapter_text,
                scene_cards=artifacts.scene_cards,
            )
            storage.save_model(storage.chapter_qa_path(chapter.chapter_number), report)
            chapter_reports.append(report)

        arc_reports: list[ArcQaReport] = []
        for arc_name, arc_focus, scene_numbers in self._editorial_arc_specs(artifacts.scene_cards):
            arc_text = "\n\n".join(
                storage.load_text(storage.scene_path(scene_number)).strip()
                for scene_number in scene_numbers
            ).strip()
            report = self.global_judge.judge_arc(
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                arc_name=arc_name,
                arc_focus=arc_focus,
                scene_numbers=scene_numbers,
                arc_text=arc_text,
            )
            storage.save_model(storage.arc_qa_path(arc_name), report)
            arc_reports.append(report)

        storage.append_log(
            "editorial_qa_completed",
            {
                "chapter_failures": sum(1 for report in chapter_reports if not report.pass_fail),
                "arc_failures": sum(1 for report in arc_reports if not report.pass_fail),
            },
        )
        return chapter_reports, arc_reports

    def _collect_editorial_repair_targets(
        self,
        chapter_reports: list[ChapterQaReport],
        arc_reports: list[ArcQaReport],
    ) -> list[RepairTarget]:
        """Collapses chapter and arc repair targets into a minimal scene list."""

        merged: dict[int, RepairTarget] = {}
        for report in [*chapter_reports, *arc_reports]:
            if report.pass_fail:
                continue
            if not report.repair_targets:
                raise RuntimeError("Editorial QA failed without scene-level repair targets.")
            for target in report.repair_targets:
                existing = merged.get(target.scene_number)
                if existing is None:
                    merged[target.scene_number] = target
                    continue
                merged[target.scene_number] = RepairTarget(
                    scene_number=target.scene_number,
                    reason="; ".join(
                        sorted(
                            {
                                existing.reason.strip(),
                                target.reason.strip(),
                            }
                        )
                    ),
                    rewrite_brief="\n".join(
                        dict.fromkeys(
                            [
                                existing.rewrite_brief.strip(),
                                target.rewrite_brief.strip(),
                            ]
                        )
                    ).strip(),
                )
        return [merged[number] for number in sorted(merged)]

    def _apply_editorial_repair_targets(
        self,
        *,
        storage: RunStorage,
        artifacts: ProjectArtifacts,
        repair_targets: list[RepairTarget],
    ) -> None:
        """Rewrites scene targets raised by chapter or arc QA."""

        for target in repair_targets:
            scene_card = get_scene_card(artifacts.scene_cards, target.scene_number)
            continuity_before = self._continuity_before_scene(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_cards=artifacts.scene_cards,
                scene_number=target.scene_number,
            )
            current_scene = storage.load_text(storage.scene_path(target.scene_number))
            repaired_scene, repaired_qa = self._run_scene_loop(
                storage=storage,
                story_spec=artifacts.story_spec,
                outline=artifacts.outline,
                book_intake=artifacts.book_intake,
                scene_card=scene_card,
                continuity_before=continuity_before,
                phase_label="editorial_repair",
                attempt_generator=lambda rewrite_brief, current_draft, scene_card=scene_card, continuity_before=continuity_before, current_scene=current_scene, target=target: self.generators.repair_scene(
                    story_spec=artifacts.story_spec,
                    outline=artifacts.outline,
                    scene_card=scene_card,
                    continuity_state=continuity_before,
                    current_scene=current_draft or current_scene,
                    global_qa_report=GlobalQaReport(
                        pass_fail=False,
                        hook_strength_score=3,
                        midpoint_turn_score=3,
                        climax_payoff_score=3,
                        ending_payoff_score=3,
                        relationship_progression_score=3,
                        antagonist_pressure_score=3,
                        continuity_score=3,
                        emotional_aftershock_score=3,
                        boredom_risk_score=3,
                        voice_consistency_score=3,
                        ai_smell_score=3,
                        major_problems=[target.reason],
                        repair_targets=[target],
                    ),
                    rewrite_brief=rewrite_brief or target.rewrite_brief,
                    book_intake=artifacts.book_intake,
                ),
                initial_rewrite_brief=target.rewrite_brief,
                initial_current_draft=current_scene,
            )
            self._approve_scene(
                storage=storage,
                story_spec=artifacts.story_spec,
                scene_card=scene_card,
                continuity_before=continuity_before,
                scene_text=repaired_scene,
                qa_report=repaired_qa,
            )
            storage.append_log(
                "scene_editorially_repaired",
                {"scene_number": target.scene_number, "reason": target.reason},
            )

    def _load_project_artifacts(self, storage: RunStorage) -> ProjectArtifacts:
        """Loads the persisted project artifacts."""

        if not storage.synopsis_path.exists():
            raise RuntimeError("Project has not been bootstrapped yet.")

        book_intake = None
        if storage.intake_json_path.exists():
            book_intake = storage.load_model(storage.intake_json_path, BookIntake)
        elif storage.intake_markdown_path.exists():
            book_intake = parse_book_intake(storage.load_text(storage.intake_markdown_path))

        return ProjectArtifacts(
            synopsis=storage.load_text(storage.synopsis_path).strip(),
            book_intake=book_intake,
            story_spec=storage.load_model(storage.story_spec_path, StorySpec),
            outline=storage.load_model(storage.outline_path, Outline),
            scene_cards=storage.load_model(storage.scene_cards_path, SceneCardCollection).scene_cards,
            continuity_state=storage.load_model(storage.continuity_path, ContinuityState),
        )

    def _recent_summaries(self, continuity_state: ContinuityState) -> list[str]:
        """Returns the recent scene summaries trimmed for drafting context."""

        summaries = continuity_state.recent_scene_summaries[-self.config.recent_scene_summaries :]
        return [truncate_text(summary, self.config.max_recent_scene_summary_chars) for summary in summaries]

    def _editorial_arc_specs(self, scene_cards: list[SceneCard]) -> list[tuple[str, str, list[int]]]:
        """Returns targeted manuscript slices for intermediate arc QA."""

        total_scenes = len(scene_cards)
        opening_numbers = list(range(1, min(total_scenes, 4) + 1))
        counterforce_numbers = [
            scene.scene_number for scene in scene_cards if self._scene_has_counterforce(scene)
        ][:6]
        relationship_numbers = [
            scene.scene_number for scene in scene_cards if self._scene_has_relationship_pressure(scene)
        ][:6]
        ending_numbers = list(range(max(1, total_scenes - 3), total_scenes + 1))

        specs: list[tuple[str, str, list[int]]] = [
            (
                "opening",
                "Does the opening create immediate compulsion, visible human cost, and a concrete shadow of pursuit or consequence?",
                opening_numbers,
            ),
            (
                "ending",
                "Does the ending convert containment, loss, and escape into emotional aftershock rather than elegant distance?",
                ending_numbers,
            ),
        ]
        if len(counterforce_numbers) >= 2:
            specs.append(
                (
                    "counterforce",
                    "Does the counterforce feel like a hunter rather than an abstract audit function, and does suspicion narrow the field?",
                    counterforce_numbers,
                )
            )
        if len(relationship_numbers) >= 2:
            specs.append(
                (
                    "relationship",
                    "Does the marriage deteriorate through refusals, absences, concealment, and cost rather than summary or theme?",
                    relationship_numbers,
                )
            )
        return specs

    def _scene_has_counterforce(self, scene_card: SceneCard) -> bool:
        """Returns True when a scene visibly advances pursuit or institutional shadow."""

        text = " ".join(
            [scene_card.counterforce_trace, scene_card.suspicion_delta, scene_card.pressure_source]
        ).lower()
        if any(
            keyword in text
            for keyword in ("marta", "conduct", "suspicion", "extract", "audit", "inquiry", "memo")
        ):
            return True
        return "review" in text and any(
            keyword in text
            for keyword in ("conduct", "marta", "extract", "suspicion", "audit", "request", "trail")
        )

    def _scene_has_relationship_pressure(self, scene_card: SceneCard) -> bool:
        """Returns True when a scene materially advances the marriage/relationship arc."""

        text = " ".join(
            [scene_card.relationship_delta, scene_card.secret_pressure, scene_card.cost_paid]
        ).lower()
        return scene_card.pov_character.lower().startswith("elena") or any(
            keyword in text
            for keyword in ("elena", "marriage", "home", "withdraw", "silence", "distance", "leave", "refuse", "cold")
        )

    def _run_scene_loop(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        outline: Outline,
        book_intake: BookIntake | None,
        scene_card: SceneCard,
        continuity_before: ContinuityState,
        phase_label: str,
        attempt_generator: Callable[[str | None, str | None], str],
        initial_rewrite_brief: str | None = None,
        initial_current_draft: str | None = None,
    ) -> tuple[str, SceneQaReport]:
        """Runs drafting, validation, judging, and rewrites until approval or failure."""

        rewrite_brief = initial_rewrite_brief
        current_draft = initial_current_draft
        latest_report: SceneQaReport | None = None
        latest_validation: DeterministicValidationReport | None = None

        for attempt_number in range(0, self.config.max_scene_rewrites + 1):
            scene_text = self._generate_scene_text(
                attempt_generator=attempt_generator,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
                scene_card=scene_card,
            )
            storage.save_text(
                storage.rewrite_path(
                    scene_card.scene_number,
                    attempt_number,
                    phase_label=phase_label,
                ),
                scene_text + "\n",
            )

            validation_report = self.validator.validate(
                scene_text=scene_text,
                scene_card=scene_card,
                story_spec=story_spec,
                continuity_state=continuity_before,
            )
            qa_report = self.scene_judge.judge(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity_before,
                validation_report=validation_report,
                scene_text=scene_text,
                book_intake=book_intake,
            )
            merged_report = self._merge_validation_into_qa(
                qa_report,
                validation_report,
                scene_card=scene_card,
                total_scenes=story_spec.expected_scenes,
            )
            if merged_report.pass_fail or not storage.has_approved_scene(scene_card.scene_number):
                storage.save_model(storage.scene_qa_path(scene_card.scene_number), merged_report)
            storage.append_log(
                "scene_attempt_completed",
                {
                    "scene_number": scene_card.scene_number,
                    "phase": phase_label,
                    "attempt_number": attempt_number,
                    "pass_fail": merged_report.pass_fail,
                },
            )

            if merged_report.pass_fail:
                return scene_text, merged_report

            latest_report = merged_report
            latest_validation = validation_report
            current_draft = scene_text
            rewrite_brief = self._compose_rewrite_brief(merged_report, validation_report)

        assert latest_report is not None
        assert latest_validation is not None
        storage.append_log(
            "scene_failed",
            {
                "scene_number": scene_card.scene_number,
                "phase": phase_label,
                "hard_fail_reasons": latest_report.hard_fail_reasons,
            },
        )
        raise SceneApprovalError(
            f"Scene {scene_card.scene_number} failed QA after {self.config.max_scene_rewrites + 1} attempts."
        )

    def _generate_scene_text(
        self,
        *,
        attempt_generator: Callable[[str | None, str | None], str],
        rewrite_brief: str | None,
        current_draft: str | None,
        scene_card: SceneCard,
    ) -> str:
        """Retries obviously truncated drafts before spending a rewrite attempt."""

        latest_scene_text = ""
        for _ in range(0, 3):
            latest_scene_text = attempt_generator(rewrite_brief, current_draft).strip()
            if not self._scene_text_looks_truncated(latest_scene_text, scene_card):
                return latest_scene_text
            logger.warning(
                "Scene %s draft looked truncated; retrying generation within the same attempt.",
                scene_card.scene_number,
            )
        return latest_scene_text

    def _scene_text_looks_truncated(self, scene_text: str, scene_card: SceneCard) -> bool:
        """Returns True when a scene draft is too short or ends like an accidental cutoff."""

        if not scene_text:
            return True
        minimum_retry_floor = max(500, int(scene_card.target_words * 0.55))
        if count_words(scene_text) < minimum_retry_floor:
            return True
        trimmed = scene_text.rstrip()
        if not trimmed:
            return True
        if trimmed[-1] in ".!?\"'”’":
            return False
        return True

    def _merge_validation_into_qa(
        self,
        qa_report: SceneQaReport,
        validation_report: DeterministicValidationReport,
        *,
        scene_card: SceneCard,
        total_scenes: int,
    ) -> SceneQaReport:
        """Applies deterministic failures and score thresholds to the model verdict."""

        hard_fail_reasons = list(qa_report.hard_fail_reasons)
        hard_fail_reasons.extend(
            finding.message for finding in validation_report.findings if finding.severity == "error"
        )

        threshold_failures = []
        thresholds = {
            "continuity_score": 4,
            "engagement_score": 4,
            "voice_score": 3,
            "pacing_score": 4,
            "specificity_score": 4,
            "prose_freshness_score": 4,
            "emotional_movement_score": 4,
            "subtext_score": 4,
            "concealment_score": 4,
            "leverage_shift_score": 4,
            "relationship_cost_score": 3,
            "commercial_hook_score": 4,
        }
        if self._is_anchor_scene(scene_card=scene_card, total_scenes=total_scenes):
            thresholds["voice_score"] = 4
        if self._scene_needs_relationship_cost(scene_card):
            thresholds["relationship_cost_score"] = 4
        if self._scene_allows_quieter_opening_drive(scene_card):
            thresholds["pacing_score"] = 3
            thresholds["commercial_hook_score"] = 3
        for field_name, minimum_score in thresholds.items():
            if getattr(qa_report, field_name) < minimum_score:
                threshold_failures.append(f"{field_name} fell below the acceptance threshold.")
        if qa_report.ai_smell_score >= 4:
            threshold_failures.append("AI-smell risk is above the acceptance threshold.")

        hard_fail_reasons.extend(threshold_failures)
        pass_fail = qa_report.pass_fail and validation_report.pass_fail and not threshold_failures
        return qa_report.model_copy(
            update={
                "pass_fail": pass_fail,
                "hard_fail_reasons": list(dict.fromkeys(hard_fail_reasons)),
                "deterministic_pass": validation_report.pass_fail,
                "deterministic_findings": validation_report.findings,
            }
        )

    def _compose_rewrite_brief(
        self,
        qa_report: SceneQaReport,
        validation_report: DeterministicValidationReport,
    ) -> str:
        """Combines model QA feedback and deterministic findings into one rewrite brief."""

        error_messages = [
            finding.message for finding in validation_report.findings if finding.severity == "error"
        ]
        warning_messages = [
            finding.message for finding in validation_report.findings if finding.severity == "warning"
        ]
        parts = [qa_report.rewrite_brief.strip()]
        if error_messages:
            parts.append("Fix deterministic hard failures: " + " ".join(error_messages))
        if warning_messages:
            parts.append("Reduce these validator risks: " + " ".join(warning_messages[:3]))
        if qa_report.ai_smell_score >= 4:
            parts.append(
                "Vary sentence openings and cadence, replace generic emotional shorthand with scene-specific detail, and cut explanatory dialogue."
            )
        score_guidance = {
            "engagement_score": "Increase immediate pressure and make the scene harder to stop reading within the first page.",
            "pacing_score": "Cut explanatory drag, tighten turn timing, and ensure each beat changes the pressure.",
            "prose_freshness_score": "Replace generic thriller phrasing with scene-specific language and sharper physical detail.",
            "emotional_movement_score": "Make the emotional turn visible through behavior, choice, and consequence rather than summary.",
            "subtext_score": "Reduce direct explanation and let omission, evasion, and behavior carry the scene's hidden argument.",
            "concealment_score": "Put an active lie, omission, or cover story under pressure on-page.",
            "leverage_shift_score": "Make the power balance clearly change by the end of the scene.",
            "relationship_cost_score": "Ensure the scene exacts a visible interpersonal cost rather than only a thematic one.",
            "commercial_hook_score": "Sharpen the opening disturbance and the closing choice so the scene creates stronger forward pull.",
        }
        for field_name, guidance in score_guidance.items():
            if getattr(qa_report, field_name) < 4:
                parts.append(guidance)
        return "\n".join(part for part in parts if part).strip()

    def _is_anchor_scene(self, *, scene_card: SceneCard, total_scenes: int) -> bool:
        """Returns True when a scene should clear a higher QA bar."""

        lower_type = scene_card.scene_type.lower()
        anchor_types = {
            "opening",
            "confrontation",
            "domestic fracture",
            "pursuit",
            "breakup",
            "midpoint",
            "climax",
            "fallout",
            "ending",
        }
        anchor_numbers = {1, max(1, total_scenes // 2), min(total_scenes, (total_scenes // 2) + 1), max(1, total_scenes - 1), total_scenes}
        return (
            lower_type in anchor_types
            or scene_card.scene_number in anchor_numbers
            or scene_card.pov_character.lower().startswith("elena")
        )

    def _scene_needs_relationship_cost(self, scene_card: SceneCard) -> bool:
        """Returns True when the scene should clear a higher interpersonal-cost bar."""

        text = " ".join([scene_card.relationship_delta, scene_card.secret_pressure, scene_card.cost_paid]).lower()
        required_entities_text = " ".join(scene_card.required_entities).lower()
        location_text = scene_card.location.lower()
        if "off-page" in text:
            return False
        if scene_card.scene_type.lower() == "domestic fracture":
            return True
        if scene_card.pov_character.lower().startswith("elena"):
            return True
        if any(keyword in required_entities_text for keyword in ("elena", "wife", "husband")):
            return True
        if any(keyword in location_text for keyword in ("apartment", "home", "kitchen", "bedroom")):
            return True
        relationship_patterns = (
            r"\belena\b",
            r"\bmarriage\b",
            r"\bhusband\b",
            r"\bwife\b",
            r"\bhome\b",
            r"\bdistance\b",
            r"\bwithdraw\b",
            r"\bwithdraws\b",
            r"\bwithdrawal\b",
            r"\bseparation\b",
            r"\bseparate\b",
            r"\bsplit\b",
            r"\bdivorce\b",
            r"\bleaves him\b",
            r"\bleaves her\b",
            r"\bleave him\b",
            r"\bleave her\b",
        )
        return any(re.search(pattern, text) for pattern in relationship_patterns)

    def _scene_allows_quieter_opening_drive(self, scene_card: SceneCard) -> bool:
        """Returns True when a scene can trade some hook velocity for intimacy or set-up."""

        lower_type = scene_card.scene_type.lower()
        quiet_types = {
            "domestic fracture",
            "cover story",
            "fallout",
        }
        return lower_type in quiet_types and not self._scene_has_counterforce(scene_card)

    def _approve_scene(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_before: ContinuityState,
        scene_text: str,
        qa_report: SceneQaReport,
    ) -> ContinuityState:
        """Persists an approved scene and advances continuity."""

        updated_continuity = self.generators.update_continuity(
            story_spec=story_spec,
            scene_card=scene_card,
            continuity_state=continuity_before,
            scene_text=scene_text,
        )
        storage.save_text(storage.scene_path(scene_card.scene_number), scene_text + "\n")
        storage.save_model(storage.scene_qa_path(scene_card.scene_number), qa_report)
        storage.save_model(storage.continuity_path, updated_continuity)
        storage.append_log(
            "scene_approved",
            {
                "scene_number": scene_card.scene_number,
                "chapter_number": scene_card.chapter_number,
                "word_count_hint": scene_card.target_words,
            },
        )
        return updated_continuity

    def _approved_prefix_length(self, storage: RunStorage, total_scenes: int) -> int:
        """Returns the length of the contiguous approved scene prefix."""

        prefix_length = 0
        for scene_number in range(1, total_scenes + 1):
            if storage.has_approved_scene(scene_number):
                prefix_length += 1
            else:
                break

        for scene_number in range(prefix_length + 1, total_scenes + 1):
            if storage.has_approved_scene(scene_number):
                raise RuntimeError(
                    "Approved scenes are not contiguous. Repair the run folder before resuming."
                )
        return prefix_length

    def _continuity_before_scene(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        scene_cards: list[SceneCard],
        scene_number: int,
    ) -> ContinuityState:
        """Replays continuity up to the scene immediately before the target scene."""

        if not storage.initial_continuity_path.exists():
            raise RuntimeError("Initial continuity snapshot is missing.")

        continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        for prior_scene_number in range(1, scene_number):
            if not storage.has_approved_scene(prior_scene_number):
                raise RuntimeError(
                    f"Scene {prior_scene_number} must be approved before drafting scene {scene_number}."
                )
            scene_card = get_scene_card(scene_cards, prior_scene_number)
            scene_text = storage.load_text(storage.scene_path(prior_scene_number))
            continuity = self.generators.update_continuity(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity,
                scene_text=scene_text,
            )
        return continuity

    def _rebuild_continuity_from_approved_scenes(
        self,
        *,
        storage: RunStorage,
        story_spec: StorySpec,
        scene_cards: list[SceneCard],
    ) -> ContinuityState:
        """Recomputes the continuity state from the approved contiguous prefix."""

        continuity = storage.load_model(storage.initial_continuity_path, ContinuityState)
        prefix_length = self._approved_prefix_length(storage, len(scene_cards))
        for scene_number in range(1, prefix_length + 1):
            scene_card = get_scene_card(scene_cards, scene_number)
            scene_text = storage.load_text(storage.scene_path(scene_number))
            continuity = self.generators.update_continuity(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity,
                scene_text=scene_text,
            )
        storage.save_model(storage.continuity_path, continuity)
        return continuity

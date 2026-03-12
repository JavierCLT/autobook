"""Model-based QA judges for scenes and full manuscripts."""

from __future__ import annotations

from novel_factory.config import AppConfig
from novel_factory.intake import build_drafting_guidance, build_planning_guidance
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.prompts import (
    arc_qa_system_prompt,
    arc_qa_user_prompt,
    chapter_qa_system_prompt,
    chapter_qa_user_prompt,
    global_qa_system_prompt,
    global_qa_user_prompt,
    scene_qa_system_prompt,
    scene_qa_user_prompt,
)
from novel_factory.schemas import (
    ArcQaReport,
    BookIntake,
    ChapterQaReport,
    ContinuityState,
    DeterministicValidationReport,
    GlobalQaReport,
    Outline,
    SceneCard,
    SceneQaReport,
    StorySpec,
)


class SceneJudge:
    """Runs the scene-level model judge."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        validation_report: DeterministicValidationReport,
        scene_text: str,
        book_intake: BookIntake | None = None,
    ) -> SceneQaReport:
        """Scores and judges a drafted scene."""

        return self.llm.structured(
            system_prompt=scene_qa_system_prompt(story_spec),
            user_prompt=scene_qa_user_prompt(
                story_spec=story_spec,
                scene_card=scene_card,
                continuity_state=continuity_state,
                validation_report=validation_report,
                scene_text=scene_text,
                intake_guidance=build_drafting_guidance(book_intake),
            ),
            schema=SceneQaReport,
            task_name=f"scene_qa_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_000,
            verbosity="low",
        )


class GlobalJudge:
    """Runs the global manuscript-level judge."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        manuscript_text: str,
        book_intake: BookIntake | None = None,
    ) -> GlobalQaReport:
        """Scores and judges the assembled manuscript."""

        return self.llm.structured(
            system_prompt=global_qa_system_prompt(story_spec),
            user_prompt=global_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                manuscript_text=manuscript_text,
                intake_guidance=build_planning_guidance(book_intake, max_chars=8_000),
            ),
            schema=GlobalQaReport,
            task_name="global_qa",
            reasoning_effort=self.config.reasoning.global_qa,
            temperature=self.config.global_qa_temperature,
            max_output_tokens=3_500,
            verbosity="low",
        )

    def judge_chapter(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        chapter_number: int,
        chapter_text: str,
        scene_cards: list[SceneCard],
    ) -> ChapterQaReport:
        """Scores one assembled chapter."""

        return self.llm.structured(
            system_prompt=chapter_qa_system_prompt(story_spec),
            user_prompt=chapter_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                chapter_number=chapter_number,
                chapter_text=chapter_text,
                scene_cards=scene_cards,
            ),
            schema=ChapterQaReport,
            task_name=f"chapter_qa_{chapter_number:02d}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_000,
            verbosity="low",
        )

    def judge_arc(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        arc_name: str,
        arc_focus: str,
        scene_numbers: list[int],
        arc_text: str,
    ) -> ArcQaReport:
        """Scores one targeted manuscript slice."""

        return self.llm.structured(
            system_prompt=arc_qa_system_prompt(story_spec),
            user_prompt=arc_qa_user_prompt(
                story_spec=story_spec,
                outline=outline,
                arc_name=arc_name,
                arc_focus=arc_focus,
                scene_numbers=scene_numbers,
                arc_text=arc_text,
            ),
            schema=ArcQaReport,
            task_name=f"arc_qa_{arc_name}",
            reasoning_effort=self.config.reasoning.qa,
            temperature=self.config.qa_temperature,
            max_output_tokens=2_000,
            verbosity="low",
        )

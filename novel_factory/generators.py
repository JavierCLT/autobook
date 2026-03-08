"""Planning, drafting, rewriting, and continuity generation."""

from __future__ import annotations

import re

from novel_factory.config import AppConfig
from novel_factory.llm import OpenAIResponsesClient
from novel_factory.prompts import (
    initial_continuity_user_prompt,
    planning_system_prompt,
    repair_scene_user_prompt,
    scene_cards_user_prompt,
    scene_draft_system_prompt,
    scene_draft_user_prompt,
    story_spec_user_prompt,
    outline_user_prompt,
)
from novel_factory.schemas import (
    ContinuityState,
    Outline,
    SceneCard,
    SceneCardCollection,
    StorySpec,
)
from novel_factory.utils import get_chapter_plan


class NovelGenerator:
    """Generates planning artifacts and prose with the Responses API."""

    def __init__(self, llm: OpenAIResponsesClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    def generate_story_spec(self, synopsis: str) -> StorySpec:
        """Creates the locked story specification from the synopsis."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=self.config.default_audience,
                rating_ceiling=self.config.default_rating_ceiling,
                market_position=self.config.default_market_position,
            ),
            user_prompt=story_spec_user_prompt(
                synopsis=synopsis,
                target_words=self.config.target_words,
                chapters=self.config.target_chapters,
                scenes=self.config.target_scenes,
                audience=self.config.default_audience,
                rating_ceiling=self.config.default_rating_ceiling,
                market_position=self.config.default_market_position,
            ),
            schema=StorySpec,
            task_name="story_spec",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=8_000,
            verbosity="low",
        )

    def generate_outline(self, synopsis: str, story_spec: StorySpec) -> Outline:
        """Creates the macro outline from the locked story spec."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=outline_user_prompt(synopsis=synopsis, story_spec=story_spec),
            schema=Outline,
            task_name="outline",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=16_000,
            verbosity="low",
        )

    def generate_scene_cards(
        self,
        synopsis: str,
        story_spec: StorySpec,
        outline: Outline,
    ) -> list[SceneCard]:
        """Creates and normalizes all scene cards."""

        collection = self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=scene_cards_user_prompt(
                synopsis=synopsis,
                story_spec=story_spec,
                outline=outline,
            ),
            schema=SceneCardCollection,
            task_name="scene_cards",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=24_000,
            verbosity="low",
        )
        return self._normalize_scene_cards(collection.scene_cards, story_spec, outline)

    def generate_initial_continuity(
        self,
        story_spec: StorySpec,
        outline: Outline,
    ) -> ContinuityState:
        """Creates the initial continuity state."""

        return self.llm.structured(
            system_prompt=planning_system_prompt(
                audience=story_spec.audience,
                rating_ceiling=story_spec.rating_ceiling,
                market_position=story_spec.subgenre or story_spec.genre,
            ),
            user_prompt=initial_continuity_user_prompt(story_spec=story_spec, outline=outline),
            schema=ContinuityState,
            task_name="initial_continuity",
            reasoning_effort=self.config.reasoning.planning,
            temperature=self.config.planning_temperature,
            max_output_tokens=3_000,
            verbosity="low",
        )

    def draft_scene(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        recent_scene_summaries: list[str],
        rewrite_brief: str | None = None,
        current_draft: str | None = None,
    ) -> str:
        """Drafts or rewrites a scene in prose."""

        is_rewrite = rewrite_brief is not None or current_draft is not None
        return self.llm.text(
            system_prompt=scene_draft_system_prompt(story_spec),
            user_prompt=scene_draft_user_prompt(
                story_spec=story_spec,
                outline=outline,
                scene_card=scene_card,
                continuity_state=continuity_state,
                recent_scene_summaries=recent_scene_summaries,
                rewrite_brief=rewrite_brief,
                current_draft=current_draft,
            ),
            task_name=f"{'rewrite' if is_rewrite else 'draft'}_scene_{scene_card.scene_number:02d}",
            reasoning_effort=(
                self.config.reasoning.rewriting if is_rewrite else self.config.reasoning.drafting
            ),
            temperature=(
                self.config.rewriting_temperature
                if is_rewrite
                else self.config.drafting_temperature
            ),
            max_output_tokens=7_000,
        )

    def update_continuity(
        self,
        *,
        story_spec: StorySpec,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        scene_text: str,
    ) -> ContinuityState:
        """Updates continuity deterministically from the locked scene contract."""

        facts_to_add = list(scene_card.continuity_outputs)

        open_threads_to_add = self._merge_candidate_updates(
            [
                scene_card.revelation_or_shift,
                scene_card.pressure_source,
                scene_card.secret_pressure,
            ]
        )

        relationship_updates = self._merge_candidate_updates(
            [
                scene_card.relationship_delta,
                f"{scene_card.pov_character}: {scene_card.emotional_turn}",
            ]
        )

        suspicion_updates = self._merge_candidate_updates(
            [scene_card.suspicion_delta]
        )

        leverage_updates = self._merge_candidate_updates(
            [scene_card.power_shift]
        )

        moral_lines_crossed_to_add = self._detect_moral_line_crossings(
            scene_card=scene_card,
            scene_text=scene_text,
        )

        costs_to_add = self._merge_candidate_updates(
            [scene_card.cost_paid]
            + self._select_matching_lines(
                scene_card.continuity_outputs
                + [scene_card.conflict, scene_card.pressure_source, scene_card.secret_pressure],
                keywords=(
                    "cost",
                    "risk",
                    "threat",
                    "exposure",
                    "loss",
                    "injury",
                    "damage",
                    "suspicion",
                    "compromise",
                    "sacrifice",
                ),
            )
        )

        evidence_to_add = self._extract_evidence_like_items(
            scene_card.required_entities
            + scene_card.continuity_outputs
            + [scene_card.sensory_anchor]
        )

        promises_to_add = self._select_matching_lines(
            scene_card.continuity_outputs
            + [scene_card.ending_mode, scene_card.scene_desire, scene_card.scene_fear],
            keywords=("will", "must", "needs to", "promise", "plan", "agrees to", "vows to"),
        )

        recent_scene_summary = self._build_scene_summary(scene_card)

        return ContinuityState(
            current_day=scene_card.time_marker.strip() or continuity_state.current_day,
            current_location=scene_card.location.strip() or continuity_state.current_location,
            known_facts=self._merge_list(continuity_state.known_facts, facts_to_add, limit=50),
            open_threads=self._merge_list(continuity_state.open_threads, open_threads_to_add, limit=25),
            relationship_state=self._merge_list(
                continuity_state.relationship_state,
                relationship_updates,
                limit=25,
            ),
            suspicion_state=self._merge_list(
                continuity_state.suspicion_state,
                suspicion_updates,
                limit=25,
            ),
            leverage_state=self._merge_list(
                continuity_state.leverage_state,
                leverage_updates,
                limit=25,
            ),
            moral_lines_crossed=self._merge_list(
                continuity_state.moral_lines_crossed,
                moral_lines_crossed_to_add,
                limit=20,
            ),
            injuries_or_costs=self._merge_list(
                continuity_state.injuries_or_costs,
                costs_to_add,
                limit=25,
            ),
            evidence_or_objects=self._merge_list(
                continuity_state.evidence_or_objects,
                evidence_to_add,
                limit=30,
            ),
            unresolved_promises=self._merge_list(
                continuity_state.unresolved_promises,
                promises_to_add,
                limit=25,
            ),
            disallowed_entities=self._merge_list(
                continuity_state.disallowed_entities,
                [],
                limit=25,
            ),
            recent_scene_summaries=(continuity_state.recent_scene_summaries + [recent_scene_summary])[-3:],
            last_approved_scene_number=scene_card.scene_number,
        )

    def repair_scene(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        current_scene: str,
        global_qa_report,
        rewrite_brief: str,
    ) -> str:
        """Runs a targeted repair on an already approved scene."""

        return self.llm.text(
            system_prompt=scene_draft_system_prompt(story_spec),
            user_prompt=repair_scene_user_prompt(
                story_spec=story_spec,
                outline=outline,
                scene_card=scene_card,
                continuity_state=continuity_state,
                current_scene=current_scene,
                global_qa_report=global_qa_report,
                rewrite_brief=rewrite_brief,
            ),
            task_name=f"repair_scene_{scene_card.scene_number:02d}",
            reasoning_effort=self.config.reasoning.repair,
            temperature=self.config.rewriting_temperature,
            max_output_tokens=7_000,
        )

    def _merge_candidate_updates(self, values: list[str]) -> list[str]:
        """Normalizes candidate status strings and drops empty placeholders."""

        cleaned: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", value or "").strip()
            if not normalized:
                continue
            if normalized.lower() in {"none", "n/a", "no change", "unchanged"}:
                continue
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    def _detect_moral_line_crossings(
        self,
        *,
        scene_card: SceneCard,
        scene_text: str,
    ) -> list[str]:
        """Infers irreversible ethical thresholds crossed in the scene."""

        candidates = [
            scene_card.cost_paid,
            scene_card.secret_pressure,
            scene_card.revelation_or_shift,
            scene_card.power_shift,
            scene_card.suspicion_delta,
            scene_card.relationship_delta,
        ]

        trigger_patterns = (
            r"\blie\b",
            r"\blies\b",
            r"\blied\b",
            r"\bforges?\b",
            r"\bsteals?\b",
            r"\btheft\b",
            r"\bdeceiv(?:e|es|ed|ing)\b",
            r"\bdeception\b",
            r"\bbetray(?:s|ed|al)?\b",
            r"\bsabotag(?:e|es|ed|ing)\b",
            r"\btamper(?:s|ed|ing)?\b",
            r"\bhide\s+evidence\b",
            r"\bconceal(?:s|ed|ing)?\b",
            r"\bcover-?up\b",
            r"\bcrosses?\s+a\s+line\b",
            r"\bbreaks?\s+protocol\b",
            r"\bviolates?\s+policy\b",
            r"\bdestroys?\s+trust\b",
        )

        def _matches_any_trigger(text: str) -> bool:
            lowered = (text or "").lower()
            return any(re.search(pattern, lowered) for pattern in trigger_patterns)

        def _contains_word(text: str, word: str) -> bool:
            return bool(re.search(rf"\b{re.escape(word.lower())}\b", text.lower()))

        hits = []
        for candidate in candidates:
            if _matches_any_trigger(candidate):
                hits.append(candidate.strip())

        text_lower = scene_text.lower()
        text_triggers = [
            ("Daniel lies to Elena", ("daniel", "elena"), (r"\blie\b", r"\blies\b", r"\blied\b")),
            ("Daniel conceals evidence", ("daniel", "evidence"), (r"\bconceal(?:s|ed|ing)?\b", r"\bhide\s+evidence\b")),
            ("Daniel tampers with bank logic", ("daniel", "bank"), (r"\btamper(?:s|ed|ing)?\b",)),
            ("Daniel crosses a professional line", ("daniel", "policy"), (r"\bviolates?\b", r"\bbreaks?\s+protocol\b")),
        ]
        for label, required_terms, trigger_regexes in text_triggers:
            if all(_contains_word(text_lower, term) for term in required_terms) and any(
                re.search(pattern, text_lower) for pattern in trigger_regexes
            ):
                hits.append(label)

        deduped: list[str] = []
        for hit in hits:
            normalized = re.sub(r"\s+", " ", hit).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    def _normalize_scene_cards(
        self,
        scene_cards: list[SceneCard],
        story_spec: StorySpec,
        outline: Outline,
    ) -> list[SceneCard]:
        """Stabilizes numbering, titles, and target lengths for scene cards."""

        if len(scene_cards) != story_spec.expected_scenes:
            raise ValueError(
                f"Expected {story_spec.expected_scenes} scene cards, received {len(scene_cards)}."
            )

        ordered = sorted(scene_cards, key=lambda card: (card.chapter_number, card.scene_number))
        normalized: list[SceneCard] = []
        for index, scene_card in enumerate(ordered, start=1):
            chapter_plan = get_chapter_plan(outline, scene_card.chapter_number)
            target_words = min(2_200, max(900, scene_card.target_words))
            normalized.append(
                scene_card.model_copy(
                    update={
                        "scene_number": index,
                        "chapter_title": chapter_plan.title,
                        "target_words": target_words,
                    }
                )
            )
        return normalized

    def _merge_list(self, current: list[str], additions: list[str], *, limit: int) -> list[str]:
        """Appends unique normalized strings and caps the list."""

        merged = list(current)
        for item in additions:
            cleaned = re.sub(r"\s+", " ", item or "").strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged[-limit:]

    def _select_matching_lines(self, lines: list[str], *, keywords: tuple[str, ...]) -> list[str]:
        """Returns lines containing operational keywords."""

        selected = []
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in keywords):
                selected.append(line)
        return selected

    def _extract_evidence_like_items(self, values: list[str]) -> list[str]:
        """Pulls object-like entities into continuity."""

        keywords = (
            "file",
            "ledger",
            "account",
            "loan",
            "email",
            "badge",
            "key",
            "phone",
            "report",
            "document",
            "api",
            "patch",
            "code",
            "provision",
            "model",
            "statement",
            "notebook",
            "spreadsheet",
            "reserve",
            "reconciliation",
            "exception",
            "ticket",
            "memo",
        )
        selected = []
        for value in values:
            lower_value = value.lower()
            if any(keyword in lower_value for keyword in keywords):
                selected.append(value)
        return selected

    def _build_scene_summary(self, scene_card: SceneCard) -> str:
        """Builds a short operational summary from the scene card."""

        summary = (
            f"{scene_card.pov_character} in {scene_card.location} pursues {scene_card.scene_desire}; "
            f"{scene_card.power_shift.lower()} after {scene_card.revelation_or_shift.lower()}."
        )
        return re.sub(r"\s+", " ", summary).strip()[:220]

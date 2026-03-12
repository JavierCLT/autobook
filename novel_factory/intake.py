"""Markdown intake parsing and prompt guidance helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from novel_factory.schemas import BookIntake
from novel_factory.utils import truncate_text

FIELD_RE = re.compile(r"^(?P<key>[A-Za-z0-9_./ -]+):(?:\s*(?P<value>.*))?$")
CHARACTER_FIELD_RE = re.compile(r"^character_\d+_[a-z0-9_]+$")
RELATIONSHIP_DETAIL_KEYS = {
    "starting_state",
    "best_memory_or_shared_ritual",
    "what_makes_it_alive_on_page",
    "how_it_deteriorates",
    "what_each_person_wants_from_the_other",
    "what_each_person_refuses_to_say",
    "end_state",
}
STATIC_ALLOWED_KEYS = {
    "project_slug",
    "working_title",
    "request_type",
    "replace_existing_project",
    "existing_project_to_replace",
    "scaffold_fit",
    "what_must_change_in_scaffolding",
    "title_working",
    "one_sentence_promise",
    "genre",
    "subgenre",
    "market_position",
    "audience",
    "rating_ceiling",
    "pov",
    "tense",
    "target_words",
    "expected_chapters",
    "expected_scenes",
    "premise_core",
    "themes",
    "setting",
    "timeline_window",
    "escalation_model",
    "emotional_engine",
    "adversarial_engine",
    "moral_fault_line",
    "ending_shape",
    "hook",
    "first_major_turn",
    "midpoint_turn",
    "dark_night_turn",
    "climax",
    "resolution",
    "final_image",
    "name",
    "age",
    "role",
    "public_face",
    "private_need",
    "fear",
    "contradiction",
    "external_goal",
    "inner_wound_or_need",
    "secret_pressure",
    "what_they_are_hiding",
    "what_they_will_lose",
    "why_they_cannot_walk_away",
    "name_or_force",
    "public_role",
    "private_goal",
    "method",
    "why_they_are_dangerous",
    "how_they_apply_pressure",
    "how_they_change_over_time",
    "what_they_correctly_understand_about_the_protagonist",
    "primary_relationship_1",
    "primary_relationship_2",
    "primary_locations",
    "social_or_institutional_environment",
    "what_the_world_rewards",
    "what_the_world_punishes",
    "world_rules_or_professional_rules_that_matter",
    "specialized_domains_the_book_must_handle_correctly",
    "research_sensitive_areas",
    "must_have_scenes",
    "must_have_reveals",
    "must_have_images_or_motifs",
    "must_keep_facts_from_synopsis",
    "must_not_happen",
    "forbidden_tropes",
    "forbidden_entities_or_plot_devices",
    "prose_traits",
    "banned_tells",
    "dialogue_rules",
    "narration_rules",
    "sensory_preferences",
    "things_that_should_feel_hotter_or_sharper",
    "things_that_should_feel_restrained",
    "things_that_make_you_say_this_sounds_ai",
    "banned_content",
    "violence_ceiling",
    "sexual_content_ceiling",
    "profanity_ceiling",
    "topics_to_handle_carefully",
    "continuity_rules",
    "facts_that_must_never_change",
    "timeline_constraints",
    "character_knowledge_constraints",
    "objects_or_evidence_that_must_track_cleanly",
    "ideal_reader",
    "primary_sales_category",
    "secondary_sales_category",
    "what_it_should_feel_like_in_the_market",
    "what_it_must_not_feel_like",
    "synopsis",
    "notes_to_codex",
}


@dataclass(frozen=True)
class IntakePlanningDefaults:
    """Resolved planning defaults derived from intake or config values."""

    audience: str
    rating_ceiling: str
    market_position: str
    target_words: int
    expected_chapters: int
    expected_scenes: int


def parse_book_intake(markdown_text: str) -> BookIntake:
    """Parses a filled markdown intake template into a normalized field map."""

    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    relationship_prefix: str | None = None

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key is None:
            return
        fields[current_key] = _normalize_value(current_lines)
        current_key = None
        current_lines = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            flush()
            relationship_prefix = None
            continue
        match = FIELD_RE.match(line.strip())
        if match and not line.lstrip().startswith("- "):
            normalized_key = _normalize_key(match.group("key"))
            if not _is_allowed_key(normalized_key):
                if current_key is not None:
                    current_lines.append(line.strip())
                continue
            flush()
            if normalized_key in {"primary_relationship_1", "primary_relationship_2"}:
                relationship_prefix = normalized_key
            elif normalized_key not in RELATIONSHIP_DETAIL_KEYS:
                relationship_prefix = None
            if relationship_prefix and normalized_key in RELATIONSHIP_DETAIL_KEYS:
                normalized_key = f"{relationship_prefix}_{normalized_key}"
            current_key = normalized_key
            first_value = match.group("value") or ""
            current_lines = [first_value] if first_value else []
            continue
        if current_key is None:
            continue
        current_lines.append(line.strip())

    flush()
    return BookIntake(raw_markdown=markdown_text.strip(), fields=fields)


def get_field(intake: BookIntake | None, key: str, default: str = "") -> str:
    """Returns a normalized string field from the parsed intake."""

    if intake is None:
        return default
    value = intake.fields.get(_normalize_key(key), default)
    return _strip_single_bullet(value).strip() or default


def get_int_field(intake: BookIntake | None, key: str, default: int) -> int:
    """Returns an integer field from the parsed intake when available."""

    raw_value = get_field(intake, key)
    if not raw_value:
        return default
    match = re.search(r"\d+", raw_value.replace(",", ""))
    return int(match.group(0)) if match else default


def resolve_planning_defaults(
    *,
    intake: BookIntake | None,
    default_audience: str,
    default_rating_ceiling: str,
    default_market_position: str,
    default_target_words: int,
    default_expected_chapters: int,
    default_expected_scenes: int,
) -> IntakePlanningDefaults:
    """Resolves planning defaults from intake first, then config defaults."""

    return IntakePlanningDefaults(
        audience=get_field(intake, "audience", default_audience),
        rating_ceiling=get_field(intake, "rating_ceiling", default_rating_ceiling),
        market_position=get_field(intake, "market_position", default_market_position),
        target_words=get_int_field(intake, "target_words", default_target_words),
        expected_chapters=get_int_field(intake, "expected_chapters", default_expected_chapters),
        expected_scenes=get_int_field(intake, "expected_scenes", default_expected_scenes),
    )


def build_planning_guidance(intake: BookIntake | None, *, max_chars: int = 14_000) -> str:
    """Builds a rich planning guidance block from a parsed intake."""

    if intake is None:
        return ""

    sections = [
        ("Request mode", _join_fields(intake, "request_type", "scaffold_fit", "what_must_change_in_scaffolding")),
        (
            "Core metadata",
            _join_fields(
                intake,
                "title_working",
                "one_sentence_promise",
                "genre",
                "subgenre",
                "market_position",
                "audience",
                "rating_ceiling",
                "pov",
                "tense",
                "target_words",
                "expected_chapters",
                "expected_scenes",
            ),
        ),
        (
            "Story contract",
            _join_fields(
                intake,
                "premise_core",
                "themes",
                "setting",
                "timeline_window",
                "escalation_model",
                "emotional_engine",
                "adversarial_engine",
                "moral_fault_line",
                "ending_shape",
            ),
        ),
        (
            "Plot anchors",
            _join_fields(
                intake,
                "hook",
                "first_major_turn",
                "midpoint_turn",
                "dark_night_turn",
                "climax",
                "resolution",
                "final_image",
            ),
        ),
        (
            "Protagonist",
            _join_fields(
                intake,
                "name",
                "age",
                "role",
                "public_face",
                "private_need",
                "fear",
                "contradiction",
                "external_goal",
                "inner_wound_or_need",
                "secret_pressure",
                "what_they_are_hiding",
                "what_they_will_lose",
                "why_they_cannot_walk_away",
            ),
        ),
        (
            "Counterforce",
            _join_fields(
                intake,
                "name_or_force",
                "public_role",
                "private_goal",
                "method",
                "why_they_are_dangerous",
                "how_they_apply_pressure",
                "how_they_change_over_time",
                "what_they_correctly_understand_about_the_protagonist",
            ),
        ),
        (
            "Relationship engine",
            _join_fields(
                intake,
                "primary_relationship_1",
                "primary_relationship_1_starting_state",
                "primary_relationship_1_best_memory_or_shared_ritual",
                "primary_relationship_1_what_makes_it_alive_on_page",
                "primary_relationship_1_how_it_deteriorates",
                "primary_relationship_1_what_each_person_wants_from_the_other",
                "primary_relationship_1_what_each_person_refuses_to_say",
                "primary_relationship_1_end_state",
                "primary_relationship_2",
                "primary_relationship_2_starting_state",
                "primary_relationship_2_best_memory_or_shared_ritual",
                "primary_relationship_2_what_makes_it_alive_on_page",
                "primary_relationship_2_how_it_deteriorates",
                "primary_relationship_2_what_each_person_wants_from_the_other",
                "primary_relationship_2_what_each_person_refuses_to_say",
                "primary_relationship_2_end_state",
            ),
        ),
        (
            "Non-negotiables",
            _join_fields(
                intake,
                "must_have_scenes",
                "must_have_reveals",
                "must_have_images_or_motifs",
                "must_keep_facts_from_synopsis",
                "must_not_happen",
                "forbidden_tropes",
                "forbidden_entities_or_plot_devices",
            ),
        ),
        (
            "Style guide",
            _join_fields(
                intake,
                "prose_traits",
                "banned_tells",
                "dialogue_rules",
                "narration_rules",
                "sensory_preferences",
                "things_that_should_feel_hotter_or_sharper",
                "things_that_should_feel_restrained",
                "things_that_make_you_say_this_sounds_ai",
            ),
        ),
        (
            "Continuity and notes",
            _join_fields(
                intake,
                "continuity_rules",
                "facts_that_must_never_change",
                "timeline_constraints",
                "character_knowledge_constraints",
                "objects_or_evidence_that_must_track_cleanly",
                "notes_to_codex",
            ),
        ),
    ]
    return truncate_text(_render_sections(sections), max_chars)


def build_drafting_guidance(intake: BookIntake | None, *, max_chars: int = 5_000) -> str:
    """Builds a compact drafting guidance block from the intake."""

    if intake is None:
        return ""

    sections = [
        (
            "Project target",
            _join_fields(
                intake,
                "title_working",
                "one_sentence_promise",
                "market_position",
                "pov",
                "tense",
            ),
        ),
        (
            "Protagonist and counterforce",
            _join_fields(
                intake,
                "name",
                "external_goal",
                "fear",
                "secret_pressure",
                "name_or_force",
                "how_they_apply_pressure",
            ),
        ),
        (
            "Relationship residue",
            _join_fields(
                intake,
                "primary_relationship_1",
                "primary_relationship_1_best_memory_or_shared_ritual",
                "primary_relationship_1_what_makes_it_alive_on_page",
                "primary_relationship_1_how_it_deteriorates",
                "primary_relationship_1_end_state",
            ),
        ),
        (
            "Style and bans",
            _join_fields(
                intake,
                "prose_traits",
                "banned_tells",
                "dialogue_rules",
                "narration_rules",
                "things_that_make_you_say_this_sounds_ai",
                "must_not_happen",
                "forbidden_tropes",
                "forbidden_entities_or_plot_devices",
            ),
        ),
        (
            "Operational notes",
            _join_fields(
                intake,
                "must_have_images_or_motifs",
                "continuity_rules",
                "notes_to_codex",
            ),
        ),
    ]
    return truncate_text(_render_sections(sections), max_chars)


def _join_fields(intake: BookIntake, *keys: str) -> str:
    parts = []
    for key in keys:
        value = get_field(intake, key)
        if value:
            parts.append(f"- {key}: {value}")
    return "\n".join(parts)


def _render_sections(sections: list[tuple[str, str]]) -> str:
    rendered = []
    for title, content in sections:
        if not content.strip():
            continue
        rendered.append(f"{title}:\n{content}")
    return "\n\n".join(rendered).strip()


def _normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    normalized = normalized.replace("/", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _normalize_value(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        value = line.rstrip()
        if not value and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(value)
    return "\n".join(cleaned).strip()


def _strip_single_bullet(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("- ") and "\n" not in stripped:
        return stripped[2:].strip()
    return stripped


def _is_allowed_key(normalized_key: str) -> bool:
    return normalized_key in STATIC_ALLOWED_KEYS or normalized_key in RELATIONSHIP_DETAIL_KEYS or bool(
        CHARACTER_FIELD_RE.match(normalized_key)
    )

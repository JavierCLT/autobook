"""Pydantic schemas for planning, drafting, QA, and checkpoint artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class ArtifactModel(BaseModel):
    """Base model for persisted artifacts."""

    model_config = ConfigDict(extra="forbid")


class StyleGuide(ArtifactModel):
    """Defines prose standards for the manuscript."""

    prose_traits: List[str] = Field(default_factory=list)
    banned_tells: List[str] = Field(default_factory=list)
    dialogue_rules: List[str] = Field(default_factory=list)
    narration_rules: List[str] = Field(default_factory=list)


class CharacterCard(ArtifactModel):
    """Defines a cast member's story function and internal pressure."""

    name: str
    age: str
    role: str
    public_face: str
    private_need: str
    fear: str
    contradiction: str
    relationships: List[str] = Field(default_factory=list)


class StorySpec(ArtifactModel):
    """Locked story contract produced from the synopsis."""

    title_working: str
    one_sentence_promise: str
    genre: str
    subgenre: str
    audience: str
    rating_ceiling: str
    pov: str
    tense: str
    target_words: int
    expected_chapters: int
    expected_scenes: int
    themes: List[str] = Field(default_factory=list)
    setting: str
    timeline_window: str
    premise_core: str
    escalation_model: str
    emotional_engine: str
    adversarial_engine: str
    moral_fault_line: str
    ending_shape: str
    style_guide: StyleGuide
    cast: List[CharacterCard] = Field(default_factory=list)
    continuity_rules: List[str] = Field(default_factory=list)
    banned_content: List[str] = Field(default_factory=list)


class ChapterSceneBeat(ArtifactModel):
    """One outline beat inside a chapter plan."""

    slot_number: int
    purpose: str
    beat_summary: str
    target_words: int


class ChapterPlan(ArtifactModel):
    """High-level chapter plan used during drafting and assembly."""

    chapter_number: int
    title: str
    purpose: str
    target_words: int
    scenes: List[ChapterSceneBeat] = Field(default_factory=list)


class Outline(ArtifactModel):
    """Complete macro outline for the novel."""

    hook: str
    midpoint_turn: str
    dark_night_turn: str
    climax: str
    resolution: str
    chapters: List[ChapterPlan] = Field(default_factory=list)


class SceneCard(ArtifactModel):
    """Locked scene contract for drafting."""

    chapter_number: int
    scene_number: int
    chapter_title: str
    location: str
    time_marker: str
    pov_character: str
    scene_type: str
    opening_disturbance: str
    mid_scene_reversal: str
    visible_decision: str
    closing_choice: str
    counterforce_trace: str

    purpose: str
    conflict: str
    pressure_source: str
    emotional_turn: str
    revelation_or_shift: str

    scene_desire: str
    scene_fear: str
    secret_pressure: str
    subtext_engine: str
    power_shift: str
    relationship_delta: str
    cost_paid: str
    suspicion_delta: str
    sensory_anchor: str

    continuity_inputs: List[str] = Field(default_factory=list)
    continuity_outputs: List[str] = Field(default_factory=list)
    required_entities: List[str] = Field(default_factory=list)
    forbidden_entities: List[str] = Field(default_factory=list)

    ending_mode: str
    target_words: int


class SceneCardCollection(ArtifactModel):
    """Wrapper model for structured scene-card generation."""

    scene_cards: List[SceneCard] = Field(default_factory=list)


class ContinuityState(ArtifactModel):
    """Accumulated continuity facts after approved scenes."""

    current_day: str
    current_location: str
    known_facts: List[str] = Field(default_factory=list)
    open_threads: List[str] = Field(default_factory=list)
    relationship_state: List[str] = Field(default_factory=list)
    suspicion_state: List[str] = Field(default_factory=list)
    leverage_state: List[str] = Field(default_factory=list)
    moral_lines_crossed: List[str] = Field(default_factory=list)
    injuries_or_costs: List[str] = Field(default_factory=list)
    evidence_or_objects: List[str] = Field(default_factory=list)
    unresolved_promises: List[str] = Field(default_factory=list)
    disallowed_entities: List[str] = Field(default_factory=list)
    recent_scene_summaries: List[str] = Field(default_factory=list)
    last_approved_scene_number: int = 0


class ContinuityUpdate(ArtifactModel):
    """Compact continuity delta merged locally after each approved scene."""

    current_day: str
    current_location: str
    facts_to_add: List[str] = Field(default_factory=list)
    open_threads_to_add: List[str] = Field(default_factory=list)
    open_threads_to_close: List[str] = Field(default_factory=list)
    relationship_updates: List[str] = Field(default_factory=list)
    suspicion_updates: List[str] = Field(default_factory=list)
    leverage_updates: List[str] = Field(default_factory=list)
    moral_lines_crossed_to_add: List[str] = Field(default_factory=list)
    injuries_or_costs_to_add: List[str] = Field(default_factory=list)
    evidence_or_objects_to_add: List[str] = Field(default_factory=list)
    unresolved_promises_to_add: List[str] = Field(default_factory=list)
    unresolved_promises_to_close: List[str] = Field(default_factory=list)
    disallowed_entities_to_add: List[str] = Field(default_factory=list)
    recent_scene_summary: str


class ValidationFinding(ArtifactModel):
    """One deterministic validator finding."""

    check_id: str
    severity: Literal["error", "warning"]
    message: str
    details: List[str] = Field(default_factory=list)


class DeterministicValidationReport(ArtifactModel):
    """Aggregated local validation results for a scene draft."""

    pass_fail: bool
    findings: List[ValidationFinding] = Field(default_factory=list)
    word_count: int
    dialogue_ratio: float
    dominant_sentence_openings: List[str] = Field(default_factory=list)
    repeated_phrases: List[str] = Field(default_factory=list)


class PlanValidationReport(ArtifactModel):
    """Aggregated local validation results for planning artifacts."""

    pass_fail: bool
    findings: List[ValidationFinding] = Field(default_factory=list)


class SceneQaReport(ArtifactModel):
    """Model-based QA verdict for a drafted scene."""

    pass_fail: bool
    continuity_score: int = Field(ge=1, le=5)
    engagement_score: int = Field(ge=1, le=5)
    voice_score: int = Field(ge=1, le=5)
    pacing_score: int = Field(ge=1, le=5)
    specificity_score: int = Field(ge=1, le=5)
    prose_freshness_score: int = Field(ge=1, le=5)
    emotional_movement_score: int = Field(ge=1, le=5)
    subtext_score: int = Field(ge=1, le=5)
    concealment_score: int = Field(ge=1, le=5)
    leverage_shift_score: int = Field(ge=1, le=5)
    relationship_cost_score: int = Field(ge=1, le=5)
    commercial_hook_score: int = Field(ge=1, le=5)
    ai_smell_score: int = Field(ge=1, le=5)
    hard_fail_reasons: List[str] = Field(default_factory=list)
    soft_issues: List[str] = Field(default_factory=list)
    rewrite_brief: str
    deterministic_pass: bool = True
    deterministic_findings: List[ValidationFinding] = Field(default_factory=list)


class RepairTarget(ArtifactModel):
    """Minimum scene-level repair needed to fix a global QA failure."""

    scene_number: int
    reason: str
    rewrite_brief: str


class ChapterQaReport(ArtifactModel):
    """Chapter-level editorial QA verdict."""

    chapter_number: int
    pass_fail: bool
    suspense_score: int = Field(ge=1, le=5)
    emotional_voltage_score: int = Field(ge=1, le=5)
    clarity_score: int = Field(ge=1, le=5)
    ending_drive_score: int = Field(ge=1, le=5)
    major_problems: List[str] = Field(default_factory=list)
    repair_targets: List[RepairTarget] = Field(default_factory=list)


class ArcQaReport(ArtifactModel):
    """Arc-level editorial QA verdict for a targeted scene cluster."""

    arc_name: str
    scene_numbers: List[int] = Field(default_factory=list)
    pass_fail: bool
    propulsion_score: int = Field(ge=1, le=5)
    pressure_score: int = Field(ge=1, le=5)
    relationship_score: int = Field(ge=1, le=5)
    payoff_readiness_score: int = Field(ge=1, le=5)
    major_problems: List[str] = Field(default_factory=list)
    repair_targets: List[RepairTarget] = Field(default_factory=list)


class GlobalQaReport(ArtifactModel):
    """Global manuscript-level QA verdict."""

    pass_fail: bool
    hook_strength_score: int = Field(ge=1, le=5)
    midpoint_turn_score: int = Field(ge=1, le=5)
    climax_payoff_score: int = Field(ge=1, le=5)
    ending_payoff_score: int = Field(ge=1, le=5)
    relationship_progression_score: int = Field(ge=1, le=5)
    antagonist_pressure_score: int = Field(ge=1, le=5)
    continuity_score: int = Field(ge=1, le=5)
    emotional_aftershock_score: int = Field(ge=1, le=5)
    boredom_risk_score: int = Field(ge=1, le=5)
    voice_consistency_score: int = Field(ge=1, le=5)
    ai_smell_score: int = Field(ge=1, le=5)
    major_problems: List[str] = Field(default_factory=list)
    repair_targets: List[RepairTarget] = Field(default_factory=list)


class BookIntake(ArtifactModel):
    """Parsed markdown intake template plus normalized field map."""

    raw_markdown: str
    fields: Dict[str, str] = Field(default_factory=dict)


class RunLogEvent(ArtifactModel):
    """One append-only JSONL log entry for the run."""

    timestamp: str
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)

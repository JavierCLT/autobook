"""Centralized prompts for planning, drafting, QA, and repair."""

from __future__ import annotations

from typing import Iterable, Optional

from novel_factory.schemas import (
    ContinuityState,
    ContinuityUpdate,
    DeterministicValidationReport,
    GlobalQaReport,
    Outline,
    SceneCard,
    StorySpec,
)
from novel_factory.utils import serialise_model


def global_prose_policy(*, audience: str, rating_ceiling: str, market_position: str) -> str:
    """Builds the standing prose policy for the current audience and market."""
    lower_audience = audience.lower()
    lower_rating = rating_ceiling.lower()

    if "young adult" in lower_audience or lower_audience == "ya" or lower_rating == "pg-13":
        audience_rules = (
            "- Keep all content YA-appropriate.\n"
            f"- Respect the requested rating ceiling: {rating_ceiling}.\n"
            "- No explicit sex, graphic violence, gore, or strong profanity.\n"
            "- Emotional intensity, danger, and obsession are allowed, but the material must remain suitable for a YA shelf."
        )
    else:
        audience_rules = (
            f"- Write for an {market_position} audience.\n"
            f"- Respect the requested rating ceiling: {rating_ceiling}.\n"
            "- Mature institutions, careers, moral ambiguity, adult relationships, and professional betrayal are allowed.\n"
            "- Avoid gratuitous gore, exploitative sexual content, or profanity that feels performative."
        )

    return (
        "You are writing a serious trade thriller.\n\n"
        "Hard constraints:\n"
        f"{audience_rules}\n"
        "- Do not imitate or reference living authors.\n\n"
        "Style priorities:\n"
        "- Precision with pressure.\n"
        "- Controlled heat: dangerous, intimate, specific.\n"
        "- Human stakes must be inseparable from institutional or plot stakes.\n"
        "- Each scene should contain active desire, resistance, concealment, leverage shift, and cost.\n"
        "- Concrete sensory grounding and behavioral subtext.\n"
        "- High readability without flattening tension into explanation.\n\n"
        "Avoid:\n"
        "- Ice-cold competence for its own sake.\n"
        "- Exposition sludge, filler, fake-deep introspection, or tidy explanatory dialogue.\n"
        "- Summary paragraphs that replace drama.\n"
        "- Decorative subtext explanation or dialogue that states exactly what the scene should imply.\n"
        "- Recycled emotional beats, canned thriller imagery, or mechanically wrapped-up endings.\n"
        "- Overuse of contrast pivots such as \"but,\" \"still,\" \"instead,\" and \"even so.\"\n"
        "- Letting restraint become anesthesia."
    )


def planning_system_prompt(
    *,
    audience: str,
    rating_ceiling: str,
    market_position: str,
) -> str:
    """Returns the system prompt for structured planning artifacts."""
    return (
        f"{global_prose_policy(audience=audience, rating_ceiling=rating_ceiling, market_position=market_position)}\n\n"
        "You are generating strict planning artifacts for a multi-stage novel pipeline.\n"
        "Honor the synopsis. Lock choices early. Prefer sharp, concrete, operational decisions.\n"
        "Build the story around pressure, secrecy, changing leverage, and consequence rather than abstract elegance.\n"
        "Do not include chain-of-thought. Return only schema-compatible content."
    )


def story_spec_user_prompt(
    synopsis: str,
    target_words: int,
    chapters: int,
    scenes: int,
    audience: str,
    rating_ceiling: str,
    market_position: str,
) -> str:
    """Builds the story-spec prompt."""
    return f"""
Create a locked StorySpec from the synopsis below.

Non-negotiable defaults:
- Audience: {audience}
- Rating ceiling: {rating_ceiling}
- Target words: about {target_words}
- Expected chapters: {chapters}
- Expected scenes: {scenes}
- The story must read like a high-quality {market_position} rather than a generic suspense template.

Requirements:
- Make the premise, antagonism, and escalation concrete.
- Lock an emotional_engine, adversarial_engine, and moral_fault_line.
- Keep the cast tight and functional.
- The protagonist must want something concrete, fear something concrete, and hide something concrete.
- Encode anti-cliche style rules in the style guide.
- Put explicit audience-appropriate bans in banned_content.
- Continuity rules should be operational and testable later.
- The one_sentence_promise should sell both the external thriller engine and the intimate personal cost.

Synopsis:
{synopsis}
""".strip()


def outline_user_prompt(synopsis: str, story_spec: StorySpec) -> str:
    """Builds the outline prompt."""
    return f"""
Create a detailed Outline from the locked StorySpec and synopsis.

Requirements:
- Deliver {story_spec.expected_chapters} chapters.
- Sustain pressure and escalation across the full novel.
- Give each chapter a distinct dramatic purpose.
- Each chapter should contain two scene beats by default unless a different split is dramatically necessary.
- Build a strong hook, a consequential midpoint turn, a dark-night turn, a payoff-rich climax, and a clean but not overly tidy resolution.
- Every 2-3 scenes, advance at least one of these engines: adversarial pressure, relationship deterioration, moral compromise.
- Do not let the middle become procedural drift. Each chapter should sharpen danger, intimacy, or irreversible consequence.
- Make sure the hunter / pursuer / institutional-counterforce grows more intelligent over time.

StorySpec:
{serialise_model(story_spec)}

Synopsis:
{synopsis}
""".strip()


def scene_cards_user_prompt(synopsis: str, story_spec: StorySpec, outline: Outline) -> str:
    """Builds the scene-card prompt."""
    return f"""
Expand the locked outline into exactly {story_spec.expected_scenes} scene cards.

Requirements:
- Each scene card must feel necessary and non-filler.
- Default to about two scenes per chapter and preserve chapter numbering.
- Each scene must have a specific conflict, pressure source, emotional turn, and end-mode.
- Each scene must also define:
  - scene_desire: what the POV character wants right now
  - scene_fear: what the POV character most wants to avoid right now
  - secret_pressure: what the POV character is hiding or suppressing
  - subtext_engine: what cannot be said openly in the scene
  - power_shift: who gains or loses leverage by the end
  - relationship_delta: how a key relationship worsens, improves, or hardens
  - cost_paid: what this scene costs someone
  - suspicion_delta: who becomes more or less suspicious, and why
  - sensory_anchor: one concrete physical detail that grounds the scene
- Required entities should list the people, objects, or facts that must appear on-page.
- Forbidden entities should list people, objects, or facts that must not appear in the scene.
- continuity_inputs should specify what prior facts the drafter must preserve.
- continuity_outputs should specify what new facts the continuity tracker must record.
- At least one meaningful thing must worsen, tighten, or become harder by the end of each scene.
- Every 2-3 scenes, at least one of these must intensify: Marta's suspicion, Elena's withdrawal, or Daniel's moral compromise. If none intensify, the scene is probably decorative and should be cut or merged.
- By scene 6 at latest, the counterforce must leave an on-page trace, rumor, memo, review footprint, or other concrete shadow.
- Keep target_words between 900 and 2200 unless the dramatic function strongly justifies otherwise.

StorySpec:
{serialise_model(story_spec)}

Outline:
{serialise_model(outline)}

Synopsis:
{synopsis}
""".strip()


def initial_continuity_user_prompt(story_spec: StorySpec, outline: Outline) -> str:
    """Builds the prompt for the initial continuity state."""
    return f"""
Create the initial ContinuityState for this project before scene 1 is drafted.

Requirements:
- Set only facts that are already locked by the story contract and outline.
- Keep known_facts concise and operational.
- open_threads should hold active unanswered pressures the novel must later resolve.
- relationship_state should be a compact list of relationship status lines describing the current status of important bonds.
- suspicion_state should identify who currently suspects what, if anything.
- leverage_state should identify who currently holds power over whom.
- moral_lines_crossed should start empty unless the synopsis already locks in an irreversible compromise before scene 1.
- disallowed_entities should include people or devices that must not appear before the outline introduces them.
- recent_scene_summaries must be empty.
- last_approved_scene_number must be 0.

StorySpec:
{serialise_model(story_spec)}

Outline:
{serialise_model(outline)}
""".strip()


def scene_draft_system_prompt(story_spec: StorySpec) -> str:
    """Returns the system prompt for prose drafting."""
    return (
        f"{global_prose_policy(audience=story_spec.audience, rating_ceiling=story_spec.rating_ceiling, market_position=story_spec.subgenre or story_spec.genre)}\n\n"
        "Write one complete scene in polished prose.\n"
        "The scene must earn its place, dramatize pressure, and move the story forward.\n"
        "Every scene must contain an active want, credible resistance, concealed pressure, a leverage shift, and a concrete cost.\n"
        "Subtext should be legible through behavior, rhythm, omission, and action rather than explained directly.\n"
        "Return only the scene text in markdown-ready prose without commentary or notes."
    )


def scene_draft_user_prompt(
    story_spec: StorySpec,
    outline: Outline,
    scene_card: SceneCard,
    continuity_state: ContinuityState,
    recent_scene_summaries: Iterable[str],
    rewrite_brief: Optional[str] = None,
    current_draft: Optional[str] = None,
) -> str:
    """Builds the prompt for initial drafts and rewrites."""
    chapter_plan = next(
        chapter for chapter in outline.chapters if chapter.chapter_number == scene_card.chapter_number
    )
    summaries_block = "\n".join(f"- {summary}" for summary in recent_scene_summaries) or "- None yet"
    rewrite_block = rewrite_brief or "None. Draft this cleanly on the first pass."
    current_draft_block = current_draft or "No prior draft."

    return f"""
Write Scene {scene_card.scene_number} of the novel.

Scene must achieve:
- Chapter: {scene_card.chapter_number} ({scene_card.chapter_title})
- POV: {scene_card.pov_character}
- Location: {scene_card.location}
- Time marker: {scene_card.time_marker}
- Purpose: {scene_card.purpose}
- Conflict: {scene_card.conflict}
- Pressure source: {scene_card.pressure_source}
- Emotional turn: {scene_card.emotional_turn}
- Revelation or shift: {scene_card.revelation_or_shift}
- Scene desire: {scene_card.scene_desire}
- Scene fear: {scene_card.scene_fear}
- Secret pressure: {scene_card.secret_pressure}
- Subtext engine: {scene_card.subtext_engine}
- Power shift: {scene_card.power_shift}
- Relationship delta: {scene_card.relationship_delta}
- Cost paid: {scene_card.cost_paid}
- Suspicion delta: {scene_card.suspicion_delta}
- Sensory anchor: {scene_card.sensory_anchor}
- Ending mode: {scene_card.ending_mode}
- Target words: about {scene_card.target_words}

Continuity requirements:
- Preserve these inputs: {scene_card.continuity_inputs}
- Produce these outputs: {scene_card.continuity_outputs}
- Required entities: {scene_card.required_entities}
- Forbidden entities: {scene_card.forbidden_entities}
- Additional disallowed entities from continuity: {continuity_state.disallowed_entities}
- continuity_inputs and continuity_outputs are planning constraints, not prose lines.
- Never quote or lightly paraphrase them as narration.
- If a required entity is abstract, realize it naturally in action, implication, or dialogue rather than quoting the scene-card wording.
- Use the sensory_anchor as grounding, not as a line to be copied verbatim.
- Realize the subtext_engine indirectly; do not explain it.
- Surface the POV character's desire and fear within the first 250 words through action or choice, not later explanation.
- The POV character must withhold, misdirect, or pay an interpersonal price on-page; if the scene can be paraphrased as competent analysis, it is under-dramatized.

Avoid:
- Plot-summary paragraphs.
- Dialogue that exists only to explain what both characters already know.
- Generic threat language, repeated emotional shorthand, filler beats, or fake introspection.
- Scenes that stay technically competent but emotionally inert.
- Endings that dissolve into reflection when the scene card requires a sharper landing.
- Lifting scene-card phrasing verbatim into dialogue or narration.
- Bullet lists, headings, numbered rules, or notebook-style outline formatting inside the prose scene.

Recent approved scene summaries:
{summaries_block}

StorySpec:
{serialise_model(story_spec)}

Relevant chapter plan:
{serialise_model(chapter_plan)}

Current continuity state:
{serialise_model(continuity_state)}

Rewrite brief:
{rewrite_block}

Current draft to revise if present:
{current_draft_block}
""".strip()


def continuity_update_system_prompt() -> str:
    """Returns the system prompt for continuity updates."""
    return (
        "Update continuity with discipline. Return only a compact delta, not the full state. "
        "Record only changes supported by the approved scene and keep every field terse and operational."
    )


def continuity_update_user_prompt(
    story_spec: StorySpec,
    scene_card: SceneCard,
    continuity_state: ContinuityState,
    scene_text: str,
) -> str:
    """Builds the prompt for continuity updates after an approved scene."""
    return f"""
Create a compact ContinuityUpdate after the approved scene below.

Requirements:
- Return only deltas that should be merged into the prior continuity.
- current_day and current_location must reflect the post-scene state.
- facts_to_add should contain only newly relevant facts.
- open_threads_to_close should list threads explicitly resolved in this scene.
- relationship_updates should be terse status lines.
- suspicion_updates should be terse status lines naming who now suspects more or less, and of what.
- leverage_updates should be terse status lines naming who now holds more or less power.
- moral_lines_crossed_to_add should record irreversible ethical thresholds crossed on-page.
- recent_scene_summary must be one concise operational sentence, 30 words or fewer.
- Do not invent off-page events.

StorySpec:
{serialise_model(story_spec)}

SceneCard:
{serialise_model(scene_card)}

Prior continuity:
{serialise_model(continuity_state)}

Approved scene:
{scene_text}
""".strip()


def scene_qa_system_prompt(story_spec: StorySpec) -> str:
    """Returns the system prompt for scene QA."""
    return (
        f"{global_prose_policy(audience=story_spec.audience, rating_ceiling=story_spec.rating_ceiling, market_position=story_spec.subgenre or story_spec.genre)}\n\n"
        "Judge one drafted scene for a strict production pipeline.\n"
        "Score continuity, engagement, voice, pacing, specificity, prose freshness, and emotional movement "
        "from 1 to 5, where 5 is best.\n"
        "Score ai_smell_score from 1 to 5, where 1 means low AI-smell risk and 5 means high AI-smell risk.\n"
        "Fail any scene with a real continuity break, filler function, weak ending against the scene card, "
        "mechanical exposition, missing required entities, forbidden entity usage, flat subtext, or pressure that does not materially move.\n"
        "The rewrite_brief must be operational and specific."
    )


def scene_qa_user_prompt(
    story_spec: StorySpec,
    scene_card: SceneCard,
    continuity_state: ContinuityState,
    validation_report: DeterministicValidationReport,
    scene_text: str,
) -> str:
    """Builds the prompt for scene-level QA."""
    return f"""
Evaluate the drafted scene against the contract below.

StorySpec:
{serialise_model(story_spec)}

SceneCard:
{serialise_model(scene_card)}

Continuity state before approval:
{serialise_model(continuity_state)}

Deterministic validation report:
{serialise_model(validation_report)}

Drafted scene:
{scene_text}
""".strip()


def global_qa_system_prompt(story_spec: StorySpec) -> str:
    """Returns the system prompt for manuscript-level QA."""
    return (
        f"{global_prose_policy(audience=story_spec.audience, rating_ceiling=story_spec.rating_ceiling, market_position=story_spec.subgenre or story_spec.genre)}\n\n"
        "Judge the full manuscript with a professional editorial mindset.\n"
        "Score hook_strength_score, midpoint_turn_score, climax_payoff_score, ending_payoff_score, "
        "relationship_progression_score, antagonist_pressure_score, continuity_score, emotional_aftershock_score, "
        "and voice_consistency_score from 1 to 5, where 5 is best.\n"
        "Score boredom_risk_score and ai_smell_score from 1 to 5, where 1 is best and 5 is worst.\n"
        "If the manuscript fails, propose the minimum set of scene-level repairs required to recover it."
    )


def global_qa_user_prompt(
    story_spec: StorySpec,
    outline: Outline,
    manuscript_text: str,
) -> str:
    """Builds the prompt for global QA."""
    return f"""
Evaluate the complete manuscript.

Requirements:
- Focus on hook strength, midpoint turn quality, climax payoff, ending payoff, continuity, relationship progression, antagonist pressure, emotional aftershock, boredom risk, voice consistency, and AI-smell risk.
- Keep repair_targets minimal and scene-specific.
- Do not ask for a total rewrite unless the manuscript is structurally unsalvageable.
- Penalize manuscripts that are intelligent but emotionally anesthetized, or technically coherent but lacking pursuit, consequence, or human cost.

StorySpec:
{serialise_model(story_spec)}

Outline:
{serialise_model(outline)}

Manuscript:
{manuscript_text}
""".strip()


def repair_scene_user_prompt(
    story_spec: StorySpec,
    outline: Outline,
    scene_card: SceneCard,
    continuity_state: ContinuityState,
    current_scene: str,
    global_qa_report: GlobalQaReport,
    rewrite_brief: str,
) -> str:
    """Builds a scene repair prompt triggered by global QA."""
    return f"""
Repair an approved scene with the minimum necessary changes.

Requirements:
- Preserve the scene's role in the novel.
- Fix the targeted issue without flattening voice.
- Respect the scene card and current continuity.
- Restore pressure, subtext, leverage movement, or relationship cost if the rewrite brief calls for it.
- Keep the scene appropriate for {story_spec.audience} readers and within a {story_spec.rating_ceiling} ceiling.
- Maintain target length near {scene_card.target_words} words.

SceneCard:
{serialise_model(scene_card)}

Continuity state at the time of repair:
{serialise_model(continuity_state)}

Global QA report:
{serialise_model(global_qa_report)}

Operational rewrite brief:
{rewrite_brief}

Current approved scene:
{current_scene}

Outline:
{serialise_model(outline)}

StorySpec:
{serialise_model(story_spec)}
""".strip()

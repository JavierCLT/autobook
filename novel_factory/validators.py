"""Deterministic validators for scene-level quality gates."""

from __future__ import annotations

import re
from collections import Counter

from novel_factory.schemas import (
    ContinuityState,
    DeterministicValidationReport,
    Outline,
    PlanValidationReport,
    SceneCard,
    StorySpec,
    ValidationFinding,
)
from novel_factory.utils import count_words, first_token, split_paragraphs, split_sentences

BANNED_CLICHE_PHRASES = [
    "heart pounded",
    "mind raced",
    "blood ran cold",
    "deafening silence",
    "couldn't help but",
    "stomach dropped",
    "for what felt like forever",
    "let out a breath",
    "every fiber of",
    "time seemed to stop",
    "eyes widened",
]

RHETORICAL_PATTERNS = [
    "as if",
    "it felt like",
    "there was",
    "there were",
    "i knew",
    "she knew",
    "he knew",
]

QUOTE_RE = re.compile(r'"([^"]+)"')
WORD_RE = re.compile(r"[A-Za-z0-9']+")
QUOTED_TERM_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


class SceneValidator:
    """Runs deterministic checks before model-based judging."""

    def validate(
        self,
        *,
        scene_text: str,
        scene_card: SceneCard,
        story_spec: StorySpec,
        continuity_state: ContinuityState,
    ) -> DeterministicValidationReport:
        """Runs all local validation checks for a scene."""

        findings: list[ValidationFinding] = []
        sentences = split_sentences(scene_text)
        paragraphs = split_paragraphs(scene_text)
        word_count = count_words(scene_text)

        findings.extend(self._word_count_findings(word_count, scene_card))
        findings.extend(self._opening_repetition_findings(sentences))
        findings.extend(self._ngram_findings(scene_text))
        findings.extend(self._cliche_findings(scene_text))
        findings.extend(self._paragraph_findings(paragraphs))
        findings.extend(self._dialogue_ratio_findings(scene_text, word_count))
        findings.extend(self._entity_findings(scene_text, scene_card, continuity_state, story_spec))
        findings.extend(self._continuity_heuristics(scene_card, continuity_state))
        findings.extend(self._rhetorical_pattern_findings(scene_text))
        findings.extend(self._punctuation_findings(scene_text))
        findings.extend(self._paragraph_start_findings(paragraphs, story_spec))

        dominant_sentence_openings = self._dominant_openings(sentences)
        repeated_phrases = self._dominant_repeated_phrases(scene_text)
        dialogue_ratio = self._dialogue_ratio(scene_text, word_count)
        has_errors = any(finding.severity == "error" for finding in findings)

        return DeterministicValidationReport(
            pass_fail=not has_errors,
            findings=findings,
            word_count=word_count,
            dialogue_ratio=round(dialogue_ratio, 3),
            dominant_sentence_openings=dominant_sentence_openings,
            repeated_phrases=repeated_phrases,
        )

    def _word_count_findings(self, word_count: int, scene_card: SceneCard) -> list[ValidationFinding]:
        floor = max(750, int(scene_card.target_words * 0.7))
        ceiling = max(2_200, int(scene_card.target_words * 1.35))
        findings: list[ValidationFinding] = []
        if word_count < floor:
            findings.append(
                ValidationFinding(
                    check_id="word_count_floor",
                    severity="error",
                    message=f"Scene is under the minimum viable word count ({word_count} < {floor}).",
                    details=[f"Target words: {scene_card.target_words}"],
                )
            )
        if word_count > ceiling:
            findings.append(
                ValidationFinding(
                    check_id="word_count_ceiling",
                    severity="warning",
                    message=f"Scene is substantially above the target range ({word_count} > {ceiling}).",
                    details=[f"Target words: {scene_card.target_words}"],
                )
            )
        return findings

    def _opening_repetition_findings(self, sentences: list[str]) -> list[ValidationFinding]:
        openings = [first_token(sentence) for sentence in sentences if first_token(sentence)]
        if len(openings) < 6:
            return []
        counts = Counter(openings)
        repeated = counts.most_common(3)
        findings: list[ValidationFinding] = []
        details = [f"{token}: {count}" for token, count in repeated if count >= 4]
        if details:
            severity = "error" if repeated[0][1] >= max(5, int(len(openings) * 0.28)) else "warning"
            findings.append(
                ValidationFinding(
                    check_id="repeated_sentence_openings",
                    severity=severity,
                    message="Sentence openings are repeating too often.",
                    details=details,
                )
            )
        return findings

    def _ngram_findings(self, scene_text: str) -> list[ValidationFinding]:
        tokens = [token.lower() for token in WORD_RE.findall(scene_text)]
        filtered = [token for token in tokens if len(token) > 2]
        findings: list[ValidationFinding] = []
        repeated_details: list[str] = []
        for size, threshold in ((2, 4), (3, 3)):
            ngrams = Counter(
                " ".join(filtered[index : index + size])
                for index in range(0, max(0, len(filtered) - size + 1))
            )
            repeated_details.extend(
                [f"{ngram}: {count}" for ngram, count in ngrams.items() if count >= threshold]
            )
        if repeated_details:
            findings.append(
                ValidationFinding(
                    check_id="repeated_ngrams",
                    severity="warning",
                    message="Repeated bigrams or trigrams suggest recycled phrasing.",
                    details=repeated_details[:8],
                )
            )
        return findings

    def _cliche_findings(self, scene_text: str) -> list[ValidationFinding]:
        lower_text = scene_text.lower()
        hits = [phrase for phrase in BANNED_CLICHE_PHRASES if phrase in lower_text]
        if not hits:
            return []
        severity = "error" if len(hits) >= 3 else "warning"
        return [
            ValidationFinding(
                check_id="banned_cliches",
                severity=severity,
                message="Scene uses banned or suspiciously generic phrases.",
                details=hits,
            )
        ]

    def _paragraph_findings(self, paragraphs: list[str]) -> list[ValidationFinding]:
        if not paragraphs:
            return [
                ValidationFinding(
                    check_id="empty_scene",
                    severity="error",
                    message="Scene draft is empty.",
                    details=[],
                )
            ]
        long_paragraphs = []
        short_paragraphs = 0
        for index, paragraph in enumerate(paragraphs, start=1):
            paragraph_words = count_words(paragraph)
            if paragraph_words > 220:
                long_paragraphs.append(f"Paragraph {index}: {paragraph_words} words")
            if paragraph_words < 8:
                short_paragraphs += 1
        findings: list[ValidationFinding] = []
        if long_paragraphs:
            findings.append(
                ValidationFinding(
                    check_id="paragraph_length_extremes",
                    severity="warning",
                    message="Paragraphs run long enough to threaten readability and pacing.",
                    details=long_paragraphs[:6],
                )
            )
        if short_paragraphs >= max(4, int(len(paragraphs) * 0.45)):
            findings.append(
                ValidationFinding(
                    check_id="too_many_short_paragraphs",
                    severity="warning",
                    message="Too many paragraphs are extremely short, which can feel mechanically chopped.",
                    details=[f"Short paragraphs: {short_paragraphs} of {len(paragraphs)}"],
                )
            )
        return findings

    def _dialogue_ratio(self, scene_text: str, word_count: int) -> float:
        if word_count <= 0:
            return 0.0
        dialogue_words = sum(count_words(match.group(1)) for match in QUOTE_RE.finditer(scene_text))
        return dialogue_words / max(1, word_count)

    def _dialogue_ratio_findings(self, scene_text: str, word_count: int) -> list[ValidationFinding]:
        ratio = self._dialogue_ratio(scene_text, word_count)
        if ratio < 0.02:
            return [
                ValidationFinding(
                    check_id="dialogue_ratio_low",
                    severity="warning",
                    message="Scene contains almost no dialogue.",
                    details=[f"Dialogue ratio: {ratio:.2f}"],
                )
            ]
        if ratio > 0.78:
            return [
                ValidationFinding(
                    check_id="dialogue_ratio_high",
                    severity="warning",
                    message="Scene is dominated by dialogue to a suspicious degree.",
                    details=[f"Dialogue ratio: {ratio:.2f}"],
                )
            ]
        return []

    def _entity_findings(
        self,
        scene_text: str,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
        story_spec: StorySpec,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        lower_text = self._normalize_match_text(scene_text)
        cast_names = {self._normalize_match_text(character.name) for character in story_spec.cast if character.name}
        forbidden_entities = list(scene_card.forbidden_entities) + list(continuity_state.disallowed_entities)
        forbidden_hits = [
            entity
            for entity in forbidden_entities
            if entity and self._normalize_match_text(entity) in lower_text
        ]
        if forbidden_hits:
            findings.append(
                ValidationFinding(
                    check_id="forbidden_entities",
                    severity="error",
                    message="Forbidden entities appear in the scene text.",
                    details=forbidden_hits,
                )
            )
        missing_required = []
        abstract_required = []
        for entity in scene_card.required_entities:
            if not entity or self._matches_required_entity(entity, lower_text, cast_names=cast_names):
                continue
            if self._is_abstract_required_entity(entity):
                abstract_required.append(entity)
            else:
                missing_required.append(entity)
        if missing_required:
            findings.append(
                ValidationFinding(
                    check_id="missing_required_entities",
                    severity="error",
                    message="Required entities are missing from the scene text.",
                    details=missing_required,
                )
            )
        if abstract_required:
            findings.append(
                ValidationFinding(
                    check_id="abstract_required_entities_unmatched",
                    severity="warning",
                    message="Some abstract scene obligations were not matched literally and should be judged semantically.",
                    details=abstract_required,
                )
            )
        return findings

    def _continuity_heuristics(
        self,
        scene_card: SceneCard,
        continuity_state: ContinuityState,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        current_location = continuity_state.current_location.lower().strip()
        scene_location = scene_card.location.lower().strip()
        transition_terms = " ".join(scene_card.continuity_inputs).lower()
        if (
            current_location
            and scene_location
            and current_location != scene_location
            and not any(term in transition_terms for term in ("travel", "arrive", "leave", "drive", "walk"))
        ):
            findings.append(
                ValidationFinding(
                    check_id="location_transition_warning",
                    severity="warning",
                    message="Location changes without an explicit transition cue in continuity_inputs.",
                    details=[f"Prior location: {continuity_state.current_location}", f"Scene location: {scene_card.location}"],
                )
            )
        return findings

    def _rhetorical_pattern_findings(self, scene_text: str) -> list[ValidationFinding]:
        lower_text = scene_text.lower()
        hits = [pattern for pattern in RHETORICAL_PATTERNS if lower_text.count(pattern) >= 4]
        if not hits:
            return []
        return [
            ValidationFinding(
                check_id="repeated_rhetorical_patterns",
                severity="warning",
                message="Repeated rhetorical scaffolding suggests AI-smelling prose habits.",
                details=hits,
            )
        ]

    def _punctuation_findings(self, scene_text: str) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        em_dashes = scene_text.count("--") + scene_text.count("\u2014")
        semicolons = scene_text.count(";")
        word_count = count_words(scene_text)
        if em_dashes > max(4, word_count // 220):
            findings.append(
                ValidationFinding(
                    check_id="emdash_overuse",
                    severity="warning",
                    message="Em dash usage is high enough to suggest stylized overcompensation.",
                    details=[f"Em dashes: {em_dashes}"],
                )
            )
        if semicolons > max(3, word_count // 260):
            findings.append(
                ValidationFinding(
                    check_id="semicolon_overuse",
                    severity="warning",
                    message="Semicolon usage is unusually high for this voice target.",
                    details=[f"Semicolons: {semicolons}"],
                )
            )
        return findings

    def _paragraph_start_findings(
        self,
        paragraphs: list[str],
        story_spec: StorySpec,
    ) -> list[ValidationFinding]:
        openings = [first_token(paragraph) for paragraph in paragraphs if first_token(paragraph)]
        if len(openings) < 5:
            return []
        counts = Counter(openings)
        dominant, dominant_count = counts.most_common(1)[0]
        cast_tokens = {character.name.split()[0].lower() for character in story_spec.cast if character.name}
        if dominant_count >= max(4, int(len(openings) * 0.55)) and (
            dominant in {"he", "she", "they", "i"} or dominant in cast_tokens
        ):
            return [
                ValidationFinding(
                    check_id="paragraph_start_repetition",
                    severity="warning",
                    message="Too many paragraphs begin with the same pronoun or character name.",
                    details=[f"{dominant}: {dominant_count} of {len(openings)} paragraphs"],
                )
            ]
        return []

    def _dominant_openings(self, sentences: list[str]) -> list[str]:
        counts = Counter(first_token(sentence) for sentence in sentences if first_token(sentence))
        return [f"{token}: {count}" for token, count in counts.most_common(5) if token]

    def _dominant_repeated_phrases(self, scene_text: str) -> list[str]:
        tokens = [token.lower() for token in WORD_RE.findall(scene_text) if len(token) > 2]
        phrases = Counter(
            " ".join(tokens[index : index + 3]) for index in range(0, max(0, len(tokens) - 2))
        )
        return [f"{phrase}: {count}" for phrase, count in phrases.most_common(5) if count > 2]

    def _is_abstract_required_entity(self, entity: str) -> bool:
        """Returns True when a required entity is really an abstract beat instruction."""

        lower_entity = entity.lower()
        abstract_markers = (
            "conversation",
            "reassurance",
            "objection",
            "language",
            "drift",
            "distance",
            "absence",
            "choice",
            "plan",
            "obligation",
            "return",
            "pressure",
            "argument",
            "confession",
            "realization",
            "shift",
            "gesture",
            "look",
            "question",
            "attempt",
            "interpretation",
            "knowledge",
            "fragment",
            "routine",
            "fact",
            "proof",
            "evidence",
            "discussion",
            "narrative",
            "explanation",
            "context",
            "decision",
            "understanding",
            "softness",
            "behavior",
            "setting",
            "demand",
            "arrangement",
            "ritual",
            "details",
            "text",
        )
        return any(marker in lower_entity for marker in abstract_markers)

    def _matches_required_entity(
        self,
        entity: str,
        lower_text: str,
        *,
        cast_names: set[str],
    ) -> bool:
        """Returns True when a required entity or one of its alternatives is present."""

        lower_entity = self._normalize_match_text(entity)
        candidate_entities = self._entity_match_candidates(lower_entity)
        if lower_entity in cast_names:
            candidate_entities.extend(self._cast_name_candidates(lower_entity))
        if any(candidate in lower_text for candidate in candidate_entities):
            return True
        if any(self._match_entity_by_tokens(candidate, lower_text) for candidate in candidate_entities):
            return True
        quoted_terms = [term for pair in QUOTED_TERM_RE.findall(lower_entity) for term in pair if term]
        if quoted_terms and any(term in lower_text for term in quoted_terms):
            return True
        if " or " in lower_entity:
            return any(
                option.strip() and any(candidate in lower_text for candidate in self._entity_match_candidates(option))
                for option in lower_entity.split(" or ")
            )
        return False

    def _entity_match_candidates(self, entity: str) -> list[str]:
        """Returns normalized match candidates for a concrete required entity."""

        candidates = [entity.strip()]
        trimmed_prefix = re.sub(r"^(archived|historical|legacy)\s+", "", entity).strip()
        if trimmed_prefix and trimmed_prefix not in candidates:
            candidates.append(trimmed_prefix)
        simplified = re.sub(r"^(figure|number|count|estimate)\s+of\s+", "", entity).strip()
        if simplified and simplified not in candidates:
            candidates.append(simplified)
        unpossessed = re.sub(r"^\w+[’']s\s+", "", entity).strip()
        if unpossessed and unpossessed not in candidates:
            candidates.append(unpossessed)
        if unpossessed:
            simplified_unpossessed = re.sub(
                r"^(figure|number|count|estimate)\s+of\s+",
                "",
                unpossessed,
            ).strip()
            if simplified_unpossessed and simplified_unpossessed not in candidates:
                candidates.append(simplified_unpossessed)
        unwrapped = re.sub(
            r"^(fact that|fact of|proof that|evidence of|knowledge of|understanding of|decision to)\s+",
            "",
            entity,
        ).strip()
        if unwrapped and unwrapped not in candidates:
            candidates.append(unwrapped)
        trimmed_suffix = re.sub(
            r"\s+(fact|record|setting|discussion|language|explanation|context|narrative|demand|softness|behavior)$",
            "",
            entity,
        ).strip()
        if trimmed_suffix and trimmed_suffix not in candidates:
            candidates.append(trimmed_suffix)
        singularized = (
            entity.replace("complaints", "complaint")
            .replace("packets", "packet")
            .replace("files", "file")
            .strip()
        )
        if singularized and singularized not in candidates:
            candidates.append(singularized)
        if trimmed_prefix:
            singularized_trimmed = (
                trimmed_prefix.replace("complaints", "complaint")
                .replace("packets", "packet")
                .replace("files", "file")
                .strip()
            )
            if singularized_trimmed and singularized_trimmed not in candidates:
                candidates.append(singularized_trimmed)
        if " showing " in entity:
            showing_prefix = entity.split(" showing ", maxsplit=1)[0].strip()
            if showing_prefix and showing_prefix not in candidates:
                candidates.append(showing_prefix)
        if entity.endswith(" request"):
            request_set = entity[: -len(" request")].strip() + " set"
            request_form = entity[: -len(" request")].strip() + " form"
            for candidate in (request_set, request_form):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _match_entity_by_tokens(self, entity: str, lower_text: str) -> bool:
        """Returns True when a multi-word entity is naturally realized across nearby tokens."""

        entity_tokens = [
            token
            for token in WORD_RE.findall(entity)
            if len(token) > 2 and token not in ENTITY_STOPWORDS
        ]
        if len(entity_tokens) < 2:
            return False
        text_tokens = set(WORD_RE.findall(lower_text))
        return all(token in text_tokens for token in entity_tokens)

    def _cast_name_candidates(self, entity: str) -> list[str]:
        """Returns looser match candidates for cast-member names."""

        parts = [part for part in entity.split() if part]
        candidates: list[str] = []
        if len(parts) >= 2:
            candidates.append(parts[0])
            candidates.append(parts[-1])
        return candidates

    def _normalize_match_text(self, value: str) -> str:
        """Normalizes punctuation variants for entity matching."""

        return (
            (value or "")
            .lower()
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u00a0", " ")
            .strip()
        )


class PlanValidator:
    """Runs deterministic checks on planning artifacts before drafting begins."""

    PLACEHOLDER_VALUES = {"", "none", "n/a", "no change", "unchanged", "not applicable"}

    def validate(
        self,
        *,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
        initial_continuity: ContinuityState,
    ) -> PlanValidationReport:
        """Runs plan-level validation for bootstrap artifacts."""

        findings: list[ValidationFinding] = []
        findings.extend(self._structure_findings(story_spec, outline, scene_cards))
        findings.extend(self._scene_contract_findings(scene_cards))
        findings.extend(self._counterforce_findings(scene_cards))
        findings.extend(self._rolling_intensity_findings(scene_cards))
        findings.extend(self._initial_continuity_findings(initial_continuity))
        has_errors = any(finding.severity == "error" for finding in findings)
        return PlanValidationReport(pass_fail=not has_errors, findings=findings)

    def _structure_findings(
        self,
        story_spec: StorySpec,
        outline: Outline,
        scene_cards: list[SceneCard],
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if len(outline.chapters) != story_spec.expected_chapters:
            findings.append(
                ValidationFinding(
                    check_id="outline_chapter_count",
                    severity="error",
                    message="Outline chapter count does not match StorySpec.",
                    details=[
                        f"Expected: {story_spec.expected_chapters}",
                        f"Actual: {len(outline.chapters)}",
                    ],
                )
            )
        if len(scene_cards) != story_spec.expected_scenes:
            findings.append(
                ValidationFinding(
                    check_id="scene_card_count",
                    severity="error",
                    message="Scene-card count does not match StorySpec.",
                    details=[
                        f"Expected: {story_spec.expected_scenes}",
                        f"Actual: {len(scene_cards)}",
                    ],
                )
            )
        ordered_scene_numbers = [scene.scene_number for scene in sorted(scene_cards, key=lambda item: item.scene_number)]
        expected_numbers = list(range(1, len(scene_cards) + 1))
        if ordered_scene_numbers and ordered_scene_numbers != expected_numbers:
            findings.append(
                ValidationFinding(
                    check_id="scene_numbering",
                    severity="error",
                    message="Scene numbering is not contiguous from 1..N.",
                    details=[f"Actual: {ordered_scene_numbers[:12]}"],
                )
            )
        return findings

    def _scene_contract_findings(self, scene_cards: list[SceneCard]) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        required_fields = (
            "scene_type",
            "opening_disturbance",
            "mid_scene_reversal",
            "visible_decision",
            "closing_choice",
            "counterforce_trace",
            "scene_desire",
            "scene_fear",
            "secret_pressure",
            "subtext_engine",
            "power_shift",
            "relationship_delta",
            "cost_paid",
            "suspicion_delta",
            "sensory_anchor",
        )
        for scene in scene_cards:
            missing = [
                field_name
                for field_name in required_fields
                if self._is_placeholder(getattr(scene, field_name))
            ]
            if missing:
                findings.append(
                    ValidationFinding(
                        check_id="thin_scene_card",
                        severity="error",
                        message=f"Scene {scene.scene_number} is missing required kinetic scene-card fields.",
                        details=missing,
                    )
                )
        return findings

    def _counterforce_findings(self, scene_cards: list[SceneCard]) -> list[ValidationFinding]:
        first_six = scene_cards[:6]
        if any(self._has_counterforce_trace(scene) for scene in first_six):
            return []
        return [
            ValidationFinding(
                check_id="early_counterforce_absent",
                severity="error",
                message="No concrete counterforce trace appears by scene 6.",
                details=["Add a rumor, review footprint, extract request, human pursuer trace, or similar signal."],
            )
        ]

    def _rolling_intensity_findings(self, scene_cards: list[SceneCard]) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if len(scene_cards) < 3:
            return findings
        for start in range(0, len(scene_cards) - 2):
            window = scene_cards[start : start + 3]
            if any(self._window_intensifier(scene) for scene in window):
                continue
            findings.append(
                ValidationFinding(
                    check_id="rolling_intensity_gap",
                    severity="error",
                    message="A 3-scene planning window fails to intensify pursuit, relationship loss, or moral compromise.",
                    details=[f"Scenes {window[0].scene_number}-{window[-1].scene_number}"],
                )
            )
        return findings

    def _initial_continuity_findings(
        self,
        initial_continuity: ContinuityState,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if initial_continuity.recent_scene_summaries:
            findings.append(
                ValidationFinding(
                    check_id="initial_recent_summaries",
                    severity="error",
                    message="Initial continuity should not include recent scene summaries.",
                    details=initial_continuity.recent_scene_summaries,
                )
            )
        if initial_continuity.last_approved_scene_number != 0:
            findings.append(
                ValidationFinding(
                    check_id="initial_last_approved_scene_number",
                    severity="error",
                    message="Initial continuity must start at scene 0.",
                    details=[str(initial_continuity.last_approved_scene_number)],
                )
            )
        return findings

    def _has_counterforce_trace(self, scene: SceneCard) -> bool:
        trace = scene.counterforce_trace
        if self._is_placeholder(trace):
            return False
        return any(
            keyword in trace.lower()
            for keyword in (
                "marta",
                "conduct",
                "review",
                "suspicion",
                "extract",
                "audit",
                "question",
                "inquiry",
                "memo",
                "trace",
                "footprint",
                "compliance",
            )
        )

    def _window_intensifier(self, scene: SceneCard) -> bool:
        suspicion_text = " ".join([scene.suspicion_delta, scene.counterforce_trace]).lower()
        relationship_text = " ".join([scene.relationship_delta, scene.cost_paid, scene.secret_pressure]).lower()
        moral_text = " ".join(
            [
                scene.secret_pressure,
                scene.cost_paid,
                scene.visible_decision,
                scene.closing_choice,
            ]
        ).lower()
        return (
            any(keyword in suspicion_text for keyword in ("marta", "conduct", "review", "suspicion", "extract", "audit", "closer"))
            or any(keyword in relationship_text for keyword in ("elena", "marriage", "withdraw", "distance", "silence", "home", "leave", "refuse", "cold"))
            or any(keyword in moral_text for keyword in ("lie", "conceal", "withhold", "tamper", "deceive", "betray", "break", "violate", "cover", "manipulate", "hide"))
        )

    def _is_placeholder(self, value: str) -> bool:
        return re.sub(r"\s+", " ", value or "").strip().lower() in self.PLACEHOLDER_VALUES

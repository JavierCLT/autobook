# Book Intake Template

Copy this file for each new book request. Fill what you know, leave unknowns blank, and keep the section labels intact.

This template is designed for the current `Autobook` pipeline. It gives me enough information to do one of two things:

1. Run a fresh book with only a new synopsis and project slug.
2. Decide whether the scaffolding itself needs to be retuned before the next run.

## 0. Request Mode

project_slug:
working_title:

request_type:
- new_book_same_scaffold
- new_book_with_prompt_retune
- new_book_with_schema_and_prompt_retune

replace_existing_project:
- yes
- no

existing_project_to_replace:

scaffold_fit:
- same_as_current_repo
- adjacent_genre
- very_different_genre

what_must_change_in_scaffolding:

## 1. Core Metadata

title_working:
one_sentence_promise:
genre:
subgenre:
market_position:
audience:
rating_ceiling:
pov:
tense:

target_words:
expected_chapters:
expected_scenes:

## 2. Story Contract

premise_core:
themes:
setting:
timeline_window:
escalation_model:
emotional_engine:
adversarial_engine:
moral_fault_line:
ending_shape:

## 3. Plot Anchors

hook:
first_major_turn:
midpoint_turn:
dark_night_turn:
climax:
resolution:
final_image:

## 4. Protagonist

name:
age:
role:
public_face:
private_need:
fear:
contradiction:
external_goal:
inner_wound_or_need:
secret_pressure:
what_they_are_hiding:
what_they_will_lose:
why_they_cannot_walk_away:

## 5. Counterforce / Antagonist

name_or_force:
public_role:
private_goal:
method:
why_they_are_dangerous:
how_they_apply_pressure:
how_they_change_over_time:
what_they_correctly_understand_about_the_protagonist:

## 6. Relationship Engine

primary_relationship_1:
starting_state:
best_memory_or_shared_ritual:
what_makes_it_alive_on_page:
how_it_deteriorates:
what_each_person_wants_from_the_other:
what_each_person_refuses_to_say:
end_state:

primary_relationship_2:
starting_state:
best_memory_or_shared_ritual:
what_makes_it_alive_on_page:
how_it_deteriorates:
what_each_person_wants_from_the_other:
what_each_person_refuses_to_say:
end_state:

## 7. Main Cast

For each important character, copy this block.

character_1_name:
character_1_age:
character_1_role:
character_1_public_face:
character_1_private_need:
character_1_fear:
character_1_contradiction:
character_1_relationships:
character_1_secret:
character_1_function_in_story:

character_2_name:
character_2_age:
character_2_role:
character_2_public_face:
character_2_private_need:
character_2_fear:
character_2_contradiction:
character_2_relationships:
character_2_secret:
character_2_function_in_story:

## 8. Setting and World Rules

primary_locations:
social_or_institutional_environment:
what_the_world_rewards:
what_the_world_punishes:
world_rules_or_professional_rules_that_matter:
specialized_domains_the_book_must_handle_correctly:
research_sensitive_areas:

## 9. Non-Negotiables

must_have_scenes:
must_have_reveals:
must_have_images_or_motifs:
must_keep_facts_from_synopsis:
must_not_happen:
forbidden_tropes:
forbidden_entities_or_plot_devices:

## 10. Style Guide

prose_traits:
banned_tells:
dialogue_rules:
narration_rules:
sensory_preferences:
things_that_should_feel_hotter_or_sharper:
things_that_should_feel_restrained:
things_that_make_you_say_this_sounds_ai:

## 11. Content Boundaries

banned_content:
violence_ceiling:
sexual_content_ceiling:
profanity_ceiling:
topics_to_handle_carefully:

## 12. Continuity Rules

continuity_rules:
facts_that_must_never_change:
timeline_constraints:
character_knowledge_constraints:
objects_or_evidence_that_must_track_cleanly:

## 13. Commercial Positioning

ideal_reader:
primary_sales_category:
secondary_sales_category:
what_it_should_feel_like_in_the_market:
what_it_must_not_feel_like:

## 14. Full Synopsis

Paste the full long-form synopsis here. This should be the most detailed section in the document.

synopsis:

## 15. Optional Notes To Codex

If the next book is a different genre or market type, tell me here exactly what you want the engine to become.

notes_to_codex:

## 16. Minimal Version

If you are in a hurry, these are the minimum fields I need for a clean new run:

- project_slug
- title_working
- genre
- subgenre
- audience
- rating_ceiling
- target_words
- one_sentence_promise
- premise_core
- protagonist block
- counterforce / antagonist block
- relationship engine block
- plot anchors
- banned_content
- continuity_rules
- full synopsis

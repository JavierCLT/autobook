# Novel Factory

Novel Factory is a production-minded Python CLI that turns a long synopsis into a checkpointed thriller manuscript pipeline. It plans first, locks a story contract and outline, generates scene cards, drafts scenes one at a time with continuity context, rejects weak scenes through deterministic and model-based QA, rewrites failed scenes, assembles chapter files, and runs manuscript-level QA plus targeted repair.

The repository is optimized for output quality rather than speed or token efficiency. It uses the official OpenAI Python SDK, the Responses API, and structured outputs for planning and QA artifacts.

The pipeline can now take either a plain synopsis file or a richer markdown intake file based on [BOOK_INTAKE_TEMPLATE.md](./BOOK_INTAKE_TEMPLATE.md). When an intake file is provided, its explicit market, relationship, style, structural, and non-negotiable constraints are threaded into planning and drafting.

## What It Produces

Each project is stored under `runs/<project_slug>/` and includes:

- `book_intake.md` and `book_intake.json` when an intake file is used
- `input_synopsis.md`
- `story_spec.json`
- `outline.json`
- `scene_cards.json`
- `initial_continuity_state.json`
- `continuity_state.json`
- `scenes/scene_01.md`, `scene_02.md`, ...
- `qa/scene_01_qa.json`, `scene_02_qa.json`, ...
- `rewrites/scene_01_draft_attempt_00.md`, ...
- `chapters/Chapter_01.md`, `Chapter_02.md`, ...
- `final_manuscript.md`
- `final_manuscript.txt`
- `global_qa_report.json`
- `run_log.jsonl`

## Pipeline

1. `bootstrap`
   Reads either a synopsis or a structured intake file and generates `StorySpec`, `Outline`, `SceneCards`, and initial continuity.
2. `draft-scene`
   Builds scene context, drafts a scene, runs deterministic validators, runs scene QA, and rewrites up to two times if needed.
3. `run-project`
   Runs the full pipeline from planning through manuscript assembly, global QA, and targeted repairs.
4. `global-qa`
   Evaluates the assembled manuscript for hook strength, payoff, continuity, boredom risk, voice consistency, and AI-smell risk.
5. `repair-project`
   Rewrites only the scenes flagged by the last failed global QA pass, then reassembles and re-runs QA.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set your API key:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.4
NOVEL_FACTORY_RUN_ROOT=runs
NOVEL_FACTORY_DEFAULT_AUDIENCE=Adult
NOVEL_FACTORY_DEFAULT_RATING_CEILING=R
NOVEL_FACTORY_DEFAULT_MARKET_POSITION=adult thriller
```

`OPENAI_MODEL` defaults to `gpt-5.4`. The audience and rating defaults can be changed through the `NOVEL_FACTORY_DEFAULT_*` variables when you want to switch between adult and YA projects.

## Usage

### Bootstrap a project from a synopsis

```bash
python main.py bootstrap --project my_book --synopsis-file synopsis.md
```

### Bootstrap a project from an intake template

```bash
python main.py bootstrap --project my_book --intake-file my_book_intake.md
```

### Draft one scene

```bash
python main.py draft-scene --project my_book --scene-index 5
```

Force a rewrite of an already approved scene:

```bash
python main.py draft-scene --project my_book --scene-index 5 --force
```

### Run the full project from a synopsis

```bash
python main.py run-project --project my_book --synopsis-file synopsis.md
```

### Run the full project from an intake template

```bash
python main.py run-project --project my_book --intake-file my_book_intake.md
```

### Run manuscript-level QA

```bash
python main.py global-qa --project my_book
```

### Apply targeted repairs from the last failed global QA

```bash
python main.py repair-project --project my_book
```

## Checkpointing And Resume Behavior

- Planning artifacts are saved once and reused on later runs.
- If an intake file is used, both the raw markdown and parsed JSON are saved with the run.
- Approved scenes are saved only after they pass deterministic validation and model QA.
- Failed attempts are preserved in `rewrites/`.
- `continuity_state.json` is updated after each approved scene.
- `run_log.jsonl` records phase transitions and scene attempts.
- `run-project` resumes from the longest contiguous prefix of approved scenes.
- Continuity can be rebuilt from `initial_continuity_state.json` plus approved scenes, which keeps the run recoverable after interruption or repair.

If a scene fails repeatedly, the run stops without saving it into `scenes/`. Fix the issue, adjust prompts or inputs if needed, and rerun `python main.py draft-scene ...` or `python main.py run-project ...`.

## Repository Layout

```text
.
|- main.py
|- BOOK_INTAKE_TEMPLATE.md
|- novel_factory/
|  |- config.py
|  |- generators.py
|  |- intake.py
|  |- judges.py
|  |- llm.py
|  |- pipeline.py
|  |- prompts.py
|  |- schemas.py
|  |- storage.py
|  |- utils.py
|  `- validators.py
|- requirements.txt
|- .env.example
`- runs/
```

## Notes

- The code uses `client.responses.create(...)` for prose generation.
- The code uses `client.responses.parse(...)` for structured artifacts and QA models.
- The pipeline reads `OPENAI_API_KEY` from environment variables through `python-dotenv`.
- Structured planning artifacts use Pydantic models to keep the run restartable and inspectable.
- Intake-template runs still produce `input_synopsis.md`; the synopsis is either read directly from `--synopsis-file` or extracted from the `synopsis:` field inside the intake file.

# The Operating Table

**One AI on the table. Another holding the scalpel.**

A small, local Rust laboratory for **weight-space programming**: inspecting a neural network and directly changing selected parameters, without gradient descent, LoRA, or a training pipeline.

Qwen3-0.6B is the patient. GPT-6 Astra, running through your authenticated **Codex CLI**, is the surgeon. You are the observing doctor: describe an objective, watch the investigation, compare results, then save the patch or revert.

This is an experiment. A change to weights is not evidence of improved capability.

## Requirements

- Current stable Rust and a C/C++ build toolchain. On macOS, install Xcode Command Line Tools.
- Codex CLI with access to `gpt-6-astra`, signed in through `codex login`. The integration was tested with Codex CLI 0.164.0; it requires `--ignore-user-config`, `--ephemeral`, and `--output-schema`.
- About 1.2 GB for the original checkpoint, additional space for Rust compilation and saved tensors, and enough memory for the model and inference. 8 GB or more is a practical starting point; CPU execution can be slow.
- Network access for the first Hugging Face download and for Codex inference.

There is **no OpenAI API key setup**. Astra uses Codex authentication and its associated usage limits. The application removes `OPENAI_API_KEY` and `CODEX_API_KEY` from the surgeon subprocess environment.

## Run

```sh
git clone https://github.com/veighnsche/the-operating-table.git
cd the-operating-table
codex login
cargo run --release
```

Open **http://127.0.0.1:3000**. Click **Load Patient**. The first load downloads original `Qwen/Qwen3-0.6B` config, tokenizer, and Safetensors weights from Hugging Face at a pinned revision. Later loads use the local cache.

For Apple Silicon:

```sh
cargo run --release --features metal
```

The Metal build selects Metal when available and falls back to CPU. Set `PATIENT_DEVICE=cpu` to force CPU execution. The CPU build works without GPU libraries. Apple Silicon execution must be verified on an actual Mac; development verification was performed on Linux CPU.

Optional configuration:

```sh
cp .env.example .env
# Edit ordinary settings in .env, then restart the app.
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_BIN` | `codex` | Codex executable path |
| `SURGEON_REASONING_EFFORT` | `medium` | Codex reasoning effort |
| `SURGEON_TIMEOUT_SECONDS` | `240` | Per-call timeout, capped at 900 seconds |
| `TABLE_PORT` | `3000` | Loopback HTTP port |
| `TABLE_DATA_DIR` | `./data` | Model cache, patches, and procedure records |
| `PATIENT_DEVICE` | `auto` | `auto`, `cpu`, or `metal` |
| `RAYON_NUM_THREADS` | library default | CPU worker count; the example uses 8 |

## Perform surgery

1. Load the patient and inspect a layer in the central diagram.
2. Enter an objective, for example: “Investigate repetitive responses. Inspect relevant tensors and try a small, justified intervention. Compare it against the original and report any regression.”
3. Set the call and edit budgets, then press **Begin Surgery**. Starting surgery authorizes bounded experiments on the disposable working copy.
4. Watch Astra's short explanations, selected instruments, inspected tensors, and observed results. The app records an initial evaluation and always attempts a final comparison, including after a surgeon error.
5. Review the paired outputs. **Accept / Save Patch** saves modified tensor data and its manifest; it does not replace the baseline. **Emergency Revert** cancels the active surgeon and restores the original weights after the current bounded model action returns. Generation checks cancellation between tokens; an initial download cannot be interrupted immediately.

The initial/final evaluation uses three fixed prompts (arithmetic, a short word problem, and instruction following), greedy decoding, thinking disabled, and a 64-token cap. Astra can choose additional prompts and enable thinking for its own tests. A comparison always uses the same prompts, decoding, thinking setting, and token limit for both models. Repeated-token rate is descriptive, not a correctness or looping benchmark.

## How it works

One Axum application serves static HTML/CSS/JavaScript and owns the patient. Candle implements Qwen3 inference; the project does not implement a new inference engine.

The operative model has 596,049,920 parameters. The checkpoint contains a duplicate tied output tensor that Candle ignores; it is excluded from inspection and editing. CPU inference uses F32 while the original and working source tensors remain BF16.

The baseline tensor map is never modified. Working tensors initially share immutable storage. An edit creates a new tensor and constructs a new Candle model with the replacement, so subsequent generation uses the edited weights and a fresh KV cache. Revert reconstructs the model from the original tensors.

Astra selects one structured action per Codex turn:

| Instrument | Actual operation |
| --- | --- |
| `inspect_model` | Architecture, layer-filtered/paginated tensor catalog, shapes, dtypes, and bounded statistics |
| `inspect_weights` | Bounded numerical samples and statistics for a selected range, rows, or columns |
| `test_patient` | Generate text using the working model |
| `edit_weights` | Scale, zero, or replace a bounded range, rows, or columns |
| `compare_patient` | Run matched generation against immutable baseline and working model |
| `revert_changes` | Restore every working tensor from the baseline |

The Rust harness checks tensor names, selections, finite values, dtype representability, and bounds before committing an edit. Each edit selects at most 262,144 parameters. Changed counts compare current values against the baseline; overlapping edits are not double counted. Model-generated commands and arbitrary code are never executed by these instruments.

Codex runs ephemerally in an empty temporary directory with a JSON output schema, neutral instructions, ignored user configuration, a read-only sandbox, and built-in shell, editing, web, plugin, and agent tools disabled. Its JSON action is parsed and dispatched by Rust. Private reasoning is not shown or saved: the UI receives only the explicit short action summary and tool results. Each new call receives the bounded investigation transcript; no agent framework is involved. The configured call/edit budgets, output limits, and timeout bound each procedure, though Codex token use still depends on the chosen effort and evidence.

The server binds to `127.0.0.1` and checks Host/Origin. Keep it local. Prompt text, bounded weight samples, and test outputs are sent to Codex. Procedure records remain under the ignored data directory. Do not put secrets in objectives or evaluation prompts.

## Saved patches

`data/patches/<id>/` contains `weights.safetensors` (only modified tensors) and `manifest.json`. The manifest identifies the model revision, baseline checksum, patch checksum, names, shapes, dtypes, and actual changed counts. The original checkpoint is never overwritten. Restarting begins from the original model.

To replay a saved patch after loading the patient:

```sh
curl http://127.0.0.1:3000/api/patch \
  -H 'Content-Type: application/json' \
  -d '{"id":"YOUR_NUMERIC_PATCH_ID"}'
```

Replay validates the baseline identity and tensor data, then replaces the working copy. Click **Compare** to evaluate it again. Patch data and procedure records are ignored by Git; only share them intentionally.

## Validation

```sh
cargo test
# Real model download, inference, editing, patch replay, and revert:
cargo test --release real_model_edit_revert -- --ignored --nocapture
```

A live Linux CPU demonstration used authenticated Codex Astra to choose and scale one attention head’s output columns by 0.95, changing 131,072 weights. All three short test outputs remained identical. This verified the experiment mechanism and established no improvement.

The ordinary tests cover edit selection bounds and the surgeon's strict action protocol. The opt-in integration test uses the actual checkpoint; it does not simulate inference or successful experiments.

## Research and limitations

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (November 2, 2026) and its [complete experiment logs](https://github.com/RikiyaT/lm_understand-logs), especially [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math).

That experiment edited **Qwen3.5-0.8B**, used substantially larger evaluation budgets and activation analysis, and found selected attention/MLP interventions that reduced thinking loops. The follow-up on Gemma did not establish an improvement. This application uses **Qwen3-0.6B** and a much smaller instrument set. It is inspired by the investigation loop, not a reproduction of those results. No particular tensor, intervention, or successful outcome is hardcoded for Astra.

This MVP has no activation capture, projection-direction estimation, benchmark suite, training, or guarantee that a small edit will help. Short greedy evaluations with thinking disabled do not reproduce the research's long sampled reasoning evaluations. A lower repetition rate or shorter output can accompany worse answers. The original baseline is the reference even when a new operation begins with an already edited working copy.

Code is MIT-licensed. Original model weights retain their upstream license and are distributed by [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B), not this repository. Research materials retain their own licenses. No affiliation with MIT, Qwen, or OpenAI is implied.

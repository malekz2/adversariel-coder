# Adversarial Coder: a Maker and a Solver that train each other

Two LoRA adapters on one language model play a game, in the spirit of a GAN:

- the **Maker** invents Python coding puzzles (statement, reference solution, test inputs);
- the **Solver** attempts each puzzle several times inside the **Pi** coding agent (it can read files,
  write `solution.py`, run the visible tests in bash, and fix its code);
- hidden tests grade the Solver, and the Solver's success rate grades the Maker;
- both adapters get a GRPO update every step;
- every 10 steps a **referee** (`vllm-sr/d3-flash`, a decision model) checks whether the game has
  stalled, and nudges a few bounded knobs if it has.

```
            topic + difficulty hint + 2 seed examples
                          │
                  ┌───────▼────────┐   JSON puzzle    ┌─────────────────────┐
                  │  MAKER adapter │ ───────────────► │ sandbox: run the     │──► invalid → reward −1
                  └───────▲────────┘                  │ reference twice,     │
                          │                           │ compute expected out │
     reward = 1−2|p̂−0.5|  │                           └─────────┬───────────┘
     (edge of ability)    │                                     │ valid puzzle
                          │                     ┌───────────────▼───────────────┐
                          │                     │ SOLVER adapter × G attempts,  │
                          └──── p̂ = pass rate ◄─│ each one a Pi agent episode   │──► reward = hidden tests passed
                                                └───────────────────────────────┘
                     both adapters: GRPO update  →  new adapters → llama.cpp  →  next step
                     every K steps: d3-flash referee reads the metrics → maybe adjusts difficulty/temperature/lr…
```

---

## 1. Quick start on the lab PC (Windows, no admin needed)

1. Put the SD card in the PC and open the `adversarial-coder` folder.
2. Double-click **`setup.bat`** and leave it running; it needs internet. It downloads everything
   into `tools\` and `.venv\` on the card: uv + Python 3.12, PyTorch (CPU or CUDA, picked automatically),
   Node.js + Pi, PortableGit (for Pi's bash), and llama.cpp. Then it:
   - detects the hardware and picks a **profile** (written to `state\hardware.json`);
   - asks where to put the **models** (press Enter for the card, or give a folder on the PC's own drive
     for the big profiles);
   - runs a **self-test** that checks llama.cpp can apply our LoRA adapters on this machine, and that
     d3-flash answers.
3. Double-click **`dryrun.bat`**: 15 steps with a fake model, done in about a minute. It proves Pi, the
   sandbox, `log.txt` and `train.png` all work before you spend hours.
4. Double-click **`run.bat`** to train. Open **`log.txt`** and **`train.png`** in the same folder; both
   are overwritten after every step. `log.txt` shows the **ETA** and the expected finish time.
5. Lab session over? Press **Ctrl+C** (it saves a checkpoint), or start with `run.bat --max-hours 5` so it
   stops by itself. Next time, double-click **`resume.bat`**.

Useful variations (append to `run.bat`, or edit `config.yaml`):

```
run.bat --steps 50                 shorter run
run.bat --max-hours 6              stop cleanly after 6 hours
run.bat --profile cpu_tiny         force the smallest model (fastest)
run.bat --no-pi                    single-shot Solver (no agent loop), much faster
run.bat --name exp2                separate run folder (runs\exp2)
```

On Linux/macOS, use `bash setup.sh`, then `bash run.sh --dry-run --steps 15`, then `bash run.sh`
(add `--resume` to continue).

**Do not run two copies at once.** They share ports 8181–8383 and the same `log.txt`.

---

## 2. What happened to the SD card

It was **not** Raspberry Pi OS. It was a 32 GB card formatted APFS (Apple-only), containing only macOS
housekeeping files. It was reformatted to **exFAT** (name `IPHONE`), which Windows, Linux and macOS all
read and which allows files over 4 GB.

---

## 3. Hardware profiles and what to expect (be patient)

`check_hardware` picks the profile; force one with `profile:` in `config.yaml` or `--profile`.

| profile | when | model trained (LoRA) | rollouts | rough time / step* |
|---|---|---|---|---|
| `cuda_large` | NVIDIA ≥ 24 GB VRAM | **Qwen3.8-27B**, 4-bit NF4 (QLoRA) | `Qwen3.8-27B-UD-IQ4_XS.gguf` | 15–30 min |
| `cuda_small` | NVIDIA 8–24 GB | Qwen3.5-4B, NF4 | Qwen3.5-4B Q4_K_M | 5–15 min |
| `cpu` | no NVIDIA, ≥ 24 GB RAM (OptiPlex with 32 GB) | Qwen3.5-2B, fp32 | Qwen3.5-2B Q4_K_M | **45–120 min** |
| `cpu_tiny` | no NVIDIA, < 24 GB RAM | Qwen3.5-0.8B, fp32 | Qwen3.5-0.8B Q4_K_M | 20–45 min |

\* Back-of-the-envelope estimates for the default settings (4 puzzles × 4 Pi attempts per step), not
measurements. The **ETA in `log.txt` is the real number**; it settles after about 3 steps. Training is
the slow part on CPU: every step back-propagates through ~50–100k tokens. To go faster, use `--no-pi`
(shorter sequences, roughly 2–3× faster), `cpu_tiny`, or fewer `grpo.solver_attempts`.

Why not 27B on a plain OptiPlex? A stock OptiPlex 5000 has a 12th-gen Intel CPU and integrated graphics.
Generating with the 27B GGUF would run at about 2 tokens/s, and back-propagating through 27B on a CPU would
take roughly half a day per step. The 4-bit training library (bitsandbytes) also needs an NVIDIA GPU.
The code is identical across profiles, so a later run on a GPU machine uses the real 27B setup.

**Disk:** tools + Python environment ≈ 3–6 GB on the card. Models: `cpu` ≈ 12.5 GB (fits on the card),
`cpu_tiny` ≈ 9 GB, `cuda_small` ≈ 19 GB, `cuda_large` ≈ 76 GB (needs the PC's own drive). Some lab PCs
wipe user folders at logout; if so, keep the models on the card or a USB drive.

---

## 4. How it works (the parts worth learning)

### 4.1 Why not a literal GAN reward
If the Maker is paid when the Solver *fails*, it learns to write impossible or ambiguous puzzles, and
the game collapses. Instead the Maker is paid for puzzles at the **edge of the Solver's ability**
(the idea from R-Zero / Absolute Zero):

```
learnability(p̂) = 1 − 2·|p̂ − 0.5|      p̂ = fraction of the Solver's G attempts that passed every test
maker_reward    = learnability − similarity_penalty      (invalid puzzle: −1)
solver_reward   = fraction of hidden tests passed (+0.5 bonus for all of them)
```

The two rewards are coupled. GRPO only learns from attempts that *disagree*, and p̂ = 0.5 is exactly
where a group of attempts disagrees most. So the Maker is literally paid for producing training signal
for the Solver. → `src/adv/rewards.py`

### 4.2 Puzzles that can't lie
The Maker writes test **inputs** only. Expected outputs come from running the Maker's own reference
solution, twice with different hash seeds (to catch nondeterminism), in a separate process with a
timeout. It also checks an import allow-list, sensible test counts, non-constant outputs, and that the
solution isn't pasted into the statement. A puzzle the Maker can't solve itself is simply invalid.
→ `src/adv/puzzles.py`, `src/adv/sandbox.py`

### 4.3 GRPO, by hand
Written out in about 60 readable lines instead of hidden inside a library, because TRL's trainer
doesn't fit two players plus multi-turn agent episodes.
`advantage = (r − mean(group)) / std(group)`; the loss pushes up the log-probability of tokens in
better-than-average attempts. The Solver's groups are "attempts at the same puzzle". The Maker has no
natural group, so its baseline is the mean of that step's batch. One update per batch keeps it
on-policy. **Watch `dead groups`** in the log: groups where every attempt scored the same give no gradient.
→ `src/adv/grpo.py`, `src/adv/policy.py`

### 4.4 One base, two adapters: QLoRA + rsLoRA
Both players share one frozen base model, each with its own LoRA adapter (r=16, α=32) on attention, the
Gated-DeltaNet projections and the MLP. That halves memory compared with two 27B models, and switching
the adapters off gives the original model for free. On GPU the base is stored in 4-bit NF4 (QLoRA);
rsLoRA scales updates by α/√r. LoRA is known to match full fine-tuning for RL-sized updates, while DoRA
is slower and fragile in 4-bit. Log-probs are computed in chunks so the 248k-word vocabulary never
materialises for the whole sequence.

### 4.5 Sampling with llama.cpp, training with PyTorch
- **Rollouts** run on llama.cpp with the GGUF (e.g. `UD-IQ4_XS`), and both adapters are loaded at once;
  each request picks one.
- **Training** runs on the safetensors weights. A GGUF can't be trained.
- After each update, the adapters are converted to GGUF and the server restarts.
- Two quantisations of the same weights means a tiny "sampler/learner mismatch", the same as vLLM vs
  FSDP in large RL systems. It's ignored here.
- **Gotcha found while building this:** llama.cpp's converter ignores rsLoRA, so it would apply the
  adapter at 1/√r strength. The export rescales α to fix this. It was verified on a tiny model:
  HF + adapter and llama.cpp + adapter produce identical greedy tokens.
- If the self-test finds that llama.cpp can't serve the adapters, training switches to the slower
  Hugging Face engine, which samples from the exact training weights.
→ `src/adv/engines/`

### 4.6 Pi + a recording proxy (agentic RL)
Pi only speaks chat messages, but RL needs the exact tokens the model produced, so Pi talks to a small
OpenAI-compatible **proxy** (`src/adv/proxy.py`). The proxy:
- renders messages with the model's own chat template;
- **chains turns**: turn t+1 = exact tokens of turn t + the newly rendered tool results, so a 4-turn
  episode is one sequence, trained in one forward pass, never re-tokenised;
- marks a **loss mask**: only model-generated tokens are trained, never tool output;
- selects the adapter, parses Qwen's XML tool calls into OpenAI JSON for Pi, and enforces the turn and
  context limits.

The rollout id travels as Pi's API key. Hidden tests never exist on disk where Pi could read them.
→ `src/adv/pi_runner.py`

### 4.7 Thinking-only overthinking penalty
Your 49 marker words get a −2 logit bias, but **only inside `<think>`**: `" or"`, `" error"` and
`" could"` are everywhere in real code. llama.cpp does it in two passes (biased until `</think>`, then
unbiased); the HF engine uses a single-pass logits processor. If thinking hits its budget, `</think>` is
forced, and that token is masked out of training. Your ids were checked and match the Qwen3.5/3.8
tokenizer 49/49 (they would be wrong for Qwen3/Qwen2.5). They're still re-derived from the words at
startup, so swapping models stays safe. Check any tokenizer with
`python -m adv.bias <folder with tokenizer.json>`. → `src/adv/bias.py`

### 4.8 The referee (d3-flash) and the controller
d3-flash doesn't write text. You send it a JSON **state** and typed questions, and it returns a
probability for each answer. It runs from a GGUF through llama.cpp's `/v1/systemone`, on CPU, only when
consulted (so it doesn't take memory from training).
- **Questions:** stalled? (yes/no) · diagnosis (too hard / too easy / invalid / mode collapse /
  plateau / overthinking) · best action · progress score.
- **Controller:** applies only **small, bounded** changes (difficulty ±1, maker temperature +0.1,
  halve the learning rate, inject seed puzzles, reset the maker to its best checkpoint), and only if
  P(stalled) ≥ 0.6, with a 15-step cooldown, so a wrong call can't wreck the run.
- **Fallback:** if d3 can't run, a transparent rule-based referee answers instead, and the log says so.
  It's also a good baseline to compare d3 against.
→ `src/adv/decision.py`

---

## 5. Reading `train.png` and `log.txt`

| panel | healthy | warning sign |
|---|---|---|
| Solver pass rate | drifts up, often within the green 0.3–0.7 band | stuck at 0 (puzzles too hard) or 1 (too easy) |
| Maker learnability / validity | validity > 0.6, learnability rising | validity collapsing → format drift; similarity → 1 → mode collapse |
| Difficulty mix | lots of green (p̂ = 0.5) | all red (p̂ = 0) or all blue (p̂ = 1): dead groups |
| Thinking / markers | thinking tokens under budget; marker rate falling | budget hit > 50 % of the time |
| Optimisation | grad norms stable; dead-group fraction < 0.7 | dead groups → 1.0: no learning is happening |
| Referee | P(stalled) mostly low; occasional interventions | interventions every check: the game is unstable |

`runs/<name>/` holds `metrics.jsonl` (every number, every step), `verdicts.json` (every referee call),
`checkpoints/` (last 3 + `best_maker`), and `logs/` (llama-server output).

---

## 6. Settings you'll most likely change (`config.yaml`)

- `run.total_steps`, `run.max_hours`, `run.log_every`
- `grpo.puzzles_per_step` (M) and `grpo.solver_attempts` (G): more = better statistics, slower steps
- `optim.lr_solver` / `optim.lr_maker`
- `generation.*.think_budget`: the main lever on speed and overthinking
- `solver.use_pi` / `solver.max_turns`
- `decision.every`, `decision.stall_threshold`, `decision.backend: rules` (to skip d3)
- `profiles.*`: which models each hardware class uses

---

## 7. Troubleshooting

- **Setup fails downloading:** run `setup.bat` again; it resumes and skips what's already there.
- **"Model not found":** run setup again, or `python -m adv.download --models-dir <folder>`.
- **Self-test says the engine is `hf`:** llama.cpp couldn't apply the LoRA on this machine. Training
  still works, just slower. Details are in `state\engine.json`.
- **PowerShell refuses to run scripts:** use the `.bat` files; they pass `-ExecutionPolicy Bypass`
  for that one process, which needs no admin rights.
- **Out of memory:** lower `ctx_per_slot`/`max_seq_len` in the profile, set `parallel_slots: 2`, or use
  a smaller profile.
- **Everything is slow:** see section 3. `--no-pi` and `cpu_tiny` are the big levers.
- **Tests:** from the project folder, `.venv\Scripts\python -m pytest tests` (Linux/macOS: `.venv/bin/python -m pytest tests`).

---

## 8. Building your own adversarial project

Every stable self-play setup has the same five parts. Swap them out and keep the plumbing:

1. **Generator** (here: the Maker) proposes tasks.
2. **Verifier** turns a task into ground truth *without trusting the generator*: here, running the
   generator's own reference solution in a sandbox. For math: a symbolic checker. For SQL: run the query.
   No verifier means no stable game.
3. **Learner** (here: the Solver) attempts the tasks in groups.
4. **Coupling reward:** pay the generator for tasks that make the learner's group *disagree*
   (learnability), not for making it fail. Add validity and novelty terms.
5. **Referee and bounded controller:** monitor dead groups, validity, novelty and difficulty; intervene
   rarely and gently.

To adapt the code: change the prompts and the format in `puzzles.py`, the grading in `train._solve_one`,
and the rewards in `rewards.py`. Everything else (two adapters, GRPO, the proxy, the logging and the
referee) is reusable as is.

---

## 9. Safety

Model-written code runs on this PC as you: the puzzle checks in the sandbox, and Pi's bash tool. The
sandbox is a separate process in a temp folder with a timeout (plus a memory cap on Linux/macOS), and
the Maker has an import allow-list. But it is **not** a real security sandbox, and Pi's bash isn't
confined to its workspace. Don't keep anything precious on the card or the account you train under.

---

## 10. What was verified before handing this over (and what wasn't)

Verified on a Mac, with no training run:
- 31 unit and integration tests, including a full fake-model run with checkpoint and resume.
- A real Pi 1.3.0 session through the proxy: read → write → `python run_tests.py`, with turns
  chained and tool output masked.
- `setup.sh` end to end (without models), and every Windows download URL resolving.
- On a miniature random Qwen3.5 with the real tokenizer: text-only loading, all 10 LoRA target types,
  chunked log-probs equal to a full forward pass, gradients only on the active adapter, the forced
  `</think>`, and llama.cpp b11700 converting and applying the LoRA with output identical to HF.
- The logit-bias ids against the real Qwen3.5 and Qwen3.8 tokenizers.

Not verified (only possible on the lab PC):
- `setup.ps1` running on Windows (the script was syntax-checked with PowerShell).
- The CUDA/bitsandbytes path.
- The real model sizes and their memory use.
- d3-flash itself: its 6.6 GB GGUF was not downloaded here.
- Actual step times.

---

## References

- GRPO: Shao et al., *DeepSeekMath* (2024) · DAPO (2025) · Dr. GRPO (2025)
- Self-play proposer/solver: *Absolute Zero* (2025), *R-Zero* (2025), *SPIRAL* (2025)
- LoRA (Hu et al., 2021), QLoRA (Dettmers et al., 2023), rsLoRA (Kalajdzievski, 2023),
  Thinking Machines, *LoRA Without Regret* (2025)
- Pi coding agent: `@earendil-works/pi-coding-agent` · llama.cpp `/v1/systemone` (ggml-org/llama.cpp PR #30383)
- Models: `unsloth/Qwen3.8-27B-GGUF`, `unsloth/Qwen3.8-27B`, `unsloth/Qwen3.5-{0.8B,2B,4B}`, `vllm-sr/d3-flash`
  (GGUF: `ggml-org/vllm-d3-flash-GGUF`)

# Latent CoDi: Distilling CoT Reasoning into Latent Tokens

EvoReason: Self-Evolving Reasoning Primitive-Guided On-Policy
Distillation for Latent Reasoning in Generative Recommendation

## Installation

Use a Linux environment with NVIDIA GPUs and a compatible CUDA/PyTorch installation. From the project directory:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

`requirements.txt` covers training, inference, primitive induction, and development. FlashAttention is installed separately because the teacher SFT script uses `flash_attention_2`. Dependency versions are not locked; validate compatibility in the target environment before training.

## Three-Phase Architecture

```text
Phase A: QwQ-32B ReAct trajectories → primitive induction → structured CoT
Phase B: Teacher SFT (a 1.7B model learns CoT reasoning)
Phase C: On-policy OPD (PPO + gated OPD + primitive distillation)
```

**Core idea:** Use reasoning primitives to organize long QwQ-32B trajectories into a five-step CoT template that a 1.7B model can learn, then distill the reasoning into latent tokens through OPD.

The sections below describe the supplied README. Training and evolution behavior has not been fully verified against source code. Paper-specific requirements and unresolved differences are summarized at the end.

---

## Phase A: QwQ-32B ReAct → Primitive Induction → Structured CoT

### A1: QwQ-32B ReAct Reasoning

Start four vLLM instances on eight GPUs. Replace the model path and create `/tmp/vllm_logs` first.

```bash
# GPUs 0,1 → port 12350; 2,3 → 12351; 4,5 → 12352; 6,7 → 12353.
for i in 0 1 2 3; do
    port=$((12350 + i))
    gpu_start=$((i * 2))
    CUDA_VISIBLE_DEVICES=${gpu_start},$((gpu_start + 1)) nohup python -m vllm.entrypoints.openai.api_server \
        --model /pub/.../QwQ-32B --served-model-name QwQ-32B \
        --port ${port} --tensor-parallel-size 2 --max-model-len 32768 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --enable-reasoning --reasoning-parser deepseek_r1 \
        --gpu-memory-utilization 0.95 > /tmp/vllm_logs/vllm_${port}.log 2>&1 &
    sleep 2
done
```

QwQ-32B freely selects tools and reasoning paths from 16 recommendation tools, including `search`, `lookup`, `search_memory`, `extract_user_profile`, `infer_intent`, and `produce_recommendation`.

### A2: Primitive Induction

Induce reasoning primitives (ptools) from successful QwQ-32B ReAct trajectories:

```text
induction.py: Load Thoughts → Categorize → Merge → Synthesize ptool
```

- **Categorize:** Assign reasoning-action labels to reasoning steps using an LLM.
- **Merge:** Consolidate synonymous categories.
- **Synthesize:** Generate a ptool specification for each category as an `@implement_via('simulate')` stub.
- **Output:** Five initial ptools, including `[extract_evidence]`, `[model_preferences]`, `[infer_intent]`, and `[recommend]`.

### A3: Structured CoT Simulation

Include ptool stubs in the prompt and instruct the model to use `[step_name]...[/step_name]` tags. Generate the complete structured CoT in one LLM call.

### A4: Training Data Conversion

Join CoT data with the original RA training parquet, add `think_content`, and save `training_RA_cot_train.parquet`.

---

## Phase B: Teacher SFT

Train the 1.7B model to generate structured CoT with primitive tags.

### Training Command

```bash
python train_beauty_RA_cot.py \
    --model_name_or_path beauty_RA/epoch_2 \
    --data_path training_RA_cot_train.parquet \
    --output_dir beauty_RA_cot/epoch_3 \
    --learning_rate 5e-6 --num_train_epochs 3
```

### Training Data Format

```text
🧐
[extract_evidence]
The user has purchased anti-aging products...
[model_preferences]
Core preference axes: anti-aging...
[infer_intent]
Given the focus on dark circles...
[recommend]
Recommend Maybelline Instant Age Rewind Eraser...

 Millennia
<|sid_begin|>...<|sid_end|>
```

This abbreviated example retains the source's markers and layout.

### Model Loading

- Initialize from Qwen3-1.7B at `beauty_RA/epoch_2`.
- Add SID tokens: `<|sid_begin|>`, `<|sid_end|>`, and `<s_{a-d}_{0-255}>`; configure `[PAD]`, `bot_id`, and `eot_id`.
- Expand embeddings with `resize_token_embeddings` and initialize new weights through mean resizing.
- Use LoRA: `r=64`, `alpha=32`, targeting `q/k/v/o/gate/up/down_proj`.
- Train for three epochs and save to `beauty_RA_cot/epoch_3`.

---

## Phase C: On-Policy OPD (GRPO + PPO Clipping + OPD Distillation)

### C1: Model Architecture: LatentRecCoDi

Teacher and student **share the same parameters**, following the stated SEED alignment. Teacher forward passes run under `no_grad`; the shared backbone still changes through student updates.

#### Initialization

```python
class LatentRecCoDi(nn.Module):
    def __init__(self, ..., tokens_per_primitive=1):
```

- Load the SFT checkpoint, such as `beauty_RA_cot/epoch_3`.
- Add SID tokens and resize token embeddings.
- Configure `[PAD]`, `bot_id`, and `eot_id`. The latter two are numeric IDs; among these three, only `[PAD]` is added to the tokenizer.
- Use LoRA: `r=64`, `alpha=32`, targeting `q/k/v/o/gate/up/down_proj`.
- Optional `prj`: `Dropout → Linear(dim, prj_dim) → GELU → Linear(prj_dim, dim) → LayerNorm`.
- Optional `primitive_slot_embeddings`: `NUM_PRIMITIVES * LATENT_PER_PRIMITIVE` learnable vectors for latent initialization.
- Use `attn_implementation="sdpa"` and `torch_dtype=torch.bfloat16`.

#### Student Inference (Implicit)

```text
[system | user | assistant | bot_id] → encoder embeddings
                                      ↓
[latent_1 | ... | latent_M] → T Jacobi refinement steps
                                      ↓
[eot_id | SID | im_end] → decoder recommendation
```

At each Jacobi step, insert the current M latent embeddings into the sequence and replace them with the model's hidden states at those positions. Inference outputs an SID without textual CoT.

#### Teacher Reasoning (Explicit OPD Path)

```text
Student: [enc | latent(M) | eot_id | SID] → forward with gradients
                 ↓ .detach()
Teacher: [enc | latent.detach() | refine_prompt | yr_cot | eot_id | SID]
         → no_grad forward
```

```python
REFINE_PROMPT = "Please re-examine the user's purchase history and provide the correct recommendation."
```

Use `output_hidden_states=True` to obtain teacher `past_key_values` and `hidden_states` for primitive distillation.

#### Complete Forward Flow (`forward_opd`)

```text
1. Student: [enc | latent(M) | dec] → T Jacobi steps
   → logits, CE loss, student_log_prob
2. Extract student_latent_kvs from the final Jacobi KV cache.
3. Teacher (no_grad): [enc | latent.detach() | refine | yr_cot | dec]
   → teacher_logits, past_key_values, hidden_states
4. Compress teacher KV by primitive segment:
   _compress_teacher_kv_cyclic(...) → _kv_distill_loss(...)
5. Extract teacher_log_prob at SID positions.
6. Compute GRPO_base = (sid_match_count - μ_group) / σ_group.
   overall_adv[t] = GRPO_base; no teacher-derived advantage.
7. Combine PPO clipping, gated OPD, primitive distillation,
   CE, diversity, and reference-policy KL regularization.
```

#### Extracting `teacher_log_prob`

```python
_get_teacher_sid_log_probs_opd(teacher_logits, labels, enc_seq_len, M, refine_len, cot_lengths)
```

- Teacher sequence: `[enc | latent(M) | refine_prompt | yr_cot | dec]`.
- Documented SID offset: `enc_seq_len + M + refine_len + cot_lengths[b]`.
- Extract SID log probabilities per sample.

### C2: Asynchronous Training (Stated SEED `ray_trainer` Alignment)

#### Process Architecture

```text
GenProcess (GPU 2)             TrainProcesses (GPUs 3–7)
gen_yo → gen_yr                rank 0: dequeue → broadcast
(actor_rollout_wg)             → reference scores → advantages
         │                    → PPO + OPD + primitive distillation
         └── mp.Queue(maxsize=4) ──►
```

A separate process continuously generates trajectories. Rank 0 of the five training processes consumes the queue and broadcasts batches to the other ranks.

#### Per-Batch Training Flow

```text
1. [GenProcess] gen_yo: latent_generate → yo_sid
2. [GenProcess] gen_yr: latent_refine_generate → yr_cot, yr_sid
3. [GenProcess] Enqueue data in mp.Queue.
4. [Train rank 0] Dequeue → dist.broadcast_object_list → all ranks.
5. [All ranks] Collate → forward_opd(no_grad) → old_log_prob.
6. [All ranks] ref_model.forward_opd(no_grad) → ref_log_prob → KL penalty.
7. [All ranks] compute_grpo_advantage:
   group-normalize per-sample SID matching counts.
8. [All ranks] forward_opd(use_ppo=True)
   → GRPO + PPO clipping + OPD + primitive distillation → backward → step.
```

#### Total Loss

```text
loss = ppo_coef * pg_loss + opd_coef * opd_gated_loss
       + primitive_distill_factor * primitive_distill_loss
       + ce_loss + div_factor * div_loss
       + kl_coef * KL(old_log_prob || ref_log_prob)
```

This is the source's schematic objective. It also describes KL as a reward penalty; verify the actual placement and direction in code.

#### Advantage and PPO Loss

```text
GRPO_base = (sid_match_count - μ_group) / σ_group
advantage = GRPO_base  # No teacher-derived advantage

ratio = exp(student_log_prob - old_log_prob)
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * clip(ratio, 1-clip_ratio, 1+clip_ratio)
pg_loss = mean(max(pg_losses1, pg_losses2))  # Aggregate over response-mask positions
```

Use `clip_ratio=0.2`. Group normalization and masked reduction are shown schematically.

#### Loss Components

**GRPO-Style SID Matching Reward**

Count position-wise matches between the predicted and ground-truth SID instead of assigning a shared binary reward:

```text
gt_sid   = "<|sid_begin|> <s_a_12> <s_b_45> <s_c_78> <s_d_90> <|sid_end|>"
pred_sid = "<|sid_begin|> <s_a_12> <s_b_45> <s_c_34> <s_d_90> <|sid_end|>"
# Matches: a ✓, b ✓, c ✗, d ✓ → score = 3
```

Each SID contains four code tokens (`s_a`, `s_b`, `s_c`, `s_d`), giving a per-sample reward from 0 to 4. Compute group-specific means and standard deviations:

```text
GRPO_base = (scores - μ_group) / σ_group
```

The teacher rescores the same trajectory with the refinement prompt to obtain `teacher_log_prob`.

**Gated OPD Loss (SEED)**

```python
opd_gated_loss = compute_opd_loss(
    log_prob=student_log_prob,
    teacher_log_prob=teacher_log_prob,
    response_mask=response_mask,
    gate_beta=5.0, loss_agg_mode="token-mean",
)
```

The documented confidence gate adjusts distillation strength based on teacher–student disagreement to reduce uncertain supervision.

**KL Penalty (`ref_policy`)**

The source provides this illustrative snippet:

```python
ref_model = LatentRecCoDi(use_lora=False)  # Frozen base model without LoRA
ref_out = ref_model.forward_opd(**inputs)  # no_grad
ref_log_prob = (ref_logits * mask).sum(-1) / mask.sum(-1)
kl_val = F.kl_div(old_log_prob.log(), ref_log_prob, reduction='batchmean')
token_rewards = rewards - kl_val * kl_coef  # apply_kl_penalty
```

This is unverified pseudocode: `ref_logits` is not extracted, and logits/log-probability handling is inconsistent. Check the actual KL computation before using it.

**Primitive Distillation (R-KV Alignment by Primitive Segment)**

```python
teacher_out = self.codi(inputs_embeds=teacher_embeds, ...,
                        use_cache=True, output_hidden_states=True)
teacher_past_kvs = teacher_out.past_key_values
teacher_hidden_states = teacher_out.hidden_states

_prim_kvs = _compress_teacher_kv_cyclic(
    teacher_past_kvs, teacher_hidden_states,
    enc_lengths, primitive_token_ranges, tokens_per_primitive,
    dec_lengths=dec_lengths,
)
_prim_kvs = _align_kv_to_M(_prim_kvs, M)
primitive_distill_loss = _kv_distill_loss(student_latent_kvs, _prim_kvs)
```

**CE Loss:** Cross-entropy on student decoder SID tokens.

**Diversity Loss:** Penalize similar latent states to discourage collapse.

```text
_lat_gram = bmm(lat_h_norm, lat_h_norm.transpose(-1, -2))
_lat_off_diag = _lat_gram * (1 - eye(M))
div_loss = mean(ReLU(_lat_off_diag - 0.3)²)
```

Monitor `lat_ev1` (largest eigenvalue share) and `lat_eff_rank` (effective rank).

### C3: On-Policy vs. Off-Policy Modes

| Mode | Objective | Use |
|------|-----------|-----|
| `use_ppo=True` | GRPO advantages + PPO clipping + gated OPD + KL + primitive distillation + CE + diversity | Asynchronous OPD, described as on-policy |
| `use_ppo=False` | CE + `opd_kl_factor * forward_kl_loss` + primitive distillation + diversity | Legacy off-policy `train_codi.py` |

### C4: R-KV Compression

#### Compression Modes

| Mode | Method | Output Shape | Use |
|------|--------|--------------|-----|
| `rkv_whole` | R-KV scoring over the full CoT | `[B, H_kv, M, head_dim]` | Stage 1 |
| `rkv_cyclic` | Independent R-KV compression per primitive segment | `[B, H_kv, N_PRIM×K_per, head_dim]` | Primitive distillation |
| `identifier` | Direct KV extraction at identifier positions | `[B, H_kv, M, head_dim]` | Skip R-KV scoring |

#### `rkv_cyclic` Steps

1. **Importance I:** Compute attention from decoder Q to segment K, with RoPE applied to Q. Apply softmax → `max_pool1d(kernel=7)` → mean over queries.
2. **Redundancy R:** Normalize K and compute `S = K_norm @ K_norm.T`. Zero the diagonal, retain β recent high-similarity tokens, and apply softmax.
3. **Selection:** Compute `Z = λ·I - (1-λ)·R`, with `mix_lambda=0.07`. Select the top `budget-w` tokens and keep the last `kv_window_size=4` window tokens.

#### KV Distillation Loss

```text
L_KV = mean_over_layers [(||sg[K_t] - K_s||_p + ||sg[V_t] - V_s||_p) / 2]
```

Default: `p=1` (L1 loss). Detach teacher KV states to stop gradients.

### C5: Launch Commands

Replace placeholder paths before running.

```bash
cd /pub/zhuang/distill/reasonrec/R3/pipeline/latent_opd

# Asynchronous version (recommended).
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 WANDB_MODE=disabled python train_opd_async.py \
    --model_name_or_path /pub/.../beauty_RA_cot/epoch_3 \
    --data_path /pub/.../training_RA_cot_train.parquet \
    --output_dir checkpoints/opd_async \
    --gen_gpu 2 --train_gpus 3,4,5,6,7 \
    --num_epochs 3 --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 --learning_rate 8e-4 \
    --ppo_coef 1.0 --opd_coef 1.0 --clip_ratio 0.2 \
    --use_primitive_distill True --kv_compress_mode rkv_cyclic \
    --tokens_per_primitive 4 --div_factor 0.02 --max_cot_length 1024 \
    --use_kl_penalty True --kl_coef 0.01

# Synchronous version for debugging; no separate generation GPU required.
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 WANDB_MODE=disabled torchrun --nproc_per_node=6 --master_port=29501 \
    train_opd_loop.py \
    --model_name_or_path /pub/.../beauty_RA_cot/epoch_3 \
    --data_path /pub/.../training_RA_cot_train.parquet \
    --output_dir checkpoints/opd_loop \
    --num_epochs 3 --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 --learning_rate 8e-4 \
    --ppo_coef 1.0 --opd_coef 1.0 --clip_ratio 0.2 \
    --use_primitive_distill True --kv_compress_mode rkv_cyclic \
    --tokens_per_primitive 4 --div_factor 0.02 --max_cot_length 1024
```

### Key Parameters

Defaults are taken from the supplied README.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_latent` (M) | 8 | Latent token count |
| `num_jacobi_steps` (T) | 3 | Jacobi iterations |
| `ppo_coef` | 1.0 | PPO loss weight |
| `opd_coef` | 1.0 | Gated OPD loss weight |
| `clip_ratio` | 0.2 | PPO clipping range |
| `use_primitive_distill` | True | Enable primitive-segment KV alignment |
| `kv_compress_mode` | `rkv_cyclic` | KV compression mode |
| `tokens_per_primitive` | 4 | Compressed KV tokens per primitive segment |
| `div_factor` | 0.02 | Diversity loss weight |
| `max_cot_length` | 1024 | Maximum teacher CoT length |
| `gen_gpu` | 2 | Asynchronous generation GPU |
| `train_gpus` | `3,4,5,6,7` | Asynchronous training GPUs |
| `use_kl_penalty` | True | Enable reference-policy KL penalty |
| `kl_coef` | 0.01 | KL penalty coefficient |
| `yo_batch_size` | 8 | Asynchronous `generate_yo` batch size |
| `yr_batch_size` | 4 | Asynchronous `generate_yr` batch size |
| `update_ptools` | True | Enable primitive-library evolution |
| `max_total_ptools` | 7 | Dynamic primitive-library size cap |

---

## Special Tokens

| Type | Purpose | Value | Reference |
|------|---------|-------|-----------|
| Thinking markers | Teacher CoT boundaries | `🧐` / ` Millennia` (documented Qwen3 IDs 151667/151668) | OneRec-Think |
| Latent markers | Student latent boundaries | `bot_id` / `eot_id` (`ori_vocab_size+1/+2`, numeric IDs) | CoDi |
| SID markers | Recommendation encoding | `<|sid_begin|>`, `<|sid_end|>`, `<s_{a-d}_{0-255}>` | Custom |
| Primitive markers | Primitive-segment boundaries | `<ptool_end_i>` | Custom |

**Teacher textual path:**

```text
[system | user | assistant | 🧐 | CoT text |  Millennia | SID | im_end]
```

The OPD teacher path additionally includes detached student latents and a refinement prompt.

**Student path:**

```text
encoder: [system | user | assistant | BOT]      ← append bot_id
latent:  [latent_1 | latent_2 | ... | latent_M] ← Jacobi refinement
decoder: [EOT | SID | im_end]                   ← prepend eot_id
```

---

## File Structure

```text
latent_opd/
├── model_codi.py          # Core LatentRecCoDi model
│   ├── __init__           # Loading, tokenizer expansion, LoRA, projection
│   ├── forward_opd        # Student/teacher forward, distillation, losses
│   ├── latent_generate    # Student inference: Jacobi + beam search
│   ├── latent_refine_generate      # Teacher CoT generation from latents
│   ├── _compress_teacher_kv_cyclic # R-KV compression per primitive segment
│   ├── _kv_distill_loss   # KV distillation
│   └── _align_kv_to_M     # KV length alignment
├── core_algos.py          # Forward KL, gated OPD, GRPO advantages
├── kl_utils.py            # KL utilities
├── rkv_utils.py           # R-KV utilities
├── train_opd_async.py     # Main entry: SEED-style asynchronous training
│   ├── gen_process        # GPU 2, actor_rollout_wg
│   ├── train_process      # GPUs 3–7, stated ray_trainer alignment
│   │   ├── ref_policy     # Frozen reference model for KL
│   │   └── actor update   # PPO + OPD + primitive distillation
│   └── main()             # Spawn two process groups
├── train_opd_loop.py      # Synchronous per-batch training for debugging
├── train_codi.py          # Legacy forward-KL OPD training
├── train_beauty_RA_cot.py # Phase B teacher SFT
├── eval_codi.py           # Evaluation
├── generate_yo.py         # Legacy full-dataset student y_o generation
├── generate_yr.py         # Legacy teacher y_r generation from latents + refinement
├── dataset_codi.py        # CoDi dataset and collator
├── dataset_opd.py         # OPD dataset and collator
├── update_ptools.py       # Primitive-library evolution utility
├── ds_config_zero2.json   # DeepSpeed ZeRO-2 configuration
├── reasoning_primitives/benchmarks/rec/  # Phase A tools
│   ├── prepare_data_parquet.py         # Prepare QwQ ReAct inputs
│   ├── react_vllm.py                   # QwQ-32B ReAct inference
│   ├── react_traces_to_results.py      # Convert traces and filter correctness
│   ├── induction.py                   # Primitive induction
│   ├── generate_induced_configs.py    # Convert ptools to Python modules
│   ├── simulate_cot_all.py            # Generate simulated CoT
│   ├── convert_traces_to_cot_parquet.py # Convert CoT to training parquet
│   └── ptool_updater.py                # Dynamic ptool updates
├── requirements.txt      # Training, inference, and development dependencies
└── README.md             # This file
```

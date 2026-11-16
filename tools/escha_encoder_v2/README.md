# L0xRE Escha/MCG V2: measured allocation and two-sided LDLQ

V2 is a separate worktree and run directory. V1's source, fitting, export,
checkpoints, evaluation and production model are untouched.

The numerical tools remain upstream EXL3 v1.6.7, commit
`8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`. No upstream optimizer, Fisher
collector, LDLQ, Viterbi encoder, codebook or inference kernel is rewritten.

## What changes

- Use original-MCG RFN anchors from V1 through `sc_rfn_probe.py`.
- Measure end-to-end sensitivity using `sc_measure.py --streaming --shaped`,
  two noise levels, 16 measurement documents and 64 separate Hessian rows.
  `measure.py` reuses the original parser, noise/capture/forward/measurement code
  but selects individual body targets. Upstream groups dense gate/up, qkv and
  qkv/z for its own equal-width fused kernels and includes the head; those are
  inappropriate measurement units for Escha's independent widths/frozen head.
- Run the original `sc_optimize.py` on feasible projections only, K2–K3,
  integer steps, with the exact body-bit budget remaining after fixed tensors.
  Its per-bit error ratio1.96 remains a model assumption, not a measured MCG
  rate curve. Sensitivity/predicted KL is not full-model measured KL.
- 336 projections may change K. The 64 qkv/q projections without a corresponding
  K3 fast path stay K2. The existing runtime supports all four FFN gate/up
  width pairs, so EXL3-specific equal-width ties are deliberately not imposed.
- Collect full YaQA **Hout**, not just its diagonal, in two-layer ranges using
  the original BF16 source. Pair it with EXL3's online calibrated Hin; this is
  two-sided LDLQ. Use `--out_scales yaqa --scaled_hessian` in conversion.
  Upstream recommends this Hout/online-Hin lane over assuming six alternating
  factor passes must always be better. The helper also supports `--sides both`.
- Keep the original vocabulary codec/bytes, small tensors, tokenizer, correction
  tensors, shared MCG LUT, dependencies, Hadamard128 and 16x16 tiles.
- Rewrite GGUF tensor dimensions/offsets when K changes. Metadata is copied as
  opaque bytes; unchanged payloads are verified byte-identical. Model bytes must
  be <=8,619,127,680. An unspent budget remainder is reported, never disguised
  with padding or new unsupported precision classes.

Allocation is a constrained **estimated-quality** optimization, not proof of a
globally best quant or a guaranteed KLD improvement. The final native measurements
decide whether the new candidate improves on V1 and B02-A2.

## Trace/calibration provenance

The default first V2 calibration is 250 documents, up to2048tokens, from the
existing pinned BF16-reference calibration split. The packed artifact contains
317,765tokens across15categories. Selection is reproducible and excludes exact
public-evaluation and known task-manifest prompts. No golden probabilities are
included. This is the packed format consumed by `sc_trace`/`sc_measure`; it is
**not** a claim that new self-sampled responses were generated.

`prepare --trace-model <verified-high-precision-EXL3-directory>` instead runs the
actual upstream `sc_trace.py`. It requires >=6bpw reporting precision and the
same tokenizer. That optional teacher must fit the selected resources; V1's
low-bit model is never silently used as the self-trace teacher. The original
BF16 model remains the Fisher/quality target in either case. No paid/external
generation API or additional GPU is assumed.

## 24GB GPU support and mandatory gates

The stock YaQA loader places the entire BF16 model on one device, which does not
fit a4090. `hf_stream.py` changes residency only: CPU-backed original parameters,
one decoder block on CUDA, saved-tensor hooks storing activation intermediates
on CPU and parameter views against their original CPU storage. View offsets are
preserved. Saved activation copies are bounded to24GiB per row. The own reservation
memory limit becomes96GiB only after V1 has finished; the 3060 service is untouched.

Before expensive sensitivity/factor collection, V2 must pass:

1. Original BF16 forward vs retained reference (8positions, KL<.005).
2. CUDA resident/offloaded forward, input-gradient and upstream full-Hout parity.
3. Changed-width native dispatch fixture: all12roles, all4FFNpairs, finite full
   inference and zero frozen-runtime fallbacks. Its silenced projections are
   **only a format fixture**, never a fit initialization or quality model.
4. An actual original Qwen/GDN full-backward YaQA pilot on the longest calibration
   row, first two layers. This verifies resource fit and architecture support.
5. Actual two-sided MCG K3 fitting of an original BF16 projection, YaQA output
   scaling activation, circular packing and native decoder equality.

Full-Hout collection can be substantially more expensive than V1. No runtime or
ETA is invented before the GPU pilot. A failed gate stops the pipeline with its
attempt log and source/input hashes; it does not downgrade to iid/diagonal-only
or uncalibrated fitting. All GPU gates remain pending until actually executed.

Packed Hout factors need approximately100GB. Factor chunks alternate between the
owned SSD run directory and `/home/sean/work/l0xre-exl3-v2-factors` on the other
volume; a flat symlink collection feeds upstream `--hessians`. Disk admission
reserves OS/conversion/export headroom. V1 artifacts are not deleted for space.

## Automation and resume

`run.py prepare` freezes code, upstream tools, runtime roles, calibration settings
and paths. `run.py run` must be the existing4090guardian's child and waits behind
V1 in its queue. It refuses GPU work until V1 has full export, KLD and completed
coding/DFlash measurements, even if V1's quality result is negative.

Stages: trace packing (or optional fresh self-trace), teacher/parity/dispatch/YaQA
pilot gates, MCG RFN probe, shaped SC measurement, constrained upstream allocation,
full YaQA factors, two-sided fitting, parallel bit-exact CPU packing, structural
GGUF export, native development116 comparisons against **both V1 and B02-A2**,
then the original paired DFlash and canonical Club150 ON/OFF checks vs B02-A2.
The existing116documents remain development data, not a new untouched holdout.
No automatic production promotion occurs.

SC measurement resumes its own partial JSON. YaQA commits per two-layer chunk.
Complete stages bind code/plan/input and output hashes. Partial/corrupt stage
outputs are archived before regeneration. Fitting uses the upstream checkpoint
plus V1's tested durable storage guard and a frozen recipe/calibration/factor
context. Independent projections pack in three CPU workers, with per-projection
hash-checked receipts. A final GGUF may be resumed only if its receipt and all
replacement payload hashes agree.

Use the existing V1 venv and pinned prebuilt extension by adding the V2 source
tree and V1 `torch-extensions/exllamav3_ext` directory to PYTHONPATH. Both use
the same pinned numerical source and ABI. No V1 packages or files are changed.

## Validation performed during build

CPU tests pass: exact no-op GGUF bytes; balanced K2↔K3 resize with unchanged
vocabulary/metadata; over-budget/protected-tensor rejection; original upstream
optimizer reallocating precision to a more sensitive tensor while pinning an
unsupported width; exact CPU forward/gradient equality including nonzero storage
offsets; V1 completion gate; Unicode JSONL and public-prompt exclusion. The real
template's2058-tensor offset/size layout matches the writer's computed layout.
The GPU gates and full V2 quality are **not yet qualified**.

# L0xRE Escha/MCG V2: measured allocation and two-sided LDLQ

V2 is a separate worktree and run directory. V1's source, fitting, export,
checkpoints, evaluation and production model are untouched.

The numerical tools remain upstream EXL3 v1.6.7, commit
`8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`. No upstream optimizer, Fisher
collector, LDLQ, Viterbi encoder, codebook or inference kernel is rewritten.

## What changes

- Use original-MCG RFN anchors from V1 through `sc_rfn_probe.py`.
- Measure end-to-end sensitivity using `sc_measure.py --streaming --shaped`,
  four noise amplitudes2.5/1.96/1/.5, two draws, and two disjoint24-document
  replicates with64shared shaping documents. Each replicate contains all15
  categories, including four code, four tools and four reasoning documents.
  Critical-domain documents must include at least16scored assistant tokens in
  the actual measured prefix; prompt-only prefixes do not count as response coverage.
  `measure.py` reuses the original parser, noise/capture/forward/measurement code
  but selects individual body targets. Upstream groups dense gate/up, qkv and
  qkv/z for its own equal-width fused kernels and includes the head; those are
  inappropriate measurement units for Escha's independent widths/frozen head.
- Run the original `sc_optimize.py` on feasible projections only, K2–K3,
  integer steps, with the exact body-bit budget remaining after fixed tensors.
  Audit/replace its discrete choices with the reviewed exact dynamic program
  over the same additive cost table. Its per-bit error ratio1.96 remains a model assumption, not a measured MCG
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
preserved. Duplicate saved views share version-aware CPU snapshots without holding
GPU storage alive. Non-reentrant block activation checkpointing recomputes each
block during backward; recomputation is excluded from Collector forward counts.
This preserves the 2048-token calibration and BF16 precision rather than shortening
the data. Saved activation copies are bounded to24GiB per row. The own reservation
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

## Reviewed quality run (SOL-HANDOFF.md / REVIEW.md, November16)

The original failed run and recovery receipts remain immutable. The reviewed run
uses a fresh `quality-run-20261116` plan with configurable stage arguments bound
to its source/data hashes. The prior temporary-Parameter binding fix was actually
GPU-qualified: original2048token backward, twelve full-Hout factors, nativeMCGK3
fit and zero-delta gradient tests. New longer-prefix teacher gates still rerun.

Before expensive measurements, actual stored-half K2/K3 YaQA fits of SSMout,
FFNgate and FFNup diagnose the rate curve and verify native decoder equality.
The measurement envelope must cover their observed RFNs relative to V1anchors.
This is a representative diagnostic, not proof every tensor follows one ratio.

Independent SC rankings must reach Spearman.75 and weighted width disagreement
<=.25. The mean-table exact recipe must not regress either replicate. These
predeclared screening thresholds are hypotheses; a failure stops for review.
An unchanged optimum is permitted and labeled. Fixed-map one-sided and YaQA
controls always run; changed maps also get one-sided and YaQA candidates,
forming a true2x2 with the same calibration data.
No forced swaps or claim that exact predicted allocation proves model quality.

Each candidate gets native paired development116 KLD vs B02-A2 **and V1**.
Selection requires a positive paired95CI, no code/tools/reasoning mean regression,
and the existing per-domain and p99/p99.9 margins. Only an eligible development
winner gets canonical Club150 OFF/ON and paired DFlash tests. Those must pass
before opening the final reserved data. No winner means retain B02-A2.

The published pool no longer has unused reasoning examples after known prior
exposures. Reserve32known-disjoint published documents (including code/tools),
plus16new deterministic arithmetic reasoning fixtures, before any V2 selection.
Their probability labels come only from the original BF16 teacher **after** a
winner is fixed; neither prompts nor outputs are used in training, factors,
sensitivity or model selection. Full native B02-A2/winner evaluation covers this
48-document final reference once. The constructed reasoning scope is limited,
and completeness of all historical public exposure is not provable; both
limitations are recorded. No automatic promotion.

Final GGUFs, native rows, source hashes, recipes, factors and failure logs are
preserved. After each candidate has a verified final model and native output,
only that new candidate's reproducible conversion/export scratch is reclaimed,
with a cleanup receipt, to fit the ablations within current disk capacity.
This never removes V1, B02-A2, calibration sources, factors or final weights.

## Preserve earned production head changes

A source audit found B02-A2 differs from the original V1 geometry template in all
three output-head tensors (codes/scales/zero-points), while embeddings and output
norm match. Primary V2 exports preserve **B02-A2's entire nonbody payload**;
the400coded body projections are still fitted from original BF16 source, within
the same K2/K3 geometry/byte contract. Native role fixtures retain the original
geometry template solely as a format/dispatch test.

`head_control.py` verifies that the only nonbody differences are those three head
tensors, then measures a labeled V1-body/production-head transplant. There is no
new vocabulary fitting, width, codebook or runtime change. Native results compare
V2 with B02-A2, raw V1, and this head-matched control. That control is eligible
for the same development/task/speed/final-holdout gates; it is not silently discarded
if costly fitting fails to improve it. Each fitted body also receives a paired
native comparison of the two **existing** original/production head payloads;
there is no assumption that a head fitted on one body must win on another.
Only the better eligible head/body variant is retained as a full GGUF; the other
is exactly reconstructable from that identical fitted body and its pinned head
donor, with all native results and hashes retained. Production remains untouched.

Compiled shards retain every encoded weight and their hash manifest before
redundant per-module quantization/state scratch is reclaimed. Final GGUFs and
native reports remain preserved. This permits the fourth factorial control
without deleting V1 or fitted final weights for space.

# EXL3-powered Escha encoder V1

Upstream algorithms are pinned to `8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`.
The added files live entirely in this directory. No EXL3 algorithm implementation,
L0xRE runtime, inference kernel, codebook or allocation has been changed.

The adapter maps the EXL3 ring's codewords to Escha tile coordinates, verifies all
overlapping windows including circular closure, and repacks into Escha's physical
bit order. It copies the already folded FP16 `suh/svh` vectors to `rin/rout`:
both formats use `diag(rin) H128 Q H128 diag(rout)`. There is no additional global
scale or Hadamard normalization to apply. Export preserves template correction
vectors (ignored by the frozen default `ESCHA_APPLY_BIAS=0` runtime). An upstream
projection bias is rejected pending an explicit compatibility implementation.

The 400-projection allocation is read from the template, including exact per-tensor
widths. Export replaces only codes and existing scale vectors. Vocabulary, small
weights, tensor schema, metadata and dependencies stay byte-identical. Output must
remain exactly **8,619,127,680 bytes**. Existing model files are never overwritten.

## Entry points

- `validate.py`: all 65,536 MCG values, stored dependency tables, walking-bit,
  random and real K2/K3 tiles, negative circularity checks, and transformed dense
  projection comparisons against upstream's independent CPU reference.
- `calibration.py`: packages retained calibration-only token prefixes into the
  exact format consumed by EXL3's calibration loader; no teacher labels are used.
- `pilot.py`: uses upstream `quantize_exl3`, including calibration finalization,
  scaled Hessian/LDLQ, Viterbi fitting, scale fitting and packing. Supports a
  projection or complete layer and hash-checked resume per projection. Checks
  retained native CUDA decoding on sampled newly encoded tiles.
- `export_pilot.py`: exports the pilot into a copy of the original GGUF.
- `workflow.py prepare/convert/export`: freezes source/calibration/tool hashes,
  generates the exact allocation recipe, invokes upstream automatic layerwise
  conversion/checkpointing, and exports with independent native reconstruction
  checks and resumable per-projection packing. Conversion requires projection and
  complete-layer runtime gates. `--resume` uses upstream checkpoint state.
- `convert_checked.py`: invokes the upstream converter with an additional guard
  that rejects uncalibrated fallback. No fitting algorithm is replaced.
- `pipeline.py`: guardian-owned automation from projection through full layer,
  both native runtime smokes, full conversion/export, and the existing 116-document
  KLD comparison. A failure stops subsequent stages. `--full` enables the full run.
- `qualify.py`: subsequent guardian job, requiring a complete full-model export.
  Uses the frozen 20-document DFlash timing fixture in paired ABBA order, and the
  canonical Club150 ON/OFF harness against both B02-A2 and the new candidate.
  Records per-pack regressions, document-paired throughput confidence intervals,
  exact loaded backend and zero-fallback checks. It performs no promotion.

`gguf_patch.py` is reused unchanged from the retained local quantization tools
(original SHA256 `6f6f156c859f0b10b0eef767767c6d8ade5f4be1db149575d1b825c189fd58fe`).
It performs atomic copy-and-patch output and hashes every tensor, including all
unchanged tensors. A no-op export was verified against the original full digest
`b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543`.

## Z840 execution

The isolated work root is `/mnt/ssd/ai-builds/l0xre-exl3-encoder-v1`.
Use its `venv/bin/python`, `CUDA_HOME=<work>/cuda`, `CC=/usr/bin/gcc-13`,
`CXX=/usr/bin/g++-13`, `NVCC_CCBIN=/usr/bin/g++-13`, `MAX_JOBS=8`,
`TORCH_CUDA_ARCH_LIST=8.9`, and `TORCH_EXTENSIONS_DIR=<work>/torch-extensions`.
The CUDA view contains symlinks to the already installed CUDA 13.3 toolkit.
The first compiler attempt used incompatible conda GCC 15 headers; preserve its
failed receipt. The isolated compiler selection fixes that mismatch.

GPU work must be admitted through the existing guardian at
`/mnt/ssd/ai-builds/l0xre-low-upgrade-20261112/leases/4090`.
Each job verifies the guardian child PID and GPU UUID. The existing RTX 3060
production service and the occupied local RTX 5090 are outside this job's scope.
Do not release the existing 4090 reservation or restore its borrowed service
without Sean's instruction; that reservation predates this task.

Example guardian command (not a direct shell launch):

```text
<work>/venv/bin/python -u <work>/exllamav3/tools/escha_encoder/pipeline.py
  --assets /mnt/ssd/ai-builds/l0xre-low-upgrade-20261112
  --work <work>/pipeline --cpu-proof <work>/cpu-compatibility.json --full
```

Full conversion checkpoints are upstream's `conversion/ckpt/`; packed Escha
checkpoints live in `full/export-checkpoints/`. Resume is bound to immutable
source, template, calibration, implementation and checkpoint hashes. A completed
projection is skipped only after its artifact hashes match.

## Qualification status and limits

CPU codec/transform checks and the no-op GGUF export passed. CUDA encoding,
projection/layer runtime compatibility and full quantization remain unqualified
until the corresponding receipts actually exist and pass.

The exact retained reference is **B02-A2 development116 KLD
0.09813687512473697**, model SHA256
`ecdd8679d7cfe023cadb3eb8726fb09a869a6f71f69c183d71144973b0785bc7`.
The existing 116 documents have already been used for development; do not call
them a new untouched holdout. Compare using the same masks, token prefixes,
native scorer and frozen SM89 runtime. Independent final holdout, coding/agentic
quality and DFlash speed still require their own measured qualification.
Neither `--full` nor a lower local reconstruction error implies those gates pass.
No production model promotion is performed by this workflow.

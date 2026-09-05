# Archify delivery evidence — GPT-OSS-Lite

The [interactive visual guide](gpt_oss_visual_guide.html) links four standalone showcase architecture and workflow diagrams.

All four: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed.**

Chrome checked at 1440×900, 1600×1000, 1920×1080 and 2048×1320 in light/dark. All required viewport measurements passed horizontal/vertical containment, minimum projected text size and viewer-control clearance.

## Artifact bindings

### Model Architecture

- Diagram type: `architecture`
- Output: [gpt-oss-lite-model-architecture.html](gpt-oss-lite-model-architecture.html)
- Specification: `docs/gpt-oss-lite-model-architecture.archify.json`
- Artifact SHA-256: `c710d2de8f1ef06711f5486591ff3bafb6e07e4887ebb0bb4024a6b1858a7e4d` (716,189 bytes)
- [Browser receipt](gpt-oss-lite-model-architecture.visual-check.json) · [Screenshot contact sheet](gpt-oss-lite-model-architecture.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Data Pipeline

- Diagram type: `dataflow`
- Output: [gpt-oss-lite-data-pipeline.html](gpt-oss-lite-data-pipeline.html)
- Specification: `docs/gpt-oss-lite-data-pipeline.archify.json`
- Artifact SHA-256: `44302a663debaa6700d1946f0bdba5dfeeb1115ab8c94bc285ea82d561be21f3` (714,641 bytes)
- [Browser receipt](gpt-oss-lite-data-pipeline.visual-check.json) · [Screenshot contact sheet](gpt-oss-lite-data-pipeline.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Training Workflow

- Diagram type: `workflow`
- Output: [gpt-oss-lite-training-workflow.html](gpt-oss-lite-training-workflow.html)
- Specification: `docs/gpt-oss-lite-training-workflow.archify.json`
- Artifact SHA-256: `2845e284382521246335248a86ce1d4cb737aa9382f3c5ed52a909fab8e57908` (716,818 bytes)
- [Browser receipt](gpt-oss-lite-training-workflow.visual-check.json) · [Screenshot contact sheet](gpt-oss-lite-training-workflow.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Optimization Stack

- Diagram type: `architecture`
- Output: [gpt-oss-lite-optimization-stack.html](gpt-oss-lite-optimization-stack.html)
- Specification: `docs/gpt-oss-lite-optimization-stack.archify.json`
- Artifact SHA-256: `6f64c4ef389ca18b38d6c18152a0dc73a6e995bcabfeaabb18e073bb3b523c67` (721,636 bytes)
- [Browser receipt](gpt-oss-lite-optimization-stack.visual-check.json) · [Screenshot contact sheet](gpt-oss-lite-optimization-stack.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

## Verification limits

Parameter count: ~502M total / ~247M active parameters per token across 12 layers. Evaluated with pure PyTorch SDPA, optional YaRN 128K context window, and Top-2-of-8 MoE. Single-GPU A100 80GB baseline budget estimated at ~32 hours for 8.0B tokens.

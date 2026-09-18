# Scope Rejection Implementation Plan

User authorized implementation after observing non-herb photos classified authentic. Codex owns design and acceptance; actual DSH headless implements. Original model project remains read-only.

## Evidence

- Current checkpoint is experimental_closed_set=true, open_set_calibrated=false.
- Existing rejection policy depends on a calibrated three-head model, which this checkpoint does not provide.
- Located original data contains positive-only and closed-set manifests. No independently labelled out-of-scope corpus has been verified. A data-path question is pending.

## Contracts

1. Add a separate scope decision with accepted/rejected/unavailable and reason/version/score. Missing or invalid scope artifact means unavailable, never proof that an image is out of scope.
2. Default scope enforcement: only accepted images may receive a public authenticity conclusion. Others become inconclusive without a fabricated authenticity prediction. Existing immutable predictions remain untouched; API/frontend expose their historical experimental status without affirmative verdicts when scope evidence is absent.
3. Keep original model flags unchanged. A scope gate is not proof of full open-set authenticity calibration. Do not overwrite original weights.
4. Scope classification baseline: frozen existing encoder features + scikit-learn StandardScaler/LogisticRegression. Manifest contains image_path, scope_label (in_scope/out_of_scope), split (train/calibration/test), group_id, source, consent. Never infer scope labels from closed-set authenticity labels or user uploads. Validate consent, duplicate content and group separation before training.
5. Fit only train; choose threshold only calibration; evaluate once on separate test. Explicit operator-supplied maximum false-accept and false-reject rates; report counts, rates and confidence intervals. Insufficient or failing evidence produces a candidate, never a deployable artifact. No claims of universal OOD protection.
6. Serialize numeric parameters/metadata in JSON (no pickle). Bind artifact to checkpoint SHA256 and feature dimension; reject corrupt/mismatched/unapproved artifacts. Publish only after held-out criteria pass. No sample count or acceptance limit silently invented as scientific validation.

## Work

- Implement scope gate, Worker outcome integration, API/frontend contract and regression tests (including old high-confidence predictions and unavailable gate).
- Implement manifest validation, frozen feature extraction, training/calibration/evaluation CLI, and JSON artifact verification with small synthetic tests expressly not evidence of real performance.
- Provide CSV template and Windows commands; avoid copying or automatically training on user case images.
- Version analysis idempotency by scope policy/artifact as well as checkpoint to avoid replaying pre-gate results. Preserve historical records.
- Verify targeted backend/frontend checks, model import isolation, migrations if needed, real missing-artifact behavior, then restart local services through existing scripts.

## Completion Boundary

Engineering can be verified without a real OOD corpus. Actual refusal accuracy cannot be claimed or enabled without independently labelled and consented data. If data remains unavailable, report the executable pipeline and unavailable gate honestly, including that conservative refusal is not learned OOD recognition.

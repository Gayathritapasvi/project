# Results

Generated deliverables (produced automatically by training + inference):

* `training_loss.png`, `validation_loss.png` — rendered by `train.py` from the
  loss CSVs in `<Lang>/logs/` (the Colab notebook sections 10 and 12 display
  them while training).
* `test_generated_audios/` — E1/E2a/E2b/E3/E4 (English) and T1/T2a/T2b/T3/T4
  (Telugu) synthesised wav files, written by `<Lang>/inference.py`.
* `variance_adaptor_plots/{duration,pitch,energy}/` — per-test-sentence plots
  of the predicted phoneme durations, F0 and energy, written by
  `<Lang>/inference.py` (raw values in `variance_adaptor_plots/<ID>_variance.json`).

**Current status (2026-11-01): pending training runs.** The folders contain
placeholder `.gitkeep` files only; no loss plots or generated audio are
shipped before the real 5-hour training runs produce them (the Colab notebook
sections 10-13 execute those runs). The only audio this repository has ever
generated comes from the synthetic smoke test (`tools/smoke_test.py`) and is
deliberately NOT included here so it cannot be mistaken for real model
output.

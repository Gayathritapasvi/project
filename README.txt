FASTSPEECH2 ENGLISH VOICE-FIX PATCH  (2026-11-04)
=================================================
Applies to the project extracted at /kaggle/working/project.

WHY: the English model was trained as ONE voice over mixed-gender Svarah
(117 speakers), so its pitch drifts between male/female registers sentence
to sentence (measured 132-210 Hz on E1-E4 vs a stable ~214 Hz for Telugu),
and inference omitted the 'sp' word-boundary phones that training data has.

FILES: all paths are relative to the project root. Unzip over the project
so they replace the existing copies.

KAGGLE STEPS (after unzipping this patch over your project):
  1. %cd /kaggle/working/project
  2. !python tools/migrate_en_gender_speakers.py --data-root data/English
     (offline; relabels your EXISTING preprocessed data by the gender
      column already in corpus/manifest.csv - no re-download, no
      re-alignment, no feature recompute. Optional: add --dry-run first.)
  3. Train FRESH (the speaker embedding table changes, so no --resume):
     !python English/train.py --total_step 15000
     (~4-5 h at your measured speed; keep checkpoints on persistent storage)
  4. !python English/inference.py --speaker_id 0
     (one fixed speaker -> one consistent voice across E1-E4 + E2b;
      --speaker_id 1 gives the other gender's voice)
  5. Optional regression check for Telugu (inference convention changed):
     !python Telugu/inference.py --speaker_id 0
     and compare with the previous T1-T4 wavs.

NOTES:
 - Do not press Ctrl-C during the first inference import (inflect/typeguard
   instrumentation takes 20-60 s; it only happens once per runtime).
 - The resume cell needs %cd /kaggle/working/project first (the earlier
   "can't open file '/kaggle/working/English/train.py'" was a wrong cwd).
 - Report: record the speaker strategy (gender -> 2 voices) and the
   excluded-row counts from migration_report.json.



## Model Checkpoints

Due to GitHub file size limits, the trained model checkpoints are hosted externally:

- **English Model Checkpoints:** [Download English Checkpoints]( https://drive.google.com/drive/folders/1eLOUMcjLmtHHHVF7zhP3Bi-pwSxVdkD9?usp=sharing )

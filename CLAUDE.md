# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

COGS 109 course project: does adding NCAA stats improve prediction of NBA PPG (first three post-combine seasons) beyond NBA Draft Combine measurements? The project is a set of Jupyter notebooks (no package, build system, linter, or test suite). `README.md` has the full data dictionary, matching policy and checkpoint requirements.

## Setup and running

- `pip install -r requirements.txt` (the builder notebook also has a `%pip` cell).
- Run Jupyter **from this folder**: every notebook uses `Path.cwd()` as `ROOT` and reads/writes `data/`, `reports/`, `output/` relative to it.
- Execute in order: `00_build_dataset.ipynb` (once) → `01_member1_combine`, `02_member2_ncaa`, `03_member3_combined` (any order) → `04_group_comparison`.
- Headless run of one notebook: `jupyter nbconvert --to notebook --execute --inplace 01_member1_combine.ipynb`.

## Architecture

- **Two-CSV split to prevent leakage:** `00_build_dataset` writes `data/predraft_features.csv` (combine + NCAA predictors) and `data/nba_outcomes.csv` (`ppg_3yr` etc.), joined on `player_id` by the model notebooks. NBA outcomes must never be used as predictors. A third file, `data/ncaa_match_audit.csv`, records every match decision.
- **Only notebook 00 touches the network.** It reuses `output/nba_early_career_ppg.csv` (downloads from the author's `COGS109_NBA` repo if missing) and caches ESPN player-box Parquet files in `data/raw_ncaa/`. If both output CSVs exist, it skips the build entirely, so **delete `data/predraft_features.csv` and `data/nba_outcomes.csv` to force a rebuild** after changing matching rules. Model notebooks make zero requests.
- **Name matching is deliberately strict:** normalized name (accents/punctuation/suffixes stripped) + exact combine year = NCAA season-ending year, a per-NBA-ID alias dictionary, no fuzzy matching. Ambiguous names and multi-school identities are excluded (audit only). NCAA predictors are the season ending in the combine year, even if the player later returned to college.
- **The three model notebooks are structurally identical** and differ only in `MODEL`, `PREDICTORS` and `FORMULA` (statsmodels OLS formula; position is a categorical with `Guard` as reference). Each: loads and joins the CSVs, asserts the matched-sample invariants (candidate count 1, team count 1, `ncaa_season == combine_year`), splits 80/20 with `random_state=42` stratified by `position_group` (616 train / 154 test, identical across models), fits on train only, plots diagnostics + VIF, optionally refits with `FIX_FORMULA`, and evaluates once on the test set. Changes to shared logic (sample filter, split, diagnostics) must be made in all three notebooks, or the comparison is no longer apples-to-apples.
- **Notebook outputs feed `04_group_comparison`:** each model writes `reports/{MODEL}_metrics.csv`, `{MODEL}_coefficients.csv` and `{MODEL}_*_diagnostics.png`; notebook 04 concatenates the metrics (asserting equal train/test sizes) and writes `reports/group_comparison.csv` and shared EDA plots.
- `FIX_FORMULA = None` by default; it is only set when a real training diagnostic or VIF (> 5) justifies it, and the change is documented for the presentation.

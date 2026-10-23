# Built for the NBA? | COGS 109 Checkpoint 1 starter package

**Question:** How well do NBA pre-draft physical measurements and athletic tests predict early-career NBA points per game (PPG), and do those associations differ between guards, forwards, and centers?

**Status:** Executable notebooks and preprocessing/modeling code are complete and passed *synthetic-data smoke tests*. This archive intentionally **does not contain a real NBA modeling CSV or observed model scores**: the live data download was unavailable inside the artifact-building environment. Run `00_build_dataset.ipynb` with Internet access (or supply real raw CSVs) to generate the real dataset. The folder `output/` is initially empty. Never report simulated test numbers as NBA results.

## Start here

1. Extract this entire ZIP to a single folder. Open the folder in JupyterLab, VS Code notebooks, or a Python-enabled environment with Internet access. Keep `.py` helper files in the **same working directory** as the notebooks. If using Colab, upload/extract the whole folder to the runtime and set the working directory to that folder.
2. Run **`00_build_dataset.ipynb`** from top to bottom. It installs its dependencies, loads official NBA Combine and NBA season scoring data (using `nba_api`), cleans data, and saves `output/nba_early_career_ppg.csv`.
3. Confirm the join audit, the eligible model sample size `n`, position distribution, and missingness. Save a screenshot or summary for the team's Slide 1 and Slide 2.
4. Give **Member 1** `01_member1_size_model.ipynb`, **Member 2** `02_member2_athletic_model.ipynb`, and **Member 3** `03_member3_interaction_model.ipynb`. Everyone uses the **exact same** modeling CSV and split seed 42.
5. Run **`04_group_comparison.ipynb`** after all three notebooks have saved their final model results. The group comparison CSV appears in `output/group_model_comparison.csv`.
6. Use the reported *real* sample size, coefficients, VIF, test MSE and test R² in your recording. **Only claim an assumption passes after examining its plots/checks.**

## Data source and research methodology

- NBA Draft Combine official fields: [nba_api DraftCombineStats documentation](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/draftcombinestats.md).
- Official regular-season totals: [nba_api LeagueDashPlayerStats documentation](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashplayerstats.md).
- Alternative all-in-one source: [Wyatt Walsh NBA Database (Kaggle)](https://www.kaggle.com/datasets/wyattowalsh/basketball). If you use this, you will need to export its equivalent *actual* combine and league-wide season totals into the source CSV formats below. Tables/columns can differ between database releases.

**Observation:** one row per unique NBA player ID using the player's **first eligible combine year**, not one row per player-season. **Input features:** combine measurements taken before NBA performance. **Outcome:** `ppg_3yr = sum(points) / sum(games)` in the first three NBA seasons following combine year (2001–2022 combine cohort, with 2001-02 through 2024-25 as the corresponding follow-up seasons). Regular-season games only. Players with zero games in the three-season period have **undefined PPG** and are excluded from this *scoring conditional on reaching the NBA* study. They remain in `combine_join_audit.csv` for transparency.

**Important scope:** The question is about *scoring among combine attendees who appeared in an NBA game in the following three seasons*, not overall prospect success, all draftees, or causation. Skipped combines, missed drills, late NBA arrivals, and undrafted attendees introduce important selection effects. Draft/combine year is a baseline proxy for NBA entry year. Validate each cohort before presentation.

**Join method:** NBA numeric `PLAYER_ID` matched to `PLAYER_ID` across the combine and official league season totals. Names are for reporting only, **not** fuzzy join keys. Each player's total points/games are aggregated over `combine_year`, `combine_year+1`, `combine_year+2`. Point totals are not averaged across seasons.

## If the NBA API doesn't work

Direct `stats.nba.com` calls may return timeouts, restricted access, or rate limits. The build script caches successfully downloaded seasons in `raw/` and will retry them on rerun. It **refuses to produce a partial sample when seasons are missing**, since missing seasons would corrupt the target.

Alternatively provide BOTH complete real source files in `raw/`:

- `raw/combine.csv`: `PLAYER_ID`, `PLAYER_NAME` or `FIRST_NAME` + `LAST_NAME`, `POSITION`, `SEASON` or `SEASON_YEAR` (e.g., `2018-19`), and measurements such as `HEIGHT_WO_SHOES`, `HEIGHT_WO_SHOES_FT_IN`, `WEIGHT`, `WINGSPAN`, `WINGSPAN_FT_IN`, `MAX_VERTICAL_LEAP`, `THREE_QUARTER_SPRINT`, `LANE_AGILITY_TIME`. NBA measurement units must be those used by the official combine fields: weight in pounds, vertical in inches, agility and sprint in seconds; heights/reach as decimal feet, inches, or feet/inches text.
- `raw/season_stats.csv`: league-wide totals with `PLAYER_ID`, `SEASON_START` or `SEASON` (starting year, e.g. `2018` or `2018-19`), `GP` and `PTS`. **GP and PTS must be season totals**, not rates. Exactly one row per player-season (aggregate team-split rows correctly, do not sum a player `TOT` aggregate plus individual team rows). The CSV must cover all 2001-02 through 2024-25 regular seasons, since otherwise some PPG labels would be incorrect.

If those two files exist, `00_build_dataset.ipynb` uses them instead of making NBA API requests. Do not substitute a per-game average for total points.

## Files in this ZIP

| File | Purpose |
|---|---|
| `00_build_dataset.ipynb` | Download/build/validate player-level dataset and PPG label |
| `01_member1_size_model.ipynb` | Height, wingspan, weight, position; training-only collinearity fix when needed |
| `02_member2_athletic_model.ipynb` | Maximum vertical, 3/4 sprint, agility, position |
| `03_member3_interaction_model.ipynb` | Wingspan, weight, vertical × position interaction |
| `04_group_comparison.ipynb` | Compare held-out MSE and R² on identical split |
| `nba_cp1_data.py` | NBA official sources, measurements cleaning, joins, outcome aggregation, audit |
| `nba_cp1_models.py` | Consistent split, OLS, imputation, VIF, diagnostics, figures, results |
| `DATA_DICTIONARY.csv` | Variables and units used by the notebooks |
| `requirements.txt` | Python package list |

## Regression assumptions (all members)

1. **Linearity:** training residual versus fitted plot.
2. **Constant error variance:** residual-versus-fitted and scale-location plot.
3. **Approximately normal errors:** residual Q-Q plot. These checks especially matter for inferential claims; prediction metrics are judged on test data.
4. **Independence:** exactly one row per player; discuss possible correlations among players/cohorts. Distinct IDs alone do not prove complete independence.
5. **No problematic multicollinearity:** correlation matrix plus training VIF for all fitted predictor terms; course guideline VIF below 5.

The pipeline imputes missing numeric combine measurements using **training-set medians only**. It reports missing fractions and warns if fewer than 10 measured training values remain in a feature. Missingness is likely systematic, so interpret these coefficients cautiously. If a serious violation is found, try a justified remedy on training data (e.g. reducing collinear predictors or a log transform) and display the before/after plot; test once at the end. The size notebook includes one transparent automatic training VIF remedy. Other models require the student to inspect plots and implement/document a remedy if needed.

## CP1 rubric quick check

- Due **Monday, October 26, 2026, 11:59 PM**. Out-of-list datasets must be **approved by the TA or project mentor**.
- Video **5–7 minutes**; shared slides + each member explains their own model.
- Slide 1: dataset/source, `n`, quantitative and categorical predictor count, continuous PPG outcome, one-sentence modeling question.
- Slide 2: PPG histogram, one motivated second figure, 80/20 split and random seed 42.
- Member slides: original mathematical formula, fitted equation using real estimated coefficients, coefficient table, quantitative and categorical coefficient interpretations with guard baseline, test MSE and R², and diagnostic status.
- Problem/fix slide if needed: one flagged assumption per person at most in deck, with a real attempted fix and before/after evidence.
- Final slide: shared comparison table with each member's predictor set, test MSE and test R², winner and why.
- Submit an **unlisted** YouTube recording and linked Drive folder containing individual model notebooks, following the course form.

## Honest study limitations

Do not say physical size **causes** scoring. A low test R² is scientifically informative. Combine test completion is non-random. These predictions are conditional on reaching NBA games within three years and use PPG as a measure of *scoring volume*, not overall player value.

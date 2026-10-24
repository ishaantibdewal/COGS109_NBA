# COGS 109: NBA + NCAA post-combine PPG

**Research question:** How much do NCAA performance statistics improve prediction of NBA PPG beyond pre-draft NBA combine measurements?

## Run order

1. Install requirements: `pip install -r requirements.txt` (or run first notebook's install cell).
2. Open **`00_build_dataset.ipynb`** **once** in the folder containing these notebooks. If you clone `ishaantibdewal/COGS109_NBA`, move these notebooks to its root or copy its `output/nba_early_career_ppg.csv` into the matching path.
3. The builder reuses existing NBA data, downloads missing ESPN college player-game Parquet files (2005–2022), then rebuilds `data/predraft_features.csv`, `data/nba_outcomes.csv`, and `data/ncaa_match_audit.csv`. Rerun it after changing matching rules. Existing source files are cached, so subsequent runs rebuild locally without downloading them again.
4. Members run the three independent notebooks, in any order, using **only the two locally saved CSVs**. Each writes coefficient estimates, held-out MSE/R², and diagnostics to `reports/`.
5. Run `04_group_comparison.ipynb`. No model notebook calls an API.

### Sources
- NBA Stats combine & league player totals: [nba_api combine](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/draftcombinestats.md), [nba_api league stats](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashplayerstats.md). Already aggregated and cached in your [existing repo](https://github.com/ishaantibdewal/COGS109_NBA/blob/main/output/nba_early_career_ppg.csv).
- NCAA: [SportsDataverse ESPN MBB player box scores](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_mens_college_basketball_player_boxscores). Free public Parquet by NCAA **season ending year**. NCAA season 2018 is 2017–18 and matches the recorded 2018 combine.
- Explicit aliases are listed by NBA player ID in the builder. Official NBA profiles provide identity references, including [Edrice/Bam Adebayo](https://www.nba.com/player/1628389/bam-adebayo/bio), [Mohamed/Mo Bamba](https://www.nba.com/player/1628964/mo-bamba/bio), and [Nah'Shon/Bones Hyland](https://www.nba.com/draft/2021/prospects/nahshon-hyland).

### Matching and season policy

- Normalize accents, punctuation, whitespace and suffixes, so `CJ`/`C.J.` and `O'Shae`/`Oshae` agree. Use a short explicit NBA-ID-specific alias dictionary for nicknames and middle names. No fuzzy matching.
- Require exactly one NCAA athlete ID for the normalized name in the recorded combine year. Drop ambiguous same-name NBA/NCAA players from both core CSVs; keep them in the audit with an explicit exclusion reason.
- Also drop NCAA identities that mix multiple schools in that season. This conservative rule excludes A.J. Lawson's mixed South Carolina/McNeese source record.
- Use the college season **ending in the recorded combine year**, rather than a later final college season. This keeps predictors available at the recorded combine. Some players subsequently return to college.
- Keep the cached outcome definition: NBA PPG over the **three post-combine seasons**, computed from NBA games actually played in that window. This is not necessarily the player's first three NBA-playing seasons.

The current build has **770 matched players**, **51 other unmatched players**, **25 ambiguous NCAA names excluded**, and **1 conflicting multi-school identity excluded**. The core CSVs each contain **821 rows**; all three models use only the same 770 matched players.

### Data dictionary
**`data/predraft_features.csv`**: NBA player ID, name, combine year, guard/forward/center, height/weight/wingspan/reach/vertical/sprint/agility, NCAA ESPN ID and player name (if matched), NCAA player games, NCAA college PPG/APG/RPG/MPG/SPG/BPG/turnovers per game, and college FG/3P/FT percentages. Matching metadata includes `ncaa_match_status`, `ncaa_match_method`, `ncaa_candidate_count`, `ncaa_season`, `ncaa_team`, and `ncaa_team_count`.

**`data/nba_outcomes.csv`**: NBA player ID, combine year, total regular-season points and games during the first three post-combine seasons, and `ppg_3yr=total_points_3yr/total_games_3yr`.

**`data/ncaa_match_audit.csv`**: all 847 source players in 2005–2022, including excluded players, matching method, candidate count, source school information when unique, and `included_in_dataset`/`included_in_model` flags. Multiple NCAA candidates are never assigned an arbitrary identity.

**Study sample**: only matched NCAA players with an NBA appearance over that three-season window, and same sample/split for all three models (616 training, 154 test; 80/20, seed 42, stratified by position). Other unmatched players remain in the core CSVs for reference; ambiguous and conflicting identities exist only in the audit. The model notebooks validate matching counts, source-school counts and season alignment before fitting. NBA outcomes are NOT predictors.

### Limitations and honesty
Matching relies on normalized player name, explicit aliases and combine year rather than a shared ESPN/NBA cross-league ID. Ambiguous matches and multi-school source conflicts are excluded; inspect the audit and verify a sample of accepted identities. ESPN season-game archives can be incomplete or have missing stat fields, and low NCAA game counts produce unstable averages. Missing numerical predictors are imputed from training data only. Some combine attendees never played NCAA, so results apply to a selected subset. There is no guarantee adding NCAA information improves predictions.

### Checkpoint 1
Use the shared 80/20 split and test MSE and R²; include one equation per member, a quantitative and qualitative coefficient interpretation, all five assumption checks, and one attempted fix if an assumption is violated. Each member explains their notebook in the recorded presentation.

**Clarification:** "one API call" here means **one data preparation run**. The first build necessarily requests an NBA CSV if absent and one NCAA file per season. After that, no downloads/API calls are made to run the models.

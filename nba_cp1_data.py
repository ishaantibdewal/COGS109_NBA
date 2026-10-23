"""Reproducible NBA combine -> early-career PPG dataset for COGS 109.

Inputs are official NBA Stats endpoint result CSVs (or downloaded using nba_api).
Outputs are real-source tables only: no example or simulated NBA players.
"""
from __future__ import annotations
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "output"
MIN_COMBINE_YEAR = 2001
MAX_COMBINE_YEAR = 2022  # last eligible follow-up is 2024-25


def _years_from_season(series):
    return pd.to_numeric(series.astype("string").str.extract(r"((?:19|20)\d{2})", expand=False), errors="coerce")


def _height_to_inches(value):
    """Parse NBA combine length formats: decimal feet, inches, or 6' 4.5''."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("’", "'").replace("′", "'")
    m = re.search(r"(?P<feet>\d+)\s*'\s*(?P<inches>\d+(?:\.\d+)?)?", s)
    if m:
        return 12 * float(m.group("feet")) + float(m.group("inches") or 0)
    try:
        v = float(s)
        # Official NBA endpoint may encode stature/reach in decimal feet.
        return v * 12 if 4.0 <= v <= 10.5 else v
    except (ValueError, TypeError):
        return np.nan


def normalize_combine(raw: pd.DataFrame):
    d = raw.copy()
    d.columns = [str(x).strip().upper() for x in d.columns]
    required = {"PLAYER_ID", "POSITION"}
    absent = required - set(d.columns)
    if absent:
        raise ValueError(f"Combine data missing: {absent}; actual columns: {list(d.columns)}")
    season_field = next((c for c in ["SEASON", "SEASON_YEAR", "YEAR", "DRAFT_YEAR"] if c in d), None)
    if season_field is None:
        raise ValueError("Combine data needs SEASON, SEASON_YEAR, YEAR, or DRAFT_YEAR")
    d["combine_year"] = _years_from_season(d[season_field])
    d["player_id"] = pd.to_numeric(d["PLAYER_ID"], errors="coerce")
    if "PLAYER_NAME" in d:
        d["player_name"] = d["PLAYER_NAME"].astype("string")
    else:
        d["player_name"] = (d.get("FIRST_NAME", pd.Series('',index=d.index)).fillna('') + ' ' +
                            d.get("LAST_NAME", pd.Series('',index=d.index)).fillna('')).str.strip()
    d["position_raw"] = d["POSITION"].astype("string").str.upper().str.strip()

    def position_group(s):
        if pd.isna(s): return np.nan
        # Hybrid positions are classified using their first-listed position.
        token = re.split(r"[-/, ]+", str(s))[0]
        if token in {"PG", "SG", "G"}: return "Guard"
        if token in {"SF", "PF", "F"}: return "Forward"
        if token in {"C", "CENTER"}: return "Center"
        return np.nan
    d["position_group"] = d["position_raw"].map(position_group)

    length_sources = {
        "height_in": ["HEIGHT_WO_SHOES_FT_IN", "HEIGHT_WO_SHOES"],
        "wingspan_in": ["WINGSPAN_FT_IN", "WINGSPAN"],
        "standing_reach_in": ["STANDING_REACH_FT_IN", "STANDING_REACH"],
    }
    for dest, cols in length_sources.items():
        d[dest] = pd.Series(np.nan, index=d.index, dtype=float)
        for source in reversed(cols):
            if source in d:
                parsed = d[source].apply(_height_to_inches)
                d[dest] = parsed.combine_first(d[dest])

    column_map = {
        "weight_lb": "WEIGHT", "vertical_max_in": "MAX_VERTICAL_LEAP",
        "vertical_standing_in": "STANDING_VERTICAL_LEAP", "sprint_sec": "THREE_QUARTER_SPRINT",
        "agility_sec": "LANE_AGILITY_TIME", "bench_reps": "BENCH_PRESS",
        "body_fat_pct": "BODY_FAT_PCT",
    }
    for dest, source in column_map.items():
        d[dest] = pd.to_numeric(d[source], errors="coerce") if source in d else np.nan
    # Reject clearly impossible numbers, but never fill physical measures using target data.
    ranges = {
        "height_in": (55, 92), "wingspan_in": (58, 105),
        "standing_reach_in": (70, 125), "weight_lb": (120, 400),
        "vertical_max_in": (15, 52), "vertical_standing_in": (10, 48),
        "sprint_sec": (2.3, 5.0), "agility_sec": (8.0, 17.0),
        "bench_reps": (0, 50), "body_fat_pct": (1, 35),
    }
    for col, (lo, hi) in ranges.items():
        d.loc[~d[col].between(lo, hi), col] = np.nan
    d = d.loc[d["combine_year"].between(MIN_COMBINE_YEAR, MAX_COMBINE_YEAR) & d["player_id"].gt(0)].copy()
    cols = ["player_id", "player_name", "combine_year", "position_raw", "position_group",
            "height_in", "wingspan_in", "standing_reach_in", "weight_lb", "vertical_max_in",
            "vertical_standing_in", "sprint_sec", "agility_sec", "bench_reps", "body_fat_pct"]
    d = d[cols]
    # Multiple rows for a player/year sometimes represent duplicated collection records.
    if d.duplicated(["player_id", "combine_year"]).any():
        d["_nonmissing"] = d.notna().sum(axis=1)
        d = (d.sort_values("_nonmissing", ascending=False).drop_duplicates(["player_id", "combine_year"])
               .drop(columns="_nonmissing"))
    assert not d.duplicated(["player_id", "combine_year"]).any()
    # A player can attend multiple years. Keep the earliest pre-draft observation.
    d = d.sort_values(["combine_year", "player_id"]).drop_duplicates("player_id", keep="first")
    assert not d.duplicated("player_id").any()
    return d.reset_index(drop=True)


def normalize_season_stats(raw: pd.DataFrame):
    d = raw.copy()
    d.columns = [str(c).upper().strip() for c in d.columns]
    yearcol = next((c for c in ["SEASON_START", "SEASON_YEAR", "SEASON", "YEAR"] if c in d), None)
    if yearcol is None:
        raise ValueError("Stats CSV needs season_start or SEASON / SEASON_YEAR")
    required = ["PLAYER_ID", "GP", "PTS"]
    if not set(required).issubset(d.columns):
        raise ValueError(f"Stats missing required columns {set(required) - set(d.columns)}")
    d["season_start"] = _years_from_season(d[yearcol])
    for c in required:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["season_start"].between(MIN_COMBINE_YEAR, MAX_COMBINE_YEAR + 2)].copy()
    # Download is per-season totals from NBA API, NOT per-game numbers.
    d = d[["PLAYER_ID", "season_start", "GP", "PTS"]].rename(
        columns={"PLAYER_ID":"player_id", "GP":"gp", "PTS":"points"})
    d = d.dropna(subset=["player_id", "season_start", "gp", "points"])
    d = d[(d["gp"] >= 0) & (d["points"] >= 0)]
    if d.duplicated(["player_id", "season_start"]).any():
        raise ValueError("Multiple rows for a player-season. Use league-wide per-player totals "+
                         "(not team split rows), or deduplicate before importing.")
    return d.reset_index(drop=True)


def make_player_dataset(combine, seasons):
    combine = normalize_combine(combine)
    seasons = normalize_season_stats(seasons)
    if combine.empty or seasons.empty:
        raise ValueError("No eligible combine or season rows. Check season formats and years.")
    joined = combine[["player_id", "combine_year"]].merge(seasons, on="player_id", how="left")
    joined = joined.loc[(joined.season_start >= joined.combine_year) &
                        (joined.season_start <= joined.combine_year + 2)].copy()
    outcome = (joined.groupby(["player_id", "combine_year"], as_index=False)
               .agg(total_games_3yr=("gp", "sum"), total_points_3yr=("points", "sum"),
                    seasons_played_3yr=("season_start", "nunique")))
    outcome["ppg_3yr"] = outcome["total_points_3yr"] / outcome["total_games_3yr"].replace(0, np.nan)
    all_players = combine.merge(outcome, on=["player_id", "combine_year"], how="left")
    all_players["made_nba_within_3yr"] = all_players["total_games_3yr"].fillna(0).gt(0).astype(int)
    included = all_players.loc[all_players["made_nba_within_3yr"] == 1].copy()
    included["total_games_3yr"] = included["total_games_3yr"].astype(int)
    included["total_points_3yr"] = included["total_points_3yr"].astype(int)
    included = included[included["position_group"].notna()].copy()
    included = included.dropna(subset=["ppg_3yr"])
    included = included.sort_values(["combine_year", "player_name"]).reset_index(drop=True)
    if len(included) < 50:
        raise ValueError(f"Only {len(included)} modeling rows produced; inspect data coverage and joins.")
    if (included["ppg_3yr"] < 0).any():
        raise ValueError("PPG cannot be negative.")
    return included, all_players


def fetch_nba_api_data():
    """Request official NBA Draft Combine + each completed season; cache sources.

    Individual failed requests are never interpreted as a zero-point season.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        from nba_api.stats.endpoints import draftcombinestats, leaguedashplayerstats
    except ImportError as exc:
        raise RuntimeError("Install nba_api first: pip install nba_api") from exc

    combine_path = RAW / "draft_combine_all_time.csv"
    if combine_path.exists():
        combine = pd.read_csv(combine_path)
        print(f"Using cached combine file: {combine_path}")
    else:
        print("Downloading official combine data (All Time) ...")
        endpoint = draftcombinestats.DraftCombineStats(season_all_time="All Time", timeout=60)
        combine = endpoint.get_data_frames()[0]
        if len(combine) < 100:
            raise RuntimeError("Combine response unexpectedly small; nothing cached.")
        combine.to_csv(combine_path, index=False)

    chunks, missing = [], []
    for year in range(MIN_COMBINE_YEAR, MAX_COMBINE_YEAR + 3):
        season = f"{year}-{(year+1)%100:02d}"
        path = RAW / f"nba_player_totals_{season}.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            print(f"Downloading {season} ...", flush=True)
            success = False
            for attempt in range(3):
                try:
                    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
                        season=season, season_type_all_star="Regular Season",
                        per_mode_detailed="Totals", measure_type_detailed_defense="Base", timeout=60)
                    frame = endpoint.get_data_frames()[0]
                    if len(frame) < 100: raise ValueError("Stats response unexpectedly small")
                    frame.to_csv(path, index=False)
                    success = True
                    time.sleep(0.5)
                    break
                except Exception as exc:
                    print(f"  attempt {attempt+1}: {type(exc).__name__}: {str(exc)[:130]}")
                    time.sleep(2 + attempt*2)
            if not success:
                missing.append(season)
                continue
        frame["SEASON_START"] = year
        chunks.append(frame)
    if missing:
        raise RuntimeError("Missing season(s): " + ", ".join(missing) +
                           ". Cache remaining NBA Stats seasons and rerun; incomplete data would bias PPG. "+
                           "Alternatively place complete raw CSVs in raw/ and rerun.")
    season_data = pd.concat(chunks, ignore_index=True)
    return combine, season_data


def build_from_cached_or_api():
    RAW.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    # When API access fails, users can provide these combined source CSVs themselves.
    custom_combine = RAW / "combine.csv"
    custom_stats = RAW / "season_stats.csv"
    if custom_combine.exists() and custom_stats.exists():
        print("Using user-provided source CSVs (raw/combine.csv + raw/season_stats.csv)")
        combine, stats = pd.read_csv(custom_combine), pd.read_csv(custom_stats)
    else:
        combine, stats = fetch_nba_api_data()
    players, audit = make_player_dataset(combine, stats)
    dest = OUT / "nba_early_career_ppg.csv"
    players.to_csv(dest, index=False)
    audit.to_csv(OUT / "combine_join_audit.csv", index=False)
    (players.isna().mean().sort_values(ascending=False)*100).round(1).rename("missing_pct").to_csv(OUT / "missingness.csv")
    print(f"Saved {len(players)} NBA players to {dest}")
    print(f"Draft/combine years: {players.combine_year.min()} - {players.combine_year.max()}")
    print(f"Combine participants evaluated: {len(audit)}; reached NBA in 3 years: {audit.made_nba_within_3yr.sum()}")
    print("Position counts:", players.position_group.value_counts().to_dict())
    print("Outcome distribution:", players.ppg_3yr.describe().round(2).to_dict())
    return players, audit


if __name__ == "__main__":
    build_from_cached_or_api()

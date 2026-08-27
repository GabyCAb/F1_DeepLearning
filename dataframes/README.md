# Formula 1 — Results & Standings (1950 → present)

A clean, **current** Formula 1 dataset covering **every World Championship race since 1950**:
race results, championship standings, and reference tables for drivers, constructors and
circuits. Built from the **jolpica-f1 API** (the community-maintained, Ergast-compatible
successor), so it stays up to date through the latest completed Grand Prix — unlike the many
F1 datasets frozen at 2020/2024.

> Tidy, analysis-ready CSVs that join on simple keys (`season`, `round`, `driver_id`,
> `constructor_id`, `circuit_id`).

---

## Files

| File | Rows (approx) | Description |
|------|------|-------------|
| `results.csv` | ~26k | One row per driver per race: grid, finishing position, points, status, fastest lap. |
| `races.csv` | ~1.1k | Race calendar: date, circuit, location, country, coordinates. |
| `driver_standings.csv` | ~1.5k | End-of-season driver championship standings (points, wins, rank). |
| `constructor_standings.csv` | ~0.8k | End-of-season constructor standings (from 1958). |
| `drivers.csv` | ~860 | Driver reference: name, nationality, date of birth, permanent number. |
| `constructors.csv` | ~210 | Constructor (team) reference. |
| `circuits.csv` | ~77 | Circuit reference with latitude/longitude. |
| `seasons.csv` | ~77 | List of seasons. |

```python
import pandas as pd
results = pd.read_csv("results.csv")
races   = pd.read_csv("races.csv")
# e.g. merge race context onto every result:
df = results.merge(races, on=["season", "round", "race_name", "date"], how="left")
```

## Key columns

**results.csv** — `season, round, race_name, date, driver_id, driver_code, driver_number,
driver_name, driver_nationality, driver_dob, constructor_id, constructor_name,
constructor_nationality, grid, position, position_text, points, laps, status, time_text,
time_millis, fastest_lap_rank, fastest_lap, fastest_lap_speed`
- `position` is numeric finishing place (empty if DNF/DNS); `position_text` keeps codes like `R` (retired), `D` (disqualified), `W` (withdrew).
- `grid` = starting position (0 = pit lane start).

**races.csv** — `season, round, race_name, date, time, circuit_id, circuit_name, locality,
country, lat, long, url`

**driver_standings.csv** — `season, position, points, wins, driver_id, driver_name,
driver_nationality, constructor` (constructor = team they finished the season with).

**constructor_standings.csv** — `season, position, points, wins, constructor_id,
constructor_name, nationality`.

---

## Notes & honest limits
- **Coverage:** 1950 → latest completed round; the in-progress season is included up to the
  most recent race. Re-run `build.py` any time to refresh.
- **Standings** are the *final* standings of each season (the current standings for the
  season in progress).
- **Not in v1:** lap-by-lap times, qualifying sessions, pit stops and sprint results. These
  exist in the API and can be added — they were left out of v1 to keep the dataset compact
  and within the API rate limit. (Open an issue / comment if you want them.)
- Some early-era fields (e.g. fastest-lap data) are sparse because they were not recorded.

## Reproducibility
```bash
pip install pandas requests
python build.py        # every API page is cached to raw/, so re-runs are free
```

## Source & license
- Data: **jolpica-f1** API (`api.jolpi.ca`), the open-source continuation of the Ergast
  Developer API.
- License: **CC BY-NC-SA 4.0** (matching the Ergast/jolpica data terms — non-commercial,
  attribution, share-alike). Not affiliated with Formula 1. For research/education.

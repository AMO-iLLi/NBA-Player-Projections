# Tableau Public setup guide

Tableau Public is free, runs natively on macOS, and publishes to a URL you
can link from a resume. Power BI Desktop is Windows-only, which is why this
project targets Tableau.

## Files

| File | Grain | Role |
|---|---|---|
| `dim_player.csv` | one row per player | dimension |
| `dim_season.csv` | one row per season | dimension |
| `dim_team_season.csv` | team x season | dimension |
| `fact_player_season.csv` | player x season | main fact |
| `fact_predictions.csv` | player x season x model | model eval fact |
| `fact_projections_2027.csv` | one row per 2027 player | forecast fact |
| `fact_residuals.csv` | player x season | error analysis fact |
| `fact_aging.csv` | age x method | analysis fact |
| `fact_stability.csv` | one row per stat | analysis fact |
| `dim_model_results.csv` | one row per model | reference |
| `dim_classification_results.csv` | one row per task | reference |

## Connecting the data

1. Open Tableau Public > Connect > To a File > Text file
2. Select `fact_player_season.csv`
3. Drag `dim_player.csv` onto the canvas. Tableau proposes a relationship;
   set it to `player_id = player_id`.
4. Drag `dim_season.csv`, relate on `season = season`.
5. Drag `dim_team_season.csv`, relate on `team_season_key = team_season_key`.

Use RELATIONSHIPS (the noodle, Tableau 2020.2+), not joins. A join would
duplicate fact rows wherever a dimension has more than one match; a
relationship keeps the grain intact and lets Tableau pick the right level
of detail per sheet.

`fact_predictions`, `fact_projections_2027`, `fact_residuals`, `fact_aging`
and `fact_stability` each have a different grain, so put them in SEPARATE
data sources rather than relating everything into one model. Mixing grains
in a single source is the most common cause of inflated Tableau numbers.

## Calculated fields

```
// headline result
MAE                 AVG([Abs Error])

Baseline MAE        { FIXED : AVG(IIF([Model] = "Persistence", [Abs Error], NULL)) }

Best Model MAE      { FIXED : AVG(IIF([Model] = "Ridge", [Abs Error], NULL)) }

MAE Improvement     [Baseline MAE] - [Best Model MAE]

Improvement Pct     [MAE Improvement] / [Baseline MAE]

// direction accuracy -- null for Persistence by design, since it
// predicts zero change and has no direction to be right about
Direction Accuracy  AVG([Direction Correct])

// volume
Players             COUNTD([Player Id])

Qualified Players   { FIXED : COUNTD(IIF([Qualified] = 1, [Player Id], NULL)) }

// aging
Avg PER             AVG([Per])

Avg PER Qualified   AVG(IIF([Qualified] = 1, [Per], NULL))

// projections
Projected Risers    COUNT(IIF([Direction] = "Rise", [Player], NULL))

Avg Projected Change  AVG([Delta])
```

## Suggested dashboard pages

**Page 1 -- Overview**
Four BANs (big numbers): Baseline MAE 1.874, Best Model MAE 1.683,
Improvement 10.2%, Qualified Players.
Bar of MAE by model from `dim_model_results`, with `ci_lo`/`ci_hi` on the
Detail shelf and a reference band showing the interval.
A text tile stating the headline claim and the significance test.

**Page 2 -- Aging**
Line chart from `fact_aging`: `age` on Columns, `value` on Rows,
`method` on Color. The two lines diverging IS the story; annotate the
peak-age gap (26 cross-sectional vs 24 within-player).

**Page 3 -- Stability**
Horizontal bar of `corr` from `fact_stability`, sorted descending,
`tier` on Color. Reference line at 0.5.
Annotate games played at 0.16 -- that is the injury ceiling on any model.

**Page 4 -- Model performance**
Scatter of `predicted` vs `actual` from `fact_predictions`, with `model`
as a filter and a 45-degree reference line.
Bar of AVG(`resid`) by `age_group` from `fact_residuals`. This shows the
systematic bias: the model is too pessimistic about older players.

**Page 5 -- 2027 projections**
Table of `fact_projections_2027` sorted by `pred_2027` descending.
Filters on `age_group` and `tier`.
Diverging bar of `delta` for the top risers and fallers.

## Publishing

File > Save to Tableau Public As. The workbook and its extracts become
public, so do not put anything private in it. The resulting URL is what
goes on a resume.

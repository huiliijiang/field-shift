# Evidence

The final tables and charts behind each claim. They describe past trials and past summers; none of them is a prediction for a farmer's field.

## Canada side (`canada/`)

| Claim | Files |
|---|---|
| In season, the yield model has skill: error 18.8% of a normal crop vs 22.2% for trend only; about 59% of year-to-year swings explained | `model_skill.csv`, `year_level_skill.csv`, `per_crop_skill.csv`, `chart_years.png` |
| Before planting there is no skill, El Niño included | `model_skill.csv` (pre-season rows) |
| Heat at flowering is the main driver: more than 15 days above 30 °C means about 20% less yield | `feature_importance.csv`, `chart_feature_importance.png`, `chart_heat.png`, `key_years.csv` |
| Of 20 ideas tested under one written rule, only MODIS NDVI and SoilGrids soil water passed the one-time exam on untouched 2018–2025 | `exam_ndvi.csv`, `exam_soil.csv` |
| Replaying 44 past summers gives an honest spread: about 83% of real outcomes fall inside the 10–90% range | `scenario_validation.csv` |
| Past-only backtest: in the 2021 drought the advice kept 61% of a normal crop vs 50% for canola → wheat; its single worst year was lower (39% vs 50%) | `backtest_confirm.csv` (6 districts never used for tuning), `backtest_pilot.csv` |
| The approach transfers to rain-fed Kansas (about 62% of swings for corn, sorghum, soybeans; about 12% for winter wheat) | `kansas_model_skill.csv`, `kansas_year_level_skill.csv` |
| An independent re-run reproduced the headline numbers and found no data leakage | `recheck_reproduced.csv`, `recheck_leakage.csv` |

Units: yields are shares of each crop's own normal (trend) yield, district averages for 10 Saskatchewan rural municipalities, 1982–2025.

## US side (`us/`)

| ID | Claim | Files |
|---|---|---|
| E1 | Marsden three-year vs two-year systems, locked test years 2012–2017: +10.22 bu/ac corn, +6.17 bu/ac soybean | `contrast_summary.csv`, `marsden_annual_contrasts.csv`, `contrast_backtest_metrics.csv` |
| E2 | Rye on five Iowa farms (PFI 2022), negative and non-significant results included | `pfi_2022_farm_summary.csv` |
| E3 | Field replay of the process model: corn bias −15.0 to −9.5 bu/ac, soil too wet; not validated | `field_replay_evaluation.json`, `admission_gates.json` |
| E4 | Six Kansas and Idaho counties: actual rotations and crop use differ | `county_evidence_cards.csv`, `rotation_transition_summary.csv`, `review_metrics.csv`, `Brief_Crop_Rotation_Six_Counties.md` |
| E5 | County transfer is not supported: 7 of 9 series worse than the county mean | `historical_validation_series.csv`, `historical_bias_transfer.csv` |
| E6 | The published economic order flips with the technology package | `historical_financial_point_order.csv`, `historical_stress_order_counts.csv` |
| E7 | Climate resemblance of 3,107 US counties to the study sites (where to look next, not suitability) | `conus_county_screening.csv` |
| | Headline numbers in one file | `summary_facts.json` |

Sources and citations: `REFERENCES.md`. Third-party rights: `RIGHTS.md`.

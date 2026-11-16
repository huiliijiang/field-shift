# Reproduce

## Canada side (`saskatchewan/`)

Python 3.11 or 3.12. Run from `04_reproduce/saskatchewan/`:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/train_models.py      # yield models, each year hidden in turn (~4 min)
.venv/bin/python scripts/analyze_results.py   # charts and feature importance
.venv/bin/python scripts/forecast.py          # 44 past summers replayed as 2027 scenarios (~3 min)
.venv/bin/python scripts/recommend.py         # money view, crop mix, insurance, in-season alerts (~1 min)
.venv/bin/python scripts/recommend_yield.py   # yield view and its past-only backtest
.venv/bin/python scripts/export_demo.py       # writes 02_demo/canada/data.json
.venv/bin/python tests/check_demo_parity.py   # the web page selects exactly what the Python layer selects (needs node)
```

The prepared inputs in `data/` let this run offline. To rebuild them from the sources, run `scripts/download_data.py`, `scripts/tidy_data.py` and `scripts/build_features.py` first (downloads NASA POWER daily weather and the Saskatchewan RM yields).

| Path | What it is |
|---|---|
| `data/processed/` | Season features from NASA POWER, RM yields, El Niño index |
| `data/experiments/` | SoilGrids soil water and MODIS NDVI features per district |
| `data/economics/` | 2026 Saskatchewan Crop Planning Guide costs and prices, soil zones, MASC yield by previous crop |
| `results/forecast/` | The replayed 2027 scenarios the planner scores |
| `app/data.py` | Selects and formats results for the planner |

## US side

Run from `04_reproduce/`:

```sh
python3 -m pip install -r requirements.txt    # numpy, pandas, openpyxl
python3 run.py                                # every step below, in order
python3 demo/build_demo.py                    # rebuilds 02_demo/us/index.html from 03_evidence/us
```

`run.py` re-runs the field diagnostic and the trial re-analysis. The field-replay step compiles the process model, so it needs a Rust toolchain (`cargo`); its last results are included in `replay_results/`, so every other step runs without Rust. Reports are written to `outputs/`.

| Path | What it is |
|---|---|
| `case-study/public-field-validation/` | Marsden long-term trial and the PFI 2022 on-farm trials: source workbooks, audit and re-analysis code, results |
| `case-study/field-admission-v2/` | Marsden field records (soil profiles, management logs, sensors), the field-replay diagnostic and its economics checks |
| `case-study/field-admission-v2/rust/` | The reduced water–carbon–nitrogen model used in the field replay |
| `fieldshift-eo-rs/` | The fuller research version of that model, with optional evapotranspiration, mulch and nitrogen-fixation switches |
| `case-study/boone-*/` | Cleaned Marsden yields and soil water, and NASA POWER daily weather for Boone County |
| `demo/` | Source of the US page |

"""Turn raw downloads into tidy CSVs in data/processed/.

  power_daily.csv  rm, date, weather + soil wetness columns (-999 -> blank)
  oni.csv          year, season (DJF..NDJ), oni
  sk_yields.csv    year, rm, crop, yield   (demo RMs, 1981+; long format)
"""
import json, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
pts = pd.read_csv(ROOT / "data" / "rm_points.csv")

frames = []
for rm in pts.rm:
    js = json.loads((RAW / "power" / f"rm{rm:03d}.json").read_text())
    p = pd.DataFrame(js["properties"]["parameter"])
    p.index = pd.to_datetime(p.index, format="%Y%m%d")
    p = p.where(p != -999)
    p.insert(0, "rm", rm)
    frames.append(p.rename_axis("date").reset_index())
power = pd.concat(frames)
power.to_csv(OUT / "power_daily.csv", index=False)

oni = pd.read_csv(RAW / "enso" / "oni.ascii.txt", sep=r"\s+")
oni = oni.rename(columns={"SEAS": "season", "YR": "year", "ANOM": "oni"})[["year", "season", "oni"]]
oni.to_csv(OUT / "oni.csv", index=False)

y = pd.read_csv(RAW / "sk_yields" / "rm_yields.csv", encoding="utf-8-sig")
y = y[y.RM.isin(pts.rm) & (y.Year >= 1981)]
y = y.melt(id_vars=["Year", "RM"], var_name="crop", value_name="yield").dropna()
y.columns = ["year", "rm", "crop", "yield"]
y.to_csv(OUT / "sk_yields.csv", index=False)
print(f"power {power.shape}, oni {oni.shape}, yields {y.shape}")

"""Download raw data for the Saskatchewan demo.

Sources:
  - Saskatchewan RM yields (Government of Saskatchewan dashboard CSV export)
  - NASA POWER daily point data (AG community), 1981-2025, one point per demo RM
  - NOAA CPC Oceanic Nino Index (ONI)

Usage: python3 scripts/download_data.py   (re-running skips files that exist)
"""
import csv, json, time, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "field-shift-crop-rotation-demo/0.1"}

SK_YIELDS = "https://dashboard.saskatchewan.ca/export/rm-yields-data/4950.csv"
ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M,GWETTOP,GWETROOT"
START, END = "19810101", "20251231"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("skip (exists)", dest.relative_to(ROOT)); return
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())
    print("saved", dest.relative_to(ROOT), dest.stat().st_size, "bytes")


def main():
    fetch(SK_YIELDS, RAW / "sk_yields" / "rm_yields.csv")
    fetch(ONI, RAW / "enso" / "oni.ascii.txt")
    with open(ROOT / "data" / "rm_points.csv") as f:
        for row in csv.DictReader(f):
            url = (f"{POWER}?parameters={POWER_PARAMS}&community=AG"
                   f"&longitude={row['lon']}&latitude={row['lat']}"
                   f"&start={START}&end={END}&format=JSON")
            fetch(url, RAW / "power" / f"rm{int(row['rm']):03d}.json")
            time.sleep(1)


if __name__ == "__main__":
    main()

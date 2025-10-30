import os, json, time, csv, pathlib, requests
import pandas as pd

# ================== CONFIG ==================
API   = "https://appeears.earthdatacloud.nasa.gov/api/"
USER  = "USERNAME"
PWD   = "PASSWORD"

CENTROIDS_CSV = "data/ca_centroids.csv"
OUT_CSV       = "data/ca_static_appeears.csv"

LAYERS = [
    {"product": "MCD12Q1.061", "layer": "LC_Type1"},
    {"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"},
    {"product": "MOD44B.061",  "layer": "Percent_Tree_Cover"},
]

# 👉 AppEEARS requires MM-DD-YYYY
DATE_START = "01-01-2024"
DATE_END   = "12-31-2024"

POLL_EVERY_SEC = 15

# ================== AUTH ==================
def get_token():
    r = requests.post(API + "login", auth=(USER, PWD), timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Login failed {r.status_code}: {r.text[:500]}")
    return r.json()["token"]

def head(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ================== HELPERS ==================
def _assert_mmddyyyy(s: str):
    # very light format check
    m, d, y = s.split("-")
    assert len(m) == 2 and len(d) == 2 and len(y) == 4, "Use MM-DD-YYYY"

def submit_task(head, task_name, coords):
    _assert_mmddyyyy(DATE_START)
    _assert_mmddyyyy(DATE_END)
    payload = {
        "task_type": "point",
        "task_name": task_name,
        "params": {
            "dates": [{"startDate": DATE_START, "endDate": DATE_END}],
            "layers": LAYERS,
            "coordinates": coords,          # [{latitude, longitude, id}]
            "output": {"format": {"type": "csv"}}
        }
    }
    r = requests.post(API + "task", headers=head, json=payload, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"Task submit failed {r.status_code}:\n{r.text}")
    return r.json()["task_id"]

def poll_until_done(head, task_id):
    while True:
        r = requests.get(API + f"task/{task_id}", headers=head, allow_redirects=False, timeout=60)
        if r.status_code == 303:  # some tasks redirect when done
            return
        if r.status_code >= 400:
            raise RuntimeError(f"Poll error {r.status_code}: {r.text[:500]}")
        status = (r.json().get("status") or "").lower()
        if status in ("done", "complete"):
            return
        if status in ("error", "failed"):
            raise RuntimeError(f"Task failed: {r.text[:500]}")
        time.sleep(POLL_EVERY_SEC)

def list_bundle(head, task_id):
    r = requests.get(API + f"bundle/{task_id}", headers=head, timeout=60)
    r.raise_for_status()
    return r.json().get("files", [])

def download_csvs(head, task_id, dest_dir):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    csv_paths = []
    files = list_bundle(head, task_id)
    for fmeta in files:
        fmt  = (fmeta.get("file_format") or "").lower()
        name = fmeta.get("file_name") or ""
        if fmt != "csv" and not name.lower().endswith(".csv"):
            continue
        file_id = fmeta.get("file_id") or fmeta.get("id")
        dest = os.path.join(dest_dir, name or f"{file_id}.csv")
        if not os.path.exists(dest):
            with requests.get(API + f"bundle/{task_id}/{file_id}",
                              headers=head, stream=True, allow_redirects=True, timeout=120) as s:
                s.raise_for_status()
                with open(dest, "wb") as w:
                    for chunk in s.iter_content(1<<20):
                        if chunk: w.write(chunk)
        csv_paths.append(dest)
    return csv_paths

# ================== MAIN ==================
def main():
    token = get_token()
    HEAD = head(token)

    df = pd.read_csv(CENTROIDS_CSV)
    if not {"region_id", "lat", "lon"}.issubset(df.columns):
        raise SystemExit("centroids file must include columns: region_id, lat, lon")

    coords = [
        {"latitude": float(r.lat), "longitude": float(r.lon), "id": str(r.region_id)}
        for r in df.itertuples(index=False)
    ]

    task_id = submit_task(HEAD, "ca_counties_static", coords)
    print("Submitted task:", task_id)
    poll_until_done(HEAD, task_id)
    print("Task complete. Downloading…")

    bundle_dir = f"data/appeears_bundle_{task_id}"
    csv_paths = download_csvs(HEAD, task_id, bundle_dir)
    if not csv_paths:
        raise SystemExit("No CSV returned in bundle.")

    frames = [pd.read_csv(p) for p in csv_paths]
    raw = pd.concat(frames, ignore_index=True)

    idcol  = "ID" if "ID" in raw.columns else "id"
    latcol = "Latitude" if "Latitude" in raw.columns else "latitude"
    loncol = "Longitude" if "Longitude" in raw.columns else "longitude"
    lyrcol = "Layer" if "Layer" in raw.columns else "layer"

    keep = raw[[idcol, latcol, loncol, lyrcol, "Date", "Value"]].copy()
    keep.rename(columns={idcol: "region_id", latcol: "lat", loncol: "lon"}, inplace=True)
    keep.sort_values("Date", inplace=True)

    latest = keep.groupby(["region_id", lyrcol], as_index=False).last()
    pivot = latest.pivot(index="region_id", columns=lyrcol, values="Value").reset_index()

    coords_latest = keep.groupby("region_id", as_index=False).last()[["region_id", "lat", "lon"]]
    out = pd.merge(coords_latest, pivot, on="region_id", how="left")

    for c in ["LC_Type1", "250m_16_days_NDVI", "Percent_Tree_Cover"]:
        if c not in out.columns:
            out[c] = float("nan")

    out = out[["region_id","lat","lon","LC_Type1","250m_16_days_NDVI","Percent_Tree_Cover"]]
    out.rename(columns={"250m_16_days_NDVI": "NDVI"}, inplace=True)
    out.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)

if __name__ == "__main__":
    main()
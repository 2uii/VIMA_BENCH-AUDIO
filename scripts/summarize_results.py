import glob
import os
import re
import pandas as pd

RESULTS_DIR = "results"

# Match filenames like:
# baseline_<task>_100eps_YYYYMMDD_HHMMSS.csv
# audiovima_<task>_100eps_YYYYMMDD_HHMMSS.csv
PAT = re.compile(r"^(baseline|audiovima)_(.+)_(\d+)eps_(\d{8}_\d{6})\.csv$")

def task_from_slug(slug: str) -> str:
    # Your filenames replace "/" with "_"
    # We can reverse only for known task prefix families.
    # Keep as slug for safety; it’s still readable.
    return slug

def load_one(path):
    fn = os.path.basename(path)
    m = PAT.match(fn)
    if not m:
        return None
    kind, task_slug, eps, ts = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    df = pd.read_csv(path)
    # success column exists in both
    success_rate = 100.0 * df["success"].mean() if "success" in df.columns else None

    out = {
        "kind": kind,
        "task": task_from_slug(task_slug),
        "episodes": eps,
        "timestamp": ts,
        "file": path,
        "success_rate": round(success_rate, 2) if success_rate is not None else None,
    }

    if kind == "audiovima":
        if "compliant" in df.columns:
            out["compliance_rate"] = round(100.0 * df["compliant"].mean(), 2)
        if "audio_events" in df.columns:
            out["avg_audio_events"] = round(float(df["audio_events"].mean()), 2)

    return out

def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    rows = [load_one(p) for p in files]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("No matching result CSVs found.")
        return

    all_df = pd.DataFrame(rows)

    # Keep only the latest run per (kind, task, episodes)
    all_df = all_df.sort_values(["kind", "task", "episodes", "timestamp"])
    latest = all_df.groupby(["kind", "task", "episodes"], as_index=False).tail(1)

    base = latest[latest["kind"] == "baseline"].set_index(["task", "episodes"])
    aud  = latest[latest["kind"] == "audiovima"].set_index(["task", "episodes"])

    merged = base[["success_rate","timestamp","file"]].rename(columns={
        "success_rate":"baseline_success",
        "timestamp":"baseline_ts",
        "file":"baseline_file"
    }).join(
        aud[["success_rate","compliance_rate","avg_audio_events","timestamp","file"]].rename(columns={
            "success_rate":"audio_success",
            "timestamp":"audio_ts",
            "file":"audio_file"
        }),
        how="outer"
    ).reset_index()

    # Delta (Audio - Baseline)
    merged["success_delta"] = (merged["audio_success"] - merged["baseline_success"]).round(2)

    out_path = os.path.join(RESULTS_DIR, "SUMMARY_COMPARISON.csv")
    merged.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("\nPreview:")
    print(merged.sort_values(["episodes","task"]).to_string(index=False))

if __name__ == "__main__":
    main()

"""Run many bench missions in parallel (one process each), resumable.
Usage: python run_sweep.py jobs.json results.jsonl [workers]
jobs.json: list of {"label": str, "args": [run_mission.py args...]}."""
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent


def main():
    jobs = json.load(open(sys.argv[1]))
    out = pathlib.Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["label"])
            except Exception:
                pass
    todo = [j for j in jobs if j["label"] not in done]
    print(f"{len(todo)} to run ({len(done)} done)", flush=True)

    def run(job):
        cmd = [sys.executable, "-B", "-W", "ignore", str(HERE / "run_mission.py"), "--label", job["label"]] + job["args"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode != 0 or not lines:
            return json.dumps({"label": job["label"], "error": (p.stderr or "")[-1500:]})
        return lines[-1]

    with ThreadPoolExecutor(workers) as pool, open(out, "a") as f:
        for i, line in enumerate(pool.map(run, todo)):
            f.write(line + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()

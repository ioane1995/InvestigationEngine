"""Probe and measure the live deployment (standard library only).

  python -m scripts.measure_deployment https://your-app.onrender.com --probe   # which layer is broken? (health, stats, bm25, dense, hybrid, ask)
  python -m scripts.measure_deployment https://your-app.onrender.com --cold    # time the first /health call (run after >15 min idle)
  python -m scripts.measure_deployment https://your-app.onrender.com           # probe, then latency / cache table

HTTP errors never crash the script: the status and the server's error body (it contains a request_id you can search for in the
Render logs) are printed. Paste the table into docs/ARCHITECTURE.md; read RAM from the Render metrics tab."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

QUESTIONS = ["Which change caused INC-0012 and which services were affected?", "How long did INC-0012 last?",
             "What was the severity of INC-0012?", "Who authored the change behind INC-0012?"]


def call(base: str, path: str, body: dict | None = None, timeout: float = 180) -> tuple[float, int, dict]:
    """-> (elapsed_ms, http_status, json_body). Network failures give status 0; HTTP errors keep their body."""
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return (time.perf_counter() - t) * 1000, 0, {"error": f"{type(e).__name__}: {e}"}
    ms = (time.perf_counter() - t) * 1000
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {"raw": raw[:300]}
    return ms, status, data if isinstance(data, dict) else {"value": data}


def probe(base: str) -> bool:
    steps = [("health", "/health", None), ("stats", "/api/stats", None),
             ("search keyword-only", "/api/search", {"query": "INC-0012", "mode": "bm25", "top_k": 3}),
             ("search dense-only", "/api/search", {"query": "payment delay", "mode": "dense", "top_k": 3}),
             ("search hybrid", "/api/search", {"query": "INC-0012", "mode": "hybrid", "top_k": 3}),
             ("ask, planner off", "/api/ask", {"question": QUESTIONS[2], "options": {"use_planner": False}}),
             ("ask, planner on", "/api/ask", {"question": QUESTIONS[2]})]
    results: dict[str, bool] = {}
    print("layer probe:")
    for name, path, body in steps:
        ms, status, data = call(base, path, body)
        ok = status == 200
        results[name] = ok
        extra = "" if ok else f"  -> {json.dumps(data)[:300]}"
        print(f"  {'OK  ' if ok else 'FAIL'} {name:<22} HTTP {status:<3} {ms:>7.0f} ms{extra}")
    if not results["health"]:
        print("\nDiagnosis: service unreachable or still waking up (free plan sleeps). Retry in a minute; check the Render dashboard.")
    elif not results["stats"] or not results["search keyword-only"]:
        print("\nDiagnosis: retrieval layer / Qdrant connection is failing -> Render logs; check QDRANT_URL / QDRANT_API_KEY and that the cluster is awake.")
    elif results["search keyword-only"] and not results["search dense-only"] and not results["search hybrid"]:
        print("\nDiagnosis: keyword search works, dense search does not -> embedding API (EMBEDDING_BASE_URL / EMBEDDING_API_KEY / model) or EMBEDDING_STORE_ID mismatch.")
    elif results["search hybrid"] and not (results["ask, planner off"] or results["ask, planner on"]):
        print("\nDiagnosis: search works, /api/ask fails with an unhandled error -> look at the Render log line 'unhandled error rid=<request_id>' (full traceback). Paste it.")
    elif results["ask, planner off"] and not results["ask, planner on"]:
        print("\nDiagnosis: only the planner path fails -> LLM planner call / model name.")
    else:
        print("\nAll layers answered.")
    return all(results.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--cold", action="store_true", help="only time the first /health call (service was asleep)")
    ap.add_argument("--probe", action="store_true", help="only probe each layer and stop")
    ap.add_argument("--pause", type=float, default=6.0, help="seconds between LLM questions (free-tier rate limits)")
    a = ap.parse_args()

    ms, status, _ = call(a.url, "/health")
    print(f"first /health: HTTP {status}, {ms / 1000:.1f} s  ({'cold start' if a.cold else 'warm or cold: use --cold after idle'})")
    if a.cold:
        return
    healthy = probe(a.url)
    if a.probe or not healthy:
        return
    lat, rows = [], []
    for q in QUESTIONS:
        ms, st, ans = call(a.url, "/api/ask", {"question": q})
        ms2, st2, ans2 = call(a.url, "/api/ask", {"question": q})  # identical question -> answer cache
        lat.append(ms)
        rows.append((q, ms, st, ans.get("status"), ans.get("degraded"), ms2, ans2.get("cached")))
        time.sleep(a.pause)
    print("\n| question | first ask (ms) | HTTP | status | degraded | repeat (ms) | cached |\n|---|---:|---:|---|---|---:|---|")
    for q, ms, st, s, dg, ms2, ch in rows:
        print(f"| {q[:60]} | {ms:.0f} | {st} | {s} | {dg} | {ms2:.0f} | {ch} |")
    print(f"\np50 first-ask latency: {statistics.median(lat):.0f} ms; max {max(lat):.0f} ms")
    _, _, m = call(a.url, "/api/metrics")
    print("metrics:", json.dumps(m)[:600])
    if any(r[4] for r in rows):
        print("WARNING: some answers were degraded (LLM rate limit / outage): rerun later before quoting these numbers.")


if __name__ == "__main__":
    main()

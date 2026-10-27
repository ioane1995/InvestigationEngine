"""Deterministic synthetic incident-investigation corpus + ground-truth questions.

Why synthetic?  The task's example is an incident-investigation scenario (changes -> incidents -> services,
reports vs. post-mortems). A seeded generator gives (a) arbitrary scale (50k+ documents), (b) exact gold labels
for retrieval AND answers, (c) *deliberately injected* contradictions, missing documents, unanswerable questions,
exact and near-duplicate documents - all things public QA sets do not label. It is fully reproducible:
same seed => byte-identical corpus and question file.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

SERVICES = {  # service -> (owner team, domain)
    "payments-api": ("Payments Core", "payments"), "ledger": ("Payments Core", "payments"),
    "card-gateway": ("Payments Integrations", "payments"), "settlement-worker": ("Payments Integrations", "payments"),
    "refund-service": ("Payments Core", "payments"), "wallet": ("Payments Core", "payments"),
    "auth-gateway": ("Identity", "auth"), "session-cache": ("Identity", "auth"), "kyc-service": ("Identity", "auth"),
    "user-profile": ("Identity", "auth"), "checkout-web": ("Storefront", "checkout"), "order-service": ("Storefront", "checkout"),
    "inventory": ("Storefront", "checkout"), "pricing": ("Storefront", "checkout"), "notifications": ("Messaging", "notify"),
    "email-relay": ("Messaging", "notify"), "sms-relay": ("Messaging", "notify"), "webhook-dispatcher": ("Messaging", "notify"),
    "search-api": ("Discovery", "search"), "search-indexer": ("Discovery", "search"), "recommendation": ("Discovery", "search"),
    "reporting": ("Data Platform", "reporting"), "data-warehouse-sync": ("Data Platform", "reporting"), "audit-log": ("Data Platform", "reporting"),
    "scheduler": ("Platform", "platform"), "config-service": ("Platform", "platform"), "feature-flags": ("Platform", "platform"),
    "rate-limiter": ("Platform", "platform"), "cdn-edge": ("Platform", "platform"), "risk-scoring": ("Risk", "payments"),
}
IMPACT = {"payments": "payment processing delays", "auth": "login failures", "checkout": "checkout errors",
          "notify": "delayed notifications", "search": "stale search results", "reporting": "report generation failures",
          "platform": "elevated API latency"}
PARAPHRASE = {"payment processing delays": "delays in processing payments", "login failures": "users failing to log in",
              "checkout errors": "errors during checkout", "delayed notifications": "notifications arriving late",
              "stale search results": "outdated search results", "report generation failures": "reports failing to generate",
              "elevated API latency": "slow API responses"}
AUTHORS = ["Nika Beridze", "Ana Kapanadze", "Giorgi Lomidze", "Mariam Gelashvili", "Luka Tsiklauri", "Salome Japaridze",
           "Davit Kvaratskhelia", "Tamar Abashidze", "Irakli Chkheidze", "Nino Mchedlishvili", "Sandro Bregvadze", "Eka Natsvlishvili"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
CHANGE_TYPES = ["configuration change", "database migration", "deployment", "feature-flag rollout", "certificate rotation", "dependency upgrade"]
ALERTS = ["HighLatencyP99", "ErrorRateAbove5Percent", "QueueBacklogGrowing", "ConnectionPoolExhausted", "HealthCheckFailing"]
ACTIONS = ["add a canary stage to the deployment pipeline", "require peer review for pool-size changes", "add an alert on queue depth",
           "document rollback steps in the runbook", "introduce a staged rollout for configuration changes", "add load-test coverage for the migration"]


def _change_summary(rng: random.Random, ctype: str, svc: str) -> str:
    a, b = rng.choice([50, 100, 200]), rng.choice([5, 10, 20])
    return {
        "configuration change": f"reduced the connection pool size of {svc} from {a} to {b}",
        "database migration": f"ran a schema migration adding an index on the {rng.choice(['orders', 'sessions', 'transactions', 'events'])} table used by {svc}",
        "deployment": f"deployed version {rng.randint(2, 9)}.{rng.randint(0, 19)}.{rng.randint(0, 9)} of {svc} with new retry logic",
        "feature-flag rollout": f"enabled feature flag {rng.choice(['fast-path', 'new-routing', 'bulk-mode', 'strict-validation'])} for all traffic on {svc}",
        "certificate rotation": f"rotated the TLS certificate used by {svc}",
        "dependency upgrade": f"upgraded the {rng.choice(['http client', 'ORM', 'serialization', 'queue'])} library of {svc} from {rng.randint(1, 4)}.x to {rng.randint(5, 8)}.x",
    }[ctype]


@dataclass
class Corpus:
    docs: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    stats: dict[str, Any]


def _doc(doc_id: str, title: str, text: str, fmt: str, **meta: Any) -> dict[str, Any]:
    return {"doc_id": doc_id, "title": title, "text": text, "format": fmt, "source": f"{doc_id}.{fmt}", "metadata": meta}


def generate(n_incidents: int = 120, n_filler: int = 600, seed: int = 7, q_scale: float = 1.0) -> Corpus:
    rng = random.Random(seed)
    svc_names = list(SERVICES)
    deps = {s: rng.sample([x for x in svc_names if x != s], rng.randint(1, 3)) for s in svc_names}
    dependents = {s: [t for t in svc_names if s in deps[t]] for s in svc_names}
    docs: list[dict[str, Any]] = []
    changes: dict[str, dict] = {}
    chg_counter = [0]

    def new_change(svc: str | None = None, incident: str | None = None) -> dict:
        chg_counter[0] += 1
        cid = f"CHG-{chg_counter[0]:04d}"
        svc = svc or rng.choice(svc_names)
        ctype = rng.choice(CHANGE_TYPES)
        y, m, d = rng.choice([2024, 2025]), rng.randint(1, 12), rng.randint(1, 28)
        ch = {"id": cid, "service": svc, "type": ctype, "author": rng.choice(AUTHORS), "date": f"{y}-{m:02d}-{d:02d}",
              "summary": _change_summary(rng, ctype, svc), "incident": incident}
        changes[cid] = ch
        return ch

    def change_doc(ch: dict) -> dict[str, Any]:
        text = (f"id: {ch['id']}\ntitle: Change {ch['id']}\ndate: {ch['date']}\nservice: {ch['service']}\ntype: {ch['type']}\n"
                f"author: {ch['author']}\nsummary: {ch['summary']}\nstatus: deployed")
        return _doc(f"change-{ch['id'].lower()}", f"Change {ch['id']}", text, "json", kind="change")

    # ---- service catalogue (html pages)
    for s in svc_names:
        team, dom = SERVICES[s]
        text = (f"# Service {s}\n\n- Owner team: {team}\n- Domain: {dom}\n- Depends on: {', '.join(deps[s])}\n- Tier: {rng.randint(1, 3)}\n\n"
                f"## Overview\n{s} is owned by the {team} team. It depends on {', '.join(deps[s])} and its failures typically cause {IMPACT[dom]}.")
        docs.append(_doc(f"service-{s}", f"Service {s}", text, "html", kind="service"))

    # ---- incidents
    incidents: list[dict] = []
    for i in range(1, n_incidents + 1):
        iid = f"INC-{i:04d}"
        root = rng.choice(svc_names)
        team, dom = SERVICES[root]
        affected = [root] + rng.sample(dependents[root], min(len(dependents[root]), rng.randint(0, 2)))
        ch = new_change(root, iid)
        year, month, day = rng.choice([2024, 2025]), rng.randint(1, 12), rng.randint(1, 28)
        impact = IMPACT[dom]
        inc = {"id": iid, "service": root, "affected": affected, "change": ch["id"], "year": year, "month": month, "day": day,
               "date": f"{year}-{month:02d}-{day:02d}", "month_name": f"{MONTHS[month - 1]} {year}", "sev": f"SEV{rng.randint(1, 3)}",
               "duration": rng.choice([12, 23, 31, 47, 58, 64, 90, 120, 155, 185]), "impact": impact, "alert": rng.choice(ALERTS),
               "has_pm": rng.random() < 0.8, "conflict": None}
        if inc["has_pm"] and rng.random() < 0.25:
            inc["conflict"] = rng.choice(["duration", "root_cause"])
        incidents.append(inc)
        inc_text = (f"# Incident Report {iid}\n\n- Date: {inc['date']}\n- Month: {inc['month_name']}\n- Severity: {inc['sev']}\n"
                    f"- Duration: {inc['duration']} minutes\n- Services affected: {', '.join(affected)}\n- Root cause change: {ch['id']}\n"
                    f"- Customer impact: {impact}\n\n## Summary\nOn {day} {inc['month_name'].split()[0]} {year} customers experienced {impact}. "
                    f"The issue originated in {root} after change {ch['id']} was deployed. Engineers rolled back the change.\n\n"
                    f"## Detection\nAlert {inc['alert']} fired for {root}. The on-call engineer opened {iid} and paged the {team} team.")
        docs.append(_doc(f"incident-report-{i:04d}", f"Incident Report {iid}", inc_text, "md", kind="incident_report", incident=iid))
        docs.append(change_doc(ch))
        docs.append(_doc(f"alerts-{i:04d}", f"Alert log {iid}",
                         "\n".join(f"time: {inc['date']}T{9 + k}:{rng.randint(10, 59)}Z; service: {s}; alert: {inc['alert']}; severity: critical"
                                   for k, s in enumerate(affected)), "csv", kind="alerts", incident=iid))
        if inc["has_pm"]:
            dur, rc = inc["duration"], ch["id"]
            if inc["conflict"] == "duration":
                dur = inc["duration"] + rng.choice([15, 28, 40])
            elif inc["conflict"] == "root_cause":
                decoy = new_change(root)
                rc = decoy["id"]
                docs.append(change_doc(decoy))
            inc["pm_duration"], inc["pm_root"] = dur, rc
            pm = (f"Postmortem: {iid}\n\nThis postmortem reviews {iid}, which caused {impact}.\n\n- Duration: {dur} minutes\n"
                  f"- Root cause change: {rc}\n- Corrective action: {rng.choice(ACTIONS)}\n- Action owner: {rng.choice(AUTHORS)}\n\n"
                  f"The {team} team reviewed the timeline. Services {', '.join(affected)} were involved.")
            docs.append(_doc(f"postmortem-{i:04d}", f"Postmortem {iid}", pm, "txt", kind="postmortem", incident=iid))

    # ---- filler: routine changes, weekly digests, alert noise, runbooks; plus exact / near duplicates
    n_changes, n_dig, n_alert = int(n_filler * 0.55), int(n_filler * 0.2), int(n_filler * 0.15)
    for _ in range(n_changes):
        docs.append(change_doc(new_change()))
    for k in range(n_dig):
        y, w = rng.choice([2024, 2025]), rng.randint(1, 52)
        sv = rng.sample(svc_names, 3)
        docs.append(_doc(f"digest-{k:05d}", f"Weekly operations digest {y} week {w}",
                         f"Weekly operations digest for week {w} of {y}.\n\nRoutine maintenance covered {sv[0]}, {sv[1]} and {sv[2]}. "
                         f"No customer-facing incidents were reported. Capacity reviews were completed by {rng.choice(AUTHORS)}.", "txt", kind="digest"))
    for k in range(n_alert):
        s = rng.choice(svc_names)
        docs.append(_doc(f"alertnoise-{k:05d}", f"Alert noise {k}", f"time: 202{rng.choice('45')}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T03:00Z; "
                         f"service: {s}; alert: {rng.choice(ALERTS)}; severity: warning; auto-resolved: true", "csv", kind="noise"))
    n_rb = max(0, n_filler - n_changes - n_dig - n_alert)
    runbooks = []
    for k in range(n_rb):
        s = svc_names[k % len(svc_names)]
        d = _doc(f"runbook-{s}-{k}", f"Runbook {s} #{k}",
                 f"# Runbook for {s}\n\n## Restart procedure\nDrain traffic from {s}, restart the instances one by one and verify health checks.\n\n"
                 f"## Escalation\nEscalate to the {SERVICES[s][0]} on-call if the restart does not recover {s} within 15 minutes. Revision {k}.", "md", kind="runbook")
        runbooks.append(d)
        docs.append(d)
    n_dupes = 0
    for d in rng.sample(docs, min(len(docs), max(3, len(docs) // 100))):  # exact duplicates under a new id
        docs.append({**d, "doc_id": d["doc_id"] + "-copy", "source": d["source"].replace(".", "-copy.", 1)})
        n_dupes += 1
    n_near = 0
    for d in rng.sample(runbooks, min(len(runbooks), max(2, len(runbooks) // 20))) if runbooks else []:  # near duplicates
        docs.append({**d, "doc_id": d["doc_id"] + "-v2", "title": d["title"] + " (mirror)",
                     "text": d["text"].replace("Revision", "Rev.") , "source": d["source"].replace(".", "-v2.", 1)})
        n_near += 1

    questions = _make_questions(rng, incidents, changes, q_scale)
    return Corpus(docs, questions, {"documents": len(docs), "incidents": n_incidents, "exact_duplicates": n_dupes,
                                    "near_duplicates": n_near, "questions": len(questions), "seed": seed})


def _make_questions(rng: random.Random, incs: list[dict], changes: dict[str, dict], scale: float) -> list[dict[str, Any]]:
    qs: list[dict[str, Any]] = []

    def add(qtype: str, question: str, gold: list[str], answers: list[str], status: str, related: list[str] | None = None, note: str = "") -> None:
        qs.append({"id": f"q{len(qs) + 1:03d}", "type": qtype, "question": question, "gold_doc_ids": gold,
                   "relevant_doc_ids": sorted(set(gold + (related or []))), "gold_answers": answers, "expected_status": status, "notes": note})

    def n(x: int) -> int:
        return max(2, round(x * scale))

    def rep(i: dict) -> str:
        return f"incident-report-{int(i['id'][4:]):04d}"

    def pm(i: dict) -> str:
        return f"postmortem-{int(i['id'][4:]):04d}"

    def chg(c: str) -> str:
        return f"change-{c.lower()}"

    plain = [i for i in incs if not i["conflict"]]
    # A. simple, single document
    for i in rng.sample(plain, min(len(plain), n(4))):
        add("simple", f"What was the severity of {i['id']}?", [rep(i)], [i["sev"]], "answered", [pm(i)] if i["has_pm"] else [])
    for i in rng.sample(plain, min(len(plain), n(3))):
        add("simple", f"On what date did {i['id']} start?", [rep(i)], [i["date"]], "answered")
    for i in rng.sample(plain, min(len(plain), n(3))):
        add("simple", f"Which services were affected by {i['id']}?", [rep(i)], i["affected"], "answered", [pm(i)] if i["has_pm"] else [])
    for c in rng.sample(list(changes.values()), n(6)):
        add("simple", f"Who authored change {c['id']}?", [chg(c["id"])], [c["author"]], "answered")
    # C. multi-hop (2 and 3 documents)
    for i in rng.sample(plain, min(len(plain), n(8))):
        c = changes[i["change"]]
        add("multi_hop", f"Which change caused {i['id']} and who authored it?", [rep(i), chg(c["id"])], [c["id"], c["author"]], "answered",
            [pm(i)] if i["has_pm"] else [])
    for i in rng.sample(plain, min(len(plain), n(5))):
        c = changes[i["change"]]
        add("multi_hop", f"Which team owns the service that was modified by the root cause change of {i['id']}?",
            [rep(i), chg(c["id"]), f"service-{c['service']}"], [SERVICES[c["service"]][0]], "answered", [pm(i)] if i["has_pm"] else [],
            "3-hop: incident -> change -> service catalogue")
    # D. natural-language month questions (no identifier in the question)
    by_key: dict[tuple, list[dict]] = {}
    for i in incs:
        by_key.setdefault((i["impact"], i["month_name"]), []).append(i)
    uniq = [v[0] for v in by_key.values() if len(v) == 1 and not v[0]["conflict"]]
    for i in rng.sample(uniq, min(len(uniq), n(6))):
        c = changes[i["change"]]
        phr = rng.choice([i["impact"], PARAPHRASE[i["impact"]]])
        add("multi_hop_nl", f"Which system change caused {phr} in {i['month_name']}, which services were affected and which documents confirm this?",
            [rep(i), chg(c["id"])], [c["id"], *i["affected"]], "answered", [pm(i)] if i["has_pm"] else [], "no identifier in question; semantic + date lookup")
    # E. unanswerable
    for i in rng.sample(incs, n(3)):
        add("unanswerable", f"What was the total financial loss caused by {i['id']}?", [], [], "insufficient_evidence", note="attribute never recorded")
    for k in range(n(3)):
        add("unanswerable", f"What was the root cause of INC-{9000 + k * 37}?", [], [], "insufficient_evidence", note="entity does not exist")
    for c in rng.sample(list(changes.values()), n(2)):
        add("unanswerable", f"Who approved change {c['id']}?", [], [], "insufficient_evidence", note="approver never recorded")
    nopm = [i for i in incs if not i["has_pm"]]
    for i in rng.sample(nopm, min(len(nopm), n(3))):
        add("unanswerable", f"Which corrective action does the postmortem of {i['id']} prescribe?", [], [], "insufficient_evidence",
            [rep(i)], "incident exists but its postmortem is missing")
    # F. conflicting
    for i in [x for x in incs if x["conflict"] == "duration"][: n(4)]:
        add("conflict", f"How long did {i['id']} last?", [rep(i), pm(i)], [str(i["duration"]), str(i["pm_duration"])], "conflicting_evidence",
            note="incident report and postmortem disagree on duration")
    for i in [x for x in incs if x["conflict"] == "root_cause"][: n(4)]:
        add("conflict", f"Which change was the root cause of {i['id']}?", [rep(i), pm(i)], [i["change"], i["pm_root"]], "conflicting_evidence",
            note="incident report and postmortem name different changes")
    return qs

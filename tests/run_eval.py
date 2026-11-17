"""Run the evaluation set against the Python bot and print every miss.   python tests/run_eval.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import Chatbot, Database  # noqa: E402
from eval_set import FOLLOWUP_TEXT, FRESH, FRESH_SCENARIOS, HELDOUT, HELDOUT_SCENARIOS, OFF_TOPIC, SCENARIOS, SINGLE, UNSEEN, UNSEEN_SCENARIOS  # noqa: E402


def fresh():
    return Chatbot(db=Database(os.path.join(tempfile.mkdtemp(), "e.db")), backend="tfidf")


def run_single(queries, label, verbose):
    bot = fresh()
    miss = []
    for i, (q, exp) in enumerate(queries):
        r = bot.respond(f"s{i}", q)
        if r.intent != exp:
            miss.append((label, q, exp, r.intent, round(r.confidence, 2)))
    if verbose:
        for m in miss: print(label.upper(), m)
    print(f"{label}: single-turn {len(queries) - len(miss)}/{len(queries)}")
    return len(miss)


def run_scenarios(scenarios, label, verbose):
    s_total = s_ok = 0
    sc_miss = []
    for name, turns in scenarios.items():
        bot = fresh()
        for turn in turns:
            q, exp, sub = (turn + (None,))[:3]
            r = bot.respond(name, q)
            s_total += 1
            if r.intent == exp and (sub is None or sub in r.text):
                s_ok += 1
            else:
                sc_miss.append((name, q, exp, r.intent, round(r.confidence, 2)))
    if verbose:
        for m in sc_miss: print(label.upper(), m)
    print(f"{label}: multi-turn {s_ok}/{s_total}")
    return len(sc_miss)


def main(verbose=True):
    bad = run_single(SINGLE, "tuning-set", verbose) + run_scenarios(SCENARIOS, "tuning-set", verbose)
    bad += run_single(HELDOUT, "held-out", verbose) + run_scenarios(HELDOUT_SCENARIOS, "held-out", verbose)
    bad += run_single(FRESH, "fresh", verbose) + run_scenarios(FRESH_SCENARIOS, "fresh", verbose)
    bad += run_single(UNSEEN, "set-D", verbose) + run_scenarios(UNSEEN_SCENARIOS, "set-D", verbose)
    bad += off_topic(verbose)
    bad += followups(verbose)
    return bad


def off_topic(verbose):
    bot = fresh()
    wrong = [(q, bot.respond(f"o{i}", q).intent) for i, q in enumerate(OFF_TOPIC)]
    wrong = [(q, t) for q, t in wrong if t is not None]
    if verbose:
        for w in wrong: print("OFF-TOPIC ANSWERED", w)
    print(f"off-topic correctly declined: {len(OFF_TOPIC) - len(wrong)}/{len(OFF_TOPIC)}")
    return len(wrong)


def followups(verbose):
    f_total = f_ok = 0
    f_miss = []
    for prev, q, sub in FOLLOWUP_TEXT:
        bot = fresh()
        bot.respond("f", prev)
        r = bot.respond("f", q) if q else bot.respond("g", prev)
        f_total += 1
        if sub in r.text:
            f_ok += 1
        else:
            f_miss.append((prev, q, sub, r.text[:90]))
    if verbose:
        for m in f_miss: print("FOLLOWUP", m)
    print(f"follow-up answers {f_ok}/{f_total}")
    return len(f_miss)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

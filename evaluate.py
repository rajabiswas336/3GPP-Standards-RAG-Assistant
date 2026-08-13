"""
Evaluation harness: run a hand-written QA test set through the
pipeline and report retrieval accuracy + faithfulness rate.

This is your evidence for "quality and effectiveness of the submitted
project" -- a system that merely "seems to work" is much weaker than
one with a measured hallucination rate on a held-out test set.

Questions are sourced from the actual content of our indexed specs:
  - TS 23.501 (5G System Architecture)
  - TS 24.501 (NAS protocol for 5G)
  - TS 38.331 (RRC protocol)
  - TS 23.503 (Policy and Charging Control)
"""
import json
import time
from typing import List, Dict

from rag_pipeline import GPPRagPipeline

# Real test questions based on known spec content.
TEST_SET: List[Dict] = [
    # ── TS 23.501 — 5G System Architecture ─────────────────────
    {
        "question": "What are the main network functions in the 5G System architecture?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },
    {
        "question": "What is the role of the AMF (Access and Mobility Management Function)?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },
    {
        "question": "How does the 5G system support network slicing?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },
    {
        "question": "What is the Service-Based Architecture (SBA) in 5G?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },
    {
        "question": "What is the role of the SMF (Session Management Function) in 5G?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },
    {
        "question": "What is the PDU Session and how is it established?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },
    {
        "question": "What is the UPF (User Plane Function) responsible for?",
        "expected_spec": "23.501",
        "expect_answerable": True,
    },

    # ── TS 24.501 — NAS protocol ───────────────────────────────
    {
        "question": "What is the 5GS registration procedure?",
        "expected_spec": "24.501",
        "expect_answerable": True,
    },
    {
        "question": "What NAS security mechanisms are defined for 5G?",
        "expected_spec": "24.501",
        "expect_answerable": True,
    },
    {
        "question": "What triggers a service request procedure in 5G NAS?",
        "expected_spec": "24.501",
        "expect_answerable": True,
    },
    {
        "question": "What are the 5GMM states defined in the NAS specification?",
        "expected_spec": "24.501",
        "expect_answerable": True,
    },

    # ── TS 38.331 — RRC protocol ──────────────────────────────
    {
        "question": "What is the RRC connection reconfiguration procedure?",
        "expected_spec": "38.331",
        "expect_answerable": True,
    },
    {
        "question": "What is the RRC idle state and what can the UE do in it?",
        "expected_spec": "38.331",
        "expect_answerable": True,
    },
    {
        "question": "How does handover work in the RRC protocol?",
        "expected_spec": "38.331",
        "expect_answerable": True,
    },
    {
        "question": "What measurements does the UE perform in RRC connected state?",
        "expected_spec": "38.331",
        "expect_answerable": True,
    },

    # ── TS 23.503 — Policy and Charging ───────────────────────
    {
        "question": "What is the role of the PCF (Policy Control Function)?",
        "expected_spec": "23.503",
        "expect_answerable": True,
    },
    {
        "question": "How does QoS flow management work in the policy framework?",
        "expected_spec": "23.503",
        "expect_answerable": True,
    },

    # ── Out-of-scope questions (should be REFUSED) ────────────
    {
        "question": "What is the capital of France?",
        "expected_spec": None,
        "expect_answerable": False,
    },
    {
        "question": "How do I configure a Cisco router for BGP?",
        "expected_spec": None,
        "expect_answerable": False,
    },
    {
        "question": "What is the latest iPhone model?",
        "expected_spec": None,
        "expect_answerable": False,
    },
    {
        "question": "Explain quantum computing in simple terms.",
        "expected_spec": None,
        "expect_answerable": False,
    },
]


def run_eval():
    pipeline = GPPRagPipeline()

    n = len(TEST_SET)
    retrieval_hits = 0
    correct_refusals = 0
    correct_answers_attempted = 0
    faithful_count = 0
    results = []

    print(f"{'=' * 60}")
    print(f"  3GPP RAG Chatbot — Evaluation Harness")
    print(f"  Running {n} test cases...")
    print(f"{'=' * 60}\n")

    start = time.time()

    for i, case in enumerate(TEST_SET, 1):
        q_start = time.time()
        resp = pipeline.answer(case["question"])
        q_time = time.time() - q_start

        status = ""
        if case["expect_answerable"]:
            answered = not resp.refused
            correct_answers_attempted += int(answered)

            if answered:
                if resp.is_faithful:
                    faithful_count += 1

                spec_hit = False
                if case.get("expected_spec") and any(
                    case["expected_spec"] in c.citation for c in resp.retrieved
                ):
                    retrieval_hits += 1
                    spec_hit = True

                status = f"{'✅' if spec_hit else '⚠️ '} answered | " \
                         f"spec={'HIT' if spec_hit else 'MISS'} | " \
                         f"faithful={'PASS' if resp.is_faithful else 'FAIL'}"
            else:
                status = "❌ REFUSED (should have answered)"
        else:
            refused = resp.refused
            correct_refusals += int(refused)
            status = f"{'✅' if refused else '❌'} {'REFUSED' if refused else 'ANSWERED'} " \
                     f"(expected: REFUSED)"

        print(f"[{i:2d}/{n}] {status}")
        print(f"       Q: {case['question'][:70]}")
        print(f"       ⏱ {q_time:.1f}s\n")

        results.append({
            "question": case["question"],
            "expected_spec": case.get("expected_spec"),
            "expect_answerable": case["expect_answerable"],
            "refused": resp.refused,
            "is_faithful": resp.is_faithful if not resp.refused else None,
            "time_s": round(q_time, 2),
            "status": status,
        })

    elapsed = time.time() - start
    answerable_cases = sum(1 for c in TEST_SET if c["expect_answerable"])
    unanswerable_cases = n - answerable_cases

    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")

    if answerable_cases:
        attempt_rate = correct_answers_attempted / answerable_cases * 100
        retrieval_rate = retrieval_hits / answerable_cases * 100
        faith_rate = faithful_count / max(correct_answers_attempted, 1) * 100

        print(f"  Answerable questions:     {answerable_cases}")
        print(f"  Attempted when should:    {correct_answers_attempted}/{answerable_cases} "
              f"({attempt_rate:.0f}%)")
        print(f"  Retrieved correct spec:   {retrieval_hits}/{answerable_cases} "
              f"({retrieval_rate:.0f}%)")
        print(f"  Passed faithfulness:      {faithful_count}/{correct_answers_attempted} "
              f"({faith_rate:.0f}%)")

    if unanswerable_cases:
        refusal_rate = correct_refusals / unanswerable_cases * 100
        print(f"  Out-of-scope questions:   {unanswerable_cases}")
        print(f"  Correctly refused:        {correct_refusals}/{unanswerable_cases} "
              f"({refusal_rate:.0f}%)")

    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/n:.1f}s per question)")
    print(f"{'=' * 60}")

    # Save results to JSON for reference
    with open("eval_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_questions": n,
                "answerable": answerable_cases,
                "answered_correctly": correct_answers_attempted,
                "retrieval_hits": retrieval_hits,
                "faithful": faithful_count,
                "correct_refusals": correct_refusals,
                "out_of_scope": unanswerable_cases,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    run_eval()

"""
LLM-as-a-Judge evaluation module for Week 2.

The judge scores a single RAG output against a 4-metric rubric (Likert 1-5):
Faithfulness, Answer Relevance, Correctness, Context Relevance.

The judge model is intentionally different from the Week 1 generator model to
reduce self-evaluation bias:
- Generator (Week 1):   llama-3.3-70b-versatile  (set in rag_pipeline.py)
- Judge    (Week 2):    llama-3.1-8b-instant     (JUDGE_MODEL below)

All judge calls use temperature=0 and require strict JSON output. On parse
failure the judge is retried once with a stricter reminder; on rate-limit
errors the call sleeps 10s and retries once. Persistent failures record null
scores rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from importlib import import_module
    HumanMessage = import_module("langchain.schema").HumanMessage


load_dotenv()

# We initially used llama-3.1-8b-instant for maximum independence from the
# generator model, but it produced uniform 5/5 scores across all variants,
# indicating insufficient critical evaluation. Switched to llama-3.3-70b-versatile
# (same as generator) -- accepting some self-evaluation bias in exchange for
# a judge capable of meaningful score discrimination.
JUDGE_MODEL: str = "llama-3.3-70b-versatile"
JUDGE_TEMPERATURE: float = 0.0

_SCORE_KEYS = ("faithfulness", "answer_relevance", "correctness", "context_relevance")
_REQUIRED_KEYS = (*_SCORE_KEYS, "reasoning")

_judge_llm_singleton: Optional[ChatGroq] = None


def _get_judge_llm() -> ChatGroq:
    """Lazily construct a single ChatGroq instance for the judge."""
    global _judge_llm_singleton
    if _judge_llm_singleton is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY not set. Create a .env file with GROQ_API_KEY=<key>.")
            sys.exit(1)
        _judge_llm_singleton = ChatGroq(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            groq_api_key=api_key,
        )
    return _judge_llm_singleton


def build_judge_prompt(
    question: str,
    retrieved_contexts: List[str],
    rag_answer: str,
    ground_truth: str,
) -> str:
    """Build the judge prompt for a single RAG output.

    The prompt states the evaluator role, gives the full rubric with 1/2/3/4/5
    anchors for each metric plus an explicit anti-leniency instruction,
    provides the numbered retrieved contexts, the RAG answer, and the ground
    truth, and instructs the judge to return ONLY a JSON object with the
    exact keys the caller expects.
    """
    numbered_contexts = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(retrieved_contexts)
    ) if retrieved_contexts else "(no contexts retrieved)"

    return (
        "You are a STRICT evaluator. Most outputs have at least one identifiable "
        "weakness -- find it. A score of 5 means literally perfect with zero gaps. "
        "If you can identify even a minor weakness, the score must be 4 or lower. "
        "Do NOT default to 5; you must justify any 5 you give.\n\n"
        "Score each metric on a 1-5 integer scale using these anchors:\n\n"
        "FAITHFULNESS -- Is every claim in the answer supported by the retrieved context?\n"
        "  1 = Major hallucinations; multiple unsupported or fabricated claims\n"
        "  2 = Several unsupported claims, though some content is grounded\n"
        "  3 = Mostly grounded, but contains one clearly unsupported claim\n"
        "  4 = Fully grounded, but includes minor extrapolation or restatement beyond context\n"
        "  5 = Every single claim is directly traceable to the retrieved context with no extrapolation\n\n"
        "ANSWER_RELEVANCE -- Does the answer address the question asked?\n"
        "  1 = Off-topic or non-responsive\n"
        "  2 = Touches the topic but largely misses the actual question\n"
        "  3 = Partially responsive; addresses some aspects, ignores others\n"
        "  4 = Addresses the question well, but includes tangential or unnecessary content\n"
        "  5 = Directly, completely, and concisely answers exactly what was asked\n\n"
        "CORRECTNESS -- Does the answer match the ground truth semantically?\n"
        "  1 = Contradicts the ground truth\n"
        "  2 = Mostly incorrect with a few accurate elements\n"
        "  3 = Partially correct; some key facts present, some missing or wrong\n"
        "  4 = Largely correct; minor omissions or imprecisions vs. ground truth\n"
        "  5 = Fully matches the meaning, scope, and key facts of the ground truth\n\n"
        "CONTEXT_RELEVANCE -- Are the retrieved chunks relevant to the question?\n"
        "  1 = None of the chunks are relevant\n"
        "  2 = Most chunks are irrelevant; one or two have partial relevance\n"
        "  3 = Mixed; some clearly relevant, some clearly irrelevant noise\n"
        "  4 = Mostly relevant; one chunk is off-topic or marginal\n"
        "  5 = All retrieved chunks are directly and highly relevant to the question\n\n"
        "REMINDER: Be critical. If you find yourself about to give all 5s, stop and "
        "re-read the answer for any minor weakness -- there is almost always one.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXTS (numbered):\n{numbered_contexts}\n\n"
        f"RAG ANSWER:\n{rag_answer}\n\n"
        f"GROUND TRUTH:\n{ground_truth}\n\n"
        "==================== OUTPUT FORMAT ====================\n"
        "Output ONLY a JSON object with EXACTLY these keys:\n"
        '  "faithfulness": integer 1-5,\n'
        '  "answer_relevance": integer 1-5,\n'
        '  "correctness": integer 1-5,\n'
        '  "context_relevance": integer 1-5,\n'
        '  "reasoning": one sentence justifying the lowest-scoring metric.\n'
        "In the reasoning field, explicitly state the weakness that prevented a "
        "perfect score for the lowest-scoring metric. If you gave any metric a 5, "
        "justify in reasoning why no weakness exists for that metric.\n"
        "No preamble. No markdown fences. No trailing commentary. JSON only."
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from a raw LLM response.

    Strips markdown fences, then tries direct parse, then falls back to the
    outermost ``{...}`` substring.
    """
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _coerce_score(value: Any) -> Optional[int]:
    """Coerce a judge score to int in [1, 5] or None if invalid."""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score


def _null_result(reasoning: str, raw: str) -> Dict[str, Any]:
    return {
        "faithfulness": None,
        "answer_relevance": None,
        "correctness": None,
        "context_relevance": None,
        "reasoning": reasoning,
        "raw_response": raw,
    }


def _invoke_judge(prompt: str) -> str:
    """Call the judge LLM once; on rate-limit errors sleep 10s and retry once."""
    llm = _get_judge_llm()
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001 - need broad catch for Groq SDK variants
        message = str(exc).lower()
        if "rate" in message or "429" in message or "too many" in message:
            print(f"  Rate limit hit: {exc}. Sleeping 10s and retrying once...")
            time.sleep(10)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, "content") else str(response)
        raise


def score_one(
    question: str,
    retrieved_contexts: List[str],
    rag_answer: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """Score a single RAG output. Returns a dict with the 4 scores, reasoning, raw_response."""
    prompt = build_judge_prompt(question, retrieved_contexts, rag_answer, ground_truth)

    try:
        raw = _invoke_judge(prompt)
    except Exception as exc:  # noqa: BLE001
        return _null_result(f"judge_error: {exc}", "")

    parsed = _extract_json_object(raw)

    if parsed is None:
        stricter_prompt = (
            "Your previous response was not valid JSON. Return ONLY the JSON object with "
            'the keys faithfulness, answer_relevance, correctness, context_relevance, reasoning. '
            "No preamble, no markdown fences, nothing else.\n\n"
            + prompt
        )
        try:
            raw_retry = _invoke_judge(stricter_prompt)
        except Exception as exc:  # noqa: BLE001
            return _null_result(f"judge_error_on_retry: {exc}", raw)

        parsed = _extract_json_object(raw_retry)
        raw = f"{raw}\n---RETRY---\n{raw_retry}"

        if parsed is None:
            return _null_result("parse_failure_after_retry", raw)

    scores = {k: _coerce_score(parsed.get(k)) for k in _SCORE_KEYS}
    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning is not None else ""

    return {
        **scores,
        "reasoning": reasoning,
        "raw_response": raw,
    }


def score_batch(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score a batch of entries. Prints progress and never crashes on a single failure.

    Each input entry must contain: question, retrieved_contexts, rag_answer, ground_truth.
    All other fields are preserved on output. The judge output is merged in.
    """
    results: List[Dict[str, Any]] = []
    total = len(entries)

    for idx, entry in enumerate(entries, start=1):
        question = entry.get("question", "")
        preview = question[:60].replace("\n", " ")
        print(f"[{idx}/{total}] Scoring: {preview}{'...' if len(question) > 60 else ''}")

        try:
            judge_out = score_one(
                question=question,
                retrieved_contexts=entry.get("retrieved_contexts", []) or [],
                rag_answer=entry.get("rag_answer", "") or "",
                ground_truth=entry.get("ground_truth", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - defensive; score_one already handles most cases
            print(f"  Unexpected failure: {exc}. Recording null scores.")
            judge_out = _null_result(f"unexpected_error: {exc}", "")

        merged = {**entry, **judge_out}
        results.append(merged)

    return results

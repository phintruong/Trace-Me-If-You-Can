"""
Human-readable AML risk report generation using Gemini LLM.

Takes the technical whitebox reports and translates them into plain English
that compliance officers can understand — no z-scores, percentiles, or formulas.

Parallel mode: fires one batch per API key concurrently (~5x speedup).
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

from src.config import GEMINI_API_KEYS, GEMINI_MODEL

log = logging.getLogger(__name__)

BATCH_SIZE = 50    # reports per API call (~50 × 1.5K = 75K tokens, well under 250K limit)
MAX_REPORTS = 2000 # cap total reports sent to Gemini (enough for all HIGH)

SYSTEM_PROMPT = """\
You are an AML (Anti-Money Laundering) compliance analyst writing risk reports \
for compliance officers who are NOT data scientists.

You will receive one or more technical risk reports for flagged bank accounts, \
separated by divider lines. Rewrite ALL of them as clear, concise summaries \
in plain English. Keep each account's summary clearly separated.

Rules:
- Keep the Account ID and Risk Level (HIGH) at the top of each summary.
- Explain WHY each account was flagged using simple, everyday language.
- Describe the suspicious behaviors concretely (e.g. "moved $1.8M in a single \
day", "85% of transactions happened outside business hours").
- DO NOT include z-scores, percentiles, contribution weights, formulas, or \
any statistical jargon.
- DO NOT mention agent names, model names, or scoring methodology.
- Use short paragraphs and bullet points for readability.
- End each account summary with a one-line recommendation (e.g. "Recommend \
escalation for enhanced due diligence").
- Keep each account summary under 250 words.
- Separate each account summary with a line of "=" characters.
"""


def _call_gemini_with_key(key: str, content: str) -> str:
    """Call Gemini with a specific API key."""
    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(content)
    return response.text


def _call_gemini_with_fallback(content: str, primary_key_idx: int) -> str:
    """Try the primary key first, then fall back to others."""
    last_error = None
    for attempt in range(len(GEMINI_API_KEYS)):
        key_idx = (primary_key_idx + attempt) % len(GEMINI_API_KEYS)
        try:
            return _call_gemini_with_key(GEMINI_API_KEYS[key_idx], content)
        except Exception as e:
            last_error = e
            log.warning(f"Key {key_idx + 1} failed: {e}")
    raise last_error


def humanize_reports(technical_reports: list[str]) -> str:
    """Translate technical AML risk reports into plain English via parallel API calls.

    Fires up to len(GEMINI_API_KEYS) batches concurrently, each using a
    different API key. ~5x faster than sequential with 5 keys.

    Args:
        technical_reports: List of technical report strings (one per account).

    Returns:
        A single string with all human-readable summaries, or None on total failure.
    """
    if not GEMINI_API_KEYS:
        log.warning("No GEMINI_KEY* env vars set — skipping LLM humanization.")
        return None

    n_keys = len(GEMINI_API_KEYS)

    # Cap reports
    if len(technical_reports) > MAX_REPORTS:
        log.info(f"  Capping from {len(technical_reports)} to {MAX_REPORTS} reports (top by risk score)")
        technical_reports = technical_reports[:MAX_REPORTS]

    # Build all batches
    batches = []
    for i in range(0, len(technical_reports), BATCH_SIZE):
        batches.append(technical_reports[i:i + BATCH_SIZE])

    total_batches = len(batches)
    log.info(f"  Processing {len(technical_reports)} reports in {total_batches} batch(es) "
             f"(batch_size={BATCH_SIZE}, parallel_keys={n_keys})")

    # Process in waves of n_keys parallel calls
    results = [None] * total_batches
    completed = 0

    for wave_start in range(0, total_batches, n_keys):
        wave_end = min(wave_start + n_keys, total_batches)
        wave_batches = list(range(wave_start, wave_end))

        log.info(f"  Wave {wave_start // n_keys + 1}: batches {wave_start + 1}-{wave_end}/{total_batches} "
                 f"({len(wave_batches)} parallel calls)")

        with ThreadPoolExecutor(max_workers=len(wave_batches)) as executor:
            futures = {}
            for i, batch_idx in enumerate(wave_batches):
                batch_text = "\n\n".join(batches[batch_idx])
                key_idx = i % n_keys
                future = executor.submit(_call_gemini_with_fallback, batch_text, key_idx)
                futures[future] = batch_idx

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    results[batch_idx] = future.result()
                    completed += 1
                    log.info(f"    Batch {batch_idx + 1} done ({completed}/{total_batches})")
                except Exception as e:
                    log.error(f"    Batch {batch_idx + 1} failed: {e}")
                    results[batch_idx] = f"[Batch {batch_idx + 1} failed — {type(e).__name__}]"
                    completed += 1

        # Wait 65s between waves so the per-minute token quota resets
        if wave_end < total_batches:
            log.info(f"  Waiting 65s for token quota reset...")
            time.sleep(65)

    # Filter out None results and join
    final = [r for r in results if r is not None]
    return ("\n\n" + "=" * 80 + "\n\n").join(final) if final else None

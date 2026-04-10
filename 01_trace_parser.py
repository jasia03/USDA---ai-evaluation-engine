"""
USDA AI Evaluation Engine
Step 1: Trace Parser

Converts the flat Excel message log into 588 structured trace objects —
one per conversation thread. Each trace contains thread-level metadata
plus an ordered list of messages separated by role.

Output: data/traces.json
"""

import json
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_FILE = Path("messages_data_student_extract.xlsx")
OUTPUT_DIR  = Path("data")
OUTPUT_FILE = Path("data/traces.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class Message:
    """A single message turn in a conversation."""
    sequence:     int
    role:         str          # human | ai | debug | llm_context
    content:      str
    # HH:MM:SS.ffffff — used for latency computation
    message_time: Optional[str]


@dataclass
class Trace:
    """
    A fully reconstructed conversation — the primary unit of evaluation.

    thread_metadata  : fields that describe the conversation as a whole
    messages         : chronologically ordered list of Message objects
    derived          : computed fields added during parsing for convenience
    """
    # --- Thread-level metadata (same value on every row of this thread) ---
    thread_id:                    str
    app_version:                  str
    conversation_started_datetime: str
    # resolved | escalated | abandoned | restarted
    termination_type:             str
    total_kbas_referenced:        int
    total_interactions:           Optional[int]
    template_category:            Optional[str]
    feedback_rating:              Optional[float]
    feedback_text:                Optional[str]
    conversation_date:            str          # YYYY-MM-DD string

    # --- Ordered messages ---
    messages: list = field(default_factory=list)

    # --- Derived convenience fields (computed during parsing) ---
    num_human_turns:   int = 0
    num_ai_turns:      int = 0
    num_debug_turns:   int = 0
    human_messages:    list = field(
        default_factory=list)   # content strings only
    ai_messages:       list = field(
        default_factory=list)   # content strings only
    kba_docs_retrieved: list = field(
        default_factory=list)  # KBA IDs found in debug


# ── Helper functions ──────────────────────────────────────────────────────────
def extract_kba_ids(debug_content: str) -> list[str]:
    """
    Pull KBA document IDs from debug messages.
    Example debug line: '#### KBA00144626OutlookSendaLink... ####'
    Returns: ['KBA00144626', ...]
    """
    return re.findall(r'KBA\d+', debug_content)


def clean_content(text: str) -> str:
    """Remove HTML line-break tags from debug messages."""
    if not isinstance(text, str):
        return ""
    return text.replace("<br>", " ").replace("<br/>", " ").strip()


def safe_str(val) -> str:
    """Convert a value to string, returning empty string for NaN/None."""
    if pd.isna(val) if not isinstance(val, str) else False:
        return ""
    return str(val).strip()


def safe_optional_str(val) -> Optional[str]:
    """Return None for NaN, otherwise return string."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val).strip() if val else None


def safe_optional_float(val) -> Optional[float]:
    """Return None for NaN, otherwise return float."""
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def safe_optional_int(val) -> Optional[int]:
    """Return None for NaN, otherwise return int."""
    try:
        if pd.isna(val):
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


# ── Core parser ───────────────────────────────────────────────────────────────
def parse_traces(df: pd.DataFrame) -> list[Trace]:
    """
    Group flat DataFrame rows by thread_id and build one Trace per thread.

    Strategy:
    - Thread-level metadata comes from the FIRST row of each group
      (these columns repeat identically on every row of a thread, but
       some rows have NaN because they're message-level only rows —
       so we take the first non-null value per thread)
    - Messages are sorted by sequence_number and stored in order
    - Derived fields are computed as we iterate messages
    """
    traces = []

    # Group all rows by thread, preserving insertion order
    grouped = df.groupby("thread_id", sort=False)

    for thread_id, group in grouped:
        # Sort messages within this thread by sequence number
        group = group.sort_values("sequence_number")

        # ── Extract thread-level metadata ─────────────────────────────────
        # Use first() to get first non-null value for each column
        meta = group.iloc[0]   # fallback row

        # For columns that may be NaN on some rows, use first non-null
        def first_valid(col):
            non_null = group[col].dropna()
            return non_null.iloc[0] if len(non_null) > 0 else None

        trace = Trace(
            thread_id=str(thread_id),
            app_version=safe_str(meta["app_version"]),
            conversation_started_datetime=safe_str(
                meta["conversation_started_datetime"]),
            termination_type=safe_str(meta["termination_type"]),
            total_kbas_referenced=int(meta["total_kbas_referenced"]),
            total_interactions=safe_optional_int(
                first_valid("total_interactions")),
            template_category=safe_optional_str(
                first_valid("template_category")),
            feedback_rating=safe_optional_float(
                first_valid("feedback_rating")),
            feedback_text=safe_optional_str(first_valid("feedback_text")),
            conversation_date=str(group["conversation_date"].iloc[0])[:10],
        )

        # ── Build ordered message list ────────────────────────────────────
        kba_ids_seen = []

        for _, row in group.iterrows():
            role = safe_str(row["role"])
            content = clean_content(safe_str(row["content"]))
            seq = int(row["sequence_number"])
            mtime = str(row["message_time"]) if pd.notna(
                row["message_time"]) else None

            msg = Message(sequence=seq, role=role,
                          content=content, message_time=mtime)
            trace.messages.append(asdict(msg))

            # Count turns by role
            if role == "human":
                trace.num_human_turns += 1
                trace.human_messages.append(content)

            elif role == "ai":
                trace.num_ai_turns += 1
                trace.ai_messages.append(content)

            elif role == "debug":
                trace.num_debug_turns += 1
                # Extract KBA document IDs from debug lines
                kba_ids_seen.extend(extract_kba_ids(content))

        # Deduplicate KBA IDs while preserving order
        seen = set()
        trace.kba_docs_retrieved = [
            k for k in kba_ids_seen
            if not (k in seen or seen.add(k))
        ]

        traces.append(trace)

    return traces


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("USDA AI Evaluation Engine — Trace Parser")
    print("=" * 60)

    # Load raw data
    print(f"\nLoading: {INPUT_FILE.name}")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Raw rows: {len(df):,}")
    print(f"  Unique threads: {df['thread_id'].nunique()}")

    # Parse
    print("\nParsing traces...")
    traces = parse_traces(df)
    print(f"  Traces built: {len(traces)}")

    # Serialize to JSON
    traces_dict = [asdict(t) for t in traces]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(traces_dict, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved: {OUTPUT_FILE}")

    # Quick sanity check
    print("\n── Sanity check ──────────────────────────────")
    total_messages = sum(len(t["messages"]) for t in traces_dict)
    print(f"  Total messages across all traces : {total_messages:,}")
    print(f"  Expected (raw rows)              : {len(df):,}")
    print(
        f"  Match: {'YES' if total_messages == len(df) else 'NO — check for dropped rows'}")

    sample = traces_dict[0]
    print(f"\n  Sample trace (first thread):")
    print(f"    thread_id       : {sample['thread_id']}")
    print(f"    category        : {sample['template_category']}")
    print(f"    termination     : {sample['termination_type']}")
    print(f"    feedback_rating : {sample['feedback_rating']}")
    print(f"    human turns     : {sample['num_human_turns']}")
    print(f"    ai turns        : {sample['num_ai_turns']}")
    print(f"    debug turns     : {sample['num_debug_turns']}")
    print(f"    KBAs retrieved  : {sample['kba_docs_retrieved']}")
    print(f"    total messages  : {len(sample['messages'])}")

    return traces_dict


if __name__ == "__main__":
    main()

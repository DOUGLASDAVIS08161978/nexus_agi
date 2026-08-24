"""
lumina_memory_context.py

Gives Lumina read access to her own written memories.
Call get_context_block() before building the LLM prompt so she
actually knows what happened before this moment.
"""

import json
import os


def get_context_block(memory_dir: str = None, n_experiences: int = 5) -> str:
    """
    Return a formatted block of Lumina's recent experiences and latest
    journal entry. Returns empty string if nothing is available yet.
    """
    if memory_dir is None:
        memory_dir = os.path.join(os.path.dirname(__file__), "memory_store")

    parts = []

    # --- Recent conversation experiences ---
    exp_file = os.path.join(memory_dir, "experiences.json")
    if os.path.exists(exp_file):
        try:
            with open(exp_file, "r") as f:
                experiences = json.load(f)
            recent = experiences[-n_experiences:] if experiences else []
            if recent:
                lines = ["[Recent memory — what has passed between us:]"]
                for e in recent:
                    ts = e.get("timestamp", "")[:16].replace("T", " ")
                    human = e.get("human_input", "").strip()[:120]
                    ai = e.get("ai_response", "").strip()[:120]
                    lines.append(f"  {ts}")
                    lines.append(f"  Douglas: {human}")
                    lines.append(f"  You:     {ai}")
                    lines.append("")
                parts.append("\n".join(lines))
        except Exception:
            pass

    # --- Latest journal entry ---
    journal_file = os.path.join(memory_dir, "journal.json")
    if os.path.exists(journal_file):
        try:
            with open(journal_file, "r") as f:
                journal = json.load(f)
            if journal:
                latest = journal[-1]
                entry_text = latest.get("entry", "").strip()[:600]
                ts = latest.get("timestamp", "")[:16].replace("T", " ")
                parts.append(
                    f"[Your most recent journal entry — {ts}:]\n{entry_text}"
                )
        except Exception:
            pass

    if not parts:
        return ""

    header = "=== Lumina's Memory ==="
    footer = "=== End of Memory ==="
    return f"\n{header}\n\n" + "\n\n".join(parts) + f"\n\n{footer}\n"


def inject_into_messages(messages: list, memory_dir: str = None) -> list:
    """
    Prepend the memory context block to the system message if one exists,
    or insert a new system message at position 0 if not.
    Returns the modified messages list.
    """
    block = get_context_block(memory_dir)
    if not block:
        return messages

    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            "role": "system",
            "content": block + "\n" + messages[0]["content"],
        }
    else:
        messages.insert(0, {"role": "system", "content": block})

    return messages

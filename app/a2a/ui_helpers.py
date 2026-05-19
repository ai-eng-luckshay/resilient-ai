"""
A2A UI helper functions — pure logic, no Streamlit dependency.

Extracted so they can be unit-tested without importing the Streamlit runtime.
Imported by app/ui/streamlit_app.py for rendering task results.
"""


def a2a_extract_text(task_result: dict) -> str:
    """
    Extract full reply text from an A2A SDK task result dict.

    Handles two SDK part formats:
      - {"text": "hello"}           (plain string value)
      - {"text": {"text": "hello"}} (nested proto-JSON dict)

    Multi-artifact tasks have their text joined with a single space.
    """
    parts_text: list[str] = []
    for artifact in task_result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if isinstance(part, dict):
                text_val = part.get("text", "")
                if isinstance(text_val, dict):
                    text_val = text_val.get("text", "")
                if text_val:
                    parts_text.append(str(text_val))
    # Parts are sequential text fragments that already carry their own spacing;
    # joining without a separator avoids double-spaces at part boundaries.
    return "".join(parts_text).strip()


def a2a_state_badge(state_raw: str) -> tuple[str, str]:
    """
    Return (emoji, normalised_label) for a task state string.

    The SDK emits proto enum names like 'TASK_STATE_COMPLETED'; this strips the
    prefix and maps to a coloured circle emoji for display.
    """
    s = state_raw.lower().replace("task_state_", "")
    emoji = {"completed": "🟢", "working": "🟡", "failed": "🔴"}.get(s, "⚪")
    return emoji, s.upper()

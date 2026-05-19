"""
Unit tests for A2A UI helper functions (app/a2a/ui_helpers.py).

Tests:
  a2a_extract_text
    - single artifact, single part
    - multiple artifacts concatenated
    - multiple parts within one artifact
    - nested proto-JSON {"text": {"text": "..."}} format
    - empty parts / missing keys return empty string
    - result.task unwrapping (SendMessage response structure)

  a2a_state_badge
    - known states: completed, working, failed
    - unknown state falls back to ⚪
    - proto enum prefix (TASK_STATE_*) stripped correctly
"""
import pytest

from app.a2a.ui_helpers import a2a_extract_text, a2a_state_badge


# ── a2a_extract_text ────────────────────────────────────────────────────────

class TestA2aExtractText:

    def test_single_artifact_single_part(self):
        task = {
            "artifacts": [
                {"parts": [{"text": "15 squared is 225."}]}
            ]
        }
        assert a2a_extract_text(task) == "15 squared is 225."

    def test_multiple_parts_within_artifact_joined(self):
        """Parts within one artifact are joined with a space."""
        task = {
            "artifacts": [
                {"parts": [{"text": "Hello,"}, {"text": "world!"}]}
            ]
        }
        result = a2a_extract_text(task)
        assert "Hello," in result
        assert "world!" in result

    def test_multiple_artifacts_joined(self):
        """Text from all artifacts is concatenated."""
        task = {
            "artifacts": [
                {"parts": [{"text": "Part one."}]},
                {"parts": [{"text": "Part two."}]},
            ]
        }
        result = a2a_extract_text(task)
        assert "Part one." in result
        assert "Part two." in result

    def test_nested_proto_json_part(self):
        """SDK may emit {"text": {"text": "..."}} (nested dict) for some versions."""
        task = {
            "artifacts": [
                {"parts": [{"text": {"text": "Nested reply"}}]}
            ]
        }
        assert a2a_extract_text(task) == "Nested reply"

    def test_empty_artifacts_returns_empty_string(self):
        assert a2a_extract_text({"artifacts": []}) == ""

    def test_no_artifacts_key_returns_empty_string(self):
        assert a2a_extract_text({}) == ""

    def test_part_with_empty_text_skipped(self):
        task = {
            "artifacts": [
                {"parts": [{"text": ""}, {"text": "Real content"}]}
            ]
        }
        assert a2a_extract_text(task) == "Real content"

    def test_non_dict_part_skipped(self):
        """Defensive: malformed parts list should not raise."""
        task = {
            "artifacts": [
                {"parts": ["not a dict", {"text": "Good part"}]}
            ]
        }
        assert a2a_extract_text(task) == "Good part"

    def test_real_sdk_response_structure(self):
        """
        Mirrors the actual SDK response seen in the A2A Inspector.

        SendMessage returns:
          { "result": { "task": { "artifacts": [{ "parts": [{"text": "..."}] }] } } }

        The UI unwraps result.task before calling a2a_extract_text, so the
        function receives the inner task dict.
        """
        sdk_response = {
            "result": {
                "task": {
                    "id": "task-123",
                    "artifacts": [
                        {
                            "artifactId": "art-1",
                            "parts": [
                                {"text": "15 squared"},
                                {"text": " is 225."},
                            ],
                        }
                    ],
                }
            }
        }
        # Replicate what the UI does: unwrap result.task
        outer = sdk_response.get("result", {})
        task_result = outer.get("task", outer)
        # Parts are joined without separator; " is 225." already has leading space.
        assert a2a_extract_text(task_result) == "15 squared is 225."


# ── a2a_state_badge ─────────────────────────────────────────────────────────

class TestA2aStateBadge:

    def test_completed_state(self):
        emoji, label = a2a_state_badge("TASK_STATE_COMPLETED")
        assert emoji == "🟢"
        assert label == "COMPLETED"

    def test_working_state(self):
        emoji, label = a2a_state_badge("TASK_STATE_WORKING")
        assert emoji == "🟡"
        assert label == "WORKING"

    def test_failed_state(self):
        emoji, label = a2a_state_badge("TASK_STATE_FAILED")
        assert emoji == "🔴"
        assert label == "FAILED"

    def test_unknown_state_falls_back(self):
        emoji, label = a2a_state_badge("TASK_STATE_CANCELED")
        assert emoji == "⚪"
        assert label == "CANCELED"

    def test_label_always_uppercase(self):
        _, label = a2a_state_badge("TASK_STATE_COMPLETED")
        assert label == label.upper()

    def test_state_without_prefix_still_works(self):
        """Defensive: bare state strings (no TASK_STATE_ prefix) handled gracefully."""
        emoji, label = a2a_state_badge("completed")
        assert emoji == "🟢"
        assert label == "COMPLETED"

"""Tests for data_prep/alix_reviews.py — PR review JSON → chat-format JSONL."""

import json

import pytest
from model_training.sft.data_prep import alix_reviews as ar


def _comment(**over):
    c = {
        "comment_type": "review_comment",
        "author": "alice",
        "path": "src/app.py",
        "line": 42,
        "context": "def foo():\n    return 1",
        "body": "This function should handle the None case explicitly.",
    }
    c.update(over)
    return c


# ── filters ───────────────────────────────────────────────────────────────────


class TestIsBot:
    def test_known_bot(self):
        assert ar.is_bot("coderabbitai[bot]") is True

    def test_bracket_bot_heuristic(self):
        assert ar.is_bot("some-new-tool[bot]") is True

    def test_human(self):
        assert ar.is_bot("alice") is False


class TestIsGenericBody:
    @pytest.mark.parametrize("body", ["lgtm", "LGTM!", "+1", "approved", "  nit  ", "done"])
    def test_generic(self, body):
        assert ar.is_generic_body(body) is True

    def test_empty(self):
        assert ar.is_generic_body("   ") is True

    def test_substantive(self):
        assert ar.is_generic_body("Consider extracting this into a helper.") is False


class TestIsReviewableComment:
    def test_good_comment(self):
        assert ar.is_reviewable_comment(_comment()) is True

    def test_bot_rejected(self):
        assert ar.is_reviewable_comment(_comment(author="dependabot[bot]")) is False

    def test_generic_body_rejected(self):
        assert ar.is_reviewable_comment(_comment(body="lgtm")) is False

    def test_missing_context_rejected(self):
        assert ar.is_reviewable_comment(_comment(context="")) is False

    def test_missing_path_rejected(self):
        assert ar.is_reviewable_comment(_comment(path="")) is False

    def test_too_short_body_rejected(self):
        assert ar.is_reviewable_comment(_comment(body="fix")) is False


# ── example construction ───────────────────────────────────────────────────────


class TestBuildExample:
    def test_messages_shape(self):
        ex = ar.build_example(_comment(), repo="meetalix/alix-mobile")
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_content_carried(self):
        ex = ar.build_example(_comment(), repo="meetalix/alix-mobile")
        system, user, assistant = ex["messages"]
        assert "meetalix/alix-mobile" in system["content"]
        assert "src/app.py" in user["content"]
        assert "def foo():" in user["content"]
        assert assistant["content"] == "This function should handle the None case explicitly."


class TestPrToExamples:
    def test_filters_and_builds(self):
        pr = {
            "repo": "alix-mobile",
            "pr_number": 246,
            "comments": [
                _comment(),                       # keep
                _comment(author="cursor[bot]"),   # drop (bot)
                _comment(body="lgtm"),            # drop (generic)
                _comment(context=""),             # drop (no context)
            ],
        }
        examples = ar.pr_to_examples(pr)
        assert len(examples) == 1
        assert "alix-mobile" in examples[0]["messages"][0]["content"]

    def test_empty_comments(self):
        assert ar.pr_to_examples({"repo": "r", "comments": []}) == []


# ── dataset conversion ─────────────────────────────────────────────────────────


class TestConvertDataset:
    def _write_pr(self, d, name, comments):
        (d / name).write_text(json.dumps({"repo": "alix-mobile", "pr_number": 1, "comments": comments}))

    def test_writes_jsonl_and_stats(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._write_pr(src, "pr-1.json", [_comment(), _comment(body="lgtm")])
        self._write_pr(src, "pr-2.json", [_comment(path="src/b.py")])
        out = tmp_path / "reviews.jsonl"

        stats = ar.convert_dataset(src, out)

        lines = out.read_text().splitlines()
        assert len(lines) == 2  # two reviewable comments kept, one generic dropped
        for line in lines:
            rec = json.loads(line)
            assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
        assert stats["examples_written"] == 2
        assert stats["prs_processed"] == 2

    def test_skips_non_pr_json(self, tmp_path):
        # e.g. pr-list.json (an enumeration file, no "comments") must be ignored
        src = tmp_path / "src"
        src.mkdir()
        (src / "pr-list.json").write_text(json.dumps([{"number": 1}, {"number": 2}]))
        self._write_pr(src, "pr-1.json", [_comment()])
        out = tmp_path / "reviews.jsonl"

        stats = ar.convert_dataset(src, out)
        assert stats["examples_written"] == 1
        assert stats["prs_processed"] == 1  # the list file was skipped, not counted

    def test_output_passes_data_validation(self, tmp_path):
        # The emitted JSONL must be accepted by the SFT loader's validator.
        from model_training.sft.data import load_jsonl, validate_examples

        src = tmp_path / "src"
        src.mkdir()
        self._write_pr(src, "pr-1.json", [_comment(), _comment(path="src/b.py")])
        out = tmp_path / "reviews.jsonl"
        ar.convert_dataset(src, out)

        validate_examples(load_jsonl(out))  # must not raise

import anthropic
import httpx

from evaluate import run_case, score_case, summarize
from rag.search import Hit


def test_run_case_records_api_error_and_continues():
    def failing_answer(question, hits):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://api"))

    case = {"question": "q", "expected_path": "a", "expected_phrase": "x"}
    row, lines = run_case(case, k=8, retrieve_only=False,
                          search_fn=lambda q, k: [Hit("a#0", "a", [], 0.9, "")],
                          answer_fn=failing_answer)
    assert row["hit"] is True
    assert row["answer_ok"] is False
    assert any("HATA" in line for line in lines)


def test_score_case_hit_rank_and_phrase_case_insensitive():
    hits = [Hit("x#0", "x", [], 0.9, ""), Hit("notlar/02.md#3", "notlar/02.md", [], 0.8, "")]
    case = {"question": "q", "expected_path": "notlar/02.md", "expected_phrase": "/api/v2/"}
    r = score_case(case, hits, "Karar: /API/V2/* [2]")
    assert r["hit"] is True and r["rank"] == 2 and r["answer_ok"] is True


def test_score_case_miss():
    case = {"question": "q", "expected_path": "a.md", "expected_phrase": "x"}
    r = score_case(case, [Hit("b#0", "b", [], 0.5, "")], "y")
    assert r["hit"] is False and r["rank"] is None and r["answer_ok"] is False


def test_score_case_trap_question_has_no_retrieval_target():
    case = {"question": "q", "expected_path": None, "expected_phrase": "bulamadım"}
    r = score_case(case, [], "Dokümanlarda bulamadım.")
    assert r["hit"] is None and r["answer_ok"] is True


def test_score_case_retrieve_only():
    case = {"question": "q", "expected_path": "a", "expected_phrase": "x"}
    assert score_case(case, [], None)["answer_ok"] is None


def test_summarize_ignores_not_applicable():
    rows = [{"hit": True, "answer_ok": True}, {"hit": False, "answer_ok": False},
            {"hit": None, "answer_ok": True}]
    s = summarize(rows)
    assert "hit@k: 1/2" in s
    assert "cevap: 2/3" in s

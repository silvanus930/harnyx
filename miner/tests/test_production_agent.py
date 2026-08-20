from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from harnyx_miner_sdk.api import LlmChatResult, ToolCallResponse
from harnyx_miner_sdk.decorators import clear_entrypoints
from harnyx_miner_sdk.llm import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmUsage,
)
from harnyx_miner_sdk.query import Query
from harnyx_miner_sdk.tools.http_models import ToolBudgetDTO, ToolResultDTO
from harnyx_miner_sdk.tools.search_models import FetchPageResponse, SearchWebSearchResponse

pytestmark = pytest.mark.anyio("asyncio")

_AGENT_PATH = Path(__file__).resolve().parents[2] / "agent.py"


def _load_agent() -> ModuleType:
    clear_entrypoints()
    spec = importlib.util.spec_from_file_location("production_agent", _AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent() -> ModuleType:
    return _load_agent()


def _budget(remaining_usd: float = 1.0) -> ToolBudgetDTO:
    return ToolBudgetDTO(
        session_budget_usd=1.0,
        session_hard_limit_usd=1.0,
        session_used_budget_usd=1.0 - remaining_usd,
        session_remaining_budget_usd=remaining_usd,
    )


def _search_response(
    results: list[dict[str, Any]],
    *,
    remaining_budget_usd: float = 1.0,
) -> ToolCallResponse[SearchWebSearchResponse]:
    parsed = tuple(ToolResultDTO.model_validate(result) for result in results)
    return ToolCallResponse(
        receipt_id="search-receipt",
        response=SearchWebSearchResponse(data=[]),
        results=parsed,
        result_policy="referenceable",
        cost_usd=0.003,
        usage=None,
        budget=_budget(remaining_budget_usd),
    )


def _fetch_response(result: dict[str, Any]) -> ToolCallResponse[FetchPageResponse]:
    parsed = (ToolResultDTO.model_validate(result),)
    return ToolCallResponse(
        receipt_id="fetch-receipt",
        response=FetchPageResponse(data=[]),
        results=parsed,
        result_policy="referenceable",
        cost_usd=0.001,
        usage=None,
        budget=_budget(),
    )


def _text_chat_result(content: str) -> LlmChatResult:
    response = LlmResponse(
        id="resp-text",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=content),),
                ),
            ),
        ),
        usage=LlmUsage(),
    )
    return LlmChatResult(
        receipt_id="chat-receipt",
        response=response,
        results=(),
        result_policy="log_only",
        cost_usd=0.01,
        usage=None,
        budget=_budget(),
    )


def _tool_call_chat_result(name: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> LlmChatResult:
    response = LlmResponse(
        id="resp-tool",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(),
                    tool_calls=(
                        LlmMessageToolCall(id=call_id, type="function", name=name, arguments=json.dumps(arguments)),
                    ),
                ),
            ),
        ),
        usage=LlmUsage(),
    )
    return LlmChatResult(
        receipt_id="chat-receipt-tool",
        response=response,
        results=(),
        result_policy="log_only",
        cost_usd=0.01,
        usage=None,
        budget=_budget(),
    )


async def test_loop_calls_search_then_finishes_with_citation(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        {
            "index": 0,
            "result_id": "r-1",
            "url": "https://example.com/a",
            "note": "Alpha evidence " * 6,
            "title": "Alpha",
        }
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**kwargs: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result("search", {"query": "the question"})
        return _text_chat_result("The answer is supported by evidence [[0]].")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What does the evidence say?"))

    # citation markers are remapped to their 1-based position in the final
    # materialized citations list, not the raw internal evidence index
    assert "[[1]]" in result.text
    assert result.citations is not None
    assert [ref.result_id for ref in result.citations] == ["r-1"]
    assert result.citations[0].receipt_id == "search-receipt"
    # call 1: search tool call, call 2: loop finish, call 3: format audit pass
    assert call_count["n"] == 3


async def test_loop_calls_open_page_after_search(agent: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    search_results = [
        {"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": "thin", "title": "Alpha"}
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(search_results)

    async def fake_fetch_page(*_: object, **__: object) -> ToolCallResponse[FetchPageResponse]:
        return _fetch_response(
            {
                "index": 0,
                "result_id": "r-1-full",
                "url": "https://example.com/a",
                "note": "The full page content answers the question. " * 5,
                "title": "Alpha",
            }
        )

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result("search", {"query": "the question"})
        if call_count["n"] == 2:
            return _tool_call_chat_result("open_page", {"url": "https://example.com/a"}, call_id="call-2")
        return _text_chat_result("Confirmed by the full page [[0]].")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "fetch_page", fake_fetch_page)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What does the full page say?"))

    # open_page fetches the same URL the search step already saw -- it
    # replaces that entry in place (richer content) rather than creating a
    # second one, so the citation still ends up at position 1 after remap.
    assert "[[1]]" in result.text
    assert result.citations is not None
    assert [ref.result_id for ref in result.citations] == ["r-1-full"]


async def test_loop_note_evidence_flows_into_final_citation(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end: the model reads a large page, calls note_evidence on the
    # decisive quote mid-loop, and the final citation must point at that
    # quote rather than falling back to a keyword-density guess or the
    # literal page head.
    quote = "the decisive value is 42"
    note = ("filler text with no matching keywords " * 300) + quote + (" more filler" * 300)
    assert len(note) > agent.MAX_CITATION_SLICE_CHARS

    search_results = [
        {"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": "thin", "title": "Alpha"}
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(search_results)

    async def fake_fetch_page(*_: object, **__: object) -> ToolCallResponse[FetchPageResponse]:
        return _fetch_response(
            {"index": 0, "result_id": "r-1-full", "url": "https://example.com/a", "note": note, "title": "Alpha"}
        )

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result("search", {"query": "the question"})
        if call_count["n"] == 2:
            return _tool_call_chat_result("open_page", {"url": "https://example.com/a"}, call_id="call-2")
        if call_count["n"] == 3:
            return _tool_call_chat_result("note_evidence", {"evidence_index": 0, "quote": quote}, call_id="call-3")
        return _text_chat_result("The decisive value is 42 [[0]].")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "fetch_page", fake_fetch_page)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What is the decisive value?"))

    assert result.citations is not None
    slice_ = result.citations[0].slices[0]
    quote_pos = note.find(quote)
    assert slice_.start <= quote_pos < slice_.end


async def test_total_provider_outage_never_raises_and_returns_hedged_response(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        raise RuntimeError("search provider unavailable")

    async def failing_llm_chat(**__: object) -> LlmChatResult:
        raise RuntimeError("llm provider unavailable")

    monkeypatch.setattr(agent, "search_web", failing_search_web)
    monkeypatch.setattr(agent, "llm_chat", failing_llm_chat)

    result = await agent.query(Query(text="What happens when every provider is down?"))

    assert result.text
    assert result.citations is None


async def test_deterministic_rescue_rung_used_when_loop_answer_unusable(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        {"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": "Alpha evidence", "title": "Alpha"}
    ]

    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # first turn gathers evidence; every synthesis attempt after that
            # (forced final answer, digest rewrite, knowledge-only) returns an
            # unusably short answer, forcing the deterministic rescue rung
            return _tool_call_chat_result("search", {"query": "the question"})
        return _text_chat_result("no")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What does the evidence say?"))

    assert "Evidence gathered for" in result.text
    assert "Alpha evidence" in result.text
    assert result.citations is not None


async def test_loop_retries_after_leaked_tool_call_markup_and_recovers_real_answer(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss: a loop turn ended with the model writing a tool
    # call out as literal prose ("<function_calls><invoke name=\"compute\">
    # ...") instead of a real structured tool_calls entry, with plenty of
    # budget/turns left unused (129s of a 200s budget, 11 of many possible
    # calls). Ending the loop there and falling to the rescue ladder would
    # ship a worse answer than the model could actually produce -- the
    # ladder can't run `compute` at all. The loop must instead nudge the
    # model to make the call for real and continue, recovering the actual
    # computed answer.
    call_count = {"n": 0}

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _text_chat_result(
                'I will now compute this.\n<function_calls>\n<invoke name="compute">\n'
                '<parameter name="code">result = str(1 + 1)</parameter>\n</invoke>\n</function_calls>'
            )
        if call_count["n"] == 2:
            return _tool_call_chat_result("compute", {"code": "result = str(1 + 1)"})
        return _text_chat_result("The computed value is 2, with no evidence needed.")

    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What is 1 plus 1?"))

    assert "<function_calls>" not in result.text
    assert "<invoke" not in result.text
    assert "2" in result.text


async def test_loop_retries_when_draft_self_admits_a_coverage_gap(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss, seen twice in real transcripts: a question named
    # four quarterly documents, the model opened two, then finalized an
    # answer that literally said the other two "were not opened, so no
    # evidence exists for those quarters" -- despite 100+ seconds of unused
    # budget and turns to spare, and despite the prompt already telling it
    # not to do this. Catching that self-admission and sending it back for
    # the missing evidence must recover a complete answer instead.
    results = [
        {"index": 0, "result_id": "r-q3", "url": "https://example.com/q3", "note": "Q3 data", "title": "Q3"}
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _text_chat_result(
                "Q1 and Q2 qualify [[1]]. The Q3 document was not opened, "
                "so no evidence exists for that quarter."
            )
        if call_count["n"] == 2:
            return _tool_call_chat_result("search", {"query": "Q3 report"})
        return _text_chat_result("Q1, Q2, and Q3 all qualify, fully checked [[1]][[2]].")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="Using all quarterly reports, which stocks qualify?"))

    assert "was not opened" not in result.text
    assert "Q3" in result.text
    assert "fully checked" in result.text


async def test_loop_retries_on_confidently_hedged_give_up_phrasing(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed BrowseComp loss, 4/5 sampled items: a multi-clue riddle
    # question drew "I cannot confidently identify..." / "...cannot
    # confidently name it" / "cannot provide a confident answer" -- none of
    # which the older, narrower verb list ("cannot determine/complete/
    # finish") matched, so the give-up shipped as the final answer instead
    # of triggering the existing retry-with-more-search mechanism.
    results = [
        {"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": "Actor bio", "title": "Bio"}
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _text_chat_result(
                "Based on the evidence gathered, I cannot confidently identify "
                "the actor described by these clues."
            )
        if call_count["n"] == 2:
            return _tool_call_chat_result("search", {"query": "actor L'Oreal catholic upbringing age 8"})
        return _text_chat_result("The actor is Jane Doe, born in 1979 [[1]].")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="Which actor fits these clues? What year were they born?"))

    assert "cannot confidently identify" not in result.text
    assert "Jane Doe" in result.text


async def test_loop_retries_on_chinese_language_give_up_phrasing(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed WebWalkerQA (Multi-Source) loss: this suite draws
    # bilingual questions, and a Chinese self-admission of incompleteness
    # ("但具体的历史意义内容在现有证据中未完整显示") shipped as the final
    # answer because every _INCOMPLETE_COVERAGE_RE pattern was English-only,
    # so the same retry mechanism that already recovers English give-ups
    # never even got a chance to fire in Chinese.
    results = [
        {"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": "History note", "title": "History"}
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _text_chat_result(
                "南京代表了中共在解放战争的重要阶段[[1]]。"
                "浙江台州市临海市紫阳街的历史意义在现有证据中未完整显示。"
            )
        if call_count["n"] == 2:
            return _tool_call_chat_result("search", {"query": "紫阳街 历史意义"})
        return _text_chat_result(
            "南京代表了中共在解放战争的重要阶段[[1]]，"
            "紫阳街象征着自新中国成立以来的历史文化变迁[[2]]。"
        )

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="南京和紫阳街分别代表了什么历史意义？"))

    assert "未完整显示" not in result.text
    assert "历史文化变迁" in result.text


async def test_leaked_tool_call_markup_is_rejected_not_shipped_as_answer(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even if the retry above also comes back garbled (or the model keeps
    # failing to make a real call), the rescue ladder must still catch it
    # at the end rather than ever shipping raw tool-call markup as the
    # final answer.
    results = [
        {"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": "Alpha evidence", "title": "Alpha"}
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result("search", {"query": "the question"})
        if call_count["n"] in (2, 3):
            return _text_chat_result(
                'I will now compute this.\n<function_calls>\n<invoke name="compute">\n'
                '<parameter name="code">result = str(1 + 1)</parameter>\n</invoke>\n</function_calls>'
            )
        return _text_chat_result("no")

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What does the evidence say?"))

    assert "<function_calls>" not in result.text
    assert "<invoke" not in result.text
    assert "Evidence gathered for" in result.text


async def test_low_budget_skips_structured_output_llm_calls(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_llm_chat(**__: object) -> LlmChatResult:
        raise AssertionError("llm_chat must not be called once the budget is exhausted")

    monkeypatch.setattr(agent, "llm_chat", unexpected_llm_chat)

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    state = agent.RunState()
    state.note_budget(0.0)
    store = agent.EvidenceStore()
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="Alpha", note="Alpha evidence")

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "fallback text", state
    )

    assert structured == {"answer": "fallback text"}


async def test_structured_output_repairs_after_invalid_first_attempt(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        {
            "index": 0,
            "result_id": "r-1",
            "url": "https://example.com/a",
            "note": "Alpha evidence " * 6,
            "title": "Alpha",
        }
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result("search", {"query": "the question"})
        if call_count["n"] == 2:
            return _text_chat_result("Plain text answer citing [[0]].")
        if call_count["n"] == 3:
            # format-audit pass -- keep the draft answer unchanged
            return _text_chat_result("Plain text answer citing [[0]].")
        if call_count["n"] == 4:
            return _text_chat_result("not valid json")
        return _text_chat_result('{"answer": "ok", "confidence": 1}')

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "integer"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    result = await agent.query(Query(text="Structured question", output_schema=schema))

    assert result.output == {"answer": "ok", "confidence": 1}
    # call 1: search, 2: loop finish, 3: format audit, 4: invalid structured
    # attempt, 5: valid structured retry
    assert call_count["n"] == 5


async def test_structured_output_extract_fallback_uses_real_text_answer_not_zero(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss: two live tasks gathered correct evidence and the
    # loop's own text answer stated the real values, but both
    # reasoning-from-evidence structured attempts failed validation, and the
    # old fallback (_best_effort_structured) discarded that real text and
    # returned bare 0/[] defaults for every number/array field even though
    # the count was right there in prose. The extract-don't-paste fallback
    # (a simpler "restate this prose as JSON" call) must catch this instead
    # of falling all the way to placeholder zeros.
    results = [
        {
            "index": 0,
            "result_id": "r-1",
            "url": "https://example.com/a",
            "note": "Roster evidence " * 6,
            "title": "Roster",
        }
    ]
    call_count = {"n": 0}

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result("search", {"query": "the question"})
        if call_count["n"] == 2:
            return _text_chat_result("Four items match: A, B, C, D [[0]].")
        if call_count["n"] == 3:
            # format-audit pass -- keep the draft answer unchanged
            return _text_chat_result("Four items match: A, B, C, D [[0]].")
        if call_count["n"] in (4, 5):
            # both reasoning-from-evidence structured attempts fail
            return _text_chat_result("not valid json")
        # extract-don't-paste fallback: a simpler restate-as-JSON call
        return _text_chat_result('{"count": 4, "items": ["A", "B", "C", "D"]}')

    monkeypatch.setattr(agent, "search_web", fake_search_web)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["count", "items"],
        "additionalProperties": False,
    }

    result = await agent.query(Query(text="How many items match?", output_schema=schema))

    assert result.output == {"count": 4, "items": ["A", "B", "C", "D"]}


async def test_structured_output_repair_retry_rejects_reasoning_dump_in_field(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss, seen twice in one live vs-champion sample: the
    # structured-output model wrote its full hedge/reasoning into a single
    # required field (once a top-level string field with every sibling left
    # empty/zero, once as the sole item of a required array) instead of
    # extracting the one short value the field asked for -- a syntactically
    # valid but semantically garbage JSON shape that the old validator
    # (type/required-presence checks only) accepted outright as final. The
    # repair-retry loop must now reject it and recover the real short value.
    call_count = {"n": 0}
    dump = (
        "I have to be honest: I cannot determine the exact light name from "
        "the evidence gathered so far.\n"
        "The evidence I found describes several candidate lighthouses but "
        "none of them exactly match every stated condition in the question, "
        "and the source table I was able to open only shows a partial "
        "listing rather than the complete set of entries the question "
        "asks about.\n"
        "Please check the original NTSB/USCG source directly for the "
        "authoritative figure, since I was not able to verify it myself."
    )

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _text_chat_result(json.dumps({"light_name": dump, "height_ft": 0}))
        return _text_chat_result(json.dumps({"light_name": "Ocracoke Light", "height_ft": 75}))

    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    schema = {
        "type": "object",
        "properties": {"light_name": {"type": "string"}, "height_ft": {"type": "integer"}},
        "required": ["light_name", "height_ft"],
    }
    state = agent.RunState()
    store = agent.EvidenceStore()

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "the answer text", state
    )

    assert structured == {"light_name": "Ocracoke Light", "height_ft": 75}


async def test_structured_output_repair_retry_rejects_all_default_object(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed pattern, three independent instances (our own local run
    # plus two from the current champion's real production history): every
    # required field simultaneously left at its type's empty/zero default --
    # a technically valid but totally uninformative object that reads as a
    # full give-up dressed as a real answer. The repair retry must reject
    # this shape and recover the real values.
    call_count = {"n": 0}

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _text_chat_result(
                json.dumps(
                    {
                        "lowest_height_feet": 0,
                        "lowest_height_aid_name": "",
                        "new_private_research_aid_name": "",
                    }
                )
            )
        return _text_chat_result(
            json.dumps(
                {
                    "lowest_height_feet": 49,
                    "lowest_height_aid_name": "Mile Rocks Light",
                    "new_private_research_aid_name": "OPT Research Buoy",
                }
            )
        )

    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    schema = {
        "type": "object",
        "properties": {
            "lowest_height_feet": {"type": "integer"},
            "lowest_height_aid_name": {"type": "string"},
            "new_private_research_aid_name": {"type": "string"},
        },
        "required": ["lowest_height_feet", "lowest_height_aid_name", "new_private_research_aid_name"],
    }
    state = agent.RunState()
    store = agent.EvidenceStore()

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "the answer text", state
    )

    assert structured == {
        "lowest_height_feet": 49,
        "lowest_height_aid_name": "Mile Rocks Light",
        "new_private_research_aid_name": "OPT Research Buoy",
    }


async def test_structured_output_allows_legitimate_all_false_booleans(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The all-defaults guard above must not fire on a schema that is
    # entirely booleans -- two required claims both correctly being false
    # is an ordinary, common real answer, not a give-up.
    async def fake_llm_chat(**__: object) -> LlmChatResult:
        return _text_chat_result(json.dumps({"claim_a_holds": False, "claim_b_holds": False}))

    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    schema = {
        "type": "object",
        "properties": {"claim_a_holds": {"type": "boolean"}, "claim_b_holds": {"type": "boolean"}},
        "required": ["claim_a_holds", "claim_b_holds"],
    }
    state = agent.RunState()
    store = agent.EvidenceStore()

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "the answer text", state
    )

    assert structured == {"claim_a_holds": False, "claim_b_holds": False}


async def test_structured_output_falls_back_to_placeholders_when_extract_also_fails(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def always_invalid_llm_chat(**__: object) -> LlmChatResult:
        return _text_chat_result("not valid json")

    monkeypatch.setattr(agent, "llm_chat", always_invalid_llm_chat)

    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    state = agent.RunState()
    store = agent.EvidenceStore()

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "fallback text", state
    )

    assert structured == {"count": 0}


async def test_structured_output_placeholder_does_not_duplicate_text_across_string_fields(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed champion loss on this exact rescue path: a schema with
    # two distinct string fields (a title, a short date span) both got the
    # entire raw text_answer dumped into them verbatim by the old
    # placeholder fallback, and the judge read the duplicated wall of text
    # as "completely hallucinated/garbage" and scored the whole answer 0.0
    # -- even though a real, correct value was sitting in the loop's own
    # prose the whole time. Only the first string field should receive the
    # raw text; the rest must stay empty rather than repeat it.
    async def always_invalid_llm_chat(**__: object) -> LlmChatResult:
        return _text_chat_result("not valid json")

    monkeypatch.setattr(agent, "llm_chat", always_invalid_llm_chat)

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "release_span": {"type": "string"},
            "pool_count": {"type": "integer"},
        },
        "required": ["title", "release_span", "pool_count"],
    }
    state = agent.RunState()
    store = agent.EvidenceStore()

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "the real answer text", state
    )

    assert structured["title"] == "the real answer text"
    assert structured["release_span"] == ""
    assert structured["pool_count"] == 0


async def test_structured_output_placeholder_does_not_empty_a_real_array_answer(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed champion loss on the same rescue path: a required array
    # field (a list of navigation aids) shipped as a bare `[]` even though
    # the model's own text answer already stated the exact three items --
    # an avoidable, silent zero on a question it had actually solved.
    async def always_invalid_llm_chat(**__: object) -> LlmChatResult:
        return _text_chat_result("not valid json")

    monkeypatch.setattr(agent, "llm_chat", always_invalid_llm_chat)

    schema = {
        "type": "object",
        "properties": {"aids": {"type": "array", "items": {"type": "string"}}},
        "required": ["aids"],
    }
    state = agent.RunState()
    store = agent.EvidenceStore()

    structured = await agent._build_structured_output(
        Query(text="q", output_schema=schema), store, "1555, 1585, 1640", state
    )

    assert structured["aids"] == ["1555, 1585, 1640"]


def _tooling_info_result(allowed: dict[str, list[str]]) -> ToolCallResponse[dict[str, Any]]:
    return ToolCallResponse(
        receipt_id="tooling-info-receipt",
        response={"allowed_llm_provider_models": allowed},
        results=(),
        result_policy="log_only",
        cost_usd=0.0,
        usage=None,
        budget=_budget(),
    )


def test_default_waterfall_tries_openrouter_before_chutes(agent: ModuleType) -> None:
    # 2026-08-20: measured across 23 real task runs on two separate days,
    # chutes/GLM-5.2-TEE succeeded on only 5 of 165 LLM calls (~3%) before
    # falling through to openrouter on a 429 capacity error. openrouter must
    # stay first so a real call doesn't pay for a doomed chutes attempt.
    assert agent.DEFAULT_MODEL_WATERFALL[0] == ("openrouter", "deepseek/deepseek-v3.2")
    assert agent.DEFAULT_MODEL_WATERFALL[1] == ("chutes", "zai-org/GLM-5.2-TEE")


async def test_model_waterfall_drops_pairs_the_platform_no_longer_allows(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_tooling_info(**__: object) -> ToolCallResponse[dict[str, Any]]:
        # chutes no longer lists our preferred model; openrouter still does.
        return _tooling_info_result(
            {
                "chutes": ["some-other-model"],
                "openrouter": ["deepseek/deepseek-v3.2"],
            }
        )

    monkeypatch.setattr(agent, "tooling_info", fake_tooling_info)

    waterfall = await agent._resolve_model_waterfall()

    assert waterfall == (("openrouter", "deepseek/deepseek-v3.2"),)


async def test_model_waterfall_falls_back_to_default_when_tooling_info_fails(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_tooling_info(**__: object) -> ToolCallResponse[dict[str, Any]]:
        raise RuntimeError("tooling_info unavailable")

    monkeypatch.setattr(agent, "tooling_info", failing_tooling_info)

    waterfall = await agent._resolve_model_waterfall()

    assert waterfall == agent.DEFAULT_MODEL_WATERFALL


async def test_query_uses_resolved_waterfall_for_loop_turns(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_tooling_info(**__: object) -> ToolCallResponse[dict[str, Any]]:
        return _tooling_info_result({"openrouter": ["deepseek/deepseek-v3.2"]})

    seen_providers: list[str] = []

    async def fake_llm_chat(**kwargs: object) -> LlmChatResult:
        seen_providers.append(str(kwargs["provider"]))
        return _text_chat_result("A short answer with no evidence needed.")

    monkeypatch.setattr(agent, "tooling_info", fake_tooling_info)
    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    async def failing_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        raise RuntimeError("no search needed for this test")

    monkeypatch.setattr(agent, "search_web", failing_search_web)

    await agent.query(Query(text="A simple question"))

    assert seen_providers
    assert all(provider == "openrouter" for provider in seen_providers)


def test_compute_tool_does_exact_decimal_arithmetic_without_precision_loss(agent: ModuleType) -> None:
    # The exact bug this tool exists to prevent: LLM-native arithmetic
    # truncated 8.31446261815324 down to 8.314462618 on a real local-eval run.
    code = "result = str(Decimal('6.02214076e23') * Decimal('1.380649e-23'))"

    output = json.loads(agent._tool_compute(code))

    assert output["result"] == "8.31446261815324"


def test_compute_tool_reports_errors_without_raising(agent: ModuleType) -> None:
    output = json.loads(agent._tool_compute("result = 1 / 0"))

    assert "error" in output


def test_compute_tool_rejects_missing_code() -> None:
    module = _load_agent()
    output = json.loads(module._tool_compute(""))

    assert "error" in output


async def test_loop_uses_compute_tool_for_arithmetic(agent: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_chat_result(
                "compute", {"code": "result = str(Decimal('2.5') * Decimal('4'))"}
            )
        return _text_chat_result("The product is 10.0, computed exactly, no evidence needed.")

    monkeypatch.setattr(agent, "llm_chat", fake_llm_chat)

    result = await agent.query(Query(text="What is 2.5 times 4?"))

    assert "10.0" in result.text


def test_build_citations_falls_back_to_topical_relevance_not_position(agent: ModuleType) -> None:
    # Real diagnosed loss: a "give only the group names" instruction made
    # the model drop its [N] markers from an otherwise perfectly correct
    # final answer ("Ro-Ro\nContainers (Lo-Lo)"). Blind positional guesses
    # (first-N, then most-recent-N) both failed live -- research sometimes
    # finds the right source early and keeps exploring past it, sometimes
    # late, so position alone is not a reliable signal. Content relevance
    # against the question/answer is: items that actually discuss the
    # cargo groups in question must outrank ones that don't, regardless of
    # when they were gathered.
    store = agent.EvidenceStore()
    store.add(
        receipt_id="r",
        result_id="irrelevant-early",
        url="https://example.com/0",
        title="Weather forecast archive",
        note="temperature rainfall wind speed daily forecast history",
    )
    store.add(
        receipt_id="r",
        result_id="relevant-middle",
        url="https://example.com/1",
        title="Port freight annual statistics 2024 overview",
        note="Cargo Group 2024 tonnage: Ro-Ro 99.3, Containers (Lo-Lo) 60.7, Liquid Bulk 163.1",
    )
    store.add(
        receipt_id="r",
        result_id="irrelevant-late",
        url="https://example.com/2",
        title="Recipe collection index",
        note="baking cookware ingredient measurement conversion chart",
    )

    question = 'The UK Department for Transport publishes "Port freight annual statistics" annually.'
    text, citations = agent._build_citations(question, "Ro-Ro\nContainers (Lo-Lo)", store)

    assert citations is not None
    result_ids = [c.result_id for c in citations]
    assert "relevant-middle" in result_ids
    assert "irrelevant-early" not in result_ids
    assert "irrelevant-late" not in result_ids
    assert text == "Ro-Ro\nContainers (Lo-Lo)"


def test_build_citations_slices_large_notes_instead_of_materializing_in_full(
    agent: ModuleType,
) -> None:
    # Real production bug: a whole response was rejected with "response
    # citations exceed 120000 materialized source-text characters" because
    # citations with no slice materialize the full note server-side, and
    # MAX_PAGE_CHARS grew large enough for a handful of citations to blow
    # past that cap.
    store = agent.EvidenceStore()
    long_note = "x" * (agent.MAX_CITATION_SLICE_CHARS + 500)
    store.add(receipt_id="r", result_id="big-note", url="https://example.com", title="T", note=long_note)

    text, citations = agent._build_citations("q", "Answer citing [[0]].", store)

    assert citations is not None
    assert len(citations[0].slices) == 1
    assert citations[0].slices[0].start == 0
    assert citations[0].slices[0].end == agent.MAX_CITATION_SLICE_CHARS
    assert text == "Answer citing [[1]]."


def test_build_citations_uses_relevant_spans_instead_of_page_head(agent: ModuleType) -> None:
    # Real diagnosed loss: citations always sliced note[0:cap], so a long
    # PDF's table-of-contents (sitting at the head) got cited instead of the
    # actual data table the model read further in, even though
    # _densest_windows had already located it. Citations must follow the
    # same relevant span, not fall back to the literal head.
    store = agent.EvidenceStore()
    head = "table of contents boilerplate " * 100
    real_data = "the actual table data goes here "
    tail = "y" * 5000
    note = head + real_data + tail
    real_start = len(head)
    real_end = real_start + len(real_data)
    store.add(
        receipt_id="r",
        result_id="pdf-note",
        url="https://example.com/report.pdf",
        title="T",
        note=note,
        relevant_spans=((real_start, real_end),),
    )

    text, citations = agent._build_citations("q", "Answer citing [[0]].", store)

    assert citations is not None
    assert len(citations[0].slices) == 1
    assert citations[0].slices[0].start == real_start
    assert citations[0].slices[0].end == real_end
    assert text == "Answer citing [[1]]."


def test_tool_note_evidence_records_span_for_verbatim_quote(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    note = "some preamble text " * 20 + "the shared date is 7th March 1916" + " trailing text" * 20
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note=note)

    result = json.loads(agent._tool_note_evidence(0, "the shared date is 7th March 1916", store))

    assert result["noted"] is True
    item = store.get(0)
    assert item is not None
    assert item.retained_spans
    pos = note.find("the shared date is 7th March 1916")
    assert any(start <= pos < end for start, end in item.retained_spans)


def test_tool_note_evidence_rejects_quote_not_present(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note="actual content here")

    result = json.loads(agent._tool_note_evidence(0, "text that was never in the source", store))

    assert result["noted"] is False
    item = store.get(0)
    assert item is not None
    assert item.retained_spans == ()


def test_tool_page_grep_finds_match_with_context_without_refetching(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    note = "padding text " * 50 + "the exact figure was 42.7 percent" + " more padding" * 50
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note=note)

    result = json.loads(agent._tool_page_grep(0, "exact figure was 42.7", store))

    assert result["matches"]
    assert "42.7 percent" in result["matches"][0]["context"]


def test_tool_page_grep_is_case_insensitive_and_caps_matches(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    note = " ".join(["Needle occurs here"] * 10)
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note=note)

    result = json.loads(agent._tool_page_grep(0, "needle", store))

    assert len(result["matches"]) == agent._PAGE_GREP_MAX_MATCHES


def test_tool_page_grep_reports_no_match_without_erroring(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note="unrelated content")

    result = json.loads(agent._tool_page_grep(0, "nowhere to be found", store))

    assert result["matches"] == []


def test_tool_page_grep_rejects_missing_index_or_pattern(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note="some content")

    missing_index = json.loads(agent._tool_page_grep("not-an-int", "content", store))
    assert "error" in missing_index

    out_of_range = json.loads(agent._tool_page_grep(5, "content", store))
    assert "error" in out_of_range

    empty_pattern = json.loads(agent._tool_page_grep(0, "  ", store))
    assert "error" in empty_pattern


async def test_execute_tool_call_dispatches_page_grep(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    store.add(receipt_id="r", result_id="r-1", url="https://example.com/a", title="T", note="the answer is 1916")
    call = LlmMessageToolCall(
        id="call-1", type="function", name="page_grep", arguments=json.dumps({"evidence_index": 0, "pattern": "1916"})
    )

    output = json.loads(await agent._execute_tool_call(call, store, agent.RunState(), set(), ()))

    assert output["matches"]


def test_loop_tools_include_page_grep(agent: ModuleType) -> None:
    names = {tool["function"]["name"] for tool in agent.LOOP_TOOLS}
    assert "page_grep" in names


def test_system_prompt_commits_instead_of_hedging(agent: ModuleType) -> None:
    prompt = agent._LOOP_SYSTEM_PROMPT
    assert "say so plainly instead of guessing" not in prompt
    assert "COMMIT TO YOUR BEST-SUPPORTED CANDIDATE" in prompt
    assert "page_grep" in prompt


def test_build_citations_prefers_retained_spans_over_relevant_spans(agent: ModuleType) -> None:
    # The note_evidence tool's model-verified quote location should win
    # over the generic keyword-density guess when both exist for the same
    # evidence item -- it's direct proof of what backs the claim, not an
    # inference about what's probably relevant.
    store = agent.EvidenceStore()
    guessed_relevant = "guessed relevant but wrong region " * 60
    real_proof = "the real proof the model actually quoted "
    note = guessed_relevant + real_proof + ("y" * 5000)
    guess_start, guess_end = 0, len(guessed_relevant)
    proof_start = len(guessed_relevant)
    proof_end = proof_start + len(real_proof)
    store.add(
        receipt_id="r",
        result_id="r-1",
        url="https://example.com/a",
        title="T",
        note=note,
        relevant_spans=((guess_start, guess_end),),
    )
    store.add_retained_span(0, (proof_start, proof_end))

    _text, citations = agent._build_citations("q", "Answer citing [[0]].", store)

    assert citations is not None
    slice_ = citations[0].slices[0]
    assert slice_.start == proof_start


async def test_tool_search_locates_relevant_span_in_oversized_note(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss: a search provider returned a full page's worth of
    # `note` (not a short snippet) -- a document's table-of-contents sat near
    # the top and the actual, question-quoted table data sat far below.
    # Search-sourced evidence never went through _densest_windows the way
    # open_page evidence did, so its citation still fell back to slicing the
    # literal head. The anchor-matched region must be found here too.
    quoted_title = "Prison facility capacity, custody population, and percent of capacity, by jurisdiction"
    head = "table of contents boilerplate " * 100
    real_data = f"{quoted_title} data: Alabama 170.7%"
    tail = "y" * 5000
    note = head + real_data + tail
    results = [{"index": 0, "result_id": "r-1", "url": "https://example.com/a", "note": note, "title": "T"}]

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    monkeypatch.setattr(agent, "search_web", fake_search_web)

    store = agent.EvidenceStore()
    state = agent.RunState()
    question = f'Look at the table titled "{quoted_title}".'
    keywords = agent._keywords_from(question)
    anchors = agent._anchors_from(question)

    await agent._tool_search("query", store, state, keywords, anchors)

    item = store.get(0)
    assert item is not None
    assert item.relevant_spans
    real_start = note.find(quoted_title)
    assert any(start <= real_start < end for start, end in item.relevant_spans)


async def test_tool_open_page_dedupes_repeat_url_and_keeps_richer_content(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss: the loop fetched the same PDF four times under
    # URL variants like "...p22st.pdf" and "...p22st.pdf#page=30" (a
    # #page= fragment is a client-side viewer hint, not something the fetch
    # provider processes), each creating a fresh evidence index. The model
    # ended up citing a shallow early fetch instead of the later one that
    # actually contained the real answer -- a numerically exact answer lost
    # purely because the citation pointed at the wrong instance of the
    # right document. Repeat opens of the same URL (fragment or not) must
    # land on the same evidence index, and a richer re-fetch must replace a
    # thinner one there.
    fetch_calls = {"n": 0}

    async def fake_fetch_page(*_: object, **__: object) -> ToolCallResponse[FetchPageResponse]:
        fetch_calls["n"] += 1
        # Content clears THIN_FETCH_RETRY_CHARS on the first provider attempt
        # so the provider-fallback loop (a separate, later fix) doesn't fire
        # here -- this test is specifically about the dedupe/richer-wins
        # behavior, not the fallback machinery.
        if fetch_calls["n"] == 1:
            note = "table of contents boilerplate " * 100
        else:
            note = "table of contents boilerplate " * 100 + "the real answer: 42"
        return _fetch_response(
            {
                "index": 0,
                "result_id": f"r-fetch-{fetch_calls['n']}",
                "url": "https://example.com/report.pdf",
                "note": note,
                "title": "Report",
            }
        )

    monkeypatch.setattr(agent, "fetch_page", fake_fetch_page)

    store = agent.EvidenceStore()
    state = agent.RunState()

    first = json.loads(await agent._tool_open_page("https://example.com/report.pdf", store, state, set()))
    second = json.loads(
        await agent._tool_open_page("https://example.com/report.pdf#page=30", store, state, set())
    )

    assert first["index"] == second["index"]
    assert len(store.items) == 1
    item = store.get(first["index"])
    assert item is not None
    assert item.result_id == "r-fetch-2"
    assert "the real answer" in (item.note or "")


async def test_tool_open_page_falls_back_to_next_provider_on_thin_fetch(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed champion loss (traced via public monitoring data): the
    # exact same URL, fetched by the exact same script under different
    # validators, came back at ~1900 chars in two runs and ~11300 chars in
    # two others -- a provider hiccup silently truncated a statutory
    # instrument, and with only one hardcoded provider and no fallback, the
    # script had no way to recover and shipped an empty answer in the runs
    # that hit the thin fetch. open_page must retry a suspiciously thin
    # fetch on the next provider rather than accepting it as final.
    calls: list[str] = []

    async def fake_fetch_page(*_: object, provider: str, **__: object) -> ToolCallResponse[FetchPageResponse]:
        calls.append(provider)
        if provider == "parallel":
            note = "thin truncated content"
        else:
            note = "the full statutory instrument text " * 200
        return _fetch_response(
            {
                "index": 0,
                "result_id": f"r-{provider}",
                "url": "https://example.com/instrument",
                "note": note,
                "title": "Instrument",
            }
        )

    monkeypatch.setattr(agent, "fetch_page", fake_fetch_page)

    store = agent.EvidenceStore()
    state = agent.RunState()

    result = json.loads(await agent._tool_open_page("https://example.com/instrument", store, state, set()))

    assert calls[0] == "parallel"
    assert len(calls) > 1
    item = store.get(result["index"])
    assert item is not None
    assert item.result_id != "r-parallel"
    assert "full statutory instrument" in (item.note or "")


def test_looks_like_binary_detects_real_xlsx_garbage_not_real_text(agent: ModuleType) -> None:
    # Real diagnosed loss: a question's source was a .xlsx spreadsheet; the
    # fetch tool returned the raw ZIP-archive bytes as text instead of
    # extracted cell data ("PK" is the ZIP magic number -- xlsx/docx/pptx
    # are all ZIP containers). The model had nothing readable and, despite
    # an explicit prompt rule against it, filled fields with -1/empty
    # placeholders.
    xlsx_garbage = (
        "PK!  [Content_Types].xml (ʖMo0\"_+b`wjEw+m+<~CJCKĞ}23m.P9[Q9dIe{"
        "|аIX)P- M//&[Xhkwαn,g.A6,R,k^;K`i@N~\\4xG@#+~^kU|m;ޡi 6U`ң"
        "`;&( ŽK7uaвtAzebJ.Xw cLc ^F|@ ^3KY<ѻ^!9mѩ.0:qO) c|&_Ќ"
    )
    real_markdown = "# Some Report\n\nThis is a real page with readable content about cargo groups."

    assert agent._looks_like_binary(xlsx_garbage) is True
    assert agent._looks_like_binary(real_markdown) is False


async def test_tool_open_page_rejects_binary_content_instead_of_citing_it(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_fetch_page(*_: object, **__: object) -> ToolCallResponse[FetchPageResponse]:
        return _fetch_response(
            {
                "index": 0,
                "result_id": "r-xlsx",
                "url": "https://example.com/data.xlsx",
                "note": "PK!  [Content_Types].xml (ʖMo0 garbled binary zip bytes here not real text",
                "title": "Data",
            }
        )

    monkeypatch.setattr(agent, "fetch_page", fake_fetch_page)

    store = agent.EvidenceStore()
    state = agent.RunState()

    result = json.loads(await agent._tool_open_page("https://example.com/data.xlsx", store, state, set()))

    assert "error" in result
    assert not store.items


async def test_tool_open_page_rejects_puzzle_spam_instead_of_citing_it(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real diagnosed loss (BrowseComp benchmark): a multi-clue, riddle-style
    # question read enough like a crossword clue to a search engine that a
    # crossword-solver aggregator site came back as a top result and got
    # cited as if it were about the actual subject.
    async def fake_fetch_page(*_: object, **__: object) -> ToolCallResponse[FetchPageResponse]:
        return _fetch_response(
            {
                "index": 0,
                "result_id": "r-puzzle",
                "url": "https://example.com/clues",
                "note": 'The Crossword Solver found 30 answers to "some riddle-like clue text"',
                "title": "Crossword Clue",
            }
        )

    monkeypatch.setattr(agent, "fetch_page", fake_fetch_page)

    store = agent.EvidenceStore()
    state = agent.RunState()

    result = json.loads(await agent._tool_open_page("https://example.com/clues", store, state, set()))

    assert "error" in result
    assert not store.items


async def test_tool_search_drops_puzzle_spam_results_before_showing_the_model(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        {
            "index": 0,
            "result_id": "r-spam",
            "url": "https://example.com/spam",
            "note": "Crossword Clue Answers: some riddle-like clue text",
            "title": "Spam",
        },
        {
            "index": 1,
            "result_id": "r-real",
            "url": "https://example.com/real",
            "note": "This is a real page about the actual subject.",
            "title": "Real",
        },
    ]

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _search_response(results)

    monkeypatch.setattr(agent, "search_web", fake_search_web)

    store = agent.EvidenceStore()
    state = agent.RunState()

    result = json.loads(await agent._tool_search("query", store, state, set()))

    urls = [item["url"] for item in result["results"]]
    assert "https://example.com/spam" not in urls
    assert "https://example.com/real" in urls
    assert len(store.items) == 1


def test_build_citations_stops_before_exceeding_total_char_budget(agent: ModuleType) -> None:
    store = agent.EvidenceStore()
    long_note = "x" * agent.MAX_CITATION_SLICE_CHARS
    # enough large-note items that citing all of them would exceed the total
    # budget, forcing the loop to stop early rather than overflow it
    count = (agent.MAX_TOTAL_CITATION_CHARS // agent.MAX_CITATION_SLICE_CHARS) + 5
    text_parts = []
    for i in range(count):
        store.add(receipt_id="r", result_id=f"item-{i}", url="https://example.com", title="T", note=long_note)
        text_parts.append(f"[{i}]")

    _text, citations = agent._build_citations("q", " ".join(text_parts), store)

    assert citations is not None
    total = len(citations) * agent.MAX_CITATION_SLICE_CHARS
    assert total <= agent.MAX_TOTAL_CITATION_CHARS
    assert len(citations) < count


def test_evidence_block_carries_far_more_than_the_old_600_char_cap(agent: ModuleType) -> None:
    # Real vs-champion loss: champion used ~9x more tokens (327K vs 37K) on
    # an exhaustive two-list comparison because the rescue ladder/audit/
    # structured-output steps only ever saw a 600-char slice of each
    # evidence item, no matter how much open_page actually captured.
    store = agent.EvidenceStore()
    long_note = "STOCK-" + "y" * 5000
    store.add(receipt_id="r", result_id="item-0", url="https://example.com", title="T", note=long_note)

    block = agent._evidence_block(store)

    assert len(block) > 600
    assert "STOCK-" in block
    assert block.count("y") == agent.EVIDENCE_BLOCK_SNIPPET_CHARS - len("STOCK-")


def test_densest_windows_drops_boilerplate_head_for_denser_content(agent: ModuleType) -> None:
    # Real diagnosed loss: unconditionally keeping the literal page head
    # cited nothing but site-navigation boilerplate on a page whose real,
    # keyword-relevant content started further down.
    nav_boilerplate = "Home Explore-collections Research-tools Help-guidance " * 400
    real_content = "vessel designated wreck sinking coordinates " * 400
    filler = "lorem ipsum unrelated padding text here " * 400
    content = nav_boilerplate + filler + real_content + filler

    focused, spans = agent._densest_windows(content, {"vessel", "designated", "wreck", "sinking"})

    assert "vessel" in focused
    assert "designated" in focused
    assert "Explore-collections" not in focused
    assert spans
    assert all(content[start:end] for start, end in spans)


def test_anchor_match_survives_platform_markdown_bold_wrapping(agent: ModuleType) -> None:
    # Real diagnosed loss: the platform's PDF-to-markdown conversion wraps
    # each clause of a table heading in its own **bold** span and inserts a
    # space before the comma between them, e.g. the real fetched text for
    # a table titled "Prison facility capacity, custody population, and
    # percent of capacity, by jurisdiction" -- so a literal contiguous
    # substring match against the raw question-quoted phrase never fired
    # even though every word was present.
    anchor = "prison facility capacity, custody population, and percent of capacity, by jurisdiction"
    bold_wrapped = (
        "**Prison facility capacity** , **custody population** , "
        "**and percent of capacity** , **by jurisdiction** , **December 31** , **2022**"
    )
    unrelated = "lorem ipsum unrelated padding text " * 50

    score_with_anchor = agent._chunk_score(bold_wrapped, set(), (anchor,))
    score_without = agent._chunk_score(unrelated, set(), (anchor,))

    assert score_with_anchor >= agent._ANCHOR_SCORE_BONUS
    assert score_without < agent._ANCHOR_SCORE_BONUS

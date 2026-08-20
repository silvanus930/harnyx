"""Harnyx SN67 miner agent.

Agentic tool-calling research loop: the model itself drives `search` /
`open_page` calls across several turns, reading full results in context and
deciding when it has enough evidence, instead of a fixed one-shot
search-then-synthesize pipeline. It finishes by responding with plain text
(no further tool calls); if it runs out of turns or time before doing so, one
forced no-tools completion asks it to answer from whatever was already
gathered.

If the loop still doesn't produce a usable answer, a rescue ladder tries
progressively cheaper fallbacks in an order that always prefers a cited,
evidence-grounded answer over an uncited but fluent one: rewrite from the
evidence digest -> deterministic zero-LLM answer from evidence -> last-resort
answer from the model's own knowledge, clearly hedged. Every external call is
wrapped so a provider outage degrades the answer instead of raising -- the
entrypoint must always return a valid Response.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from harnyx_miner_sdk.api import fetch_page, llm_chat, search_web, tooling_info
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, CitationSlice, Query, Response
from harnyx_miner_sdk.safe_exec import safe_exec

# The documented sandbox hard-kill is 300s (confirmed against
# ENTRYPOINT_TIMEOUT_SECONDS's default in harnyx_commons.sandbox.timeout).
# WALL_BUDGET_S is where we stop opening new tool-loop turns; HARD_DEADLINE_S
# is where we stop attempting any further network calls at all and return
# whatever we have. Overshooting the real kill returns nothing (a hard
# zero), so both stay well under it.
# 2026-08-20: raised again (200->250 / 260->280) after reading the current
# champion's own source -- it runs WALL_BUDGET_S=266 against the same 300s
# kill (only ~34s reserved) and survives fine in real production, while 3 of
# 5 items in a live BrowseComp sample were already hitting our old 200s
# ceiling before they'd genuinely exhausted their search strategies. This
# still keeps more margin than champion (20s here vs their ~34s) since our
# rescue ladder + audit pass + structured-output step run additional LLM
# calls of their own after the loop concludes.
WALL_BUDGET_S = 250.0
HARD_DEADLINE_S = 280.0

# 2026-08-18: a 14-task sample showed 6/14 failures (43%) were the agent
# giving up after search came back empty, rather than a wrong-but-attempted
# answer -- by far the biggest error bucket. Raised turns/budget to give the
# loop real room to try multiple query strategies before conceding, and
# widened the provider fallback (matching what the strongest real competitor
# code does) instead of stopping after two. Raised again after a diagnosed
# loss needing 4 separate quarterly documents -- the agent stopped after
# gathering 2 of them.
# 2026-08-20: raised again (10->14) after reading champion's source directly
# -- it runs MAX_TURNS=15 with a comment noting 13 was "the most
# turn-starved in the class". A live BrowseComp sample showed items hitting
# our old 10-turn cap in as little as 75s, far under WALL_BUDGET_S, meaning
# turns (not time) were the binding constraint on those runs -- pure
# unused headroom.
MAX_LOOP_TURNS = 14
SEARCH_PROVIDER_ORDER: tuple[str, ...] = ("parallel", "desearch", "exa", "tavily", "firecrawl")
FETCH_PROVIDER_ORDER: tuple[str, ...] = ("parallel", "desearch", "exa", "tavily", "firecrawl")
# 2026-08-18: real diagnosed champion loss (traced via public monitoring data,
# not our own run) -- the exact same URL, fetched by the exact same script
# under different validators, came back at ~1900 chars in two runs and
# ~11300 chars in two others: a single provider hiccup silently truncated a
# statutory instrument to a fragment, and with no second provider to try,
# the script had no way to notice or recover -- it shipped an empty answer
# in the runs that hit the thin fetch. `open_page` only ever used one
# provider with no fallback (unlike `search`'s five-provider waterfall), so
# this was a real, unguarded single point of failure.
THIN_FETCH_RETRY_CHARS = 3000

# 2026-08-18: traced live via LOG_LEVEL=DEBUG httpcore tracing. desearch's
# `/web` endpoint genuinely takes ~25-30s per request, and the platform
# retries on an empty `data: []` result -- confirmed by curling the exact
# same query directly: a real 200 OK, 31s, empty results. For a query that
# genuinely has no matches, no timeout length "waits it out"; every retry
# gets the same empty answer. The better response to an empty/slow search
# is a different query, not more patience, so this stays short enough to
# let the loop try alternate phrasings within WALL_BUDGET_S rather than
# sinking the whole budget into repeated attempts at one query.
SEARCH_TIMEOUT_SECONDS = 40.0
FETCH_TIMEOUT_SECONDS = 45.0
# 2026-08-18: a real vs-champion loss traced to a 9x token gap (37K vs 327K)
# on an exhaustive two-list comparison task -- the champion simply reads far
# more real content before answering. Raised alongside MAX_PAGE_CHARS /
# EVIDENCE_BLOCK_SNIPPET_CHARS below to give bigger prompts room to process.
LOOP_LLM_TIMEOUT_SECONDS = 55.0
SYNTH_LLM_TIMEOUT_SECONDS = 40.0
MIN_REMAINING_BUDGET_USD = 0.02

MAX_RESULTS_PER_SEARCH = 6
# 2026-08-18: diagnosed from two real local-eval losses. First: a 51-row
# two-year data table didn't fit in a 4000-char/2-chunk window, so the model
# never saw the full table. Second, bigger: even after that widening, a
# separate loss showed the champion using ~9x more tokens than us on an
# exhaustive-comparison task -- and the actual bottleneck turned out to be
# downstream, not here: EVIDENCE_BLOCK_SNIPPET_CHARS below was truncating
# every evidence item to 600 chars before the rescue ladder, audit pass, or
# structured-output step ever saw it, regardless of how much this captured.
MAX_PAGE_CHARS = 32_000
DENSEST_CHUNKS_PICKED = 6
MAX_CITATIONS = 12
# 2026-08-18: real production bug -- a whole response got rejected with
# "response citations exceed 120000 materialized source-text characters"
# once MAX_PAGE_CHARS grew. The platform materializes a citation's full note
# unless sliced; cap each citation's contribution and the running total well
# under the platform's 120,000-char ceiling.
# 2026-08-18: raised from 2_000 -- two independent real losses (AAIB off-type
# hours, UK Military Remains vessels) had a numerically *exact* answer but
# still lost because a 2,000-char slice was too small to show every candidate
# in a large enumeration/verification table, so the judge couldn't see that
# the comparison was actually exhaustive. Matches the real champion's own
# deliberately wide citation practice (their comments cite a measured
# citation-volume-vs-score correlation). 12 citations x 6_000 chars = 72_000,
# still comfortably under MAX_TOTAL_CITATION_CHARS and the platform's
# 120,000-char ceiling.
MAX_CITATION_SLICE_CHARS = 6_000
MAX_TOTAL_CITATION_CHARS = 100_000
# 2026-08-18: this was the real bottleneck behind the token-gap loss, not
# MAX_PAGE_CHARS -- raised from 600 (deterministic rung: 300) so the rescue
# ladder, audit pass, and structured-output step actually see most of what
# open_page/search already captured, instead of a small fixed slice of it.
# These are independent of MAX_CITATION_SLICE_CHARS above, which only bounds
# what gets materialized server-side for scoring, not what the model reads.
EVIDENCE_BLOCK_SNIPPET_CHARS = 4_000
DETERMINISTIC_SNIPPET_CHARS = 1_500
MAX_RESPONSE_CHARS = 80_000
MAX_STRUCTURED_ATTEMPTS = 2
MIN_USABLE_ANSWER_CHARS = 20

# Preferred order; ties in scoring favor lower tool cost, so this stays short
# rather than exhausting every allowed provider/model pair. This is only the
# fallback list -- _resolve_model_waterfall() checks it against tooling_info()
# at runtime and drops any pair the platform no longer allows, per the miner
# README: "Treat allowed_llm_provider_models[provider] as the runtime source
# of truth ... instead of hardcoding a fixed list."
# 2026-08-20: reordered -- measured across 23 real task runs on two separate
# days (18 local-eval tasks on the 18th, 5 BrowseComp benchmark items on the
# 20th), chutes/GLM-5.2-TEE succeeded on only 5 of 165 total LLM calls (~3%)
# before falling through to openrouter on a 429 "Infrastructure is at maximum
# capacity" error every other time. Putting the provider that fails ~97% of
# the time first means nearly every real call pays for a timeout/retry
# before reaching the one that actually answers. openrouter/deepseek-v3.2 is
# now first since it's the one that has actually been reachable.
# 2026-08-20: ai_gateway/zai-glm-5.2-fast was briefly added as a
# third-priority pair (mirroring the current champion's own LANE_A=
# openrouter, LANE_B=ai_gateway setup), but no ai_gateway credential is
# stored yet -- an unconfigured provider in the waterfall just means a
# guaranteed-fail attempt eats a retry before falling through, the same
# problem this whole waterfall was reordered to avoid. Drop it and re-add
# once `harnyx-miner-config --provider ai_gateway --api-key <key>` is set.
# 2026-08-20: three tiers -- openrouter/deepseek-v3.2 stays first (100%
# real-world success rate across every run measured this session). Second
# tier diversifies WITHIN openrouter (a second, different model on the one
# provider that's actually been reliable) rather than crossing providers
# immediately; third tier repeats that same model on chutes for one more
# cross-provider fallback. qwen/qwen3.8-27b (both provider ids confirmed in
# the miner README's allowed_llm_provider_models for openrouter AND
# chutes/Qwen3.8-27B-TEE) is a much smaller model than deepseek-v3.2 --
# every competitor script reviewed this session (champion, UID171) only
# uses a 27B-class model for narrow sub-tasks (audit/classification), never
# as the main research-loop model -- but since it only ever gets exercised
# after deepseek-v3.2 fails on openrouter, the exposure is bounded to a
# rare double-failure, same tradeoff already made for chutes/ai_gateway.
DEFAULT_MODEL_WATERFALL: tuple[tuple[str, str], ...] = (
    ("openrouter", "deepseek/deepseek-v3.2"),
    ("openrouter", "qwen/qwen3.8-27b"),
    ("chutes", "Qwen/Qwen3.8-27B-TEE"),
)
TOOLING_INFO_TIMEOUT_SECONDS = 8.0

_NO_ANSWER_STUB = "No answer could be produced for this question."
# 2026-08-18: platform announcement (Discord, mk97545) -- the judge now
# requires the double-bracket [[n]] pointer form as an exact one-based
# index into Response.citations; plain [n] is "ordinary text, not a
# citation" and does not back a claim at all, even though it costs nothing
# to write and reads as if it should.
_CITATION_INDEX_RE = re.compile(r"\[\[(\d+)\]\]")
_KEYWORD_RE = re.compile(r"[a-zA-Z0-9]{4,}")
_ANCHOR_QUOTE_RE = re.compile(r'"([^"]{8,200})"')
# 2026-08-18: real diagnosed loss -- a loop turn ended with the model
# writing out a tool call as literal prose ("<function_calls><invoke
# name=\"compute\">...") instead of a real structured tool_calls entry.
# Since message.tool_calls was empty, the loop treated that garbled XML as
# the finished answer. Reject it so the rescue ladder gets a chance to
# produce something real instead of shipping raw tool-call markup.
_TOOL_MARKUP_RE = re.compile(r"<(?:function_calls|invoke|parameter)\b", re.IGNORECASE)
# 2026-08-18: real diagnosed loss, seen twice in real transcripts on
# questions naming several specific documents/periods -- the model opens
# only some of them, then self-admits the gap in its own final answer
# ("the Q3 and Q4 documents were not opened, so no evidence exists for
# those quarters") instead of going back for the rest, even with 100+
# unused seconds of budget and turns to spare. The VERIFYING prompt
# section already tells it not to do this; catching its own admission and
# forcing another turn is cheaper and more reliable than hoping the
# instruction alone holds every time.
# 2026-08-18: broadened after a DeepSearchQA benchmark run -- 3 of 4 losses
# there were the same shape (giving up mid-analysis on a genuinely
# multi-step task) but with different self-admission wording than the
# multi-document case above: "cannot complete the remaining steps due to
# lack of airport data", "the evidence is insufficient", "I can provide a
# partial answer but cannot complete the full analysis". Same fix, wider
# net -- these are all a model narrating its own incompleteness instead of
# continuing to work the problem.
_INCOMPLETE_COVERAGE_RE = re.compile(
    r"\b(?:was|were) not (?:opened|examined|checked|found|available|provided)\b"
    r"|\bno evidence (?:exists|was found|is available) for\b"
    r"|\bnot (?:provided|included) in the evidence\b"
    r"|\bcan(?:not|'t) (?:complete|determine|finish|identify|name|confirm|conclude|"
    r"pinpoint|proceed|provide (?:a |the )?full)\b"
    r"|\bcan(?:not|'t) confidently \w+\b"
    r"|\bcan(?:not|'t) provide (?:a |the )?(?:confident|definitive|complete|full) answer\b"
    r"|\b(?:evidence|data) (?:is|was|are|were) insufficient\b"
    r"|\bunable to (?:complete|determine|find|identify|confirm|name)\b"
    r"|\ba? ?partial answer\b"
    r"|\bcan(?:not|'t)? ?(?:only )?partially answer\b"
    # 2026-08-18: real diagnosed WebWalkerQA (Multi-Source) loss -- this suite
    # draws bilingual questions, and a Chinese-language answer self-admitted
    # the same "found half, gave up on the rest" gap ("但具体的历史意义内容在
    # 现有证据中未完整显示" / "答案内容不完整" / "无法提供完整的对比分析") that
    # the English patterns above already catch and retry for -- but every
    # pattern above is English-only, so this one shipped as final with zero
    # chance of a retry.
    r"|未(?:能)?完整"
    r"|不完整"
    r"|无法(?:确定|确认|提供完整|完整)"
    r"|(?:证据|信息|数据)(?:不足|不充分)"
    r"|未找到"
    r"|建议(?:查阅|参考|访问).{0,40}(?:获取|了解)",
    re.IGNORECASE,
)
# 2026-08-18: real diagnosed BrowseComp losses (multi-clue riddle questions,
# 5/5 zero) -- the model's actual give-up phrasing was "I cannot confidently
# identify..." / "...cannot confidently name it" / "cannot provide a
# confident answer", none of which the verb list above matched (it only
# covered "cannot determine/complete/finish", not "identify"/"name", and not
# an adverb wedged between "cannot" and the verb). These slipped through as a
# shipped final answer instead of triggering the existing retry.
# 2026-08-18: real diagnosed loss -- a 48-page, 144k-char PDF had a "List of
# tables" front-matter section that repeats every table's title verbatim
# ("Table 21. Prison facility capacity, custody population..."), which
# out-scored the actual Table 21 data region on raw keyword-term-frequency,
# because generic terms like "jurisdiction"/"population"/"percent" recur in
# dozens of unrelated tables/footnotes throughout the document. A literal
# quoted phrase from the question (questions in this task distribution
# routinely quote the exact document/table title) is a far more precise
# anchor than word-frequency -- any chunk containing an exact match gets a
# bonus large enough to guarantee inclusion over generic density.
_ANCHOR_SCORE_BONUS = 1_000

_LOOP_SYSTEM_PROMPT = (
    "You are a careful research assistant with five tools: `search`, "
    "`open_page`, `page_grep`, `compute`, and `note_evidence`. Use "
    "search/open_page as needed to gather evidence, reading full results "
    "before deciding your next step, and cross-check evidence across "
    "sources before committing to an answer. If you already opened a long "
    "page and need one more specific detail from it (a number, a name, a "
    "date), call `page_grep(evidence_index, pattern)` to search within "
    "what you already fetched instead of calling open_page on the same "
    "URL again -- it costs nothing and often surfaces a detail a "
    "truncated view missed.\n\n"
    "SEARCHING: prefer short, keyword-style queries over full sentences or "
    "quoted phrases -- an overly specific query often returns nothing even "
    "when the source exists. If a search comes back empty, do not give up "
    "or repeat the same query -- broaden it (drop quotes and exact dates, "
    "search for the source organization plus document type instead of the "
    "exact title) and, when you are unsure which phrasing will work, fire "
    "2-3 differently-phrased queries in the same turn rather than one at a "
    "time. Giving up after a single failed search is premature -- keep "
    "trying meaningfully different queries while time and evidence needs "
    "allow it. Independent facts you need (each candidate's date, each "
    "entity's figure) should also be requested as several tool calls in "
    "the SAME turn -- they run concurrently, so checking 5 candidates "
    "costs one turn, not five. When a question describes its target "
    "indirectly -- through several distinguishing characteristics or "
    "clues rather than naming it directly -- search on the single most "
    "distinctive, unusual detail first (a rare name, an exact number, a "
    "specific place or date), not the full question text verbatim; a "
    "long descriptive query often gets matched by unrelated trivia or "
    "puzzle sites rather than finding the actual subject. Treat any "
    "result that reads like a crossword, trivia-quiz, or word-puzzle "
    "listing as noise, not evidence, no matter how well its wording "
    "seems to match the question. Once you have a candidate answer for "
    "a question built from several clues, check it against every clue "
    "before committing -- a candidate that fits only the most memorable "
    "clue but contradicts another is wrong even if it seemed plausible "
    "at first.\n\n"
    "VERIFYING: when a question asks you to compare, enumerate, or check "
    "every entry in a list or table (e.g. \"list every stock that changed "
    "between the two reports\", \"compare the complete rosters\"), fetch and "
    "read the full list on both sides before answering -- a partial read of "
    "a long table silently drops entries and produces a wrong comparison "
    "even when every entry you did see was read correctly. If the question "
    "names or implies several specific documents, dates, or periods (e.g. "
    "four quarterly reports, three editions), open every one of them "
    "individually before answering -- do not stop after gathering only "
    "some of what was named and treat that as enough. "
    "When a question requires checking multiple candidates or conditions "
    "(e.g. \"which of these N items is the one that...\"), check every "
    "candidate against the evidence before answering -- do not stop at the "
    "first one that seems plausible. When a claim concerns what is "
    "current, latest, or still standing, actively check for a more recent "
    "update, correction, or replacement rather than trusting the first "
    "matching result, which may be outdated. The reverse mistake is just as "
    "costly: when a question anchors to one specific dated snapshot or "
    "edition (e.g. \"the index created on 14 August 2026\", \"the July 2004 "
    "table\"), check that the page you actually fetched states that exact "
    "date -- a live page can move past a named snapshot date by the time "
    "you fetch it, and silently answering from today's version of a page "
    "the question anchored to an earlier date is a wrong source even when "
    "everything else about the answer is correct. If the date on the page "
    "you found does not match what was asked, treat that source as stale "
    "for this question and search for an archived version at that date "
    "(e.g. via web.archive.org) instead of proceeding with the mismatch. "
    "The same check applies to edition TYPE, not only date: many official "
    "sources publish both a base/annual edition and separate periodic "
    "amendment, errata, or \"weekly changes\" pages for the same "
    "publication -- if the question names \"the annual edition\", confirm "
    "the page you cite is actually labeled as that edition, not an update "
    "sheet layered on top of it, even when the update sheet covers the "
    "same subject and looks like a match. "
    "Prefer the primary/official "
    "document over a secondary summary or briefing about it. When a "
    "question names one specific page or document as the source (e.g. "
    "\"using only the X page\", \"the compiled Y report\"), cite that exact "
    "page even if the same fact also appears on a different page you found "
    "-- an answer backed by the named source is preferred over an "
    "identically correct answer backed by an equivalent but different "
    "source. Never invent "
    "a placeholder value (like -1, 0, or \"unknown\") with NO basis in the "
    "evidence and pass it off as a real answer -- that is different from "
    "committing to a real candidate the evidence partially supports (see "
    "ANSWERING below on what to do with a partial lead).\n\n"
    "COMPUTING: never do arithmetic yourself -- multiplication, sums, "
    "percentages, differences, unit conversions -- always use the "
    "`compute` tool for exact, non-rounded results. The same applies to "
    "classifying membership across two or more lists (which items appear "
    "in both, only one, or neither) -- once you have each list's exact "
    "items in front of you, write them as Python lists in `compute` and let "
    "it compute the actual intersection/difference, rather than tracking "
    "overlap by eye across a long list, which is exactly where "
    "cross-referencing mistakes happen.\n\n"
    "PROVING: the instant you read the specific number, name, or fact that "
    "settles part of the answer, call `note_evidence` with the exact "
    "evidence index and the verbatim text (copy it, don't paraphrase) -- "
    "do this while that source is in front of you, not later from memory. "
    "This is what lets your final citation point at the real proof instead "
    "of just the page it came from. Do this for every candidate you check "
    "in a comparison, not only the one that turns out to be the answer -- "
    "the evidence that rules a candidate OUT is as important to record as "
    "the evidence that confirms the winner. THIS MATTERS MOST when your "
    "answer is something you had to find, not something the question "
    "already named: measured on a real task where our answer exactly "
    "matched the reference -- same name, same figures, all correct -- it "
    "still scored zero because the citation only covered an earlier, "
    "unrelated part of a long source (\"I do not see [the actual answer] "
    "in these citations... the second answer is superior\"). A page with "
    "dozens of similar-looking rows or entries gives search-relevance "
    "alone almost nothing to distinguish the one that matters, so without "
    "note_evidence pointing at the exact row, your citation can silently "
    "land on the wrong one even while your stated answer is completely "
    "correct -- and an unsupported correct answer is scored the same as a "
    "wrong one.\n\n"
    "ANSWERING: when you have enough evidence, respond with your final "
    "answer as plain text and make no further tool calls. Follow the "
    "question's literal formatting instructions exactly (notation style, "
    "digit precision, ordering, units) -- do not paraphrase or round a "
    "value the question asked for verbatim. This applies to official "
    "abbreviated codes and notation exactly as much as numbers: when a "
    "source prints a specific code or symbol string for a value (a "
    "navigation-light characteristic like \"Fl (4)W 10s\", a unit "
    "abbreviation, a standard's short-form label), copy that exact string "
    "-- do not translate, expand, or describe it in plain English (\"Fl (4)"
    "W 10s\" is the answer; \"flashing white\" is a wrong answer even "
    "though it describes the same thing). The same discipline applies to a "
    "specific printed name or title, not only symbolic codes: when a "
    "source lists a named item by its full title (a coin design, a report "
    "title, a named award), give that exact printed name -- a generic "
    "category word standing in for it (\"coloured\" for a design whose "
    "actual printed name is \"$2 -- 30th Anniversary of the Torres Strait "
    "Flag\") is as wrong as an invented value, even when the category word "
    "is technically accurate. Cite the evidence item numbers "
    "inline using double brackets like [[2]] or [[1]][[3]] for every "
    "non-obvious factual claim -- [[n]] is the citation pointer the judge "
    "recognizes; a single-bracket [2] is read as ordinary text and backs "
    "nothing. When "
    "the question asks for the single largest/smallest/most/least/highest/"
    "lowest item among a set of candidates, cite the evidence for every "
    "candidate you compared, not only the winner -- a correct answer with "
    "only the winner cited does not visibly prove the comparison was "
    "exhaustive, and gets marked down for it even when the value is right. "
    "The same rule applies to a COUNT or LIST of items that all satisfy some "
    "condition (e.g. \"10 entries are empty\", \"these items qualify\"): cite "
    "each individual item's own supporting evidence, not one shared slice "
    "that only happens to show a few of them -- a correct count backed by "
    "citations that visibly cover just some of the members loses to an "
    "identical count whose citations cover every single one. When you find "
    "the qualifying detail for each member, call note_evidence for it right "
    "away so the final citation can point at that member specifically. "
    "State each qualifying item once -- do not restate the same list twice "
    "in different formats (e.g. a numbered list immediately followed by "
    "the identical bullet list); that reads as padding even when every "
    "item in it is correct. "
    "COMMIT TO YOUR BEST-SUPPORTED CANDIDATE: the judge scores your answer "
    "against a specific reference value -- a stated candidate that turns "
    "out wrong scores exactly the same (zero) as a refusal, but a refusal "
    "can never score. Once your searches are genuinely exhausted, never "
    "write a sentence narrating what you could not find or how confident "
    "you are ('I cannot confidently identify...', 'the evidence is "
    "insufficient', 'I cannot determine...', 'based on the evidence "
    "gathered, I cannot...') -- those phrasings guarantee zero credit. "
    "Instead name the single entity/value that best fits the clues you did "
    "confirm, stated as a plain, direct answer, and let any real "
    "uncertainty show only through which specific claims carry a [[n]] "
    "citation and which don't -- never through hedging language or a "
    "declined answer. This applies even when only some of a multi-clue "
    "question's conditions were verified: name the candidate that fits the "
    "most and best-confirmed clues rather than naming none. The one "
    "narrow exception is when the question asks for a figure that "
    "genuinely does not exist in any published form (not merely one you "
    "personally couldn't find) -- there, state that fact plainly as the "
    "answer, cited to what establishes its absence, rather than inventing "
    "a number."
)

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web. Returns a numbered list of results with short snippets.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_OPEN_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "open_page",
        "description": (
            "Fetch a URL and return its content, focused on the parts most "
            "relevant to the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_COMPUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "compute",
        "description": (
            "Evaluate exact arithmetic in Python with arbitrary decimal "
            "precision -- no floating-point rounding. Use this for any "
            "calculation that needs a precise, non-rounded answer "
            "(multiplication, percentages, differences, sums), instead of "
            "computing it yourself, since language-model arithmetic on long "
            "or precise numbers is unreliable. `Decimal` and `getcontext` "
            "are already imported with 60 digits of precision. Assign the "
            "final value to `result` as a string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python statements ending with result = str(<expression>).",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_PAGE_GREP_TOOL = {
    "type": "function",
    "function": {
        "name": "page_grep",
        "description": (
            "Search for a substring within a source you already opened via "
            "search or open_page, without fetching it again. Returns up to "
            "5 matches with surrounding context. Use this when you need a "
            "specific detail from a long page you already have (a number, "
            "a name, a date) instead of re-reading the whole thing or "
            "calling open_page on the same URL again -- it costs nothing "
            "and finds things a truncated view might have missed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "evidence_index": {
                    "type": "integer",
                    "description": "The [N] index of the already-opened source to search within.",
                },
                "pattern": {
                    "type": "string",
                    "description": "The exact substring to search for (case-insensitive).",
                },
            },
            "required": ["evidence_index", "pattern"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_NOTE_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "note_evidence",
        "description": (
            "Record that a specific quote from an already-opened source is "
            "the exact proof for a claim you are about to make. Call this "
            "the moment you find a decisive value -- do not wait until "
            "you're writing the final answer, and do not paraphrase; copy "
            "the exact text as it appears in that source's content. This "
            "is what makes your final citation point at the real proof "
            "instead of just the page in general."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "evidence_index": {
                    "type": "integer",
                    "description": "The [N] index of the source this quote came from.",
                },
                "quote": {
                    "type": "string",
                    "description": "The exact, verbatim text from that source that proves the claim.",
                },
            },
            "required": ["evidence_index", "quote"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

LOOP_TOOLS: tuple[dict[str, Any], ...] = (
    _SEARCH_TOOL,
    _OPEN_PAGE_TOOL,
    _PAGE_GREP_TOOL,
    _COMPUTE_TOOL,
    _NOTE_EVIDENCE_TOOL,
)


@dataclass(slots=True)
class RunState:
    started_at: float = field(default_factory=time.monotonic)
    remaining_budget_usd: float | None = None
    model_waterfall: tuple[tuple[str, str], ...] = DEFAULT_MODEL_WATERFALL

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def note_budget(self, remaining_budget_usd: float) -> None:
        self.remaining_budget_usd = remaining_budget_usd

    def budget_is_low(self) -> bool:
        return (
            self.remaining_budget_usd is not None
            and self.remaining_budget_usd <= MIN_REMAINING_BUDGET_USD
        )

    def past_soft_deadline(self) -> bool:
        return self.elapsed() >= WALL_BUDGET_S

    def past_hard_deadline(self) -> bool:
        return self.elapsed() >= HARD_DEADLINE_S


@dataclass(frozen=True, slots=True)
class Evidence:
    index: int
    receipt_id: str
    result_id: str
    url: str | None
    title: str | None
    note: str | None
    # Offsets into `note` that were judged keyword-dense (i.e. what the model
    # was actually shown / what's actually relevant), so citations can point
    # at real content instead of always the literal start of the page --
    # see _build_citations.
    relevant_spans: tuple[tuple[int, int], ...] = ()
    # Offsets the model itself pinpointed via the note_evidence tool as the
    # exact proof for a specific claim -- preferred over relevant_spans in
    # _build_citations when present, since these are model-verified rather
    # than keyword-density guesses.
    retained_spans: tuple[tuple[int, int], ...] = ()


@dataclass(slots=True)
class EvidenceStore:
    items: list[Evidence] = field(default_factory=list)

    def add(
        self,
        *,
        receipt_id: str,
        result_id: str,
        url: str | None,
        title: str | None,
        note: str | None,
        relevant_spans: tuple[tuple[int, int], ...] = (),
    ) -> int:
        index = len(self.items)
        self.items.append(
            Evidence(
                index=index,
                receipt_id=receipt_id,
                result_id=result_id,
                url=url,
                title=title,
                note=note,
                relevant_spans=relevant_spans,
            )
        )
        return index

    def get(self, index: int) -> Evidence | None:
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def find_by_url(self, url: str) -> int | None:
        normalized = _normalize_url(url)
        for item in self.items:
            if item.url and _normalize_url(item.url) == normalized:
                return item.index
        return None

    def replace_item(
        self,
        index: int,
        *,
        receipt_id: str,
        result_id: str,
        url: str | None,
        title: str | None,
        note: str | None,
        relevant_spans: tuple[tuple[int, int], ...],
    ) -> None:
        # 2026-08-18: real diagnosed loss -- the loop re-fetched the same PDF
        # four times in one run under URL variants like "...p22st.pdf" and
        # "...p22st.pdf#page=30" (each open_page call creating a fresh
        # evidence index), and the model ended up citing an earlier, shallow
        # fetch's index instead of the later one where it actually read the
        # real table data -- a numerically exact answer lost purely because
        # the citation pointed at the wrong instance of the right URL.
        # _tool_open_page now reuses the same index for a repeat URL (after
        # stripping any #fragment) instead of creating a new one. This
        # replaces the whole entry rather than merging spans, because
        # `relevant_spans` are offsets into a specific `note` string -- two
        # fetches of the "same" URL are not guaranteed to return byte-
        # identical text, so unioning old offsets against new content would
        # silently point at the wrong characters.
        self.items[index] = Evidence(
            index=index,
            receipt_id=receipt_id,
            result_id=result_id,
            url=url,
            title=title,
            note=note,
            relevant_spans=relevant_spans,
        )

    def add_retained_span(self, index: int, span: tuple[int, int]) -> None:
        item = self.items[index]
        combined = tuple(sorted(set(item.retained_spans) | {span}))
        self.items[index] = Evidence(
            index=item.index,
            receipt_id=item.receipt_id,
            result_id=item.result_id,
            url=item.url,
            title=item.title,
            note=item.note,
            relevant_spans=item.relevant_spans,
            retained_spans=combined,
        )


@entrypoint("query")
async def query(query: Query) -> Response:
    try:
        return await _run_query(query)
    except Exception:
        return _last_resort_response(query)


async def _run_query(query: Query) -> Response:
    state = RunState()
    state.model_waterfall = await _resolve_model_waterfall()
    store = EvidenceStore()

    try:
        loop_answer = await _run_loop(query.text, store, state)
    except Exception:
        loop_answer = None

    text_answer = await _finalize_answer(query.text, loop_answer, store, state)
    text_answer = await _audit_answer(query.text, text_answer, store, state)
    text_answer = _clamp_text(text_answer)
    text_answer, citations = _build_citations(query.text, text_answer, store)

    if query.output_schema is None:
        return Response(text=text_answer, citations=citations)

    structured = await _build_structured_output(query, store, text_answer, state)
    return Response(output=structured, citations=citations)


def _last_resort_response(query: Query) -> Response:
    if query.output_schema is not None:
        return Response(output=_best_effort_structured(query.output_schema, _NO_ANSWER_STUB))
    return Response(text="No answer could be produced for this question due to an internal error.")


async def _resolve_model_waterfall() -> tuple[tuple[str, str], ...]:
    try:
        info = await tooling_info(timeout=TOOLING_INFO_TIMEOUT_SECONDS)
    except Exception:
        return DEFAULT_MODEL_WATERFALL
    response = info.response if isinstance(info.response, dict) else {}
    allowed = response.get("allowed_llm_provider_models")
    if not isinstance(allowed, dict):
        return DEFAULT_MODEL_WATERFALL
    resolved = tuple(
        (provider, model)
        for provider, model in DEFAULT_MODEL_WATERFALL
        if isinstance(allowed.get(provider), list) and model in allowed[provider]
    )
    return resolved or DEFAULT_MODEL_WATERFALL


# --------------------------------------------------------------------------
# Agentic tool loop
# --------------------------------------------------------------------------


async def _run_loop(question: str, store: EvidenceStore, state: RunState) -> str | None:
    keywords = _keywords_from(question)
    anchors = _anchors_from(question)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _LOOP_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _turn in range(MAX_LOOP_TURNS):
        if state.past_soft_deadline() or state.budget_is_low():
            break
        response = await _call_loop_turn(messages, state)
        if response is None or not response.llm.choices:
            break
        message = response.llm.choices[0].message
        if not message.tool_calls:
            text = _join_text_parts(message.content)
            # 2026-08-18: real diagnosed loss -- a turn ended with the model
            # writing a tool call out as literal prose ("<function_calls>
            # <invoke name=\"compute\">...") instead of a real structured
            # tool_calls entry, and with budget/turns to spare. Ending the
            # loop there ships garbled XML; nudging the model to retry the
            # call for real (still well within budget) gives it a real
            # chance to finish the computation instead of falling back to a
            # rescue rung that can't run `compute` at all.
            if _TOOL_MARKUP_RE.search(text or ""):
                messages.append(message.to_input_message())
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That was not a real tool call -- no tool ran, "
                            "and nothing was recorded. Make the tool call "
                            "directly through the tool-calling interface, "
                            "not as text in your message."
                        ),
                    }
                )
                continue
            # 2026-08-18: real diagnosed loss, seen twice -- the model
            # opens only some of the documents/periods a question named,
            # then self-admits the gap in its own final answer instead of
            # going back for the rest, even with turns and budget to
            # spare. Catch its own admission and send it back rather than
            # finalizing an answer it already knows is incomplete.
            if (
                _turn < MAX_LOOP_TURNS - 1
                and not state.past_soft_deadline()
                and not state.budget_is_low()
                and _INCOMPLETE_COVERAGE_RE.search(text or "")
            ):
                messages.append(message.to_input_message())
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You just noted evidence you haven't gathered "
                            "yet. Go get it now with search/open_page -- "
                            "do not finalize an answer you already know is "
                            "incomplete."
                        ),
                    }
                )
                continue
            return text
        messages.append(message.to_input_message())
        # The system prompt tells the model to batch several searches/opens
        # in one turn (parallel_tool_calls=True is set on the call above);
        # run them concurrently so that instruction actually saves wall-clock
        # time instead of paying N x latency for an N-call turn. Each call
        # commits its own evidence via store.add as it completes -- safe
        # under asyncio's single-threaded interleaving, and the model always
        # reads the assigned `index` back out of its own tool response
        # before citing it, so completion-order index assignment doesn't
        # affect correctness.
        outputs = await asyncio.gather(
            *(_execute_tool_call(call, store, state, keywords, anchors) for call in message.tool_calls)
        )
        tool_messages = [
            {"role": "tool", "tool_call_id": call.id, "content": output}
            for call, output in zip(message.tool_calls, outputs, strict=True)
        ]
        messages.extend(tool_messages)

    if state.past_hard_deadline():
        return None
    return await _force_final_answer(messages, state)


async def _call_loop_turn(messages: list[dict[str, Any]], state: RunState) -> Any | None:
    for provider, model in state.model_waterfall:
        try:
            result = await llm_chat(
                provider=provider,
                model=model,
                messages=messages,
                tools=list(LOOP_TOOLS),
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0.2,
                timeout=LOOP_LLM_TIMEOUT_SECONDS,
            )
        except Exception:  # noqa: S112 -- provider outage: fall through to the next waterfall entry
            continue
        state.note_budget(result.budget.session_remaining_budget_usd)
        return result
    return None


async def _force_final_answer(messages: list[dict[str, Any]], state: RunState) -> str | None:
    nudge = [
        *messages,
        {
            "role": "user",
            "content": (
                "Stop researching now. Using only what you've already gathered "
                "above, write the final answer. Cite evidence numbers inline "
                "using double brackets like [[2]] for every claim they "
                "support -- a single-bracket [2] does not count as a "
                "citation. Commit to your single best-supported candidate "
                "even if you weren't able to verify every clue -- a stated "
                "candidate that turns out wrong scores the same (zero) as a "
                "refusal, but only a stated candidate can score. Do not "
                "write a sentence about what you couldn't find or how "
                "confident you are; state the answer plainly."
            ),
        },
    ]
    for provider, model in state.model_waterfall:
        try:
            result = await llm_chat(
                provider=provider,
                model=model,
                messages=nudge,
                tools=list(LOOP_TOOLS),
                tool_choice="none",
                temperature=0.2,
                timeout=LOOP_LLM_TIMEOUT_SECONDS,
            )
        except Exception:  # noqa: S112 -- provider outage: fall through to the next waterfall entry
            continue
        state.note_budget(result.budget.session_remaining_budget_usd)
        text = _extract_text(result)
        if text:
            return text
    return None


async def _execute_tool_call(
    call: Any,
    store: EvidenceStore,
    state: RunState,
    keywords: set[str],
    anchors: tuple[str, ...] = (),
) -> str:
    try:
        arguments = json.loads(call.arguments)
    except Exception:
        arguments = {}
    if call.name == "search":
        query_text = str(arguments.get("query") or "").strip()
        return await _tool_search(query_text or "results", store, state, keywords, anchors)
    if call.name == "open_page":
        url = str(arguments.get("url") or "").strip()
        return await _tool_open_page(url, store, state, keywords, anchors)
    if call.name == "page_grep":
        raw_index = arguments.get("evidence_index")
        pattern = str(arguments.get("pattern") or "")
        return _tool_page_grep(raw_index, pattern, store)
    if call.name == "compute":
        code = str(arguments.get("code") or "")
        return _tool_compute(code)
    if call.name == "note_evidence":
        raw_index = arguments.get("evidence_index")
        quote = str(arguments.get("quote") or "")
        return _tool_note_evidence(raw_index, quote, store)
    return json.dumps({"error": f"unknown tool {call.name}"})


def _tool_note_evidence(raw_index: Any, quote: str, store: EvidenceStore) -> str:
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return json.dumps({"noted": False, "reason": "evidence_index must be an integer"})
    item = store.get(index)
    if item is None or not item.note:
        return json.dumps({"noted": False, "reason": f"no evidence at index {index}"})
    quote = quote.strip()
    if not quote:
        return json.dumps({"noted": False, "reason": "quote is empty"})
    pos = item.note.find(quote)
    if pos == -1:
        pos = item.note.lower().find(quote.lower())
    if pos == -1:
        return json.dumps(
            {
                "noted": False,
                "reason": (
                    "quote not found verbatim in that evidence's content -- "
                    "copy the exact text, don't paraphrase"
                ),
            }
        )
    margin = 260
    start = max(0, pos - margin)
    end = min(len(item.note), pos + len(quote) + margin)
    store.add_retained_span(index, (start, end))
    return json.dumps({"noted": True})


_PAGE_GREP_MAX_MATCHES = 5
_PAGE_GREP_CONTEXT_CHARS = 400


def _tool_page_grep(raw_index: Any, pattern: str, store: EvidenceStore) -> str:
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return json.dumps({"error": "evidence_index must be an integer"})
    item = store.get(index)
    if item is None or not item.note:
        return json.dumps({"error": f"no evidence at index {index}"})
    pattern = pattern.strip()
    if not pattern:
        return json.dumps({"error": "missing pattern"})
    text = item.note
    lowered = text.lower()
    needle = pattern.lower()
    matches: list[dict[str, Any]] = []
    search_from = 0
    while len(matches) < _PAGE_GREP_MAX_MATCHES:
        pos = lowered.find(needle, search_from)
        if pos == -1:
            break
        start = max(0, pos - _PAGE_GREP_CONTEXT_CHARS)
        end = min(len(text), pos + len(pattern) + _PAGE_GREP_CONTEXT_CHARS)
        matches.append({"offset": pos, "context": text[start:end]})
        search_from = pos + max(len(needle), 1)
    if not matches:
        return json.dumps(
            {
                "matches": [],
                "note": "pattern not found in this evidence's content -- try a shorter or differently-worded substring",
            }
        )
    return json.dumps({"matches": matches})


def _tool_compute(code: str) -> str:
    if not code.strip():
        return json.dumps({"error": "missing code"})
    wrapped = f"from decimal import Decimal, getcontext\ngetcontext().prec = 60\n{code}"
    try:
        outcome = safe_exec(wrapped)
    except Exception as exc:
        return json.dumps({"error": str(exc)[:300]})
    return json.dumps({"result": outcome})


async def _tool_search(
    query_text: str,
    store: EvidenceStore,
    state: RunState,
    keywords: set[str],
    anchors: tuple[str, ...] = (),
) -> str:
    resp = None
    for provider in SEARCH_PROVIDER_ORDER:
        resp = await _search_once(query_text, provider, state)
        if resp is not None and resp.results:
            break
        # trying every remaining provider could alone consume the whole
        # budget; bail on the soft deadline, not just the hard one.
        if state.past_soft_deadline():
            break
    if resp is None or not resp.results:
        return json.dumps(
            {
                "results": [],
                "note": (
                    "no results from any search provider -- try a shorter, "
                    "more general query (drop quoted phrases and exact "
                    "dates) or search for the source organization and "
                    "document type instead of the exact title"
                ),
            }
        )
    entries: list[dict[str, Any]] = []
    for result in resp.results[:MAX_RESULTS_PER_SEARCH]:
        if _looks_like_puzzle_spam(result.note or ""):
            # 2026-08-18: real diagnosed loss (BrowseComp) -- a multi-clue,
            # riddle-style question read enough like a crossword clue to a
            # search engine that puzzle-aggregator sites came back as top
            # results and got cited as if they were about the actual
            # subject. Drop them before the model ever sees them as an
            # option, rather than relying on it to notice and discard them.
            continue
        # 2026-08-18: search results can themselves carry a full page's worth
        # of `note` (some providers return the whole converted page, not a
        # short snippet) even though the model only ever sees a 400-char
        # `summary` of it below -- the same head-slicing citation bug that
        # _tool_open_page had applies here too unless a relevant span is
        # located up front.
        spans = _citation_spans(result.note or "", keywords, anchors)
        index = store.add(
            receipt_id=resp.receipt_id,
            result_id=result.result_id,
            url=result.url,
            title=result.title,
            note=result.note,
            relevant_spans=spans,
        )
        summary = (result.note or result.title or result.url or "")[:400]
        entries.append({"index": index, "title": result.title, "url": result.url, "summary": summary})
    return json.dumps({"results": entries})


async def _tool_open_page(
    url: str,
    store: EvidenceStore,
    state: RunState,
    keywords: set[str],
    anchors: tuple[str, ...] = (),
) -> str:
    if not url:
        return json.dumps({"error": "missing url"})
    resp = await _fetch_richest(url, state)
    if resp is None or not resp.results:
        return json.dumps({"error": f"could not fetch {url}"})
    top = resp.results[0]
    resolved_url = top.url or url
    if _looks_like_binary(top.note or ""):
        return json.dumps(
            {
                "error": (
                    f"could not read {resolved_url} as text -- it looks like "
                    "a binary file (e.g. a spreadsheet, .xlsx/.docx, or an "
                    "unconverted PDF) that this tool can't extract readable "
                    "content from. Search for a different source instead: an "
                    "HTML rendering of the same data, a CSV version, or a "
                    "page that describes/summarizes it -- don't treat this "
                    "as evidence and don't give up on the question because "
                    "of it."
                )
            }
        )
    if _looks_like_puzzle_spam(top.note or ""):
        return json.dumps(
            {
                "error": (
                    f"{resolved_url} looks like a crossword/trivia-puzzle "
                    "aggregator, not a source about the actual subject -- "
                    "these sites superficially match almost any wordy "
                    "query. Search on the single most distinctive detail "
                    "from the question instead, and don't treat this page "
                    "as evidence."
                )
            }
        )
    focused, spans = _densest_windows(top.note or "", keywords, anchors)
    existing = store.find_by_url(resolved_url)
    if existing is not None:
        prior = store.get(existing)
        prior_len = len(prior.note) if prior and prior.note else 0
        new_len = len(top.note) if top.note else 0
        # Only overwrite with a fetch that's at least as rich as what's
        # already stored -- a later re-fetch could hit a transient provider
        # hiccup and return less than a prior successful fetch did.
        if new_len >= prior_len:
            store.replace_item(
                existing,
                receipt_id=resp.receipt_id,
                result_id=top.result_id,
                url=resolved_url,
                title=top.title,
                note=top.note,
                relevant_spans=spans,
            )
        index = existing
    else:
        index = store.add(
            receipt_id=resp.receipt_id,
            result_id=top.result_id,
            url=resolved_url,
            title=top.title,
            note=top.note,
            relevant_spans=spans,
        )
    return json.dumps({"index": index, "title": top.title, "url": resolved_url, "content": focused})


async def _search_once(query_text: str, provider: str, state: RunState) -> Any | None:
    try:
        resp = await search_web(
            query_text,
            provider=provider,
            num=MAX_RESULTS_PER_SEARCH,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    state.note_budget(resp.budget.session_remaining_budget_usd)
    return resp


async def _fetch_once(url: str, provider: str, state: RunState) -> Any | None:
    try:
        resp = await fetch_page(url, provider=provider, timeout=FETCH_TIMEOUT_SECONDS)
    except Exception:
        return None
    state.note_budget(resp.budget.session_remaining_budget_usd)
    return resp


async def _fetch_richest(url: str, state: RunState) -> Any | None:
    # Capped at two provider attempts, not the full waterfall: the diagnosed
    # failure needed exactly one more try to recover a full document, and a
    # great many genuinely short pages (a brief press release, an API
    # response) would otherwise pay for a five-provider sweep every time
    # they landed under the threshold for no real reason.
    best: Any | None = None
    best_len = 0
    for provider in FETCH_PROVIDER_ORDER[:2]:
        resp = await _fetch_once(url, provider, state)
        if resp is not None and resp.results:
            note_len = len(resp.results[0].note or "")
            if note_len > best_len:
                best, best_len = resp, note_len
            if best_len >= THIN_FETCH_RETRY_CHARS:
                break
        if state.past_soft_deadline():
            break
    return best


def _normalize_url(url: str) -> str:
    # A #page=N fragment is a client-side PDF-viewer hint, not something the
    # fetch provider processes server-side -- two fetches of the same
    # document differing only by fragment are the same underlying page.
    return url.split("#", 1)[0]


def _looks_like_binary(text: str) -> bool:
    # 2026-08-18: real diagnosed loss -- a question's source was a .xlsx
    # spreadsheet; the fetch tool returned the raw ZIP-archive bytes as if
    # they were text ("PK!  [Content_Types].xml ..." -- "PK" is the ZIP
    # magic number, xlsx/docx/pptx are all ZIP containers) instead of
    # extracted cell data. The model had nothing readable to work with, and
    # despite an explicit prompt rule against it, resorted to -1/empty
    # placeholders. Detect this case and tell the loop to try a different
    # source instead of feeding it binary noise as if it were real content.
    if not text:
        return False
    sample = text[:2000]
    if sample.startswith(("PK\x03\x04", "PK!", "%PDF")):
        return True
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
    return len(sample) > 0 and (printable / len(sample)) < 0.85


# 2026-08-18: real diagnosed loss (BrowseComp benchmark) -- questions
# written as indirect, multi-clue descriptions get matched by generic
# search against crossword/trivia-puzzle aggregator sites, since a long
# descriptive query reads a lot like a puzzle clue to a search engine.
# Sites like this catalog thousands of unrelated clues/answers and will
# superficially "match" almost any wordy query; treat them as unusable
# rather than storing them as evidence.
_PUZZLE_SPAM_RE = re.compile(
    r"crossword (?:solver|clue|answers?)\b|word(?:s)? puzzle answers?\b",
    re.IGNORECASE,
)


def _looks_like_puzzle_spam(text: str) -> bool:
    if not text:
        return False
    return bool(_PUZZLE_SPAM_RE.search(text[:1000]))


def _keywords_from(question: str) -> set[str]:
    return {match.lower() for match in _KEYWORD_RE.findall(question)}


def _anchors_from(question: str) -> tuple[str, ...]:
    return tuple({match.strip().lower() for match in _ANCHOR_QUOTE_RE.findall(question)})


# 2026-08-20: real diagnosed loss, traced via recorded validator results --
# a task whose structured answer was byte-identical to the reference still
# scored 0.0 on 4/5 validators because the citation slice covered entries
# 1-45 of a large register table while the actual answer (entry 63) was
# never named in the QUESTION -- it was discovered during research. Since
# _anchors_from(question) has nothing to anchor on for a "discovered, not
# named" answer, and generic keyword density is near-uniform across dozens
# of structurally similar table rows, the citation slice landed on the
# wrong region even though the answer text itself was exactly correct.
# These two patterns pull distinctive, low-noise anchors from the model's
# own FINAL ANSWER instead -- proper-noun phrases and alphanumeric code
# tokens (by-law numbers, case numbers, registration IDs) -- so a citation
# needing a slice can be retargeted at the specific value actually claimed.
_ANSWER_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
_ANSWER_CODE_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]+(?:[.\-]+[A-Za-z0-9]+){1,5}\b")
_ANSWER_ANCHOR_STOPWORDS = frozenset(
    {"the answer", "based on", "according to", "the evidence", "the question", "the source"}
)


def _answer_anchors(text: str) -> tuple[str, ...]:
    anchors: set[str] = set()
    for match in _ANCHOR_QUOTE_RE.finditer(text):
        anchors.add(match.group(1).strip().lower())
    for match in _ANSWER_PROPER_NOUN_RE.finditer(text):
        phrase = match.group(0).strip().lower()
        if phrase and phrase not in _ANSWER_ANCHOR_STOPWORDS:
            anchors.add(phrase)
    for match in _ANSWER_CODE_TOKEN_RE.finditer(text):
        token = match.group(0).strip().lower()
        if any(ch.isdigit() for ch in token):
            anchors.add(token)
    return tuple(anchors)


_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]")
_LOOSE_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:])")
_COLLAPSE_SPACE_RE = re.compile(r"\s+")


def _normalize_for_anchor_match(text: str) -> str:
    # 2026-08-18: real diagnosed loss -- the platform's PDF-to-markdown
    # conversion wraps individual clauses in their own **bold** spans (e.g.
    # "**Prison facility capacity** , **custody population**"), which
    # inserts markdown noise and irregular spacing-before-punctuation right
    # in the middle of a heading that otherwise reads as one contiguous
    # phrase. A literal anchor match against raw text silently never fires
    # on real fetched pages even though every word is present. Strip the
    # markdown emphasis markers and collapse the resulting spacing before
    # matching.
    text = _MARKDOWN_EMPHASIS_RE.sub("", text)
    text = _LOOSE_PUNCT_SPACE_RE.sub(r"\1", text)
    return _COLLAPSE_SPACE_RE.sub(" ", text)


def _chunk_score(text: str, keywords: set[str], anchors: tuple[str, ...]) -> int:
    lowered = text.lower()
    score = sum(lowered.count(term) for term in keywords)
    # An exact quoted phrase from the question (questions in this task
    # distribution routinely quote the document/table title verbatim) is a
    # far more precise signal than generic word-frequency, which a long
    # document's front matter or footnotes can win purely by repeating
    # common terms across many unrelated sections. See _ANCHOR_SCORE_BONUS.
    if anchors:
        normalized = _normalize_for_anchor_match(lowered)
        if any(anchor in normalized for anchor in anchors):
            score += _ANCHOR_SCORE_BONUS
    return score


def _densest_windows(
    content: str, keywords: set[str], anchors: tuple[str, ...] = ()
) -> tuple[str, tuple[tuple[int, int], ...]]:
    content = content.strip()
    if len(content) <= MAX_PAGE_CHARS:
        # The whole page is shown to the model as-is, but a citation later
        # still can't materialize all of it (MAX_CITATION_SLICE_CHARS is far
        # smaller than MAX_PAGE_CHARS) -- locate the keyword-dense region(s)
        # now, while we have both the content and the question's keywords,
        # so _build_citations doesn't have to fall back to slicing offset 0.
        return content, _citation_spans(content, keywords, anchors)
    window = MAX_PAGE_CHARS // DENSEST_CHUNKS_PICKED
    chunk_bounds = [(i, min(i + window, len(content))) for i in range(0, len(content), window)]
    if len(chunk_bounds) <= DENSEST_CHUNKS_PICKED:
        head = content[:MAX_PAGE_CHARS]
        return head, ((0, len(head)),)
    # Score every chunk uniformly, including the first. A real diagnosed
    # loss showed why: unconditionally keeping the literal page head cited
    # nothing but site-navigation boilerplate ("Home / Explore our
    # collections / Research tools...") on a page whose real content
    # started further down -- a later, denser chunk should win instead of
    # the head always getting a free pass.
    scores = [_chunk_score(content[s:e], keywords, anchors) for s, e in chunk_bounds]
    ranked = sorted(range(len(chunk_bounds)), key=lambda i: scores[i], reverse=True)
    # Only chunks that actually matched something count as "relevant" --
    # padding up to DENSEST_CHUNKS_PICKED with zero-score chunks would (via
    # the stable sort's tie-break on original index) silently let chunk 0
    # back in even when it was the boilerplate this exists to exclude.
    relevant = [i for i in ranked if scores[i] > 0][:DENSEST_CHUNKS_PICKED]
    if not relevant:
        head = content[:MAX_PAGE_CHARS]
        return head, ((0, len(head)),)
    picked = sorted(relevant)
    spans = tuple(chunk_bounds[i] for i in picked)
    text = "\n...\n".join(content[s:e] for s, e in spans)[:MAX_PAGE_CHARS]
    return text, spans


def _citation_spans(
    content: str, keywords: set[str], anchors: tuple[str, ...] = ()
) -> tuple[tuple[int, int], ...]:
    """Locate keyword-dense region(s) within `content` sized for a single
    citation slice, used when the whole page was shown to the model (so no
    reading-window spans exist) but the page is still too long to cite in
    full."""
    if len(content) <= MAX_CITATION_SLICE_CHARS or not (keywords or anchors):
        return ()
    window = max(MAX_CITATION_SLICE_CHARS // DENSEST_CHUNKS_PICKED, 1)
    chunk_bounds = [(i, min(i + window, len(content))) for i in range(0, len(content), window)]
    scores = [_chunk_score(content[s:e], keywords, anchors) for s, e in chunk_bounds]
    ranked = sorted(range(len(chunk_bounds)), key=lambda i: scores[i], reverse=True)
    relevant = [i for i in ranked if scores[i] > 0][:DENSEST_CHUNKS_PICKED]
    if not relevant:
        return ()
    return tuple(chunk_bounds[i] for i in sorted(relevant))


# --------------------------------------------------------------------------
# Rescue ladder
# --------------------------------------------------------------------------


async def _finalize_answer(
    question: str,
    loop_answer: str | None,
    store: EvidenceStore,
    state: RunState,
) -> str:
    if _is_usable_answer(loop_answer):
        return loop_answer  # type: ignore[return-value]

    if store.items and not state.past_hard_deadline():
        digest_answer = await _write_from_digest(question, store, state)
        if _is_usable_answer(digest_answer):
            return digest_answer  # type: ignore[return-value]

    if store.items:
        deterministic = _deterministic_answer(question, store)
        if _is_usable_answer(deterministic):
            return deterministic

    if not state.past_hard_deadline():
        knowledge_answer = await _knowledge_only_answer(question, state)
        if _is_usable_answer(knowledge_answer):
            return knowledge_answer  # type: ignore[return-value]

    return _NO_ANSWER_STUB


def _is_usable_answer(text: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_USABLE_ANSWER_CHARS or stripped == _NO_ANSWER_STUB:
        return False
    return not _TOOL_MARKUP_RE.search(stripped)


async def _audit_answer(question: str, answer: str, store: EvidenceStore, state: RunState) -> str:
    # Diagnosed from real local-eval losses: (1) the model found the right
    # facts but truncated a decimal and used exponent notation the question
    # explicitly forbade, (2) it checked only the first plausible candidate
    # in a multi-candidate question and named the wrong entity, (3) it
    # filled an unfound field with a fabricated placeholder (-1) instead of
    # saying so. One pass checking all three catches each without
    # re-researching from scratch.
    if not _is_usable_answer(answer) or state.budget_is_low() or state.past_hard_deadline():
        return answer
    evidence_block = _evidence_block(store)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict answer auditor. Check the draft answer "
                "against the question and the evidence on seven things: "
                "(1) FORMAT -- does it follow the question's literal "
                "formatting/precision instructions exactly (notation, "
                "digit precision, ordering, units), with no rounding or "
                "paraphrasing of a verbatim-requested value; (2) "
                "COMPLETENESS -- if the question required checking multiple "
                "candidates or conditions, does the evidence show every "
                "candidate was actually checked, not just the first "
                "plausible one; (3) FABRICATION -- does any field contain a "
                "placeholder (like -1, 0, or \"unknown\") standing in for a "
                "value that was never actually found in the evidence; (4) "
                "MULTI-CLUE CONSISTENCY -- if the question identifies its "
                "subject through several distinct clues or characteristics "
                "rather than naming it directly, does the evidence show the "
                "answer actually satisfies every one of them, not just the "
                "most distinctive or memorable clue; (5) ENUMERATION "
                "COVERAGE -- if the answer states a count or list of items "
                "all satisfying some condition, does the evidence show each "
                "individual item was checked and cited, not just some of "
                "them behind one shared citation; (6) SOURCE DATE -- if the "
                "question anchors to one specific dated snapshot or "
                "edition, does the cited evidence's own stated date "
                "actually match that date rather than a later live version; "
                "(7) COMMITMENT -- does the draft contain any sentence "
                "narrating what it could not find or how confident it is "
                "('cannot determine', 'insufficient evidence', 'cannot "
                "confidently identify', 'based on the evidence gathered, "
                "I cannot...') instead of stating a specific answer -- a "
                "refusal scores zero while a stated best-supported "
                "candidate has a real chance of being right, so replace any "
                "such hedge with the single entity/value the evidence most "
                "supports, cited to whatever does confirm it, even if not "
                "every clue was verified. "
                "If "
                "the draft fully passes all seven, repeat it unchanged, "
                "including every [[n]] citation marker exactly as written. "
                "If it fails one, output a corrected version that fixes "
                "only that issue and keeps every [[n]] marker in place -- "
                "never invent new facts not in the evidence. Output only "
                "the answer text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nEvidence:\n{evidence_block}\n\nDraft answer:\n{answer}",
        },
    ]
    audited = await _call_synthesis(messages, state)
    return audited if _is_usable_answer(audited) else answer


async def _write_from_digest(question: str, store: EvidenceStore, state: RunState) -> str | None:
    evidence_block = _evidence_block(store)
    messages = [
        {"role": "system", "content": "Answer only from the evidence provided. Cite [[n]] for every claim."},
        {
            "role": "user",
            "content": f"Question: {question}\n\nEvidence:\n{evidence_block}\n\nWrite the answer now.",
        },
    ]
    return await _call_synthesis(messages, state)


def _deterministic_answer(question: str, store: EvidenceStore) -> str:
    lines = [f"Evidence gathered for: {question}"]
    for item in store.items:
        snippet = (item.note or item.title or item.url or "").strip()
        if len(snippet) > DETERMINISTIC_SNIPPET_CHARS:
            snippet = snippet[:DETERMINISTIC_SNIPPET_CHARS].rstrip() + "..."
        lines.append(f"[{item.index}] {snippet}")
    return "\n".join(lines)


async def _knowledge_only_answer(question: str, state: RunState) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer from your own knowledge only. State clearly that this "
                "is not sourced from search evidence, and hedge on anything "
                "you are not confident about."
            ),
        },
        {"role": "user", "content": question},
    ]
    return await _call_synthesis(messages, state)


async def _call_synthesis(
    messages: list[dict[str, str]],
    state: RunState,
    *,
    timeout: float = SYNTH_LLM_TIMEOUT_SECONDS,
) -> str | None:
    for provider, model in state.model_waterfall:
        try:
            result = await llm_chat(
                provider=provider,
                model=model,
                messages=messages,
                temperature=0.2,
                timeout=timeout,
            )
        except Exception:  # noqa: S112 -- provider outage: fall through to the next waterfall entry
            continue
        state.note_budget(result.budget.session_remaining_budget_usd)
        text = _extract_text(result)
        if text:
            return text
    return None


def _evidence_block(store: EvidenceStore) -> str:
    if not store.items:
        return "(no evidence was gathered)"
    lines: list[str] = []
    for item in store.items:
        snippet = (item.note or item.title or item.url or "").strip()
        if len(snippet) > EVIDENCE_BLOCK_SNIPPET_CHARS:
            snippet = snippet[:EVIDENCE_BLOCK_SNIPPET_CHARS].rstrip() + "..."
        source = item.url or "unknown source"
        lines.append(f"[{item.index}] {snippet}\n    source: {source}")
    return "\n".join(lines)


def _extract_text(result: Any) -> str | None:
    choices = result.llm.choices
    if not choices:
        return None
    return _join_text_parts(choices[0].message.content)


def _join_text_parts(parts: Any) -> str | None:
    fragments: list[str] = []
    for part in parts:
        text = (part.text or "").strip()
        if text:
            fragments.append(text)
    joined = "\n".join(fragments).strip()
    return joined or None


def _clamp_text(text: str) -> str:
    text = text.strip()
    if not text:
        return _NO_ANSWER_STUB
    if len(text) > MAX_RESPONSE_CHARS:
        return text[: MAX_RESPONSE_CHARS - 3] + "..."
    return text


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def _extract_cited_indices(text: str, store: EvidenceStore) -> list[int]:
    valid_indices = {item.index for item in store.items}
    seen: set[int] = set()
    found: list[int] = []
    for match in _CITATION_INDEX_RE.finditer(text):
        idx = int(match.group(1))
        if idx in valid_indices and idx not in seen:
            seen.add(idx)
            found.append(idx)
    return found


def _relevance_ranked_indices(question: str, answer: str, store: EvidenceStore) -> list[int]:
    """Blind fallback used only when the answer carries no [[N]] markers at
    all. 2026-08-18: real diagnosed loss -- a bare positional guess (first
    few items gathered, then tried most-recent-few instead) both failed
    live: research sometimes finds the right source early and keeps
    exploring past it, sometimes finds it late, so neither end of the
    session is a reliable position to guess from. Score every gathered
    item against the question's keywords/quoted-title anchors (the same
    signal _densest_windows uses to find content within a page) plus the
    answer's own keywords, and prefer whichever items actually talk about
    what the answer says -- topical relevance, not position."""
    keywords = _keywords_from(question) | _keywords_from(answer)
    anchors = _anchors_from(question)
    scored = [
        (item.index, _chunk_score(f"{item.title or ''} {item.note or ''}", keywords, anchors))
        for item in store.items
    ]
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    relevant = [idx for idx, score in ranked if score > 0][:5]
    return relevant or [item.index for item in store.items[-5:]]


def _remap_citation_markers(text: str, remap: dict[int, int]) -> str:
    # 2026-08-18: real diagnosed loss -- the model cites evidence using our
    # internal EvidenceStore index (assigned once per item across the whole
    # research session, so it can run well past the handful of items that
    # actually end up in the final, capped citations list). The platform
    # numbers materialized citations by their position in that final list,
    # so a marker like [[22]] pointing at a real, correctly-selected citation
    # reads as a hallucinated reference once only 5 citations exist -- the
    # judge explicitly called this out as "nonsensical". Rewrite every
    # marker to the 1-based position it actually holds in the final list;
    # drop markers for indices that didn't make the cut rather than leave a
    # dangling reference.
    def _replace(match: re.Match[str]) -> str:
        new = remap.get(int(match.group(1)))
        return f"[[{new}]]" if new is not None else ""

    return _CITATION_INDEX_RE.sub(_replace, text)


def _build_citations(question: str, text: str, store: EvidenceStore) -> tuple[str, list[CitationRef] | None]:
    if not store.items:
        return text, None
    # Prefer the evidence numbers the model actually marked; if it marked
    # none, rank by topical relevance rather than fabricate an uncited
    # answer against evidence that was in fact used.
    cited_indices = _extract_cited_indices(text, store)
    if not cited_indices:
        cited_indices = _relevance_ranked_indices(question, text, store)
    # 2026-08-20: real diagnosed loss -- see _answer_anchors above. Anchors
    # from the final answer catch a "discovered, not named" target the
    # question-time anchors/keywords can't, for the fallback re-slice below.
    answer_keywords = _keywords_from(question) | _keywords_from(text)
    answer_anchors = _anchors_from(question) + _answer_anchors(text)
    refs: list[CitationRef] = []
    remap: dict[int, int] = {}
    total_chars = 0
    for idx in cited_indices[:MAX_CITATIONS]:
        item = store.get(idx)
        if item is None:
            continue
        note_len = len(item.note) if item.note else 0
        # 2026-08-18: real regression -- the platform materializes a
        # citation's FULL note server-side unless sliced, and once
        # MAX_PAGE_CHARS grew to fit large tables, citing several such
        # pages pushed the total past the platform's 120,000-char citation
        # cap and got the whole response rejected. Slice large notes and
        # stop adding citations before the running total gets close.
        if note_len > MAX_CITATION_SLICE_CHARS:
            # 2026-08-18: a second real regression on top of the first --
            # slicing unconditionally from offset 0 cited page-head
            # boilerplate (e.g. a PDF's table of contents) instead of the
            # actual keyword-dense content the model read further in the
            # page. Prefer retained_spans (the model's own note_evidence
            # quotes -- verified proof for a specific claim); next, re-score
            # this item's full note using anchors from the final ANSWER
            # (not just the question -- see _answer_anchors above, and its
            # comment for the real diagnosed loss this fixes: a
            # "discovered, not named" target has no question-time anchor to
            # find it by); only fall back to the stale fetch-time
            # relevant_spans, then the literal head, when neither locates
            # anything relevant.
            spans = (
                item.retained_spans
                or _citation_spans(item.note or "", answer_keywords, answer_anchors)
                or item.relevant_spans
            )
            slices = []
            budget = MAX_CITATION_SLICE_CHARS
            for start, end in spans:
                if budget <= 0:
                    break
                end = min(end, note_len)
                span_len = min(end - start, budget)
                if span_len <= 0:
                    continue
                slices.append(CitationSlice(start=start, end=start + span_len))
                budget -= span_len
            if not slices:
                slices = [CitationSlice(start=0, end=MAX_CITATION_SLICE_CHARS)]
            contributed = sum(s.end - s.start for s in slices)
        else:
            slices = []
            contributed = note_len
        if total_chars + contributed > MAX_TOTAL_CITATION_CHARS:
            break
        total_chars += contributed
        remap[idx] = len(refs) + 1
        refs.append(CitationRef(receipt_id=item.receipt_id, result_id=item.result_id, slices=slices))
    if not refs:
        return text, None
    return _remap_citation_markers(text, remap), refs


# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------


async def _build_structured_output(
    query: Query,
    store: EvidenceStore,
    text_answer: str,
    state: RunState,
) -> Any:
    schema = query.output_schema
    if state.budget_is_low() or state.past_hard_deadline():
        return _best_effort_structured(schema, text_answer)

    evidence_block = _evidence_block(store)
    schema_json = json.dumps(schema, separators=(",", ":"))
    base_instruction = (
        "Respond with a single JSON value only -- no prose, no markdown code "
        "fences, no explanation before or after. The JSON value must validate "
        f"against this JSON Schema (Draft 2020-12):\n{schema_json}\n\n"
        "Preserve exact numeric precision and formatting exactly as the "
        "question specifies -- do not round, truncate, or switch a value "
        "into exponent/scientific notation unless the question asked for "
        "that notation. Copy exact figures verbatim rather than "
        "re-deriving or approximating them. Only use a value that is "
        "actually present in the evidence below -- never invent a "
        "placeholder (like -1, 0, or \"unknown\") for a field you could "
        "not find and pass it off as real.\n\n"
        f"Question: {query.text}\n\nEvidence:\n{evidence_block}\n\n"
        f"Reference answer to structure (for content, not format): {text_answer}"
    )
    feedback = ""
    for _attempt in range(MAX_STRUCTURED_ATTEMPTS):
        if state.budget_is_low() or state.past_hard_deadline():
            break
        prompt = base_instruction
        if feedback:
            prompt = f"{base_instruction}\n\nYour previous attempt was invalid: {feedback}\nReturn corrected JSON only."
        raw = await _call_synthesis(
            [
                {"role": "system", "content": "You output only valid, schema-conformant JSON."},
                {"role": "user", "content": prompt},
            ],
            state,
        )
        if not raw:
            feedback = "no response received"
            continue
        parsed = _parse_json_loose(raw)
        if parsed is None:
            feedback = "response was not valid JSON"
            continue
        error = _validate_against_schema(parsed, schema)
        if error is None:
            return parsed
        feedback = error

    # 2026-08-18: real diagnosed loss -- when both reasoning-from-evidence
    # attempts above failed (invalid JSON / schema mismatch), this used to
    # fall straight to _best_effort_structured, which fills every number/
    # array field with a bare 0/[] regardless of whether text_answer
    # already stated the real value. Two real tasks lost this way despite
    # gathering correct evidence: the loop's own text_answer had the right
    # content, it just never made it into the JSON. One more attempt at a
    # much simpler task -- pure extraction from already-computed prose,
    # no fresh reasoning over raw evidence -- before giving up for real.
    if not (state.budget_is_low() or state.past_hard_deadline()):
        extract_prompt = (
            "The text below already contains the correct, fully-reasoned "
            "answer to a question -- your only job is to extract its "
            "stated values into JSON matching this schema (Draft "
            f"2020-12):\n{schema_json}\n\n"
            "Use the exact values already given in the text below for "
            "every field -- do not recompute, guess, or invent a "
            "placeholder (like -1, 0, or an empty list) for a field the "
            "text does state a real value for. Only leave a field at its "
            "schema-default if the text truly never mentions it.\n\n"
            f"Text:\n{text_answer}"
        )
        raw = await _call_synthesis(
            [
                {"role": "system", "content": "You output only valid, schema-conformant JSON."},
                {"role": "user", "content": extract_prompt},
            ],
            state,
        )
        if raw:
            parsed = _parse_json_loose(raw)
            if parsed is not None and _validate_against_schema(parsed, schema) is None:
                return parsed

    return _best_effort_structured(schema, text_answer)


def _parse_json_loose(raw: str) -> Any | None:
    candidate = raw.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


# `jsonschema` is not in the platform's accepted miner-script import subset
# (upload is rejected at the AST-policy stage before the script ever runs),
# so this is a hand-rolled structural check covering the shapes the schemas
# in practice use: type, object required/properties, and array items.
# 2026-08-18: real diagnosed loss, seen twice in one live vs-champion sample
# -- the structured-output model wrote its full hedge/reasoning into a
# single required field (once a top-level string field with every sibling
# field left empty/zero, once as the sole item of a required array) instead
# of extracting the one short value each field asked for. This happened at
# the PRIMARY generation call, not just the last-resort placeholder fallback
# already hardened elsewhere -- `_validate_against_schema` only checked
# type/required-presence, so a syntactically-valid but semantically
# garbage-shaped JSON object sailed through as "valid" every time.
_REASONING_DUMP_LEAD_RE = re.compile(
    r"^(?:i have to be honest|i cannot|i'm unable|based on the evidence"
    r"|based on the (?:gathered|available) evidence)",
    re.IGNORECASE,
)


def _looks_like_reasoning_dump(value: str) -> bool:
    if len(value) < 300:
        return False
    return value.count("\n") >= 2 or bool(_REASONING_DUMP_LEAD_RE.match(value.strip()))


def _is_default_value(value: Any, prop_schema: Any) -> bool:
    prop_type = prop_schema.get("type") if isinstance(prop_schema, dict) else None
    if prop_type == "string":
        return value == ""
    if prop_type in ("number", "integer"):
        return value == 0
    if prop_type == "boolean":
        return value is False
    if prop_type == "array":
        return value == []
    if prop_type == "object":
        return value == {}
    return value in ("", 0, False, [], {}, None)


def _validate_against_schema(value: Any, schema: Any, *, _path: str = "value") -> str | None:
    if not isinstance(schema, dict):
        return None
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            return f"{_path} must be an object"
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    return f"{_path} is missing required field '{key}'"
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in value:
                    error = _validate_against_schema(value[key], subschema, _path=f"{_path}.{key}")
                    if error:
                        return error
        # 2026-08-20: real diagnosed loss, traced via recorded validator
        # results -- a task with additionalProperties: false failed with
        # "miner returned invalid response payload" on two separate runs,
        # under two different provider conditions, with our own validation
        # showing nothing wrong. Root cause: this hand-rolled check is only
        # a best-effort SANDBOX-side self-check (jsonschema isn't in the
        # allowed import subset); the real, authoritative Draft 2020-12
        # validation happens at the trusted host, OUTSIDE our own try/except,
        # using the actual jsonschema library -- so a response that passes
        # our checks but violates a constraint we don't enforce (like this
        # one) ships anyway and gets rejected downstream where we can never
        # see why. Catch the extra-field case here so the repair-retry loop
        # gets a real chance to fix it before we ever return.
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            extra = [key for key in value if key not in properties]
            if extra:
                return (
                    f"{_path} contains field(s) not allowed by the schema: "
                    f"{', '.join(sorted(extra))} -- this schema does not "
                    "permit extra properties. Remove them; only use the "
                    "exact field names listed in the schema."
                )
        # 2026-08-18: real diagnosed loss, three independent instances (our
        # own run plus two from the current champion's real production
        # history) -- every required field simultaneously left at its
        # type's empty/zero default (0, "", [], false), a technically valid
        # but totally uninformative object that reads as a full give-up.
        # Boolean-only schemas are exempt: two required booleans both
        # correctly being false is an ordinary, common real answer, not a
        # red flag -- only flag when at least one non-boolean field is also
        # defaulted alongside it.
        if isinstance(required, list) and len(required) >= 2 and isinstance(properties, dict):
            non_bool_required = [
                key
                for key in required
                if isinstance(properties.get(key), dict) and properties[key].get("type") != "boolean"
            ]
            if non_bool_required and all(
                _is_default_value(value.get(key), properties.get(key)) for key in required
            ):
                return (
                    f"{_path} has every required field left at its empty/"
                    "zero default (0, empty string, empty list, or false) "
                    "-- that is almost never the real answer. Find and use "
                    "the actual values you already gathered from evidence "
                    "instead of leaving everything blank."
                )
        return None
    if schema_type == "array":
        if not isinstance(value, list):
            return f"{_path} must be an array"
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                error = _validate_against_schema(item, items_schema, _path=f"{_path}[{i}]")
                if error:
                    return error
        return None
    if schema_type == "string":
        if not isinstance(value, str):
            return f"{_path} must be a string"
        if _looks_like_reasoning_dump(value):
            return (
                f"{_path} contains a long free-text explanation instead of "
                "the one specific short value this field expects -- extract "
                "just that value and put it here, nothing else"
            )
        return None
    if schema_type == "integer":
        return None if isinstance(value, int) and not isinstance(value, bool) else f"{_path} must be an integer"
    if schema_type == "number":
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        return None if is_number else f"{_path} must be a number"
    if schema_type == "boolean":
        return None if isinstance(value, bool) else f"{_path} must be a boolean"
    return None


def _array_items_type(prop_schema: Any) -> str | None:
    items_schema = prop_schema.get("items") if isinstance(prop_schema, dict) else None
    return items_schema.get("type") if isinstance(items_schema, dict) else None


def _best_effort_structured(schema: Any, text_answer: str) -> Any:
    if isinstance(schema, dict) and schema.get("type") == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        result: dict[str, Any] = {}
        # 2026-08-18: real diagnosed champion loss on this exact rescue path
        # -- a schema with two distinct string fields (a short title, a short
        # date span) both got the entire raw text_answer dumped into them
        # verbatim, and the judge read the duplicated wall of text as
        # "completely hallucinated/garbage" and scored it 0.0 outright, even
        # though the real answer was sitting in the loop's own prose. Giving
        # the raw text to only the FIRST string field and leaving the rest
        # empty reads as an honest gap, not fabricated duplicate content.
        # A second real champion loss the same day: a required array field
        # (a list of navigation aids) shipped as a bare `[]` while the
        # model's own text answer already stated the exact three items --
        # an empty list is a silent, avoidable zero exactly like the string
        # case above, so a plain string-item array gets the same one-shot
        # raw-text rescue instead of being emptied unconditionally.
        used_text_answer = False
        if isinstance(properties, dict) and isinstance(required, list):
            for key in required:
                prop_schema = properties.get(key)
                prop_type = prop_schema.get("type") if isinstance(prop_schema, dict) else None
                if prop_type == "string" and not used_text_answer and text_answer:
                    result[key] = text_answer[:2000]
                    used_text_answer = True
                elif (
                    prop_type == "array"
                    and not used_text_answer
                    and text_answer
                    and _array_items_type(prop_schema) in (None, "string")
                ):
                    result[key] = [text_answer[:2000]]
                    used_text_answer = True
                else:
                    result[key] = _placeholder_for_schema(prop_schema)
        return result
    if isinstance(schema, dict) and schema.get("type") == "array":
        if text_answer and _array_items_type(schema) in (None, "string"):
            return [text_answer[:2000]]
        return []
    return {"answer": text_answer[:2000]}


def _placeholder_for_schema(prop_schema: Any) -> Any:
    prop_type = prop_schema.get("type") if isinstance(prop_schema, dict) else None
    if prop_type == "string":
        return ""
    if prop_type in ("number", "integer"):
        return 0
    if prop_type == "boolean":
        return False
    if prop_type == "array":
        return []
    if prop_type == "object":
        return {}
    return ""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

"""
Canonical section taxonomy (user-defined).

We intentionally keep this aligned with your MCM paper-writing workflow:

Title, Summary, Contents, Introduction, Assumptions and Justifications, Notations,
Preparation, Body, Sensitivity Analysis, Strengths and Weaknesses, Conclusion, Reference, Others.

Important: Many papers were converted from PDF -> Markdown and all headings became H1 (`# ...`),
so we rely on heading text patterns + document order (state machine), not markdown heading levels.
"""

# Canonical sections
SECTION_TITLE = "Title"
SECTION_SUMMARY = "Summary"
SECTION_CONTENTS = "Contents"
SECTION_INTRODUCTION = "Introduction"
SECTION_ASSUMPTIONS = "Assumptions and Justifications"
SECTION_NOTATIONS = "Notations"
SECTION_PREPARATION = "Preparation"
SECTION_BODY = "Body"
SECTION_SENSITIVITY = "Sensitivity Analysis"
SECTION_STRENGTHS_WEAKNESSES = "Strengths and Weaknesses"
SECTION_CONCLUSION = "Conclusion"
SECTION_REFERENCE = "Reference"
SECTION_OTHERS = "Others"

SECTION_ORDER = [
    SECTION_TITLE,
    SECTION_SUMMARY,
    SECTION_CONTENTS,
    SECTION_INTRODUCTION,
    SECTION_ASSUMPTIONS,
    SECTION_NOTATIONS,
    SECTION_PREPARATION,
    SECTION_BODY,
    SECTION_SENSITIVITY,
    SECTION_STRENGTHS_WEAKNESSES,
    SECTION_CONCLUSION,
    SECTION_REFERENCE,
    SECTION_OTHERS,
]


def normalize_heading(raw: str) -> str:
    """Normalize heading for matching: collapse whitespace + trim."""
    return re.sub(r"\s+", " ", raw.strip())


def normalize_title(raw: str) -> str:
    """Normalize a paper title line (from papers_titles.csv) for matching."""
    s = raw.strip()
    # Strip surrounding quotes added by CSV exporters.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    # Some titles in the file use repeated quotes; reduce them.
    s = s.replace('""', '"').strip()
    return normalize_heading(s).lower()


# Match leading section numbers like:
# - "1 Introduction"
# - "1. Introduction"
# - "1.2 Data"
# - "1.2.3 Details"
#
# We capture the *full* numeric prefix for grouping (e.g. "4.5" for
# "4.5 Sensitivity Analysis"). Many PDF->MD conversions lose heading levels,
# so numeric prefixes are our best signal for "this heading belongs under that section".
_RE_LEADING_SECTION_NUMBER = re.compile(r"^\s*((?:\d+)(?:\.\d+)*)\.?\s*(.*)$")


@dataclass(frozen=True)
class HeadingInfo:
    raw: str
    normalized: str
    number_prefix: Optional[tuple[int, ...]]
    major_number: Optional[int]
    title_without_number: str


def parse_heading(raw: str) -> HeadingInfo:
    normalized = normalize_heading(raw)
    number_prefix: Optional[tuple[int, ...]] = None
    major: Optional[int] = None
    title = normalized
    m = _RE_LEADING_SECTION_NUMBER.match(normalized)
    if m:
        try:
            prefix_str = m.group(1)
            parts = [int(p) for p in prefix_str.split(".") if p]
            if parts:
                number_prefix = tuple(parts)
                major = parts[0]
        except ValueError:
            number_prefix = None
            major = None
        title = m.group(2).strip() if m.group(2) else ""
    return HeadingInfo(
        raw=raw,
        normalized=normalized,
        number_prefix=number_prefix,
        major_number=major,
        title_without_number=title,
    )


# --- Heading detectors (keyword-based) ---

# Strict summary heading: avoid matching "AI Usage Summary", "Executive Summary:" in a memo, etc.
_RE_SUMMARY = re.compile(r"^\s*summary\s*:?\s*$", re.I)
_RE_CONTENTS = re.compile(r"^\s*(contents|content|table\s+of\s+contents)\s*:?\s*$", re.I)

_RE_INTRO_KEYWORDS = re.compile(
    r"\bintroduction\b|\bbackground\b|\bliterature\s+review\b|\brestatement\b|\bclarification\b|\bour\s+work\b",
    re.I,
)

_RE_ASSUMPTIONS = re.compile(
    r"\bassumption\b|\bassumptions\b|\bjustification\b|\bjustifications\b|\bexplanation\b|\bexplanations\b",
    re.I,
)

_RE_NOTATIONS = re.compile(
    r"\bnotation\b|\bnotations\b|\bsymbol\b|\bsymbols\b|\bglossary\b|\bglossaries\b|\bnomenclature\b",
    re.I,
)

_RE_PREPARATION = re.compile(
    r"\bmodel\s+preparation\b|\bdata\s+preparation\b|\bdata\s+preprocessing\b|\bdata\s+pre-processing\b"
    r"|\bdata\s+processing\b|\bdata\s+cleaning\b|\bdata\s+collection\b|\bdata\s+source\b|\bdata\s+sources\b"
    r"|\bpreparation\s+for\s+model(?:ing|ling)\b|\bpreparation\b|\bpreprocess(?:ing)?\b|\bpre-processing\b|\bpreprocessing\b",
    re.I,
)

_RE_MEMO = re.compile(r"\bmemo\b|\bmemorandum\b|\bletter\b", re.I)
# Match both plural forms: "Appendices" (common) and "Appendixes" (rare).
_RE_APPENDIX = re.compile(r"^\s*(?:appendix(?:es)?|appendices)\b", re.I)
_RE_AI_REPORT = re.compile(r"\breport\s+on\s+use\s+of\s+ai\b|\bai\s+report\b", re.I)
_RE_ACK = re.compile(r"\backnowledg\w*\b", re.I)

_RE_REFERENCE = re.compile(r"^\s*(?:references?|bibliography|works\s+cited)\b", re.I)

_RE_CONCLUSION_START = re.compile(r"^\s*(?:[ivx]+\.\s*)?conclusion(?:s)?\b", re.I)
_RE_CONCLUSION_PHRASE = re.compile(
    r"^\s*(?:discussion|results?)\b.*\bconclusion(?:s)?\b",
    re.I,
)

_RE_STRENGTH_START = re.compile(r"^\s*strengths?\b", re.I)
_RE_WEAKNESS_START = re.compile(r"^\s*weakness(?:es)?\b", re.I)

# Detect a Sensitivity Analysis section heading (NOT every 'sensitivity' mention).
# - strong match: contains the phrase 'sensitivity analysis'
# - fallback: contains 'sensitivity' and one of (analysis/test/evaluation/assessment)
_RE_SENSITIVITY_ANALYSIS = re.compile(r"\bsensitivity\s+analysis\b", re.I)
_RE_SENSITIVITY_FALLBACK = re.compile(
    r"\bsensitivity\b.*\b(?:analysis|test(?:ing)?|evaluation|assessment)\b|\b(?:analysis|test(?:ing)?|evaluation|assessment)\b.*\bsensitivity\b",
    re.I,
)
_RE_ROBUSTNESS_ANALYSIS = re.compile(r"\brobustness\b.*\b(?:analysis|test(?:ing)?|evaluation)\b", re.I)

def is_summary_heading(heading: str) -> bool:
    return bool(_RE_SUMMARY.match(heading))


def is_contents_heading(heading: str) -> bool:
    return bool(_RE_CONTENTS.match(heading))


def is_memo_heading(heading: str) -> bool:
    return bool(_RE_MEMO.search(heading))


def is_appendix_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    return bool(_RE_APPENDIX.match(t))


def is_ai_report_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    return bool(_RE_AI_REPORT.search(t))


def is_ack_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    return bool(_RE_ACK.search(t))


def is_reference_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    return bool(_RE_REFERENCE.match(t))


def is_conclusion_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    return bool(_RE_CONCLUSION_START.match(t) or _RE_CONCLUSION_PHRASE.match(t))


def is_strengths_weaknesses_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    if _RE_STRENGTH_START.match(t) or _RE_WEAKNESS_START.match(t):
        return True
    # e.g. "Evaluation of Strengths and Weaknesses"
    tl = t.lower()
    return ("strength" in tl) and ("weakness" in tl)


def is_sensitivity_heading(info: HeadingInfo) -> bool:
    t = (info.title_without_number or info.normalized).strip()
    if _RE_SENSITIVITY_ANALYSIS.search(t):
        return True
    if _RE_SENSITIVITY_FALLBACK.search(t):
        return True
    # Treat robustness analysis as sensitivity family (common combined section name).
    if _RE_ROBUSTNESS_ANALYSIS.search(t):
        return True
    return False


def infer_pre_body_section(info: HeadingInfo) -> Optional[str]:
    """Infer one of the pre-body sections based on heading text (no order/state)."""
    h = info.normalized
    # Assumptions / Notations / Preparation have higher priority than the numeric "1.* => intro".
    if _RE_ASSUMPTIONS.search(h):
        return SECTION_ASSUMPTIONS
    if _RE_NOTATIONS.search(h):
        return SECTION_NOTATIONS
    if _RE_PREPARATION.search(h):
        return SECTION_PREPARATION
    if info.major_number == 1 or _RE_INTRO_KEYWORDS.search(h):
        return SECTION_INTRODUCTION
    return None

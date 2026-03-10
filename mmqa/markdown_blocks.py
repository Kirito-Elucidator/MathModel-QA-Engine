from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownBlock:
    level: int
    heading: str
    content: str
    start_line: int  # 1-based
    end_line: int  # 1-based

    def as_markdown(self) -> str:
        prefix = "#" * self.level
        body = self.content.rstrip()
        if body:
            return f"{prefix} {self.heading}\n\n{body}\n"
        return f"{prefix} {self.heading}\n"


def iter_markdown_blocks(md_text: str) -> Iterable[MarkdownBlock]:
    """Split markdown into heading-defined blocks (H1..H6).

    This is intentionally simple and deterministic: a new block starts at every
    markdown heading line (e.g. '# Summary').
    """
    lines = md_text.splitlines()

    cur_level: Optional[int] = None
    cur_heading: Optional[str] = None
    cur_start: int = 1
    cur_buf: list[str] = []

    for idx0, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if not m:
            if cur_heading is not None:
                cur_buf.append(line)
            continue

        # Start of a new heading block.
        if cur_heading is not None:
            end_line = idx0  # previous line is idx0-1, but line numbers are 1-based and inclusive
            yield MarkdownBlock(
                level=cur_level or 1,
                heading=cur_heading,
                content="\n".join(cur_buf).rstrip("\n"),
                start_line=cur_start,
                end_line=end_line,
            )

        cur_level = len(m.group(1))
        cur_heading = m.group(2).strip()
        cur_start = idx0 + 1  # heading line number (1-based)
        cur_buf = []

    if cur_heading is not None:
        yield MarkdownBlock(
            level=cur_level or 1,
            heading=cur_heading,
            content="\n".join(cur_buf).rstrip("\n"),
            start_line=cur_start,
            end_line=len(lines),
        )


def find_first_summary_block_index(blocks: list[MarkdownBlock]) -> Optional[int]:
    for i, b in enumerate(blocks):
        if b.heading.strip().lower() == "summary":
            return i
    return None


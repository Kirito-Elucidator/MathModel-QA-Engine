from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from llama_index.core.schema import TextNode

from .markdown_blocks import MarkdownBlock, find_first_summary_block_index, iter_markdown_blocks
from .papers_csv import PaperRow
from .sections import (
    SECTION_BODY,
    SECTION_CONCLUSION,
    SECTION_CONTENTS,
    SECTION_ORDER,
    SECTION_OTHERS,
    SECTION_REFERENCE,
    SECTION_SENSITIVITY,
    SECTION_STRENGTHS_WEAKNESSES,
    SECTION_SUMMARY,
    SECTION_TITLE,
    infer_pre_body_section,
    is_ack_heading,
    is_ai_report_heading,
    is_appendix_heading,
    is_conclusion_heading,
    is_contents_heading,
    is_memo_heading,
    is_reference_heading,
    is_sensitivity_heading,
    is_summary_heading,
    is_strengths_weaknesses_heading,
    normalize_title,
    parse_heading,
)


NodeMode = Literal["block", "section"]


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    doc_id: str
    year: int
    problem: str
    problem_id: str
    section: str
    heading: str
    source_path: str
    start_line: int
    end_line: int
    text: str

    def to_text_node(self) -> TextNode:
        return TextNode(
            id_=self.node_id,
            text=self.text,
            metadata={
                "doc_id": self.doc_id,
                "year": self.year,
                "problem": self.problem,
                "problem_id": self.problem_id,
                "section": self.section,
                "heading": self.heading,
                "source_path": self.source_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
        )


def _make_node_id(doc_id: str, section: str, heading: str, idx: int) -> str:
    # Stable, readable ids (no hashing) to ease debugging.
    safe_section = section.replace(" ", "_").replace("/", "_")
    safe_heading = "".join(ch for ch in heading if ch.isalnum() or ch in (" ", "_", "-")).strip().replace(" ", "_")
    safe_heading = safe_heading[:48] if safe_heading else "block"
    return f"{doc_id}__{safe_section}__{idx:04d}__{safe_heading}"


def split_paper_to_nodes(
    paper: PaperRow,
    *,
    repo_root: str | Path,
    mode: NodeMode = "block",
    title_norm_set: Optional[set[str]] = None,
) -> list[NodeRecord]:
    repo_root = Path(repo_root)
    src_path = (repo_root / paper.path).resolve()
    md_text = src_path.read_text(encoding="utf-8", errors="ignore")

    blocks = list(iter_markdown_blocks(md_text))
    # Summary heading is usually '# Summary' at the start.
    summary_idx = None
    for i, b in enumerate(blocks):
        if is_summary_heading(parse_heading(b.heading).normalized):
            summary_idx = i
            break

    # Decide which block is the paper title (if present as a heading).
    title_norm_set = title_norm_set or set()
    title_idx: Optional[int] = None
    if blocks:
        h0 = normalize_title(blocks[0].heading)
        if h0 in title_norm_set:
            title_idx = 0
        elif summary_idx is not None and summary_idx > 0:
            # Fallback: treat the first non-special heading before Summary as the title.
            for i in range(summary_idx):
                if (
                    is_summary_heading(parse_heading(blocks[i].heading).normalized)
                    or is_contents_heading(parse_heading(blocks[i].heading).normalized)
                    or is_memo_heading(parse_heading(blocks[i].heading).normalized)
                    or is_appendix_heading(parse_heading(blocks[i].heading))
                    or is_ai_report_heading(parse_heading(blocks[i].heading))
                    or is_ack_heading(parse_heading(blocks[i].heading))
                    or is_reference_heading(parse_heading(blocks[i].heading))
                ):
                    continue
                title_idx = i
                break

    def assign_sections() -> list[str]:
        sections: list[str] = []
        in_reference = False
        in_ai_report = False
        in_appendix = False
        sensitivity_prefix: Optional[tuple[int, ...]] = None
        sensitivity_until_tail = False

        for i, b in enumerate(blocks):
            info = parse_heading(b.heading)
            h = info.normalized

            if title_idx is not None and i == title_idx:
                sections.append(SECTION_TITLE)
                continue

            if is_summary_heading(h):
                sections.append(SECTION_SUMMARY)
                continue

            if is_contents_heading(h):
                sections.append(SECTION_CONTENTS)
                continue

            # AI report section is always a tail section; keep everything after it as Others.
            if in_ai_report:
                sections.append(SECTION_OTHERS)
                continue

            # AI Report starts a tail region (its subheadings like "Query 1: ..." have arbitrary titles).
            if is_ai_report_heading(info):
                sections.append(SECTION_OTHERS)
                in_ai_report = True
                in_reference = False
                in_appendix = False
                sensitivity_prefix = None
                sensitivity_until_tail = False
                continue

            # Reference can (rarely) appear after appendices; always allow it to start when seen.
            if is_reference_heading(info):
                sections.append(SECTION_REFERENCE)
                in_reference = True
                in_appendix = False
                sensitivity_prefix = None
                sensitivity_until_tail = False
                continue

            # Appendices usually mean we've left the main paper; treat everything after as Others
            # unless a later explicit Reference/AI section appears (handled above).
            if is_appendix_heading(info):
                sections.append(SECTION_OTHERS)
                in_appendix = True
                in_reference = False
                sensitivity_prefix = None
                sensitivity_until_tail = False
                continue

            if in_appendix:
                sections.append(SECTION_OTHERS)
                continue

            # Always-tagged "Others" headings (not part of the main paper flow).
            if is_memo_heading(h) or is_ack_heading(info):
                sections.append(SECTION_OTHERS)
                continue

            if in_reference:
                sections.append(SECTION_REFERENCE)
                continue

            # Strength/Weakness, Conclusion are explicit tail sections.
            if is_strengths_weaknesses_heading(info):
                sections.append(SECTION_STRENGTHS_WEAKNESSES)
                sensitivity_prefix = None
                sensitivity_until_tail = False
                continue

            if is_conclusion_heading(info):
                sections.append(SECTION_CONCLUSION)
                sensitivity_prefix = None
                sensitivity_until_tail = False
                continue

            # Sensitivity Analysis: group by numeric prefix when available (PDF->MD often flattens headings).
            if sensitivity_prefix is not None:
                if info.number_prefix is not None and info.number_prefix[: len(sensitivity_prefix)] == sensitivity_prefix:
                    sections.append(SECTION_SENSITIVITY)
                    continue
                sensitivity_prefix = None

            if sensitivity_until_tail:
                sections.append(SECTION_SENSITIVITY)
                continue

            if is_sensitivity_heading(info):
                sections.append(SECTION_SENSITIVITY)
                if info.number_prefix is not None:
                    sensitivity_prefix = info.number_prefix
                    sensitivity_until_tail = False
                else:
                    sensitivity_until_tail = True
                continue

            pre = infer_pre_body_section(info)
            if pre is not None:
                sections.append(pre)
                continue

            sections.append(SECTION_BODY)
        return sections

    if mode == "block":
        out: list[NodeRecord] = []
        block_sections = assign_sections()
        for i, b in enumerate(blocks):
            section = block_sections[i]
            out.append(
                NodeRecord(
                    node_id=_make_node_id(paper.doc_id, section, b.heading, i),
                    doc_id=paper.doc_id,
                    year=paper.year,
                    problem=paper.problem,
                    problem_id=paper.problem_id,
                    section=section,
                    heading=b.heading,
                    source_path=paper.path,
                    start_line=b.start_line,
                    end_line=b.end_line,
                    text=b.as_markdown(),
                )
            )
        return out

    if mode == "section":
        # Aggregate blocks by canonical section, preserving document order.
        agg: dict[str, list[MarkdownBlock]] = {s: [] for s in SECTION_ORDER}
        block_sections = assign_sections()
        for i, b in enumerate(blocks):
            section = block_sections[i]
            agg.setdefault(section, []).append(b)

        out: list[NodeRecord] = []
        idx = 0
        for section in SECTION_ORDER:
            sec_blocks = agg.get(section, [])
            if not sec_blocks:
                continue
            text = "\n\n".join(b.as_markdown().rstrip() for b in sec_blocks).rstrip() + "\n"
            start_line = min(b.start_line for b in sec_blocks)
            end_line = max(b.end_line for b in sec_blocks)
            out.append(
                NodeRecord(
                    node_id=_make_node_id(paper.doc_id, section, section, idx),
                    doc_id=paper.doc_id,
                    year=paper.year,
                    problem=paper.problem,
                    problem_id=paper.problem_id,
                    section=section,
                    heading=section,
                    source_path=paper.path,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                )
            )
            idx += 1
        return out

    raise ValueError(f"Unsupported mode: {mode}")


def export_nodes_jsonl(nodes: Iterable[NodeRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(asdict(n), ensure_ascii=False) + "\n")


def export_text_nodes_jsonl(nodes: Iterable[NodeRecord], out_path: str | Path) -> None:
    """Export nodes in a LlamaIndex-friendly jsonl shape (text + metadata)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for n in nodes:
            tn = n.to_text_node()
            # Keep file small: store only the public fields we rely on.
            payload = {
                "id_": tn.id_,
                "text": tn.text,
                "metadata": tn.metadata,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_text_nodes_jsonl(path: str | Path) -> list[TextNode]:
    """Load TextNode objects from the JSONL produced by export_text_nodes_jsonl()."""
    path = Path(path)
    nodes: list[TextNode] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            nodes.append(TextNode(id_=obj["id_"], text=obj["text"], metadata=obj.get("metadata", {})))
    return nodes

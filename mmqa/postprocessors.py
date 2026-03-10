from __future__ import annotations

from typing import List, Optional, Set

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from .sections import SECTION_SUMMARY, SECTION_TITLE


class DropLowInfoNodesPostprocessor(BaseNodePostprocessor):
    """Drop nodes that are structurally useful but content-empty (e.g. heading-only).

    This helps avoid retrieval returning only headings like "# 3 Models" which
    provide no model details, causing the LLM to (correctly) refuse to answer.
    """

    drop_heading_only: bool = Field(default=True)
    keep_heading_only_sections: Set[str] = Field(default_factory=lambda: {SECTION_TITLE, SECTION_SUMMARY})

    drop_toc: bool = Field(default=True)
    toc_headings: Set[str] = Field(default_factory=lambda: {"contents", "content", "table of contents"})

    min_non_empty_lines: int = Field(default=2)

    @classmethod
    def class_name(cls) -> str:
        return "DropLowInfoNodesPostprocessor"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        out: List[NodeWithScore] = []
        for nws in nodes:
            meta = nws.metadata or {}
            section = str(meta.get("section", "") or "")
            heading = str(meta.get("heading", "") or "").strip()

            if self.drop_toc and heading.lower() in self.toc_headings:
                continue

            if self.drop_heading_only and section not in self.keep_heading_only_sections:
                try:
                    text = nws.text
                except Exception:
                    text = nws.get_content()
                non_empty_lines = [ln for ln in text.splitlines() if ln.strip()]
                if len(non_empty_lines) < self.min_non_empty_lines:
                    continue

            out.append(nws)
        return out

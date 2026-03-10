from __future__ import annotations

import argparse
from pathlib import Path

from .node_export import export_nodes_jsonl, export_text_nodes_jsonl, split_paper_to_nodes
from .papers_csv import read_papers_csv
from .sections import SECTION_ORDER
from .titles_csv import read_titles_csv


def _cmd_split(args: argparse.Namespace) -> int:
    rows = read_papers_csv(args.papers_csv)
    title_norm_set = read_titles_csv(args.titles_csv)
    all_nodes = []
    for r in rows:
        all_nodes.extend(
            split_paper_to_nodes(
                r,
                repo_root=args.repo_root,
                mode=args.mode,
                title_norm_set=title_norm_set,
            )
        )

    export_nodes_jsonl(all_nodes, args.out_jsonl)
    if args.out_text_nodes_jsonl:
        export_text_nodes_jsonl(all_nodes, args.out_text_nodes_jsonl)

    # Print a tiny summary for sanity.
    by_section = {s: 0 for s in SECTION_ORDER}
    for n in all_nodes:
        by_section[n.section] = by_section.get(n.section, 0) + 1
    print(f"papers: {len(rows)}")
    print(f"nodes:  {len(all_nodes)} (mode={args.mode})")
    for s in SECTION_ORDER:
        c = by_section.get(s, 0)
        if c:
            print(f"- {s}: {c}")
    print(f"wrote: {args.out_jsonl}")
    if args.out_text_nodes_jsonl:
        print(f"wrote: {args.out_text_nodes_jsonl}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmqa", description="Split math-modeling papers into canonical section nodes.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="Split all papers into nodes and export JSONL.")
    sp.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    sp.add_argument("--papers-csv", default="data/papers.csv", help="Path to papers.csv (default: data/papers.csv)")
    sp.add_argument(
        "--titles-csv",
        default="data/papers_titles.csv",
        help="Path to papers_titles.csv (default: data/papers_titles.csv)",
    )
    sp.add_argument(
        "--mode",
        choices=["block", "section"],
        default="block",
        help="block=headings as nodes; section=aggregate by canonical sections (default: block)",
    )
    sp.add_argument(
        "--out-jsonl",
        default="data/nodes/nodes.jsonl",
        help="Output JSONL path for NodeRecord (default: data/nodes/nodes.jsonl)",
    )
    sp.add_argument(
        "--out-text-nodes-jsonl",
        default="",
        help="Optional output JSONL with {id_, text, metadata} (for easier LlamaIndex ingestion).",
    )
    sp.set_defaults(func=_cmd_split)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

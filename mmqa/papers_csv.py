from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PaperRow:
    doc_id: str
    path: str
    year: int
    problem: str
    problem_id: str


def read_papers_csv(csv_path: str | Path) -> list[PaperRow]:
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"doc_id", "path", "year", "problem", "problem_id"}
        if set(reader.fieldnames or []) < required:
            raise ValueError(f"{csv_path} missing columns: {sorted(required - set(reader.fieldnames or []))}")

        rows: list[PaperRow] = []
        for r in reader:
            rows.append(
                PaperRow(
                    doc_id=r["doc_id"].strip(),
                    path=r["path"].strip(),
                    year=int(r["year"]),
                    problem=r["problem"].strip(),
                    problem_id=r["problem_id"].strip(),
                )
            )
        return rows


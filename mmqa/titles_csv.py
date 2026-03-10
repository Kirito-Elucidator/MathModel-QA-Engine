from __future__ import annotations

from pathlib import Path

from .sections import normalize_title


def read_titles_csv(path: str | Path) -> set[str]:
    """Read `papers_titles.csv` (one title per line) and return a normalized lowercase set."""
    path = Path(path)
    titles: set[str] = set()
    if not path.exists():
        return titles
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        titles.add(normalize_title(s))
    return titles


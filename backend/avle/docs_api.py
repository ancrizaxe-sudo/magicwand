"""Serve the /docs/*.md files via /api/docs so the frontend can render them."""
from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"

DOC_ORDER = [
    ("setup",              "SETUP.md",              "Setup"),
    ("project-evolution",  "PROJECT_EVOLUTION.md",  "Project Evolution"),
    ("research-paper-guide","RESEARCH_PAPER_GUIDE.md","Research Paper Guide"),
    ("model-training-guide","MODEL_TRAINING_GUIDE.md","Model Training Guide"),
    ("data-guide",         "DATA_GUIDE.md",         "Data Guide"),
    ("interview-prep",     "INTERVIEW_PREP.md",     "Interview Prep"),
]


def list_docs():
    out = []
    for slug, filename, title in DOC_ORDER:
        p = DOCS_DIR / filename
        out.append({
            "slug": slug, "filename": filename, "title": title,
            "present": p.exists(),
            "size": p.stat().st_size if p.exists() else 0,
        })
    return out


def read_doc(slug: str) -> dict | None:
    for s, filename, title in DOC_ORDER:
        if s == slug:
            p = DOCS_DIR / filename
            if not p.exists():
                return None
            return {"slug": s, "title": title, "filename": filename,
                    "markdown": p.read_text(encoding='utf-8')}
    return None

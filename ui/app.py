"""Streamlit UI entry point (coursework Q2). TODO.

Run: `streamlit run ui/app.py`
Implementation requirements in ui/README.md. Two presentation tabs (D23):
- **Paper Search** tab: calls orchestration.run_query() directly — natural-language
  paper search (understanding → parallel retrieval → review loop), papers + coverage + latency.
- **Downstream** tab: RESERVED — the downstream task type is TBD (downstream/ is an empty
  folder); the tab stays hidden/disabled until the task is defined.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PaperOdyssey", layout="wide")


def render() -> None:
    """Render the main UI (Paper Search tab active; Downstream tab reserved)."""
    raise NotImplementedError  # TODO: implement, see ui/README.md


if __name__ == "__main__":
    render()

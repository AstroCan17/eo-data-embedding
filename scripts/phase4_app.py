#!/usr/bin/env python
"""Phase 4 — Gradio demo (containerized via Dockerfile).

Minimal search UI: upload/pick a query tile -> show top-N nearest tiles from the index.

TODO:
  - load index + embeddings at startup
  - gr.Interface: query image -> search.search(...) -> gallery of nearest tiles
"""
from __future__ import annotations


def main():
    try:
        import gradio as gr
    except ImportError:
        print("Install gradio (in requirements.txt) to run the demo.")
        return

    def stub_fn(_query):
        return "Phase 4 stub — wire up FAISS retrieval. See docs/PROJECT_PLAN.md › Phase 4."

    demo = gr.Interface(fn=stub_fn, inputs=gr.Image(type="numpy"), outputs="text",
                        title="geo-embed-eo — similarity search")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

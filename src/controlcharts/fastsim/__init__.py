"""Vectorized no-LLM simulation engine for large-N experiments.

Event-level reimplementation of the `use_llm: false` simulation path.
Embeddings never run during simulation: response strings come from a finite
alphabet embedded once at finalize time. Snapshot output format is identical
to TemporalKernelHook, so run_tdkps_analysis and the figure scripts work
unchanged.
"""

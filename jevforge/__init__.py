"""jevforge: turn any Hugging Face causal LM into a Jev-style decision model
(LLM-as-Jev, arXiv 2610.02076v2)."""

from .jev import Jev

__all__ = ["Jev"]
__version__ = "0.1.0"

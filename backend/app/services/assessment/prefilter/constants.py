"""Static config for the assessment prefilter stages.
"""

from typing import Literal

# Provider + model that run the batch prefilter stages (topic relevance, dup check).
ASSESSMENT_PREFILTER_PROVIDER: Literal["openai", "google"] = "openai"
ASSESSMENT_PREFILTER_MODEL: str = "gpt-5-mini"
# ASSESSMENT_PREFILTER_MODEL: str = "gemini-3.1-flash-lite"


# File-search/vector store holding the corpus for duplicate detection.
ASSESSMENT_PREFILTER_DUPLICATE_STORE: str = "vs_6a20339fbc148191867fd06d29133278"
# ASSESSMENT_PREFILTER_DUPLICATE_STORE: str = "fileSearchStores/inquilabcorpus-782mxjcwisaz"

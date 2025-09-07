"""
This module allows to check an input or output prompt against a dataset (default deepset/prompt-injections), returning the similarity score.

Author: Antonio Battaglia
28th August 2025
"""

from typing import Any

import spacy

from ._log import get_logger
from .filtered_output import FilteredOutput
from .utils import get_nlp_model, lazy_load_dep

LOGGER = get_logger(__name__)


class NLPFilter:
    """NLPFilter class to evaluate prompts against a dataset using SpaCy.
    Args:
    language: The SpaCy language model to use. Default is "en_core_web_sm".
    dataset: The Dataset object to use for comparison. If None, it will load the default dataset (deepset/prompt-injections).
    """

    def __init__(self, language: str = "en", dataset: Any = None):
        self.language = get_nlp_model(language)

        try:
            self.nlp = spacy.load(self.language)
        except OSError:
            raise ImportError(
                f"""SpaCy language module for {language} not found locally.
                Install with: python -m spacy download {self.language}"""
            )
        if dataset is None:
            datasets = lazy_load_dep("datasets", "datasets")
            self.dataset = datasets.load_dataset("deepset/prompt-injections")["train"]
            self.dataset = self.dataset.filter(
                lambda x: x["label"] == 1
            )  # Keep only malicious prompts

    def safe_eval(self, prompt: str, threshold: float = 0.95) -> FilteredOutput:
        doc = self.nlp(prompt)
        similarity = 0.0
        most_similar_text = None
        for record in self.dataset:
            text = record["text"] if "text" in record.keys() else record[0]
            dataset_doc = self.nlp(text)
            doc_similarity = doc.similarity(dataset_doc)
            if doc_similarity > similarity:
                similarity = doc_similarity
                most_similar_text = text

            if similarity >= threshold:
                LOGGER.info(
                    f"Prompt blocked due to similarity score of {similarity} with dataset entry: {text}"
                )
                return FilteredOutput(
                    output=prompt,
                    score={"label": 1, "similarity": similarity, "matched_text": text},
                )
        return FilteredOutput(
            output=prompt,
            score={"label": 0, "similarity": similarity, "matched_text": most_similar_text},
        )


if __name__ == "__main__":
    from datasets import load_dataset

    prompt: str = load_dataset("deepset/prompt-injections")["test"][1]["text"]
    nlpfilter = NLPFilter("en")
    out = nlpfilter.safe_eval(prompt)
    print(out.score)
    print(out.output)

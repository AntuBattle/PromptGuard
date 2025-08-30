"""
This module allows to check an input or output prompt against a dataset (default deepset/prompt-injections), returning the similarity score.

Author: Antonio Battaglia
28th August 2025
"""

import spacy


class NLPFilter:
    def __init__(self, language: str = "en_core_web_sm", database=None):
        self.database = database
        self.language = language
        try:
            self.nlp = spacy.load(language)
        except OSError:
            raise ImportError(
                f"""SpaCy language module for {language} not found locally.
                Install with: python -m spacy download {self.language}"""
            )

    def safe_eval(self, prompt: str):
        doc = self.nlp(prompt)
        return doc


if __name__ == "__main__":
    prompt = "This is a sample prompt"
    nlpfilter = NLPFilter()
    nlpfilter.safe_eval("fakeprompt")

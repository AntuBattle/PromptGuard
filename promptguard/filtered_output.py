from typing import Literal


class FilteredOutput:
    """
    A class returned by the filter objects containing the sanitized or blocked prompt and an evaluation flag.

    Attributes:
        output (str): The sanitized or original prompt.
        eval (Literal["SAFE", "UNSAFE", "BLOCKED"]): The evaluation flag.
    """

    def __init__(self, output: str, score: Literal["SAFE", "UNSAFE", "BLOCKED"]):
        self.output = output
        self.score = score

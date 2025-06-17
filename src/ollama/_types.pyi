from typing import TypedDict

class Options(TypedDict, total=False):
    temperature: float
    num_predict: int
    # additional fields omitted

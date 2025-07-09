END = "END"
START = "START"


class StateGraph:
    def __init__(self) -> None:  # noqa: ANN101
        self.nodes: dict[str, object] = {}

import uuid
from typing import Any, cast
from collections.abc import Hashable

from langgraph.graph import StateGraph


def simple_condition(state: dict[str, Any]) -> str:
    # Pick option based on state value
    choice = state.get("choice")
    if choice not in {"option_a", "option_b"}:
        # Default or error handling for unexpected choice
        return "__END__"
    return cast(str, choice)


# Create minimal StateGraph using dict as state type
graph = StateGraph(dict[str, Any])
# Define nodes


def start_node(state: dict[str, Any]) -> dict[str, Any]:
    # Pass through state so 'choice' persists
    return state


def option_a_node(state: dict[str, Any]) -> dict[str, Any]:
    print("Routed to option_a_node")
    return {} # Return empty dict or updated state


def option_b_node(state: dict[str, Any]) -> dict[str, Any]:
    print("Routed to option_b_node")
    return {} # Return empty dict or updated state


# Add nodes
graph.add_node("start", start_node)
graph.add_node("node_a", option_a_node)
graph.add_node("node_b", option_b_node)
# Edges
ngraph_id = str(uuid.uuid4())
print(f"Graph ID: {ngraph_id}")
graph.set_entry_point("start")
# Add conditional edges from start node
branch_mapping: dict[Hashable, str] = {"option_a": "node_a", "option_b": "node_b"}
print("Branch mapping:", branch_mapping)
graph.add_conditional_edges("start", simple_condition, branch_mapping)
# Inspect internal branches
if hasattr(graph, "branches") and "start" in graph.branches:
    for key, branch in graph.branches["start"].items():
        ends = getattr(branch, "ends", None)
        print(f"Branch key: {key}, ends mapping: {ends}")
else:
    print("No branches found for 'start'.")

# Compile graph
compiled = graph.compile()

# Test routing
print("Testing routing for option_a result:")
result_a = compiled.invoke({"choice": "option_a"})
print(result_a)
print("Testing routing for option_b result:")
result_b = compiled.invoke({"choice": "option_b"})
print(result_b)

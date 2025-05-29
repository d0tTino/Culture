#!/usr/bin/env python
"""
Script to diagnose pydantic Field deprecation warning.
"""

import inspect
import sys
import traceback
import warnings

# Make sure to add the project root to sys.path if running this script directly
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
from pydantic import PydanticDeprecatedSince20  # Import the specific warning category

# Use the specific filter
warnings.filterwarnings(
    "error",
    category=PydanticDeprecatedSince20,
    message=r"Using extra keyword arguments on `Field` is deprecated \\(extra keys: 'required'\\\\).",
)


def main():
    """
    Attempts to trigger the Pydantic warning by importing project modules.
    """
    print("Attempting to import project modules to trigger Pydantic warnings...")
    try:
        # Import modules that define Pydantic models
        # Add more imports if necessary
        print("Importing src.agents.core.agent_state...")

        print("Successfully imported src.agents.core.agent_state")

        print("Importing src.agents.graphs.basic_agent_graph...")

        print("Successfully imported src.agents.graphs.basic_agent_graph")

        # Add other relevant imports here, for example, from dspy_programs if they define models
        # print("Importing src.agents.dspy_programs...")
        # from src.agents import dspy_programs
        # print("Successfully imported src.agents.dspy_programs")

        print(
            "\\nNo PydanticDeprecatedSince20 (Field required=True) warning was escalated to an error during imports."
        )
        print(
            "This might mean the issue is not at the direct model definition level reachable by these imports,"
        )
        print("or the warning filter is not catching it as expected when run this way.")

    except PydanticDeprecatedSince20 as e:  # Catch the specific warning if it becomes an error
        print("\\n!!! PydanticDeprecatedSince20 Exception Caught !!!")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Args: {e.args}")
        print(f"Exception Message: {e}")
        print("\\n--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")

        # Print location if possible from the exception or current frame
        frame = inspect.currentframe()
        if frame:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            print("\\nError likely occurred near or during import processed around:")
            print(f"File: {filename}, Line: {lineno}")
        else:
            # Try to get info from the exception's traceback
            tb = e.__traceback__
            if tb:
                print("\\nError location based on exception traceback:")
                print(f"  File: {tb.tb_frame.f_code.co_filename}")
                print(f"  Line: {tb.tb_lineno}")
                print(f"  Function: {tb.tb_frame.f_code.co_name}")

    except Exception as e:
        print(f"\\nAn unexpected exception occurred: {e}")
        print("--- Unexpected Exception Traceback ---")
        traceback.print_exc()
        print("--- End Unexpected Exception Traceback ---")


if __name__ == "__main__":
    # It's important to adjust sys.path if running this script standalone
    # For now, assuming it's run from a context where 'src' is importable (e.g., project root)
    # Or, uncomment the sys.path modification at the top of the script.
    print("Running Pydantic warning diagnostic script...")
    # Add src to path to allow importing src modules
    import os

    # Assuming the script is in archives, so ../../src
    project_root_diagnose = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )  # Adjust if script moves
    print(f"Adding {project_root_diagnose} to sys.path for diagnosis script execution.")
    sys.path.insert(0, project_root_diagnose)

    main()
    print("\\nDiagnostic script finished.")

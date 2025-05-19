#!/usr/bin/env python
"""
Script to diagnose pydantic Field deprecation warning.
"""

import inspect
import sys
import traceback
import warnings

from pydantic import BaseModel, Field

warnings.simplefilter("error", category=DeprecationWarning)


class TestModel(BaseModel):
    """Test model that uses Field with a direct 'required' parameter."""

    # This should trigger the warning
    field_with_issue: str = Field(default="test", required=True)


try:
    # Create an instance to trigger validation
    TestModel()
except Exception as e:
    print(f"Exception: {e}")
    traceback.print_exc()
    # Print location
    frame = inspect.currentframe()
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    print(f"File: {filename}, Line: {lineno}")

    # Look at installed packages
    print("\nInstalled packages that might use Pydantic:")
    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]
    for pkg in installed_packages:
        if any(
            name in pkg.lower()
            for name in ["pydantic", "langchain", "openai", "discord", "chromadb"]
        ):
            print(f"  - {pkg}")

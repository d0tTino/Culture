# Warning Management in Culture.ai

This document explains how warnings are managed in the Culture.ai project, particularly those from third-party dependencies.

## Overview

In Python projects with multiple dependencies, warnings can often be generated from libraries that we don't directly control. These warnings can clutter test output and make it harder to identify issues in our own code.

Culture.ai implements a two-pronged approach to managing warnings:

1. **Programmatic Warning Filters**: Code-based filters in `src/infra/warning_filters.py`
2. **Pytest Configuration**: Declaration-based filters in `pytest.ini`

## Types of Warnings

The project currently handles these categories of warnings:

1. **Pydantic Field Deprecation Warnings**: Warnings about using extra keyword arguments on `Field` directly rather than using `json_schema_extra`
2. **Discord.py Deprecation Warnings**: Warnings about audioop being deprecated in Python 3.13

## Implementation

### 1. Warning Filters Module

The `src/infra/warning_filters.py` module contains programmatic filters that are applied when imported:

```python
def configure_warning_filters():
    """
    Configure warning filters to suppress specific warnings from dependencies.
    """
    # Suppress the Pydantic Field deprecation warning
    warnings.filterwarnings(
        "ignore", 
        message=r"Using extra keyword arguments on `Field` is deprecated.*",
        category=DeprecationWarning
    )
    
    # Suppress the audioop deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=r"'audioop' is deprecated and slated for removal in Python 3\.13",
        category=DeprecationWarning
    )
```

This module is imported in `tests/conftest.py` to ensure the filters are applied early in the test initialization process.

### 2. Pytest Configuration

The `pytest.ini` file contains declarative filters for pytest to apply:

```ini
[pytest]
# Ignore specific deprecation warnings that come from third-party dependencies
filterwarnings =
    # Ignore the Pydantic Field deprecation warning
    ignore:Using extra keyword arguments on `Field` is deprecated:UserWarning
    ignore:Using extra keyword arguments on `Field` is deprecated:DeprecationWarning
    # Ignore the audioop deprecation warning
    ignore:'audioop' is deprecated and slated for removal in Python 3.13:DeprecationWarning
```

## Best Practices

When working with the project, follow these best practices regarding warnings:

1. **Don't Ignore Your Own Warnings**: Only filter warnings from third-party dependencies that we can't control
2. **Use Proper Field Definition**: In our own code, always use Pydantic V2 compliant syntax:
   ```python
   # Incorrect (will generate warning):
   field: str = Field(default="value", required=True)
   
   # Correct:
   field: str = Field(default="value", json_schema_extra={"required": True})
   ```
3. **Document New Filters**: If you need to add a new warning filter, document it in this file
4. **Periodically Review**: Periodically review the warning filters to see if they're still needed as dependencies get updated

## Adding New Filters

To add a new warning filter:

1. Add it to `src/infra/warning_filters.py`
2. Add it to `pytest.ini`
3. Document it in this file
4. Verify the filter works by running the tests

## Test Suite Warning Policy

Run the full test suite with warnings disabled:

```bash
pytest --disable-warnings
```

The summary will still show how many warnings were raised. If any appear,
either fix the source (for example by updating deprecated Pydantic `Field`
usage or closing async sessions) or extend `src/infra/warning_filters.py` and
document the change here.

## Command-Line Control

You can enable or disable these filters when running the simulation:

```bash
python -m src.app --no-warning-filters        # show all warnings
python -m src.app --log-suppressed-warnings   # keep filters but log them
```

Use `--no-warning-filters` for debugging noisy dependencies, and
`--log-suppressed-warnings` to record suppressed messages without
polluting the console.

## Potential Future Improvements

- Add command-line flag to enable/disable warning filtering for debugging
- Implement more granular filtering by module or specific warning location
- Add warning reporting tools to collect statistics on suppressed warnings

The `tests` directory contains unit and integration tests. Some tests require
optional packages such as `chromadb`, `weaviate-client`, and `langgraph`.
Those tests call `pytest.importorskip()` to skip automatically when these
packages are not installed.

Running `pytest` without the optional dependencies will report these tests as
`skipped` rather than `failed`. Install the optional packages (they are listed in
`requirements.txt` and `requirements-dev.txt`) to execute the full suite.

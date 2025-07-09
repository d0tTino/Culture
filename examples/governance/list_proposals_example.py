#!/usr/bin/env python
"""Fetch stored law proposals from the dashboard API."""
import requests


def main() -> None:
    resp = requests.get("http://localhost:8000/api/proposals")
    print("Proposals:", resp.json())


if __name__ == "__main__":  # pragma: no cover - example script
    main()

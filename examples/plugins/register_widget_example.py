"""Example plug-in registering a custom widget with the Culture UI."""

import requests


def main() -> None:
    """Register a widget named ``ExampleWidget`` using the backend API."""
    resp = requests.post(
        "http://localhost:8000/api/register_widget",
        json={"name": "ExampleWidget", "script_url": "http://localhost:5173/example.js"},
        timeout=10,
    )
    print("Registration response:", resp.json())


if __name__ == "__main__":  # pragma: no cover - example script
    main()

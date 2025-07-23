"""Example plug-in registering a custom widget with the Culture UI."""

import asyncio

from src.extensions import register_widget_backend


async def main() -> None:
    """Register a widget named ``ExampleWidget`` using the backend API."""
    await register_widget_backend(
        name="ExampleWidget",
        script_url="http://localhost:5173/example.js",
    )
    print("Widget registered via backend API")


if __name__ == "__main__":  # pragma: no cover - example script
    asyncio.run(main())

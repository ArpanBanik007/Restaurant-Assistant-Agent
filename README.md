# Restaurant Order Assistant

A standalone LangGraph CLI that extracts a single dish and quantity, checks inventory,
collects a decision for partial or unavailable orders, and simulates cooking and serving.

## Run

Install dependencies with `uv sync`, then start the app with `uv run python main.py`.
Put your rotated `GROQ_API_KEY` in the root `.env` file. The default model is
`openai/gpt-oss-120b`; override it with `GROQ_MODEL`. Without a key, a local parser
supports simple inputs such as `2 pasta` and `3 veggie burger`.

Do not commit API keys. Edit the starter inventory in `main.py` to match the menu.


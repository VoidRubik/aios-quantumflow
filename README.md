# QuantumFlow

Fully offline Windows dictation — speech to text with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) plus an instant local cleanup pass. Nothing leaves the machine: no API key, no network for the core loop.

## What it does

- **Speak, get text pasted where your cursor is.** Trigger by clicking the bottom-screen pill, double-clicking the middle mouse button, holding `F9` (record while held), or double-tapping `F9` (latch on). While recording, `✕` cancels and `✓` stops and pastes.
- **Long recordings transcribe live** in 8-second chunks, so the wait after you stop stays short regardless of length.
- **Config** — `flow_config.json` (copy from `flow_config.example.json`) holds your style, a custom dictionary that helps Whisper hit your jargon, and snippets (`"my email"` → the address gets pasted).
- **Dashboard** — the tray icon's *Open Flow* regenerates a local `dashboard.html` from your usage stats (Insights, Your Voice, Dictionary, Snippets, Style) and opens it. It's built on demand and never committed — it's your data.
- **Optional local LLM cleanup** via [Ollama](https://ollama.com) `qwen2.5:1.5b`, only when `CLEANUP_MODE = "llm"`. Offline either way.

## Stack

Single file — `flow.py` (~44 KB). Python · `faster-whisper` for transcription · optional Ollama for cleanup. No cloud services.

## Run

```
pip install faster-whisper sounddevice keyboard mouse pyperclip numpy requests pystray pillow
python -u flow.py      # debug + flow.log
pythonw flow.py        # no console
```

Tray icon: Open Flow / Style / Quit. For auto-start on login, see the `schtasks` line in the `flow.py` docstring (run once from an admin terminal).

## Status

Working, in daily personal use. This repo is the engine only — the usage dashboard, stats, and personal config stay local and are `.gitignore`d.

## License

MIT — see [LICENSE](LICENSE).

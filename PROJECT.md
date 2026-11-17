# QuantumFlow — Project Card

**What:** A fully local, offline dictation app for Windows — a Wispr Flow clone. Speech → text with `faster-whisper`, followed by an instant local cleanup pass. Nothing leaves the machine.

**Status:** working, in personal daily use. Single-file Python app (`flow.py`) plus a browser dashboard.

**Grade:** Built — but see the stale-path issue below before trusting the documented auto-start command.

## Stack

- **Python** — `faster-whisper` (transcription) · `sounddevice` · `keyboard` · `mouse` · `pyperclip` · `numpy` · `pystray` + `pillow` (tray icon) · `requests`.
- **Optional local LLM cleanup** — Ollama with `qwen2.5:1.5b`, used only when cleanup mode is set to `llm`. Offline either way.
- No cloud service, no API key, no network dependency for the core loop.

## How it runs

```
pythonw flow.py        # normal — no console window
python -u flow.py      # debug — console + writes flow.log
```

A desktop shortcut launches it; launching again opens the dashboard.

## Interaction

- A small pill sits at the bottom of the screen — **click to start dictating**.
- **Double-click the middle mouse button** → start; double-click again to stop.
- **Hold `F9`** → record while held, release to stop.
- **Double-tap `F9`** → latch recording on; tap once more to stop.
- While recording: **✕** cancels (nothing pasted), **✓** stops and pastes.
- Tray icon: Open Flow (dashboard) · Style · Quit.

**Long recordings** are transcribed live in 8-second chunks while you speak, so the wait after stopping stays short regardless of how long you talked.

## Dashboard

`dashboard.html`, opened in the browser: **Insights · Your Voice · Dictionary · Snippets · Style.**

## Files

| File | What |
|---|---|
| `flow.py` | the whole app (~44 KB) |
| `dashboard.html` | the browser dashboard |
| `flow_config.json` | style, custom dictionary words (helps Whisper recognize them), snippets (say "my email" → the address is pasted) |
| `flow_stats.json` | usage stats feeding the Insights view |
| `flow.log` | debug-mode log |
| `flow.ico` | tray/shortcut icon |

## Known issues

- **Stale auto-start path.** The setup docstring inside `flow.py` still gives a `schtasks` auto-start command pointing at `C:\Users\<you>\Documents\QuantumFlow\flow.py` — the pre-move location. The project now lives under `AIOS/projects/QuantumFlow/`, so that command would register a task against a path that does not exist. Fix the docstring (and any registered task) before relying on login auto-start.
- Auto-start registration requires an **admin** terminal, and is Bruno's hands.

## Notes

`flow_config.json` was audited for secrets and is clean — it holds an email address for the snippet feature and nothing else.

Memory node: `brain/situational/memory/quantumflow.md`.

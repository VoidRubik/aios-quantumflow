"""Self-check for the text-cleanup pipeline (fast_clean, snap_vocab,
apply_snippets, apply_style) — the ~15 pure functions between raw Whisper
output and the pasted text.

Does NOT import flow.py directly: the whole app runs at module level (tray,
hotkey, Tk mainloop, model load) with no __main__ guard, so `import flow`
launches it. Instead this pulls just the needed function/constant defs out
via ast and execs them in an isolated namespace — no mic, no model, no UI.

Run: py -3.12 test_clean.py
"""
import ast
import difflib
import re

SRC = open("flow.py", encoding="utf-8").read()
tree = ast.parse(SRC)

WANTED = {"snap_vocab", "fast_clean", "apply_snippets", "apply_style",
          "STRIP_RE", "FILLERS_RE", "SNAP_VOCAB_CUTOFF", "DEFAULT_CONFIG"}
nodes = [n for n in tree.body
         if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
         or (isinstance(n, ast.Assign) and any(
             isinstance(t, ast.Name) and t.id in WANTED for t in n.targets))]

ns = {"re": re, "difflib": difflib}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<flow-extract>", "exec"), ns)
ns["config"] = dict(ns["DEFAULT_CONFIG"])  # snap_vocab/apply_snippets/apply_style read this

snap_vocab, fast_clean = ns["snap_vocab"], ns["fast_clean"]
apply_snippets, apply_style = ns["apply_snippets"], ns["apply_style"]

n_ok = 0


def check(label, got, want):
    global n_ok
    assert got == want, f"{label}: got {got!r}, want {want!r}"
    n_ok += 1


# --- fast_clean: filler strip + stutter/punctuation cleanup ---
check("strip uh", fast_clean("Uh this is uh a test"), "This is a test")
check("strip um variants", fast_clean("Um, well, umm I think"), "Well, I think")
check("stutter collapse", fast_clean("the the cat cat sat"), "The cat sat")
check("only punctuation", fast_clean("...!?"), "")
check("empty", fast_clean(""), "")

# --- snap_vocab: spelling-level near-misses, measured cutoff 0.80 ---
# NOT the flagship "content flow"->"QuantumFlow" case — that's a phonetic
# confusion (ratio 0.52), not a spelling near-miss; character-diffing can't
# see it. hotwords= is the real lever there. Documented, not silently missed.
check("whisper flow -> Wispr Flow",
      snap_vocab("whisper flow inspired this"),
      "Wispr Flow inspired this")
check("exact match untouched", snap_vocab("QuantumFlow works"), "QuantumFlow works")
# negative — must NOT rewrite an unrelated, merely similar-length phrase
check("quantum physics untouched",
      snap_vocab("I study quantum physics"),
      "I study quantum physics")
check("content flow untouched (known miss, see cutoff comment in flow.py)",
      snap_vocab("I use content flow every day"),
      "I use content flow every day")
check("ordinary sentence untouched",
      snap_vocab("the cat sat on the mat"),
      "the cat sat on the mat")

# --- apply_snippets / apply_style: unchanged behavior, still exercised ---
check("snippet expands", apply_snippets("send my email please"),
      "send you@example.com please")
check("very_casual strips commas+periods+caps", apply_style("Hi, Bruno."), "Hi, Bruno.")
ns["config"]["style"] = "very_casual"
check("very_casual applies", apply_style("Hi, Bruno."), "hi bruno")

print(f"{n_ok} checks passed")

# Repo Consolidation Design

Goal: Make the repository navigable and self-explanatory without moving canonical artifact paths.

Decision: Keep the current directory layout stable and consolidate through documentation, folder-level guides, and ignore rules.

Approach:
1. Replace the stale root README with a GraphPilot-Edge-first overview, quickstart, and navigation guide.
2. Add a docs index and repo layout document that identify the source code, Android runtime, simulator, artifacts, truth documents, and paper outputs.
3. Add folder-level READMEs for the highest-value entry points: `artifacts/graphpilot_edge`, `scripts`, `graphpilot_edge`, `android-app`, and `Truth-docs`.
4. Add a lightweight regression test that ensures the canonical docs exist and are referenced from the root README.
5. Add ignore rules for local-only junk such as `.venv/` and `tmp/`.

Non-goals:
- No physical relocation of artifact folders, experiment outputs, or paper paths.
- No renaming of pinned directories used by checkpoints, reports, or the CASES paper.


- 2026-04-01: Environment command `python` is unavailable in this workspace shell; verification command must use `python3 -m py_compile ...`.
- 2026-04-01: `os.sched_getaffinity` is unavailable on darwin; topology discovery now uses a runtime fallback to `os.cpu_count()` when affinity syscall is missing.
- 2026-04-01: `git status --short` currently shows multiple pre-existing untracked files in this workspace; task verification relied on targeted test commands and changed-file diagnostics rather than clean-tree assumptions.
- 2026-04-01: Repository currently has task files as untracked (not staged baseline), so completion validation must rely on direct test command outcomes and file-level diagnostics instead of git diff against tracked history.
- 2026-04-01: Pyright reports optional `torch` import as missing in non-torch environments; resolved in task scope by keeping runtime optional import and adding a localized type-ignore on the import line.
- 2026-04-01: No new task-7 blockers encountered; required remeasure-focused and baseline-focused pytest selections both pass in current environment.

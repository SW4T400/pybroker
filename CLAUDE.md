# Agent instructions (Claude Code / Codex)

Server-wide global instructions also apply:
- Claude Code: `~/.claude/CLAUDE.md`
- Codex: `~/.codex/AGENTS.md`

## Cross-session memory: basic-memory

Persist work to the **basic-memory MCP** (streamable-http, default project `notes`)
-- NOT to conversation history, and NOT to the legacy local auto-memory files under
`~/.claude/projects/*/memory/`.

- **END of a session (and after major milestones) -- REQUIRED**: write what changed
  back via `write_note` (or `edit_note` to update an existing note). Capture:
    - **Findings**  -- what was discovered, measured, or concluded.
    - **Decisions** -- what was decided, and the reasoning behind it.
    - **Learnings** -- gotchas, dead-ends, and what to do differently next time.
  Tag by type: `tags: [finding | decision | learning | project | feedback | reference]`.
- **Reading prior notes is ON-DEMAND only**: search basic-memory
  (`search_notes` / `recent_activity` -> `read_note`) ONLY when the user asks about
  previous work, research, decisions, or memory. Do NOT auto-search every session.

Applies to both Claude Code and Codex.

## Git branch safety -- CRITICAL (work ONLY on `main`)

**The live Prefect workers execute flows directly from the LOCAL working tree.**
Deployments are registered with `from_source(source=.../scr)` + a
`set_working_directory` pull step, so whatever is checked out on disk is EXACTLY
what the production servers run at the next tick. A local branch switch silently
changes live trading / data-pipeline code.

- ALWAYS stay on `main`. Make every change on `main`.
- NEVER run `git checkout <branch>` / `git switch <branch>`, and NEVER
  create-and-switch to a feature branch, in any working clone under
  `/home/stefa/pythoncode/github_clones`. FORBIDDEN -- even "just to isolate" or
  "I'll switch back."
- Commit directly to `main` (only when the user asks). Do NOT create feature branches.
- If branch isolation is genuinely required, use a SEPARATE `git worktree` in a
  different directory so the primary clone STAYS on `main` -- never an in-place checkout.

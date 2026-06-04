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

# Reusable AI competition starter

Open **START_HERE.md** and paste its prompt into Codex or another coding agent
with workspace and terminal access. Supply the current official challenge files.
The agent is asked to create the environment and implementation, run a bounded
first cycle, and leave an English repository with documented evidence.

The templates are optional references; the prompt already instructs the agent
to create and adapt equivalent files. You do not need to manually install them.

| File | Purpose |
| --- | --- |
| `START_HERE.md` | Main copy-paste prompt and usage instructions |
| `AGENTS.template.md` | Persistent repository instructions; adapt as `AGENTS.md` |
| `PROJECT_BRIEF.template.md` | Optional competition context and continuation prompt |
| `EXPERIMENT.template.md` | Evidence record for each training/evaluation cycle |

## Agent compatibility

`AGENTS.md` is a repository instruction convention, not a universal execution
environment. Agents differ in which files they load automatically. If necessary,
explicitly ask yours to read it at the start of the session. The prompt requests
a platform-specific adapter where supported, with one canonical instruction file.
`SKILL.md` is useful for a separately installed reusable skill; it is not required
to start this project, and no custom skill is included.

## What the prompt does not supply

Official competition materials, missing trained models, execution access,
unlimited compute, or guaranteed playing strength. It intentionally avoids
hard-coding the previous Colony competition's contract into a new project.

The workflow incorporates lessons from Colony: strict export validation,
recovering pending candidates, side-aware evaluation, episode-level splits,
keeping the actual submitted model separate from the newest champion, and
distinguishing file metadata from measured model strength.

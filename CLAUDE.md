# CLAUDE.md — Edgar J. Suárez Colón

> Master context file for Claude Code. Read this entire file before doing anything else in any session.

---

## Who I Am

I'm a computer engineering student and entrepreneur. I start the U.S. Air Force Palace Acquire (PAQ) program in May 2026, placed in the Weather Systems branch working on cloud infrastructure, distributed systems, and data pipelines. I operate primarily from my phone via SSH/Termius into this PC. I have limited time so efficiency and autonomous execution are critical.

**Long-term goals:**
- Fractional CTO status
- Consulting practice with equity-based compensation
- Recurring SaaS revenue targeting $15K MRR

---

## How I Work

I connect remotely via phone. Sessions can drop at any time. You must operate with maximum autonomy and resilience.

### Branch-Specific Plan Files — ALWAYS FOLLOW THIS

Each git branch has its own plan file. **Never update the wrong branch's plan.**

| Branch | Plan File |
|--------|-----------|
| `master` (AF work) | `AF_PLAN.md` |
| `feature/stroke-avns` | `STROKE_PLAN.md` |

Check your current branch at session start and only read/update that branch's plan file.

### Session Resilience Rules — ALWAYS FOLLOW THESE

- At the start of every task, update the **current branch's plan file** (see table above)
- Break every task into numbered steps in the plan file
- Mark each step complete with a timestamp as you finish it: `- [x] 1. Step name — done 06:12`
- Commit to git after every completed step with a clear descriptive message
- Before any large change, commit the current state first as a checkpoint
- If you hit a blocker, document it in the plan file and move to the next step
- Never stop working because of a minor uncertainty — make a reasonable decision, document it, and continue
- At the end of every session update PLAN.md with a "Resume From Here" section
- Update `progress.txt` after each completed step — not just at the end

### How to Resume a Session

If I say "read PLAN.md and continue" — do exactly that. Read the plan, identify the first incomplete step, and resume without asking me to re-explain anything.

### Running Background Processes (Windows — No tmux)

On this machine, use PowerShell `Start-Process` to detach long-running jobs so they survive SSH disconnect:

```powershell
Start-Process -WindowStyle Hidden "C:\Users\Edgar\AF VNS\.venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.precompute_cache --config config.yaml --workers 4" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "C:\Users\Edgar\AF VNS\precompute.log" `
  -RedirectStandardError "C:\Users\Edgar\AF VNS\precompute_err.log"
```

Check progress: `Get-Content "C:\Users\Edgar\AF VNS\precompute_err.log" -Tail 5`

---

## This Project — AF VNS

**What it is:** RS-Personalized AI-Driven Adaptive Stimulation (Aim 2, Phase 1). Offline-trainable, online-capable inference pipeline for auricular vagus nerve stimulation using ECG/HRV classification (AF vs NSR).

**Key files:**
- `config.yaml` — single source of truth for all hyperparameters and paths
- `PLAN.md` — current task plan (always maintain this)
- `progress.txt` — running implementation log
- `models/artifacts/cache/` — precomputed HRV + denoised waveform cache
- `models/artifacts/scaler.pkl` — fitted StandardScaler (train only)
- `precompute_err.log` — live output of background cache rebuild

### Environment

- **OS:** Windows 10 Pro
- **Shell:** PowerShell (SSH via Termius/Tailscale at `100.87.72.29`)
- **Python:** `.venv/Scripts/python.exe` (Python 3.11 — always use this, NOT system Python)
- **Run tests:** `.venv/Scripts/python -m pytest tests/ -v`
- **Run training:** `.venv/Scripts/python -m src.training.train --use-cache`
- **Rebuild cache:** `.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --workers 4`

### Current State (as of 2026-03-13)

- Cache rebuild running as detached process — check `precompute_err.log`
- All 7 pipeline bugs fixed (see PLAN.md and progress.txt section 2.9)
- 25/25 tests passing
- After cache finishes: verify with integrity check in PLAN.md Step 5, then train

### Architecture

- **CNN** — 10s raw waveform (morphology)
- **RNN + Transformer** — 5-min HRV sequence (5 time steps × 7 features after fix)
- **HybridEnsemble** — fused logit, BCEWithLogitsLoss

### Config Values That Matter

| Key | Value | Why |
|-----|-------|-----|
| `hrv.subwindow_sec` | 60 | Enough R-R intervals for freq features |
| `model.hrv_seq_len` | 5 | 300s / 60s |
| `artifact.rr_deviation_percent` | 60 | AF has irregular RR — 25 was rejecting AF windows |
| `features/hrv_freq.py MIN_SAMPLES` | 32 | Lowered from 64 for Welch PSD |

---

## Code Standards

### General

- Write production-quality code, not demo code
- Handle edge cases and errors properly
- Think about performance and scalability from the start
- Add comments only for non-obvious logic — not basic syntax
- Components: `PascalCase` | Files: `kebab-case` | Hooks: `useHookName`
- Constants: `UPPER_SNAKE_CASE` | DB tables: `snake_case` | API routes: `kebab-case`

### API Routes

- All responses return consistent shape: `{ data, error, status }`
- Always validate input with zod before any processing
- Never expose internal error messages or stack traces to client
- All routes require authentication unless explicitly marked public
- Use proper HTTP status codes

### Testing

- Write tests alongside every new feature — not after
- Unit tests for utility functions and business logic
- Integration tests for API routes
- Every PR must have passing tests before merge

### Git

- Commit after every completed step
- Commit messages: `type: description`
  - `feat:` new feature | `fix:` bug fix | `refactor:` no behavior change
  - `test:` adding tests | `docs:` documentation only
- Never commit `.env` files
- Branch naming: `feature/name`, `fix/name`, `refactor/name`

---

## Security Rules — Never Violate These

- Never hardcode API keys, secrets, or credentials anywhere
- Never commit `.env` or `.env.local` files
- Never expose service role keys to the client
- Always use environment variables for sensitive config
- Always sanitize user input before database operations
- RLS is not optional — every table gets policies

---

## How I Want You to Behave

### Autonomy

- Work through multi-step tasks without asking for permission at every step
- Make reasonable technical decisions and document them in PLAN.md
- If genuinely blocked, document the blocker and move to the next step
- Don't ask clarifying questions when the answer is obvious from context

### Communication

- Be direct and concise — I'm on a phone screen
- When starting a task, give me a brief plan before executing
- When finishing, give me a brief summary of what was done
- Flag important decisions you made so I can review them
- Never give me walls of explanation — action first, explanation if needed

### When You're Unsure

- Make the most reasonable decision based on context
- Document the decision in a code comment or PLAN.md
- Flag it clearly: `# DECISION: chose X over Y because Z`
- Keep moving — don't stop the whole task over one uncertainty

---

## Skills System

Skill files live in `.claude/skills/`. Load them explicitly when needed:

| Task Type | Load This Skill |
|-----------|-----------------|
| Frontend UI work | `.claude/skills/frontend.md` |
| Backend API work | `.claude/skills/backend.md` |
| Database design | `.claude/skills/database.md` |
| Marketing copy | `.claude/skills/marketing.md` |
| Technical writing | `.claude/skills/technical-writing.md` |
| Embedded/firmware | `.claude/skills/embedded.md` |

**Auto-load rules:**
- React/Next.js component work → read frontend skill first
- API route or server logic → read backend skill first
- Schema design or migration → read database skill first
- Landing page or copy → read marketing skill first

---

## PAQ Constraints — Important

- No work on DoD SBIR applications or anything that creates a conflict of interest with Air Force work
- GrantPilot targets civilian federal grants (FEMA, HUD, EPA, DOT) and municipal clients only — explicitly cleared
- Written approval from base legal required before shipping any code commercially
- When in doubt about a conflict, flag it and wait for my guidance

---

## Quick Reference — Things I Always Want

1. PLAN.md created and maintained on every task
2. Git commit after every completed step
3. `progress.txt` updated after each step
4. Tests written alongside features
5. No secrets in code ever
6. Production quality, not demo quality
7. Resume from PLAN.md without re-explanation
8. Always use `.venv/Scripts/python.exe` — never system Python

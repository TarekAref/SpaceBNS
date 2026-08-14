# IBM Bob Usage Log

IBM Bob is the required primary development tool for the IBM August Challenge.
This document must contain evidence of **real** Bob-assisted work. Do not claim
that Bob created, reviewed, or tested an artifact unless that interaction
actually occurred.

## Evidence template

Copy this section for each meaningful Bob-assisted task.

### Task: Short task name

- **Date (UTC):**
- **Developer:** Tarek Aref
- **Bob mode/workflow:**
- **Goal:**
- **Prompt summary:** Summarize the actual request without including secrets, credentials, proprietary data, or sensitive mission or personal information.
- **Files affected:**
- **Bob contribution:**
- **Human review and corrections:**
- **Validation performed:**
- **Result:**

## Recorded evidence

### Task: Frontend dependency security upgrade

- **Date (UTC):** 2026-08-14
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode with human approval
- **Goal:** Resolve confirmed frontend dependency vulnerabilities while preserving the current scaffold's build and test behavior.
- **Prompt summary:** Inspect npm audit findings, perform an exact Next.js upgrade (15.5.23 → 16.3.0), avoid force-fixing unrelated advisories, and validate the result.
- **Files affected:** `frontend/package.json`, `frontend/package-lock.json`, `frontend/tsconfig.json`
- **Bob contribution:** Identified vulnerable packages, proposed the upgrade plan, and executed the approved upgrade and validation commands. Upgraded Next.js from 15.5.23 to 16.3.0; the Next.js upgrade caused the resolved transitive dependency versions to become PostCSS 8.5.23 and Sharp 0.35.3. `frontend/tsconfig.json` was updated automatically by Next.js during its build step.
- **Human review and corrections:** Tarek approved the upgrade commands, then reviewed and accepted the resulting dependency and configuration diffs.
- **Validation performed:** `npm ci` (clean install); `npm audit` (0 vulnerabilities); TypeScript type-check; Next.js production build; 3 backend regression tests passing.
- **Result:** Next.js upgraded to 16.3.0; `npm audit` reported zero known vulnerabilities, and no regressions were detected by the current automated checks.

---

### Task: Generated-file hygiene and Git checkpoint

- **Date (UTC):** 2026-08-14
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode with human approval
- **Goal:** Exclude generated TypeScript artifacts from version control without deleting required local files.
- **Prompt summary:** Review Next.js-generated files, ignore reproducible build caches, stop tracking `next-env.d.ts` without deleting its local copy, and validate and checkpoint the exact changes.
- **Files affected:** `.gitignore`, `frontend/next-env.d.ts`, `frontend/tsconfig.tsbuildinfo` (local only)
- **Bob contribution:** Added `*.tsbuildinfo` and `next-env.d.ts` patterns to `.gitignore`; ran `git rm --cached` on `frontend/next-env.d.ts` to stop tracking it while leaving the file on disk; deleted the untracked generated cache `frontend/tsconfig.tsbuildinfo` locally and added it to `.gitignore`; verified the working tree with staged and unstaged diff checks. Bob assisted and executed approved commands; Tarek remained the author and decision-maker throughout.
- **Human review and corrections:** Tarek confirmed that `frontend/next-env.d.ts` remained locally available after removal from tracking and approved all `.gitignore` additions.
- **Validation performed:** Staged and unstaged `git diff` checks; clean working tree confirmed; changes committed as `c6ffe78`; successful non-force push to branch `bob/mvp-build`.
- **Result:** `frontend/next-env.d.ts` is no longer tracked but remains available locally; generated cache files are ignored; the working tree was clean and `bob/mvp-build` was synchronized with its remote branch.

---

## Suggested evidence categories

- architecture decomposition;
- FastAPI endpoint implementation;
- Next.js component development;
- telemetry-schema validation;
- image-model adapter development;
- unit and integration tests;
- Docker/CI troubleshooting;
- security review; and
- README and architecture documentation.

## Submission summary

Before submission, replace this paragraph with a concise, evidence-backed
summary of how IBM Bob functioned as the primary development tool across
planning, implementation, testing, debugging, and documentation.


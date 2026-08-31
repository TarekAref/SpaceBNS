# SpaceBNS — Final Demo Recording Script

- **Target length:** 2 minutes 55 seconds
- **Format:** narrated screen recording, product first, architecture later
- **Claim boundary:** representative simulation, synthetic data only, no
  flight qualification, no command authority

## Before recording

1. Open [API health](https://spacebns-api.onrender.com/health) and wait for the
   Render backend to wake.
2. Open the [dashboard](https://spacebns.vercel.app/) and confirm the full result
   is visible. Use **Retry data load** if needed.
3. Open three background tabs for later cuts:
   [validation evidence](../README.md#validation-evidence),
   [architecture](architecture.md), and
   [IBM Bob evidence](bob-usage.md).
4. Record at 1080p when possible. Close developer tools, unrelated tabs,
   notifications, bookmarks, and personal information.
5. Keep the cursor still unless it is directing attention. Scroll slowly and
   stop long enough for each section heading to be read.

## Timed shot list and narration

### 0:00–0:20 — Operator problem and stakes

**Show:** Dashboard title and simulation-only boundary.

**Narrate:**

> A resource-limited spacecraft is approaching another payload window. Its
> operations team needs to know whether recent power telemetry signals a
> developing constraint—but an AI score alone is not enough to justify action.
> SpaceBNS is an explainable early-warning advisory built for that decision. This
> demonstration uses synthetic data and represents a simulation scenario only.

### 0:20–0:42 — Product and authority boundary

**Show:** Header, persistent safety strip, and top advisory.

**Narrate:**

> SpaceBNS turns six hours of telemetry into a 24-hour power-risk assessment,
> while keeping operator authority explicit. The page is permanently marked
> synthetic, not flight-qualified, simulation only, and command authority none.
> The system can advise, but it cannot issue or transmit spacecraft commands.

### 0:42–1:02 — Audit and provenance

**Show:** Audit & Provenance.

**Narrate:**

> Every result carries its provenance. This scenario contains 72 samples over a
> six-hour window, uses 12 frozen features, and identifies the model claim and
> version. These values come from the API response rather than decorative text
> in the interface.

### 1:02–1:28 — Explainable learned risk

**Show:** L1 probability, predicted class, then contribution bars.

**Narrate:**

> The learned layer estimates approximately a 33 percent probability of a power
> constraint within 24 hours and predicts no breach for this scenario. The
> contribution bars show which standardized inputs moved the estimate up or
> down. They are learned associations, not claims of physical causation.

### 1:28–1:58 — The fail-closed magic moment

**Show:** L2 omission reason. Pause on it.

**Narrate:**

> Here is the key design choice. Most AI systems are rewarded for always
> producing an answer. SpaceBNS refuses. A deterministic physical projection
> would require battery capacity, load, conversion efficiency, sunlight, and
> payload-schedule assumptions. Because those assumptions were not supplied,
> the projection is explicitly omitted instead of invented. Missing output here
> is evidence of responsible engineering, not a hidden failure.

### 1:58–2:18 — Independent rules and human advisory

**Show:** L3 findings, then L4 advisory.

**Narrate:**

> The omission does not erase the rest of the assessment. Independent safety
> thresholds are still evaluated, and the advisory remains available. This
> nominal scenario recommends continued monitoring, requires no immediate human
> action, and states again that no automated action has been or will be taken.

### 2:18–2:38 — Validation and architecture

**Show:** README validation evidence, then architecture diagram.

**Narrate:**

> The proof is reproducible: all seven held-out evaluation gates passed, 130
> focused model and prediction tests passed, and clean-source training produced
> a SHA-256-identical model artifact. The broader backend suite contains 3,686
> tests. Results remain synthetic-only, including an honestly disclosed
> precision of 0.667 and 15 false positives.

### 2:38–2:50 — IBM Bob

**Show:** Top of `docs/bob-usage.md`, then one recorded evidence entry.

**Narrate:**

> IBM Bob supported architecture planning, implementation, testing, review, and
> documentation. The public log separates Bob-assisted work from Tarek's human
> review and later corrections. Bob is not part of runtime inference.

### 2:50–2:55 — Close

**Show:** Dashboard title and advisory.

**Narrate:**

> SpaceBNS helps humans review power risk earlier—without confusing probability
> with physics, and without surrendering command authority.

## One-take quality check

- Product appears before architecture.
- The L2 omission receives a clear pause and is called the key design choice.
- `COMMAND_AUTHORITY: NONE` is both visible and spoken.
- No real mission, flight validation, certification, or operational claim is
  implied.
- Dashboard, API, repository, and Bob-evidence links are tested after upload.
- Final video is public or unlisted without requiring the judge to sign in.

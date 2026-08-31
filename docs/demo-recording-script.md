# SpaceBNS — Final Demo Recording Script

- **Target length:** 2 minutes 55 seconds
- **Primary audience:** small-satellite mission operations teams,
  mission-assurance stakeholders, and challenge judges
- **Customer need:** review developing power risk earlier without relying on an
  opaque AI answer or surrendering operator authority
- **Format:** narrated screen recording, customer problem first, product second,
  technical evidence after the working flow
- **Claim boundary:** representative simulation, synthetic data only, no flight
  qualification, no command authority

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
6. Deliver this as a customer story, not a code walkthrough. Use plain language
   before naming the model or evaluation metrics.

## Timed shot list and narration

### 0:00–0:22 — Customer need and stakes

**Show:** Dashboard title and simulation-only boundary.

**Narrate:**

> Small-satellite mission operations teams must protect limited battery margin
> while scheduling payload work. A developing constraint can force payload
> delays or more conservative operating plans, but an opaque AI score is not
> enough to justify action. SpaceBNS gives operators an explainable early
> warning. This is a representative simulation using synthetic data.

### 0:22–0:42 — Solution and authority boundary

**Show:** Header, persistent safety strip, and top advisory.

**Narrate:**

> SpaceBNS turns six hours of telemetry into a 24-hour power-risk assessment
> while keeping operator authority explicit. The interface is permanently
> marked synthetic, not flight-qualified, simulation only, and command authority
> none. It can advise a human operator, but it cannot issue or transmit
> spacecraft commands.

### 0:42–1:05 — Data and model characteristics

**Show:** Audit & Provenance, then the model version in L1.

**Narrate:**

> Every result carries its provenance. This case uses 72 synthetic samples over
> six hours and 12 frozen features. Model version 0.1.1 is a scaled logistic
> regression trained on 300 seed-42 synthetic scenarios, with 60 reserved for
> final testing. It has never been validated on a real spacecraft.

### 1:05–1:28 — Explainable learned risk

**Show:** L1 probability, predicted class, then contribution bars.

**Narrate:**

> The learned layer estimates approximately a 33 percent probability of a power
> constraint within 24 hours and predicts no breach in this scenario. The
> contribution bars show which standardized inputs moved the estimate up or
> down. They are learned associations, not claims of physical causation.

### 1:28–1:58 — The fail-closed magic moment

**Show:** L2 omission reason. Pause on it.

**Narrate:**

> Here is the key design choice. Most AI systems are rewarded for always
> producing an answer. SpaceBNS refuses. A deterministic physical projection
> requires battery capacity, load, conversion efficiency, sunlight, and
> payload-schedule assumptions. Because those assumptions were not supplied,
> the projection is explicitly omitted instead of invented. Missing output here
> is evidence of responsible engineering, not a hidden failure.

### 1:58–2:18 — Independent rules and human advisory

**Show:** L3 findings, then L4 advisory.

**Narrate:**

> The omission does not erase the rest of the assessment. Independent safety
> thresholds are still evaluated, and the advisory remains available. This
> nominal case recommends continued monitoring, requires no immediate human
> action, and states that no automated action has been or will be taken.

### 2:18–2:38 — Feasibility and validation evidence

**Show:** README validation evidence, then architecture diagram.

**Narrate:**

> The proof is reproducible: all seven held-out gates passed, 130 focused model
> and prediction tests passed, and clean-source training produced an identical
> SHA-256 model artifact. The broader backend suite contains 3,686 tests. The
> synthetic-only results also disclose 0.667 precision and 15 false positives.

### 2:38–2:50 — IBM Bob and human judgment

**Show:** Top of `docs/bob-usage.md`, then one recorded evidence entry.

**Narrate:**

> IBM Bob supported architecture planning, implementation, testing, review, and
> documentation. The evidence log separates Bob's contribution from Tarek's
> human review and corrections. Bob is not part of runtime inference.

### 2:50–2:55 — Close

**Show:** Dashboard title and advisory.

**Narrate:**

> SpaceBNS turns power risk into explainable advice—while operators keep control.

## IBM-manager guidance covered by this script

The supplied 3:50 transcript explicitly emphasizes pitching to a potential
customer, making the need visible, using storytelling and a deliberate demo
flow, communicating model and data characteristics, and focusing on what
judges, stakeholders, and users need to hear. This script covers those points
through the customer-specific opening, product-first demonstration, explicit
model/data disclosure, evidence, and human-authority close.

The supplied excerpt announces three narrative archetypes and deeper criteria
but ends before defining them. Those details cannot be claimed as reviewed
until the remaining transcript or video section is available.

## One-take quality check

- The intended customer and operational consequence are clear in the first 22
  seconds.
- Product behavior appears before architecture and test evidence.
- The data source, sample window, feature count, model type, training corpus,
  test reserve, and real-spacecraft limitation are spoken.
- The L2 omission receives a clear pause and is called the key design choice.
- `COMMAND_AUTHORITY: NONE` is both visible and spoken.
- No real mission, flight validation, certification, or operational claim is
  implied.
- Dashboard, API, repository, and Bob-evidence links are tested after upload.
- Final video is public or unlisted without requiring the judge to sign in.

# SpaceBNS — 90-Second Judge Guide

SpaceBNS is an explainable early-warning advisory for spacecraft power
constraints. This path shows the complete public proof of concept without an
account, API key, or local installation.

## Quick links

- [Live dashboard](https://spacebns.vercel.app/)
- [API health / wake-up](https://spacebns-api.onrender.com/health)
- [Public prediction response](https://spacebns-api.onrender.com/api/v1/mock/power-risk-prediction)
- [Validation evidence](README.md#validation-evidence)
- [IBM Bob evidence](docs/bob-usage.md)

## The 90-second path

| Time | Action | What to verify |
| --- | --- | --- |
| 0:00–0:15 | Open [API health](https://spacebns-api.onrender.com/health). | The free Render service may need roughly one minute to wake. Wait for the health response before opening the dashboard. |
| 0:15–0:25 | Open the [live dashboard](https://spacebns.vercel.app/). If it shows **API unavailable**, select **Retry data load** once the health endpoint is awake. | Failure is explicit and recovery is manual; the UI does not hide an unavailable API or poll indefinitely. |
| 0:25–0:40 | Review **Audit & Provenance**. | The public scenario uses `SYNTHETIC` data, 72 samples over 6 hours, 12 features, and a `NOT_FLIGHT_QUALIFIED` prototype. |
| 0:40–0:55 | Inspect **L1 — AI Power-Risk Estimate** and the contribution bars. | The public scenario reports approximately 33% breach probability, model `0.1.1`, and learned feature contributions. These are associations, not physical causes. |
| 0:55–1:10 | Find **L2 — Deterministic Energy Projection**. | This is the magic moment: the projection is deliberately omitted because required physical assumptions were not supplied. SpaceBNS refuses to fabricate physics. |
| 1:10–1:20 | Review **L3 — Deterministic Safety Findings**. | Threshold rules are evaluated independently of the learned estimate. The nominal public scenario has no active threshold findings. |
| 1:20–1:30 | Review **L4 — Human Advisory** and the persistent safety envelope. | The recommendation is `CONTINUE_MONITORING`; `COMMAND_AUTHORITY: NONE`. No automated spacecraft action has been or can be taken. |

## Why this is differentiated

Most AI demonstrations optimize for always returning an answer. SpaceBNS
separates four kinds of evidence and fails closed when one cannot be justified:

1. **L1 learned risk:** a transparent synthetic-distribution estimate.
2. **L2 physical projection:** deterministic and present only with complete
   assumptions.
3. **L3 safety rules:** explicit thresholds evaluated independently.
4. **L4 operator advisory:** human-facing guidance with no command path.

This keeps statistical risk separate from physical certainty and preserves
human authority.

## Three reproducible proof points

- **7/7 held-out evaluation gates passed.**
- **130 focused model-pipeline and prediction tests passed.**
- **Clean-source model regeneration produced a SHA-256-identical artifact.**

The broader regression suite contains 3,686 backend tests. The held-out results
are intentionally disclosed as synthetic-only: precision is `0.666667`, the
60-scenario test split includes 15 false positives, and no claim is made about
real-spacecraft performance.

## IBM Bob boundary

IBM Bob supported architecture planning, implementation, testing, review, and
documentation. Bob is not part of the deployed inference path. The
[human-reviewed usage log](docs/bob-usage.md) records what Bob did, what Tarek
reviewed, and what later work is not attributed to Bob.

## Scope and maturity

SpaceBNS uses synthetic data only. It is not connected to a spacecraft, is not
flight-qualified, and cannot authorize or transmit commands. The next credible
steps are independent simulated scenario families, hardware-in-the-loop
evaluation, authorized real-telemetry validation, operational qualification,
and only then possible flight consideration after independent verification.

# SpaceBNS — 90-Second Judge Guide

SpaceBNS turns six hours of synthetic spacecraft power telemetry into an explainable 24-hour power-risk estimate, checks deterministic safety rules, and refuses unsupported physical certainty. It is advisory only and can never issue spacecraft commands.

## Fastest review path

1. **Wake the backend:** Open the [API health endpoint](https://spacebns-api.onrender.com/health). The free Render service may take approximately one minute to wake after inactivity. Continue when it returns a healthy response.
2. **Open the product:** Visit the [live SpaceBNS dashboard](https://spacebns.vercel.app/). If it initially shows that data is unavailable, select **Retry data load** after the backend wakes.
3. **Confirm the trust boundary:** In **Audit & Provenance**, verify that the response identifies synthetic data, a non-flight-qualified prototype, simulation-only policy, and `COMMAND_AUTHORITY: NONE`.
4. **Review the AI estimate:** Inspect the learned 24-hour breach probability and its strongest feature contributions. These are disclosed as synthetic-data associations, not proven physical causes.
5. **Find the defining moment:** SpaceBNS omits the deterministic physical projection when essential assumptions are unavailable. It states why the projection was withheld instead of inventing missing physics.
6. **Check the remaining evidence:** Deterministic safety thresholds are still evaluated, and a human-readable advisory remains available even when the physical projection is omitted.
7. **Confirm human control:** The result remains advisory only. SpaceBNS cannot authorize or transmit a spacecraft command.

## Direct evidence

- [3-minute project demo](https://youtu.be/dAUYIlgKXaA)
- [Public prediction response](https://spacebns-api.onrender.com/api/v1/mock/power-risk-prediction)
- [Validation evidence](README.md#validation-evidence)
- [Architecture and assurance layers](docs/architecture.md)
- [Power-risk implementation contract](docs/power-risk-contract.md)
- [Human-reviewed IBM Bob usage record](docs/bob-usage.md)

## Judge takeaway

Most AI systems are optimized to always return an answer. SpaceBNS is designed to separate a learned risk estimate from deterministic engineering evidence, make missing assumptions visible, and preserve human command authority.

> **Scope:** Public proof of concept using synthetic data only. SpaceBNS is not flight-qualified and makes no real-spacecraft performance claim.

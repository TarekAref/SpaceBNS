# SpaceBNS Architecture

## Architectural intent

SpaceBNS is a hybrid ground-edge mission-assurance prototype. The architecture
separates probabilistic inference, physical forecasting, causal reasoning,
deterministic policy enforcement, natural-language explanation, and operator
authority. This separation is the primary safety and explainability boundary.

## Component responsibilities

| Component | Responsibility | Explicit non-responsibility |
| --- | --- | --- |
| Edge image-utility module | Detect bounded image symptoms and produce a structured utility result | Does not diagnose spacecraft hardware from an image alone |
| Telemetry gateway | Validate and normalize approved telemetry inputs | Does not accept arbitrary spacecraft commands |
| Anomaly detector | Identify deviations and attach evidence identifiers | Does not select recovery actions |
| Physics-informed PHM | Forecast energy state and threshold crossings | Does not claim certainty about component failure |
| Causal/dependency graph | Rank relationships among symptoms, commands, and subsystems | Does not guarantee root cause |
| Policy safety cage | Check allowlists, preconditions, inhibits, confidence, and action mode | Does not use free-form LLM output as executable policy |
| Granite explanation layer | Explain validated evidence and retrieved procedures | Has no command authority |
| Operator dashboard | Present evidence, uncertainty, forecast, and policy results | Is not a flight-control console |

## Decision contract

Every future AI or diagnostic result should use a structured contract containing:

- event and evidence identifiers;
- UTC timestamp and source;
- model name, version, and configuration;
- predicted class or diagnostic hypothesis;
- calibrated confidence or explicit uncertainty status;
- corroborating and contradicting evidence;
- candidate response;
- policy decision and reason;
- action mode; and
- audit status.

## Ground-edge boundary

The August MVP uses a simulated edge node. Docker Compose runs only the public
ground-side development environment. A future embedded target would require a
separate build profile, target toolchain, memory map, timing analysis, watchdog
behavior, interruption tolerance, interface-control documentation, and
processor-representative verification.

## Hera adaptation boundary

An ESA Hera Core 1 experiment would be a separate mission profile. It would use
only approved hosted interfaces, run asynchronously in bounded execution
windows, tolerate immediate termination, and produce reports or
recommendations. No direct hardware or Core 0 command access is assumed.

Earth-observation cloud masking would also be replaced by a mission-specific
asteroid-imaging task such as feature-track quality, landmark visibility,
illumination suitability, saturation, blur, or observation prioritization.

## Assurance references

NASA and ECSS materials may inform future engineering and V&V planning, but the
challenge prototype does not claim compliance, certification, qualification,
or flight heritage.


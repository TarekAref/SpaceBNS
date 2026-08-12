# BNS Innovation IP Safeguard Matrix

## Public repository content

The following categories may be published after review:

- general, non-proprietary orchestration and API contracts;
- React/Next.js dashboard layout and public presentation components;
- standard synthetic telemetry and generalized operational scenarios;
- non-sensitive architecture documentation;
- public tests and development tooling; and
- adapters that contain no private model logic or credentials.

## Excluded proprietary content

The following categories must remain outside this public repository:

- proprietary model weights, calibration methods, and custom-trained edge AI;
- private training, validation, or customer datasets;
- proprietary parsing, feature-engineering, causal-ranking, or forecast logic;
- real spacecraft or ground-network topology;
- client identities, contracts, procedures, telemetry, or command history;
- credentials, tokens, certificates, private keys, and security configuration;
- export-controlled or mission-sensitive material; and
- unpublished invention disclosures and patent-sensitive implementation detail.

## Pre-commit release gate

Before any public push, verify:

1. The dataset is synthetic or explicitly approved for public release.
2. No secret is present in current files or Git history.
3. No client, mission, or partner can be identified from the content.
4. No proprietary model artifact is included.
5. Public interfaces reveal only what is required for interoperability.
6. Claims accurately distinguish implemented, simulated, planned, and future work.
7. Every published file is intentionally offered under Apache License 2.0.

## License boundary

Copyright ownership and open-source licensing are different concepts.
BNS Innovation retains copyright in its original public contributions while
licensing those published files under Apache License 2.0. The license permits
use, modification, distribution, and commercial use of published code under
its conditions. It does not license excluded technology that was never
contributed to this repository.

Do not rely on a README statement to preserve a trade secret after disclosure.
If material must remain secret, it must never be committed to the public
repository—including in a deleted commit or earlier branch.


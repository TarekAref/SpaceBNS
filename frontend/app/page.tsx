"use client";

import { useEffect, useState } from "react";

type TelemetrySample = {
  timestamp: string;
  solar_array_current_a: number;
  payload_power_draw_w: number;
  bus_voltage_v: number;
  battery_soc_percent: number;
  command_activity: string;
  communications_status: string;
  image_utility_score: number;
};

type Assessment = {
  scenario_id: string;
  risk_level: string;
  diagnostic_hypothesis: string;
  recommendation: string;
  policy_decision: string;
  command_authority: string;
  findings: Array<{ code: string; evidence: string }>;
};

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Dashboard() {
  const [latest, setLatest] = useState<TelemetrySample | null>(null);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadScenario() {
      try {
        const [telemetryResponse, assessmentResponse] = await Promise.all([
          fetch(`${apiBaseUrl}/api/v1/mock/telemetry`),
          fetch(`${apiBaseUrl}/api/v1/mock/assessment`),
        ]);

        if (!telemetryResponse.ok || !assessmentResponse.ok) {
          throw new Error("The public demonstration API is unavailable.");
        }

        const telemetry = await telemetryResponse.json();
        const assessmentPayload = await assessmentResponse.json();
        setLatest(telemetry.samples.at(-1));
        setAssessment(assessmentPayload);
      } catch (loadError) {
        setError(
          loadError instanceof Error ? loadError.message : "Unknown API error",
        );
      }
    }

    loadScenario();
  }, []);

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">BNS Innovation · IBM August Challenge</p>
          <h1>SpaceBNS Mission Assurance</h1>
          <p className="subtitle">
            Evidence-grounded, forecast-aware, policy-constrained decisions for
            resource-limited spacecraft.
          </p>
        </div>
        <div className="mode">SIMULATION ONLY</div>
      </header>

      {error && <section className="error">{error}</section>}

      <section className="metrics" aria-label="Latest synthetic telemetry">
        <Metric label="Bus voltage" value={latest ? `${latest.bus_voltage_v} V` : "—"} />
        <Metric label="Battery SOC" value={latest ? `${latest.battery_soc_percent}%` : "—"} />
        <Metric label="Payload draw" value={latest ? `${latest.payload_power_draw_w} W` : "—"} />
        <Metric label="Image utility" value={latest ? latest.image_utility_score.toFixed(2) : "—"} />
      </section>

      <section className="grid">
        <article className="panel">
          <p className="panelLabel">Operational context</p>
          <h2>{latest?.command_activity ?? "Loading scenario…"}</h2>
          <dl>
            <div>
              <dt>Communications</dt>
              <dd>{latest?.communications_status ?? "—"}</dd>
            </div>
            <div>
              <dt>Solar-array current</dt>
              <dd>{latest ? `${latest.solar_array_current_a} A` : "—"}</dd>
            </div>
            <div>
              <dt>Latest timestamp</dt>
              <dd>{latest ? new Date(latest.timestamp).toLocaleString() : "—"}</dd>
            </div>
          </dl>
        </article>

        <article className="panel accent">
          <p className="panelLabel">Decision support</p>
          <div className={`risk risk-${assessment?.risk_level ?? "unknown"}`}>
            {assessment?.risk_level?.toUpperCase() ?? "LOADING"}
          </div>
          <h2>{assessment?.recommendation ?? "Evaluating mock evidence…"}</h2>
          <p>{assessment?.diagnostic_hypothesis}</p>
          <p className="policy">
            Policy: {assessment?.policy_decision ?? "—"} · Command authority:{" "}
            {assessment?.command_authority ?? "—"}
          </p>
        </article>
      </section>

      <section className="panel findings">
        <p className="panelLabel">Evidence findings</p>
        <ul>
          {assessment?.findings.map((finding) => (
            <li key={finding.code}>
              <strong>{finding.code}</strong>
              <span>{finding.evidence}</span>
            </li>
          )) ?? <li>Loading public demonstration evidence…</li>}
        </ul>
      </section>

      <footer>
        Synthetic data · No spacecraft connection · Not flight-qualified
      </footer>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <article className="metric">
      <p>{label}</p>
      <strong>{value}</strong>
    </article>
  );
}


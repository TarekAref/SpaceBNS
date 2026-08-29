"use client";

import { useEffect, useState } from "react";

// ---------------------------------------------------------------------------
// API types — strict mapping of the documented contract response structures
// ---------------------------------------------------------------------------

type Contribution = {
  feature: string;
  standardized_value: number;
  coefficient: number;
  contribution: number;
};

type AiPrediction = {
  label: string;
  predicted_class: number;
  breach_probability: number;
  probability_note: string;
  top_contributions: Contribution[];
  all_contributions: Contribution[];
};

type HourlyEntry = {
  hour_offset: number;
  forecast_timestamp: string;
  projected_soc_percent: number;
  projected_breach: boolean;
};

type DeterministicProjection = {
  method: string;
  not_ai_output: boolean;
  assumption_note: string;
  window_complete: boolean;
  hourly_projection: HourlyEntry[];
};

type SafetyFinding = {
  code: string;
  evidence: string;
};

type Advisory = {
  risk_summary: string;
  recommendation: string;
  basis: string;
  human_action_required: boolean;
  authority_note: string;
};

type Audit = {
  features_used: number;
  window_complete: boolean;
  samples_used: number;
  window_hours: number;
  action_mode: string;
};

// Normal response (status: "ok")
type NormalResponse = {
  data_source: "SYNTHETIC";
  prototype_status: "NOT_FLIGHT_QUALIFIED";
  command_authority: "NONE";
  policy_decision: "PERMITTED_FOR_SIMULATION_ONLY";
  scenario_id: string;
  query_timestamp: string;
  model_claim: string;
  model_version: string;
  status: "ok";
  ai_prediction: AiPrediction;
  deterministic_projection: DeterministicProjection | null;
  projection_omitted_reason: string | null;
  safety_threshold_findings: SafetyFinding[];
  advisory: Advisory;
  audit: Audit;
};

// Degraded response (status: "degraded")
type DegradedResponse = {
  data_source: "SYNTHETIC";
  prototype_status: "NOT_FLIGHT_QUALIFIED";
  command_authority: "NONE";
  policy_decision: "PERMITTED_FOR_SIMULATION_ONLY";
  status: "degraded";
  degraded_reason: string;
  ai_prediction: null;
  breach_probability: null;
  deterministic_projection: DeterministicProjection | null;
  projection_omitted_reason: string | null;
  safety_threshold_findings: SafetyFinding[];
  advisory: Advisory;
  audit: Audit;
  scenario_id?: string;
  query_timestamp?: string;
  model_claim?: string;
  model_version?: string;
};

type PowerRiskResponse = NormalResponse | DegradedResponse;

// ---------------------------------------------------------------------------
// Runtime validation — replaces all unsafe `as` casts
// ---------------------------------------------------------------------------

/** Exact required values for the safety envelope. */
const SAFETY_ENVELOPE = {
  data_source: "SYNTHETIC",
  prototype_status: "NOT_FLIGHT_QUALIFIED",
  command_authority: "NONE",
  policy_decision: "PERMITTED_FOR_SIMULATION_ONLY",
} as const;

// Logistic-regression terms are expressed in log-odds. Values beyond these
// conservative presentation bounds indicate a stale or numerically unstable
// model response and must fail closed into the existing generic error state.
const MAX_ABS_STANDARDIZED_VALUE = 100;
const MAX_ABS_COEFFICIENT = 100;
const MAX_ABS_CONTRIBUTION = 100;

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && isFinite(v);
}

function isString(v: unknown): v is string {
  return typeof v === "string";
}

function isBoolean(v: unknown): v is boolean {
  return typeof v === "boolean";
}

function isNonNegativeInteger(v: unknown): v is number {
  return typeof v === "number" && isFinite(v) && v >= 0 && Number.isInteger(v);
}

function validateAudit(a: unknown): a is Audit {
  if (!a || typeof a !== "object") return false;
  const o = a as Record<string, unknown>;
  return (
    isNonNegativeInteger(o.features_used) &&
    o.features_used <= 12 &&
    isBoolean(o.window_complete) &&
    isNonNegativeInteger(o.samples_used) &&
    o.samples_used <= 72 &&
    isFiniteNumber(o.window_hours) &&
    o.window_hours > 0 &&
    o.window_hours <= 6 &&
    o.action_mode === "simulation-only"
  );
}

function validateContribution(c: unknown): c is Contribution {
  if (!c || typeof c !== "object") return false;
  const o = c as Record<string, unknown>;
  return (
    isString(o.feature) &&
    isFiniteNumber(o.standardized_value) &&
    Math.abs(o.standardized_value) <= MAX_ABS_STANDARDIZED_VALUE &&
    isFiniteNumber(o.coefficient) &&
    Math.abs(o.coefficient) <= MAX_ABS_COEFFICIENT &&
    isFiniteNumber(o.contribution) &&
    Math.abs(o.contribution) <= MAX_ABS_CONTRIBUTION
  );
}

function validateAiPrediction(p: unknown): p is AiPrediction {
  if (!p || typeof p !== "object") return false;
  const o = p as Record<string, unknown>;
  const prob = o.breach_probability;
  return (
    isString(o.label) &&
    (o.predicted_class === 0 || o.predicted_class === 1) &&
    isFiniteNumber(prob) &&
    (prob as number) >= 0 &&
    (prob as number) <= 1 &&
    isString(o.probability_note) &&
    Array.isArray(o.top_contributions) &&
    (o.top_contributions as unknown[]).every(validateContribution) &&
    Array.isArray(o.all_contributions) &&
    (o.all_contributions as unknown[]).every(validateContribution)
  );
}

function validateHourlyEntry(e: unknown): e is HourlyEntry {
  if (!e || typeof e !== "object") return false;
  const o = e as Record<string, unknown>;
  return (
    isFiniteNumber(o.hour_offset) &&
    isString(o.forecast_timestamp) &&
    isFiniteNumber(o.projected_soc_percent) &&
    (o.projected_soc_percent as number) >= 0 &&
    (o.projected_soc_percent as number) <= 100 &&
    isBoolean(o.projected_breach)
  );
}

function validateDeterministicProjection(p: unknown): p is DeterministicProjection {
  if (!p || typeof p !== "object") return false;
  const o = p as Record<string, unknown>;
  return (
    isString(o.method) &&
    isBoolean(o.not_ai_output) &&
    isString(o.assumption_note) &&
    isBoolean(o.window_complete) &&
    Array.isArray(o.hourly_projection) &&
    (o.hourly_projection as unknown[]).every(validateHourlyEntry)
  );
}

function validateSafetyFinding(f: unknown): f is SafetyFinding {
  if (!f || typeof f !== "object") return false;
  const o = f as Record<string, unknown>;
  return isString(o.code) && isString(o.evidence);
}

function validateAdvisory(a: unknown): a is Advisory {
  if (!a || typeof a !== "object") return false;
  const o = a as Record<string, unknown>;
  return (
    isString(o.risk_summary) &&
    isString(o.recommendation) &&
    isString(o.basis) &&
    isBoolean(o.human_action_required) &&
    isString(o.authority_note)
  );
}

/**
 * Full runtime validation of the API response.
 * Returns a typed NormalResponse or DegradedResponse, or null if invalid.
 * Never throws — all type assertions have been verified by this point.
 */
function validateResponse(json: unknown): PowerRiskResponse | null {
  if (!json || typeof json !== "object") return null;
  const o = json as Record<string, unknown>;

  // Safety envelope — all four fields must be exactly correct.
  if (o.data_source !== SAFETY_ENVELOPE.data_source) return null;
  if (o.prototype_status !== SAFETY_ENVELOPE.prototype_status) return null;
  if (o.command_authority !== SAFETY_ENVELOPE.command_authority) return null;
  if (o.policy_decision !== SAFETY_ENVELOPE.policy_decision) return null;

  // Shared required fields.
  if (!validateAudit(o.audit)) return null;
  if (!Array.isArray(o.safety_threshold_findings)) return null;
  if (!(o.safety_threshold_findings as unknown[]).every(validateSafetyFinding)) return null;
  if (!validateAdvisory(o.advisory)) return null;

  // Projection is nullable but if present must be valid.
  if (
    o.deterministic_projection !== null &&
    o.deterministic_projection !== undefined &&
    !validateDeterministicProjection(o.deterministic_projection)
  ) return null;

  if (o.status === "ok") {
    if (!validateAiPrediction(o.ai_prediction)) return null;
    if (!isString(o.scenario_id)) return null;
    if (!isString(o.query_timestamp)) return null;
    if (!isString(o.model_claim)) return null;
    if (!isString(o.model_version)) return null;
    return json as NormalResponse;
  }

  if (o.status === "degraded") {
    if (o.ai_prediction !== null && o.ai_prediction !== undefined) return null;
    if (o.breach_probability !== null && o.breach_probability !== undefined) return null;
    if (!isString(o.degraded_reason)) return null;
    return json as DegradedResponse;
  }

  return null;
}

function formatProbability(probability: number): string {
  const percent = probability * 100;
  if (percent === 0) return "0%";
  if (percent < 0.1) return "<0.1%";
  if (percent < 10) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}

function formatContribution(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0.0000";
  if (magnitude >= 1000 || magnitude < 0.001) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}

// ---------------------------------------------------------------------------
// State union — all fetch outcomes
// ---------------------------------------------------------------------------

type DashboardState =
  | { phase: "loading" }
  | { phase: "ok"; data: NormalResponse }
  | { phase: "degraded"; data: DegradedResponse }
  | { phase: "error"; message: string };

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const PREDICTION_ENDPOINT = `${apiBaseUrl}/api/v1/mock/power-risk-prediction`;

// ---------------------------------------------------------------------------
// Root component
// ---------------------------------------------------------------------------

export default function Dashboard() {
  const [state, setState] = useState<DashboardState>({ phase: "loading" });

  function loadData() {
    setState({ phase: "loading" });

    fetch(PREDICTION_ENDPOINT)
      .then((res) => {
        if (!res.ok) {
          return Promise.reject(
            new Error("The public demonstration API is unavailable.")
          );
        }
        return res.json();
      })
      .then((json: unknown) => {
        const validated = validateResponse(json);
        if (validated === null) {
          setState({
            phase: "error",
            message: "Received an invalid or unexpected response from the API.",
          });
          return;
        }
        if (validated.status === "ok") {
          setState({ phase: "ok", data: validated });
        } else {
          setState({ phase: "degraded", data: validated });
        }
      })
      .catch(() => {
        setState({
          phase: "error",
          message:
            "The public demonstration API is unavailable. Verify the backend is running and retry.",
        });
      });
  }

  useEffect(() => {
    loadData();
    // No auto-poll — manual retry only (requirement 11).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const advisory =
    state.phase === "ok" || state.phase === "degraded"
      ? state.data.advisory
      : null;

  return (
    <main>
      {/* Header */}
      <header className="hero">
        <div>
          <p className="eyebrow">BNS Innovation · IBM August Challenge</p>
          <h1>SpaceBNS Mission Assurance</h1>
          <p className="subtitle">
            Evidence-grounded, forecast-aware, policy-constrained advisory for
            resource-limited spacecraft.
          </p>
        </div>
        <div className="mode" aria-label="Operating mode">
          SIMULATION ONLY
        </div>
      </header>

      {/* Permanent safety strip — always visible regardless of API state */}
      <SafetyStrip />

      {/* Advisory banner — shown when data is loaded */}
      {advisory && <AdvisoryBanner advisory={advisory} />}

      {/* Error panel */}
      {state.phase === "error" && (
        <section className="error" role="alert">
          <strong>API unavailable</strong>
          <p>{state.message}</p>
          <button className="retryBtn" onClick={loadData} type="button">
            Retry data load
          </button>
        </section>
      )}

      {/* Loading indicator */}
      {state.phase === "loading" && (
        <section className="loadingPanel" aria-live="polite" aria-busy="true">
          <span className="spinner" aria-hidden="true" />
          Loading synthetic demonstration data…
        </section>
      )}

      {/* Four-layer dashboard — only when data is present */}
      {(state.phase === "ok" || state.phase === "degraded") && (
        <FourLayerDashboard state={state} />
      )}

      <footer>
        Synthetic data · No spacecraft connection · Not flight-qualified ·
        Advisory only · IBM Bob development tool
      </footer>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Safety strip (requirement 3)
// ---------------------------------------------------------------------------

function SafetyStrip() {
  return (
    <div className="safetyStrip" role="region" aria-label="Safety boundaries">
      <span>SYNTHETIC DATA</span>
      <span aria-hidden="true">·</span>
      <span>NOT FLIGHT QUALIFIED</span>
      <span aria-hidden="true">·</span>
      <span>COMMAND AUTHORITY: NONE</span>
      <span aria-hidden="true">·</span>
      <span>SIMULATION ONLY</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Advisory banner (L4 summary — top-of-page priority)
// ---------------------------------------------------------------------------

function AdvisoryBanner({ advisory }: { advisory: Advisory }) {
  const levelClass =
    advisory.risk_summary === "HIGH"
      ? "advisoryHigh"
      : advisory.risk_summary === "ELEVATED"
        ? "advisoryElevated"
        : advisory.risk_summary === "UNKNOWN"
          ? "advisoryUnknown"
          : "advisoryNominal";

  return (
    <div
      className={`advisoryBanner ${levelClass}`}
      role="region"
      aria-label="Operator advisory"
    >
      <span className="advisoryLevel">{advisory.risk_summary}</span>
      <span className="advisoryRec">{advisory.recommendation}</span>
      <span className="advisoryNote">
        {advisory.human_action_required
          ? "Human action required"
          : "No immediate action required"}{" "}
        · {advisory.authority_note}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Four-layer dashboard
// ---------------------------------------------------------------------------

function FourLayerDashboard({
  state,
}: {
  state:
    | { phase: "ok"; data: NormalResponse }
    | { phase: "degraded"; data: DegradedResponse };
}) {
  const data = state.data;

  return (
    <>
      {/* Provenance metadata row */}
      <ProvenancePanel data={data} />

      {/* L1 — AI power-risk estimate */}
      <L1Panel data={data} />

      {/* Explainability (contributions) — only when ai_prediction is available */}
      {data.ai_prediction !== null && (
        <ExplainabilityPanel prediction={data.ai_prediction} />
      )}

      {/* L2 — deterministic energy projection */}
      <L2Panel
        projection={data.deterministic_projection}
        omittedReason={data.projection_omitted_reason ?? null}
      />

      {/* L3 — safety threshold findings */}
      <L3Panel findings={data.safety_threshold_findings} />

      {/* L4 — operator advisory (detail) */}
      <L4Panel advisory={data.advisory} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Provenance / audit panel
// ---------------------------------------------------------------------------

function ProvenancePanel({ data }: { data: PowerRiskResponse }) {
  const audit = data.audit;

  function val(v: string | number | boolean | null | undefined): string {
    if (v === null || v === undefined) return "not available";
    return String(v);
  }

  return (
    <section className="panel provenancePanel" aria-label="Audit and provenance">
      <p className="panelLabel">Audit &amp; Provenance</p>
      <dl className="provenanceDl">
        {"scenario_id" in data && (
          <div>
            <dt>Scenario ID</dt>
            <dd>{val(data.scenario_id)}</dd>
          </div>
        )}
        {"query_timestamp" in data && (
          <div>
            <dt>Query timestamp</dt>
            <dd>{data.query_timestamp ? new Date(data.query_timestamp).toLocaleString() : "not available"}</dd>
          </div>
        )}
        {"model_claim" in data && (
          <div>
            <dt>Model claim</dt>
            <dd className="monoSmall">{val(data.model_claim)}</dd>
          </div>
        )}
        <div>
          <dt>Samples used</dt>
          <dd>{val(audit.samples_used)}</dd>
        </div>
        <div>
          <dt>Window hours</dt>
          <dd>{val(audit.window_hours)}</dd>
        </div>
        <div>
          <dt>Features used</dt>
          <dd>{val(audit.features_used)}</dd>
        </div>
        <div>
          <dt>Action mode</dt>
          <dd>{val(audit.action_mode)}</dd>
        </div>
        <div>
          <dt>Data source</dt>
          <dd>{val(data.data_source)}</dd>
        </div>
        <div>
          <dt>Prototype status</dt>
          <dd>{val(data.prototype_status)}</dd>
        </div>
      </dl>
    </section>
  );
}

// ---------------------------------------------------------------------------
// L1 — AI power-risk estimate
// ---------------------------------------------------------------------------

function L1Panel({ data }: { data: PowerRiskResponse }) {
  const isNormal = data.status === "ok";
  const pred = data.ai_prediction;
  const audit = data.audit;
  const probabilityDisplay = pred
    ? formatProbability(pred.breach_probability)
    : null;

  return (
    <section
      className="panel l1Panel"
      aria-label="L1: AI power-risk estimate"
    >
      <p className="panelLabel layerTag">L1 — AI Power-Risk Estimate</p>

      {isNormal && pred !== null ? (
        <>
          <div className="probRow">
            <span className="probLabel">Breach probability</span>
            <span
              className={`probValue ${pred.breach_probability >= 0.7 ? "probHigh" : pred.breach_probability >= 0.4 ? "probElevated" : "probLow"}`}
              aria-label={`${probabilityDisplay} breach probability`}
            >
              {probabilityDisplay}
            </span>
          </div>

          <p className="disclaimerNote">
            Synthetic-distribution model estimate. Not a real-spacecraft failure
            probability. Never hardcoded — derived from the API at query time.
          </p>

          <dl className="infoGrid">
            <div>
              <dt>Predicted class</dt>
              <dd>{pred.predicted_class === 1 ? "Breach predicted (1)" : "No breach predicted (0)"}</dd>
            </div>
            <div>
              <dt>Model version</dt>
              <dd>{"model_version" in data ? (data as NormalResponse).model_version : "—"}</dd>
            </div>
            <div>
              <dt>Inference basis</dt>
              <dd>{audit.window_hours}-hour window · {audit.samples_used} samples</dd>
            </div>
            <div>
              <dt>Label</dt>
              <dd className="monoSmall">{pred.label}</dd>
            </div>
          </dl>

          <details className="probNoteDetail">
            <summary>Probability note (click to expand)</summary>
            <p className="probNoteText">{pred.probability_note}</p>
          </details>
        </>
      ) : (
        <div className="degradedBox" role="alert">
          <strong>AI estimate unavailable</strong>
          {"degraded_reason" in data && data.degraded_reason && (
            <p>{data.degraded_reason}</p>
          )}
          <p className="disclaimerNote">
            The synthetic-distribution classifier requires {audit.samples_used} samples
            ({audit.window_hours}-hour window at 5-minute cadence). The API returned no AI estimate for
            this query.
          </p>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Explainability — top contributions (requirement 6)
// ---------------------------------------------------------------------------

function ExplainabilityPanel({ prediction }: { prediction: AiPrediction }) {
  const top = prediction.top_contributions;
  if (!top || top.length === 0) return null;

  const maxAbs = Math.max(...top.map((c) => Math.abs(c.contribution)), 1e-9);

  return (
    <section
      className="panel explainPanel"
      aria-label="L1 model explainability: top feature contributions"
    >
      <p className="panelLabel layerTag">
        L1 — Model Explainability (Top Contributions)
      </p>
      <p className="disclaimerNote">
        These bars represent learned associations between input features and the
        model output — not proven physical causal relationships. Positive
        contributions increase the predicted breach probability; negative
        contributions decrease it. Values are never hardcoded.
      </p>

      <ul className="contribList" aria-label="Feature contributions">
        {top.map((c) => {
          const pct = (Math.abs(c.contribution) / maxAbs) * 100;
          const isPos = c.contribution >= 0;
          const displayValue = formatContribution(c.contribution);
          const magnitudeLabel = formatContribution(Math.abs(c.contribution));
          return (
            <li key={c.feature} className="contribItem">
              <span className="contribFeature">{c.feature}</span>
              <div className="contribBarWrap" aria-hidden="true">
                <div
                  className={`contribBar ${isPos ? "contribPos" : "contribNeg"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span
                className={`contribVal ${isPos ? "contribPos" : "contribNeg"}`}
                aria-label={`contribution ${c.contribution >= 0 ? "positive" : "negative"} ${magnitudeLabel}`}
              >
                {c.contribution > 0 ? "+" : ""}
                {displayValue}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ---------------------------------------------------------------------------
// L2 — deterministic energy projection (requirement 7)
// ---------------------------------------------------------------------------

function L2Panel({
  projection,
  omittedReason,
}: {
  projection: DeterministicProjection | null;
  omittedReason: string | null;
}) {
  if (projection === null) {
    return (
      <section
        className="panel l2Panel"
        aria-label="L2: deterministic energy projection"
      >
        <p className="panelLabel layerTag">L2 — Deterministic Energy Projection</p>
        <div className="omittedBox">
          <strong>Projection not available</strong>
          {omittedReason && <p>{omittedReason}</p>}
          <p className="disclaimerNote">
            SpaceBNS does not fabricate a projection without complete physical
            assumptions (battery capacity, base load, conversion efficiency,
            sunlight schedule, payload schedule). This is NOT an AI output.
          </p>
        </div>
      </section>
    );
  }

  const points = projection.hourly_projection;
  const hasPoints = Array.isArray(points) && points.length === 24;

  return (
    <section
      className="panel l2Panel"
      aria-label="L2: deterministic energy projection"
    >
      <p className="panelLabel layerTag">L2 — Deterministic Energy Projection</p>
      <p className="notAiTag">NOT AI OUTPUT · {projection.method}</p>

      {hasPoints ? (
        <SocChart points={points} />
      ) : (
        <p className="disclaimerNote">
          Projection data incomplete (expected 24 hourly points).
        </p>
      )}

      <details className="probNoteDetail">
        <summary>Assumption note</summary>
        <p className="probNoteText">{projection.assumption_note}</p>
      </details>
    </section>
  );
}

// Inline SVG SOC chart (dependency-free, requirement 7)
function SocChart({ points }: { points: HourlyEntry[] }) {
  const W = 600;
  const H = 200;
  const PAD = { top: 16, right: 16, bottom: 32, left: 44 };
  const SOC_THRESHOLD = 25;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  function xOf(i: number) {
    return PAD.left + (i / (points.length - 1)) * plotW;
  }

  function yOf(soc: number) {
    return PAD.top + plotH - (soc / 100) * plotH;
  }

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(i).toFixed(1)} ${yOf(p.projected_soc_percent).toFixed(1)}`)
    .join(" ");

  const thresholdY = yOf(SOC_THRESHOLD);

  const breachPoints = points.filter((p) => p.projected_breach);
  const breachCount = breachPoints.length;
  const socTitleId = "soc-chart-title";
  const socDescId = "soc-chart-desc";

  return (
    <figure className="socChart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-labelledby={`${socTitleId} ${socDescId}`}
        className="socSvg"
        style={{ width: "100%", height: "auto" }}
      >
        <title id={socTitleId}>Projected state-of-charge over {points.length} hours</title>
        <desc id={socDescId}>
          Deterministic (not AI) energy balance projection. SOC starts at{" "}
          {points[0].projected_soc_percent.toFixed(1)}% and ends at{" "}
          {points[points.length - 1].projected_soc_percent.toFixed(1)}%.{" "}
          {breachCount > 0
            ? `${breachCount} projected breach point${breachCount > 1 ? "s" : ""} below the 25% SOC safety threshold.`
            : "No projected SOC breaches below the 25% safety threshold."}
        </desc>
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((soc) => (
          <line
            key={soc}
            x1={PAD.left}
            x2={PAD.left + plotW}
            y1={yOf(soc)}
            y2={yOf(soc)}
            stroke="rgba(141,171,255,0.12)"
            strokeWidth="1"
          />
        ))}

        {/* 25% SOC safety threshold */}
        <line
          x1={PAD.left}
          x2={PAD.left + plotW}
          y1={thresholdY}
          y2={thresholdY}
          stroke="#ff6b6b"
          strokeWidth="1.5"
          strokeDasharray="5 3"
        />
        <text
          x={PAD.left + 4}
          y={thresholdY - 4}
          fontSize="10"
          fill="#ff6b6b"
          aria-hidden="true"
        >
          25% SOC safety threshold
        </text>

        {/* SOC line */}
        <path
          d={pathD}
          fill="none"
          stroke="#4ee1d0"
          strokeWidth="2"
          strokeLinejoin="round"
          aria-label="Projected SOC curve"
        />

        {/* Breach points */}
        {breachPoints.map((p) => {
          const idx = points.indexOf(p);
          return (
            <circle
              key={idx}
              cx={xOf(idx)}
              cy={yOf(p.projected_soc_percent)}
              r="4"
              fill="#ff6b6b"
              aria-label={`Projected breach at hour ${p.hour_offset}: ${p.projected_soc_percent.toFixed(1)}%`}
            />
          );
        })}

        {/* Y-axis labels */}
        {[0, 25, 50, 75, 100].map((soc) => (
          <text
            key={soc}
            x={PAD.left - 4}
            y={yOf(soc) + 4}
            fontSize="10"
            fill="rgba(170,181,213,0.8)"
            textAnchor="end"
            aria-hidden="true"
          >
            {soc}%
          </text>
        ))}

        {/* X-axis labels */}
        {[0, 6, 12, 18, 23].map((idx) => (
          <text
            key={idx}
            x={xOf(idx)}
            y={H - 6}
            fontSize="10"
            fill="rgba(170,181,213,0.8)"
            textAnchor="middle"
            aria-hidden="true"
          >
            h+{points[idx].hour_offset}
          </text>
        ))}
      </svg>
      <figcaption className="chartCaption">
        Deterministic energy balance projection · NOT AI OUTPUT · Red dots =
        projected SOC breach below 25%
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------
// L3 — safety threshold findings (requirement 8)
// ---------------------------------------------------------------------------

function L3Panel({ findings }: { findings: SafetyFinding[] }) {
  return (
    <section
      className="panel l3Panel"
      aria-label="L3: safety threshold findings"
    >
      <p className="panelLabel layerTag">L3 — Safety Threshold Findings</p>

      {findings.length === 0 ? (
        <p className="noFindings">No active threshold findings</p>
      ) : (
        <ul className="findingsList">
          {findings.map((f) => (
            <li key={f.code} className="findingItem">
              <strong className="findingCode">{f.code}</strong>
              <span className="findingEvidence">{f.evidence}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// L4 — operator advisory detail (requirement 9)
// ---------------------------------------------------------------------------

function L4Panel({ advisory }: { advisory: Advisory }) {
  const levelClass =
    advisory.risk_summary === "HIGH"
      ? "l4High"
      : advisory.risk_summary === "ELEVATED"
        ? "l4Elevated"
        : advisory.risk_summary === "UNKNOWN"
          ? "l4Unknown"
          : "l4Nominal";

  return (
    <section
      className={`panel l4Panel ${levelClass}`}
      aria-label="L4: operator advisory"
    >
      <p className="panelLabel layerTag">L4 — Operator Advisory</p>

      <div className="l4Header">
        <span className="l4Risk">{advisory.risk_summary}</span>
        <span className="l4Rec">{advisory.recommendation}</span>
      </div>

      <p className="l4Basis">{advisory.basis}</p>

      <dl className="l4Dl">
        <div>
          <dt>Human action required</dt>
          <dd>{advisory.human_action_required ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt>Authority note</dt>
          <dd>{advisory.authority_note}</dd>
        </div>
      </dl>

      <p className="advisoryOnlyNote">
        This recommendation is operator decision support only. It does not
        constitute a spacecraft command or authorisation for autonomous action.
        All outputs require human review before any operational decision.
      </p>
    </section>
  );
}

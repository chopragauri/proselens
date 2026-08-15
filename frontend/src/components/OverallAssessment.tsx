import type { AnalyzeResponse } from '../types/analysis';
import { ASSESSMENT_LABELS, BAND_LABELS } from '../types/analysis';
import { BAND_BADGE, BAND_MARKER, formatSigned } from '../utils/risk';
import type { RiskBand } from '../types/analysis';

function bandForScore(score: number): RiskBand {
  if (score <= 35) return 'low';
  if (score >= 65) return 'high';
  return 'medium';
}

interface Props {
  result: AnalyzeResponse;
}

export function OverallAssessment({ result }: Props) {
  const band = bandForScore(result.risk_score);
  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            {ASSESSMENT_LABELS[result.assessment]}
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Model {result.model_version}
          </p>
        </div>

        <div className="flex gap-6">
          <div className="text-right">
            <p className="text-xs uppercase tracking-wide text-slate-500">Risk score</p>
            <p className="text-3xl font-semibold tabular-nums text-slate-900">
              {result.risk_score}
              <span className="text-base font-normal text-slate-500"> / 100</span>
            </p>
            <span
              className={`mt-1 inline-block rounded px-2 py-0.5 text-xs font-medium ${BAND_BADGE[band]}`}
            >
              {BAND_MARKER[band]} {BAND_LABELS[band]}
            </span>
          </div>

          {/*
            Confidence is shown beside risk, never folded into it. A high risk
            score backed by two sentences is a different claim from the same
            score backed by thirty, and collapsing them would hide that.
          */}
          <div className="text-right">
            <p className="text-xs uppercase tracking-wide text-slate-500">Confidence</p>
            <p className="text-3xl font-semibold tabular-nums text-slate-900">
              {confidencePercent}
              <span className="text-base font-normal text-slate-500">%</span>
            </p>
            <span
              className={`mt-1 inline-block rounded px-2 py-0.5 text-xs font-medium ${BAND_BADGE[result.confidence_band]}`}
            >
              {BAND_LABELS[result.confidence_band]}
            </span>
          </div>
        </div>
      </div>

      <p className="mt-4 border-t border-slate-100 pt-4 text-sm leading-relaxed text-slate-700">
        {result.summary_text}
      </p>

      {result.signals.length > 0 && (
        <div className="mt-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Strongest document-level signals
          </h3>
          <ul className="mt-2 space-y-1.5">
            {result.signals.slice(0, 4).map((signal) => (
              <li
                key={signal.family}
                className="flex items-baseline justify-between gap-4 text-sm"
              >
                <span className="text-slate-800">
                  {signal.family}
                  <span className="text-slate-500"> — {signal.descriptor}</span>
                </span>
                <span
                  className={`shrink-0 tabular-nums ${
                    signal.direction === 'machine' ? 'text-rose-700' : 'text-teal-700'
                  }`}
                  title="Log-odds contributed to the document score"
                >
                  {formatSigned(signal.contribution)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-4 rounded-md bg-slate-50 p-3 text-xs leading-relaxed text-slate-600">
        This is a probabilistic measurement of writing patterns, not proof of
        authorship. It was trained on student argumentative essays and is
        weakest on short text, on unfamiliar topics, and on writing that has
        been deliberately edited to evade detection. Treat a high score as a
        prompt to look more closely, never as a verdict.
      </p>
    </section>
  );
}

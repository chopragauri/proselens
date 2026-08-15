import type { SentenceResult } from '../types/analysis';
import { BAND_LABELS } from '../types/analysis';
import { BAND_BADGE, BAND_MARKER, formatMeasurement, formatSigned } from '../utils/risk';

interface Props {
  sentence: SentenceResult | null;
}

/**
 * Shows why a sentence scored as it did.
 *
 * Every number here is a measurement or a term of the model's own arithmetic:
 * `contribution` is the log-odds that signal family added, and `z_score` is how
 * far the measured value sits from the mean of training human sentences.
 * Nothing on this panel is generated prose.
 */
export function EvidencePanel({ sentence }: Props) {
  if (!sentence) {
    return (
      <section className="rounded-lg border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-600">
        Select a sentence in the essay to see its evidence.
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Sentence analysis</h2>
        <span
          className={`shrink-0 rounded px-2 py-0.5 text-xs font-medium ${BAND_BADGE[sentence.risk_band]}`}
        >
          {BAND_MARKER[sentence.risk_band]} {BAND_LABELS[sentence.risk_band]}
        </span>
      </div>

      <blockquote className="mt-3 border-l-2 border-slate-300 pl-3 font-serif text-sm leading-relaxed text-slate-800">
        {sentence.text}
      </blockquote>

      {!sentence.is_scorable ? (
        <p className="mt-4 rounded-md bg-slate-50 p-3 text-sm text-slate-700">
          This sentence has {sentence.token_count} content{' '}
          {sentence.token_count === 1 ? 'word' : 'words'}, too few for the
          measurements to be stable. Type–token ratio is near 1.0 for almost any
          short sentence and variance is undefined, so no score is given rather
          than a misleading one.
        </p>
      ) : (
        <>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">Risk</span>
            <span className="text-2xl font-semibold tabular-nums text-slate-900">
              {sentence.risk_score}
              <span className="text-sm font-normal text-slate-500"> / 100</span>
            </span>
          </div>

          <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Signals
          </h3>
          <table className="mt-2 w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500">
                <th className="pb-1 font-medium">Signal</th>
                <th className="pb-1 text-right font-medium">Measured</th>
                <th className="pb-1 text-right font-medium" title="Standard deviations from the human training average">
                  vs human
                </th>
                <th className="pb-1 text-right font-medium" title="Log-odds contributed to this sentence's score">
                  Effect
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sentence.signals.map((signal) => (
                <tr key={signal.family}>
                  <td className="py-1.5 pr-2 align-top text-slate-800">
                    {signal.family}
                    <span className="block text-xs text-slate-500">
                      {signal.descriptor}
                    </span>
                  </td>
                  <td className="py-1.5 text-right align-top tabular-nums text-slate-700">
                    {formatMeasurement(signal.measured_value)}
                  </td>
                  <td className="py-1.5 text-right align-top tabular-nums text-slate-700">
                    {signal.z_score === null ? '—' : formatSigned(signal.z_score, 1)}
                  </td>
                  <td
                    className={`py-1.5 text-right align-top tabular-nums ${
                      signal.direction === 'machine' ? 'text-rose-700' : 'text-teal-700'
                    }`}
                  >
                    {formatSigned(signal.contribution)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {sentence.explanation && (
            <>
              <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Explanation
              </h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-700">
                {sentence.explanation}
              </p>
            </>
          )}

          <p className="mt-3 text-xs text-slate-500">
            Effect is the log-odds each signal family contributed. Positive
            values push toward machine, negative toward human; together they sum
            to the sentence&rsquo;s score.
          </p>
        </>
      )}
    </section>
  );
}

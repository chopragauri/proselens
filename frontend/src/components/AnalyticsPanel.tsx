import type { DocumentSummary } from '../types/analysis';
import { formatMeasurement } from '../utils/risk';

interface Props {
  summary: DocumentSummary;
}

/**
 * Document-level counts and measurements.
 *
 * A stacked bar is used for the risk distribution because proportion is the
 * question being asked; the raw counts sit beside it so nothing depends on
 * reading a bar accurately. The remaining figures are shown as plain numbers —
 * a chart of four unrelated scalars would decorate rather than explain.
 */
export function AnalyticsPanel({ summary }: Props) {
  const scored =
    summary.high_risk_sentences +
    summary.medium_risk_sentences +
    summary.low_risk_sentences;

  const segments = [
    { label: 'Low', count: summary.low_risk_sentences, class: 'bg-teal-600' },
    { label: 'Medium', count: summary.medium_risk_sentences, class: 'bg-amber-600' },
    { label: 'High', count: summary.high_risk_sentences, class: 'bg-rose-700' },
  ];

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">Document analytics</h2>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Statistic label="Sentences" value={summary.sentence_count.toString()} />
        <Statistic label="Paragraphs" value={summary.paragraph_count.toString()} />
        <Statistic label="Words" value={summary.word_count.toLocaleString()} />
        <Statistic
          label="Scored"
          value={`${summary.scorable_sentence_count} of ${summary.sentence_count}`}
        />
      </dl>

      {scored > 0 && (
        <div className="mt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Sentence risk distribution
          </h3>
          <div
            className="mt-2 flex h-3 overflow-hidden rounded-full bg-slate-100"
            role="img"
            aria-label={`${summary.low_risk_sentences} low risk, ${summary.medium_risk_sentences} medium risk, ${summary.high_risk_sentences} high risk sentences`}
          >
            {segments.map((segment) =>
              segment.count > 0 ? (
                <div
                  key={segment.label}
                  className={segment.class}
                  style={{ width: `${(segment.count / scored) * 100}%` }}
                />
              ) : null,
            )}
          </div>
          <ul className="mt-2 flex flex-wrap gap-4 text-xs text-slate-600">
            {segments.map((segment) => (
              <li key={segment.label} className="flex items-center gap-1.5">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${segment.class}`}
                  aria-hidden="true"
                />
                {segment.label}: <span className="tabular-nums">{segment.count}</span>
              </li>
            ))}
            {summary.unscored_sentences > 0 && (
              <li className="text-slate-500">
                Not scored:{' '}
                <span className="tabular-nums">{summary.unscored_sentences}</span>
              </li>
            )}
          </ul>
        </div>
      )}

      <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-slate-100 pt-4 sm:grid-cols-4">
        <Statistic
          label="Mean sentence risk"
          value={
            summary.mean_sentence_risk === null
              ? '—'
              : `${Math.round(summary.mean_sentence_risk * 100)} / 100`
          }
        />
        <Statistic
          label="Lexical diversity"
          hint="Document MTLD. Higher means more varied vocabulary."
          value={formatMeasurement(summary.lexical_diversity)}
        />
        <Statistic
          label="Length variation"
          hint="Coefficient of variation of sentence lengths. Lower is more even."
          value={formatMeasurement(summary.sentence_length_variation)}
        />
        <Statistic
          label="Predictability"
          hint="Mean cross-entropy under the human reference model. Lower is more predictable."
          value={formatMeasurement(summary.mean_predictability)}
        />
      </dl>
    </section>
  );
}

function Statistic({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500" title={hint}>
        {label}
      </dt>
      <dd className="mt-0.5 text-lg font-semibold tabular-nums text-slate-900">
        {value}
      </dd>
    </div>
  );
}

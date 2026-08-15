import type { AnalyzeResponse, SentenceResult } from '../types/analysis';
import { BAND_LABELS } from '../types/analysis';
import { BAND_CLASS, BAND_MARKER } from '../utils/risk';

interface Props {
  result: AnalyzeResponse;
  analyzedText: string;
  selectedIndex: number | null;
  onSelect: (sentence: SentenceResult) => void;
}

/**
 * Renders the essay with per-sentence highlighting.
 *
 * Sentences are positioned by the character offsets the backend returns, and
 * the gaps between them (whitespace, paragraph breaks) are emitted verbatim
 * from the analyzed text. That is why preprocessing was made length-preserving:
 * the offsets index into exactly the string shown here, so highlights cannot
 * drift out of alignment with the words they describe.
 */
export function HighlightedEssay({
  result,
  analyzedText,
  selectedIndex,
  onSelect,
}: Props) {
  const nodes: JSX.Element[] = [];
  let cursor = 0;

  result.sentences.forEach((sentence) => {
    if (sentence.start > cursor) {
      const gap = analyzedText.slice(cursor, sentence.start);
      nodes.push(
        <span key={`gap-${sentence.index}`} className="whitespace-pre-wrap">
          {gap}
        </span>,
      );
    }

    const isSelected = selectedIndex === sentence.index;
    const label = sentence.is_scorable
      ? `${BAND_LABELS[sentence.risk_band]} risk, ${sentence.risk_score} out of 100`
      : 'Too short to score';

    nodes.push(
      <button
        key={`sentence-${sentence.index}`}
        type="button"
        onClick={() => onSelect(sentence)}
        aria-label={`Sentence ${sentence.index + 1}. ${label}. Select for evidence.`}
        aria-pressed={isSelected}
        title={label}
        className={`sentence-button text-left ${BAND_CLASS[sentence.risk_band]} ${
          isSelected ? 'sentence-selected' : ''
        }`}
      >
        {sentence.text}
        <sup
          aria-hidden="true"
          className="ml-0.5 select-none text-[0.6rem] text-slate-500"
        >
          {BAND_MARKER[sentence.risk_band]}
        </sup>
      </button>,
    );

    cursor = sentence.end;
  });

  if (cursor < analyzedText.length) {
    nodes.push(
      <span key="gap-final" className="whitespace-pre-wrap">
        {analyzedText.slice(cursor)}
      </span>,
    );
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Annotated essay</h2>
        <Legend />
      </div>

      <p className="mt-1 text-sm text-slate-600">
        Select any sentence to see the measurements behind its score.
      </p>

      <div className="mt-4 font-serif text-[15px] leading-[1.9] text-slate-900">
        {nodes}
      </div>
    </section>
  );
}

function Legend() {
  const entries = [
    { band: 'low' as const, text: 'Low' },
    { band: 'medium' as const, text: 'Medium' },
    { band: 'high' as const, text: 'High' },
    { band: 'not_scored' as const, text: 'Not scored' },
  ];

  return (
    <ul className="flex flex-wrap gap-3 text-xs text-slate-600">
      {entries.map((entry) => (
        <li key={entry.band} className="flex items-center gap-1.5">
          <span className={`px-1 ${BAND_CLASS[entry.band]}`} aria-hidden="true">
            {BAND_MARKER[entry.band]}
          </span>
          {entry.text}
        </li>
      ))}
    </ul>
  );
}

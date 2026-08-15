import { useCallback, useState } from 'react';

import { AnalyticsPanel } from './components/AnalyticsPanel';
import { EssayInput } from './components/EssayInput';
import { EvidencePanel } from './components/EvidencePanel';
import { HighlightedEssay } from './components/HighlightedEssay';
import { OverallAssessment } from './components/OverallAssessment';
import { analyzeFile, analyzeText } from './services/api';
import type { AnalyzeResponse, SentenceResult } from './types/analysis';

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  // The text the current result describes. Kept separately from the editor's
  // contents so that editing after analysis cannot silently re-align
  // highlights against text the backend never saw.
  const [analyzedText, setAnalyzedText] = useState('');
  const [selected, setSelected] = useState<SentenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const run = useCallback(
    async (task: () => Promise<AnalyzeResponse>, sourceText: string | null) => {
      setIsAnalyzing(true);
      setError(null);
      setSelected(null);
      try {
        const response = await task();
        setResult(response);
        // For uploads the analyzed text is not what is in the editor, so it is
        // reconstructed from the sentence offsets the backend returned.
        setAnalyzedText(sourceText ?? reconstruct(response));
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Analysis failed.');
        setResult(null);
      } finally {
        setIsAnalyzing(false);
      }
    },
    [],
  );

  const handleAnalyze = () => run(() => analyzeText(text), text);

  const handleUpload = (file: File) =>
    run(async () => {
      const response = await analyzeFile(file);
      // Show the uploaded essay in the editor so the two views agree.
      setText(reconstruct(response));
      return response;
    }, null);

  const handleClear = () => {
    setText('');
    setResult(null);
    setAnalyzedText('');
    setSelected(null);
    setError(null);
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-6 py-6">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
            ProseLens
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Explainable analysis of writing patterns associated with
            machine-generated text.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-5 px-6 py-6">
        <EssayInput
          text={text}
          onTextChange={setText}
          onAnalyze={handleAnalyze}
          onUpload={handleUpload}
          onClear={handleClear}
          isAnalyzing={isAnalyzing}
        />

        {error && (
          <div
            role="alert"
            className="rounded-md border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900"
          >
            {error}
          </div>
        )}

        {result && (
          <>
            <OverallAssessment result={result} />
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
              <HighlightedEssay
                result={result}
                analyzedText={analyzedText}
                selectedIndex={selected?.index ?? null}
                onSelect={setSelected}
              />
              <div className="space-y-5">
                <EvidencePanel sentence={selected} />
              </div>
            </div>
            <AnalyticsPanel summary={result.summary} />
          </>
        )}
      </main>

      <footer className="mx-auto max-w-6xl px-6 pb-10 text-xs text-slate-500">
        ProseLens measures statistical properties of writing. It cannot prove who
        wrote a text, and it should never be the sole basis for an accusation.
      </footer>
    </div>
  );
}

/** Rebuild the analyzed text from sentence spans, for the upload path. */
function reconstruct(response: AnalyzeResponse): string {
  if (response.sentences.length === 0) return '';
  const end = response.sentences[response.sentences.length - 1].end;
  const buffer = new Array<string>(end).fill(' ');
  response.sentences.forEach((sentence) => {
    for (let index = 0; index < sentence.text.length; index += 1) {
      buffer[sentence.start + index] = sentence.text[index];
    }
  });
  return buffer.join('');
}

import { useRef } from 'react';

interface Props {
  text: string;
  onTextChange: (text: string) => void;
  onAnalyze: () => void;
  onUpload: (file: File) => void;
  onClear: () => void;
  isAnalyzing: boolean;
}

export function EssayInput({
  text,
  onTextChange,
  onAnalyze,
  onUpload,
  onClear,
  isAnalyzing,
}: Props) {
  const fileInput = useRef<HTMLInputElement>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const canAnalyze = wordCount > 0 && !isAnalyzing;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <label htmlFor="essay" className="block text-sm font-medium text-slate-700">
        Essay
      </label>
      <textarea
        id="essay"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder="Paste an admissions essay here, or upload a .txt file."
        rows={14}
        className="mt-2 w-full resize-y rounded-md border border-slate-300 p-3 font-serif text-[15px] leading-relaxed text-slate-900 placeholder:text-slate-400 focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-300"
      />

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-slate-600" aria-live="polite">
          {wordCount.toLocaleString()} words · {text.length.toLocaleString()} characters
        </p>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onAnalyze}
            disabled={!canAnalyze}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {isAnalyzing ? 'Analyzing…' : 'Analyze essay'}
          </button>

          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            disabled={isAnalyzing}
            className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            Upload .txt
          </button>

          <button
            type="button"
            onClick={onClear}
            disabled={isAnalyzing || !text}
            className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            Clear
          </button>
        </div>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept=".txt,text/plain"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onUpload(file);
          // Reset so selecting the same file twice still fires a change event.
          event.target.value = '';
        }}
      />
    </section>
  );
}

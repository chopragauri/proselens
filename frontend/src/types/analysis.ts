/**
 * Types mirroring backend/app/schemas/analysis.py.
 *
 * The backend returns `risk_score` on a 0-100 scale rather than a probability.
 * That naming is deliberate and is carried through the UI unchanged: measured
 * calibration is good at the extremes but thin in the middle, so the interface
 * must not imply a precision the evaluation does not support.
 */

export type RiskBand = 'low' | 'medium' | 'high' | 'not_scored';

export type Assessment =
  | 'likely_human'
  | 'mixed_signals'
  | 'likely_machine'
  | 'insufficient_text';

export interface SignalEvidence {
  family: string;
  direction: 'machine' | 'human';
  contribution: number;
  descriptor: string;
  measured_feature: string;
  measured_value: number | null;
  z_score: number | null;
}

export interface SentenceResult {
  index: number;
  text: string;
  start: number;
  end: number;
  paragraph_index: number;
  token_count: number;
  is_scorable: boolean;
  risk_score: number | null;
  risk_band: RiskBand;
  signals: SignalEvidence[];
  explanation: string | null;
}

export interface PassageResult {
  start_sentence_index: number;
  end_sentence_index: number;
  start: number;
  end: number;
  risk_score: number;
  risk_band: RiskBand;
}

export interface DocumentSummary {
  sentence_count: number;
  scorable_sentence_count: number;
  paragraph_count: number;
  word_count: number;
  character_count: number;
  high_risk_sentences: number;
  medium_risk_sentences: number;
  low_risk_sentences: number;
  unscored_sentences: number;
  mean_sentence_risk: number | null;
  lexical_diversity: number | null;
  sentence_length_variation: number | null;
  mean_predictability: number | null;
}

export interface AnalyzeResponse {
  risk_score: number;
  confidence: number;
  confidence_band: RiskBand;
  assessment: Assessment;
  summary_text: string;
  signals: SignalEvidence[];
  sentences: SentenceResult[];
  passages: PassageResult[];
  summary: DocumentSummary;
  model_version: string;
}

export const ASSESSMENT_LABELS: Record<Assessment, string> = {
  likely_human: 'Consistent with human writing',
  mixed_signals: 'Mixed signals',
  likely_machine: 'Machine-like patterns',
  insufficient_text: 'Not enough text to assess',
};

export const BAND_LABELS: Record<RiskBand, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  not_scored: 'Not scored',
};

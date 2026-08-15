import type { RiskBand } from '../types/analysis';

/**
 * Presentation helpers for risk bands.
 *
 * Every band carries three independent cues — colour, underline style, and a
 * text label — so none of them is load-bearing on its own.
 */

export const BAND_CLASS: Record<RiskBand, string> = {
  low: 'risk-low',
  medium: 'risk-medium',
  high: 'risk-high',
  not_scored: 'risk-not_scored',
};

export const BAND_TEXT: Record<RiskBand, string> = {
  low: 'text-teal-800',
  medium: 'text-amber-800',
  high: 'text-rose-800',
  not_scored: 'text-slate-500',
};

export const BAND_BADGE: Record<RiskBand, string> = {
  low: 'bg-teal-100 text-teal-900 ring-1 ring-teal-700/30',
  medium: 'bg-amber-100 text-amber-900 ring-1 ring-amber-700/30',
  high: 'bg-rose-100 text-rose-900 ring-1 ring-rose-700/30',
  not_scored: 'bg-slate-100 text-slate-700 ring-1 ring-slate-400/40',
};

/** Symbolic marker, so band survives greyscale printing and screen readers. */
export const BAND_MARKER: Record<RiskBand, string> = {
  low: '○',
  medium: '◐',
  high: '●',
  not_scored: '–',
};

export function formatSigned(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

export function formatMeasurement(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '—';
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

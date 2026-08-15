/**
 * Thin API client.
 *
 * Errors are surfaced with the backend's own `detail` message where one
 * exists, so a user who uploads a 2 MB file is told that, rather than being
 * shown a generic failure.
 */

import type { AnalyzeResponse } from '../types/analysis';

const API_BASE = '/api';

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') return body.detail;
    // FastAPI validation errors arrive as an array of issue objects.
    if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      return String(body.detail[0].msg);
    }
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return `Request failed (${response.status} ${response.statusText}).`;
}

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function analyzeFile(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append('file', file);
  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

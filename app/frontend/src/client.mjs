export function scoreText(score) {
  return typeof score === 'number' && Number.isFinite(score) && score >= 0 && score <= 1 ? `${(score * 100).toFixed(1)}%` : '—';
}

async function result(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `API error ${response.status}`);
  return payload;
}

export async function analyzeFile(base, file, explain = true, fetcher = fetch) {
  return result(await fetcher(`${base.replace(/\/$/, '')}/analyze?explain=${explain}`, {
    method: 'POST', headers: {'Content-Type': file.type || 'application/octet-stream'}, body: file,
  }));
}

export async function analyzeURL(base, url, explain = true, fetcher = fetch) {
  return result(await fetcher(`${base.replace(/\/$/, '')}/analyze`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({url, explain}),
  }));
}

export async function history(base, fetcher = fetch) {
  return result(await fetcher(`${base.replace(/\/$/, '')}/history`));
}

import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import {analyzeFile, analyzeURL, history, scoreText} from './client.mjs';
import './style.css';

function App() {
  const [file, setFile] = useState(null);
  const [url, setURL] = useState('');
  const [explain, setExplain] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const base = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
  async function submit(event) {
    event.preventDefault(); setError('');
    if (Boolean(file) === Boolean(url.trim())) {setError('Choose one image upload or one URL.'); return;}
    setBusy(true);
    try {
      const value = file ? await analyzeFile(base, file, explain) : await analyzeURL(base, url.trim(), explain);
      setResult(value); setLogs(await history(base));
    } catch (failure) {setError(failure.message);} finally {setBusy(false);}
  }
  return <main>
    <header><small>GRADUATION EXPERIMENT</small><h1>Explainable facial detection</h1><p>Still-image model scores, with face-level quality checks and attribution.</p></header>
    <form onSubmit={submit}>
      <label>JPEG or PNG<input type="file" accept="image/jpeg,image/png" onChange={e => setFile(e.target.files?.[0] || null)} /></label>
      <label>Or public HTTPS URL<input type="url" value={url} onChange={e => setURL(e.target.value)} placeholder="https://…" /></label>
      <label className="inline"><input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} />Generate Grad-CAM</label>
      <button disabled={busy}>{busy ? 'Analyzing…' : 'Analyze faces'}</button>
    </form>
    {error && <p role="alert" className="warning">{error}</p>}
    {result && <section>
      <h2>Results</h2><p>{result.run_id} · Calibration {result.calibration_applied ? 'applied' : 'unmeasured'} · {result.elapsed_seconds.toFixed(2)}s</p>
      {result.training_pilot && <p className="warning">Diagnostic pilot weights; not a validated detector.</p>}
      <div className="preview"><img alt="Analyzed image" src={result.image.preview_data_url} />
        <svg viewBox={`0 0 ${result.image.width} ${result.image.height}`} aria-label="Face bounding boxes">
          {result.faces.map(face => <rect key={face.face_index} x={face.box[0]} y={face.box[1]} width={face.box[2]-face.box[0]} height={face.box[3]-face.box[1]} fill="none" stroke={face.label === 'FAKE' ? '#b93434' : '#15775d'} strokeWidth="3" />)}
        </svg>
      </div>
      {!result.faces.length && <p>No faces detected; no authenticity verdict produced.</p>}
      {result.faces.map(face => <article key={face.face_index}>
        <h3>Face {face.face_index + 1}: {face.label || 'Low quality'}</h3>
        {face.status === 'accepted' ? <><p>FAKE model score: {scoreText(face.fake_score)} · REAL score: {scoreText(face.real_score)}</p>
          {face.uncertain && <p className="warning">Close to the decision threshold.</p>}
          {face.overlay_data_url && <img alt={`Attribution for face ${face.face_index+1}`} src={face.overlay_data_url} />}</> : <p>Face did not meet the evaluated quality/size gate.</p>}
      </article>)}
      <p className="note">{result.explanation_note}</p>
    </section>}
    <section><h2>Recent analyses</h2><button type="button" onClick={async () => {try {setLogs(await history(base));} catch(e) {setError(e.message);}}}>Refresh history</button>
      <ul>{logs.map(log => <li key={log.analysis_id}>#{log.analysis_id} · {log.run_id} · {log.face_count} faces · {log.status}</li>)}</ul>
    </section>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);

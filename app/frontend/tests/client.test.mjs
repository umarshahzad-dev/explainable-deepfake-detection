import test from 'node:test';
import assert from 'node:assert/strict';

async function client() {
  try { return await import('../src/client.mjs'); }
  catch (error) { if (error.code === 'ERR_MODULE_NOT_FOUND') assert.fail('frontend client not implemented'); throw error; }
}

test('uploads use raw image bodies, not unsupported multipart', async () => {
  const api = await client();
  const file = new Blob(['generated fixture'], {type: 'image/png'});
  const result = await api.analyzeFile('http://local.invalid', file, false, async (url, options) => {
    const valid = url === 'http://local.invalid/analyze?explain=false' && options.body === file && options.headers['Content-Type'] === 'image/png';
    return {ok: valid, status: valid ? 200 : 422, json: async () => valid ? {faces: [], status: 'no_faces'} : {detail: 'wrong transport'}};
  });
  assert.equal(result.status, 'no_faces');
});

test('URL requests preserve boolean explanation choice and show API errors', async () => {
  const api = await client();
  await assert.rejects(api.analyzeURL('http://local.invalid', 'https://example.org/image.png', true, async (url, options) => {
    const payload = JSON.parse(options.body);
    return {ok: false, status: 422, json: async () => ({detail: payload.explain === true ? 'fixture rejection' : 'wrong flag'})};
  }), /fixture rejection/);
});

test('missing and invalid scores are not displayed as confident predictions', async () => {
  const api = await client();
  assert.equal(api.scoreText(null), '—');
  assert.equal(api.scoreText(NaN), '—');
  assert.equal(api.scoreText(0.125), '12.5%');
});

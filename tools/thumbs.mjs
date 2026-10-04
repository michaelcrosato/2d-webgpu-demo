#!/usr/bin/env node
// Generate gallery thumbnails (assets/thumbs/<id>.webp) for the home page.
//   node tools/thumbs.mjs                # all scenes
//   node tools/thumbs.mjs noise bloom     # some scenes
//   options: --jobs=3 --frames=14 --api=webgpu
// Uses the same headless software-GPU setup as tools/check.mjs, in test mode.

import { createServer } from './serve.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, '..', 'assets', 'thumbs');
let chromium;
for (const p of ['playwright', '/opt/node22/lib/node_modules/playwright/index.mjs']) {
  try {
    ({ chromium } = await import(p));
    break;
  } catch {
    /* next */
  }
}
const args = process.argv.slice(2);
const opt = (n, d) => args.find((a) => a.startsWith(`--${n}=`))?.split('=')[1] ?? d;
const ids = args.filter((a) => !a.startsWith('--'));
const jobs = +opt('jobs', 3);
const frames = +opt('frames', 14);
const api = opt('api', 'webgpu');
fs.mkdirSync(outDir, { recursive: true });

const server = createServer();
await new Promise((r) => server.listen(0, r));
const base = `http://localhost:${server.address().port}/`;
const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--disable-vulkan-surface', '--no-sandbox'],
});

const list = await (async () => {
  const p = await browser.newPage();
  await p.goto(base + '?r=0#/');
  const l = await p.evaluate(async () => {
    const { SCENES } = await import('/src/scenes/registry.js');
    const out = [];
    for (const s of SCENES) {
      const m = await s.load().catch(() => null);
      const sc = m?.default;
      out.push({ id: s.id, ex: sc?.examples?.[0]?.id || null, thumbEx: sc?.thumbnail?.example || null, backends: sc?.backends || (sc?.initGL ? ['webgpu', 'webgl2'] : ['webgpu']) });
    }
    return out;
  });
  await p.close();
  return l.filter((s) => !ids.length || ids.includes(s.id));
})();

// Which example tab makes the most representative thumbnail (default: the first tab).
const PICK = {
  transitions: 'pixel',
  'jump-flood': 'outline',
  'trails-ribbons': 'slash',
  nbody: 'solar',
  'sdf-text': 'title',
  'fog-of-war': 'soft',
  distortion: 'blackhole',
  'color-grading': 'moods',
  'edge-detection': 'vision',
  blur: 'tiltshift',
  'chromatic-glitch': 'glitch',
  'feedback-trails': 'dream',
  'gpu-particles': 'fireworks',
  'compute-power': 'vortex',
  'god-rays': 'window',
  'masking-stencil': 'xray',
};

let i = 0;
const results = [];
async function worker() {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
  for (;;) {
    const s = list[i++];
    if (!s) break;
    const ex = PICK[s.id] || s.thumbEx || s.ex;
    const useApi = s.backends.includes(api) ? api : s.backends[0];
    const url = `${base}?r=${i}#/s/${s.id}?api=${useApi}${ex ? `&ex=${ex}` : ''}&test=1`;
    try {
      await page.goto(url, { timeout: 120000 });
      await page.waitForFunction(() => window.__showcase?.ready, null, { timeout: 120000 });
      await page.waitForFunction((f) => (window.__showcase?.frames || 0) >= f, frames, { timeout: 150000 }).catch(() => {});
      // hide UI overlays so the thumbnail is just the image
      await page.addStyleTag({ content: '.hud,.hint,.stage-tools,.scene-overlay{display:none!important}' });
      await page.waitForTimeout(300);
      const png = await page.locator('.stage-wrap').screenshot({ timeout: 120000 });
      const webp = await page.evaluate(async (b64) => {
        const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
        const bmp = await createImageBitmap(blob);
        const c = document.createElement('canvas');
        c.width = 400;
        c.height = 225;
        const g = c.getContext('2d');
        g.imageSmoothingQuality = 'high';
        // cover-fit into 16:9
        const s = Math.max(c.width / bmp.width, c.height / bmp.height);
        const w = bmp.width * s;
        const h = bmp.height * s;
        g.drawImage(bmp, (c.width - w) / 2, (c.height - h) / 2, w, h);
        return c.toDataURL('image/webp', 0.8).split(',')[1];
      }, png.toString('base64'));
      fs.writeFileSync(path.join(outDir, `${s.id}.webp`), Buffer.from(webp, 'base64'));
      results.push(`ok   ${s.id}`);
      console.log(`ok   ${s.id} (${ex || '-'}, ${useApi})`);
    } catch (e) {
      results.push(`FAIL ${s.id}`);
      console.log(`FAIL ${s.id}: ${e.message.split('\n')[0]}`);
    }
  }
  await page.close();
}
await Promise.all(Array.from({ length: jobs }, worker));
await browser.close();
server.close();
console.log(`${results.filter((r) => r.startsWith('ok')).length}/${results.length} thumbnails written to assets/thumbs/`);

#!/usr/bin/env node
// Headless smoke test for scenes: loads each scene × example × API in Chromium (SwiftShader
// software GPU), drags the mouse across the canvas, waits for frames, and reports
//   - WGSL / GLSL compile errors, WebGPU validation errors, JS exceptions
//   - blank output (a canvas that is a single flat color)
// Usage:
//   node tools/check.mjs                       # every scene, every example, both APIs
//   node tools/check.mjs noise voronoi         # specific scenes
//   node tools/check.mjs --category=post       # one category
//   options: --api=webgpu|webgl2  --first (first example only)  --shots (save PNGs to tools/out/)
//            --frames=6  --timeout=40000  --quiet
//
// Requires Playwright (globally installed here; `npm i -D playwright` elsewhere).

import { createServer } from './serve.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
let chromium;
for (const p of ['playwright', '/opt/node22/lib/node_modules/playwright/index.mjs']) {
  try {
    ({ chromium } = await import(p));
    break;
  } catch {
    /* try next */
  }
}
if (!chromium) {
  console.error('Playwright not found. Install it with: npm i -D playwright');
  process.exit(2);
}

const args = process.argv.slice(2);
const opt = (name, def) => {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.split('=')[1] : args.includes(`--${name}`) ? true : def;
};
const ids = args.filter((a) => !a.startsWith('--'));
const apiFilter = opt('api', null);
const firstOnly = !!opt('first', false);
const shots = !!opt('shots', false);
const minFrames = +opt('frames', 6);
const timeout = +opt('timeout', 45000);
const category = opt('category', null);
const quiet = !!opt('quiet', false);

const server = createServer();
await new Promise((r) => server.listen(0, r));
const port = server.address().port;
const base = `http://localhost:${port}/`;

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--disable-vulkan-surface', '--no-sandbox', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
let logs = [];
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

// discover scenes
await page.goto(base + '?r=0#/');
const all = await page.evaluate(async () => {
  const { SCENES } = await import('/src/scenes/registry.js');
  const out = [];
  for (const s of SCENES) {
    let info = { id: s.id, category: s.category, missing: false, examples: [], backends: [] };
    try {
      const m = await s.load();
      const sc = m.default;
      info.examples = (sc.examples || []).map((e) => e.id);
      info.backends = sc.backends || (sc.initGL ? ['webgpu', 'webgl2'] : ['webgpu']);
      info.badges = s.badges;
    } catch (e) {
      info.missing = true;
      info.error = String(e.message || e);
    }
    out.push(info);
  }
  return out;
});

let targets = all.filter((s) => (!ids.length || ids.includes(s.id)) && (!category || s.category === category));
if (ids.length) {
  for (const id of ids) if (!all.find((s) => s.id === id)) console.log(`?? unknown scene id "${id}"`);
}
if (shots) fs.mkdirSync(path.join(here, 'out'), { recursive: true });

const results = [];
let n = 0;
for (const s of targets) {
  if (s.missing) {
    results.push({ id: s.id, status: 'MISSING', detail: s.error });
    console.log(`MISSING ${s.id}: ${s.error}`);
    continue;
  }
  const exs = s.examples.length ? (firstOnly ? [s.examples[0]] : s.examples) : [null];
  const apis = s.backends.filter((b) => !apiFilter || b === apiFilter);
  for (const api of apis) {
    for (const ex of exs) {
      logs = [];
      const url = `${base}?r=${++n}#/s/${s.id}?api=${api}${ex ? `&ex=${ex}` : ''}&test=1`;
      const label = `${s.id}${ex ? '/' + ex : ''} [${api}]`;
      let status = 'PASS';
      let detail = '';
      let stats = null;
      const t0 = Date.now();
      try {
        await page.goto(url, { timeout });
        await page.waitForFunction(() => window.__showcase?.ready && window.__showcase.backend, null, { timeout });
        const box = await page.locator('.stage-canvas').boundingBox();
        if (box) {
          // drag across the canvas to exercise pointer code paths
          await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.4);
          await page.mouse.down();
          for (let i = 1; i <= 6; i++) await page.mouse.move(box.x + box.width * (0.3 + i * 0.06), box.y + box.height * (0.4 + i * 0.03));
          await page.mouse.up();
          await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5);
        }
        await page.waitForFunction((f) => (window.__showcase?.frames || 0) >= f || (window.__showcase?.errors || []).length > 0, minFrames, { timeout });
        const errors = await page.evaluate(() => window.__showcase.errors || []);
        const used = await page.evaluate(() => window.__showcase.backend);
        if (used !== api) {
          status = 'FAIL';
          detail += `ran on ${used} instead of ${api}\n`;
        }
        if (errors.length) {
          status = 'FAIL';
          detail += errors.join('\n') + '\n';
        }
        const el = page.locator('.stage-canvas');
        const png = await el.screenshot({ timeout });
        if (shots) fs.writeFileSync(path.join(here, 'out', `${s.id}__${ex || 'main'}__${api}.png`), png);
        stats = await page.evaluate(async (b64) => {
          const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
          const bmp = await createImageBitmap(blob);
          const c = document.createElement('canvas');
          c.width = 96;
          c.height = 54;
          const g = c.getContext('2d');
          g.drawImage(bmp, 0, 0, 96, 54);
          const d = g.getImageData(0, 0, 96, 54).data;
          let sum = 0;
          let sum2 = 0;
          const seen = new Set();
          for (let i = 0; i < d.length; i += 4) {
            const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
            sum += l;
            sum2 += l * l;
            seen.add(((d[i] >> 4) << 8) | ((d[i + 1] >> 4) << 4) | (d[i + 2] >> 4));
          }
          const N = d.length / 4;
          const mean = sum / N;
          return { mean: +mean.toFixed(1), std: +Math.sqrt(Math.max(0, sum2 / N - mean * mean)).toFixed(1), colors: seen.size };
        }, png.toString('base64'));
        if (stats.std < 1.0 && stats.colors <= 2) {
          status = status === 'PASS' ? 'BLANK' : status;
          detail += `canvas looks blank (mean ${stats.mean}, std ${stats.std})\n`;
        }
      } catch (e) {
        status = 'FAIL';
        detail += `${e.message.split('\n')[0]}\n`;
        const errors = await page.evaluate(() => window.__showcase?.errors || []).catch(() => []);
        if (errors.length) detail += errors.join('\n') + '\n';
      }
      const jsErrors = logs.filter((l) => l.startsWith('[pageerror]') || /Uncaught|TypeError|ReferenceError|SyntaxError/.test(l));
      if (jsErrors.length && status === 'PASS') {
        status = 'FAIL';
        detail += jsErrors.join('\n') + '\n';
      }
      const ms = Date.now() - t0;
      results.push({ id: s.id, ex, api, status, detail: detail.trim(), stats, ms });
      const st = stats ? ` mean=${stats.mean} std=${stats.std} colors=${stats.colors}` : '';
      if (!quiet || status !== 'PASS') console.log(`${status.padEnd(5)} ${label} (${ms} ms)${st}${detail ? '\n    ' + detail.trim().replace(/\n/g, '\n    ') : ''}`);
    }
  }
}

await browser.close();
server.close();
const bad = results.filter((r) => r.status !== 'PASS');
console.log(`\n${results.length - bad.length}/${results.length} passed${bad.length ? ` — ${bad.length} problem(s): ${[...new Set(bad.map((b) => b.id))].join(', ')}` : ''}`);
fs.writeFileSync(path.join(here, 'last-check.json'), JSON.stringify(results, null, 2));
process.exit(bad.length ? 1 : 0);

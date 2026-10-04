// Shared helpers for the "Art Styles" scenes (pixel-art, dithering, crt-retro, neon-synthwave,
// toon-comic, painterly): palettes as uniform arrays, overlay labels, a draggable split divider,
// a void-and-cluster blue-noise texture and a CPU Floyd–Steinberg ditherer (for comparison).

import { PALETTES, makeCanvas, rng } from '../../core/assets.js';
import { GAME_SCENE_CODE, GAME_SCENE_INCLUDES } from '../../core/gamescene.js';

// ------------------------------------------------------------------------- game scene input
// WORKAROUND: shaderScene's `input: 'game'` currently binds the `game` texture to the very pass
// that renders INTO it, which WebGPU rejects (TextureBinding + RenderAttachment in one pass).
// Rendering the same procedural platformer as our own first pass (double-buffered, so it never
// reads what it writes) avoids that and behaves the same. Read it as texture `scene`.
export const GAME_INCLUDES = GAME_SCENE_INCLUDES;
/**
 * The game scene as a pass. It reads its own render size (TEXSIZE of its double buffer) instead of
 * u.resolution, so it can be rendered smaller than the canvas: `scale` = number or (params, ctx) => number.
 * Under the headless test harness (software GPU) it renders at half size to keep frames fast.
 */
export function gamePass(name = 'scene', { scale = 1, ...extra } = {}) {
  const code = GAME_SCENE_CODE.replace(/u\.resolution/g, `TEXSIZE(${name})`);
  return {
    name,
    format: 'rgba8unorm',
    code,
    size: (params, ctx) => {
      let s = typeof scale === 'function' ? scale(params, ctx) : scale;
      if (ctx.testMode) s *= 0.5;
      return [Math.max(1, Math.round(ctx.width * s)), Math.max(1, Math.round(ctx.height * s))];
    },
    ...extra,
  };
}

// ------------------------------------------------------------------------------------ palettes

/** Extra palettes on top of assets.js PALETTES. */
export const EXTRA_PALETTES = {
  ega: ['#000000', '#0000aa', '#00aa00', '#00aaaa', '#aa0000', '#aa00aa', '#aa5500', '#aaaaaa', '#555555', '#5555ff', '#55ff55', '#55ffff', '#ff5555', '#ff55ff', '#ffff55', '#ffffff'],
  playdate: ['#312e28', '#b1aea7'],
  amber: ['#1c0f02', '#ffb52e'],
  phosphor: ['#03140a', '#5cff8a'],
  paper: ['#1d1a2f', '#f4ecd8'],
};

export const ALL_PALETTES = { ...PALETTES, ...EXTRA_PALETTES };

// Palettes that are a single light→dark ramp: match them by lightness only (ignore hue),
// otherwise every non-green pixel would map to whatever green happens to be closest in hue.
const RAMP = new Set(['gameboy', 'mono', 'obra', 'sepia', 'playdate', 'amber', 'phosphor', 'paper']);

const hexRgb = (h) => {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
export function srgbToOklab([r, g, b]) {
  const R = toLin(r);
  const G = toLin(g);
  const B = toLin(b);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

const palCache = new Map();
/**
 * Uniform values for a palette: { pal, lab, palN, chromaW, dark, light }.
 *   pal  : array<vec4f, 64> sRGB colors (ramp palettes are sorted dark → light)
 *   lab  : array<vec4f, 64> the same colors in OKLab (precomputed on the CPU)
 *   chromaW : 0 for single-hue ramps (map brightness → ramp index), 1 = nearest color in OKLab
 */
export function paletteUniforms(name) {
  if (palCache.has(name)) return palCache.get(name);
  const ramp = RAMP.has(name);
  let list = (ALL_PALETTES[name] || ALL_PALETTES.pico8).slice(0, 64).map((h) => {
    const c = hexRgb(h);
    return { c, L: srgbToOklab(c) };
  });
  if (ramp) list.sort((a, b) => a.L[0] - b.L[0]);
  const pal = new Float32Array(64 * 4);
  const lab = new Float32Array(64 * 4);
  let dark = null;
  let light = null;
  list.forEach(({ c, L }, i) => {
    pal.set([...c, 1], i * 4);
    lab.set([...L, 1], i * 4);
    if (!dark || L[0] < dark.L) dark = { L: L[0], c };
    if (!light || L[0] > light.L) light = { L: L[0], c };
  });
  const n = list.length;
  const out = { pal, lab, palN: n, chromaW: ramp ? 0 : 1, dark: dark.c, light: light.c };
  palCache.set(name, out);
  return out;
}

/** Uniform declarations to add to a shaderScene `uniforms` object for palette support. */
export const PALETTE_UNIFORMS = { pal: 'array<vec4f, 64>', lab: 'array<vec4f, 64>', palN: 'f32', chromaW: 'f32' };

// Portable WGSL palette mapping (needs include 'color').
//   ramp palettes (chromaW = 0): brightness picks an index on the dark→light ramp (classic Game Boy conversion)
//   other palettes: nearest color in OKLab, a perceptual color space
// palDither(c, thr, amt): ordered dithering between the TWO nearest palette colors. `thr` is the
// Bayer/blue-noise threshold (0..1) for this pixel; the fraction of the way from the nearest color
// towards the 2nd nearest decides how often this pixel picks the 2nd one. Clean 2-color patterns.
export const PALETTE_WGSL = /* wgsl */ `
fn toLab(c: vec3f) -> vec3f { return linearToOklab(srgbToLinear(clamp(c, vec3f(0.0), vec3f(1.0)))); }
fn palDither(c: vec3f, thr: f32, amt: f32) -> vec3f {
  let th: f32 = mix(0.5, thr, amt);
  if (u.chromaW < 0.5) {
    let l: f32 = clamp(toLab(c).x, 0.0, 1.0);   // perceptual lightness
    let idx: i32 = i32(clamp(floor(l * (u.palN - 1.0) + th), 0.0, u.palN - 1.0));
    return u.pal[idx].xyz;
  }
  let q: vec3f = toLab(c);
  var b1: f32 = 1.0e9;
  var b2: f32 = 1.0e9;
  var i1: i32 = 0;
  var i2: i32 = 0;
  for (var i = 0; i < 64; i++) {
    if (f32(i) >= u.palN) { break; }
    let e: vec3f = q - u.lab[i].xyz;
    let dd: f32 = dot(e, e);
    if (dd < b1) { b2 = b1; i2 = i1; b1 = dd; i1 = i; } else if (dd < b2) { b2 = dd; i2 = i; }
  }
  let a: vec3f = u.lab[i1].xyz;
  let ab: vec3f = u.lab[i2].xyz - a;
  let t: f32 = clamp(dot(q - a, ab) / max(dot(ab, ab), 1.0e-6), 0.0, 1.0);
  if (th < t) { return u.pal[i2].xyz; }
  return u.pal[i1].xyz;
}
fn palNearest(c: vec3f) -> vec3f { return palDither(c, 0.5, 0.0); }
// Contrast + saturation boost before quantizing ("punch"): small palettes need it.
fn punchColor(c: vec3f, k: f32) -> vec3f {
  let s: vec3f = adjustSaturation(c, 1.0 + (k - 1.0) * 1.6);
  return clamp((s - vec3f(0.5)) * k + vec3f(0.5), vec3f(0.0), vec3f(1.0));
}
`;

// ------------------------------------------------------------------------------- overlay tags

/** Get (or lazily create) an absolutely positioned label inside ctx.overlay. */
export function overlayTag(ctx, key, css = '') {
  if (!ctx?.overlay) return null;
  let el = ctx.overlay.querySelector(`[data-tag="${key}"]`);
  if (!el) {
    el = document.createElement('div');
    el.className = 'tag';
    el.dataset.tag = key;
    el.style.cssText = 'pointer-events:none;font-size:12px;' + css;
    ctx.overlay.append(el);
  }
  return el;
}

/** Set text + visibility of an overlay tag, touching the DOM only when something changed. */
export function setTag(ctx, key, text, css = '', visible = true) {
  const el = overlayTag(ctx, key, css);
  if (!el) return;
  if (el._text !== text) {
    el.textContent = text;
    el._text = text;
  }
  const disp = visible ? '' : 'none';
  if (el.style.display !== disp) el.style.display = disp;
  return el;
}

/** Move a tag: left in % of the canvas, top as a CSS length, optional translateX. */
export function placeTag(el, leftPct, top = '48px', tx = '-50%') {
  if (!el) return;
  const l = `${leftPct.toFixed(2)}%`;
  const t = typeof top === 'number' ? `${top.toFixed(2)}%` : top;
  if (el.style.left !== l) el.style.left = l;
  if (el.style.top !== t) el.style.top = t;
  const tr = `translateX(${tx})`;
  if (el.style.transform !== tr) el.style.transform = tr;
}

// --------------------------------------------------------------------------- split divider

/** A before/after divider the user can drag. Returns the split position 0..1. */
export function makeSplit(initial = 0.5) {
  const s = { x: initial, dragging: false };
  return (ctx) => {
    const p = ctx.pointer;
    if (p.down) {
      s.x = Math.min(0.98, Math.max(0.02, p.x / Math.max(1, ctx.width)));
    }
    return s.x;
  };
}

// ------------------------------------------------------------------------ value-noise texture
let noiseCanvas = null;
/**
 * 256×256 tiling white noise (independent R, G, B, A). Sampled with linear filtering + repeat
 * (TEXR(t, p / 256.0)) it IS value noise with lattice spacing 1 — one texture fetch instead of
 * four hash evaluations. Declare it with { filter: 'linear', wrap: 'repeat' }.
 */
export function noiseTexture() {
  if (noiseCanvas) return noiseCanvas;
  const N = 256;
  const c = makeCanvas(N, N);
  const g = c.getContext('2d');
  const img = g.createImageData(N, N);
  const r = rng(99);
  for (let i = 0; i < img.data.length; i++) img.data[i] = (r() * 256) | 0;
  for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255; // keep alpha opaque (premultiplication-safe)
  g.putImageData(img, 0, 0);
  noiseCanvas = c;
  return c;
}
export const NOISE_TEX_WGSL = /* wgsl */ `
fn tnoise(p: vec2f) -> f32 { return TEXR(noiseTex, (p + vec2f(0.5)) / 256.0).r; }
fn tnoise3(p: vec2f) -> vec3f { return TEXR(noiseTex, (p + vec2f(0.5)) / 256.0).rgb; }
fn tfbm(p: vec2f) -> f32 {
  return (tnoise(p) * 0.5 + tnoise(p * 2.03 + vec2f(17.0, 9.0)) * 0.3 + tnoise(p * 4.01 + vec2f(3.0, 41.0)) * 0.2) * 2.0 - 1.0;
}
`;

// ------------------------------------------------------------------------- blue noise (CPU)

let blueNoiseCanvas = null;
/**
 * 64×64 blue-noise threshold texture made with Ulichney's void-and-cluster method:
 * every pixel gets a rank 0..4095; consecutive ranks are spread as evenly as possible,
 * so any threshold level produces an even, clump-free dot distribution.
 */
export function blueNoiseTexture(size = 64) {
  if (blueNoiseCanvas) return blueNoiseCanvas;
  const N = size * size;
  const sigma = 1.5;
  // toroidal gaussian kernel
  const K = new Float32Array(N);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = Math.min(x, size - x);
      const dy = Math.min(y, size - y);
      K[y * size + x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    }
  }
  const energy = new Float32Array(N);
  const bits = new Uint8Array(N);
  const splat = (idx, sgn) => {
    const px = idx % size;
    const py = (idx / size) | 0;
    for (let y = 0; y < size; y++) {
      const ky = ((y - py + size) % size) * size;
      const row = y * size;
      for (let x = 0; x < size; x++) energy[row + x] += sgn * K[ky + ((x - px + size) % size)];
    }
  };
  const tightest = () => {
    let best = -1;
    let bi = 0;
    for (let i = 0; i < N; i++) if (bits[i] && energy[i] > best) (best = energy[i]), (bi = i);
    return bi;
  };
  const largestVoid = () => {
    let best = Infinity;
    let bi = 0;
    for (let i = 0; i < N; i++) if (!bits[i] && energy[i] < best) (best = energy[i]), (bi = i);
    return bi;
  };
  // 1) initial random pattern (10%), relaxed until stable
  const rand = rng(1234);
  let ones = 0;
  while (ones < N / 10) {
    const i = (rand() * N) | 0;
    if (!bits[i]) {
      bits[i] = 1;
      splat(i, 1);
      ones++;
    }
  }
  for (let iter = 0; iter < N; iter++) {
    const c = tightest();
    bits[c] = 0;
    splat(c, -1);
    const v = largestVoid();
    bits[v] = 1;
    splat(v, 1);
    if (v === c) break;
  }
  const initial = bits.slice();
  const initialEnergy = energy.slice();
  const rank = new Int32Array(N);
  // 2) phase 1: remove tightest clusters → ranks ones-1 .. 0
  let r = ones - 1;
  while (r >= 0) {
    const c = tightest();
    bits[c] = 0;
    splat(c, -1);
    rank[c] = r--;
  }
  // 3) phases 2+3: from the initial pattern, fill the largest voids → ranks ones .. N-1
  bits.set(initial);
  energy.set(initialEnergy);
  r = ones;
  while (r < N) {
    const v = largestVoid();
    bits[v] = 1;
    splat(v, 1);
    rank[v] = r++;
  }
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let i = 0; i < N; i++) {
    const v = Math.floor((rank[i] + 0.5) * (256 / N));
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  blueNoiseCanvas = c;
  return c;
}

// ------------------------------------------------------------- Floyd–Steinberg (CPU, sequential)

/**
 * Classic error diffusion. Each pixel's rounding error is pushed onto pixels that have NOT
 * been processed yet (right, and the row below) — so pixel N depends on pixel N-1. That chain
 * is why it is a CPU / serpentine-scan algorithm and doesn't map to one-thread-per-pixel GPUs.
 * `gray` is a Float32Array in 0..1, modified in place → returns Uint8Array of 0/1.
 */
export function floydSteinberg(gray, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const ltr = y % 2 === 0; // serpentine scan avoids directional "worms"
    for (let k = 0; k < w; k++) {
      const x = ltr ? k : w - 1 - k;
      const i = y * w + x;
      const old = gray[i];
      const v = old >= 0.5 ? 1 : 0;
      out[i] = v;
      const err = old - v;
      const dir = ltr ? 1 : -1;
      if (x + dir >= 0 && x + dir < w) gray[i + dir] += (err * 7) / 16;
      if (y + 1 < h) {
        if (x - dir >= 0 && x - dir < w) gray[i + w - dir] += (err * 3) / 16;
        gray[i + w] += (err * 5) / 16;
        if (x + dir >= 0 && x + dir < w) gray[i + w + dir] += (err * 1) / 16;
      }
    }
  }
  return out;
}

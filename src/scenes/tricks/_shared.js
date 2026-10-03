// Small helpers shared by the "Pseudo-3D & Game Tricks" scenes.
import { makeCanvas } from '../../core/assets.js';

/** Create (or reuse) an overlay readout tag. css: e.g. 'right:8px;bottom:8px'. */
export function overlayTag(ctx, css, cls = 'tag') {
  const el = document.createElement('div');
  el.className = cls;
  el.style.cssText = `position:absolute;pointer-events:none;${css}`;
  ctx.overlay.append(el);
  return el;
}

/** -1..1 axis from held keys. */
export function keyAxis(keys, neg, pos) {
  let v = 0;
  for (const k of neg) if (keys.has(k)) v -= 1;
  for (const k of pos) if (keys.has(k)) v += 1;
  return Math.max(-1, Math.min(1, v));
}

export const KEYS = {
  up: ['ArrowUp', 'w', 'KeyW'],
  down: ['ArrowDown', 's', 'KeyS'],
  left: ['ArrowLeft', 'a', 'KeyA'],
  right: ['ArrowRight', 'd', 'KeyD'],
  turnL: ['ArrowLeft', 'q', 'KeyQ'],
  turnR: ['ArrowRight', 'e', 'KeyE'],
};

export function anyKey(keys, lists) {
  for (const l of lists) for (const k of l) if (keys.has(k)) return true;
  return false;
}

// ------------------------------------------------------------------ noise (JS, for baking textures)

export function hash2i(x, y, s = 0) {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(s | 0, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

export function vnoise(x, y, s = 0) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  let fx = x - ix;
  let fy = y - iy;
  fx = fx * fx * (3 - 2 * fx);
  fy = fy * fy * (3 - 2 * fy);
  const a = hash2i(ix, iy, s);
  const b = hash2i(ix + 1, iy, s);
  const c = hash2i(ix, iy + 1, s);
  const d = hash2i(ix + 1, iy + 1, s);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

export function fbm2(x, y, oct = 5, s = 0) {
  let v = 0;
  let a = 0.5;
  let f = 1;
  let n = 0;
  for (let i = 0; i < oct; i++) {
    v += a * vnoise(x * f, y * f, s + i * 17);
    n += a;
    a *= 0.5;
    f *= 2;
  }
  return v / n;
}

/**
 * Pack a mip chain into one texture so plain (non-mipmapped) sampling can do trilinear filtering
 * in the shader: level 0 occupies the left N×N, levels 1.. are stacked in the right column (N/2 wide).
 * Result is (1.5·N) × N. Works identically in WebGPU and WebGL2 (no API mip support needed).
 */
export function mipPack(base) {
  const N = base.width;
  const out = makeCanvas(N + N / 2, N);
  const g = out.getContext('2d');
  g.drawImage(base, 0, 0);
  let prev = base;
  let y = 0;
  for (let s = N / 2; s >= 4; s /= 2) {
    const c = makeCanvas(s, s);
    const cg = c.getContext('2d');
    cg.imageSmoothingEnabled = true;
    cg.imageSmoothingQuality = 'high';
    cg.drawImage(prev, 0, 0, s, s);
    g.drawImage(c, N, y);
    y += s;
    prev = c;
  }
  return out;
}

/** WGSL (portable) sampler for a mipPack()ed texture. lvl 0..; n = base size in texels. */
export const MIP_WGSL = /* wgsl */ `
fn mipCoord(m: vec2f, lvl: f32, n: f32) -> vec2f {
  let s = exp2(-lvl);
  let inset = 0.5 / (n * s);
  let q = clamp(m, vec2f(inset), vec2f(1.0 - inset));
  if (lvl < 0.5) { return vec2f(q.x / 1.5, q.y); }
  return vec2f((1.0 + q.x * s) / 1.5, 1.0 - 2.0 * s + q.y * s);
}
`;

/** Pixel-art painter on a small canvas. */
export function pixelPainter(w, h) {
  const c = makeCanvas(w, h);
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  const rect = (x, y, ww, hh, col) => {
    g.fillStyle = col;
    g.fillRect(x, y, ww, hh);
  };
  return { c, g, rect };
}

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const wrapAngle = (a) => {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
};

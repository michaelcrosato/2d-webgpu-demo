// Shared helpers for the "Sprites, Tiles & Cameras" scenes.

import { getAtlas } from '../../core/assets.js';
import { WGSL_MATH_PRELUDE, resolveIncludes } from '../../core/shaderlib.js';

/** Load the shared pixel-art atlas and upload it once per scene. */
export async function atlasTexture(gpu) {
  const atlas = await getAtlas();
  const tex = gpu.textureFromImage(atlas.canvas, { label: 'atlas' });
  return { atlas, tex, view: tex.createView() };
}

/** A readout chip in the scene overlay (CSS px). Returns the element; set .innerHTML each frame. */
export function tag(ctx, css) {
  const d = document.createElement('div');
  d.className = 'tag';
  d.style.cssText = css;
  ctx.overlay.append(d);
  return d;
}

/** Library code (PI/TAU/fmod + named includes) to prepend to a custom SpriteBatch fragment. */
export const wgslLib = (...names) => WGSL_MATH_PRELUDE + resolveIncludes(names);

export const fmt = (n) => Math.round(n).toLocaleString('en-US');

/** Integer pixel-art scale so a `targetPx`-tall view fits the canvas height. */
export const pixelScale = (h, targetPx) => Math.max(1, Math.round(h / targetPx));

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, t) => a + (b - a) * t;
/** Frame-rate independent exponential smoothing factor. */
export const damp = (rate, dt) => 1 - Math.exp(-rate * dt);

// ------------------------------------------------------------------ 3×5 pixel font

const GLYPHS = {
  0: '111101101101111', 1: '010110010010111', 2: '111001111100111', 3: '111001111001111', 4: '101101111001001',
  5: '111100111001111', 6: '111100111101111', 7: '111001001010010', 8: '111101111101111', 9: '111101111001111',
  A: '010101111101101', B: '110101110101110', C: '011100100100011', D: '110101101101110', E: '111100110100111',
  F: '111100110100100', G: '011100101101011', H: '101101111101101', I: '111010010010111', J: '001001001101010',
  K: '101101110101101', L: '100100100100111', M: '101111111101101', N: '110101101101101', O: '010101101101010',
  P: '110101110100100', Q: '010101101110011', R: '110101110101101', S: '011100010001110', T: '111010010010010',
  U: '101101101101111', V: '101101101101010', W: '101101111111101', X: '101101010101101', Y: '101101010010010',
  Z: '111001010100111', '+': '000010111010000', '-': '000000111000000', '!': '010010010000010', '.': '000000000000010',
  ':': '000010000010000', '/': '001001010100100', '%': '101001010100101', '?': '111001010000010', ' ': '000000000000000',
  '<': '001010100010001', '>': '100010001010100', '=': '000111000111000', '#': '101111101111101', '*': '000101010101000',
};

/**
 * Draw text with a chunky 3×5 pixel font using a ShapeBatch (one rect per lit pixel).
 * (x, y) = top-left (align 'left'), top-center ('center') or top-right ('right'). size = one font pixel.
 */
export function pixelText(shapes, text, x, y, size, color, { align = 'left', shadow = '#000000cc', alpha = 1 } = {}) {
  const str = String(text).toUpperCase();
  const w = str.length * 4 * size - size;
  let x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  const draw = (ox, oy, col) => {
    for (let k = 0; k < str.length; k++) {
      const g = GLYPHS[str[k]] || GLYPHS['?'];
      for (let i = 0; i < 15; i++) {
        if (g[i] === '1') shapes.rect(x0 + k * 4 * size + (i % 3) * size + ox, y + Math.floor(i / 3) * size + oy, size, size, col, { alpha });
      }
    }
  };
  if (shadow) draw(Math.max(1, size * 0.5), Math.max(1, size * 0.5), shadow);
  draw(0, 0, color);
  return w;
}

/** WGSL: `fn digitMask(d: i32, cell: vec2i) -> f32` for the 3×5 digits (cell x 0..2, y 0..4). */
export const DIGITS_WGSL = (() => {
  const vals = [];
  for (let d = 0; d < 10; d++) vals.push(parseInt(GLYPHS[d], 2) + 'u');
  return /* wgsl */ `
fn digitMask(d: i32, cell: vec2i) -> f32 {
  var font = array<u32, 10>(${vals.join(', ')});
  if (cell.x < 0 || cell.y < 0 || cell.x > 2 || cell.y > 4 || d < 0 || d > 9) { return 0.0; }
  let bit = u32(14 - (cell.y * 3 + cell.x));
  return f32((font[d] >> bit) & 1u);
}
// Draw a non-negative integer (up to 3 digits) centred at 'c' with font pixel size 's'.
fn numberMask(n: i32, p: vec2f, c: vec2f, s: f32) -> f32 {
  var digits = 1;
  if (n >= 10) { digits = 2; }
  if (n >= 100) { digits = 3; }
  let w = f32(digits) * 4.0 - 1.0;
  let q = (p - c) / s + vec2f(w * 0.5, 2.5);
  if (q.x < 0.0 || q.y < 0.0 || q.x >= w || q.y >= 5.0) { return 0.0; }
  let k = i32(floor(q.x / 4.0));
  let cx = i32(floor(q.x)) - k * 4;
  var v = n;
  for (var j = 0; j < 3; j++) { if (j < digits - 1 - k) { v = v / 10; } }
  return digitMask(v % 10, vec2i(cx, i32(floor(q.y))));
}
`;
})();

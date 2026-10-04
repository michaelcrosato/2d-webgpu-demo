// Text Rendering: Bitmap vs SDF — a custom WebGPU scene.
//  1. Text is rasterized with Canvas2D at HIGH resolution (3× the final atlas).
//  2. A compute shader finds the glyph edges, then the JUMP FLOODING algorithm (log2(n) compute
//     passes) gives every pixel its nearest edge pixel → a signed distance.
//  3. That distance is stored in a SMALL texture (64 px tall glyphs, rgba16float).
//  4. A text shader turns the distance into fill, outline, glow, drop shadow and bevel — at any size.

import { createGameScene } from '../../core/gamescene.js';
import { makeCanvas } from '../../core/assets.js';
import { labels } from './_shared.js';

const LBL = 'background:#000b;font-size:11px;padding:2px 7px;color:#e2e8f0';
const LW = 1024; // atlas size (low-res, what the text shader samples)
const ROW = 96;
const FONT_PX = 64;
const SPREAD = 12; // distance range stored, in atlas texels
const FONT = (px) => `bold ${px}px "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`;
const FONT_TITLE = (px) => `900 ${px}px "Arial Black", "Segoe UI Black", "Helvetica Neue", Arial, sans-serif`;
const GLYPHS = '0123456789+-!';
const CELL = 56;

// ---------------------------------------------------------------- jump flooding (compute)
const JFA_WGSL = /* wgsl */ `
fn idxOf(p: vec2i) -> u32 { return u32(p.y) * u32(u.hiSize.x) + u32(p.x); }
fn coordOf(i: u32) -> vec2f { let w = u32(u.hiSize.x); return vec2f(f32(i % w), f32(i / w)); }
fn insideAt(p: vec2i) -> bool {
  let s = vec2i(u.hiSize);
  let q = clamp(p, vec2i(0), s - vec2i(1));
  return textureLoad(src, q, 0).a >= 0.5;
}
fn alphaAt(p: vec2i) -> f32 { return textureLoad(src, clamp(p, vec2i(0), vec2i(u.hiSize) - vec2i(1)), 0).a; }
// 1) seeds: the glyph edge = partially covered (anti-aliased) pixels, plus inside pixels that
//    touch an outside pixel. 0 = no seed.
@compute @workgroup_size(8, 8) fn seed(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  if (f32(p.x) >= u.hiSize.x || f32(p.y) >= u.hiSize.y) { return; }
  var s = 0u;
  let a = alphaAt(p);
  var edge = a > 0.02 && a < 0.98;
  if (insideAt(p)) {
    if (!insideAt(p + vec2i(1, 0)) || !insideAt(p - vec2i(1, 0)) || !insideAt(p + vec2i(0, 1)) || !insideAt(p - vec2i(0, 1))) { edge = true; }
  }
  if (edge) { s = idxOf(p) + 1u; }
  B[idxOf(p)] = s;
}
// 2) one jump-flood step: look at 9 neighbours 'step' pixels away, keep the closest seed
@compute @workgroup_size(8, 8) fn jfa(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  if (f32(p.x) >= u.hiSize.x || f32(p.y) >= u.hiSize.y) { return; }
  let st = i32(u.jump);
  let pf = vec2f(p);
  var best = A[idxOf(p)];
  var bestD = 1e20;
  if (best != 0u) { let d = coordOf(best - 1u) - pf; bestD = dot(d, d); }
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = p + vec2i(dx, dy) * st;
      if (q.x < 0 || q.y < 0 || f32(q.x) >= u.hiSize.x || f32(q.y) >= u.hiSize.y) { continue; }
      let s = A[idxOf(q)];
      if (s != 0u) {
        let d = coordOf(s - 1u) - pf;
        let dd = dot(d, d);
        if (dd < bestD) { bestD = dd; best = s; }
      }
    }
  }
  B[idxOf(p)] = best;
}
// 3) resolve: one thread per LOW-res texel → signed distance in low-res texels
@compute @workgroup_size(8, 8) fn resolve(@builtin(global_invocation_id) id: vec3u) {
  let o = vec2i(id.xy);
  if (f32(o.x) >= u.loSize.x || f32(o.y) >= u.loSize.y) { return; }
  let q = (vec2f(o) + vec2f(0.5)) * u.factor;
  let ip = vec2i(floor(q));
  let s = A[idxOf(ip)];
  var sgn = 1.0;
  if (insideAt(ip)) { sgn = -1.0; }
  var sd = sgn * (u.spread * u.factor + 1.0);
  if (s != 0u) {
    let sp = coordOf(s - 1u);
    let spi = vec2i(sp);
    let a = alphaAt(spi);
    let c = sp + vec2f(0.5);
    // sub-pixel edge: the coverage gradient gives the edge normal, the coverage itself
    // how far the edge sits from the seed's center (a = 0.5 -> right through it)
    let g = vec2f(alphaAt(spi + vec2i(1, 0)) - alphaAt(spi - vec2i(1, 0)), alphaAt(spi + vec2i(0, 1)) - alphaAt(spi - vec2i(0, 1)));
    if (dot(g, g) > 1e-6) {
      let n = -normalize(g);
      let e = c + n * (a - 0.5);
      let v = q - e;
      let lv = length(v);
      // near the edge: distance to the local tangent line; further away: distance to the point
      sd = mix(dot(v, n), sgn * lv, smoothstep(2.0 * u.factor, 5.0 * u.factor, lv));
    } else {
      sd = sgn * (length(q - c) + 0.5 * sgn);
    }
  }
  sd = clamp(sd / u.factor, -u.spread, u.spread);
  textureStore(dst, o, vec4f(sd, 0.0, 0.0, 1.0));
}`;

// ---------------------------------------------------------------- text quads
// instance: center.xy size.xy | uv rect | fill | fill2 | outline | glow | p1 | p2 | p3
//   p1 = (weight, outline width, glow radius, softness)          [atlas texels]
//   p2 = (shadow dx, shadow dy, shadow blur, bevel)
//   p3 = (mode 0 sdf / 1 bitmap linear / 2 bitmap nearest / 3 field view / 4 sdf+shine,
//         wave amplitude, layer 0 all / 1 behind / 2 fill only, rotation)
const TEXT_WGSL = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) local: vec2f,
  @location(2) @interpolate(flat) fill: vec4f,
  @location(3) @interpolate(flat) fill2: vec4f,
  @location(4) @interpolate(flat) outline: vec4f,
  @location(5) @interpolate(flat) glow: vec4f,
  @location(6) @interpolate(flat) p1: vec4f,
  @location(7) @interpolate(flat) p2: vec4f,
  @location(8) @interpolate(flat) p3: vec4f,
  @location(9) @interpolate(flat) uvr: vec4f,
  @location(10) spx: vec2f,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32,
    @location(0) rect: vec4f, @location(1) uvr: vec4f, @location(2) fill: vec4f, @location(3) fill2: vec4f,
    @location(4) outline: vec4f, @location(5) glow: vec4f, @location(6) p1: vec4f, @location(7) p2: vec4f, @location(8) p3: vec4f) -> VOut {
  var corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let l = (c - vec2f(0.5)) * rect.zw;
  let cs = cos(p3.w);
  let sn = sin(p3.w);
  let w = rect.xy + vec2f(cs * l.x - sn * l.y, sn * l.x + cs * l.y);
  var o: VOut;
  o.pos = vec4f(w / u.resolution * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.uv = mix(uvr.xy, uvr.zw, c);
  o.local = c;
  o.fill = fill; o.fill2 = fill2; o.outline = outline; o.glow = glow; o.p1 = p1; o.p2 = p2; o.p3 = p3; o.uvr = uvr;
  o.spx = w;
  return o;
}
fn over(top: vec4f, below: vec4f) -> vec4f { return top + below * (1.0 - top.a); }   // premultiplied
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let mode = i32(i.p3.x + 0.5);
  let layer = i32(i.p3.z + 0.5);
  let ts = vec2f(textureDimensions(sdfTex));
  var uv = i.uv;
  if (i.p3.y > 0.0) {
    // wavy title: shift the lookup up/down along x (each letter bobs)
    uv.y += sin(i.local.x * 16.0 - u.time * 3.0) * i.p3.y * (i.uvr.w - i.uvr.y);
  }
  if (mode == 1 || mode == 2) {
    var a: f32;
    if (mode == 1) { a = textureSampleLevel(bmpTex, sLin, uv, 0.0).a; }
    else { a = textureSampleLevel(bmpTex, sNear, uv, 0.0).a; }
    return vec4f(i.fill.rgb * a, a) * i.fill.a;
  }
  let raw = textureSampleLevel(sdfTex, sLin, uv, 0.0).r;
  let d = raw - i.p1.x;                       // weight: shift the edge (bolder / thinner)
  let aa = max(fwidth(d), 0.0001);            // texels per screen pixel -> exact 1px AA
  if (mode == 3) {
    // debug: the distance field itself
    var v = clamp(0.5 - raw / (2.0 * ${SPREAD}.0), 0.0, 1.0);
    var col = vec3f(v);
    col *= 0.85 + 0.15 * cos(raw * 2.0);
    col = mix(col, vec3f(1.0, 0.35, 0.3), clamp(1.5 - abs(raw) / aa, 0.0, 1.0));
    return vec4f(col, 1.0);
  }
  let w = aa * 0.5 + i.p1.w;
  let fillA = smoothstep(w, -w, d);
  let outW = i.p1.y;
  let outA = smoothstep(w, -w, d - outW);
  var glowA = 0.0;
  if (i.p1.z > 0.001) { glowA = exp(-max(d - outW, 0.0) / i.p1.z) * i.glow.a * (1.0 - smoothstep(${SPREAD - 3}.0, ${SPREAD}.0, raw)); }
  var shA = 0.0;
  if (abs(i.p2.x) + abs(i.p2.y) > 0.001) {
    let ds = textureSampleLevel(sdfTex, sLin, uv - i.p2.xy / ts, 0.0).r - i.p1.x - outW;
    let sw = w + i.p2.z;
    shA = smoothstep(sw, -sw, ds) * 0.7;
  }
  // fill: vertical gradient, bevel lighting from the distance gradient, optional shine
  var fc = mix(i.fill.rgb, i.fill2.rgb, clamp((i.local.y - 0.25) / 0.5, 0.0, 1.0));
  if (i.p2.w > 0.0) {
    let g = vec2f(dpdx(d), dpdy(d));
    let gl = length(g);
    if (gl > 1e-6) {
      let n = g / gl;                                   // points away from the glyph
      let light = dot(n, normalize(vec2f(-1.0, -1.2)));
      let zone = smoothstep(-3.5, -0.5, d);
      fc = fc * (1.0 + light * i.p2.w * zone * 0.7) + vec3f(max(light, 0.0) * i.p2.w * zone * 0.3);
    }
  }
  if (mode == 4) {
    let sweep = fract(u.time * 0.25) * 2.4 - 0.7;
    let band = 1.0 - smoothstep(0.0, 0.06, abs(i.local.x - sweep + (i.local.y - 0.5) * 0.25));
    fc += vec3f(1.0, 0.95, 0.8) * band * 0.75;
  }
  var col = vec4f(0.0);
  if (layer != 2) {
    col = vec4f(0.0, 0.0, 0.0, shA);
    col = over(vec4f(i.glow.rgb * glowA, glowA), col);
    if (outW > 0.001) { col = over(vec4f(i.outline.rgb, 1.0) * outA * i.outline.a, col); }
  }
  if (layer != 1) { col = over(vec4f(fc, 1.0) * fillA * i.fill.a, col); }
  if (col.a < 0.002) { discard; }
  return col;
}`;

const BG_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let mode = i32(u.mode);
  if (mode == 2) { return vec4f(TEX(game, uv).rgb, 1.0); }
  if (mode == 0) {
    let row = floor(uv.y * 3.0);
    var c = mix(vec3f(0.06, 0.07, 0.1), vec3f(0.08, 0.085, 0.12), fmod(row, 2.0));
    let g = abs(fract(px / 32.0) - vec2f(0.5));
    c += vec3f(0.015) * step(0.47, max(g.x, g.y));
    let e = abs(fract(uv.y * 3.0 + 0.5) - 0.5) * res.y / 3.0;
    c = mix(c, vec3f(0.02), clamp(2.0 - e, 0.0, 1.0));
    return vec4f(c, 1.0);
  }
  let p = (px - res * 0.5) / res.y;
  if (mode == 1) {
    var c = mix(vec3f(0.11, 0.06, 0.2), vec3f(0.03, 0.03, 0.08), length(p) * 1.4);
    let a = atan2(p.y, p.x);
    c += vec3f(0.06, 0.03, 0.1) * pow(0.5 + 0.5 * sin(a * 12.0 + u.time * 0.3), 4.0) * smoothstep(0.9, 0.1, length(p));
    return vec4f(c + vec3f((hash21(px) - 0.5) / 255.0), 1.0);
  }
  // title: night sky, nebula, stars, mountain silhouettes
  var c = mix(vec3f(0.02, 0.02, 0.07), vec3f(0.12, 0.05, 0.22), uv.y);
  let n = fbm(p * 2.5 + vec2f(u.time * 0.01, 0.0), 5) * 0.5 + 0.5;
  c += vec3f(0.25, 0.08, 0.35) * smoothstep(0.45, 0.9, n) * 0.6 + vec3f(0.05, 0.15, 0.3) * smoothstep(0.55, 0.95, fbm(p * 4.0 + vec2f(3.0), 4) * 0.5 + 0.5) * 0.5;
  let cell = floor(px / 3.0);
  let h = hash21(cell);
  c += vec3f(step(0.996, h) * (0.5 + 0.5 * sin(u.time * 2.0 + h * 60.0)));
  // a falling star
  let ft = fract(u.time * 0.18);
  let sp = vec2f(0.75 - ft * 1.2, -0.45 + ft * 0.55);
  let dir = normalize(vec2f(-1.2, 0.55));
  let rel = p - sp;
  let along = dot(rel, dir);
  let across = abs(dot(rel, vec2f(-dir.y, dir.x)));
  c += vec3f(0.8, 0.9, 1.0) * smoothstep(0.004, 0.0, across) * smoothstep(-0.25, 0.0, along) * step(along, 0.0) * smoothstep(0.0, 0.1, ft) * (1.0 - ft);
  let m1 = 0.33 - 0.08 * ridged(vec2f(p.x * 2.0, 1.0), 4);
  let m2 = 0.42 - 0.05 * ridged(vec2f(p.x * 3.0 + 5.0, 2.0), 4);
  c = mix(c, vec3f(0.06, 0.04, 0.12), clamp((p.y - m1) * res.y, 0.0, 1.0));
  c = mix(c, vec3f(0.02, 0.015, 0.05), clamp((p.y - m2) * res.y, 0.0, 1.0));
  return vec4f(c, 1.0);
}`;

// ---------------------------------------------------------------- the text atlas
function buildAtlasCanvases(F) {
  const LH = ROW * 5;
  const lo = makeCanvas(LW, LH);
  const hi = makeCanvas(LW * F, LH * F);
  const rects = {};
  const glyphs = {};
  const gl = lo.getContext('2d');
  const gh = hi.getContext('2d');
  for (const g of [gl, gh]) {
    g.fillStyle = '#fff';
    g.textBaseline = 'alphabetic';
  }
  const put = (name, text, row, font, spacing = 0) => {
    gl.font = font(FONT_PX);
    gh.font = font(FONT_PX * F);
    const chars = [...text];
    let x = 24;
    const xs = [];
    for (const ch of chars) {
      xs.push(x);
      x += gl.measureText(ch).width + spacing;
    }
    const y = row * ROW + 70;
    chars.forEach((ch, i) => {
      gl.fillText(ch, xs[i], y);
      gh.fillText(ch, xs[i] * F, y * F);
    });
    rects[name] = [8, row * ROW, Math.min(LW, x + 16), (row + 1) * ROW];
  };
  put('demo', 'SDF Text', 0, FONT);
  put('level', 'LEVEL UP!', 1, FONT, 2);
  put('title', 'STARFALL', 2, FONT_TITLE, 4);
  put('press', 'PRESS START', 4, FONT, 6);
  // row 3: individual glyphs for numbers (fixed cells), and "CRIT!"
  gl.font = FONT(FONT_PX);
  gh.font = FONT(FONT_PX * F);
  [...GLYPHS].forEach((ch, i) => {
    const cx = 8 + i * CELL;
    const w = gl.measureText(ch).width;
    const x = cx + (CELL - w) / 2;
    const y = 3 * ROW + 70;
    gl.fillText(ch, x, y);
    gh.fillText(ch, x * F, y * F);
    glyphs[ch] = { rect: [cx, 3 * ROW, cx + CELL, 4 * ROW], adv: w + 3 };
  });
  {
    const cx = 8 + GLYPHS.length * CELL + 16;
    gl.fillText('CRIT!', cx + 8, 3 * ROW + 70);
    gh.fillText('CRIT!', (cx + 8) * F, (3 * ROW + 70) * F);
    rects.crit = [cx, 3 * ROW, Math.min(LW, cx + 16 + gl.measureText('CRIT!').width + 8), 4 * ROW];
  }
  return { lo, hi, LH, rects, glyphs };
}

const hex = (h, a = 1) => {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, a];
};

export default {
  interaction: 'Click on the game scene to deal a critical hit.',
  examples: [
    {
      id: 'zoom',
      label: 'Bitmap vs SDF',
      kind: 'Comparison',
      hint: 'Drag “Zoom” up: same 64-px textures, very different results.',
      note: 'All three rows come from textures of the <b>same size</b> (glyphs 64 px tall). The bitmap stores coverage, so enlarging it gives pixel blocks (nearest) or blur (linear). The SDF stores <b>distance to the edge</b>; linear filtering of a distance is still a good distance, so a sharp edge can be rebuilt at any zoom.',
    },
    {
      id: 'effects',
      label: 'Effects from one SDF',
      kind: 'Abstract',
      hint: '',
      note: 'Every style here comes from the <b>same</b> single-channel distance texture: thresholds give fill and outline, an exponential gives glow, a second offset lookup gives the drop shadow, the distance gradient gives a bevel. No extra textures, no re-rasterizing.',
    },
    {
      id: 'damage',
      label: 'Damage numbers',
      kind: 'In a game',
      hint: 'Click the scene to deal a critical hit.',
      note: 'Floating combat text: numbers pop in with an overshoot, rise and fade. <b>Crits</b> are bigger, golden, glowing and shake. Each digit is a quad from the glyph row of the atlas; outlines are drawn in a first pass so neighbouring digits never cover each other’s outline.',
    },
    {
      id: 'title',
      label: 'Glowing title logo',
      kind: 'In a game',
      hint: '',
      note: 'A title screen logo built entirely from one SDF word: gradient fill, bevel, thick outline, pulsing outer glow, drop shadow, a wavy per-letter bob and a shine sweep — then a blinking “PRESS START”.',
    },
  ],
  controls: [
    { type: 'slider', key: 'zoom', label: 'Zoom', min: 0.5, max: 14, step: 0.01, value: 3.5, log: true, showFor: ['zoom'], format: (v) => `${v.toFixed(2)}×` },
    { type: 'toggle', key: 'showField', label: 'Show the raw distance texture', value: false, showFor: ['zoom'], help: 'Bottom row: grey = distance (white inside, black far outside), red = the edge.' },
    { type: 'heading', label: 'Style (all from one SDF)', showFor: ['effects'] },
    { type: 'slider', key: 'weight', label: 'Weight (bold ↔ thin)', min: -2, max: 3, step: 0.05, value: 0, showFor: ['effects'], help: 'Moving the threshold = font weight for free.' },
    { type: 'slider', key: 'outlineW', label: 'Outline width', min: 0, max: 5, step: 0.05, value: 2.2, showFor: ['effects'] },
    { type: 'slider', key: 'glowR', label: 'Glow radius', min: 0, max: 4, step: 0.05, value: 1.8, showFor: ['effects'] },
    { type: 'slider', key: 'shadow', label: 'Drop shadow', min: 0, max: 10, step: 0.1, value: 4, showFor: ['effects'], help: 'Offset (and blur) of a second lookup into the same texture.' },
    { type: 'slider', key: 'bevel', label: 'Bevel / emboss', min: 0, max: 1.5, step: 0.01, value: 0.8, showFor: ['effects'], help: 'Lighting from the direction of the distance gradient.' },
    { type: 'slider', key: 'softness', label: 'Softness', min: 0, max: 3, step: 0.01, value: 0, showFor: ['effects'] },
    { type: 'color', key: 'fillColor', label: 'Fill', value: '#ffd166', showFor: ['effects'] },
    { type: 'color', key: 'outlineColor', label: 'Outline', value: '#3b0d4f', showFor: ['effects'] },
    { type: 'color', key: 'glowColor', label: 'Glow', value: '#ff4fd8', showFor: ['effects'] },
    { type: 'slider', key: 'rate', label: 'Hits per second', min: 0.5, max: 12, step: 0.1, value: 4, showFor: ['damage'] },
    { type: 'slider', key: 'crit', label: 'Crit chance', min: 0, max: 1, step: 0.01, value: 0.2, showFor: ['damage'], format: (v) => `${Math.round(v * 100)}%` },
    { type: 'slider', key: 'textSize', label: 'Text size', min: 0.4, max: 2, step: 0.01, value: 1, showFor: ['damage', 'title'] },
    { type: 'toggle', key: 'wave', label: 'Wavy letters', value: true, showFor: ['title'] },
  ],
  about: {
    summary: 'Text is the hardest “shape” in a game: thousands of tiny curves that must stay readable at every size. Signed distance field (SDF) fonts store distance instead of pixels, so one small texture renders crisp text at any scale — with outlines, glows and shadows for free.',
    what: `<p><b>Bitmap vs SDF</b>: the same text from equally small textures, magnified. <b>Effects</b>: one distance texture, many styles.
      <b>Damage numbers</b> and a <b>title logo</b>: how games use it. The distance field is built on the GPU at startup with <b>jump flooding</b> in compute shaders.</p>`,
    how: `<ol>
      <li><b>Rasterize</b> the text with the browser’s Canvas2D at 3× resolution (crisp edges).</li>
      <li><b>Seed pass</b> (compute): mark pixels on the glyph edges. <b>Jump flooding</b>: in passes with step 32, 16, 8, 4, 2, 1, every pixel looks at 8 neighbours <i>step</i> pixels away and keeps the nearest seed it hears about. After log₂(n) passes every pixel knows (almost exactly) its nearest edge.</li>
      <li><b>Resolve</b> (compute): for each texel of the small atlas, distance to that edge, negative inside, clamped to ±${SPREAD} texels → an <code>rgba16float</code> texture.</li>
      <li><b>Render</b>: sample the distance with linear filtering; <code>fwidth(d)</code> tells how many texels one screen pixel covers, so <code>smoothstep(w, −w, d)</code> is an exact 1-pixel anti-aliased edge at any zoom.</li>
      <li><b>Effects</b>: outline = second threshold (<code>d − width</code>); glow = <code>exp(−d / r)</code>; shadow = the same lookup at an offset; bevel = light · gradient of <i>d</i>; weight = shift the threshold.</li>
    </ol>`,
    uses: [
      { title: 'All in-game text', text: 'Unity TextMesh Pro, Godot, Unreal and many custom engines render UI text from SDF (or multi-channel MSDF) atlases.' },
      { title: 'Combat text & titles', text: 'Damage numbers, “LEVEL UP!”, logos and quest banners with outlines and glows that animate cheaply.' },
      { title: 'World-space text', text: 'Labels and signs that scale and rotate with the camera, name plates above characters, map labels when zooming.' },
      { title: 'Icons', text: 'Any monochrome icon set can be stored as an SDF atlas the same way.' },
    ],
    try: [
      'On <b>Bitmap vs SDF</b>, raise <i>Zoom</i> to 14× — then turn on <i>Show the raw distance texture</i> to see what the SDF row really samples.',
      'On <b>Effects</b>, drag <i>Weight</i> from −2 to 3: one font, many weights.',
      'Set <i>Softness</i> to 3 and <i>Outline width</i> to 0 for a blurry “out of focus” title.',
      'Turn <i>Bevel</i> up and <i>Glow radius</i> to 0 for an embossed metal look.',
      'On <b>Damage numbers</b>, set <i>Hits per second</i> to 12 and click repeatedly.',
    ],
    ask: [
      'SDF font rendering for all UI text',
      'floating damage numbers that pop and fade, crits bigger',
      'outlined and glowing text from a single distance field',
      'a wavy, glowing title logo with a shine sweep',
      'generate the distance field on the GPU with jump flooding',
    ],
    perf: `<p>Building the field: ${'~'}7 compute passes over a few million pixels — milliseconds on a GPU, done once. Rendering: one texture lookup per pixel (two with a shadow),
      the same cost as a bitmap font. Limitations: very sharp corners round off slightly at large zooms (MSDF — 3 channels — fixes that), and effects can’t extend beyond the stored distance range.</p>`,
    api: `<p>The <b>rendering</b> part works the same in WebGL2. The <b>jump flooding</b> here uses WebGPU compute shaders and storage buffers; in WebGL2 you would do the
      same passes as full-screen fragment shaders ping-ponging two textures, or precompute the SDF offline (as most engines do).</p>`,
    code: [
      {
        title: 'One jump-flooding step (compute shader)',
        lang: 'wgsl',
        src: `var best = A[idxOf(p)];   // nearest seed known so far (0 = none)
for (var dy = -1; dy <= 1; dy++) {
  for (var dx = -1; dx <= 1; dx++) {
    let q = p + vec2i(dx, dy) * step;          // look 'step' pixels away
    let s = A[idxOf(q)];
    if (s != 0u && dist2(p, s) < bestD) { best = s; bestD = dist2(p, s); }
  }
}
B[idxOf(p)] = best;                             // then step /= 2 and swap A/B`,
      },
      {
        title: 'Rendering: fill, outline, glow from one distance',
        lang: 'wgsl',
        src: `let d = textureSampleLevel(sdfTex, sLin, uv, 0.0).r - weight;
let w = fwidth(d) * 0.5 + softness;              // 1 screen pixel, in texels
let fillA = smoothstep(w, -w, d);
let outA  = smoothstep(w, -w, d - outlineWidth);
let glowA = exp(-max(d - outlineWidth, 0.0) / glowRadius);
let shA   = smoothstep(w, -w, sampleAt(uv - shadowOffset) - outlineWidth);`,
      },
    ],
    links: [
      { title: 'Valve — Improved Alpha-Tested Magnification (SIGGRAPH 2007)', url: 'https://steamcdn-a.akamaihd.net/apps/valve/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf', note: 'the paper that popularised SDF text' },
      { title: 'Rong & Tan — Jump Flooding in GPU (2006)', url: 'https://www.comp.nus.edu.sg/~tants/jfa.html' },
      { title: 'msdfgen — multi-channel SDF', url: 'https://github.com/Chlumsky/msdfgen' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const F = ctx.testMode ? 2 : 3;
    const at = buildAtlasCanvases(F);
    const LH = at.LH;
    const HW = LW * F;
    const HH = LH * F;

    // ---------------- build the SDF on the GPU (once)
    const UJ = gpu.uniforms({ hiSize: 'vec2f', loSize: 'vec2f', jump: 'f32', factor: 'f32', spread: 'f32' }, 'J');
    const jfa = gpu.compute({
      label: 'jfa',
      bindings: {
        u: { uniform: UJ },
        src: { texture: true },
        A: { storage: 'array<u32>', access: 'read' },
        B: { storage: 'array<u32>', access: 'read_write' },
        dst: { storageTexture: 'rgba16float', access: 'write' },
      },
      code: JFA_WGSL,
    });
    const hiTex = gpu.textureFromImage(at.hi, { label: 'text-hi' });
    const bufA = gpu.storage(HW * HH * 4, 'jfa-a');
    const bufB = gpu.storage(HW * HH * 4, 'jfa-b');
    const sdf = gpu.target(LW, LH, { format: 'rgba16float', label: 'text-sdf' });
    UJ.set('hiSize', [HW, HH]).set('loSize', [LW, LH]).set('factor', F).set('spread', SPREAD);
    const run = (entry, groups, a, b) => {
      // a separate submit per pass, so each pass sees its own uniform values
      UJ.upload(true);
      const enc = device.createCommandEncoder({ label: `jfa-${entry}` });
      jfa.dispatch(enc, entry, groups, { u: UJ, src: hiTex, A: a, B: b, dst: sdf });
      device.queue.submit([enc.finish()]);
    };
    const gHi = [Math.ceil(HW / 8), Math.ceil(HH / 8)];
    UJ.set('jump', 1);
    run('seed', gHi, bufA, bufB); // writes B
    let read = bufB;
    let write = bufA;
    const steps = [];
    for (let s = 2 ** Math.ceil(Math.log2(SPREAD * F)) / 2; s >= 1; s /= 2) steps.push(s);
    steps.push(1); // the classic "JFA+1" extra pass fixes most remaining errors
    for (const s of steps) {
      UJ.set('jump', s);
      run('jfa', gHi, read, write);
      [read, write] = [write, read];
    }
    run('resolve', [Math.ceil(LW / 8), Math.ceil(LH / 8)], read, write);
    await device.queue.onSubmittedWorkDone();
    bufA.destroy();
    bufB.destroy();
    hiTex.destroy();
    const bmpTex = gpu.textureFromImage(at.lo, { label: 'text-bitmap' });

    // ---------------- rendering
    const U = gpu.uniforms({ resolution: 'vec2f', time: 'f32', mode: 'f32' }, 'T');
    const textProg = gpu.program({
      label: 'sdf-text',
      bindings: { u: { uniform: U }, sdfTex: { texture: true }, bmpTex: { texture: true }, sLin: { sampler: true }, sNear: { sampler: true } },
      code: TEXT_WGSL,
    });
    const attrs = [];
    for (let i = 0; i < 9; i++) attrs.push({ shaderLocation: i, offset: i * 16, format: 'float32x4' });
    const textPipe = textProg.renderPipeline({ format: gpu.format, blend: 'premultiplied', buffers: [{ arrayStride: 144, stepMode: 'instance', attributes: attrs }] });
    const UB = gpu.uniforms({ resolution: 'vec2f', time: 'f32', mode: 'f32' }, 'T');
    const bg = gpu.fullscreen({ label: 'text-bg', code: BG_WGSL, uniforms: UB, textures: ['game'], include: ['hash', 'noise', 'math'] });
    const game = createGameScene(gpu);
    const blank = gpu.target(1, 1, { label: 'blank' });

    let cap = 512;
    let inst = new Float32Array(cap * 36);
    let ibuf = gpu.buffer({ size: cap * 144, usage: GPUBufferUsage.VERTEX, label: 'text-instances' });
    let count = 0;
    const uvOf = (r) => [r[0] / LW, r[1] / LH, r[2] / LW, r[3] / LH];
    /** Queue a text quad. r: atlas rect (texels), cx/cy center px, scale = screen px per texel. */
    const quad = (r, cx, cy, scale, st) => {
      if (count >= cap) {
        cap *= 2;
        const n = new Float32Array(cap * 36);
        n.set(inst);
        inst = n;
        ibuf.destroy();
        ibuf = gpu.buffer({ size: cap * 144, usage: GPUBufferUsage.VERTEX, label: 'text-instances' });
      }
      const o = count++ * 36;
      inst.set([cx, cy, (r[2] - r[0]) * scale, (r[3] - r[1]) * scale], o);
      inst.set(uvOf(r), o + 4);
      inst.set(st.fill || [1, 1, 1, 1], o + 8);
      inst.set(st.fill2 || st.fill || [1, 1, 1, 1], o + 12);
      inst.set(st.outline || [0, 0, 0, 1], o + 16);
      inst.set(st.glow || [0, 0, 0, 0], o + 20);
      inst.set([st.weight || 0, st.outlineW || 0, st.glowR || 0, st.soft || 0], o + 24);
      inst.set([st.shadowX || 0, st.shadowY || 0, st.shadowBlur || 0, st.bevel || 0], o + 28);
      inst.set([st.mode || 0, st.wave || 0, st.layer || 0, st.rot || 0], o + 32);
    };
    const wordWidth = (str) => [...str].reduce((s, ch) => s + (at.glyphs[ch]?.adv || CELL * 0.6), 0);
    /** A number like "128" / "+12" from individual glyph quads, centered at cx. */
    const numberQuads = (str, cx, cy, scale, st) => {
      let x = cx - (wordWidth(str) * scale) / 2;
      for (const ch of str) {
        const g = at.glyphs[ch];
        if (!g) continue;
        quad(g.rect, x + (g.adv * scale) / 2, cy, scale, st);
        x += g.adv * scale;
      }
    };

    const numbers = [];
    let spawnAcc = 0;
    const rnd = (a, b) => a + Math.random() * (b - a);
    const spawn = (x, y, crit, t) => {
      const heal = !crit && Math.random() < 0.15;
      const v = crit ? Math.round(rnd(180, 999)) : heal ? Math.round(rnd(5, 40)) : Math.round(rnd(8, 120));
      numbers.push({ x, y, vx: rnd(-30, 30), born: t, crit, heal, text: heal ? `+${v}` : String(v), rot: crit ? rnd(-0.15, 0.15) : 0 });
    };

    const status = document.createElement('div');
    status.className = 'tag';
    status.style.cssText = 'right:8px;bottom:8px';
    status.textContent = `SDF built on the GPU: ${HW}×${HH} px → ${steps.length} jump-flood passes → ${LW}×${LH} distance texture`;
    ctx.overlay.append(status);

    const res = { sdfTex: sdf, bmpTex, sLin: 'linear', sNear: 'nearest', u: U };

    return {
      frame(ctx) {
        const P = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const t = ctx.time;
        const enc = ctx.encoder;
        const canvas = { view: ctx.target, format: gpu.format };
        const ex = ctx.example;
        count = 0;
        U.set('resolution', [W, H]).set('time', t).upload();
        UB.set('resolution', [W, H]).set('time', t);
        status.style.display = ex === 'zoom' ? '' : 'none';
        let ranges = null;

        if (ex === 'zoom') {
          labels(ctx, `zoom-${!!P.showField}`, [
            { text: 'bitmap · nearest', x: 0.985, y: 1 / 3 - 0.012, align: 'right', valign: 'bottom', style: LBL },
            { text: 'bitmap · linear', x: 0.985, y: 2 / 3 - 0.012, align: 'right', valign: 'bottom', style: LBL },
            { text: P.showField ? 'the distance texture (raw)' : 'SDF · same size texture', x: 0.985, y: 0.988 - 0.05, align: 'right', valign: 'bottom', style: LBL },
          ]);
          UB.set('mode', 0);
          bg.draw(enc, canvas, { game: blank });
          const r = at.rects.demo;
          const rh = H / 3;
          const scale = P.zoom * Math.max(1, H / 900);
          ranges = [];
          for (let k = 0; k < 3; k++) {
            const first = count;
            const mode = k === 0 ? 2 : k === 1 ? 1 : P.showField ? 3 : 0;
            quad(r, W / 2, rh * (k + 0.5), scale, { mode, fill: [0.95, 0.96, 1, 1] });
            ranges.push([first, 1, 0, Math.round(rh * k), W, Math.round(rh)]);
          }
        } else if (ex === 'effects') {
          labels(ctx, 'effects', [{ text: 'the same SDF at 3 sizes', x: 0.5, y: 0.9, style: LBL }]);
          UB.set('mode', 1);
          bg.draw(enc, canvas, { game: blank });
          const r = at.rects.level;
          const st = {
            fill: hex(P.fillColor),
            fill2: (() => {
              const c = hex(P.fillColor);
              return [c[0] * 0.75, c[1] * 0.55, c[2] * 0.6, 1];
            })(),
            outline: hex(P.outlineColor),
            glow: hex(P.glowColor, 0.9),
            weight: P.weight,
            outlineW: P.outlineW,
            glowR: P.glowR,
            soft: P.softness,
            shadowX: P.shadow * 0.6,
            shadowY: P.shadow,
            shadowBlur: P.shadow * 0.25,
            bevel: P.bevel,
          };
          const big = Math.min((W * 0.86) / (r[2] - r[0]), (H * 0.42) / (r[3] - r[1]));
          quad(r, W / 2, H * 0.4, big, { ...st, rot: Math.sin(t * 0.6) * 0.025 });
          const sizes = [0.5, 0.3, 0.18];
          let x = W * 0.08;
          for (const s of sizes) {
            const w = (r[2] - r[0]) * big * s;
            quad(r, x + w / 2, H * 0.78, big * s, st);
            x += w + W * 0.04;
          }
        } else if (ex === 'damage') {
          labels(ctx, 'damage', []);
          const gs = ctx.testMode ? 0.5 : 1;
          const gt = game.render(enc, t, Math.round(W * gs), Math.round(H * gs));
          UB.set('mode', 2);
          bg.draw(enc, canvas, { game: gt });
          const p = ctx.pointer;
          if (p.clicked) spawn(p.x, p.y, true, t);
          if (!ctx.paused) {
            spawnAcc += ctx.dt * P.rate;
            while (spawnAcc >= 1) {
              spawnAcc -= 1;
              spawn(rnd(0.22, 0.85) * W, rnd(0.42, 0.66) * H, Math.random() < P.crit, t);
            }
          }
          const base = Math.max(0.6, H / 700) * 0.78 * P.textSize;
          const layers = [1, 2];
          for (const layer of layers) {
            for (let i = numbers.length - 1; i >= 0; i--) {
              const n = numbers[i];
              const age = t - n.born;
              const life = n.crit ? 1.5 : 1.1;
              if (age > life || age < 0) {
                if (layer === 2) numbers.splice(i, 1);
                continue;
              }
              const k = Math.min(1, age / 0.18);
              const back = 1 + 2.70158 * Math.pow(k - 1, 3) + 1.70158 * Math.pow(k - 1, 2); // easeOutBack
              const settle = n.crit ? 1.35 : 1;
              const sc = base * (age < 0.18 ? back * (n.crit ? 1.6 : 1.1) : settle + (n.crit ? 0.25 : 0.1) * Math.exp(-(age - 0.18) * 8));
              const rise = (1 - Math.exp(-age * 2.2)) * H * 0.12;
              const alpha = 1 - Math.max(0, (age - life * 0.6) / (life * 0.4));
              const shake = n.crit && age < 0.4 ? Math.sin(age * 90) * 4 * (1 - age / 0.4) : 0;
              const x = n.x + n.vx * age + shake;
              const y = n.y - rise;
              let st;
              if (n.crit) {
                st = { fill: [1, 0.95, 0.45, alpha], fill2: [1, 0.45, 0.1, alpha], outline: [0.35, 0.02, 0.02, alpha], outlineW: 2.4, glow: [1, 0.4, 0.1, 0.8 * alpha], glowR: 2.2, bevel: 0.6, shadowY: 3, shadowX: 2, shadowBlur: 1, rot: n.rot, layer };
              } else if (n.heal) {
                st = { fill: [0.6, 1, 0.6, alpha], fill2: [0.2, 0.8, 0.3, alpha], outline: [0.02, 0.15, 0.05, alpha], outlineW: 2, shadowY: 2, shadowX: 1, shadowBlur: 0.5, layer };
              } else {
                st = { fill: [1, 1, 1, alpha], fill2: [0.85, 0.85, 0.9, alpha], outline: [0.08, 0.06, 0.1, alpha], outlineW: 2, shadowY: 2, shadowX: 1, shadowBlur: 0.5, layer };
              }
              numberQuads(n.text, x, y, sc, st);
              if (n.crit) quad(at.rects.crit, x, y - ROW * sc * 0.62, sc * 0.5, { ...st, fill: [1, 0.3, 0.25, alpha], fill2: [0.8, 0.1, 0.1, alpha], outline: [0.2, 0, 0, alpha], glowR: 1.5, glow: [1, 0.2, 0.1, 0.6 * alpha], bevel: 0 });
            }
          }
        } else {
          labels(ctx, 'title', []);
          UB.set('mode', 3);
          bg.draw(enc, canvas, { game: blank });
          const r = at.rects.title;
          const sc = Math.min((W * 0.8) / (r[2] - r[0]), (H * 0.36) / (r[3] - r[1])) * P.textSize;
          const pulse = 0.5 + 0.5 * Math.sin(t * 2.2);
          quad(r, W / 2, H * 0.38, sc, {
            mode: 4,
            fill: [1, 0.93, 0.55, 1],
            fill2: [1, 0.45, 0.2, 1],
            outline: [0.1, 0.04, 0.22, 1],
            outlineW: 2.4,
            glow: [0.35, 0.8, 1, 0.75 + 0.25 * pulse],
            glowR: 1.6 + pulse * 1.2,
            shadowX: 2,
            shadowY: 4,
            shadowBlur: 1.5,
            bevel: 0.9,
            wave: P.wave ? 0.045 : 0,
          });
          const pr = at.rects.press;
          const blink = 0.55 + 0.45 * Math.sin(t * 4);
          quad(pr, W / 2, H * 0.74, sc * 0.32, { fill: [1, 1, 1, blink], outline: [0.1, 0.05, 0.2, blink], outlineW: 1.6, glow: [0.6, 0.5, 1, 0.6 * blink], glowR: 2 });
        }

        device.queue.writeBuffer(ibuf, 0, inst, 0, count * 36);
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.target, loadOp: 'load', storeOp: 'store' }] });
        pass.setPipeline(textPipe);
        pass.setBindGroup(0, textProg.bind(res));
        pass.setVertexBuffer(0, ibuf);
        if (ranges) {
          for (const [first, n, x, y, w, h] of ranges) {
            pass.setScissorRect(x, y, Math.min(w, W - x), Math.max(1, Math.min(h, H - y)));
            pass.draw(6, n, 0, first);
          }
        } else if (count) pass.draw(6, count, 0, 0);
        pass.end();
      },
    };
  },
};

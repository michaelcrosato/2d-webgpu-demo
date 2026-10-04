import { shaderScene } from '../../core/shaderscene.js';
import { labels } from './_shared.js';

// Gradients & color spaces: gradient shapes and spread modes, interpolation in sRGB vs linear
// light vs OKLab vs HSV, banding vs dithering, and a sky + health-bar ramp "done right".

const TYPE_NAMES = ['linear', 'radial', 'conic (angular)', 'diamond', 'four-corner (bilinear)', 'multi-stop'];
const SPACE_NAMES = ['sRGB — naive mix', 'linear light', 'OKLab — perceptual', 'HSV — hue path'];
const BAR_NAMES = ['sRGB lerp', 'OKLab lerp', '3 stops (OKLab)'];
const LBL = 'background:#000a;font-size:11px;padding:2px 7px;color:#dbe4f3';
const DITHERS = ['none', 'white noise', 'IGN', 'Bayer 4×4', 'Bayer 8×8', 'triangular (TPDF)'];
let lowRes = false; // half-resolution rendering under the (software-GPU) test harness

// OKLab in JavaScript: per-frame constants (sky keyframes) are blended on the CPU, once,
// instead of in every pixel.
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const toSrgb = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055);
function rgbToOklab([r, g, b]) {
  [r, g, b] = [toLin(r), toLin(g), toLin(b)];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function oklabToRgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s].map((v) => Math.min(1, Math.max(0, toSrgb(v))));
}
const mixOk = (x, y, t) => {
  const a = rgbToOklab(x);
  const b = rgbToOklab(y);
  return oklabToRgb([0, 1, 2].map((i) => a[i] + (b[i] - a[i]) * t));
};
const SKY_ZEN = [[0.015, 0.02, 0.07], [0.2, 0.24, 0.5], [0.14, 0.4, 0.86], [0.13, 0.1, 0.34], [0.015, 0.02, 0.07]];
const SKY_HOR = [[0.06, 0.08, 0.2], [1.0, 0.64, 0.44], [0.68, 0.85, 0.98], [1.0, 0.45, 0.28], [0.06, 0.08, 0.2]];
function skyState(params, time) {
  const tod = params.cycle ? (0.62 + time * 0.025) % 1 : params.tod;
  const seg = tod * 4;
  const i = Math.min(3, Math.floor(seg));
  let f = seg - Math.floor(seg);
  f = f * f * (3 - 2 * f);
  const zen = mixOk(SKY_ZEN[i], SKY_ZEN[i + 1], f);
  const hor = mixOk(SKY_HOR[i], SKY_HOR[i + 1], f);
  const a = (tod - 0.25) * Math.PI * 2;
  const sunUp = Math.min(1, Math.max(0, (Math.sin(a) + 0.1) / 0.25));
  const day = 0.25 + 0.75 * sunUp;
  return {
    zen,
    hor,
    hill1: mixOk([0.22 * day, 0.27 * day, 0.55 * day], hor, 0.45),
    hill2: mixOk([0.08 * day, 0.24 * day, 0.2 * day], hor, 0.2),
    sky: [tod, a, sunUp, day],
  };
}

export default shaderScene({
  examples: [
    {
      id: 'types',
      label: 'Gradient types',
      kind: 'Abstract',
      hint: 'Hover the radial, conic or diamond tile to move its center.',
      note: 'A gradient is just “compute a number <i>t</i> from the pixel position, then look up a color for <i>t</i>”. The shape of the gradient is entirely in how <i>t</i> is computed.',
    },
    {
      id: 'spaces',
      label: 'sRGB vs linear vs OKLab',
      kind: 'Comparison',
      hint: '',
      note: 'The same two colors blended four ways. The white curve is perceived lightness (OKLab L) along each strip; the dashed line is what “even” would look like. Naive sRGB mixing sags into a dark, muddy middle; OKLab stays even.',
      params: { colA: '#e8243c', colB: '#2fd16a' },
    },
    {
      id: 'banding',
      label: 'Banding & dithering',
      kind: 'Comparison',
      hint: '',
      note: 'A dark gradient stored with few bits per channel shows visible steps (<b>banding</b>). Adding a tiny bit of noise <i>before</i> rounding (<b>dithering</b>) trades the steps for invisible grain. Left: plain rounding. Right: dithered.',
      params: { colA: '#26355e', colB: '#06070d' },
    },
    {
      id: 'game',
      label: 'Sky & health bars',
      kind: 'In a game',
      hint: '',
      note: 'A day/night sky whose color keyframes are blended in OKLab, and three health bars that fade green → red as HP drops. The sRGB one passes through a dull olive-brown; OKLab and a 3-stop ramp stay vivid.',
    },
  ],
  controls: [
    { type: 'heading', label: 'Colors' },
    { type: 'color', key: 'colA', label: 'Color A', value: '#ff4d8d', showFor: ['types', 'spaces', 'banding'] },
    { type: 'color', key: 'colB', label: 'Color B', value: '#4d6bff', showFor: ['types', 'spaces', 'banding'] },
    { type: 'color', key: 'colC', label: 'Color C (3rd stop / corner)', value: '#ffd84d', showFor: ['types'] },
    {
      type: 'select', key: 'space', label: 'Interpolate in', value: 'oklab', showFor: ['types'],
      options: [{ value: 'srgb', label: 'sRGB (naive)' }, { value: 'linear', label: 'Linear light' }, { value: 'oklab', label: 'OKLab (perceptual)' }, { value: 'hsv', label: 'HSV' }],
      help: 'The color space the two colors are mixed in.',
    },
    { type: 'heading', label: 'Shape', showFor: ['types'] },
    { type: 'slider', key: 'angle', label: 'Angle', min: 0, max: 360, step: 1, value: 30, unit: '°', showFor: ['types'], help: 'Direction of the linear gradients, start of the conic one.' },
    {
      type: 'select', key: 'spread', label: 'Spread mode', value: 'pad', showFor: ['types'],
      options: [{ value: 'pad', label: 'Pad (clamp)' }, { value: 'repeat', label: 'Repeat' }, { value: 'reflect', label: 'Reflect (mirror)' }],
      help: 'What happens when t goes past 0 or 1.',
    },
    { type: 'slider', key: 'steps', label: 'Posterize steps', min: 0, max: 12, step: 1, value: 0, showFor: ['types'], format: (v) => (v < 2 ? 'smooth' : `${v} bands`), help: 'Quantize t into flat bands — the classic toon-shading ramp.' },
    { type: 'toggle', key: 'curve', label: 'Show lightness curve', value: true, showFor: ['spaces'] },
    { type: 'heading', label: 'Quantization', showFor: ['banding'] },
    { type: 'slider', key: 'bits', label: 'Bits per channel', min: 2, max: 8, step: 1, value: 6, showFor: ['banding'], format: (v) => `${v} bits (${2 ** v} levels)`, help: 'Screens use 8. Fewer bits exaggerate the problem so it is easy to see.' },
    {
      type: 'select', key: 'dither', label: 'Dither (right half)', value: 2, showFor: ['banding'],
      options: DITHERS.map((label, value) => ({ value, label })),
      help: 'Noise added before rounding. Ordered (Bayer) patterns are regular; IGN / TPDF look like fine film grain.',
    },
    { type: 'slider', key: 'boost', label: 'Contrast boost', min: 1, max: 8, step: 0.1, value: 2.5, showFor: ['banding'], help: 'Brightens the result after rounding, to make the steps visible on any monitor.' },
    { type: 'toggle', key: 'animNoise', label: 'Animate noise', value: false, showFor: ['banding'], help: 'Temporal dithering: new noise every frame averages out even further.' },
    { type: 'heading', label: 'Game', showFor: ['game'] },
    { type: 'toggle', key: 'cycle', label: 'Day/night cycle', value: true, showFor: ['game'] },
    { type: 'slider', key: 'tod', label: 'Time of day', min: 0, max: 1, step: 0.001, value: 0.62, showFor: ['game'], format: (v) => { const h = (v * 24) % 24; return `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`; }, help: 'Used when the cycle is off.' },
    { type: 'toggle', key: 'autoHp', label: 'Animate HP', value: true, showFor: ['game'] },
    { type: 'slider', key: 'hp', label: 'HP', min: 0, max: 1, step: 0.001, value: 0.5, showFor: ['game'], format: (v) => `${Math.round(v * 100)}%`, help: 'Used when “Animate HP” is off. Try 50% — the muddiest point.' },
  ],
  uniforms: {
    colA: 'vec3f', colB: 'vec3f', colC: 'vec3f', space: 'f32', angle: 'f32', spread: 'f32', steps: 'f32', curve: 'f32',
    bits: 'f32', dither: 'f32', boost: 'f32', animNoise: 'f32', cycle: 'f32', tod: 'f32', autoHp: 'f32', hp: 'f32',
    zen: 'vec3f', hor: 'vec3f', hill1: 'vec3f', hill2: 'vec3f', sky: 'vec4f',
  },
  renderScale: () => (lowRes ? 0.5 : 1),
  include: ['color', 'hash', 'dither', 'math', 'sdf'],
  bind(params, ctx) {
    lowRes = !!ctx.testMode;
    const ex = ctx.example;
    if (ex === 'types') {
      labels(ctx, 'types', TYPE_NAMES.map((n, i) => ({ text: n, x: ((i % 3) + 0.5) / 3, y: (Math.floor(i / 3) + 0.9) / 2, valign: 'middle', style: LBL })));
    } else if (ex === 'spaces') {
      labels(ctx, 'spaces', SPACE_NAMES.map((n, i) => ({ text: n, x: 0.035, y: 0.13 + i * 0.215, align: 'left', valign: 'middle', style: LBL }))
        .concat([{ text: 'mix at t = 0.5', x: 0.9, y: 0.83, valign: 'top', style: LBL }]));
    } else if (ex === 'banding') {
      labels(ctx, 'banding', [
        { text: 'plain rounding', x: 0.25, y: 0.04, style: LBL },
        { text: 'dithered', x: 0.75, y: 0.04, style: LBL },
      ]);
    } else if (ex === 'game') {
      labels(ctx, 'game', BAR_NAMES.map((n, i) => ({ text: n, x: 0.255, y: 0.73 + i * 0.095, align: 'right', valign: 'middle', style: LBL })));
      return skyState(params, ctx.time);
    }
    return {};
  },
  code: /* wgsl */ `
fn cov(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }

// Mix two sRGB colors in a chosen space: 0 sRGB, 1 linear light, 2 OKLab, 3 HSV (shortest hue path)
fn mixSpace(a: vec3f, b: vec3f, t: f32, space: i32) -> vec3f {
  if (space == 0) { return mix(a, b, t); }
  if (space == 1) { return linearToSrgb(mix(srgbToLinear(a), srgbToLinear(b), t)); }
  if (space == 2) { return clamp(mixOklab(a, b, t), vec3f(0.0), vec3f(1.0)); }
  let ha = rgb2hsv(a);
  let hb = rgb2hsv(b);
  var dh = hb.x - ha.x;
  dh = dh - floor(dh + 0.5);
  return hsv2rgb(vec3f(fract(ha.x + dh * t), mix(ha.y, hb.y, t), mix(ha.z, hb.z, t)));
}
fn oklabL(c: vec3f) -> f32 { return linearToOklab(srgbToLinear(c)).x; }

fn spreadT(t: f32) -> f32 {
  if (u.spread > 1.5) { return 1.0 - abs(fract(t * 0.5) * 2.0 - 1.0); }
  if (u.spread > 0.5) { return fract(t); }
  return clamp(t, 0.0, 1.0);
}
fn posterize(t: f32) -> f32 {
  if (u.steps < 1.5) { return t; }
  return min(floor(t * u.steps), u.steps - 1.0) / (u.steps - 1.0);
}

fn bg(px: vec2f) -> vec3f { return mix(vec3f(0.045, 0.05, 0.075), vec3f(0.07, 0.06, 0.1), px.y / u.resolution.y); }

// ------------------------------------------------------------------ gradient types
fn typesView(px: vec2f) -> vec3f {
  let cols = 3.0;
  let rows = 2.0;
  let cs = u.resolution / vec2f(cols, rows);
  let cell = min(floor(px / cs), vec2f(cols - 1.0, rows - 1.0));
  let unit = min(cs.x, cs.y);
  let cc = (cell + vec2f(0.5)) * cs;
  let q = (px - cc) / unit;
  let id = i32(cell.y * cols + cell.x);
  let space = i32(u.space);
  let ang = radians(u.angle);

  // the mouse moves the center of the tile it is over
  var m = vec2f(0.0);
  let mcell = min(floor(u.mouse.xy / cs), vec2f(cols - 1.0, rows - 1.0));
  if (u.mouse.w > 0.5 && abs(mcell.x - cell.x) + abs(mcell.y - cell.y) < 0.5) {
    m = clamp((u.mouse.xy - cc) / unit, vec2f(-0.6), vec2f(0.6));
  }

  // Collect up to three (a, b, t) mixes, then run them through ONE mixSpace call site
  // (big inlined functions called from many places make shaders slow to compile and run).
  var ca = array<vec3f, 3>(u.colA, u.colC, u.colA);
  var cb = array<vec3f, 3>(u.colB, vec3f(0.97, 0.95, 0.88), u.colB);
  var ts = array<f32, 3>(0.0, 0.0, 0.0);
  var n = 1;
  if (id == 4) {
    let f = clamp(q * vec2f(unit / (cs.x - 24.0), unit / (cs.y - 24.0)) + vec2f(0.5), vec2f(0.0), vec2f(1.0));
    ts = array<f32, 3>(posterize(f.x), posterize(f.x), posterize(f.y));
    n = 3;
  } else {
    var t: f32;
    let dir = vec2f(cos(ang), sin(ang));
    if (id == 0) { t = spreadT(dot(q, dir) / 0.75 + 0.5); }
    else if (id == 1) { t = spreadT(length(q - m) / 0.4); }
    else if (id == 2) { let w = q - m; t = fract(atan2(w.y, w.x) / TAU - ang / TAU + 1.0); }
    else if (id == 3) { let w = rot2(-ang) * (q - m); t = spreadT((abs(w.x) + abs(w.y)) / 0.42); }
    else { t = spreadT(dot(q, dir) / 0.85 + 0.5); }
    t = posterize(t);
    ts[0] = t;
    if (id == 5) {
      if (t < 0.5) { cb[0] = u.colC; ts[0] = t * 2.0; }
      else { ca[0] = u.colC; ts[0] = t * 2.0 - 1.0; }
    }
  }
  var res = array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for (var k = 0; k < 3; k++) {
    if (k >= n) { break; }
    var a = ca[k];
    var b = cb[k];
    if (k == 2) { a = res[0]; b = res[1]; }
    res[k] = mixSpace(a, b, ts[k], space);
  }
  let col = res[n - 1];
  // rounded card
  let card = sdRoundBox(px - cc, cs * 0.5 - vec2f(8.0), 16.0);
  var c = mix(bg(px), col, cov(card, 1.0));
  c = mix(c, vec3f(1.0), cov(abs(card) - 0.5, 1.0) * 0.12);
  // center marker for the centered gradients
  if (id >= 1 && id <= 3) {
    let md = length(q - m) * unit;
    c = mix(c, vec3f(0.0), cov(abs(md - 5.0) - 2.0, 1.0) * 0.6);
    c = mix(c, vec3f(1.0), cov(abs(md - 5.0) - 1.0, 1.0) * 0.9);
  }
  return c;
}

// ------------------------------------------------------------------ color spaces
fn spacesView(px: vec2f) -> vec3f {
  let res = u.resolution;
  var c = bg(px);
  let x0 = res.x * 0.25;
  let x1 = res.x * 0.81;
  let sx0 = res.x * 0.85;
  let sx1 = res.x * 0.95;
  let hh = res.y * 0.075;
  // which strip is this pixel in?
  let i = i32(clamp(floor((px.y / res.y - 0.13 + 0.1075) / 0.215), 0.0, 3.0));
  let cy = res.y * (0.13 + f32(i) * 0.215);
  if (abs(px.y - cy) > hh + 12.0) { return c; }
  let t = clamp((px.x - x0) / (x1 - x0), 0.0, 1.0);
  let inSwatch = px.x > (x1 + sx0) * 0.5;
  var tt = t;
  if (inSwatch) { tt = 0.5; }
  let sc = mixSpace(u.colA, u.colB, tt, i);
  let box = sdRoundBox(px - vec2f((x0 + x1) * 0.5, cy), vec2f((x1 - x0) * 0.5, hh), 10.0);
  let sw = sdRoundBox(px - vec2f((sx0 + sx1) * 0.5, cy), vec2f((sx1 - sx0) * 0.5, hh), 10.0);
  c = mix(c, sc, cov(min(box, sw), 1.0));
  // lightness curve (OKLab L of the mixed color) and the ideal straight line
  if (u.curve > 0.5 && !inSwatch) {
    var cl = array<vec3f, 3>(sc, u.colA, u.colB);
    var L = array<f32, 3>(0.0, 0.0, 0.0);
    for (var k = 0; k < 3; k++) { L[k] = oklabL(cl[k]); }
    // lightness 0.35..0.95 mapped onto the strip height
    let ly = cy + hh - mix(0.12, 1.88, clamp((L[0] - 0.35) / 0.6, 0.0, 1.0)) * hh;
    let iy = cy + hh - mix(0.12, 1.88, clamp((mix(L[1], L[2], t) - 0.35) / 0.6, 0.0, 1.0)) * hh;
    let inside = cov(box, 1.0);
    let dash = step(0.5, fract(px.x / 10.0));
    c = mix(c, vec3f(0.0), cov(abs(px.y - iy) - 1.0, 1.0) * dash * inside * 0.55);
    c = mix(c, vec3f(1.0), cov(abs(px.y - iy) - 0.5, 1.0) * dash * inside * 0.55);
    c = mix(c, vec3f(0.0), cov(abs(px.y - ly) - 2.0, 1.0) * inside * 0.6);
    c = mix(c, vec3f(1.0), cov(abs(px.y - ly) - 1.0, 1.0) * inside);
  }
  // tick at t = 0.5 on the strip
  let tick = max(abs(px.x - (x0 + x1) * 0.5) - 0.75, abs(px.y - cy) - hh - 6.0);
  c = mix(c, vec3f(0.85), cov(tick, 1.0) * (1.0 - cov(box, 1.0)));
  return c;
}

// ------------------------------------------------------------------ banding
fn ditherValue(px: vec2f, method: i32) -> f32 {
  let seed = px + vec2f(fract(u.time * 7.13) * 117.0, fract(u.time * 3.71) * 71.0) * u.animNoise;
  if (method == 1) { return hash21(seed); }
  if (method == 2) { return ign(seed); }
  if (method == 3) { return bayer4(px + floor(seed - px)); }
  if (method == 4) { return bayer8(px + floor(seed - px)); }
  if (method == 5) { return 0.5 + (hash21(seed) + hash21(seed + vec2f(37.0, 11.0)) - 1.0); }
  return 0.5;
}

fn bandingView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let p = (px - vec2f(res.x * 0.5, res.y * 0.42)) / res.y;
  // a dark radial gradient (night sky / vignette), plus a faint diagonal ramp
  let t = clamp(length(p * vec2f(0.8, 1.0)) / 0.75, 0.0, 1.0);
  let base = mix(u.colA, u.colB, smoothstep(0.0, 1.0, t)) + vec3f(0.006, 0.0, 0.012) * (px.x / res.x);
  let levels = pow(2.0, u.bits) - 1.0;
  var method = 0;
  if (px.x > res.x * 0.5) { method = i32(u.dither); }
  let d = ditherValue(px, method);
  let q = floor(base * levels + vec3f(d)) / levels;
  var c = clamp(q, vec3f(0.0), vec3f(1.0)) * u.boost;
  // divider
  c = mix(c, vec3f(0.85, 0.9, 1.0), cov(abs(px.x - res.x * 0.5) - 0.75, 1.0) * 0.7);
  return c;
}

// ------------------------------------------------------------------ game: sky + health bars
fn hpColor(hp: f32, kind: i32) -> vec3f {
  let red = vec3f(0.9, 0.12, 0.2);
  let yellow = vec3f(1.0, 0.82, 0.2);
  let green = vec3f(0.22, 0.86, 0.36);
  if (kind == 0) { return mix(red, green, hp); }      // naive sRGB lerp
  var a = red;
  var b = green;
  var t = hp;
  if (kind == 2) {                                     // 3 stops: red -> yellow -> green
    if (hp < 0.5) { b = yellow; t = hp * 2.0; } else { a = yellow; t = hp * 2.0 - 1.0; }
  }
  return mixOklab(a, b, t);
}

fn gameView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let uv = px / res;
  let t = u.time;
  // u.zen / u.hor: this frame's sky keyframe colors, already blended in OKLab on the CPU
  let a = u.sky.y;
  let sunUp = u.sky.z;
  let skyH = 0.64;
  let horizonY = 0.5;
  let sy = clamp(uv.y / horizonY, 0.0, 1.0);
  var c = mixOklab(u.zen, u.hor, pow(sy, 1.6));

  // sun & moon on an arc
  let aspect = res.x / res.y;
  let pp = vec2f(uv.x * aspect, uv.y);
  let sunP = vec2f(aspect * (0.5 - 0.38 * cos(a)), horizonY - 0.42 * sin(a));
  let moonP = vec2f(aspect * (0.5 + 0.38 * cos(a)), horizonY + 0.42 * sin(a));
  let night = 1.0 - smoothstep(-0.25, 0.1, sin(a));
  let sd = length(pp - sunP);
  c += vec3f(1.0, 0.7, 0.4) * exp(-sd * 7.0) * 0.45 * sunUp;
  c = mix(c, vec3f(1.0, 0.95, 0.8), cov(sd - 0.045, 1.5 / res.y) * sunUp);
  let md = length(pp - moonP);
  c = mix(c, vec3f(0.92, 0.94, 1.0), cov(max(md - 0.035, -(length(pp - moonP - vec2f(0.015, -0.01)) - 0.03)), 1.5 / res.y) * night);
  // stars
  let sg = floor(px / 3.0);
  let star = step(0.997, hash21(sg)) * (0.6 + 0.4 * sin(t * 3.0 + hash21(sg + vec2f(5.0)) * 40.0));
  c += vec3f(star) * night * (1.0 - sy * 0.8);

  // hills: colors pre-mixed toward the horizon color (aerial perspective)
  let h1 = horizonY - 0.07 - 0.04 * sin(uv.x * 7.0 + 1.0) - 0.025 * sin(uv.x * 17.0);
  let h2 = horizonY + 0.02 - 0.05 * sin(uv.x * 4.0 + 3.0) - 0.02 * sin(uv.x * 23.0 + 1.0);
  c = mix(c, u.hill1, cov(h1 - uv.y, 1.5 / res.y));
  c = mix(c, u.hill2, cov(h2 - uv.y, 1.5 / res.y));
  c = mix(c, vec3f(0.03, 0.05, 0.05) * u.sky.w, cov(skyH - 0.02 - uv.y + 0.03 * sin(uv.x * 9.0), 1.5 / res.y));

  // UI panel
  if (uv.y > skyH) {
    c = mix(vec3f(0.05, 0.055, 0.08), vec3f(0.03, 0.035, 0.05), (uv.y - skyH) / (1.0 - skyH));
    c = mix(c, vec3f(0.3, 0.35, 0.45), cov(abs(px.y - skyH * res.y) - 1.0, 1.0));
  }
  var hp = u.hp;
  if (u.autoHp > 0.5) { hp = 0.5 + 0.5 * cos(t * 0.45 + 1.75); }
  let bx0 = res.x * 0.27;
  let bx1 = res.x * 0.93;
  let bh = res.y * 0.024;
  let bw = (bx1 - bx0) * 0.5;
  // which bar row is this pixel in?
  let i = i32(clamp(floor((uv.y - 0.73 + 0.0475) / 0.095), 0.0, 2.0));
  let cy = res.y * (0.73 + f32(i) * 0.095);
  if (uv.y > skyH && abs(px.y - cy) < bh + res.y * 0.03) {
    let ry = cy + bh + res.y * 0.014;
    let rt = clamp((px.x - bx0) / (bx1 - bx0), 0.0, 1.0);
    let inRamp = px.y > cy + bh + 2.0;
    var ht = hp;
    if (inRamp) { ht = rt; }
    let hc = hpColor(ht, i);
    let box = sdRoundBox(px - vec2f(bx0 + bw, cy), vec2f(bw, bh), bh);
    c = mix(c, vec3f(0.0), cov(box - 3.0, 1.0) * 0.8);
    c = mix(c, vec3f(0.12, 0.1, 0.13), cov(box, 1.0));
    let low = 1.0 - smoothstep(0.15, 0.3, hp);
    let pulse = 1.0 + low * 0.35 * (0.5 + 0.5 * sin(t * 12.0));
    let fill = cov(px.x - (bx0 + 2.0 * bw * hp), 1.0) * cov(box + 2.0, 1.0);
    let gloss = 1.0 + 0.25 * smoothstep(cy + bh * 0.1, cy - bh, px.y);
    c = mix(c, min(hc * gloss * pulse, vec3f(1.0)), fill);
    // the full ramp underneath, with a marker at the current HP
    let ramp = sdRoundBox(px - vec2f(bx0 + bw, ry), vec2f(bw, res.y * 0.005), res.y * 0.005);
    c = mix(c, hc, cov(ramp, 1.0));
    let mk = max(abs(px.x - (bx0 + 2.0 * bw * hp)) - 1.0, abs(px.y - ry) - res.y * 0.011);
    c = mix(c, vec3f(1.0), cov(mk, 1.0));
  }
  return c;
}

fn shade(uv: vec2f, px0: vec2f) -> vec4f {
  let px = uv * u.resolution;   // canvas pixels, even when rendering at reduced resolution
  let ex = i32(u.example);
  var c: vec3f;
  if (ex == 0) { c = typesView(px); }
  else if (ex == 1) { c = spacesView(px); }
  else if (ex == 2) { return vec4f(bandingView(px), 1.0); }
  else { c = gameView(px); }
  // a whisper of noise before the 8-bit framebuffer rounds our colors: no banding
  c += vec3f((ign(px) - 0.5) / 255.0);
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`,
  about: {
    summary: 'Gradients are everywhere in 2D games — skies, UI, lighting, health bars. How you compute the position (t) gives the shape; which color space you mix in decides whether it looks clean or muddy; and dithering decides whether it bands.',
    what: `<p><b>Gradient types</b>: six ways to turn a pixel position into <i>t</i>. <b>sRGB vs linear vs OKLab</b>: the same two colors blended in four color spaces,
      with a lightness graph. <b>Banding</b>: a dark gradient rounded to few levels, with and without dithering. <b>Sky &amp; health bars</b>: the lessons applied in a game.</p>`,
    how: `<ol>
      <li><b>Shape of t</b>: linear = <code>dot(p, direction)</code>, radial = <code>length(p − c)</code>, conic = <code>atan2(p.y, p.x)</code>, diamond = <code>|x| + |y|</code>,
        four-corner = two mixes along x, then one along y (bilinear).</li>
      <li><b>Spread</b>: when <i>t</i> leaves 0..1, <i>pad</i> clamps it, <i>repeat</i> uses <code>fract(t)</code>, <i>reflect</i> ping-pongs.</li>
      <li><b>Color space</b>: sRGB values are <i>gamma-encoded</i> — not proportional to light. Mixing them directly darkens the middle.
        Converting to <b>linear light</b> fixes brightness; <b>OKLab</b> (Björn Ottosson, 2020) is built so equal steps <i>look</i> equal, keeping hue and lightness even.</li>
      <li><b>Banding</b>: an 8-bit channel has only 256 levels. Across a wide dark gradient that is a visible step every few dozen pixels.
        <b>Dithering</b> adds noise smaller than one level before rounding, so the average is correct and the eye sees smooth tone.</li>
    </ol>`,
    uses: [
      { title: 'Skies & ambience', text: 'Day/night cycles blend color keyframes; doing it in OKLab avoids grey, desaturated dusks.' },
      { title: 'UI', text: 'Health/stamina bars, buttons, rarity glows (common → legendary ramps), heat maps.' },
      { title: 'Lighting & fog', text: 'Light falloff and fog are gradients too — they band badly in dark scenes without dithering (common in horror games).' },
      { title: 'Toon ramps', text: 'Posterized gradients give cel-shaded lighting (Zelda: Wind Waker style).' },
    ],
    try: [
      'On <b>sRGB vs linear vs OKLab</b>, pick pure red and pure green: watch the white lightness curve sag in sRGB.',
      'On <b>Gradient types</b>, switch <i>Interpolate in</i> between sRGB and OKLab and look at the four-corner tile’s center.',
      'Set <i>Spread mode</i> to Reflect and <i>Posterize steps</i> to 5 for a toon-ramp look.',
      'On <b>Banding</b>, set <i>Bits</i> to 8 and <i>Contrast boost</i> to 6: real 8-bit banding appears on the left — and vanishes on the right.',
      'On <b>Sky &amp; health bars</b>, turn off <i>Animate HP</i> and set HP to 50%.',
    ],
    ask: [
      'blend gradient colors in OKLab instead of sRGB',
      'a day/night sky gradient with color keyframes',
      'dither the gradients to remove banding',
      'a health bar that fades green → yellow → red',
      'a posterized (toon) lighting ramp',
    ],
    perf: `<p>Gradients are nearly free. OKLab costs a couple of matrix multiplies and cube roots per pixel — negligible on a GPU,
      but if you evaluate a ramp millions of times you can bake it into a small 256×1 lookup texture.</p>`,
    api: `<p>Identical on WebGPU and WebGL2 (one shader, auto-translated). One gotcha in both: if the swap chain or texture is an <code>*-srgb</code> format,
      the GPU converts linear → sRGB on write for you — do the math in linear and don’t encode twice. This scene writes sRGB values to a plain 8-bit canvas.</p>`,
    code: [
      {
        title: 'Mixing in different spaces (from this scene)',
        lang: 'wgsl',
        src: `fn mixSpace(a: vec3f, b: vec3f, t: f32, space: i32) -> vec3f {
  if (space == 0) { return mix(a, b, t); }                       // naive sRGB
  if (space == 1) {                                               // linear light
    return linearToSrgb(mix(srgbToLinear(a), srgbToLinear(b), t));
  }
  // OKLab: convert, mix, convert back
  let la = linearToOklab(srgbToLinear(a));
  let lb = linearToOklab(srgbToLinear(b));
  return linearToSrgb(oklabToLinear(mix(la, lb, t)));
}`,
      },
      {
        title: 'Quantize with and without dithering',
        lang: 'wgsl',
        src: `let levels = pow(2.0, u.bits) - 1.0;
let d = ign(px);                      // 0.5 = plain rounding, noise = dithering
let q = floor(color * levels + vec3f(d)) / levels;`,
      },
    ],
    links: [
      { title: 'Björn Ottosson — A perceptual color space for image processing (OKLab)', url: 'https://bottosson.github.io/posts/oklab/' },
      { title: 'Jorge Jimenez — Interleaved gradient noise (Call of Duty: AW, 2014)', url: 'https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/' },
    ],
  },
});

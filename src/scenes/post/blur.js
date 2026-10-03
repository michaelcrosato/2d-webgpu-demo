import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import { overlays, compareSplit, SPLIT_WGSL, GAME_POS_WGSL, clamp01 } from './_shared.js';

// Blur: box vs separable Gaussian, dual-filter (Kawase) pyramid, radial / zoom / spin blur,
// a pause menu over a blurred game and a tilt-shift "miniature" effect.
//
// Pass graph (passes that the current example doesn't need get 0 iterations = skipped):
//   game ─► d1 ─► d2 ─► d3 ─► d4 ─► d5        Kawase 5-tap downsamples (also used as cheap pre-filtered sources)
//                 u1 ◄─ u2 ◄─ u3 ◄─ u4 ◄─┘    Kawase 8-tap upsamples back up
//   game|d1|d2 ─► blurH ─► blurV              separable box / Gaussian (fixed σ, or σ(y) for tilt-shift)
//   final image: picks the right result, plus radial blur, the pause menu and the before/after split.

const cur = { ex: 'types' };
const st = { split: 0.5, open: 1, openTarget: 1, focus: 0.74, lastFocusParam: null, dashT: -10, lastAuto: 0 };
const ov = overlays();

const sepOn = (p) => cur.ex === 'tiltshift' || (cur.ex === 'types' && (p.method === 'box' || p.method === 'gauss'));
const sepDiv = (p) => (cur.ex === 'tiltshift' ? 2 : +p.workRes || 1);
const kawOn = (p) => cur.ex === 'pause' || (cur.ex === 'types' && p.method === 'kawase');
const kawL = (p) => Math.max(1, Math.min(5, Math.round(p.levels)));
const downIt = (k) => (p) => ((kawOn(p) && k <= kawL(p)) || (sepOn(p) && sepDiv(p) >= 2 ** k) ? 1 : 0);
const upIt = (k) => (p) => (kawOn(p) && k <= kawL(p) - 1 ? 1 : 0);

// ------------------------------------------------------------------------------------------ WGSL
const DOWN = (src, self) => /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  // Kawase / dual-filter downsample: centre + 4 diagonal taps. Each bilinear tap already averages 2×2 texels.
  let h: vec2f = 0.5 / TEXSIZE(${self}) * u.downOff;
  var s: vec3f = TEX(${src}, uv).rgb * 4.0;
  s += TEX(${src}, uv - h).rgb;
  s += TEX(${src}, uv + h).rgb;
  s += TEX(${src}, uv + vec2f(h.x, -h.y)).rgb;
  s += TEX(${src}, uv - vec2f(h.x, -h.y)).rgb;
  return vec4f(s / 8.0, 1.0);
}`;

const UPSAMPLE_FN = /* wgsl */ `
// Kawase / dual-filter upsample: 8 taps on a ring (diagonals weighted 2×).
fn upsample(t: texture_2d<f32>, uv: vec2f, off: f32) -> vec3f {
  let h: vec2f = 0.5 / TEXSIZE(t) * off;
  var s: vec3f = TEX(t, uv + vec2f(-h.x * 2.0, 0.0)).rgb;
  s += TEX(t, uv + vec2f(-h.x, h.y)).rgb * 2.0;
  s += TEX(t, uv + vec2f(0.0, h.y * 2.0)).rgb;
  s += TEX(t, uv + vec2f(h.x, h.y)).rgb * 2.0;
  s += TEX(t, uv + vec2f(h.x * 2.0, 0.0)).rgb;
  s += TEX(t, uv + vec2f(h.x, -h.y)).rgb * 2.0;
  s += TEX(t, uv + vec2f(0.0, -h.y * 2.0)).rgb;
  s += TEX(t, uv + vec2f(-h.x, -h.y)).rgb * 2.0;
  return s / 12.0;
}`;

// up pass at level k reads level k+1: the deepest down level when k+1 == L, else the next up pass
const UP = (k) => /* wgsl */ `
${UPSAMPLE_FN}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  ${k === 4 ? 'return vec4f(upsample(d5, uv, u.offset), 1.0);' : `if (u.kawL < ${k + 1}.5) { return vec4f(upsample(d${k + 1}, uv, u.offset), 1.0); }
  return vec4f(upsample(u${k + 1}, uv, u.offset), 1.0);`}
}`;

const SEP = (self, dirExpr, srcFn) => /* wgsl */ `
${srcFn}
// kernel weight at distance x (texels): box = flat (with a soft last texel), gaussian = bell curve
fn kw(x: f32, sig: f32) -> f32 {
  if (u.kern < 0.5) { return clamp(sig * 1.7320508 + 0.5 - x, 0.0, 1.0); }
  return exp(-0.5 * x * x / (sig * sig));
}
// σ in this pass's texels: fixed, or growing with the distance from the tilt-shift focus band
fn sepSigma(uv: vec2f) -> f32 {
  if (u.tilt > 0.5) {
    let d: f32 = abs(uv.y - u.focus);
    return u.maxBlur * smoothstep(u.band * 0.5, u.band * 0.5 + 0.22, d) / u.sepDiv;
  }
  return u.sigma / u.sepDiv;
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let texel: vec2f = 1.0 / TEXSIZE(${self});
  let dir: vec2f = ${dirExpr};
  let sig: f32 = sepSigma(uv);
  if (sig < 0.25) { return vec4f(srcS(uv), 1.0); }
  let maxK: f32 = select(3.0 * sig, sig * 1.7320508 + 0.5, u.kern < 0.5);
  var sum: vec3f = srcS(uv) * kw(0.0, sig);
  var wsum: f32 = kw(0.0, sig);
  // "linear sampling" trick: one bilinear fetch between texels k and k+1 = both weights at once
  for (var i = 0; i < 64; i++) {
    let k1: f32 = f32(i) * 2.0 + 1.0;
    if (k1 > maxK) { break; }
    let w1: f32 = kw(k1, sig);
    let w2: f32 = kw(k1 + 1.0, sig);
    let ws: f32 = w1 + w2;
    let off: f32 = (k1 * w1 + (k1 + 1.0) * w2) / max(ws, 0.00001);
    sum += (srcS(uv + dir * off) + srcS(uv - dir * off)) * ws;
    wsum += 2.0 * ws;
  }
  return vec4f(sum / wsum, 1.0);
}`;

const SRC_H = /* wgsl */ `
fn srcS(uv: vec2f) -> vec3f {
  if (u.sepDiv > 3.0) { return TEX(d2, uv).rgb; }
  if (u.sepDiv > 1.5) { return TEX(d1, uv).rgb; }
  return TEX(game, uv).rgb;
}`;
const SRC_V = /* wgsl */ `fn srcS(uv: vec2f) -> vec3f { return TEX(blurH, uv).rgb; }`;

const IMAGE = /* wgsl */ `
${SPLIT_WGSL}
${GAME_POS_WGSL}
${UPSAMPLE_FN}

fn kawaseResult(uv: vec2f) -> vec3f {
  if (u.kawL < 1.5) { return upsample(d1, uv, u.offset); }
  return upsample(u1, uv, u.offset);
}

// mode 0 = zoom (towards the centre), 1 = directional (horizontal motion), 2 = spin (around the centre)
fn radialBlur(uv: vec2f, c: vec2f, strength: f32, n: f32, mode: f32, px: vec2f) -> vec3f {
  var acc: vec3f = vec3f(0.0);
  var wsum: f32 = 0.0;
  let j: f32 = ign(px);                 // per-pixel jitter turns banding into fine noise
  let aspect: f32 = u.resolution.x / u.resolution.y;
  for (var i = 0; i < 64; i++) {
    if (f32(i) >= n) { break; }
    let t: f32 = (f32(i) + j) / n;
    let w: f32 = 1.0 - 0.6 * t;
    var q: vec2f = c + (uv - c) * (1.0 - strength * t);
    if (mode > 1.5) {
      var d: vec2f = (uv - c) * vec2f(aspect, 1.0);
      d = rot2(-strength * 1.2 * (t - 0.5)) * d;
      q = c + d / vec2f(aspect, 1.0);
    } else if (mode > 0.5) {
      q = uv - vec2f(strength * 0.35 * t, 0.0);
    }
    acc += TEX(game, q).rgb * w;
    wsum += w;
  }
  return acc / wsum;
}

fn sdRBox(p: vec2f, b: vec2f, r: f32) -> f32 {
  let q: vec2f = abs(p) - b + vec2f(r);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn pauseMenu(uv: vec2f, px: vec2f) -> vec3f {
  let sharp: vec3f = TEX(game, uv).rgb;
  let blurred: vec3f = kawaseResult(uv);
  let o: f32 = u.open;
  // background: blurred, slightly desaturated and dimmed
  var bg: vec3f = mix(blurred, vec3f(luma(blurred)), 0.25);
  bg = bg * (1.0 - u.dim) + vec3f(0.02, 0.03, 0.06) * u.dim;
  var c: vec3f = mix(sharp, bg, smoothstep(0.0, 1.0, o));
  // the panel (pops in with a little overshoot)
  let s: f32 = 0.86 + 0.14 * easeOutBack(clamp(o, 0.0, 1.0));
  let p: vec2f = centerUV(px, u.resolution) / s;
  let hs: vec2f = vec2f(0.25, 0.31);
  let d: f32 = sdRBox(p, hs, 0.035);
  let aa: f32 = 1.5 / u.resolution.y;
  let fade: f32 = smoothstep(0.35, 1.0, o);
  // soft drop shadow
  c = mix(c, c * 0.35, (1.0 - smoothstep(-0.02, 0.06, d - 0.015)) * 0.6 * fade);
  let inside: f32 = (1.0 - smoothstep(-aa, aa, d)) * fade;
  // frosted glass: the blurred image, lifted and tinted, with a vertical sheen
  var glass: vec3f = blurred * 0.55 + vec3f(0.10, 0.12, 0.20);
  glass += vec3f(0.06) * (1.0 - smoothstep(-0.31, 0.2, p.y));
  // hovered menu item highlight
  let local: vec2f = (p + hs) / (2.0 * hs);       // 0..1 inside the panel
  let item: f32 = floor((local.y - 0.305) / 0.14);
  if (item > -0.5 && item < 3.5 && abs(item - u.hoverItem) < 0.5) {
    let iy: f32 = 0.305 + (item + 0.5) * 0.14;
    let hd: f32 = sdRBox(vec2f(p.x, (local.y - iy) * 2.0 * hs.y), vec2f(0.2, 0.038), 0.03);
    glass = mix(glass, vec3f(0.42, 0.45, 0.98), (1.0 - smoothstep(-aa, aa, hd)) * 0.75);
  }
  // crisp UI text from a Canvas2D texture (alpha = glyph coverage)
  let txt: vec4f = TEX(menuTex, local);
  glass = mix(glass, txt.rgb, txt.a);
  c = mix(c, glass, inside);
  // 1px light border
  c = mix(c, vec3f(0.85, 0.9, 1.0), (1.0 - smoothstep(0.0, aa * 1.2, abs(d))) * 0.45 * fade);
  return c;
}

fn speedLines(uv: vec2f, c: vec2f, amt: f32) -> f32 {
  let d: vec2f = (uv - c) * vec2f(u.resolution.x / u.resolution.y, 1.0);
  let a: f32 = atan2(d.y, d.x) / TAU + 0.5;
  let r: f32 = length(d);
  let n: f32 = 140.0;
  let id: f32 = floor(a * n);
  let tick: f32 = floor(u.time * 18.0);
  let h: f32 = hash21(vec2f(id, tick));
  let f: f32 = abs(fract(a * n) - 0.5) * 2.0;
  let line: f32 = (1.0 - smoothstep(0.1, 0.7, f)) * step(0.72, h);
  let start: f32 = 0.30 + 0.25 * hash21(vec2f(id, tick + 7.0));
  return line * smoothstep(start, start + 0.15, r) * amt;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex: i32 = i32(u.example);
  let orig: vec3f = TEX(game, uv).rgb;
  var c: vec3f = orig;

  if (ex == 0) {
    // Blur types
    if (u.method < 1.5) { c = TEX(blurV, uv).rgb; }
    else if (u.method < 2.5) { c = kawaseResult(uv); }
    else { c = radialBlur(uv, vec2f(0.5, 0.5), u.strength, u.samples, 0.0, px); }
  } else if (ex == 1) {
    c = pauseMenu(uv, px);
  } else if (ex == 2) {
    // Tilt-shift: blurred far from the focus band, sharp inside it, then a toy-like color boost
    let d: f32 = abs(uv.y - u.focus);
    let sig: f32 = u.maxBlur * smoothstep(u.band * 0.5, u.band * 0.5 + 0.22, d);
    c = mix(orig, TEX(blurV, uv).rgb, smoothstep(0.3, 1.6, sig));
    c = adjustSaturation(c, 1.0 + u.sat);
    c = adjustContrast(c, 1.0 + u.sat * 0.35);
    c = clamp(c, vec3f(0.0), vec3f(1.0));
    if (u.mouse.z > 0.5) {
      // focus band guides while dragging
      let e: f32 = min(abs(uv.y - (u.focus - u.band * 0.5)), abs(uv.y - (u.focus + u.band * 0.5))) * u.resolution.y;
      let dash: f32 = step(0.5, fract(px.x / 14.0));
      c = mix(c, vec3f(1.0, 0.9, 0.4), (1.0 - smoothstep(0.5, 1.5, e)) * dash * 0.85);
    }
  } else {
    // Speed / zoom blur centred on the hero
    let hp: vec2f = heroP();
    let center: vec2f = vec2f(hp.x * u.resolution.y / u.resolution.x, hp.y - 0.04);
    let amt: f32 = u.dashAmt;
    if (amt > 0.001) {
      c = radialBlur(uv, center, u.strength * amt, u.samples, u.zoomMode, px);
      // keep the hero itself crisp: blend the sharp image back in near the centre
      let dc: f32 = length((uv - center) * vec2f(u.resolution.x / u.resolution.y, 1.0));
      c = mix(orig, c, smoothstep(0.03, 0.16, dc));
    }
    if (u.zoomMode < 0.5 && u.lines > 0.5) {
      c = mix(c, vec3f(1.0), speedLines(uv, center, smoothstep(0.15, 0.8, amt)) * 0.8);
    }
    c = mix(c, c * vec3f(0.85, 0.9, 1.15) + vec3f(0.02, 0.03, 0.06), amt * 0.35);
  }

  c = splitView(c, orig, px, u.split);
  return vec4f(c, 1.0);
}`;

// ------------------------------------------------------------------------------------------ UI texture
function menuCanvas() {
  const S = 512;
  const cv = makeCanvas(S, S);
  const g = cv.getContext('2d');
  g.clearRect(0, 0, S, S);
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = '#ffffff';
  g.font = '800 70px system-ui, "Segoe UI", Roboto, sans-serif';
  g.fillText('PAUSED', S / 2, S * 0.15);
  g.fillStyle = '#ffffff66';
  g.fillRect(S * 0.22, S * 0.245, S * 0.56, 3);
  const items = ['Resume', 'Settings', 'Controls', 'Quit to title'];
  g.font = '600 38px system-ui, "Segoe UI", Roboto, sans-serif';
  items.forEach((t, i) => {
    g.fillStyle = i === 3 ? '#ffb3b3' : '#f2f5ff';
    g.fillText(t, S / 2, S * (0.305 + (i + 0.5) * 0.14));
  });
  g.font = '500 22px system-ui, sans-serif';
  g.fillStyle = '#ffffff88';
  g.fillText('click Resume · click anywhere to pause', S / 2, S * 0.93);
  return cv;
}

// panel geometry (must match pauseMenu() in WGSL): centred, half size (0.25, 0.31) in height units
function hoveredItem(ctx) {
  const p = ctx.pointer;
  if (!p.over) return -1;
  const x = (p.x - ctx.width / 2) / ctx.height;
  const y = (p.y - ctx.height / 2) / ctx.height;
  if (Math.abs(x) > 0.25 || Math.abs(y) > 0.31) return -1;
  const ly = (y + 0.31) / 0.62;
  const item = Math.floor((ly - 0.305) / 0.14);
  return item >= 0 && item <= 3 ? item : -1;
}

function fetchInfo(p) {
  const m = p.method;
  if (cur.ex === 'types' && (m === 'box' || m === 'gauss')) {
    const div = sepDiv(p);
    const sig = p.sigma / div;
    const maxK = m === 'box' ? sig * Math.sqrt(3) + 0.5 : 3 * sig;
    const n = Math.min(64, Math.max(0, Math.ceil((maxK - 1) / 2 + 0.001)));
    const taps = 1 + 2 * n;
    const reach = Math.ceil(m === 'box' ? p.sigma * Math.sqrt(3) : 3 * p.sigma);
    const brute = (2 * reach + 1) ** 2;
    const res = div === 1 ? 'full res' : div === 2 ? '½ res' : '¼ res';
    return `${m === 'box' ? 'Box' : 'Gaussian'} · 2 passes × ${taps} fetches at ${res}<br>brute-force 2D kernel: ${brute.toLocaleString()} fetches/pixel`;
  }
  if (cur.ex === 'types' && m === 'radial') return `Radial: ${Math.round(p.samples)} fetches along the line to the centre`;
  if (cur.ex === 'tiltshift') return 'Separable Gaussian at ½ res · σ grows with distance from the focus band';
  if (cur.ex === 'zoom') return `${Math.round(p.samples)} fetches per pixel along the blur path`;
  const L = kawL(p);
  let total = 8;
  for (let k = 1; k <= L; k++) total += 5 * 4 ** -k;
  for (let k = 1; k < L; k++) total += 8 * 4 ** -k;
  return `Dual Kawase · ${L} level${L > 1 ? 's' : ''} · ≈${total.toFixed(1)} fetches per screen pixel<br>blur reach ≈ ${Math.round(2 ** L * p.offset * 1.5)} px`;
}

export default shaderScene({
  interaction: 'Move the mouse to slide the before/after divider.',
  examples: [
    {
      id: 'types',
      label: 'Blur types',
      kind: 'Comparison',
      note: 'Pick a method and slide the divider. <b>Box</b> and <b>Gaussian</b> have the same strength here — notice how the box turns the sun and torches into squares. <b>Dual Kawase</b> gets a huge blur from a handful of samples; <b>Radial</b> blurs towards a point.',
      params: { compare: true, method: 'gauss', sigma: 8, workRes: 2, levels: 4, offset: 1.5, strength: 0.15, samples: 24 },
      hint: 'Move the mouse to slide the before/after divider.',
    },
    {
      id: 'pause',
      label: 'Pause menu',
      kind: 'In a game',
      note: 'The classic “frosted glass” pause screen: the game keeps rendering underneath, gets a cheap Kawase blur and a dim, and a crisp UI panel sits on top. Click <b>Resume</b> to close it, click anywhere to pause again.',
      params: { levels: 4, offset: 1.5, dim: 0.45 },
      hint: 'Click “Resume” to close the menu, click anywhere to open it again.',
    },
    {
      id: 'tiltshift',
      label: 'Tilt-shift miniature',
      kind: 'Real life',
      note: 'Photographers tilt the lens so only a thin strip is in focus — our brains read that shallow depth of field as “tiny model”. Blur grows with distance from the focus band; extra saturation sells the toy look.',
      params: { focus: 0.74, band: 0.12, maxBlur: 18, sat: 0.35 },
      hint: 'Drag up/down on the canvas to move the focus band.',
    },
    {
      id: 'zoom',
      label: 'Speed / zoom blur',
      kind: 'In a game',
      note: 'A dash or boost: everything streaks away from the hero for a fraction of a second. Click to dash. Try the directional (motion) and spin variants too.',
      params: { strength: 0.22, samples: 24, zoomMode: 'zoom', auto: true, lines: true },
      hint: 'Click to dash.',
    },
  ],
  controls: [
    {
      type: 'select',
      key: 'method',
      label: 'Blur method',
      value: 'gauss',
      options: [
        { value: 'box', label: 'Box (flat weights, separable)' },
        { value: 'gauss', label: 'Gaussian (separable, 2 passes)' },
        { value: 'kawase', label: 'Dual Kawase (down/up pyramid)' },
        { value: 'radial', label: 'Radial / zoom' },
      ],
      showFor: ['types'],
      help: 'All four are real techniques used in shipped games.',
    },
    { type: 'slider', key: 'sigma', label: 'Blur size σ', min: 0.5, max: 40, step: 0.5, value: 8, unit: 'px', showFor: ['types'], help: 'Box & Gaussian. Same variance for both, so differences are purely the kernel shape.' },
    {
      type: 'select',
      key: 'workRes',
      label: 'Work resolution',
      value: 2,
      options: [
        { value: 1, label: 'Full' },
        { value: 2, label: 'Half (¼ of the pixels)' },
        { value: 4, label: 'Quarter (1/16 of the pixels)' },
      ],
      showFor: ['types'],
      help: 'Box & Gaussian. Blur at low res, then upscale — big savings, but small blurs get blocky.',
    },
    { type: 'slider', key: 'levels', label: 'Pyramid levels', min: 1, max: 5, step: 1, value: 4, showFor: ['types', 'pause'], help: 'Dual Kawase. Each level halves the resolution and roughly doubles the blur.' },
    { type: 'slider', key: 'offset', label: 'Tap spread', min: 0.5, max: 3, step: 0.05, value: 1.5, showFor: ['types', 'pause'], help: 'Dual Kawase. Spreads the taps further apart: smoother & wider, until it starts to ghost.' },
    { type: 'slider', key: 'strength', label: 'Radial strength', min: 0, max: 0.5, step: 0.005, value: 0.15, showFor: ['types', 'zoom'], help: 'How far along the line towards the centre the samples reach.' },
    { type: 'slider', key: 'samples', label: 'Samples', min: 4, max: 64, step: 1, value: 24, showFor: ['types', 'zoom'], help: 'Radial. Fewer = cheaper but visible copies (a little per-pixel jitter hides it).' },
    { type: 'toggle', key: 'compare', label: 'Before / after divider', value: true, showFor: ['types'] },
    { type: 'slider', key: 'dim', label: 'Dim background', min: 0, max: 0.9, step: 0.01, value: 0.45, showFor: ['pause'], help: 'Darken the blurred game so the UI pops.' },
    { type: 'button', key: 'toggleMenu', label: 'Open / close menu', primary: true, showFor: ['pause'] },
    { type: 'slider', key: 'focus', label: 'Focus height', min: 0.05, max: 0.95, step: 0.005, value: 0.74, showFor: ['tiltshift'], help: 'Where the sharp band sits (or drag on the canvas).' },
    { type: 'slider', key: 'band', label: 'Focus band size', min: 0, max: 0.6, step: 0.005, value: 0.12, showFor: ['tiltshift'], help: 'Height of the sharp strip, as a fraction of the screen.' },
    { type: 'slider', key: 'maxBlur', label: 'Max blur σ', min: 0, max: 30, step: 0.5, value: 18, unit: 'px', showFor: ['tiltshift'], help: 'Blur at the top and bottom of the screen.' },
    { type: 'slider', key: 'sat', label: 'Toy color boost', min: 0, max: 1, step: 0.01, value: 0.35, showFor: ['tiltshift'], help: 'Extra saturation & contrast — real miniatures are painted brightly.' },
    {
      type: 'select',
      key: 'zoomMode',
      label: 'Blur shape',
      value: 'zoom',
      options: [
        { value: 'zoom', label: 'Zoom (radial, towards the hero)' },
        { value: 'motion', label: 'Directional (horizontal motion)' },
        { value: 'spin', label: 'Spin (rotational)' },
      ],
      showFor: ['zoom'],
    },
    { type: 'toggle', key: 'auto', label: 'Auto-dash every 2 s', value: true, showFor: ['zoom'] },
    { type: 'toggle', key: 'lines', label: 'Anime speed lines', value: true, showFor: ['zoom'], help: 'Zoom shape only.' },
  ],
  uniforms: {
    method: 'f32',
    sigma: 'f32',
    levels: 'f32',
    offset: 'f32',
    strength: 'f32',
    samples: 'f32',
    dim: 'f32',
    focus: 'f32',
    band: 'f32',
    maxBlur: 'f32',
    sat: 'f32',
    zoomMode: 'f32',
    lines: 'f32',
    // derived in bind()
    kern: 'f32',
    sepDiv: 'f32',
    tilt: 'f32',
    downOff: 'f32',
    kawL: 'f32',
    split: 'f32',
    open: 'f32',
    hoverItem: 'f32',
    dashAmt: 'f32',
  },
  include: ['math', 'hash', 'color'],
  input: 'game',
  textures: { menuTex: { source: async () => menuCanvas() } },
  passes: [
    { name: 'd1', scale: 0.5, iterations: downIt(1), code: DOWN('game', 'd1') },
    { name: 'd2', scale: 0.25, iterations: downIt(2), code: DOWN('d1', 'd2') },
    { name: 'd3', scale: 0.125, iterations: downIt(3), code: DOWN('d2', 'd3') },
    { name: 'd4', scale: 0.0625, iterations: downIt(4), code: DOWN('d3', 'd4') },
    { name: 'd5', scale: 0.03125, iterations: downIt(5), code: DOWN('d4', 'd5') },
    { name: 'u4', scale: 0.0625, iterations: upIt(4), code: UP(4) },
    { name: 'u3', scale: 0.125, iterations: upIt(3), code: UP(3) },
    { name: 'u2', scale: 0.25, iterations: upIt(2), code: UP(2) },
    { name: 'u1', scale: 0.5, iterations: upIt(1), code: UP(1) },
    { name: 'blurH', scale: (p) => 1 / sepDiv(p), iterations: (p) => (sepOn(p) ? 1 : 0), code: SEP('blurH', 'vec2f(texel.x, 0.0)', SRC_H) },
    { name: 'blurV', scale: (p) => 1 / sepDiv(p), iterations: (p) => (sepOn(p) ? 1 : 0), code: SEP('blurV', 'vec2f(0.0, texel.y)', SRC_V) },
  ],
  resetOnExample: false,
  onAction(key) {
    if (key === 'toggleMenu') st.openTarget = st.openTarget > 0.5 ? 0 : 1;
  },
  bind(p, ctx) {
    cur.ex = ctx.example;
    ov.begin(ctx);
    const dt = ctx.dt;
    const out = {
      kern: p.method === 'box' && ctx.example === 'types' ? 0 : 1,
      sepDiv: sepDiv(p),
      tilt: ctx.example === 'tiltshift' ? 1 : 0,
      downOff: kawOn(p) ? p.offset : 1,
      kawL: kawL(p),
      split: compareSplit(st, ctx, ctx.example === 'types' && p.compare, ov, ['Original', 'Blurred']),
      hoverItem: -1,
      dashAmt: 0,
    };
    if (ctx.example === 'pause') {
      const hov = hoveredItem(ctx);
      out.hoverItem = st.openTarget > 0.5 ? hov : -1;
      if (ctx.pointer.clicked) {
        if (st.openTarget < 0.5) st.openTarget = 1;
        else if (hov === 0) st.openTarget = 0;
      }
      st.open += (st.openTarget - st.open) * clamp01((ctx.paused ? 1 : dt) * 7);
      if (Math.abs(st.open - st.openTarget) < 0.002) st.open = st.openTarget;
    }
    out.open = st.open;
    if (ctx.example === 'tiltshift') {
      if (p.focus !== st.lastFocusParam) {
        st.focus = p.focus;
        st.lastFocusParam = p.focus;
      }
      if (ctx.pointer.down && ctx.pointer.over) st.focus = Math.min(0.97, Math.max(0.03, ctx.pointer.ny));
      out.focus = st.focus;
    }
    if (ctx.example === 'zoom') {
      const t = ctx.time;
      if (ctx.pointer.clicked) st.dashT = t;
      if (p.auto && t - st.dashT > 2.0) st.dashT = t;
      const age = t - st.dashT;
      out.dashAmt = age < 0 ? 0 : Math.min(1, age / 0.06) * Math.exp(-Math.max(0, age - 0.06) * 2.6);
    }
    ov.show('cost', fetchInfo(p), 'right:8px;bottom:8px;text-align:right');
    ov.end();
    return out;
  },
  code: IMAGE,
  about: {
    summary:
      'Blur is the workhorse of post-processing: depth of field, bloom, frosted-glass menus, motion and speed effects all start here. The trick is getting a big, smooth blur from as few texture reads as possible.',
    what: `<p>The live game scene, blurred four ways. <b>Blur types</b> compares the classic algorithms with a before/after divider and a cost readout (bottom right).
      <b>Pause menu</b> is the frosted-glass UI backdrop, <b>Tilt-shift</b> fakes a miniature photo by blurring away from a focus band,
      and <b>Speed / zoom blur</b> is the “dash” effect from action games.</p>`,
    how: `<ol>
      <li><b>A blur is a weighted average of neighbours.</b> The weights are the <i>kernel</i>. A <b>box</b> kernel weights every neighbour equally; a <b>Gaussian</b> uses a bell curve, so far neighbours count less — no hard edges, no square “bokeh”.</li>
      <li><b>Separable</b>: a 2D Gaussian (or box) equals a horizontal 1D blur followed by a vertical one. A 49×49 kernel costs 2,401 reads per pixel; two 1D passes cost 98.</li>
      <li><b>Linear-sampling trick</b>: the GPU’s bilinear filter blends two texels for free, so one fetch placed between texels <i>k</i> and <i>k+1</i> gives both weights → half the fetches again.</li>
      <li><b>Downsample first</b>: blurring a half-resolution copy touches ¼ of the pixels, and a σ of 8 there is a σ of 16 on screen.</li>
      <li><b>Dual Kawase</b> (dual filter) goes all the way: repeatedly halve the image with a 5-tap filter, then double it back up with an 8-tap filter. Each level doubles the reach, the total cost is ~10–15 reads per screen pixel <i>whatever the radius</i>. This is what most engines use for UI blur and bloom.</li>
      <li><b>Radial / zoom</b> blur averages samples along the line from the pixel to a centre point; <b>spin</b> along a circle around it; <b>directional</b> along a fixed vector (a cheap motion blur).</li>
      <li><b>Tilt-shift</b> varies σ with the distance from a horizontal focus band. Because σ only depends on <i>y</i>, the separable trick still works.</li>
    </ol>`,
    uses: [
      { title: 'Pause & inventory screens', text: 'Frosted-glass backdrops (iOS, Windows Acrylic, countless console menus) — blur + dim + crisp panel.' },
      { title: 'Depth of field', text: 'Blur distant parallax layers; Octopath Traveler’s “HD-2D” look leans heavily on tilt-shift style DOF.' },
      { title: 'Bloom & glow', text: 'Blur the bright parts and add them back — every bloom implementation is a blur pyramid underneath.' },
      { title: 'Speed & impact', text: 'Zoom blur on dashes, boosts and big hits; spin blur on spin attacks and portals.' },
      { title: 'Focus & hiding', text: 'Blur the world behind dialogue, or blur out spoilers / unexplored areas of a map.' },
    ],
    try: [
      'On <b>Blur types</b> pick <i>Box</i>, then <i>Gaussian</i>: same blur strength, but the sun and torches become squares with the box.',
      'Set <i>Blur size</i> to 40 and compare <i>Full</i> vs <i>Quarter</i> work resolution in the cost readout — then set it to 2 and see Quarter go blocky.',
      'Pick <i>Dual Kawase</i> and step <i>Pyramid levels</i> 1→5: the blur doubles each time while the cost barely moves.',
      'On <b>Speed / zoom blur</b> switch the shape to <i>Spin</i> and set <i>Samples</i> to 6 — you can see the individual copies; that is why we jitter.',
      'On <b>Tilt-shift</b> drag the focus band onto the hero and widen <i>Max blur</i>.',
    ],
    ask: [
      'a frosted-glass pause menu with a dual Kawase blur behind it',
      'a separable Gaussian blur at half resolution',
      'tilt-shift depth of field with a draggable focus band',
      'a zoom blur with speed lines when the player dashes',
      'blur the background parallax layers for depth of field',
    ],
    perf: `<p>Cost = pixels × texture reads. A brute-force 2D kernel explodes with the radius (r² reads); separable is 2r; with the linear trick r; at half
      resolution r/4. Dual Kawase is almost constant (~12 reads per screen pixel) for any radius — the right default for big blurs and UI.
      Radial blur is N reads per pixel; 16–32 with jitter is plenty. Everything here runs comfortably at 60 fps at 1080p on integrated GPUs.</p>`,
    api: `<p>Pure fragment-shader passes, so WebGPU and WebGL2 are identical here (switch the API to check). With WebGPU a <b>compute shader</b> can do
      even better for very large Gaussians: load a row of pixels into fast <i>workgroup shared memory</i> once, then every thread reads its neighbours
      from there instead of from the texture.</p>`,
    code: [
      {
        title: 'Separable Gaussian with the linear-sampling trick (one pass; run it horizontally, then vertically)',
        lang: 'wgsl',
        src: `var sum = srcS(uv) * kw(0.0, sig);
var wsum = kw(0.0, sig);
for (var i = 0; i < 64; i++) {
  let k1 = f32(i) * 2.0 + 1.0;            // texels k1 and k1+1 ...
  if (k1 > 3.0 * sig) { break; }
  let w1 = kw(k1, sig);
  let w2 = kw(k1 + 1.0, sig);
  let ws = w1 + w2;                        // ... read with ONE bilinear fetch
  let off = (k1 * w1 + (k1 + 1.0) * w2) / ws;
  sum += (srcS(uv + dir * off) + srcS(uv - dir * off)) * ws;
  wsum += 2.0 * ws;
}
return vec4f(sum / wsum, 1.0);`,
      },
      {
        title: 'Dual Kawase: 5-tap down, 8-tap up',
        lang: 'wgsl',
        src: `// downsample (output is half the size of src)
let h = 0.5 / TEXSIZE(dst) * offset;
var s = TEX(src, uv).rgb * 4.0;
s += TEX(src, uv - h).rgb + TEX(src, uv + h).rgb;
s += TEX(src, uv + vec2f(h.x, -h.y)).rgb + TEX(src, uv - vec2f(h.x, -h.y)).rgb;
return vec4f(s / 8.0, 1.0);

// upsample (output is twice the size of src)
let h = 0.5 / TEXSIZE(src) * offset;
var s = TEX(src, uv + vec2f(-2.0 * h.x, 0.0)).rgb + TEX(src, uv + vec2f(2.0 * h.x, 0.0)).rgb
      + TEX(src, uv + vec2f(0.0, -2.0 * h.y)).rgb + TEX(src, uv + vec2f(0.0, 2.0 * h.y)).rgb;
s += 2.0 * (TEX(src, uv + h).rgb + TEX(src, uv - h).rgb
          + TEX(src, uv + vec2f(h.x, -h.y)).rgb + TEX(src, uv - vec2f(h.x, -h.y)).rgb);
return vec4f(s / 12.0, 1.0);`,
      },
      {
        title: 'Radial (zoom) blur',
        lang: 'wgsl',
        src: `let j = ign(px);                          // per-pixel jitter
for (var i = 0; i < 64; i++) {
  if (f32(i) >= n) { break; }
  let t = (f32(i) + j) / n;
  let q = center + (uv - center) * (1.0 - strength * t);
  acc += TEX(game, q).rgb * (1.0 - 0.6 * t);
  wsum += 1.0 - 0.6 * t;
}`,
      },
    ],
    links: [
      { title: 'Bandwidth-efficient rendering (Marius Bjørge, SIGGRAPH 2015)', url: 'https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_slides.pdf', note: 'the dual Kawase / dual filter blur' },
      { title: 'Efficient Gaussian blur with linear sampling', url: 'https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/', note: 'the bilinear trick' },
    ],
  },
});

import { shaderScene } from '../../core/shaderscene.js';
import { overlays, compareSplit, SPLIT_WGSL, gamePass, withGameIncludes, clamp01 } from './_shared.js';

// Color grading: every color-only adjustment is baked each frame into a 32×32×32 LUT (a 1024×32 strip),
// which the final pass applies with two texture reads. Spatial effects (vignette, grain, letterbox, glow,
// wobble…) can't live in a LUT, so they are done afterwards in the final pass.

const st = { split: 0.5, weave: [0, 0], weaveT: -1, flick: 1, flickT: -1 };
const ov = overlays();
const cur = { ex: 'grading' };

const hex = (h) => {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};
const G = [0.5, 0.5, 0.5];

// A "look" = color part (goes into the LUT) + spatial part (done per pixel afterwards).
const NEUTRAL = {
  exp: 0, con: 1, sat: 1, vib: 0, temp: 0, tint: 0, hue: 0, lift: [0, 0, 0], gamma: [1, 1, 1], gain: [1, 1, 1],
  shadow: G, high: G, split: 0, tone: 0,
  vig: 0, vigCol: [0, 0, 0], grain: 0, grainSize: 1.5, box: 0, weave: 0, hal: 0, halCol: [1, 0.45, 0.2], flicker: 0,
  wobble: 0, caustics: 0, scan: 0, noise: 0, goggles: 0, pulse: 0, edgeBlur: 0, dream: 0,
};
const P = (o) => ({ ...NEUTRAL, ...o });
const PRESETS = {
  none: NEUTRAL,
  horror: P({ exp: -0.45, con: 1.3, sat: 0.4, temp: -0.25, tint: -0.25, lift: [-0.03, -0.02, -0.02], shadow: hex('#4f8a7c'), high: hex('#a8b08a'), split: 0.6, vig: 0.85, grain: 0.3, flicker: 0.35, pulse: 0.6 }),
  sunset: P({ exp: 0.1, con: 1.1, sat: 1.15, vib: 0.35, temp: 0.65, tint: 0.12, shadow: hex('#8a5f9e'), high: hex('#ffb066'), split: 0.65, tone: 2, vig: 0.35, vigCol: [0.25, 0.08, 0.1], hal: 0.55, halCol: [1, 0.55, 0.25] }),
  underwater: P({ con: 0.92, sat: 0.9, temp: -0.55, tint: -0.2, lift: [0.0, 0.05, 0.1], gain: [0.55, 0.95, 1.05], shadow: hex('#1d5a78'), high: hex('#9fe6e0'), split: 0.45, vig: 0.6, vigCol: [0.0, 0.08, 0.16], wobble: 1, caustics: 1 }),
  poison: P({ con: 1.1, sat: 1.15, tint: -0.7, hue: -0.15, shadow: hex('#6c3f8f'), high: hex('#b5e35a'), split: 0.8, vig: 0.7, vigCol: [0.12, 0.3, 0.02], wobble: 0.55, pulse: 1 }),
  flashback: P({ con: 0.88, sat: 0.0, lift: [0.06, 0.05, 0.04], shadow: hex('#7a5a3c'), high: hex('#e8cfa0'), split: 1, vig: 0.75, vigCol: [0.12, 0.07, 0.03], grain: 0.35, grainSize: 1.6, flicker: 0.25, edgeBlur: 1, dream: 0.2 }),
  nightvision: P({ exp: 0.25, con: 1.4, sat: 0, lift: [0.02, 0.02, 0.02], shadow: hex('#3cff5a'), high: hex('#b8ffb0'), split: 1, tone: 2, vig: 0.5, hal: 0.8, halCol: [0.35, 1, 0.4], noise: 0.55, scan: 1, goggles: 1 }),
  winter: P({ exp: 0.1, con: 0.95, sat: 0.7, temp: -0.55, lift: [0.03, 0.05, 0.09], shadow: hex('#4f6aa0'), high: hex('#dfe8ff'), split: 0.5, vig: 0.25, vigCol: [0.7, 0.8, 1], dream: 0.15 }),
  noir: P({ con: 1.55, sat: 0, exp: -0.1, vig: 0.7, grain: 0.3, grainSize: 1.2 }),
  vaporwave: P({ con: 1.05, sat: 1.35, temp: -0.15, tint: 0.45, hue: 0.35, shadow: hex('#5a3fb0'), high: hex('#ff8ad8'), split: 0.85, hal: 0.4, halCol: [1, 0.4, 0.9], vig: 0.3, vigCol: [0.15, 0.0, 0.25] }),
  bleach: P({ exp: 0.15, con: 1.4, sat: 0.4, tone: 2, grain: 0.15 }),
  dream: P({ con: 0.78, sat: 0.85, lift: [0.09, 0.07, 0.11], high: hex('#ffe3f5'), shadow: hex('#8e8ac8'), split: 0.45, dream: 0.75, edgeBlur: 0.6, vig: 0.4, vigCol: [1, 0.95, 1] }),
};
const STOCKS = {
  none: NEUTRAL,
  kodak: P({ con: 1.08, sat: 1.05, temp: 0.25, lift: [0.02, 0.015, 0.0], shadow: hex('#6a6070'), high: hex('#ffd29a'), split: 0.45, tone: 3 }),
  teal: P({ con: 1.15, sat: 1.1, shadow: hex('#2f7f8f'), high: hex('#ffad6b'), split: 0.75, tone: 2 }),
  bleach: P({ exp: 0.1, con: 1.35, sat: 0.45, tone: 2 }),
  bw: P({ con: 1.25, sat: 0, tone: 3 }),
};

const lerp = (a, b, t) => (Array.isArray(a) ? a.map((v, i) => v + (b[i] - v) * t) : a + (b - a) * t);
function mixLook(look, t) {
  const o = {};
  for (const k of Object.keys(NEUTRAL)) o[k] = k === 'tone' || k === 'goggles' ? (t > 0.5 ? look[k] : 0) : lerp(NEUTRAL[k], look[k], t);
  if (look.goggles) o.goggles = t;
  if (look.tone) o.tone = look.tone;
  return o;
}

const TONE_OPTS = ['none', 'reinhard', 'aces', 'hable'];

function lookFromSliders(p) {
  const l = { ...NEUTRAL };
  l.exp = p.exposure;
  l.con = p.contrast;
  l.sat = p.saturation;
  l.vib = p.vibrance;
  l.temp = p.temperature;
  l.tint = p.tint;
  l.hue = (p.hue * Math.PI) / 180;
  l.lift = [p.lift, p.lift, p.lift];
  l.gamma = [p.gamma, p.gamma, p.gamma];
  l.gain = [p.gain, p.gain, p.gain];
  l.shadow = hex(p.shadowTint);
  l.high = hex(p.highTint);
  l.split = p.splitAmt;
  l.tone = Math.max(0, TONE_OPTS.indexOf(p.tonemap));
  return l;
}

// ------------------------------------------------------------------------------------------ WGSL
const GRADE_WGSL = /* wgsl */ `
fn hableF(x: vec3f) -> vec3f {
  let A: f32 = 0.15; let B: f32 = 0.50; let C: f32 = 0.10; let D: f32 = 0.20; let E: f32 = 0.02; let F: f32 = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
fn toneMap(x: vec3f, mode: f32) -> vec3f {
  if (mode < 0.5) { return clamp(x, vec3f(0.0), vec3f(1.0)); }       // none: hard clip
  if (mode < 1.5) { return x * (vec3f(1.0) + x / 16.0) / (vec3f(1.0) + x); }   // Reinhard (white point 4)
  if (mode < 2.5) { return tonemapACES(x); }                           // ACES (Narkowicz fit)
  return hableF(x * 2.0) / hableF(vec3f(11.2));                        // Uncharted 2 filmic (Hable)
}
fn tintLum(c: vec3f, t: vec3f) -> vec3f { return c * t / max(luma(t), 0.001); }

// The whole color grade, applied to one input color. This runs 32,768 times per frame (once per LUT cell).
fn grade(c0: vec3f) -> vec3f {
  // 1) linear light + exposure (in stops)
  var lin: vec3f = srgbToLinear(c0) * exp2(u.gExp);
  // 2) white balance: temperature (blue <-> orange), tint (green <-> magenta), brightness-neutral
  var wb: vec3f = vec3f(1.0 + 0.25 * u.gTemp, 1.0 - 0.2 * u.gTint, 1.0 - 0.3 * u.gTemp);
  wb = wb / luma(wb);
  lin = lin * wb;
  // 3) contrast around middle grey, in log space (like a film curve)
  lin = 0.18 * pow(max(lin, vec3f(0.0)) / 0.18, vec3f(u.gCon));
  // 4) tone mapping: squeeze HDR (> 1.0) into the displayable range
  var c: vec3f = linearToSrgb(toneMap(lin, u.gTone));
  // 5) lift / gamma / gain (shadows / midtones / highlights)
  c = u.gGain * (c + u.gLift * (vec3f(1.0) - c));
  c = pow(max(c, vec3f(0.0)), vec3f(1.0) / u.gGamma);
  // 6) saturation, then vibrance (boosts dull colors more than already-saturated ones)
  c = mix(vec3f(luma(c)), c, u.gSat);
  let chroma: f32 = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
  c = mix(vec3f(luma(c)), c, 1.0 + u.gVib * (1.0 - clamp(chroma, 0.0, 1.0)));
  // 7) hue rotation
  c = hueRotate(c, u.gHue);
  // 8) split toning: one color in the shadows, another in the highlights
  let l: f32 = clamp(luma(c), 0.0, 1.0);
  c = mix(c, tintLum(c, u.gShadow), (1.0 - smoothstep(0.05, 0.6, l)) * u.gSplit);
  c = mix(c, tintLum(c, u.gHigh), smoothstep(0.4, 0.95, l) * u.gSplit);
  return clamp(c, vec3f(0.0), vec3f(1.0));
}`;

const LUT_PASS = /* wgsl */ `
${GRADE_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  // 1024×32 strip = 32 slices (blue) of 32×32 (red across, green down)
  let cell: vec2f = floor(px);
  let r: f32 = fmod(cell.x, 32.0);
  let b: f32 = floor(cell.x / 32.0);
  let g: f32 = cell.y;
  return vec4f(grade(vec3f(r, g, b) / 31.0), 1.0);
}`;

const GLOW = (self, src, dir, pre) => /* wgsl */ `
fn tap(uv: vec2f, texel: vec2f) -> vec3f {
  ${pre ? 'return 0.5 * (TEX(' + src + ', uv + vec2f(0.0, 0.5) * texel).rgb + TEX(' + src + ', uv - vec2f(0.0, 0.5) * texel).rgb);' : 'return TEX(' + src + ', uv).rgb;'}
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let texel: vec2f = 1.0 / TEXSIZE(${self});
  let d: vec2f = ${dir} * texel;
  var s: vec3f = tap(uv, texel) * 0.2270;
  s += (tap(uv + d * 1.3846, texel) + tap(uv - d * 1.3846, texel)) * 0.3162;
  s += (tap(uv + d * 3.2308, texel) + tap(uv - d * 3.2308, texel)) * 0.0703;
  return vec4f(s, 1.0);
}`;

const IMAGE = /* wgsl */ `
${SPLIT_WGSL}
${GRADE_WGSL}

// Apply the 3D LUT: two bilinear reads in neighbouring blue slices, blended (= trilinear).
fn applyLUT(c0: vec3f) -> vec3f {
  let c: vec3f = clamp(c0, vec3f(0.0), vec3f(1.0)) * 31.0;
  let b0: f32 = floor(c.b);
  let b1: f32 = min(b0 + 1.0, 31.0);
  let rg: vec2f = c.rg + vec2f(0.5);
  let a: vec3f = TEX(lut, vec2f((b0 * 32.0 + rg.x) / 1024.0, rg.y / 32.0)).rgb;
  let b: vec3f = TEX(lut, vec2f((b1 * 32.0 + rg.x) / 1024.0, rg.y / 32.0)).rgb;
  return mix(a, b, c.b - b0);
}

fn caustic(p: vec2f, t: f32) -> f32 {
  // a net of soft bright lines (light focused by surface waves): borders of an animated, noise-warped Voronoi
  let w: vec2f = p + 0.12 * vec2f(perlin(p * 1.7 + vec2f(t * 0.3, 0.0)), perlin(p * 1.7 + vec2f(0.0, t * 0.3) + 5.2));
  let v: vec4f = voronoiEx(w * 4.5, 1.0, t * 0.9);
  let e: f32 = 1.0 - smoothstep(0.0, 0.2, v.y - v.x);
  let patches: f32 = smoothstep(-0.2, 0.5, perlin(p * 1.3 + vec2f(t * 0.05, 0.0)));
  return e * e * e * patches;
}

// small inset graphs: tone curves (R, G, B of a grey ramp through the LUT) or the LUT strip itself
fn insets(col: vec3f, px: vec2f) -> vec3f {
  var c: vec3f = col;
  let s: f32 = max(1.0, u.resolution.y / 540.0);
  let q: vec2f = (px - u.insetRect.xy) / u.insetRect.zw;
  let m: vec2f = vec2f(4.0 * s) / u.insetRect.zw;      // frame margin in inset units
  if (q.x < -m.x || q.x > 1.0 + m.x || q.y < -m.y || q.y > 1.0 + m.y) { return c; }
  if (u.inset > 1.5) {
    // the LUT strip itself
    c = vec3f(0.0);
    if (q.x >= 0.0 && q.x <= 1.0 && q.y >= 0.0 && q.y <= 1.0) { c = TEX(lut, q).rgb; }
    return c;
  }
  // tone curve: a grey ramp pushed through the LUT, one line per channel
  c = mix(c, vec3f(0.03, 0.035, 0.05), 0.82);
  if (q.x >= 0.0 && q.x <= 1.0 && q.y >= 0.0 && q.y <= 1.0) {
    let sz: f32 = u.insetRect.z;
    let grid: vec2f = abs(fract(q * 4.0 + 0.5) - 0.5) * sz / 4.0;
    c = mix(c, vec3f(0.2, 0.22, 0.28), (1.0 - smoothstep(0.0, 1.0, min(grid.x, grid.y))) * 0.6);
    c = mix(c, vec3f(0.45), (1.0 - smoothstep(0.5, 1.5, abs((1.0 - q.y) - q.x) * sz * 0.7071)) * 0.5);
    let dx: f32 = 1.0 / sz;
    let f0: vec3f = applyLUT(vec3f(q.x));
    let f1: vec3f = applyLUT(vec3f(q.x + dx));
    let slope: vec3f = (f1 - f0) / dx;
    let dist: vec3f = abs(f0 - vec3f(1.0 - q.y)) / sqrt(vec3f(1.0) + slope * slope) * sz;
    let lr: f32 = 1.0 - smoothstep(0.6 * s, 1.8 * s, dist.r);
    let lg: f32 = 1.0 - smoothstep(0.6 * s, 1.8 * s, dist.g);
    let lb: f32 = 1.0 - smoothstep(0.6 * s, 1.8 * s, dist.b);
    c = c + vec3f(1.0, 0.25, 0.25) * lr + vec3f(0.3, 1.0, 0.35) * lg + vec3f(0.35, 0.5, 1.0) * lb;
  }
  return c;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let t: f32 = u.time;
  let aspect: f32 = u.resolution.x / u.resolution.y;
  let cp: vec2f = (uv - 0.5) * vec2f(aspect, 1.0);
  // --- uv-space effects first: gate weave (film), wobble (underwater / poison)
  var q: vec2f = uv + u.weaveOff;
  if (u.sWobble > 0.001) {
    q += vec2f(sin(uv.y * 17.0 + t * 2.1) + 0.5 * sin(uv.y * 41.0 - t * 3.3),
               cos(uv.x * 13.0 + t * 1.7)) * 0.0035 * u.sWobble;
  }
  let orig: vec3f = TEX(game, uv).rgb;
  let src: vec3f = TEX(game, q).rgb;
  // --- the color grade: one LUT lookup
  var c: vec3f = applyLUT(src);
  // --- glow-based effects use a quarter-res blurred copy of the image
  let blur: vec3f = TEX(glowV, q).rgb;
  if (u.sEdgeBlur > 0.001) {
    let e: f32 = smoothstep(0.25, 0.75, length(cp) / (0.5 * aspect)) * u.sEdgeBlur;
    c = mix(c, applyLUT(blur), e);
  }
  if (u.sDream > 0.001) { c = mix(c, blendScreen(c, applyLUT(blur) * 0.85), u.sDream); }
  if (u.sHal > 0.001) {
    let hl: vec3f = max(blur - vec3f(0.5), vec3f(0.0)) * 2.4;
    c += luma(hl) * u.sHalCol * u.sHal;
  }
  if (u.sCaustics > 0.001) {
    let k: f32 = caustic(vec2f(uv.x * aspect, uv.y), t);
    c += vec3f(0.55, 0.95, 1.0) * k * 0.35 * u.sCaustics * (1.2 - uv.y);
    // light shafts from the surface
    let shaft: f32 = pow(0.5 + 0.5 * sin(uv.x * 23.0 + uv.y * 6.0 + t * 0.6) * sin(uv.x * 9.0 - t * 0.4), 3.0);
    c += vec3f(0.35, 0.7, 0.8) * shaft * (1.0 - uv.y) * 0.18 * u.sCaustics;
  }
  if (u.sNoise > 0.001) {
    let n: f32 = hash21(floor(px / 1.5) + vec2f(fract(t * 7.3) * 517.0, fract(t * 3.1) * 211.0)) - 0.5;
    c += vec3f(0.3, 1.0, 0.35) * n * 0.35 * u.sNoise;
    c *= 1.0 - 0.18 * u.sScan * (0.5 + 0.5 * sin(px.y * 3.14159 * 0.5));
    c += vec3f(0.2, 0.6, 0.25) * 0.04 * u.sNoise * step(0.995, hash21(vec2f(floor(px.y / 2.0), floor(t * 30.0))));
  }
  // --- vignette (optionally pulsing like a heartbeat), colored
  let beat: f32 = pow(abs(sin(t * 2.4)), 12.0) + 0.6 * pow(abs(sin(t * 2.4 - 0.45)), 12.0);
  let vr: f32 = length(cp) / (0.5 * sqrt(aspect * aspect + 1.0));
  let vig: f32 = smoothstep(0.38 - 0.12 * beat * u.sPulse, 1.05, vr) * (u.sVig + 0.25 * beat * u.sPulse);
  c = mix(c, u.sVigCol, clamp(vig, 0.0, 1.0));
  // --- night-vision goggles mask (two overlapping circles)
  if (u.sGoggles > 0.001) {
    let gd: f32 = min(length(cp - vec2f(-0.26, 0.0)), length(cp - vec2f(0.26, 0.0))) - 0.47;
    c = mix(c, vec3f(0.0), smoothstep(-0.03, 0.01, gd) * u.sGoggles);
  }
  // --- film grain: random per pixel-cell and per frame, strongest in the midtones
  if (u.sGrain > 0.001) {
    let gp: vec2f = floor(px / u.sGrainSize);
    let g: f32 = (hash21(gp + vec2f(floor(t * 24.0) * 37.0, 0.0)) + hash21(gp * 1.37 + vec2f(5.0, floor(t * 24.0) * 11.0)) - 1.0);
    let l: f32 = luma(c);
    c += vec3f(g) * u.sGrain * 0.22 * (0.35 + 2.6 * l * (1.0 - l));
  }
  c *= u.flickerK;
  // --- letterbox bars sliding in (2.39:1 "scope")
  if (u.sBox > 0.001) {
    let barH: f32 = max(0.0, 0.5 - aspect / 2.39 * 0.5) * u.sBox;
    let inBar: f32 = step(uv.y, barH) + step(1.0 - barH, uv.y);
    c = mix(c, vec3f(0.0), clamp(inBar, 0.0, 1.0));
  }
  c = clamp(c, vec3f(0.0), vec3f(1.0));
  c = splitView(c, orig, px, u.split);
  c = insets(c, px);
  return vec4f(c, 1.0);
}`;

const glowNeeded = (p) => cur.ex === 'film' || cur.ex === 'moods';

export default shaderScene({
  interaction: 'Move the mouse to slide the before/after divider.',
  examples: [
    {
      id: 'grading',
      label: 'Grading controls',
      kind: 'Abstract',
      note: 'The basic color controls every engine and photo app has. The graph (bottom right) is the <b>tone curve</b>: a grey ramp pushed through your grade, one line per channel. Temperature splits the red and blue lines; contrast bends them into an S.',
      params: { compare: true, curves: true, exposure: 0, contrast: 1.15, saturation: 1.2, vibrance: 0.2, temperature: 0.25, tint: 0, hue: 0 },
    },
    {
      id: 'pro',
      label: 'Lift/gamma/gain & tone mapping',
      kind: 'Abstract',
      note: 'How colorists work: <b>lift</b> moves the shadows, <b>gamma</b> the midtones, <b>gain</b> the highlights; <b>split toning</b> tints shadows and highlights differently (the blockbuster “teal &amp; orange”). Push <b>Exposure</b> up to create values brighter than white, then compare tone mappers.',
      params: { compare: true, curves: true, exposure: 1.2, tonemap: 'aces', lift: 0.03, gamma: 1.05, gain: 1, shadowTint: '#2f7f8f', highTint: '#ffad6b', splitAmt: 0.6 },
    },
    {
      id: 'moods',
      label: 'Mood presets (LUTs)',
      kind: 'In a game',
      note: 'One game, many moods — each preset is just a different 3D LUT (the strip at the bottom right) plus a few spatial touches. Games swap or blend LUTs for areas, status effects (poison!), flashbacks and time of day.',
      params: { compare: true, preset: 'horror', intensity: 1, showLut: true },
    },
    {
      id: 'film',
      label: 'Film look',
      kind: 'Real life',
      note: 'What makes digital look like film: a film-stock grade, <b>grain</b> that changes every frame, <b>vignette</b>, <b>letterbox</b> bars, <b>gate weave</b> (the frame jitters slightly in the projector), <b>halation</b> (red glow bleeding around highlights) and a little brightness <b>flicker</b>.',
      params: { compare: false, stock: 'kodak', grain: 0.45, grainSize: 1.5, vignette: 0.55, letterbox: 1, weave: 0.5, halation: 0.6, flicker: 0.3 },
      hint: '',
    },
  ],
  controls: [
    { type: 'heading', label: 'Basic', showFor: ['grading'] },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: -3, max: 3, step: 0.05, value: 0, unit: 'EV', showFor: ['grading', 'pro'], help: 'Brightness in photographic stops: +1 = twice as much light.' },
    { type: 'slider', key: 'contrast', label: 'Contrast', min: 0.4, max: 2, step: 0.01, value: 1, showFor: ['grading'], help: 'Spreads tones away from (or towards) middle grey.' },
    { type: 'slider', key: 'saturation', label: 'Saturation', min: 0, max: 2, step: 0.01, value: 1, showFor: ['grading'], help: '0 = black & white, 2 = cartoon colors.' },
    { type: 'slider', key: 'vibrance', label: 'Vibrance', min: -1, max: 1, step: 0.01, value: 0, showFor: ['grading'], help: 'Smart saturation: boosts dull colors, leaves already-strong ones alone.' },
    { type: 'slider', key: 'temperature', label: 'Temperature', min: -1, max: 1, step: 0.01, value: 0, showFor: ['grading'], help: 'Cool blue ↔ warm orange (white balance).' },
    { type: 'slider', key: 'tint', label: 'Tint', min: -1, max: 1, step: 0.01, value: 0, showFor: ['grading'], help: 'Green ↔ magenta (the other white-balance axis).' },
    { type: 'slider', key: 'hue', label: 'Hue shift', min: -180, max: 180, step: 1, value: 0, unit: '°', showFor: ['grading'], help: 'Rotates every color around the color wheel, keeping brightness.' },
    { type: 'heading', label: 'Shadows · midtones · highlights', showFor: ['pro'] },
    { type: 'slider', key: 'lift', label: 'Lift (shadows)', min: -0.2, max: 0.2, step: 0.005, value: 0, showFor: ['pro'], help: 'Raises or crushes the blacks. Positive = faded, matte look.' },
    { type: 'slider', key: 'gamma', label: 'Gamma (midtones)', min: 0.5, max: 2, step: 0.01, value: 1, showFor: ['pro'], help: 'Brightens or darkens the middle without touching black and white.' },
    { type: 'slider', key: 'gain', label: 'Gain (highlights)', min: 0.5, max: 1.5, step: 0.01, value: 1, showFor: ['pro'], help: 'Scales the brights.' },
    { type: 'color', key: 'shadowTint', label: 'Shadow tint', value: '#808080', showFor: ['pro'] },
    { type: 'color', key: 'highTint', label: 'Highlight tint', value: '#808080', showFor: ['pro'] },
    { type: 'slider', key: 'splitAmt', label: 'Split toning amount', min: 0, max: 1, step: 0.01, value: 0, showFor: ['pro'], help: 'How strongly the two tints are applied. Grey = no tint.' },
    {
      type: 'select',
      key: 'tonemap',
      label: 'Tone mapping',
      value: 'none',
      options: [
        { value: 'none', label: 'None (hard clip)' },
        { value: 'reinhard', label: 'Reinhard' },
        { value: 'aces', label: 'ACES filmic' },
        { value: 'hable', label: 'Hable (Uncharted 2)' },
      ],
      showFor: ['pro'],
      help: 'How values brighter than 1.0 are squeezed into the screen. Watch the sky and the curve’s top end.',
    },
    {
      type: 'select',
      key: 'preset',
      label: 'Mood',
      value: 'horror',
      options: [
        { value: 'none', label: 'Neutral' },
        { value: 'horror', label: 'Horror' },
        { value: 'sunset', label: 'Golden hour / sunset' },
        { value: 'underwater', label: 'Underwater' },
        { value: 'poison', label: 'Poisoned' },
        { value: 'flashback', label: 'Flashback (sepia)' },
        { value: 'nightvision', label: 'Night vision' },
        { value: 'winter', label: 'Frozen / winter' },
        { value: 'noir', label: 'Film noir' },
        { value: 'vaporwave', label: 'Vaporwave' },
        { value: 'bleach', label: 'Bleach bypass (war film)' },
        { value: 'dream', label: 'Dream / memory' },
      ],
      showFor: ['moods'],
    },
    { type: 'slider', key: 'intensity', label: 'Intensity', min: 0, max: 1, step: 0.01, value: 1, showFor: ['moods'], help: 'Blend from neutral to the full look — games fade LUTs in and out like this.' },
    { type: 'toggle', key: 'showLut', label: 'Show the LUT strip', value: true, showFor: ['moods'], help: '32 slices of 32×32: red →, green ↓, blue = slice.' },
    {
      type: 'select',
      key: 'stock',
      label: 'Film stock grade',
      value: 'kodak',
      options: [
        { value: 'none', label: 'Neutral' },
        { value: 'kodak', label: 'Warm print film' },
        { value: 'teal', label: 'Teal & orange' },
        { value: 'bleach', label: 'Bleach bypass' },
        { value: 'bw', label: 'Black & white' },
      ],
      showFor: ['film'],
    },
    { type: 'slider', key: 'grain', label: 'Grain', min: 0, max: 1, step: 0.01, value: 0.45, showFor: ['film'], help: 'Random speckle, new every frame, strongest in the midtones.' },
    { type: 'slider', key: 'grainSize', label: 'Grain size', min: 1, max: 4, step: 0.1, value: 1.5, unit: 'px', showFor: ['film'] },
    { type: 'slider', key: 'vignette', label: 'Vignette', min: 0, max: 1, step: 0.01, value: 0.55, showFor: ['film'], help: 'Darker corners pull the eye to the centre.' },
    { type: 'slider', key: 'letterbox', label: 'Letterbox', min: 0, max: 1, step: 0.01, value: 1, showFor: ['film'], help: 'Black bars to a 2.39:1 “cinemascope” frame — the cutscene look.' },
    { type: 'slider', key: 'weave', label: 'Gate weave', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['film'], help: 'Tiny random frame jitter, like film running through a projector.' },
    { type: 'slider', key: 'halation', label: 'Halation', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['film'], help: 'Reddish glow around bright lights (light scattering inside the film).' },
    { type: 'slider', key: 'flicker', label: 'Flicker', min: 0, max: 1, step: 0.01, value: 0.3, showFor: ['film'] },
    { type: 'toggle', key: 'compare', label: 'Before / after divider', value: true, showFor: ['grading', 'pro', 'moods', 'film'] },
    { type: 'toggle', key: 'curves', label: 'Show tone curve', value: true, showFor: ['grading', 'pro'] },
  ],
  uniforms: {
    gExp: 'f32', gCon: 'f32', gSat: 'f32', gVib: 'f32', gTemp: 'f32', gTint: 'f32', gHue: 'f32', gSplit: 'f32', gTone: 'f32',
    gLift: 'vec3f', gGamma: 'vec3f', gGain: 'vec3f', gShadow: 'vec3f', gHigh: 'vec3f',
    sVigCol: 'vec3f', sHalCol: 'vec3f',
    sVig: 'f32', sGrain: 'f32', sGrainSize: 'f32', sBox: 'f32', sHal: 'f32', sWobble: 'f32', sCaustics: 'f32', sScan: 'f32',
    sNoise: 'f32', sGoggles: 'f32', sPulse: 'f32', sEdgeBlur: 'f32', sDream: 'f32',
    weaveOff: 'vec2f', flickerK: 'f32', split: 'f32', inset: 'f32', insetRect: 'vec4f',
  },
  include: withGameIncludes(['hash', 'noise', 'color']),
  passes: [
    gamePass(),
    { name: 'lut', size: [1024, 32], code: LUT_PASS },
    { name: 'glowH', scale: 0.25, iterations: (p) => (glowNeeded(p) ? 1 : 0), code: GLOW('glowH', 'game', 'vec2f(1.5, 0.0)', true) },
    { name: 'glowV', scale: 0.25, iterations: (p) => (glowNeeded(p) ? 1 : 0), code: GLOW('glowV', 'glowH', 'vec2f(0.0, 1.5)', false) },
  ],
  resetOnExample: false,
  bind(p, ctx) {
    cur.ex = ctx.example;
    ov.begin(ctx);
    let look;
    if (ctx.example === 'moods') look = mixLook(PRESETS[p.preset] || NEUTRAL, p.intensity);
    else if (ctx.example === 'film') {
      look = { ...(STOCKS[p.stock] || NEUTRAL) };
      Object.assign(look, { grain: p.grain, grainSize: p.grainSize, vig: p.vignette, box: p.letterbox, weave: p.weave, hal: p.halation, halCol: [1, 0.42, 0.22], flicker: p.flicker });
    } else look = lookFromSliders(p);

    // gate weave & flicker change at 24 "film frames" per second
    const tick = Math.floor(ctx.time * 24);
    if (tick !== st.weaveT) {
      st.weaveT = tick;
      const w = look.weave;
      st.weave = [((Math.random() - 0.5) * 1.6 * w) / ctx.width, ((Math.random() - 0.5) * 2.4 * w + Math.sin(ctx.time * 0.9) * 1.2 * w) / ctx.height];
      st.flick = 1 - look.flicker * 0.12 * Math.random();
    }
    if (!look.weave) st.weave = [0, 0];
    if (!look.flicker) st.flick = 1;

    const showLut = ctx.example === 'moods' && p.showLut;
    const showCurve = (ctx.example === 'grading' || ctx.example === 'pro') && p.curves;
    // inset rectangle in device px (bottom right, above the hint chip); labels go just above it
    const S = Math.max(1, ctx.height / 540);
    const dpr = ctx.dpr || 1;
    let rect = [0, 0, 1, 1];
    if (showLut) {
      const w = Math.min(ctx.width * 0.6, 640 * S);
      const h = w / 32;
      rect = [ctx.width - w - 12 * S, ctx.height - h - 48 * S, w, h];
      ov.show('lut', '3D LUT · 32×32×32 — this strip <i>is</i> the whole grade', `right:${(12 * S) / dpr}px;bottom:${(h + 58 * S) / dpr}px`);
    } else if (showCurve) {
      const sz = Math.min(ctx.height * 0.36, 230 * S);
      rect = [ctx.width - sz - 12 * S, ctx.height - sz - 12 * S, sz, sz];
      ov.show('curve', 'Tone curve <span style="color:#ff6b6b">R</span> <span style="color:#6bff7a">G</span> <span style="color:#7aa0ff">B</span>', `right:${(12 * S) / dpr}px;bottom:${(sz + 22 * S) / dpr}px`);
    }
    const split = compareSplit(st, ctx, p.compare, ov, ['Original', 'Graded']);
    ov.end();
    return {
      gExp: look.exp, gCon: look.con, gSat: look.sat, gVib: look.vib, gTemp: look.temp, gTint: look.tint, gHue: look.hue,
      gSplit: look.split, gTone: look.tone, gLift: look.lift, gGamma: look.gamma, gGain: look.gain, gShadow: look.shadow, gHigh: look.high,
      sVig: look.vig, sVigCol: look.vigCol, sGrain: look.grain, sGrainSize: look.grainSize, sBox: look.box, sHal: look.hal, sHalCol: look.halCol,
      sWobble: look.wobble, sCaustics: look.caustics, sScan: look.scan, sNoise: look.noise, sGoggles: look.goggles, sPulse: look.pulse,
      sEdgeBlur: look.edgeBlur, sDream: look.dream,
      weaveOff: st.weave, flickerK: st.flick, split, inset: showLut ? 2 : showCurve ? 1 : 0, insetRect: rect,
    };
  },
  code: IMAGE,
  about: {
    summary:
      'Color grading is how games set mood: the same level can feel cosy, eerie, poisoned or nostalgic just by remapping its colors. The standard trick is to bake the whole grade into a small 3D lookup table (LUT).',
    what: `<p>The live game scene, color graded. <b>Grading controls</b> and <b>Lift/gamma/gain</b> expose the individual adjustments with a live tone curve;
      <b>Mood presets</b> show ready-made looks (and the LUT that encodes each one); <b>Film look</b> adds the spatial effects that make an image feel “shot on film”.</p>`,
    how: `<ol>
      <li><b>Order matters.</b> Exposure and white balance work on linear light; contrast is applied around middle grey; then <b>tone mapping</b> squeezes anything brighter than 1.0 into the screen range; lift/gamma/gain, saturation, hue and split toning come after.</li>
      <li><b>Bake it into a LUT.</b> Every one of those steps depends only on the input color, so we evaluate the whole chain once for a 32×32×32 grid of colors — 32,768 cells, a 1024×32 texture — every frame.</li>
      <li><b>Apply it with one lookup.</b> For each screen pixel, its red/green pick a spot inside a slice, blue picks two neighbouring slices; bilinear filtering + one mix = trilinear interpolation. Two texture reads, whatever the grade.</li>
      <li><b>Spatial effects can’t go in a LUT</b> because they depend on <i>where</i> a pixel is or what its neighbours are: vignette, grain, letterbox, halation/glow, wobble. They are applied afterwards.</li>
      <li><b>Blending moods</b> is just blending LUTs (or their parameters): fade towards “poisoned” as health drops, or between “day” and “night”.</li>
    </ol>`,
    uses: [
      { title: 'Areas & biomes', text: 'Each zone gets its own LUT; crossfade at the border. Hollow Knight, Ori and most modern games do this.' },
      { title: 'Status effects', text: 'Poison, low health, frozen, drunk, rage mode — a LUT plus a pulsing vignette says it instantly.' },
      { title: 'Story beats', text: 'Sepia flashbacks, desaturated memories, noir detective sections, horror reveals.' },
      { title: 'Time of day', text: 'Blend dawn / noon / sunset / night LUTs as the clock moves.' },
      { title: 'Artist workflow', text: 'Artists grade a screenshot in Photoshop or DaVinci Resolve with a neutral LUT strip pasted in; the graded strip becomes the game’s LUT.' },
    ],
    try: [
      'On <b>Grading controls</b> set Saturation to 0, then raise Vibrance — nothing happens, because vibrance only boosts color that exists.',
      'On <b>Lift/gamma/gain</b> push Exposure to +3 and switch tone mapping between None and ACES: the sky clips to flat white without it.',
      'On <b>Mood presets</b> slowly drag Intensity from 0 to 1 with “Poisoned” — that is how a game fades a status effect in.',
      'Look at the LUT strip while switching moods: Night vision turns it green, Noir grey, Sunset orange.',
      'On <b>Film look</b> set Grain size to 4 and Gate weave to 1 for a battered 16 mm print.',
    ],
    ask: [
      'a LUT-based color grading pass with presets I can blend between',
      'a pulsing green poison screen effect with a vignette',
      'a sepia flashback filter with grain and soft edges',
      'ACES tone mapping with an exposure control',
      'a cinematic film look: grain, letterbox, halation, gate weave',
      'per-area color grading that crossfades at zone borders',
    ],
    perf: `<p>Applying a LUT costs two texture reads per pixel, no matter how complex the grade — that is the whole point. Baking the 32³ LUT is ~33k
      shader invocations (≈1.5% of a 1080p frame), and you only need to redo it when the grade changes. Grain, vignette and letterbox are a few
      math operations; halation uses a quarter-resolution blur.</p>`,
    api: `<p>Identical on WebGPU and WebGL2. WebGPU (and WebGL2) also support real <b>3D textures</b>, so a LUT can be stored as a 32×32×32 volume and
      sampled with true trilinear filtering in one read; the 2D strip used here is the classic, universally compatible layout (and what you’d save as a PNG).</p>`,
    code: [
      {
        title: 'Applying a 32³ LUT stored as a 1024×32 strip',
        lang: 'wgsl',
        src: `fn applyLUT(c0: vec3f) -> vec3f {
  let c = clamp(c0, vec3f(0.0), vec3f(1.0)) * 31.0;
  let b0 = floor(c.b);                    // blue picks the slice...
  let b1 = min(b0 + 1.0, 31.0);           // ...and its neighbour
  let rg = c.rg + vec2f(0.5);             // red/green: position inside the slice
  let a = TEX(lut, vec2f((b0 * 32.0 + rg.x) / 1024.0, rg.y / 32.0)).rgb;
  let b = TEX(lut, vec2f((b1 * 32.0 + rg.x) / 1024.0, rg.y / 32.0)).rgb;
  return mix(a, b, c.b - b0);             // bilinear x2 + mix = trilinear
}`,
      },
      {
        title: 'The grade itself (baked into the LUT every frame)',
        lang: 'wgsl',
        src: `var lin = srgbToLinear(c0) * exp2(u.gExp);                 // exposure in stops
lin = lin * whiteBalance;                                    // temperature & tint
lin = 0.18 * pow(max(lin, vec3f(0.0)) / 0.18, vec3f(u.gCon)); // contrast around mid grey
var c = linearToSrgb(toneMap(lin, u.gTone));                 // HDR -> screen
c = u.gGain * (c + u.gLift * (vec3f(1.0) - c));              // lift & gain
c = pow(max(c, vec3f(0.0)), vec3f(1.0) / u.gGamma);          // gamma
c = mix(vec3f(luma(c)), c, u.gSat);                          // saturation
c = hueRotate(c, u.gHue);
c = mix(c, tintLum(c, u.gShadow), shadowWeight * u.gSplit);  // split toning
c = mix(c, tintLum(c, u.gHigh), highlightWeight * u.gSplit);`,
      },
    ],
    links: [
      { title: 'GPU Gems 2, ch. 24: Using Lookup Tables to Accelerate Color Transformations', url: 'https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-24-using-lookup-tables-accelerate-color', note: 'the classic LUT chapter' },
      { title: 'John Hable — Filmic tonemapping operators', url: 'http://filmicworlds.com/blog/filmic-tonemapping-operators/', note: 'the Uncharted 2 curve' },
    ],
  },
});

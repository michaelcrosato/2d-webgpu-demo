import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';

// "Three ways to make fire" — all fragment shaders, so it runs on WebGPU and WebGL2:
//   1. noise fire: scrolling, domain-warped fBm shaped by a gradient, mapped through a palette
//   2. Doom PSX fire: a cellular automaton in a feedback pass (heat rises, randomly decays/drifts)
//   3. "particle" fire faked in the fragment shader: loop over hashed particles, sum their glow
//   + an in-game scene: campfire, wall torches and burning text, with flickering light.

// The 37-colour palette of the PlayStation Doom fire (as documented by Fabien Sanglard).
const DOOM = [
  0x070707, 0x1f0707, 0x2f0f07, 0x470f07, 0x571707, 0x671f07, 0x771f07, 0x8f2707, 0x9f2f07, 0xaf3f07, 0xbf4707, 0xc74707, 0xdf4f07,
  0xdf5707, 0xdf5707, 0xd75f07, 0xd75f07, 0xd7670f, 0xcf6f0f, 0xcf770f, 0xcf7f0f, 0xcf8717, 0xc78717, 0xc78f17, 0xc7971f, 0xbf9f1f,
  0xbf9f1f, 0xbfa727, 0xbfa727, 0xbfaf2f, 0xb7af2f, 0xb7b72f, 0xb7b737, 0xcfcf6f, 0xdfdf9f, 0xefefc7, 0xffffff,
];

// Palette rows (256 wide). Row 0 = Doom (quantised), others are smooth gradients [pos, hex].
const GRADIENTS = [
  null,
  [[0, 0x000000], [0.18, 0x3a0603], [0.38, 0xa8200a], [0.58, 0xf2650f], [0.78, 0xffc23d], [0.92, 0xfff2b0], [1, 0xffffff]], // natural
  [[0, 0x000000], [0.2, 0x050a40], [0.45, 0x1032c8], [0.68, 0x2a9cff], [0.86, 0x9fe8ff], [1, 0xffffff]], // gas burner
  [[0, 0x000000], [0.2, 0x02200a], [0.45, 0x0b8a1c], [0.68, 0x52f03a], [0.86, 0xd8ff8a], [1, 0xffffff]], // cursed green
  [[0, 0x000000], [0.2, 0x1a0430], [0.45, 0x6a10b0], [0.66, 0xd040ff], [0.85, 0xffa8f0], [1, 0xffffff]], // spectral
  [[0, 0x000000], [0.25, 0x400000], [0.5, 0xc00000], [0.7, 0xff3a00], [0.88, 0xff9a40], [1, 0xfff0d0]], // hellfire
];

function paletteCanvas() {
  const W = 256;
  const H = 8;
  const c = makeCanvas(W, H);
  const g = c.getContext('2d');
  const img = g.createImageData(W, H);
  const rgb = (h) => [(h >> 16) & 255, (h >> 8) & 255, h & 255];
  for (let row = 0; row < GRADIENTS.length; row++) {
    for (let x = 0; x < W; x++) {
      const t = x / (W - 1);
      let col;
      if (row === 0) col = rgb(DOOM[Math.min(36, Math.floor(t * 36.999))]);
      else {
        const stops = GRADIENTS[row];
        let i = 0;
        while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
        const [p0, h0] = stops[i];
        const [p1, h1] = stops[i + 1];
        const k = Math.min(1, Math.max(0, (t - p0) / (p1 - p0)));
        const a = rgb(h0);
        const b = rgb(h1);
        col = [0, 1, 2].map((j) => Math.round(a[j] + (b[j] - a[j]) * k));
      }
      const o = (row * W + x) * 4;
      img.data[o] = col[0];
      img.data[o + 1] = col[1];
      img.data[o + 2] = col[2];
      img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return c;
}

function textCanvas() {
  const c = makeCanvas(1024, 256);
  const g = c.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, 1024, 256);
  g.fillStyle = '#fff';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.font = '900 172px Georgia, "Times New Roman", serif';
  g.fillText('BURN', 512, 142);
  return c;
}

// state shared between bind() and the pass iteration count (warm-up of the Doom fire)
let warm = 0;
let seen = '';
// the headless test harness (software GPU) renders the final image at half resolution
let testScale = 1;

export default shaderScene({
  interaction: 'Use the sliders: wind, turbulence, height and palette change all three techniques.',
  examples: [
    {
      id: 'noise',
      label: '1 · Noise fire',
      kind: 'Technique',
      note: 'fBm noise scrolls upward, is warped by a second noise (turbulence), multiplied by a vertical gradient (hot at the bottom), and the result — a single “heat” number — picks a color from a palette. Turn on <b>Show the layers</b> to see each step side by side.',
      params: { intensity: 1, wind: 0.15, turbulence: 1, height: 1, palette: 'natural', temperature: 1, speed: 1 },
      hint: 'Try “Show the layers”, wind and turbulence.',
    },
    {
      id: 'doom',
      label: '2 · Doom PSX fire',
      kind: 'Classic',
      note: 'The 1995 PlayStation Doom title fire: a grid of heat values. Each frame every cell copies the cell <b>below</b> it, nudged sideways at random and randomly cooled by one step. The bottom row is the fuel. Set <i>Intensity</i> to 0 and watch it die out.',
      params: { intensity: 1, wind: 0.15, height: 1, palette: 'doom', temperature: 1, pixel: 4 },
      hint: 'Set Intensity to 0 to let the fire die out.',
    },
    {
      id: 'particles',
      label: '3 · Particle fire',
      kind: 'Comparison',
      note: 'Fire as particles: each blob is born at the base, rises, drifts with the wind, shrinks and cools. Here they are faked inside the fragment shader (every pixel loops over every particle) — fine for one fire, but real games use a particle system (see GPU Particles). Toggle <b>Show particles</b>.',
      params: { intensity: 1, wind: 0.1, turbulence: 1, height: 1, palette: 'natural', temperature: 1, count: 96, speed: 1 },
      hint: 'Toggle “Show particles” and change the count.',
    },
    {
      id: 'game',
      label: 'Campfire, torches & burning text',
      kind: 'In a game',
      note: 'Noise fire shaped into flames, plus the things that sell it: flickering warm light on the walls and floor, rising embers, heat at the base, and text that is both on fire and burning away with a glowing edge.',
      params: { intensity: 1, wind: 0.1, turbulence: 1, height: 1, palette: 'natural', temperature: 1, speed: 1 },
      hint: 'Change the palette or wind — everything updates.',
    },
  ],
  controls: [
    { type: 'slider', key: 'intensity', label: 'Intensity / fuel', min: 0, max: 1.6, step: 0.01, value: 1, help: 'How much heat is fed in at the base.' },
    { type: 'slider', key: 'wind', label: 'Wind', min: -1, max: 1, step: 0.01, value: 0.15, help: 'Leans the flames; in the Doom fire it biases the sideways drift.' },
    { type: 'slider', key: 'turbulence', label: 'Turbulence', min: 0, max: 2.5, step: 0.01, value: 1, help: 'Strength of the domain warp (noise bending noise).', showFor: ['noise', 'particles', 'game'] },
    { type: 'slider', key: 'height', label: 'Flame height (decay)', min: 0.3, max: 2, step: 0.01, value: 1, help: 'How fast heat cools as it rises.' },
    {
      type: 'select',
      key: 'palette',
      label: 'Palette',
      value: 'natural',
      options: [
        { value: 'doom', label: 'Doom (37 colors)' },
        { value: 'natural', label: 'Natural (black-body)' },
        { value: 'gas', label: 'Gas burner (blue)' },
        { value: 'cursed', label: 'Cursed (green)' },
        { value: 'spectral', label: 'Spectral (purple)' },
        { value: 'hell', label: 'Hellfire (red)' },
      ],
      help: 'Heat (0..1) is just an index into this color ramp.',
    },
    { type: 'slider', key: 'temperature', label: 'Temperature', min: 0.5, max: 1.8, step: 0.01, value: 1, help: 'Scales heat before the palette lookup: hotter = whiter.' },
    { type: 'slider', key: 'speed', label: 'Speed', min: 0, max: 3, step: 0.01, value: 1, showFor: ['noise', 'particles', 'game'] },
    { type: 'toggle', key: 'layers', label: 'Show the layers', value: false, help: 'noise → gradient → heat → palette, left to right.', showFor: ['noise'] },
    { type: 'slider', key: 'pixel', label: 'Fire pixel size', min: 1, max: 12, step: 1, value: 4, help: 'Size of one automaton cell on screen.', showFor: ['doom'] },
    { type: 'slider', key: 'count', label: 'Particles', min: 8, max: 128, step: 1, value: 96, help: 'Each pixel loops over all of them: cost grows linearly.', showFor: ['particles'] },
    { type: 'toggle', key: 'showParticles', label: 'Show particles', value: false, showFor: ['particles'] },
  ],
  uniforms: {
    intensity: 'f32',
    wind: 'f32',
    turbulence: 'f32',
    height: 'f32',
    palette: 'f32',
    temperature: 'f32',
    speed: 'f32',
    layers: 'f32',
    pixel: 'f32',
    count: 'f32',
    showParticles: 'f32',
    coolStep: 'f32',
    lod: 'f32',
  },
  include: ['noise', 'sdf', 'color', 'math'],
  textures: {
    pal: { source: async () => paletteCanvas(), filter: 'linear' },
    words: { source: async () => textCanvas(), filter: 'linear' },
  },
  bind(params, ctx) {
    // warm the Doom automaton up for a few frames after (re)start so it isn't empty
    testScale = ctx.testMode ? 0.5 : 1;
    const key = `${ctx.example}|${ctx.width}x${ctx.height}|${params.pixel}`;
    if (key !== seen || ctx.frame === 0) {
      seen = key;
      warm = 4;
    }
    const rows = Math.ceil(ctx.height / Math.max(1, params.pixel));
    // the original: 168 rows, 50% chance per row of cooling one of 36 steps -> flames ~44% of
    // the screen. Keep the 50% chance and scale the step so heights match at any resolution.
    // lod: fewer noise octaves under the headless test harness (software GPU)
    return { coolStep: 2 / (0.44 * rows * Math.max(0.3, params.height)), lod: ctx.testMode ? 1 : 0 };
  },
  onAction(key) {
    if (key === 'reset') warm = 4;
  },
  renderScale: () => testScale,
  passes: [
    {
      name: 'doom',
      format: 'rgba16float',
      size: (params, ctx) => [Math.max(8, Math.ceil(ctx.width / Math.max(1, params.pixel))), Math.max(8, Math.ceil(ctx.height / Math.max(1, params.pixel)))],
      iterations: () => {
        if (warm > 0) {
          warm--;
          return 50;
        }
        return 1;
      },
      code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let size = TEXSIZE(doom);
  let ip = vec2i(floor(px));
  if (u.frame < 0.5) { return vec4f(0.0); }
  // the bottom row is the fuel line
  if (ip.y >= i32(size.y) - 1) { return vec4f(min(u.intensity, 1.0), 0.0, 0.0, 1.0); }
  // one random number per cell per step (like rand() & 3 in the original)
  let r = hash33(vec3f(px, u.frame + u.time * 7.0));
  // drift: take heat from the cell below, shifted -1/0/+1 (wind biases it)
  let shift = i32(floor(r.x * 3.0 + u.wind * 1.4)) - 1;
  let below = LOADW(doom, vec2i(ip.x - shift, ip.y + 1)).r;
  // cool by ~one palette step with a 50% chance (step size is scaled with the grid height)
  let cool = select(0.0, u.coolStep, r.y < 0.5);
  return vec4f(max(below - cool, 0.0), 0.0, 0.0, 1.0);
}`,
    },
  ],
  code: /* wgsl */ `
// ------------------------------------------------------------------ helpers
fn palRow() -> f32 { return (u.palette + 0.5) / 8.0; }
fn firePal(h: f32) -> vec3f {
  let x = clamp(h * u.temperature, 0.0, 1.0);
  if (u.palette < 0.5) { return TEXN(pal, vec2f(x, palRow())).rgb; }
  return TEX(pal, vec2f(x * 0.996 + 0.002, palRow())).rgb;
}

// Heat of a noise flame. p: local coords (y up from the base, ~0..1 tall), w: base half-width
fn flameHeat(p: vec2f, t: f32, seed: f32, w: f32) -> f32 {
  let y = max(p.y, 0.0);
  let x = p.x - u.wind * y * y * 0.55;               // lean with the wind, more at the top
  var q = vec2f(x * 3.2 + seed, y * 2.0 - t * 2.3);   // scroll the noise upward
  let warp = vec2f(fbm(q * 1.2 + vec2f(seed, -t * 0.9), 3 - i32(u.lod)), fbm(q * 1.2 + vec2f(3.7 + seed, 1.3 - t * 0.9), 3 - i32(u.lod)));
  q += warp * u.turbulence * 0.55;                    // turbulence = domain warping
  let n = fbm(q, 4 - i32(u.lod)) * 0.5 + 0.5;
  let h = 0.85 * u.height;
  let body = 1.0 - smoothstep(0.0, w * (1.1 - 0.9 * clamp(y / h, 0.0, 1.0)) + 0.02, abs(x + (n - 0.5) * 0.12));
  let grad = 1.0 - y / h;
  return clamp((n * 1.35 + 0.15) * body * grad * 1.6 - 0.25 * y / h, 0.0, 1.0) * u.intensity;
}

fn bgSky(uv: vec2f) -> vec3f { return mix(vec3f(0.01, 0.008, 0.02), vec3f(0.035, 0.012, 0.01), uv.y); }

// ------------------------------------------------------------------ 1: noise fire wall
fn noiseFire(uv: vec2f, px: vec2f) -> vec3f {
  let t = u.time * u.speed;
  let aspect = u.resolution.x / u.resolution.y;
  var x = px.x / u.resolution.y;
  let y = 1.0 - uv.y;
  var band = 3.0;
  if (u.layers > 0.5) {
    band = floor(uv.x * 4.0);
    x = fract(uv.x * 4.0) * aspect * 0.25 + 0.3;
  }
  var q = vec2f(x * 3.0, y * 2.2 - t * 1.8);
  let warp = vec2f(fbm(q * 1.1 + vec2f(0.0, -t * 0.7), 3 - i32(u.lod)), fbm(q * 1.1 + vec2f(5.2, 1.3 - t * 0.7), 3 - i32(u.lod)));
  q += warp * u.turbulence * 0.6;
  q.x -= u.wind * y * y * 2.0;
  let n = fbm(q, 5 - i32(u.lod) * 2) * 0.5 + 0.5;
  let grad = clamp(1.0 - y / (0.9 * u.height), 0.0, 1.0);
  let heat = clamp((n * grad * 1.9 - 0.36 + 0.3 * grad * grad) * u.intensity, 0.0, 1.0);
  var col = firePal(heat) * (0.25 + 0.75 * smoothstep(0.0, 0.25, heat));
  if (band < 0.5) { col = vec3f(n); }
  else if (band < 1.5) { col = vec3f(grad); }
  else if (band < 2.5) { col = vec3f(heat); }
  if (u.layers > 0.5) {
    let sep = fract(uv.x * 4.0);
    col = mix(col, vec3f(0.6), (1.0 - smoothstep(0.0, 0.004 * 4.0, min(sep, 1.0 - sep))) * 0.8);
  }
  return col + bgSky(uv) * (1.0 - heat);
}

// ------------------------------------------------------------------ 2: Doom fire
fn doomFire(uv: vec2f) -> vec3f {
  let h = TEXN(doom, uv).r;
  let c = firePal(h);
  if (u.palette < 0.5) { return c; }
  return c + bgSky(uv) * (1.0 - h);
}

// ------------------------------------------------------------------ 3: hashed "particle" fire
fn particleFire(uv: vec2f, px: vec2f) -> vec3f {
  let t = u.time * u.speed;
  let p = vec2f((px.x - 0.5 * u.resolution.x) / u.resolution.y, 1.0 - uv.y);   // x centered, y up
  let base = vec2f(0.0, 0.2);
  var col = bgSky(uv);
  // floor & logs
  if (p.y < base.y) { col = vec3f(0.02, 0.012, 0.01); }
  let H = 0.72 * u.height;
  // bounding box: skip the loop for pixels far from the fire
  if (abs(p.x) > 0.5 + abs(u.wind) * 0.4 || p.y > base.y + H + 0.2) { return col; }
  var heatSum = 0.0;
  var ring = 0.0;
  let n = select(i32(u.count), min(i32(u.count), 40), u.lod > 0.5);
  for (var i = 0; i < 128; i++) {
    if (i >= n) { break; }
    let fi = f32(i);
    let h = hash13(fi * 7.31 + 1.0);
    let life = 0.8 + 0.6 * h.z;
    let age = fract(t / life + h.x);                  // 0..1, loops forever: "respawn"
    let spawn = base + vec2f((h.y - 0.5) * 0.24, 0.0);
    let wob = (sin(age * 7.0 + fi * 1.7) * 0.03 + sin(age * 13.0 + fi * 4.1 + t * 2.0) * 0.015) * u.turbulence;
    // rise, converge toward the middle (teardrop), lean with the wind
    let pos = vec2f(spawn.x * (1.0 - 0.75 * age) + u.wind * age * age * 0.35 + wob, spawn.y + age * H);
    let size = mix(0.045, 0.02, age) * (0.7 + 0.6 * h.z) * (0.6 + 0.4 * smoothstep(0.0, 0.15, age) + 0.4);
    let d = length(p - pos);
    let fade = pow(1.0 - age, 1.3);
    heatSum += exp(-(d * d) / (size * size)) * fade * 0.5 * (64.0 / max(f32(n), 8.0) * 0.6 + 0.4);
    if (u.showParticles > 0.5) { ring += 1.0 - smoothstep(0.0, 0.0025, abs(d - size)); }
  }
  let heat = clamp(heatSum * u.intensity, 0.0, 1.0);
  col = col * (1.0 - heat) + firePal(heat) * smoothstep(0.0, 0.2, heat);
  col += vec3f(0.3, 0.8, 1.0) * clamp(ring, 0.0, 1.0) * 0.7;
  return col;
}

// ------------------------------------------------------------------ 4: in a game
fn brickWall(p: vec2f) -> vec3f {
  var b = p * vec2f(9.0, 18.0);
  b.x += 0.5 * floor(b.y);
  let f = fract(b);
  let id = floor(b);
  let mortar = smoothstep(0.0, 0.07, min(min(f.x, 1.0 - f.x) * 0.5, min(f.y, 1.0 - f.y)));
  let tone = 0.75 + 0.35 * hash21(id) + 0.15 * valueNoise(p * 40.0);
  return mix(vec3f(0.03, 0.028, 0.03), vec3f(0.17, 0.13, 0.12) * tone, mortar);
}

fn gameScene(uv: vec2f, px: vec2f) -> vec3f {
  let t = u.time * u.speed;
  let aspect = u.resolution.x / u.resolution.y;
  let p = vec2f(px.x / u.resolution.y, 1.0 - uv.y);    // x 0..aspect, y up
  let floorY = 0.24;
  let cf = vec2f(aspect * 0.5, floorY + 0.03);            // campfire base
  let tl = vec2f(aspect * 0.14, 0.62);                    // torches
  let tr = vec2f(aspect * 0.86, 0.62);
  // flicker: low-frequency noise per light
  let fl0 = 0.85 + 0.25 * valueNoise(vec2f(t * 6.0, 1.0)) * u.intensity;
  let fl1 = 0.85 + 0.25 * valueNoise(vec2f(t * 7.0, 5.0));
  let fl2 = 0.85 + 0.25 * valueNoise(vec2f(t * 7.0, 9.0));

  // ---- base color (unlit albedo)
  var alb = brickWall(p);
  var isFloor = 0.0;
  if (p.y < floorY) {
    let fp = vec2f(p.x, (floorY - p.y) * 3.0);
    alb = vec3f(0.09, 0.075, 0.07) * (0.8 + 0.4 * valueNoise(fp * 12.0)) * (0.7 + 0.3 * smoothstep(0.0, floorY, p.y));
    isFloor = 1.0;
  }
  // ---- lighting: ambient moonlight + three flickering point lights
  let fireCol = firePal(0.7) * 1.4;
  var light = vec3f(0.035, 0.04, 0.07);
  let dc = p - (cf + vec2f(0.0, 0.12));
  light += fireCol * fl0 * 1.1 * u.intensity / (1.0 + dot(dc, dc) * 16.0);
  let d1 = p - tl;
  let d2 = p - tr;
  light += fireCol * fl1 * 0.5 / (1.0 + dot(d1, d1) * 40.0);
  light += fireCol * fl2 * 0.5 / (1.0 + dot(d2, d2) * 40.0);
  var col = alb * light * 2.4;

  // ---- torches: iron bracket + flame
  for (var k = 0; k < 2; k++) {
    let tp = select(tr, tl, k == 0);
    let bd = sdBox(p - (tp - vec2f(0.0, 0.07)), vec2f(0.012, 0.05));
    col = mix(col, vec3f(0.03, 0.025, 0.02), sdfFill(bd));
    let cup = sdBox(p - (tp - vec2f(0.0, 0.012)), vec2f(0.03, 0.012));
    col = mix(col, vec3f(0.06, 0.05, 0.04), sdfFill(cup));
    let lp = (p - tp) / 0.2;
    if (abs(lp.x) < 0.6 && lp.y > -0.1 && lp.y < 1.2) {
      let th = flameHeat(lp, t * 1.3, f32(k) * 13.0, 0.17);
      col = col * (1.0 - th) + firePal(th) * 1.15 * smoothstep(0.0, 0.18, th);
    }
  }

  // ---- campfire: logs, glow, flames, embers
  let lg = p - cf;
  let log1 = sdSegment(lg, vec2f(-0.17, -0.02), vec2f(0.15, 0.03)) - 0.022;
  let log2 = sdSegment(lg, vec2f(-0.15, 0.03), vec2f(0.17, -0.02)) - 0.022;
  let logs = min(log1, log2);
  col = mix(col, vec3f(0.07, 0.035, 0.02) * (0.6 + 0.6 * fl0), sdfFill(logs));
  col += vec3f(1.0, 0.25, 0.02) * exp(-length(lg * vec2f(1.0, 3.0)) * 14.0) * 0.7 * fl0 * u.intensity;
  let fp = (p - cf) / 0.45;
  if (abs(fp.x) < 0.7 && fp.y > -0.08 && fp.y < 1.4) {
    let h = flameHeat(fp, t, 0.0, 0.3);
    col = col * (1.0 - h) + firePal(h) * 1.2 * smoothstep(0.0, 0.15, h);
  }
  // embers: hashed sparks rising from the fire (only evaluated near the fire)
  let nearFire = abs(p.x - cf.x) < 0.45 && p.y < cf.y + 0.95;
  for (var i = 0; i < 24; i++) {
    if (!nearFire) { break; }
    let fi = f32(i);
    let hh = hash13(fi * 3.7 + 11.0);
    let age = fract(t * (0.25 + 0.2 * hh.z) + hh.x);
    let ep = cf + vec2f((hh.y - 0.5) * 0.2 + sin(age * 9.0 + fi) * 0.05 + u.wind * age * 0.4, 0.1 + age * 0.7);
    let ed = length(p - ep);
    col += firePal(0.85) * exp(-ed * ed / 0.000012) * (1.0 - age) * 1.5 * step(0.15, u.intensity);
  }

  // ---- burning text: flames lick up from the letters while they burn away and regrow
  let ta = vec2f(aspect * 0.5, 0.84);
  let tsz = vec2f(0.8, 0.2);
  let tuv = vec2f((p.x - ta.x) / tsz.x + 0.5, 0.5 - (p.y - ta.y) / tsz.y);
  if (tuv.x > -0.05 && tuv.x < 1.05 && tuv.y > -0.9 && tuv.y < 1.1) {
    let m = TEX(words, tuv).r;
    // dissolve: a noise threshold sweeps up and down
    let burn = 0.5 - 0.5 * cos(u.time * 0.4);
    let dn = fbm(tuv * vec2f(9.0, 3.0), 4 - i32(u.lod) * 2) * 0.5 + 0.5 + (1.0 - tuv.y) * 0.25;
    let edge = dn - burn * 1.25 + 0.05;
    let alive = smoothstep(0.0, 0.02, edge) * m;
    let rim = (1.0 - smoothstep(0.0, 0.08, abs(edge))) * m;
    col = mix(col, vec3f(0.55, 0.42, 0.3) * (0.6 + 0.4 * dn) * light * 2.0 + vec3f(0.05, 0.03, 0.02), alive);
    col += firePal(0.9) * rim * 2.0;
    // fuel below this pixel = letters (only the unburnt parts) a little lower down
    var fuel = 0.0;
    let wv = fbm(vec2f(tuv.x * 6.0, tuv.y * 2.0 + t * 2.5), 3 - i32(u.lod)) * 0.06 * u.turbulence;
    let jit = ign(px);   // dither the sample offsets so the 8 taps don't band
    for (var s = 1; s < 9; s++) {
      let o = (f32(s) - jit) * 0.045;
      fuel = max(fuel, TEX(words, tuv + vec2f(wv - u.wind * o * 0.4, o)).r * (1.0 - f32(s) / 9.0));
    }
    let fq = vec2f(tuv.x * 7.0, tuv.y * 3.0 + t * 3.0);
    let fn2 = fbm(fq + vec2f(fbm(fq * 1.3, 2) * u.turbulence, 0.0), 4 - i32(u.lod) * 2) * 0.5 + 0.5;
    let fh = clamp(fuel * (fn2 * 1.6 - 0.35) * u.intensity * (1.4 - burn * 0.6) / u.height * 1.0 + rim * 0.4, 0.0, 1.0) * (1.0 - alive * 0.7);
    col = col * (1.0 - fh) + firePal(fh) * smoothstep(0.0, 0.2, fh) * 1.2;
  }
  // vignette
  let v = uv - 0.5;
  return col * (1.0 - dot(v, v) * 0.9);
}

fn shade(uv: vec2f, px0: vec2f) -> vec4f {
  // canvas-pixel coordinates (px0 is in render-target pixels, which differ when renderScale < 1)
  let px = uv * u.resolution;
  let ex = i32(u.example);
  var c: vec3f;
  if (ex == 0) { c = noiseFire(uv, px); }
  else if (ex == 1) { c = doomFire(uv); }
  else if (ex == 2) { c = particleFire(uv, px); }
  else { c = gameScene(uv, px); }
  return vec4f(c, 1.0);
}`,
  about: {
    summary: 'The same flame, three completely different ways: a noise shader, a 1995 cellular automaton, and particles — plus how a game dresses fire up with light and embers.',
    what: `<p><b>Noise fire</b> is a pure function of position and time. The <b>Doom fire</b> is a little simulation that remembers last frame.
      <b>Particle fire</b> is many small glowing blobs. The <b>game</b> tab combines noise flames with the details that make fire believable.</p>`,
    how: `<ol>
      <li><b>Heat, then color.</b> Every technique computes one number per pixel — heat from 0 (cold/transparent) to 1 (white hot) — and looks it up in a <b>palette</b> (color ramp). Swapping the palette turns fire into gas flames or cursed green fire for free.</li>
      <li><b>Noise fire</b>: fBm noise (several octaves of Perlin noise) is <i>scrolled upward</i> over time, <b>domain-warped</b> by another noise for turbulence, and multiplied by a vertical gradient so it is hot at the bottom and fades out at the top. A teardrop mask shapes single flames.</li>
      <li><b>Doom fire</b>: a low-resolution grid in a <b>feedback buffer</b> (the pass reads its own previous frame). The bottom row is set to max heat. Every other cell takes the heat of the cell below it, shifted left/right by a random step and cooled by one palette entry with some probability. That's the whole algorithm — about ten lines.</li>
      <li><b>Particle fire</b>: each particle has a random seed; <code>age = fract(time / life + seed)</code> makes it loop forever. Position, size and heat are functions of age. Summing soft blobs gives fire-like density.</li>
      <li><b>Selling it</b>: flicker the light intensity with low-frequency noise, light nearby surfaces with an inverse-square falloff, add embers, and let things burn away with a <i>dissolve</i> threshold plus a glowing rim.</li>
    </ol>`,
    uses: [
      { title: 'Torches & campfires', text: 'Noise fire on a small quad: no textures, infinitely varied, tiny cost — perfect for dungeon torches (Dead Cells, Hades-style ambience).' },
      { title: 'Retro & title screens', text: 'The Doom fire runs on a 320×168 grid and looks great scaled up with a palette — ideal for pixel-art menus and burning UI.' },
      { title: 'Spells & burning objects', text: 'Swap palettes for magic fire; dissolve + rim glow for paper, cards or enemies burning away (Hearthstone, Slay the Spire).' },
      { title: 'Interactive fire', text: 'Real particles (or a fluid sim) when fire must react to the world: explosions, flamethrowers, spreading grass fires (Noita).' },
    ],
    try: [
      'On <b>Noise fire</b>, enable <i>Show the layers</i>, then drag <i>Turbulence</i> to 0 and back.',
      'On <b>Doom fire</b>, set <i>Intensity</i> to 0 — the fire dies from the bottom up, exactly like the PSX title screen. Then bring it back.',
      'Push <i>Fire pixel size</i> to 1: the same automaton at full resolution looks like fine smoke.',
      'Switch every tab to the <i>Gas burner</i> or <i>Cursed</i> palette — only the color ramp changes.',
      'On <b>Particle fire</b>, enable <i>Show particles</i> and lower <i>Particles</i> to 8 to see each blob; then watch fps as you raise it.',
    ],
    ask: [
      'a scrolling fBm noise fire shader with a color ramp',
      'the Doom PSX fire effect as a feedback shader',
      'palette-swapped fire (blue gas flame, green cursed fire)',
      'flickering torch light with noise-driven intensity',
      'burning dissolve with a glowing edge for sprites or text',
      'rising embers above a campfire',
    ],
    perf: `<p><b>Noise fire</b> costs a few fBm evaluations per covered pixel — restrict it to the flame's bounding box (as done here) and it's nearly free.
      The <b>Doom fire</b> is a single texture read per cell per frame on a tiny grid: the cheapest by far. The <b>shader-particle</b> version costs
      <i>pixels × particles</i>, which explodes quickly; a real particle system only shades the pixels each particle covers.</p>`,
    api: `<p>Runs on <b>both WebGPU and WebGL2</b> from one portable-WGSL source. The Doom fire's feedback buffer is a ping-pong pair of textures —
      available in both APIs. A compute shader could update the automaton in place, but a fragment pass is just as good for a grid this small.</p>`,
    code: [
      {
        title: 'Doom fire: the whole automaton (a feedback pass)',
        lang: 'wgsl',
        src: `fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ip = vec2i(floor(px));
  if (ip.y >= i32(TEXSIZE(doom).y) - 1) { return vec4f(u.intensity, 0.0, 0.0, 1.0); } // fuel
  let r = hash33(vec3f(px, u.frame));
  let shift = i32(floor(r.x * 3.0 + u.wind * 1.4)) - 1;          // drift -1, 0, +1
  let below = LOADW(doom, vec2i(ip.x - shift, ip.y + 1)).r;      // last frame, one row down
  let cool = select(0.0, u.coolStep, r.y < 0.5);                 // 50%: lose ~1 palette step
  return vec4f(max(below - cool, 0.0), 0.0, 0.0, 1.0);
}`,
      },
      {
        title: 'Noise fire: scroll, warp, shape, palette',
        lang: 'wgsl',
        src: `var q = vec2f(x * 3.0, y * 2.2 - t * 1.8);              // scroll upward
let warp = vec2f(fbm(q * 1.1 + vec2f(0.0, -t * 0.7), 3),
                 fbm(q * 1.1 + vec2f(5.2, 1.3 - t * 0.7), 3));
q += warp * u.turbulence * 0.6;                           // domain warp = turbulence
let n = fbm(q, 5) * 0.5 + 0.5;
let grad = clamp(1.0 - y / (0.9 * u.height), 0.0, 1.0);  // hot at the bottom
let heat = clamp(n * grad * 1.9 - 0.36 + 0.3 * grad * grad, 0.0, 1.0);
let color = firePal(heat);                                 // 1D color ramp lookup`,
      },
      {
        title: 'Particle fire: particles as pure functions of time',
        lang: 'wgsl',
        src: `for (var i = 0; i < 128; i++) {
  if (i >= n) { break; }
  let h = hash13(f32(i) * 7.31 + 1.0);
  let age = fract(t / life + h.x);                // loops: born, rises, dies, reborn
  let pos = vec2f(spawn.x * (1.0 - 0.75 * age) + u.wind * age * age * 0.35 + wobble, spawn.y + age * H);
  let size = mix(0.06, 0.015, age);
  let d = length(p - pos);
  heatSum += exp(-(d * d) / (size * size)) * (1.0 - age) * (1.0 - age) * 0.36;
}`,
      },
    ],
    links: [
      { title: 'Fabien Sanglard — How DOOM fire was done', url: 'https://fabiensanglard.net/doom_fire_psx/', note: 'the original algorithm and palette' },
      { title: 'The Book of Shaders — Fractal Brownian Motion', url: 'https://thebookofshaders.com/13/' },
    ],
  },
});

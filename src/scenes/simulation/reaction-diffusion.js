import { shaderScene } from '../../core/shaderscene.js';
import { syncSlider, readout } from './_shared-a.js';

// Reaction–Diffusion (Gray–Scott). Two virtual chemicals A and B live in a float texture.
// Each step:  A' = A + (dA·∇²A − A·B² + f·(1−A))
//             B' = B + (dB·∇²B + A·B² − (k+f)·B)
// The state uses rgba32float (exact texel reads via LOAD) because half floats lose the tiny
// per-step changes and the patterns stall.

const PRESETS = {
  coral: [0.0545, 0.062],
  mitosis: [0.0367, 0.0649],
  spots: [0.03, 0.062],
  stripes: [0.037, 0.06],
  worms: [0.078, 0.061],
  maze: [0.029, 0.057],
  holes: [0.039, 0.058],
  waves: [0.014, 0.045],
};
const PRESET_NAMES = {
  coral: 'Coral growth',
  mitosis: 'Mitosis (dividing cells)',
  spots: 'Spots / solitons',
  stripes: 'Stripes / fingerprints',
  worms: 'Worms',
  maze: 'Labyrinth / maze',
  holes: 'Holes / Swiss cheese',
  waves: 'Pulsing waves (chaotic)',
};
// [feed, kill, anisotropy (0.5 = none), dB]
const ANIMALS = {
  zebra: [0.034, 0.0605, 0.82, 0.5],
  leopard: [0.0302, 0.0625, 0.5, 0.5],
  giraffe: [0.0367, 0.0649, 0.5, 0.42],
  puffer: [0.029, 0.057, 0.5, 0.5],
};
// feed × kill map: x = kill; y = feed, measured between a lower and upper line that bracket the
// band where patterns exist (the plain rectangle wastes most of the screen on dead/flooded areas)
const MAP = { k0: 0.047, k1: 0.068 };
const fLo = (k) => Math.max(0.004, (k - 0.041) * 1.1);
const fHi = (k) => 0.016 + (k - 0.047) * 4.4;

const CONTROLS = [
  { type: 'heading', label: 'Chemistry' },
  {
    type: 'select',
    key: 'preset',
    label: 'Preset (sets feed & kill)',
    value: 'coral',
    options: [...Object.entries(PRESET_NAMES).map(([value, label]) => ({ value, label })), { value: 'custom', label: 'Custom (use the sliders)' }],
    showFor: ['lab', 'growth'],
  },
  { type: 'slider', key: 'feed', label: 'Feed rate f', min: 0.005, max: 0.1, step: 0.0001, value: 0.0545, help: 'How fast A is replenished.', showFor: ['lab', 'growth'] },
  { type: 'slider', key: 'kill', label: 'Kill rate k', min: 0.04, max: 0.075, step: 0.0001, value: 0.062, help: 'How fast B is removed. Tiny changes give completely different patterns.', showFor: ['lab', 'growth'] },
  {
    type: 'select',
    key: 'animal',
    label: 'Animal',
    value: 'zebra',
    options: [
      { value: 'zebra', label: 'Zebra (stripes)' },
      { value: 'leopard', label: 'Leopard (rosettes)' },
      { value: 'giraffe', label: 'Giraffe (patches)' },
      { value: 'puffer', label: 'Pufferfish (maze)' },
    ],
    showFor: ['skins'],
  },
  { type: 'slider', key: 'dB', label: 'Diffusion of B', min: 0.25, max: 0.75, step: 0.01, value: 0.5, help: 'Relative to A (=1). Turing patterns need B to spread slower than A.' },
  { type: 'heading', label: 'Simulation' },
  { type: 'slider', key: 'steps', label: 'Steps per frame', min: 1, max: 40, step: 1, value: 14, help: 'Simulation speed. Each step is one full-screen shader pass.' },
  { type: 'slider', key: 'zoom', label: 'Cell size', min: 1, max: 4, step: 1, value: 2, unit: 'px', help: 'Pixels per simulation cell — bigger cells = bigger pattern, faster sim.' },
  { type: 'slider', key: 'brush', label: 'Brush size', min: 2, max: 40, step: 1, value: 10, unit: 'px' },
  { type: 'heading', label: 'Look' },
  {
    type: 'select',
    key: 'palette',
    label: 'Colors',
    value: 'reef',
    options: [
      { value: 'reef', label: 'Coral reef' },
      { value: 'glow', label: 'Bioluminescent' },
      { value: 'ink', label: 'Ink on paper' },
      { value: 'heat', label: 'Heat' },
    ],
    showFor: ['lab', 'map'],
  },
  { type: 'toggle', key: 'light', label: 'Fake 3D lighting', value: true, help: 'Treat B as a height map: normals from its slope, then diffuse + specular light.' },
  { type: 'button', key: 'reset', label: 'Reseed', primary: true },
];

// Shared by the simulation pass and the final image
const COMMON = /* wgsl */ `
// (feed, kill) at a point of the parameter map
fn mapFK(uv: vec2f) -> vec2f {
  let k = mix(${MAP.k0}, ${MAP.k1}, uv.x);
  let lo = max(0.004, (k - 0.041) * 1.1);
  let hi = 0.016 + (k - 0.047) * 4.4;
  return vec2f(mix(hi, lo, uv.y), k);
}
fn mapUV(f: f32, k: f32) -> vec2f {
  let lo = max(0.004, (k - 0.041) * 1.1);
  let hi = 0.016 + (k - 0.047) * 4.4;
  return vec2f((k - ${MAP.k0}) / (${MAP.k1} - ${MAP.k0}), 1.0 - (f - lo) / (hi - lo));
}
// level layout for the "creep" example: rooms + corridors as box SDFs (negative = floor)
fn levelSDF(uv: vec2f, res: vec2f) -> f32 {
  let a = res.x / res.y;
  let q = (uv - 0.5) * vec2f(a, 1.0);
  var d = sdBox(q - vec2f(-0.55, -0.2), vec2f(0.2, 0.15));
  d = min(d, sdBox(q - vec2f(0.03, -0.25), vec2f(0.16, 0.12)));
  d = min(d, sdBox(q - vec2f(0.55, -0.15), vec2f(0.19, 0.2)));
  d = min(d, sdBox(q - vec2f(-0.45, 0.27), vec2f(0.22, 0.12)));
  d = min(d, sdBox(q - vec2f(0.22, 0.27), vec2f(0.24, 0.11)));
  d = min(d, sdBox(q - vec2f(-0.27, -0.22), vec2f(0.14, 0.035)));
  d = min(d, sdBox(q - vec2f(0.3, -0.21), vec2f(0.12, 0.035)));
  d = min(d, sdBox(q - vec2f(-0.5, 0.05), vec2f(0.035, 0.12)));
  d = min(d, sdBox(q - vec2f(-0.12, 0.28), vec2f(0.12, 0.035)));
  d = min(d, sdBox(q - vec2f(0.6, 0.13), vec2f(0.035, 0.12)));
  return d;
}
`;

let lastPreset = null;
let lastAnimal = null;
let isTest = false;
let hoverTag = null;
let hoverCtx = null;

export default shaderScene({
  interaction: 'Drag to drop chemical B and start new growth. Right-drag erases.',
  examples: [
    {
      id: 'lab',
      label: 'Pattern lab',
      kind: 'Abstract',
      note: 'Pick a <b>preset</b> (it just sets <i>feed</i> and <i>kill</i>) and watch the same two equations grow coral, dividing cells, stripes or worms. Draw with the mouse to seed new growth.',
      params: { preset: 'coral', feed: 0.0545, kill: 0.062, dB: 0.5, zoom: 2, palette: 'reef', light: true },
    },
    {
      id: 'map',
      label: 'Feed × kill map',
      kind: 'Abstract',
      note: 'Here <i>feed</i> changes from top (high) to bottom (low) and <i>kill</i> from left (low) to right (high), so the whole “parameter space” is visible at once. Circles mark the presets. Hover to read <i>f</i> and <i>k</i> at the cursor.',
      params: { dB: 0.5, zoom: 2, palette: 'glow', light: true },
    },
    {
      id: 'skins',
      label: 'Animal skins',
      kind: 'Real life',
      note: 'Alan Turing proposed in 1952 that animal coats are made by exactly this kind of chemistry. Zebra uses <i>anisotropic</i> diffusion (B spreads faster in one direction) so stripes line up; the leopard’s rosettes are just spots colored by concentration.',
      params: { animal: 'zebra', dB: 0.5, zoom: 2, light: true },
    },
    {
      id: 'growth',
      label: 'Alien creep',
      kind: 'In a game',
      note: 'An infestation spreading through a sci-fi base, like StarCraft’s creep or a corruption mechanic. The walls are simply cells where the chemistry is switched off. Click anywhere on the floor to start a new outbreak.',
      params: { preset: 'coral', feed: 0.0545, kill: 0.062, dB: 0.5, zoom: 2, light: true, steps: 16 },
    },
  ],
  controls: CONTROLS,
  uniforms: {
    feed: 'f32',
    kill: 'f32',
    dB: 'f32',
    brush: 'f32',
    light: 'f32',
    palette: 'f32',
    animal: 'f32',
    aniso: 'f32',
    erase: 'f32',
    zoomPx: 'f32',
  },
  include: ['math', 'hash', 'noise', 'sdf', 'color'],
  resetOn: ['animal'],
  bind(params, ctx) {
    isTest = ctx.testMode;
    const out = {};
    // presets/animals set several sliders at once
    if (params.preset !== lastPreset) {
      lastPreset = params.preset;
      const pr = PRESETS[params.preset];
      if (pr) {
        params.feed = pr[0];
        params.kill = pr[1];
        syncSlider(CONTROLS, 'feed', pr[0]);
        syncSlider(CONTROLS, 'kill', pr[1]);
      }
    }
    let aniso = 0.5;
    if (ctx.example === 'skins') {
      const a = ANIMALS[params.animal] || ANIMALS.zebra;
      out.feed = a[0];
      out.kill = a[1];
      aniso = a[2];
      out.dB = a[3];
      if (params.animal !== lastAnimal) lastAnimal = params.animal;
    }
    out.aniso = aniso;
    out.erase = ctx.pointer.down && ctx.pointer.button === 2 ? 1 : 0;
    out.zoomPx = isTest ? 4 : params.zoom;

    // hover readout for the parameter map
    if (hoverCtx !== ctx) {
      hoverCtx = ctx;
      hoverTag = readout(ctx);
    }
    if (ctx.example === 'map') {
      const k = MAP.k0 + (MAP.k1 - MAP.k0) * ctx.pointer.nx;
      const f = fHi(k) + (fLo(k) - fHi(k)) * ctx.pointer.ny;
      let best = null;
      let bd = 1e9;
      for (const [name, [pf, pk]] of Object.entries(PRESETS)) {
        const py = 1 - (pf - fLo(pk)) / (fHi(pk) - fLo(pk));
        const px = (pk - MAP.k0) / (MAP.k1 - MAP.k0);
        const d = Math.hypot(px - ctx.pointer.nx, py - ctx.pointer.ny);
        if (d < bd) {
          bd = d;
          best = name;
        }
      }
      hoverTag.style.display = '';
      hoverTag.textContent = `f = ${f.toFixed(4)} · k = ${k.toFixed(4)}${bd < 0.06 ? ` · ≈ ${PRESET_NAMES[best]}` : ''}`;
    } else {
      const fk = ctx.example === 'skins' ? [out.feed, out.kill] : [params.feed, params.kill];
      hoverTag.textContent = `f = ${fk[0].toFixed(4)} · k = ${fk[1].toFixed(4)}`;
    }
    return out;
  },
  passes: [
    {
      name: 'state',
      format: 'rgba32float',
      filter: 'nearest',
      size: (p, ctx) => {
        const z = ctx.testMode ? 4 : p.zoom || 2;
        return [Math.max(8, Math.round(ctx.width / z)), Math.max(8, Math.round(ctx.height / z))];
      },
      iterations: (p) => (isTest ? 32 : Math.round(p.steps)),
      code:
        COMMON +
        /* wgsl */ `
fn seedB(px: vec2f, size: vec2f, ex: i32) -> f32 {
  if (ex == 1 || ex == 2) {
    // dense noise everywhere so every region of the map / skin develops
    let h = hash21(floor(px / 6.0));
    return step(0.8, h) * 0.6;
  }
  if (ex == 3) {
    // three outbreaks in different rooms
    let a = size.x / size.y;
    let q = (px / size - 0.5) * vec2f(a, 1.0);
    var d = length(q - vec2f(-0.55, -0.2));
    d = min(d, length(q - vec2f(0.55, -0.12)));
    d = min(d, length(q - vec2f(0.25, 0.27)));
    return step(d, 0.02) * step(0.3, hash21(floor(px)));
  }
  // lab: random blobs
  let cell = floor(px / 34.0);
  let h = hash22(cell);
  let center = (cell + 0.2 + 0.6 * h) * 34.0;
  let r = 2.5 + 4.0 * hash21(cell + 7.0);
  return step(length(px - center), r) * step(0.45, hash21(cell + 3.0));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let size = TEXSIZE(state);
  let ex = i32(u.example + 0.5);
  let p = vec2i(floor(px));
  var wall = 0.0;
  if (ex == 3) { wall = step(0.0, levelSDF(uv, u.resolution)); }
  if (u.frame < 0.5) {
    let b = seedB(px, size, ex) * (1.0 - wall);
    return vec4f(1.0 - b * 0.5, b, 0.0, 1.0);
  }
  let c = LOADW(state, p);
  let L = LOADW(state, p + vec2i(-1, 0)).xy;
  let R = LOADW(state, p + vec2i(1, 0)).xy;
  let T = LOADW(state, p + vec2i(0, -1)).xy;
  let B = LOADW(state, p + vec2i(0, 1)).xy;
  var lap = vec2f(0.0);
  if (abs(u.aniso - 0.5) < 0.01) {
    // isotropic 3×3 Laplacian (Karl Sims' weights: edges 0.2, corners 0.05, centre −1)
    let D = LOADW(state, p + vec2i(-1, -1)).xy + LOADW(state, p + vec2i(1, -1)).xy + LOADW(state, p + vec2i(-1, 1)).xy + LOADW(state, p + vec2i(1, 1)).xy;
    lap = 0.2 * (L + R + T + B) + 0.05 * D - c.xy;
  } else {
    // anisotropic: diffuse more along one axis; the preferred axis slowly bends across the skin
    // (the diagonal term keeps the checkerboard mode damped, like the isotropic kernel)
    let w = mix(0.5, u.aniso, 0.6 + 0.4 * sin(uv.y * 5.0 + 2.0 * sin(uv.x * 3.0)));
    let D = LOADW(state, p + vec2i(-1, -1)).xy + LOADW(state, p + vec2i(1, -1)).xy + LOADW(state, p + vec2i(-1, 1)).xy + LOADW(state, p + vec2i(1, 1)).xy;
    lap = 0.4 * (w * (L + R - 2.0 * c.xy) + (1.0 - w) * (T + B - 2.0 * c.xy)) + 0.05 * (D - 4.0 * c.xy);
  }
  var f = u.feed;
  var k = u.kill;
  if (ex == 1) {
    let fk = mapFK(uv);
    f = fk.x;
    k = fk.y;
  }
  let a = c.x;
  let b = c.y;
  let r = a * b * b;
  var na = a + (1.0 * lap.x - r + f * (1.0 - a));
  var nb = b + (u.dB * lap.y + r - (k + f) * b);
  var age = c.z;
  if (nb > 0.2) { age = min(age + 0.0015, 1.0); }

  // mouse brush (segment from last to current position, in simulation cells)
  if (u.mouse.z > 0.5) {
    let s = size / u.resolution;
    let d = sdSegment(px, u.pmouse * s, u.mouse.xy * s);
    if (d < u.brush / u.zoomPx) {
      if (u.erase > 0.5) { na = 1.0; nb = 0.0; age = 0.0; }
      else { nb = max(nb, 0.5 + 0.4 * hash21(px + u.time)); na = min(na, 0.5); }
    }
  }
  if (wall > 0.5) { na = 1.0; nb = 0.0; }
  return vec4f(clamp(na, 0.0, 1.0), clamp(nb, 0.0, 1.0), age, 1.0);
}`,
    },
  ],
  code:
    COMMON +
    /* wgsl */ `
// The state is rgba32float: not filterable, so do bilinear interpolation by hand.
fn stateAt(uv: vec2f) -> vec4f {
  let size = TEXSIZE(state);
  let t = uv * size - 0.5;
  let i = vec2i(floor(t));
  let f = fract(t);
  let a = LOADW(state, i);
  let b = LOADW(state, i + vec2i(1, 0));
  let c = LOADW(state, i + vec2i(0, 1));
  let d = LOADW(state, i + vec2i(1, 1));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn rampHeat(t: f32) -> vec3f {
  return clamp(vec3f(1.6 * t, 1.6 * t - 0.45, 3.0 * t - 2.0), vec3f(0.0), vec3f(1.0)) + vec3f(0.03, 0.02, 0.08) * (1.0 - t);
}

fn presetDot(uv: vec2f, fk: vec2f) -> f32 {
  let d = length((uv - mapUV(fk.x, fk.y)) * u.resolution);
  return 1.0 - smoothstep(1.0, 2.2, abs(d - 9.0));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example + 0.5);
  let s = stateAt(uv);
  let b = s.y;
  // slope of B → normal → lighting (B treated as a height map)
  let e = 1.0 / TEXSIZE(state);
  let bx = stateAt(uv + vec2f(e.x, 0.0)).y - b;
  let by = stateAt(uv + vec2f(0.0, e.y)).y - b;
  let n = normalize(vec3f(-bx * 6.0, -by * 6.0, 1.0));
  let Ld = normalize(vec3f(-0.45, -0.6, 0.65));
  let diff = max(dot(n, Ld), 0.0);
  let spec = pow(max(dot(reflect(-Ld, n), vec3f(0.0, 0.0, 1.0)), 0.0), 24.0);
  let lit = mix(1.0, 0.55 + 0.75 * diff, u.light);
  let sp = spec * u.light;
  let v = smoothstep(0.08, 0.42, b);
  let edge = clamp(length(vec2f(bx, by)) * 14.0, 0.0, 1.0);
  var col = vec3f(0.0);

  if (ex == 2) {
    let an = i32(u.animal + 0.5);
    let fur = 0.92 + 0.08 * valueNoise(px * vec2f(0.6, 0.12));
    if (an == 0) {
      let stripe = smoothstep(0.17, 0.25, b);
      col = mix(vec3f(0.93, 0.91, 0.86), vec3f(0.05, 0.045, 0.05), stripe) * fur;
    } else if (an == 1) {
      let base = mix(vec3f(0.93, 0.7, 0.36), vec3f(0.8, 0.55, 0.25), valueNoise(px * 0.01));
      let rim = smoothstep(0.1, 0.18, b);
      let core = smoothstep(0.27, 0.33, b);
      col = mix(base, vec3f(0.12, 0.08, 0.04), rim);
      col = mix(col, vec3f(0.68, 0.42, 0.17), core);
      col *= fur;
    } else if (an == 2) {
      let patch_ = smoothstep(0.13, 0.2, b);
      let tone = mix(vec3f(0.62, 0.34, 0.12), vec3f(0.45, 0.22, 0.08), smoothstep(0.25, 0.4, b));
      col = mix(vec3f(0.95, 0.9, 0.76), tone, patch_) * fur;
    } else {
      let m = smoothstep(0.16, 0.24, b);
      col = mix(vec3f(0.98, 0.84, 0.2), vec3f(0.06, 0.07, 0.06), m) * fur;
    }
    col = col * mix(1.0, 0.8 + 0.3 * diff, u.light) + vec3f(sp * 0.12);
    return vec4f(col, 1.0);
  }

  if (ex == 3) {
    // sci-fi floor + walls, creep on top
    let ld = levelSDF(uv, u.resolution);
    let tile = fract(px / 40.0);
    let seam = 1.0 - smoothstep(0.0, 0.04, min(min(tile.x, 1.0 - tile.x), min(tile.y, 1.0 - tile.y)));
    let tid = floor(px / 40.0);
    var floorC = vec3f(0.16, 0.18, 0.21) * (0.85 + 0.15 * hash21(tid)) * (0.93 + 0.07 * valueNoise(px * 0.2));
    floorC = mix(floorC, vec3f(0.07, 0.08, 0.1), seam * 0.8);
    let rivet = 1.0 - smoothstep(1.2, 2.2, length((tile - vec2f(0.12)) * 40.0));
    floorC += vec3f(0.12) * rivet;
    let wallC = vec3f(0.045, 0.05, 0.065) + vec3f(0.06, 0.07, 0.09) * smoothstep(0.012, 0.0, ld) * step(0.0, ld);
    col = mix(floorC, wallC, smoothstep(-0.001, 0.001, ld));
    // wall edge shadow on the floor
    col *= mix(1.0, 0.55, smoothstep(-0.02, 0.0, ld) * step(ld, 0.0));
    // creep
    let age = s.z;
    let creep = mix(vec3f(0.45, 0.06, 0.38), vec3f(0.18, 0.02, 0.2), age) * lit + vec3f(1.0, 0.35, 0.9) * edge * 0.8;
    col = mix(col, creep, smoothstep(0.08, 0.2, b));
    col += vec3f(0.9, 0.2, 0.8) * sp * 0.5 * v;
    // pulsing glow along the growth front
    let front = smoothstep(0.05, 0.15, b) * (1.0 - smoothstep(0.15, 0.3, b));
    col += vec3f(1.0, 0.3, 0.7) * front * (0.5 + 0.5 * sin(u.time * 4.0 + px.x * 0.05));
    return vec4f(col, 1.0);
  }

  let pal = i32(u.palette + 0.5);
  if (pal == 0) {
    let deep = mix(vec3f(0.02, 0.06, 0.14), vec3f(0.01, 0.03, 0.08), uv.y);
    let coral = mix(vec3f(0.95, 0.35, 0.35), vec3f(1.0, 0.75, 0.45), smoothstep(0.25, 0.45, b));
    col = mix(deep, coral * lit, v) + vec3f(sp * 0.4) * v;
  } else if (pal == 1) {
    let glow = mix(vec3f(0.0, 0.25, 0.35), vec3f(0.4, 1.0, 0.9), v);
    col = vec3f(0.005, 0.01, 0.03) + glow * v * lit + vec3f(0.5, 1.0, 0.9) * edge * 0.35 + vec3f(sp * 0.35) * v;
  } else if (pal == 2) {
    let paper = vec3f(0.95, 0.93, 0.87) * (0.96 + 0.04 * valueNoise(px * 0.4));
    col = mix(paper, vec3f(0.06, 0.06, 0.09), v) * mix(1.0, lit, 0.5);
  } else {
    col = rampHeat(smoothstep(0.0, 0.45, b)) * lit + vec3f(sp * 0.3) * v;
  }

  if (ex == 1) {
    // preset markers on the parameter map
    var m = 0.0;
    m = max(m, presetDot(uv, vec2f(0.0545, 0.062)));
    m = max(m, presetDot(uv, vec2f(0.0367, 0.0649)));
    m = max(m, presetDot(uv, vec2f(0.03, 0.062)));
    m = max(m, presetDot(uv, vec2f(0.037, 0.06)));
    m = max(m, presetDot(uv, vec2f(0.078, 0.061)));
    m = max(m, presetDot(uv, vec2f(0.029, 0.057)));
    m = max(m, presetDot(uv, vec2f(0.039, 0.058)));
    m = max(m, presetDot(uv, vec2f(0.014, 0.045)));
    col = mix(col, vec3f(1.0, 0.85, 0.3), m * 0.9);
    // cursor crosshair
    let dd = abs(px - u.mouse.xy);
    col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.0, 1.0, min(dd.x, dd.y))) * u.mouse.w * 0.35);
  }
  return vec4f(col, 1.0);
}`,
  about: {
    summary:
      'Two imaginary chemicals that spread at different speeds and react with each other. That’s all — yet they paint coral, fingerprints, leopard spots and dividing cells. Runs as a feedback shader on both WebGPU and WebGL2.',
    what: `<p>Every cell of a grid stores the concentrations of two chemicals, <b>A</b> and <b>B</b>. Each frame the grid is updated many times by a
      fragment shader that reads the previous state (a <i>feedback</i> or “ping-pong” buffer). The colors are just B, run through a palette and lit as if B were height.</p>`,
    how: `<ol>
      <li><b>Diffusion</b>: each chemical spreads to its neighbours. The <i>Laplacian</i> ∇² (neighbours minus centre, from a 3×3 kernel) measures how different a cell is from its surroundings.</li>
      <li><b>Reaction</b>: <code>A + 2B → 3B</code>. Where both are present, B eats A and multiplies (the <code>A·B²</code> term).</li>
      <li><b>Feed</b> pours fresh A in everywhere (<code>f·(1−A)</code>); <b>kill</b> removes B (<code>(k+f)·B</code>).</li>
      <li>Because B diffuses <i>slower</i> than A, a blob of B can’t get enough A in its middle, starves there and grows at its edge → spots split, fronts break into stripes. This is a <b>Turing instability</b>.</li>
      <li>The full update in one line: <code>B' = B + dB·∇²B + A·B² − (k+f)·B</code>. Run it 10–40 times per frame and patterns grow in seconds.</li>
      <li>Precision matters: the per-step changes are tiny, so the state uses 32-bit float textures read with exact texel fetches (half floats make patterns stall).</li>
    </ol>`,
    uses: [
      { title: 'Organic textures', text: 'Bake coral, skin, camouflage, cracked ground or alien surfaces into textures — or animate them live.' },
      { title: 'Spreading mechanics', text: 'Creep, corruption, fungus, mold or plague that grows over a level and avoids walls.' },
      { title: 'Generative art & UI', text: 'Hypnotic menu backgrounds, transitions that “grow” the next screen, title cards.' },
      { title: 'Biology', text: 'Used in research to explain stripes, spots, fingerprints and even seashell patterns.' },
    ],
    try: [
      'In <b>Pattern lab</b>, switch between <i>Coral</i>, <i>Mitosis</i> and <i>Worms</i> without reseeding: the existing pattern morphs into the new one.',
      'Move <i>Kill rate</i> by just 0.002 — the pattern type can change completely. The <b>Feed × kill map</b> shows why.',
      'Set <i>Diffusion of B</i> to 0.75: stripes and spots fall apart because B now spreads nearly as fast as A.',
      'On <b>Animal skins</b>, try all four animals; each is a different (f, k) plus a different color mapping.',
      'Turn off <i>Fake 3D lighting</i> to see how much of the look comes from treating B as a height map.',
    ],
    ask: [
      'a Gray–Scott reaction–diffusion shader with presets',
      'organic spreading creep that grows around walls',
      'procedural animal-skin textures (Turing patterns)',
      'ping-pong feedback buffers in a fragment shader',
      'fake lighting from a height map’s gradient',
    ],
    perf: `<p>Each step is one full-screen pass with 9 texture reads per cell. Cost = <i>cells × steps per frame</i>.
      With 2-pixel cells at 1080p that’s ~0.5M cells × 14 steps ≈ 7M cell updates per frame — easy for a GPU. Larger cells or fewer steps make it cheaper.</p>`,
    api: `<p><b>Runs on both WebGPU and WebGL2</b> (switch with the API toggle). It’s a pure fragment-shader feedback loop, which both APIs do equally well.
      The only requirement is rendering into 32-bit float textures (<code>EXT_color_buffer_float</code> in WebGL2, standard in WebGPU).
      A compute version could keep the grid in workgroup shared memory to save texture reads.</p>`,
    code: [
      {
        title: 'One Gray–Scott step (fragment shader)',
        lang: 'wgsl',
        src: `let c = LOADW(state, p);                       // previous A, B
let D = LOADW(state, p + vec2i(-1, -1)).xy + …;  // diagonals
lap = 0.2 * (L + R + T + B) + 0.05 * D - c.xy;   // 3×3 Laplacian
let a = c.x;
let b = c.y;
let r = a * b * b;                               // reaction A + 2B → 3B
var na = a + (1.0 * lap.x - r + f * (1.0 - a));
var nb = b + (u.dB * lap.y + r - (k + f) * b);
return vec4f(clamp(na, 0.0, 1.0), clamp(nb, 0.0, 1.0), age, 1.0);`,
      },
      {
        title: 'Declaring the feedback pass',
        lang: 'js',
        src: `passes: [{
  name: 'state',
  format: 'rgba32float',        // full precision; read with LOAD()
  size: (p, ctx) => [ctx.width / p.zoom, ctx.height / p.zoom],
  iterations: (p) => p.steps,   // many steps per frame
  code: /* the step above */,
}]`,
      },
    ],
    links: [
      { title: 'Karl Sims — Reaction-Diffusion Tutorial', url: 'https://www.karlsims.com/rd.html' },
      { title: 'Robert Munafo — Xmorphia (Pearson’s parameter map)', url: 'http://www.mrob.com/pub/comp/xmorphia/' },
    ],
  },
});

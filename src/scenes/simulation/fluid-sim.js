// Fluid Simulation — "Stable Fluids" (Jos Stam 1999, GPU Gems ch. 38) in WebGPU compute shaders.
// Grids are rgba16float textures; every step is a compute dispatch that reads one texture and
// writes another (ping-pong). Per frame:
//   splat forces/buoyancy → curl → vorticity confinement → (viscosity) → divergence →
//   pressure (Jacobi × N) → subtract gradient → advect velocity → splat dye → advect dye

import { readout, computeSeq } from './_shared-a.js';

const MAX_SPLATS = 8;

// Ink colors as they look at full strength on white; stored as absorbance (-ln color).
const INKS = [
  [0.12, 0.2, 0.6],
  [0.75, 0.07, 0.16],
  [0.04, 0.42, 0.48],
  [0.95, 0.62, 0.05],
  [0.1, 0.09, 0.12],
];
const PAINTS = [
  [0.07, 0.24, 0.62],
  [0.93, 0.89, 0.8],
  [0.85, 0.22, 0.16],
  [0.98, 0.72, 0.12],
  [0.05, 0.45, 0.42],
  [0.12, 0.1, 0.16],
];

const hsv = (h, s, v) => {
  const f = (n) => {
    const k = (n + h * 6) % 6;
    return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [f(5), f(3), f(1)];
};

export default {
  interaction: 'Drag across the canvas to push the fluid and inject color.',
  examples: [
    {
      id: 'ink',
      label: 'Ink in water',
      kind: 'Real life',
      note: 'Drops of ink are slightly heavier than water, so they sink and roll up into mushroom-shaped plumes (a Rayleigh–Taylor instability). Colors mix <i>subtractively</i>: the dye stores how much light each ink absorbs, like real pigment.',
      params: { velDiss: 0.15, dyeDiss: 0.02, vorticity: 18, buoyancy: -1.2, iterations: 30, splat: 2.2, force: 1 },
    },
    {
      id: 'smoke',
      label: 'Campfire smoke',
      kind: 'In a game',
      note: 'A heat source injects temperature and soot. Hot fluid is lighter, so <b>buoyancy</b> lifts it; vorticity confinement keeps the curls crisp. The fire itself is the temperature field drawn through a blackbody color ramp. Drag to blow the smoke around.',
      params: { velDiss: 0.4, dyeDiss: 0.3, vorticity: 30, buoyancy: 1.6, iterations: 25, splat: 3, force: 1 },
    },
    {
      id: 'paint',
      label: 'Paint marbling',
      kind: 'Art',
      note: 'Ebru-style paper marbling: drops of paint float on a bath and are combed into feathery patterns. Paint barely fades and is <i>pushed</i>, not added, so colors stay vivid. Lighting comes from the slope of the paint layer.',
      params: { velDiss: 0.6, dyeDiss: 0, vorticity: 4, buoyancy: 0, iterations: 30, splat: 3.5, force: 0.8 },
    },
    {
      id: 'fields',
      label: 'Under the hood',
      kind: 'Abstract',
      note: 'The hidden grids. <b>Velocity</b> (arrows; hue = direction), <b>pressure</b>, <b>divergence</b> (how much fluid would be created/destroyed — the pressure solve removes it) and <b>curl</b> (spin). Switch <i>View</i> to compare them.',
      params: { velDiss: 0.4, dyeDiss: 0.6, vorticity: 20, buoyancy: 0, iterations: 25, splat: 2.5, force: 1, view: 'velocity' },
    },
  ],
  controls: [
    { type: 'heading', label: 'Fluid' },
    { type: 'slider', key: 'velDiss', label: 'Velocity dissipation', min: 0, max: 3, step: 0.01, value: 0.15, help: 'How fast motion dies out (per second). A cheap stand-in for friction.' },
    { type: 'slider', key: 'viscosity', label: 'Viscosity', min: 0, max: 40, step: 0.5, value: 0, help: 'Real viscous diffusion: velocity is blurred into its neighbours. High = honey.' },
    { type: 'slider', key: 'dyeDiss', label: 'Dye fade', min: 0, max: 2, step: 0.01, value: 0.02, help: 'How fast the color fades (per second).' },
    { type: 'slider', key: 'vorticity', label: 'Vorticity confinement', min: 0, max: 60, step: 0.5, value: 18, help: 'Puts back the small swirls that numerical blur removes.' },
    { type: 'slider', key: 'buoyancy', label: 'Buoyancy', min: -4, max: 6, step: 0.05, value: -1.2, help: 'Positive: hot/light fluid rises (smoke). Negative: heavy dye sinks (ink).', showFor: ['ink', 'smoke'] },
    { type: 'heading', label: 'Solver' },
    { type: 'slider', key: 'iterations', label: 'Pressure iterations', min: 1, max: 80, step: 1, value: 30, help: 'Jacobi iterations per frame. Too few = fluid looks compressible and “bouncy”.' },
    { type: 'toggle', key: 'sharp', label: 'Sharp advection (MacCormack)', value: true, help: 'Off = plain semi-Lagrangian: stable but blurry. On = correct the blur using a forward+backward trace.' },
    { type: 'slider', key: 'res', label: 'Resolution', min: 48, max: 384, step: 16, value: 160, help: 'Velocity grid cells across the short side. The dye grid is 4× finer.' },
    { type: 'heading', label: 'Mouse' },
    { type: 'slider', key: 'splat', label: 'Splat radius', min: 0.5, max: 10, step: 0.1, value: 2.2, unit: '%', help: 'Size of the injected force/dye blob (% of screen height).' },
    { type: 'slider', key: 'force', label: 'Splat force', min: 0.1, max: 3, step: 0.05, value: 1 },
    {
      type: 'select',
      key: 'view',
      label: 'View',
      value: 'auto',
      options: [
        { value: 'auto', label: 'Dye (the look)' },
        { value: 'velocity', label: 'Velocity' },
        { value: 'pressure', label: 'Pressure' },
        { value: 'divergence', label: 'Divergence (before projection)' },
        { value: 'curl', label: 'Curl / vorticity' },
      ],
    },
    { type: 'button', key: 'reset', label: 'Reset fluid', primary: true },
  ],
  about: {
    summary:
      'A real-time Navier–Stokes solver (Jos Stam’s “Stable Fluids”): velocity and dye live on grids, and a handful of compute passes per frame make them swirl, mix and stay incompressible.',
    what: `<p>Two grids stored in GPU textures: a coarse <b>velocity</b> grid (which way the fluid moves at each cell) and a finer <b>dye</b> grid
      (the color/smoke/ink that the fluid carries). Everything you see is the dye being pushed around by the velocity field.</p>`,
    how: `<ol>
      <li><b>Add forces</b>: the mouse adds a Gaussian “splat” of velocity and dye. Smoke adds temperature; <b>buoyancy</b> turns temperature (or ink density) into an up/down force.</li>
      <li><b>Vorticity confinement</b>: compute the <i>curl</i> (local spin) and push velocity around high-curl spots. This restores small eddies that the grid blurs away.</li>
      <li><b>Projection</b> — the heart of the method. Real liquids and slow gases are <i>incompressible</i>: fluid can’t pile up or vanish.
        Compute the <b>divergence</b> (net outflow of each cell), solve for a <b>pressure</b> field whose gradient cancels it
        (a Poisson equation, solved with <b>Jacobi iterations</b>: each cell repeatedly becomes the average of its neighbours minus its divergence),
        then subtract the pressure gradient from the velocity.</li>
      <li><b>Advection</b> (semi-Lagrangian): for each cell, trace <i>backwards</i> along the velocity by one time step and sample what was there
        (bilinear filtering does the interpolation for free). This is unconditionally stable — that’s the “stable” in Stable Fluids.</li>
      <li>Dissipation multiplies velocity/dye by e<sup>−k·dt</sup> each step. Every step is a compute dispatch that reads one texture and writes another (ping-pong).</li>
    </ol>`,
    uses: [
      { title: 'Smoke, fire & magic', text: 'Smoke trails, spell effects and explosions that react to wind and characters — the kind of thing you see in Noita or Hollow Knight’s ambient smoke.' },
      { title: 'Interactive water/ink', text: 'Puddles you stir, rivers that carry debris, ink-wash art styles, liquid UI backgrounds.' },
      { title: 'Velocity fields for particles', text: 'Even if you never draw the dye, the velocity grid is great for pushing thousands of particles (leaves, sparks, dust) realistically.' },
      { title: 'Art toys', text: 'Marbling, paint-mixing and “fluid screensaver” apps are exactly this algorithm.' },
    ],
    try: [
      'Set <i>Pressure iterations</i> to 1: the fluid no longer stays incompressible — it squishes and bounces. Then raise it to 60.',
      'On <b>Campfire smoke</b>, set <i>Vorticity confinement</i> to 0 and compare the smooth, boring plume with the curly one at 40.',
      'On <b>Ink in water</b>, set <i>Buoyancy</i> to +2: the ink now rises like smoke.',
      'Switch <i>View</i> to <b>Divergence</b> and drag quickly: red/blue blobs appear where your splat compressed the fluid — that is what the pressure solve removes.',
      'Crank <i>Viscosity</i> on <b>Paint marbling</b> to feel the difference between water and honey.',
    ],
    ask: [
      'a GPU stable-fluids smoke simulation with buoyancy',
      'vorticity confinement to keep smoke curly',
      'an interactive fluid background that reacts to the mouse',
      'use a fluid velocity field to move particles',
      'ink-in-water / marbling effect for my menu',
    ],
    perf: `<p>Per frame ≈ 10 + N (pressure iterations) compute dispatches over a small grid (160×284 cells by default) plus dye advection on a 4× finer grid.
      Cost scales with <i>cells × iterations</i>: doubling the resolution quadruples the cells and also needs more iterations to converge.
      The pressure solve dominates; multigrid or a warm start (we reuse 80% of last frame’s pressure) speeds it up.</p>`,
    api: `<p>Written with WebGPU <b>compute shaders</b> and <code>rgba16float</code> storage textures; all ~40 dispatches are recorded into a single compute pass.
      The same algorithm runs fine in WebGL2 as a chain of fragment-shader passes rendering into float framebuffers (that’s how most web fluid demos work) —
      compute just makes it cleaner (no fullscreen triangles, scatter writes possible, fewer state changes).</p>`,
    code: [
      {
        title: 'Jacobi pressure iteration (run N times)',
        lang: 'wgsl',
        src: `@compute @workgroup_size(8, 8) fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  let pL = pres(p - vec2i(1, 0));   // a = last pressure guess
  let pR = pres(p + vec2i(1, 0));
  let pT = pres(p - vec2i(0, 1));
  let pB = pres(p + vec2i(0, 1));
  let div = texel(b, p).x;          // b = divergence of the velocity
  textureStore(dst, gid.xy, vec4f((pL + pR + pT + pB - div) * 0.25, 0.0, 0.0, 1.0));
}`,
      },
      {
        title: 'Semi-Lagrangian advection: look backwards, sample',
        lang: 'wgsl',
        src: `let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
let v = textureSampleLevel(a, samp, uv, 0.0).xy;      // velocity here (cells/s)
let back = uv - u.dt * v / u.velSize;                 // where did this fluid come from?
let q = textureSampleLevel(b, samp, back, 0.0);       // bilinear interpolation for free
textureStore(dst, gid.xy, q * keep);                  // keep = exp(-dissipation * dt)`,
      },
      {
        title: 'Remove divergence: subtract the pressure gradient',
        lang: 'wgsl',
        src: `let v = texel(b, p).xy - 0.5 * vec2f(pR - pL, pB - pT);`,
      },
    ],
    links: [
      { title: 'GPU Gems ch. 38 — Fast Fluid Dynamics Simulation on the GPU', url: 'https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu' },
      { title: 'Jos Stam — Real-Time Fluid Dynamics for Games (2003)', url: 'https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf' },
      { title: 'Pavel Dobryakov — WebGL Fluid Simulation', url: 'https://github.com/PavelDoGreat/WebGL-Fluid-Simulation', note: 'the famous web demo' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const FMT = 'rgba16float';
    const U = gpu.uniforms(
      {
        velSize: 'vec2f',
        dyeSize: 'vec2f',
        dt: 'f32',
        aspect: 'f32',
        velKeep: 'f32',
        dyeKeep: 'f32',
        alphaKeep: 'f32',
        vort: 'f32',
        buoyancy: 'f32',
        pScale: 'f32',
        visc: 'f32',
        openB: 'f32',
        nSplats: 'f32',
        splatMode: 'f32',
        wind: 'f32',
        time: 'f32',
        spos: `array<vec4f, ${MAX_SPLATS}>`,
        svel: `array<vec4f, ${MAX_SPLATS}>`,
        sdye: `array<vec4f, ${MAX_SPLATS}>`,
      },
      'Fluid',
    );

    const sim = gpu.compute({
      label: 'fluid',
      bindings: {
        u: { uniform: U },
        a: { texture: 'float' },
        b: { texture: 'float' },
        c: { texture: 'float' },
        d: { texture: 'float' },
        samp: { sampler: 'filtering' },
        dst: { storageTexture: FMT, access: 'write' },
      },
      code: /* wgsl */ `
fn texel(t: texture_2d<f32>, p: vec2i) -> vec4f {
  return textureLoad(t, clamp(p, vec2i(0), vec2i(textureDimensions(t)) - vec2i(1)), 0);
}
fn inside(p: vec2i) -> bool {
  let s = vec2i(textureDimensions(dst));
  return p.x >= 0 && p.y >= 0 && p.x < s.x && p.y < s.y;
}
fn uvOf(gid: vec2u) -> vec2f { return (vec2f(gid) + 0.5) / vec2f(textureDimensions(dst)); }
fn splatW(uv: vec2f, i: u32) -> f32 {
  let s = u.spos[i];
  var d = uv - s.xy;
  d.x *= u.aspect;
  return exp(-dot(d, d) / (s.z * s.z));
}

// 1. forces: mouse/emitter splats, buoyancy (b = dye: .a = temperature or density), wind
@compute @workgroup_size(8, 8) fn splatVel(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  var v = texel(a, p).xy;
  for (var i = 0u; i < u32(u.nSplats); i++) { v += u.svel[i].xy * splatW(uv, i); }
  let d = textureSampleLevel(b, samp, uv, 0.0);
  v.y -= u.buoyancy * d.a * u.dt;
  v.x += u.wind * u.dt;
  textureStore(dst, gid.xy, vec4f(v, 0.0, 1.0));
}

@compute @workgroup_size(8, 8) fn splatDye(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  var c = texel(a, p);
  for (var i = 0u; i < u32(u.nSplats); i++) {
    let w = splatW(uv, i);
    if (u.splatMode < 0.5) { c += u.sdye[i] * w; }
    else if (u.sdye[i].a > 0.0) { c = mix(c, vec4f(u.sdye[i].rgb, 1.0), smoothstep(0.42, 0.52, w)); }
  }
  textureStore(dst, gid.xy, c);
}

// 2. curl (spin) of the velocity field, y pointing down
@compute @workgroup_size(8, 8) fn curl(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let L = texel(a, p - vec2i(1, 0)).y;
  let R = texel(a, p + vec2i(1, 0)).y;
  let T = texel(a, p - vec2i(0, 1)).x;
  let B = texel(a, p + vec2i(0, 1)).x;
  textureStore(dst, gid.xy, vec4f(0.5 * ((R - L) - (B - T)), 0.0, 0.0, 1.0));
}

// 3. vorticity confinement: push velocity around spots of high |curl| (b = curl)
@compute @workgroup_size(8, 8) fn vorticity(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let cL = abs(texel(b, p - vec2i(1, 0)).x);
  let cR = abs(texel(b, p + vec2i(1, 0)).x);
  let cT = abs(texel(b, p - vec2i(0, 1)).x);
  let cB = abs(texel(b, p + vec2i(0, 1)).x);
  let w = texel(b, p).x;
  var N = 0.5 * vec2f(cR - cL, cB - cT);
  N = N / (length(N) + 1e-5);
  let force = u.vort * w * vec2f(N.y, -N.x);
  let v = texel(a, p).xy + force * u.dt;
  textureStore(dst, gid.xy, vec4f(clamp(v, vec2f(-3000.0), vec2f(3000.0)), 0.0, 1.0));
}

// 3b. viscosity: explicit diffusion sub-step (stable for visc <= 0.25)
@compute @workgroup_size(8, 8) fn diffuse(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let C = texel(a, p).xy;
  let lap = texel(a, p - vec2i(1, 0)).xy + texel(a, p + vec2i(1, 0)).xy + texel(a, p - vec2i(0, 1)).xy + texel(a, p + vec2i(0, 1)).xy - 4.0 * C;
  textureStore(dst, gid.xy, vec4f(C + u.visc * lap, 0.0, 1.0));
}

// 4. divergence = net outflow of every cell. Closed walls reflect the normal velocity.
@compute @workgroup_size(8, 8) fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let s = vec2i(textureDimensions(dst));
  let C = texel(a, p).xy;
  var L = texel(a, p - vec2i(1, 0)).x;
  var R = texel(a, p + vec2i(1, 0)).x;
  var T = texel(a, p - vec2i(0, 1)).y;
  var B = texel(a, p + vec2i(0, 1)).y;
  if (u.openB < 0.5) {
    if (p.x == 0) { L = -C.x; }
    if (p.x == s.x - 1) { R = -C.x; }
    if (p.y == 0) { T = -C.y; }
    if (p.y == s.y - 1) { B = -C.y; }
  }
  textureStore(dst, gid.xy, vec4f(0.5 * (R - L + B - T), 0.0, 0.0, 1.0));
}

// 5. warm start: keep part of last frame's pressure
@compute @workgroup_size(8, 8) fn scaleP(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  textureStore(dst, gid.xy, texel(a, p) * u.pScale);
}

fn pres(p: vec2i) -> f32 {
  if (u.openB > 0.5 && !inside(p)) { return 0.0; }   // open boundary: pressure 0 outside
  return texel(a, p).x;                                 // closed: zero gradient at walls
}

// 6. one Jacobi iteration of the pressure Poisson equation (a = pressure, b = divergence)
@compute @workgroup_size(8, 8) fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let pL = pres(p - vec2i(1, 0));
  let pR = pres(p + vec2i(1, 0));
  let pT = pres(p - vec2i(0, 1));
  let pB = pres(p + vec2i(0, 1));
  let div = texel(b, p).x;
  textureStore(dst, gid.xy, vec4f((pL + pR + pT + pB - div) * 0.25, 0.0, 0.0, 1.0));
}

// 7. make it divergence-free (a = pressure, b = velocity)
@compute @workgroup_size(8, 8) fn gradient(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let pL = pres(p - vec2i(1, 0));
  let pR = pres(p + vec2i(1, 0));
  let pT = pres(p - vec2i(0, 1));
  let pB = pres(p + vec2i(0, 1));
  let v = texel(b, p).xy - 0.5 * vec2f(pR - pL, pB - pT);
  textureStore(dst, gid.xy, vec4f(v, 0.0, 1.0));
}

// 8. semi-Lagrangian advection (a = velocity, b = quantity to move)
@compute @workgroup_size(8, 8) fn advectVel(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  let v = textureSampleLevel(a, samp, uv, 0.0).xy;
  let back = uv - u.dt * v / u.velSize;
  textureStore(dst, gid.xy, vec4f(textureSampleLevel(b, samp, back, 0.0).xy * u.velKeep, 0.0, 1.0));
}

@compute @workgroup_size(8, 8) fn advectDye(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  let v = textureSampleLevel(a, samp, uv, 0.0).xy;
  let back = uv - u.dt * v / u.velSize;
  let q = textureSampleLevel(b, samp, back, 0.0);
  textureStore(dst, gid.xy, vec4f(q.rgb * u.dyeKeep, q.a * u.alphaKeep));
}

// MacCormack (BFECC-style) dye advection: advect forward, advect the result backward,
// use the round-trip error to sharpen, and clamp to avoid overshoot.
@compute @workgroup_size(8, 8) fn advectFwd(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  let v = textureSampleLevel(a, samp, uv, 0.0).xy;
  textureStore(dst, gid.xy, textureSampleLevel(b, samp, uv - u.dt * v / u.velSize, 0.0));
}
@compute @workgroup_size(8, 8) fn advectBack(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  let v = textureSampleLevel(a, samp, uv, 0.0).xy;
  textureStore(dst, gid.xy, textureSampleLevel(b, samp, uv + u.dt * v / u.velSize, 0.0));
}
// a = original dye, b = forward result, c = forward-then-back result, d = velocity
@compute @workgroup_size(8, 8) fn maccormack(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inside(p)) { return; }
  let uv = uvOf(gid.xy);
  let size = vec2f(textureDimensions(dst));
  let v = textureSampleLevel(d, samp, uv, 0.0).xy;
  let back = (uv - u.dt * v / u.velSize) * size - 0.5;
  let i0 = vec2i(floor(back));
  let s00 = texel(a, i0);
  let s10 = texel(a, i0 + vec2i(1, 0));
  let s01 = texel(a, i0 + vec2i(0, 1));
  let s11 = texel(a, i0 + vec2i(1, 1));
  let mn = min(min(s00, s10), min(s01, s11));
  let mx = max(max(s00, s10), max(s01, s11));
  let r = clamp(texel(b, p) + 0.5 * (texel(a, p) - texel(c, p)), mn, mx);
  textureStore(dst, gid.xy, vec4f(r.rgb * u.dyeKeep, r.a * u.alphaKeep));
}

@compute @workgroup_size(8, 8) fn clearT(@builtin(global_invocation_id) gid: vec3u) {
  if (!inside(vec2i(gid.xy))) { return; }
  textureStore(dst, gid.xy, vec4f(0.0));
}`,
    });

    // ---------------------------------------------------------------- display
    const V = gpu.uniforms(
      { res: 'vec2f', velSize: 'vec2f', mouse: 'vec4f', time: 'f32', mode: 'f32', view: 'f32', aspect: 'f32' },
      'View',
    );
    const display = gpu.fullscreen({
      label: 'fluid-display',
      uniforms: V,
      textures: ['dye', 'vel', 'pres', 'divT', 'curlT'],
      include: ['math', 'hash', 'noise', 'sdf', 'color'],
      code: /* wgsl */ `
fn diverging(x: f32) -> vec3f {
  let t = clamp(x, -1.0, 1.0);
  let neg = vec3f(0.15, 0.45, 1.0);
  let pos = vec3f(1.0, 0.3, 0.2);
  let mid = vec3f(0.06, 0.06, 0.08);
  return mix(mix(mid, neg, smoothstep(0.0, 1.0, -t)), mix(mid, pos, smoothstep(0.0, 1.0, t)), step(0.0, t));
}

fn blackbody(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  return vec3f(smoothstep(0.0, 0.35, x), smoothstep(0.2, 0.75, x) * 0.85, smoothstep(0.6, 1.0, x) * 0.7) * (0.4 + 2.2 * x * x);
}

fn arrows(px: vec2f) -> f32 {
  let cs = max(22.0, u.res.y / 26.0);
  let c = floor(px / cs);
  let center = (c + 0.5) * cs;
  let v = TEX(vel, center / u.res).xy;
  let m = length(v) / u.velSize.y;          // screen-heights per second
  if (m < 0.002) { return 0.0; }
  let dir = normalize(v);
  let len = clamp(sqrt(m) * cs * 0.9, 3.0, cs * 0.46);
  let q = px - center;
  let lq = vec2f(dot(q, dir), dot(q, vec2f(-dir.y, dir.x)));
  let hs = clamp(len * 0.45, 2.5, 5.0);
  let shaft = sdSegment(lq, vec2f(-len, 0.0), vec2f(len - hs, 0.0)) - 0.8;
  let head = sdTriangle(lq, vec2f(len, 0.0), vec2f(len - hs * 1.6, hs), vec2f(len - hs * 1.6, -hs));
  return (1.0 - smoothstep(0.0, 1.0, min(shaft, head))) * smoothstep(0.002, 0.03, m);
}

fn fieldView(uv: vec2f, px: vec2f, view: i32) -> vec3f {
  if (view == 1) {
    let v = TEX(vel, uv).xy;
    let m = length(v) / u.velSize.y;
    var c = hsv2rgb(vec3f(fract(atan2(v.y, v.x) / TAU + 1.0), 0.7, smoothstep(0.0, 0.35, m))) * 0.85;
    c = mix(c, vec3f(1.0), arrows(px) * 0.85);
    return c;
  }
  if (view == 2) {
    // pressure with isobars, like a weather map
    let pr = TEX(pres, uv).x * 0.08;
    let f = abs(fract(pr * 3.0) - 0.5) / max(fwidth(pr * 3.0), 1e-4);
    return diverging(pr) + vec3f(0.3) * (1.0 - smoothstep(0.5, 1.5, f));
  }
  if (view == 3) { return diverging(TEX(divT, uv).x * 0.05); }
  let w = TEX(curlT, uv).x * 0.05;
  return mix(vec3f(0.05), mix(vec3f(0.1, 0.9, 0.7), vec3f(0.95, 0.3, 0.9), step(0.0, w)), clamp(abs(w), 0.0, 1.0));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let mode = i32(u.mode + 0.5);
  let view = i32(u.view + 0.5);
  if (view > 0) { return vec4f(fieldView(uv, px, view), 1.0); }
  let d = TEX(dye, uv);
  let ts = 1.0 / TEXSIZE(dye);

  if (mode == 0) {
    // ink: Beer–Lambert absorption over a lit glass tank of water
    var water = mix(vec3f(0.97, 0.97, 0.95), vec3f(0.84, 0.9, 0.93), uv.y);
    water *= 1.0 - 0.18 * pow(length(uv - vec2f(0.5, 0.4)), 2.0);
    water += vec3f(0.03) * (valueNoise(px * 0.6) - 0.5);
    var c = water * exp(-max(d.rgb, vec3f(0.0)));
    // soft refraction sparkle on dense ink edges
    let g = vec2f(TEX(dye, uv + vec2f(ts.x, 0.0)).b - TEX(dye, uv - vec2f(ts.x, 0.0)).b, TEX(dye, uv + vec2f(0.0, ts.y)).b - TEX(dye, uv - vec2f(0.0, ts.y)).b);
    c += vec3f(0.06) * clamp(-g.y * 2.0, 0.0, 1.0);
    return vec4f(c, 1.0);
  }

  if (mode == 1) {
    // campfire at dusk
    let aspect = u.res.x / u.res.y;
    let fp = vec2f((uv.x - 0.5) * aspect, uv.y - 0.82);
    var bg = mix(vec3f(0.05, 0.07, 0.16), vec3f(0.2, 0.12, 0.17), smoothstep(0.0, 0.8, uv.y));
    let starG = floor(px / 3.0);
    bg += vec3f(0.7) * step(0.997, hash21(starG)) * smoothstep(0.6, 0.0, uv.y) * (0.6 + 0.4 * sin(u.time * 3.0 + hash21(starG + 1.0) * 40.0));
    // distant tree line
    let trees = 0.74 - 0.05 * fbm(vec2f(uv.x * 5.0, 3.0), 3) - 0.04 * pow(abs(sin(uv.x * 60.0 + 1.3 * sin(uv.x * 13.0))), 3.0);
    bg = mix(bg, vec3f(0.03, 0.04, 0.07), smoothstep(trees, trees + 0.004, uv.y));
    let flick = 0.85 + 0.15 * sin(u.time * 13.0) * sin(u.time * 7.3);
    let fireLight = exp(-length(fp * vec2f(1.0, 1.5)) * 3.5) * flick;
    // ground
    let gy = 0.83 + 0.006 * perlin(vec2f(uv.x * 9.0, 0.0));
    let ground = smoothstep(gy, gy + 0.004, uv.y);
    bg = mix(bg, vec3f(0.05, 0.04, 0.035) * (0.8 + 0.4 * valueNoise(px * 0.15)) + vec3f(0.7, 0.3, 0.08) * fireLight * 0.7, ground);
    bg += vec3f(1.0, 0.45, 0.12) * fireLight * 0.22;
    // smoke: absorbs the background, lit cool from the sky and warm from the fire
    let smoke = max(d.r, 0.0);
    let k = 1.0 - exp(-smoke * 2.2);
    let lit = mix(vec3f(0.36, 0.38, 0.45), vec3f(1.0, 0.55, 0.25), clamp(fireLight * 2.0, 0.0, 1.0));
    var c = mix(bg, lit, k * 0.9);
    // flames = hot temperature through a blackbody ramp
    let temp = max(d.a, 0.0);
    c += blackbody(temp * 0.9) * smoothstep(0.08, 0.5, temp) * 1.3;
    // logs + stones
    let l1 = sdSegment(fp, vec2f(-0.09, 0.012), vec2f(0.07, -0.012)) - 0.013;
    let l2 = sdSegment(fp, vec2f(0.09, 0.016), vec2f(-0.06, -0.01)) - 0.012;
    let logs = 1.0 - smoothstep(0.0, 0.003, min(l1, l2));
    let ember = clamp(temp * 1.5 + 0.25 * flick, 0.0, 1.0);
    c = mix(c, vec3f(0.13, 0.07, 0.04) + vec3f(0.9, 0.3, 0.05) * ember * smoothstep(0.004, -0.008, min(l1, l2)), logs);
    var st = 1e5;
    for (var i = 0; i < 7; i++) {
      let a = f32(i) / 6.0;
      st = min(st, length((fp - vec2f(mix(-0.13, 0.13, a), 0.02 + 0.006 * sin(f32(i) * 2.0))) * vec2f(1.0, 1.6)) - 0.018);
    }
    let stones = 1.0 - smoothstep(0.0, 0.003, st);
    c = mix(c, vec3f(0.2, 0.19, 0.2) + vec3f(0.6, 0.3, 0.1) * fireLight * 0.5, stones);
    return vec4f(c, 1.0);
  }

  if (mode == 2) {
    // paint marbling: premultiplied paint over an off-white bath, lit by the paint's slope
    let cov = clamp(d.a, 0.0, 1.0);
    let paint = d.rgb / max(d.a, 1e-3);
    var bath = vec3f(0.93, 0.9, 0.84) * (0.96 + 0.04 * valueNoise(px * 0.3));
    var c = mix(bath, paint, cov);
    let hL = luma(TEX(dye, uv - vec2f(ts.x, 0.0)).rgb);
    let hR = luma(TEX(dye, uv + vec2f(ts.x, 0.0)).rgb);
    let hT = luma(TEX(dye, uv - vec2f(0.0, ts.y)).rgb);
    let hB = luma(TEX(dye, uv + vec2f(0.0, ts.y)).rgb);
    let n = normalize(vec3f((hL - hR) * 1.2, (hT - hB) * 1.2, 1.0));
    let L = normalize(vec3f(-0.5, -0.6, 0.65));
    let diff = 0.9 + 0.15 * (dot(n, L) - L.z);
    let spec = pow(max(dot(reflect(-L, n), vec3f(0.0, 0.0, 1.0)), 0.0), 60.0);
    c = c * diff + vec3f(spec) * 0.18 * cov;
    c *= 1.0 - 0.25 * pow(length(uv - 0.5) * 1.3, 3.0);
    return vec4f(c, 1.0);
  }

  // "under the hood" dye: glowing colours on black, shaded by their slope
  let base = d.rgb;
  let hL = luma(TEX(dye, uv - vec2f(ts.x, 0.0)).rgb);
  let hR = luma(TEX(dye, uv + vec2f(ts.x, 0.0)).rgb);
  let hT = luma(TEX(dye, uv - vec2f(0.0, ts.y)).rgb);
  let hB = luma(TEX(dye, uv + vec2f(0.0, ts.y)).rgb);
  let n = normalize(vec3f(hL - hR, hT - hB, 0.6));
  let sh = clamp(dot(n, normalize(vec3f(-0.3, -0.5, 1.0))) + 0.3, 0.6, 1.3);
  return vec4f(tonemapReinhard(base * sh * 1.4) * 1.15, 1.0);
}`,
    });

    // ---------------------------------------------------------------- textures
    const dummy = gpu.target(1, 1, { format: FMT, label: 'dummy' }); // bound as input when clearing
    let T = null;
    let sizes = null;
    const alloc = () => {
      const testMode = ctx.testMode;
      const aspect = ctx.width / ctx.height;
      const velH = Math.round(testMode ? Math.min(64, ctx.params.res) : ctx.params.res);
      const velW = Math.max(8, Math.round(velH * aspect));
      const dyeH = Math.min(testMode ? 200 : 1024, velH * 4, ctx.height);
      const dyeW = Math.max(8, Math.round(dyeH * aspect));
      const key = `${velW}x${velH}/${dyeW}x${dyeH}`;
      if (sizes === key) return false;
      if (T) for (const t of Object.values(T)) t.destroy();
      T = {
        vel: gpu.pingPong(velW, velH, { format: FMT, label: 'vel' }),
        pres: gpu.pingPong(velW, velH, { format: FMT, label: 'pressure' }),
        dye: gpu.pingPong(dyeW, dyeH, { format: FMT, label: 'dye' }),
        fwd: gpu.target(dyeW, dyeH, { format: FMT, label: 'dye-fwd' }),
        bwd: gpu.target(dyeW, dyeH, { format: FMT, label: 'dye-bwd' }),
        div: gpu.target(velW, velH, { format: FMT, label: 'divergence' }),
        curl: gpu.target(velW, velH, { format: FMT, label: 'curl' }),
      };
      sizes = key;
      return true;
    };

    // ---------------------------------------------------------------- splats
    let queue = []; // pending splats {x, y, r, fx, fy, dye:[r,g,b,a]}
    let needClear = true;
    let colorIdx = 0;
    let lastInteract = -10;
    let nextAuto = 0;
    const tag = readout(ctx);

    const absorb = (c, k = 1) => c.map((v) => -Math.log(Math.max(0.02, v)) * k);
    const initSplats = () => {
      queue = [];
      const ex = ctx.example;
      const aspect = ctx.width / ctx.height;
      if (ex === 'ink') {
        for (let i = 0; i < 4; i++) {
          const ink = INKS[i % INKS.length];
          const x = 0.2 + 0.6 * (i / 3) + (Math.random() - 0.5) * 0.08;
          queue.push({ x, y: 0.18 + Math.random() * 0.1, r: 0.035, fx: 0, fy: 60, dye: [...absorb(ink, 1.6), 1.0] });
        }
      } else if (ex === 'paint') {
        // stacked drops (the classic marbling "stone" pattern), then combing strokes across the bath
        const cols = 6;
        const rows = 3;
        for (let j = 0; j < rows; j++) {
          for (let i = 0; i < cols; i++) {
            const x = (i + 0.5 + (j % 2) * 0.5 - 0.25) / cols;
            const y = (j + 0.5) / rows;
            const base = (i * 2 + j) % PAINTS.length;
            for (let k = 0; k < 4; k++) {
              queue.push({ x, y, r: (0.15 - k * 0.032) * (1 + 0.1 * Math.random()), fx: 0, fy: 0, dye: [...PAINTS[(base + k * 3 + (k > 1 ? 1 : 0)) % PAINTS.length], 1] });
            }
          }
        }
        for (let sIdx = 0; sIdx < 10; sIdx++) {
          const y = (sIdx + 0.5) / 10;
          for (let i = 0; i < 4; i++) {
            queue.push({ x: (i + 0.5) / 4, y, r: 0.05, fx: (sIdx % 2 ? 1 : -1) * 160, fy: 0, dye: [0, 0, 0, 0] });
          }
        }
      } else if (ex === 'fields') {
        for (let i = 0; i < 7; i++) {
          const a = Math.random() * Math.PI * 2;
          const c = hsv(Math.random(), 0.85, 1);
          queue.push({ x: 0.15 + 0.7 * Math.random(), y: 0.15 + 0.7 * Math.random(), r: 0.04, fx: Math.cos(a) * 400, fy: Math.sin(a) * 400, dye: [c[0] * 1.5, c[1] * 1.5, c[2] * 1.5, 0] });
        }
      }
    };
    const resetAll = () => {
      needClear = true;
      initSplats();
    };
    alloc();
    resetAll();

    const res = (a, b, dst, c = dummy, d = dummy) => ({ u: U, a, b, c, d, samp: 'linear', dst });

    return {
      resize() {
        if (alloc()) resetAll();
      },
      onChange(key) {
        if (key === 'res' && alloc()) resetAll();
      },
      onAction(key) {
        if (key === 'reset') resetAll();
      },
      onExample() {
        resetAll();
      },
      frame(ctx) {
        const p = ctx.params;
        const ex = ctx.example;
        const enc = ctx.encoder;
        const mode = { ink: 0, smoke: 1, paint: 2, fields: 3 }[ex] ?? 3;
        const aspect = ctx.width / ctx.height;
        const vel = T.vel;
        const velSize = [vel.width, vel.height];
        const dyeSize = [T.dye.width, T.dye.height];
        const running = !ctx.paused && ctx.dt > 0;
        // the headless test GPU is very slow: take a few steps per frame there
        const substeps = ctx.testMode ? 6 : 1;
        const dt = Math.min(ctx.dt, 1 / 30) || 1 / 60;

        if (needClear) {
          const cs = computeSeq(enc, 'fluid-clear');
          for (const t of [vel.a, vel.b, T.pres.a, T.pres.b, T.dye.a, T.dye.b, T.div, T.curl, T.fwd, T.bwd]) {
            cs.run(sim, 'clearT', [Math.ceil(t.width / 8), Math.ceil(t.height / 8)], res(dummy, dummy, t));
          }
          cs.end();
          needClear = false;
        }

        if (running) {
          // ----- user input -> splats
          const ptr = ctx.pointer;
          const splats = [];
          const r = p.splat / 100;
          if (ptr.down && (Math.abs(ptr.dx) + Math.abs(ptr.dy) > 0.5 || ptr.clicked)) {
            lastInteract = ctx.time;
            const fx = ((ptr.dx / ctx.height) * velSize[1] / dt) * p.force;
            const fy = ((ptr.dy / ctx.height) * velSize[1] / dt) * p.force;
            let dye;
            if (mode === 0) dye = [...absorb(INKS[colorIdx % INKS.length], 0.35), 0.25];
            else if (mode === 1) dye = [0.15, 0.15, 0.15, 0.0];
            else if (mode === 2) dye = [...PAINTS[colorIdx % PAINTS.length], 1];
            else {
              const c = hsv((ctx.time * 0.15) % 1, 0.85, 1);
              dye = [c[0] * 0.6, c[1] * 0.6, c[2] * 0.6, 0];
            }
            if (mode === 2 && !ptr.clicked) dye = [0, 0, 0, 0]; // dragging combs the paint; a click drops new paint
            splats.push({ x: ptr.nx, y: ptr.ny, r, fx, fy, dye });
          }
          if (ptr.released) colorIdx++;

          // ----- automatic sources
          if (mode === 1) {
            // the fire: a few flickering emitters at the base of the logs
            for (let i = 0; i < 3; i++) {
              const ox = ((i - 1) * 0.03) / aspect + Math.sin(ctx.time * (5 + i * 2.3) + i) * 0.01;
              const heat = 0.35 + 0.15 * Math.sin(ctx.time * 17 + i * 1.7);
              splats.push({ x: 0.5 + ox, y: 0.805, r: 0.03, fx: Math.sin(ctx.time * 9 + i * 2) * 12, fy: -6, dye: [0.035, 0.035, 0.035, heat] });
            }
          } else if (ctx.time > nextAuto) {
            if (mode === 0) {
              const ink = INKS[Math.floor(Math.random() * INKS.length)];
              queue.push({ x: 0.12 + Math.random() * 0.76, y: 0.06, r: 0.03, fx: (Math.random() - 0.5) * 20, fy: 80, dye: [...absorb(ink, 1.6), 1.0] });
              nextAuto = ctx.time + (ctx.time - lastInteract < 3 ? 3.5 : 1.6);
            } else if (ctx.time - lastInteract > 2.5) {
              const a = Math.random() * Math.PI * 2;
              if (mode === 2) {
                const x = 0.2 + 0.6 * Math.random();
                const y = 0.2 + 0.6 * Math.random();
                queue.push({ x, y, r: 0.05 + Math.random() * 0.05, fx: 0, fy: 0, dye: [...PAINTS[Math.floor(Math.random() * PAINTS.length)], 1] });
                queue.push({ x: x - Math.cos(a) * 0.1, y: y - Math.sin(a) * 0.1, r: 0.06, fx: Math.cos(a) * 220, fy: Math.sin(a) * 220, dye: [0, 0, 0, 0] });
                nextAuto = ctx.time + 1.4;
              } else {
                const c = hsv(Math.random(), 0.85, 1);
                queue.push({ x: 0.15 + 0.7 * Math.random(), y: 0.15 + 0.7 * Math.random(), r: 0.03 + Math.random() * 0.02, fx: Math.cos(a) * 500, fy: Math.sin(a) * 500, dye: [c[0] * 1.2, c[1] * 1.2, c[2] * 1.2, 0] });
                nextAuto = ctx.time + 0.6;
              }
            }
          }
          while (splats.length < MAX_SPLATS && queue.length) splats.push(queue.shift());

          const spos = new Float32Array(MAX_SPLATS * 4);
          const svel = new Float32Array(MAX_SPLATS * 4);
          const sdye = new Float32Array(MAX_SPLATS * 4);
          splats.forEach((s, i) => {
            spos.set([s.x, s.y, s.r, 0], i * 4);
            svel.set([s.fx, s.fy, 0, 0], i * 4);
            sdye.set(s.dye, i * 4);
          });

          const visc = p.viscosity * dt;
          const viscSteps = visc > 0 ? Math.min(12, Math.ceil(visc / 0.2)) : 0;
          U.setAll({
            velSize,
            dyeSize,
            dt,
            aspect,
            velKeep: Math.exp(-p.velDiss * dt),
            dyeKeep: Math.exp(-p.dyeDiss * dt),
            alphaKeep: mode === 1 ? Math.exp(-2.2 * dt) : Math.exp(-p.dyeDiss * dt),
            vort: p.vorticity,
            buoyancy: mode <= 1 ? p.buoyancy * velSize[1] * 0.5 : 0,
            pScale: 0.8,
            visc: viscSteps ? visc / viscSteps : 0,
            openB: mode === 1 ? 1 : 0,
            nSplats: splats.length,
            splatMode: mode === 2 ? 1 : 0,
            wind: 0,
            time: ctx.time,
            spos,
            svel,
            sdye,
          });
          U.upload();

          const gx = Math.ceil(vel.width / 8);
          const gy = Math.ceil(vel.height / 8);
          const dgx = Math.ceil(T.dye.width / 8);
          const dgy = Math.ceil(T.dye.height / 8);
          const iters = Math.round(p.iterations);
          const cs = computeSeq(enc, 'fluid-step');
          for (let s = 0; s < substeps; s++) {
            // forces (splats are only applied on the first sub-step)
            if (s === 0 || mode === 1) {
              cs.run(sim, 'splatVel', [gx, gy], res(vel.read, T.dye.read, vel.write));
              vel.swap();
            }
            if (p.vorticity > 0) {
              cs.run(sim, 'curl', [gx, gy], res(vel.read, vel.read, T.curl));
              cs.run(sim, 'vorticity', [gx, gy], res(vel.read, T.curl, vel.write));
              vel.swap();
            }
            for (let k = 0; k < viscSteps; k++) {
              cs.run(sim, 'diffuse', [gx, gy], res(vel.read, vel.read, vel.write));
              vel.swap();
            }
            cs.run(sim, 'divergence', [gx, gy], res(vel.read, vel.read, T.div));
            cs.run(sim, 'scaleP', [gx, gy], res(T.pres.read, T.pres.read, T.pres.write));
            T.pres.swap();
            for (let k = 0; k < iters; k++) {
              cs.run(sim, 'jacobi', [gx, gy], res(T.pres.read, T.div, T.pres.write));
              T.pres.swap();
            }
            cs.run(sim, 'gradient', [gx, gy], res(T.pres.read, vel.read, vel.write));
            vel.swap();
            cs.run(sim, 'advectVel', [gx, gy], res(vel.read, vel.read, vel.write));
            vel.swap();
            if (s === 0 || mode === 1) {
              cs.run(sim, 'splatDye', [dgx, dgy], res(T.dye.read, T.dye.read, T.dye.write));
              T.dye.swap();
            }
            if (p.sharp) {
              cs.run(sim, 'advectFwd', [dgx, dgy], res(vel.read, T.dye.read, T.fwd));
              cs.run(sim, 'advectBack', [dgx, dgy], res(vel.read, T.fwd, T.bwd));
              cs.run(sim, 'maccormack', [dgx, dgy], res(T.dye.read, T.fwd, T.dye.write, T.bwd, vel.read));
            } else {
              cs.run(sim, 'advectDye', [dgx, dgy], res(vel.read, T.dye.read, T.dye.write));
            }
            T.dye.swap();
          }
          cs.end();
        }

        const viewIdx = { auto: 0, velocity: 1, pressure: 2, divergence: 3, curl: 4 }[p.view] ?? 0;
        V.setAll({
          res: [ctx.width, ctx.height],
          velSize,
          mouse: [ctx.pointer.x, ctx.pointer.y, ctx.pointer.down ? 1 : 0, ctx.pointer.over ? 1 : 0],
          time: ctx.time,
          mode,
          view: viewIdx,
          aspect,
        });
        display.draw(enc, { view: ctx.target, format: gpu.format }, { dye: T.dye.read, vel: vel.read, pres: T.pres.read, divT: T.div, curlT: T.curl });
        tag.textContent = `velocity ${vel.width}×${vel.height} · dye ${T.dye.width}×${T.dye.height} · ${Math.round(p.iterations)} Jacobi iterations`;
      },
    };
  },
};

// N-body gravity on the GPU: every body attracts every other body (O(N²)), computed with
// workgroup shared-memory tiling, integrated with symplectic Euler, and drawn as additive HDR
// sprites with a cheap bloom.

import { overlayTag } from './_shared-b.js';

const TILE = 256;

const SIM_WGSL = /* wgsl */ `
const TILE: u32 = ${TILE}u;
var<workgroup> tile: array<vec4f, ${TILE}>;    // one tile of bodies, shared by the 256 threads of a workgroup

@compute @workgroup_size(${TILE})
fn step(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let i = gid.x;
  let n = u.count;
  let me = pin[min(i, n - 1u)];                 // (x, y, mass, style)
  var acc = vec2f(0.0);
  let eps2 = u.soft * u.soft;
  // Walk over all "source" bodies one tile at a time. Each thread loads ONE body into shared memory,
  // then every thread reads all 256 from fast on-chip memory instead of global memory.
  let tiles = (u.sources + TILE - 1u) / TILE;
  for (var t = 0u; t < tiles; t++) {
    let j = t * TILE + lid.x;
    tile[lid.x] = select(vec4f(0.0), pin[min(j, n - 1u)], j < u.sources);
    workgroupBarrier();
    for (var k = 0u; k < TILE; k++) {
      let b = tile[k];
      let d = b.xy - me.xy;
      let r2 = dot(d, d) + eps2;                // softening: no infinite forces at tiny distances
      let inv = inverseSqrt(r2);
      acc += d * (b.z * inv * inv * inv);       // G·m·d / |d|³
    }
    workgroupBarrier();
  }
  if (i >= n) { return; }
  acc *= u.G;
  if (u.hand > 0.0) {                           // the mouse: a temporary heavy attractor
    let d = u.mouse - me.xy;
    let r2 = dot(d, d) + 0.01;
    acc += d * (u.hand * inverseSqrt(r2) / r2);
  }
  var v = vel[i];
  v = vec4f(v.xy + acc * u.dt, v.zw);           // kick
  vel[i] = v;
  pout[i] = vec4f(me.xy + v.xy * u.dt, me.zw);  // drift
}`;

const DRAW_WGSL = /* wgsl */ `
struct VOut { @builtin(position) pos: vec4f, @location(0) lp: vec2f, @location(1) @interpolate(flat) col: vec3f, @location(2) @interpolate(flat) kind: f32 };
fn planetColor(k: f32) -> vec3f {
  var P = array<vec3f, 10>(vec3f(1.0, 0.85, 0.5), vec3f(0.7, 0.65, 0.6), vec3f(0.95, 0.8, 0.55), vec3f(0.35, 0.6, 1.0),
                           vec3f(1.0, 0.45, 0.3), vec3f(0.95, 0.75, 0.55), vec3f(0.95, 0.85, 0.6), vec3f(0.6, 0.9, 0.95),
                           vec3f(0.35, 0.5, 1.0), vec3f(0.8, 0.8, 0.8));
  return P[u32(clamp(k, 0.0, 9.0))];
}
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let p = pos[ii];
  let v = vel[ii].xy;
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let style = p.w;
  var sizePx = u.size;
  var col = vec3f(0.0);
  let speed = length(v) / u.vscale;
  if (u.colorMode < 0.5) {
    // color by speed: slow = ember red, medium = warm white, fast = blue
    col = mix(vec3f(1.0, 0.32, 0.1), vec3f(1.0, 0.85, 0.6), smoothstep(0.0, 0.5, speed));
    col = mix(col, vec3f(0.45, 0.7, 1.0), smoothstep(0.5, 1.2, speed));
  } else {
    // color by origin (which galaxy / ring the body started in)
    let g = vel[ii].z;
    col = 0.55 + 0.45 * cos(6.2831853 * (g * 0.37 + vec3f(0.0, 0.33, 0.67) + 0.55));
  }
  col *= u.bright;
  if (style >= 0.5) {
    // special bodies: black holes, suns, planets
    if (style < 1.5) { sizePx = u.size * 5.0; col = vec3f(1.0, 0.9, 0.75) * 3.0; }
    else if (style < 2.5) { sizePx = 22.0; col = vec3f(1.0, 0.85, 0.5) * 4.0; }
    else { sizePx = 4.0 + vel[ii].w * 6.0; col = planetColor(style - 3.0) * 1.6; }
  }
  let sp = vec2f((p.x - u.center.x) * u.zoom / u.aspect, (p.y - u.center.y) * u.zoom);
  o.pos = vec4f(sp + c * sizePx * u.px, 0.0, 1.0);
  o.lp = c;
  o.col = col;
  o.kind = style;
  return o;
}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let r2 = dot(i.lp, i.lp);
  if (r2 > 1.0) { discard; }
  var a = exp(-r2 * 4.5);                       // soft gaussian splat
  if (i.kind >= 2.5) { a = smoothstep(1.0, 0.7, sqrt(r2)) * 0.9 + exp(-r2 * 2.0) * 0.3; }   // planets: solid discs
  return vec4f(i.col * a, 1.0);
}`;

const examples = [
  {
    id: 'galaxy',
    label: 'Galaxy collision',
    kind: 'Real life',
    note: 'Two rotating disk galaxies, each with a heavy core, fall into each other. Gravity tears long <b>tidal tails</b> out of them — exactly what we see in real galaxy mergers (the Antennae, the Mice). Every star pulls on every other star.',
    params: { soft: 0.02, speed: 1, colorMode: 'origin', size: 1.6, count: 16384 },
  },
  {
    id: 'solar',
    label: 'Solar system + asteroids',
    kind: 'In a game',
    note: 'A sun, eight planets and a belt of asteroids. Only the 9 heavy bodies pull — asteroids are “test particles” — so each body needs 9 force evaluations instead of N. That is the trick behind huge asteroid fields and bullet-hell gravity wells. Watch Jupiter carve gaps in the belt.',
    params: { soft: 0.004, speed: 1, colorMode: 'speed', size: 1.4, count: 32768 },
  },
  {
    id: 'dust',
    label: 'Dust cloud collapse',
    kind: 'Abstract',
    note: 'A slowly spinning cloud of equal-mass particles collapses under its own gravity into filaments and clumps — the way stars and galaxies form. Lower the softening for sharper, more violent clumping.',
    params: { soft: 0.012, speed: 1, colorMode: 'speed', size: 1.5, count: 16384 },
  },
];

const controls = [
  { type: 'slider', key: 'count', label: 'Bodies', min: 2048, max: 65536, step: 256, value: 16384, log: true, format: (v) => Math.round(v).toLocaleString(), help: 'Cost grows with N² (every body vs every body). Changing it restarts the simulation.' },
  { type: 'slider', key: 'G', label: 'Gravity (G)', min: 0, max: 3, step: 0.01, value: 1 },
  { type: 'slider', key: 'soft', label: 'Softening ε', min: 0.001, max: 0.1, step: 0.001, value: 0.02, log: true, help: 'Adds ε² to every distance²: avoids infinite forces when bodies pass very close. Small = sharper clumps, more chaos.' },
  { type: 'slider', key: 'speed', label: 'Time step', min: 0, max: 3, step: 0.01, value: 1, help: 'Bigger steps run faster but are less accurate (orbits drift, close encounters fling bodies away).' },
  { type: 'select', key: 'colorMode', label: 'Color by', value: 'speed', options: [{ value: 'speed', label: 'Speed (red → white → blue)' }, { value: 'origin', label: 'Where it started' }] },
  { type: 'slider', key: 'size', label: 'Point size (px)', min: 0.5, max: 4, step: 0.1, value: 1.5 },
  { type: 'slider', key: 'exposure', label: 'Exposure', min: 0.1, max: 4, step: 0.01, value: 1, help: 'Bodies add up in an HDR buffer; exposure + tone mapping bring it to the screen.' },
  { type: 'slider', key: 'bloom', label: 'Bloom', min: 0, max: 2, step: 0.01, value: 0.8 },
  { type: 'toggle', key: 'trails', label: 'Motion trails', value: false, help: 'Fade the previous frame instead of clearing it.' },
  { type: 'button', key: 'reset', label: 'Restart', primary: true },
];

// ------------------------------------------------------------------------------------------ initial conditions
function gauss() {
  return Math.sqrt(-2 * Math.log(Math.random() + 1e-9)) * Math.cos(Math.PI * 2 * Math.random());
}

// pos: (x, y, mass, style)  vel: (vx, vy, group, size)
function initBodies(example, N, soft) {
  const pos = new Float32Array(N * 4);
  const vel = new Float32Array(N * 4);
  const set = (i, x, y, m, style, vx, vy, group, size = 0) => {
    pos.set([x, y, m, style], i * 4);
    vel.set([vx, vy, group, size], i * 4);
  };
  const vcirc = (M, r) => Math.sqrt((M * r * r) / Math.pow(r * r + soft * soft, 1.5));
  let sources = N;
  let zoom = 1;
  let vscale = 1;
  if (example === 'galaxy') {
    const half = N >> 1;
    const disk = (first, n, cx, cy, vx, vy, spin, R, group) => {
      const Mbh = 0.6;
      const Md = 1.0;
      const s = R / 3.2;
      set(first, cx, cy, Mbh, 1, vx, vy, group);
      for (let k = 1; k < n; k++) {
        const r = Math.min(R * 1.4, s * -Math.log(1 - Math.random() * 0.985)) + 0.012;
        const a = Math.random() * Math.PI * 2;
        const menc = Mbh + Md * (1 - Math.exp(-r / s) * (1 + r / s));
        const v = vcirc(menc, r) * (0.97 + 0.06 * Math.random());
        set(first + k, cx + r * Math.cos(a), cy + r * Math.sin(a), Md / (n - 1), 0, vx - spin * v * Math.sin(a), vy + spin * v * Math.cos(a), group);
      }
    };
    disk(0, half, -0.78, -0.22, 0.24, 0.07, 1, 0.42, 0);
    disk(half, N - half, 0.78, 0.22, -0.24, -0.07, 1, 0.36, 1);
    zoom = 0.85;
    vscale = 1.6;
  } else if (example === 'solar') {
    // first the 9 massive bodies (the only "sources"), then massless asteroids
    set(0, 0, 0, 1, 2, 0, 0, 0);
    const planets = [
      [0.13, 3e-6, 0.3], [0.19, 6e-6, 0.45], [0.26, 8e-6, 0.5], [0.34, 3e-6, 0.4],
      [0.62, 2e-3, 1.4], [0.8, 6e-4, 1.1], [0.94, 1e-4, 0.8], [1.06, 1e-4, 0.8],
    ];
    planets.forEach(([r, m, size], k) => {
      const a = Math.random() * Math.PI * 2;
      const v = vcirc(1, r);
      set(1 + k, r * Math.cos(a), r * Math.sin(a), m, 3 + k, -v * Math.sin(a), v * Math.cos(a), 2, size);
    });
    sources = 9;
    for (let i = 9; i < N; i++) {
      // most in the main belt, some in an outer "Kuiper" ring, a few Trojans near Jupiter
      const roll = Math.random();
      let r;
      if (roll < 0.72) r = 0.4 + Math.random() * 0.17;
      else if (roll < 0.95) r = 1.12 + Math.random() * 0.12;
      else r = 0.62 + gauss() * 0.01;
      const a = Math.random() * Math.PI * 2;
      const v = vcirc(1, r) * (1 + gauss() * 0.02);
      set(i, r * Math.cos(a), r * Math.sin(a), 0, 0, -v * Math.sin(a), v * Math.cos(a), r < 0.6 ? 0 : r < 0.7 ? 1 : 3);
    }
    zoom = 0.78;
    vscale = 2.4;
  } else {
    for (let i = 0; i < N; i++) {
      const r = Math.sqrt(Math.random()) * 0.85;
      const a = Math.random() * Math.PI * 2;
      const x = r * Math.cos(a) * 1.15;
      const y = r * Math.sin(a);
      const spin = 0.35;
      set(i, x, y, 1 / N, 0, -y * spin + gauss() * 0.03, x * spin + gauss() * 0.03, r);
    }
    zoom = 1.0;
    vscale = 0.9;
  }
  return { pos, vel, sources, zoom, vscale };
}

export default {
  interaction: 'Hold the mouse to pull with gravity. Scroll to zoom.',
  examples,
  controls,
  wheel: true,
  about: {
    summary:
      'N-body simulation: every body attracts every other body. That is N² forces per step — 268 million for 16k bodies — a perfect job for thousands of GPU threads and fast workgroup <b>shared memory</b>.',
    what: `<p>Up to 65,536 stars, planets or dust grains, each pulled by all the others (Newton’s law of gravity). The whole simulation lives in
      GPU storage buffers; one compute dispatch per frame computes all the forces and moves every body. They are drawn as glowing additive
      sprites into an HDR buffer, with a little bloom.</p>`,
    how: `<ol>
      <li><b>Force</b>: body <i>i</i> feels <code>a = G Σ m<sub>j</sub> · d / (|d|² + ε²)<sup>3/2</sup></code>. The <b>softening</b> ε keeps forces finite when two
        bodies nearly collide (we pretend they are fuzzy clouds, not points).</li>
      <li><b>Tiling with shared memory</b>: a workgroup of 256 threads (one per body) loads 256 <i>source</i> bodies into
        <code>var&lt;workgroup&gt;</code> memory — each thread loads one — then waits at a <code>workgroupBarrier()</code>. Now every thread loops over all
        256 from fast on-chip memory. Repeat tile after tile. Global memory traffic drops by ~256×; the GPU becomes compute-bound.</li>
      <li><b>Integration</b> (symplectic Euler): <code>v += a·dt</code>, then <code>x += v·dt</code>. It’s cheap and keeps orbits stable for a long time,
        unlike naive Euler (which spirals outward).</li>
      <li><b>Ping-pong</b>: positions are read from buffer A and written to buffer B (then swapped), so no thread sees a half-updated world.</li>
      <li><b>Only some bodies need to pull</b>: in the solar system only the first 9 bodies have mass, so the tile loop stops after one tile —
        a 65k-asteroid belt costs almost nothing.</li>
      <li><b>Rendering</b>: one instanced quad per body, a Gaussian falloff, additive blending into <code>rgba16float</code>. Dense regions
        exceed 1.0; tone mapping and bloom turn that into glowing cores.</li>
    </ol>`,
    uses: [
      { title: 'Space games', text: 'Orbits, gravity slingshots, asteroid belts and planet-hopping (Outer Wilds, Kerbal-style 2D games).' },
      { title: 'Bullet-hell & VFX', text: 'Thousands of projectiles bent by gravity wells or black-hole power-ups.' },
      { title: 'Swarm forces', text: 'The same all-pairs pattern handles charges, magnets, flocking and “everything repels everything” layouts.' },
      { title: 'Screensavers & menus', text: 'Hypnotic galaxy backgrounds for title screens.' },
    ],
    try: [
      'Hold the mouse in the middle of the galaxy collision to swallow both cores, then let go and watch the slingshot.',
      'Raise <i>Softening</i> to 0.1: forces become gentle and the galaxies turn into fluffy clouds. Lower it to 0.002 in the dust cloud: violent clumping.',
      'Push <i>Bodies</i> to 65,536 and watch the frame time — doubling N quadruples the work. Then do the same in the solar system: nearly free.',
      'Switch <i>Color by</i> to “Where it started” after the collision to see how the two galaxies mixed.',
      'Turn on <i>Motion trails</i> in the solar system.',
    ],
    ask: [
      'GPU N-body gravity simulation with shared-memory tiling',
      'galaxy collision with tidal tails',
      'orbital mechanics with an asteroid belt of test particles',
      'additive HDR star rendering with bloom',
      'gravity well power-up that bends thousands of projectiles',
    ],
    perf: `<p>Work per step = N × (number of sources). 16,384² ≈ 268 M interactions ≈ 5 GFLOP — a few milliseconds on a mid-range GPU. 65,536 bodies
      is 16× more (~4 billion interactions): fine on desktop GPUs, too much for laptops at 60 fps. Real astrophysics codes use
      Barnes–Hut trees or particle-mesh methods (O(N log N)) to go to millions — at the cost of complexity.</p>`,
    api: `<p><b>WebGPU only.</b> The tiling trick needs compute shaders with <code>var&lt;workgroup&gt;</code> shared memory and barriers.
      WebGL2 can do N-body with fragment shaders (positions in float textures, every pixel loops over every body), but without shared memory each
      thread reads every body from texture memory — several times slower — and the code is much clumsier.</p>`,
    code: [
      {
        title: 'Tiled all-pairs gravity (one thread per body)',
        lang: 'wgsl',
        src: `var<workgroup> tile: array<vec4f, 256>;
@compute @workgroup_size(256)
fn step(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let me = pin[min(gid.x, u.count - 1u)];
  var acc = vec2f(0.0);
  for (var t = 0u; t < (u.sources + 255u) / 256u; t++) {
    let j = t * 256u + lid.x;
    tile[lid.x] = select(vec4f(0.0), pin[min(j, u.count - 1u)], j < u.sources);
    workgroupBarrier();                       // the tile is loaded
    for (var k = 0u; k < 256u; k++) {
      let d = tile[k].xy - me.xy;
      let inv = inverseSqrt(dot(d, d) + u.soft * u.soft);
      acc += d * (tile[k].z * inv * inv * inv);
    }
    workgroupBarrier();                       // everyone is done with it
  }
  if (gid.x >= u.count) { return; }           // (after the barriers!)
  vel[gid.x] = vec4f(vel[gid.x].xy + acc * u.G * u.dt, vel[gid.x].zw);
  pout[gid.x] = vec4f(me.xy + vel[gid.x].xy * u.dt, me.zw);
}`,
      },
    ],
    links: [
      { title: 'GPU Gems 3 — Fast N-Body Simulation with CUDA', url: 'https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda', note: 'the classic tiling write-up' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const test = ctx.testMode;
    const SU = gpu.uniforms({ count: 'u32', sources: 'u32', soft: 'f32', G: 'f32', dt: 'f32', hand: 'f32', mouse: 'vec2f' }, 'Sim');
    const DU = gpu.uniforms({ center: 'vec2f', zoom: 'f32', aspect: 'f32', px: 'vec2f', size: 'f32', bright: 'f32', vscale: 'f32', colorMode: 'f32' }, 'Draw');
    const TU = gpu.uniforms({ exposure: 'f32', bloom: 'f32', time: 'f32', fade: 'f32' }, 'Tone');

    const sim = gpu.compute({
      label: 'nbody',
      bindings: {
        u: { uniform: SU },
        pin: { storage: 'array<vec4f>', access: 'read' },
        pout: { storage: 'array<vec4f>', access: 'read_write' },
        vel: { storage: 'array<vec4f>', access: 'read_write' },
      },
      code: SIM_WGSL,
    });
    const draw = gpu.program({
      label: 'nbody-draw',
      bindings: { u: { uniform: DU }, pos: { storage: 'array<vec4f>', access: 'read' }, vel: { storage: 'array<vec4f>', access: 'read' } },
      code: DRAW_WGSL,
    });
    const drawPipe = draw.renderPipeline({ format: 'rgba16float', blend: 'add' });
    const MULT = { color: { srcFactor: 'zero', dstFactor: 'src', operation: 'add' }, alpha: { srcFactor: 'zero', dstFactor: 'src', operation: 'add' } };
    const fade = gpu.fullscreen({ label: 'nbody-fade', uniforms: TU, code: `fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(vec3f(u.fade), 1.0); }`, blend: MULT });
    const down = gpu.fullscreen({
      label: 'nbody-bloom-down',
      textures: ['src'],
      code: /* wgsl */ `fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let t = 1.0 / TEXSIZE(src);
  var c = vec3f(0.0);
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) { c += TEX(src, uv + vec2f(f32(x), f32(y)) * t * 1.5).rgb; } }
  c /= 9.0;
  return vec4f(max(c - vec3f(0.25), vec3f(0.0)), 1.0);
}`,
    });
    const blur = (dir) =>
      gpu.fullscreen({
        label: 'nbody-blur',
        textures: ['src'],
        code: /* wgsl */ `fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let t = vec2f(${dir}) / TEXSIZE(src);
  var w = array<f32, 5>(0.227, 0.194, 0.122, 0.054, 0.016);
  var c = TEX(src, uv).rgb * w[0];
  for (var i = 1; i < 5; i++) {
    c += (TEX(src, uv + t * f32(i) * 1.6).rgb + TEX(src, uv - t * f32(i) * 1.6).rgb) * w[i];
  }
  return vec4f(c, 1.0);
}`,
      });
    const blurH = blur('1.0, 0.0');
    const blurV = blur('0.0, 1.0');
    const tone = gpu.fullscreen({
      label: 'nbody-tone',
      uniforms: TU,
      textures: ['hdr', 'bloomTex'],
      include: ['color', 'hash'],
      code: /* wgsl */ `fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var c = TEX(hdr, uv).rgb * u.exposure + TEX(bloomTex, uv).rgb * u.bloom * 1.5;
  // deep-space backdrop with faint background stars
  var bg = mix(vec3f(0.008, 0.01, 0.025), vec3f(0.03, 0.015, 0.045), uv.y);
  let cell = floor(px / 2.0);
  let h = hash21(cell);
  bg += vec3f(0.5, 0.55, 0.7) * step(0.9985, h) * (0.15 + 0.5 * hash21(cell + 7.0));
  return vec4f(tonemapACES(c) + bg, 1.0);
}`,
    });

    let hdr = null;
    let bloomA = null;
    let bloomB = null;
    const makeTargets = () => {
      hdr?.destroy();
      bloomA?.destroy();
      bloomB?.destroy();
      hdr = gpu.target(ctx.width, ctx.height, { format: 'rgba16float', label: 'nbody-hdr' });
      const bw = Math.max(1, ctx.width >> 2);
      const bh = Math.max(1, ctx.height >> 2);
      bloomA = gpu.target(bw, bh, { format: 'rgba16float', label: 'nbody-bloomA' });
      bloomB = gpu.target(bw, bh, { format: 'rgba16float', label: 'nbody-bloomB' });
    };
    makeTargets();

    let bufs = null;
    let state = null;
    let flip = 0;
    let center = [0, 0];
    let zoomMul = 1;
    const readout = overlayTag(ctx);

    const build = () => {
      const N = Math.max(256, Math.round(test ? Math.min(ctx.params.count, 2048) : ctx.params.count));
      const init = initBodies(ctx.example, N, ctx.params.soft);
      if (bufs) Object.values(bufs).forEach((b) => b.destroy());
      bufs = {
        a: gpu.storage(init.pos, 'nbody-pos-a'),
        b: gpu.storage(init.pos.byteLength, 'nbody-pos-b'),
        vel: gpu.storage(init.vel, 'nbody-vel'),
      };
      state = { N, sources: Math.min(N, init.sources), zoom: init.zoom, vscale: init.vscale };
      flip = 0;
      center = [0, 0];
      zoomMul = 1;
    };
    build();

    return {
      resize() {
        makeTargets();
      },
      onAction(key) {
        if (key === 'reset') build();
      },
      onChange(key) {
        if (key === 'count') build();
      },
      onExample() {
        build();
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        const aspect = ctx.width / ctx.height;
        if (ctx.pointer.wheel) zoomMul = Math.min(8, Math.max(0.25, zoomMul * Math.exp(-ctx.pointer.wheel * 0.001)));
        const zoom = state.zoom * zoomMul;
        const mx = ((ctx.pointer.nx * 2 - 1) * aspect) / zoom + center[0];
        const my = -(ctx.pointer.ny * 2 - 1) / zoom + center[1];
        const baseDt = ctx.example === 'solar' ? 0.0018 : ctx.example === 'dust' ? 0.0028 : 0.0026;
        const steps = ctx.example === 'solar' ? 3 : 1;
        SU.set('count', state.N)
          .set('sources', state.sources)
          .set('soft', p.soft)
          .set('G', p.G)
          .set('dt', baseDt * p.speed)
          .set('hand', ctx.pointer.down ? (ctx.example === 'solar' ? 0.3 : 0.8) : 0)
          .set('mouse', [mx, my])
          .upload();
        const bright = (0.5 * Math.sqrt(16384 / state.N)) / Math.max(0.6, p.size * 0.6);
        DU.set('center', center)
          .set('zoom', zoom)
          .set('aspect', aspect)
          .set('px', [2 / ctx.width, 2 / ctx.height])
          .set('size', p.size * Math.max(1, ctx.dpr))
          .set('bright', bright)
          .set('vscale', state.vscale)
          .set('colorMode', p.colorMode === 'origin' ? 1 : 0)
          .upload();
        TU.set('exposure', p.exposure).set('bloom', p.bloom).set('time', ctx.time).set('fade', 0.86).upload();

        const groups = Math.ceil(state.N / TILE);
        if (!ctx.paused && p.speed > 0) {
          for (let s = 0; s < steps; s++) {
            const src = flip ? bufs.b : bufs.a;
            const dst = flip ? bufs.a : bufs.b;
            sim.dispatch(enc, 'step', groups, { u: SU, pin: src, pout: dst, vel: bufs.vel });
            flip ^= 1;
          }
        }
        const cur = flip ? bufs.b : bufs.a;
        if (p.trails) fade.draw(enc, hdr, {}, { clear: false });
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: hdr.view, loadOp: p.trails ? 'load' : 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }],
        });
        pass.setPipeline(drawPipe);
        pass.setBindGroup(0, draw.bind({ u: DU, pos: cur, vel: bufs.vel }));
        pass.draw(6, state.N);
        pass.end();
        down.draw(enc, bloomA, { src: hdr });
        blurH.draw(enc, bloomB, { src: bloomA });
        blurV.draw(enc, bloomA, { src: bloomB });
        tone.draw(enc, { view: ctx.target, format: gpu.format }, { hdr, bloomTex: bloomA });
        const inter = state.N * state.sources * steps;
        readout.textContent = `${state.N.toLocaleString()} × ${state.sources.toLocaleString()} = ${
          inter >= 1e9 ? (inter / 1e9).toFixed(2) + ' G' : (inter / 1e6).toFixed(1) + ' M'
        } interactions/frame`;
      },
    };
  },
};

// Slime mold (Physarum polycephalum), after Jeff Jones (2010) / Sage Jenson.
// Up to 2M agents each: sense the trail map at 3 points ahead → turn → move → deposit.
// Deposits are race-free: agents atomicAdd into a counter buffer; a second compute pass
// blurs + decays the trail map, adds the deposits and clears the counters.

import { readout, computeSeq } from './_shared-a.js';

const MAX_FOOD = 24;

export default {
  interaction: 'Hold the mouse to lure the slime. On “Road networks” click to place food, right-click to remove it.',
  examples: [
    {
      id: 'slime',
      label: 'Slime mold',
      kind: 'Abstract',
      note: 'A million agents that each follow a dumb rule — “turn toward the strongest smell ahead, leave smell behind” — self-organise into a living, pulsing transport network.',
      params: { count: 1000000, sensorAngle: 25, sensorDist: 14, turn: 35, speed: 1.3, deposit: 0.6, decay: 0.05, diffuse: 0.6, spawn: 'random', palette: 'gold' },
    },
    {
      id: 'roads',
      label: 'Road networks',
      kind: 'Real life',
      note: 'In 2010 researchers placed oat flakes in the pattern of cities around Tokyo, and the real slime mold grew a network remarkably like the actual rail system. Here food sources emit scent; the slime starts in the centre, explores, then thins out to efficient links. Click to add cities.',
      params: { count: 500000, sensorAngle: 30, sensorDist: 18, turn: 40, speed: 1.2, deposit: 0.4, decay: 0.08, diffuse: 0.5 },
    },
    {
      id: 'neon',
      label: 'Neon flow art',
      kind: 'Art',
      note: 'Three species, each attracted to its own trail and repelled by the others, so they carve out territories and flow around each other. Additive color makes the overlaps glow — great for menus and music visuals.',
      params: { count: 1200000, sensorAngle: 35, sensorDist: 20, turn: 30, speed: 1.6, deposit: 0.5, decay: 0.06, diffuse: 0.35 },
    },
  ],
  controls: [
    { type: 'heading', label: 'Agent behaviour' },
    { type: 'slider', key: 'sensorAngle', label: 'Sensor angle', min: 5, max: 90, step: 1, value: 25, unit: '°', help: 'Angle between the forward sensor and the left/right sensors.' },
    { type: 'slider', key: 'sensorDist', label: 'Sensor distance', min: 2, max: 50, step: 0.5, value: 14, unit: 'px', help: 'How far ahead agents “smell”. Bigger = coarser network.' },
    { type: 'slider', key: 'turn', label: 'Turn speed', min: 5, max: 90, step: 1, value: 35, unit: '°/step' },
    { type: 'slider', key: 'speed', label: 'Move speed', min: 0.3, max: 4, step: 0.05, value: 1.3, unit: 'px/step' },
    { type: 'heading', label: 'Trail map' },
    { type: 'slider', key: 'deposit', label: 'Deposit', min: 0.02, max: 3, step: 0.01, value: 0.6, log: true, help: 'Scent each agent leaves per step.' },
    { type: 'slider', key: 'decay', label: 'Decay', min: 0.005, max: 0.3, step: 0.001, value: 0.05, log: true, help: 'Fraction of scent that evaporates per step. Low = long-lived, thick veins.' },
    { type: 'slider', key: 'diffuse', label: 'Diffusion', min: 0, max: 1, step: 0.01, value: 0.6, help: 'How much the scent blurs into neighbouring cells each step.' },
    { type: 'heading', label: 'Agents' },
    {
      type: 'slider',
      key: 'count',
      label: 'Agents',
      min: 10000,
      max: 2000000,
      step: 1000,
      value: 1000000,
      log: true,
      format: (v) => Math.round(v).toLocaleString(),
    },
    { type: 'slider', key: 'steps', label: 'Steps per frame', min: 1, max: 4, step: 1, value: 1 },
    {
      type: 'select',
      key: 'spawn',
      label: 'Spawn pattern',
      value: 'random',
      options: [
        { value: 'ring', label: 'Ring facing inward' },
        { value: 'center', label: 'Burst from the centre' },
        { value: 'random', label: 'Random everywhere (fastest)' },
      ],
      showFor: ['slime'],
    },
    {
      type: 'select',
      key: 'palette',
      label: 'Colors',
      value: 'gold',
      options: [
        { value: 'gold', label: 'Gold & teal' },
        { value: 'magma', label: 'Magma' },
        { value: 'ice', label: 'Ice' },
      ],
      showFor: ['slime'],
    },
    { type: 'button', key: 'reset', label: 'Respawn agents', primary: true },
  ],
  about: {
    summary:
      'Slime-mold simulation: up to two million agents sense and follow each other’s chemical trails, and efficient branching networks grow out of nothing. Agents and trail map live entirely on the GPU.',
    what: `<p>The bright veins are a <b>trail map</b> (a scent texture). Each of the up-to-2M agents walks on it, sniffs ahead, turns toward the strongest scent and drops scent where it goes.
      Paths that many agents use get stronger and attract more agents; unused paths evaporate. It’s positive feedback plus decay — the same recipe as ant trails.</p>`,
    how: `<ol>
      <li><b>Sense</b>: each agent samples the trail map at three points: straight ahead and ±<i>sensor angle</i>, all at <i>sensor distance</i>.</li>
      <li><b>Rotate</b>: forward strongest → keep going; left or right strongest → turn that way by <i>turn speed</i>; forward weakest → turn randomly.</li>
      <li><b>Move</b> one step forward and <b>deposit</b> scent. Thousands of agents may hit the same cell at once, so the deposit is an
        <code>atomicAdd</code> into an integer counter buffer — no lost writes.</li>
      <li><b>Diffuse & decay</b>: a second compute pass blurs the trail map with a 3×3 box filter, adds this step’s deposits, multiplies by (1 − decay) and resets the counters.</li>
      <li>Food sources simply add scent every step. The network finds short paths between them because short, busy paths are reinforced the most.</li>
      <li><b>Multiple species</b> (Neon) store one scent per color channel; each species is attracted to its own channel and repelled by the others.</li>
    </ol>`,
    uses: [
      { title: 'Generative art & VFX', text: 'Organic veins, lightning-like networks, magical corruption spreading over a surface, living backgrounds.' },
      { title: 'Procedural roads & rivers', text: 'Place towns as food and let the slime suggest a road/river/cave network that connects them efficiently.' },
      { title: 'Swarm behaviour', text: 'The same sense-turn-deposit loop drives ant colonies, pheromone AI and crowd flow fields in strategy games.' },
    ],
    try: [
      'Lower <i>Sensor distance</i> to 4: the network becomes fine lace. Raise it to 40: a few thick highways.',
      'Set <i>Decay</i> very low (0.005): trails persist and the network freezes into a fixed pattern.',
      'Set <i>Sensor angle</i> above 60°: agents start circling, producing cells and bubbles.',
      'On <b>Road networks</b>, click a new city far from the others and watch a road grow to it.',
      'Lower <i>Agents</i> to 50,000: the network becomes sparse and wispy — density matters.',
    ],
    ask: [
      'a Physarum slime mold simulation with a million agents',
      'agent-based trail following (sense, rotate, move, deposit)',
      'atomic deposits into a trail map in a compute shader',
      'procedural road network that connects my towns',
      'neon multi-species flow art background',
    ],
    perf: `<p>Two dispatches per step: one thread per agent (3 texture samples + 1 atomic) and one per trail cell (9 reads). At 1M agents on a 1080p trail map that’s
      ~3M samples + 2M cells per step — a few milliseconds on a mid-range GPU. Cost scales with <i>agents + cells</i>, times steps per frame.</p>`,
    api: `<p><b>WebGPU only</b> as written: agents are scattered writes into a shared map, which needs storage buffers and atomics. In WebGL2 you could keep agents in a float texture
      and draw them as points with additive blending to deposit (that’s how older demos did it), but it’s clumsier and much slower for millions of agents.</p>`,
    code: [
      {
        title: 'Agent step: sense, rotate, move, deposit',
        lang: 'wgsl',
        src: `let F = sense(a.pos + dir(a.ang) * u.sensorDist, sp);
let L = sense(a.pos + dir(a.ang - u.sensorAngle) * u.sensorDist, sp);
let R = sense(a.pos + dir(a.ang + u.sensorAngle) * u.sensorDist, sp);
let r = rnd();
if (F > L && F > R) { }                                  // keep going
else if (F < L && F < R) { a.ang += (r - 0.5) * 2.0 * u.turn; }  // confused: random turn
else if (L > R) { a.ang -= u.turn * (0.5 + 0.5 * r); }
else if (R > L) { a.ang += u.turn * (0.5 + 0.5 * r); }
a.pos += dir(a.ang) * u.speed;
let cell = u32(a.pos.y) * u.gridW + u32(a.pos.x);
atomicAdd(&deposits[cell], 1u << (10u * sp));             // race-free deposit (10 bits per species)`,
      },
      {
        title: 'Trail pass: blur, add deposits, decay, clear',
        lang: 'wgsl',
        src: `let blur = (sum of the 3×3 neighbourhood) / 9.0;
var v = mix(old, blur, u.diffuse);
let d = atomicExchange(&deposits[i], 0u);               // read and reset the counters
v += vec4f(f32(d & 1023u), f32((d >> 10u) & 1023u), f32((d >> 20u) & 1023u), 0.0) * u.deposit;
v *= 1.0 - u.decay;`,
      },
    ],
    links: [
      { title: 'Sage Jenson — Physarum', url: 'https://cargocollective.com/sagejenson/physarum', note: 'the famous art pieces and a clear diagram' },
      { title: 'Jeff Jones — Characteristics of pattern formation… in a model of Physarum (2010)', url: 'https://doi.org/10.1162/artl.2010.16.2.16202' },
      { title: 'Tero et al. — Rules for Biologically Inspired Adaptive Network Design (Science, 2010)', url: 'https://doi.org/10.1126/science.1177894', note: 'the Tokyo rail experiment' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const MAX = ctx.testMode ? 120000 : 2000000;
    const U = gpu.uniforms(
      {
        grid: 'vec2f',
        count: 'u32',
        gridW: 'u32',
        sensorAngle: 'f32',
        sensorDist: 'f32',
        turn: 'f32',
        speed: 'f32',
        deposit: 'f32',
        decay: 'f32',
        diffuse: 'f32',
        species: 'f32',
        bounds: 'f32',
        nFood: 'f32',
        frameNo: 'u32',
        spawn: 'f32',
        seed: 'f32',
        food: `array<vec4f, ${MAX_FOOD}>`,
      },
      'Phys',
    );
    const agents = gpu.storage(MAX * 16, 'agents');
    let trail = null;
    let deposits = null;
    let gw = 0;
    let gh = 0;

    const sim = gpu.compute({
      label: 'physarum',
      bindings: {
        u: { uniform: U },
        agents: { storage: 'array<vec4f>', access: 'read_write' },
        deposits: { storage: 'array<atomic<u32>>', access: 'read_write' },
        trailIn: { texture: 'float' },
        sampR: { sampler: 'filtering' },
        trailOut: { storageTexture: 'rgba16float', access: 'write' },
      },
      include: ['hash'],
      code: /* wgsl */ `
var<private> rs: u32;
fn rnd() -> f32 { rs = pcg(rs); return f32(rs) / 4294967296.0; }
fn dir(a: f32) -> vec2f { return vec2f(cos(a), sin(a)); }

// how attractive the scent at p is for species sp
fn sense(p: vec2f, sp: u32) -> f32 {
  let t = textureSampleLevel(trailIn, sampR, p / u.grid, 0.0);
  if (u.species < 1.5) { return t.r; }
  let own = t[sp];
  return own * 1.0 - (t.r + t.g + t.b - own) * 0.55;
}

@compute @workgroup_size(256) fn spawn(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&agents)) { return; }
  rs = pcg(i * 747796405u + u32(u.seed) * 2891336453u);
  let c = u.grid * 0.5;
  let R = min(u.grid.x, u.grid.y);
  var pos = vec2f(rnd(), rnd()) * u.grid;
  var ang = rnd() * TAU;
  let mode = i32(u.spawn + 0.5);
  if (mode == 0) {
    // ring facing inward
    let a = rnd() * TAU;
    let r = R * (0.36 + 0.06 * sqrt(rnd()));
    pos = c + dir(a) * r;
    ang = a + PI + (rnd() - 0.5) * 0.4;
  } else if (mode == 1) {
    let a = rnd() * TAU;
    pos = c + dir(a) * R * 0.03 * sqrt(rnd());
    ang = a;
  }
  let sp = f32(i % 3u) * step(1.5, u.species);
  agents[i] = vec4f(pos, ang, sp);
}

@compute @workgroup_size(256) fn agentStep(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.count) { return; }
  var a = agents[i];
  let sp = u32(a.w);
  rs = pcg(i * 1664525u + u.frameNo * 1013904223u + bitcast<u32>(a.x));
  let F = sense(a.xy + dir(a.z) * u.sensorDist, sp);
  let L = sense(a.xy + dir(a.z - u.sensorAngle) * u.sensorDist, sp);
  let R = sense(a.xy + dir(a.z + u.sensorAngle) * u.sensorDist, sp);
  let r = rnd();
  if (F > L && F > R) {
    // keep going
  } else if (F < L && F < R) {
    a.z += (r - 0.5) * 2.0 * u.turn;
  } else if (L > R) {
    a.z -= u.turn * (0.5 + 0.5 * r);
  } else if (R > L) {
    a.z += u.turn * (0.5 + 0.5 * r);
  }
  var p = a.xy + dir(a.z) * u.speed;
  if (u.bounds > 0.5) {
    // petri dish: bounce off a circular wall
    let c = u.grid * 0.5;
    let rad = min(u.grid.x, u.grid.y) * 0.47;
    if (length(p - c) > rad) {
      p = a.xy;
      a.z = atan2(c.y - p.y, c.x - p.x) + (rnd() - 0.5) * 1.6;
    }
  } else {
    p = p - u.grid * floor(p / u.grid);   // wrap around (torus)
  }
  a = vec4f(p, a.z, a.w);
  agents[i] = a;
  let cell = vec2u(clamp(p, vec2f(0.0), u.grid - vec2f(1.0)));
  // one species: full 32-bit counter; three species: 10 bits each
  atomicAdd(&deposits[cell.y * u.gridW + cell.x], select(1u << (10u * sp), 1u, u.species < 1.5));
}

@compute @workgroup_size(8, 8) fn trailStep(@builtin(global_invocation_id) gid: vec3u) {
  let sz = vec2i(u.grid);
  let p = vec2i(gid.xy);
  if (p.x >= sz.x || p.y >= sz.y) { return; }
  var sum = vec4f(0.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = (p + vec2i(x, y) + sz) % sz;
      sum += textureLoad(trailIn, q, 0);
    }
  }
  let old = textureLoad(trailIn, p, 0);
  var v = mix(old, sum / 9.0, u.diffuse);
  let i = u32(p.y) * u.gridW + u32(p.x);
  let d = atomicExchange(&deposits[i], 0u);
  if (u.species < 1.5) { v.r += f32(d) * u.deposit; }
  else { v += vec4f(f32(d & 1023u), f32((d >> 10u) & 1023u), f32((d >> 20u) & 1023u), 0.0) * u.deposit; }
  // food sources / lures add scent every step
  let fp = vec2f(p) + 0.5;
  for (var k = 0u; k < u32(u.nFood); k++) {
    let f = u.food[k];
    let dd = length(fp - f.xy);
    v += vec4f(f.w * exp(-dd * dd / (f.z * f.z)));
  }
  v *= 1.0 - u.decay;
  textureStore(trailOut, gid.xy, min(v, vec4f(2000.0)));
}`,
    });

    // ---------------------------------------------------------------- display
    const V = gpu.uniforms(
      { res: 'vec2f', grid: 'vec2f', exposure: 'f32', mode: 'f32', palette: 'f32', time: 'f32', nFood: 'f32', mouseDown: 'f32', mouse: 'vec2f', food: `array<vec4f, ${MAX_FOOD}>` },
      'View',
    );
    const display = gpu.fullscreen({
      label: 'physarum-display',
      uniforms: V,
      textures: ['trail'],
      include: ['hash', 'noise', 'color'],
      code: /* wgsl */ `
fn rampGold(x: f32) -> vec3f {
  let c1 = vec3f(0.01, 0.02, 0.05);
  let c2 = vec3f(0.02, 0.28, 0.36);
  let c3 = vec3f(0.95, 0.72, 0.28);
  let c4 = vec3f(1.0, 0.98, 0.9);
  return mix(mix(c1, c2, smoothstep(0.0, 0.35, x)), mix(c3, c4, smoothstep(0.75, 1.0, x)), smoothstep(0.3, 0.75, x));
}
fn rampMagma(x: f32) -> vec3f {
  return clamp(vec3f(1.7 * x - 0.1, 2.0 * x * x - 0.2, 0.5 * sin(x * 3.5) + 0.5 * x * x * x), vec3f(0.0), vec3f(1.0)) + vec3f(0.01, 0.0, 0.03);
}
fn rampIce(x: f32) -> vec3f {
  return mix(vec3f(0.0, 0.01, 0.04), mix(vec3f(0.1, 0.35, 0.9), vec3f(0.9, 0.97, 1.0), smoothstep(0.45, 1.0, x)), smoothstep(0.0, 0.5, x));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let t = TEX(trail, uv);
  let mode = i32(u.mode + 0.5);
  let gp = uv * u.grid;
  if (mode == 2) {
    // neon: three species, additive, tone mapped
    let c = 1.0 - exp(-t.rgb * u.exposure);
    var col = c.r * vec3f(1.0, 0.15, 0.55) + c.g * vec3f(0.1, 0.9, 1.0) + c.b * vec3f(0.55, 0.3, 1.0);
    col = col * 0.9 + vec3f(0.006, 0.004, 0.015);
    col = 1.0 - exp(-col * 1.4);
    return vec4f(col, 1.0);
  }
  if (mode == 1) {
    // petri dish: agar plate, oat flakes, yellow slime
    let c = u.grid * 0.5;
    let R = min(u.grid.x, u.grid.y) * 0.47;
    let dd = length(gp - c);
    var col = mix(vec3f(0.2, 0.17, 0.12), vec3f(0.12, 0.1, 0.07), smoothstep(0.0, R, dd));
    col *= 0.9 + 0.1 * valueNoise(gp * 0.08);
    let x = 1.0 - exp(-t.r * u.exposure);
    let slime = mix(vec3f(0.85, 0.65, 0.08), vec3f(1.0, 0.93, 0.45), smoothstep(0.5, 1.0, x));
    col = mix(col, slime, smoothstep(0.05, 0.6, x));
    col += vec3f(1.0, 0.85, 0.3) * x * x * 0.15;
    // oat flakes (food)
    for (var k = 0; k < ${MAX_FOOD}; k++) {
      if (f32(k) >= u.nFood) { break; }
      let f = u.food[k];
      let q = gp - f.xy;
      let fl = length(q * vec2f(1.0, 1.4) + 0.3 * vec2f(sin(q.y * 0.5), 0.0)) - f.z * 0.55;
      col = mix(col, vec3f(0.9, 0.82, 0.62) * (0.85 + 0.15 * valueNoise(q * 0.7 + f32(k))), 1.0 - smoothstep(-0.5, 0.8, fl));
    }
    // dish rim + outside
    let rim = abs(dd - R - 2.0);
    col = mix(col, vec3f(0.55, 0.6, 0.62), (1.0 - smoothstep(0.5, 2.5, rim)) * 0.6);
    col = mix(col, vec3f(0.03, 0.035, 0.045), smoothstep(R + 3.0, R + 5.0, dd));
    col += vec3f(0.9) * pow(max(0.0, 1.0 - abs(dd - R * 0.92) / 3.0), 2.0) * smoothstep(0.5, 0.0, abs(atan2(gp.y - c.y, gp.x - c.x) + 2.2)) * 0.25;
    return vec4f(col, 1.0);
  }
  let x = 1.0 - exp(-t.r * u.exposure);
  let pal = i32(u.palette + 0.5);
  var col = rampGold(x);
  if (pal == 1) { col = rampMagma(x); }
  if (pal == 2) { col = rampIce(x); }
  if (u.mouseDown > 0.5) {
    let d = length(gp - u.mouse);
    col += vec3f(0.6, 0.9, 1.0) * (1.0 - smoothstep(0.0, 2.0, abs(d - 12.0))) * 0.5;
  }
  return vec4f(col, 1.0);
}`,
    });

    // ---------------------------------------------------------------- state
    let frameNo = 0;
    let seed = 1;
    let needSpawn = true;
    let foods = [];
    const tag = readout(ctx);

    const alloc = () => {
      const scale = ctx.testMode ? 2 : Math.max(1, ctx.dpr);
      let w = Math.round(ctx.width / scale);
      let h = Math.round(ctx.height / scale);
      const maxCells = 1920 * 1200;
      if (w * h > maxCells) {
        const k = Math.sqrt(maxCells / (w * h));
        w = Math.round(w * k);
        h = Math.round(h * k);
      }
      if (w === gw && h === gh && trail) return;
      trail?.destroy();
      deposits?.destroy();
      gw = w;
      gh = h;
      trail = gpu.pingPong(w, h, { format: 'rgba16float', label: 'trail' });
      deposits = gpu.storage(w * h * 4, 'deposits');
      needSpawn = true;
    };
    const cities = () => {
      // irregular "cities" inside the petri dish, with the starting point in the middle
      const c = [gw / 2, gh / 2];
      const R = Math.min(gw, gh) * 0.4;
      const pts = [
        [0.0, 0.0], [0.55, -0.2], [0.35, 0.55], [-0.4, 0.5], [-0.65, -0.05], [-0.3, -0.62], [0.25, -0.7], [0.8, 0.3],
        [-0.8, 0.45], [0.05, 0.85], [0.72, -0.62], [-0.72, -0.55],
      ];
      foods = pts.map(([x, y]) => [c[0] + x * R, c[1] + y * R]);
    };
    alloc();
    cities();

    const reset = () => {
      needSpawn = true;
      seed++;
      if (ctx.example === 'roads') cities();
    };

    return {
      resize() {
        alloc();
        if (ctx.example === 'roads') cities();
      },
      onAction(key) {
        if (key === 'reset') reset();
      },
      onExample() {
        reset();
      },
      onChange(key) {
        if (key === 'spawn') reset();
      },
      frame(ctx) {
        const p = ctx.params;
        const ex = ctx.example;
        const enc = ctx.encoder;
        const mode = { slime: 0, roads: 1, neon: 2 }[ex] ?? 0;
        const n = Math.max(1000, Math.min(MAX, Math.round(ctx.testMode ? Math.min(p.count, MAX) : p.count)));
        const ptr = ctx.pointer;
        const mx = ptr.nx * gw;
        const my = ptr.ny * gh;

        // food & lures
        if (mode === 1 && ptr.clicked) {
          if (ptr.button === 2) {
            let bi = -1;
            let bd = 1e9;
            foods.forEach((f, i) => {
              const d = Math.hypot(f[0] - mx, f[1] - my);
              if (d < bd) {
                bd = d;
                bi = i;
              }
            });
            if (bi >= 0 && bd < 40) foods.splice(bi, 1);
          } else {
            foods.push([mx, my]);
            if (foods.length > MAX_FOOD) foods.shift();
          }
        }
        const foodArr = new Float32Array(MAX_FOOD * 4);
        let nFood = 0;
        if (mode === 1) {
          for (const f of foods) {
            foodArr.set([f[0], f[1], 7, 2.5], nFood * 4);
            nFood++;
          }
        } else if (ptr.down) {
          foodArr.set([mx, my, 10, 6], 0);
          nFood = 1;
        }

        const spawnMode = mode === 1 ? 1 : mode === 2 ? 2 : { ring: 0, center: 1, random: 2 }[p.spawn] ?? 0;
        U.setAll({
          grid: [gw, gh],
          count: n,
          gridW: gw,
          sensorAngle: (p.sensorAngle * Math.PI) / 180,
          sensorDist: p.sensorDist,
          turn: (p.turn * Math.PI) / 180,
          speed: p.speed,
          deposit: p.deposit,
          decay: p.decay,
          diffuse: p.diffuse,
          species: mode === 2 ? 3 : 1,
          bounds: mode === 1 ? 1 : 0,
          nFood,
          frameNo,
          spawn: spawnMode,
          seed,
          food: foodArr,
        });
        U.upload();

        const cs = computeSeq(enc, 'physarum');
        if (needSpawn) {
          needSpawn = false;
          cs.run(sim, 'spawn', Math.ceil(MAX / 256), { u: U, agents, deposits, trailIn: trail.read, sampR: 'linear-repeat', trailOut: trail.write });
          cs.end();
          gpu.clear(enc, trail.a, [0, 0, 0, 0]);
          gpu.clear(enc, trail.b, [0, 0, 0, 0]);
          // deposits may hold stale counts: zero them
          enc.clearBuffer(deposits);
        } else {
          cs.end();
        }
        if (!ctx.paused) {
          const steps = ctx.testMode ? 4 : Math.round(p.steps);
          const run = computeSeq(enc, 'physarum-steps');
          for (let s = 0; s < steps; s++) {
            frameNo++;
            // (frameNo only seeds randomness; the same value for all sub-steps is fine)
            const res = { u: U, agents, deposits, trailIn: trail.read, sampR: 'linear-repeat', trailOut: trail.write };
            run.run(sim, 'agentStep', Math.ceil(n / 256), res);
            run.run(sim, 'trailStep', [Math.ceil(gw / 8), Math.ceil(gh / 8)], res);
            trail.swap();
          }
          run.end();
        }

        // exposure: equilibrium trail ≈ deposit × density / decay
        const density = n / (gw * gh);
        const exposure = (p.decay / Math.max(1e-4, p.deposit)) / Math.max(0.05, density) * (mode === 2 ? 0.9 : 0.35);
        V.setAll({
          res: [ctx.width, ctx.height],
          grid: [gw, gh],
          exposure,
          mode,
          palette: { gold: 0, magma: 1, ice: 2 }[p.palette] ?? 0,
          time: ctx.time,
          nFood,
          mouseDown: ptr.down ? 1 : 0,
          mouse: [mx, my],
          food: foodArr,
        });
        display.draw(enc, { view: ctx.target, format: gpu.format }, { trail: trail.read });
        tag.textContent = `${n.toLocaleString()} agents · trail map ${gw}×${gh}${mode === 1 ? ` · ${foods.length} food sources` : ''}`;
      },
    };
  },
};

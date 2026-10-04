// Flocking (Boids) — Craig Reynolds' three rules on the GPU with a uniform-grid spatial hash.
// Per frame (all compute, one pass):
//   clearCells → countCells (atomicAdd) → scan (prefix sum) → scatter (counting sort) → update
// then every boid is drawn as an instanced, SDF-shaped quad oriented along its velocity.

import { readout, computeSeq } from './_shared-a.js';

const BOID_WGSL = /* wgsl */ `
struct Boid { pos: vec2f, vel: vec2f, rnd: f32, team: f32, nbrs: f32, id: u32 };
`;

// example id -> behaviour of the world (not user controls)
const MODES = {
  flock: { mode: 0, wrap: 1, shape: 0, teams: 0 },
  birds: { mode: 1, wrap: 0, shape: 1, teams: 0 },
  fish: { mode: 2, wrap: 0, shape: 2, teams: 0 },
  fleets: { mode: 3, wrap: 1, shape: 3, teams: 1 },
};

export default {
  interaction: 'Hold the mouse to scare the flock (it is a predator). On “Fish school” the shark follows your cursor.',
  examples: [
    {
      id: 'flock',
      label: 'Three rules',
      kind: 'Abstract',
      note: 'Every dart only looks at neighbours inside its <b>view radius</b> and blends three steering urges: <b>separation</b> (don’t crowd), <b>alignment</b> (match heading), <b>cohesion</b> (move to the local centre). Color = heading. Turn on <i>Show spatial grid</i> to see how the GPU finds neighbours fast.',
      params: { count: 8000, size: 7, radius: 40, sep: 1.6, ali: 1.0, coh: 0.9, speed: 230, trails: 0 },
    },
    {
      id: 'birds',
      label: 'Starling murmuration',
      kind: 'Real life',
      note: 'Thousands of starlings at dusk. Strong alignment + cohesion make the whole flock fold and ripple like one fluid body. Hold the mouse: a falcon appears and the flock bursts away from it, exactly how real murmurations evade predators.',
      params: { count: 24000, size: 3.6, radius: 28, sep: 1.3, ali: 1.6, coh: 1.3, speed: 260, trails: 0 },
    },
    {
      id: 'fish',
      label: 'Fish school + shark',
      kind: 'In a game',
      note: 'An ambient reef for an underwater level. The shark follows your mouse (or patrols on its own) and the school opens up around it — the classic “fountain effect”. Note the cheap shadows: the fish layer is re-sampled with an offset.',
      params: { count: 2400, size: 10, radius: 48, sep: 1.8, ali: 1.2, coh: 1.0, speed: 190, trails: 0 },
    },
    {
      id: 'fleets',
      label: 'Space fleets',
      kind: 'In a game',
      note: 'Two factions of fighters: each flocks only with its own team and dodges the enemy, so the swarms weave through each other. Additive engine glow + motion trails. Hold the mouse to set a rally point for both fleets.',
      params: { count: 4000, size: 5.5, radius: 44, sep: 2.4, ali: 1.3, coh: 0.8, speed: 300, trails: 0.8 },
    },
  ],
  controls: [
    { type: 'heading', label: 'The three rules' },
    { type: 'slider', key: 'sep', label: 'Separation', min: 0, max: 4, step: 0.05, value: 1.6, help: 'Steer away from neighbours that are too close (inner 40% of the view radius).' },
    { type: 'slider', key: 'ali', label: 'Alignment', min: 0, max: 4, step: 0.05, value: 1.0, help: 'Steer towards the average heading of neighbours.' },
    { type: 'slider', key: 'coh', label: 'Cohesion', min: 0, max: 4, step: 0.05, value: 0.9, help: 'Steer towards the average position of neighbours.' },
    { type: 'slider', key: 'radius', label: 'View radius', min: 10, max: 100, step: 1, value: 40, help: 'How far a boid can see (world units; the screen is 1000 tall). Also the grid cell size.' },
    { type: 'slider', key: 'speed', label: 'Max speed', min: 40, max: 600, step: 5, value: 230, help: 'Units per second. Min speed is 40% of this, so boids never stall.' },
    { type: 'heading', label: 'Flock' },
    {
      type: 'slider',
      key: 'count',
      label: 'Boids',
      min: 256,
      max: 131072,
      step: 1,
      value: 8000,
      log: true,
      format: (v) => Math.round(v).toLocaleString(),
      help: 'Thanks to the spatial grid the cost grows ~linearly, not with the square of the count.',
    },
    { type: 'slider', key: 'size', label: 'Boid size', min: 2, max: 20, step: 0.1, value: 7 },
    {
      type: 'select',
      key: 'colorMode',
      label: 'Color by',
      value: 'heading',
      options: [
        { value: 'heading', label: 'Heading (direction)' },
        { value: 'speed', label: 'Speed' },
        { value: 'crowd', label: 'Neighbour count' },
      ],
      showFor: ['flock'],
    },
    { type: 'slider', key: 'trails', label: 'Motion trails', min: 0, max: 0.97, step: 0.01, value: 0, help: 'Fade the previous frame instead of clearing it.' },
    { type: 'toggle', key: 'grid', label: 'Show spatial grid', value: false, help: 'Cells are tinted by how many boids they hold. Yellow = the 3×3 cells a boid under the cursor searches.' },
    { type: 'button', key: 'reset', label: 'Respawn flock', primary: true },
  ],
  about: {
    summary:
      'Boids (Craig Reynolds, 1986): every agent follows three local rules — separation, alignment, cohesion — and realistic flocks, schools and swarms emerge. Here up to 131k of them run in a WebGPU compute shader.',
    what: `<p>Each triangle/bird/fish/ship is an independent agent. Nobody leads, nobody knows the shape of the flock — every agent only reacts to the
      few neighbours inside its <b>view radius</b>. The swirling, splitting and re-merging you see is <i>emergent</i> behaviour.</p>`,
    how: `<ol>
      <li><b>Neighbour search is the hard part.</b> Checking every pair is N² — 100k boids would be 10 billion checks per frame. Instead the world is split into a
        <b>uniform grid</b> whose cells are as big as the view radius, so a boid only has to look at the 3×3 cells around it.</li>
      <li><b>Counting sort on the GPU</b> builds that grid every frame in four compute dispatches:
        <i>clear</i> the per-cell counters → <i>count</i> (each boid does <code>atomicAdd</code> on its cell and remembers its slot) →
        <i>prefix sum</i> (turns counts into start offsets) → <i>scatter</i> (copy each boid to <code>start[cell] + slot</code>). Now boids of one cell are contiguous in memory.</li>
      <li><b>Update</b>: one thread per boid walks the 9 cells, accumulates <b>separation</b> (away from close neighbours, weighted by 1/distance),
        <b>alignment</b> (average velocity) and <b>cohesion</b> (average position). Each rule becomes a <i>steering force</i>
        <code>desired velocity − current velocity</code>, clamped and weighted by the sliders.</li>
      <li>Extra forces: flee the predator, stay inside the screen (birds, fish), or chase a rally point (fleets). Speed is clamped between a min and max.</li>
      <li><b>Drawing</b>: one instanced quad per boid, rotated along its velocity in the vertex shader (reading the same storage buffer). The fragment shader
        draws the shape as a signed-distance function, so it is anti-aliased at any size, and animates wing flaps / tail wiggles from a per-boid random phase.</li>
    </ol>`,
    uses: [
      { title: 'Ambient life', text: 'Birds over a level, fish in a reef, bats in a cave, fireflies — cheap background life that reacts to the player.' },
      { title: 'Swarm enemies', text: 'Vampire Survivors-style hordes, insect swarms, drone clouds: flocking + a seek force toward the player.' },
      { title: 'RTS & crowds', text: 'Unit groups that move in formation without overlapping (separation is the core of steering-based crowd avoidance).' },
      { title: 'Film & VFX', text: 'Boids were invented for film (Batman Returns’ bat swarms and penguin army, 1992) and still drive crowd shots.' },
    ],
    try: [
      'Set <i>Separation</i> to 0: boids collapse into dense clumps. Set <i>Cohesion</i> to 0: the flock dissolves into a gas of small groups.',
      'Set <i>Alignment</i> to 4 on <b>Three rules</b>: everyone quickly marches in parallel lanes.',
      'Turn on <i>Show spatial grid</i>, then drag <i>View radius</i>: the grid cells resize with it. Bigger radius = more boids per check = slower.',
      'Push <i>Boids</i> to 131k on <b>Starling murmuration</b> and watch the fps.',
      'On <b>Fish school</b>, move the shark slowly into the school, then fast: the school splits around it and closes behind.',
    ],
    ask: [
      'GPU boids flocking with a spatial hash grid',
      'a fish school that flees from the player',
      'a swarm of enemies that flocks and chases the player',
      'counting sort / prefix sum for neighbour search in a compute shader',
      'birds flocking in the background of my level',
    ],
    perf: `<p>Per frame: 5 compute dispatches + 1 instanced draw. The cost is roughly <i>boids × neighbours checked</i>. The neighbour count grows with
      <i>view radius²</i> and with density, so tight flocks are the expensive case — the update loop caps candidates at 320 per boid (real starlings only track ~7 neighbours!).
      100k boids is comfortable on a desktop GPU; on the CPU in JavaScript you would struggle past ~2–5k with naive N² checks.</p>`,
    api: `<p><b>WebGPU only.</b> The grid build needs <code>atomicAdd</code>, workgroup shared memory (for the prefix sum) and scattered writes into storage buffers —
      none of which exist in WebGL2. WebGL2 flocking demos either stay small (CPU) or store boids in textures with very limited neighbour search.</p>`,
    code: [
      {
        title: 'Building the grid: count with atomics, then scatter (counting sort)',
        lang: 'wgsl',
        src: `@compute @workgroup_size(256) fn countCells(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= u.count) { return; }
  let ci = cellIndex(boids[i].pos);
  let slot = atomicAdd(&counts[ci], 1u);   // my position inside the cell
  bins[i] = vec2u(ci, slot);
}
// … scan(): starts[c] = counts[0] + … + counts[c-1]  (prefix sum in shared memory)
@compute @workgroup_size(256) fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= u.count) { return; }
  let b = bins[i];
  sorted[starts[b.x] + b.y] = boids[i];     // boids of one cell are now contiguous
}`,
      },
      {
        title: 'The three rules as steering forces',
        lang: 'wgsl',
        src: `// inside the 3×3 cell loop, for each neighbour o within the view radius:
n += 1.0;
ali += o.vel;                         // alignment: sum of velocities
coh += d;                             // cohesion: sum of offsets to neighbours
if (d2 < sepR2) { sep -= d / d2; }    // separation: push away, stronger when close

// steer(dir) = normalize(dir) * maxSpeed - vel, clamped to maxForce
var acc = steer(sep, b.vel) * u.sepW;
if (n > 0.0) {
  acc += steer(ali / n, b.vel) * u.aliW;
  acc += steer(coh / n, b.vel) * u.cohW;
}
b.vel += acc * u.dt;
b.vel = clampSpeed(b.vel, minSpeed, maxSpeed);
b.pos += b.vel * u.dt;`,
      },
    ],
    links: [
      { title: 'Craig Reynolds — Boids', url: 'https://www.red3d.com/cwr/boids/', note: 'the original page by the inventor' },
      { title: 'The Nature of Code — Autonomous Agents', url: 'https://natureofcode.com/autonomous-agents/', note: 'steering behaviours explained' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const MAX = ctx.testMode ? 8192 : 131072;
    const MAXCELLS = 65536;
    const WORLD_H = 1000;

    const U = gpu.uniforms(
      {
        world: 'vec2f',
        cell: 'vec2f',
        mouse: 'vec4f', // xy world, z radius, w mode (0 none, 1 flee, 2 attract)
        count: 'u32',
        cellsX: 'u32',
        cellsY: 'u32',
        cellCount: 'u32',
        radius: 'f32',
        sepW: 'f32',
        aliW: 'f32',
        cohW: 'f32',
        maxSpeed: 'f32',
        dt: 'f32',
        time: 'f32',
        wrap: 'f32',
        frameNo: 'u32',
        maxChecks: 'u32',
        size: 'f32',
        shape: 'f32',
        colorMode: 'f32',
        teams: 'f32',
        centerPull: 'f32',
      },
      'Sim',
    );

    const boids = gpu.storage(MAX * 32, 'boids');
    const sorted = gpu.storage(MAX * 32, 'boids-sorted');
    const bins = gpu.storage(MAX * 8, 'boid-bins');
    const counts = gpu.storage(MAXCELLS * 4, 'cell-counts');
    const starts = gpu.storage(MAXCELLS * 4, 'cell-starts');

    const sim = gpu.compute({
      label: 'boids-sim',
      bindings: {
        u: { uniform: U },
        boids: { storage: 'array<Boid>', access: 'read_write' },
        sorted: { storage: 'array<Boid>', access: 'read_write' },
        bins: { storage: 'array<vec2u>', access: 'read_write' },
        counts: { storage: 'array<atomic<u32>>', access: 'read_write' },
        starts: { storage: 'array<u32>', access: 'read_write' },
      },
      include: ['hash'],
      code:
        BOID_WGSL +
        /* wgsl */ `
fn cellOf(p: vec2f) -> vec2i {
  let c = vec2i(floor(p / u.cell));
  return clamp(c, vec2i(0), vec2i(i32(u.cellsX) - 1, i32(u.cellsY) - 1));
}
fn cellIndex(p: vec2f) -> u32 { let c = cellOf(p); return u32(c.y) * u.cellsX + u32(c.x); }

@compute @workgroup_size(256) fn clearCells(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.cellCount) { atomicStore(&counts[gid.x], 0u); }
}

@compute @workgroup_size(256) fn countCells(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let ci = cellIndex(boids[i].pos);
  let slot = atomicAdd(&counts[ci], 1u);
  bins[i] = vec2u(ci, slot);
}

// Exclusive prefix sum over all cells with ONE workgroup: each thread sums a chunk,
// the 256 chunk sums are scanned in shared memory (Hillis–Steele), then written back.
var<workgroup> part: array<u32, 256>;
@compute @workgroup_size(256) fn scan(@builtin(local_invocation_id) lid: vec3u) {
  let n = u.cellCount;
  let per = (n + 255u) / 256u;
  let s0 = lid.x * per;
  var sum = 0u;
  for (var k = 0u; k < per; k++) {
    let c = s0 + k;
    if (c < n) { sum += atomicLoad(&counts[c]); }
  }
  part[lid.x] = sum;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off * 2u) {
    var add = 0u;
    if (lid.x >= off) { add = part[lid.x - off]; }
    workgroupBarrier();
    part[lid.x] = part[lid.x] + add;
    workgroupBarrier();
  }
  var run = part[lid.x] - sum;
  for (var k = 0u; k < per; k++) {
    let c = s0 + k;
    if (c < n) {
      starts[c] = run;
      run += atomicLoad(&counts[c]);
    }
  }
}

@compute @workgroup_size(256) fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let b = bins[i];
  sorted[starts[b.x] + b.y] = boids[i];
}

fn steer(dir: vec2f, vel: vec2f, maxS: f32, maxF: f32) -> vec2f {
  let l = length(dir);
  if (l < 1e-5) { return vec2f(0.0); }
  var s = dir / l * maxS - vel;
  let sl = length(s);
  if (sl > maxF) { s = s / sl * maxF; }
  return s;
}

fn rand01(s: u32) -> f32 { return f32(pcg(s)) / 4294967296.0; }

@compute @workgroup_size(256) fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.count) { return; }
  var b = sorted[i];
  let cc = cellOf(b.pos);
  let gridN = vec2i(i32(u.cellsX), i32(u.cellsY));
  let r2 = u.radius * u.radius;
  let sepR2 = r2 * 0.16;
  var sep = vec2f(0.0);
  var ali = vec2f(0.0);
  var coh = vec2f(0.0);
  var foe = vec2f(0.0);
  var n = 0.0;
  var checks = 0u;
  let seed = pcg(b.id * 747796405u + u.frameNo * 2891336453u);
  let rot = seed % 9u; // start at a random cell so the candidate cap doesn't bias direction
  for (var k = 0u; k < 9u; k++) {
    let kk = (k + rot) % 9u;
    var c = cc + vec2i(i32(kk % 3u) - 1, i32(kk / 3u) - 1);
    if (u.wrap > 0.5) {
      c = (c + gridN) % gridN;
    } else if (c.x < 0 || c.y < 0 || c.x >= gridN.x || c.y >= gridN.y) {
      continue;
    }
    let ci = u32(c.y) * u.cellsX + u32(c.x);
    let st = starts[ci];
    let cnt = atomicLoad(&counts[ci]);
    for (var j = st; j < st + cnt; j++) {
      if (checks >= u.maxChecks) { break; }
      checks++;
      if (j == i) { continue; }
      let o = sorted[j];
      var d = o.pos - b.pos;
      if (u.wrap > 0.5) { d -= u.world * round(d / u.world); }
      let d2 = dot(d, d);
      if (d2 >= r2 || d2 < 1e-4) { continue; }
      if (u.teams > 0.5 && o.team != b.team) {
        foe -= d / d2;   // other faction: only avoid
        continue;
      }
      n += 1.0;
      ali += o.vel;
      coh += d;
      if (d2 < sepR2) { sep -= d / d2; }
    }
  }

  var maxS = u.maxSpeed;
  let maxF = maxS * 2.2;
  var acc = steer(sep, b.vel, maxS, maxF) * u.sepW * 1.5;
  if (n > 0.0) {
    acc += steer(ali / n, b.vel, maxS, maxF) * u.aliW;
    acc += steer(coh / n, b.vel, maxS, maxF) * u.cohW;
  }
  acc += steer(foe, b.vel, maxS, maxF) * 2.0;

  // predator (flee) or rally point (attract)
  if (u.mouse.w > 0.5) {
    var dm = u.mouse.xy - b.pos;
    if (u.wrap > 0.5) { dm -= u.world * round(dm / u.world); }
    let dl = length(dm);
    if (u.mouse.w < 1.5) {
      if (dl < u.mouse.z) {
        let panic = 1.0 - dl / u.mouse.z;
        maxS = maxS * (1.0 + 0.6 * panic);
        acc += steer(-dm, b.vel, maxS, maxF * 3.0) * (2.0 + 6.0 * panic);
      }
    } else {
      acc += steer(dm, b.vel, maxS, maxF) * 1.1;
    }
  }

  // soft walls (birds/fish) and a gentle pull to the centre
  if (u.wrap < 0.5) {
    let m = u.world.y * 0.1;
    var inward = vec2f(0.0);
    inward.x = smoothstep(m, 0.0, b.pos.x) - smoothstep(u.world.x - m, u.world.x, b.pos.x);
    inward.y = smoothstep(m, 0.0, b.pos.y) - smoothstep(u.world.y - m, u.world.y, b.pos.y);
    acc += inward * maxF * 2.5;
  }
  if (u.centerPull > 0.0) {
    acc += steer(u.world * 0.5 - b.pos, b.vel, maxS, maxF) * u.centerPull;
  }
  // a little wander noise
  let a = rand01(seed ^ 0x9e3779b9u) * TAU;
  acc += vec2f(cos(a), sin(a)) * maxF * 0.12;

  b.vel += acc * u.dt;
  let sp = length(b.vel);
  let minS = u.maxSpeed * 0.4;
  if (sp > maxS) { b.vel = b.vel / sp * maxS; }
  else if (sp < minS) {
    if (sp < 1e-4) { b.vel = vec2f(cos(a), sin(a)) * minS; } else { b.vel = b.vel / sp * minS; }
  }
  b.pos += b.vel * u.dt;
  if (u.wrap > 0.5) {
    b.pos = b.pos - u.world * floor(b.pos / u.world);
  } else {
    b.pos = clamp(b.pos, vec2f(1.0), u.world - vec2f(1.0));
  }
  b.nbrs = n;
  boids[b.id] = b;
}`,
    });

    // ---------------------------------------------------------------- drawing the boids
    const draw = gpu.program({
      label: 'boids-draw',
      bindings: { u: { uniform: U }, boids: { storage: 'array<Boid>', access: 'read' } },
      include: ['sdf', 'color'],
      code:
        BOID_WGSL +
        /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) lp: vec2f,
  @location(1) col: vec4f,
  @location(2) info: vec4f,
};

@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let b = boids[ii];
  let sp = length(b.vel);
  var dir = vec2f(1.0, 0.0);
  if (sp > 1e-4) { dir = b.vel / sp; }
  let side = vec2f(-dir.y, dir.x);
  var ext = vec2f(1.4, 1.0);
  if (u.shape > 0.5 && u.shape < 1.5) { ext = vec2f(1.2, 2.0); }
  if (u.shape > 1.5 && u.shape < 2.5) { ext = vec2f(1.6, 0.8); }
  if (u.shape > 2.5) { ext = vec2f(2.4, 1.2); }
  let lp = corners[vi] * ext;
  let wp = b.pos + (dir * lp.x + side * lp.y) * u.size;
  var o: VOut;
  o.pos = vec4f(wp.x / u.world.x * 2.0 - 1.0, 1.0 - wp.y / u.world.y * 2.0, 0.0, 1.0);
  o.lp = lp;
  let ang = atan2(dir.y, dir.x);
  let spd = sp / u.maxSpeed;
  var col = vec4f(1.0);
  if (u.shape < 0.5) {
    if (u.colorMode < 0.5) { col = vec4f(hsv2rgb(vec3f(fract(ang / TAU + 1.0), 0.62, 1.0)), 1.0); }
    else if (u.colorMode < 1.5) { col = vec4f(mix(vec3f(0.2, 0.35, 1.0), vec3f(1.0, 0.85, 0.3), smoothstep(0.4, 1.3, spd)), 1.0); }
    else { col = vec4f(palette(clamp(b.nbrs / 40.0, 0.0, 1.0) * 0.8, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.6, 0.4, 0.2)), 1.0); }
  } else if (u.shape < 1.5) {
    col = vec4f(0.05, 0.04, 0.07, 0.95);
  } else if (u.shape < 2.5) {
    // silvery backs that flash as the fish turn (light catches the scales)
    let flash = pow(0.5 + 0.5 * sin(ang * 2.0 + b.rnd * 0.8), 8.0);
    col = vec4f(mix(vec3f(0.1, 0.16, 0.24), vec3f(0.85, 0.95, 1.0), flash * 0.8), 1.0);
  } else {
    col = vec4f(mix(vec3f(0.35, 0.85, 1.0), vec3f(1.0, 0.5, 0.2), b.team), 1.0);
  }
  o.col = col;
  o.info = vec4f(u.time * (9.0 + 5.0 * b.rnd) + b.rnd * TAU, spd, b.team, b.rnd);
  return o;
}

fn shapeDist(p: vec2f, info: vec4f) -> f32 {
  if (u.shape < 0.5) {
    // arrowhead dart
    let tri = sdTriangle(p, vec2f(1.25, 0.0), vec2f(-0.85, 0.75), vec2f(-0.85, -0.75));
    let notch = sdTriangle(p, vec2f(-0.35, 0.0), vec2f(-1.1, 0.9), vec2f(-1.1, -0.9));
    return max(tri, -notch);
  }
  if (u.shape < 1.5) {
    // bird seen from below: body + two flapping wings
    let flap = cos(info.x);
    let span = 1.75 * (0.55 + 0.45 * flap);
    let sweep = -0.35 - 0.25 * flap;
    let body = sdEllipseApprox(p - vec2f(0.1, 0.0), vec2f(0.75, 0.22));
    let wl = sdSegment(p, vec2f(0.15, 0.0), vec2f(sweep, span)) - 0.2 + 0.08 * abs(p.y) / 1.75;
    let wr = sdSegment(p, vec2f(0.15, 0.0), vec2f(sweep, -span)) - 0.2 + 0.08 * abs(p.y) / 1.75;
    let tail = sdTriangle(p, vec2f(-0.4, 0.0), vec2f(-0.95, 0.28), vec2f(-0.95, -0.28));
    return min(min(body, tail), min(wl, wr));
  }
  if (u.shape < 2.5) {
    // fish from above: body + wiggling tail
    let w = sin(info.x * 0.9) * (0.15 + 0.25 * info.y);
    var q = p;
    q.y -= w * max(0.0, -q.x) * 0.6;
    let body = sdEllipseApprox(q - vec2f(0.25, 0.0), vec2f(1.0, 0.3));
    let tail = sdTriangle(q, vec2f(-0.55, 0.0), vec2f(-1.4, 0.42), vec2f(-1.4, -0.42));
    return min(body, tail);
  }
  // fighter ship
  let hull = sdTriangle(p, vec2f(1.3, 0.0), vec2f(-0.8, 0.85), vec2f(-0.8, -0.85));
  let cut = sdTriangle(p, vec2f(-0.2, 0.0), vec2f(-1.0, 0.6), vec2f(-1.0, -0.6));
  return max(hull, -cut);
}

@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let d = shapeDist(i.lp, i.info);
  let w = max(fwidth(d), 1e-4);
  let cov = clamp(0.5 - d / w, 0.0, 1.0);
  if (u.shape > 2.5) {
    // body + additive exhaust glow (alpha 0 = purely additive with premultiplied blending)
    let q = i.lp - vec2f(-0.75, 0.0);
    let flick = 0.75 + 0.25 * sin(i.info.x * 3.0);
    let glow = exp(-dot(q * vec2f(1.2, 3.0), q * vec2f(1.2, 3.0))) * flick;
    let hot = mix(i.col.rgb, vec3f(1.0), 0.5);
    return vec4f(i.col.rgb * cov * 0.9 + hot * glow * 1.4, cov * 0.85);
  }
  var c = i.col.rgb;
  if (u.shape > 1.5) {
    // darker spine down the middle of each fish
    c *= 0.75 + 0.25 * smoothstep(0.0, 0.25, abs(i.lp.y));
  }
  let a = cov * i.col.a;
  return vec4f(c * a, a);
}`,
    });
    const drawPipe = draw.renderPipeline({ format: 'rgba16float', blend: 'premultiplied' });

    // trails: fade the layer instead of clearing it
    const FU = gpu.uniforms({ fade: 'f32' }, 'Fade');
    const fade = gpu.fullscreen({
      label: 'boids-fade',
      uniforms: FU,
      blend: 'premultiplied',
      code: 'fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(0.0, 0.0, 0.0, u.fade); }',
    });

    // ---------------------------------------------------------------- composite + backgrounds
    const V = gpu.uniforms(
      {
        world: 'vec2f',
        cell: 'vec2f',
        cells: 'vec2f',
        res: 'vec2f',
        mouse: 'vec4f',
        pred: 'vec4f',
        time: 'f32',
        mode: 'f32',
        showGrid: 'f32',
        radius: 'f32',
        fear: 'f32',
        avg: 'f32',
        predOn: 'f32',
        predPhase: 'f32',
      },
      'View',
    );
    const composite = gpu.fullscreen({
      label: 'boids-composite',
      uniforms: V,
      textures: ['layer'],
      storage: { counts: 'array<u32>' },
      include: ['math', 'hash', 'noise', 'sdf', 'color'],
      code: /* wgsl */ `
fn bgFlock(uv: vec2f, px: vec2f) -> vec3f {
  let p = centerUV(px, u.res);
  var c = mix(vec3f(0.035, 0.04, 0.075), vec3f(0.012, 0.014, 0.03), clamp(length(p) * 1.3, 0.0, 1.0));
  let g = fract(px / 32.0);
  let dotv = 1.0 - smoothstep(0.0, 1.6 / 32.0, length(g - 0.5));
  return c + vec3f(0.03, 0.035, 0.06) * dotv;
}

fn bgBirds(uv: vec2f, px: vec2f) -> vec3f {
  let aspect = u.res.x / u.res.y;
  let y = uv.y;
  var c = mix(vec3f(0.16, 0.18, 0.38), vec3f(0.62, 0.42, 0.55), smoothstep(0.0, 0.55, y));
  c = mix(c, vec3f(1.0, 0.62, 0.38), smoothstep(0.45, 0.85, y));
  c = mix(c, vec3f(1.0, 0.82, 0.55), smoothstep(0.75, 0.95, y));
  // sun
  let sp = vec2f(uv.x * aspect, uv.y) - vec2f(0.72 * aspect, 0.8);
  let sd = length(sp);
  c += vec3f(1.0, 0.75, 0.45) * (exp(-sd * 9.0) * 0.6 + smoothstep(0.055, 0.05, sd) * 0.8);
  // thin lit clouds
  let cl = fbm(vec2f(uv.x * 2.5 + u.time * 0.005, uv.y * 16.0), 4);
  let band = smoothstep(0.1, 0.45, cl) * smoothstep(0.2, 0.55, y) * smoothstep(0.85, 0.6, y);
  c = mix(c, vec3f(1.0, 0.55, 0.5), band * 0.35);
  // tree line / hills silhouette
  let h = 0.9 - 0.035 * fbm(vec2f(uv.x * 3.0, 1.7), 3) - 0.025 * pow(abs(sin(uv.x * 37.0)), 6.0) * hash11(floor(uv.x * 37.0 / PI));
  let hill = smoothstep(h, h + 0.003, y);
  c = mix(c, vec3f(0.07, 0.05, 0.09), hill);
  return c;
}

fn caustic(p: vec2f, t: f32) -> f32 {
  let w = p + 0.35 * vec2f(perlin(p * 0.7 + t * 0.2), perlin(p * 0.7 + 5.0 - t * 0.2));
  let v = voronoiEx(w, 1.0, t);
  return pow(1.0 - smoothstep(0.0, 0.18, v.y - v.x), 2.0);
}

fn bgSea(uv: vec2f, wp: vec2f) -> vec3f {
  let p = wp / 1000.0;
  let n = fbm(p * 5.0, 4);
  // sandy lagoon floor with ripples, rocks and sea grass
  var floor_ = vec3f(0.95, 0.88, 0.68) * (0.85 + 0.2 * n);
  floor_ *= 0.94 + 0.06 * sin(p.x * 90.0 + p.y * 30.0 + n * 10.0);
  let rk = fbm(p * 2.4 + vec2f(7.3, 1.1), 5);
  let rocks = smoothstep(0.38, 0.44, rk);
  let weed = smoothstep(0.24, 0.34, rk) * (1.0 - rocks);
  floor_ = mix(floor_, vec3f(0.36, 0.55, 0.28) * (0.6 + 0.6 * valueNoise(p * 90.0)), weed * 0.75);
  floor_ = mix(floor_, vec3f(0.5, 0.47, 0.42) * (0.55 + 0.5 * fbm(p * 14.0, 3)), rocks);
  // water absorbs red first: tint toward turquoise
  var c = floor_ * vec3f(0.45, 0.85, 0.85) + vec3f(0.0, 0.07, 0.1);
  let cz = caustic(p * 13.0, u.time * 0.7) * 0.6 + caustic(p * 19.0 + 3.0, -u.time * 0.5) * 0.4;
  c += vec3f(0.75, 1.0, 0.95) * cz * 0.28;
  let vig = length((uv - 0.5) * vec2f(1.0, 0.8));
  c = mix(c, vec3f(0.0, 0.2, 0.3), smoothstep(0.3, 0.85, vig) * 0.8);
  return c;
}

fn bgSpace(uv: vec2f, px: vec2f) -> vec3f {
  let p = centerUV(px, u.res);
  let neb = fbm(p * 2.2 + vec2f(3.0, 1.0), 5);
  let neb2 = fbm(p * 3.1 - vec2f(1.0, 4.0), 4);
  var c = vec3f(0.006, 0.008, 0.02);
  c += vec3f(0.18, 0.06, 0.26) * smoothstep(-0.1, 0.6, neb) * 0.35;
  c += vec3f(0.03, 0.16, 0.22) * smoothstep(0.0, 0.7, neb2) * 0.3;
  for (var l = 0; l < 2; l++) {
    let sc = 70.0 + f32(l) * 110.0;
    let g = floor(p * sc);
    let h = hash22(g + f32(l) * 17.0);
    let fp = fract(p * sc) - h;
    let star = smoothstep(0.08, 0.0, length(fp)) * step(0.9, hash21(g + 3.0 + f32(l)));
    c += vec3f(0.8, 0.85, 1.0) * star * (0.5 + 0.5 * sin(u.time * 2.0 + h.x * 30.0));
  }
  return c;
}

fn sdShark(p0: vec2f, t: f32) -> f32 {
  var p = p0;
  p.y += 0.09 * sin(t * 5.0 - p.x * 3.0) * smoothstep(0.4, -1.2, p.x);
  let body = sdEllipseApprox(p - vec2f(0.05, 0.0), vec2f(1.0, 0.19));
  let finL = sdTriangle(p, vec2f(0.3, 0.1), vec2f(-0.15, 0.56), vec2f(-0.02, 0.1));
  let finR = sdTriangle(p, vec2f(0.3, -0.1), vec2f(-0.15, -0.56), vec2f(-0.02, -0.1));
  let tail = sdTriangle(p, vec2f(-0.8, 0.0), vec2f(-1.32, 0.34), vec2f(-1.22, -0.3));
  return min(min(body, tail), min(finL, finR));
}

fn sdHawk(p: vec2f, t: f32) -> f32 {
  let flap = cos(t * 7.0);
  let span = 1.9 * (0.65 + 0.35 * flap);
  let body = sdEllipseApprox(p - vec2f(0.1, 0.0), vec2f(0.8, 0.2));
  let wl = sdBezier(p, vec2f(0.2, 0.0), vec2f(0.3, span * 0.6), vec2f(-0.5, span)) - 0.22;
  let wr = sdBezier(p, vec2f(0.2, 0.0), vec2f(0.3, -span * 0.6), vec2f(-0.5, -span)) - 0.22;
  let tail = sdTriangle(p, vec2f(-0.4, 0.0), vec2f(-1.1, 0.35), vec2f(-1.1, -0.35));
  return min(min(body, tail), min(wl, wr));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let wp = uv * u.world;
  let mode = i32(u.mode + 0.5);
  var bg = vec3f(0.0);
  if (mode == 0) { bg = bgFlock(uv, px); }
  else if (mode == 1) { bg = bgBirds(uv, px); }
  else if (mode == 2) { bg = bgSea(uv, wp); }
  else { bg = bgSpace(uv, px); }

  // predator shadow + fish shadows on the sea floor
  let pdir = u.pred.zw;
  let pside = vec2f(-pdir.y, pdir.x);
  if (mode == 2) {
    let soff = vec2f(14.0, 18.0); // light from the top-left: shadows fall down-right
    let sh = TEX(layer, uv - soff / u.world).a;
    bg *= 1.0 - 0.38 * clamp(sh, 0.0, 1.0);
    let rs = wp - (u.pred.xy + soff * 1.6);
    let ls = vec2f(dot(rs, pdir), dot(rs, pside)) / 105.0;
    let ds = sdShark(ls, u.predPhase) * 105.0;
    bg *= 1.0 - 0.4 * (1.0 - smoothstep(-6.0, 10.0, ds));
  }

  let L = TEX(layer, uv);
  var col = bg * (1.0 - clamp(L.a, 0.0, 1.0));
  if (mode == 3) { col += vec3f(1.0) - exp(-L.rgb * 1.3); } else { col += L.rgb; }

  if (u.predOn > 0.5) {
    let r = wp - u.pred.xy;
    if (mode == 2) {
      let lp = vec2f(dot(r, pdir), dot(r, pside)) / 105.0;
      let d = sdShark(lp, u.predPhase) * 105.0;
      let cov = clamp(0.5 - d / max(fwidth(d), 1e-3), 0.0, 1.0);
      let shadeS = mix(vec3f(0.2, 0.25, 0.3), vec3f(0.42, 0.48, 0.52), smoothstep(0.05, 0.3, abs(lp.y)));
      col = mix(col, shadeS, cov);
    } else if (mode == 1) {
      let lp = vec2f(dot(r, pdir), dot(r, pside)) / 16.0;
      let d = sdHawk(lp, u.predPhase) * 16.0;
      let cov = clamp(0.5 - d / max(fwidth(d), 1e-3), 0.0, 1.0);
      col = mix(col, vec3f(0.02, 0.015, 0.02), cov);
    } else if (mode == 0) {
      let ring = abs(length(r) - u.fear) - 1.5;
      col += vec3f(1.0, 0.35, 0.3) * (1.0 - smoothstep(0.0, 2.5, ring)) * 0.5;
      col += vec3f(1.0, 0.3, 0.25) * exp(-length(r) / 18.0) * 0.6;
    } else {
      let ring = abs(length(r) - 26.0 - 6.0 * sin(u.time * 6.0)) - 1.0;
      col += vec3f(0.6, 1.0, 0.7) * (1.0 - smoothstep(0.0, 2.5, ring)) * 0.7;
    }
  }

  if (u.showGrid > 0.5) {
    let cf = wp / u.cell;
    let ci = clamp(floor(cf), vec2f(0.0), u.cells - vec2f(1.0));
    let idx = u32(ci.y) * u32(u.cells.x) + u32(ci.x);
    let cnt = f32(counts[idx]);
    let heat = clamp(cnt / max(u.avg * 3.0, 1.0), 0.0, 1.0);
    let heatCol = mix(vec3f(0.1, 0.3, 1.0), vec3f(1.0, 0.25, 0.4), heat);
    col = mix(col, heatCol, 0.28 * step(0.5, cnt) * (0.4 + 0.6 * heat));
    let fw = fwidth(cf);
    let g = fract(cf);
    let gl = max(1.0 - smoothstep(0.0, fw.x * 1.2, min(g.x, 1.0 - g.x)), 1.0 - smoothstep(0.0, fw.y * 1.2, min(g.y, 1.0 - g.y)));
    col = mix(col, vec3f(0.6, 0.7, 1.0), gl * 0.25);
    if (u.mouse.w > 0.5) {
      let mc = floor(u.mouse.xy / u.cell);
      if (abs(ci.x - mc.x) <= 1.0 && abs(ci.y - mc.y) <= 1.0) {
        col = mix(col, vec3f(1.0, 0.85, 0.2), 0.12 + gl * 0.6);
      }
      let rr = abs(length(wp - u.mouse.xy) - u.radius) - 1.0;
      col = mix(col, vec3f(1.0, 0.9, 0.3), (1.0 - smoothstep(0.0, 2.0, rr)) * 0.9);
    }
  }
  return vec4f(col, 1.0);
}`,
    });

    // ---------------------------------------------------------------- state & spawning
    let layer = gpu.target(ctx.width, ctx.height, { format: 'rgba16float', label: 'boid-layer' });
    let world = [WORLD_H * (ctx.width / ctx.height), WORLD_H];
    let frameNo = 0;
    const pred = { x: world[0] * 0.3, y: world[1] * 0.5, vx: 1, vy: 0, dx: 1, dy: 0, phase: 0 };
    let lastMouse = [0, 0];
    let mouseDir = [1, 0];
    const tag = readout(ctx);

    // the layer is cleared lazily in frame() (clearing needs the frame's encoder)
    let clearLayer = true;
    const respawn = () => {
      world = [WORLD_H * (ctx.width / ctx.height), WORLD_H];
      const f = spawnData();
      gpu.queue.writeBuffer(boids, 0, f);
      clearLayer = true;
    };
    function spawnData() {
      // Spawn boids as pre-formed groups (each with a shared heading) so the scene starts out
      // looking like flocks instead of noise. The simulation takes over from there.
      const ex = ctx.example;
      const f = new Float32Array(MAX * 8);
      const u32 = new Uint32Array(f.buffer);
      const [W, H] = world;
      const speed = (ctx.params.speed || 200) * 0.75;
      const used = Math.max(1, Math.min(MAX, Math.round(ctx.params.count || 1000)));
      const nGroups = ex === 'birds' ? 1 : ex === 'fish' ? 6 : ex === 'fleets' ? 10 : 36;
      const per = used / nGroups;
      const gauss = () => {
        const u = Math.max(1e-6, Math.random());
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
      };
      const groups = [];
      for (let g = 0; g < nGroups; g++) {
        const team = g & 1;
        let cx = W * (0.1 + 0.8 * Math.random());
        let cy = H * (0.1 + 0.8 * Math.random());
        let heading = Math.random() * Math.PI * 2;
        let spread = Math.sqrt(per) * (ex === 'fish' ? 7 : ex === 'fleets' ? 4 : 6);
        if (ex === 'birds') {
          cx = W * 0.45;
          cy = H * 0.4;
          spread = H * 0.11;
        } else if (ex === 'fleets') {
          cx = team ? W * (0.78 + 0.12 * Math.random()) : W * (0.1 + 0.12 * Math.random());
          cy = H * (0.2 + 0.6 * Math.random());
          heading = Math.atan2(H * 0.5 - cy, W * 0.5 - cx) + (Math.random() - 0.5) * 0.6;
        }
        groups.push({ cx, cy, heading, spread, team });
      }
      for (let i = 0; i < MAX; i++) {
        const g = groups[i % nGroups];
        let x = g.cx + gauss() * g.spread * (ex === 'birds' ? 1.9 : 1);
        let y = g.cy + gauss() * g.spread * (ex === 'birds' ? 0.8 : 1);
        let a = g.heading + gauss() * 0.25;
        if (ex === 'birds') {
          // a slowly rotating ball of starlings
          a = Math.atan2(y - g.cy, x - g.cx) + Math.PI / 2 + gauss() * 0.2;
        }
        x = ((x % W) + W) % W;
        y = ((y % H) + H) % H;
        f[i * 8] = x;
        f[i * 8 + 1] = y;
        f[i * 8 + 2] = Math.cos(a) * speed;
        f[i * 8 + 3] = Math.sin(a) * speed;
        f[i * 8 + 4] = Math.random();
        f[i * 8 + 5] = g.team;
        f[i * 8 + 6] = 0;
        u32[i * 8 + 7] = i;
      }
      return f;
    }
    respawn();

    return {
      resize(w, h) {
        layer.destroy();
        layer = gpu.target(w, h, { format: 'rgba16float', label: 'boid-layer' });
        clearLayer = true;
      },
      onAction(key) {
        if (key === 'reset') respawn();
      },
      onExample() {
        respawn();
      },
      frame(ctx) {
        const p = ctx.params;
        const M = MODES[ctx.example] || MODES.flock;
        const enc = ctx.encoder;
        world = [WORLD_H * (ctx.width / ctx.height), WORLD_H];
        const n = Math.max(1, Math.min(MAX, Math.round(p.count)));
        const radius = p.radius;
        const cellsX = Math.max(3, Math.min(256, Math.floor(world[0] / radius)));
        const cellsY = Math.max(3, Math.min(256, Math.floor(world[1] / radius)));
        const cellCount = cellsX * cellsY;
        const cell = [world[0] / cellsX, world[1] / cellsY];
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 30);

        // pointer in world units
        const ptr = ctx.pointer;
        const mx = ptr.nx * world[0];
        const my = ptr.ny * world[1];
        const mdx = mx - lastMouse[0];
        const mdy = my - lastMouse[1];
        const ml = Math.hypot(mdx, mdy);
        if (ml > 0.5) mouseDir = [mdx / ml, mdy / ml];
        lastMouse = [mx, my];

        // predator logic
        let mouseMode = 0;
        let fear = 0;
        let predOn = 0;
        if (M.mode === 2) {
          // shark: follows the cursor, or patrols a lazy figure-eight
          const t = ctx.time;
          const tx = ptr.over ? mx : world[0] * (0.5 + 0.36 * Math.sin(t * 0.23));
          const ty = ptr.over ? my : world[1] * (0.5 + 0.3 * Math.sin(t * 0.37 + 1.0));
          let dvx = (tx - pred.x) * 2.2;
          let dvy = (ty - pred.y) * 2.2;
          const dl = Math.hypot(dvx, dvy);
          const maxv = 420;
          if (dl > maxv) {
            dvx *= maxv / dl;
            dvy *= maxv / dl;
          }
          const k = 1 - Math.exp(-dt * 2.5);
          pred.vx += (dvx - pred.vx) * k;
          pred.vy += (dvy - pred.vy) * k;
          pred.x += pred.vx * dt;
          pred.y += pred.vy * dt;
          const sl = Math.hypot(pred.vx, pred.vy);
          if (sl > 8) {
            pred.dx = pred.vx / sl;
            pred.dy = pred.vy / sl;
          }
          pred.phase += dt * (1 + sl / 150);
          mouseMode = 1;
          fear = 230;
          predOn = 1;
        } else if (ptr.down && ptr.over !== false) {
          pred.x = mx;
          pred.y = my;
          pred.dx = mouseDir[0];
          pred.dy = mouseDir[1];
          pred.phase += dt;
          if (M.mode === 3) {
            mouseMode = 2;
          } else {
            mouseMode = 1;
            fear = M.mode === 1 ? 210 : 160;
          }
          predOn = 1;
        }

        U.setAll({
          world,
          cell,
          mouse: [pred.x, pred.y, fear, mouseMode],
          count: n,
          cellsX,
          cellsY,
          cellCount,
          radius,
          sepW: p.sep,
          aliW: p.ali,
          cohW: p.coh,
          maxSpeed: p.speed,
          dt,
          time: ctx.time,
          wrap: M.wrap,
          frameNo,
          maxChecks: 320,
          size: p.size,
          shape: M.shape,
          colorMode: p.colorMode === 'speed' ? 1 : p.colorMode === 'crowd' ? 2 : 0,
          teams: M.teams,
          centerPull: M.mode === 1 ? 0.12 : M.mode === 3 ? 0.08 : 0,
        });
        U.upload();

        if (dt > 0) {
          frameNo++;
          const res = { u: U, boids, sorted, bins, counts, starts };
          const cs = computeSeq(enc, 'boids');
          cs.run(sim, 'clearCells', Math.ceil(cellCount / 256), res)
            .run(sim, 'countCells', Math.ceil(n / 256), res)
            .run(sim, 'scan', 1, res)
            .run(sim, 'scatter', Math.ceil(n / 256), res)
            .run(sim, 'update', Math.ceil(n / 256), res);
          cs.end();
        }

        // boid layer (faded for trails or cleared)
        const trails = p.trails || 0;
        let drawBoids = true;
        if (clearLayer || trails <= 0) {
          gpu.clear(enc, layer, [0, 0, 0, 0]);
          clearLayer = false;
        } else if (dt > 0) {
          FU.set('fade', 1 - Math.pow(trails, dt * 60));
          fade.draw(enc, layer, {}, { clear: false });
        } else {
          drawBoids = false; // paused with trails: the layer already holds this frame
        }
        if (drawBoids) {
          const pass = enc.beginRenderPass({
            colorAttachments: [{ view: layer.view, loadOp: 'load', storeOp: 'store' }],
          });
          pass.setPipeline(drawPipe);
          pass.setBindGroup(0, draw.bind({ u: U, boids }));
          pass.draw(6, n);
          pass.end();
        }

        V.setAll({
          world,
          cell,
          cells: [cellsX, cellsY],
          res: [ctx.width, ctx.height],
          mouse: [mx, my, ptr.down ? 1 : 0, ptr.over ? 1 : 0],
          pred: [pred.x, pred.y, pred.dx, pred.dy],
          time: ctx.time,
          mode: M.mode,
          showGrid: p.grid ? 1 : 0,
          radius,
          fear,
          avg: n / cellCount,
          predOn,
          predPhase: pred.phase * 2.0 + ctx.time,
        });
        // grid highlight follows the cursor whenever it is over the canvas
        V.set('mouse', [mx, my, ptr.down ? 1 : 0, ptr.over ? 1 : 0]);
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { layer, counts });

        tag.textContent = `${n.toLocaleString()} boids · grid ${cellsX}×${cellsY} = ${cellCount.toLocaleString()} cells · ~${(n / cellCount).toFixed(1)} per cell`;
      },
    };
  },
};

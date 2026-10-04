// Falling Sand (Noita-style) — every cell of the screen is a particle, simulated on the GPU.
// Race-free thanks to the Margolus neighbourhood: each thread owns one 2×2 block and only
// moves material inside it; the block grid shifts by one cell every step so material can
// cross block borders. No two threads ever write the same cell → no atomics, no races.

import { readout, computeSeq, stepper } from './_shared-a.js';

const M = { empty: 0, sand: 1, water: 2, stone: 3, wood: 4, fire: 5, smoke: 6, steam: 7, oil: 8, lava: 9, spout: 10, glass: 11, ember: 12 };
const MAX_SUB = 16;

const MAT_WGSL = /* wgsl */ `
const EMPTY: u32 = 0u;
const SAND: u32 = 1u;
const WATER: u32 = 2u;
const STONE: u32 = 3u;
const WOOD: u32 = 4u;
const FIRE: u32 = 5u;
const SMOKE: u32 = 6u;
const STEAM: u32 = 7u;
const OIL: u32 = 8u;
const LAVA: u32 = 9u;
const SPOUT: u32 = 10u;
const GLASS: u32 = 11u;
const EMBER: u32 = 12u;
const OUTSIDE: u32 = 255u;

// a cell is one u32: material | shade << 8 | life << 16 | flags << 24 (bit 24 = flow direction)
fn matOf(c: u32) -> u32 { return c & 0xFFu; }
fn shadeOf(c: u32) -> u32 { return (c >> 8u) & 0xFFu; }
fn lifeOf(c: u32) -> u32 { return (c >> 16u) & 0xFFu; }
fn flagsOf(c: u32) -> u32 { return c >> 24u; }
fn mk(m: u32, shade: u32, life: u32, flags: u32) -> u32 { return m | (shade << 8u) | (min(life, 255u) << 16u) | (flags << 24u); }
fn withLife(c: u32, life: u32) -> u32 { return (c & 0xFF00FFFFu) | (min(life, 255u) << 16u); }

fn density(m: u32) -> f32 {
  if (m == SAND) { return 8.0; }
  if (m == LAVA) { return 6.0; }
  if (m == WATER) { return 4.0; }
  if (m == OIL) { return 3.0; }
  if (m == FIRE) { return 0.5; }
  if (m == SMOKE) { return 0.3; }
  if (m == STEAM) { return 0.2; }
  return 1.0; // air
}
fn movable(m: u32) -> bool {
  return !(m == STONE || m == WOOD || m == SPOUT || m == GLASS || m == EMBER || m == OUTSIDE);
}
fn isLiquid(m: u32) -> bool { return m == WATER || m == OIL || m == LAVA; }
fn isGas(m: u32) -> bool { return m == FIRE || m == SMOKE || m == STEAM; }
`;

export default {
  interaction: 'Paint with the mouse (pick a material on the right). Right-drag erases.',
  examples: [
    {
      id: 'sandbox',
      label: 'Sandbox',
      kind: 'Abstract',
      note: 'Every pixel-sized cell is a particle with a material. Sand piles up, water and oil flow and layer by density, fire burns wood and oil, water puts out fire as steam, steam rains back down. A burning tree is already lit — try pouring water on it.',
      params: { material: 'sand', brush: 10, speed: 6 },
    },
    {
      id: 'hourglass',
      label: 'Hourglass',
      kind: 'Real life',
      note: 'Sand trickling through a narrow neck forms a cone below — granular flow emerges from a single “fall down, else slide diagonally” rule. Press <b>Flip</b> to turn it over (a compute pass mirrors the whole grid).',
      params: { material: 'sand', brush: 6, speed: 5 },
    },
    {
      id: 'volcano',
      label: 'Volcano',
      kind: 'In a game',
      note: 'A lava vent fills the crater and overflows. Lava sets trees alight, and where it meets the sea it hardens into dark rock and boils the water into steam — emergent gameplay from a handful of material interactions.',
      params: { material: 'lava', brush: 8, speed: 7 },
    },
  ],
  controls: [
    {
      type: 'select',
      key: 'material',
      label: 'Material',
      value: 'sand',
      options: [
        { value: 'sand', label: '🟨 Sand' },
        { value: 'water', label: '🟦 Water' },
        { value: 'oil', label: '🟫 Oil (floats, flammable)' },
        { value: 'lava', label: '🟧 Lava' },
        { value: 'fire', label: '🔥 Fire' },
        { value: 'wood', label: '🪵 Wood (static, burns)' },
        { value: 'stone', label: '⬜ Stone (static)' },
        { value: 'steam', label: '☁️ Steam' },
        { value: 'erase', label: '✖ Eraser' },
      ],
      help: 'Powders and liquids are sprayed into empty cells; solids overwrite.',
    },
    { type: 'slider', key: 'brush', label: 'Brush size', min: 2, max: 40, step: 1, value: 10, unit: 'px' },
    { type: 'slider', key: 'speed', label: 'Steps per frame', min: 1, max: 16, step: 1, value: 6, help: 'Margolus sub-steps per frame. Each step moves a particle at most one cell.' },
    { type: 'slider', key: 'cell', label: 'Cell size', min: 2, max: 8, step: 1, value: 3, unit: 'px', help: 'Pixels per simulated particle (smaller = more particles).' },
    { type: 'slider', key: 'glow', label: 'Fire glow', min: 0, max: 2, step: 0.05, value: 1, help: 'Strength of the bloom around fire and lava.' },
    { type: 'toggle', key: 'blocks', label: 'Show Margolus blocks', value: false, help: 'Overlay the 2×2 blocks of the current step (zoom in with a big cell size).' },
    { type: 'button', key: 'flip', label: 'Flip hourglass', help: 'Mirror the grid vertically.' },
    { type: 'button', key: 'reset', label: 'Reset scene', primary: true },
  ],
  about: {
    summary:
      'Noita-style “falling sand”: every cell on screen is a particle of sand, water, fire, oil, lava… updated in parallel by a compute shader using race-free 2×2 Margolus blocks.',
    what: `<p>The screen is a grid of tiny cells. Each stores a material plus a few bits of state (a random shade, a lifetime, a flow direction).
      Many times per frame the GPU applies simple local rules — <i>fall if the cell below is lighter</i>, <i>slide diagonally</i>, <i>flow sideways</i>,
      <i>burn</i>, <i>evaporate</i> — and complex behaviour emerges.</p>`,
    how: `<ol>
      <li><b>The race problem</b>: on a CPU you update cells one by one (usually bottom-up, alternating left/right). On a GPU thousands of threads run at once —
        if two grains try to move into the same empty cell, or a cell is read while its neighbour writes it, material duplicates or vanishes.</li>
      <li><b>Margolus neighbourhood</b>: split the grid into 2×2 blocks and give each block to <i>one</i> thread. The thread reads its 4 cells, rearranges them
        (swaps only — mass is conserved) and writes them back. Blocks never overlap, so there are no races and no atomics.</li>
      <li><b>Alternate the partition</b>: on odd steps the block grid is shifted by (1, 1). A grain at the bottom of one block is at the top of a block next step,
        so it can keep falling, and material crosses block borders in every direction.</li>
      <li><b>Rules inside a block</b> (top row a b, bottom row c d): heavier above lighter → swap vertically (sand sinks through water, oil floats);
        blocked → try the diagonal; liquids remember a flow direction and move sideways (lava rarely — it’s viscous); gases rise.</li>
      <li><b>Reactions</b> check pairs within the block: fire/lava ignite wood (→ glowing embers that shed flames) and oil, water + lava → rock + steam,
        water + fire → steam. Lifetimes count down: fire → smoke, smoke → nothing, steam → sometimes rain.</li>
      <li>A separate compute pass sums the light of fire and lava into a small texture; the final shader blurs it for the glow.</li>
    </ol>`,
    uses: [
      { title: 'Whole games', text: 'Noita, The Powder Toy, Sandspiel and Sandustry are built on this — every pixel is simulated.' },
      { title: 'Destructible worlds', text: 'Terrain that crumbles, collapses and flows after explosions (also see Destructible Terrain).' },
      { title: 'Small doses', text: 'Liquid puzzles, hourglass/timer UI, lava levels, falling debris, pixel snow that piles up.' },
    ],
    try: [
      'Pour <i>Water</i> on the burning tree in <b>Sandbox</b>, then pour <i>Oil</i> on it instead.',
      'Draw a <i>Stone</i> cup, fill it with water, then drop <i>Sand</i> in: sand sinks, water is displaced. Add oil: it floats on top.',
      'In <b>Hourglass</b> press <b>Flip hourglass</b>; lower <i>Steps per frame</i> to 1 to slow time down.',
      'Turn on <i>Show Margolus blocks</i> with <i>Cell size</i> 8 and pause: you see the 2×2 tiles one step uses.',
      'In <b>Volcano</b>, paint a stone dam to steer the lava away from the trees.',
    ],
    ask: [
      'a Noita-style falling sand simulation on the GPU',
      'race-free cellular automaton using Margolus 2×2 blocks',
      'materials with densities and reactions (fire, water, lava, oil)',
      'pixel-perfect destructible terrain where every pixel is simulated',
      'glow/bloom for emissive particles like fire and lava',
    ],
    perf: `<p>Each step is one compute dispatch with a thread per 2×2 block: at 3-pixel cells on a 1080p screen that’s ~230k cells → 58k threads, × 6 steps per frame.
      That’s trivial for a GPU; a JavaScript version handles maybe 50–100k cells at 60 fps. Cost grows with cells × steps; it does not depend on how much material is on screen.</p>`,
    api: `<p><b>WebGPU compute.</b> Read-write storage buffers let one thread update four cells in place. A WebGL2 port is possible as fragment passes
      (each pixel recomputes “its” 2×2 block and keeps only its own cell) but every cell redoes the block’s work and random choices must be made deterministically so all four agree.</p>`,
    code: [
      {
        title: 'One Margolus step: a thread owns a 2×2 block',
        lang: 'wgsl',
        src: `let off = i32(st.stepNo & 1u);                       // shift the partition every step
let base = vec2i(gid.xy) * 2 - vec2i(off);
var q = array<u32, 4>(load(base), load(base + vec2i(1, 0)),   // a b
                      load(base + vec2i(0, 1)), load(base + vec2i(1, 1))); // c d
react(&q);                                            // fire, steam, lava + water…
for (var col = 0u; col < 2u; col++) {                 // gravity: heavier above lighter → swap
  let t = matOf(q[col]);
  let b = matOf(q[col + 2u]);
  if (movable(t) && movable(b) && density(t) > density(b) && rnd() < fallChance(t, b)) {
    swap(&q, col, col + 2u);
  }
}
// … diagonal slides, sideways flow, then write the 4 cells back`,
      },
    ],
    links: [
      { title: 'Noita GDC talk — Exploring the Tech and Design of Noita', url: 'https://www.youtube.com/watch?v=prXuyMCgbTc' },
      { title: 'Margolus neighbourhood (Wikipedia: Block cellular automaton)', url: 'https://en.wikipedia.org/wiki/Block_cellular_automaton' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const U = gpu.uniforms(
      {
        grid: 'vec2f',
        brush: 'vec4f',
        res: 'vec2f',
        radius: 'f32',
        material: 'f32',
        erase: 'f32',
        painting: 'f32',
        frameNo: 'u32',
        time: 'f32',
        mode: 'f32',
        glow: 'f32',
        blocks: 'f32',
        lastStep: 'f32',
      },
      'Sand',
    );
    const steps = [];
    for (let i = 0; i < MAX_SUB; i++) steps.push(gpu.uniforms({ stepNo: 'u32', salt: 'u32' }, 'St'));

    const sim = gpu.compute({
      label: 'sand',
      bindings: {
        u: { uniform: U },
        st: { uniform: steps[0] },
        cells: { storage: 'array<u32>', access: 'read_write' },
        scratch: { storage: 'array<u32>', access: 'read_write' },
        glowOut: { storageTexture: 'rgba16float', access: 'write' },
      },
      include: ['hash', 'sdf'],
      code:
        MAT_WGSL +
        /* wgsl */ `
var<private> rs: u32;
fn rnd() -> f32 { rs = pcg(rs); return f32(rs) / 4294967296.0; }
fn gw() -> i32 { return i32(u.grid.x); }
fn gh() -> i32 { return i32(u.grid.y); }
fn inGrid(p: vec2i) -> bool { return p.x >= 0 && p.y >= 0 && p.x < gw() && p.y < gh(); }
fn load(p: vec2i) -> u32 {
  if (!inGrid(p)) { return OUTSIDE; }
  return cells[p.y * gw() + p.x];
}
fn store(p: vec2i, c: u32) { if (inGrid(p)) { cells[p.y * gw() + p.x] = c; } }
fn swapQ(q: ptr<function, array<u32, 4>>, i: u32, j: u32) { let t = (*q)[i]; (*q)[i] = (*q)[j]; (*q)[j] = t; }
fn rshade() -> u32 { return u32(rnd() * 255.0); }
fn newCell(m: u32) -> u32 {
  var life = 0u;
  if (m == FIRE) { life = 30u + u32(rnd() * 40.0); }
  if (m == SMOKE) { life = 60u + u32(rnd() * 80.0); }
  if (m == STEAM) { life = 100u + u32(rnd() * 120.0); }
  return mk(m, rshade(), life, u32(rnd() * 2.0));
}

// --- lifetimes: fire → smoke, smoke → air, steam → (sometimes) rain, embers burn out
fn age(c: u32) -> u32 {
  let m = matOf(c);
  let l = lifeOf(c);
  if (m == FIRE) {
    if (l <= 1u) { if (rnd() < 0.35) { return newCell(SMOKE); } return EMPTY; }
    return withLife(c, l - 1u);
  }
  if (m == SMOKE) {
    if (rnd() < 0.5) { if (l <= 1u) { return EMPTY; } return withLife(c, l - 1u); }
  }
  if (m == STEAM) {
    if (rnd() < 0.5) {
      if (l <= 1u) { if (rnd() < 0.3) { return newCell(WATER); } return EMPTY; }
      return withLife(c, l - 1u);
    }
  }
  if (m == EMBER && rnd() < 0.12) {
    if (l <= 1u) { if (rnd() < 0.5) { return newCell(SMOKE); } return EMPTY; }
    return withLife(c, l - 1u);
  }
  return c;
}

// --- reactions between the cells of a block
fn react(q: ptr<function, array<u32, 4>>) {
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      if (i == j) { continue; }
      let mi = matOf((*q)[i]);
      let mj = matOf((*q)[j]);
      let hot = mi == FIRE || mi == LAVA || mi == EMBER;
      if (hot && mj == WOOD && rnd() < select(0.03, 0.08, mi == LAVA)) {
        (*q)[j] = mk(EMBER, shadeOf((*q)[j]), 150u + u32(rnd() * 100.0), 0u);
      } else if (hot && mj == OIL && rnd() < 0.3) {
        (*q)[j] = newCell(FIRE);
      } else if (mi == WATER && (mj == FIRE || mj == EMBER) && rnd() < 0.4) {
        (*q)[i] = newCell(STEAM);
        if (mj == FIRE) { (*q)[j] = EMPTY; } else { (*q)[j] = mk(WOOD, 20u, 0u, 1u); }  // charred wood
      } else if (mi == WATER && mj == LAVA && rnd() < 0.5) {
        (*q)[i] = newCell(STEAM);
        (*q)[j] = mk(STONE, rshade(), 1u, 0u);                        // life 1 = cooled lava rock
      } else if (mi == EMBER && mj == EMPTY && (j / 2u) <= (i / 2u) && rnd() < 0.18) {
        (*q)[j] = newCell(FIRE);                                       // burning wood sheds flames
      } else if (mi == LAVA && mj == EMPTY && (j / 2u) < (i / 2u) && rnd() < 0.004) {
        (*q)[j] = newCell(FIRE);                                       // sparks above lava
      } else if (mi == SPOUT && mj == EMPTY && rnd() < select(0.12, 0.5, flagsOf((*q)[i]) == LAVA)) {
        (*q)[j] = newCell(flagsOf((*q)[i]));                           // emitters (taps, vents)
      }
    }
  }
}

fn fallChance(t: u32, b: u32) -> f32 {
  if (isGas(b) && !isGas(t)) { return 0.55; }   // gas bubbling up through air/liquid
  if (t == SAND) { return 0.97; }
  if (t == LAVA) { return 0.6; }
  return 1.0;
}
fn flowChance(m: u32) -> f32 {
  if (m == WATER) { return 0.95; }
  if (m == OIL) { return 0.8; }
  if (m == LAVA) { return 0.12; }
  if (m == STEAM || m == SMOKE) { return 0.45; }
  if (m == FIRE) { return 0.3; }
  return 0.0;
}

// sideways flow inside one row of the block (l = row, r = row + 1)
fn flowRow(q: ptr<function, array<u32, 4>>, l: u32, supL: bool, supR: bool) {
  let r = l + 1u;
  let cl = (*q)[l];
  let cr = (*q)[r];
  let ml = matOf(cl);
  let mr = matOf(cr);
  let wantR = (isGas(ml) || (isLiquid(ml) && supL)) && (flagsOf(cl) & 1u) == 1u;
  let wantL = (isGas(mr) || (isLiquid(mr) && supR)) && (flagsOf(cr) & 1u) == 0u;
  if (wantR && movable(mr) && density(mr) < density(ml) && rnd() < flowChance(ml)) {
    (*q)[l] = cr;
    (*q)[r] = cl;
  } else if (wantL && movable(ml) && density(ml) < density(mr) && rnd() < flowChance(mr)) {
    (*q)[l] = cr;
    (*q)[r] = cl;
  } else {
    // blocked: maybe turn around
    if (wantR && rnd() < 0.5) { (*q)[l] = cl ^ (1u << 24u); }
    if (wantL && rnd() < 0.5) { (*q)[r] = cr ^ (1u << 24u); }
  }
}

@compute @workgroup_size(8, 8) fn blockStep(@builtin(global_invocation_id) gid: vec3u) {
  let off = i32(st.stepNo & 1u);
  let base = vec2i(gid.xy) * 2 - vec2i(off);
  if (base.x >= gw() || base.y >= gh()) { return; }
  rs = pcg(gid.x * 1973u + gid.y * 9277u + st.salt * 26699u) ^ (u.frameNo * 2654435761u);
  let p0 = base;
  let p1 = base + vec2i(1, 0);
  let p2 = base + vec2i(0, 1);
  let p3 = base + vec2i(1, 1);
  var q = array<u32, 4>(load(p0), load(p1), load(p2), load(p3));
  for (var i = 0u; i < 4u; i++) { if (q[i] != OUTSIDE) { q[i] = age(q[i]); } }
  react(&q);

  // 1. vertical: heavier above lighter → swap (also makes gases rise)
  for (var col = 0u; col < 2u; col++) {
    let t = matOf(q[col]);
    let b = matOf(q[col + 2u]);
    if (movable(t) && movable(b) && density(t) > density(b) && rnd() < fallChance(t, b)) { swapQ(&q, col, col + 2u); }
  }
  // 2. diagonal: a grain that is blocked below slides into the lower diagonal
  let first = u32(rnd() * 2.0);
  for (var k = 0u; k < 2u; k++) {
    let i = (first + k) % 2u;          // top cell
    let j = 3u - i;                    // diagonal below
    let below = i + 2u;
    let mt = matOf(q[i]);
    let md = matOf(q[j]);
    let mb = matOf(q[below]);
    if (movable(mt) && !isGas(mt) && density(mt) > 1.5 && movable(md) && density(mt) > density(md)
        && (!movable(mb) || density(mb) >= density(mt)) && rnd() < select(0.8, 0.6, mt == SAND) * select(1.0, 0.3, mt == LAVA)) {
      swapQ(&q, i, j);
    }
  }
  // gases slide up diagonally when blocked above
  for (var k = 0u; k < 2u; k++) {
    let i = 2u + (first + k) % 2u;     // bottom cell
    let j = 3u - i;                    // diagonal above
    let above = i - 2u;
    let mg = matOf(q[i]);
    if (isGas(mg) && movable(matOf(q[j])) && density(matOf(q[j])) > density(mg)
        && (!movable(matOf(q[above])) || density(matOf(q[above])) <= density(mg)) && rnd() < 0.5) {
      swapQ(&q, i, j);
    }
  }
  // 3. sideways flow: liquids/gases remember a direction (flag bit) and flip it when blocked.
  //    Liquids only flow when supported (the cell below is not lighter) — otherwise they fall.
  //    (For the bottom row the cell below lies in another block: reading it is a harmless,
  //    read-only peek; only this thread ever writes the four cells of this block.)
  let u0 = matOf(load(p2 + vec2i(0, 1)));
  let u1 = matOf(load(p3 + vec2i(0, 1)));
  let supT0 = !movable(matOf(q[2])) || density(matOf(q[2])) >= density(matOf(q[0]));
  let supT1 = !movable(matOf(q[3])) || density(matOf(q[3])) >= density(matOf(q[1]));
  flowRow(&q, 0u, supT0, supT1);
  let supB0 = !movable(u0) || density(u0) >= density(matOf(q[2]));
  let supB1 = !movable(u1) || density(u1) >= density(matOf(q[3]));
  flowRow(&q, 2u, supB0, supB1);
  if (q[0] != OUTSIDE) { store(p0, q[0]); }
  if (q[1] != OUTSIDE) { store(p1, q[1]); }
  if (q[2] != OUTSIDE) { store(p2, q[2]); }
  if (q[3] != OUTSIDE) { store(p3, q[3]); }
}

// --- painting: one thread per cell inside the brush stroke
@compute @workgroup_size(8, 8) fn paint(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inGrid(p)) { return; }
  let d = sdSegment(vec2f(p) + 0.5, u.brush.zw, u.brush.xy);
  if (d > u.radius) { return; }
  rs = pcg(gid.x * 7919u + gid.y * 104729u + u.frameNo * 1299709u);
  let i = p.y * gw() + p.x;
  let cur = cells[i];
  let cm = matOf(cur);
  if (u.erase > 0.5) { cells[i] = EMPTY; return; }
  let m = u32(u.material);
  if (m == STONE || m == WOOD) { cells[i] = mk(m, rshade(), 0u, 0u); return; }
  if (m == FIRE && cm == WOOD) { cells[i] = mk(EMBER, shadeOf(cur), 200u, 0u); return; }
  if (cm == EMPTY && rnd() < select(0.35, 0.6, isGas(m))) { cells[i] = newCell(m); }
}

// --- flip (hourglass): copy out, then mirror back in
@compute @workgroup_size(8, 8) fn copyOut(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inGrid(p)) { return; }
  scratch[p.y * gw() + p.x] = cells[p.y * gw() + p.x];
}
@compute @workgroup_size(8, 8) fn flipIn(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inGrid(p)) { return; }
  cells[p.y * gw() + p.x] = scratch[(gh() - 1 - p.y) * gw() + p.x];
}

// --- light emitted by fire/lava, summed over 4×4 cells into a small texture
@compute @workgroup_size(8, 8) fn glowSum(@builtin(global_invocation_id) gid: vec3u) {
  let sz = vec2i(textureDimensions(glowOut));
  if (i32(gid.x) >= sz.x || i32(gid.y) >= sz.y) { return; }
  var e = vec3f(0.0);
  for (var y = 0; y < 4; y++) {
    for (var x = 0; x < 4; x++) {
      let c = load(vec2i(gid.xy) * 4 + vec2i(x, y));
      let m = matOf(c);
      if (m == FIRE) { e += vec3f(1.0, 0.45, 0.12) * (0.4 + f32(lifeOf(c)) / 70.0); }
      else if (m == LAVA) { e += vec3f(1.0, 0.32, 0.06) * 0.8; }
      else if (m == EMBER) { e += vec3f(1.0, 0.35, 0.08) * 0.5; }
    }
  }
  textureStore(glowOut, gid.xy, vec4f(e / 16.0, 1.0));
}`,
    });

    // ---------------------------------------------------------------- rendering
    const render = gpu.fullscreen({
      label: 'sand-render',
      uniforms: U,
      textures: ['glow'],
      storage: { cells: 'array<u32>' },
      include: ['hash', 'noise', 'color'],
      code:
        MAT_WGSL +
        /* wgsl */ `
fn background(uv: vec2f, px: vec2f) -> vec3f {
  let mode = i32(u.mode + 0.5);
  if (mode == 1) {
    // hourglass on a desk at night: warm vignette
    let d = length((uv - vec2f(0.5, 0.45)) * vec2f(u.res.x / u.res.y, 1.0));
    return mix(vec3f(0.16, 0.1, 0.07), vec3f(0.025, 0.02, 0.025), smoothstep(0.1, 0.9, d));
  }
  if (mode == 2) {
    var c = mix(vec3f(0.03, 0.02, 0.07), vec3f(0.32, 0.12, 0.1), smoothstep(0.0, 0.85, uv.y));
    let st = floor(px / 2.0);
    c += vec3f(0.8) * step(0.9975, hash21(st)) * smoothstep(0.6, 0.0, uv.y);
    return c;
  }
  let g = fract(px / 24.0);
  let line = step(min(g.x, g.y), 1.0 / 24.0);
  return mix(vec3f(0.06, 0.065, 0.09), vec3f(0.025, 0.028, 0.04), uv.y) + vec3f(0.012) * line;
}

fn cellColor(c: u32, bg: vec3f, cellP: vec2f) -> vec3f {
  let m = matOf(c);
  let s = f32(shadeOf(c)) / 255.0;
  let life = f32(lifeOf(c));
  let t = u.time;
  if (m == EMPTY) { return bg; }
  if (m == SAND) { return mix(vec3f(0.82, 0.66, 0.38), vec3f(0.97, 0.85, 0.56), s); }
  if (m == WATER) {
    let shimmer = 0.08 * sin(cellP.x * 0.35 + t * 3.0 + s * 6.0);
    return mix(bg, vec3f(0.12, 0.38, 0.85) * (0.85 + 0.2 * s) + shimmer, 0.85);
  }
  if (m == STONE) {
    if (life > 0.5) { return vec3f(0.17, 0.15, 0.16) * (0.8 + 0.4 * s); }   // cooled lava rock
    return vec3f(0.42, 0.42, 0.46) * (0.8 + 0.3 * s);
  }
  if (m == WOOD) {
    let grain = 0.9 + 0.1 * sin(cellP.y * 0.7 + cellP.x * 0.15);
    if ((flagsOf(c) & 1u) == 1u) { return vec3f(0.12, 0.08, 0.06) * (0.8 + 0.4 * s); } // charred
    return vec3f(0.46, 0.28, 0.14) * grain * (0.92 + 0.12 * s);
  }
  if (m == EMBER) {
    let fl = 0.6 + 0.4 * sin(t * 9.0 + s * 20.0);
    return mix(vec3f(0.18, 0.08, 0.04), vec3f(1.0, 0.45, 0.1), fl * smoothstep(0.3, 0.9, s));
  }
  if (m == FIRE) {
    let k = clamp(life / 60.0, 0.0, 1.0);
    return mix(vec3f(0.8, 0.12, 0.03), mix(vec3f(1.0, 0.55, 0.1), vec3f(1.0, 0.95, 0.6), smoothstep(0.5, 1.0, k)), smoothstep(0.0, 0.5, k)) * 1.3;
  }
  if (m == SMOKE) { return mix(bg, vec3f(0.3, 0.3, 0.33) * (0.8 + 0.4 * s), clamp(life / 110.0, 0.15, 0.75)); }
  if (m == STEAM) { return mix(bg, vec3f(0.85, 0.88, 0.93), clamp(life / 160.0, 0.12, 0.55)); }
  if (m == OIL) { return vec3f(0.2, 0.13, 0.09) * (0.8 + 0.4 * s) + vec3f(0.08, 0.05, 0.1) * step(0.92, s); }
  if (m == LAVA) {
    let fl = 0.5 + 0.5 * sin(t * 2.0 + cellP.x * 0.3 + cellP.y * 0.2 + s * 6.0);
    return mix(vec3f(0.9, 0.22, 0.03), vec3f(1.0, 0.72, 0.2), fl * 0.7 + s * 0.3) * 1.2;
  }
  if (m == SPOUT) { return vec3f(0.45, 0.47, 0.52) * (0.8 + 0.3 * s); }
  if (m == GLASS) { return mix(bg, vec3f(0.65, 0.85, 0.95), 0.35 + 0.25 * s); }
  return vec3f(1.0, 0.0, 1.0);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let cellF = uv * u.grid;
  let cp = vec2i(floor(cellF));
  let idx = cp.y * i32(u.grid.x) + cp.x;
  let c = cells[idx];
  let bg = background(uv, px);
  var col = cellColor(c, bg, vec2f(cp));
  // fire/lava bloom: blur the 4×4-cell light texture with a few bilinear taps
  let gs = 1.0 / TEXSIZE(glow);
  var g = TEX(glow, uv).rgb * 0.3;
  for (var i = 0; i < 8; i++) {
    let a = f32(i) * 0.785398;
    let o = vec2f(cos(a), sin(a));
    g += TEX(glow, uv + o * gs * 1.5).rgb * 0.06 + TEX(glow, uv + o * gs * 3.5).rgb * 0.05;
  }
  col += g * u.glow * 1.6;
  if (u.blocks > 0.5) {
    let off = u.lastStep;
    let bf = fract((cellF + vec2f(off)) * 0.5);
    let e = min(min(bf.x, 1.0 - bf.x), min(bf.y, 1.0 - bf.y)) * 2.0 * (u.res.x / u.grid.x);
    col = mix(col, vec3f(1.0, 0.85, 0.2), (1.0 - smoothstep(0.0, 1.2, e)) * 0.6);
  }
  // brush outline
  let bd = abs(length(cellF - u.brush.xy) - u.radius) * (u.res.x / u.grid.x);
  col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.0, 1.2, bd)) * 0.35 * u.painting);
  return vec4f(col, 1.0);
}`,
    });

    // ---------------------------------------------------------------- grid & scenes
    let gwid = 0;
    let ghei = 0;
    let cells = null;
    let scratch = null;
    let glowT = null;
    let frameNo = 0;
    let lastStep = 0;
    let flipQueued = false;
    const tick = stepper();
    const tag = readout(ctx);

    const cellPx = () => Math.max(ctx.testMode ? 4 : 1, Math.round((ctx.params.cell || 3) * Math.max(1, Math.min(ctx.dpr, 2))));

    function build(ex, w, h) {
      const g = new Uint32Array(w * h);
      let seed = 12345;
      const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
      const put = (x, y, m, life = 0, flags = 0) => {
        x = Math.round(x);
        y = Math.round(y);
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        g[y * w + x] = m | (Math.floor(rand() * 255) << 8) | (Math.min(255, life) << 16) | (flags << 24);
      };
      const fill = (fn, m, life = 0, chance = 1) => {
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (fn(x, y) && rand() < chance) put(x, y, m, life, rand() < 0.5 ? 1 : 0);
      };
      const disc = (cx, cy, r, m, chance = 1) => fill((x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r, m, 0, chance);
      const rect = (x0, y0, x1, y1, m, chance = 1) => fill((x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1, m, 0, chance);
      const line = (x0, y0, x1, y1, r, m) => {
        const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0));
        for (let i = 0; i <= n; i++) {
          const t = i / Math.max(1, n);
          const cx = x0 + (x1 - x0) * t;
          const cy = y0 + (y1 - y0) * t;
          for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r) put(cx + dx, cy + dy, m);
        }
      };
      const tree = (x, yBase, H, burning = false) => {
        line(x, yBase, x, yBase - H, Math.max(1, Math.round(H * 0.06)), M.wood);
        line(x, yBase - H * 0.55, x - H * 0.3, yBase - H * 0.85, Math.max(1, Math.round(H * 0.03)), M.wood);
        line(x, yBase - H * 0.6, x + H * 0.32, yBase - H * 0.9, Math.max(1, Math.round(H * 0.03)), M.wood);
        disc(x, yBase - H, H * 0.28, M.wood, 0.85);
        disc(x - H * 0.3, yBase - H * 0.88, H * 0.18, M.wood, 0.85);
        disc(x + H * 0.32, yBase - H * 0.92, H * 0.2, M.wood, 0.85);
        if (burning) for (let i = 0; i < H * 0.4; i++) put(x + (rand() - 0.5) * H * 0.12, yBase - rand() * H * 0.25, M.ember, 220);
      };
      const spout = (x, y, emits, wide = 1) => {
        for (let dx = -wide; dx <= wide; dx++) put(x + dx, y, M.spout, 0, emits);
      };

      if (ex === 'hourglass') {
        const cx = w / 2;
        const cy = h * 0.5;
        const H = h * 0.42;
        const rx = H * 0.42;
        const ry = H * 0.5;
        const neck = Math.max(1.5, h / 110);
        const inside = (x, y) => {
          const e1 = ((x - cx) / rx) ** 2 + ((y - (cy - ry)) / ry) ** 2;
          const e2 = ((x - cx) / rx) ** 2 + ((y - (cy + ry)) / ry) ** 2;
          const n = Math.abs(x - cx) <= neck && Math.abs(y - cy) <= H * 0.2;
          return (e1 < 1 && y > cy - 2 * ry + 3) || (e2 < 1 && y < cy + 2 * ry - 3) || n;
        };
        const shell = (x, y) => {
          if (inside(x, y)) return false;
          for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (inside(x + dx, y + dy)) return true;
          return false;
        };
        fill((x, y) => Math.abs(x - cx) < rx + 8 && Math.abs(y - cy) < H + 6 && shell(x, y), M.glass);
        // wooden frame
        const fx0 = Math.round(cx - rx - 11);
        const fx1 = Math.round(cx + rx + 11);
        rect(fx0, cy - H - 9, fx1, cy - H - 3, M.wood);
        rect(fx0, cy + H + 3, fx1, cy + H + 9, M.wood);
        rect(fx0 + 1, cy - H - 3, fx0 + 4, cy + H + 3, M.wood);
        rect(fx1 - 4, cy - H - 3, fx1 - 1, cy + H + 3, M.wood);
        // sand in the top bulb (and a little cone already below)
        fill((x, y) => inside(x, y) && y < cy - H * 0.12 && y > cy - H * 0.85, M.sand);
        fill((x, y) => inside(x, y) && y > cy + H * 0.9 - (rx * 0.25 - Math.abs(x - cx)) * 0.4 && y > cy + H * 0.6, M.sand);
      } else if (ex === 'volcano') {
        const cx = w / 2;
        const sea = h * 0.8;
        const surf = (x) => h * 0.3 + Math.abs(x - cx) * 1.05 + Math.sin(x * 0.11) * 2 + Math.sin(x * 0.037) * 4;
        const vent = Math.max(3, h * 0.03);
        const crater = (x, y) => Math.abs(x - cx) < vent + Math.max(0, (h * 0.42 - y) * 0.9) && y < h * 0.9;
        fill((x, y) => y > surf(x) && !crater(x, y), M.stone);
        rect(0, h * 0.95, w, h, M.stone);
        // the sea on both sides
        fill((x, y) => y > sea && g[y * w + x] === 0 && Math.abs(x - cx) > h * 0.3, M.water);
        // beaches
        fill((x, y) => y > surf(x) - 3 && y <= surf(x) && y > sea - 6 && y < sea + 4, M.sand);
        // lava already rising in the vent + the vent itself
        fill((x, y) => crater(x, y) && y > h * 0.55 && y < h * 0.9, M.lava);
        for (let i = -1; i <= 1; i++) spout(cx + i * 3, Math.round(h * 0.9) - 1, M.lava);
        // trees on the slopes
        for (const fx of [-0.36, -0.27, -0.17, 0.19, 0.28, 0.37]) {
          const x = cx + fx * h;
          if (x < 4 || x > w - 4) continue;
          tree(x, surf(x) + 1, h * 0.09);
        }
      } else {
        // sandbox
        const ground = (x) => h * 0.92 + Math.sin(x * 0.05) * 2;
        fill((x, y) => y > ground(x), M.stone);
        // basin with water (left)
        const bx0 = w * 0.06;
        const bx1 = w * 0.3;
        rect(bx0, h * 0.64, bx0 + 2, h * 0.93, M.stone);
        rect(bx1, h * 0.64, bx1 + 2, h * 0.93, M.stone);
        rect(bx0, h * 0.9, bx1 + 2, h * 0.93, M.stone);
        rect(bx0 + 3, h * 0.72, bx1 - 1, h * 0.895, M.water);
        rect(bx0 + 3, h * 0.67, bx1 - 1, h * 0.715, M.oil);
        // shelf + sand pile
        line(w * 0.36, h * 0.38, w * 0.56, h * 0.46, Math.max(1, Math.round(h * 0.008)), M.stone);
        fill((x, y) => y < ground(x) && y > h * 0.92 - (h * 0.16 - Math.abs(x - w * 0.43) * 0.7), M.sand);
        // a burning tree (right)
        tree(w * 0.68, ground(w * 0.68), h * 0.3, true);
        // oil dish (far right)
        const ox = w * 0.84;
        rect(ox, h * 0.84, ox + 1, h * 0.92, M.stone);
        rect(w * 0.97, h * 0.84, w * 0.97 + 1, h * 0.92, M.stone);
        rect(ox + 2, h * 0.87, w * 0.97 - 1, h * 0.915, M.oil);
        // emitters: a sand chute and a water tap
        spout(w * 0.46, h * 0.05, M.sand, 0);
        spout(w * 0.18, h * 0.05, M.water, 0);
      }
      return g;
    }

    const rebuild = () => {
      const cp = cellPx();
      const w = Math.max(16, Math.floor(ctx.width / cp));
      const h = Math.max(16, Math.floor(ctx.height / cp));
      if (w !== gwid || h !== ghei || !cells) {
        cells?.destroy();
        scratch?.destroy();
        glowT?.destroy();
        gwid = w;
        ghei = h;
        cells = gpu.storage(w * h * 4, 'sand-cells');
        scratch = gpu.storage(w * h * 4, 'sand-scratch');
        glowT = gpu.target(Math.ceil(w / 4), Math.ceil(h / 4), { format: 'rgba16float', label: 'sand-glow' });
      }
      gpu.queue.writeBuffer(cells, 0, build(ctx.example, w, h));
    };
    rebuild();

    let prevBrush = null;
    return {
      resize() {
        rebuild();
      },
      onChange(key) {
        if (key === 'cell') rebuild();
      },
      onAction(key) {
        if (key === 'reset') rebuild();
        if (key === 'flip') flipQueued = true;
      },
      onExample() {
        rebuild();
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        const ptr = ctx.pointer;
        const bx = ptr.nx * gwid;
        const by = ptr.ny * ghei;
        if (!prevBrush || !ptr.down) prevBrush = [bx, by];
        const radius = Math.max(0.8, p.brush / cellPx() * Math.max(1, Math.min(ctx.dpr, 2)));
        const matKey = p.material === 'erase' ? 'empty' : p.material;
        const erase = p.material === 'erase' || (ptr.down && ptr.button === 2) ? 1 : 0;
        const painting = ptr.down && !ctx.paused ? 1 : 0;
        U.setAll({
          grid: [gwid, ghei],
          brush: [bx, by, prevBrush[0], prevBrush[1]],
          res: [ctx.width, ctx.height],
          radius,
          material: M[matKey] ?? M.sand,
          erase,
          painting: ptr.over || ptr.down ? 1 : 0,
          frameNo,
          time: ctx.time,
          mode: { sandbox: 0, hourglass: 1, volcano: 2 }[ctx.example] ?? 0,
          glow: p.glow,
          blocks: p.blocks ? 1 : 0,
          lastStep,
        });
        U.upload();
        const groups = [Math.ceil(gwid / 8), Math.ceil(ghei / 8)];
        const res = (st) => ({ u: U, st, cells, scratch, glowOut: glowT });
        const cs = computeSeq(enc, 'sand');
        if (flipQueued) {
          flipQueued = false;
          cs.run(sim, 'copyOut', groups, res(steps[0])).run(sim, 'flipIn', groups, res(steps[0]));
        }
        if (!ctx.paused) {
          frameNo++;
          if (painting) cs.run(sim, 'paint', groups, res(steps[0]));
          // fixed rate: speed steps per 1/60 s (capped)
          const cap = ctx.testMode ? 6 : MAX_SUB;
          const n = tick.take(Math.min(ctx.dt, 0.1), p.speed * 60, cap);
          const bgroups = [Math.ceil((gwid + 1) / 2 / 8), Math.ceil((ghei + 1) / 2 / 8)];
          for (let i = 0; i < n; i++) {
            const st = steps[i];
            lastStep = (lastStep + 1) & 1;
            st.set('stepNo', lastStep).set('salt', (frameNo * 31 + i * 7) >>> 0);
            st.upload();
            cs.run(sim, 'blockStep', bgroups, res(st));
          }
        }
        cs.run(sim, 'glowSum', [Math.ceil(glowT.width / 8), Math.ceil(glowT.height / 8)], res(steps[0]));
        cs.end();
        prevBrush = [bx, by];
        render.draw(enc, { view: ctx.target, format: gpu.format }, { glow: glowT, cells });
        tag.textContent = `${gwid}×${ghei} = ${(gwid * ghei).toLocaleString()} cells · ${p.speed} Margolus steps/frame`;
      },
    };
  },
};

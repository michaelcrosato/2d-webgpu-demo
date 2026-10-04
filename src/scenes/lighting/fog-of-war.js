// Fog of war: unexplored (black / clouds) · explored but not visible (dim, grey "memory") · visible (full color).
//   vis (compute)  : per fog texel (4 per tile): visible if within a unit's radius AND the straight line to it is not
//                    blocked by a wall tile (march through the tile grid). Or: per-tile CPU shadowcasting result.
//                    explored = max(explored, visible) kept in a persistent storage buffer.
//   blur (2 frag)  : separable Gaussian -> soft fog edges
//   composite      : terrain from the atlas, units layer only where visible, memory = desaturated terrain

import { Camera2D, SpriteBatch } from '../../core/batch.js';
import { getAtlas, rng } from '../../core/assets.js';
import { tag } from './_shared.js';

const SUB = 4; // fog texels per tile
const MAX_UNITS = 64;
const MH = 30; // map height in tiles

// tile types
const T = { GRASS: 0, FOREST: 1, ROCK: 2, WATER: 3, ROAD: 4, WALL: 5, FLOOR: 6, HALL: 7, BRIDGE: 8, TOWER: 9 };
const BLOCKS_SIGHT = new Set([T.FOREST, T.ROCK, T.WALL]);
const WALKABLE = new Set([T.GRASS, T.ROAD, T.FLOOR, T.HALL, T.BRIDGE]);
const TILE_ART = [
  ['tile_sand', '#7fc35a'],
  ['tile_leaves', '#6aa860'],
  ['tile_stone', '#a8a49c'],
  ['tile_water_0', '#ffffff'],
  ['tile_dirt', '#d8b890'],
  ['tile_brick', '#9a8a90'],
  ['tile_stone', '#8a8e9e'],
  ['tile_wood', '#b8a088'],
  ['tile_wood', '#d0b090'],
  ['tile_metal', '#c0c8d8'],
];

const VIS_WGSL = /* wgsl */ `
fn blocksAt(q: vec2i) -> bool {
  if (q.x < 0 || q.y < 0 || q.x >= i32(u.mapSize.x) || q.y >= i32(u.mapSize.y)) { return true; }
  return walls[u32(q.y) * u32(u.mapSize.x) + u32(q.x)] > 0u;
}

@compute @workgroup_size(8, 8)
fn vis(@builtin(global_invocation_id) id: vec3u) {
  let fs = vec2u(u.fogSize);
  if (id.x >= fs.x || id.y >= fs.y) { return; }
  let c = (vec2f(id.xy) + 0.5) / u.sub;            // this fog texel's center, in tiles
  let ti = vec2i(floor(c));
  var v = 0.0;
  if (u.mode < 0.5) {
    // GPU line of sight: walk from every unit to this texel through the wall grid
    for (var k = 0u; k < u.nUnits; k++) {
      let un = units[k];
      let d = distance(un.xy, c);
      if (d > un.z) { continue; }
      let fall = smoothstep(un.z, un.z - u.edge, d);     // soft rim at the edge of the vision radius
      if (fall <= v) { continue; }
      let steps = i32(ceil(d / 0.25));
      var blocked = false;
      for (var s = 1; s < 512; s++) {
        if (s >= steps) { break; }
        let q = vec2i(floor(mix(un.xy, c, f32(s) / f32(steps))));
        if (q.x == ti.x && q.y == ti.y) { break; }       // reached our own tile: a wall's face is visible
        if (blocksAt(q)) { blocked = true; break; }
      }
      if (!blocked) { v = fall; }
    }
  } else {
    // CPU shadowcasting result, one value per tile
    v = cpuVis[u32(ti.y) * u32(u.mapSize.x) + u32(ti.x)];
  }
  let idx = id.y * fs.x + id.x;
  var e = max(explored[idx], v);                    // memory: once seen, stays explored
  if (u.resetExplored > 0.5) { e = v; }
  explored[idx] = e;
  textureStore(visOut, vec2i(id.xy), vec4f(v, e, 0.0, 1.0));
}`;

const BLUR_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let r = u.radius;
  if (r < 0.25) { return TEX(src, uv); }
  let ts = u.dir / TEXSIZE(src);
  var s = vec4f(0.0);
  var wsum = 0.0;
  for (var i = -16; i <= 16; i++) {
    let fi = f32(i);
    if (abs(fi) > r * 2.0) { continue; }
    let w = exp(-fi * fi / (2.0 * r * r));
    s += TEX(src, uv + ts * fi) * w;
    wsum += w;
  }
  return s / wsum;
}`;

const COMPOSITE_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let tc = (px - u.origin) / u.tilePx;
  let ms = u.mapSize;
  if (tc.x < 0.0 || tc.y < 0.0 || tc.x >= ms.x || tc.y >= ms.y) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let ti = vec2i(floor(tc));
  let ty = i32(tiles[u32(ti.y) * u32(ms.x) + u32(ti.x)]);
  let r = u.rects[ty];
  let lc = fract(tc);
  var terrain = TEXN(atlas, mix(r.xy, r.zw, lc)).rgb * u.tints[ty].rgb;
  terrain *= 0.9 + 0.1 * hash21(vec2f(ti));
  if (u.tints[ty].a > 0.5) {
    // raised tiles (walls, rocks, forest): darken the bottom edge so they read as blocks
    terrain *= 1.0 - 0.35 * smoothstep(0.7, 1.0, lc.y);
  }

  let fuv = tc / ms;
  var fog = TEX(fogT, fuv);
  if (u.hard > 0.5) { fog = TEXN(fogT, fuv); }
  let v = fog.r;
  let e = fog.g;
  let view = i32(u.view + 0.5);
  if (view == 1) { let raw = TEXN(rawT, fuv); return vec4f(raw.r, raw.g * 0.5, 0.15, 1.0); }
  if (view == 2) { return vec4f(v, e * 0.5, 0.15, 1.0); }

  let unitsC = TEX(unitsT, uv);                                  // premultiplied unit sprites
  let live = terrain * (1.0 - unitsC.a) + unitsC.rgb;
  let g = luma(terrain);
  let memory = mix(vec3f(g), terrain, 0.2) * u.memory * vec3f(0.8, 0.88, 1.15);
  var unexplored = vec3f(0.01, 0.012, 0.02) + 0.015 * valueNoise(px / 6.0);
  if (u.style > 0.5) {
    // drifting clouds instead of black
    let k = u.resolution.y;
    let n = fbm(px / (k * 0.18) + vec2f(u.time * 0.03, u.time * 0.01), 5) * 0.5 + 0.5;
    let n2 = fbm(px / (k * 0.07) - vec2f(u.time * 0.05, 0.0), 3) * 0.5 + 0.5;
    unexplored = mix(vec3f(0.12, 0.13, 0.17), vec3f(0.55, 0.57, 0.62), smoothstep(0.3, 0.8, n * 0.75 + n2 * 0.25));
  }
  var col = mix(unexplored, memory, smoothstep(0.0, 1.0, e));
  if (u.style > 0.5) { col = mix(col, unexplored, 0.25 * (1.0 - v)); }  // thin mist over the remembered area
  col = mix(col, live, v);
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}`;

// ----------------------------------------------------------------------------------- CPU helpers

function valueNoise2(seed) {
  const h = (x, y) => {
    let t = Math.imul(x * 374761393 + y * 668265263 + seed * 1442695041, 1274126177);
    t ^= t >>> 13;
    t = Math.imul(t, 1103515245);
    return ((t ^ (t >>> 16)) >>> 0) / 4294967296;
  };
  return (x, y) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const fx = x - xi;
    const fy = y - yi;
    const sx = fx * fx * (3 - 2 * fx);
    const sy = fy * fy * (3 - 2 * fy);
    const a = h(xi, yi) + (h(xi + 1, yi) - h(xi, yi)) * sx;
    const b = h(xi, yi + 1) + (h(xi + 1, yi + 1) - h(xi, yi + 1)) * sx;
    return a + (b - a) * sy;
  };
}

function genOverworld(MW, seed) {
  const map = new Uint8Array(MW * MH);
  const n1 = valueNoise2(seed);
  const n2 = valueNoise2(seed + 7);
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      const f = n1(x * 0.16, y * 0.16) * 0.7 + n1(x * 0.4, y * 0.4) * 0.3;
      const r = n2(x * 0.22, y * 0.22);
      let t = T.GRASS;
      if (f > 0.6) t = T.FOREST;
      if (r > 0.74) t = T.ROCK;
      if (x === 0 || y === 0 || x === MW - 1 || y === MH - 1) t = T.FOREST;
      map[y * MW + x] = t;
    }
  // a river with bridges
  for (let y = 0; y < MH; y++) {
    const rx = MW * 0.58 + Math.sin(y * 0.28) * 2.5;
    for (let x = Math.floor(rx - 1); x <= Math.floor(rx + 1); x++) if (x > 0 && x < MW - 1 && y > 0 && y < MH - 1) map[y * MW + x] = y % 9 === 4 ? T.BRIDGE : T.WATER;
  }
  // a winding road
  for (let x = 1; x < MW - 1; x++) {
    const ry = Math.round(MH * 0.5 + Math.sin(x * 0.17) * 4);
    for (const y of [ry, ry + 1]) if (map[y * MW + x] !== T.WATER && map[y * MW + x] !== T.BRIDGE) map[y * MW + x] = T.ROAD;
    if (map[ry * MW + x] === T.WATER) map[ry * MW + x] = T.BRIDGE;
    if (map[(ry + 1) * MW + x] === T.WATER) map[(ry + 1) * MW + x] = T.BRIDGE;
  }
  // clear the start area, add a watchtower
  for (let y = 10; y < 20; y++) for (let x = 2; x < 9; x++) if (map[y * MW + x] !== T.ROAD) map[y * MW + x] = T.GRASS;
  const tower = [Math.round(MW * 0.82), 7];
  for (let y = tower[1] - 1; y <= tower[1] + 1; y++) for (let x = tower[0] - 1; x <= tower[0] + 1; x++) map[y * MW + x] = T.GRASS;
  map[tower[1] * MW + tower[0]] = T.TOWER;
  return { map, start: [5, 15], tower };
}

function genDungeon(MW, seed) {
  const R = rng(seed);
  const map = new Uint8Array(MW * MH).fill(T.WALL);
  const rooms = [];
  for (let tries = 0; tries < 200 && rooms.length < 12; tries++) {
    const w = 4 + Math.floor(R() * 6);
    const h = 3 + Math.floor(R() * 5);
    const x = 1 + Math.floor(R() * (MW - w - 2));
    const y = 1 + Math.floor(R() * (MH - h - 2));
    if (rooms.some((r) => x < r.x + r.w + 1 && x + w + 1 > r.x && y < r.y + r.h + 1 && y + h + 1 > r.y)) continue;
    rooms.push({ x, y, w, h, cx: x + Math.floor(w / 2), cy: y + Math.floor(h / 2) });
  }
  rooms.sort((a, b) => a.cx - b.cx);
  for (const r of rooms) for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) map[y * MW + x] = T.FLOOR;
  const carve = (x, y) => {
    if (map[y * MW + x] === T.WALL) map[y * MW + x] = T.HALL;
  };
  for (let i = 1; i < rooms.length; i++) {
    const a = rooms[i - 1];
    const b = rooms[i];
    const horizFirst = R() < 0.5;
    if (horizFirst) {
      for (let x = Math.min(a.cx, b.cx); x <= Math.max(a.cx, b.cx); x++) carve(x, a.cy);
      for (let y = Math.min(a.cy, b.cy); y <= Math.max(a.cy, b.cy); y++) carve(b.cx, y);
    } else {
      for (let y = Math.min(a.cy, b.cy); y <= Math.max(a.cy, b.cy); y++) carve(a.cx, y);
      for (let x = Math.min(a.cx, b.cx); x <= Math.max(a.cx, b.cx); x++) carve(x, b.cy);
    }
  }
  return { map, start: [rooms[0].cx, rooms[0].cy], rooms };
}

/** Recursive shadowcasting (Björn Bergström / RogueBasin), 8 octants. */
function shadowcast(cx, cy, radius, blocks, mark) {
  mark(cx, cy, 0);
  const M = [
    [1, 0, 0, -1, -1, 0, 0, 1],
    [0, 1, -1, 0, 0, -1, 1, 0],
    [0, 1, 1, 0, 0, -1, -1, 0],
    [1, 0, 0, 1, -1, 0, 0, -1],
  ];
  const cast = (row, start, end, xx, xy, yx, yy) => {
    if (start < end) return;
    const r2 = radius * radius;
    let newStart = 0;
    for (let j = row; j <= radius; j++) {
      let dx = -j - 1;
      const dy = -j;
      let blocked = false;
      while (dx <= 0) {
        dx += 1;
        const X = cx + dx * xx + dy * xy;
        const Y = cy + dx * yx + dy * yy;
        const lSlope = (dx - 0.5) / (dy + 0.5);
        const rSlope = (dx + 0.5) / (dy - 0.5);
        if (start < rSlope) continue;
        if (end > lSlope) break;
        if (dx * dx + dy * dy < r2) mark(X, Y, Math.sqrt(dx * dx + dy * dy));
        if (blocked) {
          if (blocks(X, Y)) {
            newStart = rSlope;
            continue;
          }
          blocked = false;
          start = newStart;
        } else if (blocks(X, Y) && j < radius) {
          blocked = true;
          cast(j + 1, start, lSlope, xx, xy, yx, yy);
          newStart = rSlope;
        }
      }
      if (blocked) break;
    }
  };
  for (let o = 0; o < 8; o++) cast(1, 1, 0, M[0][o], M[1][o], M[2][o], M[3][o]);
}

function bfs(map, MW, from, to) {
  const [fx, fy] = from.map(Math.round);
  const [tx, ty] = to;
  if (!WALKABLE.has(map[ty * MW + tx])) return null;
  const prev = new Int32Array(MW * MH).fill(-1);
  const q = [fy * MW + fx];
  prev[fy * MW + fx] = fy * MW + fx;
  while (q.length) {
    const c = q.shift();
    if (c === ty * MW + tx) break;
    const x = c % MW;
    const y = (c / MW) | 0;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= MW || ny >= MH) continue;
      const n = ny * MW + nx;
      if (prev[n] >= 0 || !WALKABLE.has(map[n])) continue;
      prev[n] = c;
      q.push(n);
    }
  }
  if (prev[ty * MW + tx] < 0) return null;
  const path = [];
  for (let c = ty * MW + tx; c !== fy * MW + fx; c = prev[c]) path.push([c % MW, (c / MW) | 0]);
  return path.reverse();
}

export default {
  keys: true,
  interaction: 'Click (or drag) to send your units there · Roguelike: WASD / arrow keys or click.',
  examples: [
    {
      id: 'rts',
      label: 'RTS fog',
      kind: 'In a game',
      note: 'Classic real-time-strategy fog: <b>black</b> where you have never been, <b>grey memory</b> of the terrain where you have been, <b>full color</b> where your units see right now. Enemies (slimes, bats) only show up in the live area. Forests and rocks block line of sight; the watchtower sees far.',
      params: { algorithm: 'gpu', radius: 7, blur: 2, memory: 0.55, style: 'black', view: 'final' },
    },
    {
      id: 'rogue',
      label: 'Roguelike shadowcasting',
      kind: 'Classic',
      note: 'Tile-exact field of view computed on the CPU with <b>recursive shadowcasting</b> — the algorithm most roguelikes use. No blur: a tile is either seen or not. Walk with WASD / arrows or click a tile.',
      params: { algorithm: 'cpu', radius: 8, blur: 0, memory: 0.5, style: 'black', view: 'final' },
    },
    {
      id: 'soft',
      label: 'Soft fog',
      kind: 'Stylized',
      note: 'The same GPU visibility, blurred heavily, with drifting <b>clouds</b> instead of black and a thin mist over remembered ground — the look of strategy games like <i>Northgard</i> or <i>Bad North</i>.',
      params: { algorithm: 'gpu', radius: 8, blur: 6, memory: 0.5, style: 'clouds', view: 'final' },
    },
  ],
  controls: [
    { type: 'slider', key: 'radius', label: 'Vision radius', min: 3, max: 16, step: 0.5, value: 7, format: (v) => `${v} tiles`, help: 'How far each unit sees.' },
    { type: 'slider', key: 'blur', label: 'Edge softness (blur)', min: 0, max: 8, step: 0.1, value: 2, help: 'Gaussian blur of the fog texture, in fog texels (4 per tile). 0 = hard tile edges.' },
    { type: 'slider', key: 'memory', label: 'Explored brightness', min: 0, max: 1, step: 0.01, value: 0.55, help: 'How visible remembered (explored but not currently seen) areas are.' },
    {
      type: 'select',
      key: 'algorithm',
      label: 'Line-of-sight algorithm',
      value: 'gpu',
      options: [
        { value: 'gpu', label: 'GPU: ray march per fog texel (compute)' },
        { value: 'cpu', label: 'CPU: recursive shadowcasting per tile' },
      ],
      help: 'Both decide which tiles a unit can see past the walls.',
    },
    {
      type: 'select',
      key: 'style',
      label: 'Unexplored look',
      value: 'black',
      options: [
        { value: 'black', label: 'Black (classic)' },
        { value: 'clouds', label: 'Drifting clouds' },
      ],
    },
    { type: 'slider', key: 'speed', label: 'Unit speed', min: 1, max: 12, step: 0.1, value: 4, format: (v) => `${v} tiles/s` },
    {
      type: 'select',
      key: 'view',
      label: 'View',
      value: 'final',
      options: [
        { value: 'final', label: 'Final image' },
        { value: 'raw', label: 'Raw fog texture (R visible, G explored)' },
        { value: 'blurred', label: 'Blurred fog texture' },
      ],
    },
    { type: 'button', key: 'reset', label: 'Forget explored map', primary: true },
  ],
  about: {
    summary:
      'Fog of war hides what your units cannot currently see. It needs three states per spot — never seen, seen before, seen now — plus a line-of-sight test against walls.',
    what: `<p>A tile map, units with a vision radius, and a low-resolution <b>fog texture</b> (4 texels per tile) holding two values:
      <i>visible now</i> and <i>explored</i>. The final shader shows full color where visible, a desaturated memory where explored,
      and darkness or clouds elsewhere. Live things (enemies) are only drawn in the visible part.</p>`,
    how: `<ol>
      <li><b>Visibility (GPU)</b>: a compute shader runs one thread per fog texel. For every unit within range it walks the straight line
        from the unit to the texel through the tile grid; if a wall tile is crossed, that unit cannot see it. A smoothstep near the
        radius gives a soft rim.</li>
      <li><b>Visibility (CPU)</b>: <b>recursive shadowcasting</b> scans the 8 octants around the hero row by row, tracking the angular
        range that is still unobstructed; when a wall interrupts it, the range splits. It is exact per tile and symmetric enough for games.</li>
      <li><b>Memory</b>: an <code>explored</code> storage buffer is updated with <code>max(explored, visible)</code> every frame, so it only
        ever grows (until you reset).</li>
      <li><b>Soft edges</b>: a separable Gaussian blur (horizontal then vertical pass) on the tiny fog texture; it is then sampled with
        bilinear filtering at screen resolution — smooth curves from a 216×120 texture.</li>
      <li><b>Composite</b>: <code>mix(unexplored, memory, explored)</code> then <code>mix(…, live, visible)</code>. The units layer is multiplied by visibility so hidden enemies never leak.</li>
    </ol>`,
    uses: [
      { title: 'Strategy', text: 'StarCraft, Age of Empires, Warcraft III, Northgard: scouting and map control are core gameplay.' },
      { title: 'Roguelikes', text: 'NetHack, Brogue, DCSS use shadowcasting FOV with remembered tiles.' },
      { title: 'Tactics & stealth', text: 'XCOM-style cover and hidden enemies; Darkwood-like vision cones.' },
      { title: 'Exploration', text: 'Revealing a map as you go (maps in Hollow Knight or Zelda dungeons).' },
    ],
    try: [
      'On <b>RTS fog</b>, send the squad into the forest edge: trees cut the vision into wedges.',
      'Set <i>Edge softness</i> to 0 and then 8 — same data, very different mood.',
      'Switch <i>Line-of-sight algorithm</i> to CPU on the RTS map: visibility snaps to whole tiles.',
      'View → <i>Raw fog texture</i> to see the tiny texture that drives everything (red = visible, green = explored).',
      'Press <i>Forget explored map</i> and watch memory build up again as the units move.',
    ],
    ask: [
      'RTS fog of war with explored and visible areas',
      'roguelike field of view with recursive shadowcasting',
      'soft blurred fog edges with a low-res fog texture',
      'hide enemies outside the player’s line of sight',
      'a GPU compute shader for line of sight on a tile map',
      'cloudy fog of war like Northgard',
    ],
    perf: `<p>The fog texture is tiny (≈26k texels) and independent of screen resolution, so the compute pass costs well under 0.1 ms even with
      dozens of units. CPU shadowcasting touches only tiles within the radius — microseconds per unit. The full-screen composite is one
      texture lookup per pixel. Fog of war is cheap; the expensive part in real games is <i>gameplay</i> visibility for many units, which
      is why it is often computed on a coarse grid at a few Hz.</p>`,
    api: `<p>The per-texel line-of-sight pass uses a <b>compute shader</b> with a <b>read-write storage buffer</b> for the persistent explored
      state — natural in WebGPU. In WebGL2 you would do the same as a fragment pass with ping-pong textures for memory, or compute
      visibility on the CPU (shadowcasting) and upload a small texture each frame.</p>`,
    code: [
      {
        title: 'Line of sight per fog texel (compute)',
        lang: 'wgsl',
        src: `for (var k = 0u; k < u.nUnits; k++) {
  let un = units[k];                                  // xy = position (tiles), z = radius
  let d = distance(un.xy, c);
  if (d > un.z) { continue; }
  let fall = smoothstep(un.z, un.z - u.edge, d);
  let steps = i32(ceil(d / 0.25));
  var blocked = false;
  for (var s = 1; s < 512; s++) {
    if (s >= steps) { break; }
    let q = vec2i(floor(mix(un.xy, c, f32(s) / f32(steps))));
    if (q.x == ti.x && q.y == ti.y) { break; }        // our own tile: wall faces are visible
    if (blocksAt(q)) { blocked = true; break; }
  }
  if (!blocked) { v = max(v, fall); }
}
let e = max(explored[idx], v);                        // memory only grows
explored[idx] = e;
textureStore(visOut, vec2i(id.xy), vec4f(v, e, 0.0, 1.0));`,
      },
      {
        title: 'Composite: three states',
        lang: 'wgsl',
        src: `let memory = mix(vec3f(luma(terrain)), terrain, 0.2) * u.memory;
var col = mix(unexplored, memory, smoothstep(0.0, 1.0, e));
col = mix(col, terrain * (1.0 - units.a) + units.rgb, v);   // live view incl. units`,
      },
    ],
    links: [
      { title: 'RogueBasin — FOV using recursive shadowcasting', url: 'https://www.roguebasin.com/index.php/FOV_using_recursive_shadowcasting' },
      { title: 'Red Blob Games — 2D visibility', url: 'https://www.redblobgames.com/articles/visibility/' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasTex = gpu.textureFromImage(atlas.canvas, { label: 'atlas' });
    const atlasView = atlasTex.createView();
    const cam = new Camera2D();
    const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest' });

    const U = gpu.uniforms(
      {
        resolution: 'vec2f',
        mapSize: 'vec2f',
        fogSize: 'vec2f',
        origin: 'vec2f',
        tilePx: 'f32',
        sub: 'f32',
        nUnits: 'u32',
        mode: 'f32',
        edge: 'f32',
        resetExplored: 'f32',
        memory: 'f32',
        style: 'f32',
        view: 'f32',
        hard: 'f32',
        time: 'f32',
        rects: 'array<vec4f, 12>',
        tints: 'array<vec4f, 12>',
      },
      'FogU',
    );
    const blurH = gpu.uniforms({ dir: 'vec2f', radius: 'f32' }, 'BlurU');
    const blurV = gpu.uniforms({ dir: 'vec2f', radius: 'f32' }, 'BlurU');
    const unitData = new Float32Array(MAX_UNITS * 4);
    const unitBuf = gpu.storage(unitData.byteLength, 'units');

    const visProg = gpu.compute({
      label: 'fog-vis',
      bindings: {
        u: { uniform: U },
        units: { storage: 'array<vec4f>', access: 'read' },
        walls: { storage: 'array<u32>', access: 'read' },
        cpuVis: { storage: 'array<f32>', access: 'read' },
        explored: { storage: 'array<f32>', access: 'read_write' },
        visOut: { storageTexture: 'rgba16float', access: 'write' },
      },
      code: VIS_WGSL,
    });
    // two passes need two uniform buffers (all writeBuffer calls land before the frame's commands)
    const blurFxH = gpu.fullscreen({ label: 'fog-blur-h', uniforms: blurH, textures: ['src'], code: BLUR_WGSL });
    const blurFxV = gpu.fullscreen({ label: 'fog-blur-v', uniforms: blurV, textures: ['src'], code: BLUR_WGSL });
    const compFx = gpu.fullscreen({
      label: 'fog-composite',
      uniforms: U,
      textures: ['atlas', 'fogT', 'rawT', 'unitsT'],
      storage: { tiles: 'array<u32>' },
      include: ['color', 'noise'],
      code: COMPOSITE_WGSL,
    });

    // rects & tints for every tile type (alpha of tint = "raised" tile)
    const rects = new Float32Array(48);
    const tints = new Float32Array(48);
    TILE_ART.forEach(([name, hex], i) => {
      rects.set(atlas.uv(name), i * 4);
      const n = parseInt(hex.slice(1), 16);
      tints.set([((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, BLOCKS_SIGHT.has(i) || i === T.TOWER ? 1 : 0], i * 4);
    });

    const readout = tag(ctx, 'right:8px;bottom:8px');

    // ------------------------------------------------------------------ world state
    let world = null; // {kind, MW, map, ...}
    let bufs = null;
    let units = [];
    let enemies = [];
    let resetFlag = 2;
    let seedRoute = null;
    let lastOrder = -100;
    let keyT = 0;
    let unitsT = null;

    function build(kind) {
      const MW = Math.max(24, Math.round((MH * ctx.width) / ctx.height));
      const gen = kind === 'rogue' ? genDungeon(MW, 11) : genOverworld(MW, 5);
      world = { kind, MW, ...gen, key: `${kind}|${MW}` };
      const fogW = MW * SUB;
      const fogH = MH * SUB;
      if (bufs) for (const b of Object.values(bufs)) b.destroy?.();
      const walls = new Uint32Array(MW * MH);
      const tiles = new Uint32Array(MW * MH);
      for (let i = 0; i < MW * MH; i++) {
        walls[i] = BLOCKS_SIGHT.has(gen.map[i]) ? 1 : 0;
        tiles[i] = gen.map[i];
      }
      bufs = {
        walls: gpu.storage(walls, 'walls'),
        tiles: gpu.storage(tiles, 'tiles'),
        cpuVis: gpu.storage(MW * MH * 4, 'cpu-vis'),
        explored: gpu.storage(fogW * fogH * 4, 'explored'),
        visT: gpu.target(fogW, fogH, { format: 'rgba16float', label: 'fog-vis' }),
        blurA: gpu.target(fogW, fogH, { format: 'rgba16float', label: 'fog-blur-a' }),
        blurB: gpu.target(fogW, fogH, { format: 'rgba16float', label: 'fog-blur-b' }),
      };
      world.fogW = fogW;
      world.fogH = fogH;
      world.cpuVis = new Float32Array(MW * MH);
      // units
      const [sx, sy] = gen.start;
      units = [];
      const R = rng(9);
      if (kind === 'rogue') units.push({ x: sx, y: sy, path: [], hero: true, flip: false, moving: 0 });
      else {
        const offs = kind === 'soft' ? [[0, 0], [1, 1]] : [[0, 0], [1, 0], [0, 1], [1, 1]];
        for (const [ox, oy] of offs) units.push({ x: sx + ox, y: sy + oy, path: [], flip: false, moving: 0, off: [ox, oy] });
      }
      enemies = [];
      for (let i = 0; i < 18; i++) {
        for (let tries = 0; tries < 50; tries++) {
          const x = 1 + Math.floor(R() * (MW - 2));
          const y = 1 + Math.floor(R() * (MH - 2));
          if (WALKABLE.has(gen.map[y * MW + x]) && Math.hypot(x - sx, y - sy) > 6) {
            enemies.push({ x, y, path: [], kind: i % 3 === 2 ? 'bat' : 'slime', flip: false, wait: R() * 3 });
            break;
          }
        }
      }
      // pretend the player already scouted a little: a route of past positions seeds the explored memory
      seedRoute = [];
      if (kind === 'rogue') {
        for (const r of gen.rooms.slice(0, 3)) seedRoute.push([r.cx, r.cy]);
      } else {
        for (let k = 0; k < 6; k++) seedRoute.push([sx + k * 3, MH * 0.5 + Math.sin((sx + k * 3) * 0.17) * 4]);
      }
      resetFlag = 2;
    }

    const walkable = (x, y) => x >= 0 && y >= 0 && x < world.MW && y < MH && WALKABLE.has(world.map[y * world.MW + x]);
    const order = (u, tx, ty) => {
      let best = null;
      for (let r = 0; r < 4 && !best; r++)
        for (let dy = -r; dy <= r && !best; dy++)
          for (let dx = -r; dx <= r && !best; dx++) {
            const x = tx + dx;
            const y = ty + dy;
            if (walkable(x, y)) best = [x, y];
          }
      if (!best) return;
      const path = bfs(world.map, world.MW, [u.x, u.y], best);
      if (path) u.path = path;
    };
    const moveAlong = (u, speed, dt) => {
      if (!u.path.length) {
        u.moving = 0;
        return;
      }
      const [nx, ny] = u.path[0];
      const dx = nx - u.x;
      const dy = ny - u.y;
      const d = Math.hypot(dx, dy);
      const step = speed * dt;
      if (d <= step) {
        u.x = nx;
        u.y = ny;
        u.path.shift();
      } else {
        u.x += (dx / d) * step;
        u.y += (dy / d) * step;
      }
      if (Math.abs(dx) > 0.01) u.flip = dx < 0;
      u.moving = 1;
    };

    return {
      resize() {
        world = null;
      },
      onExample() {
        world = null;
      },
      onAction(key) {
        if (key === 'reset') resetFlag = 1;
      },
      frame(ctx) {
        const W = ctx.width;
        const H = ctx.height;
        const p = ctx.params;
        const dt = ctx.dt;
        const t = ctx.time;
        const kind = ctx.example;
        if (!world || world.kind !== kind) build(kind);
        if (!unitsT || unitsT.width !== W || unitsT.height !== H) {
          unitsT?.destroy();
          unitsT = gpu.target(W, H, { label: 'units' });
        }
        const MW = world.MW;
        const tilePx = Math.min(W / MW, H / MH);
        const origin = [(W - MW * tilePx) / 2, (H - MH * tilePx) / 2];
        const toTile = (px, py) => [Math.floor((px - origin[0]) / tilePx), Math.floor((py - origin[1]) / tilePx)];

        // ---- input & movement
        const ptr = ctx.pointer;
        if (ptr.over && (ptr.clicked || (ptr.down && ctx.frame % 10 === 0))) {
          const [tx, ty] = toTile(ptr.x, ptr.y);
          units.forEach((u) => order(u, tx + (u.off ? u.off[0] : 0), ty + (u.off ? u.off[1] : 0)));
          lastOrder = t;
        }
        if (kind === 'rogue' && dt > 0) {
          keyT -= dt;
          const k = ctx.keys;
          let mx = 0;
          let my = 0;
          if (k.has('ArrowLeft') || k.has('KeyA')) mx = -1;
          else if (k.has('ArrowRight') || k.has('KeyD')) mx = 1;
          else if (k.has('ArrowUp') || k.has('KeyW')) my = -1;
          else if (k.has('ArrowDown') || k.has('KeyS')) my = 1;
          const h = units[0];
          if ((mx || my) && keyT <= 0 && !h.path.length) {
            const nx = Math.round(h.x) + mx;
            const ny = Math.round(h.y) + my;
            if (walkable(nx, ny)) h.path = [[nx, ny]];
            keyT = 0.13;
            lastOrder = t;
          }
        }
        if (dt > 0) {
          // idle units go exploring on their own after a while (keeps the demo alive)
          const idle = units.every((u) => !u.path.length);
          if (idle && t - lastOrder > 2.5) {
            const R = Math.random;
            for (let tries = 0; tries < 30; tries++) {
              const tx = 1 + Math.floor(R() * (MW - 2));
              const ty = 1 + Math.floor(R() * (MH - 2));
              if (walkable(tx, ty) && Math.hypot(tx - units[0].x, ty - units[0].y) > 8) {
                units.forEach((u) => order(u, tx + (u.off ? u.off[0] : 0), ty + (u.off ? u.off[1] : 0)));
                break;
              }
            }
            lastOrder = t;
          }
          for (const u of units) moveAlong(u, p.speed, dt);
          for (const e of enemies) {
            e.wait -= dt;
            if (!e.path.length && e.wait <= 0) {
              const tx = Math.round(e.x) + Math.floor(Math.random() * 9) - 4;
              const ty = Math.round(e.y) + Math.floor(Math.random() * 9) - 4;
              if (walkable(tx, ty)) e.path = bfs(world.map, MW, [e.x, e.y], [tx, ty]) || [];
              e.wait = 1 + Math.random() * 3;
            }
            moveAlong(e, 1.5, dt);
          }
        }

        // ---- vision sources
        const sources = [];
        const useCpu = p.algorithm === 'cpu';
        const seeding = resetFlag > 0 && seedRoute && seedRoute.length;
        if (seeding) for (const [x, y] of seedRoute) sources.push([x + 0.5, y + 0.5, p.radius * 0.9]);
        else {
          for (const u of units) sources.push([u.x + 0.5, u.y + 0.5, p.radius]);
          if (world.tower) sources.push([world.tower[0] + 0.5, world.tower[1] + 0.5, p.radius * 1.6]);
        }
        sources.slice(0, MAX_UNITS).forEach((s, i) => unitData.set([s[0], s[1], s[2], 0], i * 4));
        gpu.queue.writeBuffer(unitBuf, 0, unitData, 0, Math.max(4, sources.length * 4));
        if (useCpu) {
          const cv = world.cpuVis;
          cv.fill(0);
          const blocks = (x, y) => x < 0 || y < 0 || x >= MW || y >= MH || BLOCKS_SIGHT.has(world.map[y * MW + x]);
          for (const [sx, sy, r] of sources) {
            shadowcast(Math.floor(sx), Math.floor(sy), Math.round(r), blocks, (x, y, d) => {
              if (x < 0 || y < 0 || x >= MW || y >= MH) return;
              const v = 1 - 0.45 * Math.pow(d / r, 2);
              cv[y * MW + x] = Math.max(cv[y * MW + x], v);
            });
          }
          gpu.queue.writeBuffer(bufs.cpuVis, 0, cv);
        }

        // ---- uniforms
        const time = t;
        const water = atlas.uv(`tile_water_${Math.floor(time * 3) % 4}`);
        rects.set(water, T.WATER * 4);
        U.setAll({
          resolution: [W, H],
          mapSize: [MW, MH],
          fogSize: [world.fogW, world.fogH],
          origin,
          tilePx,
          sub: SUB,
          nUnits: Math.min(sources.length, MAX_UNITS),
          mode: useCpu ? 1 : 0,
          edge: Math.min(2, p.radius * 0.3),
          resetExplored: resetFlag === 1 ? 1 : 0,
          memory: p.memory,
          style: p.style === 'clouds' ? 1 : 0,
          view: { final: 0, raw: 1, blurred: 2 }[p.view] ?? 0,
          hard: p.blur < 0.25 ? 1 : 0,
          time,
          rects,
          tints,
        });
        U.upload();
        if (resetFlag > 0) resetFlag--;
        blurH.setAll({ dir: [1, 0], radius: p.blur }).upload();
        blurV.setAll({ dir: [0, 1], radius: p.blur }).upload();

        const enc = ctx.encoder;
        visProg.dispatch(enc, 'vis', [Math.ceil(world.fogW / 8), Math.ceil(world.fogH / 8)], {
          u: U,
          units: unitBuf,
          walls: bufs.walls,
          cpuVis: bufs.cpuVis,
          explored: bufs.explored,
          visOut: bufs.visT,
        });
        blurFxH.draw(enc, bufs.blurA, { src: bufs.visT });
        blurFxV.draw(enc, bufs.blurB, { src: bufs.blurA });

        // ---- units layer (only shown where visible)
        cam.setViewport(W, H);
        cam.x = W / 2;
        cam.y = H / 2;
        sprites.begin();
        const S = tilePx * 1.25;
        const at = (x, y) => [origin[0] + (x + 0.5) * tilePx, origin[1] + (y + 0.5) * tilePx];
        for (const e of enemies) {
          const [x, y] = at(e.x, e.y);
          const fr = e.kind === 'bat' ? `bat_${Math.floor(t * 8) % 2}` : `slime_${Math.floor(t * 4) % 3}`;
          sprites.draw(x, y + tilePx * 0.3, S, S, { uv: atlas.uv(fr), anchor: [0.5, 1], flipX: e.flip });
        }
        for (const u of units) {
          const [x, y] = at(u.x, u.y);
          const fr = u.moving ? `hero_run_${Math.floor(t * 10) % 4}` : `hero_idle_${Math.floor(t * 2) % 2}`;
          sprites.draw(x, y + tilePx * 0.35, S, S, { uv: atlas.uv(fr), anchor: [0.5, 1], flipX: u.flip, color: kind === 'rts' && u !== units[0] ? '#b8d8ff' : '#ffffff' });
        }
        sprites.flush(enc, unitsT, cam, { clear: [0, 0, 0, 0] });

        compFx.draw(enc, { view: ctx.target, format: gpu.format }, { atlas: atlasView, fogT: bufs.blurB, rawT: bufs.visT, unitsT, tiles: bufs.tiles });

        readout.textContent = `fog texture ${world.fogW}×${world.fogH} (${SUB}/tile) · ${sources.length} vision source${sources.length > 1 ? 's' : ''} · ${useCpu ? 'CPU shadowcasting' : 'GPU ray march'}`;
      },
    };
  },
};

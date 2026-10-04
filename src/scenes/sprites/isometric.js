// Isometric Worlds & Depth Sorting — 2:1 isometric projection explained (morphing a square grid into a
// diamond grid), an iso city builder with mouse picking by inverse projection, and correct vs naive draw order.
// The iso pixel art (ground tiles, houses, towers, trees) is painted procedurally at startup.

import { ShapeBatch, SpriteBatch, Camera2D } from '../../core/batch.js';
import { makeCanvas, rng, textures } from '../../core/assets.js';
import { atlasTexture, tag, pixelText, clamp, fmt } from './_shared.js';

// ------------------------------------------------------------------ procedural iso sprites
function hexRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function shadeHex(h, k) {
  const [r, g, b] = hexRgb(h);
  const f = (v) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}

/** Paint one face (parallelogram above a ground edge a→b, height h) pixel by pixel. */
function face(g, a, b, h, colorAt) {
  const x0 = Math.min(a[0], b[0]);
  const x1 = Math.max(a[0], b[0]);
  for (let x = x0; x < x1; x++) {
    const u = (x + 0.5 - a[0]) / (b[0] - a[0]);
    const yg = a[1] + (b[1] - a[1]) * u;
    for (let y = Math.floor(yg - h); y < Math.floor(yg); y++) {
      const v = (yg - (y + 0.5)) / h;
      g.fillStyle = colorAt(u, v);
      g.fillRect(x, y, 1, 1);
    }
  }
}

function groundTile(top, left, right, seed, opts = {}) {
  const p = textures.painter(32, 20);
  const r = rng(seed);
  p.poly([[0, 8], [16, 16], [16, 20], [0, 12]], left);
  p.poly([[16, 16], [32, 8], [32, 12], [16, 20]], right);
  p.poly([[16, 0], [32, 8], [16, 16], [0, 8]], top);
  // texture speckles inside the diamond
  const img = p.g.getImageData(0, 0, 32, 16);
  for (let y = 0; y < 16; y++)
    for (let x = 0; x < 32; x++) {
      const i = (y * 32 + x) * 4;
      if (!img.data[i + 3]) continue;
      const q = r();
      const k = q < (opts.dark ?? 0.12) ? 0.86 : q > 1 - (opts.light ?? 0.08) ? 1.12 : 1;
      for (let c = 0; c < 3; c++) img.data[i + c] = Math.min(255, img.data[i + c] * k);
    }
  p.g.putImageData(img, 0, 0);
  if (opts.edge) {
    p.line(16, 0, 31, 7, opts.edge);
    p.line(0, 8, 15, 1, opts.edge);
  }
  return { canvas: p.c, cy: 8 };
}

function building(levels, { wall, roof, window: win, kind }) {
  const lh = 12;
  const roofH = kind === 'house' ? 10 : 3;
  const H = 16 + levels * lh + roofH + 2;
  const p = textures.painter(32, H);
  const g = p.g;
  const base = H - 16;
  const m = 4;
  const f = (32 - 2 * m) / 32;
  const C = [16, base + 8];
  const corner = (dx, dy) => [Math.round(C[0] + dx * f), Math.round(C[1] + dy * f)];
  const L = corner(-16, 0);
  const B = corner(0, 8);
  const R = corner(16, 0);
  const T = corner(0, -8);
  const wh = levels * lh;
  const winAt = (dark) => (u, v) => {
    const lv = v * levels;
    const inWin = (u * 3) % 1 > 0.28 && (u * 3) % 1 < 0.72 && lv % 1 > 0.3 && lv % 1 < 0.75 && lv < levels - 0.05;
    const lit = Math.sin(Math.floor(u * 3) * 12.9 + Math.floor(lv) * 78.2 + levels) > 0.1;
    if (inWin) return lit ? win : shadeHex(win, 0.35);
    return shadeHex(wall, dark ? 0.68 : 0.95 - 0.1 * v);
  };
  face(g, L, B, wh, winAt(true));
  face(g, B, R, wh, winAt(false));
  const up = (q, d) => [q[0], q[1] - d];
  if (kind === 'house') {
    const apex = [C[0], C[1] - wh - roofH - 4];
    p.poly([up(L, wh), up(T, wh), apex], shadeHex(roof, 0.8));
    p.poly([up(T, wh), up(R, wh), apex], shadeHex(roof, 0.95));
    p.poly([up(L, wh), up(B, wh), apex], shadeHex(roof, 0.7));
    p.poly([up(B, wh), up(R, wh), apex], roof);
    // door on the right face
    face(g, [B[0] + 4, B[1] - 2], [B[0] + 7, B[1] - 3.5], 6, () => '#4a2f22');
  } else {
    p.poly([up(L, wh), up(T, wh), up(R, wh), up(B, wh)], roof);
    p.poly([up(L, wh + 2), up(T, wh + 2), up(R, wh + 2), up(B, wh + 2)], shadeHex(roof, 1.15));
    const inner = (q) => [Math.round(C[0] + (q[0] - C[0]) * 0.7), Math.round(C[1] - wh - 2 + (q[1] - C[1]) * 0.7)];
    p.poly([inner(L), inner(T), inner(R), inner(B)], shadeHex(roof, 0.8));
    if (levels >= 3) p.rect(C[0] - 1, C[1] - wh - 9, 1, 7, '#cbd5e1');
  }
  textures.outline(p.c, '#1a1c2c');
  return { canvas: p.c, cy: H - 8 };
}

function isoTree(seed) {
  const p = textures.painter(32, 40);
  const r = rng(seed);
  p.ellipse(16, 33, 7, 3, 'rgba(0,0,0,0.25)');
  p.rect(15, 22, 3, 11, '#6b4628');
  p.rect(15, 22, 1, 11, '#5b3a29');
  p.ellipse(16, 16, 10, 9, '#257179');
  p.ellipse(15, 14, 9, 8, '#38b764');
  p.ellipse(13, 11, 4, 3, '#a7f070');
  for (let i = 0; i < 6; i++) p.px(9 + Math.floor(r() * 14), 9 + Math.floor(r() * 12), '#2a9d55');
  textures.outline(p.c, '#1a1c2c');
  return { canvas: p.c, cy: 32 };
}

function buildIsoAtlas(atlas) {
  const list = {
    grass: groundTile('#5cb85c', '#6b4628', '#8f563b', 1, { dark: 0.14 }),
    grass2: groundTile('#52ad57', '#6b4628', '#8f563b', 2, { dark: 0.18, light: 0.1 }),
    road: groundTile('#6e7488', '#3f4458', '#566078', 3, { dark: 0.08, light: 0.04, edge: '#8b93a8' }),
    water: groundTile('#3b7dd8', '#2a4f8f', '#335fa8', 4, { dark: 0.05, light: 0.12 }),
    plaza: groundTile('#c9b48a', '#7a6a4f', '#9a8866', 5, { dark: 0.1 }),
    tree: isoTree(9),
    tree2: isoTree(17),
  };
  const schemes = [
    { wall: '#e8d5b5', roof: '#c0453f', window: '#ffe08a', kind: 'house' },
    { wall: '#d9e3ec', roof: '#3b5dc9', window: '#ffe08a', kind: 'house' },
    { wall: '#9fb3c8', roof: '#566c86', window: '#a5f3fc', kind: 'tower' },
    { wall: '#c9a27a', roof: '#6b4628', window: '#ffd27a', kind: 'tower' },
  ];
  list.house0 = building(1, schemes[0]);
  list.house1 = building(1, schemes[1]);
  for (let l = 2; l <= 5; l++) {
    list[`tower${l}a`] = building(l, schemes[2]);
    list[`tower${l}b`] = building(l, schemes[3]);
  }
  // the villagers come from the main atlas: copy them in, so ONE texture holds everything we sort together
  for (const n of atlas.anim('hero_run')) {
    const f = atlas.frames[n];
    const c = makeCanvas(f.w, f.h);
    c.getContext('2d').drawImage(atlas.canvas, f.x, f.y, f.w, f.h, 0, 0, f.w, f.h);
    list[n] = { canvas: c, cy: f.h };
  }
  // pack in a row
  const names = Object.keys(list);
  const W = names.length * 36;
  const Hh = Math.max(...names.map((n) => list[n].canvas.height)) + 4;
  const c = makeCanvas(W, Hh);
  const g = c.getContext('2d');
  const frames = {};
  names.forEach((n, i) => {
    const s = list[n];
    g.drawImage(s.canvas, i * 36 + 2, 2);
    frames[n] = { x: i * 36 + 2, y: 2, w: s.canvas.width, h: s.canvas.height, cy: s.cy };
  });
  const uv = (n) => {
    const f = frames[n];
    return [f.x / W, f.y / Hh, (f.x + f.w) / W, (f.y + f.h) / Hh];
  };
  return { canvas: c, frames, uv };
}

const N_CITY = 10;

export default {
  wheel: true,
  interaction: 'Hover tiles · click to build · wheel to zoom',
  examples: [
    {
      id: 'projection',
      label: 'Projection explained',
      kind: 'Abstract',
      note: 'A square top-down grid becomes isometric in two steps: <b>rotate 45°</b>, then <b>squash vertically by half</b>. The result is the classic 2:1 diamond (a tile is twice as wide as it is tall), and heights are simply drawn upward on screen. Hover a tile: the inverse transform finds it at any point of the morph.',
      hint: 'Turn off “Animate” and drag “Morph” · hover a tile',
    },
    {
      id: 'city',
      label: 'Iso city builder',
      kind: 'In a game',
      note: 'The mouse position is converted back into grid coordinates with the <b>inverse projection</b> — that’s how you pick a diamond tile. Click to place the selected building (heights vary), trees or roads. Everything is drawn back to front, sorted by <code>x + y</code>.',
      hint: 'Hover to highlight · click to build',
    },
    {
      id: 'sorting',
      label: 'Depth sorting',
      kind: 'Comparison',
      note: 'Villagers walk between trees and houses. With <b>correct</b> sorting, everything is drawn in order of its depth key <code>x + y</code> (farther first). With <b>naive</b> order (scenery first, characters last) the villagers are painted <i>on top of</i> houses they are standing behind — the most common isometric bug.',
      hint: 'Set “Draw order” to Naive to see the bug',
    },
  ],
  controls: [
    { type: 'toggle', key: 'animate', label: 'Animate', value: true, showFor: ['projection'] },
    { type: 'slider', key: 'morph', label: 'Morph: top-down → isometric', min: 0, max: 1, step: 0.01, value: 1, showFor: ['projection'], help: '0 = square grid, 0.5 = rotated, 1 = rotated + squashed (2:1).' },
    { type: 'toggle', key: 'heights', label: 'Extrude block heights', value: true, showFor: ['projection'] },
    {
      type: 'select',
      key: 'tool',
      label: 'Build tool (click)',
      value: 'tower',
      showFor: ['city'],
      options: [
        { value: 'tower', label: 'Tower (random height)' },
        { value: 'house', label: 'House' },
        { value: 'tree', label: 'Tree' },
        { value: 'road', label: 'Road' },
        { value: 'water', label: 'Water' },
        { value: 'bulldoze', label: 'Bulldoze' },
      ],
    },
    {
      type: 'select',
      key: 'order',
      label: 'Draw order',
      value: 'sorted',
      showFor: ['sorting', 'city'],
      options: [
        { value: 'sorted', label: 'Correct: sorted by x + y' },
        { value: 'naive', label: 'Naive: scenery first, characters last' },
      ],
    },
    { type: 'slider', key: 'walkers', label: 'Villagers', min: 1, max: 40, step: 1, value: 12, showFor: ['sorting', 'city'] },
    { type: 'toggle', key: 'sortKeys', label: 'Show depth keys (x + y)', value: false, showFor: ['sorting', 'city'] },
    { type: 'toggle', key: 'grid', label: 'Grid lines', value: false, showFor: ['city', 'sorting'] },
  ],
  about: {
    summary: 'Isometric games fake 3D with a 2D projection: rotate the map 45°, squash it to half height, draw heights straight up — and draw everything back to front.',
    what: `<p>Three views: the projection itself morphing from a square grid to the iso diamond grid, a small city builder that picks tiles with the inverse
      projection, and a depth-sorting comparison where villagers walk among houses and trees.</p>`,
    how: `<ol>
      <li><b>Projection</b> (2:1 “pixel-art isometric”): <code>screenX = (x − y) × tileW/2</code>, <code>screenY = (x + y) × tileH/2 − z</code>, with
        tileW = 2 × tileH (here 32×16 pixels). It’s a 45° rotation followed by a vertical squash of 0.5 — strictly speaking dimetric, but everyone calls it isometric.</li>
      <li><b>Picking</b> (screen → grid): invert it: <code>a = sx / (tileW/2)</code>, <code>b = sy / (tileH/2)</code>, <code>x = (a + b) / 2</code>,
        <code>y = (b − a) / 2</code>, then <code>floor()</code>.</li>
      <li><b>Depth sorting</b>: draw ground first, then all objects ordered by <code>x + y</code> (the diagonal row; farther rows first). Moving characters use their
        fractional position, so they slide correctly between rows. Ties can be broken by x, then by height.</li>
      <li>Objects bigger than one tile need splitting into 1-tile slices or a smarter topological sort; the depth buffer (z = x + y) also works with alpha-tested pixel art.</li>
      <li>The 2:1 ratio makes edges perfect pixel stair-steps (2 across, 1 down) — that’s why pixel artists love it.</li>
    </ol>`,
    uses: [
      { title: 'City builders & sims', text: 'SimCity 2000, RollerCoaster Tycoon, Theme Hospital, Townscaper-style toys.' },
      { title: 'Tactics & RPGs', text: 'Final Fantasy Tactics, Disgaea, Diablo II, Hades (which uses a similar 3/4 projection).' },
      { title: 'Strategy', text: 'Age of Empires II, Civilization II — iso tiles with sorted units and buildings.' },
      { title: 'Puzzle & cozy', text: 'Monument Valley’s impossible geometry is isometric projection exploited on purpose.' },
    ],
    try: [
      'In <b>Projection explained</b>, turn off <i>Animate</i> and drag <i>Morph</i> from 0 to 1: first the rotation, then the squash.',
      'Hover tiles mid-morph — the inverse transform still finds the right one.',
      'In <b>Depth sorting</b>, switch <i>Draw order</i> to Naive and watch villagers walking “through” houses.',
      'Turn on <i>Show depth keys</i>: every sprite is drawn in ascending order of x + y.',
      'In the <b>city builder</b>, build a row of tall towers and walk villagers behind them.',
    ],
    ask: [
      '2:1 isometric projection with mouse picking',
      'depth sorting for isometric sprites by x + y',
      'an isometric city builder grid with buildings of different heights',
      'convert screen coordinates to isometric tile coordinates',
      'procedural isometric pixel-art buildings',
    ],
    perf: `<p>The projection is free (two multiply-adds). The cost is <b>sorting</b> every visible object each frame — trivial for thousands of objects in JavaScript,
      but for 100k+ use the depth buffer (write depth = x + y and alpha-test the pixel art) and let the GPU resolve it, as in
      <a href="#/s/sprite-batching">Sprite Batching → Crowd</a>.</p>`,
    api: `<p>Identical in WebGL2 and WebGPU: it’s sprite drawing with a different coordinate formula and a sort. (This scene uses the WebGPU sprite batcher.)</p>`,
    code: [
      {
        title: 'Projection and its inverse',
        lang: 'js',
        src: `const iso = (x, y, z = 0) => [
  ox + (x - y) * (TW / 2),
  oy + (x + y) * (TH / 2) - z,
];
function pick(sx, sy) {                 // screen -> tile
  const a = (sx - ox) / (TW / 2);
  const b = (sy - oy) / (TH / 2);
  return [Math.floor((a + b) / 2), Math.floor((b - a) / 2)];
}`,
      },
      {
        title: 'Back-to-front drawing',
        lang: 'js',
        src: `const list = [...buildings, ...trees, ...villagers];
list.sort((a, b) => (a.x + a.y) - (b.x + b.y) || a.x - b.x);  // farther rows first
for (const o of list) {
  const [sx, sy] = iso(o.x, o.y);
  sprites.draw(sx, sy, o.w * k, o.h * k, { uv: o.uv, anchor: o.anchor });
}`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const { atlas } = await atlasTexture(gpu);
    const iso = buildIsoAtlas(atlas);
    const isoTex = gpu.textureFromImage(iso.canvas, { label: 'iso-atlas' }).createView();
    const sprites = new SpriteBatch(gpu, { texture: isoTex, filter: 'nearest' });
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    const info = tag(ctx, 'right:8px;top:48px;text-align:right;line-height:1.5');
    const canvasOf = () => ({ view: ctx.target, format: gpu.format });
    const r = rng(3);

    // ---------------------------------------------------------------- city map
    // ground: 0 grass, 1 road, 2 water, 3 plaza; object: null | {kind, name}
    const N = N_CITY;
    const ground = new Uint8Array(N * N);
    const objs = new Array(N * N).fill(null);
    const towerName = () => `tower${2 + Math.floor(r() * 4)}${r() < 0.5 ? 'a' : 'b'}`;
    const resetCity = () => {
      ground.fill(0);
      objs.fill(null);
      for (let i = 0; i < N; i++) {
        ground[4 * N + i] = 1;
        ground[i * N + 6] = 1;
      }
      for (const [x, y] of [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]]) ground[y * N + x] = 2;
      for (let y = 5; y <= 6; y++) for (let x = 7; x <= 8; x++) ground[y * N + x] = 3;
      const put = (x, y, kind, name) => (objs[y * N + x] = { kind, name });
      put(2, 2, 'house', 'house0');
      put(3, 2, 'house', 'house1');
      put(2, 3, 'tree', 'tree');
      put(7, 2, 'tower', 'tower4a');
      put(8, 2, 'tower', 'tower3b');
      put(8, 3, 'tower', 'tower5a');
      put(7, 3, 'tower', 'tower2b');
      put(4, 7, 'house', 'house0');
      put(3, 8, 'house', 'house1');
      put(5, 8, 'tree', 'tree2');
      put(8, 8, 'tower', 'tower3a');
      put(9, 7, 'tree', 'tree');
      put(4, 1, 'tree', 'tree2');
      put(1, 6, 'tree', 'tree');
      put(2, 7, 'house', 'house1');
      put(9, 5, 'tree', 'tree2');
    };
    resetCity();
    const walkable = (x, y) => x >= 0 && y >= 0 && x < N && y < N && !objs[y * N + x] && ground[y * N + x] !== 2;

    // villagers move tile to tile on the grid, interpolating in between
    const walkers = [];
    const spawnWalker = () => {
      for (let tries = 0; tries < 50; tries++) {
        const x = Math.floor(r() * N);
        const y = Math.floor(r() * N);
        if (walkable(x, y)) return { x: x + 0.5, y: y + 0.5, from: [x, y], to: [x, y], t: 1, speed: 1.4 + r() * 0.8, phase: r() * 4, face: 1, tint: r() };
      }
      return null;
    };
    const stepWalker = (w, dt) => {
      w.t += dt * w.speed;
      if (w.t >= 1) {
        w.from = w.to;
        const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter(([dx, dy]) => walkable(w.from[0] + dx, w.from[1] + dy));
        if (dirs.length) {
          // prefer to keep going straight
          const last = [w.to[0] - (w.prev?.[0] ?? w.to[0]), w.to[1] - (w.prev?.[1] ?? w.to[1])];
          const straight = dirs.find(([dx, dy]) => dx === last[0] && dy === last[1]);
          const d = straight && r() < 0.7 ? straight : dirs[Math.floor(r() * dirs.length)];
          w.prev = w.from;
          w.to = [w.from[0] + d[0], w.from[1] + d[1]];
        }
        w.t = 0;
      }
      const k = Math.min(1, w.t);
      w.x = w.from[0] + 0.5 + (w.to[0] - w.from[0]) * k;
      w.y = w.from[1] + 0.5 + (w.to[1] - w.from[1]) * k;
      const sdx = w.to[0] - w.from[0] - (w.to[1] - w.from[1]); // screen-x direction
      if (sdx) w.face = Math.sign(sdx);
      w.phase += dt * 8;
    };

    let zoomStep = 0;
    let projT = 1;
    let clock = 0;

    // ---------------------------------------------------------------- projection explained
    function projection(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const G = 6;
      if (p.animate) {
        const c = (clock * 0.25 + 1) % 2; // start at the isometric end
        const tri = c < 1 ? c : 2 - c;
        projT = clamp((tri - 0.15) / 0.7, 0, 1);
        projT = projT * projT * (3 - 2 * projT);
      } else projT = p.morph;
      const t = projT;
      const S = Math.min(W, H) * 0.095;
      const a = (Math.PI / 4) * Math.min(1, t * 2);
      const sq = 1 - 0.5 * clamp(t * 2 - 1, 0, 1);
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      const ox = W * 0.42;
      const oy = H * 0.55;
      const zs = 0.62 * S * t;
      const P = (x, y, z = 0) => {
        const u = (x - G / 2) * S;
        const v = (y - G / 2) * S;
        return [ox + ca * u - sa * v, oy + (sa * u + ca * v) * sq - z * zs];
      };
      const heights = [[1, 1, 1], [1, 2, 2], [4, 1, 1.5], [4, 2, 1], [2, 4, 0.5], [3, 4, 0.5], [4, 4, 2.5], [0, 5, 1]];
      const hmap = new Map(p.heights ? heights.map(([x, y, h]) => [`${x},${y}`, h]) : []);
      // inverse: screen -> grid (ground plane)
      const ptr = ctx.pointer;
      const du = ptr.x - ox;
      const dv = (ptr.y - oy) / sq;
      const gx = Math.floor((ca * du + sa * dv) / S + G / 2);
      const gy = Math.floor((-sa * du + ca * dv) / S + G / 2);
      const hoverOk = ptr.over && gx >= 0 && gy >= 0 && gx < G && gy < G;
      cam.setViewport(W, H).reset();
      shapes.rect(0, 0, W, H, '#10141f');
      const quad = (A, B, C, D, col) => {
        shapes.triangle(A[0], A[1], B[0], B[1], C[0], C[1], col);
        shapes.triangle(A[0], A[1], C[0], C[1], D[0], D[1], col);
      };
      // ground tiles (checkerboard)
      for (let y = 0; y < G; y++)
        for (let x = 0; x < G; x++) {
          const on = hoverOk && x === gx && y === gy;
          const col = on ? '#fcd34d' : (x + y) % 2 ? '#3f8f4f' : '#4aa35a';
          quad(P(x, y), P(x + 1, y), P(x + 1, y + 1), P(x, y + 1), col);
        }
      for (let i = 0; i <= G; i++) {
        shapes.line(...P(i, 0), ...P(i, G), 1, '#ffffff30');
        shapes.line(...P(0, i), ...P(G, i), 1, '#ffffff30');
      }
      // blocks, back to front by x + y
      const blocks = [...hmap.entries()].map(([k, h]) => [...k.split(',').map(Number), h]).sort((A, B) => A[0] + A[1] - (B[0] + B[1]));
      for (const [x, y, h] of blocks) {
        const hov = hoverOk && x === gx && y === gy;
        quad(P(x + 1, y, 0), P(x + 1, y + 1, 0), P(x + 1, y + 1, h), P(x + 1, y, h), hov ? '#d9a514' : '#8f9bb5');
        quad(P(x, y + 1, 0), P(x + 1, y + 1, 0), P(x + 1, y + 1, h), P(x, y + 1, h), hov ? '#b88a0f' : '#66728c');
        quad(P(x, y, h), P(x + 1, y, h), P(x + 1, y + 1, h), P(x, y + 1, h), hov ? '#fde68a' : '#c3cde0');
        shapes.polyline([P(x, y, h), P(x + 1, y, h), P(x + 1, y + 1, h), P(x, y + 1, h)], 1.2, '#1a1c2c', { closed: true });
        shapes.line(...P(x + 1, y + 1, 0), ...P(x + 1, y + 1, h), 1.2, '#1a1c2c');
      }
      // axes
      const O = P(0, 0);
      const X = P(G + 0.9, 0);
      const Y = P(0, G + 0.9);
      shapes.line(...O, ...X, 2.5, '#f87171');
      shapes.line(...O, ...Y, 2.5, '#4ade80');
      shapes.circle(...X, 4, '#f87171');
      shapes.circle(...Y, 4, '#4ade80');
      const fs = Math.max(2, Math.round(H / 170));
      pixelText(shapes, 'X', X[0] + 6, X[1] - 4 * fs, fs, '#f87171');
      pixelText(shapes, 'Y', Y[0] - 6 - 3 * fs, Y[1] - 4 * fs, fs, '#4ade80');
      // one unit tile under the same transform, with its width : height ratio
      const bx = W * 0.84;
      const by = H * 0.42;
      const US = Math.min(W * 0.1, S * 1.8);
      const Ut = (x, y) => [bx + (ca * x - sa * y) * US, by + (sa * x + ca * y) * sq * US];
      const dia = [Ut(-0.5, -0.5), Ut(0.5, -0.5), Ut(0.5, 0.5), Ut(-0.5, 0.5)];
      shapes.triangle(...dia[0], ...dia[1], ...dia[2], '#fcd34d33');
      shapes.triangle(...dia[0], ...dia[2], ...dia[3], '#fcd34d33');
      shapes.polyline(dia, 2, '#fcd34d', { closed: true });
      const label = t < 0.02 ? 'TOP-DOWN' : t < 0.5 ? 'ROTATE 45' : t < 0.98 ? 'SQUASH Y' : 'ISO 2:1';
      pixelText(shapes, label, bx, by + US * 0.95, fs, '#fcd34d', { align: 'center' });
      pixelText(shapes, `W:H = ${(1 / sq).toFixed(2)}:1`, bx, by + US * 0.95 + 8 * fs, Math.max(1, fs - 1), '#94a3b8', { align: 'center' });
      shapes.flush(enc, canvasOf(), cam, { clear: [0.06, 0.08, 0.12, 1] });
      info.innerHTML =
        `screen = rotate(45°·${Math.min(1, t * 2).toFixed(2)}) · squashY(${sq.toFixed(2)}) · (x, y) − z<br>` +
        (hoverOk ? `mouse → inverse → tile <b style="color:#fcd34d">(${gx}, ${gy})</b>${hmap.get(`${gx},${gy}`) ? ` · height ${hmap.get(`${gx},${gy}`)}` : ''}` : '<span style="color:#94a3b8">hover a tile</span>');
    }

    // ---------------------------------------------------------------- shared iso renderer (city & sorting)
    function isoWorld(ctx, { editable }) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 20);
      const ptr = ctx.pointer;
      if (ptr.wheel) zoomStep = clamp(zoomStep - Math.sign(ptr.wheel), -2, 4);
      const fit = Math.min(W / (N * 32), H / (N * 16 + 60)) * 1.12;
      const k = Math.max(1, Math.round(fit) + zoomStep);
      const ox = Math.round(W / 2);
      const oy = Math.round(H / 2 - (N * 16 * k) / 2 + 14 * k);
      const isoP = (x, y) => [ox + (x - y) * 16 * k, oy + (x + y) * 8 * k];
      // picking: screen -> grid
      const a = (ptr.x - ox) / (16 * k);
      const b = (ptr.y - oy) / (8 * k);
      const hx = Math.floor((a + b) / 2);
      const hy = Math.floor((b - a) / 2);
      const hoverOk = ptr.over && hx >= 0 && hy >= 0 && hx < N && hy < N;
      if (editable && hoverOk && ptr.clicked && ptr.button === 0) {
        const i = hy * N + hx;
        const tool = p.tool;
        if (tool === 'bulldoze') {
          objs[i] = null;
          if (ground[i] !== 0) ground[i] = 0;
        } else if (tool === 'road' || tool === 'water') {
          objs[i] = null;
          ground[i] = tool === 'road' ? 1 : 2;
        } else {
          ground[i] = ground[i] === 2 ? 0 : ground[i];
          objs[i] = { kind: tool, name: tool === 'tower' ? towerName() : tool === 'house' ? `house${r() < 0.5 ? 0 : 1}` : `tree${r() < 0.5 ? '' : 2}` };
        }
        for (const w of walkers) if (!walkable(w.to[0], w.to[1])) Object.assign(w, spawnWalker() || {});
      }
      // villagers
      const want = Math.round(p.walkers);
      while (walkers.length < want) {
        const w = spawnWalker();
        if (!w) break;
        walkers.push(w);
      }
      walkers.length = Math.min(walkers.length, want);
      for (const w of walkers) stepWalker(w, dt);

      cam.setViewport(W, H).reset();
      shapes.rect(0, 0, W, H, '#0f1a24');
      shapes.flush(enc, canvasOf(), cam, { clear: [0.06, 0.1, 0.14, 1] });
      // ground (no sorting needed: it's all flat), drawn back to front anyway
      const f0 = iso.frames.grass;
      for (let s = 0; s < 2 * N - 1; s++)
        for (let x = 0; x < N; x++) {
          const y = s - x;
          if (y < 0 || y >= N) continue;
          const g = ground[y * N + x];
          const name = g === 1 ? 'road' : g === 2 ? 'water' : g === 3 ? 'plaza' : (x * 7 + y * 3) % 5 === 0 ? 'grass2' : 'grass';
          const [sx, sy] = isoP(x + 0.5, y + 0.5);
          sprites.draw(sx, sy, f0.w * k, f0.h * k, { uv: iso.uv(name), anchor: [0.5, f0.cy / f0.h] });
        }
      sprites.flush(enc, canvasOf(), cam);
      // grid & hover diamond
      if (p.grid)
        for (let i = 0; i <= N; i++) {
          shapes.line(...isoP(i, 0), ...isoP(i, N), 1, '#ffffff38');
          shapes.line(...isoP(0, i), ...isoP(N, i), 1, '#ffffff38');
        }
      if (hoverOk) {
        const dia = [isoP(hx, hy), isoP(hx + 1, hy), isoP(hx + 1, hy + 1), isoP(hx, hy + 1)];
        shapes.triangle(...dia[0], ...dia[1], ...dia[2], '#fcd34d40');
        shapes.triangle(...dia[0], ...dia[2], ...dia[3], '#fcd34d40');
        shapes.polyline(dia, 2, '#fcd34d', { closed: true });
      }
      shapes.flush(enc, canvasOf(), cam);
      // objects + villagers
      const list = [];
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          const o = objs[y * N + x];
          if (o) list.push({ key: x + y + 1, x: x + 0.5, y: y + 0.5, name: o.name, scenery: true });
        }
      if (editable && hoverOk && !objs[hy * N + hx] && ['tower', 'house', 'tree'].includes(p.tool))
        list.push({ key: hx + hy + 1, x: hx + 0.5, y: hy + 0.5, name: p.tool === 'tower' ? 'tower3a' : p.tool === 'house' ? 'house0' : 'tree', ghost: true, scenery: true });
      for (const w of walkers) list.push({ key: w.x + w.y, x: w.x, y: w.y, walker: w });
      if (p.order === 'naive') list.sort((A, B) => (A.scenery === B.scenery ? A.key - B.key : A.scenery ? -1 : 1));
      else list.sort((A, B) => A.key - B.key || A.x - B.x);
      // shadows under villagers
      for (const o of list) if (o.walker) {
        const [sx, sy] = isoP(o.x, o.y);
        shapes.box(sx, sy, 5 * k, 2 * k, '#00000055', { radius: 2 * k });
      }
      shapes.flush(enc, canvasOf(), cam);
      let order = 0;
      for (const o of list) {
        const [sx, sy] = isoP(o.x, o.y);
        o.sx = sx;
        o.sy = sy;
        o.order = order++;
        if (o.walker) {
          const w = o.walker;
          const fr = `hero_run_${Math.floor(w.phase) % 4}`;
          const f = iso.frames[fr];
          sprites.draw(sx, sy + k, f.w * k, f.h * k, { uv: iso.uv(fr), anchor: [0.5, 1], flipX: w.face < 0, color: w.tint < 0.33 ? '#ffd0d0' : w.tint < 0.66 ? '#d0ffd8' : '#ffffff' });
        } else {
          const f = iso.frames[o.name];
          sprites.draw(sx, sy, f.w * k, f.h * k, { uv: iso.uv(o.name), anchor: [0.5, f.cy / f.h], alpha: o.ghost ? 0.5 : 1 });
        }
      }
      sprites.flush(enc, canvasOf(), cam); // the whole sorted list in one draw call
      // depth keys
      if (p.sortKeys) {
        const fs = Math.max(1, Math.round(k * 0.75));
        for (const o of list) {
          const tall = o.walker ? 18 : iso.frames[o.name].cy + 2;
          pixelText(shapes, o.key.toFixed(o.walker ? 1 : 0), o.sx, o.sy - tall * k - 6 * fs, fs, o.walker ? '#fde047' : '#e2e8f0', { align: 'center' });
        }
        shapes.flush(enc, canvasOf(), cam);
      }
      return { hover: hoverOk ? [hx, hy] : null, count: list.length, k };
    }

    return {
      onAction(key) {
        if (key === 'reset') resetCity();
      },
      onExample() {
        zoomStep = 0;
      },
      frame(ctx) {
        shapes.begin();
        sprites.begin();
        clock += ctx.paused ? 0 : Math.min(ctx.dt, 1 / 20);
        if (ctx.example === 'projection') projection(ctx);
        else {
          const r2 = isoWorld(ctx, { editable: ctx.example === 'city' });
          const p = ctx.params;
          info.innerHTML =
            `${fmt(r2.count)} sprites drawn ${p.order === 'naive' ? '<b style="color:#fca5a5">naive order</b>' : 'sorted by <b style="color:#86efac">x + y</b>'} · zoom ${r2.k}×<br>` +
            (r2.hover ? `mouse → tile <b style="color:#fcd34d">(${r2.hover[0]}, ${r2.hover[1]})</b> · depth key ${r2.hover[0] + r2.hover[1] + 1}` : '<span style="color:#94a3b8">hover the map</span>');
        }
      },
    };
  },
};

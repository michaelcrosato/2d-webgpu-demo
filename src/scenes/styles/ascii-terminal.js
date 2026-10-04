import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas, rng } from '../../core/assets.js';
import { GAME_SCENE_CODE, GAME_SCENE_INCLUDES } from '../../core/gamescene.js';

// ASCII & text-mode rendering.
//  * A glyph atlas (font texture) is painted with Canvas2D at start-up: printable ASCII, CP437-style
//    shade/box characters drawn by hand, and mirrored katakana for the Matrix rain.
//  * Pass "cells" runs at ONE PIXEL PER CHARACTER CELL: it averages the source image over the cell and
//    fits a luminance plane to find edge direction (for / \ | - glyphs), or ray-marches donut.c's torus.
//  * The final pass maps each cell's brightness to a glyph from a "ramp" and copies that glyph from the atlas.
//  * The roguelike is a data texture (tile ids + text) + a field-of-view pass with memory (feedback).

// ------------------------------------------------------------------------------------------------
// Glyph atlas: 16 x 12 cells of 32 x 48 px. Slot = charCode - 32 for printable ASCII.
// ------------------------------------------------------------------------------------------------
const GW = 32;
const GH = 48;
const GCOLS = 16;
const GROWS = 12;
const SPECIAL = {
  '░': 96, '▒': 97, '▓': 98, '█': 99, '▀': 100, '▄': 101, '▌': 102, '▐': 103,
  '─': 104, '│': 105, '┌': 106, '┐': 107, '└': 108, '┘': 109, '├': 110, '┤': 111, '┬': 112, '┴': 113, '┼': 114,
  '═': 115, '║': 116, '╔': 117, '╗': 118, '╚': 119, '╝': 120, '·': 121, '♣': 122, '≈': 123, '♥': 124, '☺': 125, '♦': 126, '•': 127,
};
const gi = (ch) => SPECIAL[ch] ?? ch.charCodeAt(0) - 32;
const FONT = '"DejaVu Sans Mono", Menlo, Consolas, "Liberation Mono", "Courier New", monospace';

let glyphInfo = null;

function sameGlyph(a, b) {
  const c = makeCanvas(48, 48);
  const g = c.getContext('2d');
  g.font = `40px "Noto Sans CJK JP", "Hiragino Kaku Gothic ProN", "MS Gothic", "Yu Gothic", ${FONT}`;
  g.textBaseline = 'middle';
  g.fillStyle = '#fff';
  g.fillText(a, 4, 24);
  const da = g.getImageData(0, 0, 48, 48).data;
  g.clearRect(0, 0, 48, 48);
  g.fillText(b, 4, 24);
  const db = g.getImageData(0, 0, 48, 48).data;
  let diff = 0;
  let ink = 0;
  for (let i = 3; i < da.length; i += 4) {
    diff += Math.abs(da[i] - db[i]);
    ink += da[i];
  }
  return diff < 400 || ink < 400; // identical "tofu" boxes or nothing drawn -> unsupported
}

function buildGlyphAtlas() {
  if (glyphInfo) return glyphInfo;
  const c = makeCanvas(GW * GCOLS, GH * GROWS);
  const g = c.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = '#fff';
  g.strokeStyle = '#fff';
  const xy = (i) => [(i % GCOLS) * GW, Math.floor(i / GCOLS) * GH];
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.font = `bold 41px ${FONT}`;
  for (let code = 33; code < 127; code++) {
    const [x, y] = xy(code - 32);
    g.fillText(String.fromCharCode(code), x + GW / 2, y + GH / 2 + 1);
  }
  // --- CP437-style block & shade glyphs, drawn as pixels so they tile perfectly
  const blocks = (slot, fn) => {
    const [x, y] = xy(slot);
    for (let by = 0; by < GH / 4; by++) for (let bx = 0; bx < GW / 4; bx++) if (fn(bx, by)) g.fillRect(x + bx * 4, y + by * 4, 4, 4);
  };
  blocks(96, (x, y) => x % 2 === 0 && y % 2 === 0); // ░ 25%
  blocks(97, (x, y) => (x + y) % 2 === 0); // ▒ 50%
  blocks(98, (x, y) => !(x % 2 === 1 && y % 2 === 1)); // ▓ 75%
  blocks(99, () => true); // █
  blocks(100, (x, y) => y < GH / 8); // ▀
  blocks(101, (x, y) => y >= GH / 8); // ▄
  blocks(102, (x) => x < GW / 8); // ▌
  blocks(103, (x) => x >= GW / 8); // ▐
  // --- box drawing (single 104..114, double 115..120)
  const box = (slot, up, down, left, right, dbl = false) => {
    const [x, y] = xy(slot);
    const cx = x + GW / 2;
    const cy = y + GH / 2;
    const offs = dbl ? [-5, 5] : [0];
    for (const o of offs) {
      // horizontal strokes
      if (left || right) {
        const x0 = left ? x : cx + (dbl ? -o * (up || down ? 1 : 0) : 0) - 2;
        const x1 = right ? x + GW : cx + (dbl ? o * (up || down ? -1 : 0) : 0) + 2;
        g.fillRect(Math.min(x0, x1), cy + o - 2, Math.abs(x1 - x0), 4);
      }
      if (up || down) {
        const y0 = up ? y : cy - 2;
        const y1 = down ? y + GH : cy + 2;
        g.fillRect(cx + o - 2, y0, 4, y1 - y0);
      }
    }
  };
  box(104, 0, 0, 1, 1);
  box(105, 1, 1, 0, 0);
  box(106, 0, 1, 0, 1);
  box(107, 0, 1, 1, 0);
  box(108, 1, 0, 0, 1);
  box(109, 1, 0, 1, 0);
  box(110, 1, 1, 0, 1);
  box(111, 1, 1, 1, 0);
  box(112, 0, 1, 1, 1);
  box(113, 1, 0, 1, 1);
  box(114, 1, 1, 1, 1);
  box(115, 0, 0, 1, 1, true);
  box(116, 1, 1, 0, 0, true);
  box(117, 0, 1, 0, 1, true);
  box(118, 0, 1, 1, 0, true);
  box(119, 1, 0, 0, 1, true);
  box(120, 1, 0, 1, 0, true);
  // --- small symbols
  const at = (slot) => {
    const [x, y] = xy(slot);
    return [x + GW / 2, y + GH / 2];
  };
  let [sx, sy] = at(121); // ·
  g.beginPath();
  g.arc(sx, sy, 3.5, 0, Math.PI * 2);
  g.fill();
  [sx, sy] = at(122); // ♣
  g.beginPath();
  g.arc(sx, sy - 9, 7, 0, Math.PI * 2);
  g.arc(sx - 8, sy + 1, 7, 0, Math.PI * 2);
  g.arc(sx + 8, sy + 1, 7, 0, Math.PI * 2);
  g.fill();
  g.fillRect(sx - 2, sy, 4, 15);
  [sx, sy] = at(123); // ≈
  g.lineWidth = 3.5;
  for (const oy of [-6, 6]) {
    g.beginPath();
    for (let i = 0; i <= 24; i++) {
      const px = sx - 12 + i;
      const py = sy + oy + Math.sin((i / 24) * Math.PI * 2) * 3.5;
      if (i) g.lineTo(px, py);
      else g.moveTo(px, py);
    }
    g.stroke();
  }
  [sx, sy] = at(124); // ♥
  g.beginPath();
  g.moveTo(sx, sy + 12);
  g.bezierCurveTo(sx - 18, sy - 2, sx - 8, sy - 16, sx, sy - 6);
  g.bezierCurveTo(sx + 8, sy - 16, sx + 18, sy - 2, sx, sy + 12);
  g.fill();
  [sx, sy] = at(125); // ☺
  g.lineWidth = 3;
  g.beginPath();
  g.arc(sx, sy, 12, 0, Math.PI * 2);
  g.stroke();
  g.fillRect(sx - 6, sy - 5, 3, 4);
  g.fillRect(sx + 3, sy - 5, 3, 4);
  g.beginPath();
  g.arc(sx, sy + 1, 6, 0.2, Math.PI - 0.2);
  g.stroke();
  [sx, sy] = at(126); // ♦
  g.beginPath();
  g.moveTo(sx, sy - 14);
  g.lineTo(sx + 10, sy);
  g.lineTo(sx, sy + 14);
  g.lineTo(sx - 10, sy);
  g.fill();
  [sx, sy] = at(127); // •
  g.beginPath();
  g.arc(sx, sy, 6, 0, Math.PI * 2);
  g.fill();

  // --- Matrix glyphs (slots 128..191): mirrored half-width katakana + digits, or a Latin fallback
  const kana = [];
  for (let cp = 0xff66; cp <= 0xff9d; cp++) kana.push(String.fromCodePoint(cp));
  const hasKana = !sameGlyph('ｱ', 'ﾝ');
  const fallback = 'ZXCVBNMASDFGHJKLQWERTYUP0123456789:=*+-<>|$#@&%?!'.split('');
  const pool = hasKana ? [...kana.slice(0, 54), ...'0123456789'.split('')] : fallback;
  g.font = hasKana ? `38px "Noto Sans CJK JP", "Hiragino Kaku Gothic ProN", "MS Gothic", "Yu Gothic", ${FONT}` : `bold 36px ${FONT}`;
  for (let i = 0; i < 64; i++) {
    const [x, y] = xy(128 + i);
    g.save();
    g.translate(x + GW / 2, y + GH / 2 + 1);
    g.scale(-1, 1); // the film's code is mirrored
    g.fillText(pool[i % pool.length], 0, 0);
    g.restore();
  }

  // --- measure ink coverage of every printable ASCII glyph -> an auto-sorted 16-step ramp
  const data = g.getImageData(0, 0, c.width, c.height).data;
  const cover = [];
  for (let code = 32; code < 127; code++) {
    const [x, y] = xy(code - 32);
    let s = 0;
    for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) s += data[((y + j) * c.width + x + i) * 4];
    cover.push({ slot: code - 32, ch: String.fromCharCode(code), v: s / (GW * GH * 255) });
  }
  cover.sort((a, b) => a.v - b.v);
  const maxV = cover[cover.length - 1].v;
  const measured = [];
  const used = new Set();
  for (let k = 0; k < 16; k++) {
    const want = (k / 15) * maxV;
    let best = null;
    for (const cv of cover) if (!used.has(cv.slot) && cv.ch !== '_' && (!best || Math.abs(cv.v - want) < Math.abs(best.v - want))) best = cv;
    used.add(best.slot);
    measured.push(best);
  }
  measured.sort((a, b) => a.v - b.v);
  glyphInfo = { canvas: c, measured: measured.map((m) => m.slot), measuredText: measured.map((m) => m.ch).join(''), hasKana };
  return glyphInfo;
}

const RAMPS = {
  classic: ' .:-=+*#%@'.split('').map(gi),
  donut: ' .,-~:;=!*#$@'.split('').map(gi),
  blocks: [0, 96, 97, 98, 99],
};
function rampFor(name) {
  if (name === 'measured') return buildGlyphAtlas().measured;
  return RAMPS[name] || RAMPS.classic;
}

// ------------------------------------------------------------------------------------------------
// Roguelike: an 80 x 32 "terminal" encoded in a data texture.
//   B = 0   map tile   (R = tile type, G = random variation / item type)
//   B = 100 text       (R = glyph slot, G = color id)
//   B = 200 bar + text (R = glyph slot, G = bar id)
// ------------------------------------------------------------------------------------------------
const TW = 80;
const TH = 32;
const MX = 20; // map offset inside the terminal
const MY = 2;
const MW = 60;
const MH = 28;
const ROCK = 0, WALL = 1, FLOOR = 2, CORR = 3, DOOR = 4, WATER = 5, GRASS = 6, STAIRS = 7, TORCH = 8, ITEM = 9, BONES = 10;
const MONSTERS = [
  { ch: 'g', color: 5, name: 'goblin mystic', state: '(Wandering)' },
  { ch: 'r', color: 1, name: 'rat', state: '(Hunting)' },
  { ch: 'k', color: 8, name: 'kobold', state: '(Sleeping)' },
  { ch: 'o', color: 11, name: 'ogre', state: '(Wandering)' },
  { ch: 'j', color: 10, name: 'jackal', state: '(Hunting)' },
  { ch: 'B', color: 9, name: 'vampire bat', state: '(Flitting)' },
];

let dungeon = null;
function buildDungeon() {
  if (dungeon) return dungeon;
  const R = rng(1337);
  const T = new Uint8Array(MW * MH);
  const V = new Uint8Array(MW * MH);
  const idx = (x, y) => y * MW + x;
  const get = (x, y) => (x < 0 || y < 0 || x >= MW || y >= MH ? -1 : T[idx(x, y)]);
  const rooms = [];
  for (let tries = 0; tries < 600 && rooms.length < 9; tries++) {
    const w = 6 + Math.floor(R() * 9);
    const h = 4 + Math.floor(R() * 4);
    const x = 2 + Math.floor(R() * (MW - w - 4));
    const y = 2 + Math.floor(R() * (MH - h - 4));
    if (rooms.some((r) => x < r.x + r.w + 3 && x + w + 3 > r.x && y < r.y + r.h + 2 && y + h + 2 > r.y)) continue;
    rooms.push({ x, y, w, h, cx: x + (w >> 1), cy: y + (h >> 1) });
  }
  rooms.sort((a, b) => a.cx - b.cx);
  for (const r of rooms) for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) T[idx(x, y)] = FLOOR;
  const carve = (x, y) => {
    if (T[idx(x, y)] === ROCK) T[idx(x, y)] = CORR;
  };
  const corridor = (a, b) => {
    let x = a.cx;
    let y = a.cy;
    const hFirst = R() < 0.5;
    const walkX = () => {
      while (x !== b.cx) {
        carve(x, y);
        x += Math.sign(b.cx - x);
      }
    };
    const walkY = () => {
      while (y !== b.cy) {
        carve(x, y);
        y += Math.sign(b.cy - y);
      }
    };
    if (hFirst) {
      walkX();
      walkY();
    } else {
      walkY();
      walkX();
    }
    carve(x, y);
  };
  for (let i = 0; i + 1 < rooms.length; i++) corridor(rooms[i], rooms[i + 1]);
  if (rooms.length > 5) {
    corridor(rooms[0], rooms[2]);
    corridor(rooms[rooms.length - 3], rooms[rooms.length - 1]);
  }
  // walls around everything walkable
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      if (T[idx(x, y)] !== ROCK) continue;
      let near = false;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) if (get(x + i, y + j) === FLOOR || get(x + i, y + j) === CORR) near = true;
      if (near) T[idx(x, y)] = WALL;
    }
  // doors: corridor cells squeezed between two walls next to a room
  for (let y = 1; y < MH - 1; y++)
    for (let x = 1; x < MW - 1; x++) {
      if (T[idx(x, y)] !== CORR) continue;
      const nextToRoom = [get(x + 1, y), get(x - 1, y), get(x, y + 1), get(x, y - 1)].includes(FLOOR);
      const ew = get(x - 1, y) === WALL && get(x + 1, y) === WALL;
      const ns = get(x, y - 1) === WALL && get(x, y + 1) === WALL;
      if (nextToRoom && (ew || ns) && R() < 0.7) T[idx(x, y)] = DOOR;
    }
  // a pond, grass patches, stairs, torches, items
  const pond = rooms[Math.floor(rooms.length / 2)];
  for (let y = pond.y; y < pond.y + pond.h; y++)
    for (let x = pond.x; x < pond.x + pond.w; x++) {
      const dx = (x - pond.cx - 0.5) / (pond.w * 0.42);
      const dy = (y - pond.cy) / (pond.h * 0.45);
      if (dx * dx + dy * dy + R() * 0.35 < 1) T[idx(x, y)] = WATER;
    }
  for (const r of rooms) {
    if (R() < 0.35) continue;
    const gx = r.x + Math.floor(R() * r.w);
    const gy = r.y + Math.floor(R() * r.h);
    for (let y = r.y - 1; y <= r.y + r.h; y++)
      for (let x = r.x - 1; x <= r.x + r.w; x++) {
        const d = Math.hypot(x - gx, (y - gy) * 1.4);
        if (T[idx(x, y)] === FLOOR && d < 3.5 + R() * 1.5) T[idx(x, y)] = GRASS;
      }
  }
  const last = rooms[rooms.length - 1];
  T[idx(last.cx + 1, last.cy)] = STAIRS;
  const torches = [];
  for (const r of rooms) {
    if (torches.length >= 7) break;
    for (let k = 0; k < 6; k++) {
      const x = r.x + 1 + Math.floor(R() * Math.max(1, r.w - 2));
      const y = r.y - 1;
      if (get(x, y) === WALL && get(x, y + 1) === FLOOR) {
        T[idx(x, y)] = TORCH;
        torches.push([x + MX, y + MY]);
        break;
      }
    }
  }
  const floors = [];
  for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) if (T[idx(x, y)] === FLOOR) floors.push([x, y]);
  for (let k = 0; k < 14; k++) {
    const [x, y] = floors[Math.floor(R() * floors.length)];
    if (k < 11) {
      T[idx(x, y)] = ITEM;
      V[idx(x, y)] = k % 7;
    } else T[idx(x, y)] = BONES;
  }
  for (let i = 0; i < T.length; i++) if (T[i] !== ITEM) V[i] = Math.floor(R() * 255);

  // ---- path finding for the animated @ and monsters (BFS on walkable tiles)
  const walk = (x, y) => {
    const t = get(x, y);
    return t >= FLOOR && t !== TORCH;
  };
  const bfs = (sx, sy, tx, ty) => {
    const prev = new Int32Array(MW * MH).fill(-1);
    const q = [idx(sx, sy)];
    prev[idx(sx, sy)] = idx(sx, sy);
    while (q.length) {
      const cur = q.shift();
      if (cur === idx(tx, ty)) break;
      const cx = cur % MW;
      const cy = (cur / MW) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (!walk(nx, ny) || prev[idx(nx, ny)] >= 0) continue;
        prev[idx(nx, ny)] = cur;
        q.push(idx(nx, ny));
      }
    }
    const path = [];
    let cur = idx(tx, ty);
    if (prev[cur] < 0) return path;
    while (cur !== idx(sx, sy)) {
      path.push([cur % MW, (cur / MW) | 0]);
      cur = prev[cur];
    }
    return path.reverse();
  };
  const randIn = (r) => {
    for (let k = 0; k < 30; k++) {
      const x = r.x + Math.floor(R() * r.w);
      const y = r.y + Math.floor(R() * r.h);
      if (walk(x, y) && T[idx(x, y)] !== WATER) return [x, y];
    }
    return [r.cx, r.cy];
  };
  const tour = [];
  let cur = [rooms[0].cx, rooms[0].cy];
  const visit = [...rooms.slice(1), rooms[0]];
  for (const r of visit) {
    const tgt = randIn(r);
    const p = bfs(cur[0], cur[1], tgt[0], tgt[1]);
    tour.push(...p);
    if (p.length) cur = p[p.length - 1];
  }
  // pre-seed the map memory: everything the @ saw during the first part of its walk is "remembered"
  const start = Math.floor(tour.length * 0.35);
  const opaque = (x, y) => {
    const t = get(x, y);
    return t <= WALL || t === TORCH;
  };
  const seenFlag = new Uint8Array(MW * MH);
  for (let s = 0; s < start; s++) {
    const [px, py] = tour[s];
    for (let y = py - 6; y <= py + 6; y++)
      for (let x = px - 9; x <= px + 9; x++) {
        if (x < 0 || y < 0 || x >= MW || y >= MH || Math.hypot(x - px, (y - py) * 1.5) > 9) continue;
        const dx = x - px;
        const dy = y - py;
        const n = Math.max(Math.abs(dx), Math.abs(dy));
        let ok = true;
        for (let i = 1; i < n && ok; i++) if (opaque(Math.floor(px + (dx * i) / n + 0.5001), Math.floor(py + (dy * i) / n + 0.4999))) ok = false;
        if (ok) seenFlag[idx(x, y)] = 1;
      }
  }
  for (let i = 0; i < V.length; i++) V[i] = (V[i] & 127) | (seenFlag[i] ? 128 : 0);

  const monsters = MONSTERS.map((m, i) => {
    const r = rooms[1 + (i % (rooms.length - 1))];
    const pts = [randIn(r), randIn(r), randIn(r)];
    const loop = [];
    for (let k = 0; k < pts.length; k++) {
      const a = pts[k];
      const b = pts[(k + 1) % pts.length];
      loop.push(...bfs(a[0], a[1], b[0], b[1]));
    }
    if (!loop.length) loop.push(pts[0]);
    return { ...m, loop, phase: Math.floor(R() * 50), rate: m.ch === 'B' ? 1.6 : m.ch === 'r' || m.ch === 'j' ? 0.9 : 0.5 };
  });

  // ---- encode the terminal into a canvas
  const c = makeCanvas(TW, TH);
  const g = c.getContext('2d');
  const img = g.createImageData(TW, TH);
  const put = (x, y, r, gg, b) => {
    const o = (y * TW + x) * 4;
    img.data[o] = r;
    img.data[o + 1] = gg;
    img.data[o + 2] = b;
    img.data[o + 3] = 255;
  };
  for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) put(x, y, 0, 0, 100); // blank text
  for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) put(x + MX, y + MY, T[idx(x, y)], V[idx(x, y)], 0);
  const text = (x, y, str, color = 0) => {
    for (let i = 0; i < str.length; i++) if (x + i < TW) put(x + i, y, gi(str[i]), color, 100);
  };
  const bar = (y, id, label) => {
    for (let i = 0; i < 19; i++) put(i, y, gi(' '), id, 200);
    const s = Math.floor((19 - label.length) / 2);
    for (let i = 0; i < label.length; i++) put(s + i, y, gi(label[i]), id, 200);
  };
  text(MX, 0, 'Welcome, adventurer, to the Dungeons of Doom!', 0);
  text(MX, 1, 'You hear the distant splash of water.', 1);
  text(3, 1, '-- Depth: 3 --', 1);
  text(1, 3, '@', 3);
  text(2, 3, ': Rogue', 0);
  bar(4, 0, 'Health');
  bar(5, 1, 'Nutrition');
  text(1, 6, 'Str: 12  Armor: 4', 1);
  MONSTERS.forEach((m, i) => {
    const y = 8 + i * 3;
    text(1, y, m.ch, m.color);
    text(2, y, ': ' + m.name, 0);
    bar(y + 1, 2 + i, m.state);
  });
  text(3, 28, 'Gold: 47', 3);
  text(MX + 1, 31, '[E]xplore   [Z] Rest   [S]earch   [I]nventory   [M]enu', 2);
  for (const k of [MX + 2, MX + 14, MX + 23, MX + 34, MX + 49]) put(k, 31, img.data[(31 * TW + k) * 4], 3, 100);
  g.putImageData(img, 0, 0);
  dungeon = { canvas: c, tour, start, monsters, torches };
  return dungeon;
}

// ------------------------------------------------------------------------------------------------

const SHARED_WGSL = /* wgsl */ `
fn glyphAt(idx: f32, luv: vec2f) -> f32 {
  let cellXY = vec2f(fmod(idx, 16.0), floor(idx / 16.0));
  let q = clamp(luv, vec2f(0.03), vec2f(0.97));
  return TEX(glyphs, (cellXY + q) / vec2f(16.0, 12.0)).r;
}
// 4 taps spread over the pixel footprint (fw = one screen pixel in cell units) = cheap anti-aliasing
fn glyphAA(idx: f32, luv: vec2f, fw: vec2f) -> f32 {
  let o = fw * 0.25;
  return 0.25 * (glyphAt(idx, luv + vec2f(-o.x, -o.y)) + glyphAt(idx, luv + vec2f(o.x, -o.y)) +
                 glyphAt(idx, luv + vec2f(-o.x, o.y)) + glyphAt(idx, luv + vec2f(o.x, o.y)));
}
fn rampGlyph(k: i32) -> f32 {
  let v = u.rampG[k / 4];
  let c = k % 4;
  var r = v.x;
  if (c == 1) { r = v.y; } else if (c == 2) { r = v.z; } else if (c == 3) { r = v.w; }
  return r;
}
fn rotX(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x, c * p.y - s * p.z, s * p.y + c * p.z); }
fn rotZ(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(c * p.x - s * p.y, s * p.x + c * p.y, p.z); }
fn donutSdf(p0: vec3f) -> f32 {
  let p = rotZ(rotX(p0, u.time * 0.9), u.time * 0.45);
  let q = vec2f(length(p.xz) - 1.0, p.y);
  return length(q) - 0.45;
}
fn tileAt(c: vec2i) -> vec4f { return floor(LOAD(dungeon, c) * 255.0 + 0.5); }
fn opaqueAt(c: vec2i) -> f32 {
  let t = tileAt(c);
  if (t.b > 0.5) { return 1.0; }
  if (t.r < 0.5 || t.r == 1.0 || t.r == 8.0) { return 1.0; }
  return 0.0;
}
// line of sight between two cell centers: step along the segment, any opaque cell in between blocks
fn losTo(a: vec2f, b: vec2f) -> f32 {
  let d = b - a;
  let n = max(abs(d.x), abs(d.y));
  for (var i = 1; i < 48; i++) {
    if (f32(i) >= n) { break; }
    let p = a + d * (f32(i) / n);
    if (opaqueAt(vec2i(floor(p + vec2f(0.5001, 0.4999)))) > 0.5) { return 0.0; }
  }
  return 1.0;
}
`;

const CELLS_PASS = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  let cs = vec2f(u.cell, u.cell * 1.5);
  let base = floor(px) * cs;
  if (ex == 0) {
    // average the source over the cell (4x4 taps) and fit a plane to the luminance -> gradient
    var acc = vec3f(0.0);
    var gx = 0.0;
    var gy = 0.0;
    for (var j = 0; j < 4; j++) {
      for (var i = 0; i < 4; i++) {
        let o = (vec2f(f32(i), f32(j)) + 0.5) / 4.0;
        let c = TEX(game, (base + o * cs) / u.resolution).rgb;
        acc += c;
        let l = luma(c);
        gx += l * (o.x - 0.5);
        gy += l * (o.y - 0.5);
      }
    }
    let avg = acc / 16.0;
    let grad = vec2f(gx, gy) / 1.25;                  // luminance change across one cell
    var code = 0.0;
    if (length(grad) > u.edgeThresh) {
      let g = grad / cs;                              // to pixel units so the angle is right on screen
      let tang = vec2f(-g.y, g.x);                     // the edge runs perpendicular to the gradient
      let ang = fmod(atan2(tang.y, tang.x) + PI, PI);  // 0..PI
      code = 1.0;                                      // '-'
      if (ang > PI * 0.125 && ang <= PI * 0.375) { code = 2.0; } // '\\'
      if (ang > PI * 0.375 && ang <= PI * 0.625) { code = 3.0; } // '|'
      if (ang > PI * 0.625 && ang <= PI * 0.875) { code = 4.0; } // '/'
    }
    return vec4f(avg, code);
  }
  if (ex == 1) {
    // donut.c: ray-march a spinning torus at the CENTER of this character cell
    let cc = base + 0.5 * cs;
    let p = (cc - 0.5 * u.resolution) / u.resolution.y;
    let ro = vec3f(0.0, 0.0, -4.2);
    let rd = normalize(vec3f(p.x, -p.y, 1.25));
    var t = 0.0;
    var hit = 0.0;
    for (var i = 0; i < 64; i++) {
      let d = donutSdf(ro + rd * t);
      if (d < 0.002) { hit = 1.0; break; }
      t += d;
      if (t > 9.0) { break; }
    }
    if (hit < 0.5) { return vec4f(0.0); }
    let pos = ro + rd * t;
    let e = 0.002;
    let n = normalize(vec3f(donutSdf(pos + vec3f(e, 0.0, 0.0)) - donutSdf(pos - vec3f(e, 0.0, 0.0)),
                            donutSdf(pos + vec3f(0.0, e, 0.0)) - donutSdf(pos - vec3f(0.0, e, 0.0)),
                            donutSdf(pos + vec3f(0.0, 0.0, e)) - donutSdf(pos - vec3f(0.0, 0.0, e))));
    let lightDir = normalize(vec3f(0.0, 1.0, -1.0));   // the same light as donut.c
    let lum = 0.08 + 0.92 * clamp(dot(n, lightDir), 0.0, 1.0);
    return vec4f(vec3f(lum), 0.0);
  }
  return vec4f(0.0);
}`;

const FOV_PASS = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  if (i32(u.example) != 3) { return vec4f(0.0); }
  let cell = floor(px);
  let t = tileAt(vec2i(cell));
  if (t.b > 0.5) { return vec4f(0.0); }
  var prevSeen = LOAD(fov, vec2i(cell)).a;
  if (u.frame < 0.5) { prevSeen = step(127.5, t.g); }   // high bit of G = "seen before the demo started"
  // the player's own light (warm white) with line of sight
  let pd = length((cell - u.player) * vec2f(1.0, 1.5));
  var vis = 0.0;
  if (pd < 26.0) { vis = losTo(u.player, cell); }
  var light = vec3f(1.0, 0.92, 0.75) * pow(clamp(1.0 - pd / u.light, 0.0, 1.0), 0.75) * 1.3;
  // wall torches: each one casts its own line-of-sight light
  for (var i = 0; i < 8; i++) {
    let tl = u.torches[i];
    if (tl.z < 0.5) { continue; }
    let d = length((cell - tl.xy) * vec2f(1.0, 1.5));
    if (d > 9.0) { continue; }
    let flick = 0.85 + 0.15 * sin(u.time * 9.0 + f32(i) * 2.1) * sin(u.time * 13.3 + f32(i));
    light += vec3f(1.0, 0.5, 0.18) * pow(clamp(1.0 - d / 9.0, 0.0, 1.0), 1.6) * flick * 1.4 * losTo(tl.xy, cell);
  }
  let seenNow = vis * step(0.03, dot(light, vec3f(0.33)));
  return vec4f(light * seenNow, max(prevSeen * u.memory, seenNow));
}`;

const IMAGE = /* wgsl */ `
fn edgeGlyph(code: f32) -> f32 {
  var g = 13.0;                          // '-'
  if (code > 1.5) { g = 60.0; }          // '\\'
  if (code > 2.5) { g = 92.0; }          // '|'
  if (code > 3.5) { g = 15.0; }          // '/'
  return g;
}

fn asciiFilter(px: vec2f, ex: i32) -> vec3f {
  let cs = vec2f(u.cell, u.cell * 1.5);
  let cid = floor(px / cs);
  let luv = (px - cid * cs) / cs;
  let cv = LOAD(cells, vec2i(cid));
  var lum = clamp((luma(cv.rgb) - 0.5) * u.contrast + 0.5, 0.0, 1.0);
  if (ex == 1) { lum = clamp(luma(cv.rgb) * u.contrast * 0.8, 0.0, 1.0); }
  let mode = i32(u.colorMode);
  if (mode == 3) { lum = 1.0 - lum; }
  let k = i32(min(floor(lum * u.rampLen), u.rampLen - 1.0));
  var gidx = rampGlyph(k);
  if (u.edges > 0.5 && ex == 0 && cv.a > 0.5) { gidx = edgeGlyph(cv.a); }
  let cov = glyphAA(gidx, luv, 1.0 / cs);

  var fg = vec3f(1.0);
  var bg = vec3f(0.0);
  if (mode == 0) {
    if (ex == 1) {
      fg = mix(vec3f(0.42, 0.25, 1.0), vec3f(1.0, 0.82, 0.45), smoothstep(0.1, 0.9, lum)) + vec3f(0.25) * smoothstep(0.85, 1.0, lum);
      bg = vec3f(0.02, 0.015, 0.05);
    } else {
      let mx = max(max(cv.r, cv.g), max(cv.b, 0.04));
      fg = mix(cv.rgb, cv.rgb / mx, 0.65);             // keep the hue, push to full brightness
      bg = cv.rgb * u.bgFill * 0.6;
    }
  } else if (mode == 1) {
    fg = vec3f(0.35, 1.0, 0.5);
    bg = vec3f(0.0, 0.035, 0.015) + vec3f(0.0, 0.25, 0.1) * lum * u.bgFill;
  } else if (mode == 2) {
    fg = vec3f(1.0, 0.7, 0.22);
    bg = vec3f(0.035, 0.02, 0.0) + vec3f(0.25, 0.14, 0.0) * lum * u.bgFill;
  } else {
    bg = vec3f(0.95, 0.93, 0.87) - vec3f(0.18, 0.2, 0.22) * (1.0 - lum) * u.bgFill;
    fg = vec3f(0.1, 0.09, 0.12);
  }
  var col = mix(bg, fg, cov);
  // compare lens: show the original image under the mouse
  if (ex == 0 && u.lens > 0.5 && u.mouse.w > 0.5) {
    let r = u.resolution.y * 0.13;
    let d = length(px - u.mouse.xy);
    let src = TEX(game, px / u.resolution).rgb;
    col = mix(col, src, 1.0 - smoothstep(r - 1.0, r, d));
    col = mix(col, vec3f(1.0), (1.0 - smoothstep(1.0, 2.0, abs(d - r))) * 0.9);
  }
  return col;
}

// one layer of digital rain; returns color
fn rainLayer(px: vec2f, cw: f32, seed: f32) -> vec3f {
  let cs = vec2f(cw, cw * 1.5);
  let cid = floor(px / cs);
  let luv = (px - cid * cs) / cs;
  let rows = u.resolution.y / cs.y;
  let t = u.time + 40.0;
  var bright = 0.0;
  var head = 0.0;
  var glow = 0.0;
  for (var k = 0; k < 2; k++) {
    let h = hash23(vec2f(cid.x * 1.7 + f32(k) * 31.0, seed));
    let speed = mix(6.0, 16.0, h.x) * u.speed;          // rows per second
    let len = u.trail * mix(0.5, 1.5, h.y);              // trail length in rows
    let period = rows + len + 8.0 + 30.0 * h.z;
    let headRow = fmod(t * speed + h.y * 200.0, period);
    let d = headRow - cid.y;                             // how many rows behind the head
    if (d >= 0.0 && d < len) {
      bright = max(bright, pow(1.0 - d / len, 1.6));
      if (d < 1.0) { head = 1.0; }
    }
    // soft glow around the head
    let hp = vec2f((cid.x + 0.5) * cs.x, (floor(headRow) + 0.5) * cs.y);
    glow += exp(-length((px - hp) / cs) * 1.6) * step(0.0, headRow) * 0.5;
  }
  // every cell flips to a random glyph at its own rate (heads flip fastest)
  let gh = hash21(cid + vec2f(seed, 3.0));
  let rate = mix(0.6, 5.0, gh * gh) + head * 12.0;
  let gidx = 128.0 + floor(hash21(cid * 1.31 + vec2f(floor(t * rate + gh * 10.0), seed)) * 64.0);
  let cov = glyphAA(gidx, luv, 1.0 / cs);
  var reveal = 1.0;
  if (u.reveal > 0.0) {
    let src = luma(TEX(game, (cid + 0.5) * cs / u.resolution).rgb);
    reveal = mix(1.0, 0.15 + 1.6 * smoothstep(0.25, 0.85, src), u.reveal);
  }
  let base = u.rainColor;
  var c = base * cov * bright * reveal * 1.15;
  c = mix(c, vec3f(0.85, 1.0, 0.9) * cov * 1.4, head);
  c += base * glow * 0.2;
  return c;
}

fn tcolor(id: f32) -> vec3f {
  var pal = array<vec3f, 12>(
    vec3f(0.92, 0.92, 0.9), vec3f(0.62, 0.62, 0.68), vec3f(0.42, 0.42, 0.5), vec3f(1.0, 0.85, 0.3),
    vec3f(1.0, 0.35, 0.3), vec3f(0.45, 0.92, 0.35), vec3f(0.4, 0.6, 1.0), vec3f(0.4, 0.95, 0.95),
    vec3f(1.0, 0.58, 0.2), vec3f(0.78, 0.45, 1.0), vec3f(0.82, 0.6, 0.38), vec3f(0.62, 0.78, 0.3));
  return pal[i32(clamp(id, 0.0, 11.0))];
}

fn roguelike(px: vec2f) -> vec3f {
  // fit 80 x 32 cells (+2 rows of margin top & bottom) into the canvas, centered
  let ch = min(u.resolution.y / 36.0, u.resolution.x / 81.0 * 1.5);
  let cs = vec2f(ch / 1.5, ch);
  let org = floor((u.resolution - vec2f(80.0, 32.0) * cs) * 0.5);
  let tc = floor((px - org) / cs);
  if (tc.x < 0.0 || tc.y < 0.0 || tc.x > 79.0 || tc.y > 31.0) { return vec3f(0.0); }
  let luv = (px - org - tc * cs) / cs;
  let fw = 1.0 / cs;
  let t = tileAt(vec2i(tc));
  let kind = floor(t.b / 100.0 + 0.5);
  var glyph = t.r;
  var fg = vec3f(0.0);
  var bg = vec3f(0.0);
  if (kind > 1.5) {
    // status bars: background fill + label
    let frac = (tc.x + luv.x) / 19.0;
    var fills = array<f32, 8>(0.0, 0.82, 1.0, 0.6, 0.9, 0.45, 0.75, 1.0);
    var fill = fills[i32(t.g)];
    var bc = vec3f(0.55, 0.12, 0.12);
    if (t.g < 0.5) { fill = u.hp; }
    else if (t.g < 1.5) { bc = vec3f(0.45, 0.4, 0.1); }
    else { bc = vec3f(0.18, 0.22, 0.35); }
    bg = bc * mix(0.35, 1.0, step(frac, fill));
    fg = vec3f(0.95);
  } else if (kind > 0.5) {
    fg = tcolor(t.g);
  } else {
    let ty = t.r;
    let vr = fmod(t.g, 128.0) / 127.0;
    let lv = LOAD(fov, vec2i(tc));
    var light = lv.rgb;
    let seen = lv.a;
    let visible = step(0.03, dot(light, vec3f(0.33)));
    if (u.lighting < 0.5) { light = vec3f(visible); }
    glyph = 0.0;
    if (ty == 1.0 || ty == 8.0) { glyph = 3.0; fg = vec3f(0.58, 0.52, 0.46) + vr * 0.08; bg = vec3f(0.24, 0.21, 0.19) + vr * 0.04; }
    if (ty == 2.0) { glyph = 14.0; fg = vec3f(0.62, 0.6, 0.66); bg = vec3f(0.16, 0.145, 0.17) + vr * 0.035; }
    if (ty == 3.0) { glyph = 14.0; fg = vec3f(0.5, 0.5, 0.55); bg = vec3f(0.1, 0.095, 0.11); }
    if (ty == 4.0) { glyph = 11.0; fg = vec3f(0.85, 0.58, 0.28); bg = vec3f(0.3, 0.17, 0.07); }
    if (ty == 5.0) {
      let w = sin(tc.x * 0.9 + u.time * 2.0 + sin(tc.y * 1.3 + u.time)) * 0.5 + 0.5;
      glyph = select(94.0, 123.0, w > 0.55);  // '~' or '≈'
      fg = mix(vec3f(0.35, 0.55, 1.0), vec3f(0.75, 0.9, 1.0), w * 0.6);
      bg = vec3f(0.04, 0.12, 0.38) + vec3f(0.02, 0.06, 0.12) * w;
    }
    if (ty == 6.0) { glyph = select(2.0, 12.0, vr > 0.5); fg = vec3f(0.3, 0.72, 0.24) + vec3f(0.1, 0.15, 0.0) * vr; bg = vec3f(0.05, 0.12, 0.05); }
    if (ty == 7.0) { glyph = 30.0; fg = vec3f(1.0); bg = vec3f(0.2, 0.2, 0.32); }
    if (ty == 9.0) {
      var items = array<f32, 7>(4.0, 1.0, 31.0, 9.0, 59.0, 5.0, 10.0);   // $ ! ? ) [ % *
      var icol = array<vec3f, 7>(vec3f(1.0, 0.85, 0.2), vec3f(1.0, 0.4, 0.8), vec3f(0.92, 0.9, 0.78), vec3f(0.75, 0.82, 0.95),
                                 vec3f(0.6, 0.72, 0.85), vec3f(0.85, 0.6, 0.3), vec3f(0.4, 1.0, 1.0));
      let it = i32(clamp(fmod(t.g, 128.0), 0.0, 6.0));
      glyph = items[it];
      fg = icol[it];
      bg = vec3f(0.11, 0.1, 0.12);
    }
    if (ty == 10.0) { glyph = 12.0; fg = vec3f(0.8, 0.78, 0.7); bg = vec3f(0.11, 0.1, 0.12); }
    if (ty == 8.0) {
      let fl = 0.8 + 0.2 * sin(u.time * 11.0 + tc.x) * sin(u.time * 7.0);
      glyph = 10.0;  // '*'
      fg = vec3f(1.0, 0.75, 0.3) * fl * 1.3;
      bg = vec3f(0.55, 0.2, 0.05) * fl;
      light = max(light, vec3f(visible));
    }
    // monsters are only drawn where you can see them
    for (var i = 0; i < 8; i++) {
      let m = u.mons[i];
      if (m.z > 0.5 && abs(m.x - tc.x) < 0.5 && abs(m.y - tc.y) < 0.5 && visible > 0.5) {
        glyph = m.z;
        fg = tcolor(m.w) * 1.15;
      }
    }
    var isPlayer = 0.0;
    if (abs(u.player.x - tc.x) < 0.5 && abs(u.player.y - tc.y) < 0.5) {
      glyph = 32.0;
      fg = vec3f(1.0, 0.95, 0.75);
      isPlayer = 1.0;
    }
    if (visible > 0.5 || isPlayer > 0.5) {
      let lt = min(light + vec3f(0.06), vec3f(1.6));
      fg = fg * lt + vec3f(isPlayer);
      bg = bg * lt;
    } else if (seen > 0.5) {
      // remembered but not currently visible: desaturated "memory" colors
      let m = luma(fg) * 0.55;
      fg = vec3f(0.35, 0.42, 0.75) * m;
      bg = vec3f(0.02, 0.025, 0.05) + luma(bg) * vec3f(0.08, 0.1, 0.2);
    } else {
      fg = vec3f(0.0);
      bg = vec3f(0.0);
    }
  }
  let cov = glyphAA(glyph, luv, fw);
  return mix(bg, fg, cov);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  if (ex == 2) {
    var c = vec3f(0.0, 0.012, 0.006);
    c += rainLayer(px, u.cell * 0.62, 7.0) * vec3f(0.32, 0.38, 0.4);   // far layer: smaller & dimmer
    c += rainLayer(px, u.cell, 1.0);
    let v = uv - 0.5;
    c *= 1.0 - dot(v, v) * 0.9;
    return vec4f(c, 1.0);
  }
  if (ex == 3) {
    let c = roguelike(px);
    return vec4f(c, 1.0);
  }
  return vec4f(asciiFilter(px, ex), 1.0);
}`;

export default shaderScene({
  interaction: 'Hover the ASCII filter: a lens shows the original image.',
  examples: [
    {
      id: 'filter',
      label: 'ASCII filter',
      kind: 'In a game',
      note: 'The platformer is re-drawn with characters. Each cell’s average brightness picks a glyph from a <i>ramp</i> (sparse <code>.</code> → dense <code>@</code>); where the cell contains a strong edge, a line glyph <code>/ \\ | -</code> follows its direction.',
      params: { cell: 9, rampSet: 'classic', contrast: 1.25 },
    },
    {
      id: 'donut',
      label: 'Spinning donut (donut.c)',
      kind: 'Abstract',
      note: 'A homage to Andy Sloane’s famous <code>donut.c</code>: a 3D torus is lit, then each character cell shows the brightness at its center with the ramp <code>.,-~:;=!*#$@</code>. Same pipeline as the filter — only the “source image” is different.',
      params: { cell: 11, rampSet: 'donut', contrast: 1.2 },
    },
    {
      id: 'matrix',
      label: 'Matrix rain',
      kind: 'Classic',
      note: 'Each column runs a couple of falling “drops”: a bright head and a fading trail. Glyphs flip at random rates. Everything is a pure function of <code>(column, row, time)</code> — no state, no particles.',
      params: { cell: 15 },
    },
    {
      id: 'rogue',
      label: 'Roguelike terminal',
      kind: 'In a game',
      note: 'A tiny Brogue-style dungeon: the map lives in an 80×32 <b>data texture</b> (one texel per character). A per-cell pass computes line-of-sight lighting from the @ and wall torches, and remembers explored tiles with a feedback buffer.',
    },
  ],
  controls: [
    { type: 'slider', key: 'cell', label: 'Cell width (px)', min: 5, max: 32, step: 1, value: 9, showFor: ['filter', 'donut', 'matrix'], help: 'Character size. Height is 1.5× the width, like a terminal font.' },
    {
      type: 'select', key: 'rampSet', label: 'Character ramp', value: 'classic', showFor: ['filter', 'donut'],
      options: [
        { value: 'classic', label: 'Classic  " .:-=+*#%@"' },
        { value: 'donut', label: 'donut.c  " .,-~:;=!*#$@"' },
        { value: 'blocks', label: 'CP437 blocks  " ░▒▓█"' },
        { value: 'measured', label: 'Measured: 16 glyphs sorted by ink' },
      ],
      help: 'Glyphs ordered from least to most ink. Brightness picks a position on the ramp.',
    },
    {
      type: 'select', key: 'colorMode', label: 'Color', value: 'color', showFor: ['filter', 'donut'],
      options: [
        { value: 'color', label: 'Full color' },
        { value: 'green', label: 'Green phosphor' },
        { value: 'amber', label: 'Amber terminal' },
        { value: 'paper', label: 'Ink on paper (inverted)' },
      ],
    },
    { type: 'slider', key: 'contrast', label: 'Contrast', min: 0.5, max: 3, step: 0.01, value: 1.25, showFor: ['filter', 'donut'], help: 'Stretches brightness before choosing a glyph — uses more of the ramp.' },
    { type: 'toggle', key: 'edges', label: 'Edge-aware glyphs  / \\ | -', value: true, showFor: ['filter'], help: 'Where a cell contains a strong edge, use a line character that follows it (like Acerola’s ASCII shader).' },
    { type: 'slider', key: 'edgeThresh', label: 'Edge threshold', min: 0.05, max: 1, step: 0.01, value: 0.32, showFor: ['filter'], help: 'Lower = more line glyphs.' },
    { type: 'slider', key: 'bgFill', label: 'Background tint', min: 0, max: 1, step: 0.01, value: 0.4, showFor: ['filter', 'donut'], help: 'Fill each cell’s background with a dim version of its color (many ASCII renderers do this).' },
    { type: 'toggle', key: 'lens', label: 'Compare lens under the mouse', value: true, showFor: ['filter'] },
    { type: 'slider', key: 'speed', label: 'Rain speed', min: 0.1, max: 3, step: 0.01, value: 1, showFor: ['matrix'] },
    { type: 'slider', key: 'trail', label: 'Trail length (rows)', min: 3, max: 40, step: 1, value: 16, showFor: ['matrix'] },
    { type: 'color', key: 'rainColor', label: 'Code color', value: '#2bff6a', showFor: ['matrix'] },
    { type: 'slider', key: 'reveal', label: 'Hidden image', min: 0, max: 1, step: 0.01, value: 0, showFor: ['matrix'], help: 'Modulate the rain by the game image — the code “draws” a picture.' },
    { type: 'slider', key: 'light', label: 'Torch radius (tiles)', min: 3, max: 16, step: 0.5, value: 8, showFor: ['rogue'] },
    { type: 'slider', key: 'walkSpeed', label: 'Turns per second', min: 1, max: 20, step: 1, value: 6, showFor: ['rogue'] },
    { type: 'toggle', key: 'lighting', label: 'Colored lighting (Brogue-style)', value: true, showFor: ['rogue'], help: 'Off = classic flat colors: everything in view is fully bright.' },
    { type: 'toggle', key: 'memory', label: 'Remember explored tiles', value: true, showFor: ['rogue'], help: 'A feedback buffer keeps “seen” flags: explored tiles stay on the map in dim blue.' },
  ],
  uniforms: {
    cell: 'f32', colorMode: 'f32', contrast: 'f32', edges: 'f32', edgeThresh: 'f32', bgFill: 'f32', lens: 'f32',
    speed: 'f32', trail: 'f32', rainColor: 'vec3f', reveal: 'f32',
    light: 'f32', walkSpeed: 'f32', lighting: 'f32', memory: 'f32', hp: 'f32',
    rampLen: 'f32', player: 'vec2f', rampG: 'array<vec4f, 4>', mons: 'array<vec4f, 8>', torches: 'array<vec4f, 8>',
  },
  include: [...new Set(['hash', 'noise', 'color', 'math', ...GAME_SCENE_INCLUDES])],
  textures: {
    glyphs: { source: async () => buildGlyphAtlas().canvas },
    dungeon: { source: async () => buildDungeon().canvas, filter: 'nearest' },
  },
  // WORKAROUND (core bug): input: 'game' binds the game target while rendering into it on WebGPU,
  // so the platformer image is rendered as an ordinary first pass named "game" instead.
  passes: [
    { name: 'game', format: 'rgba8unorm', code: GAME_SCENE_CODE },
    {
      name: 'cells',
      size: (p, ctx) => [Math.max(1, Math.ceil(ctx.width / p.cell)), Math.max(1, Math.ceil(ctx.height / (p.cell * 1.5)))],
      code: SHARED_WGSL + CELLS_PASS,
    },
    { name: 'fov', size: [TW, TH], code: SHARED_WGSL + FOV_PASS },
  ],
  bind(p, ctx) {
    const ramp = rampFor(p.rampSet);
    const rampG = new Array(16).fill(0);
    ramp.forEach((v, i) => (rampG[i] = v));
    const D = buildDungeon();
    const step = D.start + Math.floor(ctx.time * p.walkSpeed);
    const pl = D.tour[step % D.tour.length] || [0, 0];
    const mons = new Array(32).fill(0);
    D.monsters.forEach((m, i) => {
      const s = Math.floor(ctx.time * p.walkSpeed * m.rate) + m.phase;
      const q = m.loop[s % m.loop.length];
      mons.splice(i * 4, 4, q[0] + MX, q[1] + MY, gi(m.ch), m.color);
    });
    const torches = new Array(32).fill(0);
    D.torches.slice(0, 8).forEach((t, i) => torches.splice(i * 4, 4, t[0], t[1], 1, 0));
    return {
      rampG,
      rampLen: ramp.length,
      player: [pl[0] + MX, pl[1] + MY],
      mons,
      torches,
      hp: 0.62 + 0.25 * Math.sin(ctx.time * 0.4),
    };
  },
  code: SHARED_WGSL + IMAGE,
  about: {
    summary: 'Render the world with characters: brightness picks a glyph, edges pick a line character, and a whole game can live in a grid of text cells.',
    what: `<p>Four uses of the same idea — <b>a grid of character cells, each showing one glyph from a font texture</b>:
      a live ASCII filter over the platformer, the famous spinning <code>donut.c</code>, Matrix-style digital rain,
      and a little Brogue-like roguelike with line-of-sight lighting.</p>`,
    how: `<ol>
      <li><b>Glyph atlas.</b> At start-up, Canvas2D paints every printable ASCII character (plus hand-drawn CP437 shade/box glyphs and mirrored katakana) into a 16×12 grid texture.
        We even <i>measure</i> how much ink each glyph has, which gives the auto-sorted “Measured” ramp.</li>
      <li><b>One pixel per cell.</b> A small render pass with exactly one pixel per character cell averages the source image over that cell (16 taps)
        and fits a plane to the luminance — the plane’s slope is the edge direction. For donut.c this pass ray-marches the torus instead.</li>
      <li><b>Ramp lookup.</b> The full-screen pass finds which cell a pixel belongs to, reads that cell’s brightness, and maps it to a glyph index:
        <code>glyph = ramp[floor(brightness × rampLength)]</code>.</li>
      <li><b>Edge glyphs.</b> If the cell’s gradient is strong, the angle of the edge (perpendicular to the gradient) chooses <code>- \\ | /</code>.</li>
      <li><b>Copy the glyph.</b> The pixel’s position inside its cell (0..1) becomes a UV inside that glyph’s atlas cell. Four taps over the pixel footprint anti-alias small text.</li>
      <li><b>Matrix rain</b> is pure math per column: <code>head = fmod(time × speed + random, period)</code>; cells above the head fade out, glyphs flip at random rates.</li>
      <li><b>Roguelike</b>: the terminal is an 80×32 data texture (tile id, text, colors). A pass at 80×32 marches a line from each cell to the @ and to each torch
        to compute visibility and colored light, and keeps a “seen” bit from its own previous frame (feedback) — that’s fog-of-war memory.</li>
    </ol>`,
    uses: [
      { title: 'Roguelikes', text: 'Rogue, NetHack, Brogue, Dwarf Fortress, Caves of Qud and Cogmind are drawn as character grids — GPU glyph rendering lets them add lighting, color and smooth animation.' },
      { title: 'Stylised games', text: 'Stone Story RPG is fully animated ASCII; many jam games use an ASCII post-filter as an instant distinctive look.' },
      { title: 'Hacking & sci-fi UI', text: 'Matrix rain, terminals and “decrypting” text for hacking minigames, title screens and transitions.' },
      { title: 'Debug views', text: 'Show tile ids, pathfinding costs or AI state directly on top of a level as glyphs.' },
    ],
    try: [
      'On <b>ASCII filter</b>, drop <i>Cell width</i> to 5 — at small sizes the eye blends glyphs back into a picture. Push it to 24 and it becomes typography.',
      'Switch the ramp to <b>CP437 blocks</b>: the image turns into chunky DOS-style shading.',
      'Turn <i>Edge-aware glyphs</i> off and on — outlines suddenly read as drawn lines.',
      'On <b>Matrix rain</b>, raise <i>Hidden image</i>: the falling code starts drawing the platformer.',
      'On <b>Roguelike</b>, turn off <i>Colored lighting</i> to see the classic flat NetHack look, and off <i>Remember</i> to lose your map memory.',
    ],
    ask: [
      'an ASCII post-processing filter with edge-aware glyphs',
      'render text from a glyph atlas texture in a shader',
      'Matrix digital rain background',
      'a roguelike terminal renderer with colored lighting and field of view',
      'fog-of-war memory with a feedback buffer',
      'brightness-to-character ramp sorted by glyph coverage',
    ],
    perf: `<p>Very cheap. The expensive part (averaging the image, edge detection, ray marching, line-of-sight) runs once per <i>cell</i>, not per pixel —
      at a 9 px cell that is ~1/130th of the pixels. The full-screen pass only does one cell lookup and four texture taps into the font.
      The roguelike’s line-of-sight pass is 2,560 cells × 8 lights × ≤48 steps: nothing for a GPU.</p>`,
    api: `<p>Works identically on WebGPU and WebGL2 — it’s fragment shaders and textures only. The trick of a low-resolution “one pixel per cell” pass
      is the fragment-shader equivalent of a compute shader writing one value per cell; on WebGPU you could also do it with a compute pass and a storage buffer.</p>`,
    code: [
      {
        title: 'Brightness → glyph → pixel (final pass)',
        lang: 'wgsl',
        src: `let cs = vec2f(u.cell, u.cell * 1.5);          // cell size in pixels
let cid = floor(px / cs);                        // which character cell
let luv = (px - cid * cs) / cs;                  // 0..1 inside the cell
let cv = LOAD(cells, vec2i(cid));                // averaged color + edge code
var lum = clamp((luma(cv.rgb) - 0.5) * u.contrast + 0.5, 0.0, 1.0);
let k = i32(min(floor(lum * u.rampLen), u.rampLen - 1.0));
var gidx = rampGlyph(k);                         // e.g. " .:-=+*#%@"[k]
if (u.edges > 0.5 && cv.a > 0.5) { gidx = edgeGlyph(cv.a); }  // - \\ | /
let cov = glyphAA(gidx, luv, 1.0 / cs);          // ink coverage from the atlas
col = mix(bg, fg, cov);`,
      },
      {
        title: 'Edge direction from a plane fit (cells pass)',
        lang: 'wgsl',
        src: `for (var j = 0; j < 4; j++) { for (var i = 0; i < 4; i++) {
  let o = (vec2f(f32(i), f32(j)) + 0.5) / 4.0;
  let c = TEX(game, (base + o * cs) / u.resolution).rgb;
  acc += c;
  gx += luma(c) * (o.x - 0.5);   // least-squares slope in x
  gy += luma(c) * (o.y - 0.5);   // ... and in y
} }
let tang = vec2f(-gy, gx);       // the edge is perpendicular to the gradient
let ang = fmod(atan2(tang.y, tang.x) + PI, PI);   // bucket into - \\ | /`,
      },
      {
        title: 'Matrix rain: one column',
        lang: 'wgsl',
        src: `let h = hash23(vec2f(cid.x, seed));               // per-column randomness
let speed = mix(6.0, 16.0, h.x) * u.speed;         // rows per second
let headRow = fmod(t * speed + h.y * 200.0, period);
let d = headRow - cid.y;                           // rows behind the head
if (d >= 0.0 && d < len) { bright = pow(1.0 - d / len, 1.6); }`,
      },
    ],
    links: [
      { title: 'donut.c — Andy Sloane', url: 'https://www.a1k0n.net/2011/07/20/donut-math.html', note: 'the math behind the spinning ASCII donut' },
      { title: 'Character ramps — Paul Bourke', url: 'http://paulbourke.net/dataformats/asciiart/', note: 'the classic grey-scale character sequences' },
    ],
  },
});

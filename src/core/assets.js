// Procedural game assets — no image files needed.
// Pixel-art sprites & tiles are painted with code at startup (Sweetie-16 palette + a few extras),
// auto-outlined, packed into one atlas with padding, and a matching NORMAL MAP atlas is generated
// so lighting scenes can do per-pixel normal-mapped lighting.
//
//   const atlas = await getAtlas();
//   const tex = gpu.textureFromImage(atlas.canvas);       // or kit.textureFromImage for WebGL2
//   const ntex = gpu.textureFromImage(atlas.normalCanvas);
//   sprites.draw(x, y, 64, 64, { uv: atlas.uv('hero_run_2') });
//   atlas.anim('hero_run') -> ['hero_run_0', ..., 'hero_run_3']
//   atlas.names() -> every frame name

export const PAL = {
  black: '#1a1c2c',
  plum: '#5d275d',
  red: '#b13e53',
  orange: '#ef7d57',
  yellow: '#ffcd75',
  lime: '#a7f070',
  green: '#38b764',
  teal: '#257179',
  navy: '#29366f',
  blue: '#3b5dc9',
  sky: '#41a6f6',
  cyan: '#73eff7',
  white: '#f4f4f4',
  silver: '#94b0c2',
  slate: '#566c86',
  gunmetal: '#333c57',
  // extras
  brown: '#8f563b',
  darkbrown: '#5b3a29',
  skin: '#f2c3a0',
  gold: '#f7c64b',
  darkgold: '#c28b1f',
  pink: '#f59bb5',
  dirt: '#9a6a43',
  darkdirt: '#6b4628',
};

/** Classic palettes, handy for palette-swap / quantization scenes. Colors as '#rrggbb'. */
export const PALETTES = {
  sweetie16: Object.values(PAL).slice(0, 16),
  gameboy: ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'],
  pico8: ['#000000', '#1d2b53', '#7e2553', '#008751', '#ab5236', '#5f574f', '#c2c3c7', '#fff1e8', '#ff004d', '#ffa300', '#ffec27', '#00e436', '#29adff', '#83769c', '#ff77a8', '#ffccaa'],
  nes: ['#000000', '#fcfcfc', '#f8f8f8', '#bcbcbc', '#7c7c7c', '#a4e4fc', '#3cbcfc', '#0078f8', '#0000fc', '#b8b8f8', '#6888fc', '#0058f8', '#0000bc', '#d8b8f8', '#9878f8', '#6844fc', '#4428bc', '#f8b8f8', '#f878f8', '#d800cc', '#940084', '#f8a4c0', '#f85898', '#e40058', '#a80020', '#f0d0b0', '#f87858', '#f83800', '#a81000', '#fce0a8', '#fca044', '#e45c10', '#881400', '#f8d878', '#f8b800', '#ac7c00', '#503000', '#d8f878', '#b8f818', '#00b800', '#007800', '#b8f8b8', '#58d854', '#00a800', '#006800', '#b8f8d8', '#58f898', '#00a844', '#005800', '#00fcfc', '#00e8d8', '#008888', '#004058', '#f8d8f8', '#787878'],
  cga: ['#000000', '#55ffff', '#ff55ff', '#ffffff'],
  mono: ['#000000', '#ffffff'],
  obra: ['#333319', '#e5ffff'],
  vaporwave: ['#ff71ce', '#01cdfe', '#05ffa1', '#b967ff', '#fffb96', '#2d1b4e'],
  sepia: ['#2b1d0e', '#5e4426', '#8f6a3e', '#c19a63', '#efd9a7'],
};

export function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined' && !globalThis.document) return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Deterministic RNG (mulberry32). */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hexRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function shadeHex(h, k) {
  const [r, g, b] = hexRgb(h);
  const f = (v) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}

/** Pixel painter on a small canvas. */
function painter(w, h) {
  const c = makeCanvas(w, h);
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  const api = {
    c,
    g,
    w,
    h,
    px(x, y, col) {
      g.fillStyle = col;
      g.fillRect(x | 0, y | 0, 1, 1);
    },
    rect(x, y, rw, rh, col) {
      g.fillStyle = col;
      g.fillRect(x | 0, y | 0, rw | 0, rh | 0);
    },
    ellipse(cx, cy, rx, ry, col) {
      g.fillStyle = col;
      for (let y = Math.floor(cy - ry); y <= Math.ceil(cy + ry); y++)
        for (let x = Math.floor(cx - rx); x <= Math.ceil(cx + rx); x++) {
          const dx = (x + 0.5 - cx) / rx;
          const dy = (y + 0.5 - cy) / ry;
          if (dx * dx + dy * dy <= 1) g.fillRect(x, y, 1, 1);
        }
    },
    line(x0, y0, x1, y1, col) {
      g.fillStyle = col;
      x0 |= 0;
      y0 |= 0;
      x1 |= 0;
      y1 |= 0;
      const dx = Math.abs(x1 - x0);
      const dy = -Math.abs(y1 - y0);
      const sx = x0 < x1 ? 1 : -1;
      const sy = y0 < y1 ? 1 : -1;
      let err = dx + dy;
      for (;;) {
        g.fillRect(x0, y0, 1, 1);
        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
          err += dy;
          x0 += sx;
        }
        if (e2 <= dx) {
          err += dx;
          y0 += sy;
        }
      }
    },
    poly(points, col) {
      g.fillStyle = col;
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          let inside = false;
          const px = x + 0.5;
          const py = y + 0.5;
          for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
            const [xi, yi] = points[i];
            const [xj, yj] = points[j];
            if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
          }
          if (inside) g.fillRect(x, y, 1, 1);
        }
    },
    bitmap(rows, map, ox = 0, oy = 0) {
      rows.forEach((row, y) =>
        [...row].forEach((ch, x) => {
          if (map[ch]) api.px(ox + x, oy + y, map[ch]);
        }),
      );
    },
  };
  return api;
}

/** Add a 1px outline around opaque pixels. */
function outline(c, col = PAL.black) {
  const g = c.getContext('2d');
  const { width: w, height: h } = c;
  const img = g.getImageData(0, 0, w, h);
  const a = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : img.data[(y * w + x) * 4 + 3]);
  const [r, gg, b] = hexRgb(col);
  const out = new ImageData(new Uint8ClampedArray(img.data), w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (a(x, y) > 0) continue;
      if (a(x - 1, y) || a(x + 1, y) || a(x, y - 1) || a(x, y + 1)) {
        const i = (y * w + x) * 4;
        out.data[i] = r;
        out.data[i + 1] = gg;
        out.data[i + 2] = b;
        out.data[i + 3] = 255;
      }
    }
  g.putImageData(out, 0, 0);
  return c;
}

// ------------------------------------------------------------------------------- characters

function hero({ leg = 0, bob = 0, arm = 0, jump = false, hurt = false } = {}) {
  const p = painter(16, 16);
  const y0 = 1 + bob;
  const armor = hurt ? PAL.white : PAL.silver;
  const tunic = hurt ? PAL.white : PAL.blue;
  // plume
  p.rect(9, y0, 2, 1, PAL.red);
  p.rect(10, y0 + 1, 1, 1, PAL.red);
  // helmet
  p.rect(6, y0 + 1, 4, 1, armor);
  p.rect(5, y0 + 2, 6, 4, armor);
  p.rect(5, y0 + 2, 1, 4, PAL.slate);
  p.rect(7, y0 + 3, 4, 1, PAL.gunmetal);
  p.px(9, y0 + 3, PAL.cyan);
  p.rect(6, y0 + 5, 5, 1, PAL.slate);
  // body
  p.rect(5, y0 + 6, 6, 4, tunic);
  p.rect(5, y0 + 6, 1, 4, PAL.navy);
  p.rect(5, y0 + 9, 6, 1, PAL.brown);
  p.px(8, y0 + 9, PAL.gold);
  // arms
  p.rect(4, y0 + 6 + arm, 1, 3, armor);
  p.rect(11, y0 + 6 - arm, 1, 3, armor);
  p.px(11, y0 + 8 - arm, PAL.skin);
  // legs
  const lx = [6, 8];
  const ly = y0 + 10;
  if (jump) {
    p.rect(5, ly, 2, 2, PAL.navy);
    p.rect(9, ly, 2, 2, PAL.navy);
    p.rect(5, ly + 2, 2, 1, PAL.darkbrown);
    p.rect(9, ly + 2, 2, 1, PAL.darkbrown);
  } else {
    const off = [
      [-1, 1],
      [0, 0],
      [1, -1],
      [0, 0],
    ][leg];
    const lift = [
      [0, 1],
      [0, 0],
      [1, 0],
      [0, 0],
    ][leg];
    for (let i = 0; i < 2; i++) {
      const x = lx[i] + off[i];
      const hgt = 4 - bob - lift[i];
      p.rect(x, ly, 2, hgt - 1, PAL.navy);
      p.rect(x, ly + hgt - 1, 2, 1, PAL.darkbrown);
    }
  }
  return outline(p.c);
}

function slime(squash) {
  const p = painter(16, 16);
  const rx = 6 + squash;
  const ry = 5 - squash;
  const cy = 15 - ry;
  p.ellipse(8, cy, rx, ry, PAL.green);
  p.ellipse(8, cy + ry * 0.45, rx * 0.85, ry * 0.5, PAL.teal);
  p.ellipse(8, cy, rx - 1, ry - 1.2, PAL.green);
  p.px(5, cy - ry + 2, PAL.white);
  p.px(6, cy - ry + 1, PAL.lime);
  p.rect(6, cy - 1, 1, 2, PAL.black);
  p.rect(10, cy - 1, 1, 2, PAL.black);
  return outline(p.c);
}

function bunny(frame) {
  const p = painter(16, 16);
  const b = frame ? 1 : 0;
  p.rect(5, 1 + b, 2, 5, PAL.white);
  p.rect(9, 1 + b, 2, 5, PAL.white);
  p.rect(5, 2 + b, 1, 3, PAL.pink);
  p.rect(10, 2 + b, 1, 3, PAL.pink);
  p.ellipse(8, 9 + b, 4.5, 3.5, PAL.white);
  p.ellipse(8, 13, 5, 3, PAL.white);
  p.px(6, 9 + b, PAL.black);
  p.px(10, 9 + b, PAL.black);
  p.px(8, 10 + b, PAL.pink);
  p.rect(3, 13, 2, 2, PAL.silver);
  return outline(p.c);
}

function bat(frame) {
  const p = painter(16, 16);
  const up = frame === 0;
  p.ellipse(8, 8, 3, 3, PAL.plum);
  if (up) {
    p.poly([[5, 8], [0, 3], [2, 9]], PAL.plum);
    p.poly([[11, 8], [16, 3], [14, 9]], PAL.plum);
  } else {
    p.poly([[5, 8], [0, 12], [3, 9]], PAL.plum);
    p.poly([[11, 8], [16, 12], [13, 9]], PAL.plum);
  }
  p.px(7, 7, PAL.yellow);
  p.px(9, 7, PAL.yellow);
  p.px(6, 5, PAL.plum);
  p.px(10, 5, PAL.plum);
  return outline(p.c);
}

// ----------------------------------------------------------------------------------- items

function coin(i, n) {
  const p = painter(16, 16);
  const k = Math.abs(Math.cos((i / n) * Math.PI));
  const rx = Math.max(1, 5.5 * k);
  p.ellipse(8, 8, rx, 6, PAL.darkgold);
  if (rx > 1.5) p.ellipse(8, 8, rx - 1, 5, PAL.gold);
  if (rx > 3) p.rect(8, 5, 1, 6, PAL.darkgold);
  if (rx > 2) p.px(Math.round(8 - rx * 0.4), 5, PAL.white);
  return outline(p.c);
}

function gem() {
  const p = painter(16, 16);
  p.poly([[8, 2], [14, 7], [8, 14], [2, 7]], PAL.blue);
  p.poly([[8, 2], [14, 7], [8, 7]], PAL.sky);
  p.poly([[8, 2], [2, 7], [8, 7]], PAL.cyan);
  p.px(6, 5, PAL.white);
  return outline(p.c);
}

function heart() {
  const p = painter(16, 16);
  p.bitmap(['.rr...rr.', 'rrrr.rrrr', 'rwrrrrrrr', 'rrrrrrrrr', '.rrrrrrr.', '..rrrrr..', '...rrr...', '....r....'], { r: PAL.red, w: PAL.white }, 3, 4);
  return outline(p.c);
}

function potion() {
  const p = painter(16, 16);
  p.rect(7, 2, 2, 2, PAL.brown);
  p.rect(7, 4, 2, 2, PAL.silver);
  p.ellipse(8, 10, 4.5, 4.5, PAL.silver);
  p.ellipse(8, 10.5, 3.8, 3.5, PAL.red);
  p.px(6, 9, PAL.white);
  return outline(p.c);
}

function key() {
  const p = painter(16, 16);
  p.ellipse(5, 8, 3, 3, PAL.gold);
  p.ellipse(5, 8, 1.2, 1.2, 'rgba(0,0,0,0)');
  p.g.clearRect(4, 7, 2, 2);
  p.rect(7, 7, 7, 2, PAL.gold);
  p.rect(11, 9, 1, 2, PAL.gold);
  p.rect(13, 9, 1, 2, PAL.gold);
  return outline(p.c);
}

function chest() {
  const p = painter(16, 16);
  p.rect(2, 5, 12, 9, PAL.brown);
  p.rect(2, 5, 12, 3, PAL.darkbrown);
  p.rect(2, 8, 12, 1, PAL.gold);
  p.rect(2, 5, 1, 9, PAL.gold);
  p.rect(13, 5, 1, 9, PAL.gold);
  p.rect(7, 8, 2, 3, PAL.yellow);
  return outline(p.c);
}

function crate() {
  const p = painter(16, 16);
  p.rect(1, 1, 14, 14, PAL.brown);
  p.rect(1, 1, 14, 2, PAL.dirt);
  p.rect(1, 13, 14, 2, PAL.darkbrown);
  p.line(3, 3, 12, 12, PAL.darkbrown);
  p.line(12, 3, 3, 12, PAL.darkbrown);
  p.rect(1, 1, 2, 14, PAL.dirt);
  p.rect(13, 1, 2, 14, PAL.darkbrown);
  return outline(p.c);
}

function mushroom() {
  const p = painter(16, 16);
  p.rect(6, 9, 4, 5, PAL.white);
  p.ellipse(8, 8, 6, 4.5, PAL.red);
  p.g.clearRect(0, 9, 16, 1);
  p.rect(6, 9, 4, 5, PAL.white);
  p.px(5, 6, PAL.white);
  p.px(10, 5, PAL.white);
  p.px(8, 7, PAL.white);
  p.px(7, 11, PAL.black);
  p.px(9, 11, PAL.black);
  return outline(p.c);
}

function star() {
  const p = painter(16, 16);
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const r = i % 2 ? 3 : 7;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    pts.push([8 + r * Math.cos(a), 8.5 + r * Math.sin(a)]);
  }
  p.poly(pts, PAL.yellow);
  p.px(7, 6, PAL.white);
  return outline(p.c);
}

function torch(i) {
  const p = painter(16, 16);
  p.rect(7, 9, 2, 6, PAL.brown);
  p.rect(6, 8, 4, 2, PAL.slate);
  const h = [4, 5, 3][i];
  p.ellipse(8, 7 - h / 2, 2.5, h / 1.4 + 1, PAL.orange);
  p.ellipse(8, 7 - h / 2 + 1, 1.4, h / 2.2, PAL.yellow);
  p.px(8 + (i === 1 ? -1 : 1), 1 + (3 - h / 2), PAL.orange);
  return outline(p.c, PAL.darkbrown);
}

function bomb() {
  const p = painter(16, 16);
  p.ellipse(8, 10, 5, 5, PAL.gunmetal);
  p.ellipse(7, 9, 3, 3, PAL.slate);
  p.px(6, 7, PAL.silver);
  p.rect(8, 3, 1, 3, PAL.brown);
  p.px(9, 2, PAL.yellow);
  return outline(p.c);
}

function sword() {
  const p = painter(16, 16);
  p.line(4, 12, 12, 4, PAL.silver);
  p.line(5, 12, 12, 5, PAL.white);
  p.line(2, 11, 5, 14, PAL.gold);
  p.line(2, 14, 4, 12, PAL.brown);
  return outline(p.c);
}

function shield() {
  const p = painter(16, 16);
  p.poly([[3, 2], [13, 2], [13, 8], [8, 14], [3, 8]], PAL.silver);
  p.poly([[4, 3], [12, 3], [12, 8], [8, 12.5], [4, 8]], PAL.blue);
  p.rect(7, 4, 2, 7, PAL.yellow);
  p.rect(5, 6, 6, 2, PAL.yellow);
  return outline(p.c);
}

// ------------------------------------------------------------------------------------ tiles

function noiseTile(seed, base, dark, light, density = 0.25) {
  const p = painter(16, 16);
  const r = rng(seed);
  p.rect(0, 0, 16, 16, base);
  for (let y = 0; y < 16; y++)
    for (let x = 0; x < 16; x++) {
      const v = r();
      if (v < density * 0.6) p.px(x, y, dark);
      else if (v > 1 - density * 0.3) p.px(x, y, light);
    }
  return p;
}

function tileGrass() {
  const p = noiseTile(11, PAL.dirt, PAL.darkdirt, '#b07d55');
  const r = rng(12);
  for (let x = 0; x < 16; x++) {
    const h = 3 + Math.floor(r() * 3);
    p.rect(x, 0, 1, h, PAL.green);
    p.px(x, h, '#2a8a4f');
    if (r() > 0.6) p.px(x, Math.max(0, h - 2), PAL.lime);
  }
  return p.c;
}
const tileDirt = () => noiseTile(21, PAL.dirt, PAL.darkdirt, '#b07d55').c;
const tileSand = () => noiseTile(31, '#e8c98a', '#c9a76a', '#fbe3b0').c;
const tileLeaves = () => noiseTile(41, PAL.green, PAL.teal, PAL.lime, 0.4).c;

function tileStone() {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, PAL.gunmetal);
  const stones = [
    [0, 0, 7, 5], [8, 0, 8, 5], [0, 6, 4, 4], [5, 6, 6, 4], [12, 6, 4, 4], [0, 11, 9, 5], [10, 11, 6, 5],
  ];
  const r = rng(51);
  for (const [x, y, w, h] of stones) {
    const base = r() > 0.5 ? PAL.slate : '#6e7f96';
    p.rect(x + 1, y, w - 1, h, base);
    p.rect(x + 1, y, w - 1, 1, PAL.silver);
    p.rect(x + w - 1, y, 1, h, PAL.gunmetal);
  }
  return p.c;
}

function tileBrick() {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, '#d8c3a5');
  const r = rng(61);
  for (let row = 0; row < 4; row++) {
    const off = row % 2 ? 4 : 0;
    for (let i = -1; i < 3; i++) {
      const x = i * 8 + off;
      const col = r() > 0.5 ? PAL.red : '#c44d4d';
      p.rect(x + 1, row * 4 + 1, 7, 3, col);
      p.rect(x + 1, row * 4 + 1, 7, 1, '#e07070');
    }
  }
  return p.c;
}

function tileWood() {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, PAL.brown);
  const r = rng(71);
  for (let x = 0; x < 16; x += 4) {
    p.rect(x, 0, 1, 16, PAL.darkbrown);
    for (let y = 0; y < 16; y++) if (r() > 0.8) p.px(x + 1 + Math.floor(r() * 3), y, PAL.dirt);
  }
  p.px(2, 3, PAL.darkbrown);
  p.px(10, 11, PAL.darkbrown);
  return p.c;
}

function tileWater(i) {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, PAL.blue);
  for (let y = 0; y < 16; y += 4) {
    for (let x = 0; x < 16; x++) {
      const yy = y + Math.round(Math.sin(((x + i * 4) / 16) * Math.PI * 2) * 1.2) + 2;
      if (((x + i * 2 + y) & 7) < 4) p.px(x, ((yy % 16) + 16) % 16, PAL.sky);
    }
  }
  p.px((3 + i * 4) % 16, 2, PAL.cyan);
  p.px((11 + i * 4) % 16, 10, PAL.cyan);
  return p.c;
}

function tileLava(i) {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, '#d8401f');
  const r = rng(81);
  for (let k = 0; k < 6; k++) {
    const x = r() * 16;
    const y = (r() * 16 - i * 4 + 64) % 16;
    p.ellipse(x, y, 1.5 + r() * 2, 1 + r() * 1.5, PAL.orange);
    p.px(x, y, PAL.yellow);
  }
  return p.c;
}

function tileIce() {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, '#a8e4f7');
  p.line(2, 13, 13, 2, PAL.white);
  p.line(5, 15, 15, 5, '#dff7ff');
  p.rect(0, 15, 16, 1, PAL.sky);
  p.rect(15, 0, 1, 16, PAL.sky);
  return p.c;
}

function tileMetal() {
  const p = painter(16, 16);
  p.rect(0, 0, 16, 16, PAL.slate);
  p.rect(0, 0, 16, 1, PAL.silver);
  p.rect(0, 0, 1, 16, PAL.silver);
  p.rect(15, 0, 1, 16, PAL.gunmetal);
  p.rect(0, 15, 16, 1, PAL.gunmetal);
  for (const [x, y] of [[3, 3], [12, 3], [3, 12], [12, 12]]) {
    p.px(x, y, PAL.white);
    p.px(x + 1, y + 1, PAL.gunmetal);
  }
  return p.c;
}

// --------------------------------------------------------------------------------- scenery

function tree() {
  const p = painter(32, 48);
  p.rect(14, 28, 5, 19, PAL.brown);
  p.rect(14, 28, 2, 19, PAL.darkbrown);
  p.rect(12, 45, 9, 2, PAL.darkbrown);
  const blobs = [[16, 18, 12, 10], [9, 24, 7, 6], [23, 24, 7, 6], [16, 10, 8, 7]];
  for (const [x, y, rx, ry] of blobs) p.ellipse(x, y, rx, ry, PAL.teal);
  for (const [x, y, rx, ry] of blobs) p.ellipse(x - 1, y - 1, rx - 1.5, ry - 1.5, PAL.green);
  p.ellipse(13, 9, 3, 2, PAL.lime);
  p.ellipse(9, 20, 2, 2, PAL.lime);
  return outline(p.c);
}

function bush() {
  const p = painter(16, 16);
  p.ellipse(5, 11, 4.5, 4, PAL.teal);
  p.ellipse(11, 11, 4.5, 4, PAL.teal);
  p.ellipse(8, 8, 5, 5, PAL.green);
  p.px(6, 6, PAL.lime);
  p.px(10, 8, PAL.lime);
  p.rect(1, 14, 14, 2, PAL.teal);
  return outline(p.c);
}

function flower() {
  const p = painter(16, 16);
  p.rect(7, 8, 1, 7, PAL.green);
  p.px(6, 11, PAL.green);
  p.px(8, 12, PAL.green);
  p.ellipse(7.5, 6, 3, 3, PAL.pink);
  p.px(7, 5, PAL.yellow);
  p.px(8, 6, PAL.yellow);
  return outline(p.c);
}

function cloud() {
  const p = painter(32, 16);
  p.ellipse(10, 10, 7, 5, PAL.silver);
  p.ellipse(20, 9, 9, 6, PAL.silver);
  p.ellipse(15, 6, 6, 5, PAL.silver);
  p.ellipse(10, 9, 6, 4, PAL.white);
  p.ellipse(19, 8, 8, 5, PAL.white);
  p.ellipse(15, 5, 5, 4, PAL.white);
  return p.c;
}

function rock() {
  const p = painter(16, 16);
  p.poly([[2, 15], [3, 8], [7, 4], [12, 5], [15, 11], [14, 15]], PAL.slate);
  p.poly([[4, 9], [7, 5], [11, 6], [8, 9]], PAL.silver);
  return outline(p.c);
}

// ------------------------------------------------------------------------------ particles/UI

function softCircle(size = 32) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.35, 'rgba(255,255,255,0.6)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  return c;
}

function sparkTex(size = 32) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const grd = g.createLinearGradient(0, 0, size, 0);
  grd.addColorStop(0, 'rgba(255,255,255,0)');
  grd.addColorStop(0.7, 'rgba(255,255,255,0.9)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.beginPath();
  g.ellipse(size / 2, size / 2, size / 2, size / 10, 0, 0, Math.PI * 2);
  g.fill();
  return c;
}

function starTex(size = 32) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const s = size / 2;
  const grd = g.createRadialGradient(s, s, 0, s, s, s);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.2, 'rgba(255,255,255,0.5)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.beginPath();
  for (let i = 0; i < 8; i++) {
    const r = i % 2 ? s * 0.18 : s;
    const a = (i * Math.PI) / 4;
    g.lineTo(s + r * Math.cos(a), s + r * Math.sin(a));
  }
  g.closePath();
  g.fill();
  return c;
}

function smokeTex(size = 32) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const r = rng(99);
  for (let i = 0; i < 14; i++) {
    const x = size / 2 + (r() - 0.5) * size * 0.4;
    const y = size / 2 + (r() - 0.5) * size * 0.4;
    const rad = size * (0.15 + r() * 0.2);
    const grd = g.createRadialGradient(x, y, 0, x, y, rad);
    grd.addColorStop(0, 'rgba(255,255,255,0.35)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, size, size);
  }
  return c;
}

function ringTex(size = 32) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  g.strokeStyle = 'rgba(255,255,255,1)';
  g.lineWidth = size / 10;
  g.beginPath();
  g.arc(size / 2, size / 2, size * 0.38, 0, Math.PI * 2);
  g.stroke();
  return c;
}

function uiPanel() {
  const p = painter(24, 24);
  p.rect(1, 1, 22, 22, PAL.gold);
  p.rect(2, 2, 20, 20, PAL.darkgold);
  p.rect(3, 3, 18, 18, PAL.navy);
  p.rect(3, 3, 18, 2, PAL.blue);
  p.rect(0, 2, 1, 20, PAL.black);
  p.rect(23, 2, 1, 20, PAL.black);
  p.rect(2, 0, 20, 1, PAL.black);
  p.rect(2, 23, 20, 1, PAL.black);
  p.px(1, 1, PAL.black);
  p.px(22, 1, PAL.black);
  p.px(1, 22, PAL.black);
  p.px(22, 22, PAL.black);
  for (const [x, y] of [[2, 2], [21, 2], [2, 21], [21, 21]]) p.px(x, y, PAL.yellow);
  return p.c;
}

function uiButton(pressed) {
  const p = painter(24, 16);
  p.rect(1, 0, 22, 16, PAL.black);
  p.rect(0, 1, 24, 14, PAL.black);
  p.rect(1, 1, 22, 14, pressed ? PAL.teal : PAL.green);
  if (!pressed) p.rect(1, 1, 22, 2, PAL.lime);
  p.rect(1, 13, 22, 2, pressed ? PAL.green : PAL.teal);
  return p.c;
}

// -------------------------------------------------------------------------------- normal maps

/** Height from luminance + bevel near transparent edges, then Sobel -> tangent-space normal. */
function normalMapFor(c, { seamless = false, bevel = 2.5, strength = 2.0 } = {}) {
  const w = c.width;
  const h = c.height;
  const src = c.getContext('2d').getImageData(0, 0, w, h).data;
  const H = new Float32Array(w * h);
  const A = (x, y) => {
    if (seamless) return 255;
    if (x < 0 || y < 0 || x >= w || y >= h) return 0;
    return src[(y * w + x) * 4 + 3];
  };
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      if (src[i + 3] < 8) continue;
      const l = (0.2126 * src[i] + 0.7152 * src[i + 1] + 0.0722 * src[i + 2]) / 255;
      let d = bevel;
      if (!seamless) {
        for (let r = 1; r <= Math.ceil(bevel); r++) {
          if (!A(x - r, y) || !A(x + r, y) || !A(x, y - r) || !A(x, y + r)) {
            d = r - 0.5;
            break;
          }
        }
      }
      H[y * w + x] = 0.55 * l + 0.45 * (d / bevel);
    }
  const get = (x, y) => {
    if (seamless) {
      x = (x + w) % w;
      y = (y + h) % h;
    } else {
      x = Math.max(0, Math.min(w - 1, x));
      y = Math.max(0, Math.min(h - 1, y));
    }
    return H[y * w + x];
  };
  const out = makeCanvas(w, h);
  const og = out.getContext('2d');
  const img = og.createImageData(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const dx =
        get(x + 1, y - 1) + 2 * get(x + 1, y) + get(x + 1, y + 1) - (get(x - 1, y - 1) + 2 * get(x - 1, y) + get(x - 1, y + 1));
      const dy =
        get(x - 1, y + 1) + 2 * get(x, y + 1) + get(x + 1, y + 1) - (get(x - 1, y - 1) + 2 * get(x, y - 1) + get(x + 1, y - 1));
      // tangent space, +x right, +y UP (so screen-down gradient flips), +z toward viewer
      let nx = -dx * strength;
      let ny = dy * strength;
      let nz = 1;
      const l = Math.hypot(nx, ny, nz);
      nx /= l;
      ny /= l;
      nz /= l;
      img.data[i] = Math.round((nx * 0.5 + 0.5) * 255);
      img.data[i + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      img.data[i + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      img.data[i + 3] = src[i + 3];
    }
  og.putImageData(img, 0, 0);
  return out;
}

// ------------------------------------------------------------------------------------- atlas

function buildSpriteList() {
  const list = [];
  const add = (name, canvas, opts = {}) => list.push({ name, canvas, ...opts });
  add('hero_idle_0', hero({}));
  add('hero_idle_1', hero({ bob: 1, arm: 0 }));
  for (let i = 0; i < 4; i++) add(`hero_run_${i}`, hero({ leg: i, bob: i % 2, arm: i < 2 ? 1 : -1 }));
  add('hero_jump_0', hero({ jump: true, arm: -1 }));
  add('hero_hurt_0', hero({ hurt: true }));
  add('slime_0', slime(0));
  add('slime_1', slime(1));
  add('slime_2', slime(-1));
  add('bunny_0', bunny(0));
  add('bunny_1', bunny(1));
  add('bat_0', bat(0));
  add('bat_1', bat(1));
  for (let i = 0; i < 8; i++) add(`coin_${i}`, coin(i, 8));
  add('gem', gem());
  add('heart', heart());
  add('potion', potion());
  add('key', key());
  add('chest', chest());
  add('crate', crate());
  add('mushroom', mushroom());
  add('star', star());
  for (let i = 0; i < 3; i++) add(`torch_${i}`, torch(i));
  add('bomb', bomb());
  add('sword', sword());
  add('shield', shield());
  add('tile_grass', tileGrass(), { seamless: true });
  add('tile_dirt', tileDirt(), { seamless: true });
  add('tile_sand', tileSand(), { seamless: true });
  add('tile_leaves', tileLeaves(), { seamless: true });
  add('tile_stone', tileStone(), { seamless: true });
  add('tile_brick', tileBrick(), { seamless: true });
  add('tile_wood', tileWood(), { seamless: true });
  add('tile_ice', tileIce(), { seamless: true });
  add('tile_metal', tileMetal(), { seamless: true });
  for (let i = 0; i < 4; i++) add(`tile_water_${i}`, tileWater(i), { seamless: true });
  for (let i = 0; i < 4; i++) add(`tile_lava_${i}`, tileLava(i), { seamless: true });
  add('tree', tree());
  add('bush', bush());
  add('flower', flower());
  add('cloud', cloud());
  add('rock', rock());
  add('ui_panel', uiPanel());
  add('ui_button_0', uiButton(false));
  add('ui_button_1', uiButton(true));
  add('p_soft', softCircle(32), { smooth: true });
  add('p_spark', sparkTex(32), { smooth: true });
  add('p_star', starTex(32), { smooth: true });
  add('p_smoke', smokeTex(32), { smooth: true });
  add('p_ring', ringTex(32), { smooth: true });
  return list;
}

let atlasPromise = null;

/** Build (once) and return the shared sprite atlas. */
export function getAtlas() {
  if (!atlasPromise) atlasPromise = Promise.resolve().then(buildAtlas);
  return atlasPromise;
}

function buildAtlas() {
  const list = buildSpriteList();
  const PAD = 2;
  const W = 512;
  // shelf packing, tallest first
  const order = [...list].sort((a, b) => b.canvas.height - a.canvas.height);
  let x = PAD;
  let y = PAD;
  let shelf = 0;
  for (const s of order) {
    const w = s.canvas.width;
    const h = s.canvas.height;
    if (x + w + PAD > W) {
      x = PAD;
      y += shelf + PAD * 2;
      shelf = 0;
    }
    s.x = x;
    s.y = y;
    x += w + PAD * 2;
    shelf = Math.max(shelf, h);
  }
  const H = Math.pow(2, Math.ceil(Math.log2(y + shelf + PAD)));
  const canvas = makeCanvas(W, H);
  const normalCanvas = makeCanvas(W, H);
  const g = canvas.getContext('2d');
  const ng = normalCanvas.getContext('2d');
  g.imageSmoothingEnabled = false;
  ng.imageSmoothingEnabled = false;
  ng.fillStyle = 'rgb(128,128,255)';
  ng.fillRect(0, 0, W, H);
  ng.clearRect(0, 0, W, H);
  const frames = {};
  for (const s of list) {
    const { canvas: c, x: sx, y: sy } = s;
    const w = c.width;
    const h = c.height;
    // extrude edges by 1px so linear filtering doesn't bleed neighbours
    const ex = (dst, src) => {
      dst.drawImage(src, 0, 0, w, 1, sx, sy - 1, w, 1);
      dst.drawImage(src, 0, h - 1, w, 1, sx, sy + h, w, 1);
      dst.drawImage(src, 0, 0, 1, h, sx - 1, sy, 1, h);
      dst.drawImage(src, w - 1, 0, 1, h, sx + w, sy, 1, h);
      dst.drawImage(src, sx, sy, w, h);
    };
    if (s.seamless) ex(g, c);
    g.drawImage(c, sx, sy);
    const n = normalMapFor(c, { seamless: !!s.seamless });
    if (s.seamless) ex(ng, n);
    ng.drawImage(n, sx, sy);
    frames[s.name] = { x: sx, y: sy, w, h };
  }
  const atlas = {
    canvas,
    normalCanvas,
    width: W,
    height: H,
    frames,
    names: () => Object.keys(frames),
    /** UV rect [u0, v0, u1, v1] for a frame (top-left origin). */
    uv(name) {
      const f = frames[name];
      if (!f) throw new Error(`atlas: no frame "${name}"`);
      return [f.x / W, f.y / H, (f.x + f.w) / W, (f.y + f.h) / H];
    },
    /** Frame names of an animation: anim('hero_run') -> ['hero_run_0', ...]. */
    anim(prefix) {
      return Object.keys(frames)
        .filter((k) => k.startsWith(prefix + '_') && /_\d+$/.test(k))
        .sort((a, b) => +a.split('_').pop() - +b.split('_').pop());
    },
  };
  return atlas;
}

/** Extra standalone textures (not in the atlas). */
export const textures = { softCircle, sparkTex, starTex, smokeTex, ringTex, normalMapFor, outline, painter };

import { shaderScene } from '../../core/shaderscene.js';
import { getAtlas, makeCanvas } from '../../core/assets.js';
import { overlayTag, keyAxis, KEYS, anyKey, clamp, wrapAngle } from './_shared.js';

// Wolfenstein-3D-style raycasting: for each screen column, march a ray through a grid map with DDA,
// draw a textured wall slice whose height is 1/distance, then cast the floor & ceiling per pixel.
// The DDA runs per pixel in a portable fragment shader (WebGPU + WebGL2); movement, collision,
// the top-down rays and sprite projection run in JavaScript and are passed in via bind().

const MAP = [
  '222222222222222222222222',
  '2......................2',
  '2......................2',
  '2...5.............5....2',
  '2......................2',
  '2......................2',
  '2...5.............5....2',
  '2......................2',
  '2222.2222111..1112222.22',
  '2......1.........1.....2',
  '2......1..3...3..1.....2',
  '2..44..1.........1..44.2',
  '2..44..1..3...3..1..44.2',
  '2......1.........1.....2',
  '2......111.....111.....2',
  '2......................2',
  '22.22222222.2222222222.2',
  '2.....3.......3........2',
  '2.....3...5...3...5....2',
  '2.....3.......3........2',
  '2..3333...5...3333.....2',
  '2......................2',
  '2......................2',
  '222222222222222222222222',
];
const MW = 24;
const MH = 24;
const cell = (x, y) => {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  if (xi < 0 || yi < 0 || xi >= MW || yi >= MH) return 2;
  const c = MAP[yi][xi];
  return c === '.' ? 0 : +c;
};

function mapCanvas() {
  const c = makeCanvas(MW, MH);
  const g = c.getContext('2d');
  const img = g.createImageData(MW, MH);
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      const i = (y * MW + x) * 4;
      img.data[i] = cell(x, y) * 32;
      img.data[i + 3] = 255;
    }
  g.putImageData(img, 0, 0);
  return c;
}

const PATH = [
  [12.5, 4.5], [20.5, 4.5], [21.5, 7.5], [21.5, 15.5], [22.5, 17.5], [20.5, 21.5], [11.5, 21.5],
  [11.5, 16.2], [10.5, 15.5], [5.5, 15.5], [5.5, 9.5], [4.5, 8.2], [4.5, 4.5],
];
const LIGHTS = [
  [3.5, 4.5], [19.5, 4.5], [12.5, 11.5], [12.5, 19.5], [4.5, 21.5], [21.5, 12.5],
];
const NSPR = 16;
const NRAYS = 160;
const FRAMES = ['torch_0', 'torch_1', 'torch_2', 'chest', 'crate', 'slime_0', 'slime_1', 'slime_2', 'potion', 'gem', 'key', 'heart'];

/** DDA, exactly like the shader: returns {dist (perpendicular), side, type, hx, hy}. */
function castJS(px, py, rx, ry) {
  let mx = Math.floor(px);
  let my = Math.floor(py);
  const ddx = Math.abs(1 / (rx || 1e-9));
  const ddy = Math.abs(1 / (ry || 1e-9));
  const sx = rx < 0 ? -1 : 1;
  const sy = ry < 0 ? -1 : 1;
  let sdx = rx < 0 ? (px - mx) * ddx : (mx + 1 - px) * ddx;
  let sdy = ry < 0 ? (py - my) * ddy : (my + 1 - py) * ddy;
  let side = 0;
  let type = 0;
  for (let i = 0; i < 64; i++) {
    if (sdx < sdy) {
      sdx += ddx;
      mx += sx;
      side = 0;
    } else {
      sdy += ddy;
      my += sy;
      side = 1;
    }
    type = cell(mx, my);
    if (type) break;
  }
  const dist = side === 0 ? sdx - ddx : sdy - ddy;
  return { dist, side, type, hx: px + rx * dist, hy: py + ry * dist };
}

let S = null;
let atlasRef = null;
let TEST = false;

function newState(ctx) {
  const st = {
    ctx,
    x: 12.5,
    y: 5.5,
    a: -Math.PI / 2,
    wp: 1,
    idle: 99,
    bob: 0,
    walk: 0,
    tags: {},
    lastEx: null,
    frames: new Float32Array(12 * 4),
    aspect: [],
    sprites: [
      { x: 12.5, y: 11.5, f: 3, h: 0.42 },
      { x: 1.6, y: 1.6, f: 4, h: 0.5 }, { x: 22.4, y: 1.6, f: 4, h: 0.5 }, { x: 2.6, y: 1.6, f: 4, h: 0.5 },
      { x: 1.6, y: 22.4, f: 4, h: 0.5 }, { x: 15.5, y: 18.5, f: 8, h: 0.32 }, { x: 8.5, y: 21.5, f: 9, h: 0.28, float: true },
      { x: 20.5, y: 12.5, f: 10, h: 0.28, float: true }, { x: 9.0, y: 11.0, f: 11, h: 0.26, float: true },
      { slime: 0, x: 0, y: 0, f: 5, h: 0.42 }, { slime: 1, x: 0, y: 0, f: 5, h: 0.42 }, { slime: 2, x: 0, y: 0, f: 5, h: 0.42 },
    ],
  };
  for (const [x, y] of LIGHTS) st.sprites.push({ x, y, f: 0, h: 0.75, torch: true });
  if (atlasRef) {
    FRAMES.forEach((n, i) => {
      st.frames.set(atlasRef.uv(n), i * 4);
      const f = atlasRef.frames[n];
      st.aspect[i] = f.w / f.h;
    });
  }
  return st;
}

function tryMove(st, dx, dy) {
  const r = 0.22;
  const nx = st.x + dx;
  if (!cell(nx + Math.sign(dx) * r, st.y - r) && !cell(nx + Math.sign(dx) * r, st.y + r)) st.x = nx;
  const ny = st.y + dy;
  if (!cell(st.x - r, ny + Math.sign(dy) * r) && !cell(st.x + r, ny + Math.sign(dy) * r)) st.y = ny;
}

function step(st, p, ctx, dt) {
  const keys = ctx.keys;
  const manual = anyKey(keys, [KEYS.up, KEYS.down, KEYS.left, KEYS.right, KEYS.turnL, KEYS.turnR]);
  st.idle = manual ? 0 : st.idle + dt;
  let fwd = keyAxis(keys, KEYS.down, KEYS.up);
  let turn = keyAxis(keys, ['ArrowLeft', 'q', 'KeyQ'], ['ArrowRight', 'e', 'KeyE']);
  const strafe = keyAxis(keys, ['a', 'KeyA'], ['d', 'KeyD']);
  // mouse drag turns too
  if (ctx.pointer.down) turn += clamp(ctx.pointer.dx * 0.25, -3, 3);
  if (p.autopilot && st.idle > 2.5 && !ctx.pointer.down) {
    if (ctx.example === 'fisheye') {
      // stand still and sweep the view across the long flat wall
      const want = -Math.PI / 2 + Math.sin(ctx.time * 0.45) * 0.7;
      turn = clamp(wrapAngle(want - st.a) * 3, -1, 1);
      fwd = 0;
      const tx = 12.5 - st.x;
      const ty = 2.6 - st.y;
      if (Math.hypot(tx, ty) > 0.2) tryMove(st, tx * dt, ty * dt);
    } else {
      const [wx, wy] = PATH[st.wp];
      if (Math.hypot(wx - st.x, wy - st.y) < 0.45) st.wp = (st.wp + 1) % PATH.length;
      const want = Math.atan2(wy - st.y, wx - st.x);
      const err = wrapAngle(want - st.a);
      turn = clamp(err * 2.5, -1, 1);
      fwd = Math.abs(err) < 0.9 ? 0.75 : 0.15;
    }
  }
  st.a += turn * 2.4 * dt;
  const sp = 2.6 * dt;
  const ca = Math.cos(st.a);
  const sa = Math.sin(st.a);
  tryMove(st, (ca * fwd - sa * strafe) * sp, (sa * fwd + ca * strafe) * sp);
  const moving = Math.abs(fwd) + Math.abs(strafe) > 0.05;
  st.walk += moving ? dt * 9 : 0;
  st.bob += ((moving ? 1 : 0) - st.bob) * Math.min(1, dt * 6);
}

function bindFn(params, ctx) {
  TEST = !!ctx.testMode;
  if (!S || S.ctx !== ctx) S = newState(ctx);
  const st = S;
  const ex = ctx.example;
  if (st.lastEx !== ex) {
    for (const t of Object.values(st.tags)) t.remove();
    st.tags = {};
    st.lastEx = ex;
    // a good starting spot for each example
    const start = { topdown: [11.5, 6.2, -0.75, 1], fps: [8.5, 4.5, 0, 1], fisheye: [12.5, 2.6, -Math.PI / 2, 1] }[ex];
    if (start) [st.x, st.y, st.a, st.wp] = start;
    if (ex === 'fisheye') {
      st.tags.l = overlayTag(ctx, 'left:25%;top:40px;transform:translateX(-50%)');
      st.tags.l.textContent = 'Euclidean distance → fish-eye';
      st.tags.r = overlayTag(ctx, 'left:75%;top:40px;transform:translateX(-50%)');
      st.tags.r.textContent = 'Perpendicular distance → straight walls';
    }
    if (ex === 'topdown') {
      st.tags.l = overlayTag(ctx, 'left:25%;top:40px;transform:translateX(-50%)');
      st.tags.l.textContent = 'The grid map, one line per ray';
      st.tags.r = overlayTag(ctx, 'left:75%;top:40px;transform:translateX(-50%)');
    }
  }
  const dt = Math.min(ctx.dt, 1 / 20);
  step(st, params, ctx, dt);

  const W = ctx.width;
  const H = ctx.height;
  const split = ex !== 'fps';
  const x0 = split ? Math.floor(W / 2) : 0;
  const vw = W - x0;
  const fov = (params.fov * Math.PI) / 180;
  const plane = Math.tan(fov / 2);
  const f = vw / 2 / plane;
  const bobY = Math.sin(st.walk) * st.bob * H * 0.012;
  const dirX = Math.cos(st.a);
  const dirY = Math.sin(st.a);
  const plX = -dirY * plane;
  const plY = dirX * plane;

  // top-down ray fan: the exact same DDA, in JavaScript
  const nr = Math.round(params.rays);
  const rayHits = new Float32Array(NRAYS * 4);
  if (ex === 'topdown') {
    for (let i = 0; i < nr; i++) {
      const cx = (2 * (i + 0.5)) / nr - 1;
      const h = castJS(st.x, st.y, dirX + plX * cx, dirY + plY * cx);
      rayHits.set([h.hx, h.hy, h.side, h.type], i * 4);
    }
    if (st.tags.r) st.tags.r.textContent = `${nr} rays → ${nr} vertical wall strips`;
  }

  // sprites (billboards): camera-space transform, sorted far -> near
  const sprRect = new Float32Array(NSPR * 4);
  const sprInfo = new Float32Array(NSPR * 4);
  if (ex !== 'fisheye') {
    const invDet = 1 / (plX * dirY - dirX * plY);
    const list = [];
    const t = ctx.time;
    for (const s of st.sprites) {
      let sx = s.x;
      let sy = s.y;
      let frame = s.f;
      let lift = 0;
      if (s.slime !== undefined) {
        // slimes patrol back and forth
        const k = s.slime;
        const base = [[12.5, 2.5, 6, 0], [3.5, 18.5, 0, 2.2], [17.5, 21.5, 3.5, 0]][k];
        const ph = Math.sin(t * 0.5 + k * 2.1);
        sx = base[0] + base[2] * ph;
        sy = base[1] + base[3] * ph;
        frame = 5 + (Math.floor(t * 6 + k) % 3);
        lift = Math.abs(Math.sin(t * 4 + k)) * 0.12;
      }
      if (s.torch) frame = Math.floor(t * 8 + s.x) % 3;
      if (s.float) lift = 0.18 + Math.sin(t * 2 + s.x) * 0.05;
      const rx = sx - st.x;
      const ry = sy - st.y;
      const tx = invDet * (dirY * rx - dirX * ry);
      const tz = invDet * (-plY * rx + plX * ry);
      if (tz < 0.15) continue;
      const scrX = x0 + (vw / 2) * (1 - tx / tz);
      const hpx = (f * s.h) / tz;
      const wpx = hpx * (st.aspect[frame] || 1);
      const ground = H / 2 + bobY + (f * 0.5) / tz;
      const lp = (f * lift) / tz;
      if (scrX + wpx / 2 < x0 || scrX - wpx / 2 > W) continue;
      list.push({ tz, rect: [scrX - wpx / 2, ground - hpx - lp, wpx, hpx], info: [tz, frame, s.torch ? 1 : 0, 0] });
    }
    list.sort((a, b) => b.tz - a.tz);
    list.slice(-NSPR).forEach((s, i) => {
      sprRect.set(s.rect, i * 4);
      sprInfo.set(s.info, i * 4);
    });
  }
  const flick = 0.85 + 0.15 * Math.sin(ctx.time * 17) * Math.sin(ctx.time * 11.3);
  return {
    player: [st.x, st.y, st.a, bobY],
    view: [x0, vw, f, plane],
    rayHits,
    sprRect,
    sprInfo,
    frames: st.frames,
    flick,
  };
}

const FRAG = /* wgsl */ `
const NSPR: i32 = ${NSPR};

fn cellAt(c: vec2f) -> i32 { return i32(LOAD(level, vec2i(c)).r * 255.0 / 32.0 + 0.5); }

// DDA through the grid. Returns (perpendicular distance, side 0=x/1=y, wall type, wallX 0..1)
fn castRay(pos: vec2f, rd: vec2f) -> vec4f {
  var mp = floor(pos);
  let dd = vec2f(1.0 / max(abs(rd.x), 0.00001), 1.0 / max(abs(rd.y), 0.00001));
  var sd: vec2f;
  var stp: vec2f;
  if (rd.x < 0.0) { stp.x = -1.0; sd.x = (pos.x - mp.x) * dd.x; } else { stp.x = 1.0; sd.x = (mp.x + 1.0 - pos.x) * dd.x; }
  if (rd.y < 0.0) { stp.y = -1.0; sd.y = (pos.y - mp.y) * dd.y; } else { stp.y = 1.0; sd.y = (mp.y + 1.0 - pos.y) * dd.y; }
  var side = 0.0;
  var kind = 2;
  for (var i = 0; i < 64; i++) {
    if (sd.x < sd.y) { sd.x += dd.x; mp.x += stp.x; side = 0.0; }
    else { sd.y += dd.y; mp.y += stp.y; side = 1.0; }
    let c = cellAt(mp);
    if (c > 0) { kind = c; break; }
  }
  var dist = sd.y - dd.y;
  var wx = pos.x + dist * rd.x;
  if (side < 0.5) { dist = sd.x - dd.x; wx = pos.y + dist * rd.y; }
  return vec4f(dist, side, f32(kind), fract(wx));
}

// ---- procedural 64x64 "pixel art" wall textures (q = integer texel coordinate) ----
fn brickTex(q: vec2f) -> vec3f {
  let row = floor(q.y / 8.0);
  let off = fmod(row, 2.0) * 8.0;
  let lx = fmod(q.x + off, 16.0);
  let ly = fmod(q.y, 8.0);
  let id = vec2f(floor((q.x + off) / 16.0), row);
  var c = mix(vec3f(0.5, 0.17, 0.12), vec3f(0.74, 0.3, 0.19), hash21(id));
  c *= 0.85 + 0.3 * hash21(q);
  c *= 1.0 - 0.28 * f32(ly > 6.0) + 0.15 * f32(ly < 2.0);
  return mix(c, vec3f(0.45, 0.42, 0.38) * (0.8 + 0.3 * hash21(q + 9.0)), f32(lx < 1.0 || ly < 1.0));
}
fn stoneTex(q: vec2f) -> vec3f {
  let row = floor(q.y / 16.0);
  let off = floor(hash11(row + 1.0) * 16.0);
  let bw = 21.0;
  let lx = fmod(q.x + off, bw);
  let ly = fmod(q.y, 16.0);
  let h = hash21(vec2f(floor((q.x + off) / bw), row));
  var c = vec3f(0.5 + 0.14 * h, 0.49 + 0.14 * h, 0.53 + 0.14 * h);
  c *= 0.86 + 0.26 * hash21(floor(q / 2.0));
  let gap = f32(lx < 1.0 || ly < 1.0);
  let hi = f32(lx < 2.0 || ly < 2.0) * (1.0 - gap);
  let lo = f32(lx > bw - 2.0 || ly > 14.0);
  c = c * (1.0 + 0.3 * hi) * (1.0 - 0.3 * lo);
  return mix(c, vec3f(0.14, 0.14, 0.16), gap);
}
fn woodTex(q: vec2f) -> vec3f {
  let pl = floor(q.x / 16.0);
  let lx = fmod(q.x, 16.0);
  let grain = 0.5 + 0.5 * sin(q.x * 1.3 + 5.0 * valueNoise(vec2f(q.x * 0.25, q.y * 0.06 + pl * 7.0)));
  var c = mix(vec3f(0.36, 0.21, 0.11), vec3f(0.62, 0.4, 0.2), hash11(pl + 3.0) * 0.5 + grain * 0.5);
  c *= 0.92 + 0.16 * hash21(q);
  let nail = f32(abs(lx - 8.0) < 1.0 && abs(fmod(q.y, 32.0) - 5.0) < 1.0);
  c = mix(c, vec3f(0.12, 0.07, 0.04), f32(lx < 1.0));
  c = mix(c, vec3f(0.75, 0.75, 0.7), nail);
  // horizontal iron band
  let band = f32(abs(q.y - 40.0) < 3.0);
  return mix(c, vec3f(0.22, 0.22, 0.26) * (0.8 + 0.4 * f32(abs(q.y - 40.0) < 1.0)), band);
}
fn techTex(q: vec2f) -> vec3f {
  let lp = fmod2(q, vec2f(32.0));
  var c = vec3f(0.16, 0.24, 0.48) * (0.88 + 0.2 * hash21(floor(q / 2.0)));
  let edge = f32(lp.x < 1.0 || lp.y < 1.0);
  let hi = f32(lp.x < 2.0 || lp.y < 2.0) * (1.0 - edge);
  c = c * (1.0 + 0.4 * hi);
  let rivet = f32(length(lp - vec2f(4.5, 4.5)) < 1.6 || length(lp - vec2f(27.5, 4.5)) < 1.6 || length(lp - vec2f(4.5, 27.5)) < 1.6 || length(lp - vec2f(27.5, 27.5)) < 1.6);
  c = mix(c, vec3f(0.65, 0.72, 0.85), rivet);
  let glow = f32(abs(lp.y - 16.0) < 2.0 && lp.x > 8.0 && lp.x < 24.0);
  c = mix(c, vec3f(0.45, 0.95, 1.0), glow);
  return mix(c, vec3f(0.06, 0.08, 0.16), edge);
}
fn mossTex(q: vec2f) -> vec3f {
  var c = stoneTex(q + vec2f(7.0, 3.0));
  let m = valueNoise(q * 0.12) * 0.7 + valueNoise(q * 0.4) * 0.3 + q.y / 64.0 * 0.35;
  let moss = step(0.62, m);
  return mix(c, vec3f(0.22, 0.42, 0.16) * (0.8 + 0.4 * hash21(q)), moss);
}
fn wallTex(kind: i32, t: vec2f, textured: f32) -> vec3f {
  let q = floor(clamp(t, vec2f(0.0), vec2f(0.9999)) * 64.0);
  if (textured < 0.5) {
    var flatC = vec3f(0.5);
    if (kind == 1) { flatC = vec3f(0.65, 0.25, 0.17); }
    if (kind == 3) { flatC = vec3f(0.5, 0.32, 0.16); }
    if (kind == 4) { flatC = vec3f(0.2, 0.3, 0.6); }
    if (kind == 5) { flatC = vec3f(0.3, 0.45, 0.25); }
    return flatC;
  }
  if (kind == 1) { return brickTex(q); }
  if (kind == 3) { return woodTex(q); }
  if (kind == 4) { return techTex(q); }
  if (kind == 5) { return mossTex(q); }
  return stoneTex(q);
}
fn floorTex(w: vec2f) -> vec3f {
  let q = floor(w * 32.0);                    // 32 texels per cell, 2x2 tiles per cell
  let lt = fmod2(q, vec2f(16.0));
  let id = floor(q / 16.0);
  var c = mix(vec3f(0.36, 0.34, 0.31), vec3f(0.47, 0.44, 0.4), hash21(id)) * (0.88 + 0.22 * hash21(q));
  c = mix(c, c * 0.55, f32(lt.x < 1.0 || lt.y < 1.0));
  return c;
}
fn ceilTex(w: vec2f) -> vec3f {
  let q = floor(w * 32.0);
  let lx = fmod(q.x, 8.0);
  var c = vec3f(0.24, 0.17, 0.11) * (0.85 + 0.25 * hash21(vec2f(floor(q.x / 8.0), floor(q.y / 64.0))));
  c *= 0.9 + 0.15 * hash21(q);
  c = mix(c, vec3f(0.08, 0.05, 0.03), f32(lx < 1.0));
  let beam = f32(fmod(q.y, 32.0) < 4.0);
  return mix(c, vec3f(0.14, 0.1, 0.07), beam);
}

fn lightAt(p3: vec3f) -> vec3f {
  var L = vec3f(0.0);
  var lp = array<vec2f, 6>(${LIGHTS.map(([x, y]) => `vec2f(${x.toFixed(1)}, ${y.toFixed(1)})`).join(', ')});
  for (var i = 0; i < 6; i++) {
    let d = length(vec3f(lp[i].x, lp[i].y, 0.62) - p3);
    L += vec3f(1.0, 0.62, 0.3) * u.torch * u.flick / (1.0 + d * d * 1.6);
  }
  return L;
}

fn fogMul(d: f32) -> f32 { return exp(-d * u.fog * 0.3); }

// One first-person view. x0/vw = the screen columns it occupies. fish = 1 uses Euclidean distance.
fn render3D(p: vec2f, x0: f32, vw: f32, fish: f32, strips: f32) -> vec3f {
  let H = u.resolution.y;
  let pos = u.player.xy;
  let dir = vec2f(cos(u.player.z), sin(u.player.z));
  let pl = vec2f(-dir.y, dir.x) * u.view.w;
  let f = u.view.z;
  var colX = p.x;
  var stripEdge = 0.0;
  if (strips > 0.5) {
    // one ray per strip: quantize the column
    let sw = vw / strips;
    let si = floor((p.x - x0) / sw);
    colX = x0 + (si + 0.5) * sw;
    stripEdge = f32(fmod(p.x - x0, sw) < 1.0);
  }
  let camX = 2.0 * (colX - x0) / vw - 1.0;
  let rd = dir + pl * camX;
  let hit = castRay(pos, rd);
  var dist = hit.x;
  if (fish > 0.5) { dist = hit.x * length(rd); }    // no cos() correction -> bulging walls
  let hy = H * 0.5 + u.player.w;
  let top = hy - f * 0.5 / dist;
  let bot = hy + f * 0.5 / dist;
  var col: vec3f;
  if (p.y >= top && p.y <= bot) {
    let kind = i32(hit.z);
    var tx = hit.w;
    if ((hit.y < 0.5 && rd.x > 0.0) || (hit.y > 0.5 && rd.y < 0.0)) { tx = 1.0 - tx; }
    let tv = (p.y - top) / (bot - top);
    col = wallTex(kind, vec2f(tx, tv), u.textured);
    if (u.sideShade > 0.5 && hit.y > 0.5) { col *= 0.68; }
    let hw = pos + rd * hit.x;
    col *= vec3f(0.62) + lightAt(vec3f(hw, 1.0 - tv));
    col *= fogMul(hit.x * length(rd));
    if (strips > 0.5) {
      let si = floor((p.x - x0) / (vw / strips));
      let ray = hsv2rgb(vec3f(si / strips, 0.75, 1.0));
      col = mix(col, ray, f32(p.y - top < 3.0) * 0.9);
      col = mix(col, vec3f(0.0), stripEdge * 0.6);
    }
  } else {
    // floor & ceiling casting: each row is at a known distance from the eye
    let below = p.y > hy;
    let rowD = f * 0.5 / max(abs(p.y - hy), 0.5);
    var w = pos + rd * rowD;
    var d = rowD * length(rd);
    if (fish > 0.5) { w = pos + normalize(rd) * rowD; d = rowD; }
    if (u.floors > 0.5) {
      if (below) { col = floorTex(w); } else { col = ceilTex(w); }
    } else {
      if (below) { col = vec3f(0.33, 0.33, 0.33); } else { col = vec3f(0.18, 0.18, 0.2); }
    }
    var hz = 1.0;
    if (below) { hz = 0.0; }
    col *= vec3f(0.62) + lightAt(vec3f(w, hz));
    col *= fogMul(d);
  }
  // billboard sprites, hidden behind closer walls (a 1D depth buffer = this column's wall distance)
  if (fish < 0.5) {
    for (var i = 0; i < NSPR; i++) {
      let r = u.sprRect[i];
      if (r.z <= 0.0) { continue; }
      let q = (p - r.xy) / r.zw;
      if (q.x < 0.0 || q.y < 0.0 || q.x >= 1.0 || q.y >= 1.0) { continue; }
      let inf = u.sprInfo[i];
      if (inf.x > hit.x) { continue; }
      let fr = u.frames[i32(inf.y)];
      let c = TEXN(atlas, mix(fr.xy, fr.zw, q));
      var sc = c.rgb * fogMul(inf.x);
      if (inf.z < 0.5) { sc *= vec3f(0.7) + lightAt(vec3f(pos + dir * inf.x, 0.3)); }
      col = mix(col, sc, step(0.5, c.a));
    }
  }
  return col;
}

// Top-down grid map. o = top-left, s = pixels per cell.
fn mapView(p: vec2f, o: vec2f, s: f32, rays: f32) -> vec3f {
  let w = (p - o) / s;
  var col = vec3f(0.07, 0.08, 0.1);
  if (w.x >= 0.0 && w.y >= 0.0 && w.x < ${MW}.0 && w.y < ${MH}.0) {
    let k = cellAt(floor(w));
    col = vec3f(0.14, 0.15, 0.19);
    if (k > 0) { col = wallTex(k, vec2f(0.5, 0.5), 0.0) * 0.9; }
    let g = fract(w);
    col = mix(col, col * 0.6, f32(min(g.x, g.y) * s < 1.0));
  }
  let pos = u.player.xy;
  let dir = vec2f(cos(u.player.z), sin(u.player.z));
  let pl = vec2f(-dir.y, dir.x) * u.view.w;
  let pp = o + pos * s;
  if (rays > 0.5) {
    // which rays could pass near this pixel? invert the camera-plane coordinate
    let d = w - pos;
    let fz = dot(d, dir);
    if (fz > 0.0) {
      let cxp = dot(d, pl) / dot(pl, pl) / fz;
      let fi = (cxp + 1.0) * 0.5 * rays - 0.5;
      for (var kk = 0; kk < 2; kk++) {
        let idx = clamp(floor(fi) + f32(kk), 0.0, rays - 1.0);
        let h = u.rayHits[i32(idx)];
        let ds = sdSegment(p, pp, o + h.xy * s);
        let rc = hsv2rgb(vec3f(idx / rays, 0.75, 1.0));
        col = mix(col, rc, (1.0 - smoothstep(0.4, 1.4, ds)) * 0.85);
        col = mix(col, vec3f(1.0), 1.0 - smoothstep(2.0, 3.0, length(p - (o + h.xy * s))));
      }
    }
  } else {
    // minimap: highlight what the player can see (one DDA per minimap pixel)
    let d = w - pos;
    let ld = length(d);
    let cxp = dot(d, pl) / dot(pl, pl) / max(dot(d, dir), 0.0001);
    if (dot(d, dir) > 0.0 && abs(cxp) <= 1.0 && ld < 9.0) {
      let h = castRay(pos, d / ld);
      if (ld < h.x) { col = mix(col, vec3f(1.0, 0.85, 0.45), 0.28 * (1.0 - ld / 9.0)); }
    }
  }
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(s * 0.35 + 1.5, s * 0.35 + 2.5, length(p - pp)));
  col = mix(col, vec3f(1.0, 0.85, 0.3), 1.0 - smoothstep(s * 0.35, s * 0.35 + 1.0, length(p - pp)));
  col = mix(col, vec3f(1.0, 0.85, 0.3), 1.0 - smoothstep(0.8, 1.8, sdSegment(p, pp, pp + dir * s * 1.1)));
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  let p = uv * u.resolution;
  let W = u.resolution.x;
  let H = u.resolution.y;
  let half = floor(W * 0.5);
  var col: vec3f;
  if (ex == 0) {
    if (p.x < half) {
      let s = floor(min(half * 0.92, H - 84.0) / ${MW}.0);
      let o = vec2f((half - s * ${MW}.0) * 0.5, 70.0 + (H - 84.0 - s * ${MH}.0) * 0.5);
      col = mapView(p, o, s, u.rays);
    } else {
      col = render3D(p, half, W - half, 0.0, u.rays);
    }
    col = mix(col, vec3f(0.35, 0.38, 0.5), 1.0 - smoothstep(0.5, 1.5, abs(p.x - half)));
  } else if (ex == 1) {
    col = render3D(p, 0.0, W, u.fisheye, 0.0);
    // minimap in the bottom-right corner
    let s = floor(H * 0.3 / ${MW}.0);
    let o = vec2f(W - s * ${MW}.0 - 10.0, H - s * ${MH}.0 - 10.0);
    let inMap = p.x > o.x - 2.0 && p.y > o.y - 2.0 && p.x < o.x + s * ${MW}.0 + 2.0 && p.y < o.y + s * ${MH}.0 + 2.0;
    if (inMap) { col = mix(col, mapView(p, o, s, 0.0), 0.92); }
  } else {
    if (p.x < half) { col = render3D(p, 0.0, half, 1.0, 0.0); }
    else { col = render3D(p, half, W - half, 0.0, 0.0); }
    col = mix(col, vec3f(0.9), 1.0 - smoothstep(0.5, 1.5, abs(p.x - half)));
  }
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: 'W/S move · ←/→ (or drag) turn · A/D strafe · let go for autopilot',
  keys: true,
  examples: [
    {
      id: 'topdown',
      label: 'Top-down rays',
      kind: 'Abstract',
      note: 'Left: the level is just a 24×24 grid of numbers. One ray per screen column is marched from the player until it hits a wall. Right: each ray becomes one <b>vertical strip</b>; the closer the hit, the taller the strip. Raise <i>Rays</i> until the strips become a smooth picture.',
      params: { pixel: 1, rays: 32, fov: 66 },
    },
    {
      id: 'fps',
      label: 'First person',
      kind: 'In a game',
      note: 'The full Wolfenstein recipe: textured walls, floor & ceiling casting, torch light, distance fog, darker north/south faces, billboard sprites hidden behind walls, and a minimap that only lights up what the player can see.',
      params: { pixel: 2, fov: 66 },
    },
    {
      id: 'fisheye',
      label: 'Fish-eye vs corrected',
      kind: 'Comparison',
      note: 'Both halves look at the same flat wall. Left uses the straight-line (Euclidean) distance to each hit — rays at the screen edges travel farther, so walls bulge like a fish-eye lens. Right multiplies by <code>cos(ray angle)</code> (the perpendicular distance), and the wall is straight.',
      params: { pixel: 1, fov: 90 },
    },
  ],
  controls: [
    { type: 'slider', key: 'fov', label: 'Field of view (°)', min: 40, max: 130, step: 1, value: 66, help: 'Wolfenstein 3D used about 66°. Wide angles exaggerate the fish-eye.' },
    { type: 'slider', key: 'rays', label: 'Rays (columns)', min: 4, max: 160, step: 1, value: 32, showFor: ['topdown'], help: 'A real game casts one ray per pixel column (320 in Wolf3D).' },
    { type: 'slider', key: 'pixel', label: 'Pixel size', min: 1, max: 4, step: 1, value: 2, help: 'Render at lower resolution for the chunky 320×200 look.' },
    { type: 'toggle', key: 'fisheye', label: 'Fish-eye (no cos correction)', value: false, showFor: ['fps'] },
    { type: 'toggle', key: 'textured', label: 'Textured walls', value: true, help: 'Off = flat colors, like the very first raycasters.' },
    { type: 'toggle', key: 'floors', label: 'Floor & ceiling casting', value: true, help: 'Wolf3D used flat colors here; Doom-era engines textured them.' },
    { type: 'toggle', key: 'sideShade', label: 'Darker N/S faces', value: true, help: 'A free fake light direction: it makes corners readable.' },
    { type: 'slider', key: 'fog', label: 'Distance darkness', min: 0, max: 2, step: 0.01, value: 0.45 },
    { type: 'slider', key: 'torch', label: 'Torch light', min: 0, max: 3, step: 0.01, value: 1.2, help: 'Six flickering point lights, applied to walls, floor and sprites.' },
    { type: 'toggle', key: 'autopilot', label: 'Autopilot when idle', value: true },
  ],
  uniforms: {
    fov: 'f32', rays: 'f32', fisheye: 'f32', textured: 'f32', floors: 'f32', sideShade: 'f32', fog: 'f32', torch: 'f32', flick: 'f32',
    player: 'vec4f', view: 'vec4f',
    rayHits: `array<vec4f, ${NRAYS}>`, sprRect: `array<vec4f, ${NSPR}>`, sprInfo: `array<vec4f, ${NSPR}>`, frames: 'array<vec4f, 12>',
  },
  include: ['hash', 'noise', 'sdf', 'color'],
  renderScale: (p) => 1 / Math.max(TEST ? 3 : 1, p.pixel || 1),
  textures: {
    level: { source: async () => mapCanvas(), filter: 'nearest' },
    atlas: {
      source: async () => {
        atlasRef = await getAtlas();
        return atlasRef.canvas;
      },
      filter: 'nearest',
    },
  },
  bind: bindFn,
  code: FRAG,
  about: {
    summary: 'How 1992’s Wolfenstein 3D drew a 3D maze on a 386: the level is a flat grid, and the screen is drawn one column at a time by shooting a single ray per column.',
    what: `<p>A first-person maze that is really a <b>24×24 grid of numbers</b>. There are no polygons: every vertical strip of the screen is one ray’s
      answer to “how far is the nearest wall in this direction?”. Height on screen = 1 / distance.</p>`,
    how: `<ol>
      <li>For each screen column compute a ray direction: <code>dir + plane · x</code>, where <code>x</code> goes from −1 (left edge) to +1 (right edge) and the length of <code>plane</code> sets the field of view.</li>
      <li><b>DDA</b> (digital differential analyzer): step the ray from grid line to grid line — always to whichever of the next vertical or horizontal line is closer — until the cell is a wall. It never misses a wall and costs only a few additions per cell.</li>
      <li>Use the <b>perpendicular</b> distance to the camera plane (= Euclidean distance × cos(ray angle)), otherwise flat walls bulge (see <i>Fish-eye vs corrected</i>).</li>
      <li>Wall slice height = <code>focal / distance</code>. Where the ray hit the wall (<code>wallX</code>, 0..1) picks the texture column; the pixel’s height on the slice picks the row.</li>
      <li><b>Floor/ceiling casting</b>: a screen row below the horizon always sees the floor at distance <code>focal·eyeHeight / rowsBelowHorizon</code>; walk that far along the column’s ray.</li>
      <li><b>Sprites</b>: transform into camera space, draw scaled by 1/depth, and skip pixels where this column’s wall is closer — the wall distances act as a 1D depth buffer.</li>
    </ol>
    <p>Here the whole thing runs in a fragment shader: every pixel casts its column’s ray (redundant, but GPUs don’t mind).</p>`,
    uses: [
      { title: 'Classic FPS', text: 'Wolfenstein 3D, Blake Stone, Rise of the Triad, Catacomb 3-D; Doom and Duke 3D extended the idea (sectors/portals, not grids).' },
      { title: 'Dungeon crawlers', text: 'Grid-based crawlers (Eye of the Beholder style, Legend of Grimrock’s ancestors) and many modern retro “boomer shooters”.' },
      { title: 'Tiny games & jams', text: 'Raycasters fit in a few hundred lines — popular in PICO-8, TIC-80, Arduboy and js13k games.' },
      { title: '2D line-of-sight', text: 'The same DDA grid walk powers visibility, bullets and AI vision checks in top-down tile games.' },
    ],
    try: [
      'On <b>Top-down rays</b>, drag <i>Rays</i> down to 8: the 3D view becomes 8 fat strips, each matching a colored line on the map.',
      'Switch to <b>Fish-eye vs corrected</b> and widen the <i>Field of view</i>: the left half curves more and more.',
      'Turn off <i>Darker N/S faces</i> and notice how corners lose definition.',
      'Turn off <i>Floor & ceiling casting</i> for the authentic 1992 flat-colored floor.',
      'Set <i>Distance darkness</i> to 0 and <i>Torch light</i> to 0 for a full-bright, sterile look — then crank both up for horror.',
    ],
    ask: [
      'a Wolfenstein-style raycaster with DDA and textured walls',
      'floor and ceiling casting for a grid raycaster',
      'billboard sprites with a 1D z-buffer per column',
      'fish-eye correction using perpendicular wall distance',
      'a minimap with a visibility cone',
    ],
    perf: `<p>Classic CPU cost: one DDA per <i>column</i> (320 rays) — tiny. In this shader every <i>pixel</i> repeats its column’s DDA (~10–30 grid steps),
      which is still cheap at 1080p. Floor casting is O(1) per pixel. Sprites cost a loop per pixel over the visible sprites.</p>`,
    api: `<p>A pure fragment shader, so it runs on WebGPU and WebGL2 from one WGSL source. The level is a 24×24 texture read with exact texel loads.
      A WebGPU version could cast one ray per column in a compute shader and store the distances in a buffer, then draw from it.</p>`,
    code: [
      {
        title: 'DDA: walk the grid until a wall (from the scene’s shader)',
        lang: 'wgsl',
        src: `for (var i = 0; i < 64; i++) {
  if (sd.x < sd.y) { sd.x += dd.x; mp.x += stp.x; side = 0.0; }   // cross a vertical grid line
  else             { sd.y += dd.y; mp.y += stp.y; side = 1.0; }   // cross a horizontal one
  let c = cellAt(mp);
  if (c > 0) { kind = c; break; }
}
var dist = sd.y - dd.y;               // perpendicular distance: no fish-eye
if (side < 0.5) { dist = sd.x - dd.x; }`,
      },
      {
        title: 'Wall slice and floor casting',
        lang: 'wgsl',
        src: `let top = horizon - focal * 0.5 / dist;   // eye at half wall height
let bot = horizon + focal * 0.5 / dist;
if (p.y >= top && p.y <= bot) {
  col = wallTex(kind, vec2f(wallX, (p.y - top) / (bot - top)), ...);
} else {
  let rowD = focal * 0.5 / abs(p.y - horizon);  // distance seen by this row
  let w = pos + rayDir * rowD;                   // floor/ceiling point
  col = floorTex(w);
}`,
      },
    ],
    links: [
      { title: 'Lode’s raycasting tutorial', url: 'https://lodev.org/cgtutor/raycasting.html', note: 'the classic step-by-step reference' },
      { title: 'Game Engine Black Book: Wolfenstein 3D', url: 'https://fabiensanglard.net/gebbwolf3d/', note: 'how the original worked on a 386' },
    ],
  },
});

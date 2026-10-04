import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas, rng } from '../../core/assets.js';
import { overlayTag, keyAxis, KEYS, anyKey, clamp } from './_shared.js';

// OutRun / "Lou's pseudo 3D" road. The road is a 1D list of segments, each with a curve and a height.
// JavaScript projects the visible segments every frame (curves = accumulated x offsets, hills = clip
// line), exactly like the classic algorithm; the fragment shader then binary-searches the visible
// segment for its screen row and paints road, rumble strips, lanes, grass, fog and the scaled sprites.

const SEG = 200; // segment length (world units)
const ROAD = 2000; // road HALF width
const RUMBLE = 3; // segments per color band
const CAM_H = 1000;
const NSEG = 150; // draw distance (segments) = uniform slots
const NSPR = 40;
const MAX_SPEED = SEG * 60;

// ------------------------------------------------------------------------------------- sprites

const SHEET_W = 512;
const SHEET_H = 256;
const SPR = {
  car0: [0, 0, 80, 44], car1: [80, 0, 80, 44], car2: [160, 0, 80, 44],
  van0: [0, 48, 64, 40], van1: [64, 48, 64, 40], van2: [128, 48, 64, 40],
  palm: [240, 0, 56, 96], tree: [296, 0, 64, 88], bush: [360, 0, 56, 36], rock: [416, 0, 48, 32],
  board: [0, 96, 96, 64], chevL: [96, 96, 40, 40], chevR: [136, 96, 40, 40], lamp: [176, 96, 32, 112],
  board2: [208, 96, 96, 64],
};
// world height of each sprite kind (ROAD = half the road width)
const WORLD_H = { palm: 3600, tree: 3000, bush: 900, rock: 700, board: 1900, board2: 1900, chevL: 900, chevR: 900, lamp: 3800, van: 650 };

function paintSheet() {
  const c = makeCanvas(SHEET_W, SHEET_H);
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  const r = (x, y, w, h, col) => {
    g.fillStyle = col;
    g.fillRect(x, y, w, h);
  };
  // ---- player convertible, 3 steering frames
  for (let f = 0; f < 3; f++) {
    const ox = f * 80;
    const lean = (f - 1) * 2;
    r(ox + 3, 28, 15, 15, '#16161e');
    r(ox + 62, 28, 15, 15, '#16161e');
    r(ox + 5, 30, 3, 11, '#3a3a48');
    r(ox + 72, 30, 3, 11, '#3a3a48');
    g.fillStyle = '#1a1c2c';
    g.beginPath();
    g.roundRect(ox + 4, 14, 72, 24, 7);
    g.fill();
    g.fillStyle = '#d42a3a';
    g.beginPath();
    g.roundRect(ox + 5, 15, 70, 21, 6);
    g.fill();
    r(ox + 8, 15, 64, 4, '#ff5a5f');
    r(ox + 6, 31, 68, 5, '#7a1622');
    r(ox + 16, 9, 48, 7, '#22222c');
    // driver & passenger
    g.fillStyle = '#f2c3a0';
    g.beginPath();
    g.arc(ox + 31 + lean, 8, 5, 0, Math.PI * 2);
    g.arc(ox + 50 + lean, 8, 5, 0, Math.PI * 2);
    g.fill();
    r(ox + 25 + lean, 2, 12, 5, '#5b3a29');
    r(ox + 44 + lean, 2, 13, 6, '#ffcd75');
    r(ox + 54 + lean, 6, 4, 8, '#ffcd75');
    // tail lights, plate, exhausts
    r(ox + 8, 21, 14, 6, '#5d0a12');
    r(ox + 58, 21, 14, 6, '#5d0a12');
    r(ox + 9, 22, 12, 4, '#ff3b30');
    r(ox + 59, 22, 12, 4, '#ff3b30');
    r(ox + 11, 23, 4, 2, '#ffd0c0');
    r(ox + 61, 23, 4, 2, '#ffd0c0');
    r(ox + 34, 26, 12, 6, '#f4f4f4');
    r(ox + 35, 27, 10, 1, '#41a6f6');
    r(ox + 22, 36, 5, 3, '#94b0c2');
    r(ox + 53, 36, 5, 3, '#94b0c2');
  }
  // ---- traffic: three rear views in different colors
  const vans = [['#3b5dc9', '#73a0ff'], ['#f7c64b', '#fff0a0'], ['#38b764', '#a7f070']];
  vans.forEach(([body, hi], k) => {
    const ox = k * 64;
    const oy = 48;
    r(ox + 4, oy + 28, 12, 11, '#16161e');
    r(ox + 48, oy + 28, 12, 11, '#16161e');
    g.fillStyle = '#1a1c2c';
    g.beginPath();
    g.roundRect(ox + 3, oy + 2, 58, 34, 5);
    g.fill();
    g.fillStyle = body;
    g.beginPath();
    g.roundRect(ox + 4, oy + 3, 56, 32, 4);
    g.fill();
    r(ox + 6, oy + 3, 52, 3, hi);
    r(ox + 9, oy + 7, 46, 10, '#22303c');
    r(ox + 11, oy + 8, 18, 3, '#4a6a7c');
    r(ox + 6, oy + 21, 9, 5, '#ff3b30');
    r(ox + 49, oy + 21, 9, 5, '#ff3b30');
    r(ox + 26, oy + 24, 12, 5, '#f4f4f4');
    r(ox + 4, oy + 30, 56, 4, '#1a1c2c');
  });
  // ---- palm tree
  {
    const ox = 240;
    g.strokeStyle = '#6b4628';
    g.lineWidth = 6;
    g.beginPath();
    g.moveTo(ox + 32, 96);
    g.quadraticCurveTo(ox + 36, 60, ox + 26, 26);
    g.stroke();
    g.strokeStyle = '#8f563b';
    g.lineWidth = 2;
    for (let i = 0; i < 9; i++) {
      const t = i / 9;
      const y = 92 - t * 64;
      const x = ox + 32 + 4 * t * (1 - t) * 4 - t * 6;
      g.beginPath();
      g.moveTo(x - 3, y);
      g.lineTo(x + 3, y - 1);
      g.stroke();
    }
    const frond = (ang, len, col, w) => {
      g.strokeStyle = col;
      g.lineWidth = w;
      g.beginPath();
      g.moveTo(ox + 26, 26);
      const ex = ox + 26 + Math.cos(ang) * len;
      const ey = 26 + Math.sin(ang) * len * 0.7;
      g.quadraticCurveTo(ox + 26 + Math.cos(ang) * len * 0.5, 26 + Math.sin(ang) * len * 0.3 - 10, ex, ey + 10);
      g.stroke();
    };
    for (let i = 0; i < 9; i++) frond(-Math.PI + (i / 8) * Math.PI + 0.1 * Math.sin(i * 3), 24 + (i % 3) * 3, i % 2 ? '#257a3e' : '#1d5c33', 6);
    for (let i = 0; i < 7; i++) frond(-Math.PI + 0.2 + (i / 6) * (Math.PI - 0.4), 20, '#38b764', 3);
    g.fillStyle = '#5b3a29';
    g.beginPath();
    g.arc(ox + 24, 30, 3, 0, Math.PI * 2);
    g.arc(ox + 29, 31, 3, 0, Math.PI * 2);
    g.fill();
  }
  // ---- round tree
  {
    const ox = 296;
    r(ox + 28, 56, 8, 32, '#5b3a29');
    r(ox + 28, 56, 3, 32, '#8f563b');
    const blob = (x, y, rad, col) => {
      g.fillStyle = col;
      g.beginPath();
      g.arc(ox + x, y, rad, 0, Math.PI * 2);
      g.fill();
    };
    blob(32, 36, 26, '#1d5c33');
    blob(20, 44, 16, '#1d5c33');
    blob(44, 44, 16, '#1d5c33');
    blob(28, 30, 18, '#257a3e');
    blob(40, 34, 14, '#257a3e');
    blob(24, 24, 10, '#38b764');
    blob(36, 22, 8, '#38b764');
    blob(22, 20, 4, '#a7f070');
  }
  // ---- bush and rock
  {
    const ox = 360;
    const blob = (x, y, rad, col) => {
      g.fillStyle = col;
      g.beginPath();
      g.arc(ox + x, y, rad, 0, Math.PI * 2);
      g.fill();
    };
    blob(16, 24, 12, '#1d5c33');
    blob(38, 24, 13, '#1d5c33');
    blob(27, 18, 14, '#257a3e');
    blob(22, 14, 7, '#38b764');
    r(ox + 2, 32, 52, 4, '#1d5c33');
    const rx = 416;
    g.fillStyle = '#566c86';
    g.beginPath();
    g.moveTo(rx + 2, 32);
    g.lineTo(rx + 8, 10);
    g.lineTo(rx + 22, 2);
    g.lineTo(rx + 38, 8);
    g.lineTo(rx + 46, 32);
    g.fill();
    g.fillStyle = '#94b0c2';
    g.beginPath();
    g.moveTo(rx + 9, 12);
    g.lineTo(rx + 22, 4);
    g.lineTo(rx + 30, 10);
    g.lineTo(rx + 18, 20);
    g.fill();
  }
  // ---- billboards
  const board = (ox, oy, bg1, bg2, text, sub) => {
    r(ox + 14, oy + 40, 6, 24, '#333c57');
    r(ox + 76, oy + 40, 6, 24, '#333c57');
    r(ox, oy, 96, 44, '#1a1c2c');
    const grad = g.createLinearGradient(0, oy, 0, oy + 44);
    grad.addColorStop(0, bg1);
    grad.addColorStop(1, bg2);
    g.fillStyle = grad;
    g.fillRect(ox + 2, oy + 2, 92, 40);
    g.fillStyle = '#ffffff';
    g.font = 'bold 20px sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(text, ox + 48, oy + 18);
    g.font = 'bold 10px sans-serif';
    g.fillStyle = '#ffcd75';
    g.fillText(sub, ox + 48, oy + 35);
  };
  board(0, 96, '#ef7d57', '#b13e53', 'WEBGPU', '60 FPS AHEAD');
  board(208, 96, '#41a6f6', '#3b5dc9', 'TURBO', 'PSEUDO 3D');
  // ---- chevrons
  const chev = (ox, dir) => {
    r(ox + 18, 128, 4, 8, '#333c57');
    r(ox, 96, 40, 32, '#1a1c2c');
    r(ox + 2, 98, 36, 28, '#ffcd75');
    g.fillStyle = '#1a1c2c';
    for (let k = 0; k < 2; k++) {
      const cx = ox + 12 + k * 14;
      g.beginPath();
      if (dir < 0) {
        g.moveTo(cx + 6, 101);
        g.lineTo(cx - 4, 112);
        g.lineTo(cx + 6, 123);
        g.lineTo(cx + 10, 123);
        g.lineTo(cx, 112);
        g.lineTo(cx + 10, 101);
      } else {
        g.moveTo(cx - 6, 101);
        g.lineTo(cx + 4, 112);
        g.lineTo(cx - 6, 123);
        g.lineTo(cx - 10, 123);
        g.lineTo(cx, 112);
        g.lineTo(cx - 10, 101);
      }
      g.fill();
    }
  };
  chev(96, -1);
  chev(136, 1);
  // ---- lamp post (arm reaching left, toward the road when placed on the right)
  {
    const ox = 176;
    r(ox + 22, 104, 5, 104, '#333c57');
    r(ox + 22, 104, 2, 104, '#566c86');
    r(ox + 6, 102, 21, 4, '#333c57');
    r(ox + 2, 104, 12, 5, '#566c86');
    r(ox + 3, 109, 10, 3, '#fff6c8');
  }
  return c;
}

// ------------------------------------------------------------------------------------- track

const easeIn = (a, b, t) => a + (b - a) * t * t;
const easeInOut = (a, b, t) => a + (b - a) * (-Math.cos(t * Math.PI) / 2 + 0.5);

function buildTrack(hilly, seed) {
  const segs = [];
  const lastY = () => (segs.length ? segs[segs.length - 1].y2 : 0);
  const add = (curve, y) => {
    const y1 = lastY();
    segs.push({ i: segs.length, curve, y1, y2: y, sprites: [], lamp: 0 });
  };
  const road = (enter, hold, leave, curve, hill) => {
    const y0 = lastY();
    const y1 = y0 + (hilly ? hill : 0) * SEG;
    const n = enter + hold + leave;
    for (let k = 0; k < enter; k++) add(easeIn(0, curve, k / enter), easeInOut(y0, y1, k / n));
    for (let k = 0; k < hold; k++) add(curve, easeInOut(y0, y1, (enter + k) / n));
    for (let k = 0; k < leave; k++) add(easeInOut(curve, 0, k / leave), easeInOut(y0, y1, (enter + hold + k) / n));
  };
  road(25, 25, 25, 0, 0);
  road(25, 25, 25, 0, 20);
  road(50, 50, 50, 4, 0);
  road(50, 50, 50, 0, -30);
  road(25, 25, 25, -3, 20);
  road(25, 25, 25, 3, -10);
  road(25, 25, 25, -4, 0);
  road(100, 100, 100, 3, 50);
  road(50, 50, 50, 0, -20);
  road(100, 100, 100, -4, -10);
  road(10, 10, 10, 0, 5);
  road(10, 10, 10, 0, -5);
  road(10, 10, 10, 0, 5);
  road(10, 10, 10, 0, -5);
  road(50, 50, 50, 6, 30);
  road(50, 50, 50, -2, 0);
  road(100, 100, 100, -3, -40);
  road(80, 80, 80, 2, -lastY() / SEG);
  // decorations
  const R = rng(seed);
  const n = segs.length;
  for (let i = 0; i < n; i++) {
    const s = segs[i];
    if (i % 9 === 0) s.sprites.push({ k: 'palm', off: -1.55 - R() * 0.6 });
    if (i % 9 === 4) s.sprites.push({ k: 'palm', off: 1.55 + R() * 0.6 });
    if (i % 23 === 7) s.sprites.push({ k: 'tree', off: (R() < 0.5 ? -1 : 1) * (2.6 + R() * 2) });
    if (i % 17 === 3) s.sprites.push({ k: R() < 0.6 ? 'bush' : 'rock', off: (R() < 0.5 ? -1 : 1) * (1.4 + R() * 1.5) });
    if (i % 160 === 40) s.sprites.push({ k: 'board', off: -1.45 });
    if (i % 160 === 120) s.sprites.push({ k: 'board2', off: 1.45 });
    if (i % 12 === 6) {
      s.sprites.push({ k: 'lamp', off: 1.18, flip: false });
      s.lampSide = 1;
    }
    // warning chevrons on the outside of upcoming curves
    const ahead = segs[(i + 30) % n].curve;
    if (i % 6 === 0 && Math.abs(ahead) > 2.5) s.sprites.push({ k: ahead > 0 ? 'chevL' : 'chevR', off: ahead > 0 ? -1.22 : 1.22 });
  }
  // light pools under the lamps (for the night drive)
  for (let i = 0; i < n; i++) {
    let L = 0;
    for (let d = -2; d <= 2; d++) if (segs[(i + d + n) % n].lampSide) L = Math.max(L, 1 - Math.abs(d) / 3);
    segs[i].lamp = L;
  }
  return segs;
}

// ------------------------------------------------------------------------------------- state

let S = null;
let TEST = false;

function newState(ctx) {
  const tracks = { flat: buildTrack(false, 3), hilly: buildTrack(true, 4) };
  const cars = [];
  const R = rng(17);
  for (let i = 0; i < 26; i++) {
    cars.push({ z: R() * tracks.hilly.length * SEG, off: (R() < 0.5 ? -1 : 1) * (0.2 + R() * 0.45), speed: MAX_SPEED * (0.25 + R() * 0.35), kind: i % 3 });
  }
  return { ctx, tracks, cars, pos: 0, x: 0, speed: MAX_SPEED * 0.55, steer: 0, idle: 99, skyOff: 0, hillOff: 0, tags: {}, lastEx: null, laneT: 0.35, laneTimer: 0 };
}

function stepCar(st, p, ctx, dt, segs) {
  const n = segs.length;
  const len = n * SEG;
  const keys = ctx.keys;
  const manual = anyKey(keys, [KEYS.up, KEYS.down, KEYS.left, KEYS.right]);
  st.idle = manual ? 0 : st.idle + dt;
  let steer = keyAxis(keys, KEYS.left, KEYS.right);
  let gas = keyAxis(keys, KEYS.down, KEYS.up);
  const playerZ = CAM_H * st.camDepth;
  const pseg = segs[Math.floor((st.pos + playerZ) / SEG) % n];
  if (p.autopilot && st.idle > 2.5) {
    // stay in a lane, counter-steer the curves, change lanes around slower cars
    st.laneTimer -= dt;
    for (const c of st.useCars ? st.cars : []) {
      const dz = (c.z - st.pos - playerZ + len) % len;
      if (dz > 0 && dz < SEG * 25 && Math.abs(c.off - st.laneT) < 0.35 && st.laneTimer <= 0) {
        st.laneT = st.laneT > 0 ? -0.4 : 0.4;
        st.laneTimer = 2;
      }
    }
    const want = st.laneT + pseg.curve * 0.06;
    steer = clamp((want - st.x) * 4, -1, 1);
    gas = st.speed < MAX_SPEED * 0.85 ? 1 : 0;
  }
  const sp = st.speed / MAX_SPEED;
  const dx = dt * 2 * sp;
  st.x += steer * dx;
  st.x -= dx * sp * pseg.curve * 0.3 * p.centrifugal; // centrifugal force on curves
  st.steer += (steer - st.steer) * Math.min(1, dt * 8);
  if (gas > 0) st.speed += (MAX_SPEED / 5) * dt;
  else if (gas < 0) st.speed -= MAX_SPEED * dt;
  else st.speed -= (MAX_SPEED / 5) * dt;
  if ((st.x < -1 || st.x > 1) && st.speed > MAX_SPEED / 4) st.speed -= (MAX_SPEED / 2) * dt;
  st.x = clamp(st.x, -2.5, 2.5);
  st.speed = clamp(st.speed, 0, MAX_SPEED * p.topSpeed);
  // traffic
  if (st.useCars) {
    for (const c of st.cars) {
      c.z = (c.z + c.speed * dt) % len;
      const dz = (c.z - st.pos - playerZ + len) % len;
      if (dz < SEG * 0.8 && Math.abs(c.off - st.x) < 0.42 && st.speed > c.speed) {
        st.speed = c.speed * 0.8;
        st.pos = (c.z - playerZ - SEG * 0.8 + len) % len;
      }
    }
  }
  st.pos = (st.pos + st.speed * dt + len) % len;
  st.skyOff += 0.001 * pseg.curve * sp * dt * 60;
  st.hillOff += 0.002 * pseg.curve * sp * dt * 60;
}

function bindFn(params, ctx) {
  TEST = !!ctx.testMode;
  if (!S || S.ctx !== ctx) S = newState(ctx);
  const st = S;
  const ex = ctx.example;
  const W = ctx.width;
  const H = ctx.height;
  if (st.lastEx !== ex) {
    for (const t of Object.values(st.tags)) t.remove();
    st.tags = { hud: overlayTag(ctx, 'right:8px;bottom:8px') };
    if (ex === 'flat') {
      st.tags.plan = overlayTag(ctx, 'right:8px;top:48px');
      st.tags.plan.textContent = 'the same road, unrolled from above';
    }
    st.lastEx = ex;
  }
  const segs = ex === 'flat' ? st.tracks.flat : st.tracks.hilly;
  st.useCars = ex !== 'flat';
  const fov = (params.fov * Math.PI) / 180;
  st.camDepth = 1 / Math.tan(fov / 2);
  const dt = Math.min(ctx.dt, 1 / 20);
  stepCar(st, params, ctx, dt, segs);

  const n = segs.length;
  const len = n * SEG;
  const f = (W / 2) * st.camDepth; // focal length in pixels
  const hy = H * params.horizon;
  const playerZ = CAM_H * st.camDepth;
  const baseI = Math.floor(st.pos / SEG) % n;
  const basePct = (st.pos % SEG) / SEG;
  const pI = Math.floor((st.pos + playerZ) / SEG) % n;
  const pPct = ((st.pos + playerZ) % SEG) / SEG;
  const playerY = segs[pI].y1 + (segs[pI].y2 - segs[pI].y1) * pPct;
  const camY = playerY + CAM_H * params.camHeight;
  const camX = st.x * ROAD;

  const segA = new Float32Array(NSEG * 4);
  const segB = new Float32Array(NSEG * 4);
  const clipOf = new Float32Array(NSEG);
  const projOf = [];
  let maxy = H;
  let x = 0;
  let dxs = -(segs[baseI].curve * basePct);
  let count = 0;
  const drawN = Math.round(params.draw);
  for (let k = 0; k < drawN; k++) {
    const s = segs[(baseI + k) % n];
    const looped = s.i < baseI;
    const cz = st.pos - (looped ? len : 0);
    const z1 = s.i * SEG - cz;
    const z2 = (s.i + 1) * SEG - cz;
    const p1 = { z: z1, x: W / 2 + (f * (x - camX)) / z1, y: hy - (f * (s.y1 - camY)) / z1, w: (f * ROAD) / z1 };
    const p2 = { z: z2, x: W / 2 + (f * (x + dxs - camX)) / z2, y: hy - (f * (s.y2 - camY)) / z2, w: (f * ROAD) / z2 };
    // curves: every segment shifts the road sideways a bit more than the last (dx += curve)
    x += dxs;
    dxs += s.curve;
    projOf[k] = { p1, clip: maxy, s };
    if (z1 <= st.camDepth || p2.y >= p1.y || p2.y >= maxy) continue; // behind us, facing away, or hidden by a hill
    const fog = Math.exp(-Math.pow(k / drawN, 2) * params.fogAmt * 5);
    const parity = Math.floor(s.i / RUMBLE) % 2;
    segA.set([p1.y, p1.x, p1.w, parity + 2 * Math.round(s.lamp * 15) + (s.i % 12) * 32], count * 4);
    segB.set([p2.y, p2.x, p2.w, fog], count * 4);
    clipOf[k] = maxy;
    count++;
    maxy = p2.y;
    if (count >= NSEG) break;
  }

  // sprites (scenery + traffic), far -> near, each clipped by the hill line at its segment
  const list = [];
  const sprite = (k, kind, offset, extraZ, flip, emissive) => {
    const pr = projOf[k];
    if (!pr || pr.p1.z <= st.camDepth) return;
    const fr = SPR[kind];
    const scale = f / pr.p1.z;
    const hw = WORLD_H[kind] ?? WORLD_H.van;
    const h = hw * scale;
    const w = (h * fr[2]) / fr[3];
    const sx = pr.p1.x + scale * offset * ROAD;
    if (sx + w < 0 || sx - w > W || h < 1) return;
    const ax = offset < 0 ? -1 : 0; // left-side sprites hang left of their anchor
    const fogK = Math.exp(-Math.pow(k / drawN, 2) * params.fogAmt * 5);
    list.push({ z: pr.p1.z + extraZ, rect: [kind.startsWith('van') ? sx - w / 2 : sx + ax * w, pr.p1.y - h, w, h], uv: [fr[0] / SHEET_W, fr[1] / SHEET_H, (fr[0] + fr[2]) / SHEET_W, (fr[1] + fr[3]) / SHEET_H], info: [pr.clip, fogK, emissive, flip ? 1 : 0] });
  };
  for (let k = drawN - 1; k > 0; k--) {
    const pr = projOf[k];
    if (!pr) continue;
    for (const sp of pr.s.sprites) {
      if (sp.k === 'lamp' && ex !== 'night') continue;
      sprite(k, sp.k, sp.off, 0, sp.k === 'lamp' && sp.off < 0, sp.k === 'lamp' || sp.k.startsWith('board') ? 1 : 0);
    }
  }
  if (st.useCars) {
    for (const c of st.cars) {
      let dz = c.z - st.pos;
      if (dz < 0) dz += len;
      const k = Math.floor((c.z / SEG) % n) - baseI;
      const kk = (k + n) % n;
      if (kk <= 0 || kk >= drawN) continue;
      sprite(kk, 'van' + c.kind, c.off, -1, false, 0);
    }
  }
  list.sort((a, b) => b.z - a.z);
  const keep = list.slice(-NSPR);
  const sprRect = new Float32Array(NSPR * 4);
  const sprUV = new Float32Array(NSPR * 4);
  const sprInfo = new Float32Array(NSPR * 4);
  keep.forEach((s, i) => {
    sprRect.set(s.rect, i * 4);
    sprUV.set(s.uv, i * 4);
    sprInfo.set(s.info, i * 4);
  });

  // player car (screen space)
  const cw = Math.min(W * 0.3, H * 0.55);
  const chh = (cw * 44) / 80;
  const frame = st.steer < -0.3 ? 0 : st.steer > 0.3 ? 2 : 1;
  const bounce = st.speed > 100 && (Math.abs(st.x) > 1 ? Math.sin(ctx.time * 40) > 0 : Math.sin(ctx.time * 25) > 0.6) ? chh * 0.025 : 0;
  const car = SPR['car' + frame];
  const kmh = Math.round((st.speed / MAX_SPEED) * 290);
  if (st.tags.hud) st.tags.hud.textContent = `${kmh} km/h · ${count} visible segments${st.idle > 2.5 && params.autopilot ? ' · autopilot' : ''}`;
  return {
    view: [W / 2, hy, f, count],
    segA,
    segB,
    sprRect,
    sprUV,
    sprInfo,
    carRect: [W / 2 - cw / 2, H - chh - H * 0.03 + bounce, cw, chh],
    carUV: [car[0] / SHEET_W, car[1] / SHEET_H, (car[0] + car[2]) / SHEET_W, (car[1] + car[3]) / SHEET_H],
    bg: [st.skyOff, st.hillOff, (playerY / SEG) * 0.002, st.speed / MAX_SPEED],
  };
}

const FRAG = /* wgsl */ `
const NSEG: i32 = ${NSEG};
const NSPR: i32 = ${NSPR};

// smallest k with segB[k].y (far edge) <= y; visible segments are sorted bottom -> top
fn findSeg(y: f32) -> i32 {
  let count = i32(u.view.w);
  var lo = 0;
  var hi = count;
  for (var it = 0; it < 9; it++) {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2;
    if (u.segB[mid].x <= y) { hi = mid; } else { lo = mid + 1; }
  }
  return lo;
}

fn skyDay(p: vec2f) -> vec3f {
  let H = u.resolution.y;
  let hz = u.view.y + u.bg.z * H;
  let t = clamp(p.y / max(hz, 1.0), 0.0, 1.0);
  var c = mix(vec3f(0.18, 0.42, 0.85), vec3f(0.72, 0.86, 0.98), t);
  let x = p.x / u.resolution.x;
  // clouds (slow layer)
  let cq = vec2f(x * 3.0 + u.bg.x * 3.0 + u.time * 0.01, p.y / H * 8.0);
  let cl = smoothstep(0.55, 0.75, fbm(cq, 4) * 0.5 + 0.5) * smoothstep(0.65, 0.3, t);
  c = mix(c, vec3f(1.0), cl * 0.9);
  // far mountains
  let mx = x * 2.0 + u.bg.x * 2.0;
  let mh = hz - H * (0.11 + 0.07 * (0.5 + 0.5 * sin(mx * 6.0 + 1.0)) * (0.6 + 0.4 * sin(mx * 13.0)) + 0.015 * sin(mx * 41.0));
  if (p.y > mh) {
    var m = mix(vec3f(0.42, 0.5, 0.7), vec3f(0.55, 0.62, 0.8), smoothstep(mh, hz, p.y));
    m = mix(m, vec3f(0.95), smoothstep(mh + H * 0.015, mh + H * 0.002, p.y) * 0.7);
    c = m;
  }
  // near hills with trees
  let hx = x * 3.0 + u.bg.y * 3.0;
  let hh = hz - H * (0.04 + 0.025 * sin(hx * 5.0) + 0.01 * sin(hx * 17.0));
  let trees = H * 0.018 * smoothstep(0.2, 1.0, sin(hx * 80.0) * 0.5 + 0.5) * step(0.4, valueNoise(vec2f(hx * 8.0, 1.0)));
  if (p.y > hh - trees) { c = mix(vec3f(0.18, 0.5, 0.22), vec3f(0.12, 0.38, 0.18), smoothstep(hh, hz, p.y)); }
  return c;
}

fn skyNeon(p: vec2f) -> vec3f {
  let H = u.resolution.y;
  let W = u.resolution.x;
  let hz = u.view.y + u.bg.z * H;
  let t = clamp(p.y / max(hz, 1.0), 0.0, 1.0);
  var c = mix(vec3f(0.03, 0.01, 0.1), vec3f(0.55, 0.08, 0.45), t * t);
  c = mix(c, vec3f(1.0, 0.45, 0.35), smoothstep(0.75, 1.0, t) * 0.6);
  // stars
  let sg = floor(p / 3.0);
  let star = step(0.997, hash21(sg)) * (0.5 + 0.5 * sin(u.time * 3.0 + hash21(sg + 3.0) * 20.0)) * (1.0 - t);
  c += vec3f(star);
  // striped synthwave sun
  let sc = vec2f(W * 0.5 - u.bg.x * W * 0.6, hz - H * 0.12);
  let sd = length(p - sc);
  let sr = H * 0.17;
  let stripes = step(0.0, sin((p.y - sc.y) / H * 140.0 - u.time * 2.0) + (sc.y - p.y) / sr * 2.0 + 0.3);
  let sunCol = mix(vec3f(1.0, 0.85, 0.3), vec3f(1.0, 0.25, 0.55), clamp((p.y - sc.y + sr) / (2.0 * sr), 0.0, 1.0));
  c = mix(c, sunCol, (1.0 - smoothstep(sr - 1.0, sr + 1.0, sd)) * stripes);
  c += vec3f(1.0, 0.3, 0.5) * exp(-max(sd - sr, 0.0) / (H * 0.08)) * 0.35;
  // wireframe mountains
  let mx = p.x / W * 2.0 + u.bg.x * 2.0;
  let tri = abs(fract(mx * 3.0) - 0.5) * 2.0;
  let mh = hz - H * (0.04 + 0.1 * (1.0 - tri) * (0.4 + 0.6 * hash11(floor(mx * 3.0))));
  if (p.y > mh) {
    c = vec3f(0.06, 0.0, 0.12);
    let gl = abs(fract((p.y - mh) / (H * 0.02)) - 0.5);
    c += vec3f(0.2, 0.05, 0.5) * smoothstep(0.08, 0.0, gl) * 0.6;
  }
  c += vec3f(1.0, 0.2, 0.8) * smoothstep(2.0, 0.0, abs(p.y - mh)) * step(mh, hz);
  // city skyline
  let bx = floor((p.x + u.bg.y * W * 1.4) / (W * 0.035));
  let bh = hz - H * (0.02 + 0.09 * hash11(bx) * hash11(bx + 7.0));
  if (p.y > bh) {
    c = vec3f(0.02, 0.01, 0.05);
    let wq = floor(vec2f(p.x + u.bg.y * W * 1.4, p.y) / vec2f(W * 0.007, H * 0.012));
    let lit = step(0.72, hash21(wq)) * step(0.3, fract((p.x + u.bg.y * W * 1.4) / (W * 0.007))) * step(0.35, fract(p.y / (H * 0.012)));
    c += mix(vec3f(1.0, 0.75, 0.3), vec3f(0.3, 0.9, 1.0), hash21(wq + 5.0)) * lit * 0.8;
  }
  return c;
}

fn roadAt(p: vec2f, k: i32, ex: i32) -> vec3f {
  let a = u.segA[k];
  let b = u.segB[k];
  let t = (p.y - b.x) / max(a.x - b.x, 0.0001);
  let cx = mix(b.y, a.y, t);
  let w = mix(b.z, a.z, t);
  let fog = b.w;
  let parity = fmod(a.w, 2.0);
  let lamp = fmod(floor(a.w * 0.5), 16.0) / 15.0;
  let segId = floor(a.w / 32.0);
  let d = (p.x - cx) / w;           // -1..1 across the road
  let ad = abs(d);
  let px1 = 1.0 / w;                // one pixel in road units (for anti-aliasing)
  let rumbleW = 0.16;
  let laneW = 0.035;
  let laneLine = (1.0 - smoothstep(laneW - px1, laneW + px1, abs(ad - 0.333))) * parity;
  let roadM = 1.0 - smoothstep(1.0 - px1, 1.0 + px1, ad);
  let rumbleM = (1.0 - smoothstep(1.0 + rumbleW - px1, 1.0 + rumbleW + px1, ad)) * (1.0 - roadM);
  var grass: vec3f;
  var asphalt: vec3f;
  var rumble: vec3f;
  var lane: vec3f;
  var fogCol: vec3f;
  if (ex == 2) {
    grass = mix(vec3f(0.05, 0.0, 0.1), vec3f(0.08, 0.01, 0.14), parity);
    // neon grid on the ground: one bright line per segment boundary + lines along the road
    let gx = abs(fract(d * 0.5) - 0.5);
    grass += vec3f(0.5, 0.1, 0.9) * (smoothstep(0.06, 0.0, abs(t - 0.0)) + smoothstep(0.012 + px1, 0.0, gx)) * 0.6;
    asphalt = mix(vec3f(0.07, 0.06, 0.1), vec3f(0.09, 0.07, 0.12), parity);
    rumble = mix(vec3f(1.0, 0.2, 0.75), vec3f(0.2, 0.9, 1.0), parity) * 1.2;
    lane = vec3f(0.4, 1.0, 1.0);
    fogCol = vec3f(0.35, 0.05, 0.35);
    // street lamps & headlights
    asphalt += vec3f(1.0, 0.8, 0.5) * lamp * 0.25 * (1.0 - smoothstep(0.3, 1.0, ad));
    grass += vec3f(1.0, 0.7, 0.4) * lamp * 0.12 * smoothstep(1.2, 1.0, ad);
    let hl = exp(-pow(abs(p.x - u.resolution.x * 0.5) / (u.resolution.y - p.y + 40.0), 2.0) * 3.0);
    let near = smoothstep(u.view.y + (u.resolution.y - u.view.y) * 0.15, u.resolution.y, p.y);
    asphalt += vec3f(0.9, 0.85, 0.7) * hl * near * 0.35;
  } else {
    grass = mix(vec3f(0.06, 0.6, 0.06), vec3f(0.0, 0.54, 0.0), parity);
    asphalt = mix(vec3f(0.42, 0.42, 0.42), vec3f(0.4, 0.4, 0.41), parity);
    rumble = mix(vec3f(0.95, 0.95, 0.95), vec3f(0.8, 0.15, 0.12), parity);
    lane = vec3f(0.9);
    fogCol = vec3f(0.72, 0.86, 0.98);
  }
  if (ex == 0 && u.showSegs > 0.5) {
    // every segment in its own color + its near edge drawn as a line
    let hue = hsv2rgb(vec3f(fract(segId / 12.0 + 0.05), 0.55, 0.95));
    asphalt = mix(asphalt, hue * 0.75, 0.75);
    grass = mix(grass, hue * 0.5, 0.35);
  }
  var col = mix(grass, rumble, rumbleM);
  col = mix(col, mix(asphalt, lane, laneLine), roadM);
  if (ex == 0 && u.showSegs > 0.5) {
    col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.5, 1.5, abs(p.y - a.x))) * 0.7);
  }
  return mix(fogCol, col, fog);
}

fn plan(p: vec2f) -> vec4f {
  // top-down "unrolled" plot of the visible segments: x = lateral offset, y = segment number
  let W = u.resolution.x;
  let H = u.resolution.y;
  let pw = min(W * 0.18, 200.0);
  let ph = min(H * 0.42, 240.0);
  let o = vec2f(W - pw - 10.0, 74.0);
  let q = (p - o) / vec2f(pw, ph);
  if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) { return vec4f(0.0); }
  let count = u.view.w;
  if (count < 2.0) { return vec4f(0.0); }
  let kf = (1.0 - q.y) * (count - 1.0);
  let k0 = i32(floor(kf));
  let k1 = min(k0 + 1, i32(count) - 1);
  let l0 = (u.segB[k0].y - u.view.x) / u.segB[k0].z;
  let l1 = (u.segB[k1].y - u.view.x) / u.segB[k1].z;
  let lat = mix(l0, l1, fract(kf));
  let xr = (q.x - 0.5) * 16.0;
  var c = vec3f(0.06, 0.08, 0.12);
  let dr = abs(xr - lat);
  let pxw = 16.0 / pw;
  c = mix(c, vec3f(0.35, 0.37, 0.42), 1.0 - smoothstep(1.0 - pxw, 1.0 + pxw, dr));
  let segHue = hsv2rgb(vec3f(fract(floor(u.segA[k0].w / 32.0) / 12.0 + 0.05), 0.55, 0.95));
  c = mix(c, segHue * 0.8, (1.0 - smoothstep(1.0 - pxw, 1.0 + pxw, dr)) * u.showSegs * 0.6);
  // the car at the bottom
  let carP = vec2f(0.5 * pw, ph - 6.0);
  c = mix(c, vec3f(1.0, 0.3, 0.3), 1.0 - smoothstep(3.0, 4.0, length((p - o) - carP)));
  let fb = sdBox(p - o - vec2f(pw, ph) * 0.5, vec2f(pw, ph) * 0.5);
  c = mix(c, vec3f(0.55, 0.6, 0.75), 1.0 - smoothstep(0.5, 1.5, abs(fb)));
  return vec4f(c, 0.92);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  let p = uv * u.resolution;
  var col: vec3f;
  if (ex == 2) { col = skyNeon(p); } else { col = skyDay(p); }
  let k = findSeg(p.y);
  if (k < i32(u.view.w)) { col = roadAt(p, k, ex); }
  // sprites, far -> near; each one is hidden below the hill line of its own segment
  for (var i = 0; i < NSPR; i++) {
    let r = u.sprRect[i];
    if (r.z <= 0.0) { continue; }
    var q = (p - r.xy) / r.zw;
    if (q.x < 0.0 || q.y < 0.0 || q.x >= 1.0 || q.y >= 1.0) { continue; }
    let inf = u.sprInfo[i];
    if (p.y > inf.x) { continue; }
    if (inf.w > 0.5) { q.x = 1.0 - q.x; }
    let s = TEXN(sheet, mix(u.sprUV[i].xy, u.sprUV[i].zw, q));
    if (s.a < 0.5) { continue; }
    var sc = s.rgb;
    var fogCol = vec3f(0.72, 0.86, 0.98);
    if (ex == 2) {
      // night: silhouettes, except emissive signs and lamps
      sc = mix(sc * vec3f(0.12, 0.08, 0.22), sc * 1.3, inf.z);
      fogCol = vec3f(0.35, 0.05, 0.35);
    }
    col = mix(fogCol, sc, inf.y);
  }
  // lamp glows at night (additive halos around the lamp heads)
  if (ex == 2) {
    for (var i = 0; i < NSPR; i++) {
      let r = u.sprRect[i];
      let inf = u.sprInfo[i];
      if (r.z <= 0.0 || inf.z < 0.5 || r.w / r.z < 2.0) { continue; }
      var lx = r.x + r.z * 0.25;
      if (inf.w > 0.5) { lx = r.x + r.z * 0.75; }
      let lp = vec2f(lx, r.y + r.w * 0.1);
      let d = length(p - lp) / max(r.w * 0.12, 1.0);
      col += vec3f(1.0, 0.7, 0.35) * (exp(-d * 2.0) * 0.5 + exp(-d * 8.0) * 0.8) * inf.y;
    }
  }
  // player car with a soft shadow
  let cr = u.carRect;
  let sd = length((p - vec2f(cr.x + cr.z * 0.5, cr.y + cr.w * 0.95)) / vec2f(cr.z * 0.55, cr.w * 0.14));
  col *= 1.0 - 0.45 * smoothstep(1.0, 0.5, sd);
  let cq = (p - cr.xy) / cr.zw;
  if (cq.x >= 0.0 && cq.y >= 0.0 && cq.x < 1.0 && cq.y < 1.0) {
    let s = TEXN(sheet, mix(u.carUV.xy, u.carUV.zw, cq));
    var cc = s.rgb;
    if (ex == 2) { cc = cc * vec3f(0.45, 0.4, 0.6) + vec3f(1.0, 0.1, 0.1) * step(0.9, s.r) * step(s.g, 0.4); }
    col = mix(col, cc, step(0.5, s.a));
  }
  if (ex == 2) {
    // tail-light glow
    let tl1 = length(p - vec2f(cr.x + cr.z * 0.19, cr.y + cr.w * 0.55));
    let tl2 = length(p - vec2f(cr.x + cr.z * 0.81, cr.y + cr.w * 0.55));
    col += vec3f(1.0, 0.15, 0.15) * (exp(-tl1 / (cr.z * 0.05)) + exp(-tl2 / (cr.z * 0.05))) * 0.6;
  }
  if (ex == 0) {
    let pl = plan(p);
    col = mix(col, pl.rgb, pl.a);
  }
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: '↑/↓ (W/S) throttle & brake · ←/→ steer · let go for autopilot',
  keys: true,
  examples: [
    {
      id: 'flat',
      label: 'Flat curves (how it works)',
      kind: 'Abstract',
      note: 'Each colored band is one road <b>segment</b>. A segment only stores a <i>curve</i> number. While drawing from near to far, every segment is shifted sideways by <code>dx</code>, and <code>dx</code> grows by the segment’s curve — so offsets add up into a smooth bend. The inset shows the same segments unrolled from above.',
      params: { draw: 110, horizon: 0.5, showSegs: true, fogAmt: 0.6 },
    },
    {
      id: 'hills',
      label: 'Hills & curves',
      kind: 'In a game',
      note: 'Segments also store a height. Projecting it moves rows up and down, and a <b>clip line</b> (the highest row drawn so far) hides whatever is behind a crest — the classic OutRun hill. Trees, signs and traffic are sprites scaled by 1/distance and clipped by the same line.',
      params: { draw: 150, horizon: 0.5, showSegs: false, fogAmt: 0.5 },
    },
    {
      id: 'night',
      label: 'Night drive / neon',
      kind: 'Art style',
      note: 'The same road with a synthwave palette: glowing rumble strips, a neon grid, a striped sun, emissive billboards and street lamps that light pools on the asphalt, plus headlights and tail-light glow.',
      params: { draw: 150, horizon: 0.5, showSegs: false, fogAmt: 0.7 },
    },
  ],
  controls: [
    { type: 'slider', key: 'fov', label: 'Field of view (°)', min: 60, max: 140, step: 1, value: 100, help: 'Sets cameraDepth = 1/tan(fov/2). Wider = more speed sensation.' },
    { type: 'slider', key: 'camHeight', label: 'Camera height', min: 0.4, max: 3, step: 0.01, value: 1, help: 'Low cameras make hills more dramatic.' },
    { type: 'slider', key: 'horizon', label: 'Horizon height', min: 0.3, max: 0.7, step: 0.01, value: 0.5 },
    { type: 'slider', key: 'draw', label: 'Draw distance (segments)', min: 20, max: NSEG, step: 1, value: 150, help: 'How many segments are projected each frame.' },
    { type: 'slider', key: 'fogAmt', label: 'Fog', min: 0, max: 2, step: 0.01, value: 0.5 },
    { type: 'slider', key: 'centrifugal', label: 'Curve push (centrifugal)', min: 0, max: 2, step: 0.01, value: 1, help: 'Curves push the car outward; you must steer against them.' },
    { type: 'slider', key: 'topSpeed', label: 'Top speed', min: 0.2, max: 1.4, step: 0.01, value: 1 },
    { type: 'slider', key: 'pixel', label: 'Pixel size', min: 1, max: 4, step: 1, value: 2, help: 'Arcade boards ran at ~320×224.' },
    { type: 'toggle', key: 'showSegs', label: 'Color each segment', value: true, showFor: ['flat'] },
    { type: 'toggle', key: 'autopilot', label: 'Autopilot when idle', value: true },
  ],
  uniforms: {
    fov: 'f32', camHeight: 'f32', horizon: 'f32', draw: 'f32', fogAmt: 'f32', showSegs: 'f32',
    view: 'vec4f', carRect: 'vec4f', carUV: 'vec4f', bg: 'vec4f',
    segA: `array<vec4f, ${NSEG}>`, segB: `array<vec4f, ${NSEG}>`,
    sprRect: `array<vec4f, ${NSPR}>`, sprUV: `array<vec4f, ${NSPR}>`, sprInfo: `array<vec4f, ${NSPR}>`,
  },
  include: ['hash', 'noise', 'sdf', 'color'],
  renderScale: (p) => 1 / Math.max(TEST ? 3 : 1, p.pixel || 1),
  textures: { sheet: { source: async () => paintSheet(), filter: 'nearest' } },
  bind: bindFn,
  code: FRAG,
  about: {
    summary: 'OutRun, Pole Position and Top Gear drew fast 3D roads with no 3D at all: the road is a list of short segments, each with a curve and a height, projected from near to far every frame.',
    what: `<p>A road that curves and climbs over hills, roadside palms, billboards and traffic — all flat 2D sprites. The road itself is a
      list of about 2,000 segments; only the next ~150 are projected each frame.</p>`,
    how: `<ol>
      <li>The track is a 1D array of segments (200 units long). Each stores a <b>curve</b> amount and a <b>height</b>.</li>
      <li>Every frame, walk the next N segments from the camera. Project each edge with <code>screenY = horizon − f·(y − camY)/z</code>,
        <code>screenX = center + f·(x − camX)/z</code>, <code>width = f·roadWidth/z</code>.</li>
      <li><b>Curves</b> are faked: keep an offset <code>x</code> and its rate <code>dx</code>. For each segment: <code>x += dx; dx += curve</code>.
        Constant curve → offsets grow quadratically → a smooth bend. No angles or rotation needed.</li>
      <li><b>Hills</b>: keep a clip line (the highest row drawn so far). Draw segments near→far; a segment whose far edge is below the clip line is hidden behind a crest.</li>
      <li>Alternate colors every 3 segments for grass, rumble strips and lane dashes — this striping is what sells the speed.</li>
      <li><b>Sprites</b> are drawn far→near at their segment’s scale, clipped by that segment’s clip line. The background scrolls sideways with the curve.</li>
      <li>Here JavaScript does the projection (cheap: 150 segments) and uploads the visible list; the fragment shader binary-searches it for each pixel row.</li>
    </ol>`,
    uses: [
      { title: 'Arcade racers', text: 'OutRun, Pole Position, Hang-On, Super Hang-On, Top Gear, Lotus Turbo Challenge, Road Rash.' },
      { title: 'Modern retro', text: 'Slipstream, Horizon Chase and many jam games use exactly this technique for its look and speed.' },
      { title: 'Beyond cars', text: 'Space Harrier-style shooters, endless runners, and tunnel/corridor sections in 2D games.' },
    ],
    try: [
      'On <b>Flat curves</b>, watch the inset: the road bends only because each segment’s sideways offset keeps growing.',
      'Lower <i>Draw distance</i> to 30 — the road pops in at the horizon, like on old hardware.',
      'On <b>Hills & curves</b>, drop <i>Camera height</i> to 0.4: crests now hide long stretches of road and traffic.',
      'Raise <i>Curve push</i> to 2 and take the wheel: you have to steer into every bend.',
      'Set <i>Pixel size</i> to 1 and <i>Field of view</i> to 140 for a modern, hyper-speed look.',
    ],
    ask: [
      'an OutRun-style pseudo-3D road with curves and hills',
      'roadside sprites scaled by distance and clipped by hills',
      'a synthwave night drive with neon rumble strips',
      'alternating rumble strips and lane markers for speed sensation',
      'traffic cars and lane changes on a pseudo-3D road',
    ],
    perf: `<p>CPU: projecting ~150 segments and a few dozen sprites — microseconds. GPU: one binary search (≤ 8 steps) per pixel plus a loop over 40 sprite rectangles. Trivial at 1080p.</p>`,
    api: `<p>Runs on WebGPU and WebGL2: a single fragment shader fed by uniform arrays (≈ 7 KB per frame). A WebGPU-only version could upload the projected
      segments to a storage buffer and draw them as instanced trapezoids instead.</p>`,
    code: [
      {
        title: 'Projecting the segments (JavaScript, from this scene)',
        lang: 'js',
        src: `let maxy = H, x = 0, dx = -(segs[base].curve * basePct);
for (let k = 0; k < drawN; k++) {
  const s = segs[(base + k) % n];
  const z1 = s.i * SEG - camZ, z2 = (s.i + 1) * SEG - camZ;
  const p1 = { x: W/2 + f * (x - camX) / z1,      y: hy - f * (s.y1 - camY) / z1, w: f * ROAD / z1 };
  const p2 = { x: W/2 + f * (x + dx - camX) / z2, y: hy - f * (s.y2 - camY) / z2, w: f * ROAD / z2 };
  x += dx;  dx += s.curve;                         // curves = accumulated offsets
  if (z1 <= camDepth || p2.y >= p1.y || p2.y >= maxy) continue;  // hidden by a hill
  visible.push(p1, p2);
  maxy = p2.y;                                     // the new clip line
}`,
      },
      {
        title: 'Shader: which segment covers this row?',
        lang: 'wgsl',
        src: `var lo = 0; var hi = count;              // segments sorted bottom -> top
for (var it = 0; it < 9; it++) {
  if (lo >= hi) { break; }
  let mid = (lo + hi) / 2;
  if (u.segB[mid].x <= y) { hi = mid; } else { lo = mid + 1; }
}
let t = (y - farY) / (nearY - farY);          // position inside the trapezoid
let d = (p.x - mix(farX, nearX, t)) / mix(farW, nearW, t); // -1..1 across the road`,
      },
    ],
    links: [
      { title: 'Lou’s Pseudo 3D Page', url: 'http://www.extentofthejam.com/pseudo/', note: 'the definitive explanation' },
      { title: 'Jake Gordon — Javascript Racer', url: 'https://jakesgordon.com/writing/javascript-racer/', note: 'a classic step-by-step implementation' },
    ],
  },
});

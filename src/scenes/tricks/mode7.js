import { shaderScene } from '../../core/shaderscene.js';
import { getAtlas, makeCanvas, rng } from '../../core/assets.js';
import { overlayTag, keyAxis, KEYS, anyKey, fbm2, vnoise, mipPack, MIP_WGSL, clamp, wrapAngle } from './_shared.js';

// Mode 7: every screen row below the horizon is a straight line through a flat 2D map texture.
// The per-pixel inverse projection runs in a portable fragment shader (WebGPU + WebGL2); the kart /
// airship physics, autopilot and billboard projection run in JavaScript and arrive via bind().

const TRACK_N = 1024; // track map size (texels = world units)
const WORLD_N = 1024; // airship world map size
const NS = 24; // billboard slots

// ------------------------------------------------------------------------------------ track bake

const TRACK_PTS = [
  [0.2, 0.78], [0.13, 0.5], [0.2, 0.22], [0.42, 0.13], [0.55, 0.3], [0.68, 0.17],
  [0.87, 0.24], [0.88, 0.55], [0.72, 0.66], [0.64, 0.86], [0.42, 0.87], [0.36, 0.66],
];
const ROAD_W = 64;
const KERB = 7;

function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return [0, 1].map((k) =>
    0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3),
  );
}

function trackCenterline() {
  const P = TRACK_PTS.map(([x, y]) => [x * TRACK_N, y * TRACK_N]);
  const n = P.length;
  const dense = [];
  for (let i = 0; i < n; i++) {
    for (let s = 0; s < 64; s++) dense.push(catmull(P[(i - 1 + n) % n], P[i], P[(i + 1) % n], P[(i + 2) % n], s / 64));
  }
  // resample by arc length every 4 texels
  const out = [dense[0]];
  let acc = 0;
  for (let i = 1; i <= dense.length; i++) {
    const a = dense[i - 1];
    const b = dense[i % dense.length];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    acc += d;
    if (acc >= 4) {
      out.push(b);
      acc = 0;
    }
  }
  return out;
}

let bakePromise = null;
function bake() {
  if (bakePromise) return bakePromise;
  bakePromise = (async () => {
    const atlas = await getAtlas();
    const center = trackCenterline();
    const w = paintWorld();
    return { atlas, center, track: mipPack(paintTrack(center)), world: mipPack(w.canvas), worldTrees: w.trees, props: paintProps() };
  })();
  return bakePromise;
}

function pathOf(center) {
  const p = new Path2D();
  p.moveTo(center[0][0], center[0][1]);
  for (const c of center) p.lineTo(c[0], c[1]);
  p.closePath();
  return p;
}

const BOOST_AT = [0.18, 0.62]; // fractions along the lap
const COIN_GROUPS = [0.08, 0.3, 0.45, 0.75, 0.9];

function paintTrack(center) {
  const N = TRACK_N;
  const c = makeCanvas(N, N);
  const g = c.getContext('2d');
  const R = rng(7);
  // mowed lawn stripes + speckle
  for (let y = 0; y < N; y += 32) {
    g.fillStyle = (y / 32) % 2 ? '#4f9a3c' : '#5aa846';
    g.fillRect(0, y, N, 32);
  }
  for (let i = 0; i < 26000; i++) {
    g.fillStyle = R() < 0.5 ? 'rgba(30,70,30,0.35)' : 'rgba(150,210,110,0.3)';
    g.fillRect((R() * N) | 0, (R() * N) | 0, 2, 2);
  }
  // flower patches
  for (let k = 0; k < 40; k++) {
    const cx = R() * N;
    const cy = R() * N;
    const col = ['#ffd75e', '#ff8fb1', '#ffffff', '#a2c8ff'][k % 4];
    for (let i = 0; i < 40; i++) {
      g.fillStyle = col;
      g.fillRect((cx + (R() - 0.5) * 40) | 0, (cy + (R() - 0.5) * 40) | 0, 2, 2);
    }
  }
  // a pond in the infield
  g.fillStyle = '#d9c38a';
  g.beginPath();
  g.ellipse(N * 0.4, N * 0.48, 92, 62, 0.3, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = '#3d7fc8';
  g.beginPath();
  g.ellipse(N * 0.4, N * 0.48, 82, 53, 0.3, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = '#5fa3e6';
  g.beginPath();
  g.ellipse(N * 0.39, N * 0.47, 60, 36, 0.3, 0, Math.PI * 2);
  g.fill();
  // forest border so the map edge blends into the procedural forest outside
  for (let i = 0; i < 2600; i++) {
    const side = i % 4;
    const t = R() * N;
    const d = Math.pow(R(), 1.6) * 70;
    const x = side === 0 ? d : side === 1 ? N - d : t;
    const y = side === 2 ? d : side === 3 ? N - d : t;
    g.fillStyle = R() < 0.5 ? '#1c4d24' : '#2e6b31';
    g.beginPath();
    g.arc(x, y, 6 + R() * 8, 0, Math.PI * 2);
    g.fill();
  }
  const path = pathOf(center);
  g.lineJoin = 'round';
  g.lineCap = 'round';
  // sand run-off
  g.strokeStyle = '#d8bf86';
  g.lineWidth = ROAD_W + KERB * 2 + 30;
  g.stroke(path);

  // kerbs: red base + white dashes
  g.strokeStyle = '#d23a3a';
  g.lineWidth = ROAD_W + KERB * 2;
  g.stroke(path);
  g.strokeStyle = '#f4f4f4';
  g.lineCap = 'butt'; // round caps would grow every dash by half the line width
  g.setLineDash([10, 10]);
  g.stroke(path);
  g.setLineDash([]);
  g.lineCap = 'round';
  // asphalt
  g.strokeStyle = '#6b6f78';
  g.lineWidth = ROAD_W;
  g.stroke(path);
  // asphalt grain
  const off = makeCanvas(N, N);
  const og = off.getContext('2d');
  for (let i = 0; i < 60000; i++) {
    og.fillStyle = R() < 0.5 ? 'rgba(40,40,48,0.35)' : 'rgba(150,150,160,0.25)';
    og.fillRect((R() * N) | 0, (R() * N) | 0, 2, 2);
  }
  og.globalCompositeOperation = 'destination-in';
  og.strokeStyle = '#000';
  og.lineJoin = 'round';
  og.lineWidth = ROAD_W - 2;
  og.stroke(path);
  g.drawImage(off, 0, 0);
  // center line
  g.strokeStyle = 'rgba(255,255,255,0.55)';
  g.lineWidth = 2;
  g.lineCap = 'butt';
  g.setLineDash([14, 18]);
  g.stroke(path);
  g.setLineDash([]);
  const n = center.length;
  const dirAt = (i) => {
    const a = center[(i - 2 + n) % n];
    const b = center[(i + 2) % n];
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  };
  // start / finish checker
  {
    const [x, y] = center[0];
    g.save();
    g.translate(x, y);
    g.rotate(dirAt(0));
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < ROAD_W / 8; j++) {
        g.fillStyle = (i + j) % 2 ? '#111' : '#fff';
        g.fillRect(i * 8 - 8, j * 8 - ROAD_W / 2, 8, 8);
      }
    g.restore();
  }
  // boost pads (yellow chevrons)
  for (const f of BOOST_AT) {
    const i = Math.floor(f * n);
    const [x, y] = center[i];
    g.save();
    g.translate(x, y);
    g.rotate(dirAt(i));
    g.fillStyle = '#ffb22e';
    g.fillRect(-18, -16, 36, 32);
    g.fillStyle = '#fff36b';
    for (let k = 0; k < 3; k++) {
      g.beginPath();
      g.moveTo(-14 + k * 10, -12);
      g.lineTo(-6 + k * 10, 0);
      g.lineTo(-14 + k * 10, 12);
      g.lineTo(-10 + k * 10, 12);
      g.lineTo(-2 + k * 10, 0);
      g.lineTo(-10 + k * 10, -12);
      g.fill();
    }
    g.restore();
  }
  return c;
}

function paintWorld() {
  const N = WORLD_N;
  const c = makeCanvas(N, N);
  const g = c.getContext('2d');
  const img = g.createImageData(N, N);
  const d = img.data;
  const H = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = x / N - 0.5;
      const v = y / N - 0.5;
      const r = Math.hypot(u * 1.1, v);
      let h = fbm2(x / 140, y / 140, 6, 3) * 1.15 - 0.2 + 0.35 * (0.55 - r * 1.25);
      h += 0.12 * (fbm2(x / 40, y / 40, 3, 9) - 0.5);
      H[y * N + x] = h;
    }
  }
  const sea = 0.36;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = y * N + x;
      const h = H[i];
      const hx = H[y * N + Math.min(N - 1, x + 1)] - H[y * N + Math.max(0, x - 1)];
      const hy = H[Math.min(N - 1, y + 1) * N + x] - H[Math.max(0, y - 1) * N + x];
      const shade = 1 - (hx + hy) * 18; // light from the north-west
      const moist = vnoise(x / 80, y / 80, 21);
      let col;
      if (h < sea - 0.07) col = [28, 78, 150];
      else if (h < sea) col = [52, 120, 190];
      else if (h < sea + 0.025) col = [226, 205, 145];
      else if (h < 0.55) col = moist > 0.55 ? [44, 112, 56] : [104, 170, 76];
      else if (h < 0.66) col = [128, 120, 88];
      else if (h < 0.72) col = [150, 146, 140];
      else col = [242, 244, 250];
      let s = h < sea ? 1 : shade;
      // forest canopy texture
      if (h >= sea + 0.025 && h < 0.55 && moist > 0.55 && vnoise(x / 3, y / 3, 5) > 0.55) s *= 0.75;
      // shallow-water ripples near the shore
      if (h < sea && h > sea - 0.03) s *= 1.08;
      d[i * 4] = clamp(col[0] * s, 0, 255);
      d[i * 4 + 1] = clamp(col[1] * s, 0, 255);
      d[i * 4 + 2] = clamp(col[2] * s, 0, 255);
      d[i * 4 + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  // candidate spots for billboard trees: forests and meadows
  const trees = [];
  const RT = rng(23);
  for (let k = 0; k < 9000 && trees.length < 420; k++) {
    const x = RT() * N;
    const y = RT() * N;
    const h = H[(y | 0) * N + (x | 0)];
    if (h > sea + 0.04 && h < 0.55 && vnoise(x / 80, y / 80, 21) > 0.5) trees.push([x, y]);
  }
  // towns, a castle and roads
  const R = rng(11);
  const towns = [];
  for (let k = 0; k < 400 && towns.length < 9; k++) {
    const x = 60 + R() * (N - 120);
    const y = 60 + R() * (N - 120);
    const h = H[(y | 0) * N + (x | 0)];
    if (h > sea + 0.04 && h < 0.52 && towns.every((t) => Math.hypot(t[0] - x, t[1] - y) > 120)) towns.push([x, y]);
  }
  g.strokeStyle = '#b78e57';
  g.lineWidth = 3;
  for (let i = 1; i < towns.length; i++) {
    g.beginPath();
    g.moveTo(towns[i - 1][0], towns[i - 1][1]);
    g.lineTo(towns[i][0], towns[i][1]);
    g.stroke();
  }
  towns.forEach(([x, y], k) => {
    for (let j = 0; j < 9; j++) {
      const hx = x + (R() - 0.5) * 16;
      const hy = y + (R() - 0.5) * 16;
      g.fillStyle = '#f2ead8';
      g.fillRect(hx | 0, hy | 0, 4, 4);
      g.fillStyle = j % 3 ? '#c4473a' : '#4864b8';
      g.fillRect(hx | 0, hy | 0, 4, 2);
    }
    if (k === 0) {
      g.fillStyle = '#9ea4b4';
      g.fillRect(x - 7, y - 7, 14, 14);
      g.fillStyle = '#6b7184';
      for (const [ox, oy] of [[-8, -8], [5, -8], [-8, 5], [5, 5]]) g.fillRect(x + ox, y + oy, 4, 4);
      g.fillStyle = '#e04848';
      g.fillRect(x - 1, y - 12, 2, 5);
    }
  });
  return { canvas: c, trees };
}

/** Kart (3 steering frames) + airship, painted as pixel art. */
function paintProps() {
  const c = makeCanvas(160, 48);
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  const r = (x, y, w, h, col) => {
    g.fillStyle = col;
    g.fillRect(x, y, w, h);
  };
  for (let f = 0; f < 3; f++) {
    const ox = f * 32;
    const lean = f - 1; // -1 left, 0, +1 right
    // rear tyres
    r(ox + 2, 15, 7, 8, '#1a1c2c');
    r(ox + 23, 15, 7, 8, '#1a1c2c');
    r(ox + 3, 16, 5, 1, '#566c86');
    r(ox + 24, 16, 5, 1, '#566c86');
    // body
    r(ox + 6, 12, 20, 9, '#1a1c2c');
    r(ox + 7, 12, 18, 8, '#b13e53');
    r(ox + 7, 12, 18, 2, '#ef7d57');
    r(ox + 9, 17, 14, 2, '#5d275d');
    // exhausts
    r(ox + 11, 20, 3, 2, '#94b0c2');
    r(ox + 18, 20, 3, 2, '#94b0c2');
    // driver
    const hx = ox + 11 + lean;
    r(hx, 2, 10, 10, '#1a1c2c');
    r(hx + 1, 3, 8, 8, '#f4f4f4');
    r(hx + 1, 3, 8, 3, '#41a6f6');
    r(hx + 2, 8, 6, 1, '#94b0c2');
    r(ox + 9 + lean, 10, 3, 3, '#b13e53');
    r(ox + 20 + lean, 10, 3, 3, '#b13e53');
    // number plate
    r(ox + 14, 14, 4, 3, '#ffcd75');
  }
  // airship (48x44 at x=96)
  const ax = 100;
  g.fillStyle = '#1a1c2c';
  g.beginPath();
  g.ellipse(ax + 24, 15, 23, 14, 0, 0, Math.PI * 2);
  g.fill();
  for (let i = 0; i < 6; i++) {
    g.fillStyle = i % 2 ? '#ef7d57' : '#ffcd75';
    g.beginPath();
    g.ellipse(ax + 24, 15, 22 - i * 3.6, 13, 0, 0, Math.PI * 2);
    g.fill();
  }
  g.fillStyle = 'rgba(255,255,255,0.35)';
  g.beginPath();
  g.ellipse(ax + 18, 9, 9, 4, -0.3, 0, Math.PI * 2);
  g.fill();
  r(ax + 15, 27, 2, 8, '#5b3a29');
  r(ax + 31, 27, 2, 8, '#5b3a29');
  r(ax + 10, 33, 28, 9, '#1a1c2c');
  r(ax + 11, 34, 26, 7, '#8f563b');
  r(ax + 11, 34, 26, 2, '#c28b1f');
  for (let i = 0; i < 4; i++) r(ax + 14 + i * 6, 37, 3, 2, '#ffcd75');
  r(ax + 2, 35, 7, 2, '#94b0c2');
  r(ax + 39, 35, 7, 2, '#94b0c2');
  r(ax + 4, 31, 2, 9, '#566c86');
  r(ax + 42, 31, 2, 9, '#566c86');
  return c;
}

// ------------------------------------------------------------------------------------ simulation

const ATLAS_FRAMES = ['tree', 'bush', 'rock', 'gem', 'star', 'mushroom', 'chest', 'flower'];
const COIN0 = 8; // frames 8..15 = coin_0..7

let S = null;

function newState(ctx, data) {
  const center = data.center;
  const n = center.length;
  const props = [];
  const R = rng(5);
  // roadside trees, bushes, rocks
  for (let k = 0; k < 140; k++) {
    const i = Math.floor(R() * n);
    const a = Math.atan2(center[(i + 2) % n][1] - center[(i - 2 + n) % n][1], center[(i + 2) % n][0] - center[(i - 2 + n) % n][0]);
    const side = R() < 0.5 ? -1 : 1;
    const off = ROAD_W / 2 + 34 + R() * 90;
    const x = center[i][0] - Math.sin(a) * off * side;
    const y = center[i][1] + Math.cos(a) * off * side;
    if (x < 30 || y < 30 || x > TRACK_N - 30 || y > TRACK_N - 30) continue;
    // keep away from the road everywhere (the track bends back on itself)
    let ok = true;
    for (let j = 0; j < n; j += 3) if (Math.hypot(center[j][0] - x, center[j][1] - y) < ROAD_W / 2 + 26) ok = false;
    if (!ok) continue;
    const roll = R();
    const frame = roll < 0.6 ? 0 : roll < 0.82 ? 1 : roll < 0.92 ? 2 : 5;
    const size = frame === 0 ? 34 + R() * 14 : frame === 5 ? 8 : 12 + R() * 6;
    props.push({ x, y, size, frame, kind: 'prop' });
  }
  for (const f of COIN_GROUPS) {
    const i0 = Math.floor(f * n);
    const a = Math.atan2(center[(i0 + 2) % n][1] - center[(i0 - 2 + n) % n][1], center[(i0 + 2) % n][0] - center[(i0 - 2 + n) % n][0]);
    for (let k = 0; k < 4; k++) {
      const i = (i0 + k * 5) % n;
      const lat = (k % 2 ? 1 : -1) * 10;
      props.push({ x: center[i][0] - Math.sin(a) * lat, y: center[i][1] + Math.cos(a) * lat, size: 5, frame: COIN0, kind: 'coin', hidden: 0 });
    }
  }
  // airship world: castles / landmarks as billboards
  const R2 = rng(9);
  const worldProps = data.worldTrees.map(([x, y]) => ({ x, y, size: 7 + R2() * 4, frame: R2() < 0.75 ? 0 : 1, kind: 'prop' }));
  return {
    ctx,
    data,
    props,
    worldProps,
    kart: { x: center[4][0], y: center[4][1], a: Math.atan2(center[8][1] - center[0][1], center[8][0] - center[0][0]), v: 60, idx: 4, steer: 0, camA: 0 },
    ship: { x: WORLD_N * 0.5, y: WORLD_N * 0.62, a: -1.2, v: 42, h: 0, t: 0 },
    camA: null,
    idle: 99,
    coins: 0,
    boost: 0,
    tags: {},
    lastEx: null,
  };
}

function nearestCenter(center, x, y, hint) {
  const n = center.length;
  let best = hint;
  let bd = Infinity;
  for (let k = -40; k <= 40; k++) {
    const i = (hint + k + n) % n;
    const d = (center[i][0] - x) ** 2 + (center[i][1] - y) ** 2;
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  if (bd > 140 * 140) {
    for (let i = 0; i < n; i++) {
      const d = (center[i][0] - x) ** 2 + (center[i][1] - y) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
  }
  return [best, Math.sqrt(bd)];
}

function stepKart(st, p, ctx, dt) {
  const k = st.kart;
  const center = st.data.center;
  const n = center.length;
  const keys = ctx.keys;
  const manual = anyKey(keys, [KEYS.up, KEYS.down, KEYS.left, KEYS.right]);
  st.idle = manual ? 0 : st.idle + dt;
  const auto = p.autopilot && st.idle > 2.5;
  let throttle = keyAxis(keys, KEYS.down, KEYS.up);
  let steer = keyAxis(keys, KEYS.left, KEYS.right);
  const [idx, dist] = nearestCenter(center, k.x, k.y, k.idx);
  k.idx = idx;
  if (auto) {
    const look = 9 + Math.floor(k.v / 22);
    const t = center[(idx + look) % n];
    const want = Math.atan2(t[1] - k.y, t[0] - k.x);
    steer = clamp(wrapAngle(want - k.a) * 2.4, -1, 1);
    const ahead = center[(idx + look * 2) % n];
    const curve = Math.abs(wrapAngle(Math.atan2(ahead[1] - t[1], ahead[0] - t[0]) - want));
    throttle = k.v > 210 - curve * 160 ? -0.3 : 1;
  }
  const onRoad = dist < ROAD_W / 2 + KERB;
  const top = onRoad ? 230 : 95;
  k.v += (throttle > 0 ? throttle * 150 : throttle * 260) * dt;
  if (throttle === 0) k.v -= Math.sign(k.v) * Math.min(Math.abs(k.v), 40 * dt);
  if (k.v > top) k.v = Math.max(top, k.v - 300 * dt);
  k.v = clamp(k.v, -60, 340);
  // boost pads
  for (const f of BOOST_AT) {
    const b = center[Math.floor(f * n)];
    if (Math.hypot(b[0] - k.x, b[1] - k.y) < 22) {
      k.v = Math.max(k.v, 330);
      st.boost = 0.6;
    }
  }
  st.boost = Math.max(0, st.boost - dt);
  k.steer += (steer - k.steer) * Math.min(1, dt * 10);
  k.a += k.steer * 2.3 * Math.min(1, Math.abs(k.v) / 90) * Math.sign(k.v || 1) * dt;
  k.x += Math.cos(k.a) * k.v * dt;
  k.y += Math.sin(k.a) * k.v * dt;
  k.x = clamp(k.x, 20, TRACK_N - 20);
  k.y = clamp(k.y, 20, TRACK_N - 20);
  // coins
  for (const pr of st.props) {
    if (pr.kind !== 'coin') continue;
    if (pr.hidden > 0) pr.hidden -= dt;
    else if (Math.hypot(pr.x - k.x, pr.y - k.y) < 12) {
      pr.hidden = 8;
      st.coins++;
    }
  }
  // chase camera lags behind the kart's heading (that lag is what sells the turns)
  if (st.camA === null) st.camA = k.a;
  st.camA += wrapAngle(k.a - st.camA) * Math.min(1, dt * 5);
}

function stepShip(st, p, ctx, dt) {
  const s = st.ship;
  const keys = ctx.keys;
  const manual = anyKey(keys, [KEYS.up, KEYS.down, KEYS.left, KEYS.right]);
  st.idle = manual ? 0 : st.idle + dt;
  let turn = keyAxis(keys, KEYS.left, KEYS.right);
  const climb = keyAxis(keys, KEYS.down, KEYS.up);
  s.t += dt;
  if (p.autopilot && st.idle > 2.5) {
    // lazy figure-eight, pulled back toward the middle of the map
    const toC = Math.atan2(WORLD_N * 0.5 - s.y, WORLD_N * 0.5 - s.x);
    const far = Math.hypot(WORLD_N * 0.5 - s.x, WORLD_N * 0.5 - s.y) / (WORLD_N * 0.5);
    turn = clamp(0.35 * Math.sin(s.t * 0.21) + wrapAngle(toC - s.a) * far * 1.4, -1, 1);
  }
  s.h = clamp(s.h + climb * 30 * dt, -30, 60);
  s.a += turn * 0.7 * dt;
  s.roll = (s.roll || 0) + (turn - (s.roll || 0)) * Math.min(1, dt * 3);
  s.x += Math.cos(s.a) * s.v * dt;
  s.y += Math.sin(s.a) * s.v * dt;
  if (st.camA === null) st.camA = s.a;
  st.camA += wrapAngle(s.a - st.camA) * Math.min(1, dt * 2.5);
}

function project(view, cam, wx, wy) {
  const ca = Math.cos(cam.a);
  const sa = Math.sin(cam.a);
  const dx = wx - cam.x;
  const dy = wy - cam.y;
  const z = dx * ca + dy * sa;
  const x = -dx * sa + dy * ca;
  return { z, sx: view.cx + (x * view.f) / z, sy: view.hy + (cam.h * view.f) / z, k: view.f / z };
}

function computeBind(params, ctx) {
  const st = S;
  const ex = ctx.example;
  if (st.lastEx !== ex) {
    for (const t of Object.values(st.tags)) t.remove();
    st.tags = {};
    st.camA = null;
    st.lastEx = ex;
    if (ex === 'explain') {
      st.tags.l = overlayTag(ctx, 'left:25%;top:40px;transform:translateX(-50%)');
      st.tags.l.textContent = 'The map (a flat texture), seen from above';
      st.tags.r = overlayTag(ctx, 'left:75%;top:40px;transform:translateX(-50%)');
      st.tags.r.textContent = 'The same map, projected row by row';
    }
    st.tags.hud = overlayTag(ctx, 'right:8px;bottom:8px');
  }
  const dt = Math.min(ctx.dt, 1 / 20);
  const W = ctx.width;
  const H = ctx.height;
  const isShip = ex === 'airship';
  if (isShip) stepShip(st, params, ctx, dt);
  else stepKart(st, params, ctx, dt);

  const x0 = ex === 'explain' ? Math.floor(W / 2) : 0;
  const vw = W - x0;
  const fov = (params.fov * Math.PI) / 180;
  const f = vw / 2 / Math.tan(fov / 2);
  const hy = H * params.horizon;
  const view = { cx: x0 + vw / 2, f, hy };
  // The chase camera sits exactly far enough behind the player that its ground position lands at
  // a fixed screen row (z = h·f / rowsBelowHorizon) — the same formula the shader inverts.
  const groundRow = isShip ? 0.95 : 0.86;
  const camH = isShip ? params.height + st.ship.h : params.height;
  const chase = (camH * f) / Math.max(8, (groundRow - params.horizon) * H);
  st.chase = chase;
  const pl = isShip ? st.ship : st.kart;
  const cam = { x: pl.x - Math.cos(st.camA) * chase, y: pl.y - Math.sin(st.camA) * chase, a: st.camA, h: camH };

  // billboards: project, cull, sort far -> near
  const list = [];
  const src = isShip ? st.worldProps : st.props;
  const coinFrame = COIN0 + (Math.floor(ctx.time * 12) % 8);
  for (const pr of src) {
    if (pr.hidden > 0) continue;
    let wx = pr.x;
    let wy = pr.y;
    if (isShip) {
      // the world map repeats as ocean outside; keep landmarks only on the map
      if (wx < 0 || wy < 0 || wx > WORLD_N || wy > WORLD_N) continue;
    }
    const q = project(view, cam, wx, wy);
    if (q.z < 4 || q.z > 2600) continue;
    const h = pr.size * q.k;
    const fr = pr.kind === 'coin' ? coinFrame : pr.frame;
    const asp = st.frameAspect[fr];
    const w = h * asp;
    if (q.sx + w / 2 < x0 || q.sx - w / 2 > W || h < 0.6) continue;
    list.push({ z: q.z, rect: [q.sx - w / 2, q.sy - h, w, h], world: [wx, wy, pr.size * 0.32 * Math.max(asp, 0.6), fr] });
  }
  list.sort((a, b) => b.z - a.z);
  const sprRect = new Float32Array(NS * 4);
  const sprWorld = new Float32Array(NS * 4);
  // keep the NS nearest, still drawn far -> near
  const keep = list.slice(Math.max(0, list.length - NS));
  keep.forEach((s, i) => {
    sprRect.set(s.rect, i * 4);
    sprWorld.set(s.world, i * 4);
  });

  // player sprite (kart or airship) in screen space
  let kartRect = [0, 0, 0, 0];
  let kartUV = [0, 0, 0, 0];
  let shadow = [0, 0, 0, 0];
  const PW = 160;
  const PH = 48;
  if (isShip) {
    // the airship hovers above its own shadow: fixed size & place on screen, shadow on the ground
    const s = st.ship;
    const q = project(view, cam, s.x, s.y);
    const w = Math.min(vw * 0.24, H * 0.42);
    const h = (w * 44) / 48;
    const bob = Math.sin(ctx.time * 1.7) * h * 0.03;
    const cy = H * (0.6 + 0.04 * Math.min(1, s.h / 60)) + bob;
    kartRect = [q.sx - w / 2, cy - h / 2, w, h];
    kartUV = [96 / PW, 0, 144 / PW, 44 / PH];
    shadow = [s.x, s.y, (w / q.k) * 0.3, 0.45];
  } else {
    const k = st.kart;
    const q = project(view, cam, k.x, k.y);
    const w = 7 * q.k; // the kart is 7 world units wide
    const h = (w * 24) / 32;
    const frame = k.steer < -0.35 ? 0 : k.steer > 0.35 ? 2 : 1;
    const bounce = Math.abs(k.v) > 20 && Math.sin(ctx.time * 30) > 0 ? h / 24 : 0;
    kartRect = [q.sx - w / 2, q.sy - h * 0.92 + bounce, w, h];
    kartUV = [(frame * 32) / PW, 0, (frame * 32 + 32) / PW, 24 / PH];
    shadow = [k.x, k.y, 4.2, 0.55];
  }

  const hud = st.tags.hud;
  if (hud) {
    if (isShip) hud.textContent = `heading ${(((st.ship.a * 180) / Math.PI) % 360 + 360) % 360 | 0}° · altitude ${cam.h.toFixed(0)} · ${keep.length} billboards`;
    else hud.textContent = `${Math.round(Math.abs(st.kart.v) * 0.6)} km/h · coins ${st.coins}${st.boost > 0 ? ' · BOOST!' : ''}${st.idle > 2.5 && params.autopilot ? ' · autopilot' : ''}`;
  }
  return {
    cam: [cam.x, cam.y, cam.a, cam.h],
    view: [x0, vw, hy, f],
    chase,
    sprRect,
    sprWorld,
    frames: st.frames,
    kartRect,
    kartUV,
    shadow,
    boostFx: st.boost,
  };
}

let bakeCache = null;
let TEST = false;

const FRAG = /* wgsl */ `
${MIP_WGSL}
const NS: i32 = ${NS};

// "Sharp bilinear": when a texel covers several pixels, only blend across a 1-pixel band at texel
// edges, so magnified pixel art stays crisp while minified texels still filter smoothly.
fn sharpen(m: vec2f, n: f32, pxPerTexel: f32) -> vec2f {
  let t = m * n - vec2f(0.5);
  let i = floor(t);
  let s = max(pxPerTexel, 1.0);
  let g = clamp((fract(t) - vec2f(0.5)) * s + vec2f(0.5), vec2f(0.0), vec2f(1.0));
  return (i + g + vec2f(0.5)) / n;
}

fn mapSample(t: texture_2d<f32>, m: vec2f, lod: f32, n: f32, mode: f32) -> vec3f {
  if (mode < 0.5) { return TEXN(t, mipCoord(m, 0.0, n)).rgb; }
  if (mode < 1.5) { return TEX(t, mipCoord(m, 0.0, n)).rgb; }
  let l = clamp(lod, 0.0, 6.0);
  let l0 = floor(l);
  var m0 = m;
  if (l0 < 0.5) { m0 = sharpen(m, n, exp2(-lod)); }
  let a = TEX(t, mipCoord(m0, l0, n)).rgb;
  let b = TEX(t, mipCoord(m, min(l0 + 1.0, 7.0), n)).rgb;
  return mix(a, b, l - l0);
}

// What lies beyond the edge of the map: a repeating pattern (Super Mario Kart did exactly this).
fn forestPattern(w: vec2f, lod: f32) -> vec3f {
  let g = w / 22.0;
  let id = floor(g);
  let f = fract(g) - vec2f(0.5);
  let j = (hash22(id) - vec2f(0.5)) * 0.35;
  let d = length(f - j);
  let crown = smoothstep(0.46, 0.34, d);
  var c = mix(vec3f(0.07, 0.2, 0.1), vec3f(0.16, 0.42, 0.19) * (0.8 + 0.4 * hash21(id)), crown);
  c += vec3f(0.08, 0.1, 0.04) * smoothstep(0.22, 0.0, length(f - j + vec2f(0.12))) * crown;
  return mix(c, vec3f(0.12, 0.31, 0.15), clamp((lod - 1.0) / 2.0, 0.0, 1.0));
}

fn oceanPattern(w: vec2f, lod: f32, t: f32) -> vec3f {
  let wv = sin(w.x * 0.31 + 2.0 * sin(w.y * 0.13 + t * 0.6) + t * 1.2) * sin(w.y * 0.27 - t * 0.9);
  let crest = smoothstep(0.7, 0.95, wv) * (1.0 - clamp((lod - 0.5) / 2.0, 0.0, 1.0));
  return vec3f(0.11, 0.3, 0.58) + vec3f(0.3, 0.35, 0.3) * crest;
}

fn skyColor(p: vec2f, hy: f32, f: f32, cx: f32, ang: f32, ex: i32) -> vec3f {
  let el = (hy - p.y) / f;                 // tangent of the elevation angle
  let phi = ang + atan2(p.x - cx, f);       // azimuth: the background scrolls with the camera angle
  let ship = f32(ex == 2);
  var c = mix(vec3f(0.66, 0.84, 0.98), vec3f(0.24, 0.47, 0.88), clamp(el * 2.4, 0.0, 1.0));
  c = mix(c, mix(vec3f(0.98, 0.86, 0.7), vec3f(0.3, 0.42, 0.78), clamp(el * 2.0, 0.0, 1.0)), ship * 0.55);
  // sun
  let sunDir = vec2f(fmod(phi - 0.6 + PI, TAU) - PI, el - 0.22);
  c += vec3f(1.0, 0.85, 0.6) * exp(-length(sunDir) * 9.0) * 0.6;
  c = mix(c, vec3f(1.0, 0.97, 0.88), smoothstep(0.045, 0.04, length(sunDir)));
  // clouds: noise on a circle so they wrap around 360 degrees
  let cp = vec3f(cos(phi) * 3.0, sin(phi) * 3.0, el * 9.0);
  let cn = perlin3(cp + vec3f(0.0, 0.0, u.time * 0.03)) * 0.6 + perlin3(cp * 2.3) * 0.3;
  let cloud = smoothstep(0.08, 0.35, cn) * smoothstep(0.03, 0.1, el) * smoothstep(0.42, 0.18, el);
  c = mix(c, vec3f(1.0, 0.98, 0.96), cloud * 0.85);
  // far mountains: sums of integer-frequency sines are periodic in phi
  let mh = 0.05 + 0.035 * (0.5 + 0.5 * sin(phi * 3.0 + 1.3)) * (0.7 + 0.3 * sin(phi * 7.0)) + 0.01 * sin(phi * 23.0) + 0.005 * sin(phi * 41.0 + 2.0);
  let snow = smoothstep(mh - 0.012, mh - 0.004, el) * smoothstep(0.075, 0.09, mh);
  let mcol = mix(vec3f(0.42, 0.48, 0.66), vec3f(0.95, 0.96, 1.0), snow);
  c = mix(c, mix(mcol, c, 0.35), step(el, mh) * (1.0 - ship));
  // from altitude: the curved haze line of the far sea instead of mountains
  c = mix(c, vec3f(0.93, 0.86, 0.84), smoothstep(0.06, 0.0, el) * ship);
  // near hills with a tree line (not on the airship: it flies above them)
  let hh = 0.018 + 0.01 * sin(phi * 5.0 + 0.7) + 0.004 * sin(phi * 61.0) + 0.006 * smoothstep(0.3, 1.0, sin(phi * 37.0));
  c = mix(c, mix(vec3f(0.16, 0.42, 0.2), vec3f(0.3, 0.5, 0.35), ship), step(el, hh) * (1.0 - ship));
  return c;
}

fn rowFrac(k: i32) -> f32 {
  var fr = array<f32, 6>(0.035, 0.075, 0.14, 0.26, 0.47, 0.85);
  return fr[k];
}
fn rowColor(k: i32) -> vec3f { return hsv2rgb(vec3f(f32(k) / 6.0 + 0.02, 0.75, 1.0)); }

// Mode 7 for one pixel: returns rgb. view = (x0, width, horizonY, focal), cam = (x, y, angle, height)
fn mode7(p: vec2f, ex: i32) -> vec3f {
  let cam = u.cam;
  let v = u.view;
  let cx = v.x + v.y * 0.5;
  let sy = p.y - v.z;
  let fwd = vec2f(cos(cam.z), sin(cam.z));
  let rgt = vec2f(-sin(cam.z), cos(cam.z));
  // inverse projection: screen row -> distance, screen column -> sideways offset
  let z = cam.w * v.w / max(sy, 0.0001);
  let lat = (p.x - cx) * z / v.w;
  let w = cam.xy + fwd * z + rgt * lat;
  let isShip = ex == 2;
  var n = ${TRACK_N}.0;
  if (isShip) { n = ${WORLD_N}.0; }
  let m = w / n;
  let fw = fwidth(w);
  let lod = log2(max(max(fw.x, fw.y), 0.0001)) - 0.35;
  if (sy <= 0.0) {
    return skyColor(p, v.z, v.w, cx, cam.z, ex);
  }
  var col: vec3f;
  let inside = m.x >= 0.0 && m.y >= 0.0 && m.x <= 1.0 && m.y <= 1.0;
  if (isShip) {
    if (inside) {
      col = mapSample(world, m, lod, n, u.texFilter);
      // animate water: blue-dominant texels get rolling wave crests
      let water = smoothstep(0.08, 0.2, col.b - max(col.r, col.g));
      col = mix(col, oceanPattern(w, lod, u.time) * (col.b / 0.6), water * 0.45);
    } else {
      col = oceanPattern(w, lod, u.time);
    }
  } else {
    if (inside) { col = mapSample(track, m, lod, n, u.texFilter); }
    else { col = forestPattern(w, lod); }
  }
  // blob shadows of billboards (and the player) on the floor
  var sh = 1.0;
  for (var i = 0; i < NS; i++) {
    let sw = u.sprWorld[i];
    if (sw.z <= 0.0) { continue; }
    let d = length((w - sw.xy) / vec2f(sw.z, sw.z * 0.7));
    sh *= 1.0 - 0.35 * smoothstep(1.0, 0.6, d);
  }
  if (u.shadow.z > 0.0) {
    var sp = u.shadow.xy;
    let d = length(w - sp) / u.shadow.z;
    sh *= 1.0 - u.shadow.w * smoothstep(1.0, 0.55, d);
  }
  col *= sh;
  // airship: a second, semi-transparent Mode 7 plane for clouds (plus their shadows on the ground)
  if (isShip && u.clouds > 0.0) {
    let ch = cam.w * 0.45;
    let zc = (cam.w - ch) * v.w / max(sy, 0.0001);
    let wc = cam.xy + fwd * zc + rgt * (p.x - cx) * zc / v.w + vec2f(u.time * 6.0, u.time * 2.0);
    let cl = fwidth(wc);
    let fade = 1.0 - clamp((log2(max(cl.x, cl.y)) - 1.0) / 3.0, 0.0, 1.0);
    let cn = fbm(wc / 70.0, 5) * 0.5 + 0.5;
    let dens = smoothstep(0.66 - u.clouds * 0.22, 0.9, cn);
    let gshadow = smoothstep(0.66 - u.clouds * 0.22, 0.9, fbm((w + vec2f(u.time * 6.0, u.time * 2.0) + vec2f(25.0, -18.0)) / 70.0, 5) * 0.5 + 0.5);
    col *= 1.0 - 0.4 * gshadow;
    // lit tops, grey-blue thin edges
    let cc = mix(vec3f(0.62, 0.68, 0.82), vec3f(1.0, 0.98, 0.96), smoothstep(0.0, 0.7, dens));
    col = mix(col, cc, dens * mix(0.55, 0.8, fade));
  }
  // distance haze toward the horizon color
  var haze = vec3f(0.66, 0.8, 0.95);
  if (isShip) { haze = vec3f(0.86, 0.82, 0.86); }
  let fogAmt = 1.0 - exp(-z * u.fog * 0.0025);
  col = mix(col, haze, clamp(fogAmt, 0.0, 1.0));
  // explainer: colour the sample rows
  if (ex == 0 && u.rows > 0.5) {
    for (var k = 0; k < 6; k++) {
      let ry = v.z + (u.resolution.y - v.z) * rowFrac(k);
      let a = 1.0 - smoothstep(1.0, 2.2, abs(p.y - ry));
      col = mix(col, rowColor(k), a * 0.9);
    }
  }
  return col;
}

fn sprites(p: vec2f, colIn: vec3f, ex: i32) -> vec3f {
  var col = colIn;
  for (var i = 0; i < NS; i++) {
    let r = u.sprRect[i];
    if (r.z <= 0.0) { continue; }
    let q = (p - r.xy) / r.zw;
    if (q.x < 0.0 || q.y < 0.0 || q.x >= 1.0 || q.y >= 1.0) { continue; }
    let fr = u.frames[i32(u.sprWorld[i].w)];
    let c = TEXN(atlas, mix(fr.xy, fr.zw, q));
    let d = length(u.sprWorld[i].xy - u.cam.xy);
    var haze = vec3f(0.66, 0.8, 0.95);
    if (ex == 2) { haze = vec3f(0.86, 0.82, 0.86); }
    let sc = mix(c.rgb, haze, clamp(1.0 - exp(-d * u.fog * 0.0025), 0.0, 1.0));
    col = mix(col, sc, step(0.5, c.a));
  }
  let kr = u.kartRect;
  if (kr.z > 0.0) {
    let q = (p - kr.xy) / kr.zw;
    if (q.x >= 0.0 && q.y >= 0.0 && q.x < 1.0 && q.y < 1.0) {
      let c = TEXN(props, mix(u.kartUV.xy, u.kartUV.zw, q));
      col = mix(col, c.rgb, step(0.5, c.a));
    }
  }
  return col;
}

fn mapPanel(p: vec2f) -> vec3f {
  let half = u.view.x;
  let side = min(half * 0.92, u.resolution.y - 84.0);
  let o = vec2f((half - side) * 0.5, 70.0 + (u.resolution.y - 84.0 - side) * 0.5);
  let box = sdBox(p - (o + vec2f(side * 0.5)), vec2f(side * 0.5));
  let clipM = 1.0 - smoothstep(-1.0, 0.0, box);
  let m = (p - o) / side;
  var col = vec3f(0.06, 0.07, 0.1);
  let n = ${TRACK_N}.0;
  let px2m = n / side;               // map texels per screen pixel
  let cam = u.cam;
  let v = u.view;
  let fwd = vec2f(cos(cam.z), sin(cam.z));
  let rgt = vec2f(-sin(cam.z), cos(cam.z));
  let w = m * n;
  if (m.x >= 0.0 && m.y >= 0.0 && m.x <= 1.0 && m.y <= 1.0) {
    col = mapSample(track, m, log2(px2m), n, 2.0);
    // the region the camera can see (inside the frustum, below the horizon)
    let d = w - cam.xy;
    let zz = dot(d, fwd);
    let xx = dot(d, rgt);
    let zNear = cam.w * v.w / (u.resolution.y - v.z);
    let inF = step(zNear, zz) * step(abs(xx), zz * (v.y * 0.5) / v.w);
    col = mix(col * 0.55, col, max(inF, 0.0));
  }
  // frustum edges
  let tanH = (v.y * 0.5) / v.w;
  let far = 1400.0;
  let e1 = cam.xy + (fwd + rgt * tanH) * far;
  let e2 = cam.xy + (fwd - rgt * tanH) * far;
  let cp = o + cam.xy / n * side;
  let d1 = min(sdSegment(p, cp, o + e1 / n * side), sdSegment(p, cp, o + e2 / n * side));
  col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.8, 1.8, d1)) * 0.85 * clipM);
  // each colored screen row is a straight line across the map
  if (u.rows > 0.5) {
    for (var k = 0; k < 6; k++) {
      let ry = v.z + (u.resolution.y - v.z) * rowFrac(k);
      let zr = cam.w * v.w / (ry - v.z);
      let a = cam.xy + fwd * zr - rgt * zr * tanH;
      let b = cam.xy + fwd * zr + rgt * zr * tanH;
      let ds = sdSegment(p, o + a / n * side, o + b / n * side);
      col = mix(col, rowColor(k), (1.0 - smoothstep(1.0, 2.2, ds)) * clipM);
    }
  }
  // camera and kart markers
  let kp = o + (cam.xy + fwd * u.chase) / n * side;
  col = mix(col, vec3f(0.1), 1.0 - smoothstep(4.0, 5.0, length(p - kp)));
  col = mix(col, vec3f(0.95, 0.3, 0.35), 1.0 - smoothstep(3.0, 4.0, length(p - kp)));
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(5.0, 6.0, length(p - cp)));
  col = mix(col, vec3f(1.0, 0.9, 0.3), 1.0 - smoothstep(3.5, 4.5, length(p - cp)));
  // frame
  col = mix(col, vec3f(0.5, 0.55, 0.7), 1.0 - smoothstep(0.5, 1.5, abs(box)));
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  let p = uv * u.resolution;      // canvas pixels (independent of the retro render scale)
  if (ex == 0 && p.x < u.view.x) {
    var c = mapPanel(p);
    c = mix(c, vec3f(0.3, 0.33, 0.45), 1.0 - smoothstep(0.5, 1.5, abs(p.x - u.view.x)));
    return vec4f(c, 1.0);
  }
  var col = mode7(p, ex);
  col = sprites(p, col, ex);
  // speed lines when boosting
  if (u.boostFx > 0.0) {
    let cxy = vec2f(u.view.x + u.view.y * 0.5, u.view.z);
    let a = atan2(p.y - cxy.y, p.x - cxy.x);
    let streak = smoothstep(0.92, 1.0, hash11(floor(a * 60.0) + floor(u.time * 20.0)));
    let r = length(p - cxy) / u.resolution.y;
    col = mix(col, vec3f(1.0), streak * smoothstep(0.35, 0.8, r) * u.boostFx);
  }
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: 'Arrows / WASD: drive · let go for autopilot',
  keys: true,
  examples: [
    {
      id: 'explain',
      label: 'Mode 7 explained',
      kind: 'Abstract',
      note: 'Left: the flat map texture with the camera (yellow), its field of view (white lines) and the area it can see. Right: the same map projected. Each <b>colored screen row</b> samples exactly one <b>straight line</b> of the map — rows near the horizon reach far away, rows at the bottom are close.',
      params: { pixel: 1, height: 22, horizon: 0.36, fov: 70, fog: 0.4 },
    },
    {
      id: 'kart',
      label: 'Kart racer',
      kind: 'In a game',
      note: 'A Super-Mario-Kart-style circuit: the whole track is one 1024² texture. Trees, coins and the kart are flat <b>billboard sprites</b> scaled by 1/distance. Drive over the yellow boost pads, collect coins.',
      params: { pixel: 2, height: 20, horizon: 0.34, fov: 70, fog: 0.35 },
    },
    {
      id: 'airship',
      label: 'Airship over the world map',
      kind: 'In a game',
      note: 'The Final Fantasy VI trick: a world map tilted toward the horizon, with a second Mode 7 plane for clouds that drift at a different height (so they parallax against the ground) and cast shadows.',
      params: { pixel: 2, height: 64, horizon: 0.3, fov: 65, fog: 0.35 },
      hint: '←/→ steer · ↑/↓ climb & descend · let go for autopilot',
    },
  ],
  controls: [
    { type: 'heading', label: 'Camera' },
    { type: 'slider', key: 'height', label: 'Camera height', min: 4, max: 140, step: 0.5, value: 22, help: 'Higher = steeper view, rows sample a bigger area. Low = racing-game feel.' },
    { type: 'slider', key: 'fov', label: 'Field of view (°)', min: 30, max: 120, step: 1, value: 70, help: 'Sets the focal length: how fast columns fan out.' },
    { type: 'slider', key: 'horizon', label: 'Horizon height', min: 0.12, max: 0.6, step: 0.005, value: 0.36, help: 'Moving the horizon = tilting the camera (pitch).' },
    { type: 'heading', label: 'Look' },
    {
      type: 'select', key: 'texFilter', label: 'Texture filtering', value: 'mip',
      options: [{ value: 'nearest', label: 'Nearest (real SNES)' }, { value: 'linear', label: 'Bilinear' }, { value: 'mip', label: 'Mipmapped (no shimmer)' }],
      help: 'Far rows squeeze many texels into one pixel: without mipmaps they sparkle and shimmer.',
    },
    { type: 'slider', key: 'pixel', label: 'Pixel size', min: 1, max: 4, step: 1, value: 2, help: '2–3 ≈ the SNES’s 256×224 chunky look.' },
    { type: 'slider', key: 'fog', label: 'Distance haze', min: 0, max: 2, step: 0.01, value: 0.5 },
    { type: 'slider', key: 'clouds', label: 'Cloud layer', min: 0, max: 1, step: 0.01, value: 0.4, showFor: ['airship'], help: 'A second Mode 7 plane halfway between camera and ground.' },
    { type: 'toggle', key: 'rows', label: 'Show sampled rows', value: true, showFor: ['explain'] },
    { type: 'toggle', key: 'autopilot', label: 'Autopilot when idle', value: true },
  ],
  uniforms: {
    height: 'f32', fov: 'f32', horizon: 'f32', chase: 'f32', texFilter: 'f32', fog: 'f32', clouds: 'f32', rows: 'f32', boostFx: 'f32',
    cam: 'vec4f', view: 'vec4f', kartRect: 'vec4f', kartUV: 'vec4f', shadow: 'vec4f',
    sprRect: `array<vec4f, ${NS}>`, sprWorld: `array<vec4f, ${NS}>`, frames: 'array<vec4f, 16>',
  },
  include: ['hash', 'noise', 'sdf', 'color', 'math'],
  renderScale: (p) => 1 / Math.max(TEST ? 3 : 1, p.pixel || 1),
  textures: {
    track: { source: async () => (await bake()).track, filter: 'linear' },
    world: { source: async () => (await bake()).world, filter: 'linear' },
    atlas: { source: async () => (await bake()).atlas.canvas, filter: 'nearest' },
    props: {
      source: async () => {
        const data = await bake();
        bakeCache = data;
        return data.props;
      },
      filter: 'nearest',
    },
  },
  bind(params, ctx) {
    TEST = !!ctx.testMode;
    if (!bakeCache) return {};
    if (!S || S.ctx !== ctx) {
      S = newState(ctx, bakeCache);
      const at = bakeCache.atlas;
      const names = [...ATLAS_FRAMES, ...at.anim('coin')];
      S.frames = new Float32Array(16 * 4);
      S.frameAspect = [];
      names.forEach((nm, i) => {
        S.frames.set(at.uv(nm), i * 4);
        const fr = at.frames[nm];
        S.frameAspect[i] = fr.w / fr.h;
      });
    }
    return computeBind(params, ctx) || {};
  },
  code: FRAG,
  about: {
    summary: 'The SNES could only scale and rotate one flat background layer — but by changing that transform on every scanline, games faked a 3D floor stretching to the horizon. A fragment shader does the same per pixel.',
    what: `<p>A flat 2D map texture (the race track or the world map) tilted away from the camera so it reaches the horizon, with a sky band above it.
      Sprites (trees, coins, the kart) are flat pictures that only get <b>bigger when closer</b>. Nothing here is 3D geometry.</p>`,
    how: `<ol>
      <li>Every pixel below the horizon asks: <i>which point of the flat ground plane do I see?</i> For a camera at height <code>h</code>
        with focal length <code>f</code>, a pixel <code>sy</code> rows below the horizon sees the ground at distance <code>z = h·f / sy</code>.</li>
      <li>Its column gives a sideways offset <code>x = (px − center) · z / f</code>. Rotate <code>(z, x)</code> by the camera angle and add the camera position → map coordinate.</li>
      <li>So each screen row is a straight line across the map, sampled at evenly spaced points. The SNES did this in hardware by
        reprogramming the background’s scale/rotation matrix during every horizontal blank (HDMA).</li>
      <li>Far rows squeeze hundreds of texels into one pixel and <b>shimmer</b>. The fix is <b>mipmapping</b>: pre-shrunk copies of the map,
        picked from the screen-space derivative (<code>fwidth</code>) of the map coordinate. Here the mip chain is packed into one texture so the
        same code runs on WebGL2.</li>
      <li>Billboards: project a point with the same formulas (<code>screenY = horizon + h·f/z</code>, size ∝ <code>f/z</code>), sort far→near and draw.</li>
      <li>The sky band scrolls horizontally with the camera angle; the airship’s clouds are a <b>second</b> Mode 7 plane at a different height.</li>
    </ol>`,
    uses: [
      { title: 'Kart & racing games', text: 'Super Mario Kart, F-Zero, and countless GBA racers: one big track texture plus billboard karts.' },
      { title: 'World maps & flying', text: 'Final Fantasy VI’s airship, Secret of Mana’s flammie, Pilotwings — a tilted world map with a horizon.' },
      { title: 'Boss arenas & effects', text: 'Rotating/zooming floors (Super Castlevania IV’s rotating room, Contra III’s top-down stages).' },
      { title: 'Modern retro games', text: 'Indie racers and “SNES-feel” games still use it because it is cheap and instantly recognisable.' },
    ],
    try: [
      'On <b>Mode 7 explained</b>, lower <i>Camera height</i> and watch the white wedge on the map get longer: low cameras see far, high cameras see a smaller, steeper area.',
      'Set <i>Texture filtering</i> to <b>Nearest</b> and look at the kerbs near the horizon — sparkling noise. Switch to <b>Mipmapped</b>.',
      'Push <i>Field of view</i> to 120: the fish-bowl stretch at the edges is the price of a wide lens.',
      'Drive off the track into the grass (it slows you down) and over a yellow boost pad.',
      'On the airship, raise <i>Cloud layer</i> and fly low with ↓: the clouds slide past the ground because they are a separate plane.',
    ],
    ask: [
      'a Mode 7 floor with mipmapped sampling',
      'SNES-style kart racer with billboard sprites',
      'an airship flying over a tilted world map with a cloud layer',
      'per-scanline perspective floor (Mode 7) in a fragment shader',
      'a rotating, zooming background layer for a boss fight',
    ],
    perf: `<p>Practically free: a handful of multiply-adds and one or two texture reads per pixel (trilinear = two). Billboards cost a loop over the
      visible sprites per pixel (24 here). On real hardware it was 0% CPU — just one matrix per scanline.</p>`,
    api: `<p>Pure fragment shader, so it runs on WebGPU and WebGL2 from the same WGSL. WebGPU could use real hardware mipmaps with
      <code>textureSampleGrad</code>; to stay portable this scene packs the mip chain into one texture and picks levels itself.</p>`,
    code: [
      {
        title: 'The whole trick: screen pixel → map coordinate',
        lang: 'wgsl',
        src: `let sy = p.y - horizonY;                 // rows below the horizon
let z = camHeight * focal / sy;           // distance to the ground point
let lat = (p.x - centerX) * z / focal;    // sideways offset
let fwd = vec2f(cos(angle), sin(angle));
let rgt = vec2f(-sin(angle), cos(angle));
let w = camPos + fwd * z + rgt * lat;     // world/map position
let fw = fwidth(w);                       // texels per pixel -> mip level
let lod = log2(max(fw.x, fw.y));
col = mapSample(track, w / mapSize, lod, mapSize, filterMode);`,
      },
      {
        title: 'Billboards use the same projection (JavaScript)',
        lang: 'js',
        src: `const z = dx * cos(a) + dy * sin(a);        // depth along the view
const x = -dx * sin(a) + dy * cos(a);       // sideways
const screenX = centerX + x * focal / z;
const groundY = horizonY + camHeight * focal / z;
const size = worldSize * focal / z;          // 1/z scaling
// sort far -> near, draw bottom-anchored at groundY`,
      },
    ],
    links: [
      { title: 'Mode 7 — Wikipedia', url: 'https://en.wikipedia.org/wiki/Mode_7' },
      { title: 'Lode’s raycasting tutorial (floor casting)', url: 'https://lodev.org/cgtutor/raycasting2.html', note: 'the same per-row floor math' },
    ],
  },
});

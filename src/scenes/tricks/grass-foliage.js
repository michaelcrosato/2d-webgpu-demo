// Wind, grass & foliage: tens of thousands of instanced grass blades whose bend is simulated per blade
// in a compute shader (wind torque + push-away from the player/mouse + spring back), then bent in the
// vertex shader along a Bézier curve. Trees & bushes are subdivided quads whose vertices sway with
// a weight that grows with height. Everything is drawn into a 4× MSAA target for smooth thin blades.

import { getAtlas, makeCanvas, rng } from '../../core/assets.js';
import { overlayTag, keyAxis, KEYS, anyKey, clamp } from './_shared.js';

const ROWS = 32;
const SEGS = 4; // blade segments (strip of 2*SEGS+1 vertices)
const MAX_BLADES = 200000;
const MAX_QUADS = 64;
const LEAVES = 220;

// ------------------------------------------------------------------------------------ tree art

const TREE_UV = {
  oak: [0, 0, 256, 384],
  pine: [256, 0, 192, 384],
  birch: [448, 0, 192, 384],
  bush: [640, 224, 256, 160],
  bush2: [640, 40, 256, 160],
};

function paintTrees() {
  const c = makeCanvas(1024, 512);
  const g = c.getContext('2d');
  const R = rng(9);
  const blobTree = (cx, cy, rx, ry, n, cols, ox) => {
    // three passes: dark volume, mid tone up-left, highlights further up-left
    const passes = [
      [cols[0], 1, 0, 0],
      [cols[1], 0.8, -0.12, -0.14],
      [cols[2], 0.55, -0.25, -0.3],
      [cols[3], 0.3, -0.35, -0.42],
    ];
    for (const [col, k, dx, dy] of passes) {
      for (let i = 0; i < n; i++) {
        const a = R() * Math.PI * 2;
        const r = Math.sqrt(R());
        const x = ox + cx + Math.cos(a) * r * rx * k + dx * rx;
        const y = cy + Math.sin(a) * r * ry * k + dy * ry;
        g.fillStyle = col;
        g.beginPath();
        g.arc(x, y, (0.1 + R() * 0.1) * rx * (0.6 + k * 0.4), 0, Math.PI * 2);
        g.fill();
      }
    }
  };
  const trunk = (x0, y0, x1, y1, w0, w1, col) => {
    g.fillStyle = col;
    g.beginPath();
    g.moveTo(x0 - w0 / 2, y0);
    g.lineTo(x1 - w1 / 2, y1);
    g.lineTo(x1 + w1 / 2, y1);
    g.lineTo(x0 + w0 / 2, y0);
    g.fill();
  };
  // oak
  trunk(128, 384, 126, 170, 30, 14, '#4a3121');
  trunk(124, 384, 122, 200, 10, 5, '#6b4a33');
  trunk(126, 250, 70, 170, 9, 4, '#4a3121');
  trunk(128, 240, 190, 160, 9, 4, '#4a3121');
  blobTree(128, 150, 112, 112, 150, ['#1f4a24', '#2f6b2f', '#4f9440', '#8cc865'], 0);
  // pine: stacked jagged tiers
  {
    const ox = 256;
    trunk(ox + 96, 384, ox + 96, 300, 16, 12, '#4a3121');
    for (let t = 0; t < 7; t++) {
      const y = 330 - t * 44;
      const w = 90 - t * 11;
      for (const [col, sh] of [['#163b26', 0], ['#22573a', -4], ['#35744a', -9]]) {
        g.fillStyle = col;
        g.beginPath();
        g.moveTo(ox + 96, y - 70 + sh);
        for (let k = 0; k <= 8; k++) {
          const xx = ox + 96 - w + (k / 8) * w * 2;
          g.lineTo(xx, y + (k % 2 ? 10 : 0) + sh * 0.3);
        }
        g.closePath();
        g.fill();
      }
    }
  }
  // birch
  {
    const ox = 448;
    trunk(ox + 96, 384, ox + 98, 110, 14, 7, '#e8e4da');
    for (let i = 0; i < 16; i++) {
      g.fillStyle = '#2a2a2a';
      g.fillRect(ox + 89 + R() * 8, 130 + R() * 240, 5 + R() * 6, 2);
    }
    trunk(ox + 97, 220, ox + 60, 150, 5, 2, '#e8e4da');
    blobTree(96, 150, 72, 120, 120, ['#3c6b2a', '#5d9136', '#93c24c', '#cfe680'], ox);
  }
  // bushes
  blobTree(128, 330, 112, 54, 110, ['#1f4a24', '#2f6b2f', '#4f9440', '#8cc865'], 640);
  blobTree(128, 130, 112, 60, 110, ['#2b4d1f', '#46732e', '#79a548', '#b8d870'], 640);
  // small berries/flowers on bush2
  for (let i = 0; i < 26; i++) {
    g.fillStyle = i % 2 ? '#f59bb5' : '#ffffff';
    g.beginPath();
    g.arc(640 + 40 + R() * 176, 100 + R() * 70, 3, 0, Math.PI * 2);
    g.fill();
  }
  return c;
}

// ------------------------------------------------------------------------------------ shaders

const COMMON = /* wgsl */ `
struct Blade { root: vec2f, hgt: f32, wid: f32, depth: f32, shade: f32, flower: f32, stiff: f32 };

fn rowY(d: f32) -> f32 { return u.horizon + (u.res.y * 1.04 - u.horizon) * pow(d, 1.35); }

fn bladeAt(i: u32) -> Blade {
  let rows = f32(${ROWS});
  let perRow = ceil(u.count / rows);
  let r = floor(f32(i) / perRow);
  let k = f32(i) - r * perRow;
  let h = hash13(f32(i) * 0.731 + 1.3);
  let h2 = hash13(f32(i) * 1.917 + 7.1);
  var b: Blade;
  b.depth = clamp((r + h.x) / rows, 0.0, 1.0);
  let sc = mix(0.3, 1.3, b.depth) * u.res.y / 700.0;
  b.root = vec2f((k + h.y) / perRow * (u.res.x + 60.0) - 30.0, rowY(b.depth));
  b.hgt = (34.0 + 30.0 * h.z * h.z) * sc * u.grassH;
  b.wid = (3.2 + 2.2 * h2.x) * sc;
  b.shade = h2.y;
  b.flower = step(1.0 - u.flowerRate, h2.z);
  b.stiff = u.stiffness * (0.75 + 0.5 * h.x);
  if (b.flower > 0.5) { b.hgt *= 1.25; }
  return b;
}

// traveling gusts: noise in (x - t·speed) moves across the screen in the wind direction
fn windAt(x: f32, t: f32) -> f32 {
  let xs = x / u.res.y;
  let ph = xs * u.gustFreq - t * u.gustSpeed * u.direction;
  let g = valueNoise(vec2f(ph, 0.5)) * 0.65 + valueNoise(vec2f(ph * 2.7, 3.1)) * 0.35;
  let gust = smoothstep(0.25, 0.85, g);
  return u.strength * (u.direction * (0.3 + 0.9 * gust) + 0.12 * sin(t * 3.1 + xs * 9.0));
}

fn toClip(p: vec2f) -> vec4f { return vec4f(p.x / u.res.x * 2.0 - 1.0, 1.0 - p.y / u.res.y * 2.0, 0.0, 1.0); }

fn pushTorque(root: vec2f, mid: vec2f, pz: vec4f) -> f32 {
  if (pz.z <= 0.0) { return 0.0; }
  let d = (mid - pz.xy) * vec2f(1.0, 1.6);
  let dist = length(d);
  let k = 1.0 - smoothstep(0.0, pz.z, dist);
  var sgn = 1.0;
  if (d.x < 0.0) { sgn = -1.0; }
  return sgn * k * k * pz.w * step(abs(root.y - pz.y), pz.z * 0.9);
}
`;

const SIM = /* wgsl */ `
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u32(u.count)) { return; }
  let b = bladeAt(i);
  var s = st[i];
  let mid = b.root + vec2f(0.0, -b.hgt * 0.5);
  var torque = windAt(b.root.x, u.time) * 7.0 * (u.stiffness / 14.0);
  torque += pushTorque(b.root, mid, u.p0) + pushTorque(b.root, mid, u.p1);
  let damp = 2.0 * sqrt(b.stiff) * 0.22;
  s.y += (torque - b.stiff * s.x - damp * s.y) * u.dt;
  s.x = clamp(s.x + s.y * u.dt, -1.45, 1.45);
  st[i] = s;
}
`;

const BLADE = /* wgsl */ `
struct VO {
  @builtin(position) pos: vec4f,
  @location(0) t: f32,
  @location(1) side: f32,
  @location(2) @interpolate(flat) col: vec3f,
  @location(3) @interpolate(flat) depth: f32,
};
fn bendPoint(b: Blade, a: f32, t: f32) -> vec2f {
  let tip = b.root + b.hgt * vec2f(sin(a) * 0.95, -cos(a));
  let ctrl = b.root + vec2f(0.0, -b.hgt * 0.55);
  let it = 1.0 - t;
  return it * it * b.root + 2.0 * it * t * ctrl + t * t * tip;
}
fn bladeAngle(i: u32, b: Blade) -> f32 {
  // simulated bend + a fast flutter that grows with the local wind
  let w = windAt(b.root.x, u.time);
  return st[i].x + 0.06 * sin(u.time * (7.0 + b.shade * 4.0) + b.root.x * 0.05) * (0.3 + abs(w));
}
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let b = bladeAt(ii);
  let a = bladeAngle(ii, b);
  let k = vi / 2u;
  let t = min(f32(k) / f32(${SEGS}), 1.0);
  let p = bendPoint(b, a, t);
  let p2 = bendPoint(b, a, min(t + 0.05, 1.0));
  let p1 = bendPoint(b, a, max(t - 0.05, 0.0));
  let tg = normalize(p2 - p1 + vec2f(0.0, -0.0001));
  let nrm = vec2f(-tg.y, tg.x);
  var sd = -0.5;
  if (vi % 2u == 1u) { sd = 0.5; }
  let w = b.wid * (1.0 - t * 0.92);
  var o: VO;
  o.pos = toClip(p + nrm * w * sd);
  o.t = t;
  o.side = sd;
  let v = mix(0.75, 1.15, b.shade);
  o.col = vec3f(v);
  o.depth = b.depth;
  return o;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  var c = mix(u.cBase, u.cTip, smoothstep(0.0, 1.0, i.t)) * i.col;
  c *= 1.0 - 0.25 * abs(i.side) * 2.0 * (1.0 - i.t);   // rounded blade
  c *= mix(0.55, 1.0, smoothstep(0.0, 0.35, i.t));       // ambient occlusion at the root
  c = mix(u.haze, c, mix(0.35, 1.0, smoothstep(0.0, 0.7, i.depth)));
  return vec4f(c, 1.0);
}
`;

const FLOWER = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) @interpolate(flat) col: vec3f, @location(2) @interpolate(flat) depth: f32 };
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let b = bladeAt(ii);
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  var o: VO;
  let a = st[ii].x + 0.06 * sin(u.time * (7.0 + b.shade * 4.0) + b.root.x * 0.05) * (0.3 + abs(windAt(b.root.x, u.time)));
  let tip = b.root + b.hgt * vec2f(sin(a) * 0.95, -cos(a));
  let size = b.wid * 2.4 * b.flower;            // non-flowers collapse to nothing
  o.pos = toClip(tip + corners[vi] * size);
  o.q = corners[vi];
  let hcol = floor(b.shade * 4.0);
  var c = vec3f(1.0, 0.55, 0.75);
  if (hcol < 1.0) { c = vec3f(1.0, 0.86, 0.3); }
  else if (hcol < 2.0) { c = vec3f(0.97, 0.97, 1.0); }
  else if (hcol < 3.0) { c = vec3f(0.72, 0.55, 1.0); }
  o.col = c;
  o.depth = b.depth;
  return o;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  let r = length(i.q);
  let ang = atan2(i.q.y, i.q.x);
  let petal = 0.62 + 0.3 * cos(ang * 5.0);
  if (r > petal) { discard; }
  var c = i.col * (0.85 + 0.15 * (1.0 - r));
  c = mix(c, vec3f(1.0, 0.75, 0.2), 1.0 - smoothstep(0.18, 0.26, r));
  c = mix(u.haze, c, mix(0.35, 1.0, smoothstep(0.0, 0.7, i.depth)));
  return vec4f(c, 1.0);
}
`;

const QUAD = /* wgsl */ `
struct Quad { a: vec4f, uv: vec4f, m: vec4f };   // a: anchor x,y (bottom center), w, h. m: flex, phase, fog, flipX
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) fog: f32 };
const GX: u32 = 6u;
const GY: u32 = 10u;
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let q = quads[ii];
  // a GX x GY grid of cells, 6 vertices each, so the sprite can bend smoothly
  let cellI = vi / 6u;
  var corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let cell = vec2f(f32(cellI % GX), f32(cellI / GX));
  let l = (cell + corners[vi % 6u]) / vec2f(f32(GX), f32(GY));   // 0..1, y down
  let hf = 1.0 - l.y;                                            // 0 at the roots, 1 at the top
  var p = vec2f(q.a.x + (l.x - 0.5) * q.a.z, q.a.y - hf * q.a.w);
  // sway: displacement grows with height squared, so the trunk base never moves
  let w = windAt(q.a.x, u.time - 0.15 * hf);
  let sway = (w * 0.9 + 0.25 * sin(u.time * 1.3 + q.m.y) * u.strength) * q.m.x * hf * hf * q.a.w * 0.08;
  // leaf flutter: small fast wobble in the upper part
  let flutter = sin(u.time * 6.0 + l.x * 9.0 + l.y * 13.0 + q.m.y) * q.a.w * 0.006 * hf * (0.3 + abs(w)) * q.m.x;
  p.x += sway + flutter;
  p.y += abs(sway) * 0.12 * hf;
  var o: VO;
  o.pos = toClip(p);
  var lu = l.x;
  if (q.m.w > 0.5) { lu = 1.0 - lu; }
  o.uv = mix(q.uv.xy, q.uv.zw, vec2f(lu, l.y));
  o.fog = q.m.z;
  return o;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  let c = textureSample(tex, samp, i.uv);
  if (c.a < 0.5) { discard; }
  return vec4f(mix(c.rgb, u.haze, i.fog), 1.0);
}
`;

const LEAF = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) @interpolate(flat) col: vec3f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let h = hash13(f32(ii) * 3.17 + 0.5);
  let h2 = hash13(f32(ii) * 7.31 + 2.5);
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let W = u.res.x + 120.0;
  let w = windAt(h.x * u.res.x, u.time);
  let travel = u.time * (20.0 + 90.0 * h.y) * u.strength * u.direction;
  let x = fmod(h.x * W + travel + sin(u.time * 0.7 + h.z * 6.0) * 30.0, W) - 60.0;
  let y = u.horizon * (0.25 + 1.1 * h.y) + sin(u.time * (1.0 + h.z) + h.x * 20.0) * 26.0 + cos(u.time * 2.3 + h.y * 9.0) * 8.0;
  let rot = u.time * (2.0 + 4.0 * h.z) * (0.4 + abs(w)) + h.x * 10.0;
  let s = (5.0 + 5.0 * h2.x) * max(u.res.y / 700.0, 0.6);
  let c = corners[vi];
  let rc = vec2f(c.x * cos(rot) - c.y * 0.5 * sin(rot), c.x * sin(rot) + c.y * 0.5 * cos(rot)) * s * u.leaves;
  var o: VO;
  o.pos = toClip(vec2f(x, y) + rc);
  o.q = c;
  o.col = mix(vec3f(0.85, 0.45, 0.12), vec3f(0.55, 0.75, 0.2), h2.y);
  return o;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  if (abs(i.q.y) > 1.0 - i.q.x * i.q.x) { discard; }
  return vec4f(i.col * (0.8 + 0.2 * i.q.y), 1.0);
}
`;

const BG = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  let p = i.pos.xy;
  let H = u.res.y;
  let hz = u.horizon;
  let ty = clamp(p.y / hz, 0.0, 1.0);
  var c = mix(u.sky0, u.sky1, ty);
  // sun
  let sp = vec2f(u.res.x * 0.78, hz * 0.32);
  c += vec3f(1.0, 0.9, 0.6) * exp(-length(p - sp) / (H * 0.12)) * 0.4;
  // clouds drifting with the wind
  let drift = u.time * 18.0 * u.strength * u.direction;
  let cq = vec2f((p.x - drift) / H * 2.2, p.y / H * 6.0);
  let cl = smoothstep(0.52, 0.75, fbm(cq, 4) * 0.5 + 0.5) * smoothstep(0.75, 0.2, ty);
  c = mix(c, vec3f(1.0, 0.99, 0.97), cl * 0.85 * u.cloudAmt);
  // distant hills (two layers) and a far tree line
  let x = p.x / H;
  let h1 = hz - H * (0.12 + 0.05 * sin(x * 1.7 + 1.0) + 0.03 * sin(x * 4.3));
  c = mix(c, mix(u.haze, u.hill1, 0.55), step(h1, p.y));
  let h2 = hz - H * (0.05 + 0.025 * sin(x * 2.9 + 2.0)) - H * 0.02 * smoothstep(0.3, 0.9, valueNoise(vec2f(x * 22.0, 1.0)));
  c = mix(c, mix(u.haze, u.hill2, 0.75), step(h2, p.y));
  // ground behind the blades
  if (p.y > hz) {
    let gy = (p.y - hz) / (H - hz);
    c = mix(mix(u.haze, u.cBase * 1.2, 0.6), u.cBase * 0.55, gy);
    c *= 0.9 + 0.1 * valueNoise(p / 7.0);
  }
  // wind field visualisation (the "Wind field" example)
  if (u.showWind > 0.5 && p.y < hz) {
    let w = windAt(p.x, u.time);
    let heat = clamp(abs(w) / 2.0, 0.0, 1.0);
    c = mix(c, mix(vec3f(0.1, 0.3, 0.8), vec3f(1.0, 0.45, 0.15), heat), 0.35);
    // arrows on a grid, length = local wind
    let cs = 46.0;
    let cid = floor(p / cs);
    let cc = (cid + 0.5) * cs;
    let wc = windAt(cc.x, u.time);
    let len = clamp(wc * 14.0, -20.0, 20.0);
    let a0 = cc - vec2f(len, 0.0);
    let a1 = cc + vec2f(len, 0.0);
    let hd = sign(len + 0.0001) * 6.0;
    var d = sdSegment(p, a0, a1);
    d = min(d, sdSegment(p, a1, a1 - vec2f(hd, 4.5)));
    d = min(d, sdSegment(p, a1, a1 - vec2f(hd, -4.5)));
    c = mix(c, vec3f(1.0), (1.0 - smoothstep(0.8, 1.8, d)) * 0.85 * step(cs * 0.5, hz - cc.y + cs * 0.5));
    // graph of wind(x) along a baseline
    let base = hz - H * 0.07;
    let gy = base - w * H * 0.045;
    c = mix(c, vec3f(1.0, 0.9, 0.3), 1.0 - smoothstep(1.0, 2.2, abs(p.y - gy)));
    c = mix(c, vec3f(1.0), (1.0 - smoothstep(0.4, 1.2, abs(p.y - base))) * 0.35);
  }
  return vec4f(c, 1.0);
}
`;

// ------------------------------------------------------------------------------------ palettes

const LOOKS = {
  wind: { sky0: '#1d2b53', sky1: '#5a7bb5', haze: '#7d93bf', hill1: '#33406b', hill2: '#2a4a4a', cBase: '#1d5c33', cTip: '#a7f070', clouds: 0.4 },
  meadow: { sky0: '#3b7dd8', sky1: '#bfe3ff', haze: '#b7d9ee', hill1: '#5f8fb8', hill2: '#5e9c55', cBase: '#2c6e2a', cTip: '#c8ec6a', clouds: 1 },
  forest: { sky0: '#4d7fa8', sky1: '#e9e2c8', haze: '#c9d3c0', hill1: '#6f8c8a', hill2: '#3f6a45', cBase: '#24502a', cTip: '#9cc95a', clouds: 0.8 },
};

export default {
  interaction: 'Sweep the mouse through the grass',
  keys: true,
  examples: [
    {
      id: 'wind',
      label: 'Wind field',
      kind: 'Abstract',
      note: 'The sky shows the <b>wind field</b>: noise that slides sideways over time, so gusts travel across the screen (arrows & yellow graph = local wind). Each blade feels the wind at its root as a torque, springs back, and overshoots — that delay is why the gusts look like waves rolling through.',
      params: { count: 30000, strength: 1.2, gustFreq: 1.6, gustSpeed: 1.2, flowerRate: 0.0, showWind: true },
    },
    {
      id: 'meadow',
      label: 'Interactive meadow',
      kind: 'In a game',
      note: 'A hero walking through a flowery meadow: blades near the player (and the mouse) are pushed away and spring back with a wobble, because every blade has its own little physics state updated by a compute shader each frame.',
      params: { count: 60000, strength: 0.6, gustFreq: 1.0, gustSpeed: 0.8, flowerRate: 0.018, showWind: false },
      hint: '←/→ (A/D) walk · sweep the mouse through the grass',
    },
    {
      id: 'forest',
      label: 'Swaying forest',
      kind: 'In a game',
      note: 'Trees and bushes are ordinary sprites drawn as a grid of small triangles. The vertex shader moves each vertex sideways by <code>wind × height²</code>, so trunks stay planted while crowns sway, plus a fast flutter for the leaves.',
      params: { count: 40000, strength: 1.0, gustFreq: 0.7, gustSpeed: 0.7, flowerRate: 0.02, showWind: false },
    },
  ],
  controls: [
    { type: 'heading', label: 'Wind' },
    { type: 'slider', key: 'strength', label: 'Wind strength', min: 0, max: 2.5, step: 0.01, value: 1 },
    { type: 'slider', key: 'direction', label: 'Direction', min: -1, max: 1, step: 0.01, value: 1, help: '−1 = blowing left, +1 = blowing right.' },
    { type: 'slider', key: 'gustFreq', label: 'Gust size', min: 0.2, max: 5, step: 0.01, value: 1, help: 'Spatial frequency of the gust noise: small = many short gusts.' },
    { type: 'slider', key: 'gustSpeed', label: 'Gust speed', min: 0, max: 4, step: 0.01, value: 1, help: 'How fast gusts travel across the field.' },
    { type: 'heading', label: 'Grass' },
    { type: 'slider', key: 'count', label: 'Blades', min: 2000, max: MAX_BLADES, step: 1000, value: 60000, log: true, format: (v) => Math.round(v).toLocaleString(), help: 'Each blade is one instance: 9 vertices, 7 triangles.' },
    { type: 'slider', key: 'stiffness', label: 'Stiffness', min: 2, max: 40, step: 0.1, value: 14, help: 'Spring constant of each blade: soft grass lags and sways more.' },
    { type: 'slider', key: 'grassH', label: 'Grass height', min: 0.4, max: 2, step: 0.01, value: 1 },
    { type: 'slider', key: 'push', label: 'Push radius', min: 0, max: 220, step: 1, value: 90, help: 'How far the player / mouse parts the grass.' },
    { type: 'slider', key: 'flowerRate', label: 'Flowers', min: 0, max: 0.2, step: 0.001, value: 0.018, help: 'Fraction of blades that carry a flower head.' },
    { type: 'toggle', key: 'showWind', label: 'Show wind field', value: false },
  ],
  about: {
    summary: 'Grass and trees that move with the wind are a vertex-shader trick: every blade or leaf vertex is displaced by a wind function, weighted by how high it is above its root. Add per-blade physics on the GPU and the player can part the grass.',
    what: `<p>Tens of thousands of grass blades, flowers and swaying trees. Gusts roll across the field, and the grass bends away from the player
      and the mouse, then springs back with a wobble.</p>`,
    how: `<ol>
      <li><b>Instancing</b>: one draw call renders every blade. Its position, height, color and stiffness come from a hash of the instance index — no per-blade data uploaded.</li>
      <li><b>Wind field</b>: <code>wind(x, t) = noise(x · gustSize − t · gustSpeed)</code>. Because the noise is sampled at <i>x − t·speed</i>, patterns slide sideways: gusts travel.</li>
      <li><b>Per-blade physics (compute shader)</b>: each blade stores a bend angle and angular speed. Every frame: torque from the wind + push away from the player/mouse,
        a spring pulling back to upright, and damping. The overshoot makes the grass ripple naturally.</li>
      <li><b>Bending (vertex shader)</b>: the blade is a quadratic Bézier from root to tip; the tip rotates by the bend angle while the root stays put. Width tapers to zero at the tip.</li>
      <li><b>Trees</b>: a sprite drawn as a 6×10 grid of triangles; each vertex moves by <code>wind · height²</code> (so the trunk base never slides), plus a fast flutter for leaves.</li>
      <li>Back-to-front order comes from the instance order (rows from the horizon forward); a 4× MSAA target keeps the thin blades smooth.</li>
    </ol>`,
    uses: [
      { title: 'Platformers & adventure games', text: 'Ori, Rayman, Hollow Knight and many indies sway foliage and part grass around the player for life and feedback.' },
      { title: 'Top-down & farming games', text: 'Stardew-like grass tufts that rustle when walked through; Zelda-style tall grass.' },
      { title: 'Atmosphere & storytelling', text: 'Storm build-up (wind rising), calm mornings, magic shockwaves flattening the grass.' },
    ],
    try: [
      'On <b>Wind field</b>, set <i>Gust speed</i> to 0: the gusts freeze in place and the grass just leans.',
      'Lower <i>Stiffness</i> to 3: the grass becomes floppy and lags far behind the gusts.',
      'In the <b>Meadow</b>, walk with ←/→ and swipe the mouse through the grass.',
      'Drag <i>Direction</i> slowly from +1 to −1 and watch trees, grass, clouds and leaves all turn around.',
      'Push <i>Blades</i> to 200,000 — still one draw call and one compute dispatch.',
    ],
    ask: [
      'instanced grass blades bent by a wind field in the vertex shader',
      'grass that bends away from the player and springs back',
      'traveling wind gusts made from scrolling noise',
      'trees that sway with vertex displacement weighted by height',
      'GPU per-blade spring simulation in a compute shader',
    ],
    perf: `<p>Per blade: one compute thread (a few dozen flops) and 9 vertices. 60k blades ≈ 0.5M vertices — easy for any GPU. The cost that grows fastest is
      <b>overdraw</b>: many thin overlapping blades. MSAA ×4 multiplies fragment-store bandwidth; on weak GPUs use fewer, wider blades.</p>`,
    api: `<p>WebGPU: blade physics lives in a <b>storage buffer</b> updated by a compute shader and read directly by the vertex shader. In WebGL2 you could keep
      only the stateless part (wind displacement in the vertex shader) — interactive bending with springs would need transform feedback or float-texture tricks.</p>`,
    code: [
      {
        title: 'Compute: one thread per blade',
        lang: 'wgsl',
        src: `let b = bladeAt(i);                       // position/height/stiffness from a hash
var s = st[i];                            // (angle, angular speed)
var torque = windAt(b.root.x, u.time) * 7.0;
torque += pushTorque(b.root, mid, u.p0) + pushTorque(b.root, mid, u.p1);
s.y += (torque - b.stiff * s.x - damp * s.y) * u.dt;  // spring + damping
s.x = clamp(s.x + s.y * u.dt, -1.45, 1.45);
st[i] = s;`,
      },
      {
        title: 'Vertex: bend along a Bézier, sway trees by height²',
        lang: 'wgsl',
        src: `let tip  = b.root + b.hgt * vec2f(sin(a) * 0.95, -cos(a));
let ctrl = b.root + vec2f(0.0, -b.hgt * 0.55);   // keeps the base upright
let p = mix(mix(b.root, ctrl, t), mix(ctrl, tip, t), t);
// trees:
let hf = 1.0 - l.y;                            // 0 at the trunk base
p.x += windAt(q.a.x, u.time) * flex * hf * hf * height * 0.08;`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const atlas = await getAtlas();
    const treeTex = gpu.textureFromImage(paintTrees(), { label: 'trees', mips: true });
    const heroTex = gpu.textureFromImage(atlas.canvas, { label: 'atlas' });
    const U = gpu.uniforms(
      {
        res: 'vec2f', time: 'f32', dt: 'f32', count: 'f32', horizon: 'f32', strength: 'f32', direction: 'f32',
        gustFreq: 'f32', gustSpeed: 'f32', stiffness: 'f32', grassH: 'f32', flowerRate: 'f32', showWind: 'f32', cloudAmt: 'f32', leaves: 'f32',
        p0: 'vec4f', p1: 'vec4f', cBase: 'vec3f', cTip: 'vec3f', haze: 'vec3f', sky0: 'vec3f', sky1: 'vec3f', hill1: 'vec3f', hill2: 'vec3f',
      },
      'G',
    );
    const state = gpu.storage(MAX_BLADES * 8, 'blade-state');
    const quadBuf = gpu.storage(MAX_QUADS * 48, 'quads');
    const inc = ['hash', 'noise', 'sdf'];
    const sim = gpu.program({ label: 'grass-sim', bindings: { u: { uniform: U }, st: { storage: 'array<vec2f>', access: 'read_write' } }, include: inc, code: COMMON + SIM });
    const bladeProg = gpu.program({ label: 'grass-blades', bindings: { u: { uniform: U }, st: { storage: 'array<vec2f>', access: 'read' } }, include: inc, code: COMMON + BLADE });
    const flowerProg = gpu.program({ label: 'grass-flowers', bindings: { u: { uniform: U }, st: { storage: 'array<vec2f>', access: 'read' } }, include: inc, code: COMMON + FLOWER });
    const quadProg = gpu.program({
      label: 'sway-quads',
      bindings: { u: { uniform: U }, quads: { storage: 'array<Quad>', access: 'read' }, tex: { texture: true }, samp: { sampler: true } },
      include: inc,
      code: COMMON + QUAD,
    });
    const leafProg = gpu.program({ label: 'leaves', bindings: { u: { uniform: U } }, include: inc, code: COMMON + LEAF });
    const bgProg = gpu.program({ label: 'grass-bg', bindings: { u: { uniform: U } }, include: inc, code: COMMON + BG });
    const fmt = gpu.format;
    const MS = 4;
    const pBlade = bladeProg.renderPipeline({ format: fmt, topology: 'triangle-strip', sampleCount: MS });
    const pFlower = flowerProg.renderPipeline({ format: fmt, sampleCount: MS });
    const pQuad = quadProg.renderPipeline({ format: fmt, sampleCount: MS });
    const pLeaf = leafProg.renderPipeline({ format: fmt, sampleCount: MS });
    const pBg = bgProg.renderPipeline({ format: fmt, sampleCount: MS });
    let msaa = null;
    const makeMsaa = (w, h) => {
      msaa?.destroy();
      msaa = gpu.texture({ size: [w, h], format: fmt, sampleCount: MS, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'grass-msaa' });
    };
    makeMsaa(ctx.width, ctx.height);
    const treeView = treeTex.createView();
    const heroView = heroTex.createView();
    const hud = overlayTag(ctx, 'right:8px;bottom:8px');

    // scenery: depth 0 = horizon, 1 = bottom of the screen
    const R = rng(4);
    const forest = [];
    for (let i = 0; i < 26; i++) {
      const d = 0.03 + Math.pow(i / 26, 1.3) * 0.75 + R() * 0.03;
      const kinds = ['oak', 'pine', 'birch', 'oak', 'bush', 'bush2'];
      const kind = d > 0.55 ? (R() < 0.6 ? 'bush' : 'bush2') : kinds[Math.floor(R() * kinds.length)];
      forest.push({ x: R(), d, kind, phase: R() * 6.28, flip: R() < 0.5, scale: 0.8 + R() * 0.45 });
    }
    const meadowTrees = [
      { x: 0.12, d: 0.04, kind: 'oak', phase: 1, flip: false, scale: 1 },
      { x: 0.3, d: 0.02, kind: 'pine', phase: 2, flip: false, scale: 0.9 },
      { x: 0.86, d: 0.06, kind: 'birch', phase: 3, flip: true, scale: 1 },
      { x: 0.68, d: 0.03, kind: 'bush2', phase: 4, flip: false, scale: 1 },
    ];
    const hero = { x: 0.5, dir: 1, idle: 99, walk: 0, auto: 1 };
    const quadData = new Float32Array(MAX_QUADS * 12);

    return {
      resize(w, h) {
        makeMsaa(w, h);
      },
      frame(ctx) {
        const p = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const ex = ctx.example;
        const look = LOOKS[ex] || LOOKS.meadow;
        const count = Math.min(Math.round(p.count), ctx.testMode ? 8000 : MAX_BLADES);
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 30);
        const horizon = H * (ex === 'forest' ? 0.5 : 0.46);
        const rowY = (d) => horizon + (H * 1.04 - horizon) * Math.pow(d, 1.35);
        const scaleAt = (d) => (0.3 + d) * (H / 700);

        // hero (meadow): keyboard or a lazy autopilot stroll
        const heroD = 0.72;
        let heroOn = ex === 'meadow';
        if (heroOn && dt > 0) {
          const ax = keyAxis(ctx.keys, KEYS.left, KEYS.right);
          hero.idle = anyKey(ctx.keys, [KEYS.left, KEYS.right]) ? 0 : hero.idle + dt;
          let mv = ax;
          if (hero.idle > 2.5) {
            if (hero.x > 0.85) hero.auto = -1;
            if (hero.x < 0.15) hero.auto = 1;
            mv = hero.auto * (Math.sin(ctx.time * 0.4) > -0.6 ? 1 : 0);
          }
          hero.x = clamp(hero.x + mv * dt * 0.09, 0.03, 0.97);
          if (mv) hero.dir = Math.sign(mv);
          hero.walk += Math.abs(mv) * dt * 10;
          hero.moving = Math.abs(mv) > 0;
        }
        const heroPx = [hero.x * W, rowY(heroD)];
        const heroH = 64 * scaleAt(heroD) * 2.2;
        const pr = p.push * (H / 700);
        U.set('res', [W, H])
          .set('time', ctx.time)
          .set('dt', dt)
          .set('count', count)
          .set('horizon', horizon)
          .set('strength', p.strength)
          .set('direction', p.direction)
          .set('gustFreq', p.gustFreq)
          .set('gustSpeed', p.gustSpeed)
          .set('stiffness', p.stiffness)
          .set('grassH', p.grassH * (ex === 'forest' ? 0.8 : 1))
          .set('flowerRate', p.flowerRate)
          .set('showWind', p.showWind ? 1 : 0)
          .set('cloudAmt', look.clouds)
          .set('leaves', ex === 'meadow' ? 0 : 1)
          .set('p0', heroOn ? [heroPx[0], heroPx[1], pr * 0.8, 26] : [0, 0, 0, 0])
          .set('p1', ctx.pointer.over ? [ctx.pointer.x, ctx.pointer.y, pr, ctx.pointer.down ? 40 : 26] : [0, 0, 0, 0])
          .set('cBase', look.cBase)
          .set('cTip', look.cTip)
          .set('haze', look.haze)
          .set('sky0', look.sky0)
          .set('sky1', look.sky1)
          .set('hill1', look.hill1)
          .set('hill2', look.hill2);
        U.upload();

        const enc = ctx.encoder;
        if (dt > 0) sim.dispatch(enc, 'main', Math.ceil(count / 256), { u: U, st: state });

        // build sprite quads, grouped by the grass band they stand in
        const items = [];
        const scenery = ex === 'forest' ? forest : ex === 'meadow' ? meadowTrees : [];
        for (const t of scenery) {
          const r = TREE_UV[t.kind];
          const s = scaleAt(t.d) * t.scale * (t.kind.startsWith('bush') ? 0.75 : 1.25);
          const w = r[2] * s;
          const h = r[3] * s;
          const flex = t.kind === 'pine' ? 0.6 : t.kind.startsWith('bush') ? 0.7 : 1;
          items.push({ d: t.d, data: [t.x * W, rowY(t.d) + 4 * s, w, h, r[0] / 1024, r[1] / 512, (r[0] + r[2]) / 1024, (r[1] + r[3]) / 512, flex, t.phase, (1 - Math.min(1, t.d * 2.2)) * 0.45, t.flip ? 1 : 0], tex: 'tree' });
        }
        if (heroOn) {
          const frame = hero.moving ? `hero_run_${Math.floor(hero.walk) % 4}` : `hero_idle_${Math.floor(ctx.time * 2) % 2}`;
          const uv = atlas.uv(frame);
          const fr = atlas.frames[frame];
          items.push({ d: heroD, data: [heroPx[0], heroPx[1] + heroH * 0.04, (heroH * fr.w) / fr.h, heroH, ...uv, 0, 0, 0, hero.dir < 0 ? 1 : 0], tex: 'hero' });
        }
        items.sort((a, b) => a.d - b.d);
        items.forEach((it, i) => quadData.set(it.data, i * 12));
        if (items.length) gpu.queue.writeBuffer(quadBuf, 0, quadData, 0, items.length * 12);

        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: msaa.createView(), resolveTarget: ctx.target, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'discard' }],
        });
        pass.setPipeline(pBg);
        pass.setBindGroup(0, bgProg.bind({ u: U }));
        pass.draw(3);
        const perRow = Math.ceil(count / ROWS);
        const drawBlades = (r0, r1) => {
          const a = Math.min(count, r0 * perRow);
          const b = Math.min(count, r1 * perRow);
          if (b <= a) return;
          pass.setPipeline(pBlade);
          pass.setBindGroup(0, bladeProg.bind({ u: U, st: state }));
          pass.draw(SEGS * 2 + 1, b - a, 0, a);
          if (p.flowerRate > 0) {
            pass.setPipeline(pFlower);
            pass.setBindGroup(0, flowerProg.bind({ u: U, st: state }));
            pass.draw(6, b - a, 0, a);
          }
        };
        const drawQuad = (i, it) => {
          pass.setPipeline(pQuad);
          pass.setBindGroup(0, quadProg.bind({ u: U, quads: quadBuf, tex: it.tex === 'hero' ? heroView : treeView, samp: it.tex === 'hero' ? 'nearest' : 'linear-mip' }));
          pass.draw(6 * 6 * 10, 1, 0, i);
        };
        // interleave: grass rows behind each sprite, then the sprite, then the rows in front
        let row = 0;
        items.forEach((it, i) => {
          const r = Math.floor(it.d * ROWS);
          if (r > row) {
            drawBlades(row, r);
            row = r;
          }
          drawQuad(i, it);
        });
        drawBlades(row, ROWS);
        if (ex !== 'meadow') {
          pass.setPipeline(pLeaf);
          pass.setBindGroup(0, leafProg.bind({ u: U }));
          pass.draw(6, ctx.testMode ? 60 : LEAVES);
        }
        pass.end();
        hud.textContent = `${count.toLocaleString()} blades · 1 compute dispatch · ${items.length} swaying sprites`;
      },
    };
  },
};

// Lines, Curves & Vector Graphics — a custom WebGPU scene.
//  * a JavaScript stroke tessellator (thick polylines -> triangles with miter/round/bevel joins,
//    butt/square/round caps, dashes) rendered into a 4× MSAA target,
//  * quadratic & cubic Bézier curves flattened into segments, with the de Casteljau construction,
//  * SDF capsules with glow (ShapeBatch) for lasers bouncing off mirrors,
//  * a verlet grappling rope and a marching-dash route on a treasure map.

import { Camera2D, ShapeBatch } from '../../core/batch.js';
import { labels } from './_shared.js';

// ------------------------------------------------------------------------------- tessellator
const FPV = 16; // floats per vertex: pos2, line2 (along, across), bary4 (xyz + unused), color4, style4

class Mesh {
  constructor() {
    this.data = new Float32Array(FPV * 8192);
    this.n = 0;
    this.tris = 0;
  }
  reset() {
    this.n = 0;
    this.tris = 0;
  }
  _grow() {
    const d = new Float32Array(this.data.length * 2);
    d.set(this.data);
    this.data = d;
  }
  _v(p, bx, by, bz, st) {
    if ((this.n + 1) * FPV > this.data.length) this._grow();
    const d = this.data;
    const o = this.n++ * FPV;
    d[o] = p[0];
    d[o + 1] = p[1];
    d[o + 2] = p[2];
    d[o + 3] = p[3];
    d[o + 4] = bx;
    d[o + 5] = by;
    d[o + 6] = bz;
    d[o + 7] = 0;
    d.set(st.color, o + 8);
    d[o + 12] = st.hw;
    d[o + 13] = st.dash;
    d[o + 14] = st.gap;
    d[o + 15] = st.flags;
  }
  /** a, b, c = [x, y, along, across] */
  tri(a, b, c, st) {
    this._v(a, 1, 0, 0, st);
    this._v(b, 0, 1, 0, st);
    this._v(c, 0, 0, 1, st);
    this.tris++;
  }
}

/**
 * Stroke a polyline into triangles.
 * o: { width, join: 'miter'|'round'|'bevel', cap: 'butt'|'square'|'round', closed, miterLimit,
 *      color: [r,g,b,a], dash (px, 0 = solid), gap, roundDash, wire, rope, offset (dash phase) }
 */
function stroke(mesh, pts, o) {
  const n = pts.length;
  if (n < 2) return;
  const hw = o.width / 2;
  const closed = !!o.closed && n > 2;
  const off = o.offset || 0;
  let dash = o.dash || 0;
  let gap = o.gap ?? dash * 0.7;
  if (dash > 0 && o.roundDash) {
    gap += Math.min(dash, 2 * hw);
    dash = Math.max(0.001, dash - 2 * hw);
  }
  const flags = (o.wire ? 1 : 0) | (o.roundDash ? 2 : 0) | (o.rope ? 4 : 0);
  const st = { color: o.color, hw, dash, gap, flags };
  const segs = closed ? n : n - 1;
  const along = [off];
  for (let i = 0; i < segs; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    along.push(along[i] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const dirs = [];
  for (let i = 0; i < segs; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-6;
    dirs.push([(b[0] - a[0]) / L, (b[1] - a[1]) / L]);
  }
  // segment bodies: one quad (two triangles) each
  for (let i = 0; i < segs; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const [dx, dy] = dirs[i];
    const nx = -dy * hw;
    const ny = dx * hw;
    const s0 = along[i];
    const s1 = along[i + 1];
    const a0 = [a[0] + nx, a[1] + ny, s0, hw];
    const a1 = [a[0] - nx, a[1] - ny, s0, -hw];
    const b0 = [b[0] + nx, b[1] + ny, s1, hw];
    const b1 = [b[0] - nx, b[1] - ny, s1, -hw];
    mesh.tri(a0, a1, b0, st);
    mesh.tri(a1, b1, b0, st);
  }
  // joins
  const fan = (cx, cy, s, ang0, ang1, sideSign) => {
    let delta = ang1 - ang0;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;
    const steps = Math.max(1, Math.ceil(Math.abs(delta) / (Math.PI / Math.max(6, Math.min(24, hw * 0.6)))));
    const c = [cx, cy, s, 0];
    let prev = [cx + Math.cos(ang0) * hw, cy + Math.sin(ang0) * hw, s, hw * sideSign];
    for (let k = 1; k <= steps; k++) {
      const a = ang0 + (delta * k) / steps;
      const cur = [cx + Math.cos(a) * hw, cy + Math.sin(a) * hw, s, hw * sideSign];
      mesh.tri(c, prev, cur, st);
      prev = cur;
    }
  };
  const first = closed ? 0 : 1;
  const last = closed ? n - 1 : n - 2;
  for (let i = first; i <= last; i++) {
    const d0 = dirs[(i - 1 + segs) % segs];
    const d1 = dirs[i % segs];
    const p = pts[i];
    const crs = d0[0] * d1[1] - d0[1] * d1[0];
    const dt = d0[0] * d1[0] + d0[1] * d1[1];
    if (Math.abs(crs) < 1e-5 && dt > 0) continue;
    const n0 = [-d0[1], d0[0]];
    const n1 = [-d1[1], d1[0]];
    const s = n0[0] * d1[0] + n0[1] * d1[1] > 0 ? -1 : 1; // outer side
    const s0 = along[i] ?? along[0];
    const sAt = closed && i === 0 ? along[0] : s0;
    const c = [p[0], p[1], sAt, 0];
    const o0 = [p[0] + n0[0] * hw * s, p[1] + n0[1] * hw * s, sAt, hw * s];
    const o1 = [p[0] + n1[0] * hw * s, p[1] + n1[1] * hw * s, sAt, hw * s];
    if (o.join === 'round') {
      fan(p[0], p[1], sAt, Math.atan2(n0[1] * s, n0[0] * s), Math.atan2(n1[1] * s, n1[0] * s), s);
    } else if (o.join === 'miter') {
      let mx = n0[0] + n1[0];
      let my = n0[1] + n1[1];
      const ml = Math.hypot(mx, my);
      const cosHalf = ml / 2;
      const limit = o.miterLimit ?? 4;
      if (ml > 1e-6 && 1 / cosHalf <= limit) {
        mx /= ml;
        my /= ml;
        const len = hw / cosHalf;
        const tip = [p[0] + mx * len * s, p[1] + my * len * s, sAt, hw * s];
        mesh.tri(c, o0, tip, st);
        mesh.tri(c, tip, o1, st);
      } else mesh.tri(c, o0, o1, st);
    } else {
      mesh.tri(c, o0, o1, st);
    }
  }
  // caps
  if (!closed && o.cap && o.cap !== 'butt') {
    const ends = [
      [pts[0], dirs[0], -1, along[0]],
      [pts[n - 1], dirs[segs - 1], 1, along[segs]],
    ];
    for (const [p, d, sign, s] of ends) {
      const nx = -d[1] * hw;
      const ny = d[0] * hw;
      if (o.cap === 'square') {
        const ex = d[0] * hw * sign;
        const ey = d[1] * hw * sign;
        const se = s + hw * sign;
        const a0 = [p[0] + nx, p[1] + ny, s, hw];
        const a1 = [p[0] - nx, p[1] - ny, s, -hw];
        const b0 = [p[0] + nx + ex, p[1] + ny + ey, se, hw];
        const b1 = [p[0] - nx + ex, p[1] - ny + ey, se, -hw];
        mesh.tri(a0, a1, b0, st);
        mesh.tri(a1, b1, b0, st);
      } else {
        const base = Math.atan2(ny, nx);
        // half circle from +n through the outward direction to -n
        const steps = Math.max(4, Math.min(24, Math.ceil(hw * 0.5)));
        const c = [p[0], p[1], s, 0];
        let prev = [p[0] + nx, p[1] + ny, s, hw];
        for (let k = 1; k <= steps; k++) {
          const a = base + (sign > 0 ? -1 : 1) * (Math.PI * k) / steps;
          const cur = [p[0] + Math.cos(a) * hw, p[1] + Math.sin(a) * hw, s, hw];
          mesh.tri(c, prev, cur, st);
          prev = cur;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------- curves
const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
/** de Casteljau: returns all construction levels [[P0..Pn], [..], ..., [point]] */
function casteljau(ctrl, t) {
  const levels = [ctrl];
  let cur = ctrl;
  while (cur.length > 1) {
    const next = [];
    for (let i = 0; i < cur.length - 1; i++) next.push(lerp2(cur[i], cur[i + 1], t));
    levels.push(next);
    cur = next;
  }
  return levels;
}
function flatten(ctrl, segments) {
  const out = [];
  for (let i = 0; i <= segments; i++) {
    const lv = casteljau(ctrl, i / segments);
    out.push(lv[lv.length - 1][0]);
  }
  return out;
}
function catmullRom(pts, perSeg) {
  const out = [];
  const P = (i) => pts[Math.max(0, Math.min(pts.length - 1, i))];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = P(i - 1);
    const p1 = P(i);
    const p2 = P(i + 1);
    const p3 = P(i + 2);
    for (let k = 0; k < perSeg; k++) {
      const t = k / perSeg;
      const t2 = t * t;
      const t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

// ------------------------------------------------------------------------------- shaders
const MESH_WGSL = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) line: vec2f,
  @location(1) bary: vec4f,
  @location(2) color: vec4f,
  @location(3) @interpolate(flat) style: vec4f,
};
@vertex fn vs_main(@location(0) pos: vec2f, @location(1) line: vec2f, @location(2) bary: vec4f,
                   @location(3) color: vec4f, @location(4) style: vec4f) -> VOut {
  var o: VOut;
  o.pos = vec4f(pos / u.resolution * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.line = line; o.bary = bary; o.color = color; o.style = style;
  return o;
}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  var c = i.color;
  let hw = i.style.x;
  let flags = u32(i.style.w + 0.5);
  if (i.style.y > 0.0) {
    // dashes: distance to the nearest dash along the line
    let dashLen = i.style.y;
    let period = dashLen + i.style.z;
    let s = fmod(i.line.x, period);
    var dist: f32;
    if ((flags & 2u) != 0u) {
      var dd = 0.0;
      if (s > dashLen) { dd = min(s - dashLen, period - s); }
      dist = length(vec2f(dd, i.line.y)) - hw;      // round dash ends
    } else {
      if (s <= dashLen) { dist = -min(s, dashLen - s); } else { dist = min(s - dashLen, period - s); }
    }
    c.a *= clamp(0.5 - dist / max(fwidth(dist), 0.0001), 0.0, 1.0);
  }
  if ((flags & 4u) != 0u) {
    // twisted rope: diagonal stripes + cylinder shading from the across coordinate
    let k = clamp(i.line.y / max(hw, 0.001), -1.0, 1.0);
    let stripe = smoothstep(0.35, 0.6, abs(fract((i.line.x + i.line.y * 1.3) / (hw * 2.2)) - 0.5) * 2.0);
    let shadeK = 0.55 + 0.45 * sqrt(max(1.0 - k * k, 0.0)) - 0.15 * k;
    c = vec4f(mix(c.rgb, c.rgb * 0.62, stripe) * shadeK, c.a);
  }
  if ((flags & 1u) != 0u) {
    // wireframe: distance to the triangle's edges from barycentric coordinates
    let b = i.bary.xyz;
    let w = fwidth(b);
    let e = min(min(b.x / max(w.x, 1e-5), b.y / max(w.y, 1e-5)), b.z / max(w.z, 1e-5));
    let edge = 1.0 - smoothstep(0.0, 1.2, e);
    c = vec4f(mix(c.rgb, vec3f(1.0), edge * 0.85), max(c.a, edge * 0.85));
  }
  if (c.a < 0.003) { discard; }
  return c;
}`;

const BG_WGSL = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
  return o;
}
fn cov(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }
fn grid(px: vec2f) -> vec3f {
  var c = mix(vec3f(0.035, 0.045, 0.075), vec3f(0.055, 0.05, 0.09), px.y / u.resolution.y);
  let g1 = abs(fract(px / 24.0 + 0.5) - 0.5) * 24.0;
  let g2 = abs(fract(px / 120.0 + 0.5) - 0.5) * 120.0;
  c += vec3f(0.03, 0.04, 0.07) * cov(min(g1.x, g1.y) - 0.5, 1.0);
  c += vec3f(0.05, 0.07, 0.12) * cov(min(g2.x, g2.y) - 0.6, 1.0);
  return c;
}
fn cubicPt(t: f32) -> vec2f {
  let s = 1.0 - t;
  return s * s * s * u.p0 + 3.0 * s * s * t * u.p1 + 3.0 * s * t * t * u.p2 + t * t * t * u.p3;
}
fn floorTiles(px: vec2f) -> vec3f {
  let ts = 72.0;
  let id = floor(px / ts);
  let f = fract(px / ts);
  let h = hash21(id);
  var c = mix(vec3f(0.05, 0.07, 0.085), vec3f(0.07, 0.09, 0.105), h);
  let e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * ts;
  c *= 0.6 + 0.4 * smoothstep(0.0, 3.0, e);
  c += vec3f(0.08, 0.1, 0.12) * cov(abs(f.y * ts - 2.0) - 0.7, 1.0) * 0.4;
  // floor lamps
  let lamp = step(0.86, h) * (0.6 + 0.4 * sin(u.time * 2.0 + h * 30.0));
  c += vec3f(0.1, 0.35, 0.4) * lamp * cov(length(f - vec2f(0.5)) * ts - 5.0, 1.0);
  let v = length((px - u.resolution * 0.5) / u.resolution.y);
  return c * (1.1 - 0.8 * v * v);
}
fn cave(px: vec2f) -> vec3f {
  let res = u.resolution;
  let p = px / res.y;
  var c = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.1, 0.13, 0.12), p.y);
  // distant columns
  let cx = p.x * 3.0;
  let col = abs(fract(cx) - 0.5) - 0.12 - 0.04 * sin(p.y * 9.0 + floor(cx) * 2.0);
  c = mix(c, vec3f(0.08, 0.1, 0.13), cov(col, 2.0 / res.y * 3.0) * 0.6);
  // ceiling and floor with noise edges
  let ceilY = 0.1 + 0.03 * fbm(vec2f(p.x * 6.0, 1.0), 3);
  let floorY = 0.9 + 0.03 * fbm(vec2f(p.x * 5.0, 7.0), 3);
  let pw = 1.5 / res.y;
  c = mix(c, vec3f(0.17, 0.15, 0.17) * (0.7 + 0.3 * valueNoise(px / 9.0)), cov(p.y - ceilY, pw));
  // stalactites
  let sx = fract(p.x * 7.0) - 0.5;
  let sid = floor(p.x * 7.0);
  let sl = 0.05 + 0.08 * hash11(sid);
  let stal = sdTriangle(vec2f(sx, p.y), vec2f(-0.22, ceilY - 0.01), vec2f(0.22, ceilY - 0.01), vec2f(0.0, ceilY + sl));
  c = mix(c, vec3f(0.14, 0.12, 0.15), cov(stal, pw * 7.0));
  // a soft light shaft from a crack in the ceiling
  let shaft = exp(-abs(p.x - 0.55 - (p.y - 0.1) * 0.35) * 9.0) * smoothstep(0.95, 0.15, p.y);
  c += vec3f(0.12, 0.16, 0.14) * shaft;
  c = mix(c, vec3f(0.12, 0.2, 0.11) * (0.8 + 0.2 * valueNoise(px / 6.0)), cov(floorY - p.y, pw));
  c = mix(c, vec3f(0.25, 0.45, 0.2), cov(abs(p.y - floorY) - 0.004, pw));
  // glowing mushrooms on the floor
  for (var k = 0; k < 6; k++) {
    let mxp = (0.06 + f32(k) * 0.155 + 0.04 * hash11(f32(k) * 3.7)) * u.split / res.y;
    let mp = vec2f(mxp, 0.9 + 0.03 * fbm(vec2f(mxp * 5.0, 7.0), 3) - 0.012);
    let gcol = mix(vec3f(0.2, 0.9, 1.0), vec3f(0.8, 0.4, 1.0), hash11(f32(k) * 9.1));
    let md = length((p - mp) * vec2f(1.0, 1.6));
    c += gcol * exp(-md * 28.0) * (0.35 + 0.15 * sin(u.time * 2.0 + f32(k)));
    c = mix(c, gcol, cov(md - 0.008, pw));
  }
  return c;
}
fn parchment(px: vec2f) -> vec3f {
  let res = u.resolution;
  let lp = (px - vec2f(u.split, 0.0)) / res.y;
  let n = fbm(px / 140.0, 4);
  var c = mix(vec3f(0.86, 0.76, 0.55), vec3f(0.74, 0.6, 0.4), smoothstep(-0.3, 0.6, n));
  // land masses
  let h = fbm(lp * 3.2 + vec2f(5.3, 1.1), 5) + 0.05;
  let pw = 1.5 / res.y;
  let land = cov(-h, pw * 3.0);
  c = mix(c, c * vec3f(0.78, 0.86, 0.95), (1.0 - land) * 0.45);
  c = mix(c, vec3f(0.42, 0.3, 0.18), cov(abs(h) - 0.004, pw * 3.0) * 0.8);
  c = mix(c, c * 0.88, cov(abs(h - 0.08) - 0.002, pw * 3.0) * land * 0.6);
  // waves in the sea
  let wv = abs(fract(h * 14.0) - 0.5);
  c = mix(c, c * 0.9, (1.0 - land) * cov(wv - 0.04, 0.03) * 0.5);
  // grid & frame
  let g = abs(fract((px - vec2f(u.split, 0.0)) / 90.0 + 0.5) - 0.5) * 90.0;
  c = mix(c, vec3f(0.45, 0.33, 0.2), cov(min(g.x, g.y) - 0.5, 1.0) * 0.18);
  let w = res.x - u.split;
  let q = px - vec2f(u.split + w * 0.5, res.y * 0.5);
  let frame = sdBox(q, vec2f(w * 0.5 - 10.0, res.y * 0.5 - 10.0));
  c = mix(c, vec3f(0.35, 0.22, 0.12), cov(abs(frame) - 2.0, 1.0));
  c *= 1.0 - 0.35 * smoothstep(-60.0, 0.0, frame);
  // compass rose
  let cr = px - vec2f(res.x - 70.0, 74.0);
  let rose = min(sdStar5(cr, 0.0, 0.5) + 1.0, sdRhombus(cr, vec2f(10.0, 44.0)));
  let rose2 = sdRhombus(cr, vec2f(40.0, 8.0));
  c = mix(c, vec3f(0.4, 0.26, 0.15), cov(min(rose, rose2), 1.0) * 0.85);
  c = mix(c, vec3f(0.62, 0.18, 0.12), cov(max(sdRhombus(cr, vec2f(10.0, 44.0)), cr.y), 1.0));
  c = mix(c, vec3f(0.4, 0.26, 0.15), cov(abs(length(cr) - 30.0) - 1.0, 1.0) * 0.7);
  return c;
}
@fragment fn fs_main(i: VSOut) -> @location(0) vec4f {
  let px = i.pos.xy;
  let mode = i32(u.mode);
  var c: vec3f;
  if (mode == 0) { c = grid(px); }
  else if (mode == 1) {
    c = grid(px);
    // reference: the exact curve, as a distance field
    var d = 1e9;
    if (u.curveKind > 0.5) {
      var prev = u.p0;
      for (var k = 1; k <= 48; k++) {
        let cur = cubicPt(f32(k) / 48.0);
        d = min(d, sdSegment(px, prev, cur));
        prev = cur;
      }
    } else {
      d = sdBezier(px, u.p0, u.p1, u.p2);
    }
    c += vec3f(0.2, 0.7, 1.0) * exp(-d / 10.0) * 0.18;
    c = mix(c, vec3f(0.45, 0.85, 1.0), cov(d - 0.8, 1.0) * 0.55);
  } else if (mode == 2) { c = floorTiles(px); }
  else {
    if (px.x < u.split) { c = cave(px); } else { c = parchment(px); }
    c = mix(c, vec3f(0.02), cov(abs(px.x - u.split) - 2.0, 1.0));
  }
  c += vec3f((ign(px) - 0.5) / 255.0);
  return vec4f(c, 1.0);
}`;

// ------------------------------------------------------------------------------- scene
const LBL = 'background:#000a;font-size:11px;padding:2px 7px;color:#dbe4f3';

export default {
  interaction: 'Drag the white handles.',
  examples: [
    {
      id: 'joins',
      label: 'Thick lines: joins & caps',
      kind: 'Abstract',
      hint: 'Drag the white handles. Turn on the wireframe to see the triangles.',
      note: 'GPUs can only fill triangles, so a thick line must be <b>tessellated</b>: one quad per segment, plus extra triangles at corners (<b>joins</b>) and ends (<b>caps</b>). The top row shows each option side by side.',
    },
    {
      id: 'bezier',
      label: 'Bézier curves',
      kind: 'Abstract',
      hint: 'Drag the control points. Lower “Segments” to see the curve flatten.',
      note: 'A Bézier curve is defined by control points. <b>de Casteljau’s algorithm</b> finds a point on it by repeated linear interpolation (the colored scaffolding). To draw it, we sample it into straight segments and stroke those. The thin blue line is the exact curve for reference.',
      params: { width: 10, wire: false },
    },
    {
      id: 'lasers',
      label: 'Laser mirrors',
      kind: 'In a game',
      hint: 'Aim the laser with the mouse.',
      note: 'Each bounce is a ray–segment intersection plus a reflection: <code>d − 2(d·n)n</code>. The beam is drawn as <b>SDF capsules</b> with additive glow — no tessellation needed, and the glow comes for free from the distance.',
    },
    {
      id: 'rope',
      label: 'Grapple rope & map route',
      kind: 'In a game',
      hint: 'Click in the cave to grapple a peg. Drag the map’s waypoints.',
      note: 'Left: a grappling rope simulated as a chain of points (verlet) and drawn as one tessellated stroke with round joins and a twisted-rope pattern. Right: a route through waypoints (Catmull–Rom spline), drawn with marching round-cap dashes.',
      params: { width: 8, wire: false },
    },
  ],
  controls: [
    { type: 'heading', label: 'Stroke' },
    { type: 'slider', key: 'width', label: 'Line width', min: 1, max: 90, step: 0.5, value: 36, unit: 'px', showFor: ['joins', 'bezier', 'rope'] },
    {
      type: 'select', key: 'join', label: 'Join', value: 'miter', showFor: ['joins'],
      options: [{ value: 'miter', label: 'Miter (sharp)' }, { value: 'round', label: 'Round' }, { value: 'bevel', label: 'Bevel (cut)' }],
      help: 'How two segments meet at a corner.',
    },
    {
      type: 'select', key: 'cap', label: 'Cap', value: 'round', showFor: ['joins'],
      options: [{ value: 'butt', label: 'Butt (flat)' }, { value: 'square', label: 'Square' }, { value: 'round', label: 'Round' }],
      help: 'How the open ends are finished.',
    },
    { type: 'slider', key: 'miterLimit', label: 'Miter limit', min: 1, max: 10, step: 0.1, value: 4, showFor: ['joins'], help: 'Very sharp corners make endless spikes; past this ratio the miter falls back to a bevel.' },
    { type: 'slider', key: 'dash', label: 'Dash length', min: 0, max: 120, step: 1, value: 0, unit: 'px', showFor: ['joins', 'bezier'], format: (v) => (v ? `${v} px` : 'solid'), help: 'Dashes are cut in the fragment shader from the distance along the line.' },
    { type: 'toggle', key: 'roundDash', label: 'Round dash ends', value: true, showFor: ['joins', 'bezier'] },
    { type: 'toggle', key: 'wire', label: 'Show triangles (wireframe)', value: true, showFor: ['joins', 'bezier', 'rope'] },
    { type: 'toggle', key: 'closed', label: 'Closed shape', value: false, showFor: ['joins'] },
    { type: 'heading', label: 'Curve', showFor: ['bezier'] },
    { type: 'select', key: 'curve', label: 'Curve type', value: 'cubic', showFor: ['bezier'], options: [{ value: 'quadratic', label: 'Quadratic (3 points)' }, { value: 'cubic', label: 'Cubic (4 points)' }] },
    { type: 'slider', key: 'segments', label: 'Segments', min: 1, max: 64, step: 1, value: 16, showFor: ['bezier'], help: 'How many straight pieces approximate the curve (“flattening”).' },
    { type: 'toggle', key: 'construct', label: 'Show de Casteljau construction', value: true, showFor: ['bezier'] },
    { type: 'slider', key: 't', label: 't (animated when 0)', min: 0, max: 1, step: 0.001, value: 0, showFor: ['bezier'], format: (v) => (v ? v.toFixed(3) : 'auto') },
    { type: 'heading', label: 'Laser', showFor: ['lasers'] },
    { type: 'slider', key: 'bounces', label: 'Max bounces', min: 0, max: 16, step: 1, value: 8, showFor: ['lasers'] },
    { type: 'slider', key: 'beam', label: 'Beam width', min: 1, max: 16, step: 0.5, value: 5, unit: 'px', showFor: ['lasers'] },
    { type: 'slider', key: 'glow', label: 'Glow', min: 0, max: 2, step: 0.01, value: 1, showFor: ['lasers'], help: 'exp(−distance) halo added on top with additive blending.' },
    { type: 'color', key: 'laserColor', label: 'Laser color', value: '#ff3b6b', showFor: ['lasers'] },
    { type: 'toggle', key: 'spin', label: 'Rotate mirrors', value: true, showFor: ['lasers'] },
  ],
  about: {
    summary: 'Lines look trivial but GPUs only draw triangles. Thick lines, corners, end caps, dashes and smooth curves all have to be built — either by tessellating them into triangles or by evaluating a distance function per pixel.',
    what: `<p><b>Joins &amp; caps</b>: a draggable polyline tessellated on the CPU into triangles (wireframe on), with a legend of every join and cap style.
      <b>Bézier</b>: quadratic/cubic curves with their de Casteljau construction and adjustable flattening. <b>Lasers</b>: reflections drawn with glowing SDF capsules.
      <b>Rope &amp; route</b>: the same techniques in game situations.</p>`,
    how: `<ol>
      <li><b>Tessellation</b>: for each segment, offset both endpoints by ±half-width along the normal → a quad (2 triangles).</li>
      <li><b>Joins</b> fill the wedge on the outside of each corner: <i>bevel</i> = one triangle, <i>miter</i> = extend both edges to a point (limited, or it spikes to infinity on sharp angles), <i>round</i> = a triangle fan.</li>
      <li><b>Caps</b>: <i>butt</i> stops at the endpoint, <i>square</i> extends by half the width, <i>round</i> adds a half-disc fan.</li>
      <li>Each vertex also carries its <b>distance along the line</b>; the fragment shader turns it into dashes with <code>fmod</code> (round dash ends use a tiny 2D distance).</li>
      <li><b>Curves</b>: evaluate <code>B(t)</code> by de Casteljau (lerp, lerp, lerp…) at N values of <i>t</i>, then stroke the resulting polyline. Too few segments = visible corners.</li>
      <li>Triangle edges are aliased, so this scene renders the tessellated geometry into a <b>4× MSAA</b> target. The SDF approach (lasers) computes coverage analytically instead.</li>
    </ol>`,
    uses: [
      { title: 'Lasers, beams & tethers', text: 'Bouncing lasers in puzzle games (e.g. The Talos Principle style), tractor beams, chain lightning, enemy tethers.' },
      { title: 'Ropes & cables', text: 'Grappling hooks (Terraria, Celeste-likes), power cables, fishing lines — a physics chain drawn as one smooth stroke.' },
      { title: 'Maps & UI', text: 'Routes and quest paths, graphs, radar sweeps, selection marquees (“marching ants”), node editors with Bézier wires.' },
      { title: 'Vector art', text: 'SVG-style shapes and fonts, hand-drawn outlines, trails and sword slashes (a polyline that fades along its length).' },
    ],
    try: [
      'On <b>Joins &amp; caps</b>, drag a handle to make a very sharp corner with <i>Miter</i> joins, then lower <i>Miter limit</i>.',
      'Turn on <i>Closed shape</i> and switch joins — the closing corner gets a join too.',
      'On <b>Bézier</b>, set <i>Segments</i> to 3 and watch the polyline cut corners against the blue reference curve.',
      'Set <i>Dash length</i> to 40 on a thick curve — dashes follow the curve because they are based on arc length.',
      'On <b>Laser mirrors</b>, raise <i>Max bounces</i> to 16 and sweep the mouse slowly.',
    ],
    ask: [
      'thick polylines with round joins and caps',
      'a dashed route line with marching ants',
      'a bouncing laser beam that reflects off mirrors',
      'a grappling hook rope with verlet physics',
      'Bézier curve editor with draggable control points',
      'anti-aliased lines with MSAA',
    ],
    perf: `<p>Tessellation runs on the CPU here: a few hundred triangles per frame is nothing. For thousands of dynamic lines, do it in a
      <b>vertex shader</b> (instanced quads that read the segment endpoints from a storage buffer) or a compute shader. The SDF capsule approach costs
      a little per pixel but needs no joins at all — overlapping round capsules hide the seams.</p>`,
    api: `<p>MSAA works in both APIs (WebGL2: <code>antialias: true</code> or a multisampled renderbuffer + blit; WebGPU: a <code>sampleCount: 4</code> texture with a
      <code>resolveTarget</code>, as here). WebGPU also lets a compute shader tessellate thousands of lines straight into a vertex buffer.</p>`,
    code: [
      {
        title: 'Tessellating one segment and a bevel / miter join (JavaScript)',
        lang: 'js',
        src: `const nx = -dy * hw, ny = dx * hw;                 // normal × half width
mesh.tri([a.x+nx, a.y+ny], [a.x-nx, a.y-ny], [b.x+nx, b.y+ny]);
mesh.tri([a.x-nx, a.y-ny], [b.x-nx, b.y-ny], [b.x+nx, b.y+ny]);

// at a corner: outer side s, miter direction m = normalize(n0 + n1)
const cosHalf = |n0 + n1| / 2;
if (1 / cosHalf <= miterLimit) {               // miter: extend to the tip
  const tip = p + m * s * (hw / cosHalf);
  mesh.tri(p, o0, tip); mesh.tri(p, tip, o1);
} else mesh.tri(p, o0, o1);                    // bevel`,
      },
      {
        title: 'de Casteljau & dashes',
        lang: 'js',
        src: `function casteljau(ctrl, t) {          // repeated lerp → point on the curve
  const levels = [ctrl];
  let cur = ctrl;
  while (cur.length > 1) {
    const next = [];
    for (let i = 0; i < cur.length - 1; i++) next.push(lerp2(cur[i], cur[i + 1], t));
    levels.push(next); cur = next;
  }
  return levels;                         // last level = [B(t)]
}
// WGSL: dash from distance along the line (round ends)
// let s = fmod(along, dash + gap);
// let dist = length(vec2f(max(s - dash, 0.0), across)) - halfWidth;`,
      },
    ],
    links: [
      { title: 'Matt DesLauriers — Drawing lines is hard', url: 'https://mattdesl.svbtle.com/drawing-lines-is-hard' },
      { title: 'A Primer on Bézier Curves (Pomax)', url: 'https://pomax.github.io/bezierinfo/' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const U = gpu.uniforms(
      { resolution: 'vec2f', time: 'f32', mode: 'f32', p0: 'vec2f', p1: 'vec2f', p2: 'vec2f', p3: 'vec2f', curveKind: 'f32', split: 'f32' },
      'P',
    );
    const bgProg = gpu.program({ label: 'lines-bg', bindings: { u: { uniform: U } }, include: ['math', 'hash', 'noise', 'sdf'], code: BG_WGSL });
    const bgPipe = bgProg.renderPipeline({ format: gpu.format, sampleCount: 4 });
    const meshProg = gpu.program({ label: 'lines-mesh', bindings: { u: { uniform: U } }, code: MESH_WGSL });
    const attrs = [
      { shaderLocation: 0, offset: 0, format: 'float32x2' },
      { shaderLocation: 1, offset: 8, format: 'float32x2' },
      { shaderLocation: 2, offset: 16, format: 'float32x4' },
      { shaderLocation: 3, offset: 32, format: 'float32x4' },
      { shaderLocation: 4, offset: 48, format: 'float32x4' },
    ];
    const meshPipe = meshProg.renderPipeline({ format: gpu.format, blend: 'alpha', sampleCount: 4, buffers: [{ arrayStride: FPV * 4, attributes: attrs }] });
    let vbuf = null;
    let vcap = 0;
    const mesh = new Mesh();
    const shapes = new ShapeBatch(gpu); // solid shapes, alpha blended
    const light = new ShapeBatch(gpu); // beams & sparks, additive
    const cam = new Camera2D();

    let msaa = null;
    let msaaView = null;
    const makeMsaa = (w, h) => {
      msaa?.destroy();
      msaa = gpu.texture({ size: [w, h], format: gpu.format, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'lines-msaa' });
      msaaView = msaa.createView();
    };
    makeMsaa(ctx.width, ctx.height);

    const readout = document.createElement('div');
    readout.className = 'tag';
    readout.style.cssText = 'right:8px;bottom:8px';
    ctx.overlay.append(readout);

    // ---- editable points (normalized 0..1 coordinates, so they survive resizes)
    const state = {
      joins: [[0.12, 0.72], [0.3, 0.42], [0.44, 0.8], [0.6, 0.47], [0.7, 0.86], [0.88, 0.5]],
      cubic: [[0.12, 0.78], [0.25, 0.18], [0.72, 0.22], [0.88, 0.75]],
      quad: [[0.14, 0.78], [0.5, 0.12], [0.86, 0.78]],
      route: [[0.08, 0.84], [0.26, 0.64], [0.2, 0.38], [0.48, 0.3], [0.6, 0.62], [0.84, 0.58], [0.74, 0.34]],
    };
    let drag = null;
    const toPx = (p) => [p[0] * ctx.width, p[1] * ctx.height];
    const handleDrag = (list, region = [0, 0, 1, 1]) => {
      // region: [x0, y0, w, h] in normalized canvas coords; list coords are relative to it
      const p = ctx.pointer;
      const lx = (p.nx - region[0]) / region[2];
      const ly = (p.ny - region[1]) / region[3];
      if (p.clicked) {
        let best = -1;
        let bd = 26 * Math.max(1, ctx.dpr);
        list.forEach((q, i) => {
          const d = Math.hypot((q[0] - lx) * region[2] * ctx.width, (q[1] - ly) * region[3] * ctx.height);
          if (d < bd) {
            bd = d;
            best = i;
          }
        });
        drag = best >= 0 ? { list, i: best } : null;
      }
      if (!p.down) drag = null;
      if (drag && drag.list === list) list[drag.i] = [Math.min(0.99, Math.max(0.01, lx)), Math.min(0.99, Math.max(0.01, ly))];
      return drag && drag.list === list ? drag.i : -1;
    };
    const handle = (x, y, active, r = 7) => {
      const s = Math.max(1, ctx.dpr);
      shapes.circle(x, y, (r + 3) * s, [0, 0, 0, 0.55]);
      shapes.circle(x, y, r * s, active ? '#fde047' : '#ffffff');
      shapes.circle(x, y, (r - 3) * s, active ? '#a16207' : '#334155');
    };

    // ---- lasers
    const mirrors = [
      { x: 0.3, y: 0.22, len: 0.22, a: 0.5, spin: 0.25 },
      { x: 0.52, y: 0.78, len: 0.24, a: -0.4, spin: -0.2 },
      { x: 0.76, y: 0.26, len: 0.24, a: 1.2, spin: 0.3 },
      { x: 0.66, y: 0.5, len: 0.2, a: 2.0, spin: -0.35 },
      { x: 0.88, y: 0.74, len: 0.2, a: 0.3, spin: 0.15 },
      { x: 0.36, y: 0.66, len: 0.18, a: 0.9, spin: 0.18 },
      { x: 0.92, y: 0.36, len: 0.18, a: 1.7, spin: -0.22 },
    ];
    const crystals = [
      { x: 0.66, y: 0.12, hit: -10 },
      { x: 0.2, y: 0.86, hit: -10 },
      { x: 0.94, y: 0.52, hit: -10 },
    ];
    const sparks = [];
    let sparkAcc = 0;

    // ---- rope (verlet)
    const ROPE_N = 18;
    const rope = { pts: [], prev: [], anchor: 1, len: 0, lastClick: -100, nextAuto: 2.5, shoot: 1 };
    const pegs = [0.12, 0.27, 0.42, 0.55].map((x) => [x, 0.17]);
    const initRope = (W, H) => {
      const a = [pegs[rope.anchor][0] * W, pegs[rope.anchor][1] * H];
      const hero = [a[0] + 0.16 * W, a[1] + 0.35 * H];
      rope.len = Math.hypot(hero[0] - a[0], hero[1] - a[1]);
      rope.pts = [];
      for (let i = 0; i < ROPE_N; i++) rope.pts.push(lerp2(a, hero, i / (ROPE_N - 1)));
      rope.prev = rope.pts.map((p) => [...p]);
    };
    let ropeSize = [0, 0];
    const attach = (pegIndex, W, H, t) => {
      const hero = rope.pts[ROPE_N - 1];
      const vel = [hero[0] - rope.prev[ROPE_N - 1][0], hero[1] - rope.prev[ROPE_N - 1][1]];
      rope.anchor = pegIndex;
      const a = [pegs[pegIndex][0] * W, pegs[pegIndex][1] * H];
      rope.len = Math.max(0.12 * H, Math.hypot(hero[0] - a[0], hero[1] - a[1]) * 0.85);
      for (let i = 0; i < ROPE_N; i++) {
        rope.pts[i] = lerp2(a, hero, i / (ROPE_N - 1));
        rope.prev[i] = [rope.pts[i][0] - vel[0] * (i / (ROPE_N - 1)), rope.pts[i][1] - vel[1] * (i / (ROPE_N - 1))];
      }
      rope.shoot = 0;
    };
    const stepRope = (dt, W, H) => {
      const g = 1.6 * H;
      const a = [pegs[rope.anchor][0] * W, pegs[rope.anchor][1] * H];
      const seg = rope.len / (ROPE_N - 1);
      for (let i = 1; i < ROPE_N; i++) {
        const p = rope.pts[i];
        const q = rope.prev[i];
        const vx = (p[0] - q[0]) * 0.999;
        const vy = (p[1] - q[1]) * 0.999;
        rope.prev[i] = [p[0], p[1]];
        const mass = i === ROPE_N - 1 ? 1 : 0.3;
        p[0] += vx;
        p[1] += vy + g * dt * dt * mass;
      }
      rope.pts[0] = a;
      for (let it = 0; it < 12; it++) {
        for (let i = 0; i < ROPE_N - 1; i++) {
          const p = rope.pts[i];
          const q = rope.pts[i + 1];
          const dx = q[0] - p[0];
          const dy = q[1] - p[1];
          const d = Math.hypot(dx, dy) || 1e-6;
          const diff = (d - seg) / d;
          const wp = i === 0 ? 0 : 0.5;
          const wq = i === 0 ? 1 : 0.5;
          p[0] += dx * diff * wp;
          p[1] += dy * diff * wp;
          q[0] -= dx * diff * wq;
          q[1] -= dy * diff * wq;
        }
      }
      // keep the hero inside the cave area
      const h = rope.pts[ROPE_N - 1];
      h[0] = Math.min(W * 0.56, Math.max(W * 0.02, h[0]));
      h[1] = Math.min(H * 0.86, h[1]);
    };

    const intersectSeg = (o, d, a, b) => {
      const ex = b[0] - a[0];
      const ey = b[1] - a[1];
      const den = d[0] * ey - d[1] * ex;
      if (Math.abs(den) < 1e-9) return null;
      const ax = a[0] - o[0];
      const ay = a[1] - o[1];
      const t = (ax * ey - ay * ex) / den;
      const s = (ax * d[1] - ay * d[0]) / den;
      if (t > 1e-3 && s >= 0 && s <= 1) return t;
      return null;
    };
    const intersectCircle = (o, d, c, r) => {
      const ox = o[0] - c[0];
      const oy = o[1] - c[1];
      const b = ox * d[0] + oy * d[1];
      const cc = ox * ox + oy * oy - r * r;
      const h = b * b - cc;
      if (h < 0) return null;
      const t = -b - Math.sqrt(h);
      return t > 1e-3 ? t : null;
    };

    const pack = (c) => {
      const n = parseInt(c.slice(1), 16);
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    };

    return {
      resize(w, h) {
        makeMsaa(w, h);
      },
      frame(ctx) {
        const P = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const s = Math.max(1, ctx.dpr);
        const ex = ctx.example;
        const t = ctx.time;
        const dt = Math.min(ctx.dt, 1 / 30);
        cam.setViewport(W, H);
        mesh.reset();
        shapes.begin();
        light.begin();
        const split = W * 0.58;
        let mode = 0;
        let info = '';

        if (ex === 'joins') {
          // legend row: three joins and three caps
          const lw = Math.min(P.width, 22) * s;
          const items = [];
          const top = H * 0.17;
          for (let k = 0; k < 6; k++) {
            const cx = W * ((k + 0.5) / 6);
            const sc = Math.min(W / 6, H * 0.22) * 0.32;
            let pts;
            let opts = { width: lw, color: [0.55, 0.62, 0.78, 1], wire: P.wire, join: 'miter', cap: 'butt', miterLimit: 10 };
            if (k < 3) {
              pts = [[cx - sc, top + sc * 0.55], [cx, top - sc * 0.55], [cx + sc, top + sc * 0.55]];
              opts.join = ['miter', 'round', 'bevel'][k];
            } else {
              pts = [[cx - sc, top], [cx + sc, top]];
              opts.cap = ['butt', 'square', 'round'][k - 3];
            }
            stroke(mesh, pts, opts);
            items.push({ text: k < 3 ? `${opts.join} join` : `${opts.cap} cap`, x: (k + 0.5) / 6, y: 0.3, valign: 'top', style: LBL });
            // show the true endpoints for caps
            if (k >= 3) for (const q of pts) shapes.circle(q[0], q[1], 2.5 * s, '#fde047');
          }
          labels(ctx, 'joins', items);
          // the editable polyline (lower area)
          const region = [0, 0.36, 1, 0.64];
          const active = handleDrag(state.joins, region);
          const pts = state.joins.map((q) => [q[0] * W, (region[1] + q[1] * region[3]) * H]);
          const col = [0.96, 0.45, 0.71, 1];
          stroke(mesh, pts, {
            width: P.width * s, join: P.join, cap: P.cap, miterLimit: P.miterLimit, closed: P.closed, color: col,
            dash: P.dash * s, roundDash: P.roundDash, wire: P.wire,
          });
          // the centerline the stroke was built from
          for (let i = 0; i < pts.length - 1 + (P.closed ? 1 : 0); i++) {
            const a = pts[i];
            const b = pts[(i + 1) % pts.length];
            shapes.line(a[0], a[1], b[0], b[1], 1.2 * s, [1, 1, 1, 0.55]);
          }
          pts.forEach((q, i) => handle(q[0], q[1], i === active));
          info = `${mesh.tris} triangles · ${mesh.n} vertices`;
        } else if (ex === 'bezier') {
          mode = 1;
          labels(ctx, 'bezier', []);
          const cubic = P.curve === 'cubic';
          const list = cubic ? state.cubic : state.quad;
          const active = handleDrag(list);
          const ctrl = list.map(toPx);
          const segs = Math.round(P.segments);
          const poly = flatten(ctrl, segs);
          // control polygon (dashed)
          stroke(mesh, ctrl, { width: 1.6 * s, join: 'bevel', color: [0.7, 0.75, 0.9, 0.7], dash: 7 * s, gap: 6 * s });
          stroke(mesh, poly, { width: P.width * s, join: 'round', cap: 'round', color: [0.98, 0.62, 0.2, 1], dash: P.dash * s, roundDash: P.roundDash, wire: P.wire });
          const tri = mesh.tris;
          // sample points
          for (const q of poly) shapes.circle(q[0], q[1], 3 * s, '#0b1020', { stroke: 0 }), shapes.circle(q[0], q[1], 2 * s, '#ffe4b5');
          // de Casteljau scaffolding
          if (P.construct) {
            const tt = P.t > 0 ? P.t : 0.5 + 0.48 * Math.sin(t * 0.7);
            const lv = casteljau(ctrl, tt);
            const cols = ['#94a3b8', '#22d3ee', '#a78bfa', '#4ade80'];
            for (let L = 1; L < lv.length; L++) {
              const pts = lv[L];
              for (let i = 0; i < pts.length - 1; i++) shapes.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 2 * s, cols[L]);
              for (const q of pts) shapes.circle(q[0], q[1], (L === lv.length - 1 ? 7 : 4) * s, L === lv.length - 1 ? '#ffffff' : cols[L], L === lv.length - 1 ? { glow: 10 * s, glowStrength: 0.5 } : {});
            }
            info = `t = ${tt.toFixed(2)} · ${segs} segments → ${tri} triangles`;
          } else info = `${segs} segments → ${tri} triangles`;
          ctrl.forEach((q, i) => handle(q[0], q[1], i === active));
          U.set('p0', ctrl[0]).set('p1', ctrl[1]).set('p2', ctrl[2]).set('p3', ctrl[cubic ? 3 : 2]).set('curveKind', cubic ? 1 : 0);
        } else if (ex === 'lasers') {
          mode = 2;
          labels(ctx, 'lasers', []);
          const col = pack(P.laserColor);
          const E = [W * 0.08, H * 0.5];
          let dir;
          if (ctx.pointer.over) {
            const dx = ctx.pointer.x - E[0];
            const dy = ctx.pointer.y - E[1];
            const L = Math.hypot(dx, dy) || 1;
            dir = [dx / L, dy / L];
          } else {
            const a = 0.45 * Math.sin(t * 0.4) - 0.1;
            dir = [Math.cos(a), Math.sin(a)];
          }
          const segsM = mirrors.map((m) => {
            const a = m.a + (P.spin ? m.spin * t : 0);
            const hl = (m.len * H) / 2;
            const c = [m.x * W, m.y * H];
            return [[c[0] - Math.cos(a) * hl, c[1] - Math.sin(a) * hl], [c[0] + Math.cos(a) * hl, c[1] + Math.sin(a) * hl], c];
          });
          const cr = H * 0.035;
          // raycast with reflections
          const path = [E];
          let o = E;
          let d = dir;
          let end = null;
          for (let b = 0; b <= P.bounces + 1; b++) {
            let best = Infinity;
            let hit = null;
            segsM.forEach((sg, i) => {
              const tt = intersectSeg(o, d, sg[0], sg[1]);
              if (tt !== null && tt < best) {
                best = tt;
                hit = { kind: 'mirror', i };
              }
            });
            crystals.forEach((c, i) => {
              const tt = intersectCircle(o, d, [c.x * W, c.y * H], cr);
              if (tt !== null && tt < best) {
                best = tt;
                hit = { kind: 'crystal', i };
              }
            });
            // walls
            const tw = Math.min(d[0] > 0 ? (W - o[0]) / d[0] : d[0] < 0 ? -o[0] / d[0] : Infinity, d[1] > 0 ? (H - o[1]) / d[1] : d[1] < 0 ? -o[1] / d[1] : Infinity);
            if (tw < best) {
              best = tw;
              hit = { kind: 'wall' };
            }
            const q = [o[0] + d[0] * best, o[1] + d[1] * best];
            path.push(q);
            if (hit.kind === 'mirror' && b < P.bounces) {
              const sg = segsM[hit.i];
              const ex2 = sg[1][0] - sg[0][0];
              const ey2 = sg[1][1] - sg[0][1];
              const L = Math.hypot(ex2, ey2);
              const n = [-ey2 / L, ex2 / L];
              const dn = d[0] * n[0] + d[1] * n[1];
              d = [d[0] - 2 * dn * n[0], d[1] - 2 * dn * n[1]];
              o = q;
              continue;
            }
            if (hit.kind === 'crystal') crystals[hit.i].hit = t;
            end = { p: q, n: hit.kind === 'mirror' ? null : [-d[0], -d[1]], kind: hit.kind };
            break;
          }
          // sparks at the end point
          if (end && !ctx.paused) {
            sparkAcc += dt * (ctx.testMode ? 60 : 160);
            while (sparkAcc > 1) {
              sparkAcc -= 1;
              const a = Math.random() * Math.PI * 2;
              const v = (60 + Math.random() * 260) * s;
              let vx = Math.cos(a) * v;
              let vy = Math.sin(a) * v;
              if (end.n && vx * end.n[0] + vy * end.n[1] < 0) {
                vx = -vx;
                vy = -vy;
              }
              sparks.push({ x: end.p[0], y: end.p[1], vx, vy, life: 0.25 + Math.random() * 0.35, age: 0 });
            }
          }
          for (let i = sparks.length - 1; i >= 0; i--) {
            const sp = sparks[i];
            if (!ctx.paused) {
              sp.age += dt;
              sp.x += sp.vx * dt;
              sp.y += sp.vy * dt;
              sp.vy += 400 * s * dt;
            }
            if (sp.age > sp.life || sparks.length > 400) sparks.splice(i, 1);
          }
          // solid objects (alpha pass): mirrors, emitter, crystals
          for (const [a, b, c] of segsM) {
            shapes.line(a[0], a[1], b[0], b[1], 11 * s, '#0b0f17');
            shapes.line(a[0], a[1], b[0], b[1], 7 * s, '#c7d2e3');
            shapes.line(a[0], a[1], b[0], b[1], 2 * s, '#ffffff');
            shapes.circle(c[0], c[1], 5 * s, '#475569');
          }
          for (const c of crystals) {
            const lit = Math.max(0, 1 - (t - c.hit) * 2.5);
            const cx = c.x * W;
            const cy = c.y * H;
            const rot = Math.PI / 4 + 0.3 * Math.sin(t * 0.8 + c.x * 9);
            shapes.box(cx, cy, cr * 0.78, cr * 0.78, [0.03, 0.06, 0.08, 1], { rotation: rot, radius: 3 * s });
            shapes.box(cx, cy, cr * 0.64, cr * 0.64, [0.12 + 0.6 * lit, 0.62 + 0.3 * lit, 0.55 + 0.4 * lit, 1], { rotation: rot, radius: 2 * s });
            shapes.box(cx - cr * 0.12, cy - cr * 0.12, cr * 0.22, cr * 0.22, [0.75, 1, 0.95, 0.85], { rotation: rot, radius: 1 * s });
            light.circle(cx, cy, cr * 0.3, [0.2, 0.9, 0.75, 0.35 + 0.1 * Math.sin(t * 3 + c.y * 7)], { glow: 18 * s, glowStrength: 0.35 });
          }
          shapes.box(E[0] - 30 * s, E[1], 14 * s, 34 * s, '#0b0f17', { radius: 6 * s });
          shapes.box(E[0] - 30 * s, E[1], 11 * s, 31 * s, '#334155', { radius: 5 * s });
          shapes.circle(E[0], E[1], 25 * s, '#0b0f17');
          shapes.circle(E[0], E[1], 21 * s, '#475569');
          shapes.circle(E[0], E[1], 21 * s, '#64748b', { stroke: 3 * s });
          shapes.line(E[0], E[1], E[0] + dir[0] * 36 * s, E[1] + dir[1] * 36 * s, 15 * s, '#0b0f17');
          shapes.line(E[0], E[1], E[0] + dir[0] * 34 * s, E[1] + dir[1] * 34 * s, 10 * s, '#94a3b8');
          shapes.line(E[0] + dir[0] * 12 * s, E[1] + dir[1] * 12 * s, E[0] + dir[0] * 34 * s, E[1] + dir[1] * 34 * s, 3 * s, '#e2e8f0');
          shapes.circle(E[0], E[1], 10 * s, '#1e293b');
          shapes.circle(E[0], E[1], 4 * s, P.laserColor);
          // light: additive SDF capsules with glow
          const flick = 1 + 0.08 * Math.sin(t * 50) + 0.05 * Math.sin(t * 31);
          const bw = P.beam * s * flick;
          for (let i = 0; i < path.length - 1; i++) {
            const a = path[i];
            const b = path[i + 1];
            const fade = Math.pow(0.9, i);
            light.line(a[0], a[1], b[0], b[1], bw * 3, [col[0], col[1], col[2], 0.35 * fade], { glow: 16 * s * P.glow, glowStrength: 0.55 * P.glow });
            light.line(a[0], a[1], b[0], b[1], bw, [col[0] * 0.5 + 0.5, col[1] * 0.5 + 0.5, col[2] * 0.5 + 0.5, fade]);
            light.line(a[0], a[1], b[0], b[1], bw * 0.35, [1, 1, 1, fade]);
          }
          for (let i = 1; i < path.length; i++) {
            const q = path[i];
            light.circle(q[0], q[1], bw * 0.9, [1, 1, 1, 0.9], { glow: 20 * s * P.glow, glowStrength: 0.8 * Math.min(1, P.glow) });
          }
          light.circle(E[0] + dir[0] * 36 * s, E[1] + dir[1] * 36 * s, bw * 1.2, [col[0], col[1], col[2], 1], { glow: 18 * s, glowStrength: 0.9 });
          for (const sp of sparks) {
            const k = 1 - sp.age / sp.life;
            light.line(sp.x, sp.y, sp.x - sp.vx * 0.02, sp.y - sp.vy * 0.02, 2.2 * s * k, [1, 0.6 + 0.4 * k, 0.4 + 0.5 * k, k]);
          }
          for (const c of crystals) {
            const lit = Math.max(0, 1 - (t - c.hit) * 2.5);
            if (lit > 0) light.circle(c.x * W, c.y * H, cr * 0.5, [0.4, 1, 0.8, lit], { glow: 40 * s, glowStrength: lit });
          }
          info = `${path.length - 1} beam segment${path.length > 2 ? 's' : ''} · ${path.length - 2} bounce${path.length === 3 ? '' : 's'}`;
        } else {
          mode = 3;
          labels(ctx, 'rope', [
            { text: 'verlet rope · one tessellated stroke', x: 0.29, y: 0.2, valign: 'top', style: LBL },
            { text: 'Catmull–Rom route · marching dashes', x: 0.79, y: 0.965, valign: 'bottom', style: LBL },
          ]);
          // ---- rope
          if (!rope.pts.length || ropeSize[0] !== W || ropeSize[1] !== H) {
            initRope(W, H);
            ropeSize = [W, H];
          }
          const p = ctx.pointer;
          if (p.clicked && p.x < split) {
            let bi = 0;
            let bd = Infinity;
            pegs.forEach((q, i) => {
              const d = Math.hypot(q[0] * W - p.x, q[1] * H - p.y);
              if (d < bd) {
                bd = d;
                bi = i;
              }
            });
            attach(bi, W, H, t);
            rope.lastClick = t;
          }
          if (!ctx.paused) {
            if (t > rope.nextAuto && t - rope.lastClick > 6) {
              const hero = rope.pts[ROPE_N - 1];
              const vx = hero[0] - rope.prev[ROPE_N - 1][0];
              let next = rope.anchor + (vx >= 0 ? 1 : -1);
              if (next < 0 || next >= pegs.length) next = rope.anchor - (vx >= 0 ? 1 : -1);
              attach(Math.max(0, Math.min(pegs.length - 1, next)), W, H, t);
              rope.nextAuto = t + 1.6 + Math.random();
            }
            const sub = 4;
            for (let k = 0; k < sub; k++) stepRope(dt / sub, W, H);
            rope.shoot = Math.min(1, rope.shoot + dt * 8);
          }
          // pegs
          pegs.forEach((q, i) => {
            shapes.circle(q[0] * W, q[1] * H, 9 * s, '#0b0f17');
            shapes.circle(q[0] * W, q[1] * H, 6 * s, i === rope.anchor ? '#fbbf24' : '#94a3b8');
          });
          // rope stroke (shooting animation: only the first part is visible at first)
          let rp = rope.pts.map((q) => [q[0], q[1]]);
          if (rope.shoot < 1) {
            const hero = rp[ROPE_N - 1];
            const a = rp[0];
            rp = [hero, lerp2(hero, a, rope.shoot)];
          }
          stroke(mesh, rp, { width: P.width * s, join: 'round', cap: 'round', color: [0.78, 0.6, 0.38, 1], rope: true, wire: P.wire });
          // hero
          const h = rope.pts[ROPE_N - 1];
          const prev = rope.prev[ROPE_N - 1];
          const lean = Math.max(-0.6, Math.min(0.6, (h[0] - prev[0]) * 0.08));
          const hr = 18 * s;
          shapes.circle(h[0], h[1] + hr * 0.4, hr * 1.08, '#0b0f17');
          shapes.circle(h[0], h[1] + hr * 0.4, hr, '#ef4444');
          shapes.circle(h[0] + hr * (0.25 + lean * 0.3), h[1] + hr * 0.2, hr * 0.32, '#ffffff');
          shapes.circle(h[0] + hr * (0.35 + lean * 0.4), h[1] + hr * 0.22, hr * 0.15, '#0b0f17');
          shapes.circle(h[0], h[1], 4 * s, '#78350f');
          // ---- route on the map
          const mx0 = split / W;
          const region = [mx0 + 0.02, 0.05, 1 - mx0 - 0.04, 0.82];
          const active = handleDrag(state.route, region);
          const wps = state.route.map((q) => [(region[0] + q[0] * region[2]) * W, (region[1] + q[1] * region[3]) * H]);
          const curve = catmullRom(wps, 14);
          stroke(mesh, curve, { width: (P.width + 7) * s, join: 'round', cap: 'round', color: [0.96, 0.9, 0.75, 0.55] });
          stroke(mesh, curve, { width: P.width * s, join: 'round', cap: 'round', color: [0.55, 0.14, 0.1, 1], dash: 3.2 * P.width * s, gap: 1.6 * P.width * s, roundDash: true, offset: -t * 40 * s, wire: P.wire });
          // traveller moving along the route by arc length
          let total = 0;
          const acc = [0];
          for (let i = 1; i < curve.length; i++) acc.push((total += Math.hypot(curve[i][0] - curve[i - 1][0], curve[i][1] - curve[i - 1][1])));
          const target = ((t * 60 * s) % total + total) % total;
          let k = 1;
          while (k < acc.length - 1 && acc[k] < target) k++;
          const f = (target - acc[k - 1]) / Math.max(1e-6, acc[k] - acc[k - 1]);
          const tp = lerp2(curve[k - 1], curve[k], f);
          // start & X marks the spot
          const st = wps[0];
          const en = wps[wps.length - 1];
          shapes.circle(st[0], st[1], 11 * s, '#3f2a14');
          shapes.circle(st[0], st[1], 7 * s, '#f5e6c8');
          const xs = 13 * s;
          shapes.line(en[0] - xs, en[1] - xs, en[0] + xs, en[1] + xs, 7 * s, '#b91c1c');
          shapes.line(en[0] + xs, en[1] - xs, en[0] - xs, en[1] + xs, 7 * s, '#b91c1c');
          wps.forEach((q, i) => {
            if (i > 0 && i < wps.length - 1) {
              shapes.circle(q[0], q[1], (i === active ? 8 : 6) * s, i === active ? '#fde047' : '#3f2a14', { stroke: 2.5 * s });
            }
          });
          shapes.circle(tp[0], tp[1], 9 * s, '#1d4ed8', { glow: 8 * s, glowStrength: 0.35 });
          shapes.circle(tp[0], tp[1], 4 * s, '#ffffff');
          info = '';
        }

        // ---- upload & draw: MSAA pass (background + tessellated meshes), resolved to the canvas
        U.set('resolution', [W, H]).set('time', t).set('mode', mode).set('split', split);
        U.upload();
        const bytes = mesh.n * FPV * 4;
        if (bytes > vcap) {
          vbuf?.destroy();
          vcap = Math.max(bytes, 1 << 16) * 2;
          vbuf = gpu.buffer({ size: vcap, usage: GPUBufferUsage.VERTEX, label: 'line-mesh' });
        }
        if (mesh.n) device.queue.writeBuffer(vbuf, 0, mesh.data, 0, mesh.n * FPV);
        const enc = ctx.encoder;
        const pass = enc.beginRenderPass({
          label: 'lines-msaa',
          colorAttachments: [{ view: msaaView, resolveTarget: ctx.target, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'discard' }],
        });
        pass.setPipeline(bgPipe);
        pass.setBindGroup(0, bgProg.bind({ u: U }));
        pass.draw(3);
        if (mesh.n) {
          pass.setPipeline(meshPipe);
          pass.setBindGroup(0, meshProg.bind({ u: U }));
          pass.setVertexBuffer(0, vbuf);
          pass.draw(mesh.n);
        }
        pass.end();
        // SDF shapes on top (anti-aliased analytically, no MSAA needed)
        const canvas = { view: ctx.target, format: gpu.format };
        shapes.flush(enc, canvas, cam, { blend: 'alpha' });
        light.flush(enc, canvas, cam, { blend: 'additive' });
        readout.textContent = info;
        readout.style.display = info ? '' : 'none';
      },
    };
  },
};

import { shaderScene } from '../../core/shaderscene.js';
import { setLabels, quadLabels } from './_shared.js';

// Patterns & tiling: repetition with fract (grid, brick, mirror, polar), Truchet tiles (arcs,
// 10 PRINT maze, flowing pipes — click to flip a tile), a hex-grid strategy board with axial
// coordinates, range and a hex line, and a kaleidoscope. One portable-WGSL shader.

const MAX_FLIPS = 32;
const flips = []; // clicked truchet tiles: [tx, ty]
const flipData = new Float32Array(MAX_FLIPS * 4);
let lastTiles = 0;

// --- hex math (shared logic with the shader): pointy-top axial coordinates, origin at canvas center
function pixelToHex(x, y, s) {
  const fq = ((Math.sqrt(3) / 3) * x - y / 3) / s;
  const fr = ((2 / 3) * y) / s;
  const fs = -fq - fr;
  let q = Math.round(fq);
  let r = Math.round(fr);
  const s3 = Math.round(fs);
  const dq = Math.abs(q - fq);
  const dr = Math.abs(r - fr);
  const ds = Math.abs(s3 - fs);
  if (dq > dr && dq > ds) q = -r - s3;
  else if (dr > ds) r = -q - s3;
  return [q, r];
}
const hexSizePx = (params, ctx) => ctx.height / (params.hexRows * 1.5);

function hexInfo(params, ctx) {
  const p = ctx.pointer;
  if (!p.over) return [];
  let x = p.x - ctx.width / 2;
  let y = p.y - ctx.height / 2;
  if (params.orient === 'flat') [x, y] = [y, x];
  const [q, r] = pixelToHex(x, y, hexSizePx(params, ctx));
  const s = -q - r;
  const dist = (Math.abs(q) + Math.abs(r) + Math.abs(s)) / 2;
  const fmt = (v) => (v === 0 ? '0' : v > 0 ? `+${v}` : `−${-v}`);
  const left = Math.min(78, (p.x / ctx.width) * 100);
  const top = Math.min(88, (p.y / ctx.height) * 100);
  return [
    {
      text: `hex <b>q ${fmt(q)}</b>, <b>r ${fmt(r)}</b> <span style="opacity:.7">(s ${fmt(s)})</span> · ${dist} from the unit`,
      style: `left:${left.toFixed(2)}%;top:${top.toFixed(2)}%;transform:translate(16px, 16px);font-size:11px`,
    },
  ];
}

const CODE = /* wgsl */ `
// ================================================================== motif for the repetition demo
// an asymmetric motif in a cell (q in -0.5..0.5) so flips and rotations are visible
fn motif(q: vec2f, id: vec2f, t: f32) -> vec3f {
  let gb = mix(vec3f(0.14, 0.15, 0.25), vec3f(0.22, 0.18, 0.32), q.y + 0.5);
  var c = gb;
  if (u.showCells > 0.5) { c = mix(c, vec3f(q.x + 0.5, q.y + 0.5, 0.25), 0.55); }
  // leaf: two circles intersected, rotated 45 degrees
  let r = rot2(0.785 + t) * (q - vec2f(0.04, 0.04));
  let leaf = max(length(r - vec2f(0.0, 0.16)), length(r + vec2f(0.0, 0.16))) - 0.28;
  let vein = abs(r.y) - 0.008;
  let lc = mix(vec3f(0.25, 0.85, 0.6), vec3f(0.15, 0.6, 0.75), q.x + 0.5);
  c = mix(c, lc, sdfFill(leaf));
  c = mix(c, vec3f(0.08, 0.3, 0.3), sdfFill(max(vein, leaf + 0.02)));
  // a dot in one corner + a little triangle pointing right: breaks the symmetry
  c = mix(c, vec3f(1.0, 0.75, 0.3), sdfFill(length(q - vec2f(-0.3, -0.3)) - 0.075));
  let tri = sdTriangle(q, vec2f(0.26, 0.18), vec2f(0.26, 0.38), vec2f(0.42, 0.28));
  c = mix(c, vec3f(1.0, 0.45, 0.55), sdfFill(tri));
  // cell border
  let e = 0.5 - max(abs(q.x), abs(q.y));
  c = mix(c, vec3f(0.0), (1.0 - smoothstep(0.0, fwidth(e) * 1.5, e)) * 0.8);
  return c;
}

fn exRepeat(px: vec2f) -> vec3f {
  let hs = u.resolution * 0.5;
  let qd = vec2f(step(hs.x, px.x), step(hs.y, px.y));
  let k = i32(qd.x + 2.0 * qd.y);
  let lp = (px - qd * hs) / hs.y;
  let t = u.time * u.spin;
  var col = vec3f(0.0);
  if (k == 0) {
    // plain grid: the cell id and the position inside the cell
    let p = lp * u.cells;
    col = motif(fract(p) - 0.5, floor(p), 0.0);
  } else if (k == 1) {
    // brick: shift every other row by half a cell (rows slide when animated)
    var p = lp * u.cells;
    let row = floor(p.y);
    p.x += 0.5 * fmod(row, 2.0) + (fmod(row, 2.0) - 0.5) * t * 0.2;
    col = motif(fract(p) - 0.5, floor(p), 0.0);
  } else if (k == 2) {
    // mirror: flip every other cell so neighbours match seamlessly
    let p = lp * u.cells;
    let id = floor(p);
    var q = fract(p) - 0.5;
    if (fmod(id.x, 2.0) > 0.5) { q.x = -q.x; }
    if (fmod(id.y, 2.0) > 0.5) { q.y = -q.y; }
    col = motif(q, id, 0.0);
  } else {
    // polar: repeat around the center (sectors) and outward (rings)
    let c = (lp - vec2f(hs.x / hs.y * 0.5, 0.5));
    let r = length(c) * u.cells * 1.2;
    let sector = TAU / u.sectors;
    let a = atan2(c.y, c.x) + t * 0.15;
    let ai = floor(a / sector + 0.5);
    let al = a - ai * sector;
    let ring = floor(r);
    let q = vec2f(fract(r) - 0.5, al / sector);
    col = motif(q, vec2f(ai, ring), 0.0);
    col *= smoothstep(0.0, 0.6, r);
  }
  let edge = min(abs(px.x - hs.x), abs(px.y - hs.y));
  return mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, edge));
}

// ================================================================== Truchet
fn isFlipped(id: vec2f) -> f32 {
  var f = step(0.5, hash21(id + vec2f(u.seed * 13.0, u.seed * 7.0)));
  for (var i = 0; i < ${MAX_FLIPS}; i++) {
    if (f32(i) >= u.nflip) { break; }
    let e = u.flips[i];
    if (abs(e.x - id.x) + abs(e.y - id.y) < 0.5) { f = 1.0 - f; }
  }
  return f;
}

fn exTruchet(px: vec2f) -> vec3f {
  let ts = u.resolution.y / u.tiles;
  let p = px / ts;
  let id = floor(p);
  var q = fract(p);
  let fl = isFlipped(id);
  if (fl > 0.5) { q.x = 1.0 - q.x; }
  let t = u.time * u.spin;
  let style = i32(u.style);
  var col = vec3f(0.0);
  let parity = fmod(id.x + id.y, 2.0);
  if (style == 1) {
    // 10 PRINT: one diagonal per tile
    let d = abs(q.x - q.y) * 0.70710678 * ts;
    col = vec3f(0.16, 0.2, 0.6);
    col = mix(col, vec3f(0.62, 0.7, 1.0), 1.0 - smoothstep(ts * 0.08, ts * 0.08 + 1.2, d));
  } else {
    // two quarter-circle arcs around opposite corners
    let d0 = length(q);
    let d1 = length(q - vec2f(1.0));
    let da = abs(d0 - 0.5);
    let db = abs(d1 - 0.5);
    let d = min(da, db);
    if (style == 0) {
      // the arcs split the plane into two regions: color them by parity -> blobs & mazes
      let side = step(0.5, d0) * step(0.5, d1);
      let region = fmod(side + parity + fl, 2.0);
      col = mix(vec3f(0.96, 0.62, 0.32), vec3f(0.18, 0.1, 0.28), region);
      col = mix(col, vec3f(0.98, 0.95, 0.88), 1.0 - smoothstep(0.03, 0.03 + 1.5 / ts, d));
    } else {
      // pipes: shaded tubes with liquid flowing along them
      let w = 0.17;
      let x = clamp(d / w, 0.0, 1.0);
      let nz = sqrt(1.0 - x * x);
      col = vec3f(0.05, 0.06, 0.09) + vec3f(0.03) * step(0.5, fmod(floor(p.x * 4.0) + floor(p.y * 4.0), 2.0));
      // arc angle -> distance along the pipe; direction alternates on a checkerboard so flow stays continuous
      var ang = atan2(q.y, q.x);
      if (db < da) { ang = atan2(1.0 - q.y, 1.0 - q.x); }
      var s = ang / (PI * 0.5);
      if (db < da) { s = 1.0 - s; }
      if (fl > 0.5) { s = 1.0 - s; }
      if (parity > 0.5) { s = 1.0 - s; }
      let flow = smoothstep(0.55, 0.9, sin((s * 2.0 - t * 1.5) * TAU) * 0.5 + 0.5);
      let metal = mix(vec3f(0.2, 0.32, 0.35), vec3f(0.7, 0.85, 0.85), nz * nz);
      var tube = metal + vec3f(0.3, 1.0, 0.75) * flow * nz * 0.9;
      tube += vec3f(1.0) * pow(max(1.0 - abs(x - 0.35) * 4.0, 0.0), 3.0) * 0.35;
      let cov = 1.0 - smoothstep(w - 1.2 / ts, w, d);
      // shadow under the tube
      col *= 1.0 - 0.6 * (1.0 - smoothstep(w, w + 0.1, d));
      col = mix(col, tube, cov);
    }
  }
  if (u.showCells > 0.5) {
    let e = min(min(q.x, 1.0 - q.x), min(q.y, 1.0 - q.y)) * ts;
    col = mix(col, vec3f(1.0, 0.85, 0.3), (1.0 - smoothstep(0.5, 1.5, e)) * 0.7);
    col = mix(col, col * vec3f(1.2, 0.8, 0.8), fl * 0.25);
  }
  // hovered tile
  if (u.mouse.w > 0.5) {
    let hm = floor(u.mouse.xy / ts);
    if (abs(hm.x - id.x) + abs(hm.y - id.y) < 0.5) {
      let e = min(min(q.x, 1.0 - q.x), min(q.y, 1.0 - q.y)) * ts;
      col = mix(col, vec3f(1.0), (1.0 - smoothstep(1.0, 2.5, e)) * 0.9);
      col += vec3f(0.06);
    }
  }
  return col;
}

// ================================================================== hex board
fn hexRound(fq: f32, fr: f32) -> vec2f {
  let fs = -fq - fr;
  var q = floor(fq + 0.5);
  var r = floor(fr + 0.5);
  let s = floor(fs + 0.5);
  let dq = abs(q - fq);
  let dr = abs(r - fr);
  let ds = abs(s - fs);
  if (dq > dr && dq > ds) { q = -r - s; } else if (dr > ds) { r = -q - s; }
  return vec2f(q, r);
}
fn toHex(p: vec2f, s: f32) -> vec2f { return hexRound((0.57735027 * p.x - 0.33333333 * p.y) / s, (0.66666667 * p.y) / s); }
fn hexCenter(h: vec2f, s: f32) -> vec2f { return vec2f(s * (1.7320508 * h.x + 0.8660254 * h.y), s * 1.5 * h.y); }
fn hexDist(a: vec2f, b: vec2f) -> f32 {
  let d = a - b;
  return (abs(d.x) + abs(d.y) + abs(d.x + d.y)) * 0.5;
}
// distance from rel (relative to the hex center) to the hex's edge, positive inside
fn hexEdge(rel: vec2f, s: f32) -> f32 {
  let m = max(max(abs(rel.x), abs(dot(rel, vec2f(0.5, 0.8660254)))), abs(dot(rel, vec2f(-0.5, 0.8660254))));
  return s * 0.8660254 - m;
}

// terrain id at a hex: 0 water, 1 grass, 2 forest, 3 hills, 4 mountain, 5 desert
fn hexTerrain(h: vec2f) -> f32 {
  let p = h * 0.16 + vec2f(u.seed * 3.7, u.seed * 1.3);
  let e = fbm(p, 4) + 0.08;
  let m = fbm(p * 1.3 + vec2f(20.0, 3.0), 3);
  if (e < -0.06) { return 0.0; }
  if (e > 0.3) { return 4.0; }
  if (e > 0.18) { return 3.0; }
  if (m < -0.12) { return 5.0; }
  if (m > 0.05) { return 2.0; }
  return 1.0;
}

fn terrainColor(k: f32) -> vec3f {
  if (k < 0.5) { return vec3f(0.18, 0.42, 0.62); }
  if (k < 1.5) { return vec3f(0.48, 0.7, 0.33); }
  if (k < 2.5) { return vec3f(0.28, 0.52, 0.28); }
  if (k < 3.5) { return vec3f(0.62, 0.62, 0.38); }
  if (k < 4.5) { return vec3f(0.55, 0.52, 0.5); }
  return vec3f(0.88, 0.78, 0.5);
}

fn terrainIcon(c0: vec3f, k: f32, l: vec2f, s: f32) -> vec3f {
  var c = c0;
  let aa = 1.2 / s;
  if (k > 1.5 && k < 2.5) {
    for (var i = 0; i < 3; i++) {
      let o = vec2f(f32(i) * 0.28 - 0.28, 0.1 - 0.12 * f32(i % 2));
      let tr = sdTriangle(l, o + vec2f(0.0, -0.25), o + vec2f(-0.13, 0.08), o + vec2f(0.13, 0.08));
      c = mix(c, vec3f(0.12, 0.33, 0.15), 1.0 - smoothstep(0.0, aa, tr));
    }
  } else if (k > 3.5 && k < 4.5) {
    let tr = sdTriangle(l, vec2f(0.0, -0.42), vec2f(-0.38, 0.25), vec2f(0.38, 0.25));
    c = mix(c, vec3f(0.42, 0.38, 0.38), 1.0 - smoothstep(0.0, aa, tr));
    let cap = sdTriangle(l, vec2f(0.0, -0.42), vec2f(-0.14, -0.17), vec2f(0.14, -0.17));
    c = mix(c, vec3f(0.97), 1.0 - smoothstep(0.0, aa, cap));
  } else if (k > 2.5 && k < 3.5) {
    let a1 = abs(length(l - vec2f(-0.15, 0.2)) - 0.2) - 0.03;
    let a2 = abs(length(l - vec2f(0.2, 0.12)) - 0.17) - 0.03;
    let upper = step(l.y, 0.12);
    c = mix(c, vec3f(0.42, 0.42, 0.25), (1.0 - smoothstep(0.0, aa, min(a1, a2))) * upper);
  } else if (k < 0.5) {
    let wv = abs(l.y - 0.06 * sin(l.x * 14.0 + u.time * 2.0)) - 0.02;
    c = mix(c, vec3f(0.55, 0.75, 0.9), (1.0 - smoothstep(0.0, aa, wv)) * step(abs(l.x), 0.35) * 0.8);
  }
  return c;
}

fn exHex(px: vec2f) -> vec3f {
  let s = u.resolution.y / (u.hexRows * 1.5);
  var p = px - u.resolution * 0.5;
  var m = u.mouse.xy - u.resolution * 0.5;
  if (u.orient > 0.5) { p = p.yx; m = m.yx; }
  let h = toHex(p, s);
  let hc = hexCenter(h, s);
  let rel = p - hc;
  let edge = hexEdge(rel, s);
  let k = hexTerrain(h);
  var col = terrainColor(k);
  col *= 0.9 + 0.1 * hash21(h);
  // bevel: lighter toward the top-left, darker near the edge
  col *= 0.9 + 0.15 * clamp(-dot(rel / s, vec2f(0.6, 0.8)), -1.0, 1.0);
  col = terrainIcon(col, k, rel / s, s);
  let hm = toHex(m, s);
  let unit = vec2f(0.0, 0.0);
  let dh = hexDist(h, hm);
  if (u.showCoords > 0.5) {
    // debug view: color by axial coordinates; highlight the three axes (q = 0, r = 0, s = 0)
    let cc = 0.5 + 0.5 * vec3f(sin(h.x * 0.55), sin(h.y * 0.55 + 2.0), sin((-h.x - h.y) * 0.55 + 4.0));
    col = mix(col, cc, 0.75);
    let onAxis = max(max(1.0 - step(0.5, abs(h.x)), 1.0 - step(0.5, abs(h.y))), 1.0 - step(0.5, abs(h.x + h.y)));
    col = mix(col, vec3f(1.0), onAxis * 0.35);
  }
  if (u.mouse.w > 0.5) {
    // movement range: every hex within N steps of the hovered hex
    let inRange = 1.0 - step(u.range + 0.5, dh);
    col = mix(col, col * 0.7 + vec3f(0.25, 0.3, 0.45), inRange * (0.35 + 0.1 * sin(u.time * 4.0)));
    // outline the range: edges whose neighbour is out of range
    let ang = atan2(rel.y, rel.x);
    let k0 = floor(ang / (PI / 3.0));
    var bd = 1e9;
    for (var e = 0; e < 2; e++) {
      let a = (k0 + f32(e)) * PI / 3.0;
      let nd = vec2f(cos(a), sin(a));
      let nb = toHex(hc + nd * s * 1.7320508, s);
      if (hexDist(nb, hm) > u.range + 0.5) { bd = min(bd, s * 0.8660254 - dot(rel, nd)); }
    }
    col = mix(col, vec3f(0.55, 0.75, 1.0), inRange * (1.0 - smoothstep(1.5, 3.0, bd)));
    // hex line from the unit to the hovered hex: round(lerp(a, b, i / n)) for i = 0..n
    let n = hexDist(unit, hm);
    var onLine = 0.0;
    for (var i = 0; i <= 40; i++) {
      if (f32(i) > n) { break; }
      let tt = f32(i) / max(n, 1.0);
      let lh = hexRound(mix(unit.x, hm.x, tt) + 0.000001, mix(unit.y, hm.y, tt) + 0.000002);
      if (abs(lh.x - h.x) + abs(lh.y - h.y) < 0.5) { onLine = 1.0; }
    }
    col = mix(col, vec3f(1.0, 0.85, 0.35), onLine * 0.45 * smoothstep(0.0, s * 0.2, edge));
    // the hovered hex
    if (dh < 0.5) { col = mix(col, vec3f(1.0), 0.25 + 0.65 * (1.0 - smoothstep(1.5, 3.5, edge))); }
  }
  // hex outlines (gap between tiles)
  col = mix(col, vec3f(0.06, 0.07, 0.1), 1.0 - smoothstep(0.6, 1.8, edge));
  // the unit token at hex (0, 0)
  if (abs(h.x) + abs(h.y) < 0.5) {
    let d = length(rel) - s * 0.42;
    col = mix(col, vec3f(0.0), (1.0 - smoothstep(0.0, s * 0.18, d)) * 0.4);
    col = mix(col, vec3f(0.85, 0.2, 0.25), 1.0 - smoothstep(-1.0, 0.5, d));
    let star = sdStar5(rel / s, 0.26, 0.45) * s;
    col = mix(col, vec3f(1.0, 0.92, 0.7), 1.0 - smoothstep(-0.5, 0.8, star));
  }
  return col;
}

// ================================================================== kaleidoscope
// the "object" inside the kaleidoscope: drifting pieces of colored glass and sticks on a dark field
fn beads(p: vec2f, t: f32) -> vec3f {
  let n = fbm(p * 1.2 + vec2f(t * 0.05, -t * 0.03), 3);
  var c = vec3f(0.02, 0.015, 0.05) + palette(n * 0.6 + t * 0.02, vec3f(0.5, 0.45, 0.55), vec3f(0.4, 0.35, 0.35), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.33, 0.67)) * 0.3;
  for (var i = 0; i < 22; i++) {
    let fi = f32(i);
    let h = hash13(fi + 7.0);
    let pos = vec2f(sin(t * (0.12 + h.x * 0.25) + fi * 1.7), cos(t * (0.1 + h.y * 0.2) + fi * 2.3)) * (0.35 + 0.6 * h.y);
    let rad = 0.12 + 0.2 * h.z;
    let q = rot2(t * (h.y - 0.5) * 1.5 + fi) * (p - pos);
    var d = length(q) - rad;
    if (i % 3 == 1) { d = sdEquilateralTriangle(q, rad * 0.9); }
    if (i % 3 == 2) { d = sdRhombus(q, vec2f(rad * 1.2, rad * 0.6)); }
    let bc = palette(h.x + h.z * 0.4, vec3f(0.6, 0.5, 0.5), vec3f(0.45, 0.45, 0.45), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.2, 0.5));
    let inner = clamp(-d / rad, 0.0, 1.0);
    let fill = 1.0 - smoothstep(-0.004, 0.004, d);
    // translucent glass: overlapping pieces mix their colors; bright rim
    c = mix(c, bc * (0.55 + 0.7 * inner), fill * 0.85);
    c += bc * (1.0 - smoothstep(0.0, 0.012, abs(d))) * 0.6;
  }
  for (var i = 0; i < 4; i++) {
    let fi = f32(i);
    let a = t * (0.2 + 0.1 * fi) + fi * 1.9;
    let ctr = vec2f(cos(fi * 2.1 + t * 0.07), sin(fi * 1.3 - t * 0.05)) * 0.7;
    let dir = vec2f(cos(a), sin(a)) * 0.45;
    let d = sdSegment(p, ctr - dir, ctr + dir) - 0.018;
    c = mix(c, palette(fi * 0.27 + 0.1, vec3f(0.7), vec3f(0.3), vec3f(1.0), vec3f(0.0, 0.33, 0.67)), 1.0 - smoothstep(-0.004, 0.004, d));
  }
  return c;
}

fn kfold(c: vec2f) -> vec2f {
  // fold the angle into one wedge, mirroring every other wedge (that's what two mirrors do)
  let seg = TAU / u.segments;
  var a = atan2(c.y, c.x) + u.time * u.spin * 0.1;
  a = fmod(a, seg);
  a = min(a, seg - a);
  return vec2f(cos(a), sin(a)) * length(c);
}

fn exKaleido(px: vec2f) -> vec3f {
  let H = u.resolution.y;
  let t = u.time * u.spin;
  let c = (px - u.resolution * 0.5) / H * u.kzoom;
  let f = kfold(c);
  var col = beads(f + vec2f(0.15, 0.08), t);
  col *= 1.0 - 0.35 * smoothstep(0.45, 0.95, length((px - u.resolution * 0.5) / H));
  // inset: the unfolded source with the wedge that gets mirrored
  if (u.showSource > 0.5) {
    let iw = H * 0.34;
    let io = u.resolution - vec2f(iw + 12.0, iw + 12.0);
    let lq = (px - io) / iw;
    if (lq.x > 0.0 && lq.x < 1.0 && lq.y > 0.0 && lq.y < 1.0) {
      let sp = (lq - 0.5) * u.kzoom * 1.6;
      var sc = beads(sp + vec2f(0.15, 0.08), t);
      let seg = TAU / u.segments;
      var a = atan2(sp.y, sp.x) + u.time * u.spin * 0.1;
      a = fmod(a, TAU);
      let inW = step(a, seg * 0.5);
      sc = mix(sc * 0.35, sc, inW);
      let r = length(sp);
      let a1 = abs(sin(-u.time * u.spin * 0.1) * sp.x - cos(-u.time * u.spin * 0.1) * sp.y);
      let a2 = abs(sin(seg * 0.5 - u.time * u.spin * 0.1) * sp.x - cos(seg * 0.5 - u.time * u.spin * 0.1) * sp.y);
      let ln = min(a1, a2) / (u.kzoom * 1.6) * iw;
      sc = mix(sc, vec3f(1.0), (1.0 - smoothstep(0.5, 1.5, ln)) * step(0.0, dot(sp, vec2f(cos(seg * 0.25 - u.time * u.spin * 0.1), sin(seg * 0.25 - u.time * u.spin * 0.1)))));
      col = sc;
    }
    let e = max(abs(lq.x - 0.5), abs(lq.y - 0.5));
    col = mix(col, vec3f(0.9), (1.0 - smoothstep(0.0, 1.5 / iw, abs(e - 0.5))) * 0.8);
  }
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = exRepeat(px); }
  else if (ex == 1) { col = exTruchet(px); }
  else if (ex == 2) { col = exHex(px); }
  else { col = exKaleido(px); }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: 'Hover a hex: range, line and axial coordinates.',
  examples: [
    {
      id: 'repeat',
      label: 'Repetition with fract()',
      kind: 'Abstract',
      note: '<code>fract(p × cells)</code> gives every pixel its position inside a cell, and <code>floor()</code> gives the cell’s id — draw one motif and it repeats forever, for free. Offset every other row (brick), mirror every other cell (seamless), or repeat in angle instead of x/y (polar).',
      hint: '',
    },
    {
      id: 'truchet',
      label: 'Truchet tiles & pipes',
      kind: 'Abstract',
      note: 'One tile with two quarter-circle arcs, randomly flipped per cell — and long winding paths emerge. The arcs also split the plane into two colors (left), or become a maze with diagonals (the famous one-line program <code>10 PRINT CHR$(205.5+RND(1))</code>), or pipes with liquid flowing through. <b>Click a tile to flip it</b> — the core of every pipe-connecting puzzle game.',
      hint: 'Click a tile to flip it.',
      params: { style: 'pipes' },
    },
    {
      id: 'hex',
      label: 'Hex strategy board',
      kind: 'In a game',
      note: 'Hex grids use <b>axial coordinates</b> (q, r) — two axes at 60°, with s = −q − r as an optional third. Converting pixel → hex is a matrix multiply plus a careful rounding step; distance is <code>(|Δq| + |Δr| + |Δs|) / 2</code>. Hover to see the hex under the cursor, the movement range around it and the hex line back to the unit (★).',
    },
    {
      id: 'kaleido',
      label: 'Kaleidoscope',
      kind: 'Real life',
      note: 'A kaleidoscope is two mirrors at an angle: fold the angle of every pixel into a single wedge (mirroring every other wedge) and sample the moving image there. The inset shows the unfolded source and the wedge that is being repeated.',
      hint: '',
    },
  ],
  controls: [
    { type: 'slider', key: 'cells', label: 'Cells', min: 1, max: 12, step: 0.1, value: 3.5, showFor: ['repeat'], help: 'Repetitions per panel height (fractional values cut a cell in half — fract keeps working).' },
    { type: 'slider', key: 'sectors', label: 'Polar sectors', min: 3, max: 24, step: 1, value: 8, showFor: ['repeat'], help: 'How many times the motif repeats around the circle.' },
    { type: 'toggle', key: 'showCells', label: 'Show cells / local uv', value: false, showFor: ['repeat', 'truchet'], help: 'Visualize each cell’s local coordinates (red = x, green = y) or tile borders.' },
    { type: 'slider', key: 'tiles', label: 'Tiles', min: 3, max: 40, step: 1, value: 9, showFor: ['truchet'], help: 'Tiles per screen height.' },
    {
      type: 'select', key: 'style', label: 'Tile style', value: 'pipes', showFor: ['truchet'],
      options: [{ value: 'arcs', label: 'Arcs, two-color regions' }, { value: 'maze', label: '10 PRINT maze (diagonals)' }, { value: 'pipes', label: 'Pipes with flowing liquid' }],
    },
    { type: 'slider', key: 'hexRows', label: 'Hex rows on screen', min: 4, max: 30, step: 0.5, value: 9, showFor: ['hex'] },
    { type: 'select', key: 'orient', label: 'Orientation', value: 'pointy', showFor: ['hex'], options: [{ value: 'pointy', label: 'Pointy-top' }, { value: 'flat', label: 'Flat-top' }] },
    { type: 'slider', key: 'range', label: 'Movement range', min: 0, max: 6, step: 1, value: 2, showFor: ['hex'], help: 'Hexes with distance ≤ range from the hovered hex.' },
    { type: 'toggle', key: 'showCoords', label: 'Color by axial coordinates', value: false, showFor: ['hex'], help: 'Debug view: hue from q, r and s; the three axes (q=0, r=0, s=0) are highlighted.' },
    { type: 'slider', key: 'segments', label: 'Mirror segments', min: 2, max: 16, step: 1, value: 6, showFor: ['kaleido'], help: 'Number of wedge pairs around the circle.' },
    { type: 'slider', key: 'kzoom', label: 'Zoom', min: 0.5, max: 4, step: 0.01, value: 1.0, showFor: ['kaleido'], help: 'How much of the source image each wedge shows.' },
    { type: 'toggle', key: 'showSource', label: 'Show source & wedge', value: true, showFor: ['kaleido'] },
    { type: 'slider', key: 'spin', label: 'Animation speed', min: 0, max: 3, step: 0.01, value: 1, showFor: ['repeat', 'truchet', 'kaleido'] },
    { type: 'slider', key: 'seed', label: 'Seed', min: 0, max: 99, step: 1, value: 0, showFor: ['truchet', 'hex'] },
    { type: 'button', key: 'clearFlips', label: 'Undo all flips', showFor: ['truchet'] },
  ],
  uniforms: {
    cells: 'f32', sectors: 'f32', showCells: 'f32', tiles: 'f32', style: 'f32', hexRows: 'f32', orient: 'f32', range: 'f32', showCoords: 'f32',
    segments: 'f32', kzoom: 'f32', showSource: 'f32', spin: 'f32', seed: 'f32', nflip: 'f32', flips: `array<vec4f, ${MAX_FLIPS}>`,
  },
  include: ['noise', 'sdf', 'color'],
  onAction(key) {
    if (key === 'clearFlips' || key === 'reset') flips.length = 0;
  },
  bind(params, ctx) {
    const ex = ctx.example;
    if (ex === 'truchet') {
      if (params.tiles !== lastTiles) {
        flips.length = 0;
        lastTiles = params.tiles;
      }
      if (ctx.pointer.clicked) {
        const ts = ctx.height / params.tiles;
        const tx = Math.floor(ctx.pointer.x / ts);
        const ty = Math.floor(ctx.pointer.y / ts);
        const i = flips.findIndex((f) => f[0] === tx && f[1] === ty);
        if (i >= 0) flips.splice(i, 1);
        else {
          flips.push([tx, ty]);
          if (flips.length > MAX_FLIPS) flips.shift();
        }
      }
      setLabels(ctx, []);
    } else if (ex === 'repeat') {
      setLabels(ctx, quadLabels(['<b>Grid</b> · fract(p)', '<b>Brick</b> · offset odd rows', '<b>Mirror</b> · flip odd cells', '<b>Polar</b> · repeat the angle']));
    } else if (ex === 'hex') {
      setLabels(ctx, hexInfo(params, ctx));
    } else {
      setLabels(ctx, params.showSource ? [{ text: 'source + mirrored wedge', style: 'right:12px;bottom:calc(34% + 18px);font-size:11px;opacity:.85' }] : []);
    }
    flipData.fill(0);
    flips.forEach((f, i) => flipData.set([f[0], f[1], 0, 0], i * 4));
    return { nflip: flips.length, flips: flipData };
  },
  code: CODE,
  about: {
    summary: 'Draw something once, get it everywhere: fract() repetition, flipped Truchet tiles, hexagonal grids and mirror folds turn a single motif into infinite patterns at zero extra cost.',
    what: `<p><b>Repetition</b>: one motif repeated four ways. <b>Truchet</b>: one randomly-flipped tile creating paths, mazes and pipes. <b>Hex board</b>: a strategy-game grid with
      axial coordinates, movement range and line drawing. <b>Kaleidoscope</b>: angular mirror folds of a moving picture.</p>`,
    how: `<ol>
      <li><b>Cells</b>: multiply the position by the number of cells; <code>floor(p)</code> is the cell id, <code>fract(p) − 0.5</code> the position inside it.
        Every pixel only ever draws <i>one</i> motif — the one for its own cell.</li>
      <li><b>Variation per cell</b>: hash the cell id to get a random number per cell (rotate, flip, recolor). Brick = add 0.5 to x on odd rows. Mirror = negate x on odd columns.</li>
      <li><b>Polar repetition</b>: convert to (angle, radius), then do the same <code>floor/fract</code> trick on the angle.</li>
      <li><b>Truchet</b>: each tile contains two arcs around opposite corners; flipping the tile (x → 1 − x) swaps the corners. Arcs always meet the tile edges at their midpoints, so
        neighbours always connect. For flow along the pipes, the direction is reversed on a checkerboard so it stays continuous.</li>
      <li><b>Hexes</b>: pixel → fractional axial coords with a 2×2 matrix, then <i>cube rounding</i> (round q, r, s; fix the one with the largest rounding error). Neighbours are 6 fixed
        offsets; distance = (|Δq|+|Δr|+|Δs|)/2; a hex line rounds evenly spaced points between two hexes.</li>
      <li><b>Kaleidoscope</b>: <code>a = mod(angle, wedge); a = min(a, wedge − a)</code> — the second step is the mirror.</li>
    </ol>`,
    uses: [
      { title: 'Backgrounds & UI', text: 'Animated menu backgrounds, loading screens, patterned panels, transitions — tiny shaders, no textures.' },
      { title: 'Puzzle games', text: 'Pipe-connecting puzzles, mazes and path tiles (Truchet logic), match-3 boards, tile flipping.' },
      { title: 'Strategy & tactics', text: 'Hex boards (Civilization V/VI, Battle for Wesnoth, Into the Breach-style grids): movement range, line of sight, pathfinding.' },
      { title: 'Magic & psychedelia', text: 'Summoning circles, mandalas, rune rings and kaleidoscopic boss-fight backgrounds.' },
    ],
    try: [
      'On <b>Repetition</b>, turn on <i>Show cells / local uv</i>, then look at the mirror quadrant: neighbouring cells’ gradients meet seamlessly.',
      'On <b>Truchet</b>, switch the style between arcs, maze and pipes — it’s the same random flip per tile each time. Then click tiles to reroute the pipes.',
      'On <b>Hex board</b>, toggle <i>Color by axial coordinates</i> and watch the hover label: moving along a highlighted axis changes only one coordinate.',
      'Switch hexes to <b>Flat-top</b>: the same math with x and y swapped.',
      'On <b>Kaleidoscope</b>, set <i>Mirror segments</i> to 2 and grow it slowly while watching the wedge in the inset.',
    ],
    ask: [
      'seamless repeating pattern shader with fract()',
      'Truchet tile maze / pipe puzzle',
      'hex grid with axial coordinates, hover highlight and movement range',
      'pixel-to-hex conversion with cube rounding',
      'kaleidoscope shader effect',
      'polar (radial) repetition for a magic circle',
    ],
    perf: `<p>Repetition is free: each pixel still evaluates exactly one motif. The cost is the motif itself. The hex board checks two neighbours for range borders and up to
      ~40 points for the hex line — trivial. For huge boards in a real game you'd draw hexes as instanced sprites and keep per-hex data in a texture or buffer.</p>`,
    api: `<p>Pure math, identical in WebGL2 and WebGPU. The clicked Truchet tiles are a small uniform array updated from JavaScript; a game with a large board would
      store tile states in a data texture (WebGL2) or a storage buffer (WebGPU) instead.</p>`,
    code: [
      {
        title: 'Grid, brick, mirror and polar repetition',
        lang: 'wgsl',
        src: `let p = lp * u.cells;
// grid
col = motif(fract(p) - 0.5, floor(p));
// brick: shift every other row by half a cell
p.x += 0.5 * fmod(floor(p.y), 2.0);
// mirror: flip odd cells
if (fmod(id.x, 2.0) > 0.5) { q.x = -q.x; }
// polar: repeat the angle
let sector = TAU / u.sectors;
let ai = floor(a / sector + 0.5);
let q = vec2f(fract(r) - 0.5, (a - ai * sector) / sector);`,
      },
      {
        title: 'Truchet arcs',
        lang: 'wgsl',
        src: `let id = floor(p);  var q = fract(p);
if (isFlipped(id) > 0.5) { q.x = 1.0 - q.x; }    // random per tile (hash) XOR clicked
let d = min(abs(length(q) - 0.5),               // arc around corner (0,0)
            abs(length(q - vec2f(1.0)) - 0.5)); // arc around corner (1,1)
col = mix(bg, lineColor, 1.0 - smoothstep(w, w + aa, d));`,
      },
      {
        title: 'Pixel → hex (pointy-top axial) with cube rounding',
        lang: 'wgsl',
        src: `fn toHex(p: vec2f, s: f32) -> vec2f {
  return hexRound((0.57735027 * p.x - 0.33333333 * p.y) / s, (0.66666667 * p.y) / s);
}
fn hexRound(fq: f32, fr: f32) -> vec2f {
  let fs = -fq - fr;
  var q = floor(fq + 0.5);  var r = floor(fr + 0.5);  let s = floor(fs + 0.5);
  let dq = abs(q - fq);  let dr = abs(r - fr);  let ds = abs(s - fs);
  if (dq > dr && dq > ds) { q = -r - s; } else if (dr > ds) { r = -q - s; }
  return vec2f(q, r);
}
fn hexDist(a: vec2f, b: vec2f) -> f32 {
  let d = a - b;  return (abs(d.x) + abs(d.y) + abs(d.x + d.y)) * 0.5;
}`,
      },
    ],
    links: [
      { title: 'Red Blob Games — Hexagonal Grids', url: 'https://www.redblobgames.com/grids/hexagons/', note: 'the definitive guide' },
      { title: 'The Book of Shaders — Patterns', url: 'https://thebookofshaders.com/09/' },
      { title: '10 PRINT (the book)', url: 'https://10print.org/', note: 'a whole book about one line of maze code' },
    ],
  },
});

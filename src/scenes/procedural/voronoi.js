import { shaderScene } from '../../core/shaderscene.js';
import { setLabels, quadLabels } from './_shared.js';

// Voronoi / cellular patterns: F1, F2, F2-F1, exact border distance, distance metrics and the
// 3×3 neighbour search; natural patterns (stained glass, cracked earth, giraffe, dragon scales);
// and a Civilization-style territory map where every click founds a city.

const MAX_CITIES = 32;
const TEAMS = ['red', 'blue', 'green', 'gold', 'purple', 'teal'];
const NAMES = [
  'Avalon', 'Brightwater', 'Corvin', 'Dunmere', 'Eastholm', 'Fairhaven', 'Glimmer', 'Highcrest', 'Ironford', 'Jadeport',
  'Kingsreach', 'Lowmarsh', 'Moonfall', 'Northwatch', 'Oakheart', 'Pinecliff', 'Queensgate', 'Ravenhold', 'Stonebridge',
  'Thornbury', 'Umberlee', 'Valewood', 'Westmarch', 'Yarrow', 'Zephyr', 'Ashgrove', 'Bramble', 'Coldharbor', 'Dawnstar',
  'Emberly', 'Frostvale', 'Goldmoor',
];
const DEFAULT_CITIES = [
  { x: 0.2, y: 0.3, team: 0 },
  { x: 0.33, y: 0.62, team: 0 },
  { x: 0.52, y: 0.28, team: 1 },
  { x: 0.66, y: 0.5, team: 1 },
  { x: 0.82, y: 0.3, team: 2 },
  { x: 0.84, y: 0.74, team: 2 },
  { x: 0.48, y: 0.8, team: 3 },
  { x: 0.13, y: 0.78, team: 3 },
];
let cities = DEFAULT_CITIES.map((c, i) => ({ ...c, name: NAMES[i] }));
let nameCounter = cities.length;
const cityData = new Float32Array(MAX_CITIES * 4);

function handleMapInput(params, ctx) {
  const p = ctx.pointer;
  if (!p.clicked) return;
  const x = p.x / ctx.width;
  const y = p.y / ctx.height;
  if (p.button === 2) {
    // right click: remove the nearest city
    let best = -1;
    let bd = 1e9;
    cities.forEach((c, i) => {
      const d = Math.hypot((c.x - x) * ctx.width, (c.y - y) * ctx.height);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    if (best >= 0 && bd < 60 * (ctx.dpr || 1)) cities.splice(best, 1);
    return;
  }
  if (cities.length >= MAX_CITIES) {
    ctx.status?.(`The shader array holds ${MAX_CITIES} cities — right-click to remove one first.`);
    return;
  }
  const team = Math.max(0, TEAMS.indexOf(params.team));
  cities.push({ x, y, team, name: NAMES[nameCounter++ % NAMES.length] });
}

function cityLabels() {
  return cities.map((c) => ({
    text: c.name,
    style: `left:${(c.x * 100).toFixed(2)}%;top:${(c.y * 100).toFixed(2)}%;transform:translate(-50%, 12px);font-size:10px;padding:1px 6px;opacity:.9`,
  }));
}

const CODE = /* wgsl */ `
// ------------------------------------------------------------------ a configurable Voronoi
struct Vor { f1: f32, f2: f32, id: vec2f, pt: vec2f };

fn vdist(r: vec2f) -> f32 {
  if (u.metric > 1.5) { return max(abs(r.x), abs(r.y)); }
  if (u.metric > 0.5) { return abs(r.x) + abs(r.y); }
  return length(r);
}

// feature point of a cell (in cell-local 0..1), orbiting when animated
fn fpoint(cell: vec2f, jit: f32, t: f32) -> vec2f {
  let h = hash22(cell + vec2f(u.seed * 17.0, u.seed * 3.0));
  return 0.5 + jit * 0.5 * sin(t * (0.6 + 0.8 * h.yx) + TAU * h);
}

fn vor(p: vec2f, jit: f32, t: f32) -> Vor {
  let n = floor(p);
  let f = fract(p);
  var v: Vor;
  v.f1 = 8.0;
  v.f2 = 8.0;
  v.id = n;
  v.pt = n;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let g = vec2f(f32(i), f32(j));
      let o = fpoint(n + g, jit, t);
      let d = vdist(g + o - f);
      if (d < v.f1) {
        v.f2 = v.f1;
        v.f1 = d;
        v.id = n + g;
        v.pt = n + g + o;
      } else if (d < v.f2) {
        v.f2 = d;
      }
    }
  }
  return v;
}

// exact distance to the cell border (Euclidean): second pass over the 5x5 neighbourhood (IQ)
fn vborder(p: vec2f, jit: f32, t: f32) -> f32 {
  let n = floor(p);
  let f = fract(p);
  var mg = vec2f(0.0);
  var mr = vec2f(0.0);
  var md = 8.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let g = vec2f(f32(i), f32(j));
      let r = g + fpoint(n + g, jit, t) - f;
      let d = dot(r, r);
      if (d < md) { md = d; mr = r; mg = g; }
    }
  }
  md = 8.0;
  for (var j = -2; j <= 2; j++) {
    for (var i = -2; i <= 2; i++) {
      let g = mg + vec2f(f32(i), f32(j));
      let r = g + fpoint(n + g, jit, t) - f;
      if (dot(mr - r, mr - r) > 0.00001) {
        md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
      }
    }
  }
  return md;
}

fn cellColor(id: vec2f) -> vec3f {
  let h = hash23(id + vec2f(u.seed * 5.0, 1.0));
  return palette(h.x, vec3f(0.55, 0.5, 0.55), vec3f(0.35, 0.35, 0.3), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.25, 0.55)) * (0.75 + 0.35 * h.y);
}

fn distRamp(x: f32) -> vec3f {
  // magma-like ramp + contour lines every 0.1
  let c = palette(clamp(x, 0.0, 1.2) * 0.8, vec3f(0.5, 0.4, 0.45), vec3f(0.5, 0.4, 0.4), vec3f(1.0, 0.9, 0.7), vec3f(0.6, 0.75, 0.9));
  let iso = abs(fract(x * 10.0 + 0.5) - 0.5) / max(fwidth(x * 10.0), 0.0001);
  return mix(c * (0.25 + 0.75 * smoothstep(0.0, 0.9, x)), c * 0.35, (1.0 - smoothstep(0.5, 1.5, iso)) * 0.6);
}

// ------------------------------------------------------------------ example 0: cells & distance fields
fn exCells(px: vec2f) -> vec3f {
  let H = u.resolution.y;
  let ppu = H / u.scale;
  let t = u.time * u.anim;
  let p = px / ppu;
  let v = vor(p, u.jitter, t);
  var col = vec3f(0.0);
  let mode = i32(u.viz);
  if (mode == 0) {
    col = cellColor(v.id);
    col *= 0.8 + 0.2 * smoothstep(0.0, 0.6, v.f2 - v.f1);
  } else if (mode == 1) {
    col = distRamp(v.f1);
  } else if (mode == 2) {
    col = distRamp(v.f2 * 0.7);
  } else if (mode == 3) {
    col = distRamp((v.f2 - v.f1) * 1.4);
  } else {
    var bd = v.f2 - v.f1;
    if (u.metric < 0.5) { bd = vborder(p, u.jitter, t); }
    col = distRamp(bd * 2.2);
  }
  if (u.points > 0.5) {
    // feature points + cell outlines (from F2 - F1, valid for every metric)
    let edge = (v.f2 - v.f1) * ppu;
    col = mix(col, vec3f(0.02), (1.0 - smoothstep(0.6, 1.6, edge)) * 0.7);
    let dp = length(p - v.pt) * ppu;
    col = mix(col, vec3f(0.0), 1.0 - smoothstep(4.0, 5.0, dp));
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(2.5, 3.5, dp));
  }
  // hover: the 3x3 neighbourhood search for the point under the mouse
  if (u.mouse.w > 0.5) {
    let m = u.mouse.xy / ppu;
    let mn = floor(m);
    let mf = fract(m);
    let rel = floor(p) - mn;
    let inBlock = step(-1.0, rel.x) * step(rel.x, 1.0) * step(-1.0, rel.y) * step(rel.y, 1.0);
    let gl = abs(fract(p + 0.5) - 0.5) * ppu;
    let gridLine = (1.0 - smoothstep(0.4, 1.2, min(gl.x, gl.y))) * inBlock;
    col = mix(col, vec3f(1.0, 0.85, 0.4), gridLine * 0.55);
    col = mix(col, col * 0.75 + vec3f(0.05, 0.04, 0.0), inBlock * 0.25);
    var best = 8.0;
    var bp = vec2f(0.0);
    var lines = 1e9;
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let g = vec2f(f32(i), f32(j));
        let pt = mn + g + fpoint(mn + g, u.jitter, t);
        let d = vdist(pt - m);
        if (d < best) { best = d; bp = pt; }
        lines = min(lines, sdSegment(p, m, pt) * ppu);
      }
    }
    col = mix(col, vec3f(1.0, 0.85, 0.4), (1.0 - smoothstep(0.3, 1.1, lines)) * 0.6);
    let bl = sdSegment(p, m, bp) * ppu;
    col = mix(col, vec3f(0.0), (1.0 - smoothstep(1.5, 3.5, bl)) * 0.5);
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(0.8, 1.8, bl));
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(3.0, 4.0, length(p - m) * ppu));
  }
  return col;
}

// ------------------------------------------------------------------ example 1: patterns from nature
fn stainedGlass(lp: vec2f, t: f32) -> vec3f {
  let p = lp * u.scale * 0.8 + vec2f(u.seed * 7.0, 0.0);
  let v = vor(p, 0.9, 0.0);
  let bd = vborder(p, 0.9, 0.0);
  let h = hash23(v.id + 31.0);
  // pick from a small set of rich glass colors
  var g = vec3f(0.75, 0.08, 0.12);
  if (h.x > 0.2) { g = vec3f(0.08, 0.2, 0.7); }
  if (h.x > 0.42) { g = vec3f(0.95, 0.65, 0.08); }
  if (h.x > 0.6) { g = vec3f(0.1, 0.55, 0.25); }
  if (h.x > 0.78) { g = vec3f(0.5, 0.15, 0.6); }
  if (h.x > 0.9) { g = vec3f(0.85, 0.85, 0.75); }
  // light shining through: brighter in the middle, streaky glass texture
  let back = 0.55 + 0.6 * exp(-dot(lp - vec2f(0.95, 0.35), lp - vec2f(0.95, 0.35)) * 2.0);
  let streak = 0.85 + 0.15 * perlin(p * vec2f(3.0, 12.0) + h.yz * 10.0);
  var c = g * back * streak * (0.8 + 0.3 * h.y);
  c *= 0.55 + 0.45 * smoothstep(0.0, 0.25, bd);
  c += g * 0.25 * pow(max(0.0, sin(t * 0.7 + h.z * 6.0)), 8.0);
  // lead came with a rounded highlight
  let w = 0.07;
  let lead = 1.0 - smoothstep(w - 0.015, w, bd);
  let prof = clamp(bd / w, 0.0, 1.0);
  let leadC = vec3f(0.12, 0.12, 0.14) + vec3f(0.35) * (1.0 - prof) * (1.0 - prof) * 0.6;
  return mix(c, leadC, lead);
}

fn crackedEarth(lp: vec2f) -> vec3f {
  var p = lp * u.scale * 0.7 + vec2f(u.seed * 7.0, 3.0);
  p += 0.18 * vec2f(fbm(p * 1.5, 3), fbm(p * 1.5 + 5.2, 3));
  let v = vor(p, 0.95, 0.0);
  let bd = vborder(p, 0.95, 0.0);
  let h = hash21(v.id);
  var c = mix(vec3f(0.62, 0.45, 0.3), vec3f(0.72, 0.56, 0.38), h);
  c *= 0.85 + 0.15 * fbm(p * 6.0, 3);
  // plates curl up at the edges: fake height from the border distance, lit from the top-left
  let hgt = 1.0 - smoothstep(0.0, 0.22, bd);
  let gx = dpdx(hgt);
  let gy = dpdy(hgt);
  c *= clamp(1.0 - (gx + gy) * 6.0, 0.6, 1.35);
  // small secondary cracks
  let v2 = vor(p * 3.0 + 11.0, 0.9, 0.0);
  c *= 1.0 - 0.35 * (1.0 - smoothstep(0.0, 0.06, v2.f2 - v2.f1)) * smoothstep(0.08, 0.2, bd);
  // main cracks: width varies with noise
  let cw = 0.035 + 0.03 * perlin(p * 2.0);
  let crack = 1.0 - smoothstep(cw * 0.5, cw, bd);
  return mix(c, vec3f(0.13, 0.08, 0.05), crack);
}

fn giraffe(lp: vec2f) -> vec3f {
  var p = lp * u.scale * 0.75 + vec2f(u.seed * 7.0, 9.0);
  p += 0.08 * vec2f(perlin(p * 2.0), perlin(p * 2.0 + 4.0));
  let v = vor(p, 0.8, 0.0);
  let bd = vborder(p, 0.8, 0.0);
  let h = hash21(v.id + 4.0);
  let cream = vec3f(0.95, 0.88, 0.72) * (0.92 + 0.08 * perlin(lp * 40.0));
  var spot = mix(vec3f(0.62, 0.33, 0.12), vec3f(0.5, 0.25, 0.09), h);
  spot = mix(spot * 0.8, spot * 1.1, smoothstep(0.05, 0.35, bd));
  spot *= 0.9 + 0.1 * perlin(p * 9.0);
  let w = 0.1 + 0.02 * perlin(p * 3.0);
  return mix(cream, spot, smoothstep(w, w + 0.025, bd));
}

// overlapping scales: rows of circles on a staggered lattice; the first row (top) is in FRONT,
// so every scale shows its rounded bottom edge over the next row
fn dragonScales(lp: vec2f, t: f32) -> vec3f {
  let p = lp * u.scale * 0.6 + vec2f(u.seed * 7.0, 0.0);
  let rs = 0.55;
  let rad = 0.62;
  let row0 = floor(p.y / rs);
  var found = 0.0;
  var cq = vec2f(0.0);
  var cid = vec2f(0.0);
  var dEdge = 1.0;
  for (var j = -2; j <= 0; j++) {
    let row = row0 + f32(j);
    let sh = 0.5 * fmod(row, 2.0);
    let c0 = floor(p.x - sh);
    for (var i = -1; i <= 1; i++) {
      let cx = c0 + f32(i);
      let ctr = vec2f(cx + sh + 0.5, row * rs + rs);
      let d = length(p - ctr);
      if (found < 0.5 && d < rad) {
        found = 1.0;
        cq = (p - ctr) / rad;
        cid = vec2f(cx, row);
        dEdge = rad - d;
      }
    }
  }
  let hh = hash21(cid);
  // dome normal from the position inside the scale
  let n = normalize(vec3f(cq * 1.1, 0.8));
  let L = normalize(vec3f(-0.45, -0.7, 0.6));
  let diff = max(dot(n, L), 0.0);
  let spec = pow(max(dot(reflect(-L, n), vec3f(0.0, 0.0, 1.0)), 0.0), 20.0);
  let irid = palette(0.42 + 0.22 * cq.x + 0.18 * cq.y + hh * 0.12 + 0.04 * sin(t + hh * 6.0), vec3f(0.3, 0.45, 0.32), vec3f(0.25, 0.3, 0.22), vec3f(1.0, 1.0, 1.0), vec3f(0.1, 0.25, 0.45));
  var c = irid * (0.4 + 1.05 * diff) + vec3f(1.0, 0.95, 0.8) * spec * 0.6;
  // the upper part tucks under the previous row: shadow it
  c *= 0.45 + 0.55 * smoothstep(-0.55, 0.25, cq.y);
  // dark rim + a thin bright lip just inside it
  let ew = dEdge * u.resolution.y * 0.5 / (u.scale * 0.6);
  c = mix(c, c * 1.5 + 0.08, (1.0 - smoothstep(1.5, 4.0, ew)) * smoothstep(0.2, 0.6, cq.y) * 0.5);
  c *= smoothstep(0.0, 1.6, ew) * 0.85 + 0.15;
  return c;
}

fn exNature(px: vec2f) -> vec3f {
  let hs = u.resolution * 0.5;
  let qd = vec2f(step(hs.x, px.x), step(hs.y, px.y));
  let k = i32(qd.x + 2.0 * qd.y);
  let lp = (px - qd * hs) / hs.y;
  let t = u.time * u.anim;
  var col = vec3f(0.0);
  if (k == 0) { col = stainedGlass(lp, t); }
  else if (k == 1) { col = crackedEarth(lp); }
  else if (k == 2) { col = giraffe(lp); }
  else { col = dragonScales(lp, t); }
  let edge = min(abs(px.x - hs.x), abs(px.y - hs.y));
  return mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, edge));
}

// ------------------------------------------------------------------ example 2: territory map
fn teamColor(t: f32) -> vec3f {
  let k = i32(t + 0.5);
  if (k == 0) { return vec3f(0.92, 0.26, 0.24); }
  if (k == 1) { return vec3f(0.24, 0.5, 0.98); }
  if (k == 2) { return vec3f(0.3, 0.82, 0.36); }
  if (k == 3) { return vec3f(0.98, 0.78, 0.2); }
  if (k == 4) { return vec3f(0.72, 0.38, 0.95); }
  return vec3f(0.2, 0.85, 0.85);
}

fn mapTerrain(px: vec2f) -> vec3f {
  let H = u.resolution.y;
  let p = px / H * 2.2 + vec2f(u.seed * 3.3, 1.7);
  let e = fbm(p, 6) + 0.12 - 0.35 * pow(length((px - u.resolution * 0.5) / u.resolution) * 1.5, 3.0);
  let e2 = fbm(p + vec2f(0.004, 0.004), 6) + 0.12 - 0.35 * pow(length((px + vec2f(1.6) - u.resolution * 0.5) / u.resolution) * 1.5, 3.0);
  let moist = fbm(p * 1.3 + 9.0, 3);
  if (e < 0.0) {
    var w = mix(vec3f(0.16, 0.36, 0.52), vec3f(0.07, 0.17, 0.32), smoothstep(0.0, -0.35, e));
    w = mix(w, vec3f(0.55, 0.75, 0.75), (1.0 - smoothstep(0.0, 0.025, -e)) * 0.6);
    return w;
  }
  var land = mix(vec3f(0.55, 0.62, 0.36), vec3f(0.32, 0.5, 0.26), smoothstep(-0.1, 0.2, moist));
  land = mix(land, vec3f(0.55, 0.5, 0.4), smoothstep(0.18, 0.3, e));
  land = mix(land, vec3f(0.92, 0.92, 0.94), smoothstep(0.34, 0.4, e));
  land = mix(land, vec3f(0.85, 0.8, 0.6), 1.0 - smoothstep(0.0, 0.02, e));
  let shadeF = clamp(1.0 + (e - e2) * 40.0, 0.6, 1.4);
  return land * shadeF;
}

// nearest city (within the culture radius): (team, index, distance px)
fn nearestCity(p: vec2f) -> vec3f {
  var best = 1e9;
  var bi = -1.0;
  var bt = -1.0;
  for (var i = 0; i < ${MAX_CITIES}; i++) {
    if (f32(i) >= u.ncity) { break; }
    let c = u.cities[i];
    let d = length(p - c.xy * u.resolution);
    if (d < best) { best = d; bi = f32(i); bt = c.z; }
  }
  if (best > u.culture * u.resolution.y) { return vec3f(-1.0, bi, best); }
  return vec3f(bt, bi, best);
}

// distance (px) from p to the edge of the territory of team 'team' owned via city 'ci'
fn borderDist(p: vec2f, team: f32, ci: f32, d1: f32) -> f32 {
  let ca = u.cities[i32(ci)].xy * u.resolution;
  var bd = u.culture * u.resolution.y - d1;
  for (var i = 0; i < ${MAX_CITIES}; i++) {
    if (f32(i) >= u.ncity) { break; }
    let c = u.cities[i];
    if (abs(c.z - team) > 0.5) {
      let cp = c.xy * u.resolution;
      let dj = (dot(p - cp, p - cp) - d1 * d1) / (2.0 * max(length(cp - ca), 0.001));
      bd = min(bd, dj);
    }
  }
  return bd;
}

// hex helpers (pointy-top, size s): pixel -> axial (q, r) of the containing hex, and back
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
fn pixelToHex(p: vec2f, s: f32) -> vec2f {
  return hexRound((0.57735027 * p.x - 0.33333333 * p.y) / s, (0.66666667 * p.y) / s);
}
fn hexToPixel(h: vec2f, s: f32) -> vec2f {
  return vec2f(s * (1.7320508 * h.x + 0.8660254 * h.y), s * 1.5 * h.y);
}

fn exTerritory(px: vec2f) -> vec3f {
  var col = mapTerrain(px);
  let R = u.culture * u.resolution.y;
  var team = -1.0;
  var bd = 1e9;
  var ci = -1.0;
  if (u.hexes > 0.5) {
    // own whole hexes: ownership is decided at the hex center; borders run along hex edges
    let s = u.resolution.y / 26.0;
    let hx = pixelToHex(px, s);
    let hc = hexToPixel(hx, s);
    let own = nearestCity(hc);
    team = own.x;
    ci = own.y;
    // the two nearest edges: check the neighbours across them
    let rel = px - hc;
    let ang = atan2(rel.y, rel.x);
    let k0 = floor(ang / (PI / 3.0));
    for (var e = 0; e < 2; e++) {
      let k = k0 + f32(e);
      let a = k * PI / 3.0;
      let nd = vec2f(cos(a), sin(a));
      let nb = nearestCity(hc + nd * s * 1.7320508);
      let distEdge = s * 0.8660254 - dot(rel, nd);
      if (abs(nb.x - team) > 0.5) { bd = min(bd, distEdge); }
    }
    let gl = s * 0.8660254 - max(max(abs(dot(rel, vec2f(1.0, 0.0))), abs(dot(rel, vec2f(0.5, 0.8660254)))), abs(dot(rel, vec2f(-0.5, 0.8660254))));
    col = mix(col, col * 0.82, (1.0 - smoothstep(0.0, 1.2, gl)) * 0.6);
  } else {
    let own = nearestCity(px);
    team = own.x;
    ci = own.y;
    if (team > -0.5) { bd = borderDist(px, team, ci, own.z); }
  }
  if (team > -0.5) {
    let tc = teamColor(team);
    let glow = exp(-max(bd, 0.0) / (u.resolution.y * 0.025));
    col = mix(col, tc, 0.16 + 0.42 * glow);
    let bw = u.borderW;
    col = mix(col, tc * 1.15 + 0.1, 1.0 - smoothstep(bw, bw + 1.2, bd));
    col = mix(col, tc * 0.25, 1.0 - smoothstep(0.6, 1.4, bd));
  }
  // city markers
  for (var i = 0; i < ${MAX_CITIES}; i++) {
    if (f32(i) >= u.ncity) { break; }
    let c = u.cities[i];
    let d = length(px - c.xy * u.resolution);
    let r0 = u.resolution.y * 0.014;
    col = mix(col, vec3f(0.05), 1.0 - smoothstep(r0 + 2.0, r0 + 3.0, d));
    col = mix(col, vec3f(0.97), 1.0 - smoothstep(r0, r0 + 1.0, d));
    col = mix(col, teamColor(c.z), 1.0 - smoothstep(r0 * 0.55, r0 * 0.55 + 1.0, d));
  }
  // preview ring under the mouse: where a new city would claim land
  if (u.mouse.w > 0.5) {
    let dm = abs(length(px - u.mouse.xy) - R);
    let dash = step(0.5, fract(atan2(px.y - u.mouse.y, px.x - u.mouse.x) * 12.0 / PI));
    col = mix(col, teamColor(u.team), (1.0 - smoothstep(0.5, 1.5, dm)) * dash * 0.8);
  }
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = exCells(px); }
  else if (ex == 1) { col = exNature(px); }
  else { col = exTerritory(px); }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;

const VIZ_NAMES = ['Cells (random color per cell)', 'F1: distance to the nearest point', 'F2: distance to the 2nd nearest', 'F2 − F1: “how close to an edge”', 'Exact distance to the border'];

export default shaderScene({
  interaction: 'Hover to see the 3×3 neighbour search for that point.',
  examples: [
    {
      id: 'cells',
      label: 'Cells & distance fields',
      kind: 'Abstract',
      note: 'Scatter one random point in every grid cell. Each pixel finds its nearest point (only the 3×3 neighbouring cells need checking — hover to see the search). The distance to the nearest point is <b>F1</b>, to the second nearest <b>F2</b>. Coloring by the nearest point’s id gives cells; F2 − F1 is ~0 on the borders.',
    },
    {
      id: 'nature',
      label: 'Glass · Earth · Giraffe · Scales',
      kind: 'Real life',
      note: 'The same cells, dressed four ways: stained glass (lead came drawn where the border distance is small), cracked dry mud (warped cells, curled edges, secondary cracks), giraffe coat (thick cream borders), and dragon scales (a staggered lattice, each cell shaded as a little shiny dome).',
      params: { scale: 5, anim: 1 },
      hint: '',
    },
    {
      id: 'territory',
      label: 'Territory map',
      kind: 'In a game',
      note: 'Civilization-style borders: every pixel belongs to its nearest city (a Voronoi diagram over the city list), limited to a culture radius. Borders are only drawn between <i>different</i> nations, so a nation’s cities merge into one realm. <b>Click</b> to found a city for the selected nation, <b>right-click</b> to raze one. The city list lives in JavaScript and is sent to the shader as a uniform array.',
      hint: 'Click: found a city · right-click: remove · hexes toggle on the right.',
    },
  ],
  controls: [
    { type: 'slider', key: 'scale', label: 'Cells across', min: 2, max: 30, step: 0.1, value: 6, showFor: ['cells', 'nature'], help: 'Grid cells per screen height (one random point per cell).' },
    { type: 'slider', key: 'jitter', label: 'Jitter', min: 0, max: 1, step: 0.01, value: 1, showFor: ['cells'], help: '0 = every point in its cell’s center (a regular grid of squares). 1 = fully random.' },
    {
      type: 'select', key: 'viz', label: 'Visualize', value: 'cells', showFor: ['cells'],
      options: [{ value: 'cells', label: 'Cell id colors' }, { value: 'f1', label: 'F1 (nearest distance)' }, { value: 'f2', label: 'F2 (2nd nearest)' }, { value: 'f21', label: 'F2 − F1 (edges)' }, { value: 'border', label: 'Exact border distance' }],
      help: 'Contour lines show equal distance.',
    },
    {
      type: 'select', key: 'metric', label: 'Distance metric', value: 'euclid', showFor: ['cells'],
      options: [{ value: 'euclid', label: 'Euclidean (round)' }, { value: 'manhattan', label: 'Manhattan |x|+|y|' }, { value: 'cheb', label: 'Chebyshev max(|x|,|y|)' }],
      help: 'Changing how distance is measured changes the shape of the cells.',
    },
    { type: 'toggle', key: 'points', label: 'Show points & outlines', value: true, showFor: ['cells'] },
    { type: 'slider', key: 'anim', label: 'Animation speed', min: 0, max: 3, step: 0.01, value: 0.6, showFor: ['cells', 'nature'], help: 'Feature points orbit inside their cells.' },
    { type: 'select', key: 'team', label: 'Nation for new cities', value: 'red', showFor: ['territory'], options: TEAMS.map((t) => ({ value: t, label: t[0].toUpperCase() + t.slice(1) })) },
    { type: 'slider', key: 'culture', label: 'Culture radius', min: 0.05, max: 1, step: 0.005, value: 0.26, showFor: ['territory'], help: 'How far (× screen height) a city can claim land. Larger = borders meet.' },
    { type: 'slider', key: 'borderW', label: 'Border width (px)', min: 1, max: 8, step: 0.5, value: 3, showFor: ['territory'] },
    { type: 'toggle', key: 'hexes', label: 'Snap to hex tiles', value: false, showFor: ['territory'], help: 'Decide ownership per hex (at its center) — the Civilization V/VI look.' },
    { type: 'button', key: 'clear', label: 'Clear cities', showFor: ['territory'] },
    { type: 'button', key: 'defaults', label: 'Default cities', showFor: ['territory'] },
    { type: 'slider', key: 'seed', label: 'Seed', min: 0, max: 99, step: 1, value: 0 },
  ],
  uniforms: {
    scale: 'f32', jitter: 'f32', viz: 'f32', metric: 'f32', points: 'f32', anim: 'f32', team: 'f32', culture: 'f32', borderW: 'f32',
    hexes: 'f32', seed: 'f32', ncity: 'f32', cities: `array<vec4f, ${MAX_CITIES}>`,
  },
  include: ['noise', 'sdf', 'color'],
  onAction(key) {
    if (key === 'clear') cities = [];
    if (key === 'defaults' || key === 'reset') {
      cities = DEFAULT_CITIES.map((c, i) => ({ ...c, name: NAMES[i] }));
      nameCounter = cities.length;
    }
  },
  bind(params, ctx) {
    const ex = ctx.example;
    if (ex === 'territory') {
      handleMapInput(params, ctx);
      setLabels(ctx, cityLabels());
    } else if (ex === 'nature') {
      setLabels(ctx, quadLabels(['<b>Stained glass</b>', '<b>Cracked dry earth</b>', '<b>Giraffe coat</b>', '<b>Dragon scales</b>']));
    } else {
      const i = ['cells', 'f1', 'f2', 'f21', 'border'].indexOf(params.viz);
      setLabels(ctx, [{ text: VIZ_NAMES[Math.max(0, i)], style: 'left:8px;top:44px' }]);
    }
    cityData.fill(0);
    cities.forEach((c, i) => cityData.set([c.x, c.y, c.team, 1], i * 4));
    return { ncity: cities.length, cities: cityData };
  },
  code: CODE,
  about: {
    summary: 'A Voronoi diagram splits space into cells: every point belongs to the nearest “seed”. With random seeds it makes organic cells; with a list of cities it makes territories.',
    what: `<p><b>Cells & distance fields</b>: the raw diagram and its distance fields (F1, F2, F2−F1, border distance), with different distance metrics and the
      3×3 search visualised under the mouse. <b>Real life</b>: stained glass, cracked earth, giraffe coat and dragon scales. <b>In a game</b>: a territory map whose
      borders update instantly as you found or remove cities.</p>`,
    how: `<ol>
      <li>Divide space into a grid. In each grid cell, hash the cell coordinate to get <b>one random point</b> (jitter scales how far it can move from the center).</li>
      <li>For a pixel, only the points in the <b>3×3 surrounding cells</b> can be nearest — check those 9, keep the smallest distance (<b>F1</b>), the second smallest (<b>F2</b>) and the winner’s id.</li>
      <li><b>Cells</b>: color = hash(id). <b>Borders</b>: F2 − F1 ≈ 0 near an edge (cheap, approximate). For exactly even-width lines, a second pass measures the true
        distance to the bisector between the winner and each neighbour (Inigo Quilez’s method).</li>
      <li><b>Metric</b>: “distance” can be Euclidean (round cells), Manhattan (diamond-ish, axis-aligned cuts) or Chebyshev (blocky).</li>
      <li><b>Territory map</b>: no grid at all — just loop over the city list (up to ${MAX_CITIES}, sent as a uniform array). Each pixel finds its nearest city; the distance to the
        border with another nation is the distance to the perpendicular bisector: <code>(|p−b|² − |p−a|²) / (2|b−a|)</code>.</li>
    </ol>`,
    uses: [
      { title: 'Strategy maps', text: 'Territories and zones of control (Civilization, Stellaris, Europa Universalis-like borders), influence maps for AI.' },
      { title: 'Textures', text: 'Stone walls, cobblestones, cells, scales, leather, cracked ground, crystals, foam, caustics.' },
      { title: 'Destruction', text: 'Shatter glass or rocks into Voronoi fragments (common in physics-based breaking).' },
      { title: 'Level generation', text: 'Region graphs for world maps (Amit Patel’s polygon map generation), placing towns and biomes, nav meshes.' },
    ],
    try: [
      'Set <b>Jitter</b> to 0: the cells become a perfect square grid. Raise it slowly to see randomness grow.',
      'Switch the <b>Distance metric</b> to Manhattan or Chebyshev with <i>Visualize: F1</i> to see the iso-distance contours turn into diamonds and squares.',
      'Compare <i>F2 − F1</i> with <i>Exact border distance</i>: the exact one has equally wide contours everywhere.',
      'On <b>Territory map</b>, select Purple and click inside the blue nation: watch it lose land instantly. Then toggle <i>Snap to hex tiles</i>.',
      'Raise the <b>Culture radius</b> until all borders meet — that is the full Voronoi diagram of the cities.',
    ],
    ask: [
      'Voronoi / cellular noise texture (F1, F2, F2−F1)',
      'Civilization-style territory borders from a list of cities',
      'stained glass shader with lead lines',
      'cracked dry earth / mud texture',
      'shatter a sprite into Voronoi fragments',
      'influence map for strategy AI',
    ],
    perf: `<p>Grid Voronoi checks 9 cells per pixel (the exact border distance adds a 25-cell pass): cheap. The territory map loops over all cities for every pixel
      (32 × a few operations, ×2 for the border pass) — fine for dozens of seeds. For thousands of seeds, use a <a href="#/s/jump-flood">Jump Flood</a> pass
      instead: it builds the diagram in log₂(size) passes regardless of the seed count.</p>`,
    api: `<p>Works identically in WebGL2 and WebGPU — the city list is a uniform array (<code>array&lt;vec4f, ${MAX_CITIES}&gt;</code> / <code>vec4 cities[${MAX_CITIES}]</code>,
      same std140 layout). With WebGPU you could switch to a <b>storage buffer</b> to hold thousands of seeds, and compute the diagram once in a compute shader.</p>`,
    code: [
      {
        title: 'Voronoi: check the 3×3 neighbouring cells',
        lang: 'wgsl',
        src: `fn vor(p: vec2f, jit: f32, t: f32) -> Vor {
  let n = floor(p);  let f = fract(p);
  var v: Vor;  v.f1 = 8.0;  v.f2 = 8.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let g = vec2f(f32(i), f32(j));
      let o = fpoint(n + g, jit, t);          // the random point of that cell
      let d = vdist(g + o - f);               // euclidean / manhattan / chebyshev
      if (d < v.f1) { v.f2 = v.f1; v.f1 = d; v.id = n + g; }
      else if (d < v.f2) { v.f2 = d; }
    }
  }
  return v;
}`,
      },
      {
        title: 'Territory: nearest city, then distance to the border with other nations',
        lang: 'wgsl',
        src: `for (var i = 0; i < 32; i++) {
  if (f32(i) >= u.ncity) { break; }
  let c = u.cities[i];                       // xy = position, z = nation
  if (abs(c.z - team) > 0.5) {
    let cp = c.xy * u.resolution;
    // distance from p to the perpendicular bisector of (our city ca, their city cp)
    let dj = (dot(p - cp, p - cp) - d1 * d1) / (2.0 * length(cp - ca));
    bd = min(bd, dj);
  }
}`,
      },
      {
        title: 'JavaScript: the city list → a uniform array',
        lang: 'js',
        src: `bind(params, ctx) {
  if (ctx.pointer.clicked) cities.push({ x: ctx.pointer.nx, y: ctx.pointer.ny, team });
  cityData.fill(0);
  cities.forEach((c, i) => cityData.set([c.x, c.y, c.team, 1], i * 4));
  return { ncity: cities.length, cities: cityData };   // uniform array<vec4f, 32>
}`,
      },
    ],
    links: [
      { title: 'Inigo Quilez — Voronoi edges', url: 'https://iquilezles.org/articles/voronoilines/', note: 'exact border distance' },
      { title: 'Red Blob Games — Polygonal map generation', url: 'http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/', note: 'Voronoi-based world maps' },
      { title: 'The Book of Shaders — Cellular noise', url: 'https://thebookofshaders.com/12/' },
    ],
  },
});

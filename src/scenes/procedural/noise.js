import { shaderScene } from '../../core/shaderscene.js';
import { setLabels, quadLabels } from './_shared.js';

// Noise: Value vs Perlin vs Simplex vs Worley, natural textures made from noise, and
// "random jitter vs smooth noise" for motion in games. One portable-WGSL shader (WebGPU + WebGL2).

const CODE = /* wgsl */ `
fn accent(k: i32) -> vec3f {
  if (k == 0) { return vec3f(1.0, 0.66, 0.3); }
  if (k == 1) { return vec3f(0.38, 0.8, 1.0); }
  if (k == 2) { return vec3f(0.52, 0.95, 0.56); }
  return vec3f(1.0, 0.5, 0.78);
}

// grayscale-ish ramp with a hint of warmth for displaying a 0..1 noise value
fn ramp(n: f32) -> vec3f {
  let x = clamp(n, 0.0, 1.0);
  return mix(vec3f(0.03, 0.04, 0.08), vec3f(0.97, 0.95, 0.89), x * x * (3.0 - 2.0 * x) * 0.35 + x * 0.65);
}

// ---------------------------------------------------------------- 3D noises (z = time slice)
fn vnoise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  let a = hash31(i);
  let b = hash31(i + vec3f(1.0, 0.0, 0.0));
  let c = hash31(i + vec3f(0.0, 1.0, 0.0));
  let d = hash31(i + vec3f(1.0, 1.0, 0.0));
  let e = hash31(i + vec3f(0.0, 0.0, 1.0));
  let g = hash31(i + vec3f(1.0, 0.0, 1.0));
  let h = hash31(i + vec3f(0.0, 1.0, 1.0));
  let k = hash31(i + vec3f(1.0, 1.0, 1.0));
  return mix(mix(mix(a, b, w.x), mix(c, d, w.x), w.y), mix(mix(e, g, w.x), mix(h, k, w.x), w.y), w.z);
}
// unit-length random gradient for 3D simplex
fn sgrad3(i: vec3f) -> vec3f {
  let h = hash33(i);
  let z = h.x * 2.0 - 1.0;
  let a = h.y * TAU;
  let r = sqrt(max(1.0 - z * z, 0.0));
  return vec3f(r * cos(a), r * sin(a), z);
}
fn simplex3(v: vec3f) -> f32 {
  let i = floor(v + dot(v, vec3f(1.0 / 3.0)));
  let x0 = v - i + dot(i, vec3f(1.0 / 6.0));
  let g = step(x0.yzx, x0.xyz);
  let l = vec3f(1.0) - g;
  let i1 = min(g, l.zxy);
  let i2 = max(g, l.zxy);
  let x1 = x0 - i1 + 1.0 / 6.0;
  let x2 = x0 - i2 + 1.0 / 3.0;
  let x3 = x0 - 0.5;
  var m = max(vec4f(0.6) - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
  m = m * m;
  m = m * m;
  let d = vec4f(dot(x0, sgrad3(i)), dot(x1, sgrad3(i + i1)), dot(x2, sgrad3(i + i2)), dot(x3, sgrad3(i + vec3f(1.0))));
  return 38.0 * dot(m, d);
}
fn worley3(p: vec3f) -> f32 {
  let n = floor(p);
  let f = fract(p);
  var d1 = 8.0;
  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let g = vec3f(f32(i), f32(j), f32(k));
        let r = g + hash33(n + g) - f;
        d1 = min(d1, dot(r, r));
      }
    }
  }
  return sqrt(d1);
}

// Noise of kind k, normalised to 0..1. When animating we take a 2D slice of 3D noise at depth z.
fn noise01(k: i32, p: vec2f, z: f32) -> f32 {
  if (u.animate > 0.5) {
    let q = vec3f(p, z);
    if (k == 0) { return vnoise3(q); }
    if (k == 1) { return perlin3(q) * 0.5 + 0.5; }
    if (k == 2) { return simplex3(q) * 0.5 + 0.5; }
    return clamp(worley3(q), 0.0, 1.0);
  }
  if (k == 0) { return valueNoise(p); }
  if (k == 1) { return perlin(p) * 0.5 + 0.5; }
  if (k == 2) { return simplex(p) * 0.5 + 0.5; }
  return clamp(voronoi(p).x, 0.0, 1.0);
}

fn arrowTo(c: vec3f, p: vec2f, o: vec2f, g: vec2f, ppu: f32, col: vec3f) -> vec3f {
  let tip = o + g * 0.42;
  let sd = vec2f(-g.y, g.x);
  let d0 = sdSegment(p, o, tip);
  let d1 = sdSegment(p, tip, tip - g * 0.13 + sd * 0.08);
  let d2 = sdSegment(p, tip, tip - g * 0.13 - sd * 0.08);
  let d = min(d0, min(d1, d2)) * ppu;
  var r = mix(c, vec3f(0.0), (1.0 - smoothstep(1.0, 3.5, d)) * 0.55);
  r = mix(r, col, 1.0 - smoothstep(0.7, 1.5, d));
  let dd = length(p - o) * ppu;
  r = mix(r, vec3f(0.0), 1.0 - smoothstep(4.0, 5.0, dd));
  r = mix(r, col, 1.0 - smoothstep(2.5, 3.5, dd));
  return r;
}

fn gridLines(c: vec3f, p: vec2f, ppu: f32, col: vec3f, a: f32) -> vec3f {
  let g = abs(fract(p + 0.5) - 0.5) * ppu;
  return mix(c, col, (1.0 - smoothstep(0.4, 1.3, min(g.x, g.y))) * a);
}

// Draw the lattice each noise is built on: random values (value), random gradients (Perlin),
// the skewed triangle grid (simplex) and the random feature points (Worley).
fn latticeOverlay(c0: vec3f, k: i32, p: vec2f, ppu: f32) -> vec3f {
  let acc = accent(k);
  var c = c0;
  if (k == 2) {
    let K1 = 0.366025404;
    let K2 = 0.211324865;
    let s = p + (p.x + p.y) * K1;
    let dd = s.x - s.y;
    let lx = abs(fract(s.x + 0.5) - 0.5) / fwidth(s.x);
    let ly = abs(fract(s.y + 0.5) - 0.5) / fwidth(s.y);
    let ld = abs(fract(dd + 0.5) - 0.5) / fwidth(dd);
    c = mix(c, acc, (1.0 - smoothstep(0.4, 1.3, min(lx, min(ly, ld)))) * 0.6);
    if (u.animate < 0.5) {
      let i = floor(s);
      let a = p - i + (i.x + i.y) * K2;
      let m = step(a.y, a.x);
      let o = vec2f(m, 1.0 - m);
      let i1 = i + o;
      let i2 = i + vec2f(1.0);
      c = arrowTo(c, p, i - (i.x + i.y) * K2, grad2(i), ppu, acc);
      c = arrowTo(c, p, i1 - (i1.x + i1.y) * K2, grad2(i1), ppu, acc);
      c = arrowTo(c, p, i2 - (i2.x + i2.y) * K2, grad2(i2), ppu, acc);
    }
    return c;
  }
  c = gridLines(c, p, ppu, acc, 0.55);
  if (u.animate > 0.5) { return c; }
  let ip = floor(p + 0.5);
  if (k == 0) {
    let d = length(p - ip) * ppu;
    c = mix(c, acc, 1.0 - smoothstep(6.0, 7.0, d));
    c = mix(c, ramp(hash21(ip)), 1.0 - smoothstep(4.0, 5.0, d));
  } else if (k == 1) {
    c = arrowTo(c, p, ip, grad2(ip), ppu, acc);
  } else {
    let v = voronoi(p);
    let d = v.x * ppu;
    c = mix(c, vec3f(0.0), 1.0 - smoothstep(4.5, 5.5, d));
    c = mix(c, acc, 1.0 - smoothstep(3.0, 4.0, d));
    c = mix(c, acc, (1.0 - smoothstep(0.0, 1.5, (v.y - v.x) * ppu)) * 0.5);
  }
  return c;
}

// ---------------------------------------------------------------- example 0: comparison
fn exCompare(px: vec2f) -> vec3f {
  let hs = u.resolution * 0.5;
  let qd = vec2f(step(hs.x, px.x), step(hs.y, px.y));
  let k = i32(qd.x + 2.0 * qd.y);
  let lp = px - qd * hs;
  let ppu = hs.y / u.freq;
  let off = vec2f(u.seed * 31.7, u.seed * 17.3);
  let z = u.seed * 7.0 + u.time * u.speed * 0.4;
  let graphTop = hs.y * 0.7;
  var sliceY = hs.y * 0.38;
  if (u.mouse.w > 0.5) { sliceY = clamp(fmod(u.mouse.y, hs.y), 8.0, graphTop - 8.0); }
  let acc = accent(k);
  var col = vec3f(0.0);
  if (lp.y < graphTop) {
    let p = lp / ppu + off;
    col = ramp(noise01(k, p, z));
    if (u.lattice > 0.5) { col = latticeOverlay(col, k, p, ppu); }
    let dash = step(0.45, fract(lp.x / 16.0));
    col = mix(col, acc, (1.0 - smoothstep(0.6, 1.6, abs(lp.y - sliceY))) * (0.25 + 0.75 * dash));
    col = mix(col, col * 0.55, smoothstep(graphTop - 18.0, graphTop, lp.y));
  } else {
    let gy0 = graphTop + 12.0;
    let gy1 = hs.y - 12.0;
    col = mix(vec3f(0.05, 0.055, 0.08), vec3f(0.025, 0.03, 0.045), (lp.y - graphTop) / (hs.y - graphTop));
    // guides at 0, 0.5 and 1
    let gm = min(abs(lp.y - gy0), min(abs(lp.y - gy1), abs(lp.y - 0.5 * (gy0 + gy1))));
    col = mix(col, vec3f(0.2, 0.22, 0.3), (1.0 - smoothstep(0.3, 1.0, gm)) * 0.6);
    let ps = vec2f(lp.x, sliceY) / ppu + off;
    // lattice ticks: integer x in noise space
    if (u.lattice > 0.5 && k != 2) {
      let tx = abs(fract(ps.x + 0.5) - 0.5) * ppu;
      col = mix(col, acc * 0.5, (1.0 - smoothstep(0.3, 1.2, tx)) * step(gy0, lp.y) * step(lp.y, gy1));
    }
    let e = 1.5 / ppu;
    let ya = mix(gy1, gy0, noise01(k, ps - vec2f(e, 0.0), z));
    let yb = mix(gy1, gy0, noise01(k, ps, z));
    let yc = mix(gy1, gy0, noise01(k, ps + vec2f(e, 0.0), z));
    let d = min(sdSegment(lp, vec2f(lp.x - 1.5, ya), vec2f(lp.x, yb)), sdSegment(lp, vec2f(lp.x, yb), vec2f(lp.x + 1.5, yc)));
    let under = step(yb, lp.y) * step(lp.y, gy1);
    col = mix(col, acc * 0.35, under * (0.25 + 0.4 * (lp.y - yb) / max(gy1 - yb, 1.0)));
    col = mix(col, acc * 0.25, (1.0 - smoothstep(1.5, 6.0, d)) * 0.6);
    col = mix(col, mix(acc, vec3f(1.0), 0.25), 1.0 - smoothstep(0.7, 1.7, d));
  }
  let edge = min(abs(px.x - hs.x), abs(px.y - hs.y));
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, edge));
  return col;
}

// ---------------------------------------------------------------- example 1: natural textures
fn turb(p: vec2f, oct: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 8; i++) {
    if (i >= oct) { break; }
    s += a * abs(perlin(q));
    q = FBM_ROT * q * 2.0 + vec2f(5.3, 1.7);
    a *= 0.5;
  }
  return s;
}

fn wood(lp: vec2f, sc: f32, off: vec2f, oct: i32) -> vec3f {
  let boards = 3.0;
  let bi = floor(lp.y * boards);
  let by = fract(lp.y * boards);
  let rnd = hash22(vec2f(bi, u.seed));
  // staggered end joints between boards
  let len = 1.15;
  let jx = lp.x / len + rnd.x;
  let piece = floor(jx);
  let rnd2 = hash22(vec2f(bi * 7.0 + piece, u.seed + 3.0));
  let q = vec2f(lp.x * sc * 0.5, (by - 0.5)) + off * 0.1 + rnd2 * 40.0;
  // distance to the log's axis: rings are circles around it, seen from the side
  let wob = fbm(vec2f(q.x * 0.7, rnd2.y * 9.0), 3) * 0.6 * u.warp;
  let yy = (by - 0.5 + (rnd2.y - 0.5) * 0.8 + wob) * 1.6;
  let depth = 0.15 + rnd2.x * 0.5;
  let r = sqrt(yy * yy + depth * depth);
  let ring = fract(r * sc * 2.2 + fbm(q * vec2f(1.0, 6.0), oct) * 0.25 * u.warp);
  let late = smoothstep(0.55, 0.92, ring) * (1.0 - smoothstep(0.92, 1.0, ring));
  var c = mix(vec3f(0.78, 0.53, 0.3), vec3f(0.46, 0.25, 0.12), late);
  // fine grain streaks along the board
  let grain = perlin(vec2f(q.x * 6.0, by * 90.0 + rnd2.y * 20.0));
  c *= 0.9 + 0.1 * grain;
  c = mix(c, c * vec3f(1.05, 0.95, 0.9), rnd.y * 0.5);
  // seams between boards and between pieces
  let seamY = min(by, 1.0 - by) * u.resolution.y * 0.5 / boards;
  let seamX = min(fract(jx), 1.0 - fract(jx)) * len * u.resolution.y * 0.5;
  let seam = min(seamY, seamX);
  c *= 0.35 + 0.65 * smoothstep(0.5, 2.5, seam);
  c *= 0.85 + 0.15 * smoothstep(0.0, 0.5, by);
  return c;
}

fn marble(lp: vec2f, sc: f32, off: vec2f, oct: i32) -> vec3f {
  let p = lp * sc * 0.9 + off;
  let tb = turb(p, oct);
  let v = sin((p.x * 0.8 + p.y * 0.6) * 2.2 + tb * u.warp * 7.0);
  let vein = 1.0 - smoothstep(0.0, 0.16, abs(v));
  let thin = 1.0 - smoothstep(0.0, 0.05, abs(sin(p.y * 1.3 - p.x * 0.4 + turb(p * 1.7 + 9.0, oct) * u.warp * 9.0)));
  let cloud = fbm(p * 0.7 + 3.0, oct) * 0.5 + 0.5;
  var c = mix(vec3f(0.93, 0.92, 0.9), vec3f(0.82, 0.83, 0.86), cloud);
  c = mix(c, vec3f(0.42, 0.44, 0.5), vein * 0.75);
  c = mix(c, vec3f(0.55, 0.5, 0.42), thin * 0.45);
  // polished highlight
  c += vec3f(0.07) * smoothstep(0.3, 0.0, abs(lp.x - lp.y * 0.6 - 0.35));
  return c;
}

fn clouds(lp: vec2f, sc: f32, off: vec2f, oct: i32, t: f32) -> vec3f {
  let wind = vec2f(t * 0.05, 0.0);
  let p = lp * sc * 0.55 + off + wind;
  let warp = vec2f(fbm(p * 0.5 + t * 0.02, 3), fbm(p * 0.5 + 7.3, 3)) * 0.35 * u.warp;
  let n = fbm(p + warp, oct) * 0.5 + 0.5;
  let n2 = fbm(p + warp + vec2f(-0.06, -0.08), oct) * 0.5 + 0.5;
  let dens = smoothstep(0.47, 0.75, n);
  let lit = clamp(0.62 + (n2 - n) * 7.0, 0.35, 1.05);
  let sky = mix(vec3f(0.22, 0.45, 0.82), vec3f(0.62, 0.8, 0.96), lp.y);
  let cl = mix(vec3f(0.55, 0.6, 0.72), vec3f(1.0, 0.99, 0.96), lit);
  return mix(sky, cl, dens);
}

fn caust(p: vec2f, t: f32) -> f32 {
  let w = p + 0.18 * vec2f(perlin(p * 0.7 + vec2f(t * 0.3, 0.0)), perlin(p * 0.7 + vec2f(5.2, 1.3 - t * 0.3)));
  let a = voronoiEx(w, 1.0, t * 1.1);
  let b = voronoiEx(w * 1.6 + vec2f(3.1, 7.7), 1.0, -t * 0.9);
  let la = pow(1.0 - clamp(a.y - a.x, 0.0, 1.0), 9.0);
  let lb = pow(1.0 - clamp(b.y - b.x, 0.0, 1.0), 9.0);
  return la * 0.75 + lb * 0.5;
}

fn caustics(lp: vec2f, sc: f32, off: vec2f, t: f32) -> vec3f {
  // pool floor tiles
  let tp = lp * 9.0;
  let tf = abs(fract(tp) - 0.5);
  let grout = smoothstep(0.43, 0.47, max(tf.x, tf.y));
  let th = hash21(floor(tp));
  var floorc = mix(vec3f(0.42, 0.72, 0.8), vec3f(0.36, 0.64, 0.74), th);
  floorc = mix(floorc, vec3f(0.25, 0.45, 0.52), grout);
  let p = lp * sc * 0.55 + off;
  let e = 0.012;
  let cr = caust(p + vec2f(e, 0.0), t);
  let cg = caust(p, t);
  let cb = caust(p - vec2f(e, 0.0), t);
  var c = floorc * vec3f(0.55, 0.75, 0.8) + vec3f(cr, cg, cb) * vec3f(0.9, 1.0, 1.0) * 0.75;
  c = mix(c, vec3f(0.05, 0.25, 0.35), 0.18);
  return c;
}

fn exTextures(px: vec2f) -> vec3f {
  let hs = u.resolution * 0.5;
  let qd = vec2f(step(hs.x, px.x), step(hs.y, px.y));
  let k = i32(qd.x + 2.0 * qd.y);
  let lp = (px - qd * hs) / hs.y;
  let sc = u.freq * 0.5;
  let oct = i32(u.detail);
  let off = vec2f(u.seed * 31.7, u.seed * 17.3);
  let t = u.time * u.speed;
  var col = vec3f(0.0);
  if (k == 0) { col = wood(lp, sc, off, oct); }
  else if (k == 1) { col = marble(lp, sc, off, oct); }
  else if (k == 2) { col = clouds(lp, sc, off, oct, t); }
  else { col = caustics(lp, sc, off, t); }
  let edge = min(abs(px.x - hs.x), abs(px.y - hs.y));
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, edge));
  return col;
}

// ---------------------------------------------------------------- example 2: noise-driven motion
// side 0: a NEW random number every 1/rate seconds (what Math.random() per frame gives you)
// side 1: smooth noise sampled along time
fn sig(side: f32, t: f32, ch: f32) -> f32 {
  if (side < 0.5) { return hash21(vec2f(floor(t * u.rate), ch * 13.7)) * 2.0 - 1.0; }
  return clamp(perlin(vec2f(t * u.noiseFreq, ch * 7.31 + 0.5)) * 1.45, -1.0, 1.0);
}

fn dungeon(sp: vec2f, side: f32, t: f32) -> vec3f {
  // brick wall
  let bp = sp * vec2f(7.0, 14.0);
  let row = floor(bp.y);
  let bx = bp.x + 0.5 * fmod(row, 2.0);
  let cell = vec2f(floor(bx), row);
  let f = vec2f(fract(bx), fract(bp.y));
  let mortar = min(min(f.x, 1.0 - f.x) * 2.0, min(f.y, 1.0 - f.y));
  let bh = hash21(cell);
  var wall = mix(vec3f(0.36, 0.3, 0.3), vec3f(0.46, 0.38, 0.34), bh);
  wall *= 0.85 + 0.25 * valueNoise(sp * 60.0);
  wall *= 0.35 + 0.65 * smoothstep(0.02, 0.09, mortar);
  // floor
  let fl = smoothstep(0.3, 0.31, sp.y);
  wall = mix(wall, vec3f(0.22, 0.2, 0.22) * (0.8 + 0.2 * valueNoise(sp * vec2f(8.0, 30.0))), fl);
  // torch light with flicker
  let flick = 0.8 + 0.2 * sig(side, t, 3.0);
  let tp = vec2f(0.0, -0.08);
  let d = length(sp - tp);
  let light = flick * 1.25 / (1.0 + 18.0 * d * d / (flick * flick));
  var c = wall * (vec3f(0.06, 0.07, 0.12) + vec3f(1.0, 0.62, 0.3) * light);
  // torch: handle + flame
  let hd = sdBox(sp - vec2f(0.0, 0.0), vec2f(0.012, 0.07));
  c = mix(c, vec3f(0.25, 0.15, 0.08), 1.0 - smoothstep(0.0, 0.004, hd));
  let fq = (sp - tp) * vec2f(1.0, 0.65) / (0.9 + 0.25 * flick);
  let fd = length(fq + vec2f(0.0, 0.02)) - 0.035;
  let flame = 1.0 - smoothstep(-0.01, 0.012, fd);
  c = mix(c, mix(vec3f(1.0, 0.45, 0.1), vec3f(1.0, 0.92, 0.6), smoothstep(0.0, -0.03, fd)), flame);
  c += vec3f(1.0, 0.55, 0.2) * exp(-d * 9.0) * 0.35 * flick;
  // fireflies
  for (var i = 0; i < 7; i++) {
    let fi = f32(i);
    let base = vec2f((fi - 3.0) * 0.17, -0.22 + 0.1 * sin(fi * 2.3));
    let o = vec2f(sig(side, t * 0.6, 10.0 + fi), sig(side, t * 0.6, 30.0 + fi)) * vec2f(0.1, 0.08);
    let fp = base + o;
    let fd2 = length(sp - fp);
    let blink = 0.65 + 0.35 * sig(side, t, 50.0 + fi);
    c += vec3f(0.75, 1.0, 0.35) * (exp(-fd2 * 160.0) * 1.4 + exp(-fd2 * 30.0) * 0.25) * blink;
  }
  return c;
}

fn exMotion(px: vec2f) -> vec3f {
  let W = u.resolution.x * 0.5;
  let H = u.resolution.y;
  let side = step(W, px.x);
  let lp = vec2f(px.x - side * W, px.y);
  let sceneH = H * 0.7;
  let t = u.time * u.speed;
  var col = vec3f(0.0);
  if (lp.y < sceneH) {
    let shake = vec2f(sig(side, t, 1.0), sig(side, t, 2.0)) * u.shake * 0.04;
    let sp = (lp - vec2f(W * 0.5, sceneH * 0.5)) / sceneH + shake;
    col = dungeon(sp, side, t);
    // vignette per panel
    let vq = (lp - vec2f(W * 0.5, sceneH * 0.5)) / vec2f(W, sceneH);
    col *= 1.0 - 0.6 * dot(vq, vq);
  } else {
    // signal graph: the torch flicker channel over the last 3 seconds
    let gy0 = sceneH + 18.0;
    let gy1 = H - 14.0;
    col = vec3f(0.035, 0.04, 0.055);
    let gm = min(abs(lp.y - gy0), min(abs(lp.y - gy1), abs(lp.y - 0.5 * (gy0 + gy1))));
    col = mix(col, vec3f(0.18, 0.2, 0.27), (1.0 - smoothstep(0.3, 1.0, gm)) * 0.6);
    let span = 3.0;
    let x0 = W - 16.0;
    let tt = t - (x0 - lp.x) / x0 * span;
    let e = 1.5 / x0 * span;
    let a = 0.5 + 0.5 * sig(side, tt - e, 3.0);
    let b = 0.5 + 0.5 * sig(side, tt, 3.0);
    let c2 = 0.5 + 0.5 * sig(side, tt + e, 3.0);
    let ya = mix(gy1, gy0, a);
    let yb = mix(gy1, gy0, b);
    let yc = mix(gy1, gy0, c2);
    let d = min(sdSegment(lp, vec2f(lp.x - 1.5, ya), vec2f(lp.x, yb)), sdSegment(lp, vec2f(lp.x, yb), vec2f(lp.x + 1.5, yc)));
    var acc = vec3f(1.0, 0.45, 0.4);
    if (side > 0.5) { acc = vec3f(0.45, 0.95, 0.6); }
    let live = step(lp.x, x0);
    col = mix(col, acc * 0.3, step(yb, lp.y) * step(lp.y, gy1) * 0.35 * live);
    col = mix(col, acc, (1.0 - smoothstep(0.7, 1.7, d)) * live);
    // "now" marker
    let yNow = mix(gy1, gy0, 0.5 + 0.5 * sig(side, t, 3.0));
    let dn = length(lp - vec2f(x0, yNow));
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(3.0, 4.5, dn));
  }
  let edge = min(abs(px.x - W), abs(lp.y - sceneH));
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, edge));
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = exCompare(px); }
  else if (ex == 1) { col = exTextures(px); }
  else { col = exMotion(px); }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: 'Move the mouse over a noise panel: the dashed line picks the row that is plotted as a 1D graph below it.',
  examples: [
    {
      id: 'compare',
      label: 'Value · Perlin · Simplex · Worley',
      kind: 'Comparison',
      note: 'Four classic noise functions side by side. Each graph is a <b>1D slice</b> along the dashed line: notice how value noise has blocky bumps that peak <i>on</i> grid points, Perlin and simplex are rounder, and Worley measures the distance to scattered points. Turn on <i>Show the lattice</i> to see the hidden grid each one is built from.',
    },
    {
      id: 'textures',
      label: 'Wood · Marble · Clouds · Caustics',
      kind: 'Real life',
      note: 'Natural materials are mostly “noise + a shaping function”. Wood = rings (distance to the log axis) wobbled by noise. Marble = <code>sin(x + turbulence)</code>. Clouds = fBm through a soft threshold. Pool caustics = two layers of moving Worley cell edges.',
      params: { freq: 4, warp: 1, detail: 5, speed: 1 },
    },
    {
      id: 'motion',
      label: 'Jitter vs smooth noise',
      kind: 'In a game',
      note: 'Noise isn’t only for pictures — sample it along <b>time</b> to drive motion. Left: a fresh random number every tick (camera shake, torch flicker and fireflies all jitter and teleport). Right: the same effects driven by smooth noise — organic and pleasant. The graphs show the torch brightness over the last 3 seconds.',
      params: { speed: 1 },
    },
  ],
  controls: [
    { type: 'slider', key: 'freq', label: 'Frequency', min: 1, max: 24, step: 0.1, value: 5, showFor: ['compare', 'textures'], help: 'How many noise cells fit across a panel. Higher = smaller features.' },
    { type: 'slider', key: 'seed', label: 'Seed', min: 0, max: 99, step: 1, value: 0, help: 'A different seed = a different (but repeatable) random world. Here it simply offsets into the infinite noise field.' },
    { type: 'toggle', key: 'animate', label: 'Animate (slide through 3D noise)', value: true, showFor: ['compare'], help: 'Uses 3D noise and moves the z slice with time, so the 2D image evolves smoothly instead of scrolling.' },
    { type: 'toggle', key: 'lattice', label: 'Show the lattice', value: false, showFor: ['compare'], help: 'Grid, random values (value), gradient arrows (Perlin, simplex) and feature points (Worley). Dots & arrows appear when Animate is off.' },
    { type: 'slider', key: 'detail', label: 'Detail (octaves)', min: 1, max: 8, step: 1, value: 5, showFor: ['textures'], help: 'Layers of noise summed together (fBm). See the next scene for the details.' },
    { type: 'slider', key: 'warp', label: 'Turbulence', min: 0, max: 2, step: 0.01, value: 1, showFor: ['textures'], help: 'How strongly noise bends the rings, veins and clouds.' },
    { type: 'slider', key: 'shake', label: 'Camera shake', min: 0, max: 1, step: 0.01, value: 0.35, showFor: ['motion'], help: 'Shake offset = shake × (signal X, signal Y).' },
    { type: 'slider', key: 'rate', label: 'Random updates / second', min: 2, max: 60, step: 1, value: 20, showFor: ['motion'], help: 'Left side: how often a new random value is picked. Even slow updates look like jumps.' },
    { type: 'slider', key: 'noiseFreq', label: 'Noise frequency (Hz)', min: 0.2, max: 8, step: 0.05, value: 1.6, showFor: ['motion'], help: 'Right side: how fast we move along the noise. High values start to look jittery too.' },
    { type: 'slider', key: 'speed', label: 'Time speed', min: 0, max: 3, step: 0.01, value: 1, help: 'Multiplies time for all animation.' },
  ],
  uniforms: { freq: 'f32', seed: 'f32', animate: 'f32', lattice: 'f32', detail: 'f32', warp: 'f32', shake: 'f32', rate: 'f32', noiseFreq: 'f32', speed: 'f32' },
  include: ['noise', 'sdf'],
  bind(params, ctx) {
    if (ctx.example === 'compare') {
      setLabels(ctx, quadLabels(['<b>Value noise</b> · random values, blended', '<b>Perlin noise</b> · random gradients', '<b>Simplex noise</b> · triangle grid', '<b>Worley noise</b> · distance to points']));
    } else if (ctx.example === 'textures') {
      setLabels(ctx, quadLabels(['<b>Wood</b> · rings + noise wobble', '<b>Marble</b> · sin(x + turbulence)', '<b>Clouds</b> · fBm + soft threshold', '<b>Caustics</b> · Worley cell edges']));
    } else {
      setLabels(ctx, [
        { text: '<b>random()</b> every tick — jittery', style: 'left:8px;top:8px' },
        { text: '<b>noise(time)</b> — smooth & organic', style: 'left:calc(50% + 8px);top:8px' },
        { text: 'torch brightness, last 3 s', style: 'left:calc(50% - 175px);top:calc(70% + 4px);opacity:.75' },
      ]);
    }
    return {};
  },
  code: CODE,
  about: {
    summary: 'Noise is “random, but smooth”: a function that returns the same value for the same input, changes gradually, and never repeats. It is the raw material for terrain, clouds, fire, water and organic motion.',
    what: `<p><b>Comparison</b>: the four classic noise functions, each with a 1D cross-section graph. <b>Real life</b>: wood, marble, clouds and pool
      caustics, each built from noise plus a small shaping formula. <b>In a game</b>: the same idea used over <i>time</i> for camera shake,
      flickering torch light and wandering fireflies — compared against plain random numbers.</p>`,
    how: `<ol>
      <li><b>Hash</b>: everything starts with a hash — a function that scrambles a grid coordinate into a repeatable pseudo-random number.
        (<code>Math.random()</code> can’t be used: a shader needs the <i>same</i> answer every time it asks about the same spot.)</li>
      <li><b>Value noise</b>: put a random value on every integer grid point and blend between the 4 neighbours with a smooth curve.
        Cheap, but the grid shows (blobs line up, extremes sit on grid points).</li>
      <li><b>Perlin (gradient) noise</b>: put a random <i>direction</i> (gradient) on every grid point. Each pixel takes the dot product of the gradient with
        its offset from that point, then blends. The value is 0 at every grid point and the features are rounder and less “blocky”.</li>
      <li><b>Simplex noise</b>: Ken Perlin’s improvement: a triangle grid (in 2D) so only 3 corners are needed instead of 4 (4 instead of 8 in 3D) —
        cheaper in higher dimensions and with fewer directional artifacts.</li>
      <li><b>Worley (cellular) noise</b>: scatter one random point per grid cell; the value is the distance to the nearest point. Gives cells, scales,
        stones and caustics (see the <a href="#/s/voronoi">Voronoi</a> scene).</li>
      <li><b>Animating</b>: use 3D noise and treat time as the third axis — the 2D image is a slice that slides through a 3D noise block.</li>
      <li><b>Motion</b>: sample noise along time (<code>noise(t × frequency, channel)</code>) to get a smooth random signal. Use a different “channel”
        (y offset) for each property: shake X, shake Y, light intensity, each firefly…</li>
    </ol>`,
    uses: [
      { title: 'Textures without image files', text: 'Wood, stone, marble, clouds, grass variation, dirt and rust — generated at any resolution, infinitely varied.' },
      { title: 'Terrain & worlds', text: 'Height maps and biomes (Minecraft, Terraria, No Man’s Sky, Dwarf Fortress all lean on noise).' },
      { title: 'Organic motion', text: 'Camera shake (trauma-based shake uses Perlin noise), idle sway, wandering creatures, flickering fires and lights.' },
      { title: 'VFX', text: 'Fire, smoke, dissolve effects, water surfaces, magic auras, heat haze distortion.' },
    ],
    try: [
      'Turn on <b>Show the lattice</b> and turn <b>Animate</b> off: value noise peaks on the dots, while Perlin passes through the middle grey (zero) at every grid point.',
      'Drag <b>Frequency</b> down to 1–2 to see the shape of a single noise cell, then up to 20 to see why tiny features look like TV static.',
      'On <b>Jitter vs smooth noise</b>, set <i>Random updates / second</i> to 4: still looks like teleporting. Then push <i>Noise frequency</i> to 8 Hz to see smooth noise become jittery.',
      'On <b>Wood · Marble…</b> set <i>Turbulence</i> to 0: wood becomes perfect stripes and marble becomes a plain sine wave — the noise is what makes it natural.',
      'Change the <b>Seed</b> — every value is a whole new, but repeatable, pattern. Save the seed, get the same world back.',
    ],
    ask: [
      'Perlin noise based camera shake (trauma system)',
      'procedural wood / marble / cloud textures from noise',
      'animated 3D noise (time as the third dimension)',
      'Worley / cellular noise for caustics and cells',
      'seeded noise so the same seed gives the same world',
      'smooth noise instead of random jitter for flickering lights',
    ],
    perf: `<p>Value noise is a handful of operations; Perlin ~2× that; 2D simplex is similar to Perlin but scales much better to 3D and 4D;
      Worley checks 9 cells in 2D (27 in 3D). All are cheap enough to evaluate per pixel every frame — the cost explodes only when you stack
      many octaves (see fBm) or evaluate many layers. For static textures, generate once into a texture.</p>`,
    api: `<p>Identical in WebGPU and WebGL2: these are pure fragment-shader math. One portability note: the hash used here relies on <b>integer bit
      operations</b> (PCG hash via <code>bitcast</code>), which both WebGL2 (GLSL ES 3.0) and WebGPU support. Old WebGL1 shaders used
      <code>fract(sin(x) × 43758.5)</code>, which gives different results on different GPUs.</p>`,
    code: [
      {
        title: 'Value noise vs Perlin noise (2D)',
        lang: 'wgsl',
        src: `fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);            // smooth blend curve
  let a = hash21(i);                  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0)); let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);   // random VALUES at corners
}
fn perlin(p: vec2f) -> f32 {
  let i = floor(p);  let f = fract(p);
  let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic fade
  let va = dot(grad2(i), f);                          // random GRADIENTS at corners
  let vb = dot(grad2(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0));
  let vc = dot(grad2(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0));
  let vd = dot(grad2(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0));
  return mix(mix(va, vb, w.x), mix(vc, vd, w.x), w.y) * 1.4142;
}`,
      },
      {
        title: 'Smooth vs jittery motion signals',
        lang: 'wgsl',
        src: `// side 0: a new random value every 1/rate seconds
// side 1: smooth noise sampled along time (channel = which property)
fn sig(side: f32, t: f32, ch: f32) -> f32 {
  if (side < 0.5) { return hash21(vec2f(floor(t * u.rate), ch * 13.7)) * 2.0 - 1.0; }
  return clamp(perlin(vec2f(t * u.noiseFreq, ch * 7.31 + 0.5)) * 1.45, -1.0, 1.0);
}
let shake = vec2f(sig(side, t, 1.0), sig(side, t, 2.0)) * u.shake * 0.04;
let flicker = 0.8 + 0.2 * sig(side, t, 3.0);`,
      },
      {
        title: 'Marble and wood in two lines each',
        lang: 'wgsl',
        src: `// marble: a sine wave whose phase is disturbed by turbulence (sum of |noise|)
let v = sin((p.x * 0.8 + p.y * 0.6) * 2.2 + turb(p, oct) * u.warp * 7.0);
let vein = 1.0 - smoothstep(0.0, 0.16, abs(v));
// wood: rings = distance to the log axis, wobbled by fBm
let r = sqrt(yy * yy + depth * depth);
let ring = fract(r * sc * 2.2 + fbm(q * vec2f(1.0, 6.0), oct) * 0.25 * u.warp);`,
      },
    ],
    links: [
      { title: 'The Book of Shaders — Noise', url: 'https://thebookofshaders.com/11/', note: 'gentle visual introduction' },
      { title: 'Inigo Quilez — articles', url: 'https://iquilezles.org/articles/', note: 'value noise derivatives, Voronoi, warping' },
      { title: 'Red Blob Games — Making maps with noise', url: 'https://www.redblobgames.com/maps/terrain-from-noise/', note: 'noise for game worlds, interactive' },
    ],
  },
});

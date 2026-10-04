import { shaderScene } from '../../core/shaderscene.js';
import { setLabels } from './_shared.js';

// fBm (fractal Brownian motion) and domain warping: octaves visualised one by one, ridged and
// turbulence variants, IQ's f(p + f(p + f(p))) warping, a gas giant and an ink/portal pair.
// One portable-WGSL shader (WebGPU + WebGL2).

const CODE = /* wgsl */ `
// ------------------------------------------------------------------ configurable fBm
// One octave's noise, shaped by the variant: 0 fBm (signed), 1 turbulence |n|, 2 ridged (1-|n|)^2
fn shapeOct(n: f32) -> f32 {
  if (u.variant > 1.5) { let r = 1.0 - abs(n); return r * r; }
  if (u.variant > 0.5) { return abs(n); }
  return n;
}

fn fbmV(p: vec2f) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  var prev = 1.0;
  let oc = i32(u.octaves);
  for (var i = 0; i < 10; i++) {
    if (i >= oc) { break; }
    let n = shapeOct(perlin(q));
    if (u.variant > 1.5) {
      // ridged multifractal: each octave is weighted by the previous one -> detail gathers on ridges
      sum += amp * n * prev;
      prev = clamp(n * 1.6, 0.0, 1.0);
    } else {
      sum += amp * n;
    }
    q = FBM_ROT * q * u.lacunarity + vec2f(17.13, 3.71);
    amp *= u.gain;
  }
  return sum;
}

fn ampSum() -> f32 {
  var s = 0.0;
  var a = 0.5;
  let oc = i32(u.octaves);
  for (var i = 0; i < 10; i++) {
    if (i >= oc) { break; }
    s += a;
    a *= u.gain;
  }
  return s;
}

// fBm remapped to 0..1 for display
fn fbm01(p: vec2f, A: f32) -> f32 {
  let s = fbmV(p);
  if (u.variant > 1.5) { return clamp(s / (A * 0.8), 0.0, 1.0); }
  if (u.variant > 0.5) { return clamp(s / (A * 0.62), 0.0, 1.0); }
  return clamp(0.5 + s / (A * 1.25), 0.0, 1.0);
}

// position of octave i: rotate + scale by lacunarity^i
fn octPos(p: vec2f, i: i32) -> vec2f {
  var q = p;
  for (var k = 0; k < 10; k++) {
    if (k >= i) { break; }
    q = FBM_ROT * q * u.lacunarity + vec2f(17.13, 3.71);
  }
  return q;
}

fn variantRamp(v0: f32) -> vec3f {
  let v = clamp(v0, 0.0, 1.0);
  if (u.variant > 1.5) {
    // ridged: slate valleys -> lilac -> white ridges
    let c = mix(vec3f(0.05, 0.06, 0.12), vec3f(0.42, 0.42, 0.66), smoothstep(0.0, 0.6, v));
    return mix(c, vec3f(1.0, 0.97, 0.94), smoothstep(0.55, 1.0, v));
  }
  if (u.variant > 0.5) {
    // turbulence: fire
    let c = mix(vec3f(0.02, 0.0, 0.02), vec3f(0.7, 0.08, 0.04), smoothstep(0.0, 0.45, v));
    return mix(c, vec3f(1.0, 0.85, 0.35), smoothstep(0.4, 1.0, v));
  }
  // fBm: deep blue -> teal -> warm white
  let c = mix(vec3f(0.03, 0.05, 0.13), vec3f(0.12, 0.46, 0.55), smoothstep(0.15, 0.55, v));
  return mix(c, vec3f(0.98, 0.95, 0.86), smoothstep(0.5, 0.9, v));
}

// ------------------------------------------------------------------ example 0: octaves
fn exOctaves(px: vec2f) -> vec3f {
  let W = u.resolution.x;
  let H = u.resolution.y;
  let mainH = floor(H * 0.56);
  let bandH = floor(H * 0.19);
  let off = vec2f(u.seed * 31.7, u.seed * 17.3) + vec2f(u.time * u.speed * 0.06, 0.0);
  let A = ampSum();
  let oc = i32(u.octaves);
  var col = vec3f(0.0);
  if (px.y < mainH) {
    // the sum
    let p = px / H * u.scale + off;
    col = variantRamp(fbm01(p, A));
  } else if (px.y < mainH + bandH) {
    // one panel per octave, all looking at the same window of the world
    let pw = W / f32(oc);
    let i = min(i32(floor(px.x / pw)), oc - 1);
    let lx = px.x - f32(i) * pw;
    let ly = px.y - mainH;
    let p = vec2f(lx, ly) / H * u.scale + off;
    let n = shapeOct(perlin(octPos(p, i)));
    var v = n;
    if (u.variant < 0.5) { v = 0.5 + 0.5 * n; }
    col = variantRamp(v) * 0.9;
    // amplitude bar: this octave's weight relative to the first
    let ampRel = pow(u.gain, f32(i));
    let bar = step(bandH - 7.0, ly) * step(ly, bandH - 3.0) * step(lx, 4.0 + (pw - 8.0) * ampRel) * step(4.0, lx);
    col = mix(col * 0.75, vec3f(1.0, 0.8, 0.3), bar);
    col = mix(col, vec3f(0.0), (1.0 - smoothstep(0.5, 1.5, min(lx, pw - lx))) * step(0.5, f32(i)));
  } else {
    // 1D profile of the row under the mouse: a mountain silhouette
    let y0 = mainH + bandH + 8.0;
    let y1 = H - 6.0;
    var row = mainH * 0.5;
    if (u.mouse.w > 0.5 && u.mouse.y < mainH) { row = u.mouse.y; }
    let p = vec2f(px.x, row) / H * u.scale + off;
    let e = 1.5 / H * u.scale;
    let va = fbm01(p - vec2f(e, 0.0), A);
    let vb = fbm01(p, A);
    let vc = fbm01(p + vec2f(e, 0.0), A);
    let ya = mix(y1, y0, va);
    let yb = mix(y1, y0, vb);
    let yc = mix(y1, y0, vc);
    col = mix(vec3f(0.07, 0.08, 0.13), vec3f(0.02, 0.025, 0.04), (px.y - y0) / (y1 - y0));
    let fillA = step(yb, px.y);
    col = mix(col, variantRamp(vb) * (0.45 + 0.55 * (1.0 - (px.y - yb) / max(y1 - yb, 1.0))), fillA);
    // the first octave alone, for comparison
    var o1 = shapeOct(perlin(p));
    if (u.variant < 0.5) { o1 = 0.5 + 0.5 * o1; }
    let yo = mix(y1, y0, o1);
    let dash = step(0.5, fract(px.x / 10.0));
    col = mix(col, vec3f(1.0, 0.8, 0.3), (1.0 - smoothstep(0.5, 1.5, abs(px.y - yo))) * dash * 0.8);
    let d = min(sdSegment(px, vec2f(px.x - 1.5, ya), vec2f(px.x, yb)), sdSegment(px, vec2f(px.x, yb), vec2f(px.x + 1.5, yc)));
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(0.6, 1.5, d));
  }
  // row marker on the main image
  var row2 = mainH * 0.5;
  if (u.mouse.w > 0.5 && u.mouse.y < mainH) { row2 = u.mouse.y; }
  if (px.y < mainH) {
    let dash = step(0.5, fract(px.x / 14.0));
    col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.5, 1.5, abs(px.y - row2))) * dash * 0.7);
  }
  let sep = min(abs(px.y - mainH), abs(px.y - mainH - bandH));
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, sep));
  return col;
}

// ------------------------------------------------------------------ plain fBm for the warping examples
fn fb(p: vec2f) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  let oc = i32(u.octaves);
  for (var i = 0; i < 10; i++) {
    if (i >= oc) { break; }
    sum += amp * perlin(q);
    q = FBM_ROT * q * 2.02 + vec2f(17.13, 3.71);
    amp *= 0.5;
  }
  return sum;
}

// IQ-style domain warping. stage 0: f(p), 1: f(p + k q), 2: f(p + k r), r = q-warped again.
struct Warp { f: f32, q: vec2f, r: vec2f };
fn warpPattern(p: vec2f, t: f32, stage: f32, k: f32) -> Warp {
  var w: Warp;
  w.q = vec2f(0.0);
  w.r = vec2f(0.0);
  if (stage < 0.5) {
    w.f = fb(p);
    return w;
  }
  w.q = vec2f(fb(p + vec2f(0.0, 0.0) + t * 0.03), fb(p + vec2f(5.2, 1.3) - t * 0.02));
  if (stage < 1.5) {
    w.f = fb(p + k * w.q);
    return w;
  }
  w.r = vec2f(fb(p + k * w.q + vec2f(1.7, 9.2) + 0.15 * t), fb(p + k * w.q + vec2f(8.3, 2.8) + 0.126 * t));
  w.f = fb(p + k * w.r);
  return w;
}

fn warpColor(w: Warp) -> vec3f {
  let f = clamp(0.5 + w.f * 1.1, 0.0, 1.0);
  var col = mix(vec3f(0.1, 0.62, 0.67), vec3f(0.67, 0.67, 0.5), clamp(f * f * 4.0, 0.0, 1.0));
  col = mix(col, vec3f(0.0, 0.0, 0.16), clamp(length(w.q) * 1.1, 0.0, 1.0));
  col = mix(col, vec3f(0.67, 1.0, 1.0), clamp(abs(w.r.x) * 1.4, 0.0, 1.0));
  return col * (f * f * f * 1.4 + 0.6 * f * f + 0.45 * f) * 1.25;
}

fn exWarp(px: vec2f) -> vec3f {
  let H = u.resolution.y;
  let off = vec2f(u.seed * 31.7, u.seed * 17.3);
  let t = u.time * u.speed;
  let p = px / H * u.scale + off;
  let w = warpPattern(p, t, u.stages, u.warp);
  var col = warpColor(w);
  if (u.showVec > 0.5 && u.stages > 0.5) {
    // arrows: where each grid point looks up its noise value (the warp offset)
    let cs = 34.0;
    let cc = (floor(px / cs) + 0.5) * cs;
    let wc = warpPattern(cc / H * u.scale + off, t, u.stages, u.warp);
    var o = wc.q;
    if (u.stages > 1.5) { o = wc.r; }
    var v = o * u.warp / u.scale * H;
    let L = length(v);
    v = v / max(L, 0.001) * min(L, cs * 0.46);
    let tip = cc + v;
    let dir = v / max(length(v), 0.001);
    let sd = vec2f(-dir.y, dir.x);
    let d = min(sdSegment(px, cc, tip), min(sdSegment(px, tip, tip - dir * 5.0 + sd * 3.5), sdSegment(px, tip, tip - dir * 5.0 - sd * 3.5)));
    col = mix(col, vec3f(0.0), (1.0 - smoothstep(1.0, 3.0, d)) * 0.5);
    col = mix(col, vec3f(1.0, 0.95, 0.8), 1.0 - smoothstep(0.5, 1.3, d));
    col = mix(col, vec3f(1.0, 0.95, 0.8), 1.0 - smoothstep(1.5, 2.5, length(px - cc)));
  }
  return col;
}

// ------------------------------------------------------------------ example 2: gas giant
fn rotY3(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn fbm3n(p: vec3f, oc: i32) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < 8; i++) {
    if (i >= oc) { break; }
    sum += amp * perlin3(q);
    q = q * 2.03 + vec3f(17.13, 3.71, 9.17);
    amp *= 0.5;
  }
  return sum;
}

fn bandPalette(b: f32) -> vec3f {
  var c = palette(b, vec3f(0.66, 0.55, 0.43), vec3f(0.28, 0.22, 0.17), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.07, 0.15));
  c = mix(c, vec3f(0.95, 0.9, 0.8), smoothstep(0.75, 1.0, sin(b * TAU * 0.5 + 1.3)) * 0.35);
  return clamp(hueRotate(c, u.hue * TAU), vec3f(0.0), vec3f(1.0));
}

fn starfield(px: vec2f) -> vec3f {
  var c = vec3f(0.006, 0.008, 0.02);
  let H = u.resolution.y;
  let nb = fb(px / H * 1.6 + vec2f(3.0, 1.0));
  c += vec3f(0.08, 0.04, 0.13) * smoothstep(-0.1, 0.5, nb) * 0.6;
  for (var l = 0; l < 2; l++) {
    let cs = 22.0 + 17.0 * f32(l);
    let cell = floor(px / cs);
    let h = hash23(cell + f32(l) * 71.0);
    let sp = (cell + 0.15 + 0.7 * h.xy) * cs;
    let b = pow(h.z, 18.0) * 2.5;
    let tw = 0.75 + 0.25 * sin(u.time * (1.0 + 3.0 * h.x) + h.y * 30.0);
    c += vec3f(0.85, 0.9, 1.0) * b * tw * exp(-length(px - sp) * 1.3);
  }
  return c;
}

fn ringDensity(rho: f32) -> f32 {
  let x = (rho - 1.32) / 0.88;
  if (x < 0.0 || x > 1.0) { return 0.0; }
  var d = 0.55 + 0.3 * valueNoise(vec2f(x * 60.0, 0.5)) + 0.15 * sin(x * 140.0);
  d *= smoothstep(0.0, 0.04, x) * smoothstep(1.0, 0.86, x);
  d *= 1.0 - 0.85 * (smoothstep(0.55, 0.57, x) - smoothstep(0.62, 0.64, x));
  d *= 0.6 + 0.4 * smoothstep(0.2, 0.3, x);
  return clamp(d, 0.0, 1.0);
}

fn exPlanet(px: vec2f) -> vec3f {
  let H = u.resolution.y;
  let ctr = u.resolution * 0.5;
  let R = H * 0.33;
  let q = (px - ctr) / R;
  let r = length(q);
  let t = u.time * u.speed;
  let L = normalize(vec3f(-0.62, -0.38, 0.68));
  let axis = normalize(vec3f(0.22, -1.0, 0.3));
  let E = normalize(cross(axis, vec3f(0.0, 0.0, 1.0)));
  let F = cross(E, axis);
  let atmo = clamp(hueRotate(vec3f(0.55, 0.75, 1.0), u.hue * TAU), vec3f(0.0), vec3f(1.0));
  var col = starfield(px);

  // rings: intersect the view ray with the ring plane (normal = spin axis)
  var ringA = 0.0;
  var ringC = vec3f(0.0);
  var ringFront = 0.0;
  if (u.rings > 0.5) {
    let rz = -(axis.x * q.x + axis.y * q.y) / axis.z;
    let Q = vec3f(q.x, q.y, rz);
    let rho = length(Q);
    let dens = ringDensity(rho);
    // planet shadow on the rings
    let along = dot(Q, L);
    let perp = length(Q - along * L);
    let sh = select(1.0, smoothstep(0.96, 1.04, perp), along < 0.0);
    ringC = mix(vec3f(0.72, 0.64, 0.52), vec3f(0.9, 0.86, 0.78), valueNoise(vec2f(rho * 30.0, 0.5))) * (0.15 + 0.85 * sh);
    ringC = clamp(hueRotate(ringC, u.hue * TAU), vec3f(0.0), vec3f(1.0));
    ringA = dens * 0.85;
    ringFront = step(0.0, rz);
  }
  // rings behind the planet
  col = mix(col, ringC, ringA * (1.0 - ringFront));

  // outer atmosphere glow (brighter on the lit side)
  let dirL = dot(normalize(q + vec2f(0.0001)), normalize(L.xy));
  let halo = exp(-max(r - 1.0, 0.0) * 9.0) * (0.25 + 0.75 * smoothstep(-0.6, 0.8, dirL));
  col += atmo * halo * 0.55 * step(1.0, r);

  if (r < 1.02) {
    let z = sqrt(max(1.0 - r * r, 0.0));
    let n = vec3f(q, z);
    // planet-local frame: y = spin axis, then spin around it
    var P = vec3f(dot(n, E), dot(n, axis), dot(n, F));
    P = rotY3(P, t * 0.12);
    // great storm: swirl the sample position inside an oval
    let lat0 = -0.36;
    let S = vec3f(0.0, lat0, sqrt(1.0 - lat0 * lat0));
    let S2 = rotY3(S, 0.35);
    let east = normalize(cross(vec3f(0.0, 1.0, 0.0), S2));
    let north = cross(S2, east);
    let d3 = P - S2;
    let lx = dot(d3, east);
    let ly = dot(d3, north);
    let e = length(vec2f(lx / 0.24, ly / 0.13));
    let ang = 2.6 * exp(-e * e * 1.4) * step(0.0, dot(P, S2));
    let rl = rot2(ang) * vec2f(lx, ly);
    var Ps = P + east * (rl.x - lx) + north * (rl.y - ly);
    // stretched sample space -> long east-west features, then domain-warp it twice
    let oc = i32(u.octaves);
    let s = vec3f(Ps.x, Ps.y * 3.2, Ps.z) * 1.8;
    let wq = vec2f(fbm3n(s + vec3f(t * 0.02, 0.0, 0.0), oc), fbm3n(s + vec3f(5.2, 1.3, 2.7), oc));
    let wf = fbm3n(s + u.warp * 0.45 * vec3f(wq.x, wq.y * 0.25, wq.y), oc);
    let b = Ps.y * u.bands + wf * u.warp * 0.32 + wq.x * 0.2;
    var c = bandPalette(b);
    c = mix(c, c * vec3f(1.15, 0.75, 0.6), smoothstep(1.0, 0.55, e) * step(0.0, dot(P, S2)) * 0.8);
    // lighting: soft terminator, limb darkening, rim scattering
    let ndl = dot(n, L);
    let lit = smoothstep(-0.12, 0.65, ndl);
    c *= 0.03 + 1.12 * lit;
    c *= 0.55 + 0.45 * pow(z, 0.35);
    let fres = pow(1.0 - z, 3.0);
    c = mix(c, atmo * (0.15 + 1.1 * lit), fres * 0.7);
    // ring shadow on the planet
    if (u.rings > 0.5) {
      let tt = -dot(axis, n) / dot(axis, L);
      if (tt > 0.0) {
        let hit = n + tt * L;
        c *= 1.0 - 0.75 * ringDensity(length(hit));
      }
    }
    let aa = 1.5 / R;
    let cov = 1.0 - smoothstep(1.0 - aa, 1.0, r);
    col = mix(col, c, cov);
  }
  // rings in front of the planet
  col = mix(col, ringC, ringA * ringFront);
  return col;
}

// ------------------------------------------------------------------ example 3: ink marbling + magic portal
fn ink(i: f32) -> vec3f {
  let k = i32(fmod(i, 5.0));
  if (k == 0) { return vec3f(0.95, 0.91, 0.82); }
  if (k == 1) { return vec3f(0.13, 0.2, 0.45); }
  if (k == 2) { return vec3f(0.95, 0.91, 0.82); }
  if (k == 3) { return vec3f(0.72, 0.15, 0.17); }
  return vec3f(0.85, 0.62, 0.22);
}

fn marbling(lp: vec2f, H: f32, t: f32) -> vec3f {
  var p = lp / H * u.scale * 0.55 + vec2f(u.seed * 3.1, 0.0);
  // stir around the mouse like a stylus dragged through the ink
  let mp = u.mouse.xy / H * u.scale * 0.55 + vec2f(u.seed * 3.1, 0.0);
  if (u.mouse.w > 0.5 && u.mouse.x < u.resolution.x * 0.5) {
    let d = p - mp;
    let ang = 2.5 * exp(-dot(d, d) * 9.0);
    p = mp + rot2(ang) * d;
  }
  let k = u.warp * 0.3;
  let q = vec2f(fb(p + t * 0.02), fb(p + vec2f(5.2, 1.3)));
  let r = vec2f(fb(p + k * q + vec2f(1.7, 9.2)), fb(p + k * q + vec2f(8.3, 2.8) - t * 0.015));
  // stripes of ink, combed into waves, then pushed around by the warp
  let comb = 0.12 * sin(p.y * 9.0 + p.x * 0.5);
  let s = (p.x * 1.3 + comb + k * 1.1 * r.x + 0.4 * q.y) * 4.0;
  let i = floor(s);
  let f = fract(s);
  var c = mix(ink(i), ink(i + 1.0), smoothstep(0.82, 0.98, f));
  // fine dark contour where two inks meet
  c *= 1.0 - 0.35 * (1.0 - smoothstep(0.0, 0.035, min(f, 1.0 - f)));
  // paper grain
  c *= 0.93 + 0.07 * valueNoise(lp * 0.7);
  return c;
}

fn portal(lp: vec2f, W: f32, H: f32, t: f32) -> vec3f {
  let ctr = vec2f(W * 0.5, H * 0.52);
  let R = H * 0.34;
  let q = (lp - ctr) / R * vec2f(1.15, 0.92);
  let r = length(q);
  // stone wall lit by the portal
  let bp = lp / H * vec2f(9.0, 16.0);
  let row = floor(bp.y);
  let bx = bp.x + 0.5 * fmod(row, 2.0);
  let fq = vec2f(fract(bx), fract(bp.y));
  let mortar = min(min(fq.x, 1.0 - fq.x) * 1.8, min(fq.y, 1.0 - fq.y));
  var wall = vec3f(0.2, 0.19, 0.22) * (0.75 + 0.35 * hash21(vec2f(floor(bx), row)));
  wall *= 0.4 + 0.6 * smoothstep(0.02, 0.08, mortar);
  let glowC = clamp(hueRotate(vec3f(0.55, 0.3, 1.0), u.hue * TAU), vec3f(0.0), vec3f(1.0));
  let glowC2 = clamp(hueRotate(vec3f(0.3, 0.95, 1.0), u.hue * TAU), vec3f(0.0), vec3f(1.0));
  let pulse = 0.85 + 0.15 * sin(t * 2.3);
  var col = wall * (0.12 + glowC * 2.2 * pulse / (1.0 + 3.0 * r * r));
  // the vortex: rotate each ring of the disc by an angle that grows toward the center
  if (r < 1.0) {
    let twist = u.warp * 0.55 / (r + 0.18) + t * 1.2;
    let sp = rot2(twist) * q;
    let n1 = fb(sp * 2.2 + vec2f(0.0, t * 0.2));
    let n2 = fb(sp * 3.0 + n1 * 1.5 - vec2f(t * 0.1, 0.0));
    let v = clamp(0.5 + n2 * 1.3, 0.0, 1.0);
    var c = mix(vec3f(0.02, 0.0, 0.05), glowC, smoothstep(0.2, 0.7, v));
    c = mix(c, glowC2, smoothstep(0.62, 0.95, v));
    c += vec3f(1.0) * smoothstep(0.85, 1.0, v) * 0.6;
    // dark eye in the middle, bright inner rim
    c *= smoothstep(0.05, 0.35, r);
    c += glowC2 * smoothstep(0.75, 0.99, r) * 0.8;
    col = mix(col, c, 1.0 - smoothstep(0.98, 1.0, r));
  }
  // stone frame: voussoirs (wedge-shaped stones) around the opening
  let fr0 = 1.0;
  let fr1 = 1.2;
  if (r > fr0 - 0.01 && r < fr1 + 0.01) {
    let a = atan2(q.y, q.x);
    let segs = 18.0;
    let sa = a / TAU * segs;
    let fs = fract(sa);
    let edgeA = min(fs, 1.0 - fs) * TAU / segs * r * R;
    let edgeR = min(r - fr0, fr1 - r) * R;
    let ed = min(edgeA, edgeR);
    var st = vec3f(0.36, 0.33, 0.36) * (0.8 + 0.3 * hash11(floor(sa)));
    st *= 0.7 + 0.3 * smoothstep(0.0, 6.0, ed);
    // lit from the portal side
    st *= 0.5 + glowC * 1.2 * smoothstep(fr1, fr0, r) + 0.3;
    let cov = smoothstep(0.0, 1.2, ed);
    col = mix(col, st, cov * step(fr0, r) * step(r, fr1) + 0.0);
    col = mix(col, vec3f(0.03), (1.0 - smoothstep(0.0, 1.2, ed)) * step(fr0 - 0.005, r) * step(r, fr1 + 0.005));
  }
  // orbiting sparks being pulled in
  for (var i = 0; i < 14; i++) {
    let fi = f32(i);
    let h = hash12(fi + 3.0);
    let life = fract(t * (0.15 + 0.1 * h.x) + h.y);
    let rr = mix(1.35, 0.15, life * life);
    let aa = h.x * TAU + t * (0.6 + h.y) + life * 5.0;
    let spk = vec2f(cos(aa), sin(aa)) * rr / vec2f(1.15, 0.92) * R + ctr;
    let ds = length(lp - spk);
    col += glowC2 * exp(-ds * 0.45) * 1.2 * smoothstep(0.0, 0.2, life) * smoothstep(1.0, 0.8, life);
  }
  return col;
}

fn exSwirl(px: vec2f) -> vec3f {
  let W = u.resolution.x * 0.5;
  let H = u.resolution.y;
  let t = u.time * u.speed;
  var col = vec3f(0.0);
  if (px.x < W) { col = marbling(px, H, t); }
  else { col = portal(px - vec2f(W, 0.0), W, H, t); }
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(1.0, 2.5, abs(px.x - W)));
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = exOctaves(px); }
  else if (ex == 1) { col = exWarp(px); }
  else if (ex == 2) { col = exPlanet(px); }
  else { col = exSwirl(px); }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;

const STAGE_TEXT = ['<b>f(p)</b> · plain fBm', '<b>f(p + k·q)</b> · q = (f(p), f(p+o₁))', '<b>f(p + k·r)</b> · r = (f(p + k·q + o₂), …)'];

export default shaderScene({
  interaction: 'Hover the image: the dashed row is drawn as a 1D profile.',
  examples: [
    {
      id: 'octaves',
      label: 'Octaves, one by one',
      kind: 'Abstract',
      note: 'fBm adds up several copies of noise (“octaves”). Each one has <b>lacunarity×</b> the frequency and <b>gain×</b> the amplitude of the previous one. The strip shows each octave alone (yellow bar = its weight); the top image is their sum; the bottom is a 1D profile — the dashed line is octave 1 alone.',
    },
    {
      id: 'warp',
      label: 'Domain warping',
      kind: 'Abstract',
      note: 'Instead of <code>f(p)</code>, evaluate <code>f(p + offset)</code> where the offset itself comes from noise. Doing it twice gives Inigo Quilez’s famous <code>f(p + f(p + f(p)))</code>: flowing, marbled, smoke-like shapes. Turn on the arrows to see where each point looks up its value.',
      params: { scale: 1.2, octaves: 4, warp: 3.5 },
      hint: '',
    },
    {
      id: 'planet',
      label: 'Gas giant',
      kind: 'In a game',
      note: 'A whole planet from one shader: latitude bands whose edges are domain-warped by 3D fBm sampled <i>on the sphere</i> (no seams), a swirled storm oval, soft day/night terminator, atmospheric rim light, and rings found by intersecting each pixel’s view ray with the ring plane — including shadows both ways.',
      params: { octaves: 5, warp: 2.4, bands: 6 },
      hint: '',
    },
    {
      id: 'swirl',
      label: 'Marbled ink & magic portal',
      kind: 'Real life',
      note: 'Left: paper marbling (ebru) — stripes of ink pushed around by a double domain warp; hover to stir it with a stylus. Right: a magic portal — the same noise, but each ring of the disc is rotated by an angle that grows toward the center (a <i>twist</i> warp) to make a vortex.',
      params: { warp: 4, octaves: 5, scale: 3 },
      hint: 'Hover the ink on the left to stir it.',
    },
  ],
  controls: [
    { type: 'slider', key: 'scale', label: 'Scale', min: 0.5, max: 10, step: 0.05, value: 3, showFor: ['octaves', 'warp', 'swirl'], help: 'Zoom: how many base-noise features fit on screen.' },
    { type: 'slider', key: 'octaves', label: 'Octaves', min: 1, max: 10, step: 1, value: 6, help: 'How many layers of noise are summed. Each extra octave costs one more noise evaluation per pixel.' },
    { type: 'slider', key: 'lacunarity', label: 'Lacunarity', min: 1.2, max: 4, step: 0.01, value: 2, showFor: ['octaves'], help: 'Frequency multiplier between octaves (2 = each octave is twice as detailed).' },
    { type: 'slider', key: 'gain', label: 'Gain (persistence)', min: 0.1, max: 0.9, step: 0.01, value: 0.5, showFor: ['octaves'], help: 'Amplitude multiplier between octaves. Low = smooth, high = rough.' },
    {
      type: 'select', key: 'variant', label: 'Variant', value: 'fbm', showFor: ['octaves'],
      options: [{ value: 'fbm', label: 'Standard fBm (clouds)' }, { value: 'turb', label: 'Turbulence |n| (fire, billows)' }, { value: 'ridged', label: 'Ridged (1−|n|)² (mountains)' }],
      help: 'What is done to each octave before summing.',
    },
    {
      type: 'select', key: 'stages', label: 'Warp stages', value: 2, showFor: ['warp'],
      options: [{ value: 0, label: '0 · f(p)' }, { value: 1, label: '1 · f(p + q)' }, { value: 2, label: '2 · f(p + r(q))' }],
      help: 'How many times the domain is warped.',
    },
    { type: 'slider', key: 'warp', label: 'Warp strength', min: 0, max: 8, step: 0.01, value: 4, showFor: ['warp', 'planet', 'swirl'], help: 'How far (in noise units) points are pushed before sampling.' },
    { type: 'toggle', key: 'showVec', label: 'Show warp arrows', value: false, showFor: ['warp'], help: 'Each arrow: the offset added to that point before the final lookup.' },
    { type: 'slider', key: 'bands', label: 'Band frequency', min: 1, max: 12, step: 0.1, value: 5, showFor: ['planet'], help: 'How many color bands from pole to pole.' },
    { type: 'toggle', key: 'rings', label: 'Rings', value: true, showFor: ['planet'], help: 'Ray–plane intersection per pixel, with planet ↔ ring shadows.' },
    { type: 'slider', key: 'hue', label: 'Hue shift', min: 0, max: 1, step: 0.005, value: 0, showFor: ['planet', 'swirl'], help: 'Rotate the palette: Jupiter → alien worlds.' },
    { type: 'slider', key: 'seed', label: 'Seed', min: 0, max: 99, step: 1, value: 0, showFor: ['octaves', 'warp', 'swirl'] },
    { type: 'slider', key: 'speed', label: 'Time speed', min: 0, max: 3, step: 0.01, value: 1, help: 'Animation speed (0 = frozen).' },
  ],
  uniforms: {
    scale: 'f32', octaves: 'f32', lacunarity: 'f32', gain: 'f32', variant: 'f32', stages: 'f32', warp: 'f32', showVec: 'f32',
    bands: 'f32', rings: 'f32', hue: 'f32', seed: 'f32', speed: 'f32',
  },
  include: ['noise', 'sdf', 'color'],
  bind(params, ctx) {
    const ex = ctx.example;
    if (ex === 'octaves') {
      const name = { fbm: 'fBm', turb: 'Turbulence', ridged: 'Ridged' }[params.variant] || 'fBm';
      setLabels(ctx, [
        { text: `<b>Sum</b> · ${name}, ${Math.round(params.octaves)} octaves`, style: 'left:8px;top:44px' },
        { text: 'each octave alone →', style: 'left:8px;top:calc(56% + 6px);opacity:.85' },
        { text: '1D profile of the dashed row', style: 'right:8px;top:calc(75% + 6px);opacity:.85' },
      ]);
    } else if (ex === 'warp') {
      setLabels(ctx, [{ text: STAGE_TEXT[Math.round(params.stages)] || STAGE_TEXT[2], style: 'left:8px;top:44px' }]);
    } else if (ex === 'swirl') {
      setLabels(ctx, [
        { text: '<b>Marbled ink</b> · double warp', style: 'left:8px;top:44px' },
        { text: '<b>Magic portal</b> · twist warp', style: 'left:calc(50% + 8px);top:44px' },
      ]);
    } else setLabels(ctx, []);
    return {};
  },
  code: CODE,
  about: {
    summary: 'One layer of noise looks like blurry blobs. Stack several at increasing frequency (fBm) and you get natural detail; feed noise into its own input (domain warping) and you get flowing, organic shapes.',
    what: `<p><b>Octaves</b>: the individual layers of an fBm and their sum, as an image and as a 1D mountain profile, with ridged and turbulence variants.
      <b>Domain warping</b>: the same fBm sampled at positions that are pushed around by other fBm calls. <b>Gas giant</b> and
      <b>ink & portal</b> show how far these two ideas go in real art.</p>`,
    how: `<ol>
      <li><b>fBm</b> (fractal Brownian motion): <code>sum += amp × noise(p); p *= lacunarity; amp *= gain;</code> repeated for each octave.
        Big, strong octaves give the overall shape; small, faint ones add detail — like mountains, then hills, then rocks.</li>
      <li>Each octave is also <b>rotated</b> a little so their grids don’t line up (that would create visible axis-aligned artifacts).</li>
      <li><b>Turbulence</b> sums <code>|noise|</code>: creases at every zero crossing → billowing smoke and fire.
        <b>Ridged</b> uses <code>(1 − |noise|)²</code>: the creases flip into sharp ridges; weighting each octave by the previous one keeps detail on the peaks.</li>
      <li><b>Domain warping</b>: compute <code>q = (fbm(p), fbm(p + offset))</code>, then look up <code>fbm(p + k·q)</code>. Repeat for a second level.
        Nothing new is drawn — the <i>coordinates</i> are distorted, so all the detail gets stretched and swirled.</li>
      <li><b>On a sphere</b>: sample 3D noise at the surface point instead of 2D noise at (longitude, latitude) — no seam, no pinching at the poles.</li>
    </ol>`,
    uses: [
      { title: 'Terrain', text: 'Height maps are fBm (plus ridged noise for mountain ranges) — see World & Terrain Generation.' },
      { title: 'Skies & space', text: 'Clouds, nebulae, planets and gas giants in space games, star maps and title screens.' },
      { title: 'Materials', text: 'Marble, malachite, agate, lava, rust, ink, smoke — anything with flowing, layered structure.' },
      { title: 'Magic & VFX', text: 'Portals, energy shields, auras, vortexes, and animated fog layers.' },
    ],
    try: [
      'Set <b>Octaves</b> to 1, then step it up one at a time and watch the profile go from rolling hills to jagged rock.',
      'Push <b>Gain</b> to 0.8: the small octaves become as loud as the big ones — rough static. At 0.3 it’s smooth and blobby.',
      'Switch <b>Variant</b> to Ridged: the profile becomes a mountain range with sharp peaks.',
      'On <b>Domain warping</b>, flip <i>Warp stages</i> 0 → 1 → 2 with the arrows on, then sweep <i>Warp strength</i>.',
      'On <b>Gas giant</b>, raise <i>Band frequency</i> and <i>Hue shift</i> for an alien world; turn the rings off to see the ring shadow disappear.',
    ],
    ask: [
      'fBm noise with adjustable octaves, lacunarity and gain',
      'ridged multifractal noise for mountains',
      'domain-warped noise (Inigo Quilez style)',
      'procedural gas giant planet shader with rings and atmosphere',
      'swirling magic portal shader',
      'animated paper marbling / ink effect',
    ],
    perf: `<p>Cost ≈ <b>octaves × noise evaluations</b> per pixel. A two-level warp calls fBm 5 times — 30 noise lookups at 6 octaves —
      still fine for a full screen on a mid-range GPU, but it adds up. The gas giant uses 3D noise (8 corners instead of 4) for 3 fBm calls.
      Tips: drop octaves that are smaller than a pixel (fewer when zoomed out), and render slow-changing things (a planet) into a texture once.</p>`,
    api: `<p>Pure fragment math: identical in WebGPU and WebGL2. With WebGPU you could additionally bake noise into a 3D texture
      with a compute shader once and sample it — trading ALU work for memory bandwidth.</p>`,
    code: [
      {
        title: 'fBm with lacunarity, gain and the ridged/turbulence variants',
        lang: 'wgsl',
        src: `fn fbmV(p: vec2f) -> f32 {
  var sum = 0.0;  var amp = 0.5;  var q = p;  var prev = 1.0;
  for (var i = 0; i < 10; i++) {
    if (i >= i32(u.octaves)) { break; }
    let n = shapeOct(perlin(q));       // n, |n| or (1-|n|)^2
    if (u.variant > 1.5) { sum += amp * n * prev; prev = clamp(n * 1.6, 0.0, 1.0); }
    else { sum += amp * n; }
    q = FBM_ROT * q * u.lacunarity + vec2f(17.13, 3.71);   // rotate + scale
    amp *= u.gain;
  }
  return sum;
}`,
      },
      {
        title: 'Domain warping: f(p + k·r), r = f(p + k·q), q = f(p)',
        lang: 'wgsl',
        src: `let q = vec2f(fb(p + t * 0.03), fb(p + vec2f(5.2, 1.3) - t * 0.02));
let r = vec2f(fb(p + k * q + vec2f(1.7, 9.2) + 0.15 * t),
              fb(p + k * q + vec2f(8.3, 2.8) + 0.126 * t));
let f = fb(p + k * r);
// color from f, plus |q| and r.x for extra variation (IQ's trick)`,
      },
      {
        title: 'Portal vortex: twist each ring by an angle that grows toward the center',
        lang: 'wgsl',
        src: `let twist = u.warp * 0.55 / (r + 0.18) + t * 1.2;
let sp = rot2(twist) * q;                 // q = position inside the portal disc
let n1 = fb(sp * 2.2 + vec2f(0.0, t * 0.2));
let n2 = fb(sp * 3.0 + n1 * 1.5 - vec2f(t * 0.1, 0.0));`,
      },
    ],
    links: [
      { title: 'Inigo Quilez — Domain warping', url: 'https://iquilezles.org/articles/warp/', note: 'the original article' },
      { title: 'Inigo Quilez — fBm', url: 'https://iquilezles.org/articles/fbm/', note: 'why gain 0.5 looks natural' },
      { title: 'The Book of Shaders — Fractal Brownian Motion', url: 'https://thebookofshaders.com/13/' },
    ],
  },
});

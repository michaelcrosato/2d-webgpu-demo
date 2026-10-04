import { shaderScene } from '../../core/shaderscene.js';

// Silhouette & atmosphere (Limbo / Inside / sunset postcards).
// Everything is drawn back-to-front in ONE fragment shader: each depth layer is a set of signed
// distance shapes (trees, ground, vines, a boy) filled with a single flat value, then pushed toward
// the fog color by an amount that grows with distance ("aerial perspective"). Far layers are drawn with a
// wider anti-aliasing width = cheap depth-of-field blur. Grain, flicker and vignette finish the film look.

const CODE = /* wgsl */ `
fn cover(d: f32, w: f32) -> f32 { return clamp(0.5 - d / w, 0.0, 1.0); }

// tapered capsule: radius ra at a, rb at b
fn sdTaper(p: vec2f, a: vec2f, b: vec2f, ra: f32, rb: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - mix(ra, rb, h);
}

fn trunkX(cx: f32, y: f32, hy: f32, amp: f32) -> f32 { return cx + amp * sin(y * 3.7 + hy * 19.0) + amp * 0.5 * sin(y * 9.0 + hy * 7.0); }

// A layer of bare, crooked trees (one candidate per cell, neighbours checked so branches can cross cells).
fn treesD(q: vec2f, cellW: f32, seed: f32, thick: f32, groundY: f32, density: f32, branches: f32) -> f32 {
  var d = 10.0;
  let c0 = floor(q.x / cellW);
  for (var k = -1; k <= 1; k++) {
    let c = c0 + f32(k);
    let h = hash23(vec2f(c, seed));
    if (h.x > density) { continue; }
    let cx = (c + 0.25 + 0.5 * h.y) * cellW;
    let tw = thick * (0.7 + 0.6 * h.z);
    let amp = thick * 2.2;
    let flare = 1.0 + 2.2 * exp(-max(groundY - q.y, 0.0) / (thick * 2.5));
    let w = tw * (0.55 + 0.45 * clamp(q.y, 0.0, 1.0)) * flare;
    var dt = abs(q.x - trunkX(cx, q.y, h.y, amp)) - w;
    dt = max(dt, q.y - groundY - thick * 4.0);
    if (branches > 0.5) {
      for (var b = 0; b < 3; b++) {
        let hb = hash23(vec2f(c * 3.1 + f32(b), seed + 9.0));
        if (hb.z < 0.3) { continue; }
        let by = 0.05 + (groundY - 0.32) * (f32(b) + hb.x) / 3.0;   // spread along the trunk
        let side = select(-1.0, 1.0, hb.y > 0.5);
        let len = cellW * (0.2 + 0.3 * hb.z);
        let bw = tw * (0.55 + 0.45 * by);
        let a = vec2f(trunkX(cx, by, h.y, amp), by);
        let m = a + vec2f(side * len * 0.55, -len * (0.12 + 0.25 * hb.x));    // out...
        let e = m + vec2f(side * len * 0.4, -len * (0.4 + 0.5 * fract(hb.y * 7.0)));  // ...then curving up
        dt = min(dt, sdTaper(q, a, m, bw * 0.75, bw * 0.38));
        dt = min(dt, sdTaper(q, m, e, bw * 0.38, bw * 0.05));
        dt = min(dt, sdTaper(q, m, m + vec2f(side * len * 0.3, len * 0.08), bw * 0.2, bw * 0.04));
      }
    }
    d = min(d, dt);
  }
  return d;
}

// Straight thin trunks (Inside-style pine forest).
fn polesD(q: vec2f, cellW: f32, seed: f32, thick: f32, groundY: f32, density: f32) -> f32 {
  var d = 10.0;
  let c0 = floor(q.x / cellW);
  for (var k = -1; k <= 1; k++) {
    let c = c0 + f32(k);
    let h = hash23(vec2f(c, seed));
    if (h.x > density) { continue; }
    let cx = (c + 0.2 + 0.6 * h.y) * cellW + 0.004 * (q.y - groundY) * (h.z - 0.5);
    let w = thick * (0.6 + 0.8 * h.z) * (0.75 + 0.25 * q.y);
    d = min(d, max(abs(q.x - cx) - w, q.y - groundY - 0.02));
  }
  return d;
}

// Vines / roots hanging from above.
fn vinesD(q: vec2f, cellW: f32, seed: f32, maxLen: f32, thick: f32, t: f32) -> f32 {
  var d = 10.0;
  let c0 = floor(q.x / cellW);
  for (var k = -1; k <= 1; k++) {
    let c = c0 + f32(k);
    let h = hash23(vec2f(c, seed + 3.0));
    if (h.x > 0.55) { continue; }
    let len = maxLen * (0.35 + 0.65 * h.y);
    let yy = clamp(q.y, -0.1, len);
    let sway = 0.012 * sin(t * 0.7 + h.z * 6.0) * yy / max(len, 0.01);
    let vx = (c + 0.5) * cellW + sway + 0.006 * sin(yy * 23.0 + h.z * 9.0);
    var dv = length(vec2f(q.x - vx, q.y - yy)) - thick * (1.0 - 0.4 * yy / max(len, 0.01));
    // a few leaves along the vine
    let lf = fract(q.y * 28.0 + h.z * 5.0);
    let leaf = length(vec2f((q.x - vx - 0.004 * select(-1.0, 1.0, fract(floor(q.y * 28.0) * 0.5) > 0.25)) * 1.0, (lf - 0.5) / 28.0 * 0.7)) - thick * 1.9;
    if (q.y < len && q.y > 0.0) { dv = min(dv, leaf); }
    d = min(d, dv);
  }
  return d;
}

// Grass blades on top of a ground line (returns the blade height to subtract from groundY).
fn grassH(x: f32, seed: f32, hgt: f32) -> f32 {
  let g1 = x * 150.0;
  let f1 = fract(g1);
  let h1 = hash21(vec2f(floor(g1), seed));
  let b1 = hgt * (0.25 + 0.75 * h1 * h1) * pow(max(0.0, 1.0 - abs(f1 - 0.5) * 2.0), 3.0);
  let g2 = x * 97.0 + 0.37;
  let f2 = fract(g2);
  let h2 = hash21(vec2f(floor(g2), seed + 1.0));
  let b2 = hgt * 1.3 * step(0.55, h2) * h2 * pow(max(0.0, 1.0 - abs(f2 - 0.5) * 2.0), 4.0);
  return max(b1, b2);
}

fn groundY(x: f32, base: f32, amp: f32, seed: f32) -> f32 {
  return base + amp * (perlin(vec2f(x * 1.3, seed)) * 0.7 + perlin(vec2f(x * 4.1, seed + 3.0)) * 0.3);
}

// The boy: feet at the origin, facing right, returns distance; also outputs eye distance via eyes()
fn boyD(p: vec2f, s: f32, walk: f32) -> vec3f {
  let q = p / s;
  let ph = walk;
  let bob = abs(sin(ph)) * 0.035;
  let hip = vec2f(0.0, -0.36 - bob);
  let sh = vec2f(0.02, -0.68 - bob);
  // legs
  let f1 = vec2f(0.17 * sin(ph), -max(0.0, cos(ph)) * 0.06);
  let f2 = vec2f(-0.17 * sin(ph), -max(0.0, -cos(ph)) * 0.06);
  var legs = min(sdTaper(q, hip, f1, 0.075, 0.055), sdTaper(q, hip, f2, 0.075, 0.055));
  legs = min(legs, min(sdTaper(q, f1, f1 + vec2f(0.09, 0.0), 0.05, 0.045), sdTaper(q, f2, f2 + vec2f(0.09, 0.0), 0.05, 0.045)));
  // torso & arms
  var torso = sdTaper(q, hip + vec2f(0.0, -0.02), sh, 0.11, 0.1);
  let h1 = sh + vec2f(-0.15 * sin(ph) + 0.03, 0.3);
  let h2 = sh + vec2f(0.15 * sin(ph) + 0.03, 0.3);
  torso = min(torso, min(sdTaper(q, sh, h1, 0.06, 0.045), sdTaper(q, sh, h2, 0.06, 0.045)));
  // head (big, Limbo-like) with a few spiky hairs
  let hc = vec2f(0.06, -0.92 - bob);
  var head = length((q - hc) * vec2f(1.0, 0.95)) - 0.2;
  head = min(head, sdTaper(q, hc + vec2f(-0.05, -0.14), hc + vec2f(-0.22, -0.2), 0.05, 0.01));
  head = min(head, sdTaper(q, hc + vec2f(-0.12, -0.08), hc + vec2f(-0.27, -0.05), 0.04, 0.01));
  // eyes: two dots facing right
  let e = min(length(q - (hc + vec2f(0.08, -0.01))), length(q - (hc + vec2f(0.165, -0.015)))) - 0.034;
  return vec3f(min(legs, min(torso, head)) * s, e * s, torso * s);
}

fn filmGrain(px: vec2f) -> f32 { return hash21(px + vec2f(fract(u.time * 7.13) * 931.0, fract(u.time * 3.7) * 517.0)) - 0.5; }

fn finish(c0: vec3f, uv: vec2f, px: vec2f, vigK: f32) -> vec3f {
  var c = c0;
  let v = uv - 0.5;
  c *= clamp(1.0 - dot(v, v) * u.vignette * vigK, 0.0, 1.0);
  c += vec3f(filmGrain(px)) * u.grain;
  let fl = hash11(floor(u.time * 18.0));
  c *= 1.0 + u.flicker * 0.12 * (fl - 0.5);
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

// ---------------------------------------------------------------------------------- Limbo
fn limbo(p: vec2f, px: vec2f) -> vec3f {
  let t = u.time;
  let aspect = u.resolution.x / u.resolution.y;
  let pw = 1.0 / u.resolution.y;
  let cam = t * u.drift * 0.06;
  let lp = vec2f(aspect * 0.62, 0.22);
  // the bright fog "sky": a soft light from behind and above
  let ld = length((p - lp) * vec2f(0.75, 1.0));
  var fogL = 0.13 + 0.62 * exp(-ld * 2.4) + 0.14 * (1.0 - p.y);
  fogL += 0.05 * (valueNoise(vec2f(p.x * 3.0 + t * 0.03, p.y * 5.0)) - 0.5);
  let fogC = vec3f(fogL);
  var col = fogC + vec3f(0.25) * exp(-ld * 9.0) * u.light;

  // layer 0 & 1: far forests, almost lost in fog
  var depthF = array<f32, 5>(0.82, 0.62, 0.38, 0.1, 0.0);
  var blurF = array<f32, 5>(5.0, 3.0, 1.2, 0.0, 16.0);
  for (var i = 0; i < 2; i++) {
    let fi = f32(i);
    let par = 0.15 + 0.2 * fi;
    let q = vec2f(p.x + cam * par + fi * 13.7, p.y);
    let gy = groundY(q.x, 0.74 + 0.04 * fi, 0.03, 4.0 + fi);
    var d = min(treesD(q, 0.085 + 0.07 * fi, 11.0 + fi, 0.0028 + 0.0025 * fi, gy, 0.6, fi), gy - p.y);
    let w = pw * (1.5 + u.blur * blurF[i]);
    let val = mix(vec3f(0.02), fogC, clamp(u.fog * depthF[i] * 1.15, 0.0, 0.97));
    col = mix(col, val, cover(d, w));
    // fog wisps creeping over the ground of this layer
    let wisp = smoothstep(0.35, 0.8, valueNoise(vec2f(q.x * 2.5 + t * 0.04, p.y * 9.0))) * smoothstep(gy - 0.18, gy, p.y);
    col = mix(col, fogC * 1.05, wisp * u.fog * 0.35);
  }

  // god rays from the light (drawn before the near layers so they occlude them)
  let ang = atan2(p.y - lp.y, p.x - lp.x);
  let rays = pow(valueNoise(vec2f(ang * 9.0, t * 0.15)), 3.0) * smoothstep(0.0, 0.1, p.y - lp.y + 0.05);
  col += vec3f(rays * exp(-ld * 1.4) * 0.3 * u.light);

  // dust motes in the light
  let dq = p * 22.0 + vec2f(t * 0.25, -t * 0.12);
  let dc = floor(dq);
  let dh = hash22(dc);
  let mote = exp(-length(fract(dq) - dh) * 40.0) * step(0.6, hash21(dc + 4.0));
  col += vec3f(mote * exp(-ld * 2.5) * 0.6 * u.light);

  // layer 2: mid trees with hanging vines
  {
    let q = vec2f(p.x + cam * 0.55 + 5.3, p.y);
    let gy = groundY(q.x, 0.82, 0.03, 7.0);
    var d = min(treesD(q, 0.24, 21.0, 0.008, gy, 0.6, 1.0), gy - p.y);
    d = min(d, vinesD(q, 0.15, 2.0, 0.34, 0.0018, t));
    let w = pw * (1.5 + u.blur * blurF[2]);
    let val = mix(vec3f(0.02), fogC, clamp(u.fog * depthF[2], 0.0, 0.95));
    col = mix(col, val, cover(d, w));
    let wisp = smoothstep(0.4, 0.85, valueNoise(vec2f(q.x * 3.5 - t * 0.05, p.y * 12.0))) * smoothstep(gy - 0.12, gy, p.y);
    col = mix(col, fogC, wisp * u.fog * 0.3);
  }

  // layer 3: the play layer — ground, grass, a big tree, vines, the boy
  var eyeGlow = 0.0;
  {
    let q = vec2f(p.x + cam, p.y);
    let gy = groundY(q.x, 0.86, 0.025, 9.0);
    let gg = gy - grassH(q.x + (gy - p.y) * 0.5, 3.0, 0.03);
    var d = gg - p.y;
    d = min(d, treesD(q, 0.85, 31.0, 0.022, gy, 0.55, 1.0));
    d = min(d, vinesD(q, 0.17, 5.0, 0.22, 0.0024, t));
    // the boy stands at a fixed screen position on the ground below him
    let bx = aspect * 0.3;
    let by = groundY(bx + cam, 0.86, 0.025, 9.0) + 0.004;
    let walk = t * u.drift * 6.0;
    let boy = boyD(p - vec2f(bx, by), 0.1, walk);
    d = min(d, boy.x);
    let w = pw * (1.2 + u.blur * blurF[3]);
    let val = mix(vec3f(0.015), fogC, clamp(u.fog * depthF[3], 0.0, 0.9));
    col = mix(col, val, cover(d, w));
    // glowing eyes (+ a small halo)
    eyeGlow = cover(boy.y, pw * 1.2) + exp(-max(boy.y, 0.0) / (pw * 3.0)) * 0.35;
  }
  col = mix(col, vec3f(1.0), clamp(eyeGlow, 0.0, 1.0));

  // layer 4: out-of-focus foreground grass & a branch (strong DOF blur)
  {
    let q = vec2f(p.x + cam * 1.7 + 2.0, p.y);
    let fy = 1.02 - 0.09 * smoothstep(0.3, 0.8, valueNoise(vec2f(q.x * 2.2, 1.0))) - grassH(q.x * 0.35, 8.0, 0.09);
    var d = fy - p.y;
    let br = sdTaper(p, vec2f(-0.05, 0.02), vec2f(0.32, 0.11 + 0.01 * sin(t * 0.5)), 0.018, 0.004);
    d = min(d, br);
    let w = pw * (1.5 + u.blur * blurF[4]);
    col = mix(col, vec3f(0.0), cover(d, w));
  }
  // contrast curve (Limbo is very high contrast)
  col = smoothstep(vec3f(0.02), vec3f(0.95), col);
  return col;
}

// ---------------------------------------------------------------------------------- Layers + fog (abstract)
fn layered(p: vec2f) -> vec3f {
  let t = u.time;
  let aspect = u.resolution.x / u.resolution.y;
  let pw = 1.0 / u.resolution.y;
  let cam = t * u.drift * 0.08;
  let fogC = mix(vec3f(0.86, 0.9, 0.93), vec3f(0.98, 0.86, 0.78), smoothstep(0.1, 0.6, p.y));
  var col = mix(vec3f(0.62, 0.74, 0.86), fogC, smoothstep(0.0, 0.55, p.y));
  let sp = vec2f(aspect * 0.7, 0.3);
  col = mix(col, vec3f(1.0, 0.97, 0.9), cover(length(p - sp) - 0.07, pw * 1.5));
  col += vec3f(1.0, 0.85, 0.7) * exp(-length(p - sp) * 6.0) * 0.25;
  let n = i32(u.layers);
  let nearC = vec3f(0.05, 0.09, 0.15);
  for (var i = 0; i < 8; i++) {
    if (i >= n) { break; }
    let fi = f32(i) / max(f32(n) - 1.0, 1.0);   // 0 = farthest, 1 = nearest
    let par = mix(0.08, 1.3, fi * fi);
    let x = p.x + cam * par + f32(i) * 7.31;
    let base = mix(0.42, 0.86, fi);
    let amp = mix(0.12, 0.05, fi);
    let gy = base - amp * (0.6 * ridged(vec2f(x * mix(1.2, 2.2, fi), f32(i)), 3) + 0.2 * perlin(vec2f(x * 6.0, f32(i) * 3.0)));
    // aerial perspective: the farther the layer, the more fog is mixed in (Beer-Lambert style)
    let depth = 1.0 - fi;
    let fogAmt = 1.0 - exp(-u.fog * 4.0 * depth);
    var lc = mix(nearC, fogC, fogAmt);
    if (u.tint > 0.5) { lc = mix(hsv2rgb(vec3f(fract(fi * 0.8 + 0.55), 0.65, 0.95)) * 0.85, fogC, fogAmt * 0.3); }
    // mist pooling in the valley above this ridge
    col = mix(col, fogC, exp(-max(gy - p.y, 0.0) * 30.0) * 0.35 * u.fog * depth);
    col = mix(col, lc, cover(gy - p.y, pw * (1.2 + depth * 3.0 * u.blur)));
  }
  return col;
}

// ---------------------------------------------------------------------------------- Sunset
fn sunsetSky(p: vec2f, aspect: f32) -> vec3f {
  let hz = 0.64;
  let y = clamp(p.y / hz, 0.0, 1.0);
  var c = mix(vec3f(0.13, 0.06, 0.26), vec3f(0.48, 0.13, 0.42), smoothstep(0.0, 0.42, y));
  c = mix(c, vec3f(0.93, 0.35, 0.3), smoothstep(0.35, 0.72, y));
  c = mix(c, vec3f(1.0, 0.66, 0.3), smoothstep(0.66, 0.92, y));
  c = mix(c, vec3f(1.0, 0.86, 0.55), smoothstep(0.9, 1.0, y));
  let sp = vec2f(aspect * 0.6, hz - 0.06 - u.sunHeight * 0.32);
  let sd = length(p - sp);
  c += vec3f(1.0, 0.6, 0.3) * exp(-sd * 5.0) * 0.45 * u.light;
  c = mix(c, vec3f(1.0, 0.93, 0.7), cover(sd - 0.075, 2.0 / u.resolution.y));
  // thin streaky clouds (dark with a hot rim on the side facing the sun)
  let cn = valueNoise(vec2f(p.x * 2.2 + u.time * 0.01, p.y * 34.0)) * valueNoise(vec2f(p.x * 5.0 - u.time * 0.006, p.y * 11.0));
  let band = smoothstep(0.12, 0.26, p.y) * (1.0 - smoothstep(0.4, 0.58, p.y));
  let cl = smoothstep(0.22, 0.42, cn) * band;
  let rim = exp(-length((p - sp) * vec2f(0.6, 2.5)) * 2.5);
  c = mix(c, mix(vec3f(0.32, 0.1, 0.3), vec3f(1.0, 0.55, 0.35), rim), cl * 0.85);
  return c;
}

fn sunsetBack(p: vec2f, aspect: f32, cam: f32) -> vec3f {
  let pw = 1.0 / u.resolution.y;
  var col = sunsetSky(p, aspect);
  // far mountains (purple, hazy) & mid hills
  let x0 = p.x + cam * 0.12;
  let m0 = 0.64 - 0.16 * ridged(vec2f(x0 * 1.4, 2.0), 4);
  col = mix(col, mix(vec3f(0.36, 0.12, 0.34), col, 0.45 * u.fog), cover(m0 - p.y, pw * 2.0));
  let x1 = p.x + cam * 0.3 + 3.0;
  let m1 = 0.66 - 0.07 * ridged(vec2f(x1 * 2.3, 5.0), 3);
  col = mix(col, mix(vec3f(0.17, 0.05, 0.17), col, 0.3 * u.fog), cover(m1 - p.y, pw * 1.5));
  return col;
}

fn acaciaD(q: vec2f, cellW: f32, seed: f32, gy: f32, sc: f32) -> f32 {
  var d = 10.0;
  let c0 = floor(q.x / cellW);
  for (var k = -1; k <= 1; k++) {
    let c = c0 + f32(k);
    let h = hash23(vec2f(c, seed));
    if (h.x > 0.75) { continue; }
    let s = sc * (0.75 + 0.5 * h.z);
    let base = vec2f((c + 0.3 + 0.4 * h.y) * cellW, gy + 0.01);
    let fork = base + vec2f(0.01 * s, -0.1 * s);
    let top = base + vec2f(0.0, -0.2 * s);
    var dt = sdTaper(q, base, fork, 0.012 * s, 0.008 * s);
    dt = min(dt, sdTaper(q, fork, top + vec2f(-0.06 * s, 0.0), 0.007 * s, 0.004 * s));
    dt = min(dt, sdTaper(q, fork, top + vec2f(0.07 * s, 0.01 * s), 0.007 * s, 0.004 * s));
    // umbrella canopy: flattened ellipses with a leafy noisy edge
    var cd = sdEllipseApprox(q - (top + vec2f(0.0, -0.01 * s)), vec2f(0.17, 0.028) * s);
    cd = min(cd, sdEllipseApprox(q - (top + vec2f(-0.07 * s, 0.008 * s)), vec2f(0.09, 0.02) * s));
    cd = min(cd, sdEllipseApprox(q - (top + vec2f(0.08 * s, 0.01 * s)), vec2f(0.08, 0.018) * s));
    cd += (valueNoise(q * 140.0) - 0.5) * 0.008 * s;
    d = min(d, min(dt, cd));
  }
  return d;
}

fn birdD(p: vec2f, c: vec2f, s: f32, flap: f32) -> f32 {
  let q = (p - c) / s;
  let ax = abs(q.x);
  let elbow = vec2f(0.5, -0.25 - 0.35 * flap);
  let tip = vec2f(1.0, 0.05 - 0.6 * flap);
  let qq = vec2f(ax, q.y);
  let w = min(sdTaper(qq, vec2f(0.0, 0.05), elbow, 0.09, 0.06), sdTaper(qq, elbow, tip, 0.06, 0.01));
  return min(w, length(q * vec2f(1.0, 1.6) - vec2f(0.0, 0.1)) - 0.12) * s;
}

fn sunset(p: vec2f, uv: vec2f) -> vec3f {
  let t = u.time;
  let aspect = u.resolution.x / u.resolution.y;
  let pw = 1.0 / u.resolution.y;
  let cam = t * u.drift * 0.05;
  var col = sunsetBack(p, aspect, cam);
  // lake: mirror the background with ripples + sun glitter
  let lakeTop = 0.7;
  if (p.y > lakeTop) {
    let ry = 2.0 * lakeTop - p.y;
    let rip = 0.004 * sin(p.y * 260.0 - t * 2.0) * (p.y - lakeTop) * 6.0;
    var rc = sunsetBack(vec2f(p.x + rip, ry), aspect, cam) * vec3f(0.75, 0.68, 0.8);
    let sx = aspect * 0.6;
    let glit = smoothstep(0.55, 1.0, valueNoise(vec2f(p.x * 90.0, p.y * 500.0 - t * 3.0))) * exp(-abs(p.x - sx) * 9.0);
    rc += vec3f(1.0, 0.75, 0.45) * glit * 0.9 * u.light;
    col = mix(col, rc, cover(lakeTop - p.y, pw * 1.5));
  }
  // birds
  for (var i = 0; i < 7; i++) {
    let fi = f32(i);
    let h = hash13(fi * 3.7 + 1.0);
    let bx = fract(h.x + t * (0.012 + 0.01 * h.y)) * (aspect + 0.4) - 0.2;
    let by = 0.16 + 0.22 * h.z + 0.012 * sin(t * 0.8 + fi);
    let flap = sin(t * (6.0 + 3.0 * h.y) + fi * 2.0);
    let bd = birdD(p, vec2f(bx, by), 0.012 + 0.01 * h.y, flap);
    col = mix(col, vec3f(0.08, 0.02, 0.08), cover(bd, pw * 1.2));
  }
  // near shore: black ground, acacias, grass
  let q = vec2f(p.x + cam, p.y);
  let gy = groundY(q.x, 0.8, 0.03, 2.0);
  var d = gy - grassH(q.x + (gy - p.y) * 0.4, 6.0, 0.022) - p.y;
  d = min(d, acaciaD(q, 0.62, 4.0, gy, 1.15));
  col = mix(col, vec3f(0.03, 0.008, 0.035), cover(d, pw * 1.2));
  // foreground tall grass, slightly blurred
  let q2 = vec2f(p.x + cam * 1.6, p.y);
  let fy = 1.0 - grassH(q2.x * 0.5 + (1.0 - p.y) * 0.2, 9.0, 0.12) - 0.02 * valueNoise(vec2f(q2.x * 3.0, 0.0));
  col = mix(col, vec3f(0.0), cover(fy - p.y, pw * (2.0 + 6.0 * u.blur)));
  return col;
}

// ---------------------------------------------------------------------------------- Inside-style
fn insideScene(p: vec2f) -> vec3f {
  let t = u.time;
  let aspect = u.resolution.x / u.resolution.y;
  let pw = 1.0 / u.resolution.y;
  let cam = t * u.drift * 0.05;
  let fogC = mix(vec3f(0.6, 0.65, 0.63), vec3f(0.4, 0.45, 0.46), p.y * 0.6) + vec3f(0.05) * (valueNoise(vec2f(p.x * 2.0 + t * 0.02, p.y * 4.0)) - 0.5);
  var col = fogC;
  // far: factory silhouette & chimneys
  {
    let x = p.x + cam * 0.12;
    let cellX = floor(x / 0.5);
    let fx = fract(x / 0.5) * 0.5;
    let hb = hash21(vec2f(cellX, 1.0));
    let roof = 0.6 - 0.12 * hb - 0.025 * fract(fx * 12.0) * step(0.5, hb);   // flat or saw-tooth roofs
    var d = max(roof - p.y, p.y - 0.85);                                        // building blocks
    d = max(d, abs(fx - 0.25) - 0.22 - 0.03 * hb);
    let chim = max(abs(fx - 0.15 - 0.2 * hb) - 0.012, p.y - roof);
    d = min(d, max(chim, (0.3 + 0.1 * hb) - p.y));
    let smoke = smoothstep(0.55, 0.8, valueNoise(vec2f(x * 6.0 - t * 0.1, p.y * 4.0 + t * 0.2))) * (1.0 - smoothstep(0.1, 0.32, p.y)) * step(abs(fx - 0.15 - 0.2 * hb), 0.04 + (0.3 - p.y) * 0.3);
    col = mix(col, mix(vec3f(0.24, 0.27, 0.28), fogC, 0.62 * u.fog), cover(d, pw * 3.0));
    col = mix(col, fogC * 1.08, smoke * 0.5);
  }
  // power poles with sagging wires
  {
    let x = p.x + cam * 0.3;
    let sp = 0.6;
    let cx = floor(x / sp);
    let fx = x - cx * sp;
    var d = max(abs(fx - 0.3) - 0.004, max(0.2 - p.y, p.y - 0.8));
    d = min(d, max(abs(p.y - 0.23) - 0.003, abs(fx - 0.3) - 0.035));
    for (var k = 0; k < 2; k++) {
      let wy = 0.235 + f32(k) * 0.015;
      let s = (fx - 0.3) / sp;
      let sag = 0.06 * (0.25 - (fract(s + 0.5) - 0.5) * (fract(s + 0.5) - 0.5)) * 4.0;
      d = min(d, abs(p.y - wy - sag) - 0.0012);
    }
    col = mix(col, mix(vec3f(0.17, 0.19, 0.2), fogC, 0.45 * u.fog), cover(d, pw * 2.0));
  }
  // mid: straight pine trunks
  {
    let q = vec2f(p.x + cam * 0.5 + 3.0, p.y);
    let gy = groundY(q.x, 0.8, 0.02, 3.0);
    let d = min(polesD(q, 0.06, 5.0, 0.004, gy, 0.7), gy - p.y);
    col = mix(col, mix(vec3f(0.12, 0.14, 0.15), fogC, 0.38 * u.fog), cover(d, pw * (1.5 + 2.0 * u.blur)));
  }
  // sweeping searchlight cone through the fog
  {
    let src = vec2f(aspect * 0.98, -0.15);
    let a = 2.0 + 0.32 * sin(t * 0.35);
    let dir = vec2f(cos(a), sin(a));
    let v = p - src;
    let along = dot(v, dir);
    let perp = abs(v.x * dir.y - v.y * dir.x);
    let cone = (1.0 - smoothstep(0.03, 0.11, perp / max(along, 0.01))) * step(0.0, along);
    let haze = 0.6 + 0.4 * valueNoise(vec2f(along * 6.0 - t * 0.3, perp * 20.0));
    col += vec3f(0.85, 0.92, 0.95) * cone * haze * 0.32 * u.light * exp(-along * 0.6);
  }
  // near: ground, grass, a few dark trunks, the boy in a red shirt
  {
    let q = vec2f(p.x + cam, p.y);
    let gy = groundY(q.x, 0.86, 0.02, 7.0);
    var d = gy - grassH(q.x + (gy - p.y) * 0.4, 2.0, 0.02) - p.y;
    d = min(d, polesD(q, 0.5, 9.0, 0.014, gy, 0.5));
    let bx = aspect * 0.42;
    let by = groundY(bx + cam, 0.86, 0.02, 7.0) + 0.003;
    let boy = boyD(p - vec2f(bx, by), 0.085, t * u.drift * 6.0);
    let dark = mix(vec3f(0.06, 0.07, 0.08), fogC, 0.1 * u.fog);
    col = mix(col, dark, cover(d, pw * 1.2));
    col = mix(col, vec3f(0.1, 0.11, 0.12), cover(boy.x, pw * 1.2));
    // the single color accent: the shirt
    col = mix(col, u.accent * 0.85, cover(boy.z, pw * 1.2));
  }
  // foreground blur: dark grass along the bottom
  let fy = 1.03 - grassH(p.x * 0.4 + cam * 0.7, 4.0, 0.08);
  col = mix(col, vec3f(0.02, 0.025, 0.03), cover(fy - p.y, pw * (2.0 + 10.0 * u.blur)));
  // muted grade: low saturation, slightly green-cyan shadows
  col = mix(vec3f(luma(col)), col, 0.85);
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  let p = px / u.resolution.y;
  var col = vec3f(0.0);
  var vk = 1.0;
  if (ex == 0) { col = limbo(p, px); vk = 1.4; }
  else if (ex == 1) { col = layered(p); vk = 0.5; }
  else if (ex == 2) { col = sunset(p, uv); }
  else { col = insideScene(p); }
  return vec4f(finish(col, uv, px, vk), 1.0);
}`;

export default shaderScene({
  examples: [
    {
      id: 'limbo',
      label: 'Limbo-style forest',
      kind: 'In a game',
      note: 'Pure black shapes, layered in depth. Fog pushes far layers toward the bright background, far & foreground layers are blurred (cheap depth of field), and grain, flicker and a heavy vignette make it look like old film. Only the eyes are white.',
    },
    {
      id: 'layers',
      label: 'Layers + fog = depth',
      kind: 'Abstract',
      note: 'The whole trick in its simplest form: N flat silhouettes, each mixed toward the fog color by <code>1 − exp(−fog × distance)</code> and scrolled at a different parallax speed. Turn on <i>Tint layers</i> to see them separately.',
      params: { fog: 0.55, drift: 0.8, blur: 0.4, grain: 0.03, vignette: 0.6 },
    },
    {
      id: 'sunset',
      label: 'Sunset silhouettes',
      kind: 'In a game',
      note: 'Same technique, warm palette: a graded sky, purple far hills that take on sky color, a lake that mirrors the background, and a pitch-black foreground with acacias and birds.',
      params: { fog: 0.6, drift: 0.5, grain: 0.04, vignette: 0.7, light: 1.0 },
    },
    {
      id: 'inside',
      label: 'Inside-style (one accent)',
      kind: 'In a game',
      note: 'Playdead’s INSIDE uses a muted, foggy palette where the only saturated thing is the boy’s red shirt — your eye always finds him. A sweeping searchlight lights up the fog volume.',
      params: { fog: 0.7, drift: 0.45, grain: 0.05, vignette: 0.9, light: 1.0, blur: 0.5 },
    },
  ],
  controls: [
    { type: 'slider', key: 'fog', label: 'Fog density', min: 0, max: 1.2, step: 0.01, value: 0.75, help: 'How strongly distant layers fade into the fog color (aerial perspective).' },
    { type: 'slider', key: 'drift', label: 'Parallax drift', min: 0, max: 2.5, step: 0.01, value: 0.6, help: 'Camera speed. Each layer scrolls at its own rate: far = slow, near = fast.' },
    { type: 'slider', key: 'layers', label: 'Number of layers', min: 2, max: 8, step: 1, value: 6, showFor: ['layers'] },
    { type: 'toggle', key: 'tint', label: 'Tint layers (debug view)', value: false, showFor: ['layers'], help: 'Give every layer its own hue to see how the image is built.' },
    { type: 'slider', key: 'light', label: 'Backlight & rays', min: 0, max: 2, step: 0.01, value: 1, showFor: ['limbo', 'sunset', 'inside'], help: 'Light glow, god rays, dust motes / searchlight / sun glitter.' },
    { type: 'slider', key: 'sunHeight', label: 'Sun height', min: 0, max: 1, step: 0.01, value: 0.3, showFor: ['sunset'] },
    { type: 'color', key: 'accent', label: 'Accent color', value: '#d8261b', showFor: ['inside'], help: 'The one saturated color in the frame.' },
    { type: 'slider', key: 'blur', label: 'Depth-of-field blur', min: 0, max: 1.5, step: 0.01, value: 0.6, help: 'Far and very near layers get a wider anti-aliasing edge — a nearly free fake blur.' },
    { type: 'slider', key: 'grain', label: 'Film grain', min: 0, max: 0.3, step: 0.005, value: 0.07 },
    { type: 'slider', key: 'vignette', label: 'Vignette', min: 0, max: 2, step: 0.01, value: 0.9 },
    { type: 'slider', key: 'flicker', label: 'Projector flicker', min: 0, max: 1, step: 0.01, value: 0.4, showFor: ['limbo'] },
  ],
  uniforms: {
    fog: 'f32', drift: 'f32', layers: 'f32', tint: 'f32', light: 'f32', sunHeight: 'f32', accent: 'vec3f',
    blur: 'f32', grain: 'f32', vignette: 'f32', flicker: 'f32',
  },
  bind(p, ctx) {
    // flicker only exists in the Limbo example
    return { flicker: ctx.example === 'limbo' ? p.flicker : 0 };
  },
  include: ['hash', 'noise', 'sdf', 'color', 'math'],
  code: CODE,
  about: {
    summary: 'Flat black shapes + fog that grows with distance = instant mood and depth. The look of Limbo, Inside and every sunset poster.',
    what: `<p>Every layer here is a single flat value — no textures, no lighting on the shapes themselves. Depth comes entirely from
      <b>how much fog</b> is mixed into each layer, <b>how fast</b> it scrolls (parallax) and <b>how blurry</b> its edges are.</p>`,
    how: `<ol>
      <li><b>Shapes as distance functions.</b> Trees are tapered capsules (trunk + branches, randomised per cell), grass is a row of sharp spikes
        added to the ground line, vines are wavy lines hanging from the top, the boy is a dozen capsules animated with <code>sin(walkPhase)</code>.</li>
      <li><b>Paint back to front.</b> <code>col = mix(col, layerColor, coverage)</code> for each layer, farthest first.</li>
      <li><b>Aerial perspective.</b> <code>layerColor = mix(black, fogColor, 1 − exp(−fog × depth))</code> — distant layers become almost the sky color.</li>
      <li><b>Parallax.</b> Each layer samples its shapes at <code>x + camera × speed</code>, with speed growing toward the viewer.</li>
      <li><b>Fake depth of field.</b> The edge coverage uses <code>clamp(0.5 − d / w)</code>; a wider <code>w</code> on far and foreground layers looks like lens blur at zero cost.</li>
      <li><b>Atmosphere.</b> God rays are noise sampled by <i>angle</i> around the light; fog wisps are drifting value noise near each layer’s ground.</li>
      <li><b>Film finish.</b> Per-frame random grain, a brightness flicker that changes 18 times a second, and a strong vignette.</li>
    </ol>`,
    uses: [
      { title: 'Atmospheric platformers', text: 'Limbo and Inside (Playdead), Little Nightmares, Black The Fall, Feist and Unravel use silhouettes, fog and grain to tell stories without words.' },
      { title: 'Mobile & minimal games', text: 'Badland, Alto’s Adventure / Odyssey and Sword & Sworcery rely on layered silhouettes over graded skies — cheap to render, gorgeous at any resolution.' },
      { title: 'Readability', text: 'A silhouette-first design makes gameplay readable: the player, hazards and platforms are clear shapes against a bright background.' },
      { title: 'Menus & loading screens', text: 'A slowly drifting parallax silhouette landscape is a classic, inexpensive title-screen backdrop.' },
    ],
    try: [
      'On <b>Layers + fog</b>, turn on <i>Tint layers</i>, then slide <i>Fog density</i> from 0 to max — watch the far layers dissolve into the sky.',
      'Set <i>Depth-of-field blur</i> to 0 on the Limbo forest: everything gets sharp and the depth illusion weakens noticeably.',
      'Push <i>Parallax drift</i> high: near layers race by, distant ones barely move — your brain reads that as depth.',
      'On <b>Inside-style</b>, change the <i>Accent color</i>: one saturated color in a grey world steers the eye.',
      'Set <i>Film grain</i> and <i>Flicker</i> to 0 — the image instantly looks more “digital”.',
    ],
    ask: [
      'Limbo-style black silhouettes with layered fog and film grain',
      'aerial perspective: fade distant parallax layers into the fog color',
      'fake depth-of-field by blurring far and foreground layers',
      'god rays and dust motes behind silhouettes',
      'a muted palette with a single accent color for the player',
      'sunset silhouette parallax background with birds',
    ],
    perf: `<p>Cheap to moderate: one full-screen pass, ~60 distance evaluations per pixel (trees check three neighbouring cells × trunk and branches).
      Cost scales with the number of tree layers and branches, not with the screen content. In a real game you would usually draw each layer
      once into a texture (or use sprites) and just scroll them — then it’s nearly free.</p>`,
    api: `<p>Pure fragment-shader work, identical on WebGPU and WebGL2. Nothing here needs compute.</p>`,
    code: [
      {
        title: 'Painting a fogged layer',
        lang: 'wgsl',
        src: `let q = vec2f(p.x + cam * par, p.y);                   // parallax: each layer scrolls at its own speed
let gy = groundY(q.x, base, 0.03, seed);                // the ground line of this layer
var d = min(treesD(q, cellW, seed, thick, gy, 0.6, 1.0), gy - p.y);
let w = pw * (1.5 + u.blur * blurAmount);               // wider edge = depth-of-field blur
let val = mix(vec3f(0.02), fogC, u.fog * depth);        // aerial perspective
col = mix(col, val, cover(d, w));                       // cover(d, w) = clamp(0.5 - d / w, 0, 1)`,
      },
      {
        title: 'Grass = spikes on the ground line',
        lang: 'wgsl',
        src: `fn grassH(x: f32, seed: f32, hgt: f32) -> f32 {
  let g1 = x * 150.0;
  let h1 = hash21(vec2f(floor(g1), seed));               // random height per blade
  return hgt * (0.25 + 0.75 * h1 * h1) * pow(max(0.0, 1.0 - abs(fract(g1) - 0.5) * 2.0), 3.0);
}
// lean the blades by shifting x with height:
let gg = gy - grassH(q.x + (gy - p.y) * 0.5, 3.0, 0.03);`,
      },
      {
        title: 'Film finish',
        lang: 'wgsl',
        src: `c *= clamp(1.0 - dot(v, v) * u.vignette, 0.0, 1.0);          // vignette
c += vec3f(hash21(px + fract(u.time * 7.13) * 931.0) - 0.5) * u.grain;  // grain
c *= 1.0 + u.flicker * 0.12 * (hash11(floor(u.time * 18.0)) - 0.5);    // flicker`,
      },
    ],
  },
});

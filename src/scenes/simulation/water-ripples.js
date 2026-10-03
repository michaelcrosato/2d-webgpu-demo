import { shaderScene } from '../../core/shaderscene.js';

// Water ripples: the 2D wave equation on a grid (two-state "ping-pong" height field),
// rendered with refraction, specular highlights and curvature caustics.
// One portable-WGSL source → runs on WebGPU and (translated) WebGL2.

// ------------------------------------------------------------------------------------------
// Shared WGSL used by BOTH the simulation pass and the final image (walls, fish, rocks …).
// q = aspect-correct coordinates: (0,0) = screen center, y in [-0.5, 0.5] (down), x in ±aspect/2.
// ------------------------------------------------------------------------------------------
const COMMON = /* wgsl */ `
const WATERLINE: f32 = 0.64;   // rain example: water below this uv.y
const NFISH: i32 = 6;

fn segDist(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.0000001), 0.0, 1.0);
  return length(pa - ba * h);
}

// ---- ripple tank: a wall with slits (signed distance, < 0 = solid) -------------------
fn tankWall(q: vec2f, aspect: f32, mode: i32, sep: f32) -> f32 {
  if (mode == 0) { return 1.0; }
  let wx = -aspect * 0.5 + aspect * 0.42;
  let dx = abs(q.x - wx) - 0.007;
  var opening = abs(q.y) - 0.022;                                   // single slit
  if (mode == 1) { opening = min(abs(q.y - sep * 0.5), abs(q.y + sep * 0.5)) - 0.012; } // double slit
  return max(dx, -opening);
}

// ---- koi pond: rocks (walls for the waves) ---------------------------------------------
fn rockSD(q: vec2f, aspect: f32) -> f32 {
  let hx = aspect * 0.5;
  var d = length(q - vec2f(-hx - 0.02, 0.52)) - 0.19;
  d = min(d, length(q - vec2f(-hx + 0.17, 0.5)) - 0.085);
  d = min(d, length(q - vec2f(-hx + 0.05, 0.3)) - 0.07);
  d = min(d, length(q - vec2f(hx + 0.03, -0.5)) - 0.17);
  d = min(d, length(q - vec2f(hx - 0.16, -0.5)) - 0.075);
  return d + 0.012 * (valueNoise(q * 22.0) - 0.5) + 0.006 * (valueNoise(q * 60.0) - 0.5);
}

// ---- koi: each fish follows a smooth looping path; body = chain of tapered capsules ------
fn koiPath(i: f32, t: f32, aspect: f32) -> vec2f {
  let ph = hash11(i * 7.31 + 1.0) * TAU;
  let sp = 0.7 + 0.5 * hash11(i * 3.17 + 2.0);
  let ax = aspect * 0.5 * (0.5 + 0.3 * hash11(i * 1.9 + 3.0));
  let ay = 0.26 + 0.1 * hash11(i * 5.3 + 4.0);
  let tt = t * 0.11 * sp + ph;
  var dirx = 1.0;
  if (hash11(i * 2.3) > 0.5) { dirx = -1.0; }
  return vec2f(dirx * ax * sin(tt + 0.5 * sin(tt * 0.7 + i)), ay * sin(tt * 1.43 + ph * 0.5 + 0.4 * cos(tt * 0.5)));
}

fn koiLen(i: f32) -> f32 { return 0.12 * (0.8 + 0.45 * hash11(i * 9.1 + 5.0)); }

fn koiRadius(s: f32, L: f32) -> f32 {
  return L * max(0.15 * pow(sin(PI * (0.1 + 0.9 * s)), 0.75), 0.022);
}

// returns (body distance, s along body 0..1, side -1..1, fin distance)
fn koiDist(q: vec2f, i: f32, t: f32, aspect: f32) -> vec4f {
  let head = koiPath(i, t, aspect);
  let L = koiLen(i);
  if (length(q - head) > L * 1.45) { return vec4f(1.0, 0.0, 0.0, 1.0); }
  let d0 = normalize(koiPath(i, t + 0.05, aspect) - head + vec2f(0.00001, 0.0));
  let tb = t - 1.6;
  let d1 = normalize(koiPath(i, tb + 0.05, aspect) - koiPath(i, tb, aspect) + vec2f(0.00001, 0.0));
  var best = 100.0;
  var bs = 0.0;
  var bside = 0.0;
  var prev = head;
  var prevR = koiRadius(0.0, L);
  var tailDir = d0;
  var finD = 100.0;
  for (var k = 1; k <= 6; k++) {
    let s = f32(k) / 6.0;
    let dir = normalize(mix(d0, d1, s * s));
    let perp = vec2f(-dir.y, dir.x);
    let wig = L * 0.07 * s * s * sin(t * 4.5 + i * 1.7 - s * 4.0);
    let sp = head - dir * (L * s) + perp * wig;
    let r = koiRadius(s, L);
    let pa = q - prev;
    let ba = sp - prev;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.0000001), 0.0, 1.0);
    let off = pa - ba * h;
    let rr = mix(prevR, r, h);
    let dd = length(off) - rr;
    if (dd < best) {
      best = dd;
      bs = (f32(k - 1) + h) / 6.0;
      let cr = ba.x * off.y - ba.y * off.x;
      bside = clamp(sign(cr) * length(off) / max(rr, 0.00001), -1.0, 1.0);
    }
    if (k == 2) {
      // pectoral fins, flapping
      let flap = 0.6 + 0.4 * sin(t * 3.0 + i);
      let fa = sp + perp * r * 0.8;
      let fb = sp - perp * r * 0.8;
      finD = min(finD, segDist(q, fa, fa + (perp * 0.22 * flap - dir * 0.16) * L) - L * 0.035);
      finD = min(finD, segDist(q, fb, fb + (-perp * 0.22 * flap - dir * 0.16) * L) - L * 0.035);
    }
    prev = sp;
    prevR = r;
    tailDir = dir;
  }
  // forked tail fin
  let tp = vec2f(-tailDir.y, tailDir.x);
  let sw = sin(t * 4.5 + i * 1.7 - 4.0);
  let tipA = prev - tailDir * L * 0.32 + tp * L * (0.17 + 0.05 * sw);
  let tipB = prev - tailDir * L * 0.32 - tp * L * (0.17 - 0.05 * sw);
  finD = min(finD, segDist(q, prev, tipA) - L * 0.04);
  finD = min(finD, segDist(q, prev, tipB) - L * 0.04);
  return vec4f(best, bs, bside, finD);
}

// ---- lily pads: (center.xy, radius, notch angle). x is in units of aspect/2. -------------
fn padInfo(i: i32) -> vec4f {
  var P = array<vec4f, 6>(
    vec4f(-0.62, -0.26, 0.085, 0.6),
    vec4f(-0.42, -0.34, 0.055, 2.4),
    vec4f(-0.78, -0.05, 0.06, 4.0),
    vec4f(0.55, 0.22, 0.1, 3.6),
    vec4f(0.74, 0.08, 0.06, 1.2),
    vec4f(0.36, 0.34, 0.065, 5.1));
  return P[i];
}

fn padSD(q: vec2f, i: i32, aspect: f32, t: f32) -> vec2f {
  let P = padInfo(i);
  let fi = f32(i);
  let c = vec2f(P.x * aspect * 0.5, P.y) + 0.006 * vec2f(sin(t * 0.21 + fi), cos(t * 0.17 + fi * 2.0));
  let rot = P.w + 0.08 * sin(t * 0.1 + fi);
  let d = q - c;
  let a = atan2(d.y, d.x) - rot;
  let aw = abs(atan2(sin(a), cos(a)));
  let circle = length(d) - P.z;
  // notch: a thin wedge cut from the center to the rim
  let notch = (0.16 - aw) * length(d);
  return vec2f(max(circle, notch), aw);
}
`;

// ------------------------------------------------------------------------------------------
// Simulation pass: rgba16float, r = height now, g = height one step ago.
// ------------------------------------------------------------------------------------------
const SIM = /* wgsl */ `
${COMMON}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  if (u.frame < 0.5) { return vec4f(0.0); }
  let ip = vec2i(px);
  let c = LOAD(sim, ip);
  let h = c.r;
  let hp = c.g;
  // 4 neighbours (LOAD clamps at the border = reflecting edges)
  let n = LOAD(sim, ip + vec2i(0, -1)).r;
  let s = LOAD(sim, ip + vec2i(0, 1)).r;
  let e = LOAD(sim, ip + vec2i(1, 0)).r;
  let w = LOAD(sim, ip + vec2i(-1, 0)).r;
  let lap = n + s + e + w - 4.0 * h;               // discrete Laplacian = "curvature"
  var hn = (2.0 * h - hp + u.speed * lap) * u.damping; // wave equation, explicit (Verlet) step

  let ex = i32(u.example + 0.5);
  let res = TEXSIZE(sim);
  let aspect = u.resolution.x / u.resolution.y;
  let q = centerUV(px, res);
  let pxScale = u.resolution.y;                   // q units -> canvas px

  // ---- finger: hold the water down along the mouse drag ----
  if (u.mouse.z > 0.5) {
    let m = centerUV(u.mouse.xy, u.resolution);
    let pm = centerUV(u.pmouse.xy, u.resolution);
    let d = segDist(q, pm, m) * pxScale;
    let k = exp(-(d * d) / (u.brush * u.brush));
    hn = mix(hn, -u.strength, k * 0.5);
  }

  // ---- random rain drops ----
  if (u.rain > 0.001 && ex > 0) {
    let chance = clamp(u.rain * 12.0 * u.dt, 0.0, 1.0);
    for (var i = 0; i < 3; i++) {
      let r = hash33(vec3f(u.frame, f32(i), 3.7));
      if (r.z < chance) {
        let rp = hash23(vec2f(u.frame * 1.618, f32(i) + 9.0));
        var dp = vec2f((rp.x - 0.5) * aspect, rp.y - 0.5);
        if (ex == 2) { dp.y = mix(WATERLINE + 0.03, 1.0, rp.y) - 0.5; }
        let d = length(q - dp) * pxScale;
        let rad = 2.5 + 3.0 * rp.z;
        hn = mix(hn, -0.9 - rp.z, exp(-(d * d) / (rad * rad)));
      }
    }
  }

  if (ex == 0) {
    // ---- ripple tank: oscillating sources + absorbing "sponge" edges + slit wall ----
    let mode = i32(u.tank + 0.5);
    let src = sin(u.frame * u.freq);
    if (mode == 0) {
      let sx = -aspect * 0.5 + aspect * 0.32;
      let d1 = length(q - vec2f(sx, -u.sep * 0.5)) * pxScale;
      let d2 = length(q - vec2f(sx, u.sep * 0.5)) * pxScale;
      hn = mix(hn, src, exp(-(d1 * d1) / 30.0));
      hn = mix(hn, src, exp(-(d2 * d2) / 30.0));
    } else {
      let lx = -aspect * 0.5 + 0.09;
      let d = abs(q.x - lx) * pxScale;
      hn = mix(hn, src * 0.8, exp(-(d * d) / 12.0));
    }
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    hn *= mix(0.9, 1.0, smoothstep(0.0, 0.05, edge));
    if (tankWall(q, aspect, mode, u.sep) < 0.0) { hn = 0.0; }
  } else if (ex == 1) {
    // ---- koi pond: fish leave gentle wakes; rocks are walls ----
    for (var i = 0; i < NFISH; i++) {
      let hd = koiPath(f32(i), u.time, aspect);
      let d = length(q - hd) * pxScale;
      hn -= 0.006 * exp(-(d * d) / 40.0);
    }
    if (rockSD(q, aspect) < 0.0) { hn = 0.0; }
  } else {
    // ---- rainy shore: no water above the waterline ----
    if (uv.y < WATERLINE) { hn = 0.0; }
  }
  return vec4f(hn, h, 0.0, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// Final image
// ------------------------------------------------------------------------------------------
const IMAGE = /* wgsl */ `
${COMMON}
fn hAt(uv: vec2f) -> f32 { return TEX(sim, uv).r; }
fn aaq(d: f32) -> f32 { return clamp(0.5 - d * u.resolution.y, 0.0, 1.0); }

fn koiColor(i: f32, s: f32, side: f32) -> vec3f {
  let ty = floor(hash11(i * 4.7 + 6.0) * 4.99);
  let white = vec3f(0.97, 0.94, 0.88);
  let red = vec3f(0.92, 0.28, 0.07);
  let blk = vec3f(0.05, 0.05, 0.07);
  let pn = valueNoise(vec2f(s * 7.0, side * 1.4) + vec2f(i * 17.0, i * 5.0));
  let pn2 = valueNoise(vec2f(s * 11.0, side * 2.5) + vec2f(i * 3.0, 40.0));
  var c = white;
  if (ty < 0.5) {
    c = mix(white, red, smoothstep(0.46, 0.52, pn));                         // kohaku
  } else if (ty < 1.5) {
    c = mix(vec3f(1.0, 0.66, 0.18), vec3f(1.0, 0.88, 0.5), smoothstep(0.3, 0.9, pn2)); // ogon (gold)
  } else if (ty < 2.5) {
    c = mix(white, red, smoothstep(0.45, 0.5, pn));                          // showa
    c = mix(c, blk, smoothstep(0.55, 0.6, pn2));
  } else if (ty < 3.5) {
    c = mix(white, red, 1.0 - smoothstep(0.42, 0.5, length(vec2f((s - 0.14) * 3.2, side * 0.8)))); // tancho
  } else {
    c = mix(vec3f(1.0, 0.45, 0.1), vec3f(1.0, 0.62, 0.25), pn2);             // orange
  }
  let rnd = sqrt(max(1.0 - side * side, 0.0));
  c *= 0.5 + 0.5 * rnd;
  c += vec3f(0.18, 0.16, 0.12) * pow(rnd, 10.0) * smoothstep(0.9, 0.2, s);   // wet sheen along the back
  // eyes
  let eye = length(vec2f((s - 0.07) * 9.0, abs(side) - 0.62));
  c = mix(c, vec3f(0.05), 1.0 - smoothstep(0.1, 0.18, eye));
  return c;
}

fn pondBed(q: vec2f, aspect: f32, t: f32) -> vec3f {
  let depth = 1.0 - smoothstep(0.15, 0.62, length(q / vec2f(aspect * 0.5, 0.5)) * 0.62);
  let vor = voronoi(q * 13.0);
  let cid = hash21(vor.zw);
  var peb = mix(vec3f(0.46, 0.42, 0.33), vec3f(0.33, 0.36, 0.31), cid);
  peb = mix(peb, vec3f(0.62, 0.56, 0.46), step(0.82, hash21(vor.zw + 3.1)));
  peb = mix(peb, vec3f(0.28, 0.22, 0.2), step(0.9, hash21(vor.zw + 7.7)));
  peb *= 0.45 + 0.55 * smoothstep(0.0, 0.3, vor.y - vor.x);
  let mud = fbm(q * 2.5 + vec2f(3.0, 1.0), 4);
  var c = mix(peb, vec3f(0.17, 0.25, 0.13), smoothstep(-0.15, 0.35, mud) * 0.75);
  c = mix(c, vec3f(0.02, 0.1, 0.12), depth * 0.7);
  return c;
}

// cheap animated "ambient" caustic network
fn ambientCaustic(q: vec2f, t: f32) -> f32 {
  let a = voronoiEx(q * 9.0 + vec2f(t * 0.05, 0.0), 1.0, t * 0.6);
  let b = voronoiEx(q * 14.0 + vec2f(0.0, t * 0.04) + 5.0, 1.0, -t * 0.5);
  let ca = 1.0 - smoothstep(0.0, 0.12, a.y - a.x);
  let cb = 1.0 - smoothstep(0.0, 0.1, b.y - b.x);
  return ca * 0.6 + cb * 0.4;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example + 0.5);
  let aspect = u.resolution.x / u.resolution.y;
  let q = centerUV(px, u.resolution);
  let t = u.time;
  let ts = 1.0 / TEXSIZE(sim);

  // ---- surface shape from the height field ----
  let h0 = hAt(uv);
  let hl = hAt(uv - vec2f(ts.x, 0.0));
  let hr = hAt(uv + vec2f(ts.x, 0.0));
  let hu = hAt(uv - vec2f(0.0, ts.y));
  let hd = hAt(uv + vec2f(0.0, ts.y));
  let grad = vec2f(hr - hl, hd - hu) * 0.5;       // slope
  let lap = hl + hr + hu + hd - 4.0 * h0;         // curvature: < 0 on crests (they focus light)
  let nrm = normalize(vec3f(-grad * 7.0, 1.0));
  let L = normalize(vec3f(-0.45, -0.6, 0.66));
  let Hv = normalize(L + vec3f(0.0, 0.0, 1.0));
  let spec = pow(max(dot(nrm, Hv), 0.0), 160.0) * 3.0 + pow(max(dot(nrm, Hv), 0.0), 24.0) * 0.12;
  let caust = clamp(-lap * 9.0, -0.6, 2.0);

  if (u.showHeight > 0.5) {
    let v = clamp(h0 * 1.5, -1.0, 1.0);
    var c = mix(vec3f(0.04, 0.05, 0.07), mix(vec3f(0.2, 0.45, 1.0), vec3f(1.0, 0.45, 0.2), step(0.0, v)), abs(v));
    c += vec3f(0.12) * (1.0 - smoothstep(0.0, 0.06, abs(fract(h0 * 4.0 + 0.5) - 0.5)));
    return vec4f(c, 1.0);
  }

  if (ex == 0) {
    // ---------------------------------------------------------------- ripple tank
    let mode = i32(u.tank + 0.5);
    var col = mix(vec3f(0.01, 0.04, 0.1), vec3f(0.08, 0.42, 0.62), clamp(0.5 + h0 * 0.55, 0.0, 1.0));
    col *= 0.85 + 0.3 * dot(nrm, L);
    // a real ripple tank projects light through the water: crests focus it into bright bands
    col += vec3f(0.45, 0.85, 1.0) * clamp(caust * 0.35, 0.0, 1.0) * u.caustics;
    col += vec3f(1.0) * spec * u.shine * 0.4;
    // faint grid on the tank floor
    let g = abs(fract(q * 10.0) - 0.5);
    col += vec3f(0.03, 0.06, 0.08) * (1.0 - smoothstep(0.0, 0.02, min(g.x, g.y)));
    // wall
    let wd = tankWall(q, aspect, mode, u.sep);
    col = mix(col, vec3f(0.0), aaq(wd - 0.004) * 0.5);
    col = mix(col, mix(vec3f(0.55, 0.6, 0.68), vec3f(0.8, 0.84, 0.9), smoothstep(0.004, -0.004, wd + 0.003)), aaq(wd));
    // sources
    let ph = 0.5 + 0.5 * sin(u.frame * u.freq);
    if (mode == 0) {
      let sx = -aspect * 0.5 + aspect * 0.32;
      let d = min(length(q - vec2f(sx, -u.sep * 0.5)), length(q - vec2f(sx, u.sep * 0.5)));
      col += vec3f(1.0, 0.8, 0.4) * exp(-d * 90.0) * (0.6 + 0.6 * ph);
      col = mix(col, vec3f(1.0, 0.95, 0.8), aaq(d - 0.006));
    } else {
      let lx = -aspect * 0.5 + 0.09;
      let d = abs(q.x - lx);
      col += vec3f(1.0, 0.8, 0.4) * exp(-d * 120.0) * (0.4 + 0.5 * ph);
      col = mix(col, vec3f(1.0, 0.95, 0.8), aaq(d - 0.003) * 0.9);
    }
    // vignette
    col *= 1.0 - 0.35 * dot(uv - 0.5, uv - 0.5);
    return vec4f(col, 1.0);
  }

  if (ex == 1) {
    // ---------------------------------------------------------------- koi pond
    let qb = q - nrm.xy * u.refraction / u.resolution.y;   // refraction: look at the bed through a tilted surface
    var col = pondBed(qb, aspect, t);
    // shadows of fish and lily pads on the bed (light from the top-left -> shadows to the bottom-right)
    let so = vec2f(0.022, 0.03);
    var shadow = 0.0;
    for (var i = 0; i < NFISH; i++) {
      let k = koiDist(qb - so, f32(i), t, aspect);
      shadow = max(shadow, 1.0 - smoothstep(-0.004, 0.012, min(k.x, k.w + 0.004)));
    }
    for (var i = 0; i < 6; i++) {
      let pd = padSD(qb - so * 1.6, i, aspect, t);
      shadow = max(shadow, 1.0 - smoothstep(-0.006, 0.02, pd.x));
    }
    col *= 1.0 - 0.45 * shadow;
    // fish (they swim under the surface, so they are refracted too)
    for (var i = 0; i < NFISH; i++) {
      let fi = f32(i);
      let k = koiDist(qb, fi, t, aspect);
      if (k.x < 0.05) {
        let fc = koiColor(fi, 0.1, 0.0) * 0.6 + vec3f(0.35);
        col = mix(col, fc, aaq(k.w) * 0.45);
        col = mix(col, koiColor(fi, k.y, k.z), aaq(k.x));
      }
    }
    // light: ripple caustics + gentle ambient caustic net, then water absorption
    let lightOnBed = 1.0 + caust * 0.55 * u.caustics + ambientCaustic(qb, t) * 0.22 * u.caustics;
    col *= lightOnBed * (1.0 - 0.6 * shadow * 0.0);
    col = mix(col, vec3f(0.03, 0.17, 0.2), 0.32);
    // surface reflections: sky sheen (fresnel grows as the surface tilts) + sun glints
    let fres = 0.04 + 0.6 * pow(1.0 - nrm.z, 1.5);
    col += vec3f(0.55, 0.7, 0.8) * fres * 0.5;
    col += vec3f(1.0, 0.97, 0.9) * spec * u.shine;

    // lily pads float ON the surface: not refracted, but tilted by the waves
    for (var i = 0; i < 6; i++) {
      let pd = padSD(q, i, aspect, t);
      if (pd.x < 0.01) {
        let P = padInfo(i);
        let c = vec2f(P.x * aspect * 0.5, P.y);
        let rr = clamp(length(q - c) / P.z, 0.0, 1.0);
        let ang = atan2(q.y - c.y, q.x - c.x);
        var pc = mix(vec3f(0.36, 0.6, 0.2), vec3f(0.17, 0.4, 0.13), rr);
        pc *= 0.88 + 0.12 * smoothstep(0.0, 0.25, abs(sin(ang * 9.0 + f32(i))));   // veins
        pc = mix(pc, vec3f(0.52, 0.55, 0.22), smoothstep(-0.008, 0.0, pd.x) * 0.8); // curled rim
        pc *= 0.75 + 0.45 * max(dot(nrm, L), 0.0);
        pc += vec3f(0.9) * pow(max(dot(nrm, Hv), 0.0), 40.0) * 0.12;
        col = mix(col, vec3f(0.02, 0.06, 0.03), aaq(pd.x - 0.004) * 0.5);
        col = mix(col, pc, aaq(pd.x));
        // lotus flowers on two pads
        if (i == 0 || i == 3) {
          let fq = q - c - vec2f(0.012, -0.008);
          let fr = length(fq);
          let fa = atan2(fq.y, fq.x) + t * 0.05;
          let petal = P.z * 0.55 * (0.62 + 0.38 * abs(cos(fa * 4.0)));
          let inner = P.z * 0.38 * (0.6 + 0.4 * abs(cos(fa * 4.0 + 0.8)));
          var fc = mix(vec3f(1.0, 0.85, 0.92), vec3f(0.95, 0.42, 0.62), smoothstep(petal, petal * 0.3, fr));
          col = mix(col, vec3f(0.1, 0.15, 0.05), aaq(fr - petal - 0.004) * 0.4);
          col = mix(col, fc, aaq(fr - petal));
          fc = mix(vec3f(1.0, 0.93, 0.96), vec3f(1.0, 0.55, 0.7), smoothstep(inner, inner * 0.2, fr));
          col = mix(col, fc, aaq(fr - inner));
          col = mix(col, vec3f(1.0, 0.82, 0.25), aaq(fr - P.z * 0.1));
        }
      }
    }
    // rocks (above water)
    let rd = rockSD(q, aspect);
    if (rd < 0.02) {
      let rn = fbm(q * 9.0, 4);
      var rc = mix(vec3f(0.42, 0.4, 0.37), vec3f(0.6, 0.58, 0.52), rn * 0.5 + 0.5);
      rc = mix(rc, vec3f(0.24, 0.38, 0.16), smoothstep(0.1, 0.35, fbm(q * 5.0 + 7.0, 3)) * 0.8); // moss
      let rim = smoothstep(-0.05, 0.0, rd);
      rc *= 1.0 - 0.45 * rim;                       // wet dark band at the waterline
      // fake bevel lighting from the rock's distance gradient
      let e = 0.004;
      let gx = rockSD(q + vec2f(e, 0.0), aspect) - rd;
      let gy = rockSD(q + vec2f(0.0, e), aspect) - rd;
      let rnrm = normalize(vec3f(gx, gy, e * 1.2 * (1.0 + 4.0 * smoothstep(-0.0, -0.06, rd))));
      rc *= 0.55 + 0.6 * max(dot(rnrm, L), 0.0);
      col = mix(col, vec3f(0.85, 0.95, 1.0), (1.0 - smoothstep(0.0, 0.006, abs(rd - 0.003))) * 0.25); // foam line
      col = mix(col, rc, aaq(rd));
    }
    col *= 1.0 - 0.3 * dot(uv - 0.5, uv - 0.5);
    return vec4f(col, 1.0);
  }

  // ---------------------------------------------------------------- rainy shore (game scene + reflection)
  let shift = 0.8 - WATERLINE;      // the game's ground line (y = 0.8) sits on the waterline
  var col: vec3f;
  if (uv.y < WATERLINE) {
    col = TEX(game, vec2f(uv.x, uv.y + shift)).rgb;
  } else {
    // mirror the world about the waterline; ripples bend the reflection
    let off = -nrm.xy * u.refraction / u.resolution.y;
    let ry = 0.8 - (uv.y - WATERLINE) + off.y * 1.5;
    var refl = TEX(game, vec2f(uv.x + off.x, clamp(ry, 0.0, 0.8))).rgb;
    let depthK = smoothstep(WATERLINE, 1.0, uv.y);
    let fres = mix(0.75, 0.45, depthK);             // reflection fades as we look more "down" into the water
    let deep = mix(vec3f(0.08, 0.14, 0.2), vec3f(0.03, 0.07, 0.11), depthK);
    col = mix(deep, refl * vec3f(0.85, 0.92, 1.0), fres);
    col *= 1.0 + caust * 0.25 * u.caustics;
    col += vec3f(1.0, 0.95, 0.85) * spec * u.shine * 0.8;
    // bank: a dark muddy edge right at the waterline
    let bank = 1.0 - smoothstep(0.0, 0.012, uv.y - WATERLINE);
    col = mix(col, vec3f(0.12, 0.1, 0.08), bank * 0.85);
  }
  // overcast, rainy mood
  col = mix(col, vec3f(dot(col, vec3f(0.3, 0.55, 0.15))), 0.35) * vec3f(0.78, 0.84, 0.95);
  // falling rain streaks (three layers, slanted)
  var streaks = 0.0;
  for (var l = 0; l < 3; l++) {
    let fl = f32(l);
    let sc = vec2f(70.0 + fl * 45.0, 6.0 + fl * 3.0);
    var rq = vec2f(q.x + q.y * 0.12, q.y) * sc;
    rq.y -= t * (9.0 + fl * 4.0);
    let cell = floor(rq);
    let f = fract(rq);
    let rnd = hash22(cell + vec2f(fl * 13.0, 0.0));
    let on = step(1.0 - u.rain * 0.55, rnd.y);
    let line = (1.0 - smoothstep(0.0, 0.08, abs(f.x - 0.2 - 0.6 * rnd.x))) * smoothstep(0.0, 0.6, f.y) * (1.0 - smoothstep(0.75, 1.0, f.y));
    streaks += line * on * (0.35 - fl * 0.08);
  }
  col += vec3f(0.75, 0.82, 0.95) * streaks;
  col *= 1.0 - 0.35 * dot(uv - 0.5, uv - 0.5);
  return vec4f(col, 1.0);
}`;

export default shaderScene({
  interaction: 'Click & drag to touch the water. The ripples spread, reflect and interfere.',
  examples: [
    {
      id: 'tank',
      label: 'Ripple tank',
      kind: 'Abstract',
      note: 'A physics-class ripple tank. Two oscillating sources make an <b>interference</b> pattern; switch “Tank setup” to the slit walls to see <b>diffraction</b> — waves bending around obstacles. The bright bands are light focused by the curved surface.',
      params: { speed: 0.4, damping: 0.999, steps: 2, refraction: 20, shine: 1, rain: 0 },
    },
    {
      id: 'koi',
      label: 'Koi pond',
      kind: 'In a game',
      note: 'The same height field drives <b>refraction</b> of the pond bed (fish wobble as ripples pass), <b>caustics</b> on the pebbles and <b>specular</b> glints. The fish leave wakes, the rocks reflect the waves. Drag through the water!',
      params: { speed: 0.35, damping: 0.99, steps: 2, refraction: 26, shine: 1.2, rain: 0.08 },
    },
    {
      id: 'rain',
      label: 'Rain on a lake',
      kind: 'In a game',
      note: 'Rain drops are just random “pokes” into the height field. The water mirrors the game world (the procedural platformer scene) and the ripples distort the reflection — a classic look for side-scrollers.',
      params: { speed: 0.3, damping: 0.985, steps: 2, refraction: 18, shine: 1.4, rain: 0.7 },
    },
  ],
  controls: [
    { type: 'heading', label: 'Wave physics' },
    { type: 'slider', key: 'speed', label: 'Wave speed (c²)', min: 0.05, max: 0.5, step: 0.005, value: 0.35, help: 'How far a wave travels per step. 0.5 is the stability limit of this explicit solver — beyond it the simulation explodes.' },
    { type: 'slider', key: 'damping', label: 'Damping', min: 0.95, max: 1, step: 0.0005, value: 0.99, format: (v) => v.toFixed(4), help: 'Energy kept each step. 1.0 = waves bounce forever; lower = thick, syrupy water.' },
    { type: 'slider', key: 'steps', label: 'Steps per frame', min: 1, max: 8, step: 1, value: 2, help: 'Simulation sub-steps per frame (= faster waves, more GPU work).' },
    { type: 'slider', key: 'brush', label: 'Finger size', min: 3, max: 50, step: 1, value: 12, help: 'Radius (px) of the mouse disturbance.' },
    { type: 'slider', key: 'strength', label: 'Push depth', min: 0.2, max: 3, step: 0.05, value: 1.4 },
    { type: 'heading', label: 'Look' },
    { type: 'slider', key: 'refraction', label: 'Refraction', min: 0, max: 80, step: 1, value: 26, help: 'How much the tilted surface bends the view of what lies under (or is reflected in) the water.' },
    { type: 'slider', key: 'shine', label: 'Specular glints', min: 0, max: 3, step: 0.05, value: 1.2 },
    { type: 'toggle', key: 'caustics', label: 'Caustics', value: true, help: 'Light focused by curved crests: brightness from the surface curvature (Laplacian).' },
    { type: 'toggle', key: 'showHeight', label: 'Show raw height field', value: false, help: 'Debug view: orange = above rest level, blue = below.' },
    { type: 'heading', label: 'Sources', showFor: ['tank', 'koi', 'rain'] },
    {
      type: 'select',
      key: 'tank',
      label: 'Tank setup',
      value: 'two',
      showFor: ['tank'],
      options: [
        { value: 'two', label: 'Two point sources (interference)' },
        { value: 'double', label: 'Plane wave + double slit' },
        { value: 'single', label: 'Plane wave + single slit' },
      ],
    },
    { type: 'slider', key: 'freq', label: 'Source frequency', min: 0.05, max: 0.6, step: 0.005, value: 0.25, showFor: ['tank'], help: 'Higher = shorter wavelength = narrower interference fringes.' },
    { type: 'slider', key: 'sep', label: 'Source / slit separation', min: 0.04, max: 0.4, step: 0.005, value: 0.16, showFor: ['tank'] },
    { type: 'slider', key: 'rain', label: 'Rain', min: 0, max: 1, step: 0.01, value: 0.1, showFor: ['koi', 'rain'], help: 'Random drops per second.' },
    { type: 'button', key: 'reset', label: 'Calm the water', primary: true },
  ],
  uniforms: {
    speed: 'f32',
    damping: 'f32',
    steps: 'f32',
    brush: 'f32',
    strength: 'f32',
    refraction: 'f32',
    shine: 'f32',
    caustics: 'f32',
    showHeight: 'f32',
    tank: 'f32',
    freq: 'f32',
    sep: 'f32',
    rain: 'f32',
  },
  include: ['math', 'noise', 'color'],
  input: 'game',
  resetOn: ['tank'],
  passes: [{ name: 'sim', format: 'rgba16float', scale: 0.5, iterations: 'steps', code: SIM }],
  code: IMAGE,
  about: {
    summary:
      'Water ripples are the <b>wave equation</b> solved on a grid: each cell’s height accelerates toward the average of its neighbours. Two textures, a few lines of shader, and you get interference, reflection, refraction and caustics.',
    what: `<p>A grid of water heights (here at half the screen resolution) is updated every frame on the GPU. Everything you see — the
      wobbling pond bed, glints of light, bright caustic lines, the distorted reflection — is derived from that one height texture.</p>`,
    how: `<ol>
      <li><b>State</b>: each cell stores its height <i>now</i> and its height <i>one step ago</i> (two channels of an <code>rgba16float</code> texture).
        Velocity is implicit: <code>h − h<sub>prev</sub></code>. This is <b>Verlet integration</b>.</li>
      <li><b>Update</b> (a fragment “pass” that renders into the other half of a ping-pong pair):
        <code>h<sub>new</sub> = (2h − h<sub>prev</sub> + c²·∇²h) · damping</code>, where <code>∇²h</code> (the Laplacian) =
        sum of the 4 neighbours − 4h. Crests get pulled down, troughs pushed up, and the disturbance travels outward.</li>
      <li><b>Stability</b>: this explicit scheme only works when <code>c² ≤ 0.5</code> (the CFL condition). For faster waves, run more steps per frame instead.</li>
      <li><b>Input</b>: the mouse, rain drops, fish and wave sources simply overwrite heights inside a small Gaussian blob. Walls/rocks force the height to 0.</li>
      <li><b>Rendering</b>: the slope (finite differences of the height) gives a surface <b>normal</b>. The normal offsets the texture lookup of the
        bed or reflection (<b>refraction</b>), lights the surface (<b>specular</b> via a half-vector), and the curvature (Laplacian) brightens
        convex crests that act like tiny lenses (<b>caustics</b>).</li>
    </ol>`,
    uses: [
      { title: 'Ponds, puddles & lakes', text: 'Interactive water that reacts to the player, rain and projectiles — e.g. walking through shallows in top-down RPGs.' },
      { title: 'Reflections in side-scrollers', text: 'Mirror the level below a waterline and wobble it with ripples (Ori, Rayman Legends style).' },
      { title: 'Shockwave / impact distortion', text: 'The same height field can distort the whole screen: drop an impulse where an explosion happens.' },
      { title: 'UI & menus', text: 'A calm ripple background that responds to the cursor — a cheap, classy title screen.' },
    ],
    try: [
      'In the <b>Ripple tank</b>, switch to “double slit” and watch the waves fan out behind the slits and interfere. Change <i>Source frequency</i>: shorter waves → more fringes.',
      'Set <i>Damping</i> to 1.0 in the koi pond and splash around: the waves never die and the pond turns into chaos.',
      'Push <i>Wave speed</i> to 0.5 and <i>Steps per frame</i> to 8: very fast water. The CFL limit is why the slider stops at 0.5.',
      'Turn on <b>Show raw height field</b> to see the actual data the GPU is simulating.',
      'Turn <i>Refraction</i> to 0 in the koi pond: the ripples almost disappear — refraction and glints are what sell “water”.',
    ],
    ask: [
      'interactive 2D water ripples using the wave equation on the GPU',
      'refraction of the floor through a rippling water surface',
      'caustics from the curvature of a height field',
      'rain drops creating ripple rings on a puddle',
      'reflection of the level in water below a waterline, distorted by ripples',
      'the player leaves a wake when walking through water',
    ],
    perf: `<p>Per step, each cell reads 5 texels and does ~10 math operations: a 960×540 grid with 2 steps is ~1M cell updates per frame —
      far below 1 ms on any GPU. The render pass is the heavier part here (fish, pebbles and lily pads are all procedural).
      Simulating at half resolution and sampling with bilinear filtering is invisible in practice.</p>`,
    api: `<p>Works identically in <b>WebGL2</b> and <b>WebGPU</b>: it’s pure fragment-shader ping-pong between two float textures (WebGL2 needs
      <code>EXT_color_buffer_float</code>, which nearly every device has). A WebGPU compute version could store heights in a storage buffer and
      use workgroup shared memory to share neighbour reads, but for a simple 5-point stencil the gain is small.</p>`,
    code: [
      {
        title: 'Simulation step (one fragment per cell)',
        lang: 'wgsl',
        src: `let c = LOAD(sim, ip);            // r = height now, g = height one step ago
let h = c.r;  let hp = c.g;
let lap = LOAD(sim, ip + vec2i(0,-1)).r + LOAD(sim, ip + vec2i(0,1)).r
        + LOAD(sim, ip + vec2i(1,0)).r  + LOAD(sim, ip + vec2i(-1,0)).r - 4.0 * h;
var hn = (2.0 * h - hp + u.speed * lap) * u.damping;   // wave equation (Verlet step)
// finger / rain drop: pull the height toward a target inside a Gaussian blob
hn = mix(hn, -u.strength, exp(-(d * d) / (u.brush * u.brush)) * 0.5);
return vec4f(hn, h, 0.0, 1.0);    // new "now", and "now" becomes "previous"`,
      },
      {
        title: 'Shading: normal → refraction, glints, caustics',
        lang: 'wgsl',
        src: `let grad = vec2f(hr - hl, hd - hu) * 0.5;          // surface slope
let lap  = hl + hr + hu + hd - 4.0 * h0;            // curvature
let nrm  = normalize(vec3f(-grad * 7.0, 1.0));
let qb   = q - nrm.xy * u.refraction / u.resolution.y; // bent view ray
var col  = pondBed(qb, aspect, t);                  // look up the bed there
col *= 1.0 + clamp(-lap * 9.0, -0.6, 2.0) * 0.55;   // crests focus light
col += pow(max(dot(nrm, halfVec), 0.0), 160.0) * 3.0; // sun glints`,
      },
    ],
    links: [
      { title: 'Hugo Elias — 2D Water (archived classic)', url: 'https://web.archive.org/web/20160418004149/http://freespace.virgin.net/hugo.elias/graphics/x_water.htm', note: 'the original two-buffer ripple trick' },
      { title: 'Evan Wallace — WebGL Water', url: 'https://madebyevan.com/webgl-water/', note: 'heightfield water with real caustics' },
    ],
  },
});

import { shaderScene } from '../../core/shaderscene.js';

// Parallax scrolling & depth — every layer is a procedural silhouette at a distance z. The camera moves,
// each layer scrolls by camera × (1/z)^strength, and atmospheric fog pushes far layers toward the sky colour.
// One fragment shader, runs on WebGPU and (translated) WebGL2.

let camX = 0;
let camY = 0;
let testMode = false;

export default shaderScene({
  interaction: 'Mouse up/down: vertical parallax · drag: scrub the camera.',
  examples: [
    {
      id: 'forest',
      label: 'Forest side-scroller',
      kind: 'In a game',
      note: 'Mountains, three bands of pine forest, rolling hills and a foreground of grass and trunks. Far layers barely move and fade into the warm haze; the foreground races past. That difference in speed is all your brain needs to read depth.',
    },
    {
      id: 'city',
      label: 'Night city skyline',
      kind: 'In a game',
      note: 'Building rows at increasing distance with lit windows, neon signs and a moon. Far rows are small, purple and hazy (light pollution); near rows are dark, big and detailed. The moon and stars are “at infinity” — they don’t scroll at all.',
      params: { speed: 0.08, fog: 0.55 },
    },
    {
      id: 'space',
      label: 'Space',
      kind: 'Abstract',
      note: 'Star fields at several depths over a nebula, with planets in the middle distance and asteroids up close. Even simple dots become convincingly 3D when each layer moves at its own speed.',
      params: { speed: 0.15, layers: 7, fog: 0.4 },
    },
  ],
  controls: [
    { type: 'slider', key: 'speed', label: 'Camera speed', min: 0, max: 1, step: 0.01, value: 0.12, help: 'Screen heights per second. 0 = stop (you can still drag to scrub).' },
    { type: 'slider', key: 'strength', label: 'Parallax strength', min: 0, max: 2, step: 0.01, value: 1, help: '0 = flat (everything moves together), 1 = speed ∝ 1/distance, 2 = exaggerated.' },
    { type: 'slider', key: 'layers', label: 'Layer count', min: 2, max: 8, step: 1, value: 6, help: 'The same depth range split into more or fewer layers.' },
    { type: 'slider', key: 'fog', label: 'Atmospheric fog', min: 0, max: 1, step: 0.01, value: 0.65, help: 'Far layers get lighter, bluer and lower-contrast (aerial perspective).' },
    { type: 'slider', key: 'vertical', label: 'Vertical parallax (mouse)', min: 0, max: 1, step: 0.01, value: 0.5, help: 'Mouse height shifts near layers more than far ones.' },
    { type: 'toggle', key: 'debug', label: 'Debug: tint layers by depth', value: false, help: 'Purple = far … red = near. Silhouette edges are outlined.' },
  ],
  uniforms: { speed: 'f32', strength: 'f32', layers: 'f32', fog: 'f32', vertical: 'f32', debug: 'f32', camX: 'f32', camY: 'f32' },
  include: ['noise', 'sdf', 'color'],
  renderScale: () => (testMode ? 0.3 : 1),
  bind(params, ctx) {
    testMode = ctx.testMode;
    camX += params.speed * ctx.dt;
    if (ctx.pointer.down) camX -= (ctx.pointer.dx / Math.max(1, ctx.height)) * 1.5;
    const target = ctx.pointer.over ? ctx.pointer.ny - 0.5 : 0;
    camY += (target - camY) * (1 - Math.exp(-4 * Math.max(ctx.dt, 1 / 120)));
    return { camX, camY: camY * params.vertical };
  },
  code: /* wgsl */ `
struct LayerOut { col: vec3f, cov: f32, glow: vec3f };

fn aaf(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }

// value-noise fBm / ridged noise in [0, 1]: no sin/cos per octave, so it's cheap even on weak GPUs
fn fbmV(p0: vec2f, oct: i32) -> f32 {
  var p = p0;
  var s = 0.0;
  var a = 0.5;
  var n = 0.0;
  for (var i = 0; i < 6; i++) {
    if (i >= oct) { break; }
    s += valueNoise(p) * a;
    n += a;
    p = p * 2.03 + vec2f(1.7, 9.2);
    a *= 0.5;
  }
  return s / n;
}
fn ridgedV(p0: vec2f, oct: i32) -> f32 {
  var p = p0;
  var s = 0.0;
  var a = 0.5;
  var n = 0.0;
  for (var i = 0; i < 6; i++) {
    if (i >= oct) { break; }
    let v = 1.0 - abs(valueNoise(p) * 2.0 - 1.0);
    s += v * v * a;
    n += a;
    p = p * 2.07 + vec2f(3.1, 5.3);
    a *= 0.5;
  }
  return s / n;
}

// ------------------------------------------------------------------ forest
// cheap pine silhouette: two stacked triangles + trunk as an approximate signed distance (no full triangle SDF)
fn pine(p: vec2f, base: vec2f, h: f32) -> f32 {
  let dx = abs(p.x - base.x);
  let w = h * 0.36;
  let s1 = w / (h * 0.56);                       // half-width per unit of height, lower tier
  let s2 = w * 0.75 / (h * 0.58);                // upper tier
  let k1 = 1.0 / sqrt(1.0 + s1 * s1);
  let k2 = 1.0 / sqrt(1.0 + s2 * s2);
  let lower = max((dx - (p.y - (base.y - h * 0.68)) * s1) * k1, p.y - (base.y - h * 0.12));
  let upper = max((dx - (p.y - (base.y - h)) * s2) * k2, p.y - (base.y - h * 0.42));
  let trunk = max(dx - w * 0.12, abs(p.y - (base.y - h * 0.02)) - h * 0.12);
  return min(min(lower, upper), trunk);
}

fn skyForest(p: vec2f, aspect: f32) -> vec3f {
  var c = mix(vec3f(0.30, 0.52, 0.84), vec3f(1.0, 0.85, 0.64), smoothstep(0.0, 0.62, p.y));
  let sun = vec2f(aspect * 0.7, 0.34);
  let d = length(p - sun);
  c += vec3f(1.0, 0.72, 0.42) * exp(-d * 5.0) * 0.5;
  c = mix(c, vec3f(1.0, 0.97, 0.88), aaf(d - 0.055, 0.004));
  // soft clouds
  if (p.y > 0.04 && p.y < 0.46) {
    let cq = vec2f(p.x * 1.3 + u.camX * 0.02 + u.time * 0.01, p.y * 3.5);
    let cl = smoothstep(0.52, 0.72, fbmV(cq * 3.0, 4)) * smoothstep(0.05, 0.2, p.y) * smoothstep(0.45, 0.25, p.y);
    c = mix(c, vec3f(1.0, 0.95, 0.9), cl * 0.75);
  }
  return c;
}

fn forestLayer(t: f32, x: f32, y: f32, seed: f32, pw: f32) -> LayerOut {
  let base = mix(0.5, 0.9, t);
  // bounding check: nothing of this layer reaches above 'top' -> skip all the noise
  var top = base - 0.03 - mix(0.08, 0.32, t) * 1.15;
  if (t < 0.15) { top = base - 0.27; }
  if (t < 0.95 && y < top) { return LayerOut(vec3f(0.0), 0.0, vec3f(0.0)); }
  var h = 0.0;
  if (t < 0.15) {
    h = base - 0.03 - 0.22 * ridgedV(vec2f(x * 1.6, seed), 4);
  } else {
    h = base - mix(0.1, 0.04, t) * fbmV(vec2f(x * mix(2.4, 7.0, t), seed), 3);
  }
  var cov = aaf(h - y, pw);
  if (t >= 0.15 && t < 0.95) {
    let cw = mix(0.03, 0.15, t);
    let cell = floor(x / cw);
    for (var k = -1; k <= 1; k++) {
      let c = cell + f32(k);
      if (hash11(c * 1.37 + seed) > 0.2) {
        let tx = (c + 0.2 + 0.6 * hash11(c * 7.1 + seed)) * cw;
        let th = mix(0.08, 0.32, t) * (0.55 + 0.6 * hash11(c * 3.3 + seed));
        cov = max(cov, aaf(pine(vec2f(x, y), vec2f(tx, base - 0.01), th), pw));
      }
    }
  }
  var glow = vec3f(0.0);
  if (t >= 0.95) {
    // foreground: grass blades swaying + huge dark trunks
    let blade = 0.03 * hash11(floor(x * 110.0)) + 0.012 * sin(x * 37.0 + u.time * 1.7);
    cov = max(cov, aaf((base - blade) - y, pw));
    let tw = 0.95;
    let tc = floor(x / tw);
    if (hash11(tc * 5.3 + seed) > 0.4) {
      let txc = (tc + 0.3 + 0.4 * hash11(tc * 2.1 + seed)) * tw;
      let trunk = abs(x - txc) - 0.04 * (1.0 + 0.6 * smoothstep(0.75, 1.0, y));
      cov = max(cov, aaf(trunk, pw));
    }
    // drifting pollen catching the light
    if (y > 0.5) {
      let pc = floor(vec2f(x, y + u.time * 0.02) * 9.0);
      let pp = (pc + hash22(pc)) / 9.0;
      let pd = length(vec2f(x, y + u.time * 0.02) - pp);
      glow = vec3f(1.0, 0.9, 0.6) * exp(-pd * 300.0) * step(0.85, hash21(pc)) * 0.8;
    }
  }
  var col = mix(vec3f(0.43, 0.55, 0.64), vec3f(0.05, 0.11, 0.08), pow(t, 0.7));
  col = mix(col, col * vec3f(1.25, 1.1, 0.85), (1.0 - t) * 0.4);
  col *= 1.0 - 0.3 * t * smoothstep(base - 0.15, base + 0.25, y);
  return LayerOut(col, cov, glow);
}

// ------------------------------------------------------------------ night city
fn skyCity(p: vec2f, aspect: f32) -> vec3f {
  var c = mix(vec3f(0.012, 0.014, 0.05), vec3f(0.30, 0.11, 0.32), smoothstep(0.0, 0.8, p.y));
  let sc = floor(p * 320.0);
  let sh = hash21(sc);
  let tw = 0.6 + 0.4 * sin(u.time * 3.0 + sh * 60.0);
  c += vec3f(0.9, 0.9, 1.0) * step(0.9965, sh) * tw * (1.0 - smoothstep(0.2, 0.6, p.y));
  let moon = vec2f(aspect * 0.78, 0.2);
  let d = length(p - moon);
  c += vec3f(0.65, 0.6, 0.9) * exp(-d * 9.0) * 0.35;
  let crater = valueNoise((p - moon) * 60.0) * 0.18 + valueNoise((p - moon) * 140.0) * 0.08;
  c = mix(c, vec3f(0.97, 0.95, 0.88) - crater, aaf(d - 0.06, 0.003));
  return c;
}

fn cityLayer(t: f32, x: f32, y: f32, seed: f32, pw: f32) -> LayerOut {
  let base = mix(0.72, 1.04, t);
  if (y < base - mix(0.1, 0.44, t) * 1.55 - 0.03) { return LayerOut(vec3f(0.0), 0.0, vec3f(0.0)); }
  let bw = mix(0.028, 0.16, t);
  let cell = floor(x / bw);
  let r1 = hash11(cell * 1.91 + seed);
  let r2 = hash11(cell * 7.37 + seed * 3.1);
  let r3 = hash11(cell * 3.13 + seed * 1.7);
  let w = bw * (0.62 + 0.34 * r1);
  let lx = x - cell * bw - (bw - w) * 0.5;
  let hgt = mix(0.1, 0.44, t) * (0.3 + 0.7 * r2);
  let top = base - hgt;
  var d = max(abs(lx - w * 0.5) - w * 0.5, top - y);
  if (r3 > 0.55) {
    d = min(d, max(abs(lx - w * 0.5) - w * 0.28, (top - hgt * 0.16) - y));
  }
  if (r3 > 0.82) {
    d = min(d, max(abs(lx - w * 0.5) - 0.0012 - 0.002 * t, (top - hgt * 0.5) - y));
  }
  let cov = aaf(d, pw);
  var col = mix(vec3f(0.2, 0.15, 0.33), vec3f(0.03, 0.028, 0.06), pow(t, 0.55));
  // windows: a grid inside the facade, each lit or not by a hash (a few flicker)
  let ws = mix(0.0065, 0.024, t);
  let wq = vec2f(lx, y - top) / ws;
  let wid = floor(wq);
  let wf = fract(wq);
  let inWin = step(0.28, wf.x) * step(wf.x, 0.72) * step(0.3, wf.y) * step(wf.y, 0.75);
  let inside = step(ws, lx) * step(lx, w - ws) * step(top + ws, y);
  let h = hash21(wid + vec2f(cell * 31.0, seed * 17.0));
  var lit = step(0.6, h);
  if (h > 0.93) { lit = step(0.5, hash21(wid + vec2f(floor(u.time * 0.4 + h * 10.0), cell))); }
  let wc = mix(vec3f(1.0, 0.76, 0.42), vec3f(0.55, 0.8, 1.0), step(0.82, hash11(h * 91.0)));
  col = mix(col, wc * mix(0.55, 1.15, t), inWin * inside * lit);
  // neon signs with glow
  var glow = vec3f(0.0);
  if (t > 0.3 && r1 > 0.62) {
    let sc = vec2f(cell * bw + bw * 0.5, top + hgt * 0.28);
    let nd = sdBox(vec2f(x, y) - sc, vec2f(w * 0.3, hgt * 0.05 + 0.004));
    let ncol = mix(vec3f(1.0, 0.22, 0.68), vec3f(0.2, 0.9, 1.0), step(0.5, r3));
    let blink = 0.7 + 0.3 * step(0.15, fract(u.time * 0.6 + r2 * 3.0));
    col = mix(col, ncol * 1.4, aaf(abs(nd) - 0.0016 * (1.0 + t), pw) * blink * cov);
    glow = ncol * exp(-max(nd, 0.0) * mix(55.0, 22.0, t)) * 0.3 * blink;
  }
  return LayerOut(col, cov, glow);
}

// ------------------------------------------------------------------ space
fn skySpace(p: vec2f, aspect: f32) -> vec3f {
  let q = vec2f(p.x + u.camX * 0.015, p.y + u.camY * 0.01);
  let warp = vec2f(valueNoise(q * 2.4), valueNoise(q * 2.4 + vec2f(5.2, 1.3))) * 2.0 - 1.0;
  let n = fbmV(q * 2.4 + warp * 0.8, 4);
  var c = mix(vec3f(0.01, 0.01, 0.03), vec3f(0.18, 0.06, 0.28), smoothstep(0.35, 0.8, n));
  c = mix(c, vec3f(0.05, 0.3, 0.4), smoothstep(0.55, 0.8, fbmV(q * 3.5 - warp, 2)) * 0.4);
  c += vec3f(0.9, 0.45, 0.3) * smoothstep(0.72, 0.95, n) * 0.35;
  return c;
}

fn spaceLayer(t: f32, q: vec2f, seed: f32, pw: f32) -> LayerOut {
  var col = vec3f(0.0);
  var cov = 0.0;
  var glow = vec3f(0.0);
  let gs = mix(0.03, 0.12, t);
  let cell = floor(q / gs);
  let r = hash22(cell + vec2f(seed * 13.0, seed * 7.0));
  let sp = (cell + 0.15 + 0.7 * r) * gs;
  let d = length(q - sp);
  let rad = mix(0.0008, 0.0032, t) * (0.4 + 1.2 * hash21(cell + vec2f(seed, 1.0)));
  let tw = 0.75 + 0.25 * sin(u.time * (2.0 + 3.0 * r.x) + r.y * 20.0);
  let sc = mix(vec3f(0.65, 0.78, 1.0), vec3f(1.0, 0.85, 0.65), r.y);
  glow += sc * (exp(-d / rad) * 1.3 + exp(-d / (rad * 6.0)) * 0.1) * tw * mix(0.45, 1.0, t);
  if (t > 0.25 && t < 0.8) {
    let pcw = 1.6;
    let pcell = floor(q.x / pcw);
    if (hash11(pcell * 4.7 + seed) > 0.3) {
      let ppos = vec2f((pcell + 0.25 + 0.5 * hash11(pcell * 2.3 + seed)) * pcw, 0.2 + 0.55 * hash11(pcell * 9.1 + seed));
      let prad = mix(0.035, 0.13, t) * (0.6 + 0.6 * hash11(pcell * 5.5 + seed));
      let rel = (q - ppos) / prad;
      let pd = (length(rel) - 1.0) * prad;
      if (pd < prad * 1.2) {
      let hue = hash11(pcell * 8.8 + seed);
      let pc = palette(hue, vec3f(0.5), vec3f(0.45), vec3f(1.0), vec3f(0.0, 0.33, 0.67));
      let nz = sqrt(max(0.0, 1.0 - dot(rel, rel)));
      let light = clamp(dot(vec3f(rel, nz), normalize(vec3f(-0.6, -0.5, 0.65))), 0.0, 1.0);
      let bands = 0.8 + 0.2 * sin(rel.y * 12.0 + (fbmV(rel * 4.0 + vec2f(hue * 9.0, 0.0), 3) * 2.0 - 1.0) * 4.0);
      col = pc * (0.05 + 0.95 * light) * bands;
      cov = aaf(pd, pw);
      glow += pc * exp(-max(pd, 0.0) / (prad * 0.15)) * 0.25 * (1.0 - cov);
      // a ring on some planets (drawn in front of the planet's lower half)
      if (hue > 0.55) {
        let rq = vec2f(rel.x, rel.y * 3.2 + rel.x * 0.5);
        let rd = abs(length(rq) - 1.75) - 0.22;
        let front = step(0.0, rel.y * 3.2 + rel.x * 0.5) + (1.0 - cov);
        let ra = aaf(rd * prad, pw) * clamp(front, 0.0, 1.0);
        col = mix(col, pc * 0.9 + vec3f(0.15), ra * 0.85);
        cov = max(cov, ra * 0.85);
      }
      }
    }
  }
  if (t > 0.85) {
    let ac = 0.3;
    let acell = floor(q / ac);
    let ar = hash22(acell + vec2f(seed, 3.0));
    if (ar.x > 0.64) {
      let apos = (acell + 0.25 + 0.5 * ar) * ac;
      let arad = 0.018 + 0.035 * ar.y;
      let rel = q - apos;
      let ang = atan2(rel.y, rel.x);
      let bump = 0.22 * valueNoise(vec2f(ang * 1.7 + ar.x * 10.0, ar.y * 5.0)) + 0.08 * valueNoise(vec2f(ang * 5.0, 2.0));
      let ad = length(rel) - arad * (0.8 + bump);
      let sh = 0.35 + 0.65 * clamp(dot(normalize(rel + vec2f(0.0001)), vec2f(-0.7, -0.7)) * 0.5 + 0.5, 0.0, 1.0);
      let rock = vec3f(0.42, 0.37, 0.34) * sh * (0.8 + 0.4 * valueNoise(rel * 160.0));
      let a = aaf(ad, pw);
      col = mix(col, rock, a);
      cov = max(cov, a);
    }
  }
  return LayerOut(col, cov, glow);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let aspect = res.x / res.y;
  let p = vec2f(uv.x * aspect, uv.y);          // y 0..1 down, x 0..aspect (independent of render scale)
  let pw = max(fwidth(p.y), 0.0005) * 1.6;      // ~1.6 pixels, for anti-aliased edges
  let ex = i32(u.example + 0.5);
  let n = i32(u.layers + 0.5);
  var haze = vec3f(0.1, 0.06, 0.18);
  if (ex == 0) { haze = mix(vec3f(0.74, 0.82, 0.92), vec3f(1.0, 0.87, 0.7), smoothstep(0.2, 0.75, p.y)); }
  if (ex == 1) { haze = mix(vec3f(0.16, 0.08, 0.25), vec3f(0.55, 0.24, 0.45), smoothstep(0.45, 1.0, p.y)); }
  // composite FRONT to back: stop as soon as the nearer layers fully cover the pixel
  var acc = vec3f(0.0);
  var T = 1.0;                                 // how much of what's behind still shows through
  for (var j = 0; j < 8; j++) {
    if (j >= n) { break; }
    let i = n - 1 - j;
    let t = f32(i) / max(f32(n - 1), 1.0);   // 0 = farthest … 1 = nearest
    let z = mix(10.0, 1.0, t);                 // distance of this layer
    let par = pow(1.0 / z, u.strength);        // how much it moves with the camera
    let lx = p.x + u.camX * par;
    let ly = p.y + u.camY * par * 0.35;
    let seed = f32(i) * 17.3 + 3.0;
    var L: LayerOut;
    if (ex == 0) { L = forestLayer(t, lx, ly, seed, pw); }
    else if (ex == 1) { L = cityLayer(t, lx, ly, seed, pw); }
    else { L = spaceLayer(t, vec2f(lx, ly), seed, pw); }
    // atmospheric perspective: the farther, the more the layer dissolves into the haze
    let fogAmt = u.fog * pow(1.0 - t, 1.35) * 0.92;
    var lc = mix(L.col, haze, fogAmt);
    var lg = L.glow * (1.0 - fogAmt * 0.8);
    if (u.debug > 0.5) {
      lc = mix(lc, hsv2rgb(vec3f(0.78 - t * 0.78, 0.75, 1.0)), 0.65);
      lg = lg * 0.25;
      lc += vec3f(1.0) * (1.0 - L.cov) * 3.0;
    }
    acc += T * (lc * L.cov + lg);
    T *= 1.0 - L.cov;
    if (T < 0.003) { break; }
  }
  // the sky (at infinity: it never scrolls) only where something still shows through
  var col = acc;
  if (T > 0.003) {
    var sky = vec3f(0.0);
    if (ex == 0) { sky = skyForest(p, aspect); }
    else if (ex == 1) { sky = skyCity(p, aspect); }
    else { sky = skySpace(p, aspect); }
    if (u.debug > 0.5) { sky = vec3f(luma(sky)) * 0.35; }
    col = acc + T * sky;
  }
  let v = length(uv - 0.5);
  col *= 1.0 - 0.35 * v * v;
  return vec4f(col, 1.0);
}`,
  about: {
    summary: 'Layers that move at different speeds trick the eye into seeing depth. Add atmospheric fog and the flattest 2D game suddenly has miles of distance.',
    what: `<p>A camera slides sideways over several hand-made-looking layers that are actually generated in one fragment shader. The farther a layer is,
      the slower it moves and the more it fades into the haze. Move the mouse vertically to shift the near layers more than the far ones.</p>`,
    how: `<ol>
      <li>Each layer has a <b>distance z</b> (here from 10 down to 1). When the camera moves by Δx, a layer moves by <code>Δx / z</code>
        (that’s what perspective does to real objects). The <i>Parallax strength</i> slider raises it to a power: 0 = flat, 2 = exaggerated.</li>
      <li>In practice: <code>layerX = screenX + cameraX × factor</code> before evaluating the layer — the shader “looks up” the layer at a shifted position.</li>
      <li>Each layer returns a colour and a coverage (alpha). They are composited <b>front to back</b>: a running “transmittance” says how much of the
        layers behind still shows through, and the loop stops early once a pixel is fully covered — the sky is only evaluated where it’s visible.</li>
      <li><b>Atmospheric perspective</b>: air scatters light, so distant things take on the sky colour and lose contrast:
        <code>mix(layerColor, hazeColor, fog × (1 − nearness)^1.35)</code>.</li>
      <li>Things at infinity (sun, moon, stars) don’t move at all; the nebula moves at 1.5% of the camera speed.</li>
    </ol>`,
    uses: [
      { title: 'Side-scrollers', text: 'Every platformer since Moon Patrol (1982): Sonic, Rayman Legends, Ori, Hollow Knight’s foregrounds.' },
      { title: 'Menus & title screens', text: 'Slow-scrolling skylines and starfields that react to the mouse or device tilt.' },
      { title: 'Shmups & runners', text: 'Endless runners and space shooters sell speed with fast foreground layers.' },
      { title: 'Top-down games', text: 'Cloud shadows or foreground leaves on a faster layer add height above the map.' },
    ],
    try: [
      'Set <i>Parallax strength</i> to 0: the scene instantly becomes a flat painting. Then try 2.',
      'Turn <i>Atmospheric fog</i> to 0 in the forest: the layers look like cardboard cut-outs stacked on top of each other.',
      'Enable <i>Debug: tint layers by depth</i> and drag <i>Layer count</i> from 2 to 8.',
      'Move the mouse to the top and bottom of the canvas slowly — near layers swing much more than far ones.',
      'Drag horizontally to scrub the camera backwards.',
    ],
    ask: [
      'multi-layer parallax background with atmospheric perspective',
      'parallax factor = 1 / depth for each background layer',
      'mouse-reactive vertical parallax for the title screen',
      'a procedural night-city skyline with lit windows and neon',
      'a parallax starfield with planets and asteroids',
    ],
    perf: `<p>Here every layer is evaluated for every pixel (≈8 noise evaluations per layer), still cheap for a GPU. In a real game the layers are usually
      textures: then each layer costs one texture read per pixel, plus overdraw where layers overlap. Drawing front-to-back with early-out, or skipping fully
      covered regions, saves fill-rate on low-end GPUs.</p>`,
    api: `<p>Pure fragment shader work — identical on WebGL2 and WebGPU (this scene runs on both; the WGSL is translated to GLSL automatically).</p>`,
    code: [
      {
        title: 'The parallax loop (front to back, with early exit)',
        lang: 'wgsl',
        src: `for (var j = 0; j < 8; j++) {
  if (j >= n) { break; }
  let i = n - 1 - j;                         // nearest layer first
  let t = f32(i) / max(f32(n - 1), 1.0);   // 0 = farthest … 1 = nearest
  let z = mix(10.0, 1.0, t);                 // distance of this layer
  let par = pow(1.0 / z, u.strength);        // how much it moves with the camera
  let lx = p.x + u.camX * par;               // shift the lookup, not the pixels
  let ly = p.y + u.camY * par * 0.35;        // vertical parallax from the mouse
  let L = forestLayer(t, lx, ly, seed, pw);  // colour + coverage of this layer
  let fogAmt = u.fog * pow(1.0 - t, 1.35);   // atmospheric perspective
  acc += T * (mix(L.col, haze, fogAmt) * L.cov + L.glow);
  T *= 1.0 - L.cov;                          // what still shows through
  if (T < 0.003) { break; }                  // fully covered: skip the rest
}
col = acc + T * sky;`,
      },
      {
        title: 'With textures instead (typical engine code)',
        lang: 'js',
        src: `for (const layer of layers) {            // sorted far -> near
  const factor = 1 / layer.depth;
  const offset = (camera.x * factor) % layer.width;
  drawTiledHorizontally(layer.texture, -offset, layer.y - camera.y * factor);
}`,
      },
    ],
  },
});

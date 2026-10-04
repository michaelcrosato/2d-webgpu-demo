import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import { GAME_SCENE_CODE, GAME_SCENE_INCLUDES } from '../../core/gamescene.js';

// Hand-drawn looks: line boil, wobbly SDF outlines, paper grain, crayon / pencil / hatching fills,
// a chalkboard, a ballpoint notebook page and the game image redrawn as a boiling sketch.
//
// Line boil = the drawing is "re-traced" a few times per second: a noise offset (domain warp) is added
// to the position BEFORE evaluating the shapes, and the noise seed only changes 8–12 times a second,
// cycling through a few "drawings" — exactly like hand-animated cartoons (Ed, Edd n Eddy, Squigglevision).

// Hand-lettered labels, painted once with Canvas2D. Top half: chalkboard, bottom half: notebook.
// Design space for each half is 16:9 and maps to the centered screen space used in the shader.
function buildLabels() {
  const W = 1024;
  const H = 576;
  const c = makeCanvas(W, H * 2);
  const g = c.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, W, H * 2);
  g.fillStyle = '#fff';
  const hand = '"Comic Sans MS", "Chalkboard SE", "Marker Felt", "Segoe Print", "Bradley Hand", "Noteworthy", cursive';
  const label = (txt, x, y, size, half, rot = 0) => {
    g.save();
    g.translate(x * W, y * H + half * H);
    g.rotate(rot);
    g.font = `${size}px ${hand}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(txt, 0, 0);
    g.restore();
  };
  // chalkboard (coordinates in 0..1 of the 16:9 design space)
  label('Level 1-1', 0.5, 0.1, 50, 0, -0.02);
  label('y = sin(x)', 0.25, 0.25, 32, 0, -0.03);
  label('jump!', 0.66, 0.36, 34, 0, 0.05);
  label('+100', 0.86, 0.43, 26, 0, -0.04);
  // notebook
  label('Game idea #7', 0.32, 0.12, 42, 1, -0.02);
  label('to the moon!!', 0.43, 0.83, 30, 1, -0.06);
  label('boss?', 0.82, 0.75, 28, 1, 0.08);
  return c;
}

const CODE = /* wgsl */ `
// ---------------------------------------------------------------- boil & wobble
fn boilSeed() -> f32 {
  if (u.boilFps < 0.5) { return 0.0; }
  return fmod(floor(u.time * u.boilFps), u.cycle) + 1.0;   // cycles through a few "drawings"
}
// time that only advances in drawing steps (animating "on twos/threes")
fn steppedTime() -> f32 {
  if (u.boilFps < 0.5) { return u.time; }
  return floor(u.time * u.boilFps) / u.boilFps;
}
// wobble offset in pixels for a pixel position and a drawing seed
fn wob(px: vec2f, seed: f32) -> vec2f {
  let q = px / u.resolution.y * 7.0;
  let s = vec2f(seed * 3.17, seed * 1.73);
  let big = vec2f(perlin(q + s), perlin(q + vec2f(7.3, 2.9) + s.yx));
  let small = vec2f(perlin(q * 5.0 + s * 2.0), perlin(q * 5.0 + vec2f(3.1, 8.4) - s));
  return (big + small * 0.25) * u.wobble * u.resolution.y / 900.0;
}

// ---------------------------------------------------------------- media
fn tooth(px: vec2f) -> f32 { return valueNoise(px * 0.55) * 0.6 + valueNoise(px * 1.7) * 0.4; }
fn paperCol(px: vec2f, base: vec3f) -> vec3f {
  let g1 = valueNoise(px * 0.9);
  let g2 = valueNoise(px * 0.012 + vec2f(7.0, 1.0));
  let fib = valueNoise(vec2f(px.x * 0.04, px.y * 0.5));
  return base * (1.0 - u.paper * (0.07 * g1 + 0.09 * g2 + 0.05 * fib));
}
// anti-aliased line from a distance in pixels
fn lineCov(d: f32, w: f32) -> f32 { return 1.0 - smoothstep(w * 0.5 - 0.6, w * 0.5 + 0.6, abs(d)); }
// graphite/ink line: thins out where the paper tooth is high
fn pencil(d: f32, w: f32, px: vec2f) -> f32 { return lineCov(d, w) * mix(1.0, smoothstep(0.12, 0.5, tooth(px)), u.paper * 0.7); }
// crayon fill: streaky coverage along a stroke direction, the paper shows through
fn crayon(inside: f32, px: vec2f, angle: f32) -> f32 {
  let r = rot2(angle) * px;
  let streak = valueNoise(vec2f(r.x * 0.05, r.y * 0.8)) * 0.55 + valueNoise(px * 0.9) * 0.45;
  return inside * smoothstep(0.22, 0.55, streak);
}
// hatching: 1, 2 or 3 layers of wobbly parallel strokes depending on the tone (0 light .. 1 dark)
fn hatch(tone: f32, px: vec2f, spacing: f32, w: f32) -> f32 {
  var cov = 0.0;
  for (var k = 0; k < 3; k++) {
    let fk = f32(k);
    if (tone < 0.2 + fk * 0.27) { break; }
    let r = rot2(0.75 + fk * 1.1) * px;
    let jit = perlin(r * 0.015 + vec2f(fk * 4.0, boilSeed())) * 0.35;
    let h = abs(fract(r.y / spacing + jit) - 0.5) * spacing;
    cov = max(cov, 1.0 - smoothstep(w * 0.5 - 0.5, w * 0.5 + 0.5, h));
  }
  return cov * (0.55 + 0.45 * tooth(px));
}

fn sdHeartAt(p: vec2f, c: vec2f, s: f32) -> f32 { return sdHeart((p - c) / s + vec2f(0.0, 0.55)) * s; }

// ---------------------------------------------------------------- 0. line boil (abstract)
fn blobD(p: vec2f) -> f32 {
  var d = opSmoothUnion(sdCircle(p - vec2f(0.0, -0.03), 0.14), sdRoundBox(p - vec2f(0.0, 0.07), vec2f(0.15, 0.08), 0.06), 0.07);
  d = opSmoothUnion(d, sdCircle(p - vec2f(-0.1, 0.15), 0.035), 0.02);
  d = opSmoothUnion(d, sdCircle(p - vec2f(0.1, 0.15), 0.035), 0.02);
  return d;
}

fn boilScene(px: vec2f) -> vec3f {
  let res = u.resolution;
  let s1 = boilSeed();
  let ts = steppedTime();
  var col = paperCol(px, vec3f(0.97, 0.95, 0.9));
  let ink = vec3f(0.09, 0.08, 0.1);
  let lw = u.lineWidth * res.y / 700.0;
  // two tracings of the same drawing: the main line and a lighter "sketch" line
  for (var tr = 0; tr < 2; tr++) {
    let seed = s1 + f32(tr) * 11.5;
    let wpx = px + wob(px, seed) * (1.0 + f32(tr) * 0.6);
    let p = (wpx - 0.5 * res) / res.y;
    let bob = 0.015 * sin(ts * 3.0);
    let bc = vec2f(-0.26, 0.06 + bob);
    let dBlob = blobD(p - bc) * res.y;
    let starC = vec2f(0.3, -0.13 - bob * 0.6);
    let dStar = sdStar5(rot2(0.1 * sin(ts * 1.5)) * (p - starC), 0.13, 0.45) * res.y;
    let dHeart = sdHeartAt(p, vec2f(0.28, 0.22), 0.13) * res.y;
    // motion lines, a little cloud and a ground line
    var ml = min(sdSegment(p, vec2f(-0.5, -0.12), vec2f(-0.44, -0.12)), sdSegment(p, vec2f(-0.52, -0.06), vec2f(-0.45, -0.06)));
    let cl = vec2f(-0.02, -0.3);
    var cloud = min(sdCircle(p - cl, 0.045), sdCircle(p - cl - vec2f(0.05, 0.012), 0.035));
    cloud = min(cloud, min(sdCircle(p - cl - vec2f(-0.05, 0.015), 0.03), sdBox(p - cl - vec2f(0.0, 0.03), vec2f(0.07, 0.015))));
    ml = min(ml, abs(cloud));
    ml = min(ml, sdSegment(p, vec2f(-0.62, 0.3), vec2f(0.55, 0.3)));
    ml = ml * res.y;
    if (tr == 0) {
      // fills only once (with the first tracing)
      col = mix(col, vec3f(0.55, 0.78, 0.95), crayon(1.0 - smoothstep(-1.0, 1.0, dBlob + 3.0), px, 0.5) * 0.9);
      col = mix(col, vec3f(1.0, 0.8, 0.2), crayon(1.0 - smoothstep(-1.0, 1.0, dStar + 3.0), px, -0.4) * 0.95);
      col = mix(col, vec3f(0.85, 0.15, 0.2), hatch(0.35 + 0.65 * u.hatch, px, 7.0 * res.y / 900.0, 1.4) * (1.0 - smoothstep(-1.0, 1.0, dHeart + 2.0)));
      // a scribbled shadow under the blob
      let shd = sdEllipseApprox(p - vec2f(bc.x, 0.3), vec2f(0.17 - bob, 0.025));
      col = mix(col, vec3f(0.45, 0.42, 0.45), hatch(0.9, px, 4.0 * res.y / 900.0, 1.1) * (1.0 - smoothstep(-0.002, 0.002, shd)) * 0.7);
      // cheeks
      let ck = min(length(p - bc - vec2f(-0.085, 0.0)), length(p - bc - vec2f(0.085, 0.0))) - 0.025;
      col = mix(col, vec3f(0.98, 0.5, 0.55), crayon(1.0 - smoothstep(-0.002, 0.002, ck), px, 0.2) * 0.6);
    }
    let a = select(1.0, u.sketchy * 0.55, tr == 1);
    var d = min(dBlob, min(dStar, dHeart));
    d = min(d, ml);
    col = mix(col, ink, pencil(d, lw, px) * a);
    if (tr == 0) {
      // eyes & smile
      let eyes = min(length(p - bc - vec2f(-0.05, -0.05)), length(p - bc - vec2f(0.05, -0.05))) - 0.024;
      col = mix(col, ink, 1.0 - smoothstep(-0.5, 0.5, eyes * res.y));
      let hl = min(length(p - bc - vec2f(-0.042, -0.06)), length(p - bc - vec2f(0.058, -0.06))) - 0.007;
      col = mix(col, vec3f(1.0), 1.0 - smoothstep(-0.5, 0.5, hl * res.y));
      // sdArc opens toward screen-up in our y-down space, so flip y for a smile
      let sq = p - bc - vec2f(0.0, -0.02);
      let smile = sdArc(vec2f(sq.x, -sq.y), vec2f(sin(0.9), cos(0.9)), 0.05, 0.0) * res.y;
      col = mix(col, ink, pencil(smile, lw, px));
    }
  }
  return col;
}

// ---------------------------------------------------------------- 1. notebook doodles (ballpoint)
fn labelAt(p: vec2f, half: f32) -> f32 {
  let uvl = vec2f(p.x / (16.0 / 9.0) + 0.5, (p.y + 0.5) * 0.5 + half * 0.5);
  if (uvl.x < 0.0 || uvl.x > 1.0 || p.y < -0.5 || p.y > 0.5) { return 0.0; }
  return TEX(labels, uvl).r;
}

fn spiralD(p: vec2f, k: f32, turns: f32) -> f32 {
  let r = length(p);
  let th = atan2(p.y, p.x);
  let n = clamp(floor((r / k - th / TAU) + 0.5), 0.0, turns);
  return abs(r - k * (th / TAU + n) * 1.0);
}

fn notebook(px: vec2f) -> vec3f {
  let res = u.resolution;
  let base = (px - 0.5 * res) / res.y;
  var col = paperCol(px, vec3f(0.985, 0.98, 0.96));
  // printed ruling: straight lines (printed, so they do NOT boil)
  let ls = res.y / 22.0;
  let ly = abs(fract((px.y - res.y * 0.12) / ls + 0.5) - 0.5) * ls;
  col = mix(col, vec3f(0.62, 0.75, 0.92), (1.0 - smoothstep(0.4, 1.2, ly)) * step(res.y * 0.1, px.y) * 0.8);
  let mx = res.x * 0.5 - res.y * 0.62;
  col = mix(col, vec3f(0.92, 0.45, 0.45), 1.0 - smoothstep(0.5, 1.4, abs(px.x - mx)));
  // punched holes
  for (var h = 0; h < 3; h++) {
    let hc = vec2f(mx - res.y * 0.075, res.y * (0.2 + 0.3 * f32(h)));
    let hd = length(px - hc) - res.y * 0.022;
    col = mix(col, vec3f(0.25, 0.22, 0.2) * (0.8 + 0.2 * clamp(-hd / 8.0, 0.0, 1.0)), 1.0 - smoothstep(-0.7, 0.7, hd));
  }
  let pen = vec3f(0.12, 0.2, 0.62);
  let lw = u.lineWidth * res.y / 900.0;
  let s1 = boilSeed();
  let ts = steppedTime();
  for (var tr = 0; tr < 2; tr++) {
    let seed = s1 + f32(tr) * 7.3;
    let wpx = px + wob(px, seed) * (0.7 + f32(tr) * 0.6);
    let p = (wpx - 0.5 * res) / res.y;
    var d = 1.0e3;
    // rocket (tilted) with window, fins and a flame
    let rc = vec2f(-0.12, 0.12) + vec2f(0.01, -0.01) * sin(ts * 2.0);
    let rq = rot2(-0.6) * (p - rc);
    d = min(d, abs(sdVesica(rq.yx, 0.17, 0.07)));
    d = min(d, abs(length(rq - vec2f(0.0, -0.05)) - 0.028));
    let fin1 = sdTriangle(rq, vec2f(-0.05, 0.05), vec2f(-0.11, 0.14), vec2f(-0.035, 0.11));
    let fin2 = sdTriangle(rq, vec2f(0.05, 0.05), vec2f(0.11, 0.14), vec2f(0.035, 0.11));
    d = min(d, min(abs(fin1), abs(fin2)));
    let fy = rq.y - 0.15;
    let flame = abs(rq.x - 0.018 * sin(fy * 90.0 + ts * 20.0) * clamp(fy * 12.0, 0.0, 1.0)) - 0.0;
    d = min(d, max(flame, max(-fy, fy - 0.07)));
    // planet with a ring
    let pc = vec2f(0.33, -0.12);
    let pl = length(p - pc) - 0.09;
    let rg = sdEllipseApprox(rot2(-0.25) * (p - pc), vec2f(0.17, 0.035));
    let ringFront = max(abs(rg), -(rot2(-0.25) * (p - pc)).y);
    d = min(d, abs(pl));
    d = min(d, select(ringFront, abs(rg), pl > 0.0));
    // smiley sun
    let sc = vec2f(-0.55, -0.27);
    let sd = length(p - sc) - 0.06;
    d = min(d, abs(sd));
    let sa = atan2(p.y - sc.y, p.x - sc.x);
    let ray = abs(fract(sa / TAU * 10.0 + ts * 0.1) - 0.5);
    d = min(d, max(ray * length(p - sc) * 0.6, abs(length(p - sc) - 0.095) - 0.018));
    d = min(d, min(length(p - sc - vec2f(-0.02, -0.012)), length(p - sc - vec2f(0.02, -0.012))) - 0.004);
    let sq = p - sc - vec2f(0.0, -0.005);
    d = min(d, abs(sdArc(vec2f(sq.x, -sq.y), vec2f(sin(0.8), cos(0.8)), 0.032, 0.0)));
    // stars, a spiral, a cube and a dashed flight path
    for (var k = 0; k < 6; k++) {
      let hk = hash12(f32(k) * 7.1 + 2.0);
      let c = vec2f(-0.05 + hk.x * 0.65, -0.42 + hk.y * 0.22 + select(0.0, 0.62, k > 3));
      d = min(d, abs(sdStar5(p - c, 0.025 + 0.012 * hk.x, 0.45)));
    }
    d = min(d, spiralD(p - vec2f(0.6, 0.27), 0.012, 4.0));
    let cq = p - vec2f(-0.5, 0.25);
    let cube = min(abs(sdBox(cq, vec2f(0.06))), abs(sdBox(cq - vec2f(0.035, -0.035), vec2f(0.06))));
    var edges = min(sdSegment(cq, vec2f(-0.06, -0.06), vec2f(-0.025, -0.095)), sdSegment(cq, vec2f(0.06, -0.06), vec2f(0.095, -0.095)));
    edges = min(edges, min(sdSegment(cq, vec2f(0.06, 0.06), vec2f(0.095, 0.025)), sdSegment(cq, vec2f(-0.06, 0.06), vec2f(-0.025, 0.025))));
    d = min(d, min(cube, edges));
    let arc = sdBezier(p, rc + vec2f(0.08, -0.12), vec2f(0.12, -0.38), pc + vec2f(-0.1, -0.04));
    let dash = step(0.5, fract(length(p - rc) * 40.0));
    d = min(d, max(arc, -0.0001 + dash * 1.0));
    // ink with pressure variation
    let pressure = 0.7 + 0.3 * valueNoise(px * 0.03 + vec2f(seed));
    let a = select(1.0, u.sketchy * 0.4, tr == 1);
    col = mix(col, pen, pencil(d * res.y, lw, px) * pressure * a);
    if (tr == 0) {
      // ballpoint hatching on the fins and the planet's shadow side
      let fins = 1.0 - smoothstep(-0.002, 0.002, min(fin1, fin2));
      let shadowSide = (1.0 - smoothstep(-0.002, 0.002, pl)) * smoothstep(0.0, 0.05, dot(p - pc, vec2f(0.6, 0.8)));
      col = mix(col, pen, hatch(0.3 + 0.6 * u.hatch, px, 6.0 * res.y / 900.0, lw * 0.9) * max(fins, shadowSide) * 0.9);
      // hand lettering
      let lab = labelAt(p, 1.0);
      col = mix(col, pen, smoothstep(0.3, 0.7, lab) * pressure);
    }
  }
  return col;
}

// ---------------------------------------------------------------- 2. chalkboard
fn chalk(d: f32, w: f32, px: vec2f) -> f32 {
  let core = lineCov(d, w);
  let grit = smoothstep(0.25, 0.7, valueNoise(px * 0.4) * 0.6 + valueNoise(px * 1.6) * 0.4);
  return core * mix(1.0, grit, 0.85);
}

fn chalkboard(px: vec2f) -> vec3f {
  let res = u.resolution;
  let base = (px - 0.5 * res) / res.y;
  // slate with smudges and eraser swipes
  var col = vec3f(0.13, 0.21, 0.18) * (0.9 + 0.2 * valueNoise(px * 0.004));
  let smudge = fbm(px / res.y * 3.0, 4) * 0.5 + 0.5;
  col += vec3f(0.06, 0.07, 0.07) * smoothstep(0.45, 0.9, smudge) * u.paper;
  let swipe = valueNoise(vec2f(px.x * 0.003 + 1.0, px.y * 0.02)) * valueNoise(vec2f(px.x * 0.01, px.y * 0.004));
  col += vec3f(0.05) * smoothstep(0.25, 0.6, swipe) * u.paper;
  col *= 0.97 + 0.06 * valueNoise(px * 0.8) * u.paper;
  let lw = u.lineWidth * 2.2 * res.y / 900.0;
  let s1 = boilSeed();
  let ts = steppedTime();
  let white = vec3f(0.93, 0.95, 0.92);
  let yellow = vec3f(0.98, 0.9, 0.45);
  let pink = vec3f(0.98, 0.6, 0.72);
  var dustAcc = 0.0;
  for (var tr = 0; tr < 2; tr++) {
    let seed = s1 + f32(tr) * 5.1;
    let wpx = px + wob(px, seed) * (0.8 + 0.6 * f32(tr));
    let p = (wpx - 0.5 * res) / res.y;
    let a = select(1.0, u.sketchy * 0.35, tr == 1);
    // left: axes + sine graph
    let o = vec2f(-0.62, 0.05);
    var dw = min(sdSegment(p, o + vec2f(0.0, 0.17), o + vec2f(0.0, -0.2)), sdSegment(p, o, o + vec2f(0.5, 0.0)));
    dw = min(dw, min(sdSegment(p, o + vec2f(0.5, 0.0), o + vec2f(0.47, -0.015)), sdSegment(p, o + vec2f(0.5, 0.0), o + vec2f(0.47, 0.015))));
    dw = min(dw, min(sdSegment(p, o + vec2f(0.0, -0.2), o + vec2f(-0.015, -0.17)), sdSegment(p, o + vec2f(0.0, -0.2), o + vec2f(0.015, -0.17))));
    let gx = clamp(p.x - o.x, 0.0, 0.46);
    let sy = o.y - 0.11 * sin(gx * 18.0);
    let slope = 0.11 * 18.0 * cos(gx * 18.0);
    let dsin = abs(p.y - sy) / sqrt(1.0 + slope * slope) + max(abs(p.x - o.x - 0.23) - 0.23, 0.0);
    // the game level sketch on the right
    let gnd = sdSegment(p, vec2f(-0.05, 0.33), vec2f(0.82, 0.33));
    let plat1 = abs(sdBox(p - vec2f(0.28, 0.16), vec2f(0.09, 0.022)));
    let plat2 = abs(sdBox(p - vec2f(0.62, 0.02), vec2f(0.1, 0.022)));
    dw = min(dw, min(gnd, min(plat1, plat2)));
    // a stick figure that hops (on twos)
    let hop = abs(sin(ts * 2.2)) * 0.05;
    let sf = vec2f(0.02, 0.33 - hop);
    var fig = abs(length(p - sf - vec2f(0.0, -0.13)) - 0.025);
    fig = min(fig, sdSegment(p, sf + vec2f(0.0, -0.105), sf + vec2f(0.0, -0.05)));
    fig = min(fig, min(sdSegment(p, sf + vec2f(0.0, -0.05), sf + vec2f(-0.025, 0.0)), sdSegment(p, sf + vec2f(0.0, -0.05), sf + vec2f(0.025, 0.0))));
    fig = min(fig, min(sdSegment(p, sf + vec2f(0.0, -0.09), sf + vec2f(-0.03, -0.065)), sdSegment(p, sf + vec2f(0.0, -0.09), sf + vec2f(0.035, -0.11))));
    dw = min(dw, fig);
    // dashed jump arc + arrow head (pink)
    let jp = sdBezier(p, sf + vec2f(0.03, -0.16), vec2f(0.33, -0.3), vec2f(0.55, -0.03));
    let dashes = step(0.45, fract((p.x + 0.2) * 32.0));
    var dp = max(jp, dashes * 0.01);
    dp = min(dp, min(sdSegment(p, vec2f(0.55, -0.03), vec2f(0.52, -0.065)), sdSegment(p, vec2f(0.55, -0.03), vec2f(0.505, -0.02))));
    // coins (yellow) and a spiky enemy
    var dy = dsin;
    for (var k = 0; k < 3; k++) {
      let cc = vec2f(0.56 + 0.05 * f32(k), -0.06 + 0.01 * sin(ts * 3.0 + f32(k)));
      dy = min(dy, abs(length((p - cc) * vec2f(1.4, 1.0)) - 0.016));
    }
    let en = p - vec2f(0.45, 0.33);
    let spikes = abs(en.y + 0.035 + 0.015 * abs(fract(en.x * 30.0) - 0.5) * 2.0) + max(abs(en.x) - 0.05, 0.0);
    var de = min(abs(sdBox(en - vec2f(0.0, -0.017), vec2f(0.05, 0.017))), spikes);
    de = min(de, length(en - vec2f(0.018, -0.022)) - 0.005);
    // hand-written labels
    let lab = labelAt(p, 0.0);
    col = mix(col, white, chalk(dw * res.y, lw, px) * a);
    col = mix(col, yellow, chalk(dy * res.y, lw, px) * a);
    col = mix(col, pink, chalk(min(dp, de) * res.y, lw, px) * a);
    col = mix(col, white, smoothstep(0.25, 0.75, lab) * mix(1.0, smoothstep(0.2, 0.6, valueNoise(px * 0.5)), 0.7) * a);
    if (tr == 0) { dustAcc = exp(-min(min(dw, dy), min(dp, de)) * res.y / (6.0 * lw)); }
  }
  // chalk dust settling around the strokes
  col += vec3f(0.08) * dustAcc * valueNoise(px * 0.25) * u.paper;
  // wooden frame + tray
  let fr = min(min(px.x, res.x - px.x), min(px.y, res.y - px.y));
  let wood = vec3f(0.45, 0.28, 0.15) * (0.85 + 0.15 * valueNoise(vec2f(px.x * 0.01, px.y * 0.3))) * (0.9 + 0.1 * valueNoise(vec2f(px.y * 0.01, px.x * 0.3)));
  col = mix(col, wood, 1.0 - smoothstep(res.y * 0.028, res.y * 0.028 + 1.5, fr));
  col *= 1.0 - 0.35 * (1.0 - smoothstep(res.y * 0.028, res.y * 0.06, fr));
  return col;
}

// ---------------------------------------------------------------- 3. the game as a sketch
fn lumAt(sp: vec2f) -> f32 { return luma(TEX(game, sp / u.resolution).rgb); }
fn edgeAt(sp: vec2f, k: f32) -> f32 {
  let l00 = lumAt(sp + vec2f(-k, -k)); let l10 = lumAt(sp + vec2f(0.0, -k)); let l20 = lumAt(sp + vec2f(k, -k));
  let l01 = lumAt(sp + vec2f(-k, 0.0)); let l21 = lumAt(sp + vec2f(k, 0.0));
  let l02 = lumAt(sp + vec2f(-k, k)); let l12 = lumAt(sp + vec2f(0.0, k)); let l22 = lumAt(sp + vec2f(k, k));
  let gx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
  let gy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
  return length(vec2f(gx, gy));
}

fn sketchGame(px: vec2f) -> vec3f {
  let res = u.resolution;
  let s1 = boilSeed();
  let paper = paperCol(px, vec3f(0.97, 0.95, 0.89));
  // colored-pencil fill: the image, slightly posterised, laid down with crayon texture
  let fo = wob(px, s1 + 3.0) * 0.5;
  var src = TEX(game, (px + fo) / res).rgb;
  src = floor(src * 5.0 + 0.5) / 5.0;
  let fillCov = crayon(1.0, px, 0.55) * 0.85 + 0.1;
  var col = mix(paper, paper * mix(vec3f(1.0), src * 1.1, 0.95), fillCov);
  // shadows get pencil hatching
  let tone = 1.0 - luma(src);
  col = mix(col, vec3f(0.18, 0.16, 0.2), hatch((tone - 0.35) * 1.6 * u.hatch, px, 6.0 * res.y / 900.0, 1.2) * 0.75);
  // ink outlines from edge detection, sampled at boiling (wobbled) positions, traced twice
  let k = max(1.0, u.lineWidth * 0.8);
  for (var tr = 0; tr < 2; tr++) {
    let sp = px + wob(px, s1 + f32(tr) * 9.0) * (1.0 + f32(tr) * 0.5);
    let e = edgeAt(sp, k);
    let line = smoothstep(0.18, 0.45, e) * mix(1.0, smoothstep(0.1, 0.5, tooth(px)), u.paper * 0.6);
    let a = select(1.0, u.sketchy * 0.5, tr == 1);
    col = mix(col, vec3f(0.08, 0.07, 0.1), line * a);
  }
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = boilScene(px); }
  else if (ex == 1) { col = notebook(px); }
  else if (ex == 2) { col = chalkboard(px); }
  else { col = sketchGame(px); }
  // soft vignette, like a photographed page
  let v = uv - 0.5;
  col *= 1.0 - dot(v, v) * 0.35;
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export default shaderScene({
  examples: [
    {
      id: 'boil',
      label: 'Line boil & wobble',
      kind: 'Abstract',
      note: 'Every line is re-drawn 8 times a second with a different wobble, cycling through 3 “drawings” — the signature shimmer of hand-animated cartoons. Fills are crayon (streaky) and pencil hatching; a second, lighter tracing makes it sketchy. Set <i>Boil rate</i> to 0 to freeze it.',
    },
    {
      id: 'notebook',
      label: 'Notebook doodles',
      kind: 'Real life',
      note: 'Blue ballpoint on lined paper. The printed ruling and margin stay perfectly still — only the ink “boils”. Pressure varies along the strokes, hatching shades the fins and the planet.',
      params: { lineWidth: 1.7, wobble: 2.0, sketchy: 0.35 },
    },
    {
      id: 'chalk',
      label: 'Chalkboard',
      kind: 'Real life',
      note: 'Chalk = thick lines broken up by high-frequency noise (the board’s grain catches some chalk and skips other spots), colored accents, dust around the strokes and eraser smudges in the background.',
      params: { lineWidth: 2.0, wobble: 2.5, sketchy: 0.4 },
    },
    {
      id: 'game',
      label: 'Sketchy game',
      kind: 'In a game',
      note: 'The platformer redrawn as a living sketch: edge detection for ink outlines (sampled at wobbling positions so the outlines boil), colored-pencil fills with paper showing through, and hatching in the shadows.',
      params: { lineWidth: 1.6, wobble: 2.0, sketchy: 0.5 },
    },
  ],
  controls: [
    { type: 'slider', key: 'boilFps', label: 'Boil rate (drawings/s)', min: 0, max: 24, step: 1, value: 8, help: '0 = still. Cartoons usually re-trace at 8–12 per second (“on twos/threes”).' },
    { type: 'slider', key: 'cycle', label: 'Drawings in the loop', min: 2, max: 12, step: 1, value: 3, help: 'How many different tracings before it repeats. 2–3 is the classic boil.' },
    { type: 'slider', key: 'wobble', label: 'Wobble (px)', min: 0, max: 8, step: 0.1, value: 2.5, help: 'Strength of the noise offset added to every position before drawing.' },
    { type: 'slider', key: 'lineWidth', label: 'Line width', min: 0.6, max: 5, step: 0.1, value: 1.8 },
    { type: 'slider', key: 'sketchy', label: 'Sketchy double lines', min: 0, max: 1, step: 0.01, value: 0.5, help: 'A second, lighter tracing with its own wobble.' },
    { type: 'slider', key: 'hatch', label: 'Hatching density', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['boil', 'notebook', 'game'], help: 'More tone = more crossing layers of strokes.' },
    { type: 'slider', key: 'paper', label: 'Paper / board texture', min: 0, max: 1, step: 0.01, value: 0.7, help: 'Grain, fibers and how much the surface breaks up the strokes.' },
  ],
  uniforms: { boilFps: 'f32', cycle: 'f32', wobble: 'f32', lineWidth: 'f32', sketchy: 'f32', hatch: 'f32', paper: 'f32' },
  include: [...new Set(['hash', 'noise', 'sdf', 'color', 'math', ...GAME_SCENE_INCLUDES])],
  textures: { labels: { source: async () => buildLabels() } },
  // WORKAROUND (core bug): input: 'game' binds the game target while rendering into it on WebGPU,
  // so the platformer image is rendered as an ordinary first pass named "game" instead.
  passes: [{ name: 'game', format: 'rgba8unorm', code: GAME_SCENE_CODE }],
  code: CODE,
  about: {
    summary: 'Make clean vector shapes look drawn by hand: wobbling lines that “boil”, paper grain, crayon, pencil hatching, ballpoint and chalk.',
    what: `<p>Shapes are signed distance functions (perfect math circles, stars, hearts…). Everything that makes them look hand-made happens
      <i>around</i> them: the position is jittered before drawing, the line is broken up by paper texture, fills are streaky or hatched,
      and the jitter changes a few times per second.</p>`,
    how: `<ol>
      <li><b>Wobble (domain warp).</b> Before evaluating a shape, add a smooth noise offset to the pixel position: <code>p += noise(p, seed) × wobble</code>. Straight lines become slightly wavy, circles lumpy.</li>
      <li><b>Line boil.</b> The noise <code>seed</code> only changes <code>boilFps</code> times per second and cycles through a few values: <code>seed = floor(time × fps) mod N</code>. Each value is one “drawing”.</li>
      <li><b>Stepped motion.</b> Animation also uses <code>floor(time × fps) / fps</code>, so things move in the same choppy, hand-animated rhythm.</li>
      <li><b>Sketchy lines.</b> The same shapes are traced a second time with a different seed and lower opacity.</li>
      <li><b>Media.</b> Pencil/chalk lines are multiplied by a paper “tooth” noise; crayon fills use noise stretched along a stroke direction; hatching is
        <code>fract(rotated.y / spacing)</code> — 1, 2 or 3 crossing layers depending on how dark the area should be.</li>
      <li><b>Sketch filter.</b> For the game image, a Sobel edge detector runs on wobbled sample positions → boiling ink outlines; the colors are posterised and laid down as colored pencil.</li>
    </ol>`,
    uses: [
      { title: 'Hand-drawn games', text: 'Cuphead (1930s cartoon), Don’t Starve, Hollow Knight, Scribblenauts, Crayon Physics Deluxe and Doodle Jump (literally on notebook paper) all lean on drawn linework.' },
      { title: 'Line boil', text: 'Classic in TV animation — Ed, Edd n Eddy and “Squigglevision” (Dr. Katz) — and used in games like Wobbledogs, Line Rider and many jam games to make still frames feel alive.' },
      { title: 'UI & menus', text: 'Sketchy buttons, notebook-paper menus, chalkboard tutorials and hand-drawn maps.' },
      { title: 'Filters', text: 'Comic-book or storyboard modes, “memory” and dream sequences, photo mode filters.' },
    ],
    try: [
      'Set <i>Boil rate</i> to 0 and back to 8: the same picture goes from dead to alive.',
      'Set <i>Drawings in the loop</i> to 2 and <i>Boil rate</i> to 4 — a slow, deliberate Ed-Edd-n-Eddy shimmer.',
      'Push <i>Wobble</i> to 8 on the notebook: the doodles get drunk, the printed lines stay straight.',
      'On <b>Sketchy game</b>, raise <i>Hatching density</i> — the shadows fill with pencil strokes.',
      'Set <i>Paper texture</i> to 0: lines become clean and digital.',
    ],
    ask: [
      'line boil on outlines at 8–12 fps (cycle 3 drawings)',
      'wobbly hand-drawn SDF outlines with domain-warp noise',
      'paper texture with crayon fill and pencil hatching',
      'chalkboard shader with broken chalk lines and dust',
      'sketch filter: boiling ink outlines over the game',
      'animate on twos (stepped time) for a hand-animated feel',
    ],
    perf: `<p>Cheap to moderate. The doodle scenes evaluate a few dozen distance functions twice (two tracings) plus some value noise per pixel.
      The sketch filter takes ~18 texture reads per pixel for the two Sobel tracings. Boil itself is free — it’s just a different noise seed.</p>`,
    api: `<p>Plain fragment shaders: identical on WebGPU and WebGL2. In a sprite-based game you would apply the wobble in the <i>vertex</i> shader
      (jitter the outline mesh) or as a screen-space UV offset in a post pass, with the same stepped seed.</p>`,
    code: [
      {
        title: 'Line boil: a seed that changes a few times per second',
        lang: 'wgsl',
        src: `fn boilSeed() -> f32 {
  if (u.boilFps < 0.5) { return 0.0; }
  return fmod(floor(u.time * u.boilFps), u.cycle) + 1.0;   // cycles through N drawings
}
fn wob(px: vec2f, seed: f32) -> vec2f {                      // offset in pixels
  let q = px / u.resolution.y * 7.0;
  let s = vec2f(seed * 3.17, seed * 1.73);
  let big = vec2f(perlin(q + s), perlin(q + vec2f(7.3, 2.9) + s.yx));
  return big * u.wobble * u.resolution.y / 900.0;
}
// draw the shapes at the wobbled position:
let wpx = px + wob(px, boilSeed());
let d = sdStar5(rot2(0.1) * (wpx / u.resolution.y - c), 0.13, 0.45) * u.resolution.y;`,
      },
      {
        title: 'Media: pencil, crayon, hatching',
        lang: 'wgsl',
        src: `fn pencil(d: f32, w: f32, px: vec2f) -> f32 {        // line broken up by paper tooth
  return lineCov(d, w) * mix(1.0, smoothstep(0.12, 0.5, tooth(px)), u.paper * 0.7);
}
fn crayon(inside: f32, px: vec2f, angle: f32) -> f32 { // streaks along a direction
  let r = rot2(angle) * px;
  let streak = valueNoise(vec2f(r.x * 0.05, r.y * 0.8)) * 0.55 + valueNoise(px * 0.9) * 0.45;
  return inside * smoothstep(0.22, 0.55, streak);
}
// hatching layer k: parallel lines at angle 0.75 + 1.1 k
let h = abs(fract(r.y / spacing + jit) - 0.5) * spacing;`,
      },
    ],
  },
});

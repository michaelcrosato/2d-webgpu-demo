// Masks & stencils: a soft light mask (flashlight in the dark), a real hardware STENCIL buffer
// (depth24plus-stencil8) for an x-ray lens and shaped windows into another world, and a persistent
// mask texture painted with the mouse (scratch card).

import { Camera2D, ShapeBatch } from '../../core/batch.js';
import { getAtlas, makeCanvas } from '../../core/assets.js';
import { createGameScene } from '../../core/gamescene.js';
import { overlayTag, clamp } from './_shared.js';

const DS = 'depth24plus-stencil8';
const CARD_W = 640;
const CARD_H = 400;

// ------------------------------------------------------------------------------------ x-ray world
const XRAY = /* wgsl */ `
fn seg(p: vec2f, a: vec2f, b: vec2f) -> f32 { return sdSegment(p, a, b); }
fn aaf(d: f32, w: f32) -> f32 { return clamp(0.5 - d / w, 0.0, 1.0); }
// glow line for the x-ray look
fn xl(d: f32, w: f32) -> f32 { return exp(-max(d, 0.0) / w) * 0.6 + aaf(abs(d) - w * 0.4, w); }

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let H = u.res.y;
  let p = px / H;                       // y 0..1 down, x 0..aspect
  let a = u.res.x / H;
  let pw = 1.5 / H;
  let gy = 0.62;                        // ground line
  let hx = a * 0.3;                     // house left
  let hw = 0.55;
  let houseD = sdBox(p - vec2f(hx + hw * 0.5, 0.46), vec2f(hw * 0.5, 0.16));
  let roofD = sdTriangle(p, vec2f(hx - 0.04, 0.305), vec2f(hx + hw + 0.04, 0.305), vec2f(hx + hw * 0.5, 0.14));
  let treeX = hx + hw + 0.32;
  let carX = a * 0.86;
  let skel = vec2f(hx + 0.15, 0.86);
  let chest = vec2f(treeX + 0.12, 0.84);
  if (u.layer < 0.5) {
    // ---------------- the normal world
    var c = mix(vec3f(0.35, 0.5, 0.85), vec3f(0.95, 0.75, 0.6), smoothstep(0.0, gy, p.y));
    c += vec3f(1.0, 0.8, 0.5) * exp(-length(p - vec2f(a * 0.12, 0.2)) * 7.0) * 0.4;
    if (p.y > gy) {
      let band = 0.5 + 0.5 * sin(p.y * 60.0 + valueNoise(p * 9.0) * 4.0);
      c = mix(vec3f(0.45, 0.3, 0.18), vec3f(0.36, 0.23, 0.14), band) * (0.85 + 0.15 * valueNoise(p * 40.0));
      c *= mix(1.0, 0.55, smoothstep(gy, 1.0, p.y));
      c = mix(c, vec3f(0.3, 0.6, 0.25), aaf(p.y - gy - 0.018, pw));
      let peb = voronoi(p * 30.0);
      c = mix(c, vec3f(0.55, 0.5, 0.45), smoothstep(0.2, 0.12, peb.x) * step(0.75, hash21(peb.zw)));
    }
    // tree
    let trunk = sdBox(p - vec2f(treeX, gy - 0.1), vec2f(0.018, 0.1));
    c = mix(c, vec3f(0.35, 0.22, 0.13), aaf(trunk, pw));
    let crown = min(length(p - vec2f(treeX, gy - 0.27)) - 0.12, min(length(p - vec2f(treeX - 0.08, gy - 0.2)) - 0.08, length(p - vec2f(treeX + 0.09, gy - 0.21)) - 0.08));
    c = mix(c, mix(vec3f(0.15, 0.4, 0.18), vec3f(0.3, 0.6, 0.25), smoothstep(0.1, -0.1, p.y - (gy - 0.3))), aaf(crown, pw));
    // house: brick wall, roof, door, windows
    if (houseD < pw) {
      var bq = p / vec2f(0.04, 0.02);
      bq.x += 0.5 * floor(bq.y);
      let bf = fract(bq);
      let mortar = smoothstep(0.0, 0.1, min(min(bf.x, 1.0 - bf.x), min(bf.y, 1.0 - bf.y)));
      var hc = mix(vec3f(0.85, 0.8, 0.72), mix(vec3f(0.62, 0.27, 0.2), vec3f(0.74, 0.35, 0.24), hash21(floor(bq))), mortar);
      for (var i = 0; i < 4; i++) {
        let wp = vec2f(hx + 0.08 + f32(i % 2) * 0.24 + step(1.5, f32(i)) * 0.15 * 0.0, 0.38 + step(1.5, f32(i)) * 0.15);
        let wd = sdBox(p - wp, vec2f(0.045, 0.04));
        hc = mix(hc, vec3f(0.95, 0.85, 0.5), aaf(wd, pw));
        hc = mix(hc, vec3f(0.25, 0.2, 0.18), aaf(abs(wd) - 0.004, pw));
        hc = mix(hc, vec3f(0.25, 0.2, 0.18), aaf(min(abs(p.x - wp.x), abs(p.y - wp.y)) - 0.003, pw) * step(wd, 0.0));
      }
      let door = sdBox(p - vec2f(hx + hw - 0.12, gy - 0.07), vec2f(0.045, 0.07));
      hc = mix(hc, vec3f(0.35, 0.2, 0.12), aaf(door, pw));
      c = mix(c, hc, aaf(houseD, pw));
    }
    c = mix(c, vec3f(0.3, 0.15, 0.15), aaf(roofD, pw));
    c = mix(c, vec3f(0.45, 0.22, 0.2), aaf(roofD + 0.01, pw) * step(0.5, fract(p.x * 40.0 + p.y * 20.0)));
    // car
    let carB = sdRoundBox(p - vec2f(carX, gy - 0.05), vec2f(0.13, 0.035), 0.02);
    let carT = sdRoundBox(p - vec2f(carX - 0.01, gy - 0.1), vec2f(0.07, 0.03), 0.02);
    c = mix(c, vec3f(0.2, 0.45, 0.8), aaf(min(carB, carT), pw));
    c = mix(c, vec3f(0.7, 0.85, 0.95), aaf(sdBox(p - vec2f(carX - 0.01, gy - 0.1), vec2f(0.055, 0.018)), pw));
    let wh = min(length(p - vec2f(carX - 0.08, gy - 0.015)), length(p - vec2f(carX + 0.08, gy - 0.015))) - 0.028;
    c = mix(c, vec3f(0.1), aaf(wh, pw));
    return vec4f(c, 1.0);
  }
  // ---------------- the x-ray layer: same layout, what is hidden inside
  var c = vec3f(0.02, 0.04, 0.08) + vec3f(0.0, 0.05, 0.1) * (1.0 - p.y);
  let g = abs(fract(p * 20.0) - vec2f(0.5));
  c += vec3f(0.0, 0.06, 0.08) * smoothstep(0.47, 0.5, max(g.x, g.y));
  let cy = vec3f(0.3, 0.85, 1.0);
  c += cy * xl(abs(p.y - gy), 0.002) * 0.5;
  c += cy * xl(abs(houseD), 0.002) * 0.6 + cy * xl(abs(roofD), 0.002) * 0.6;
  // second floor & furniture
  c += cy * xl(seg(p, vec2f(hx, 0.46), vec2f(hx + hw, 0.46)), 0.002) * 0.5;
  let bed = sdBox(p - vec2f(hx + 0.12, gy - 0.025), vec2f(0.08, 0.02));
  let lamp = min(seg(p, vec2f(hx + 0.3, gy), vec2f(hx + 0.3, gy - 0.09)), length(p - vec2f(hx + 0.3, gy - 0.1)) - 0.02);
  let shelf = sdBox(p - vec2f(hx + 0.42, 0.39), vec2f(0.05, 0.06));
  c += cy * (xl(abs(bed), 0.002) + xl(lamp, 0.002) + xl(abs(shelf), 0.002)) * 0.55;
  // wiring (yellow) from the fuse box
  let y1 = vec3f(1.0, 0.85, 0.25);
  let fuse = sdBox(p - vec2f(hx + 0.03, 0.52), vec2f(0.012, 0.02));
  var wire = min(seg(p, vec2f(hx + 0.03, 0.5), vec2f(hx + 0.03, 0.33)), seg(p, vec2f(hx + 0.03, 0.33), vec2f(hx + hw - 0.03, 0.33)));
  wire = min(wire, min(seg(p, vec2f(hx + 0.2, 0.33), vec2f(hx + 0.2, 0.36)), seg(p, vec2f(hx + 0.44, 0.33), vec2f(hx + 0.44, 0.36))));
  wire = min(wire, seg(p, vec2f(hx + 0.03, 0.54), vec2f(hx + 0.03, gy + 0.05)));
  let pulse = 0.6 + 0.4 * sin(u.time * 6.0 - p.x * 30.0);
  c += y1 * (xl(wire, 0.0015) * pulse + xl(abs(fuse), 0.002));
  // plumbing (blue) down into the water main
  let b1 = vec3f(0.3, 0.55, 1.0);
  var pipe = min(seg(p, vec2f(hx + hw - 0.2, 0.4), vec2f(hx + hw - 0.2, 0.72)), seg(p, vec2f(0.0, 0.72), vec2f(a, 0.72)));
  pipe = min(pipe, seg(p, vec2f(hx + hw - 0.2, 0.4), vec2f(hx + hw - 0.08, 0.4)));
  c += b1 * xl(abs(pipe - 0.006), 0.0015);
  let flow = step(0.5, fract(p.x * 12.0 - u.time * 1.5)) * step(abs(p.y - 0.72), 0.004);
  c += b1 * flow * 0.6;
  // gas line (orange)
  let o1 = vec3f(1.0, 0.5, 0.2);
  c += o1 * xl(abs(abs(p.y - 0.78) - 0.004), 0.0015) * 0.8;
  // tree roots
  var roots = 1e3;
  for (var i = 0; i < 5; i++) {
    let fi = f32(i) - 2.0;
    let e = vec2f(treeX + fi * 0.07, gy + 0.1 + abs(fi) * 0.02);
    roots = min(roots, seg(p, vec2f(treeX, gy), e));
    roots = min(roots, seg(p, e, e + vec2f(fi * 0.03, 0.06)));
  }
  c += vec3f(0.6, 0.9, 0.5) * xl(roots, 0.0015) * 0.6;
  // a buried skeleton
  let w1 = vec3f(0.92, 0.97, 1.0);
  var bone = length(p - skel) - 0.022;
  bone = min(bone, seg(p, skel + vec2f(0.03, 0.0), skel + vec2f(0.17, 0.005)) - 0.004);
  for (var r = 0; r < 4; r++) {
    let rx = skel.x + 0.06 + f32(r) * 0.025;
    bone = min(bone, abs(length((p - vec2f(rx, skel.y)) * vec2f(3.0, 1.0)) - 0.06) - 0.002);
  }
  bone = min(bone, seg(p, skel + vec2f(0.17, 0.005), skel + vec2f(0.28, -0.02)) - 0.004);
  bone = min(bone, seg(p, skel + vec2f(0.17, 0.005), skel + vec2f(0.28, 0.03)) - 0.004);
  bone = min(bone, seg(p, skel + vec2f(0.07, 0.0), skel + vec2f(0.12, 0.05)) - 0.003);
  let eye = min(length(p - skel - vec2f(-0.007, -0.005)), length(p - skel - vec2f(0.008, -0.005))) - 0.005;
  c += w1 * (aaf(bone, pw) * 0.85 + exp(-max(bone, 0.0) / 0.004) * 0.3) * (1.0 - aaf(eye, pw));
  // treasure
  let ch = sdBox(p - chest, vec2f(0.05, 0.03));
  c += vec3f(1.0, 0.8, 0.3) * (xl(abs(ch), 0.002) + exp(-length(p - chest) / 0.05) * 0.35 * (0.7 + 0.3 * sin(u.time * 3.0)));
  // car engine & wheels
  c += cy * xl(abs(sdRoundBox(p - vec2f(carX, gy - 0.07), vec2f(0.13, 0.06), 0.02)), 0.002) * 0.4;
  c += o1 * xl(abs(sdBox(p - vec2f(carX + 0.07, gy - 0.05), vec2f(0.035, 0.02))), 0.002);
  let whx = min(length(p - vec2f(carX - 0.08, gy - 0.015)), length(p - vec2f(carX + 0.08, gy - 0.015)));
  c += cy * xl(abs(whx - 0.028), 0.0015) + cy * xl(abs(whx - 0.012), 0.0015) * 0.6;
  return vec4f(c, 1.0);
}
`;

// ------------------------------------------------------------------------------------ space (the "other world")
const SPACE = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let H = u.res.y;
  let p = (px - 0.5 * u.res) / H;
  var c = vec3f(0.01, 0.0, 0.04);
  let n = fbm(p * 2.2 + vec2f(u.time * 0.01, 0.0), 5);
  let n2 = fbm(p * 4.0 - vec2f(0.0, u.time * 0.015) + n, 4);
  c += vec3f(0.5, 0.1, 0.6) * smoothstep(0.0, 0.8, n2) * 0.7 + vec3f(0.1, 0.3, 0.7) * smoothstep(-0.2, 0.6, n) * 0.5;
  // stars
  for (var k = 0; k < 3; k++) {
    let sc = 40.0 + f32(k) * 50.0;
    let id = floor(p * sc);
    let f = fract(p * sc) - vec2f(0.5);
    let h = hash22(id + f32(k) * 13.0);
    let st = exp(-length(f - (h - vec2f(0.5)) * 0.7) * (18.0 + f32(k) * 8.0)) * step(0.8, hash21(id + 3.0));
    c += vec3f(0.9, 0.95, 1.0) * st * (0.6 + 0.4 * sin(u.time * 3.0 + h.x * 20.0));
  }
  // ringed planet
  let pp = p - vec2f(0.25, -0.05);
  let pd = length(pp) - 0.16;
  let lit = clamp(dot(normalize(vec3f(pp, sqrt(max(0.0, 0.0256 - dot(pp, pp))))), normalize(vec3f(-0.6, -0.5, 0.6))), 0.0, 1.0);
  let bands = 0.5 + 0.5 * sin(pp.y * 60.0 + sin(pp.x * 10.0) * 2.0);
  let pc = mix(vec3f(0.9, 0.55, 0.3), vec3f(1.0, 0.85, 0.6), bands) * (0.15 + lit);
  let rq = rot2(0.35) * pp;
  let ring = abs(length(rq * vec2f(1.0, 3.5)) - 0.27) - 0.035;
  let ringFront = step(0.0, rq.y);
  c = mix(c, vec3f(0.9, 0.8, 0.6) * 0.8, clamp(0.5 - ring * H * 0.5, 0.0, 1.0) * (1.0 - ringFront));
  c = mix(c, pc, clamp(0.5 - pd * H, 0.0, 1.0));
  c = mix(c, vec3f(0.95, 0.85, 0.65), clamp(0.5 - ring * H * 0.5, 0.0, 1.0) * ringFront * 0.9);
  c += vec3f(0.6, 0.4, 1.0) * exp(-max(pd, 0.0) * 30.0) * 0.25;
  return vec4f(c, 1.0);
}
`;

// ------------------------------------------------------------------------------------ flashlight composite
const SPOT = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let s = TEX(scene, uv).rgb;
  let L = TEX(light, uv).r;
  if (u.showMask > 0.5) { return vec4f(vec3f(L), 1.0); }
  // darkness is not black: a dim, blue "moonlight" version of the scene
  let night = s * vec3f(0.07, 0.09, 0.2) + vec3f(0.0, 0.005, 0.02);
  let lit = s * vec3f(1.05, 0.95, 0.8);
  var c = mix(night, lit, clamp(L, 0.0, 1.0));
  // dust motes in the beam
  let d = valueNoise(px / 6.0 + vec2f(u.time * 4.0, sin(u.time) * 3.0));
  c += vec3f(1.0, 0.9, 0.7) * smoothstep(0.82, 0.95, d) * clamp(L - 0.3, 0.0, 1.0) * 0.25;
  return vec4f(c, 1.0);
}
`;

// ------------------------------------------------------------------------------------ x-ray (alpha mask variant)
const LENS_ALPHA = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let a = TEX(normal, uv).rgb;
  let b = TEX(xray, uv).rgb;
  let d = length(px - u.lens.xy) - u.lens.z;
  let m = 1.0 - smoothstep(-u.lens.w, u.lens.w * 0.2, d);    // soft, feathered edge
  return vec4f(mix(a, b, m), 1.0);
}
`;

// ------------------------------------------------------------------------------------ scratch card
const CARD = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let r = u.card;                                   // x, y, w, h in px
  let q = (px - r.xy) / r.zw;
  // the table
  var c = mix(vec3f(0.1, 0.32, 0.22), vec3f(0.05, 0.18, 0.13), length(uv - vec2f(0.5)) * 1.3);
  c *= 0.92 + 0.08 * valueNoise(px / 3.0);
  let sh = sdRoundBox(px - (r.xy + r.zw * 0.5 + vec2f(8.0, 12.0)), r.zw * 0.5, 18.0);
  c *= 1.0 - 0.45 * exp(-max(sh, 0.0) / 14.0);
  let cd = sdRoundBox(px - (r.xy + r.zw * 0.5), r.zw * 0.5, 16.0);
  if (cd < 1.0) {
    let prizeC = TEX(prize, q).rgb;
    let foilR = TEX(foil, q);                       // alpha = scratchable panel
    let ms = clamp(TEX(mask, q).r + u.reveal, 0.0, 1.0);
    // metallic sheen that follows the mouse (fake tilt)
    let sheen = 0.5 + 0.5 * sin((q.x * 1.3 + q.y * 0.7) * 9.0 - u.tilt.x * 4.0 + u.time * 0.4);
    let holo = hsv2rgb(vec3f(fract(q.x * 0.8 - q.y * 0.5 + u.tilt.y * 0.6), 0.35, 1.0));
    let foilC = foilR.rgb * (0.85 + 0.25 * sheen) + holo * 0.12;
    // scratched edge: a thin darker lip where the mask changes
    let e = u.texel;
    let mx = TEX(mask, q + vec2f(e.x * 2.0, 0.0)).r - TEX(mask, q - vec2f(e.x * 2.0, 0.0)).r;
    let my = TEX(mask, q + vec2f(0.0, e.y * 2.0)).r - TEX(mask, q - vec2f(0.0, e.y * 2.0)).r;
    let lip = clamp(length(vec2f(mx, my)) * 1.5, 0.0, 1.0) * foilR.a * (1.0 - u.reveal);
    let cover = foilR.a * (1.0 - ms);
    var cc = mix(prizeC, foilC, cover);
    cc = mix(cc, cc * 0.55, lip * (1.0 - cover * 0.5));
    c = mix(c, cc, clamp(0.5 - cd, 0.0, 1.0));
  }
  return vec4f(c, 1.0);
}
`;

// textured fullscreen blit used inside the stencil pass
const BLIT = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
  return o;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f { return textureSampleLevel(src, samp, i.uv, 0.0); }
`;
// stencil-only shapes: triangles in pixel space, color writes disabled
const SHAPES = /* wgsl */ `
@vertex fn vs_main(@location(0) p: vec2f) -> @builtin(position) vec4f {
  return vec4f(p.x / su.res.x * 2.0 - 1.0, 1.0 - p.y / su.res.y * 2.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

// ------------------------------------------------------------------------------------ shape polygons
function fan(out, cx, cy, pts) {
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    out.push(cx, cy, a[0], a[1], b[0], b[1]);
  }
}
function starPts(cx, cy, r, rot, n = 5, inner = 0.45) {
  const pts = [];
  for (let i = 0; i < n * 2; i++) {
    const a = rot + (i / (n * 2)) * Math.PI * 2 - Math.PI / 2;
    const rr = i % 2 ? r * inner : r;
    pts.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]);
  }
  return pts;
}
function circlePts(cx, cy, r, n = 64) {
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([cx + Math.cos((i / n) * Math.PI * 2) * r, cy + Math.sin((i / n) * Math.PI * 2) * r]);
  return pts;
}
function heartPts(cx, cy, s, n = 64) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const x = 16 * Math.sin(t) ** 3;
    const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    pts.push([cx + (x / 17) * s, cy - (y / 17) * s]);
  }
  return pts;
}

// ------------------------------------------------------------------------------------ card art
function paintCards(atlas) {
  const prize = makeCanvas(CARD_W, CARD_H);
  const g = prize.getContext('2d');
  let grad = g.createLinearGradient(0, 0, CARD_W, CARD_H);
  grad.addColorStop(0, '#ffcd75');
  grad.addColorStop(0.5, '#ef7d57');
  grad.addColorStop(1, '#b13e53');
  g.fillStyle = grad;
  g.fillRect(0, 0, CARD_W, CARD_H);
  g.globalAlpha = 0.15;
  for (let i = 0; i < 18; i++) {
    g.fillStyle = '#fff';
    g.beginPath();
    g.moveTo(CARD_W / 2, CARD_H * 0.55);
    const a0 = (i / 18) * Math.PI * 2;
    g.arc(CARD_W / 2, CARD_H * 0.55, 600, a0, a0 + 0.17);
    g.fill();
  }
  g.globalAlpha = 1;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.font = 'bold 34px sans-serif';
  g.fillStyle = '#1a1c2c';
  g.fillText('SHADER LOTTO', CARD_W / 2, 42);
  g.imageSmoothingEnabled = false;
  const sym = ['gem', 'gem', 'gem'];
  sym.forEach((n, i) => {
    const f = atlas.frames[n];
    const x = CARD_W / 2 + (i - 1) * 150;
    const y = 210;
    g.fillStyle = 'rgba(255,255,255,0.85)';
    g.beginPath();
    g.roundRect(x - 62, y - 70, 124, 140, 16);
    g.fill();
    g.drawImage(atlas.canvas, f.x, f.y, f.w, f.h, x - 48, y - 48, 96, 96);
  });
  g.font = 'bold 40px sans-serif';
  g.fillStyle = '#fff';
  g.strokeStyle = '#1a1c2c';
  g.lineWidth = 6;
  g.strokeText('YOU WIN 1,000 GEMS!', CARD_W / 2, 330);
  g.fillText('YOU WIN 1,000 GEMS!', CARD_W / 2, 330);

  const foil = makeCanvas(CARD_W, CARD_H);
  const f = foil.getContext('2d');
  // the scratchable area is the panel; the header stays printed (alpha 0 = not scratchable)
  f.fillStyle = 'rgba(0,0,0,0)';
  f.fillRect(0, 0, CARD_W, CARD_H);
  f.save();
  f.beginPath();
  f.roundRect(28, 92, CARD_W - 56, 272, 18);
  f.clip();
  grad = f.createLinearGradient(0, 92, CARD_W, 364);
  grad.addColorStop(0, '#c9d1db');
  grad.addColorStop(0.45, '#f4f6f8');
  grad.addColorStop(0.55, '#aeb7c2');
  grad.addColorStop(1, '#dfe4ea');
  f.fillStyle = grad;
  f.fillRect(0, 0, CARD_W, CARD_H);
  f.strokeStyle = 'rgba(80,90,110,0.18)';
  f.lineWidth = 3;
  for (let x = -CARD_H; x < CARD_W; x += 14) {
    f.beginPath();
    f.moveTo(x, 364);
    f.lineTo(x + 272, 92);
    f.stroke();
  }
  f.font = 'bold 30px sans-serif';
  f.textAlign = 'center';
  f.textBaseline = 'middle';
  f.fillStyle = 'rgba(60,70,90,0.55)';
  f.fillText('SCRATCH HERE', CARD_W / 2, 228);
  f.font = 'bold 14px sans-serif';
  for (let i = 0; i < 6; i++) f.fillText('★ ✦ ★', 90 + i * 92, 120);
  f.restore();
  // printed header area on the foil layer (not scratchable): copy of the prize header
  const card = makeCanvas(CARD_W, CARD_H);
  const cg = card.getContext('2d');
  cg.drawImage(prize, 0, 0);
  return { prize, foil };
}

export default {
  interaction: 'Move the mouse: flashlight / x-ray lens / keyhole. Scratch card: drag to scratch.',
  examples: [
    {
      id: 'spotlight',
      label: 'Flashlight in the dark',
      kind: 'In a game',
      note: 'An <b>alpha (soft) mask</b>: the flashlight cone, the hero’s glow and the torches are drawn additively into an offscreen “light texture”. The final shader mixes a dark, moonlit version of the scene with the lit version using that texture.',
      hint: 'Move the mouse to aim the flashlight',
    },
    {
      id: 'xray',
      label: 'X-ray vision',
      kind: 'In a game',
      note: 'A <b>stencil</b> mask: the lens circle is drawn into the stencil buffer only (no color). Then the x-ray layer is drawn where <code>stencil == 1</code> and the normal world where <code>stencil != 1</code>. Switch <i>Mask type</i> to compare with a soft alpha mask.',
      hint: 'Move the mouse to scan',
    },
    {
      id: 'scratch',
      label: 'Scratch card',
      kind: 'Real life',
      note: 'A <b>persistent mask texture</b>: every drag paints white circles into a texture that is never cleared. The card shader mixes the foil and the prize with it, and darkens the edges of the scratched area for a 3D lip.',
      hint: 'Drag to scratch',
    },
    {
      id: 'windows',
      label: 'Shaped windows',
      kind: 'Abstract',
      note: 'Arbitrary <b>polygons</b> — a keyhole, stars, a heart — are written into the stencil buffer, and a different world is drawn only inside them. This is also how “iris wipe” transitions and Mario-style keyhole endings work.',
      hint: 'Move the keyhole · press “Iris wipe”',
    },
  ],
  controls: [
    { type: 'select', key: 'maskType', label: 'Mask type', value: 'stencil', options: [{ value: 'stencil', label: 'Stencil buffer (hard, binary)' }, { value: 'alpha', label: 'Alpha mask (soft)' }], showFor: ['xray'] },
    { type: 'toggle', key: 'msaa', label: 'MSAA ×4 (smooth stencil edges)', value: true, showFor: ['xray', 'windows'], help: 'A stencil is per sample: without MSAA its edges are jagged.' },
    { type: 'slider', key: 'lens', label: 'Lens / light size', min: 0.05, max: 0.5, step: 0.005, value: 0.18 },
    { type: 'slider', key: 'feather', label: 'Soft edge (alpha masks)', min: 0, max: 80, step: 1, value: 24, showFor: ['spotlight', 'xray'] },
    { type: 'toggle', key: 'torches', label: 'Torches & hero glow', value: true, showFor: ['spotlight'] },
    { type: 'toggle', key: 'showMask', label: 'Show the light mask', value: false, showFor: ['spotlight'] },
    { type: 'slider', key: 'brush', label: 'Scratch brush size', min: 4, max: 60, step: 1, value: 22, showFor: ['scratch'] },
    { type: 'toggle', key: 'animate', label: 'Animate the windows', value: true, showFor: ['windows'] },
    { type: 'button', key: 'iris', label: 'Iris wipe', showFor: ['windows'] },
    { type: 'button', key: 'reset', label: 'New card', showFor: ['scratch'] },
  ],
  about: {
    summary: 'Masks decide where something is drawn. An alpha mask is a grayscale texture you blend with (soft edges, any shape, any amount); a stencil buffer is a per-pixel yes/no tag the GPU tests before drawing (hard edges, free to test, perfect for windows and portals).',
    what: `<p>Four classic uses: a flashlight cutting through darkness, an x-ray lens revealing a hidden layer, a scratch card that remembers where you scratched,
      and windows shaped like keyholes and stars that look into a different world.</p>`,
    how: `<ol>
      <li><b>Alpha mask</b>: draw the “where” into a texture (white = show, black = hide, gray = partial). The final shader does <code>mix(a, b, mask)</code>. Soft edges, gradients and accumulation are free. The flashlight and the scratch card work like this.</li>
      <li><b>Persistent mask</b>: don’t clear the mask texture between frames, keep painting into it (blend mode <code>max</code>) — scratch cards, fog-of-war, paint, footprints.</li>
      <li><b>Stencil buffer</b>: an 8-bit integer per pixel (here <code>depth24plus-stencil8</code>). Pass 1 draws the shapes with color writes off and <code>passOp: 'replace'</code> → stencil = 1 inside.
        Pass 2 draws layer B with <code>compare: 'equal'</code>, layer A with <code>compare: 'not-equal'</code>. The test happens in fixed-function hardware before the fragment shader.</li>
      <li>A stencil is binary per sample, so edges are aliased — enable MSAA (one stencil value per sample) for smooth edges, or use an alpha mask when you need a feathered edge.</li>
    </ol>`,
    uses: [
      { title: 'Horror & stealth', text: 'Flashlights and vision cones in the dark (Darkwood, Teleglitch, Lone Survivor), sonar/x-ray reveals.' },
      { title: 'UI', text: 'Scroll views and minimaps clipped to rounded or circular frames, scratch-card rewards, reveal animations.' },
      { title: 'Transitions', text: 'Iris wipes (Looney Tunes, Mario’s keyhole/star endings), shaped wipes between levels.' },
      { title: 'Portals & windows', text: 'Stencil-masked views into another layer: magic windows, mirrors, x-ray goggles, “dream world” overlays.' },
    ],
    try: [
      'In <b>X-ray vision</b>, switch <i>Mask type</i> to alpha and raise <i>Soft edge</i> — the stencil can never do a gradient.',
      'Turn off <i>MSAA</i> and look closely at the lens edge: stair-steps.',
      'Turn on <i>Show the light mask</i> in the flashlight example: that grayscale texture is all the shader needs.',
      'In <b>Shaped windows</b>, press <i>Iris wipe</i>.',
      'Scratch more than 60% of the card…',
    ],
    ask: [
      'a flashlight cone that reveals a dark level using a light mask texture',
      'x-ray lens using the stencil buffer',
      'a scratch card with a persistent mask texture',
      'star / keyhole shaped windows into another scene (stencil)',
      'iris wipe transition with the stencil buffer',
    ],
    perf: `<p>Stencil tests are nearly free (fixed function, early rejection before the fragment shader). Alpha masks cost one texture read per pixel
      plus whatever renders the mask. MSAA multiplies the color/stencil storage by the sample count.</p>`,
    api: `<p>WebGPU: stencil state lives in the render pipeline (<code>depthStencil.stencilFront/back</code>) and the reference value is set per pass with
      <code>setStencilReference</code>. WebGL2 has the same features (<code>gl.stencilFunc/stencilOp</code>, a <code>DEPTH24_STENCIL8</code> renderbuffer).</p>`,
    code: [
      {
        title: 'Stencil pipelines (WebGPU, from this scene)',
        lang: 'js',
        src: `const write = { format: 'depth24plus-stencil8', depthWriteEnabled: false, depthCompare: 'always',
  stencilFront: { compare: 'always', passOp: 'replace' }, stencilBack: { compare: 'always', passOp: 'replace' } };
const inside  = { ...write, stencilWriteMask: 0,
  stencilFront: { compare: 'equal' }, stencilBack: { compare: 'equal' } };
const outside = { ...inside, stencilFront: { compare: 'not-equal' }, stencilBack: { compare: 'not-equal' } };
// pass: shapes with color writeMask 0 -> stencil = 1, then layer B (inside), then layer A (outside)
pass.setStencilReference(1);`,
      },
      {
        title: 'Alpha mask: one mix in the shader',
        lang: 'wgsl',
        src: `let L = TEX(light, uv).r;                       // the painted mask
let night = s * vec3f(0.07, 0.09, 0.2);          // dark version
c = mix(night, s * vec3f(1.05, 0.95, 0.8), L);   // lit where the mask is white`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const fmt = gpu.format;
    const atlas = await getAtlas();
    const game = createGameScene(gpu);
    const cards = paintCards(atlas);
    const prizeTex = gpu.textureFromImage(cards.prize, { label: 'prize' });
    const foilTex = gpu.textureFromImage(cards.foil, { label: 'foil' });
    const hud = overlayTag(ctx, 'right:8px;bottom:8px');
    const cam = new Camera2D();
    const shapes = new ShapeBatch(gpu);
    const lights = new ShapeBatch(gpu);
    const brush = new ShapeBatch(gpu);

    // layers
    const mkU = () => gpu.uniforms({ res: 'vec2f', time: 'f32', layer: 'f32' }, 'U');
    const UN = mkU();
    const UX = mkU();
    const US = mkU();
    const fxNormal = gpu.fullscreen({ label: 'xray-normal', code: XRAY, uniforms: UN, include: ['sdf', 'noise', 'hash'] });
    const fxXray = gpu.fullscreen({ label: 'xray-hidden', code: XRAY, uniforms: UX, include: ['sdf', 'noise', 'hash'] });
    const fxSpace = gpu.fullscreen({ label: 'space', code: SPACE, uniforms: US, include: ['noise', 'hash', 'math'] });
    const SU2 = gpu.uniforms({ time: 'f32', showMask: 'f32' }, 'Spot');
    const fxSpot = gpu.fullscreen({ label: 'spot', code: SPOT, uniforms: SU2, textures: ['scene', 'light'], include: ['noise'] });
    const LU = gpu.uniforms({ lens: 'vec4f' }, 'Lens');
    const fxLens = gpu.fullscreen({ label: 'lens-alpha', code: LENS_ALPHA, uniforms: LU, textures: ['normal', 'xray'] });
    const CUf = gpu.uniforms({ card: 'vec4f', tilt: 'vec2f', texel: 'vec2f', time: 'f32', reveal: 'f32' }, 'Card');
    const fxCard = gpu.fullscreen({ label: 'card', code: CARD, uniforms: CUf, textures: ['prize', 'foil', 'mask'], include: ['noise', 'sdf', 'color'] });

    // stencil machinery
    const blitProg = gpu.program({ label: 'stencil-blit', bindings: { src: { texture: true }, samp: { sampler: true } }, code: BLIT });
    const SHU = gpu.uniforms({ res: 'vec2f' }, 'ShapeU');
    const shapeProg = gpu.program({ label: 'stencil-shapes', bindings: { su: { uniform: SHU } }, code: SHAPES });
    const base = { format: DS, depthWriteEnabled: false, depthCompare: 'always' };
    const writeState = { ...base, stencilFront: { compare: 'always', passOp: 'replace' }, stencilBack: { compare: 'always', passOp: 'replace' } };
    const eqState = { ...base, stencilWriteMask: 0, stencilFront: { compare: 'equal', passOp: 'keep' }, stencilBack: { compare: 'equal', passOp: 'keep' } };
    const neState = { ...base, stencilWriteMask: 0, stencilFront: { compare: 'not-equal', passOp: 'keep' }, stencilBack: { compare: 'not-equal', passOp: 'keep' } };
    const pipes = {};
    const pipesFor = (samples) => {
      if (pipes[samples]) return pipes[samples];
      const sc = samples > 1 ? samples : undefined;
      return (pipes[samples] = {
        write: shapeProg.renderPipeline({ targets: [{ format: fmt, writeMask: 0 }], depth: writeState, sampleCount: sc, buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }] }),
        inside: blitProg.renderPipeline({ format: fmt, depth: eqState, sampleCount: sc }),
        outside: blitProg.renderPipeline({ format: fmt, depth: neState, sampleCount: sc }),
      });
    };
    let shapeBuf = gpu.vertexBuffer(new Float32Array(6 * 2 * 4096), 'stencil-shapes');
    let size = [0, 0];
    let msaaColor = null;
    let dsTex = { 1: null, 4: null };
    let tA = null;
    let tB = null;
    let lightT = null;
    let maskT = null;
    const ensure = (W, H) => {
      if (size[0] === W && size[1] === H) return;
      size = [W, H];
      for (const t of [msaaColor, dsTex[1], dsTex[4], tA, tB, lightT]) t?.destroy();
      msaaColor = gpu.texture({ size: [W, H], format: fmt, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'mask-msaa' });
      dsTex = {
        1: gpu.texture({ size: [W, H], format: DS, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'stencil' }),
        4: gpu.texture({ size: [W, H], format: DS, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'stencil-msaa' }),
      };
      tA = gpu.target(W, H, { label: 'layer-a' });
      tB = gpu.target(W, H, { label: 'layer-b' });
      lightT = gpu.target(W, H, { label: 'light-mask' });
    };
    const newCard = () => {
      maskT?.destroy();
      maskT = gpu.target(CARD_W, CARD_H, { label: 'scratch-mask' });
      scratch.grid.fill(0);
      scratch.cleared = true;
      scratch.revealT = 0;
      scratch.won = false;
      scratch.confetti = [];
    };
    const scratch = { grid: new Uint8Array(64 * 40), cleared: true, revealT: 0, won: false, confetti: [], last: null, auto: 0 };
    let iris = -1;
    let idle = 99;
    newCard();

    // polygons drawn into the stencil (and outlined afterwards)
    let outlines = [];
    const buildShapes = (ex, W, H, t, mx, my, p) => {
      const out = [];
      outlines = [];
      const s = Math.min(W, H);
      const poly = (cx, cy, pts, rim = true) => {
        fan(out, cx, cy, pts);
        if (rim) outlines.push(pts);
      };
      if (ex === 'xray') {
        poly(mx, my, circlePts(mx, my, p.lens * s * 1.2, 96), false);
      } else {
        const anim = p.animate ? t : 0;
        // keyhole following the mouse: one polygon (arc + trapezoid), star-shaped around a point inside
        const kr = p.lens * s * 0.9;
        const ccy = my - kr * 0.25;
        const rr = kr * 0.55;
        const a0 = Math.atan2(0.49, 0.25);
        const key = [];
        for (let i = 0; i <= 48; i++) {
          const a = a0 - (i / 48) * (Math.PI * 2 - 2 * (Math.PI / 2 - a0));
          key.push([mx + Math.cos(a) * rr, ccy + Math.sin(a) * rr]);
        }
        key.push([mx - kr * 0.45, my + kr * 0.95], [mx + kr * 0.45, my + kr * 0.95]);
        poly(mx, my + kr * 0.1, key);
        // a spinning star, a beating heart, a drifting circle
        const sx = W * 0.2;
        const sy = H * 0.32;
        poly(sx, sy, starPts(sx, sy, s * 0.17 * (1 + 0.08 * Math.sin(anim * 2)), anim * 0.6));
        const hx = W * 0.8;
        const hy = H * 0.62;
        poly(hx, hy + s * 0.02, heartPts(hx, hy, s * 0.13 * (1 + 0.12 * Math.max(0, Math.sin(anim * 5)) ** 4)));
        const cx = W * (0.5 + 0.3 * Math.sin(anim * 0.4));
        const cy = H * 0.15 + s * 0.06;
        poly(cx, cy, circlePts(cx, cy, s * 0.07, 48));
        if (iris >= 0) {
          const k = iris < 1 ? iris : 2 - iris;
          const ir = Math.hypot(W, H) * 0.55 * (k * k * (3 - 2 * k));
          poly(W / 2, H / 2, circlePts(W / 2, H / 2, Math.max(1, ir), 96));
        }
      }
      return new Float32Array(out);
    };

    return {
      onAction(key) {
        if (key === 'iris') iris = 0;
        if (key === 'reset') newCard();
      },
      frame(ctx) {
        const p = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const ex = ctx.example;
        const t = ctx.time;
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 30);
        const enc = ctx.encoder;
        const ptr = ctx.pointer;
        ensure(W, H);
        cam.setViewport(W, H);
        shapes.begin();
        lights.begin();
        brush.begin();
        const canvas = { view: ctx.target, format: fmt };
        idle = ptr.over && (ptr.dx || ptr.dy || ptr.down) ? 0 : idle + dt;
        // the lens follows the mouse, or wanders when idle
        const autoX = W * (0.5 + 0.32 * Math.sin(t * 0.5));
        const autoY = H * (0.55 + 0.25 * Math.sin(t * 0.83 + 1));
        const useAuto = idle > 2;
        const mx = useAuto ? autoX : ptr.x;
        const my = useAuto ? autoY : ptr.y;
        if (iris >= 0) {
          iris += dt * 0.7;
          if (iris > 2) iris = -1;
        }
        const s = Math.min(W, H);

        if (ex === 'spotlight') {
          const gs = ctx.testMode ? 0.5 : 1; // the scene is sampled by uv, any resolution works
          const g = game.render(enc, t, Math.round(W * gs), Math.round(H * gs));
          // light mask: additive soft shapes
          const hx = 0.45 * H;
          const hop = Math.abs(Math.sin(t * 2.6));
          const hy = H * (0.8 - 0.042 - hop * 0.13);
          const ang = Math.atan2(my - hy, mx - hx);
          const len = Math.max(W, H) * 1.2;
          const spread = 0.25 + p.lens;
          const feather = p.feather * (H / 700) + 0.5;
          for (let k = 0; k < 4; k++) {
            const sp = spread * (1 - k * 0.18);
            const a = 0.32;
            lights.triangle(hx, hy, hx + Math.cos(ang - sp) * len, hy + Math.sin(ang - sp) * len, hx + Math.cos(ang + sp) * len, hy + Math.sin(ang + sp) * len, [1, 1, 1, a], { glow: feather, glowStrength: 0.9 });
          }
          lights.circle(mx, my, s * p.lens * 0.6, [1, 1, 1, 0.35], { glow: feather * 2 + 10, glowStrength: 0.8 });
          if (p.torches) {
            lights.circle(hx, hy, s * 0.06, [1, 1, 1, 0.5], { glow: s * 0.06, glowStrength: 0.7 });
            const scroll = t * 0.12;
            for (let k = -1; k < 6; k++) {
              const gx = 0.15 + k * 0.9 + Math.ceil(scroll / 0.9) * 0.9;
              const x = (gx - scroll) * H;
              if (x < -100 || x > W + 100) continue;
              const fl = 0.85 + 0.15 * Math.sin(t * 17 + k * 3) * Math.sin(t * 23);
              lights.circle(x, H * 0.69, s * 0.05 * fl, [1, 1, 1, 0.6], { glow: s * 0.07 * fl, glowStrength: 0.8 });
            }
          }
          lights.flush(enc, lightT, cam, { clear: [0, 0, 0, 1], blend: 'additive' });
          SU2.set('time', t).set('showMask', p.showMask ? 1 : 0);
          fxSpot.draw(enc, canvas, { scene: g, light: lightT });
          hud.textContent = 'alpha mask: 1 offscreen light texture';
        } else if (ex === 'scratch') {
          // card placement
          const cw = Math.min(W * 0.82, (H * 0.82 * CARD_W) / CARD_H);
          const chh = (cw * CARD_H) / CARD_W;
          const cx0 = (W - cw) / 2;
          const cy0 = (H - chh) / 2;
          const toCard = (x, y) => [((x - cx0) / cw) * CARD_W, ((y - cy0) / chh) * CARD_H];
          const strokes = [];
          if (ptr.down) {
            strokes.push(toCard(ptr.x, ptr.y));
            if (scratch.last) strokes.push(scratch.last);
            scratch.last = toCard(ptr.x, ptr.y);
            idle = 0;
          } else scratch.last = null;
          if (idle > 2 && !scratch.won && dt > 0) {
            // an invisible hand scratches in zig-zags
            scratch.auto += dt;
            const a = scratch.auto;
            const x = CARD_W * (0.12 + 0.76 * (0.5 + 0.5 * Math.sin(a * 2.3)));
            const y = CARD_H * (0.3 + 0.5 * ((a * 0.05) % 1));
            strokes.push([x, y]);
            if (scratch.autoLast) strokes.push(scratch.autoLast);
            scratch.autoLast = [x, y];
          }
          // paint the persistent mask: many small rough dabs along the stroke
          const bcam = new Camera2D();
          bcam.setViewport(CARD_W, CARD_H);
          const br = p.brush * 0.8;
          if (strokes.length >= 1) {
            const [a, b] = strokes.length > 1 ? strokes : [strokes[0], strokes[0]];
            const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / (br * 0.3)));
            for (let k = 0; k <= n; k++) {
              const x = a[0] + ((b[0] - a[0]) * k) / n;
              const y = a[1] + ((b[1] - a[1]) * k) / n;
              for (let j = 0; j < 3; j++) brush.circle(x + (Math.random() - 0.5) * br * 0.6, y + (Math.random() - 0.5) * br * 0.6, br * (0.5 + Math.random() * 0.3), [1, 1, 1, 1]);
              // coverage bookkeeping on a coarse CPU grid (scratch area only)
              const gx = Math.floor((x / CARD_W) * 64);
              const gy = Math.floor((y / CARD_H) * 40);
              const rr = Math.ceil((br / CARD_W) * 64);
              for (let yy = gy - rr; yy <= gy + rr; yy++) for (let xx = gx - rr; xx <= gx + rr; xx++) if (xx >= 0 && yy >= 0 && xx < 64 && yy < 40) scratch.grid[yy * 64 + xx] = 1;
            }
          }
          brush.flush(enc, maskT, bcam, scratch.cleared ? { clear: [0, 0, 0, 1], blend: 'max' } : { blend: 'max' });
          scratch.cleared = false;
          // revealed fraction of the scratch panel (rows 9..36 of the grid ≈ the foil panel)
          let on = 0;
          let tot = 0;
          for (let yy = 10; yy < 36; yy++) for (let xx = 3; xx < 61; xx++) {
            tot++;
            on += scratch.grid[yy * 64 + xx];
          }
          const frac = on / tot;
          if (frac > 0.6 && !scratch.won) {
            scratch.won = true;
            for (let k = 0; k < 160; k++) scratch.confetti.push({ x: W / 2, y: H * 0.45, vx: (Math.random() - 0.5) * 900, vy: -300 - Math.random() * 600, c: [Math.random(), 0.5 + Math.random() * 0.5, Math.random(), 1], r: 3 + Math.random() * 4, rot: Math.random() * 6 });
          }
          if (scratch.won) scratch.revealT = Math.min(1, scratch.revealT + dt * 1.5);
          CUf.set('card', [cx0, cy0, cw, chh])
            .set('tilt', [ptr.nx - 0.5, ptr.ny - 0.5])
            .set('texel', [1 / CARD_W, 1 / CARD_H])
            .set('time', t)
            .set('reveal', scratch.revealT);
          fxCard.draw(enc, canvas, { prize: prizeTex, foil: foilTex, mask: maskT });
          for (const c of scratch.confetti) {
            c.vy += 900 * dt;
            c.x += c.vx * dt;
            c.y += c.vy * dt;
            c.rot += dt * 8;
            shapes.rect(c.x - c.r, c.y - c.r * 0.5, c.r * 2, c.r, c.c, { rotation: c.rot });
          }
          scratch.confetti = scratch.confetti.filter((c) => c.y < H + 20);
          if (ptr.over && !useAuto) shapes.circle(ptr.x, ptr.y, (br / CARD_W) * cw, [1, 1, 1, 0.5], { stroke: 1.5 });
          shapes.flush(enc, canvas, cam);
          hud.textContent = scratch.won ? 'WINNER! — press “New card”' : `scratched ${(frac * 100).toFixed(0)}% (60% reveals the prize)`;
        } else {
          // ---- stencil examples: render both layers, then composite with the stencil test
          let layerA;
          let layerB;
          if (ex === 'xray') {
            UN.set('res', [W, H]).set('time', t).set('layer', 0);
            UX.set('res', [W, H]).set('time', t).set('layer', 1);
            fxNormal.draw(enc, tA);
            fxXray.draw(enc, tB);
            layerA = tA;
            layerB = tB;
          } else {
            const gs = ctx.testMode ? 0.5 : 1;
            layerA = game.render(enc, t, Math.round(W * gs), Math.round(H * gs));
            US.set('res', [W, H]).set('time', t).set('layer', 0);
            fxSpace.draw(enc, tB);
            layerB = tB;
          }
          const lensR = p.lens * s * 1.2;
          if (ex === 'xray' && p.maskType === 'alpha') {
            LU.set('lens', [mx, my, lensR, p.feather * (H / 700) + 0.5]);
            fxLens.draw(enc, canvas, { normal: layerA, xray: layerB });
            hud.textContent = 'alpha mask: mix(normal, x-ray, smoothstep(edge))';
          } else {
            const verts = buildShapes(ex, W, H, t, mx, my, p);
            if (verts.byteLength > shapeBuf.size) shapeBuf = gpu.vertexBuffer(new Float32Array(verts.length * 2), 'stencil-shapes');
            gpu.queue.writeBuffer(shapeBuf, 0, verts);
            SHU.set('res', [W, H]);
            SHU.upload();
            const samples = p.msaa ? 4 : 1;
            const pp = pipesFor(samples);
            const color = samples > 1 ? { view: msaaColor.createView(), resolveTarget: ctx.target } : { view: ctx.target };
            const pass = enc.beginRenderPass({
              colorAttachments: [{ ...color, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: samples > 1 ? 'discard' : 'store' }],
              depthStencilAttachment: {
                view: dsTex[samples].createView(),
                depthClearValue: 1,
                depthLoadOp: 'clear',
                depthStoreOp: 'discard',
                stencilClearValue: 0,
                stencilLoadOp: 'clear',
                stencilStoreOp: 'discard',
              },
            });
            pass.setStencilReference(1);
            // 1) shapes -> stencil only
            pass.setPipeline(pp.write);
            pass.setBindGroup(0, shapeProg.bind({ su: SHU }));
            pass.setVertexBuffer(0, shapeBuf);
            pass.draw(verts.length / 2);
            // 2) the other world where stencil == 1, 3) the normal world where stencil != 1
            pass.setPipeline(pp.inside);
            pass.setBindGroup(0, blitProg.bind({ src: layerB, samp: 'linear' }));
            pass.draw(3);
            pass.setPipeline(pp.outside);
            pass.setBindGroup(0, blitProg.bind({ src: layerA, samp: 'linear' }));
            pass.draw(3);
            pass.end();
            hud.textContent = `stencil: ${verts.length / 6} triangles written · ${samples}× sampling`;
          }
          // rims drawn on top
          if (ex === 'windows' && !(ex === 'xray' && p.maskType === 'alpha')) {
            for (const o of outlines) {
              shapes.polyline(o, 5, [0.08, 0.06, 0.12, 0.85], { closed: true });
              shapes.polyline(o, 2, [1, 0.85, 0.55, 1], { closed: true });
            }
          }
          if (ex === 'xray') {
            shapes.circle(mx, my, lensR + 3, [0.1, 0.12, 0.16, 1], { stroke: 6 });
            shapes.circle(mx, my, lensR + 3, [0.4, 0.9, 1, 0.9], { stroke: 1.5, glow: 10, glowStrength: 0.5 });
            const hx = mx + Math.cos(0.8) * (lensR + 6);
            const hy = my + Math.sin(0.8) * (lensR + 6);
            shapes.line(hx, hy, hx + lensR * 0.5, hy + lensR * 0.5, 10, [0.1, 0.12, 0.16, 1]);
          }
          shapes.flush(enc, canvas, cam);
        }
      },
    };
  },
};

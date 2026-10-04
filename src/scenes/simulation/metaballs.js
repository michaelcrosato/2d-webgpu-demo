import { shaderScene } from '../../core/shaderscene.js';

// Metaballs: sum a "field" from every blob, call everything above a threshold "inside".
// The blob motion (bouncing balls, lava wax, a jiggly slime, gooey UI) is simulated in JavaScript
// and handed to the shader as a uniform array; the shader evaluates the field for every pixel.
// Portable WGSL → runs on WebGPU and WebGL2.

const MAXB = 24;

// ------------------------------------------------------------------------------------------
// WGSL shared by every example: field evaluation, color blending, fake-3D shading, field view
// q = aspect-correct coords: (0,0) = screen center, y in [-0.5, 0.5] (down), x in ±aspect/2
// ------------------------------------------------------------------------------------------
const FIELD = /* wgsl */ `
const MAXB: i32 = ${MAXB};
fn aaq(d: f32) -> f32 { return clamp(0.5 - d * u.resolution.y, 0.0, 1.0); }
fn ballCol(w: f32) -> vec3f {
  if (w < 0.0) { return u.tint; }
  return hsv2rgb(vec3f(fract(w), 0.6, 1.0));
}
// smooth-min union of circles (polynomial smin, k = blend radius)
fn sdUnion(q: vec2f) -> f32 {
  var d = 100000.0;
  for (var i = 0; i < MAXB; i++) {
    if (f32(i) >= u.nBalls) { break; }
    let b = u.balls[i];
    d = opSmoothUnion(d, length(q - b.xy) - b.z * u.blobSize, max(u.smoothK, 0.0005));
  }
  return d;
}
// returns (signed distance estimate [<0 inside], outward unit normal xy, field value)
fn evalField(q: vec2f) -> vec4f {
  let mode = i32(u.fieldFn + 0.5);
  if (mode == 2) {
    let e = 0.0015;
    let d = sdUnion(q) + (1.0 - u.threshold) * 0.04;
    let g = vec2f(sdUnion(q + vec2f(e, 0.0)) - sdUnion(q - vec2f(e, 0.0)), sdUnion(q + vec2f(0.0, e)) - sdUnion(q - vec2f(0.0, e)));
    return vec4f(d, normalize(g + vec2f(0.0000001, 0.0)), -d);
  }
  var f = 0.0;
  var g = vec2f(0.0);
  for (var i = 0; i < MAXB; i++) {
    if (f32(i) >= u.nBalls) { break; }
    let b = u.balls[i];
    let r = b.z * u.blobSize;
    let dv = q - b.xy;
    let d2 = dot(dv, dv) + 0.0000001;
    if (mode == 0) {
      let c = r * r / d2;                      // classic inverse-square falloff (Blinn)
      f += c;
      g += -2.0 * c / d2 * dv;
    } else {
      let R2 = 4.0 * r * r;                    // Wyvill: (1 - d²/R²)³, zero beyond R = 2r
      let x = d2 / R2;
      if (x < 1.0) {
        let k = 1.0 - x;
        f += k * k * k * 2.370370;             // normalised so a lone ball's surface is at d = r
        g += -6.0 * k * k * 2.370370 / R2 * dv;
      }
    }
  }
  // Distance estimate. (T - F)/|grad F| is only accurate right at the surface, so first remap the field
  // with G(F) that is exactly linear in distance for a lone blob (G = d/r), then divide by |grad G|.
  let gl = max(length(g), 0.0000001);
  var G = 0.0;
  var GT = 0.0;
  var dG = 0.0;
  if (mode == 0) {
    G = inverseSqrt(max(f, 0.000001));
    GT = inverseSqrt(max(u.threshold, 0.000001));
    dG = 0.5 * pow(max(f, 0.000001), -1.5);
  } else {
    let c = pow(max(f, 0.000001) / 2.370370, 1.0 / 3.0);
    let ct = pow(max(u.threshold, 0.000001) / 2.370370, 1.0 / 3.0);
    G = sqrt(max(1.0 - c, 0.0001));
    GT = sqrt(max(1.0 - ct, 0.0001));
    dG = c / (6.0 * max(f, 0.000001) * G);
  }
  return vec4f((G - GT) / max(dG * gl, 0.0001), -g, f);   // yz = outward (unnormalised) field gradient
}
// each ball's color, weighted by its share of the field
fn evalColor(q: vec2f) -> vec3f {
  var cs = vec3f(0.0);
  var ws = 0.0;
  let pw = mix(10.0, 2.0, u.colorBlend);
  for (var i = 0; i < MAXB; i++) {
    if (f32(i) >= u.nBalls) { break; }
    let b = u.balls[i];
    let r = b.z * u.blobSize;
    let dv = q - b.xy;
    let c = min(r * r / (dot(dv, dv) + 0.000001), 60.0);
    let w = pow(c, pw);
    cs += ballCol(b.w) * w;
    ws += w;
  }
  return cs / max(ws, 0.000001);
}
// Fake 3D: give the blob a height and light it.
//  - field modes: height H = 1 - exp(-(F - T)) rises smoothly from the edge; its gradient is smooth everywhere
//  - smooth-min SDF: a dome whose height comes from the distance to the edge
fn blobNormal(d: f32, g: vec2f, f: f32, R0: f32) -> vec3f {
  var n = vec3f(0.0, 0.0, 1.0);
  if (i32(u.fieldFn + 0.5) == 2) {
    let s = clamp(R0 + d, 0.0, R0);
    let z = sqrt(max(R0 * R0 - s * s, 0.0));
    n = normalize(vec3f(normalize(g + vec2f(0.0000001, 0.0)) * s, z + 0.00001));
  } else {
    let T = max(u.threshold, 0.001);
    let tilt = g * exp(-max(f - T, 0.0) * 1.5 / T) * 1.5 / T * R0;
    n = normalize(vec3f(tilt, 1.0));
  }
  return normalize(mix(vec3f(0.0, 0.0, 1.0), n, u.shading));
}
fn shadeBlob(n: vec3f, base: vec3f) -> vec3f {
  let L = normalize(vec3f(-0.5, -0.65, 0.6));
  let H = normalize(L + vec3f(0.0, 0.0, 1.0));
  let diff = max(dot(n, L), 0.0);
  let spec = pow(max(dot(n, H), 0.0), u.gloss) * u.shading;
  let rim = pow(1.0 - n.z, 2.0) * u.rim;
  return base * (0.42 + 0.7 * diff) + vec3f(1.0) * spec * 0.85 + mix(base, vec3f(1.0), 0.4) * rim * 0.6;
}
// Debug view: heat map of the field, iso-lines, and the threshold contour
fn fieldViz(f: f32, d: f32) -> vec3f {
  let mode = i32(u.fieldFn + 0.5);
  var v = f / max(u.threshold, 0.001);
  if (mode == 2) { v = 1.0 - d * 10.0; }
  let h = clamp(v * 0.45, 0.0, 1.0);
  var c = mix(vec3f(0.02, 0.015, 0.06), vec3f(0.32, 0.05, 0.42), smoothstep(0.0, 0.3, h));
  c = mix(c, vec3f(0.92, 0.3, 0.16), smoothstep(0.28, 0.6, h));
  c = mix(c, vec3f(1.0, 0.92, 0.45), smoothstep(0.58, 1.0, h));
  let vv = v * 4.0;
  let iso = (1.0 - smoothstep(0.0, 1.2, abs(fract(vv + 0.5) - 0.5) / max(fwidth(vv), 0.0001))) * (1.0 - smoothstep(2.0, 3.5, v));
  c = mix(c, c * 0.45 + vec3f(0.12), iso * 0.6);
  return c;
}
`;

// ------------------------------------------------------------------------------------------
// Per-example image shaders
// ------------------------------------------------------------------------------------------
const HEAD = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let q = centerUV(uv * res, res);
  let aspect = res.x / res.y;
  let t = u.time;
  let F = evalField(q);
  let d = F.x;
  let g = F.yz;
  let f = F.w;
`;

const FIELD_IMG = /* wgsl */ `
${FIELD}
${HEAD}
  var col = mix(vec3f(0.05, 0.05, 0.08), vec3f(0.08, 0.07, 0.12), uv.y);
  let gg = abs(fract(q * 20.0) - 0.5);
  col += vec3f(0.02) * (1.0 - smoothstep(0.0, 0.04, min(gg.x, gg.y)));
  if (u.showField > 0.5) { col = fieldViz(f, d); }
  let a = aaq(d);
  let base = evalColor(q);
  let lit = shadeBlob(blobNormal(d, g, f, 0.05), base);
  col = mix(col, lit, a * mix(1.0, 0.55, u.showField));
  // the threshold iso-line = the visible surface
  col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.6, 1.6, abs(d) * res.y)) * 0.9 * u.showField);
  // centers and each ball's own radius (where it alone would reach the threshold)
  if (u.showField > 0.5) {
    for (var i = 0; i < MAXB; i++) {
      if (f32(i) >= u.nBalls) { break; }
      let b = u.balls[i];
      let dd = length(q - b.xy);
      let ring = abs(dd - b.z * u.blobSize);
      let dash = step(0.0, sin(atan2(q.y - b.y, q.x - b.x) * 24.0));
      col = mix(col, ballCol(b.w), (1.0 - smoothstep(0.0, 1.5 / res.y, ring)) * dash * 0.8);
      col = mix(col, vec3f(1.0), 1.0 - smoothstep(2.0 / res.y, 3.5 / res.y, dd));
    }
  }
  col *= 1.0 - 0.3 * dot(uv - 0.5, uv - 0.5);
  return vec4f(col, 1.0);
}`;

const LAVA_IMG = /* wgsl */ `
${FIELD}
// IQ's trapezoid: half-width r1 at the TOP (y = -he), r2 at the bottom (y = +he); y down.
fn sdTrap(p0: vec2f, r1: f32, r2: f32, he: f32) -> f32 {
  let k1 = vec2f(r2, he);
  let k2 = vec2f(r2 - r1, 2.0 * he);
  let p = vec2f(abs(p0.x), p0.y);
  let ca = vec2f(p.x - min(p.x, select(r2, r1, p.y < 0.0)), abs(p.y) - he);
  let cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
fn glassSD(q: vec2f) -> f32 {
  let upper = sdTrap(q - vec2f(0.0, -0.13), 0.068, 0.145, 0.23);
  let lower = sdTrap(q - vec2f(0.0, 0.16), 0.145, 0.115, 0.06);
  return opSmoothUnion(upper, lower, 0.03);
}
fn metal(q: vec2f, w: f32, glow: vec3f) -> vec3f {
  let tt = clamp(q.x / max(w, 0.001), -1.0, 1.0);
  var c = mix(vec3f(0.16, 0.13, 0.18), vec3f(0.92, 0.86, 0.9), exp(-pow((tt + 0.35) * 2.6, 2.0)));
  c += vec3f(0.5, 0.45, 0.55) * exp(-pow((tt - 0.55) * 5.0, 2.0)) * 0.4;
  c += glow * 0.35 * smoothstep(0.3, 1.0, abs(tt));
  return c * vec3f(1.0, 0.92, 0.82);
}
${HEAD}
  let wax = u.tint;
  let liq = u.liquid;
  // room + glow of the lamp on the wall
  let lampGlow = mix(liq, wax, 0.5);
  var col = mix(vec3f(0.035, 0.025, 0.06), vec3f(0.06, 0.03, 0.07), uv.y);
  col += lampGlow * exp(-length((q - vec2f(0.0, 0.0)) * vec2f(1.6, 0.9)) * 3.2) * 0.55;
  // table
  let table = smoothstep(0.425, 0.43, q.y);
  col = mix(col, mix(vec3f(0.12, 0.07, 0.06), vec3f(0.05, 0.03, 0.03), smoothstep(0.43, 0.5, q.y)) + lampGlow * 0.12 * exp(-abs(q.x) * 6.0), table);
  // cap and base (metal)
  let cap = sdTrap(q - vec2f(0.0, -0.405), 0.03, 0.068, 0.045);
  let base = sdTrap(q - vec2f(0.0, 0.33), 0.115, 0.2, 0.11);
  col = mix(col, metal(q, mix(0.03, 0.068, (q.y + 0.45) / 0.09), lampGlow), aaq(cap));
  col = mix(col, metal(q, mix(0.115, 0.2, (q.y - 0.22) / 0.22), lampGlow), aaq(base));
  // inside the glass: glowing liquid, lit from the bulb below
  let gd = glassSD(q);
  if (gd < 0.01) {
    let hgt = smoothstep(0.24, -0.38, q.y);
    var inside = liq * mix(1.25, 0.45, hgt) + vec3f(1.0, 0.8, 0.5) * exp(-length(q - vec2f(0.0, 0.25)) * 7.0) * 0.35;
    // wax: translucent, glowing more where it is hot (near the bottom)
    let a = aaq(d);
    if (a > 0.0) {
      let n = blobNormal(d, g, f, 0.05);
      let thick = clamp(-d / 0.05, 0.0, 1.0);
      var wc = shadeBlob(n, wax * mix(0.7, 1.15, thick));
      wc += wax * vec3f(1.0, 0.75, 0.5) * (1.0 - hgt) * 0.45;           // heat glow
      inside = mix(inside, wc, a);
    }
    // glass: darker edges (fresnel), a vertical reflection streak
    let edgeK = smoothstep(-0.04, 0.0, gd);
    inside = mix(inside, inside * 0.55 + liq * 0.1, edgeK * 0.6);
    let streak = exp(-pow((q.x / max(0.068 + (q.y + 0.36) * 0.17, 0.01) + 0.55) * 7.0, 2.0)) * smoothstep(0.2, -0.3, q.y);
    inside += vec3f(1.0) * streak * 0.22;
    col = mix(col, inside, aaq(gd));
    col = mix(col, vec3f(0.85, 0.8, 0.9), (1.0 - smoothstep(0.0, 1.5 / res.y, abs(gd))) * 0.25);
  }
  if (u.showField > 0.5) { col = mix(col, fieldViz(f, d), 0.75); }
  col *= 1.0 - 0.35 * dot(uv - 0.5, uv - 0.5);
  return vec4f(col, 1.0);
}`;

const SLIME_IMG = /* wgsl */ `
${FIELD}
const GROUND: f32 = 0.3;
${HEAD}
  // dungeon wall: offset stone bricks + a warm torch glow
  var bq = vec2f(q.x, q.y) * vec2f(7.0, 12.0);
  bq.x += 0.5 * step(1.0, fmod(floor(bq.y), 2.0));
  let bid = floor(bq);
  let bf = fract(bq);
  let mortar = smoothstep(0.0, 0.07, min(min(bf.x, 1.0 - bf.x) * 0.6, min(bf.y, 1.0 - bf.y)));
  var col = mix(vec3f(0.1, 0.1, 0.14), vec3f(0.17, 0.16, 0.21), hash21(bid)) * (0.75 + 0.25 * valueNoise(q * 30.0));
  col = mix(vec3f(0.04, 0.04, 0.06), col, mortar);
  col += vec3f(1.0, 0.55, 0.2) * exp(-length(q - vec2f(-aspect * 0.32, -0.22)) * 4.5) * 0.4;
  col += vec3f(1.0, 0.55, 0.2) * exp(-length(q - vec2f(aspect * 0.32, -0.22)) * 4.5) * 0.4;
  // torches
  for (var k = 0; k < 2; k++) {
    let tp = vec2f(aspect * 0.32 * (f32(k) * 2.0 - 1.0), -0.22);
    let fl = 0.85 + 0.15 * sin(t * 13.0 + f32(k) * 3.0) * sin(t * 7.0);
    col = mix(col, vec3f(0.25, 0.16, 0.1), aaq(max(abs(q.x - tp.x) - 0.008, abs(q.y - tp.y - 0.05) - 0.04)));
    col = mix(col, vec3f(1.0, 0.8, 0.35), aaq(length((q - tp) * vec2f(1.0, 0.6)) - 0.016 * fl));
  }
  // floor
  if (q.y > GROUND) {
    var fq = vec2f(q.x * 5.0, (q.y - GROUND) * 9.0);
    fq.x += 0.5 * step(1.0, fmod(floor(fq.y), 2.0));
    let ff = fract(fq);
    var fc = mix(vec3f(0.26, 0.24, 0.28), vec3f(0.34, 0.31, 0.35), hash21(floor(fq)));
    fc *= 0.55 + 0.45 * smoothstep(0.0, 0.08, min(min(ff.x, 1.0 - ff.x) * 0.5, min(ff.y, 1.0 - ff.y)));
    fc *= mix(1.0, 0.45, smoothstep(GROUND, 0.5, q.y));
    fc = mix(fc, vec3f(0.42, 0.62, 0.3), (1.0 - smoothstep(0.0, 0.012, q.y - GROUND)) * 0.7);
    col = fc;
  }
  // slime shadow on the floor (wider when squashed)
  let sp = u.extra[2];
  let sh = length((q - vec2f(sp.x, GROUND + 0.006)) / vec2f(sp.z, 0.018));
  col *= 1.0 - 0.55 * (1.0 - smoothstep(0.6, 1.0, sh)) * sp.w;
  // the slime: glossy, slightly translucent goo
  let a = aaq(d);
  if (a > 0.0) {
    let n = blobNormal(d, g, f, 0.06);
    let thick = clamp(-d / 0.08, 0.0, 1.0);
    var sc = shadeBlob(n, mix(u.tint * 1.25, u.tint * 0.55, thick));
    sc = mix(col * u.tint * 1.6, sc, 0.55 + 0.45 * thick);          // see-through near the edges
    sc += vec3f(0.85, 1.0, 0.8) * exp(-length(q - u.extra[0].xy - vec2f(-0.01, 0.06)) * 22.0) * 0.12; // inner glow
    // eyes (extra[0] = left eye, right eye xy; extra[1] = pupil offset xy, blink, mouth)
    let e = u.extra[0];
    let lk = u.extra[1];
    for (var k = 0; k < 2; k++) {
      var ep = e.xy;
      if (k == 1) { ep = e.zw; }
      let eq = (q - ep) * vec2f(1.0, 1.0 / max(lk.z, 0.08));
      let ed = length(eq) - 0.022;
      sc = mix(sc, vec3f(0.02, 0.06, 0.03), aaq(ed - 0.004) * 0.5);
      sc = mix(sc, vec3f(0.98), aaq(ed));
      let pd = length(eq - lk.xy) - 0.011;
      sc = mix(sc, vec3f(0.03, 0.04, 0.06), aaq(pd));
      sc = mix(sc, vec3f(1.0), aaq(length(eq - lk.xy - vec2f(-0.004, -0.005)) - 0.0035));
    }
    // mouth: a little arc
    let mq = q - (e.xy + e.zw) * 0.5 - vec2f(0.0, 0.018);
    let mr = 0.014 + lk.w * 0.006;
    let smile = max(abs(length(mq) - mr) - 0.0028, -mq.y);
    let open = max(length(mq) - mr, -mq.y) + (1.0 - lk.w) * 0.02;   // mouth opens when it jumps
    sc = mix(sc, vec3f(0.03, 0.08, 0.04), aaq(min(smile, open)) * 0.9);
    col = mix(col, sc, a * 0.97);
  }
  if (u.showField > 0.5) { col = mix(col, fieldViz(f, d), 0.7); }
  col *= 1.0 - 0.3 * dot(uv - 0.5, uv - 0.5);
  return vec4f(col, 1.0);
}`;

const UI_IMG = /* wgsl */ `
${FIELD}
fn sdBoxF(p: vec2f, b: vec2f) -> f32 { let d = abs(p) - b; return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0); }
// simple SDF icons, p in units of the button radius
fn icon(p: vec2f, id: i32, t: f32) -> f32 {
  if (id == 0) {   // home
    let roof = sdTriangle(p, vec2f(-0.55, -0.05), vec2f(0.55, -0.05), vec2f(0.0, -0.55));
    let body = sdBoxF(p - vec2f(0.0, 0.2), vec2f(0.36, 0.26));
    let door = sdBoxF(p - vec2f(0.0, 0.3), vec2f(0.1, 0.17));
    return max(min(roof, body), -door);
  }
  if (id == 1) {   // search
    let ring = abs(length(p - vec2f(-0.1, -0.1)) - 0.3) - 0.08;
    return min(ring, sdSegment(p, vec2f(0.12, 0.12), vec2f(0.45, 0.45)) - 0.09);
  }
  if (id == 2) { return sdHeart(p * 1.6 + vec2f(0.0, -0.45)) / 1.6; }  // heart
  if (id == 3) { return sdStar5(p, 0.52, 0.45); }                       // star
  if (id == 4) {   // gear
    let a = atan2(p.y, p.x) + t * 0.5;
    let teeth = 0.42 + 0.1 * smoothstep(-0.3, 0.3, sin(a * 8.0));
    return max(length(p) - teeth, -(length(p) - 0.16));
  }
  if (id == 5) {   // plus (rotates into an x when the menu is open)
    let r = rot2(u.extra[0].x * 0.785398) * p;
    return min(sdBoxF(r, vec2f(0.45, 0.09)), sdBoxF(r, vec2f(0.09, 0.45)));
  }
  if (id == 6) { return sdTriangle(p, vec2f(-0.25, -0.38), vec2f(-0.25, 0.38), vec2f(0.4, 0.0)); }  // play
  if (id == 7) { return abs(length(p) - 0.3) - 0.1; }                                               // ring
  if (id == 8) { return sdBoxF(p, vec2f(0.3, 0.3)) - 0.04; }                                        // square
  return sdEquilateralTriangle(p * vec2f(1.0, -1.0), 0.38);                                         // triangle
}
${HEAD}
  // app background + a frosted dock panel
  var col = mix(vec3f(0.1, 0.08, 0.2), vec3f(0.2, 0.08, 0.26), uv.x * 0.6 + uv.y * 0.4);
  col += vec3f(0.3, 0.2, 0.6) * exp(-length(q - vec2f(-aspect * 0.3, -0.4)) * 2.5) * 0.25;
  let panel = sdBoxF(q - u.extra[1].xy, u.extra[1].zw) - 0.06;
  col = mix(col, vec3f(0.0), (1.0 - smoothstep(0.0, 0.05, panel)) * 0.25);
  col = mix(col, col * 1.25 + vec3f(0.04, 0.03, 0.08), aaq(panel));
  col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.0, 1.5 / res.y, abs(panel))) * 0.12);
  // goo
  let a = aaq(d);
  if (a > 0.0) {
    let n = blobNormal(d, g, f, 0.035);
    var gc = shadeBlob(n, evalColor(q));
    // icons on top of their buttons
    for (var i = 0; i < 13; i++) {
      if (i == 5 || i == 6 || i == 7) { continue; }
      if (i >= 9 && u.extra[0].x < 0.35) { continue; }      // action-menu icons only when it is open
      let b = u.balls[i];
      let r = max(b.z * u.blobSize, 0.001);
      var id = i;
      if (i >= 8) { id = i - 3; }
      let p = (q - b.xy) / (r * 0.95);
      if (length(p) < 1.2) {
        let idd = icon(p, id, t) * r * 0.95;
        gc = mix(gc, vec3f(0.06, 0.04, 0.12), aaq(idd - 0.002) * 0.35);
        gc = mix(gc, vec3f(1.0), aaq(idd));
      }
    }
    col = mix(col, gc, a);
  } else {
    col = mix(col, vec3f(0.0), (1.0 - smoothstep(0.0, 0.04, d)) * 0.25);   // soft drop shadow
  }
  if (u.showField > 0.5) { col = mix(col, fieldViz(f, d), 0.7); }
  return vec4f(col, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// JavaScript blob simulations (one per example). All positions in q units.
// ------------------------------------------------------------------------------------------
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function mouseQ(ctx) {
  return [(ctx.pointer.x - ctx.width / 2) / ctx.height, (ctx.pointer.y - ctx.height / 2) / ctx.height];
}

function fieldSim() {
  const balls = [];
  let grab = -1;
  const make = (i) => {
    const a = rand(0, Math.PI * 2);
    const s = rand(0.08, 0.2);
    return { x: rand(-0.5, 0.5), y: rand(-0.3, 0.3), vx: Math.cos(a) * s, vy: Math.sin(a) * s, r: rand(0.045, 0.1), hue: 0.55 + ((i * 0.13) % 0.5) };
  };
  return {
    step(ctx, dt, p) {
      const n = Math.round(p.numBlobs);
      while (balls.length < n) balls.push(make(balls.length));
      balls.length = n;
      const A = ctx.width / ctx.height;
      const [mx, my] = mouseQ(ctx);
      if (ctx.pointer.clicked) {
        let best = 0.15;
        grab = -1;
        balls.forEach((b, i) => {
          const dd = Math.hypot(b.x - mx, b.y - my);
          if (dd < best) (best = dd), (grab = i);
        });
      }
      if (!ctx.pointer.down) grab = -1;
      balls.forEach((b, i) => {
        if (i === grab) {
          b.vx = (mx - b.x) / Math.max(dt, 1 / 120) * 0.5;
          b.vy = (my - b.y) / Math.max(dt, 1 / 120) * 0.5;
          b.x = mx;
          b.y = my;
          return;
        }
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        const sp = Math.hypot(b.vx, b.vy);
        if (sp > 0.3) (b.vx *= 0.3 / sp), (b.vy *= 0.3 / sp);
        if (b.x < -A / 2 + b.r) (b.x = -A / 2 + b.r), (b.vx = Math.abs(b.vx));
        if (b.x > A / 2 - b.r) (b.x = A / 2 - b.r), (b.vx = -Math.abs(b.vx));
        if (b.y < -0.5 + b.r) (b.y = -0.5 + b.r), (b.vy = Math.abs(b.vy));
        if (b.y > 0.5 - b.r) (b.y = 0.5 - b.r), (b.vy = -Math.abs(b.vy));
      });
      return balls.map((b) => [b.x, b.y, b.r, b.hue]);
    },
  };
}

// lava lamp: wax heats up at the bottom (rises), cools near the top (sinks)
function lavaSim() {
  const halfW = (y) => (y < 0.1 ? 0.068 + (0.145 - 0.068) * clamp((y + 0.36) / 0.46, 0, 1) : 0.145 + (0.115 - 0.145) * clamp((y - 0.1) / 0.12, 0, 1));
  const wax = [];
  for (let i = 0; i < 8; i++) {
    wax.push({ x: rand(-0.04, 0.04), y: rand(-0.3, 0.18), vx: 0, vy: 0, r: rand(0.026, 0.046), T: Math.random(), seed: rand(0, 100) });
  }
  let t = 0;
  return {
    step(ctx, dt) {
      t += dt;
      for (const b of wax) {
        if (b.y > 0.13) b.T += dt * 0.32;
        else b.T -= dt * (0.05 + Math.max(0, -b.y - 0.05) * 0.5);
        b.T = clamp(b.T, 0, 1);
        b.vy += -(b.T - 0.5) * 0.28 * dt;          // buoyancy (y is down)
        b.vy *= Math.exp(-1.2 * dt);
        b.vy = clamp(b.vy, -0.09, 0.09);
        b.vx += Math.sin(t * 0.35 + b.seed) * 0.02 * dt;
        b.vx *= Math.exp(-0.8 * dt);
        b.y += b.vy * dt;
        b.x += b.vx * dt;
        if (b.y < -0.31 + b.r * 0.5) (b.y = -0.31 + b.r * 0.5), (b.vy = Math.abs(b.vy) * 0.2), (b.T = Math.min(b.T, 0.35));
        if (b.y > 0.19) (b.y = 0.19), (b.vy = 0);
        const w = halfW(b.y) - b.r * 0.75;
        if (Math.abs(b.x) > w) (b.x = Math.sign(b.x) * w), (b.vx *= -0.3);
      }
      // gentle separation so the wax spreads out
      for (let i = 0; i < wax.length; i++)
        for (let j = i + 1; j < wax.length; j++) {
          const a = wax[i];
          const c = wax[j];
          const dx = c.x - a.x;
          const dy = c.y - a.y;
          const dd = Math.hypot(dx, dy) + 1e-6;
          const m = (a.r + c.r) * 0.9;
          if (dd < m) {
            const k = ((m - dd) / dd) * 0.5 * Math.min(1, dt * 2);
            a.x -= dx * k * 0.3;
            a.y -= dy * k;
            c.x += dx * k * 0.3;
            c.y += dy * k;
          }
        }
      const pool = [
        [-0.07, 0.228, 0.055, -1],
        [0.0, 0.235, 0.06, -1],
        [0.07, 0.228, 0.055, -1],
      ];
      return [...pool, ...wax.map((b) => [b.x, b.y, b.r, -1])];
    },
  };
}

// slime: a core body + jiggly satellite lobes on springs, hopping toward the mouse, dripping goo
function slimeSim() {
  const GROUND = 0.3;
  const core = { x: 0, y: GROUND - 0.07, vx: 0, vy: 0 };
  const lobes = [
    [-0.075, 0.025, 0.05], [0.075, 0.025, 0.05], [-0.058, -0.035, 0.05], [0.058, -0.035, 0.05], [0, -0.06, 0.055], [-0.03, 0.04, 0.045], [0.03, 0.04, 0.045],
  ].map(([ox, oy, r]) => ({ ox, oy, r, x: ox, y: GROUND - 0.07 + oy, vx: 0, vy: 0 }));
  const drips = [];
  let squash = 0;
  let onGround = true;
  let cooldown = 0.5;
  let t = 0;
  let blinkT = 2;
  let look = [0, 0];
  let lastMx = 0;
  return {
    step(ctx, dt) {
      t += dt;
      const A = ctx.width / ctx.height;
      const [mx, my] = mouseQ(ctx);
      const target = ctx.pointer.over ? clamp(mx, -A / 2 + 0.14, A / 2 - 0.14) : Math.sin(t * 0.25) * A * 0.28;
      lastMx = target;
      const dx = target - core.x;
      cooldown -= dt;
      if (onGround && cooldown <= 0 && dt > 0) {
        const high = ctx.pointer.over && my < core.y - 0.15 && Math.abs(dx) < 0.25;
        if (Math.abs(dx) > 0.04 || high) {
          core.vy = high ? -1.35 : -0.75 - Math.min(Math.abs(dx), 0.4) * 0.6;
          core.vx = clamp(dx * 2.2, -0.75, 0.75);
          onGround = false;
          squash = -0.18;
        }
        cooldown = 0.25;
      }
      core.vy += 3.2 * dt;
      core.x += core.vx * dt;
      core.y += core.vy * dt;
      if (core.y > GROUND - 0.07) {
        if (!onGround) squash = Math.min(0.32, 0.12 + Math.abs(core.vy) * 0.18);
        core.y = GROUND - 0.07;
        core.vy = 0;
        core.vx *= Math.exp(-14 * dt);
        onGround = true;
      }
      core.x = clamp(core.x, -A / 2 + 0.1, A / 2 - 0.1);
      squash *= Math.exp(-6 * dt);
      const stretch = onGround ? squash : -clamp(Math.abs(core.vy) * 0.12, 0, 0.18);
      const sx = 1 + stretch;
      const sy = 1 - stretch;
      for (const l of lobes) {
        const tx = core.x + l.ox * sx;
        const ty = core.y + l.oy * sy + (l.oy > 0 ? stretch * 0.03 : 0);
        l.vx += ((tx - l.x) * 260 - l.vx * 12) * dt;
        l.vy += ((ty - l.y) * 260 - l.vy * 12) * dt;
        l.x += l.vx * dt;
        l.y += l.vy * dt;
        if (l.y > GROUND - l.r * 0.75) (l.y = GROUND - l.r * 0.75), (l.vy = 0);
      }
      // drips fly off when moving fast
      if (!onGround && Math.abs(core.vx) > 0.2 && Math.random() < dt * 6 && drips.length < 10) {
        drips.push({ x: core.x - Math.sign(core.vx) * 0.07, y: core.y + 0.02, vx: -core.vx * 0.25, vy: core.vy * 0.3, r: rand(0.014, 0.024), life: 1 });
      }
      for (const dp of drips) {
        dp.vy += 3 * dt;
        dp.x += dp.vx * dt;
        dp.y += dp.vy * dt;
        if (dp.y > GROUND - dp.r * 0.3) (dp.y = GROUND - dp.r * 0.3), (dp.vy = 0), (dp.vx *= Math.exp(-10 * dt)), (dp.life -= dt * 0.35);
        // drips near the slime get sucked back in
        const ddx = core.x - dp.x;
        if (Math.abs(ddx) < 0.12 && onGround) dp.vx += ddx * 20 * dt;
        if (Math.abs(ddx) < 0.05 && dp.y > core.y) dp.life -= dt * 2;
      }
      for (let i = drips.length - 1; i >= 0; i--) if (drips[i].life <= 0) drips.splice(i, 1);
      // eyes look at the mouse, blink now and then
      const eyeC = [core.x, core.y - 0.035 * sy];
      const lx = ctx.pointer.over ? mx - eyeC[0] : target - eyeC[0];
      const ly = ctx.pointer.over ? my - eyeC[1] : 0.1;
      const ll = Math.hypot(lx, ly) + 1e-6;
      look[0] += ((lx / ll) * 0.009 - look[0]) * Math.min(1, dt * 10);
      look[1] += ((ly / ll) * 0.009 - look[1]) * Math.min(1, dt * 10);
      blinkT -= dt;
      if (blinkT < -0.14) blinkT = rand(2, 5);
      const blink = blinkT < 0 ? 0.1 : 1;
      const balls = [[core.x, core.y, 0.085, -1], ...lobes.map((l) => [l.x, l.y, l.r, -1]), ...drips.map((dp) => [dp.x, dp.y, dp.r * Math.min(1, dp.life * 2), -1])];
      const extra = [
        [eyeC[0] - 0.032 * sx, eyeC[1], eyeC[0] + 0.032 * sx, eyeC[1]],
        [look[0], look[1], blink, clamp(Math.abs(core.vx) * 2 + (onGround ? 0 : 0.6), 0, 1)],
        [core.x, core.y, 0.11 * sx * (onGround ? 1 : 0.7), onGround ? 1 : clamp(1 - (GROUND - 0.07 - core.y) * 3, 0.2, 1)],
        [lastMx, 0, 0, 0],
      ];
      return { balls, extra };
    },
  };
}

// gooey UI: a dock of buttons + a selection blob + a cursor blob + an expanding action menu
function uiSim() {
  const spring = (o, tx, ty, k, c, dt) => {
    o.vx += ((tx - o.x) * k - o.vx * c) * dt;
    o.vy += ((ty - o.y) * k - o.vy * c) * dt;
    o.x += o.vx * dt;
    o.y += o.vy * dt;
  };
  const P = () => ({ x: 0, y: 0, vx: 0, vy: 0 });
  const buttons = Array.from({ length: 5 }, () => ({ ...P(), hover: 0 }));
  const lead = P();
  const tail = P();
  const cursor = P();
  const fab = P();
  const kids = Array.from({ length: 4 }, () => P());
  let selected = 0;
  let open = 0;
  let openTarget = 0;
  let idle = 0;
  let autoT = 0;
  let init = false;
  return {
    step(ctx, dt) {
      const A = ctx.width / ctx.height;
      const [mx, my] = mouseQ(ctx);
      const dockC = -A * 0.17;
      const pos = (i) => [dockC + (i - 2) * 0.15, -0.02];
      const fabP = [A * 0.31, 0.24];
      const kidP = (k) => {
        const ang = -Math.PI / 2 - (k / 3) * (Math.PI / 2);   // a quarter circle up and to the left
        return [fabP[0] + Math.cos(ang) * 0.21, fabP[1] + Math.sin(ang) * 0.21];
      };
      if (!init) {
        init = true;
        buttons.forEach((b, i) => ([b.x, b.y] = pos(i)));
        [lead.x, lead.y] = pos(0);
        [tail.x, tail.y] = pos(0);
        [fab.x, fab.y] = fabP;
        kids.forEach((k) => ([k.x, k.y] = fabP));
        [cursor.x, cursor.y] = [mx, my];
      }
      idle += dt;
      if (ctx.pointer.clicked) {
        idle = 0;
        buttons.forEach((b, i) => {
          if (Math.hypot(b.x - mx, b.y - my) < 0.07) selected = i;
        });
        if (Math.hypot(fab.x - mx, fab.y - my) < 0.08) openTarget = openTarget ? 0 : 1;
        kids.forEach((k) => {
          if (open > 0.5 && Math.hypot(k.x - mx, k.y - my) < 0.06) openTarget = 0;
        });
      }
      // demo mode: when nobody clicks, animate by itself
      if (idle > 3 && dt > 0) {
        autoT += dt;
        if (autoT > 1.6) {
          autoT = 0;
          if (Math.random() < 0.6) selected = (selected + 1 + Math.floor(Math.random() * 3)) % 5;
          else openTarget = openTarget ? 0 : 1;
        }
      }
      open += (openTarget - open) * Math.min(1, dt * 6);
      buttons.forEach((b, i) => {
        const [tx, ty] = pos(i);
        const near = ctx.pointer.over && Math.hypot(tx - mx, ty - my) < 0.075 ? 1 : 0;
        b.hover += (near - b.hover) * Math.min(1, dt * 10);
        spring(b, tx, ty - (i === selected ? 0.035 : 0) - b.hover * 0.01, 300, 16, dt);
      });
      const [sx, sy] = pos(selected);
      spring(lead, sx, sy + 0.05, 260, 20, dt);
      spring(tail, lead.x, lead.y, 45, 9, dt);
      if (ctx.pointer.over) spring(cursor, mx, my, 220, 22, dt);
      else spring(cursor, cursor.x, 0.9, 30, 8, dt);
      spring(fab, fabP[0], fabP[1], 200, 14, dt);
      kids.forEach((k, i) => {
        const [kx, ky] = kidP(i);
        const o = clamp(open * 1.3 - i * 0.1, 0, 1);
        spring(k, fabP[0] + (kx - fabP[0]) * o, fabP[1] + (ky - fabP[1]) * o, 150 + i * 30, 11, dt);
      });
      const hues = [0.58, 0.68, 0.95, 0.12, 0.45];
      const balls = [
        ...buttons.map((b, i) => [b.x, b.y, 0.048 * (1 + 0.25 * b.hover + (i === selected ? 0.12 : 0)), hues[i]]),
        [cursor.x, cursor.y, 0.026, 0.8],
        [lead.x, lead.y, 0.03, -1],
        [tail.x, tail.y, 0.022, -1],
        [fab.x, fab.y, 0.065, -1],
        ...kids.map((k, i) => [k.x, k.y, 0.042 * clamp(open * 2, 0.35, 1), [0.5, 0.62, 0.75, 0.85][i]]),
      ];
      const extra = [
        [open, 0, 0, 0],
        [dockC, -0.02, 0.36, 0.0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      return { balls, extra };
    },
  };
}

const SIMS = { field: fieldSim, lava: lavaSim, slime: slimeSim, ui: uiSim };
const IMAGES = { field: FIELD_IMG, lava: LAVA_IMG, slime: SLIME_IMG, ui: UI_IMG };

// ------------------------------------------------------------------------------------------
const examples = [
  {
    id: 'field',
    label: 'Field & threshold',
    kind: 'Abstract',
    note: 'The background is the <b>field</b>: every blob adds <code>r²/d²</code>. Where the sum crosses the <b>threshold</b> (white line) is the surface. Dashed rings show each blob alone; when two get close their fields add up and the surface bridges between them. Drag a blob.',
    params: { showField: true, fieldFn: 'inv', threshold: 1, shading: true, numBlobs: 7 },
  },
  {
    id: 'lava',
    label: 'Lava lamp',
    kind: 'Real life',
    note: 'Wax heats up at the bottom, rises, cools and sinks. The blobs are simple circles moving in JavaScript — the gooey merging and splitting is entirely the metaball field.',
    params: { showField: false, fieldFn: 'inv', threshold: 1, gloss: 24, rim: 0.6, tint: '#ff5a3c' },
  },
  {
    id: 'slime',
    label: 'Slime character',
    kind: 'In a game',
    note: 'A core blob plus seven springy “lobes” and a few drips. Each part is simulated as a simple point; the metaball field fuses them into one wobbly, squashy body. Move the mouse — it hops toward it.',
    params: { showField: false, fieldFn: 'inv', threshold: 1, gloss: 60, rim: 1, tint: '#5ee05a' },
  },
  {
    id: 'ui',
    label: 'Gooey UI',
    kind: 'In a game',
    note: 'Menu buttons rendered as a <b>smooth-min union</b> of circles: the selection bubble, the cursor and the pop-out action menu melt into each other. Hover and click the buttons, and the round “+” button.',
    params: { showField: false, fieldFn: 'smin', smoothK: 0.05, threshold: 1, gloss: 50, rim: 0.5, tint: '#ff5fa2' },
  },
];

const controls = [
  { type: 'heading', label: 'Field' },
  {
    type: 'select',
    key: 'fieldFn',
    label: 'Field function',
    value: 'inv',
    options: [
      { value: 'inv', label: 'Inverse square  r²/d²  (classic)' },
      { value: 'wyvill', label: 'Wyvill  (1−d²/R²)³  (finite reach)' },
      { value: 'smin', label: 'Smooth-min of circle SDFs' },
    ],
    help: 'How each blob’s influence falls off with distance.',
  },
  { type: 'slider', key: 'threshold', label: 'Threshold', min: 0.3, max: 3, step: 0.01, value: 1, help: 'The iso-level that counts as “inside”. Lower = fatter blobs that merge from further away.' },
  { type: 'slider', key: 'smoothK', label: 'Smooth-min blend (k)', min: 0.002, max: 0.2, step: 0.001, value: 0.05, help: 'Only for “Smooth-min”: the distance over which shapes melt together.' },
  { type: 'slider', key: 'blobSize', label: 'Blob size', min: 0.4, max: 1.8, step: 0.01, value: 1 },
  { type: 'slider', key: 'numBlobs', label: 'Blobs', min: 2, max: 16, step: 1, value: 7, showFor: ['field'] },
  { type: 'toggle', key: 'showField', label: 'Show the field', value: false, help: 'Heat map of the summed field with iso-lines.' },
  { type: 'heading', label: 'Look' },
  { type: 'toggle', key: 'shading', label: 'Fake 3D lighting', value: true, help: 'Light the blob as a dome whose height comes from the distance to its edge.' },
  { type: 'slider', key: 'gloss', label: 'Gloss', min: 4, max: 128, step: 1, value: 40, help: 'Specular exponent: higher = smaller, sharper highlight.' },
  { type: 'slider', key: 'rim', label: 'Rim light', min: 0, max: 2, step: 0.01, value: 0.8 },
  { type: 'toggle', key: 'colorBlend', label: 'Blend colors', value: true, help: 'Mix blob colors by their share of the field (off = nearest blob wins).' },
  { type: 'color', key: 'tint', label: 'Wax / goo color', value: '#ff5a3c', showFor: ['lava', 'slime', 'ui'] },
  { type: 'color', key: 'liquid', label: 'Liquid color', value: '#5a2a8a', showFor: ['lava'] },
  { type: 'slider', key: 'speed', label: 'Speed', min: 0, max: 3, step: 0.01, value: 1 },
];

const uniforms = {
  balls: `array<vec4f, ${MAXB}>`,
  extra: 'array<vec4f, 4>',
  nBalls: 'f32',
  fieldFn: 'f32',
  threshold: 'f32',
  smoothK: 'f32',
  blobSize: 'f32',
  showField: 'f32',
  shading: 'f32',
  gloss: 'f32',
  rim: 'f32',
  colorBlend: 'f32',
  tint: 'vec3f',
  liquid: 'vec3f',
};


function build(ctx) {
  const ex = SIMS[ctx.example] ? ctx.example : 'field';
  const sim = SIMS[ex]();
  const ballData = new Float32Array(MAXB * 4);
  const extraData = new Float32Array(16);
  return shaderScene({
    controls,
    uniforms,
    include: ['math', 'noise', 'sdf', 'color'],
    code: IMAGES[ex],
    renderScale: ctx.testMode ? 0.5 : 1,
    bind(params, c) {
      const dt = Math.min(c.dt, 1 / 30) * params.speed;
      const out = sim.step(c, dt, params);
      const balls = Array.isArray(out) ? out : out.balls;
      ballData.fill(0);
      const n = Math.min(MAXB, balls.length);
      for (let i = 0; i < n; i++) ballData.set(balls[i], i * 4);
      extraData.fill(0);
      if (out.extra) out.extra.forEach((v, i) => extraData.set(v, i * 4));
      return { balls: ballData, nBalls: n, extra: extraData };
    },
  });
}

export default {
  interaction: 'Move the mouse. Drag blobs in “Field & threshold”; click buttons in “Gooey UI”.',
  examples,
  controls,
  backends: ['webgpu', 'webgl2'],
  reinitOnExample: true,
  init: (ctx) => build(ctx).init(ctx),
  initGL: (ctx) => build(ctx).initGL(ctx),
  about: {
    summary:
      'Metaballs are blobs that melt together. Each blob radiates an invisible <b>field</b>; add the fields up and draw everything above a <b>threshold</b>. Where two fields overlap, the sum bridges the gap — instant goo.',
    what: `<p>A handful of circles move around (in JavaScript). For every pixel the fragment shader adds up each circle’s influence and
      compares the total to a threshold. Shading, rims and color blending are all derived from that same field.</p>`,
    how: `<ol>
      <li><b>Field</b>: each blob contributes <code>f<sub>i</sub>(p) = r²/|p − c|²</code> (1 at its radius, growing toward its center).
        The total field is <code>F(p) = Σ f<sub>i</sub>(p)</code>.</li>
      <li><b>Threshold</b>: the surface is the iso-line <code>F(p) = T</code>. A lone blob gives a circle of radius r (at T = 1);
        two nearby blobs add up between them and the surface swells into a neck that eventually connects them.</li>
      <li><b>Anti-aliasing</b>: dividing <code>(T − F)</code> by the field’s gradient length gives an approximate distance to the surface
        in pixels — a crisp, smooth edge at any resolution.</li>
      <li><b>Fake 3D</b>: turn the field into a height (0 at the edge, rising inside). Its gradient tilts the surface normal outward near the
        rim and flattens it deep inside — then use ordinary diffuse + specular + rim lighting. (For the smooth-min SDF the height comes from the distance to the edge: a dome.)</li>
      <li><b>Color blending</b>: weight each blob’s color by its share of the field, so colors flow into each other where blobs merge.</li>
      <li><b>Relation to SDFs</b>: a smooth-minimum of circle distance functions (<code>smin(d₁, d₂, k)</code>) gives a very similar
        look with a <i>true</i> distance field (great for outlines and glows) — pick “Smooth-min” to compare. The Wyvill kernel has
        finite reach, so far-away blobs cost nothing (good for spatial grids).</li>
    </ol>`,
    uses: [
      { title: 'Characters', text: 'Slimes, jelly bosses, liquid creatures and amoebas that squash and split.' },
      { title: 'Liquids & lava', text: 'Particle-based water or lava rendered as one surface (2D “screen-space fluids”). Used for blood, goo and paint.' },
      { title: 'UI', text: 'Gooey menus, tab indicators and loaders — the CSS “gooey effect” does the same with blur + threshold.' },
      { title: 'Organic VFX', text: 'Cells dividing, magic orbs merging, oil in water, mercury.' },
    ],
    try: [
      'In <b>Field & threshold</b>, drag one blob slowly toward another and watch the white contour form a bridge before they touch.',
      'Lower the <i>Threshold</i> to 0.4: everything inflates and merges. Raise it to 2: blobs shrink and separate.',
      'Switch the <i>Field function</i> to Wyvill: blobs only interact when close, and merging is snappier.',
      'Turn off <i>Fake 3D lighting</i> to see how much the shading sells the goo.',
      'In <b>Gooey UI</b> raise the <i>Smooth-min blend</i> to 0.15: everything becomes one big puddle.',
    ],
    ask: [
      'metaball rendering for a slime character with squash and stretch',
      'gooey UI menu that melts buttons together (smooth-min)',
      'render fluid particles as metaballs with a threshold',
      'lava lamp effect',
      'fake 3D shading on 2D blobs from their distance field',
      'color blending between merged blobs',
    ],
    perf: `<p>Cost is <i>pixels × blobs</i>: each pixel loops over every blob (here up to ${MAXB}). That is fine for dozens of blobs. For thousands
      (e.g. fluid particles) splat each particle’s kernel additively into a low-resolution field texture instead, then threshold that
      texture in one full-screen pass — or bin blobs into tiles so each pixel only visits nearby ones.</p>`,
    api: `<p>Identical in <b>WebGL2</b> and <b>WebGPU</b>: a single fragment shader reading a uniform array of blob positions.
      With WebGPU you could move the blob simulation itself to a compute shader and store thousands of particles in a storage buffer.</p>`,
    code: [
      {
        title: 'The field, its threshold and a distance estimate',
        lang: 'wgsl',
        src: `var f = 0.0;  var g = vec2f(0.0);
for (var i = 0; i < MAXB; i++) {
  if (f32(i) >= u.nBalls) { break; }
  let b  = u.balls[i];                       // xy = center, z = radius
  let r  = b.z * u.blobSize;
  let dv = q - b.xy;
  let d2 = dot(dv, dv) + 0.0000001;
  let c  = r * r / d2;                       // this blob's influence
  f += c;                                    // the field
  g += -2.0 * c / d2 * dv;                   // its gradient (for normals & AA)
}
let gl = max(length(g), 0.0001);
let dist = (u.threshold - f) / gl;           // < 0 inside, ≈ distance to the surface`,
      },
      {
        title: 'Fake 3D: a smooth height from the field',
        lang: 'wgsl',
        src: `// height H = 1 - exp(-(F - T)): 0 at the edge, rising smoothly inside.
// Its gradient (chain rule) tilts the normal outward near the rim and flattens deep inside.
let T = max(u.threshold, 0.001);
let tilt = g * exp(-max(f - T, 0.0) * 1.5 / T) * 1.5 / T * R0;   // g = outward field gradient
let n = normalize(vec3f(tilt, 1.0));
let diff = max(dot(n, L), 0.0);
let spec = pow(max(dot(n, H), 0.0), u.gloss);
let rim  = pow(1.0 - n.z, 2.0) * u.rim;`,
      },
    ],
    links: [
      { title: 'Jamie Wong — Metaballs and Marching Squares', url: 'https://jamie-wong.com/2014/08/19/metaballs-and-marching-squares/', note: 'a great visual explanation' },
      { title: 'Inigo Quilez — smooth minimum', url: 'https://iquilezles.org/articles/smin/' },
    ],
  },
};

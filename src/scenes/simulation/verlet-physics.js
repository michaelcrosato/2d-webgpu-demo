// Verlet physics on the GPU: points + distance constraints, solved with compute shaders.
// Constraints are "graph colored" on the CPU once, so every color is a batch of constraints that share
// no points — a whole batch can then be solved in parallel (one GPU thread per constraint) without
// two threads ever writing the same point. This is parallel Gauss–Seidel.
//
// Everything (simulation, picking, tearing, cutting) stays on the GPU; nothing is read back.

import { FULLSCREEN_VS } from '../../core/webgpu.js';
import { overlayTag } from './_shared-b.js';

const WORLD_H = 1000; // world units: the view is always 1000 units tall, W = 1000 * aspect wide
const MAX_SUB = 8;
const MAXRING = 32;

// point draw styles (meta.y)
const PS = { none: 0, pin: 4, lantern: 2, ball: 3, ring: 5, knot: 6 };
// constraint draw styles (low 4 bits of Con.style)
const CS = { hidden: 0, rope: 1, chain: 2, plank: 3, string: 4, cloth: 5, jelly: 6 };
// point kinds (meta.x)
const KIND = { plain: 0, cloth: 1, blob: 2, curtain: 3 };

// ------------------------------------------------------------------------------------------ WGSL
const STRUCTS = /* wgsl */ `
struct Pt { p: vec4f, o: vec4f };     // p.xyz = position, p.w = inverse mass (0 = pinned); o.xyz = previous position, o.w = radius
struct Con { a: u32, b: u32, rest: f32, stiff: f32, style: f32, tear: f32, alive: f32, cut: f32 };
struct Blob { first: u32, count: u32, restArea: f32, style: f32, c: vec2f, area: f32, radius: f32 };
struct Quad { pts: vec4u, edges: vec4u, info: vec4f };
`;

const SIM_WGSL = /* wgsl */ `
${STRUCTS}
fn isGrabbed(i: u32) -> bool {
  let pk = atomicLoad(&pick[0]);
  return u.grab > 0.5 && pk != 0xffffffffu && (pk & 0xffffu) == i;
}
fn ccw(a: vec2f, b: vec2f, c: vec2f) -> f32 { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }
fn segCross(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  return ccw(a, b, c) * ccw(a, b, d) < 0.0 && ccw(c, d, a) * ccw(c, d, b) < 0.0;
}

// ---- 1. per blob: area and centroid (for the pressure force and collisions)
@compute @workgroup_size(64) fn blobArea(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= u.nBlobs) { return; }
  var bl = blobs[b];
  var A = 0.0;
  var c = vec2f(0.0);
  for (var k = 0u; k < bl.count; k++) {
    let p0 = pts[bl.first + k].p.xy;
    let p1 = pts[bl.first + (k + 1u) % bl.count].p.xy;
    A += p0.x * p1.y - p1.x * p0.y;   // shoelace formula
    c += p0;
  }
  c /= f32(bl.count);
  var r = 0.0;
  for (var k = 0u; k < bl.count; k++) { r = max(r, length(pts[bl.first + k].p.xy - c)); }
  bl.area = max(abs(A) * 0.5, 1.0);
  bl.c = c;
  bl.radius = r;
  blobs[b] = bl;
}

// ---- 2. per point: external forces that depend on neighbours (wind on cloth, pressure in blobs)
@compute @workgroup_size(64) fn forces(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.nPts) { return; }
  let m = pmeta[i];
  let nb = nbr[i];
  let pt = pts[i];
  var f = vec3f(0.0);
  if (m.x == 1 || m.x == 3) {
    // cloth: aerodynamic force from the wind relative to the moving cloth
    let p = pt.p.xyz;
    let pl = select(p, pts[u32(max(nb.x, 0))].p.xyz, nb.x >= 0);
    let pr = select(p, pts[u32(max(nb.y, 0))].p.xyz, nb.y >= 0);
    let pu = select(p, pts[u32(max(nb.z, 0))].p.xyz, nb.z >= 0);
    let pd = select(p, pts[u32(max(nb.w, 0))].p.xyz, nb.w >= 0);
    let nn = cross(pr - pl, pd - pu);
    let ln = length(nn);
    if (ln > 0.0001) {
      let n = nn / ln;
      let gust = 0.65 + 0.35 * sin(u.time * 1.3 + p.x * 0.004) + 0.25 * sin(u.time * 3.1 + p.y * 0.01 + p.x * 0.007);
      var wind = vec3f(u.wind.x * gust, 0.0, u.wind.x * 0.18 * sin(u.time * 2.3 + p.x * 0.013 + p.y * 0.005));
      if (m.x == 3) { wind *= 0.18; }            // the curtain hangs indoors: only a draft
      let vel = (p - pt.o.xyz) / max(u.dt, 0.00001);
      let rel = wind - vel;
      f = n * dot(n, rel) * u.wind.w + rel * u.wind.y;   // pressure on the surface + skin drag along it
    }
  } else if (m.x == 2) {
    // blob ring point: gas pressure pushes along the outward edge normal, ∝ (rest area / area − 1)
    let bl = blobs[u32(nb.z)];
    let e = pts[u32(nb.y)].p.xy - pts[u32(nb.x)].p.xy;
    var n2 = vec2f(e.y, -e.x);
    if (dot(n2, pt.p.xy - bl.c) < 0.0) { n2 = -n2; }
    f = vec3f(n2 * (bl.restArea / bl.area - 1.0) * u.pressure * 260.0, 0.0);
  }
  frc[i] = vec4f(f, 0.0);
}

// ---- 3. per point: Verlet integration  x' = x + (x − x_prev)·damping + a·dt²
@compute @workgroup_size(64) fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.nPts) { return; }
  var pt = pts[i];
  if (isGrabbed(i)) {
    // the grabbed point follows the mouse (interpolated across sub-steps so a fling keeps its speed)
    let tgt = mix(u.pmouse, u.mouse, sub.frac);
    pt.o = vec4f(pt.p.xyz, pt.o.w);
    pt.p = vec4f(tgt, pt.p.z, pt.p.w);
    pts[i] = pt;
    return;
  }
  if (pt.p.w <= 0.0) { return; }                 // pinned
  let pos = pt.p.xyz;
  let vel = (pos - pt.o.xyz) * u.damping;
  let acc = vec3f(u.gravity, 0.0) + frc[i].xyz;
  var np = pos + vel + acc * u.dt * u.dt;
  let kd = pmeta[i].x;
  if (kd != 1 && kd != 3) { np.z = 0.0; }        // only cloth moves in depth
  pt.o = vec4f(pos, pt.o.w);
  pt.p = vec4f(np, pt.p.w);
  pts[i] = pt;
}

// ---- 4. per constraint (one color at a time): move both ends to restore the rest length
@compute @workgroup_size(64) fn solve(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= rg.count) { return; }
  let ci = rg.first + k;
  let c = cons[ci];
  if (c.alive < 0.5) { return; }
  let A = pts[c.a].p;
  let B = pts[c.b].p;
  let d = B.xyz - A.xyz;
  let len = length(d);
  if (c.tear > 0.0 && len > c.rest * c.tear * u.tearMul) { cons[ci].alive = 0.0; return; }        // tearing
  if (u.cutting > 0.5 && c.cut > 0.5 && segCross(A.xy, B.xy, u.pmouse, u.mouse)) { cons[ci].alive = 0.0; return; } // knife
  var wa = A.w;
  var wb = B.w;
  if (isGrabbed(c.a)) { wa = 0.0; }
  if (isGrabbed(c.b)) { wb = 0.0; }
  let w = wa + wb;
  if (w <= 0.0 || len < 0.000001) { return; }
  let corr = d * ((len - c.rest) / len) * min(c.stiff * u.stiffMul, 1.0);
  pts[c.a].p = vec4f(A.xyz + corr * (wa / w), A.w);
  pts[c.b].p = vec4f(B.xyz - corr * (wb / w), B.w);
}

// ---- 5. per point: collisions with the floor, walls, static obstacles and other blobs
@compute @workgroup_size(64) fn collide(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.nPts) { return; }
  var pt = pts[i];
  if (pt.p.w <= 0.0 || isGrabbed(i)) { return; }
  var p = pt.p.xyz;
  let r = max(pt.o.w, 2.0);
  var hit = false;
  if (p.y > u.floorY - r) { p.y = u.floorY - r; hit = true; }
  if (p.x < r) { p.x = r; hit = true; }
  if (p.x > u.worldW - r) { p.x = u.worldW - r; hit = true; }
  for (var k = 0u; k < u.nObs; k++) {
    let ob = u.obs[k];
    if (ob.w < 0.0) {                                  // circle
      let dv = p.xy - ob.xy;
      let dl = length(dv);
      if (dl < ob.z + r && dl > 0.0001) { p = vec3f(ob.xy + dv / dl * (ob.z + r), p.z); hit = true; }
    } else {                                           // box (center, half size)
      let q = p.xy - ob.xy;
      let pen = vec2f(ob.z + r, ob.w + r) - abs(q);
      if (pen.x > 0.0 && pen.y > 0.0) {
        if (pen.x < pen.y) { p.x += sign(q.x) * pen.x; } else { p.y += sign(q.y) * pen.y; }
        hit = true;
      }
    }
  }
  let m = pmeta[i];
  if (m.x == 2) {
    // soft body vs soft body: if I'm inside another blob's polygon, move me to its closest edge
    let own = u32(nbr[i].z);
    for (var b = 0u; b < u.nBlobs; b++) {
      if (b == own) { continue; }
      let bl = blobs[b];
      if (length(p.xy - bl.c) > bl.radius + r) { continue; }
      var inside = false;
      var best = 1000000.0;
      var bestP = p.xy;
      for (var k = 0u; k < bl.count; k++) {
        let a = pts[bl.first + k].p.xy;
        let c2 = pts[bl.first + (k + 1u) % bl.count].p.xy;
        if ((a.y > p.y) != (c2.y > p.y)) {
          if (p.x < (c2.x - a.x) * (p.y - a.y) / (c2.y - a.y) + a.x) { inside = !inside; }
        }
        let ba = c2 - a;
        let h = clamp(dot(p.xy - a, ba) / max(dot(ba, ba), 0.0001), 0.0, 1.0);
        let cp = a + ba * h;
        let dd = length(p.xy - cp);
        if (dd < best) { best = dd; bestP = cp; }
      }
      if (inside) { p = vec3f(mix(p.xy, bestP, 0.6), p.z); hit = true; }
    }
  }
  if (hit) {
    // friction: kill part of the tangential motion
    let v = p - pt.o.xyz;
    pt.o = vec4f(pt.o.xyz + v * u.friction * 0.5, pt.o.w);
  }
  pt.p = vec4f(p, pt.p.w);
  pts[i] = pt;
}

// ---- picking: the nearest point to the mouse wins an atomicMin on (distance << 16 | index)
@compute @workgroup_size(64) fn pickPoint(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.nPts) { return; }
  let d = length(pts[i].p.xy - u.mouse);
  if (d < u.pickR) { atomicMin(&pick[0], (min(u32(d * 8.0), 0x7fffu) << 16u) | i); }
}
`;

const DRAW_WGSL = /* wgsl */ `
${STRUCTS}
${FULLSCREEN_VS}
const LD: vec3f = vec3f(-0.42, -0.55, 0.72);
fn toClip(w: vec2f) -> vec4f { return vec4f(w.x / u.worldW * 2.0 - 1.0, 1.0 - w.y / 500.0, 0.0, 1.0); }
fn aa(d: f32) -> f32 { return clamp(0.5 - d / max(fwidth(d), 0.0001), 0.0, 1.0); }
fn hash1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }
fn grabbedIndex() -> u32 { if (u.grab < 0.5) { return 0xffffffffu; } return pick[0] & 0xffffu; }

// ---------------------------------------------------------------- background
fn sdBoxB(p: vec2f, b: vec2f) -> f32 { let d = abs(p) - b; return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0); }
fn rockColor(w: vec2f) -> vec3f {
  let strata = 0.5 + 0.5 * sin(w.y * 0.06 + sin(w.x * 0.01) * 2.0);
  return mix(vec3f(0.33, 0.22, 0.17), vec3f(0.45, 0.31, 0.22), strata) * (0.85 + 0.15 * hash1(floor(w.x / 9.0) + floor(w.y / 9.0) * 57.0));
}
@fragment fn fs_bg(i: VSOut) -> @location(0) vec4f {
  let w = vec2f(i.uv.x * u.worldW, i.uv.y * 1000.0);
  let ex = i32(u.example + 0.5);
  var col = vec3f(0.0);
  if (ex == 0) {
    // dusk sky, mountains, a wooden beam on top, two cliffs and the chasm
    col = mix(vec3f(0.08, 0.1, 0.25), vec3f(0.95, 0.55, 0.35), smoothstep(150.0, 760.0, w.y));
    col += vec3f(1.0, 0.7, 0.4) * exp(-length(w - vec2f(u.worldW * 0.5, 700.0)) * 0.004) * 0.35;
    let mtn = 560.0 + 70.0 * sin(w.x * 0.004 + 1.0) + 35.0 * sin(w.x * 0.011 + 2.0) + 15.0 * sin(w.x * 0.03);
    col = mix(col, vec3f(0.3, 0.22, 0.35), smoothstep(mtn, mtn + 2.0, w.y) * 0.85);
    let mtn2 = 650.0 + 40.0 * sin(w.x * 0.007 + 4.0) + 20.0 * sin(w.x * 0.02);
    col = mix(col, vec3f(0.18, 0.13, 0.22), smoothstep(mtn2, mtn2 + 2.0, w.y));
    // chasm
    col = mix(col, vec3f(0.05, 0.04, 0.07), smoothstep(700.0, 900.0, w.y));
    // cliffs
    let xL = u.scene.x;
    let xR = u.scene.y;
    let top = u.scene.z;
    if (w.y > top && (w.x < xL || w.x > xR)) {
      var rc = rockColor(w);
      rc *= mix(1.0, 0.45, smoothstep(top, 1000.0, w.y));
      rc = mix(rc, vec3f(0.35, 0.6, 0.25), smoothstep(top + 14.0, top + 2.0, w.y));
      let edge = min(abs(w.x - xL), abs(w.x - xR));
      rc *= 0.7 + 0.3 * smoothstep(0.0, 18.0, edge);
      col = rc;
    }
    // posts (bridge anchors)
    for (var k = 0; k < 2; k++) {
      let px = select(xR, xL, k == 0);
      let d = sdBoxB(w - vec2f(px, top - 48.0), vec2f(9.0, 50.0));
      col = mix(col, vec3f(0.36, 0.22, 0.12) * (0.8 + 0.2 * sin(w.y * 0.3)), aa(d));
    }
    // beam on top
    let beam = sdBoxB(w - vec2f(u.worldW * 0.5, 62.0), vec2f(u.worldW, 22.0));
    var wood = mix(vec3f(0.42, 0.26, 0.14), vec3f(0.55, 0.36, 0.2), 0.5 + 0.5 * sin(w.x * 0.05 + sin(w.y * 0.4) * 2.0));
    wood *= 0.75 + 0.25 * smoothstep(40.0, 70.0, w.y);
    col = mix(col, wood, aa(beam));
    col = mix(col, vec3f(0.0), (1.0 - smoothstep(84.0, 100.0, w.y)) * step(84.0, w.y) * 0.35);
  } else if (ex == 1) {
    // dusk sky; right half a stone wall with an arched window behind the curtain
    col = mix(vec3f(0.18, 0.3, 0.6), vec3f(0.98, 0.72, 0.5), smoothstep(0.0, 1000.0, w.y));
    col += vec3f(1.0, 0.9, 0.6) * exp(-length(w - vec2f(u.worldW * 0.3, 760.0)) * 0.006) * 0.4;
    let cl = sin(w.x * 0.006 + u.time * 0.05) * sin(w.y * 0.02 + 1.0) + sin(w.x * 0.013 - u.time * 0.03 + 2.0) * 0.5;
    col = mix(col, vec3f(1.0, 0.92, 0.88), smoothstep(0.6, 1.2, cl) * smoothstep(500.0, 150.0, w.y) * 0.6);
    let hillY = 860.0 + 30.0 * sin(w.x * 0.004);
    col = mix(col, vec3f(0.2, 0.34, 0.25), smoothstep(hillY, hillY + 2.0, w.y));
    // the wall
    let wallX = u.scene.x;
    if (w.x > wallX) {
      let win = sdBoxB(w - vec2f(u.scene.y, 500.0), vec2f(u.scene.z, 270.0));
      let arch = length(w - vec2f(u.scene.y, 230.0)) - u.scene.z;
      let opening = min(win, max(arch, w.y - 230.0));
      var bq = vec2f(w.x / 70.0, w.y / 36.0);
      bq.x += 0.5 * (floor(bq.y) - 2.0 * floor(bq.y * 0.5));
      let bf = fract(bq);
      var stone = mix(vec3f(0.42, 0.38, 0.36), vec3f(0.55, 0.5, 0.46), hash1(floor(bq.x) * 13.0 + floor(bq.y) * 71.0));
      stone *= 0.6 + 0.4 * smoothstep(0.0, 0.06, min(min(bf.x, 1.0 - bf.x) * 0.5, min(bf.y, 1.0 - bf.y)));
      stone *= 1.0 - 0.35 * smoothstep(40.0, 0.0, w.x - wallX);
      // inner shadow around the opening
      stone *= 1.0 - 0.5 * (1.0 - smoothstep(0.0, 26.0, opening));
      col = mix(col, stone, smoothstep(-1.0, 1.0, opening));
    }
    // flag pole with a golden finial
    let pole = sdBoxB(w - vec2f(u.scene.w - 7.0, 600.0), vec2f(6.0, 470.0));
    let pc = mix(vec3f(0.35, 0.36, 0.4), vec3f(0.85, 0.86, 0.9), exp(-pow((w.x - u.scene.w + 9.0) * 0.4, 2.0)));
    col = mix(col, pc, aa(pole));
    let fin = length(w - vec2f(u.scene.w - 7.0, 122.0)) - 13.0;
    col = mix(col, mix(vec3f(0.75, 0.5, 0.1), vec3f(1.0, 0.9, 0.5), smoothstep(10.0, -10.0, w.x - u.scene.w + 12.0 + w.y - 118.0)), aa(fin));
    // curtain rail
    let rail = sdBoxB(w - vec2f((u.scene.y), 120.0), vec2f(u.scene.z + 70.0, 5.0));
    col = mix(col, mix(vec3f(0.6, 0.42, 0.12), vec3f(1.0, 0.85, 0.45), smoothstep(124.0, 116.0, w.y)), aa(rail));
    for (var k = 0; k < 2; k++) {
      let fx = u.scene.y + (f32(k) * 2.0 - 1.0) * (u.scene.z + 75.0);
      col = mix(col, vec3f(0.9, 0.7, 0.3), aa(length(w - vec2f(fx, 120.0)) - 11.0));
    }
  } else {
    // playroom: pastel wall with polka dots, striped floor, candy pegs
    col = mix(vec3f(0.98, 0.93, 0.86), vec3f(0.93, 0.84, 0.95), w.y / 1000.0);
    let dq = fract(w / 60.0 + vec2f(0.0, 0.5 * floor(w.x / 60.0))) - 0.5;
    col = mix(col, col * 0.93, 1.0 - smoothstep(0.1, 0.13, length(dq)));
    if (w.y > u.floorY) {
      let st = step(0.5, fract((w.x + w.y) / 80.0));
      col = mix(vec3f(0.55, 0.4, 0.62), vec3f(0.62, 0.47, 0.7), st) * mix(1.0, 0.7, smoothstep(u.floorY, 1000.0, w.y));
    }
    col *= 1.0 - 0.25 * smoothstep(12.0, 0.0, abs(w.y - u.floorY - 6.0)) * step(w.y, u.floorY);
  }
  // static obstacles (pegs)
  for (var k = 0u; k < u.nObs; k++) {
    let ob = u.obs[k];
    if (ob.w < 0.0) {
      let d = length(w - ob.xy) - ob.z;
      col = mix(col, vec3f(0.0), (1.0 - smoothstep(0.0, 22.0, d - 6.0)) * 0.18);
      let a = atan2(w.y - ob.y, w.x - ob.x);
      let swirl = step(0.5, fract(a / 6.2831853 * 6.0 + length(w - ob.xy) / ob.z * 0.6));
      var pc = mix(vec3f(1.0, 0.45, 0.5), vec3f(1.0, 0.97, 0.95), swirl);
      pc *= 0.7 + 0.45 * clamp(dot(normalize(vec3f(w - ob.xy, ob.z * 0.6)), normalize(LD)), 0.0, 1.0);
      col = mix(col, pc, aa(d));
    }
  }
  return vec4f(col, 1.0);
}

// ---------------------------------------------------------------- cloth (two triangles per grid quad)
struct ClothOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) n: vec3f, @location(2) @interpolate(flat) style: f32, @location(3) w: vec2f };
fn vnormal(i: u32) -> vec3f {
  let nb = nbr[i];
  let p = pts[i].p.xyz;
  let pl = select(p, pts[u32(max(nb.x, 0))].p.xyz, nb.x >= 0);
  let pr = select(p, pts[u32(max(nb.y, 0))].p.xyz, nb.y >= 0);
  let pu = select(p, pts[u32(max(nb.z, 0))].p.xyz, nb.z >= 0);
  let pd = select(p, pts[u32(max(nb.w, 0))].p.xyz, nb.w >= 0);
  let n = cross(pr - pl, pd - pu);
  return n / max(length(n), 0.0001);
}
@vertex fn vs_cloth(@builtin(vertex_index) vi: u32, @builtin(instance_index) qi: u32) -> ClothOut {
  let q = quads[qi];
  var corner = array<u32, 6>(0u, 1u, 3u, 0u, 3u, 2u);   // TL TR BR | TL BR BL
  let c = corner[vi];
  var o: ClothOut;
  // a triangle exists only while both of its grid edges are intact (that is how tears open up)
  var ok = true;
  if (vi < 3u) { ok = cons[q.edges.x].alive > 0.5 && cons[q.edges.w].alive > 0.5; }
  else { ok = cons[q.edges.z].alive > 0.5 && cons[q.edges.y].alive > 0.5; }
  if (!ok) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  var idx = q.pts.x;
  if (c == 1u) { idx = q.pts.y; }
  if (c == 2u) { idx = q.pts.z; }
  if (c == 3u) { idx = q.pts.w; }
  let m = pmeta[idx];
  let p = pts[idx].p;
  o.pos = toClip(p.xy);
  o.uv = vec2f(f32(m.z) / q.info.x, f32(m.w) / q.info.y);
  o.n = vnormal(idx);
  o.style = q.info.z;
  o.w = p.xy;
  return o;
}
@fragment fn fs_cloth(i: ClothOut, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  var n = normalize(i.n);
  var back = 1.0;
  if (n.z < 0.0) { n = -n; back = 0.75; }
  let L = normalize(LD);
  let diff = max(dot(n, L), 0.0);
  let H = normalize(L + vec3f(0.0, 0.0, 1.0));
  var base = vec3f(0.0);
  if (i.style < 1.5) {
    // a heraldic banner
    let uv = i.uv;
    base = vec3f(0.1, 0.22, 0.55);
    base = mix(base, vec3f(0.95, 0.75, 0.2), step(abs(uv.y - 0.5), 0.12));
    base = mix(base, vec3f(0.75, 0.1, 0.16), step(abs(uv.y - 0.5), 0.07));
    let star = sdStar5(vec2f((uv.x - 0.24) * 1.6, uv.y - 0.5), 0.2, 0.45);
    base = mix(base, vec3f(0.98, 0.95, 0.85), smoothstep(0.01, -0.01, star));
    base = mix(base, vec3f(0.95, 0.75, 0.2), smoothstep(0.012, 0.0, abs(star) - 0.012));
    base *= 0.92 + 0.08 * sin(uv.x * 400.0) * sin(uv.y * 400.0);    // weave
  } else {
    // velvet curtain with a golden hem
    let uv = i.uv;
    base = mix(vec3f(0.42, 0.04, 0.08), vec3f(0.62, 0.08, 0.12), 0.5 + 0.5 * sin(uv.x * 60.0));
    base = mix(base, vec3f(0.85, 0.62, 0.2), step(0.94, uv.y));
    base = mix(base, vec3f(0.6, 0.4, 0.1), step(0.965, uv.y) * step(0.5, fract(uv.x * 40.0)));
  }
  let sheen = pow(1.0 - n.z, 2.0) * 0.35;
  var col = base * (0.3 + 0.8 * diff) * back + vec3f(1.0, 0.95, 0.9) * pow(max(dot(n, H), 0.0), 30.0) * 0.12 + base * sheen;
  return vec4f(col, 1.0);
}

// ---------------------------------------------------------------- constraints (ropes, chains, planks)
struct SegOut { @builtin(position) pos: vec4f, @location(0) lp: vec2f, @location(1) @interpolate(flat) info: vec4f };
fn segWidth(st: u32) -> f32 {
  if (st == 1u) { return 4.5; }
  if (st == 2u) { return 7.0; }
  if (st == 3u) { return 9.0; }
  if (st == 4u) { return 2.2; }
  return 1.2;
}
@vertex fn vs_seg(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> SegOut {
  var o: SegOut;
  let c = cons[ci];
  var st = u32(c.style) % 16u;
  if (u.debug > 0.5) { st = 7u; }
  if (c.alive < 0.5 || st == 0u || st == 5u || st == 6u) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  let a = pts[c.a].p.xy;
  let b = pts[c.b].p.xy;
  var dir = b - a;
  let len = max(length(dir), 0.0001);
  dir /= len;
  let perp = vec2f(-dir.y, dir.x);
  let hw = segWidth(st) + 2.0;
  var corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let cc = corners[vi];
  let along = mix(-hw, len + hw, cc.x);
  let w = a + dir * along + perp * cc.y * hw;
  o.pos = toClip(w);
  o.lp = vec2f(along, cc.y * hw);
  o.info = vec4f(len, f32(st), f32(ci), floor(c.style / 16.0));
  return o;
}
@fragment fn fs_seg(i: SegOut) -> @location(0) vec4f {
  let len = i.info.x;
  let st = u32(i.info.y);
  let t = i.lp.x;
  let s = i.lp.y;
  let hw = segWidth(st);
  if (st == 7u) {
    // debug: every constraint, colored by its graph color (= the batch it is solved in)
    let d = length(vec2f(t - clamp(t, 0.0, len), s)) - 0.9;
    let col = 0.55 + 0.45 * cos(6.2831853 * (i.info.w * 0.13 + vec3f(0.0, 0.33, 0.67)));
    return vec4f(col, aa(d));
  }
  if (st == 3u) {
    // plank: a wooden board
    let q = vec2f(t - len * 0.5, s);
    let d = sdBoxB(q, vec2f(len * 0.5 - 1.5, hw)) - 1.5;
    let grain = 0.5 + 0.5 * sin(t * 0.25 + sin(s * 0.9 + i.info.z) * 2.0);
    var c = mix(vec3f(0.45, 0.28, 0.14), vec3f(0.62, 0.42, 0.24), grain * 0.6 + 0.2 * hash1(i.info.z));
    c *= 0.75 + 0.35 * smoothstep(hw, -hw, s);
    c = mix(c, vec3f(0.2, 0.2, 0.22), smoothstep(2.6, 1.6, length(vec2f(abs(q.x) - len * 0.5 + 5.0, q.y))));
    c = mix(vec3f(0.15, 0.08, 0.04), c, smoothstep(0.0, -1.5, d));
    return vec4f(c, aa(d));
  }
  if (st == 2u) {
    // chain link: alternate "face-on" rings and "edge-on" bars
    let q = vec2f(t - len * 0.5, s);
    var d = 0.0;
    if ((u32(i.info.z) % 2u) == 0u) {
      let e = vec2f(len * 0.5 + 4.0, hw);
      let k = length(q / e);
      d = (abs(k - 0.72) - 0.2) * min(e.x, e.y);
    } else {
      d = sdBoxB(q, vec2f(len * 0.5 + 4.0, 2.4)) - 1.6;
    }
    let lit = 0.5 + 0.5 * cos(s / hw * 2.6 + 0.6);
    let c = mix(vec3f(0.22, 0.23, 0.27), vec3f(0.85, 0.87, 0.92), lit);
    return vec4f(c, aa(d));
  }
  // ropes & strings: twisted fibres
  let d = length(vec2f(t - clamp(t, 0.0, len), s)) - hw;
  let twist = 0.5 + 0.5 * sin((t + s * 1.6) * 0.9);
  var c = mix(vec3f(0.5, 0.36, 0.2), vec3f(0.8, 0.64, 0.4), twist);
  if (st == 4u) { c = mix(vec3f(0.55, 0.42, 0.26), vec3f(0.75, 0.6, 0.4), twist); }
  c *= 0.65 + 0.45 * smoothstep(hw, -hw, s);
  return vec4f(c, aa(d));
}

// ---------------------------------------------------------------- points (pins, lantern, wrecking ball)
struct PtOut { @builtin(position) pos: vec4f, @location(0) lp: vec2f, @location(1) @interpolate(flat) info: vec4f };
@vertex fn vs_pt(@builtin(vertex_index) vi: u32, @builtin(instance_index) pi: u32) -> PtOut {
  var o: PtOut;
  let m = pmeta[pi];
  let pt = pts[pi];
  var st = m.y;
  let grabbed = pi == grabbedIndex();
  if (u.debug > 0.5 && st == 0) { st = 9; }
  if (grabbed) { st = max(st, 1); }
  if (st == 0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  var r = max(pt.o.w, 3.0);
  if (st == 9) { r = 2.2; }
  var ext = r + 3.0;
  if (st == 2) { ext = r * 4.0; }          // lantern glow
  if (grabbed) { ext = max(ext, r + 14.0); }
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let cc = corners[vi] * ext;
  o.pos = toClip(pt.p.xy + cc);
  o.lp = cc;
  o.info = vec4f(r, f32(st), select(0.0, 1.0, grabbed), f32(pi));
  return o;
}
@fragment fn fs_pt(i: PtOut) -> @location(0) vec4f {
  let r = i.info.x;
  let st = i32(i.info.y);
  let d = length(i.lp) - r;
  var col = vec4f(0.0);
  let n3 = normalize(vec3f(i.lp / r, sqrt(max(1.0 - dot(i.lp, i.lp) / (r * r), 0.0)) + 0.001));
  let diff = max(dot(n3, normalize(LD)), 0.0);
  let spec = pow(max(dot(n3, normalize(normalize(LD) + vec3f(0.0, 0.0, 1.0))), 0.0), 40.0);
  if (st == 2) {
    // lantern: warm glowing glass in an iron frame
    let flick = 0.9 + 0.1 * sin(u.time * 13.0) * sin(u.time * 7.3);
    let glow = exp(-max(d, 0.0) / (r * 0.9)) * 0.55 * flick;
    col = vec4f(vec3f(1.0, 0.7, 0.3), glow * (1.0 - aa(d)));
    var body = mix(vec3f(1.0, 0.75, 0.35), vec3f(1.0, 0.97, 0.8), exp(-length(i.lp) / r * 2.0));
    let bars = smoothstep(0.85, 0.95, abs(sin(i.lp.x / r * 3.1)));
    body = mix(body, vec3f(0.15, 0.12, 0.1), bars * 0.8);
    col = mix(col, vec4f(body, 1.0), aa(d));
    let cap = sdBoxB(i.lp - vec2f(0.0, -r * 1.05), vec2f(r * 0.55, r * 0.22));
    col = mix(col, vec4f(0.18, 0.15, 0.13, 1.0), aa(cap));
  } else if (st == 3) {
    // wrecking ball: dark iron
    var c = vec3f(0.16, 0.16, 0.19) * (0.35 + 0.9 * diff) + vec3f(1.0) * spec * 0.6;
    c += vec3f(0.6, 0.45, 0.4) * pow(1.0 - n3.z, 3.0) * 0.4;
    col = vec4f(c, aa(d));
  } else if (st == 4) {
    // pin / bolt
    var c = vec3f(0.55, 0.56, 0.6) * (0.5 + 0.6 * diff) + vec3f(1.0) * spec * 0.4;
    c = mix(c, vec3f(0.2), 1.0 - smoothstep(r * 0.25, r * 0.4, length(i.lp)));
    col = vec4f(c, aa(d));
  } else if (st == 5) {
    // curtain ring
    let dr = abs(length(i.lp) - r * 0.75) - r * 0.25;
    col = vec4f(mix(vec3f(0.7, 0.5, 0.15), vec3f(1.0, 0.88, 0.5), diff), aa(dr));
  } else if (st == 9) {
    col = vec4f(1.0, 1.0, 1.0, aa(d) * 0.9);
  } else {
    col = vec4f(vec3f(0.85, 0.85, 0.9) * (0.5 + 0.6 * diff), aa(d));
  }
  if (i.info.z > 0.5) {
    // grabbed: a highlight ring
    let ring = abs(length(i.lp) - r - 7.0) - 1.8;
    col = mix(col, vec4f(1.0, 1.0, 1.0, 1.0), aa(ring) * 0.9);
  }
  return col;
}

// ---------------------------------------------------------------- soft bodies (triangle fan per blob)
struct BlobOut { @builtin(position) pos: vec4f, @location(0) w: vec2f, @location(1) rim: f32, @location(2) @interpolate(flat) info: vec4f, @location(3) @interpolate(flat) frame: vec4f };
@vertex fn vs_blob(@builtin(vertex_index) vi: u32, @builtin(instance_index) bi: u32) -> BlobOut {
  var o: BlobOut;
  let bl = blobs[bi];
  let tri = vi / 3u;
  let corner = vi % 3u;
  if (tri >= bl.count) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  var w = bl.c;
  var rim = 0.0;
  if (corner == 1u) { w = pts[bl.first + tri].p.xy; rim = 1.0; }
  if (corner == 2u) { w = pts[bl.first + (tri + 1u) % bl.count].p.xy; rim = 1.0; }
  o.pos = toClip(w);
  o.w = w;
  o.rim = rim;
  // orientation: where ring point 0 (which started at the top) is now
  let top = pts[bl.first].p.xy - bl.c;
  o.frame = vec4f(bl.c, normalize(top + vec2f(0.0, -0.0001)));
  o.info = vec4f(sqrt(bl.restArea / 3.14159), bl.style, bl.area / bl.restArea, f32(bi));
  return o;
}
@fragment fn fs_blob(i: BlobOut) -> @location(0) vec4f {
  let R = i.info.x;
  let hue = i.info.y;
  let base = 0.55 + 0.45 * cos(6.2831853 * (hue + vec3f(0.0, 0.33, 0.67)));
  let rr = clamp(i.rim, 0.0, 1.0);
  let dir = normalize(i.w - i.frame.xy + vec2f(0.0001, 0.0));
  let n = normalize(vec3f(dir * rr, sqrt(max(1.0 - rr * rr, 0.0)) * 0.8 + 0.05));
  let L = normalize(LD);
  let diff = max(dot(n, L), 0.0);
  let spec = pow(max(dot(n, normalize(L + vec3f(0.0, 0.0, 1.0))), 0.0), 50.0);
  var col = base * (0.45 + 0.65 * diff) + vec3f(1.0) * spec * 0.7;
  col = mix(col, base * 0.55, smoothstep(0.75, 1.0, rr) * 0.6);                     // darker jelly edge
  col += base * (1.0 - rr) * 0.15;                                                  // inner glow
  // a face, rotating with the body
  let up = -i.frame.zw;
  let rt = vec2f(-up.y, up.x);
  let lp = vec2f(dot(i.w - i.frame.xy, rt), dot(i.w - i.frame.xy, up)) / R;        // local, x right, y up
  for (var k = 0; k < 2; k++) {
    let ec = vec2f((f32(k) * 2.0 - 1.0) * 0.32, 0.12);
    let ed = length(lp - ec) - 0.17;
    col = mix(col, vec3f(1.0), smoothstep(0.02, -0.02, ed));
    col = mix(col, vec3f(0.08, 0.06, 0.12), smoothstep(0.02, -0.02, length(lp - ec - vec2f(0.0, -0.04)) - 0.08));
    col = mix(col, vec3f(1.0), smoothstep(0.02, -0.02, length(lp - ec - vec2f(-0.03, 0.0)) - 0.025));
  }
  let squash = clamp(1.0 - i.info.z, -0.3, 0.3);
  let mq = lp - vec2f(0.0, -0.1);
  let mouth = max(abs(length(mq) - 0.12 - squash * 0.2) - 0.025, mq.y);
  col = mix(col, vec3f(0.12, 0.05, 0.1), smoothstep(0.02, -0.02, mouth));
  return vec4f(col, 1.0);
}
`;

// ------------------------------------------------------------------------------------------ world building
class Builder {
  constructor() {
    this.pts = []; // [x, y, z, invMass, radius]
    this.meta = []; // [kind, style, gi, gj]
    this.nbr = []; // [l, r, u, d]
    this.cons = []; // {a, b, rest, stiff, style, tear, cut}
    this.blobs = []; // {first, count, restArea, style}
    this.quads = []; // {pts:[tl,tr,bl,br], edges:[top,bottom,left,right], info:[gw-1, gh-1, style, 0]}
    this.obs = [];
    this.layout = [0, 0, 0, 0];
  }
  point(x, y, invMass = 1, radius = 3, style = 0, kind = 0, z = 0) {
    this.pts.push([x, y, z, invMass, radius]);
    this.meta.push([kind, style, 0, 0]);
    this.nbr.push([-1, -1, -1, -1]);
    return this.pts.length - 1;
  }
  link(a, b, { stiff = 1, style = 0, tear = 0, cut = 1, rest } = {}) {
    const pa = this.pts[a];
    const pb = this.pts[b];
    const r = rest ?? Math.hypot(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
    this.cons.push({ a, b, rest: r, stiff, style, tear, cut });
    return this.cons.length - 1;
  }
  chain(x0, y0, x1, y1, n, opts = {}) {
    const ids = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      ids.push(this.point(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 1, opts.radius ?? 4, 0));
    }
    for (let i = 0; i < n - 1; i++) this.link(ids[i], ids[i + 1], { style: opts.style ?? CS.rope, rest: opts.rest, cut: 1 });
    return ids;
  }
  cloth(x0, y0, gw, gh, sp, { style = 1, pinned = () => false, tear = 0, zf = () => 0 } = {}) {
    const id = (i, j) => base + j * gw + i;
    const base = this.pts.length;
    for (let j = 0; j < gh; j++)
      for (let i = 0; i < gw; i++) {
        const kind = style === 1 ? KIND.cloth : KIND.curtain;
        const p = this.point(x0 + i * sp, y0 + j * sp, pinned(i, j) ? 0 : 1, 2, 0, kind, zf(i, j));
        this.meta[p] = [kind, 0, i, j];
        this.nbr[p] = [i > 0 ? id(i - 1, j) : -1, i < gw - 1 ? id(i + 1, j) : -1, j > 0 ? id(i, j - 1) : -1, j < gh - 1 ? id(i, j + 1) : -1];
      }
    const H = [];
    const V = [];
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw - 1; i++) H[j * gw + i] = this.link(id(i, j), id(i + 1, j), { style: CS.cloth, tear });
    for (let j = 0; j < gh - 1; j++) for (let i = 0; i < gw; i++) V[j * gw + i] = this.link(id(i, j), id(i, j + 1), { style: CS.cloth, tear });
    // shear springs keep the cloth from collapsing like a net
    for (let j = 0; j < gh - 1; j++)
      for (let i = 0; i < gw - 1; i++) {
        this.link(id(i, j), id(i + 1, j + 1), { style: CS.cloth, stiff: 0.6, tear: tear ? tear * 1.4 : 0 });
        this.link(id(i + 1, j), id(i, j + 1), { style: CS.cloth, stiff: 0.6, tear: tear ? tear * 1.4 : 0 });
      }
    for (let j = 0; j < gh - 1; j++)
      for (let i = 0; i < gw - 1; i++)
        this.quads.push({
          pts: [id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1)],
          edges: [H[j * gw + i], H[(j + 1) * gw + i], V[j * gw + i], V[j * gw + i + 1]],
          info: [gw - 1, gh - 1, style, 0],
        });
  }
  blob(cx, cy, R, n, hue) {
    const first = this.pts.length;
    let area = 0;
    for (let k = 0; k < n; k++) {
      const a = -Math.PI / 2 + (k / n) * Math.PI * 2;
      this.point(cx + Math.cos(a) * R, cy + Math.sin(a) * R, 1, 3, 0, KIND.blob);
    }
    for (let k = 0; k < n; k++) {
      const p0 = this.pts[first + k];
      const p1 = this.pts[first + ((k + 1) % n)];
      area += p0[0] * p1[1] - p1[0] * p0[1];
      this.nbr[first + k] = [first + ((k - 1 + n) % n), first + ((k + 1) % n), this.blobs.length, -1];
    }
    for (let k = 0; k < n; k++) this.link(first + k, first + ((k + 1) % n), { style: CS.jelly, cut: 0 });
    for (let k = 0; k < n; k++) this.link(first + k, first + ((k + 2) % n), { style: CS.jelly, stiff: 0.35, cut: 0 });
    this.blobs.push({ first, count: n, restArea: Math.abs(area) / 2, style: hue });
  }
}

function buildWorld(example, W, test) {
  const b = new Builder();
  if (example === 'ropes') {
    // rope with a lantern
    const r0 = b.point(W * 0.13, 84, 0, 7, PS.pin);
    const n = 24;
    let prev = r0;
    for (let i = 1; i <= n; i++) {
      const last = i === n;
      const p = b.point(W * 0.13 + i * 13.5 * 0.8, 84 + i * 13.5 * 0.6, last ? 0.35 : 1, last ? 20 : 4, last ? PS.lantern : 0);
      b.link(prev, p, { style: CS.rope });
      prev = p;
    }
    // chain with a wrecking ball
    const c0 = b.point(W * 0.86, 84, 0, 7, PS.pin);
    prev = c0;
    for (let i = 1; i <= 15; i++) {
      const last = i === 15;
      const p = b.point(W * 0.86 - i * 22 * 0.35, 84 + i * 22 * 0.94, last ? 0.07 : 0.8, last ? 42 : 5, last ? PS.ball : 0);
      b.link(prev, p, { style: CS.chain, rest: last ? 22 + 30 : 22 });
      prev = p;
    }
    // rope bridge between two cliffs
    const xL = W * 0.3;
    const xR = W * 0.7;
    const top = 690;
    const N = 15;
    const seg = ((xR - xL) / (N - 1)) * 1.05;
    const deck = [];
    const rail = [];
    for (let i = 0; i < N; i++) {
      const x = xL + ((xR - xL) * i) / (N - 1);
      const pinned = i === 0 || i === N - 1;
      deck.push(b.point(x, top - 8, pinned ? 0 : 1, 6, pinned ? PS.pin : 0));
      rail.push(b.point(x, top - 92, pinned ? 0 : 1, 3, pinned ? PS.pin : 0));
    }
    for (let i = 0; i < N - 1; i++) {
      b.link(deck[i], deck[i + 1], { style: CS.plank, rest: seg });
      b.link(rail[i], rail[i + 1], { style: CS.string, rest: seg });
    }
    for (let i = 1; i < N - 1; i++) b.link(deck[i], rail[i], { style: CS.string, rest: 84 });
    b.obs.push([xL / 2, top + 500, xL / 2, 500]);
    b.obs.push([(W + xR) / 2, top + 500, (W - xR) / 2, 500]);
    b.layout = [xL, xR, top, 0];
    b.floorY = 1400;
  } else if (example === 'cloth') {
    const s = test ? 1.8 : 1;
    // flag on a pole: pinned along the left edge
    const fx = W * 0.09;
    const fgw = Math.round(34 / s);
    const fgh = Math.round(21 / s);
    b.cloth(fx, 150, fgw, fgh, 14 * s, { style: 1, pinned: (i) => i === 0, zf: (i) => 10 * Math.sin(i * 0.45 * s) });
    // curtain: pinned at its rings on the rail, tearable
    const cgw = Math.round(36 / s);
    const cgh = Math.round(45 / s);
    const csp = 13.5 * s;
    const cw = (cgw - 1) * csp;
    const cx0 = Math.max(W * 0.5, W * 0.72 - cw / 2);
    // pleats: the curtain starts folded in depth (rest lengths are measured in 3D, so the folds persist)
    b.cloth(cx0, 128, cgw, cgh, csp, { style: 2, pinned: (i, j) => j === 0 && i % 3 === 0, tear: 1, zf: (i) => 9 * Math.sin((i * Math.PI) / 3) });
    // curtain rings
    for (let i = 0; i < b.pts.length; i++)
      if (b.meta[i][0] === KIND.curtain && b.meta[i][3] === 0 && b.pts[i][3] === 0) {
        b.meta[i][1] = PS.ring;
        b.pts[i][4] = 7;
      }
    b.layout = [cx0 - 90, cx0 + cw / 2, cw / 2 - 10, fx];
    b.floorY = 990;
  } else {
    // jelly: soft bodies with pressure, dropped on candy pegs
    const n = test ? 16 : 22;
    const hues = [0.0, 0.15, 0.35, 0.55, 0.72, 0.85, 0.25, 0.62];
    const count = test ? 6 : 10;
    for (let k = 0; k < count; k++) {
      const R = 46 + ((k * 37) % 28);
      b.blob(W * (0.1 + 0.8 * ((k * 0.37) % 1)), 110 + (k % 3) * 140 + ((k * 13) % 40), R, n, hues[k % hues.length]);
    }
    b.obs.push([W * 0.28, 560, 55, -1]);
    b.obs.push([W * 0.6, 470, 68, -1]);
    b.obs.push([W * 0.84, 640, 45, -1]);
    b.floorY = 860;
  }
  b.floorY ??= 990;
  return b;
}

// greedy graph coloring of constraints: two constraints sharing a point get different colors
function colorConstraints(cons, nPts) {
  const used = new Uint32Array(nPts);
  const color = new Int32Array(cons.length);
  let maxC = 0;
  cons.forEach((c, i) => {
    const m = used[c.a] | used[c.b];
    let k = 0;
    while (m & (1 << k)) k++;
    color[i] = k;
    used[c.a] |= 1 << k;
    used[c.b] |= 1 << k;
    maxC = Math.max(maxC, k + 1);
  });
  const order = [];
  const ranges = [];
  const remap = new Int32Array(cons.length);
  for (let k = 0; k < maxC; k++) {
    const first = order.length;
    cons.forEach((c, i) => {
      if (color[i] === k) {
        remap[i] = order.length;
        order.push({ ...c, color: k });
      }
    });
    ranges.push([first, order.length - first]);
  }
  return { order, ranges, remap };
}

// ------------------------------------------------------------------------------------------ scene
const examples = [
  {
    id: 'ropes',
    label: 'Ropes, chains & a bridge',
    kind: 'In a game',
    note: 'A rope with a lantern, a chain with a wrecking ball, and a rope bridge whose planks are just links between points. Grab anything and swing it. <b>Right-drag</b> (or turn on “Left button cuts”) to cut.',
  },
  {
    id: 'cloth',
    label: 'Cloth: flag & curtain',
    kind: 'In a game',
    note: 'A grid of points joined by links. The flag catches the wind (a force along each point’s normal); the curtain hangs from rings and <b>tears</b> when a link is stretched too far — pull it hard, or right-drag to slash it.',
  },
  {
    id: 'jelly',
    label: 'Soft bodies / jelly',
    kind: 'Abstract',
    note: 'Each jelly is a ring of points with links around the edge plus <b>gas pressure</b>: a force pushing outward that grows when the ring’s area shrinks. Grab one and fling it at the others.',
  },
];

const controls = [
  { type: 'heading', label: 'Solver' },
  { type: 'slider', key: 'substeps', label: 'Sub-steps per frame', min: 1, max: 8, step: 1, value: 3, help: 'Smaller time steps = stiffer, more stable constraints.' },
  { type: 'slider', key: 'iterations', label: 'Constraint iterations', min: 1, max: 30, step: 1, value: 8, help: 'Relaxation passes per sub-step. Few = stretchy rubber, many = stiff rope.' },
  { type: 'slider', key: 'stiffness', label: 'Stiffness', min: 0.05, max: 1, step: 0.01, value: 1, help: 'Fraction of each constraint’s error fixed per pass.' },
  { type: 'slider', key: 'gravity', label: 'Gravity', min: -1, max: 3, step: 0.01, value: 1 },
  { type: 'slider', key: 'damping', label: 'Air damping', min: 0.9, max: 1, step: 0.001, value: 0.995, help: 'Velocity kept per 1/60 s.' },
  { type: 'heading', label: 'Forces & breaking' },
  { type: 'slider', key: 'wind', label: 'Wind', min: 0, max: 1500, step: 10, value: 850, showFor: ['cloth'] },
  { type: 'slider', key: 'tear', label: 'Tear at stretch ×', min: 1.2, max: 5, step: 0.05, value: 1.9, showFor: ['cloth'], help: 'A curtain link breaks when it gets longer than this × its rest length.' },
  { type: 'slider', key: 'pressure', label: 'Pressure', min: 0, max: 6, step: 0.05, value: 3, showFor: ['jelly'], help: 'Gas pressure inside the jellies. 0 = deflated bags.' },
  { type: 'toggle', key: 'cutMode', label: 'Left button cuts', value: false, help: 'Same as right-dragging: slice through ropes and cloth.' },
  { type: 'toggle', key: 'debug', label: 'Show points & constraint colors', value: false, help: 'Every link, colored by its batch (graph color). Same-colored links never share a point, so they are solved in parallel.' },
  { type: 'button', key: 'reset', label: 'Reset', primary: true },
];

export default {
  interaction: 'Drag to grab & swing. Right-drag to cut. (Reset with ↻)',
  examples,
  controls,
  reinitOnExample: true,
  about: {
    summary:
      'Verlet physics models everything as <b>points</b> connected by <b>distance constraints</b>. Ropes, chains, bridges, cloth and jelly all come from the same two ingredients — here solved entirely on the GPU with compute shaders.',
    what: `<p>Thousands of points, each knowing only its current and previous position, plus thousands of “stay this far apart” links.
      Every frame a chain of compute passes moves the points, then nudges every link back toward its rest length, over and over.</p>`,
    how: `<ol>
      <li><b>Verlet integration</b>: velocity isn’t stored — it’s <code>x − x<sub>prev</sub></code>. New position =
        <code>x + (x − x<sub>prev</sub>)·damping + a·dt²</code>. Because of this, simply <i>moving</i> a point also changes its velocity correctly.</li>
      <li><b>Constraints</b>: for each link, measure its length, and move both ends along the link so it returns to its rest length
        (heavier / pinned points move less). Repeating this a few times per step makes the whole structure consistent (Jakobsen, Hitman 2001).</li>
      <li><b>Parallel solving</b>: two GPU threads must never move the same point at once. The links are <b>graph-colored</b> on the CPU:
        links of one color share no points, so a color is one parallel batch (one compute dispatch). Turn on “Show constraint colors”.</li>
      <li><b>Sub-steps</b>: several small steps per frame beat many iterations of one big step — stiffer ropes for the same cost.</li>
      <li><b>Tearing</b>: a link longer than <i>k ×</i> rest length is switched off. <b>Cutting</b>: links crossing the mouse’s path are switched off.
        Cloth triangles only draw while their edges are alive, so holes open up.</li>
      <li><b>Wind</b> pushes each cloth point along its normal (computed from its grid neighbours) by how much the wind blows into it.
        <b>Pressure</b> pushes jelly edges outward ∝ (rest area ÷ current area − 1) — the shoelace formula gives the area.</li>
      <li><b>Picking</b>: on click, every point checks its distance to the mouse and an <code>atomicMin</code> on (distance, index) picks the nearest — no CPU read-back.</li>
    </ol>`,
    uses: [
      { title: 'Ropes & grappling hooks', text: 'Cut the Rope, Worms ninja rope, swinging lanterns, hanging bridges, chain physics.' },
      { title: 'Cloth', text: 'Capes, flags, banners, curtains and sails that react to wind and characters — tearable for destruction.' },
      { title: 'Soft bodies', text: 'Jelly enemies, squishy characters (Gish, JellyCar), bouncy UI, tires.' },
      { title: 'Ragdolls', text: 'A ragdoll is points (joints) + links (bones) + angle limits — the same solver.' },
    ],
    try: [
      'Set <i>Constraint iterations</i> to 1 and drag the chain: it turns into a rubber band. Then raise <i>Sub-steps</i> instead.',
      'Pull the curtain hard to the side until it rips, or right-drag across it.',
      'Turn on <b>Show points & constraint colors</b> on the cloth: the colors are the parallel batches.',
      'Set <i>Pressure</i> to 0 for deflated jelly bags, or 6 for bouncy balls.',
      'Flip <i>Gravity</i> negative and watch everything fall upward.',
    ],
    ask: [
      'Verlet rope physics with a grabbable end',
      'tearable cloth simulation on the GPU',
      'a rope bridge the player can walk on',
      'pressure-based soft body (jelly) physics',
      'graph-colored parallel constraint solver in a compute shader',
      'flag waving in the wind',
    ],
    perf: `<p>Cost ≈ (sub-steps × iterations × colors) dispatches per frame, each touching every link once. The cloth example has ~10k links in ~8 colors:
      3 × 8 × 8 ≈ 200 tiny dispatches per frame — trivial for the GPU; the per-dispatch overhead dominates. A CPU (JavaScript) solver handles a few
      thousand links comfortably; the GPU version scales to hundreds of thousands (big cloths, hair, many ropes).</p>`,
    api: `<p><b>WebGPU only.</b> Constraint solving needs scattered writes to arbitrary points (storage buffers written from compute shaders) and
      atomics for picking. WebGL2 has neither, so a WebGL2 version would run the solver in JavaScript and upload positions every frame.</p>`,
    code: [
      {
        title: 'Verlet integration (one thread per point)',
        lang: 'wgsl',
        src: `let pos = pt.p.xyz;
let vel = (pos - pt.o.xyz) * u.damping;          // implicit velocity
let acc = vec3f(u.gravity, 0.0) + frc[i].xyz;
pt.o = vec4f(pos, pt.o.w);                       // remember where we were
pt.p = vec4f(pos + vel + acc * u.dt * u.dt, pt.p.w);`,
      },
      {
        title: 'Solve one color of constraints (one thread per link)',
        lang: 'wgsl',
        src: `let c = cons[rg.first + k];
let A = pts[c.a].p;  let B = pts[c.b].p;        // w = inverse mass (0 = pinned)
let d = B.xyz - A.xyz;
let len = length(d);
if (c.tear > 0.0 && len > c.rest * c.tear * u.tearMul) { cons[ci].alive = 0.0; return; }
let w = A.w + B.w;
let corr = d * ((len - c.rest) / len) * c.stiff;
pts[c.a].p = vec4f(A.xyz + corr * (A.w / w), A.w);  // safe: no other thread in this
pts[c.b].p = vec4f(B.xyz - corr * (B.w / w), B.w);  // color touches these points`,
      },
      {
        title: 'Graph coloring on the CPU (once)',
        lang: 'js',
        src: `cons.forEach((c, i) => {
  const taken = used[c.a] | used[c.b];      // colors already used at either end
  let k = 0; while (taken & (1 << k)) k++;   // lowest free color
  color[i] = k;
  used[c.a] |= 1 << k;  used[c.b] |= 1 << k;
});`,
      },
    ],
    links: [
      { title: 'Thomas Jakobsen — Advanced Character Physics', url: 'https://www.cs.cmu.edu/afs/cs/academic/class/15462-s13/www/lec_slides/Jakobsen.pdf', note: 'the classic Verlet + constraints paper (Hitman)' },
      { title: 'Matthias Müller — Ten Minute Physics', url: 'https://matthias-research.github.io/pages/tenMinutePhysics/', note: 'PBD, sub-stepping, cloth & soft bodies' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const test = ctx.testMode;
    const U = gpu.uniforms(
      {
        gravity: 'vec2f', dt: 'f32', damping: 'f32',
        mouse: 'vec2f', pmouse: 'vec2f',
        grab: 'f32', cutting: 'f32', time: 'f32', nPts: 'u32',
        wind: 'vec4f',
        nBlobs: 'u32', pressure: 'f32', tearMul: 'f32', floorY: 'f32',
        worldW: 'f32', friction: 'f32', nObs: 'u32', pickR: 'f32',
        debug: 'f32', example: 'f32', nCons: 'u32', stiffMul: 'f32',
        scene: 'vec4f',
        obs: 'array<vec4f, 8>',
      },
      'Sim',
    );
    const subs = Array.from({ length: MAX_SUB }, () => gpu.uniforms({ frac: 'f32' }, 'Sub'));
    const RANGE_FIELDS = { first: 'u32', count: 'u32' };

    const simBindings = (rangeBlock, subBlock) => ({
      u: { uniform: U },
      rg: { uniform: rangeBlock },
      sub: { uniform: subBlock },
      pts: { storage: 'array<Pt>', access: 'read_write' },
      cons: { storage: 'array<Con>', access: 'read_write' },
      blobs: { storage: 'array<Blob>', access: 'read_write' },
      frc: { storage: 'array<vec4f>', access: 'read_write' },
      pmeta: { storage: 'array<vec4i>', access: 'read' },
      nbr: { storage: 'array<vec4i>', access: 'read' },
      pick: { storage: 'array<atomic<u32>>', access: 'read_write' },
    });

    let W = 1000 * (ctx.width / ctx.height);
    let world = null;
    let sim = null;
    let draw = null;
    let pipes = null;
    let bgs = null;
    let drawBG = null;
    let msaa = null;
    let pickBuf = null;
    const readout = overlayTag(ctx);

    const build = () => {
      W = 1000 * (ctx.width / ctx.height);
      const b = buildWorld(ctx.example, W, test);
      const nPts = b.pts.length;
      const { order, ranges, remap } = colorConstraints(b.cons, nPts);
      // GPU data
      const P = new Float32Array(nPts * 8);
      b.pts.forEach((p, i) => {
        P.set([p[0], p[1], p[2], p[3], p[0], p[1], p[2], p[4]], i * 8);
      });
      const M = new Int32Array(b.meta.flat());
      const NB = new Int32Array(b.nbr.flat());
      const nCons = Math.max(1, order.length);
      const CB = new ArrayBuffer(nCons * 32);
      const cu = new Uint32Array(CB);
      const cf = new Float32Array(CB);
      order.forEach((c, i) => {
        cu[i * 8] = c.a;
        cu[i * 8 + 1] = c.b;
        cf[i * 8 + 2] = c.rest;
        cf[i * 8 + 3] = c.stiff;
        cf[i * 8 + 4] = c.style + 16 * c.color;
        cf[i * 8 + 5] = c.tear;
        cf[i * 8 + 6] = 1;
        cf[i * 8 + 7] = c.cut;
      });
      const nBlobs = b.blobs.length;
      const BB = new ArrayBuffer(Math.max(1, nBlobs) * 32);
      const bu = new Uint32Array(BB);
      const bf = new Float32Array(BB);
      b.blobs.forEach((bl, i) => {
        bu[i * 8] = bl.first;
        bu[i * 8 + 1] = bl.count;
        bf[i * 8 + 2] = bl.restArea;
        bf[i * 8 + 3] = bl.style;
        bf[i * 8 + 6] = bl.restArea;
      });
      const nQuads = b.quads.length;
      const QB = new ArrayBuffer(Math.max(1, nQuads) * 48);
      const qu = new Uint32Array(QB);
      const qf = new Float32Array(QB);
      b.quads.forEach((q, i) => {
        qu.set(q.pts, i * 12);
        qu.set(q.edges.map((e) => remap[e]), i * 12 + 4);
        qf.set(q.info, i * 12 + 8);
      });
      const bufs = {
        pts: gpu.storage(P, 'verlet-pts'),
        cons: gpu.storage(new Uint8Array(CB), 'verlet-cons'),
        blobs: gpu.storage(new Uint8Array(BB), 'verlet-blobs'),
        frc: gpu.storage(nPts * 16, 'verlet-forces'),
        meta: gpu.storage(M, 'verlet-meta'),
        nbr: gpu.storage(NB, 'verlet-nbr'),
        quads: gpu.storage(new Uint8Array(QB), 'verlet-quads'),
      };
      pickBuf = gpu.storage(new Uint32Array([0xffffffff, 0, 0, 0]), 'verlet-pick');
      const rangeBlocks = ranges.map(([first, count]) => {
        const r = gpu.uniforms(RANGE_FIELDS, 'Range');
        r.set('first', first).set('count', count).upload();
        return r;
      });
      if (!rangeBlocks.length) {
        const r = gpu.uniforms(RANGE_FIELDS, 'Range');
        r.set('first', 0).set('count', 0).upload();
        rangeBlocks.push(r);
      }
      if (!sim) {
        sim = gpu.compute({ label: 'verlet-sim', bindings: simBindings(rangeBlocks[0], subs[0]), code: SIM_WGSL });
        draw = gpu.program({
          label: 'verlet-draw',
          bindings: {
            u: { uniform: U },
            pts: { storage: 'array<Pt>', access: 'read' },
            cons: { storage: 'array<Con>', access: 'read' },
            blobs: { storage: 'array<Blob>', access: 'read' },
            pmeta: { storage: 'array<vec4i>', access: 'read' },
            nbr: { storage: 'array<vec4i>', access: 'read' },
            quads: { storage: 'array<Quad>', access: 'read' },
            pick: { storage: 'array<u32>', access: 'read' },
          },
          include: ['sdf'],
          code: DRAW_WGSL,
        });
        const mk = (vs, fs, blend) => draw.renderPipeline({ vs, fs, format: gpu.format, blend, sampleCount: 4 });
        pipes = {
          bg: mk('vs_main', 'fs_bg'),
          cloth: mk('vs_cloth', 'fs_cloth'),
          seg: mk('vs_seg', 'fs_seg', 'alpha'),
          pt: mk('vs_pt', 'fs_pt', 'alpha'),
          blob: mk('vs_blob', 'fs_blob'),
        };
      }
      const common = { u: U, pts: bufs.pts, cons: bufs.cons, blobs: bufs.blobs, frc: bufs.frc, pmeta: bufs.meta, nbr: bufs.nbr, pick: pickBuf };
      // pre-create every bind group we will need (sub-step × color)
      bgs = subs.map((s) => rangeBlocks.map((r) => sim.bind({ ...common, rg: r, sub: s })));
      drawBG = draw.bind({ u: U, pts: bufs.pts, cons: bufs.cons, blobs: bufs.blobs, pmeta: bufs.meta, nbr: bufs.nbr, quads: bufs.quads, pick: pickBuf });
      world = { b, nPts, nCons: order.length, nColors: ranges.length, ranges, nBlobs, nQuads, bufs, W };
      const obs = new Float32Array(32);
      b.obs.forEach((o, i) => obs.set(o, i * 4));
      U.set('obs', obs).set('nObs', b.obs.length).set('scene', b.layout).set('floorY', b.floorY);
    };

    const makeMsaa = () => {
      msaa?.destroy();
      const tex = gpu.texture({ size: [ctx.width, ctx.height], format: gpu.format, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'verlet-msaa' });
      msaa = { texture: tex, view: tex.createView(), destroy: () => tex.destroy() };
    };

    build();
    makeMsaa();
    let grabbing = false;
    let lastMouse = null;

    return {
      resize(w, h) {
        makeMsaa();
        if (Math.abs(1000 * (w / h) - world.W) > world.W * 0.03) build();
      },
      onAction(key) {
        if (key === 'reset') build();
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        const ptr = ctx.pointer;
        const toW = (x, y) => [(x / ctx.height) * 1000, (y / ctx.height) * 1000];
        const m = toW(ptr.x, ptr.y);
        const pm = lastMouse || m;
        lastMouse = m;
        const cutting = ptr.down && (ptr.button === 2 || p.cutMode);
        const wantGrab = ptr.down && !cutting;
        const startGrab = wantGrab && (ptr.clicked || !grabbing);
        grabbing = wantGrab;

        const nSub = Math.max(1, Math.min(MAX_SUB, Math.round(test ? Math.min(p.substeps, 2) : p.substeps)));
        const iters = Math.max(1, Math.round(test ? Math.min(p.iterations, 4) : p.iterations));
        const frameDt = Math.min(ctx.dt, 1 / 30);
        const dt = frameDt / nSub;
        U.set('gravity', [0, 1500 * p.gravity])
          .set('dt', Math.max(dt, 1e-5))
          .set('damping', Math.pow(p.damping, dt * 60))
          .set('mouse', m)
          .set('pmouse', pm)
          .set('grab', wantGrab ? 1 : 0)
          .set('cutting', cutting && !ctx.paused ? 1 : 0)
          .set('time', ctx.time)
          .set('nPts', world.nPts)
          .set('wind', [p.wind, 2.6, 0, 2.2])
          .set('nBlobs', world.nBlobs)
          .set('pressure', p.pressure)
          .set('tearMul', p.tear)
          .set('worldW', world.W)
          .set('friction', 0.6)
          .set('pickR', 45)
          .set('debug', p.debug ? 1 : 0)
          .set('example', Math.max(0, examples.findIndex((e) => e.id === ctx.example)))
          .set('nCons', world.nCons)
          .set('stiffMul', p.stiffness);
        U.upload();
        for (let s = 0; s < nSub; s++) subs[s].set('frac', (s + 1) / nSub).upload();
        if (startGrab) gpu.queue.writeBuffer(pickBuf, 0, new Uint32Array([0xffffffff]));

        const groups = (n) => Math.max(1, Math.ceil(n / 64));
        if (startGrab || (!ctx.paused && dt > 0)) {
          const pass = enc.beginComputePass({ label: 'verlet' });
          const bg0 = bgs[0][0];
          if (startGrab) {
            pass.setPipeline(sim.computePipeline('pickPoint'));
            pass.setBindGroup(0, bg0);
            pass.dispatchWorkgroups(groups(world.nPts));
          }
          if (!ctx.paused && dt > 0) {
            for (let s = 0; s < nSub; s++) {
              const bg = bgs[s][0];
              if (world.nBlobs) {
                pass.setPipeline(sim.computePipeline('blobArea'));
                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(groups(world.nBlobs));
              }
              pass.setPipeline(sim.computePipeline('forces'));
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(groups(world.nPts));
              pass.setPipeline(sim.computePipeline('integrate'));
              pass.dispatchWorkgroups(groups(world.nPts));
              pass.setPipeline(sim.computePipeline('solve'));
              for (let it = 0; it < iters; it++) {
                for (let c = 0; c < world.nColors; c++) {
                  pass.setBindGroup(0, bgs[s][c]);
                  pass.dispatchWorkgroups(groups(world.ranges[c][1]));
                }
              }
              pass.setPipeline(sim.computePipeline('collide'));
              pass.setBindGroup(0, bg);
              pass.dispatchWorkgroups(groups(world.nPts));
            }
          }
          pass.end();
        }

        const rp = enc.beginRenderPass({
          label: 'verlet-draw',
          colorAttachments: [{ view: msaa.view, resolveTarget: ctx.target, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'discard' }],
        });
        rp.setBindGroup(0, drawBG);
        rp.setPipeline(pipes.bg);
        rp.draw(3);
        if (world.nQuads) {
          rp.setPipeline(pipes.cloth);
          rp.draw(6, world.nQuads);
        }
        if (world.nBlobs) {
          rp.setPipeline(pipes.blob);
          rp.draw(MAXRING * 3, world.nBlobs);
        }
        rp.setPipeline(pipes.seg);
        rp.draw(6, world.nCons);
        rp.setPipeline(pipes.pt);
        rp.draw(6, world.nPts);
        rp.end();

        const disp = nSub * iters * world.nColors;
        readout.textContent = `${world.nPts.toLocaleString()} points · ${world.nCons.toLocaleString()} links in ${world.nColors} colors → ${disp} solve dispatches/frame`;
      },
    };
  },
};

// Weather: rain, snow, wind and lightning.
// One GPU particle pool (compute) holds raindrops, splash droplets, puddle ripples, snowflakes and
// leaves. Every particle has a depth z (0 = far, 1 = near) that sets its size, speed, brightness
// and where it hits the ground. Snow that lands near the camera is counted per screen column with
// atomics and drawn as a growing snow drift. The backdrop is a procedural fragment shader that
// also draws fog layers, puddle reflections, street lamps and lightning.

import { makeStrip, createBloom, TONEMAP_WGSL, tag, prng, TEX } from './_shared.js';

const COLS = 512; // snow accumulation columns

const STRUCTS = /* wgsl */ `
struct Drop {
  pos: vec2f, vel: vec2f,
  age: f32, life: f32, z: f32, kind: f32,   // z: depth 0 far .. 1 near
  rot: f32, spin: f32, seed: f32, state: f32,
};
const K_RAIN: f32 = 0.0;
const K_SPLASH: f32 = 1.0;
const K_RIPPLE: f32 = 2.0;
const K_FLAKE: f32 = 3.0;
const K_LEAF: f32 = 4.0;
const K_SETTLED: f32 = 5.0;
fn groundAt(z: f32) -> f32 { return mix(u.groundTop, u.groundBot, z); }
fn puddle(x: f32, z: f32) -> f32 { return smoothstep(0.05, 0.25, valueNoise(vec2f(x * 0.0045, z * 7.0 + 3.0)) - 0.42); }
fn windAt(x: f32, z: f32) -> f32 {
  // base wind + gusts that vary a little across the screen
  return u.wind + u.gust * (0.55 + 0.45 * valueNoise(vec2f(x * 0.002 - u.time * 0.6, z * 3.0)));
}
`;

const SIM_WGSL = /* wgsl */ `
${STRUCTS}
struct Counters { head: atomic<u32>, p0: u32, p1: u32, p2: u32 };

var<private> rs: u32;
fn seedRng(s: u32) { rs = pcg(s ^ 0x85ebca6bu); }
fn rnd() -> f32 { rs = rs * 747796405u + 2891336453u; let w = ((rs >> ((rs >> 28u) + 4u)) ^ rs) * 277803737u; return f32((w >> 22u) ^ w) / 4294967296.0; }

fn spawn(d: Drop) { ps[atomicAdd(&ctr.head, 1u) & (u.maxCount - 1u)] = d; }

fn newDrop(kind: f32, pos: vec2f, vel: vec2f, z: f32, life: f32) -> Drop {
  var d: Drop;
  d.pos = pos; d.vel = vel; d.age = 0.0; d.life = life; d.z = z; d.kind = kind;
  d.rot = rnd() * TAU; d.spin = (rnd() - 0.5) * 8.0; d.seed = rnd(); d.state = 0.0;
  return d;
}

fn snowSurface(x: f32) -> f32 {
  let c = clamp(i32(x / u.world.x * f32(${COLS})), 0, ${COLS - 1});
  return u.accY - f32(atomicLoad(&snowCols[c])) * u.accK;
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.emitCount) { return; }
  seedRng(i * 2654435u + u.seed * 97u + 5u);
  let margin = 300.0 + abs(u.wind) * 500.0;
  var z = rnd();
  if (u.mode != 2u) { z = 1.0 - sqrt(rnd()); }   // more far drops than near ones, like real depth
  let x = -margin + rnd() * (u.world.x + 2.0 * margin);
  var y = -30.0 - rnd() * 120.0;
  if (u.fill != 0u) { y = mix(-60.0, groundAt(z), rnd()); }
  var d: Drop;
  if (u.mode == 1u) {
    let fall = mix(45.0, 150.0, z) * u.speedMul;
    d = newDrop(K_FLAKE, vec2f(x, y), vec2f(u.wind * 120.0, fall), z, 60.0);
  } else if (u.mode == 2u) {
    // leaves: from the tree canopy, or blown in from the upwind edge
    var p = vec2f(u.world.x * (0.06 + 0.3 * rnd()), 120.0 + 300.0 * rnd());
    if (rnd() < 0.45) { p = vec2f(select(u.world.x + 40.0, -40.0, u.wind > 0.0), 100.0 + 600.0 * rnd()); }
    if (u.fill != 0u) { p = vec2f(rnd() * u.world.x, 100.0 + 600.0 * rnd()); }
    z = 0.25 + 0.75 * rnd();
    d = newDrop(K_LEAF, p, vec2f(u.wind * 150.0, 40.0), z, 18.0 + 10.0 * rnd());
    d.spin = (rnd() - 0.5) * 7.0;
    d.state = rnd() * 4.0;   // leaf color variant
  } else {
    let fall = mix(1100.0, 2300.0, z) * u.speedMul;
    d = newDrop(K_RAIN, vec2f(x, y), vec2f(u.wind * fall * 0.35, fall), z, 30.0);
  }
  spawn(d);
}

@compute @workgroup_size(128)
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.maxCount) { return; }
  var d = ps[i];
  if (d.age >= d.life) { return; }
  seedRng(i * 7919u + u.seed * 31337u);
  let dt = u.dt;
  let margin = 300.0 + abs(u.wind) * 500.0;
  let w = windAt(d.pos.x, d.z);
  if (d.kind == K_RAIN) {
    // terminal velocity: rain doesn't accelerate, it just leans with the wind
    d.vel.x = mix(d.vel.x, w * d.vel.y * 0.35, 1.0 - exp(-dt * 6.0));
    d.pos += d.vel * dt;
    let g = groundAt(d.z);
    if (d.pos.y > g) {
      d.age = d.life;                       // the drop dies...
      let hit = vec2f(d.pos.x, g);
      let s = 0.35 + 0.65 * d.z;
      if (u.splashes != 0u) {               // ...and becomes 2-3 splash droplets
        let n = 2u + u32(rnd() * 2.0);
        for (var k = 0u; k < n; k++) {
          var sp = newDrop(K_SPLASH, hit, vec2f((rnd() - 0.5) * 260.0, -120.0 - 200.0 * rnd()) * s, d.z, 0.22 + 0.2 * rnd());
          spawn(sp);
        }
        // and a ripple if it fell into a puddle
        if (puddle(hit.x, d.z) > 0.3 && rnd() < 0.7) { spawn(newDrop(K_RIPPLE, hit, vec2f(0.0), d.z, 0.55 + 0.35 * rnd())); }
      }
    }
  } else if (d.kind == K_SPLASH) {
    d.vel.y += 1700.0 * (0.4 + 0.6 * d.z) * dt;
    d.pos += d.vel * dt;
  } else if (d.kind == K_FLAKE) {
    // flutter: each flake sways on its own sine, wind pushes harder on near flakes
    let ph = u.time * (1.2 + d.seed * 1.6) + d.seed * 40.0;
    let target = vec2f(w * mix(80.0, 220.0, d.z) + sin(ph) * 45.0 * (0.4 + d.z), mix(45.0, 150.0, d.z) * u.speedMul + cos(ph * 1.3) * 12.0);
    d.vel = mix(d.vel, target, 1.0 - exp(-dt * 2.5));
    d.pos += d.vel * dt;
    d.rot += d.spin * dt * 0.3;
    if (d.z > 0.72 && u.accumulate != 0u) {
      let surf = snowSurface(d.pos.x);
      if (d.pos.y > surf && d.pos.x > 0.0 && d.pos.x < u.world.x) {
        let c = clamp(i32(d.pos.x / u.world.x * f32(${COLS})), 0, ${COLS - 1});
        atomicAdd(&snowCols[c], 1u);         // pile up: one more flake in this column
        d.kind = K_SETTLED; d.age = 0.0; d.life = 0.6; d.vel = vec2f(0.0); d.pos.y = surf;
      }
    } else if (d.pos.y > groundAt(d.z)) {
      d.kind = K_SETTLED; d.age = 0.0; d.life = 0.8; d.vel = vec2f(0.0);
    }
  } else if (d.kind == K_LEAF) {
    let g = groundAt(d.z);
    if (d.pos.y >= g - 1.0 && abs(d.vel.y) < 1.0) {
      // resting on the ground until a gust lifts it again
      d.vel = vec2f(0.0);
      if (abs(w) > 0.9 && rnd() < dt * (abs(w) - 0.8) * 1.5) { d.vel = vec2f(w * 160.0, -140.0 - 220.0 * rnd()); d.pos.y = g - 2.0; }
    } else {
      // tumbling: drag toward the wind velocity + gravity + flutter lift
      let ph = u.time * 3.0 + d.seed * 30.0;
      let airV = vec2f(w * 260.0 * (0.6 + 0.4 * d.z), 70.0 + sin(ph) * 60.0);
      d.vel = mix(d.vel, airV, 1.0 - exp(-dt * 1.6));
      d.vel.y += 40.0 * dt;
      d.pos += d.vel * dt;
      d.rot += d.spin * dt * (0.5 + abs(w));
      if (d.pos.y > g) { d.pos.y = g; d.vel = vec2f(0.0); d.spin *= 0.4; }
    }
  }
  // wrap horizontally so a steady wind doesn't blow the screen empty
  if (d.kind == K_FLAKE || d.kind == K_LEAF) {
    if (d.pos.x < -margin) { d.pos.x += u.world.x + 2.0 * margin; }
    if (d.pos.x > u.world.x + margin) { d.pos.x -= u.world.x + 2.0 * margin; }
  }
  d.age += dt;
  ps[i] = d;
}

@compute @workgroup_size(1)
fn args() {
  drawArgs[0] = 6u; drawArgs[1] = min(atomicLoad(&ctr.head), u.maxCount); drawArgs[2] = 0u; drawArgs[3] = 0u;
}

@compute @workgroup_size(64)
fn meltSnow(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= ${COLS}u) { return; }
  // slowly settle the pile: average with neighbours so it forms drifts, not spikes
  let l = atomicLoad(&snowCols[max(c, 1u) - 1u]);
  let r = atomicLoad(&snowCols[min(c + 1u, ${COLS - 1}u)]);
  let m = atomicLoad(&snowCols[c]);
  let avg = (l + r + m * 2u) / 4u;
  if (m > avg + 2u) { atomicSub(&snowCols[c], 1u); }
  if (u.accumulate == 0u && m > 0u) { atomicSub(&snowCols[c], min(m, 4u)); }
}
`;

const DRAW_WGSL = /* wgsl */ `
${STRUCTS}
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) col: vec4f,
  @location(2) @interpolate(flat) info: vec4f,   // x: texture cell (-1 = analytic streak), y: additive, z: back side
};

fn lampLight(p: vec2f) -> vec3f {
  var l = vec3f(0.0);
  for (var k = 0; k < 3; k++) {
    let lp = u.lamps[k];
    if (lp.w > 0.0) {
      let d = p - lp.xy;
      // a cone pointing down
      let cone = smoothstep(0.55, 0.85, d.y / max(length(d), 1.0)) * step(0.0, d.y);
      l += vec3f(1.0, 0.72, 0.4) * lp.w * cone * 90000.0 / (dot(d, d) + 9000.0);
    }
  }
  return l;
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let d = ps[ii];
  if (d.age >= d.life) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  let t = d.age / d.life;
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  var ax = vec2f(cos(d.rot), sin(d.rot));
  var half = vec2f(4.0);
  var col = vec4f(1.0);
  var info = vec4f(${TEX.soft}.0, 1.0, 0.0, 0.0);
  let light = u.ambient + lampLight(d.pos) + vec3f(0.8, 0.85, 1.0) * u.flash * 3.0;
  var dbg = vec3f(1.0);
  if (u.mode == 3u) { dbg = mix(vec3f(0.25, 0.45, 1.0), vec3f(1.0, 0.3, 0.25), d.z) * 2.0; }
  if (d.kind == K_RAIN) {
    let sp = length(d.vel);
    ax = d.vel / max(sp, 1.0);
    // streak length = distance travelled while a camera shutter is open (motion blur)
    half = vec2f(sp * 0.011 * u.sizeMul + 3.0, mix(0.7, 2.0, d.z) * u.sizeMul);
    col = vec4f(light * dbg * mix(0.10, 0.32, d.z), 1.0);
    info.x = -1.0;
  } else if (d.kind == K_SPLASH) {
    half = vec2f(mix(1.0, 2.6, d.z) * u.sizeMul);
    col = vec4f(light * dbg * 0.5 * (1.0 - t), 1.0);
  } else if (d.kind == K_RIPPLE) {
    let r = mix(6.0, 30.0, d.z) * (0.25 + 1.2 * t);
    half = vec2f(r, r * 0.26);
    ax = vec2f(1.0, 0.0);
    col = vec4f(light * dbg * 0.5 * (1.0 - t) * (1.0 - t), 1.0);
    info.x = ${TEX.ring}.0;
  } else if (d.kind == K_FLAKE || d.kind == K_SETTLED) {
    let s = mix(1.6, 6.5, d.z * d.z) * u.sizeMul * (0.7 + 0.6 * d.seed);
    half = vec2f(s);
    let fade = select(1.0, 1.0 - t, d.kind == K_SETTLED);
    let lit = min(light * 1.1 + vec3f(0.05), vec3f(3.0));
    col = vec4f(lit * dbg, mix(0.45, 0.95, d.z) * fade);
    info = vec4f(select(${TEX.soft}.0, ${TEX.flake}.0, d.z > 0.55), 0.0, 0.0, 0.0);
  } else {
    // leaf: flip around its long axis to fake a 3D tumble
    let s = mix(9.0, 20.0, d.z) * u.sizeMul;
    let flip = cos(d.rot * 1.7 + d.seed * 6.0);
    half = vec2f(s, s * max(abs(flip), 0.12));
    var tint = vec3f(0.9, 0.38, 0.06);
    if (d.state > 3.0) { tint = vec3f(0.85, 0.62, 0.08); }
    else if (d.state > 2.0) { tint = vec3f(0.62, 0.12, 0.04); }
    else if (d.state > 1.0) { tint = vec3f(0.42, 0.22, 0.08); }
    let back = select(1.0, 0.65, flip < 0.0);
    let fade = 1.0 - smoothstep(0.85, 1.0, t);
    col = vec4f(tint * back * (u.ambient * 1.4 + vec3f(0.25)) * 2.2, fade);
    info = vec4f(${TEX.leaf}.0, 0.0, 0.0, 0.0);
  }
  let ay = vec2f(-ax.y, ax.x);
  let w = d.pos + ax * c.x * half.x + ay * c.y * half.y;
  o.pos = vec4f(w.x / u.world.x * 2.0 - 1.0, 1.0 - w.y / u.world.y * 2.0, 0.0, 1.0);
  o.local = c * 0.5 + 0.5;
  o.col = col;
  o.info = info;
  return o;
}

@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  var a: f32;
  var rgb = i.col.rgb;
  if (i.info.x < 0.0) {
    // analytic raindrop streak: soft across, tapered along
    let q = i.local * 2.0 - 1.0;
    a = exp(-q.y * q.y * 3.0) * (1.0 - q.x * q.x) * smoothstep(-1.0, -0.2, q.x);
  } else {
    let s = textureSample(strip, stripSamp, vec2f((i.info.x + clamp(i.local.x, 0.01, 0.99)) / 8.0, i.local.y));
    a = s.a;
    rgb *= s.rgb;
  }
  a *= i.col.a;
  return vec4f(rgb * a, a * (1.0 - i.info.y));
}
`;

const BG_WGSL = /* wgsl */ `
fn aa(d: f32) -> f32 { return clamp(0.5 - d / max(u.pxw, 0.0001), 0.0, 1.0); }

fn skyline(x: f32, seed: f32, w: f32, hmin: f32, hmax: f32) -> vec2f {
  let id = floor(x / w);
  let h = hmin + (hmax - hmin) * hash11(id * 3.1 + seed) * (0.4 + 0.6 * hash11(id * 1.3 + seed + 7.0));
  return vec2f(h, id);
}

fn boltDist(p: vec2f) -> f32 {
  var d = 1e9;
  for (var k = 0; k < 17; k++) {
    let a4 = u.bolt[k / 2];
    let b4 = u.bolt[(k + 1) / 2];
    var a = a4.xy; if (k % 2 == 1) { a = a4.zw; }
    var b = b4.xy; if ((k + 1) % 2 == 1) { b = b4.zw; }
    d = min(d, sdSegment(p, a, b));
  }
  return d;
}

// ---------------------------------------------------------------- rainy city
fn cityAt(w: vec2f, reflectFlash: f32) -> vec3f {
  // sky with storm clouds lit by the flash
  let cl = fbm(vec2f(w.x * 0.0016 - u.time * 0.02 * (1.0 + u.wind), w.y * 0.004), 4) * 0.5 + 0.5;
  var c = mix(vec3f(0.012, 0.014, 0.022), vec3f(0.03, 0.033, 0.045), smoothstep(0.35, 0.8, cl)) * (1.0 + 0.4 * (1.0 - w.y / 760.0));
  c += vec3f(0.55, 0.6, 0.8) * u.flash * (0.25 + 0.75 * smoothstep(0.3, 0.75, cl)) * 1.6 * reflectFlash;
  // far skyline: hazy; revealed against the sky when lightning flashes
  let far = skyline(w.x, 11.0, 70.0, 120.0, 330.0);
  if (w.y > u.groundTop - far.x) {
    c = mix(vec3f(0.018, 0.02, 0.028), vec3f(0.01), 0.3) + vec3f(0.02, 0.022, 0.03) * u.flash * 0.3;
    let wc = floor(vec2f(w.x / 11.0, w.y / 16.0));
    let lit = step(0.86, hash21(wc + far.y * 3.0)) * step(2.0, fmod(w.x, 11.0)) * step(fmod(w.x, 11.0), 7.0) * step(fmod(w.y, 16.0), 9.0);
    c += vec3f(0.5, 0.36, 0.16) * lit * 0.22;
  }
  // near buildings: darker, taller
  let near = skyline(w.x + 37.0, 23.0, 150.0, 60.0, 420.0);
  if (w.y > u.groundTop - near.x + 25.0) {
    c = vec3f(0.008, 0.009, 0.012) + vec3f(0.012, 0.013, 0.02) * u.flash;
    let wc = floor(vec2f(w.x / 18.0, w.y / 26.0));
    let lit = step(0.8, hash21(wc + near.y * 5.0)) * step(4.0, fmod(w.x, 18.0)) * step(fmod(w.x, 18.0), 13.0) * step(fmod(w.y, 26.0), 15.0);
    let warm = mix(vec3f(1.0, 0.65, 0.3), vec3f(0.6, 0.8, 1.0), step(0.7, hash21(wc * 1.7)));
    c += warm * lit * 0.35;
  }
  return c;
}

fn rainCity(w: vec2f) -> vec3f {
  var c: vec3f;
  if (w.y < u.groundTop) {
    c = cityAt(w, 1.0);
  } else {
    // wet street: reflect the city about the horizon line, rippled; puddles reflect sharply
    let z = (w.y - u.groundTop) / (u.groundBot - u.groundTop);
    let pd = puddle(w.x, z);
    let wob = (valueNoise(vec2f(w.x * 0.05, w.y * 0.3 + u.time * 2.0)) - 0.5) * mix(14.0, 3.0, pd);
    let ry = u.groundTop - (w.y - u.groundTop) * mix(2.2, 1.6, z);
    let refl = cityAt(vec2f(w.x + wob, ry), 0.6);
    let asphalt = vec3f(0.012, 0.012, 0.014) * (0.8 + 0.4 * valueNoise(w * 0.08));
    c = asphalt + refl * mix(0.25, 0.8, pd);
    c += vec3f(0.02, 0.022, 0.028) * u.flash * 1.5;
  }
  // street lamps: poles, heads and pools of light on the wet street
  for (var k = 0; k < 3; k++) {
    let lp = u.lamps[k];
    let pole = sdBox(w - vec2f(lp.x - 26.0, (lp.y + 940.0) * 0.5 - 6.0), vec2f(4.0, (940.0 - lp.y) * 0.5 + 6.0));
    let arm = sdSegment(w, vec2f(lp.x - 26.0, lp.y - 14.0), vec2f(lp.x, lp.y - 8.0)) - 3.0;
    c = mix(c, vec3f(0.006), aa(min(pole, arm)));
    let hd = length((w - lp.xy) * vec2f(0.6, 1.6));
    c += vec3f(1.0, 0.75, 0.45) * (exp(-hd / 4.0) * 6.0 + exp(-hd / 40.0) * 0.25);
    let pool = length((w - vec2f(lp.x, 950.0)) * vec2f(0.012, 0.05));
    if (w.y > u.groundTop) { c += vec3f(1.0, 0.7, 0.4) * exp(-pool * pool) * 0.12 * (0.5 + puddle(w.x, (w.y - u.groundTop) / (u.groundBot - u.groundTop))); }
  }
  return c;
}

// ---------------------------------------------------------------- blizzard
fn pine(p: vec2f, base: vec2f, h: f32, sway: f32) -> f32 {
  let q = p - base;
  let lean = sway * (-q.y / h) * (-q.y / h) * h * 0.08;
  let qq = vec2f(q.x - lean, q.y);
  var d = 1e9;
  for (var k = 0; k < 3; k++) {
    let fk = f32(k);
    let y0 = -h * (0.15 + 0.25 * fk);
    let wdt = h * (0.36 - 0.08 * fk);
    d = min(d, sdTriangle(qq, vec2f(-wdt, y0), vec2f(wdt, y0), vec2f(0.0, y0 - h * 0.42)));
  }
  return min(d, sdBox(qq - vec2f(0.0, -h * 0.08), vec2f(h * 0.04, h * 0.08)));
}

fn blizzard(w: vec2f) -> vec3f {
  var c = mix(vec3f(0.07, 0.08, 0.11), vec3f(0.16, 0.17, 0.2), smoothstep(0.0, 700.0, w.y));
  // mountains
  let mh = 430.0 + 120.0 * (ridged(vec2f(w.x * 0.0015, 2.0), 4) - 0.5) * -1.0;
  if (w.y > mh) { c = mix(c, vec3f(0.2, 0.22, 0.27), 0.6 * aa(mh - w.y + 0.5)); }
  // far forest band
  let sway = u.wind + u.gust;
  for (var layer = 0; layer < 2; layer++) {
    let fl = f32(layer);
    let cw = mix(42.0, 70.0, fl);
    let id = floor(w.x / cw);
    var d = 1e9;
    for (var k = -1; k <= 1; k++) {
      let cid = id + f32(k);
      let h = mix(70.0, 150.0, fl) * (0.6 + 0.6 * hash11(cid * 3.7 + fl * 9.0));
      let bx = (cid + 0.5 + (hash11(cid * 1.9 + fl) - 0.5) * 0.6) * cw;
      d = min(d, pine(w, vec2f(bx, mix(650.0, 720.0, fl)), h, sway * (0.6 + 0.4 * sin(u.time * 2.0 + cid))));
    }
    let tc = mix(vec3f(0.11, 0.13, 0.16), vec3f(0.05, 0.065, 0.08), fl);
    c = mix(c, tc, aa(d));
  }
  // snowfield
  if (w.y > u.groundTop) { c = mix(c, mix(vec3f(0.36, 0.38, 0.44), vec3f(0.52, 0.55, 0.62), (w.y - u.groundTop) / 300.0), aa(u.groundTop - w.y)); }
  // cabin with a warm window
  let cb = vec2f(u.world.x * 0.68, 712.0);
  let body = sdBox(w - (cb + vec2f(0.0, -28.0)), vec2f(58.0, 28.0));
  let roof = sdTriangle(w, cb + vec2f(-74.0, -54.0), cb + vec2f(74.0, -54.0), cb + vec2f(0.0, -104.0));
  c = mix(c, vec3f(0.06, 0.045, 0.04), aa(min(body, roof)));
  c = mix(c, vec3f(0.62, 0.64, 0.7), aa(roof + 2.0) * step(w.y, cb.y - 54.0 - (74.0 - abs(w.x - cb.x)) * 0.5 + 9.0));
  let win = sdBox(w - (cb + vec2f(-18.0, -30.0)), vec2f(12.0, 10.0));
  c = mix(c, vec3f(2.2, 1.3, 0.5), aa(win));
  c += vec3f(1.0, 0.55, 0.2) * exp(-length(w - (cb + vec2f(-18.0, -30.0))) / 60.0) * 0.12;
  // accumulated snow drift along the bottom (from the atomic column counters)
  let fc = clamp(w.x / u.world.x * f32(${COLS}), 0.0, f32(${COLS - 1}));
  let c0 = i32(floor(fc));
  let c1 = min(c0 + 1, ${COLS - 1});
  let hgt = mix(f32(snowCols[c0]), f32(snowCols[c1]), fract(fc)) * u.accK;
  let surf = u.accY - hgt;
  if (w.y > surf - 1.0) {
    let depth = w.y - surf;
    let sc = mix(vec3f(0.85, 0.88, 0.95), vec3f(0.55, 0.6, 0.72), smoothstep(0.0, 60.0, depth));
    c = mix(c, sc * (0.9 + 0.1 * valueNoise(w * 0.3)), aa(surf - w.y));
  }
  // drifting fog banks
  let fg = fbm(vec2f(w.x * 0.002 - u.time * (0.05 + 0.1 * abs(sway)) * sign(sway + 0.001), w.y * 0.006), 4) * 0.5 + 0.5;
  c = mix(c, vec3f(0.3, 0.32, 0.37), smoothstep(0.35, 0.85, fg) * u.fog * 0.6 * smoothstep(300.0, 700.0, w.y));
  return c;
}

// ---------------------------------------------------------------- autumn
fn autumn(w: vec2f) -> vec3f {
  var c = mix(vec3f(0.18, 0.12, 0.25), vec3f(1.1, 0.5, 0.2), smoothstep(100.0, 700.0, w.y));
  let sun = vec2f(u.world.x * 0.7, 600.0);
  let sd = length(w - sun);
  c += vec3f(1.6, 0.8, 0.35) * (exp(-sd / 40.0) * 2.0 + exp(-sd / 220.0) * 0.35);
  // hills
  let h1 = 640.0 + 40.0 * sin(w.x * 0.004 + 1.0) + 25.0 * perlin(vec2f(w.x * 0.006, 1.0));
  c = mix(c, vec3f(0.35, 0.16, 0.12), 0.85 * aa(h1 - w.y));
  let h2 = 700.0 + 30.0 * sin(w.x * 0.003 + 4.0) + 20.0 * perlin(vec2f(w.x * 0.008, 5.0));
  c = mix(c, vec3f(0.15, 0.08, 0.07), aa(h2 - w.y));
  // ground with grass blades that bend with the wind
  let sway = u.wind + u.gust * 0.8;
  if (w.y > u.groundTop - 30.0) {
    let z = clamp((w.y - u.groundTop) / (u.groundBot - u.groundTop), 0.0, 1.0);
    var g = mix(vec3f(0.12, 0.07, 0.04), vec3f(0.08, 0.05, 0.03), z);
    let bx = floor(w.x / 6.0);
    let bh = 18.0 + 22.0 * hash11(bx);
    let tip = (w.y - (u.groundTop - bh)) / bh;
    let bend = sway * 10.0 * (1.0 - tip) * (1.0 - tip) + sin(u.time * 3.0 + bx) * 2.0;
    let blade = abs(fmod(w.x + bend, 6.0) - 3.0) - (1.0 - tip) * 1.6;
    let grass = aa(blade) * step(0.0, tip);
    c = mix(c, mix(g, vec3f(0.25, 0.16, 0.06), 0.4), grass);
    c = mix(c, g, aa(u.groundTop - w.y));
  }
  // the tree: trunk, branches and a swaying canopy
  let tb = vec2f(u.world.x * 0.18, u.groundTop + 20.0);
  let sw = sway * 0.6 + sin(u.time * 1.3) * 0.08 * (0.3 + abs(sway));
  var td = sdSegment(w, tb, tb + vec2f(sw * 40.0, -380.0)) - mix(26.0, 12.0, clamp((tb.y - w.y) / 380.0, 0.0, 1.0));
  td = min(td, sdSegment(w, tb + vec2f(sw * 20.0, -220.0), tb + vec2f(-140.0 + sw * 70.0, -380.0)) - 8.0);
  td = min(td, sdSegment(w, tb + vec2f(sw * 30.0, -280.0), tb + vec2f(150.0 + sw * 80.0, -430.0)) - 7.0);
  c = mix(c, vec3f(0.05, 0.03, 0.025), aa(td));
  let cp = w - (tb + vec2f(sw * 90.0, -470.0));
  let cn = fbm(cp * 0.012 + vec2f(sw * 0.4, 0.0), 4);
  let canopy = length(cp * vec2f(0.75, 1.1)) - 190.0 - cn * 90.0;
  let cc = mix(vec3f(0.55, 0.16, 0.04), vec3f(0.95, 0.45, 0.08), smoothstep(-0.4, 0.5, fbm(cp * 0.03, 3)));
  c = mix(c, cc * mix(0.45, 1.0, smoothstep(150.0, -150.0, cp.x + cp.y * 0.5)), aa(canopy));
  // warm haze
  let fg = fbm(vec2f(w.x * 0.0025 - u.time * 0.06 * sign(sway + 0.001), w.y * 0.008), 3) * 0.5 + 0.5;
  c = mix(c, vec3f(0.9, 0.5, 0.3), smoothstep(0.4, 0.9, fg) * u.fog * 0.35 * smoothstep(350.0, 750.0, w.y));
  return c;
}

// ---------------------------------------------------------------- abstract layers
fn layersBg(w: vec2f) -> vec3f {
  var c = mix(vec3f(0.01, 0.012, 0.02), vec3f(0.025, 0.02, 0.035), w.y / 1000.0);
  let gp = abs(fract(w / 100.0 + 0.5) - 0.5) * 100.0;
  c += vec3f(0.015, 0.018, 0.03) * (1.0 - smoothstep(0.0, 1.5 * u.pxw, min(gp.x, gp.y)));
  for (var k = 0; k <= 4; k++) {
    let z = f32(k) / 4.0;
    let gy = mix(u.groundTop, u.groundBot, z);
    let lc = mix(vec3f(0.25, 0.45, 1.0), vec3f(1.0, 0.3, 0.25), z);
    c += lc * 0.5 * exp(-abs(w.y - gy) / (1.2 * u.pxw));
  }
  return c;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let w = uv * u.world;
  var c: vec3f;
  if (u.mode == 0u) { c = rainCity(w); }
  else if (u.mode == 1u) { c = blizzard(w); }
  else if (u.mode == 2u) { c = autumn(w); }
  else { c = layersBg(w); }
  // lightning bolt + flash
  if (u.boltOn > 0.0 && w.y < u.groundTop + 20.0) {
    let bd = boltDist(w);
    c += vec3f(0.75, 0.82, 1.0) * u.boltOn * (exp(-bd / (1.2 * u.pxw)) * 8.0 + exp(-bd / 30.0) * 0.6);
  }
  return vec4f(c, 1.0);
}
`;

const COMPOSITE_WGSL = /* wgsl */ `
${TONEMAP_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var c = TEX(hdr, uv).rgb + TEX(bloomTex, uv).rgb * u.bloomAmt;
  // near fog layer drifting over everything (in front of the particles)
  let w = uv * u.world;
  let f = fbm(vec2f(w.x * 0.0018 - u.time * 0.08 * (u.wind + u.gust), w.y * 0.004 + u.time * 0.02), 4) * 0.5 + 0.5;
  let fogCol = select(u.fogColor.rgb, u.fogColor.rgb + vec3f(0.3, 0.32, 0.4) * u.flash, u.mode == 0u);
  c = mix(c, fogCol, smoothstep(0.45, 0.95, f) * u.fog * 0.35 * smoothstep(0.25, 0.85, uv.y));
  let v = uv - 0.5;
  c *= 1.0 - dot(v, v) * 0.8;
  return vec4f(vfxTonemap(c * u.exposure * (1.0 + u.flash * 0.6), px), 1.0);
}
`;

const MODES = { rain: 0, snow: 1, leaves: 2, layers: 3 };

export default {
  interaction: 'Use the sliders and the ⚡ button. Wind and gusts push everything.',
  examples: [
    {
      id: 'rain',
      label: 'Rainstorm',
      kind: 'In a game',
      note: 'Thousands of streaks at different depths: near drops are longer, thicker, faster and brighter. Each drop that hits the street turns into splash droplets, and — if it lands in a puddle — an expanding ripple. Drops passing through the lamp light glow. Lightning flashes reveal the skyline.',
      hint: 'Press ⚡ Lightning! — or wait for the storm.',
      params: { amount: 9000, wind: 0.25, gusts: 0.35, fog: 0.5, speed: 1, size: 1, storm: 0.5, splashes: true, exposure: 1.2 },
    },
    {
      id: 'snow',
      label: 'Blizzard',
      kind: 'In a game',
      note: 'Flakes flutter on their own sine waves and are pushed by gusty wind. Flakes landing near the camera are counted per screen column with <code>atomicAdd</code>, so a snow drift slowly builds up along the bottom. Fog banks drift through the forest.',
      hint: 'Raise Wind and Gusts for a whiteout. Snow piles up over time.',
      params: { amount: 5000, wind: 0.55, gusts: 0.6, fog: 0.7, speed: 1, size: 1, storm: 0, accumulate: true, exposure: 1.1 },
    },
    {
      id: 'leaves',
      label: 'Windy autumn leaves',
      kind: 'In a game',
      note: 'Leaves tumble: drag pulls them toward the wind speed, a flutter force lifts and drops them, and flipping the quad’s width with <code>cos(angle)</code> fakes a 3D spin. Leaves rest on the ground until a strong gust picks them up again.',
      hint: 'Push Gusts up and watch the leaves on the ground take off.',
      params: { amount: 14, wind: 0.45, gusts: 0.8, fog: 0.5, speed: 1, size: 1, storm: 0, exposure: 1.1 },
    },
    {
      id: 'layers',
      label: 'How depth layers work',
      kind: 'Abstract',
      note: 'The same rain, color-coded by depth: <b style="color:#6b8cff">far</b> → <b style="color:#ff5a4a">near</b>. Depth z sets size, speed, brightness and the ground line it hits (the colored lines), which is all it takes to read as a 3D volume of rain in a 2D game.',
      params: { amount: 5000, wind: 0.2, gusts: 0, fog: 0, speed: 0.5, size: 1.4, storm: 0, splashes: true, exposure: 1.2 },
    },
  ],
  controls: [
    { type: 'slider', key: 'amount', label: 'Amount', min: 0, max: 60000, step: 10, value: 9000, help: 'Particles spawned per second (leaves: a few per second).', format: (v) => Math.round(v).toLocaleString() + '/s', showFor: ['rain', 'snow', 'layers'] },
    { type: 'slider', key: 'amount', label: 'Leaves per second', min: 0, max: 80, step: 1, value: 14, showFor: ['leaves'] },
    { type: 'slider', key: 'wind', label: 'Wind', min: -1.5, max: 1.5, step: 0.01, value: 0.25, help: 'Steady wind. Rain leans, flakes and leaves drift.' },
    { type: 'slider', key: 'gusts', label: 'Gusts', min: 0, max: 1.5, step: 0.01, value: 0.35, help: 'Extra wind that comes and goes over time and varies across the screen.' },
    { type: 'slider', key: 'speed', label: 'Fall speed', min: 0.2, max: 2, step: 0.01, value: 1, showFor: ['rain', 'snow', 'layers'] },
    { type: 'slider', key: 'size', label: 'Size', min: 0.4, max: 2.5, step: 0.01, value: 1 },
    { type: 'slider', key: 'fog', label: 'Fog', min: 0, max: 1, step: 0.01, value: 0.5, help: 'Drifting fbm fog layers behind and in front of the particles.', showFor: ['rain', 'snow', 'leaves'] },
    { type: 'slider', key: 'storm', label: 'Lightning frequency', min: 0, max: 1, step: 0.01, value: 0.5, help: '0 = only when you press the button.', showFor: ['rain'] },
    { type: 'toggle', key: 'splashes', label: 'Splashes & ripples', value: true, help: 'Drops that hit the ground spawn new particles on the GPU.', showFor: ['rain', 'layers'] },
    { type: 'toggle', key: 'accumulate', label: 'Snow accumulation', value: true, help: 'Count landed flakes per column with atomics; draw a drift.', showFor: ['snow'] },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: 0.3, max: 3, step: 0.01, value: 1.2 },
    { type: 'button', key: 'lightning', label: '⚡ Lightning!', primary: true },
    { type: 'button', key: 'reset', label: 'Reset' },
  ],
  about: {
    summary: 'Rain, snow and wind are particles with a depth value. Add splashes, ripples, fog, gusts and a lightning flash, and a flat 2D scene suddenly has weather and atmosphere.',
    what: `<p>A GPU particle pool (compute shaders) holds every raindrop, splash, ripple, snowflake and leaf. The backdrop — city, forest, autumn hill —
      is a procedural fragment shader that also draws puddle reflections, fog banks, street lamps and the lightning bolt.</p>`,
    how: `<ol>
      <li><b>Depth layers</b>: every particle gets a depth <i>z</i>. Near = bigger, faster, brighter and lower on screen when it lands; far = small, slow, faint. More far drops than near ones (<code>z = 1 − √rand</code>) mimics a real volume.</li>
      <li><b>Rain = motion blur</b>: a drop is a quad stretched along its velocity by the distance it travels while a camera shutter would be open. It falls at constant (terminal) speed and leans with the wind.</li>
      <li><b>Collision → new particles</b>: when a drop passes its ground line, the compute thread kills it and spawns 2–3 splash droplets (ballistic) plus, if a noise-based puddle is there, a ripple — an expanding, squashed ring.</li>
      <li><b>Snow</b>: flakes relax toward <i>wind + sin(time·f + seed)</i>, so each flutters differently. Near flakes landing on the drift do <code>atomicAdd(&amp;snowCols[column], 1)</code>; a small pass smooths the columns into drifts, and the backdrop shader reads the same buffer to draw the snow.</li>
      <li><b>Wind & gusts</b>: wind = base + gust envelope (from JS) × a noise that scrolls across the screen, so gusts visibly travel. Trees and grass in the backdrop bend with the same value.</li>
      <li><b>Lightning</b>: a midpoint-displacement bolt + a flash envelope with a double flicker. The flash brightens the sky behind the skyline (instant silhouettes), the rain, the fog and the exposure.</li>
    </ol>`,
    uses: [
      { title: 'Mood & storytelling', text: 'Rain for noir and horror, snow for quiet, wind and leaves for melancholy (Celeste, Hollow Knight’s City of Tears, Gris).' },
      { title: 'Gameplay weather', text: 'Wind that pushes the player and projectiles, lightning that reveals enemies, snow that builds up and slows movement.' },
      { title: 'Cheap depth', text: 'Layered particles + fog give a flat 2D game parallax depth without any 3D.' },
    ],
    try: [
      'Press <b>⚡ Lightning!</b> a few times in the Rainstorm — notice the skyline silhouettes and the bright rain during the flash.',
      'Drag <i>Wind</i> from −1.5 to 1.5: the rain angle changes smoothly because each drop relaxes toward the wind instead of snapping.',
      'In <b>Blizzard</b>, set <i>Amount</i> to 30,000 and <i>Gusts</i> to 1.5 for a whiteout; watch the drift grow along the bottom.',
      'In <b>How depth layers work</b>, turn splashes off and on, and compare the far (blue) and near (red) drops.',
      'In <b>Windy autumn leaves</b>, set <i>Gusts</i> to 0 and wait until leaves settle, then push it to 1.5.',
    ],
    ask: [
      'layered 2D rain with depth, splashes and puddle ripples',
      'snow with flutter and per-column accumulation using atomics',
      'wind gusts that travel across the screen and bend grass',
      'lightning flash that reveals background silhouettes',
      'drifting fog layers in front of and behind the action',
      'tumbling leaves faked by flipping sprite width with cos()',
    ],
    perf: `<p>Each particle costs one compute thread and one small quad. Rain streaks are thin, so even 50 k drops are cheap; big soft snowflakes near the camera
      cost more fill rate. The backdrop is one full-screen shader (a few noise lookups per pixel); in a real game it would be your normal background art.</p>`,
    api: `<p><b>WebGPU only</b> as written: collisions spawn splashes from inside the compute shader, and snow accumulation uses atomic counters in a storage buffer that
      the backdrop's fragment shader reads directly. In WebGL2 you'd simulate in textures and do splashes/accumulation on the CPU or with tricks.</p>`,
    code: [
      {
        title: 'A raindrop hits the ground: die, splash and ripple',
        lang: 'wgsl',
        src: `d.vel.x = mix(d.vel.x, w * d.vel.y * 0.35, 1.0 - exp(-dt * 6.0)); // lean with the wind
d.pos += d.vel * dt;
let g = groundAt(d.z);                       // depth decides where "the ground" is
if (d.pos.y > g) {
  d.age = d.life;                            // kill the drop
  let hit = vec2f(d.pos.x, g);
  for (var k = 0u; k < 2u + u32(rnd() * 2.0); k++) {
    spawn(newDrop(K_SPLASH, hit, vec2f((rnd() - 0.5) * 260.0, -120.0 - 200.0 * rnd()) * s, d.z, 0.3));
  }
  if (puddle(hit.x, d.z) > 0.3) { spawn(newDrop(K_RIPPLE, hit, vec2f(0.0), d.z, 0.7)); }
}`,
      },
      {
        title: 'Snow piles up: one atomic counter per screen column',
        lang: 'wgsl',
        src: `@group(0) @binding(4) var<storage, read_write> snowCols: array<atomic<u32>>;

let surf = u.accY - f32(atomicLoad(&snowCols[col])) * u.accK;
if (d.pos.y > surf) {
  atomicAdd(&snowCols[col], 1u);            // safe even if 100 flakes land at once
  d.kind = K_SETTLED;                       // fade out where it landed
}
// the backdrop fragment shader binds the same buffer read-only:
let hgt = mix(f32(snowCols[c0]), f32(snowCols[c1]), fract(fc)) * u.accK;`,
      },
      {
        title: 'Rain streak = motion blur along the velocity',
        lang: 'wgsl',
        src: `let sp = length(d.vel);
ax = d.vel / sp;                                            // quad's long axis
half = vec2f(sp * 0.011 + 3.0, mix(0.7, 2.0, d.z));        // length from speed, width from depth
col = light * mix(0.10, 0.32, d.z);                        // near drops brighter
// fragment: soft across, tapered along
a = exp(-q.y * q.y * 3.0) * (1.0 - q.x * q.x) * smoothstep(-1.0, -0.2, q.x);`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const MAX = ctx.testMode ? 1 << 14 : 1 << 18;
    const RATE = ctx.testMode ? 0.35 : 1;
    const U = gpu.uniforms(
      {
        dt: 'f32',
        time: 'f32',
        emitCount: 'u32',
        maxCount: 'u32',
        seed: 'u32',
        mode: 'u32',
        fill: 'u32',
        splashes: 'u32',
        accumulate: 'u32',
        world: 'vec2f',
        wind: 'f32',
        gust: 'f32',
        groundTop: 'f32',
        groundBot: 'f32',
        speedMul: 'f32',
        sizeMul: 'f32',
        accY: 'f32',
        accK: 'f32',
        flash: 'f32',
        boltOn: 'f32',
        fog: 'f32',
        exposure: 'f32',
        bloomAmt: 'f32',
        pxw: 'f32',
        ambient: 'vec3f',
        fogColor: 'vec4f',
        lamps: 'array<vec4f, 3>',
        bolt: 'array<vec4f, 9>',
      },
      'Weather',
    );
    const drops = gpu.storage(MAX * 48, 'weather-drops');
    const counters = gpu.storage(16, 'weather-counters');
    const snowCols = gpu.storage(COLS * 4, 'snow-columns');
    const drawArgs = gpu.storage(16, 'weather-args', GPUBufferUsage.INDIRECT);
    const sim = gpu.compute({
      label: 'weather-sim',
      include: ['noise'],
      bindings: {
        u: { uniform: U },
        ps: { storage: 'array<Drop>', access: 'read_write' },
        ctr: { storage: 'Counters', access: 'read_write' },
        drawArgs: { storage: 'array<u32, 4>', access: 'read_write' },
        snowCols: { storage: `array<atomic<u32>, ${COLS}>`, access: 'read_write' },
      },
      code: SIM_WGSL,
    });
    const strip = makeStrip(gpu);
    const draw = gpu.program({
      label: 'weather-draw',
      include: ['noise'],
      bindings: {
        u: { uniform: U },
        ps: { storage: 'array<Drop>', access: 'read' },
        strip: { texture: true },
        stripSamp: { sampler: true },
      },
      code: DRAW_WGSL,
    });
    const drawPipe = draw.renderPipeline({ format: 'rgba16float', blend: 'premultiplied' });
    const bg = gpu.fullscreen({
      label: 'weather-bg',
      uniforms: U,
      include: ['noise', 'sdf'],
      storage: { snowCols: `array<u32, ${COLS}>` },
      code: STRUCTS.replace(/struct Drop[\s\S]*?};/, '') + BG_WGSL,
      format: 'rgba16float',
    });
    const composite = gpu.fullscreen({ label: 'weather-composite', textures: ['hdr', 'bloomTex'], uniforms: U, include: ['color', 'hash', 'noise'], code: COMPOSITE_WGSL });
    const bloom = createBloom(gpu);
    let hdr = gpu.target(ctx.width, ctx.height, { format: 'rgba16float', label: 'weather-hdr' });

    const readout = tag(ctx);
    let seed = 1;
    let emitAcc = 0;
    let needReset = true;
    let simTime = 0;
    let rand = prng(3);
    // lightning state
    let strikeAt = -10;
    let nextStrike = 2.5;
    let boltPts = new Float32Array(36);
    const makeBolt = (W) => {
      const x0 = W * (0.15 + 0.7 * rand());
      let pts = [
        [x0, 60],
        [x0 + (rand() - 0.5) * 300, 640],
      ];
      let disp = 120;
      for (let lvl = 0; lvl < 4; lvl++) {
        const next = [];
        for (let i = 0; i < pts.length - 1; i++) {
          const [ax, ay] = pts[i];
          const [bx, by] = pts[i + 1];
          next.push(pts[i], [(ax + bx) / 2 + (rand() - 0.5) * disp * 2, (ay + by) / 2 + (rand() - 0.5) * disp * 0.3]);
        }
        next.push(pts[pts.length - 1]);
        pts = next;
        disp *= 0.55;
      }
      // 17 points -> pad to 18
      pts.push(pts[pts.length - 1]);
      for (let i = 0; i < 18; i++) {
        boltPts[i * 2] = pts[i][0];
        boltPts[i * 2 + 1] = pts[i][1];
      }
    };

    const SCENES = {
      rain: { groundTop: 760, groundBot: 1000, ambient: [0.05, 0.055, 0.07], fog: [0.06, 0.065, 0.08, 1] },
      snow: { groundTop: 700, groundBot: 1000, ambient: [0.55, 0.58, 0.68], fog: [0.45, 0.47, 0.52, 1] },
      leaves: { groundTop: 760, groundBot: 1000, ambient: [0.5, 0.36, 0.3], fog: [0.8, 0.45, 0.3, 1] },
      layers: { groundTop: 700, groundBot: 980, ambient: [0.5, 0.5, 0.5], fog: [0, 0, 0, 1] },
    };

    const step = (dt, p, fillCount = 0) => {
      const ex = ctx.example;
      const mode = MODES[ex] ?? 0;
      const sc = SCENES[ex] || SCENES.rain;
      const W = (1000 * ctx.width) / ctx.height;
      simTime += dt;
      // gust envelope: a few overlapping slow bumps
      const g = Math.pow(Math.max(0, Math.sin(simTime * 0.45) * 0.6 + Math.sin(simTime * 1.13 + 1.3) * 0.4), 2);
      const gust = p.gusts * g * Math.sign(p.wind || 1);
      let emitCount = fillCount;
      if (!fillCount) {
        emitAcc += p.amount * dt * (mode === 2 ? 1 : RATE);
        emitCount = Math.floor(emitAcc);
        emitAcc -= emitCount;
      }
      // lightning
      if (mode === 0 && p.storm > 0 && simTime > nextStrike) {
        strikeAt = simTime;
        makeBolt(W);
        nextStrike = simTime + (3 + 9 * rand()) / (0.2 + p.storm);
      }
      const ts = simTime - strikeAt;
      const flash = ts >= 0 && ts < 1.2 ? Math.exp(-ts * 7) + 0.7 * Math.exp(-Math.abs(ts - 0.16) * 14) * (ts > 0.12 ? 1 : 0) + 0.25 * Math.exp(-Math.abs(ts - 0.38) * 10) : 0;
      const boltOn = ts >= 0 && ts < 0.5 ? (ts < 0.07 || (ts > 0.14 && ts < 0.24) || (ts > 0.36 && ts < 0.42) ? 1 : 0.15) : 0;
      const lamps = mode === 0 ? [W * 0.2, 600, 0, 1, W * 0.56, 600, 0, 1, W * 0.9, 600, 0, 1] : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      U.set('dt', dt)
        .set('time', simTime)
        .set('emitCount', emitCount)
        .set('maxCount', MAX)
        .set('seed', seed++)
        .set('mode', mode)
        .set('fill', fillCount ? 1 : 0)
        .set('splashes', p.splashes ? 1 : 0)
        .set('accumulate', p.accumulate ? 1 : 0)
        .set('world', [W, 1000])
        .set('wind', p.wind)
        .set('gust', gust)
        .set('groundTop', sc.groundTop)
        .set('groundBot', sc.groundBot)
        .set('speedMul', p.speed ?? 1)
        .set('sizeMul', p.size ?? 1)
        .set('accY', 985)
        .set('accK', ctx.testMode ? 1.6 : 0.55)
        .set('flash', mode === 0 || mode === 3 ? flash : 0)
        .set('boltOn', mode === 0 ? boltOn : 0)
        .set('fog', p.fog ?? 0)
        .set('exposure', p.exposure)
        .set('bloomAmt', 0.35)
        .set('pxw', 1000 / ctx.height)
        .set('ambient', sc.ambient)
        .set('fogColor', sc.fog)
        .set('lamps', lamps)
        .set('bolt', boltPts);
      return emitCount;
    };

    const encodeSim = (enc, n) => {
      const res = { u: U, ps: drops, ctr: counters, drawArgs, snowCols };
      if (n > 0) sim.dispatch(enc, 'emit', Math.ceil(n / 64), res);
      sim.dispatch(enc, 'update', Math.ceil(MAX / 128), res);
      if (ctx.frame % 6 === 0) sim.dispatch(enc, 'meltSnow', Math.ceil(COLS / 64), res);
    };

    const reset = () => {
      const enc = device.createCommandEncoder();
      enc.clearBuffer(drops);
      enc.clearBuffer(counters);
      enc.clearBuffer(snowCols);
      device.queue.submit([enc.finish()]);
      simTime = 0;
      emitAcc = 0;
      rand = prng(3);
      nextStrike = 2.5;
      strikeAt = -10;
      // pre-fill the sky so the scene starts mid-storm: fall time ≈ height / speed
      const p = ctx.params;
      const mode = MODES[ctx.example] ?? 0;
      const fallTime = mode === 1 ? 1000 / (95 * (p.speed || 1)) : mode === 2 ? 4 : 1000 / (1500 * (p.speed || 1));
      const fill = Math.min(MAX * 0.8, Math.round(p.amount * (mode === 2 ? 1 : RATE) * fallTime));
      const n = step(1 / 60, p, Math.max(1, fill));
      U.upload();
      const e2 = device.createCommandEncoder();
      encodeSim(e2, n);
      device.queue.submit([e2.finish()]);
      if (mode === 0 && ctx.testMode) {
        // show a strike in the test screenshot
        nextStrike = 0.35;
      }
    };

    return {
      resize(w, h) {
        hdr.destroy();
        hdr = gpu.target(w, h, { format: 'rgba16float', label: 'weather-hdr' });
      },
      onAction(key) {
        if (key === 'reset') needReset = true;
        if (key === 'lightning') {
          strikeAt = simTime;
          makeBolt((1000 * ctx.width) / ctx.height);
        }
      },
      onExample() {
        needReset = true;
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        if (needReset) {
          needReset = false;
          reset();
        }
        const dt = Math.min(ctx.dt, 1 / 30);
        if (!ctx.paused && dt > 0) {
          const n = step(dt, p);
          U.upload();
          encodeSim(enc, n);
        } else {
          U.set('exposure', p.exposure).set('emitCount', 0).set('dt', 0);
          U.upload();
        }
        sim.dispatch(enc, 'args', 1, { u: U, ps: drops, ctr: counters, drawArgs, snowCols });
        bg.draw(enc, hdr, { snowCols }, { clear: [0, 0, 0, 1] });
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: hdr.view, loadOp: 'load', storeOp: 'store' }] });
        pass.setPipeline(drawPipe);
        pass.setBindGroup(0, draw.bind({ u: U, ps: drops, strip, stripSamp: gpu.sampler('linear-mip') }));
        pass.drawIndirect(drawArgs, 0);
        pass.end();
        const bl = bloom.render(enc, hdr, { threshold: 1.0, knee: 0.5 });
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { hdr, bloomTex: bl });
        const g = U.f32[U.map.gust.offset >> 2];
        readout.textContent = `wind ${(p.wind + g).toFixed(2)}${Math.abs(g) > 0.15 ? ' · gust!' : ''}`;
      },
    };
  },
};

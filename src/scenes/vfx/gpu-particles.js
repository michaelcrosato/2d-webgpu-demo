// GPU Particle Systems — a real emitter system that lives entirely on the GPU:
//   emit (compute)   : atomic ring-buffer allocation, sub-frame spawn positions
//   update (compute) : gravity, drag, curl-noise turbulence, attractors, ground bounce,
//                      GPU-side spawning (rocket trails, firework bursts, crackle) via atomics,
//                      alive count via a workgroup reduction
//   args (compute)   : writes the indirect draw arguments
//   draw             : instanced, velocity-stretched, textured soft quads into an HDR target
//   post             : bloom pyramid + ACES tone mapping (+ water reflection for the fireworks)

import { createGameScene } from '../../core/gamescene.js';
import { Camera2D, ShapeBatch } from '../../core/batch.js';
import { makeStrip, createBloom, TONEMAP_WGSL, tag, hexLin, prng, TEX } from './_shared.js';

const STRUCTS = /* wgsl */ `
struct Particle {
  pos: vec2f, vel: vec2f,
  age: f32, life: f32, size: f32, seed: f32,
  color: vec4f,   // rgb tint (linear, HDR), a = additive amount (1 additive, 0 alpha-blended)
  misc: vec4f,    // x rotation, y spin (or burst pattern), z texture cell, w kind
};
`;

const SIM_WGSL = /* wgsl */ `
${STRUCTS}
struct Counters { head: atomic<u32>, alive: atomic<u32>, p0: u32, p1: u32 };

const K_GENERIC: u32 = 0u;
const K_ROCKET: u32 = 1u;
const K_BURST: u32 = 2u;
const K_CRACKLE: u32 = 3u;
const K_TRAIL: u32 = 4u;
const K_EMBER: u32 = 5u;

var<private> rs: u32;
fn seedRng(s: u32) { rs = pcg(s ^ 0x9e3779b9u); }
fn rnd() -> f32 { rs = rs * 747796405u + 2891336453u; let w = ((rs >> ((rs >> 28u) + 4u)) ^ rs) * 277803737u; return f32((w >> 22u) ^ w) / 4294967296.0; }
fn rnd2() -> vec2f { return vec2f(rnd(), rnd()); }
fn dirOf(a: f32) -> vec2f { return vec2f(cos(a), sin(a)); }

fn spawn(p: Particle) {
  let slot = atomicAdd(&ctr.head, 1u) & (u.maxCount - 1u);
  ps[slot] = p;
}

fn mkParticle(pos: vec2f, vel: vec2f, life: f32, size: f32, color: vec4f, tex: f32, kind: u32) -> Particle {
  var p: Particle;
  p.pos = pos; p.vel = vel; p.age = 0.0; p.life = life; p.size = size; p.seed = rnd();
  p.color = color; p.misc = vec4f(rnd() * TAU, (rnd() - 0.5) * 6.0, tex, f32(kind));
  return p;
}

// ---------------------------------------------------------------- emit: one thread per new particle
@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.emitCount) { return; }
  seedRng(i * 1973u + u.seed * 9277u + 17u);
  let f = (f32(i) + rnd()) / f32(max(u.emitCount, 1u));      // where in this frame it was born
  let born = mix(u.emitPrev, u.emitPos, f);                    // interpolate along the emitter path
  let ageOffset = (1.0 - f) * u.dt;                             // sub-frame age: no clumping
  var p: Particle;

  if (u.mode == 1u) {
    // fireworks: a rocket launched by the CPU (pattern in misc.y)
    p = mkParticle(u.emitPos, u.rocketVel, u.rocketLife, 9.0, u.rocketColor, ${TEX.soft}.0, K_ROCKET);
    p.misc.y = u.pattern;
    ps[atomicAdd(&ctr.head, 1u) & (u.maxCount - 1u)] = p;
    return;
  }

  var tex = f32(u.texCell);
  var kind = K_GENERIC;
  var pos = born;
  var vel: vec2f;
  var size = u.size * (0.6 + 0.8 * rnd());
  var life = u.life * (0.7 + 0.6 * rnd());
  var color = vec4f(1.0, 1.0, 1.0, u.additive);

  if (u.mode == 2u) {
    let a = rnd() * TAU;
    if (i < u.split) {
      // aura: born on a ring around the caster, moving tangentially + drifting up
      let r = u.emitRadius * (0.75 + 0.5 * rnd());
      pos = born + dirOf(a) * r;
      vel = vec2f(-sin(a), cos(a)) * u.speed * (0.7 + 0.6 * rnd()) + vec2f(0.0, -20.0 - 50.0 * rnd());
      tex = select(${TEX.soft}.0, ${TEX.star}.0, rnd() < 0.25);
      size = select(size, size * 2.2, tex > 1.5);
    } else {
      // cast burst: a fast expanding ring of stars
      pos = born + dirOf(a) * 10.0;
      vel = dirOf(a) * (700.0 + 500.0 * rnd());
      life = 0.6 + 0.5 * rnd();
      tex = ${TEX.star}.0;
      size = u.size * 2.5;
      color = vec4f(1.6, 1.6, 2.2, 1.0);
    }
  } else if (u.mode == 3u && i >= u.split) {
    // forge embers: born across the coal bed, buoyant
    kind = K_EMBER;
    pos = u.emitPos2 + vec2f((rnd() - 0.5) * u.emitRadius, -rnd() * 10.0);
    vel = vec2f((rnd() - 0.5) * 40.0, -40.0 - 80.0 * rnd());
    life = 1.8 + 2.5 * rnd();
    size = 3.0 + 4.0 * rnd();
    tex = ${TEX.soft}.0;
  } else {
    // point / cone emitter
    let a = radians(u.emitDir + (rnd() - 0.5) * u.spread);
    vel = dirOf(a) * u.speed * (0.65 + 0.7 * rnd());
    if (u.emitRadius > 0.0) { pos += dirOf(rnd() * TAU) * u.emitRadius * sqrt(rnd()); }
  }
  p = mkParticle(pos + vel * ageOffset, vel, life, size, color, tex, kind);
  p.age = ageOffset;
  let slot = atomicAdd(&ctr.head, 1u) & (u.maxCount - 1u);
  ps[slot] = p;
}

// ---------------------------------------------------------------- firework burst (runs in the dying rocket's thread)
fn heart(t: f32) -> vec2f {
  let s = sin(t);
  return vec2f(16.0 * s * s * s, -(13.0 * cos(t) - 5.0 * cos(2.0 * t) - 2.0 * cos(3.0 * t) - cos(4.0 * t))) / 17.0;
}

fn explode(r: Particle) {
  let n = u32(u.burst);
  let pattern = u32(r.misc.y);
  let tilt = 0.25 + 0.75 * rnd();
  let rot = rnd() * TAU;
  let cr = cos(rot); let sr = sin(rot);
  let second = vec4f(hsv2rgb(vec3f(rnd(), 0.7, 1.0)) * 3.0, 1.0);
  for (var k = 0u; k < 4000u; k++) {
    if (k >= n) { break; }
    var v: vec2f;
    var speed = u.speed;
    var life = 1.3 + 0.9 * rnd();
    var col = r.color;
    var flags = 0.0;
    if (pattern == 0u) {
      // peony: a 3D sphere shell seen from the side -> denser at the rim
      let z = rnd() * 2.0 - 1.0;
      v = dirOf(rnd() * TAU) * sqrt(1.0 - z * z) * (0.92 + 0.08 * rnd());
    } else if (pattern == 1u) {
      // tilted ring
      let a = rnd() * TAU;
      let q = vec2f(cos(a), sin(a) * tilt) * (0.97 + 0.06 * rnd());
      v = vec2f(q.x * cr - q.y * sr, q.x * sr + q.y * cr);
    } else if (pattern == 2u) {
      // willow: long-lived golden drooping strands
      let z = rnd() * 2.0 - 1.0;
      v = dirOf(rnd() * TAU) * sqrt(1.0 - z * z) * 0.8;
      life = 2.6 + 1.4 * rnd();
      col = vec4f(2.6, 1.5, 0.45, 1.0);
      flags = 2.0;
    } else if (pattern == 3u) {
      // two-tone: inner sphere in a second color
      let z = rnd() * 2.0 - 1.0;
      let inner = rnd() < 0.4;
      v = dirOf(rnd() * TAU) * sqrt(1.0 - z * z) * select(1.0, 0.5, inner);
      if (inner) { col = second; }
    } else {
      // heart
      let h = heart(rnd() * TAU);
      v = vec2f(h.x * cr * 0.3 + h.x * 0.7, h.y) * (0.95 + 0.1 * rnd());
    }
    var p = mkParticle(r.pos, r.vel * 0.15 + v * speed, life, 5.0 + 3.0 * rnd(), col, ${TEX.spark}.0, K_BURST);
    // misc.y holds flags for sparks: 1 = crackles at the end, 2 = heavy drag (willow)
    if ((u.flags & 1u) != 0u && rnd() < 0.35 && pattern != 2u) { flags += 1.0; }
    p.misc.y = flags;
    spawn(p);
  }
  // the flash of the burst itself
  var f = mkParticle(r.pos, vec2f(0.0), 0.35, 160.0, vec4f(r.color.rgb * 1.2 + vec3f(1.0), 1.0), ${TEX.soft}.0, K_CRACKLE);
  f.misc.y = 9.0;
  spawn(f);
}

// ---------------------------------------------------------------- update: one thread per slot
@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < u.maxCount) {
    var p = ps[i];
    if (p.age < p.life) {
      seedRng(i * 7919u + u.seed * 104729u);
      let dt = u.dt;
      let kind = u32(p.misc.w);
      var acc = u.gravity;
      var drag = u.drag;
      if (kind == K_EMBER) { acc = vec2f(0.0, -70.0); drag = 1.2; }
      if (kind == K_BURST) { drag = select(u.drag, u.drag * 2.2, p.misc.y >= 2.0); acc *= 0.6; }
      if (kind == K_TRAIL || kind == K_CRACKLE) { acc *= 0.15; drag = 2.0; }
      if (kind == K_ROCKET) { drag = 0.0; }

      // curl-noise turbulence: swirly, divergence-free flow (never clumps into sinks)
      let turb = select(u.turb, max(u.turb, 450.0), kind == K_EMBER);
      if (turb > 0.0) {
        acc += curl(p.pos * u.turbScale + vec2f(p.seed * 0.3, u.time * 0.25)) * turb;
      }
      // attractor
      if (u.mode == 2u && kind == K_GENERIC && p.color.a > 0.5 && length(p.color.rgb - vec3f(1.0)) < 0.01) {
        // aura: spring toward a ring of radius R around the caster + upward lift = spirals
        let d = p.pos - u.attractor;
        let r = max(length(d), 1.0);
        let n = d / r;
        acc += -n * (r - u.emitRadius) * 14.0 * u.attract + vec2f(-n.y, n.x) * 60.0 * u.attract;
        acc.y -= 35.0;
      } else if (u.attract != 0.0 && kind == K_GENERIC) {
        let d = u.attractor - p.pos;
        let r2 = dot(d, d) + 2500.0;
        acc += d * (u.attract * 3.0e6 / (r2 * sqrt(r2)));
        acc += vec2f(-d.y, d.x) * (u.attract * 0.8e6 / (r2 * sqrt(r2)));
      }

      p.vel += acc * dt;
      p.vel *= exp(-drag * dt);                // frame-rate independent drag
      p.pos += p.vel * dt;
      p.misc.x += p.misc.y * dt * select(1.0, 0.0, kind == K_BURST || kind == K_ROCKET);

      // ground collision: reflect, lose energy, add friction
      if (u.bounce > 0.0 && p.pos.y > u.groundY && kind != K_ROCKET) {
        p.pos.y = u.groundY - (p.pos.y - u.groundY) * u.bounce;
        if (p.vel.y > 0.0) {
          p.vel.y = -p.vel.y * u.bounce * (0.7 + 0.6 * rnd());
          p.vel.x = p.vel.x * (1.0 - 0.3 * u.bounce) + (rnd() - 0.5) * 40.0 * u.bounce;
        }
      }

      // GPU-side spawning (fireworks)
      if (kind == K_ROCKET) {
        if (rnd() < min(1.0, dt * 120.0)) {
          spawn(mkParticle(p.pos + rnd2() * 4.0, -p.vel * 0.1 + (rnd2() - 0.5) * 40.0, 0.35 + 0.4 * rnd(), 6.0, vec4f(2.5, 1.2, 0.4, 1.0), ${TEX.soft}.0, K_TRAIL));
        }
        if (p.age + dt >= p.life) { explode(p); }
      }
      if (kind == K_BURST) {
        if ((u.flags & 2u) != 0u && rnd() < dt * select(10.0, 40.0, p.misc.y >= 2.0) && p.age < p.life * 0.85) {
          spawn(mkParticle(p.pos, p.vel * 0.05, 0.4 + 0.5 * rnd(), p.size * 0.7, p.color, ${TEX.soft}.0, K_TRAIL));
        }
        if (p.misc.y == 1.0 && p.age + dt >= p.life) {
          for (var k = 0u; k < 5u; k++) {
            spawn(mkParticle(p.pos, (rnd2() - 0.5) * 220.0, 0.1 + 0.25 * rnd(), 10.0, vec4f(1.0), ${TEX.star}.0, K_CRACKLE));
          }
        }
      }
      p.age += dt;
      ps[i] = p;
      // live count for the readout (only on frames the CPU reads it back)
      if (u.countAlive != 0u && p.age < p.life) { atomicAdd(&ctr.alive, 1u); }
    }
  }
}

@compute @workgroup_size(1)
fn resetAlive() { atomicStore(&ctr.alive, 0u); }

// indirect draw arguments: draw only slots that have ever been used
@compute @workgroup_size(1)
fn args() {
  let used = min(atomicLoad(&ctr.head), u.maxCount);
  drawArgs[0] = 6u; drawArgs[1] = used; drawArgs[2] = 0u; drawArgs[3] = 0u;
}
`;

const DRAW_WGSL = /* wgsl */ `
${STRUCTS}
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) col: vec4f,   // rgb, a = alpha
  @location(2) @interpolate(flat) info: vec2f,  // x tex cell, y additive
};

fn ramp(t: f32) -> vec4f {
  // 3-key color-over-life gradient (rgb) with alpha fade in/out
  let c = select(mix(u.c1.rgb, u.c2.rgb, (t - u.rampMid) / (1.0 - u.rampMid)), mix(u.c0.rgb, u.c1.rgb, t / u.rampMid), t < u.rampMid);
  let a = smoothstep(0.0, 0.04, t) * (1.0 - smoothstep(0.55, 1.0, t));
  return vec4f(c, a);
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let p = ps[ii];
  if (p.age >= p.life || p.life <= 0.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  let t = clamp(p.age / p.life, 0.0, 1.0);
  let kind = u32(p.misc.w);
  var col = vec3f(1.0);
  var alpha = 1.0;
  var size = p.size;
  var stretch = 0.0;
  var tex = p.misc.z;
  var additive = p.color.a;
  let flick = hash11(p.seed * 91.7 + floor(u.time * 30.0));

  if (kind == ${0}u) {
    let r = ramp(t);
    col = r.rgb * p.color.rgb; alpha = r.a;
    size = p.size * mix(1.0, u.sizeEnd / max(u.size, 0.001), t);
    stretch = u.stretch;
  } else if (kind == 1u) {
    col = vec3f(5.0, 3.0, 1.2); size = 7.0; stretch = 0.025; tex = ${TEX.spark}.0;
  } else if (kind == 2u) {
    let hot = 1.0 - smoothstep(0.0, 0.15, t);
    col = mix(p.color.rgb, vec3f(4.0, 3.6, 3.0), hot);
    // late sparks flicker before dying
    col *= mix(1.0, 0.25 + 1.2 * flick, smoothstep(0.6, 0.9, t));
    alpha = 1.0 - smoothstep(0.7, 1.0, t);
    size = p.size * (1.0 - 0.4 * t);
    stretch = 0.045;
  } else if (kind == 3u) {
    if (p.misc.y > 8.0) { col = p.color.rgb * 0.5; alpha = (1.0 - t) * (1.0 - t); size = p.size * (0.6 + 0.6 * t); }
    else { col = vec3f(6.0, 5.0, 3.5) * step(0.35, flick); alpha = 1.0 - t; size = 14.0; }
  } else if (kind == 4u) {
    col = p.color.rgb * 0.45; alpha = (1.0 - t) * (1.0 - t); size = p.size * (1.0 - 0.5 * t);
  } else {
    // ember: orange -> deep red, flickering
    col = mix(vec3f(4.0, 1.6, 0.35), vec3f(1.2, 0.18, 0.04), t) * (0.6 + 0.6 * flick);
    alpha = smoothstep(0.0, 0.1, t) * (1.0 - smoothstep(0.6, 1.0, t));
  }

  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  // orient: velocity-aligned (stretched by speed) or free rotation
  let sp = length(p.vel);
  var ax = vec2f(cos(p.misc.x), sin(p.misc.x));
  var half = vec2f(size * 0.5);
  if (stretch > 0.0 && sp > 1.0) {
    ax = p.vel / sp;
    half.x = size * 0.5 + sp * stretch;
  }
  let ay = vec2f(-ax.y, ax.x);
  let w = p.pos + ax * c.x * half.x + ay * c.y * half.y;
  o.pos = vec4f(w.x / u.world.x * 2.0 - 1.0, 1.0 - w.y / u.world.y * 2.0, 0.0, 1.0);
  o.local = c * 0.5 + 0.5;
  o.col = vec4f(col, alpha);
  o.info = vec2f(tex, additive);
  return o;
}

@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let uv = vec2f((i.info.x + clamp(i.local.x, 0.01, 0.99)) / ${8}.0, i.local.y);
  let s = textureSample(strip, stripSamp, uv);
  let a = s.a * i.col.a;
  // premultiplied output: additive particles write alpha 0, alpha-blended ones write a
  return vec4f(s.rgb * i.col.rgb * a, a * (1.0 - i.info.y));
}
`;

const BG_WGSL = /* wgsl */ `
fn ring(d: f32, r: f32, w: f32) -> f32 { let x = (d - r) / w; return exp(-x * x); }

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let wpos = uv * u.world;               // world units (height 1000)
  let mode = u.mode;
  if (mode == 1u) {
    // ---- fireworks night: sky, stars, skyline, water
    var c = mix(vec3f(0.002, 0.003, 0.012), vec3f(0.03, 0.016, 0.05), smoothstep(0.0, 0.85, uv.y));
    let cell = floor(px / 3.0);
    let h = hash21(cell);
    if (h > 0.9975) { c += vec3f(0.6, 0.65, 0.8) * (0.4 + 0.6 * sin(u.time * 2.0 + h * 900.0)); }
    let water = u.waterY;
    // skyline silhouette
    let bx = floor(wpos.x / 46.0);
    let bh = 60.0 + 150.0 * hash11(bx * 3.1) * hash11(bx * 1.7 + 4.0);
    let top = water - bh;
    if (wpos.y > top && wpos.y < water) {
      c = vec3f(0.004, 0.004, 0.009);
      let wc = floor(vec2f(wpos.x / 9.0, (wpos.y - top) / 14.0));
      let lit = step(0.72, hash21(wc + bx * 13.0)) * step(4.0, fmod(wpos.x, 46.0)) * step(fmod(wpos.x, 9.0), 5.0) * step(fmod(wpos.y - top, 14.0), 7.0) * step(10.0, wpos.y - top);
      c += vec3f(0.5, 0.3, 0.08) * lit * 0.35;
    }
    if (wpos.y >= water) { c = vec3f(0.002, 0.004, 0.012); }
    return vec4f(c, 1.0);
  }
  if (mode == 2u) {
    // ---- the game scene at night, lit by the spell
    let g = TEX(game, uv).rgb;
    let lin = pow(g, vec3f(2.2));
    let lum = dot(lin, vec3f(0.3, 0.5, 0.2));
    var c = mix(lin, vec3f(lum), 0.55) * vec3f(0.10, 0.13, 0.26);
    let d = length(wpos - u.attractor);
    let light = exp(-d * d / (2.0 * 220.0 * 220.0)) * u.lightAmt;
    c += lin * mix(u.c0.rgb, u.c1.rgb, 0.5) * light * 0.9;
    // magic circle under the caster
    // a rotating rune circle on the ground under the caster (squashed for perspective)
    let q = wpos - (u.attractor + vec2f(0.0, 135.0));
    let a = atan2(q.y / 0.32, q.x) + u.time * 0.6;
    let rq = length(vec2f(q.x, q.y / 0.32));
    let R = u.emitRadius * 1.25;
    var glyph = ring(rq, R, 2.6) + ring(rq, R * 0.82, 1.6);
    glyph += smoothstep(0.6, 0.95, sin(a * 28.0)) * step(R * 0.82, rq) * step(rq, R) * 0.5;
    // hexagram: distance to a regular hexagon edge, twice rotated
    let hx = abs(fract(a * 3.0 / TAU) - 0.5) * TAU / 3.0;
    glyph += ring(rq * cos(hx) / cos(PI / 6.0), R * 0.7, 1.4) * 0.9;
    let hy = abs(fract(a * 3.0 / TAU + 0.5) - 0.5) * TAU / 3.0;
    glyph += ring(rq * cos(hy) / cos(PI / 6.0), R * 0.7, 1.4) * 0.9;
    c += mix(u.c0.rgb, u.c1.rgb, 0.4) * glyph * 1.6;
    // the caster's core orb
    let dc = length(wpos - u.attractor);
    c += u.c0.rgb * (exp(-dc / 9.0) * 4.0 + exp(-dc / 40.0) * 0.5);
    return vec4f(c, 1.0);
  }
  if (mode == 3u) {
    // ---- workshop: dark brick wall, concrete floor, warm light from the sparks
    var c = mix(vec3f(0.012, 0.010, 0.012), vec3f(0.03, 0.022, 0.02), uv.y);
    var bq = wpos / vec2f(64.0, 28.0);
    bq.x += 0.5 * floor(bq.y);
    let bf = fract(bq);
    let mortar = smoothstep(0.0, 0.06, min(min(bf.x, 1.0 - bf.x) * 0.45, min(bf.y, 1.0 - bf.y)));
    c *= 0.7 + 0.3 * mortar * (0.8 + 0.4 * hash21(floor(bq)));
    if (wpos.y > u.groundY) {
      c = vec3f(0.02, 0.019, 0.018) * (0.85 + 0.3 * hash21(floor(wpos / 3.0)) * 0.3);
      c *= 1.0 - 0.4 * smoothstep(u.groundY, 1000.0, wpos.y);
    }
    let dl = length((wpos - u.emitPos) * vec2f(0.7, 1.0));
    c += vec3f(1.0, 0.45, 0.12) * exp(-dl / 160.0) * 0.12 * u.lightAmt;
    let df = length((wpos - u.emitPos2) * vec2f(0.6, 1.0));
    c += vec3f(1.0, 0.35, 0.08) * exp(-df / 180.0) * 0.12;
    return vec4f(c, 1.0);
  }
  // ---- playground: dark gradient, grid, ground, emitter & attractor markers
  var c = mix(vec3f(0.006, 0.007, 0.016), vec3f(0.022, 0.014, 0.034), uv.y);
  let gp = abs(fract(wpos / 100.0 + 0.5) - 0.5) * 100.0;
  c += vec3f(0.012, 0.014, 0.03) * (1.0 - smoothstep(0.0, 1.4 * u.pxw, min(gp.x, gp.y)));
  if (u.bounce > 0.0 && wpos.y > u.groundY) {
    c = vec3f(0.018, 0.016, 0.03) * (1.0 - 0.5 * smoothstep(u.groundY, 1000.0, wpos.y));
  }
  if (u.bounce > 0.0) { c += vec3f(0.25, 0.22, 0.4) * exp(-abs(wpos.y - u.groundY) / (1.2 * u.pxw)) * 0.5; }
  let de = length(wpos - u.emitPos);
  c += vec3f(0.5, 0.5, 0.7) * ring(de, 14.0, 1.5 * u.pxw) * 0.6;
  if (abs(u.attract) > 0.001) {
    let da = length(wpos - u.attractor);
    let ac = select(vec3f(1.0, 0.3, 0.25), vec3f(0.3, 0.8, 1.0), u.attract > 0.0);
    c += ac * (ring(da, 22.0, 2.0 * u.pxw) * 0.9 + exp(-da / 30.0) * 0.15);
  }
  return vec4f(c, 1.0);
}`;

const COMPOSITE_WGSL = /* wgsl */ `
${TONEMAP_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var c = TEX(hdr, uv).rgb;
  var b = TEX(bloomTex, uv).rgb;
  if (u.mode == 1u) {
    let wl = u.waterY / u.world.y;
    if (uv.y > wl) {
      // water: mirror the sky about the waterline with a ripple wobble, darker further down
      let depth = uv.y - wl;
      let wob = sin(uv.y * 900.0 - u.time * 3.0) * 0.0025 + sin(uv.y * 310.0 + u.time * 1.7 + uv.x * 40.0) * 0.003;
      let ruv = vec2f(uv.x + wob * (0.3 + depth * 12.0), wl - depth * 1.15);
      let refl = TEX(hdr, ruv).rgb + TEX(bloomTex, ruv).rgb * u.bloom;
      c += refl * (0.42 - depth * 1.4) * (0.8 + 0.2 * sin(uv.x * 600.0 + uv.y * 2000.0));
    }
  }
  c += b * u.bloom;
  let v = uv - 0.5;
  c *= 1.0 - dot(v, v) * 0.7;
  return vec4f(vfxTonemap(c * u.exposure, px), 1.0);
}`;

const MODES = { playground: 0, fireworks: 1, magic: 2, sparks: 3 };
const SHAPES = { soft: TEX.soft, spark: TEX.spark, star: TEX.star, smoke: TEX.smoke, ring: TEX.ring };

export default {
  interaction: 'Drag on the canvas to move the emitter. Everything you see is simulated by compute shaders.',
  examples: [
    {
      id: 'playground',
      label: 'Particle playground',
      kind: 'Abstract',
      note: 'One emitter, every knob. Drag to move the emitter. Try the attractor, turbulence and ground-bounce sliders, and switch the texture and blending.',
      hint: 'Drag to move the emitter. Use the sliders — every particle is a GPU thread.',
      params: { rate: 14000, life: 2.8, speed: 650, spread: 30, dir: -90, gravity: 900, drag: 0.3, turb: 0, attract: 0, bounce: 0.5, size: 9, sizeEnd: 3, stretch: 0, colA: '#ffe08a', colB: '#ff4f8b', colC: '#3b2bff', shape: 'soft', additive: 1, exposure: 1, bloom: 0.9 },
    },
    {
      id: 'fireworks',
      label: 'Fireworks',
      kind: 'In a game',
      note: 'Rockets are particles too. Each rocket <b>spawns its own trail on the GPU</b>, and when its lifetime ends the dying thread allocates hundreds of sparks with an atomic counter — peony, ring, willow, two-tone and heart patterns. Some sparks crackle into glitter. Click to launch at the cursor.',
      hint: 'Click anywhere to launch a rocket that bursts at that spot.',
      params: { rockets: 1.3, burst: 520, speed: 300, gravity: 260, drag: 1.1, crackle: true, trails: true, exposure: 1.1, bloom: 1.0 },
    },
    {
      id: 'magic',
      label: 'Magic spell / aura',
      kind: 'In a game',
      note: 'An aura around the caster: particles are born on a ring, orbit with a spring force pulling them back to the ring, and drift upward — spirals. Their light also tints the night scene behind them (a cheap fake light). Click to cast a burst.',
      hint: 'Move the mouse to move the caster. Click to cast a burst.',
      params: { rate: 5000, life: 2.2, speed: 170, attract: 1, turb: 160, size: 7, sizeEnd: 1.5, colA: '#9ffcff', colB: '#a46bff', colC: '#2a0b6e', exposure: 1.1, bloom: 1.1 },
    },
    {
      id: 'sparks',
      label: 'Sparks & embers',
      kind: 'Real life',
      note: 'A grinder throws hot sparks: fast, <b>velocity-stretched</b> particles that cool from white to deep red, fall under gravity and <b>bounce</b> off the floor with friction. Embers from the forge on the right are buoyant and ride curl-noise turbulence.',
      hint: 'Drag to move the grinder. Hold the button to press harder (3× sparks).',
      params: { rate: 1600, life: 1.3, speed: 950, spread: 28, dir: 4, gravity: 1500, drag: 0.4, bounce: 0.35, stretch: 0.035, size: 5, exposure: 1.1, bloom: 0.9 },
    },
  ],
  controls: [
    { type: 'heading', label: 'Emitter' },
    { type: 'slider', key: 'rate', label: 'Emission rate', min: 100, max: 400000, step: 100, value: 20000, log: true, format: (v) => `${Math.round(v).toLocaleString()}/s`, help: 'New particles per second. Each new one is written by its own compute thread.', showFor: ['playground', 'magic', 'sparks'] },
    { type: 'slider', key: 'rockets', label: 'Rockets per second', min: 0, max: 5, step: 0.1, value: 1.3, help: 'Automatic launches (click to launch your own).', showFor: ['fireworks'] },
    { type: 'slider', key: 'burst', label: 'Sparks per burst', min: 50, max: 3000, step: 10, value: 520, log: true, format: (v) => Math.round(v), help: 'Allocated on the GPU by the dying rocket via an atomic counter.', showFor: ['fireworks'] },
    { type: 'slider', key: 'life', label: 'Lifetime (s)', min: 0.2, max: 8, step: 0.05, value: 2.4, help: 'Age → 0..1 drives the color and size ramps.', showFor: ['playground', 'magic', 'sparks'] },
    { type: 'slider', key: 'speed', label: 'Initial speed', min: 0, max: 1600, step: 5, value: 620, help: 'Units per second (screen height = 1000 units).' },
    { type: 'slider', key: 'dir', label: 'Direction (°)', min: -180, max: 180, step: 1, value: -90, showFor: ['playground', 'sparks'] },
    { type: 'slider', key: 'spread', label: 'Spread (°)', min: 0, max: 360, step: 1, value: 34, help: 'Cone angle. 360 = all directions.', showFor: ['playground', 'sparks'] },
    { type: 'heading', label: 'Forces' },
    { type: 'slider', key: 'gravity', label: 'Gravity', min: -800, max: 2500, step: 10, value: 700, help: 'Negative = particles float up (smoke, bubbles).', showFor: ['playground', 'fireworks', 'sparks'] },
    { type: 'slider', key: 'drag', label: 'Drag', min: 0, max: 5, step: 0.01, value: 0.35, help: 'Air resistance: velocity *= exp(-drag·dt).', showFor: ['playground', 'fireworks', 'sparks'] },
    { type: 'slider', key: 'turb', label: 'Curl-noise turbulence', min: 0, max: 3000, step: 10, value: 0, help: 'Swirly flow field from the curl of Perlin noise.', showFor: ['playground', 'magic'] },
    { type: 'slider', key: 'attract', label: 'Attractor', min: -3, max: 3, step: 0.01, value: 0, help: 'Pull (+) or push (−). In the playground the attractor orbits the screen.', showFor: ['playground', 'magic'] },
    { type: 'slider', key: 'bounce', label: 'Ground bounce', min: 0, max: 1, step: 0.01, value: 0.45, help: '0 = no ground. Otherwise the fraction of speed kept when bouncing.', showFor: ['playground', 'sparks'] },
    { type: 'toggle', key: 'crackle', label: 'Crackle', value: true, help: 'Some sparks pop into glitter when they die (more GPU spawning).', showFor: ['fireworks'] },
    { type: 'toggle', key: 'trails', label: 'Spark trails', value: true, help: 'Every spark keeps spawning short-lived trail particles.', showFor: ['fireworks'] },
    { type: 'heading', label: 'Look' },
    { type: 'slider', key: 'size', label: 'Start size', min: 1, max: 60, step: 0.5, value: 11, showFor: ['playground', 'magic', 'sparks'] },
    { type: 'slider', key: 'sizeEnd', label: 'End size', min: 0, max: 60, step: 0.5, value: 3, help: 'Size-over-life: interpolated from start to end.', showFor: ['playground', 'magic'] },
    { type: 'slider', key: 'stretch', label: 'Velocity stretch', min: 0, max: 0.08, step: 0.001, value: 0, help: 'Stretch quads along their velocity — fake motion blur.', showFor: ['playground', 'sparks'] },
    { type: 'color', key: 'colA', label: 'Color: birth', value: '#ffe08a', showFor: ['playground', 'magic'] },
    { type: 'color', key: 'colB', label: 'Color: middle', value: '#ff4f8b', showFor: ['playground', 'magic'] },
    { type: 'color', key: 'colC', label: 'Color: death', value: '#3b2bff', showFor: ['playground', 'magic'] },
    {
      type: 'select',
      key: 'shape',
      label: 'Texture',
      value: 'soft',
      options: [
        { value: 'soft', label: 'Soft glow' },
        { value: 'spark', label: 'Spark streak' },
        { value: 'star', label: 'Twinkle star' },
        { value: 'smoke', label: 'Smoke puff' },
        { value: 'ring', label: 'Ring' },
      ],
      showFor: ['playground'],
    },
    { type: 'slider', key: 'additive', label: 'Additive ↔ alpha', min: 0, max: 1, step: 0.01, value: 1, help: '1 = light adds up (fire, magic). 0 = normal transparency (smoke, dust). Both in one batch thanks to premultiplied alpha.', showFor: ['playground'] },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: 0.1, max: 4, step: 0.01, value: 1, help: 'Particles add up in a float (HDR) buffer; tone mapping squeezes it to the screen.' },
    { type: 'slider', key: 'bloom', label: 'Bloom', min: 0, max: 3, step: 0.01, value: 0.9, help: 'Bright pixels bleed light (see Lighting → Bloom).' },
    { type: 'button', key: 'reset', label: 'Clear particles', primary: true },
  ],
  about: {
    summary: 'A complete particle engine on the GPU: emitters, lifetimes, color & size over life, forces, collisions and particles that spawn particles — for up to a million at a time.',
    what: `<p>Four uses of one GPU particle system. Each particle is a 64-byte struct in a <b>storage buffer</b>; compute shaders create, move and kill them,
      and an instanced draw call renders every live one as a textured quad. The CPU only says “emit 300 particles here this frame”.</p>`,
    how: `<ol>
      <li><b>Pool & ring buffer</b>: a fixed array of N slots (a power of two). To spawn, a thread does <code>slot = atomicAdd(&amp;head, 1) &amp; (N − 1)</code>:
        an <b>atomic counter</b> hands out slots without two threads ever getting the same one, wrapping around like a ring. The oldest particles are overwritten first.</li>
      <li><b>Emit pass</b>: one thread per new particle. It picks a random direction inside the cone, speed, lifetime and size, and is placed
        <i>along the path the emitter moved this frame</i> with a sub-frame age — so fast emitters leave a continuous stream instead of clumps.</li>
      <li><b>Update pass</b>: one thread per slot. Dead slots return immediately. Live ones integrate forces: gravity, drag
        (<code>v *= exp(−drag·dt)</code>, frame-rate independent), <b>curl noise</b> (the rotated gradient of Perlin noise — a swirling flow with no sinks),
        attractors, and a ground plane that reflects velocity with a restitution factor.</li>
      <li><b>GPU spawning</b>: fireworks rockets spawn trail particles every step, and on death run a loop that allocates hundreds of sparks with the same atomic counter. Sparks can spawn crackle. No CPU round trip.</li>
      <li><b>Draw</b>: a tiny compute pass writes <i>indirect draw arguments</i> (how many slots were ever used), then <code>drawIndirect</code> renders 6 vertices per slot.
        The vertex shader reads the particle, evaluates the <b>color-over-life</b> and <b>size-over-life</b> ramps and optionally stretches the quad along the velocity.</li>
      <li><b>HDR + premultiplied alpha</b>: quads are blended into a 16-bit float target with <code>one, one−srcAlpha</code>. Additive particles output alpha 0, smoke outputs its alpha — both kinds in one draw. Then bloom and ACES tone mapping.</li>
    </ol>`,
    uses: [
      { title: 'Every action game', text: 'Muzzle flashes, impacts, blood, dust, magic — Diablo, Hades, Vampire Survivors and Nova Drift live on particles.' },
      { title: 'Celebrations', text: 'Fireworks, confetti and level-up bursts; GPU spawning makes “particles that spawn particles” cheap.' },
      { title: 'Ambience', text: 'Embers, dust motes, fireflies and pollen drifting on curl noise give scenes life (Ori, Hollow Knight).' },
      { title: 'Feedback', text: 'Sparks that bounce off floors and walls sell physicality: grinders, sword clashes, bullet hits.' },
    ],
    try: [
      'In the <b>playground</b>, set <i>Ground bounce</i> to 0.9 and <i>Gravity</i> to 2000 — then switch the texture to <i>Spark streak</i> and raise <i>Velocity stretch</i>.',
      'Turn <i>Curl-noise turbulence</i> up to ~1200 with gravity 0: the fountain becomes smoke-like swirls.',
      'Set <i>Attractor</i> to 2 and watch particles slingshot around the orbiting attractor; −2 pushes them away.',
      'Choose <i>Smoke puff</i>, set <i>Additive ↔ alpha</i> to 0, gravity −200, and pick grey colors: instant smoke.',
      'In <b>Fireworks</b>, raise <i>Sparks per burst</i> to 3000 and click rapidly — watch the live particle counter.',
    ],
    ask: [
      'GPU compute particle system with an atomic ring-buffer allocator',
      'color-over-life and size-over-life ramps for particles',
      'curl-noise turbulence for smoke and embers',
      'particles that bounce off the ground with restitution and friction',
      'fireworks where the burst particles are spawned on the GPU',
      'velocity-stretched spark particles with additive HDR blending',
    ],
    perf: `<p>The update costs one thread per <i>slot</i> (≈ 1 M per millisecond on a mid-range GPU, more with curl noise). The real limit is <b>fill rate</b>:
      big, overlapping additive quads touch many pixels — 100 k small sparks are cheaper than 10 k huge smoke puffs.
      Indirect drawing skips never-used slots; a production engine would also compact live particles into a list. Alpha-blended particles strictly need
      back-to-front sorting; the ring buffer’s age order is a good-enough approximation here.</p>`,
    api: `<p><b>WebGPU only.</b> Compute passes, read/write storage buffers, <code>atomicAdd</code> and indirect draws don’t exist in WebGL2.
      There you would keep particles in float textures and update them with a fragment shader (“ping-pong”), and spawning from the GPU would be very awkward.</p>`,
    code: [
      {
        title: 'Spawning: an atomic counter hands out ring-buffer slots',
        lang: 'wgsl',
        src: `struct Counters { head: atomic<u32>, alive: atomic<u32>, p0: u32, p1: u32 };

fn spawn(p: Particle) {
  let slot = atomicAdd(&ctr.head, 1u) & (u.maxCount - 1u); // unique, wraps around
  ps[slot] = p;
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.emitCount) { return; }
  let f = (f32(i) + rnd()) / f32(u.emitCount);    // when in this frame it was born
  let born = mix(u.emitPrev, u.emitPos, f);       // along the emitter's path
  let a = radians(u.emitDir + (rnd() - 0.5) * u.spread);
  let vel = vec2f(cos(a), sin(a)) * u.speed * (0.65 + 0.7 * rnd());
  ...
}`,
      },
      {
        title: 'Update: forces, curl noise and bouncing',
        lang: 'wgsl',
        src: `var acc = u.gravity;
acc += curl(p.pos * u.turbScale + vec2f(p.seed * 0.3, u.time * 0.25)) * u.turb;
p.vel += acc * dt;
p.vel *= exp(-drag * dt);              // frame-rate independent drag
p.pos += p.vel * dt;
if (u.bounce > 0.0 && p.pos.y > u.groundY) {
  p.pos.y = u.groundY - (p.pos.y - u.groundY) * u.bounce;
  if (p.vel.y > 0.0) { p.vel.y = -p.vel.y * u.bounce; p.vel.x *= 1.0 - 0.3 * u.bounce; }
}
if (kind == K_ROCKET && p.age + dt >= p.life) { explode(p); } // spawns hundreds of sparks
p.age += dt;`,
      },
      {
        title: 'Vertex shader: color & size over life, velocity stretch',
        lang: 'wgsl',
        src: `let t = clamp(p.age / p.life, 0.0, 1.0);
let r = ramp(t);                                   // 3-key gradient + fade in/out
col = r.rgb * p.color.rgb;
size = p.size * mix(1.0, u.sizeEnd / u.size, t);
var ax = vec2f(cos(p.misc.x), sin(p.misc.x));      // free rotation...
var half = vec2f(size * 0.5);
if (stretch > 0.0 && speed > 1.0) {                // ...or aligned to velocity
  ax = p.vel / speed;
  half.x = size * 0.5 + speed * stretch;
}`,
      },
    ],
    links: [
      { title: 'Wicked Engine — GPU-based particle simulation', url: 'https://wickedengine.net/2017/11/gpu-based-particle-simulation/', note: 'dead/alive lists and indirect draws' },
      { title: 'Robert Bridson — Curl-noise for procedural fluid flow', url: 'https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const MAX = ctx.testMode ? 1 << 14 : 1 << 20;
    // the headless test GPU is a CPU rasterizer: emit fewer particles there
    const RATE = ctx.testMode ? 0.5 : 1;
    const U = gpu.uniforms(
      {
        dt: 'f32',
        time: 'f32',
        emitCount: 'u32',
        maxCount: 'u32',
        seed: 'u32',
        countAlive: 'u32',
        mode: 'u32',
        split: 'u32',
        flags: 'u32',
        texCell: 'u32',
        additive: 'f32',
        world: 'vec2f',
        emitPos: 'vec2f',
        emitPrev: 'vec2f',
        emitPos2: 'vec2f',
        attractor: 'vec2f',
        gravity: 'vec2f',
        emitDir: 'f32',
        spread: 'f32',
        speed: 'f32',
        life: 'f32',
        size: 'f32',
        sizeEnd: 'f32',
        drag: 'f32',
        turb: 'f32',
        turbScale: 'f32',
        attract: 'f32',
        groundY: 'f32',
        bounce: 'f32',
        emitRadius: 'f32',
        stretch: 'f32',
        rampMid: 'f32',
        burst: 'f32',
        rocketVel: 'vec2f',
        rocketLife: 'f32',
        pattern: 'f32',
        rocketColor: 'vec4f',
        c0: 'vec4f',
        c1: 'vec4f',
        c2: 'vec4f',
        waterY: 'f32',
        lightAmt: 'f32',
        pxw: 'f32',
        exposure: 'f32',
        bloom: 'f32',
      },
      'Sim',
    );
    const particles = gpu.storage(MAX * 64, 'particles');
    const counters = gpu.storage(16, 'counters');
    const drawArgs = gpu.storage(16, 'draw-args', GPUBufferUsage.INDIRECT);
    const staging = gpu.buffer({ size: 16, usage: GPUBufferUsage.MAP_READ, label: 'counter-readback' });

    const sim = gpu.compute({
      label: 'particles-sim',
      include: ['noise', 'color'],
      bindings: {
        u: { uniform: U },
        ps: { storage: 'array<Particle>', access: 'read_write' },
        ctr: { storage: 'Counters', access: 'read_write' },
        drawArgs: { storage: 'array<u32, 4>', access: 'read_write' },
      },
      code: SIM_WGSL,
    });
    const strip = makeStrip(gpu);
    const draw = gpu.program({
      label: 'particles-draw',
      include: ['hash'],
      bindings: {
        u: { uniform: U },
        ps: { storage: 'array<Particle>', access: 'read' },
        strip: { texture: true },
        stripSamp: { sampler: true },
      },
      code: DRAW_WGSL,
    });
    const drawPipe = draw.renderPipeline({ format: 'rgba16float', blend: 'premultiplied' });
    const bg = gpu.fullscreen({ label: 'particles-bg', textures: ['game'], uniforms: U, include: ['hash', 'noise'], code: BG_WGSL, format: 'rgba16float' });
    const composite = gpu.fullscreen({ label: 'particles-composite', textures: ['hdr', 'bloomTex'], uniforms: U, include: ['color', 'hash'], code: COMPOSITE_WGSL });
    const bloom = createBloom(gpu);
    const game = createGameScene(gpu);
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    let hdr = gpu.target(ctx.width, ctx.height, { format: 'rgba16float', label: 'hdr' });
    const blankGame = gpu.target(4, 4, { label: 'nogame' });

    const readout = tag(ctx);
    let alive = 0;
    let reading = false;
    let rand = prng(7);
    let seed = 1;
    let emitAcc = 0;
    let emberAcc = 0;
    let simTime = 0;
    let emitPos = [((1000 * ctx.width) / ctx.height) * 0.5, 700];
    let emitPrev = emitPos.slice();
    let grinder = null;
    let fw = { next: 0.15, queue: [] };
    let castQueued = 0;
    let lightAmt = 0;
    let lastPointer = { x: -1, y: -1 };
    let needWarm = true;

    const clearAll = (enc) => {
      enc.clearBuffer(particles);
      enc.clearBuffer(counters);
    };

    const worldSize = () => [(1000 * ctx.width) / ctx.height, 1000];

    const PALETTE = [
      [3.2, 0.5, 0.35],
      [3.0, 1.9, 0.45],
      [0.6, 3.0, 0.7],
      [0.45, 2.2, 3.2],
      [0.8, 0.9, 3.4],
      [2.4, 0.6, 3.2],
      [3.2, 0.8, 1.9],
      [2.6, 2.6, 2.8],
    ];

    /** CPU side of one simulation step: decide what to emit and fill the uniforms. */
    const prepareStep = (dt, p) => {
      const [W, H] = worldSize();
      const mode = MODES[ctx.example] ?? 0;
      const k = 1000 / ctx.height;
      const ptr = ctx.pointer;
      const mx = ptr.x * k;
      const my = ptr.y * k;
      simTime += dt;
      let emitCount = 0;
      let split = 0;
      let attractor = [W * 0.5, 450];
      let emitRadius = 0;
      let emitPos2 = [0, 0];
      let dir = p.dir ?? -90;
      let rocketVel = [0, 0];
      let rocketLife = 1;
      let rocketColor = [1, 1, 1, 1];
      let pattern = 0;
      emitPrev = emitPos.slice();

      if (mode === 0) {
        if (ptr.down) emitPos = [mx, Math.min(my, 980)];
        else if (lastPointer.x < 0) emitPos = [W * 0.5, 700];
        attractor = [W * 0.5 + Math.cos(simTime * 0.6) * W * 0.28, 420 + Math.sin(simTime * 1.1) * 180];
        emitAcc += p.rate * RATE * dt;
      } else if (mode === 1) {
        if (simTime >= fw.next && p.rockets > 0) {
          fw.queue.push({ x: W * (0.15 + 0.7 * rand()), y: 120 + 330 * rand() });
          fw.next = simTime + (0.4 + 1.2 * rand()) / p.rockets;
        }
        if (ptr.clicked) fw.queue.push({ x: mx, y: Math.min(my, 700), click: true });
        if (fw.queue.length) {
          const r = fw.queue.shift();
          const x0 = r.click ? r.x + (rand() - 0.5) * 60 : r.x + (rand() - 0.5) * 300;
          const y0 = 880;
          const g = Math.max(50, p.gravity);
          const vy = -Math.sqrt(2 * g * Math.max(50, y0 - r.y));
          const T = -vy / g;
          rocketVel = [(r.x - x0) / T, vy];
          rocketLife = T * 0.97;
          emitPos = [x0, y0];
          emitPrev = emitPos.slice();
          rocketColor = [...PALETTE[Math.floor(rand() * PALETTE.length)], 1];
          pattern = Math.floor(rand() * 5);
          emitCount = 1;
        }
      } else if (mode === 2) {
        const target = ptr.over || lastPointer.x >= 0 ? [mx, my] : [W * 0.5 + Math.sin(simTime * 0.7) * W * 0.25, 520 + Math.sin(simTime * 1.4) * 80];
        const kf = 1 - Math.exp(-dt * 10);
        emitPos = [emitPos[0] + (target[0] - emitPos[0]) * kf, emitPos[1] + (target[1] - emitPos[1]) * kf];
        attractor = emitPos.slice();
        emitRadius = 95;
        emitAcc += p.rate * RATE * dt;
        if (ptr.clicked) castQueued = ctx.testMode ? 800 : 2500;
        lightAmt = 1;
      } else {
        if (!grinder) grinder = [W * 0.3, 560];
        if (ptr.down) grinder = [Math.min(Math.max(mx, 120), W - 120), Math.min(Math.max(my, 200), 720)];
        const contact = [grinder[0] + 18, grinder[1] + 92];
        emitPos = contact;
        if (ptr.clicked) emitPrev = emitPos.slice();
        emitAcc += p.rate * RATE * dt * (ptr.down ? 3 : 1);
        emitPos2 = [W * 0.8, 780];
        emitRadius = 150;
        dir = p.dir;
        lightAmt = 0.6 + 0.4 * Math.random();
      }

      if (mode !== 1) {
        const n = Math.floor(emitAcc);
        emitAcc -= n;
        emitCount = Math.min(n, Math.floor(MAX / 2));
        if (mode === 2 && castQueued) {
          split = emitCount;
          emitCount += castQueued;
          castQueued = 0;
        }
        if (mode === 3) {
          split = emitCount;
          emberAcc += 90 * dt;
          const e = Math.floor(emberAcc);
          emberAcc -= e;
          emitCount += e;
        }
      }

      const ramp = (hex, i) => {
        const c = hexLin(hex || '#ffffff', mode === 2 ? 1.3 : 1.1);
        return [c[0], c[1], c[2], 1];
      };
      const sparkRamp = mode === 3;
      U.set('dt', dt)
        .set('time', simTime)
        .set('emitCount', emitCount)
        .set('maxCount', MAX)
        .set('seed', seed++)
        .set('mode', mode)
        .set('split', split)
        .set('flags', (p.crackle ? 1 : 0) | (p.trails ? 2 : 0))
        .set('texCell', mode === 3 ? TEX.spark : SHAPES[p.shape] ?? 0)
        .set('additive', mode === 0 ? p.additive : 1)
        .set('world', [W, H])
        .set('emitPos', emitPos)
        .set('emitPrev', emitPrev)
        .set('emitPos2', emitPos2)
        .set('attractor', attractor)
        .set('gravity', [0, mode === 2 ? 0 : p.gravity])
        .set('emitDir', dir)
        .set('spread', p.spread ?? 30)
        .set('speed', p.speed)
        .set('life', p.life ?? 2)
        .set('size', p.size ?? 8)
        .set('sizeEnd', mode === 3 ? 1.5 : p.sizeEnd ?? 2)
        .set('drag', mode === 2 ? 1.6 : p.drag ?? 0.3)
        .set('turb', mode === 3 || mode === 1 ? 0 : p.turb ?? 0)
        .set('turbScale', 0.004)
        .set('attract', mode === 0 || mode === 2 ? p.attract ?? 0 : 0)
        .set('groundY', mode === 3 ? 850 : 900)
        .set('bounce', mode === 0 || mode === 3 ? p.bounce ?? 0 : 0)
        .set('emitRadius', emitRadius)
        .set('stretch', mode === 0 || mode === 3 ? p.stretch ?? 0 : 0)
        .set('rampMid', sparkRamp ? 0.25 : 0.4)
        .set('burst', Math.min(4000, p.burst ?? 500))
        .set('rocketVel', rocketVel)
        .set('rocketLife', rocketLife)
        .set('pattern', pattern)
        .set('rocketColor', rocketColor)
        .set('c0', sparkRamp ? [6, 5, 3.6, 1] : ramp(p.colA, 0))
        .set('c1', sparkRamp ? [4, 1.4, 0.3, 1] : ramp(p.colB, 1))
        .set('c2', sparkRamp ? [0.9, 0.08, 0.02, 1] : ramp(p.colC, 2))
        .set('waterY', 880)
        .set('lightAmt', lightAmt)
        .set('pxw', k)
        .set('exposure', p.exposure)
        .set('bloom', p.bloom * 0.35);
      return emitCount;
    };

    const encodeStep = (enc, emitCount, count = false) => {
      if (emitCount > 0) sim.dispatch(enc, 'emit', Math.ceil(emitCount / 64), { u: U, ps: particles, ctr: counters, drawArgs });
      if (count) sim.dispatch(enc, 'resetAlive', 1, { u: U, ps: particles, ctr: counters, drawArgs });
      sim.dispatch(enc, 'update', Math.ceil(MAX / 256), { u: U, ps: particles, ctr: counters, drawArgs });
    };

    /** Run ~1.5 s of simulation immediately so the scene starts "in progress".
     *  Each step is its own submit, because uniform writes land before a whole command buffer. */
    const warmUp = () => {
      // fireworks need ~3.5 s so the first rockets have already burst
      const fwk = ctx.example === 'fireworks';
      const stepDt = fwk ? (ctx.testMode ? 1 / 15 : 1 / 24) : 1 / 30;
      const steps = fwk ? Math.round(3.4 / stepDt) : ctx.testMode ? 24 : 50;
      const e0 = device.createCommandEncoder();
      clearAll(e0);
      device.queue.submit([e0.finish()]);
      simTime = 0;
      fw = { next: 0.05, queue: [] };
      emitAcc = 0;
      rand = prng(7);
      for (let s = 0; s < steps; s++) {
        const n = prepareStep(stepDt, ctx.params);
        U.set('countAlive', 0);
        U.upload();
        const enc = device.createCommandEncoder();
        encodeStep(enc, n);
        device.queue.submit([enc.finish()]);
      }
    };

    const readBack = () => {
      if (reading) return;
      reading = true;
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(counters, 0, staging, 0, 16);
      device.queue.submit([e.finish()]);
      staging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const a = new Uint32Array(staging.getMappedRange().slice(0));
          alive = a[1];
          staging.unmap();
          reading = false;
        })
        .catch(() => (reading = false));
    };

    return {
      resize(w, h) {
        hdr.destroy();
        hdr = gpu.target(w, h, { format: 'rgba16float', label: 'hdr' });
      },
      onAction(key) {
        if (key === 'reset') needWarm = true;
      },
      onExample() {
        grinder = null;
        lastPointer = { x: -1, y: -1 };
        emitPos = [((1000 * ctx.width) / ctx.height) * 0.5, 700];
        needWarm = true;
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        if (ctx.pointer.over) lastPointer = { x: ctx.pointer.x, y: ctx.pointer.y };
        if (needWarm) {
          needWarm = false;
          warmUp();
        } else if (ctx.frame % 8 === 0) {
          readBack();
        }
        const dt = Math.min(ctx.dt, 1 / 30);
        if (!ctx.paused && dt > 0) {
          const n = prepareStep(dt, p);
          // count live particles on the frame before a read-back (it reads last frame's counter)
          const count = ctx.frame % 8 === 7;
          U.set('countAlive', count ? 1 : 0);
          U.upload();
          encodeStep(enc, n, count);
        } else {
          U.set('exposure', p.exposure).set('bloom', p.bloom * 0.35).set('emitCount', 0);
          U.upload();
        }
        sim.dispatch(enc, 'args', 1, { u: U, ps: particles, ctr: counters, drawArgs });

        // background -> HDR
        const mode = MODES[ctx.example] ?? 0;
        let gameTex = blankGame;
        if (mode === 2) gameTex = game.render(enc, ctx.time, Math.max(64, ctx.width >> 1), Math.max(36, ctx.height >> 1));
        bg.draw(enc, hdr, { game: gameTex }, { clear: [0, 0, 0, 1] });

        // grinder & forge props (sparks example)
        if (mode === 3 && grinder) {
          const k = ctx.height / 1000;
          cam.setViewport(ctx.width, ctx.height);
          shapes.begin();
          const gx = grinder[0] * k;
          const gy = grinder[1] * k;
          const r = 92 * k;
          const W = ctx.width;
          // workbench + steel bar whose top touches the disc
          const by = gy + r;
          const bx0 = gx - 300 * k;
          const bx1 = gx + 70 * k;
          shapes.rect(bx0 - 20 * k, by + 26 * k, bx1 - bx0 + 40 * k, 16 * k, [0.06, 0.035, 0.022, 1], { radius: 3 * k });
          shapes.rect(bx0, by + 42 * k, 16 * k, 850 * k - by - 42 * k, [0.035, 0.022, 0.016, 1]);
          shapes.rect(bx1 - 16 * k, by + 42 * k, 16 * k, 850 * k - by - 42 * k, [0.035, 0.022, 0.016, 1]);
          shapes.rect(bx0 + 30 * k, by, bx1 - bx0 - 60 * k, 26 * k, [0.13, 0.135, 0.15, 1], { radius: 3 * k });
          shapes.rect(bx0 + 30 * k, by, bx1 - bx0 - 60 * k, 5 * k, [0.42, 0.43, 0.46, 1], { radius: 2 * k });
          shapes.circle(gx + 18 * k, by + 4 * k, 12 * k, [5, 2.0, 0.5, 1], { glow: 14 * k, glowStrength: 0.9 });
          // disc (spinning), rim lit orange by the sparks, hub
          shapes.circle(gx, gy, r * 1.04, [0.05, 0.05, 0.055, 1]);
          shapes.circle(gx, gy, r, [0.2, 0.2, 0.21, 1]);
          shapes.circle(gx, gy, r * 0.97, [0.13, 0.13, 0.14, 1]);
          const spin = ctx.time * 40;
          for (let i = 0; i < 6; i++) {
            const a = spin + (i * Math.PI) / 3;
            shapes.line(gx + Math.cos(a) * r * 0.3, gy + Math.sin(a) * r * 0.3, gx + Math.cos(a) * r * 0.9, gy + Math.sin(a) * r * 0.9, 2.5 * k, [0.24, 0.24, 0.26, 1]);
          }
          shapes.circle(gx + r * 0.55, gy + r * 0.6, r * 0.5, [1.2, 0.4, 0.1, 0.18]);
          shapes.circle(gx, gy, r * 0.24, [0.32, 0.32, 0.34, 1]);
          shapes.circle(gx, gy, r * 0.1, [0.08, 0.08, 0.09, 1]);
          // motor housing + handle
          shapes.rect(gx - 34 * k, gy - r * 1.75, 68 * k, r * 1.35, [0.05, 0.13, 0.42, 1], { radius: 14 * k });
          shapes.rect(gx - 26 * k, gy - r * 1.68, 14 * k, r * 1.15, [0.12, 0.25, 0.65, 1], { radius: 7 * k });
          shapes.rect(gx - 16 * k, gy - r * 2.9, 32 * k, r * 1.2, [0.03, 0.03, 0.035, 1], { radius: 10 * k });
          // forge: brick box with glowing coals
          const fx = W * 0.8;
          const fy = 780 * k;
          shapes.rect(fx - 200 * k, fy - 4 * k, 400 * k, 74 * k, [0.09, 0.05, 0.04, 1], { radius: 6 * k });
          shapes.rect(fx - 200 * k, fy - 4 * k, 400 * k, 8 * k, [0.2, 0.1, 0.07, 1], { radius: 4 * k });
          for (let i = 0; i < 15; i++) {
            const t = ctx.time * 2 + i * 1.7;
            const glow = 1.1 + 0.5 * Math.sin(t) * Math.sin(t * 1.7);
            shapes.circle(fx - 165 * k + i * 23.5 * k, fy - 8 * k + Math.sin(i * 2.3) * 5 * k, (11 + 4 * Math.sin(i * 1.3)) * k, [2.6 * glow, 0.75 * glow, 0.12 * glow, 1], { glow: 10 * k, glowStrength: 0.55 });
          }
          shapes.flush(enc, hdr, cam, {});
        }

        // particles -> HDR (indirect draw)
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: hdr.view, loadOp: 'load', storeOp: 'store' }] });
        pass.setPipeline(drawPipe);
        pass.setBindGroup(0, draw.bind({ u: U, ps: particles, strip, stripSamp: gpu.sampler('linear-mip') }));
        pass.drawIndirect(drawArgs, 0);
        pass.end();

        const bl = bloom.render(enc, hdr, { threshold: 0.9, knee: 0.6 });
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { hdr, bloomTex: bl });
        readout.textContent = `${alive.toLocaleString()} alive · pool ${MAX.toLocaleString()}`;
      },
    };
  },
};

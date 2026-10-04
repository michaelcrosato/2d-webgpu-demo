import { shaderScene } from '../../core/shaderscene.js';
import { setTag } from './_shared-a.js';

// Neon & synthwave: everything is drawn procedurally in one fragment shader with signed distance
// fields (SDFs). "Glow" is not a blur here: every shape knows its distance d to each pixel, and
// light falls off as exp(-d / radius), added on top (additive = light).
//
// The Geometry-Wars arena runs a tiny game simulation in JavaScript (ship, enemies, bullets,
// explosions) and passes the results as uniform arrays; the shader draws them and warps the grid.

// ------------------------------------------------------------------------------ arena simulation
const N_EN = 12;
const N_BL = 16;
const N_EV = 8;
const sim = {
  lastT: -1,
  look: 0,
  ship: { x: 0, y: 0, vx: 0, vy: 0, a: 0 },
  en: [],
  bl: [],
  ev: [],
  fireT: 0,
  lastKill: 0,
  seed: 1,
};
const rand = () => {
  sim.seed = (sim.seed * 16807) % 2147483647;
  return (sim.seed - 1) / 2147483646;
};

function resetSim(asp) {
  sim.seed = 7;
  sim.ship = { x: 0, y: 0, vx: 0, vy: 0, a: 0 };
  sim.en = [];
  for (let i = 0; i < N_EN; i++) sim.en.push(spawnEnemy(asp, i % 4, 1));
  sim.bl = [];
  sim.ev = [];
  sim.fireT = 0;
  sim.lastKill = 0;
}
function spawnEnemy(asp, type, grown = 0) {
  const side = rand() * 4;
  const hx = asp * 0.5 - 0.04;
  let x = (rand() * 2 - 1) * hx;
  let y = (rand() * 2 - 1) * 0.46;
  if (side < 1) x = -hx;
  else if (side < 2) x = hx;
  else if (side < 3) y = -0.46;
  else y = 0.46;
  return { x, y, vx: 0, vy: 0, type, grow: grown, rot: rand() * 6.28, wander: rand() * 6.28 };
}
const HUES = [0.92, 0.36, 0.6, 0.08];
function explode(x, y, hue, t, big = 0) {
  sim.ev.push({ x, y, t0: t, hue, big });
  if (sim.ev.length > N_EV) sim.ev.shift();
}

function stepArena(ctx) {
  const asp = ctx.width / Math.max(1, ctx.height);
  const t = ctx.time;
  if (sim.lastT < 0 || t < sim.lastT - 0.001 || sim.en.length !== N_EN) resetSim(asp);
  const dt = Math.min(0.05, Math.max(0, t - sim.lastT));
  sim.lastT = t;
  const hx = asp * 0.5 - 0.03;
  const p = ctx.pointer;
  const s = sim.ship;
  // target: the mouse, or an autopilot figure-eight when the mouse is away
  let tx = Math.sin(t * 0.6) * asp * 0.3;
  let ty = Math.sin(t * 1.2) * 0.22;
  if (p.over) {
    tx = (p.x - ctx.width / 2) / ctx.height;
    ty = (p.y - ctx.height / 2) / ctx.height;
  }
  if (dt > 0) {
    s.vx += (tx - s.x) * 30 * dt;
    s.vy += (ty - s.y) * 30 * dt;
    const damp = Math.pow(0.0015, dt);
    s.vx *= damp;
    s.vy *= damp;
    s.x += s.vx * dt;
    s.y += s.vy * dt;
    if (Math.hypot(s.vx, s.vy) > 0.05) s.a = Math.atan2(s.vy, s.vx);
    // enemies: seek the ship + wander, softly separated
    for (const e of sim.en) {
      e.grow = Math.min(1, e.grow + dt * 1.5);
      e.rot += dt * (e.type === 2 ? 4 : 1.5);
      e.wander += (rand() - 0.5) * dt * 6;
      let ax = s.x - e.x;
      let ay = s.y - e.y;
      const d = Math.hypot(ax, ay) + 1e-4;
      const seek = e.type === 3 ? 0.9 : 0.45;
      ax = (ax / d) * seek + Math.cos(e.wander) * 0.6;
      ay = (ay / d) * seek + Math.sin(e.wander) * 0.6;
      e.vx = (e.vx + ax * dt * 2) * Math.pow(0.3, dt);
      e.vy = (e.vy + ay * dt * 2) * Math.pow(0.3, dt);
      e.x = Math.max(-hx, Math.min(hx, e.x + e.vx * dt * e.grow));
      e.y = Math.max(-0.47, Math.min(0.47, e.y + e.vy * dt * e.grow));
    }
    // auto-fire at the nearest enemy
    sim.fireT -= dt;
    if (sim.fireT <= 0) {
      sim.fireT = 0.11;
      let best = null;
      let bd = 1e9;
      for (const e of sim.en) {
        const d = Math.hypot(e.x - s.x, e.y - s.y);
        if (e.grow > 0.6 && d < bd) (bd = d), (best = e);
      }
      if (best) {
        const a = Math.atan2(best.y - s.y, best.x - s.x) + (rand() - 0.5) * 0.12;
        sim.bl.push({ x: s.x, y: s.y, dx: Math.cos(a), dy: Math.sin(a), life: 1.2 });
        if (sim.bl.length > N_BL) sim.bl.shift();
      }
    }
    for (const b of sim.bl) {
      b.x += b.dx * 1.5 * dt;
      b.y += b.dy * 1.5 * dt;
      b.life -= dt;
      for (let i = 0; i < sim.en.length; i++) {
        const e = sim.en[i];
        if (b.life > 0 && e.grow > 0.6 && Math.hypot(e.x - b.x, e.y - b.y) < 0.03) {
          b.life = 0;
          explode(e.x, e.y, HUES[e.type], t);
          sim.en[i] = spawnEnemy(asp, e.type);
          sim.lastKill = t;
        }
      }
    }
    sim.bl = sim.bl.filter((b) => b.life > 0 && Math.abs(b.x) < hx + 0.1 && Math.abs(b.y) < 0.6);
    if (t - sim.lastKill > 1.6) {
      // keep it lively: a random enemy pops
      const i = Math.floor(rand() * sim.en.length);
      const e = sim.en[i];
      explode(e.x, e.y, HUES[e.type], t);
      sim.en[i] = spawnEnemy(asp, e.type);
      sim.lastKill = t;
    }
  }
  if (p.clicked) explode(p.over ? tx : s.x, p.over ? ty : s.y, 0.15, t, 1);
  // pack uniforms
  const en = new Float32Array(N_EN * 4);
  sim.en.forEach((e, i) => en.set([e.x, e.y, e.type + Math.min(0.999, e.rot / 1000 - Math.floor(e.rot / 1000)), e.grow], i * 4));
  const rot = new Float32Array(N_EN * 4);
  sim.en.forEach((e, i) => (rot[i] = e.rot));
  const bl = new Float32Array(N_BL * 4);
  sim.bl.forEach((b, i) => bl.set([b.x, b.y, Math.atan2(b.dy, b.dx), 1], i * 4));
  const ev = new Float32Array(N_EV * 4);
  for (let i = 0; i < N_EV; i++) ev.set([0, 0, -1, 0], i * 4);
  sim.ev.forEach((e, i) => ev.set([e.x, e.y, t - e.t0, e.hue + (e.big ? 10 : 0)], i * 4));
  return { ship: [s.x, s.y, s.a, 1], en, enRot: rot, bl, ev };
}

// ----------------------------------------------------------------------------------- shaders
const CODE = /* wgsl */ `
fn pxSize() -> f32 { return 1.0 / u.resolution.y; }

// A glowing tube/line: white-hot core + colored halo. d = distance to the centre line.
fn neonLine(d: f32, col: vec3f, width: f32) -> vec3f {
  let aa = pxSize() * 1.2;
  let core = 1.0 - smoothstep(width, width + aa, d);
  let r = 0.01 * u.glowSize;
  let halo = exp(-d / r) * 0.55 + exp(-d / (r * 4.0)) * 0.22;
  return mix(col, vec3f(1.0), 0.55) * core + col * halo * u.glow;
}

// ------------------------------------------------------------------ synthwave sunset
fn mtnHeight(k: f32) -> f32 {
  let x = k * 0.085;
  let env = 0.12 + smoothstep(0.08, 0.75, abs(x)) * 0.9;
  return (0.025 + 0.16 * hash11(k * 3.17 + 11.0)) * env;
}
fn sunsetView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let asp = res.x / res.y;
  let p = (px - 0.5 * res) / res.y;
  let t = u.time * u.speed;
  let hz = 0.1;
  var col = vec3f(0.0);
  if (p.y < hz) {
    // sky gradient: deep violet → magenta → hot orange at the horizon
    let k = clamp((hz - p.y) / (hz + 0.5), 0.0, 1.0);
    col = mix(vec3f(1.0, 0.42, 0.32), vec3f(0.55, 0.07, 0.45), smoothstep(0.0, 0.35, k));
    col = mix(col, vec3f(0.05, 0.01, 0.12), smoothstep(0.3, 1.0, k));
    // stars
    let sp = p * 90.0;
    let cell = floor(sp);
    let h = hash22(cell);
    let sd = length(fract(sp) - h);
    let tw = 0.6 + 0.4 * sin(t * 3.0 + h.x * 40.0);
    col += u.colB * step(0.93, hash21(cell + 7.0)) * exp(-sd * 18.0) * tw * smoothstep(0.25, 0.7, k) * 1.4;
    // sun with scrolling stripe gaps
    let sc = vec2f(0.0, hz - 0.17);
    let sr = 0.23;
    let sdist = length(p - sc) - sr;
    let sy = (p.y - sc.y) / sr;
    let sunCol = mix(vec3f(1.0, 0.93, 0.35), mix(vec3f(1.0, 0.45, 0.3), u.colA, 0.6), smoothstep(-0.9, 0.7, sy));
    let gap = step(-0.15, sy) * step(fract(sy * 4.5 - t * 0.25), (sy + 0.15) * 0.42);
    let sunMask = (1.0 - smoothstep(-pxSize(), pxSize(), sdist)) * (1.0 - gap);
    col += u.colA * exp(-max(sdist, 0.0) * 7.0) * 0.55 * u.glow;
    col = mix(col, sunCol, sunMask);
    // wireframe mountains in front of the sun
    let mw = 0.085;
    let mx = p.x / mw + 0.5;
    let kk = floor(mx);
    let fr = fract(mx);
    let h0 = mtnHeight(kk);
    let h1 = mtnHeight(kk + 1.0);
    let ridge = hz - mix(h0, h1, fr);
    if (p.y > ridge - 0.02) {
      var dw = 1.0;
      for (var j = -1; j <= 1; j++) {
        let c = kk + f32(j);
        let a = vec2f((c - 0.5) * mw, hz - mtnHeight(c));
        let b = vec2f((c + 0.5) * mw, hz - mtnHeight(c + 1.0));
        dw = min(dw, sdSegment(p, a, b));
        // triangulation: each peak connects down to the horizon on both sides
        dw = min(dw, sdSegment(p, a, vec2f(c * mw, hz)));
        dw = min(dw, sdSegment(p, a, vec2f((c - 1.0) * mw, hz)));
      }
      let inside = smoothstep(ridge - pxSize(), ridge + pxSize(), p.y);
      col = mix(col, mix(vec3f(0.07, 0.01, 0.13), vec3f(0.02, 0.0, 0.05), smoothstep(ridge, hz, p.y)), inside);
      col += neonLine(dw, u.colB, 0.0012) * (0.35 + 0.65 * inside);
    }
  } else {
    // perspective floor: depth from the screen y, world x scaled by depth
    let dy = max(p.y - hz, 0.0001);
    let z = 0.22 / dy;
    let wp = vec2f((p.x + u.look * 0.4) * z, z + t * 1.2) * (u.grid * 0.12);
    let fw = max(fwidth(wp), vec2f(0.0001));
    let dl = abs(fract(wp + 0.5) - 0.5);
    let lw = 0.035;
    let cov = clamp((vec2f(lw) - dl) / fw + 0.5, vec2f(0.0), vec2f(1.0));
    let farFade = clamp(1.4 - max(fw.x, fw.y) * 3.0, 0.0, 1.0);
    let lines = max(cov.x, cov.y);
    let avg = lw * 2.0;
    let gl = mix(avg, lines, farFade);
    let halo = (exp(-dl.x / (fw.x * 3.0 + 0.04)) + exp(-dl.y / (fw.y * 3.0 + 0.04))) * 0.25 * farFade;
    col = mix(vec3f(0.04, 0.0, 0.08), vec3f(0.12, 0.02, 0.2), exp(-dy * 6.0));
    col += u.colA * (gl * 1.6 + halo * u.glow);
    // horizon haze and the sun's reflection on the floor
    col += mix(u.colA, vec3f(1.0, 0.5, 0.3), 0.4) * exp(-dy * 22.0) * 0.8;
    col += vec3f(1.0, 0.55, 0.35) * exp(-abs(p.x) * 9.0) * exp(-dy * 7.0) * 0.35;
  }
  // bright horizon line
  col += mix(u.colA, vec3f(1.0), 0.4) * exp(-abs(p.y - hz) / (0.003 * u.glowSize)) * 0.8;
  return col;
}

// ------------------------------------------------------------------ geometry wars arena
fn warpAt(q: vec2f) -> vec2f {
  var d = vec2f(0.0);
  // the ship is a small gravity well
  let s = q - u.ship.xy;
  let rs = length(s) + 0.0001;
  d -= s / rs * 0.022 * exp(-rs * 9.0) * u.warp;
  for (var i = 0; i < ${N_EV}; i++) {
    let e = u.ev[i];
    let age = e.z;
    if (age >= 0.0 && age < 3.0) {
      let big = step(5.0, e.w);
      let v = q - e.xy;
      let r = length(v) + 0.0001;
      let ringR = age * (0.55 + 0.4 * big);
      let amp = (0.045 + 0.05 * big) * exp(-age * 1.4) * u.warp;
      // a pushed-out ring of space, plus a pull in the middle (it springs back)
      d += v / r * amp * exp(-pow((r - ringR) / 0.07, 2.0));
      d -= v / r * amp * 0.5 * exp(-r * 14.0) * sin(age * 12.0) * exp(-age * 3.0);
    }
  }
  return d;
}
fn shipSd(q0: vec2f) -> f32 {
  let q = rot2(-u.ship.z) * q0;
  let a = vec2f(0.03, 0.0);
  let b = vec2f(-0.022, 0.024);
  let c = vec2f(-0.008, 0.0);
  let e = vec2f(-0.022, -0.024);
  return min(min(sdSegment(q, a, b), sdSegment(q, b, c)), min(sdSegment(q, c, e), sdSegment(q, e, a)));
}
fn enemySd(q0: vec2f, kind: f32, r: f32, s: f32) -> f32 {
  let q = rot2(r) * q0 / max(s, 0.01);
  var d = 0.0;
  if (kind < 0.5) { d = abs(sdRhombus(q, vec2f(0.02, 0.03))); }
  else if (kind < 1.5) { d = abs(sdBox(q, vec2f(0.019))); d = min(d, abs(sdBox(rot2(0.785) * q, vec2f(0.011)))); }
  else if (kind < 2.5) { d = abs(sdCross(q, vec2f(0.028, 0.006), 0.0)); }
  else { d = abs(sdEquilateralTriangle(q, 0.024)); }
  return d * max(s, 0.01);
}
fn arenaView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let asp = res.x / res.y;
  let p = (px - 0.5 * res) / res.y;
  var col = vec3f(0.005, 0.008, 0.03);
  // warped grid: sample the grid at the un-warped position
  let w = warpAt(p);
  let q = p - w;
  let gp = q * u.grid;
  let fw = max(fwidth(gp), vec2f(0.0001));
  let dl = abs(fract(gp + 0.5) - 0.5);
  let major = abs(fract(gp / 4.0 + 0.5) - 0.5) * 4.0;
  let thin = max(clamp((vec2f(0.03) - dl) / fw + 0.5, vec2f(0.0), vec2f(1.0)).x, clamp((vec2f(0.03) - dl) / fw + 0.5, vec2f(0.0), vec2f(1.0)).y);
  let thick = max(clamp((vec2f(0.06) - major) / fw + 0.5, vec2f(0.0), vec2f(1.0)).x, clamp((vec2f(0.06) - major) / fw + 0.5, vec2f(0.0), vec2f(1.0)).y);
  let energy = clamp(length(w) * 40.0, 0.0, 2.0);
  let gcol = mix(u.colB * 0.55, mix(u.colB, vec3f(1.0), 0.5), clamp(energy, 0.0, 1.0));
  let gh = (exp(-min(dl.x, dl.y) / (fw.x * 2.0 + 0.02)) * 0.12);
  col += gcol * (thin * 0.45 + thick * 0.55 + gh * u.glow) * (0.6 + energy);
  // arena border
  let bd = abs(sdBox(p, vec2f(asp * 0.5 - 0.02, 0.48)));
  col += neonLine(bd, u.colB, 0.0015) * 0.6;
  // enemies
  var hues = array<vec3f, 4>(vec3f(1.0, 0.25, 0.75), vec3f(0.3, 1.0, 0.35), vec3f(0.3, 0.55, 1.0), vec3f(1.0, 0.55, 0.15));
  for (var i = 0; i < ${N_EN}; i++) {
    let e = u.en[i];
    let v = p - e.xy;
    if (dot(v, v) < 0.02) {
      let kind = floor(e.z);
      let d = enemySd(v, kind, u.enRot[i / 4][i % 4], e.w);
      col += neonLine(d, hues[i32(kind)], 0.0016) * e.w;
    }
  }
  // bullets
  for (var i = 0; i < ${N_BL}; i++) {
    let b = u.bl[i];
    if (b.w > 0.5) {
      let dir = vec2f(cos(b.z), sin(b.z));
      let d = sdSegment(p, b.xy, b.xy - dir * 0.018);
      col += neonLine(d, vec3f(1.0, 0.85, 0.4), 0.0015) * 0.8;
    }
  }
  // explosions: shockwave ring + streaking sparks
  for (var i = 0; i < ${N_EV}; i++) {
    let e = u.ev[i];
    let age = e.z;
    if (age >= 0.0 && age < 1.6) {
      let v = p - e.xy;
      let r = length(v);
      let big = step(5.0, e.w);
      let hue = fract(e.w);
      let ec = hsv2rgb(vec3f(hue, 0.75, 1.0));
      let fade = exp(-age * 2.5);
      let ringR = age * (0.55 + 0.4 * big);
      col += neonLine(abs(r - ringR), ec, 0.001) * fade * 0.7;
      if (r < 0.32) {
        var ds = 1.0;
        for (var k = 0; k < 20; k++) {
          let fk = f32(k);
          let ang = fk / 20.0 * TAU + hash11(fk + e.x * 31.0) * 0.6;
          let spd = 0.35 + 0.55 * hash11(fk * 1.7 + e.y * 17.0);
          let dist = spd * (1.0 - exp(-age * 3.5)) / 3.5 * 2.2;
          let len = 0.006 + 0.05 * spd * exp(-age * 3.5);
          let dir = vec2f(cos(ang), sin(ang));
          ds = min(ds, sdSegment(v, dir * dist, dir * max(dist - len, 0.0)));
        }
        col += neonLine(ds, mix(ec, vec3f(1.0, 0.95, 0.7), 0.3), 0.0012) * fade * 1.2;
      }
      col += ec * exp(-r * 30.0) * exp(-age * 8.0) * 2.0;
    }
  }
  // the ship
  col += neonLine(shipSd(p - u.ship.xy), mix(u.colA, vec3f(1.0), 0.35), 0.0018);
  return col;
}

// ------------------------------------------------------------------ neon sign
fn halfRing(q: vec2f, r: f32) -> f32 {
  if (q.x >= 0.0) { return abs(length(q) - r); }
  return min(length(q - vec2f(0.0, -r)), length(q - vec2f(0.0, r)));
}
fn letterSd(q: vec2f, which: i32) -> f32 {
  // letter box: x in [-0.3, 0.3], y in [-0.5, 0.5] (y down)
  if (which == 0) { return abs(sdRoundBox(q, vec2f(0.3, 0.5), 0.29)); }
  if (which == 1) {
    var d = sdSegment(q, vec2f(-0.28, 0.5), vec2f(-0.28, -0.5));
    d = min(d, sdSegment(q, vec2f(-0.28, -0.5), vec2f(0.02, -0.5)));
    d = min(d, sdSegment(q, vec2f(-0.28, 0.04), vec2f(0.02, 0.04)));
    return min(d, halfRing(q - vec2f(0.02, -0.23), 0.27));
  }
  if (which == 2) {
    var d = sdSegment(q, vec2f(-0.26, -0.5), vec2f(-0.26, 0.5));
    d = min(d, sdSegment(q, vec2f(-0.26, -0.5), vec2f(0.28, -0.5)));
    d = min(d, sdSegment(q, vec2f(-0.26, 0.0), vec2f(0.16, 0.0)));
    return min(d, sdSegment(q, vec2f(-0.26, 0.5), vec2f(0.28, 0.5)));
  }
  var d = sdSegment(q, vec2f(-0.27, 0.5), vec2f(-0.27, -0.5));
  d = min(d, sdSegment(q, vec2f(-0.27, -0.5), vec2f(0.27, 0.5)));
  return min(d, sdSegment(q, vec2f(0.27, 0.5), vec2f(0.27, -0.5)));
}
fn flickerOn(id: f32, strength: f32) -> f32 {
  let t = u.time * u.speed;
  // random short drop-outs; the "E" (id 2) is a dying tube that buzzes much more
  let rate = select(7.0, 17.0, id > 1.5 && id < 2.5);
  let chance = select(0.04, 0.35, id > 1.5 && id < 2.5) * strength;
  let off = step(1.0 - chance, hash11(floor(t * rate) + id * 13.1));
  let hum = 1.0 - strength * 0.06 * (0.5 + 0.5 * sin(t * 120.0 + id));
  return (1.0 - off) * hum;
}
fn tube(d: f32, col: vec3f, on: f32) -> vec4f {
  // returns rgb = light emitted by the tube (core + glow), a = tube body coverage
  let tr = 0.0075;
  let aa = pxSize() * 1.2;
  let body = 1.0 - smoothstep(tr, tr + aa, d);
  let core = 1.0 - smoothstep(0.0, tr * 0.9, d);
  let lit = mix(col, vec3f(1.0), core * 0.7) * (0.65 + 0.6 * core);
  let unlit = col * 0.12 + vec3f(0.05);
  let surface = mix(unlit, lit, on) * body;
  let r = 0.012 * u.glowSize;
  let glow = col * (exp(-d / r) * 0.6 + exp(-d / (r * 5.0)) * 0.3) * on * u.glow;
  return vec4f(surface + glow * (1.0 - body), body);
}
fn signView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let p = (px - 0.5 * res) / res.y;
  let s = 1.15;                                     // sign scale
  let sp = p / s;
  // ---- distances to every tube (in screen units)
  let frameD = abs(sdRoundBox(sp - vec2f(0.0, 0.0), vec2f(0.66, 0.27), 0.08)) * s;
  // cocktail glass (left)
  let gq = (sp - vec2f(-0.44, -0.01)) / 0.36;
  var glassD = sdSegment(gq, vec2f(-0.42, -0.5), vec2f(0.42, -0.5));
  glassD = min(glassD, sdSegment(gq, vec2f(-0.42, -0.5), vec2f(0.0, 0.08)));
  glassD = min(glassD, sdSegment(gq, vec2f(0.42, -0.5), vec2f(0.0, 0.08)));
  glassD = min(glassD, sdSegment(gq, vec2f(0.0, 0.08), vec2f(0.0, 0.44)));
  glassD = min(glassD, sdSegment(gq, vec2f(-0.24, 0.46), vec2f(0.24, 0.46)));
  glassD = min(glassD, sdSegment(gq, vec2f(0.12, -0.36), vec2f(0.44, -0.78)));
  glassD = glassD * 0.36 * s;
  let oliveD = abs(length(gq - vec2f(-0.06, -0.3)) - 0.085) * 0.36 * s;
  // OPEN
  var lettersD = array<f32, 4>(1.0, 1.0, 1.0, 1.0);
  for (var i = 0; i < 4; i++) {
    let lq = (sp - vec2f(-0.08 + f32(i) * 0.2, 0.0)) / 0.26;
    lettersD[i] = letterSd(lq, i) * 0.26 * s;
  }
  // ---- brick wall lit by the tubes
  let bsz = vec2f(0.11, 0.05);
  var bq = p / bsz;
  bq.x += 0.5 * floor(bq.y);
  let bid = floor(bq);
  let bf = fract(bq);
  let edge = min(min(bf.x, 1.0 - bf.x) * bsz.x, min(bf.y, 1.0 - bf.y) * bsz.y);
  let mortar = smoothstep(0.003, 0.006, edge);
  let rough = valueNoise(p * 180.0) * 0.25 + valueNoise(p * 40.0) * 0.2;
  let brick = mix(vec3f(0.33, 0.13, 0.09), vec3f(0.45, 0.2, 0.13), hash21(bid)) * (0.75 + rough);
  var wall = mix(vec3f(0.16, 0.15, 0.14) * (0.8 + rough), brick, mortar);
  // bevel: the top edge of each brick catches light from above (the sign is above most bricks)
  let bevel = smoothstep(0.012, 0.0, bf.y * bsz.y) * mortar * 0.5;

  let onFrame = flickerOn(5.0, u.flicker);
  let onGlass = flickerOn(4.0, u.flicker);
  var light = vec3f(0.012, 0.01, 0.02);
  light += u.colB * exp(-frameD / 0.18) * 0.45 * onFrame;
  light += u.colB * exp(-glassD / 0.18) * 0.6 * onGlass;
  light += vec3f(0.4, 1.0, 0.3) * exp(-oliveD / 0.12) * 0.2 * onGlass;
  for (var i = 0; i < 4; i++) {
    light += u.colA * exp(-lettersD[i] / 0.16) * 0.42 * flickerOn(f32(i), u.flicker);
  }
  var col = wall * light * (2.2 + bevel * 3.0) * (0.6 + 0.4 * u.glow);
  // ---- tubes on top (back-plate shadow first)
  var minD = min(min(frameD, glassD), oliveD);
  for (var i = 0; i < 4; i++) { minD = min(minD, lettersD[i]); }
  col *= 0.55 + 0.45 * smoothstep(0.004, 0.02, minD);
  var tb = tube(frameD, mix(u.colB, vec3f(0.55, 0.3, 1.0), 0.6), onFrame);
  col = col * (1.0 - tb.a) + tb.rgb;
  tb = tube(glassD, u.colB, onGlass);
  col = col * (1.0 - tb.a) + tb.rgb;
  tb = tube(oliveD, vec3f(0.45, 1.0, 0.3), onGlass);
  col = col * (1.0 - tb.a) + tb.rgb;
  for (var i = 0; i < 4; i++) {
    tb = tube(lettersD[i], u.colA, flickerOn(f32(i), u.flicker));
    col = col * (1.0 - tb.a) + tb.rgb;
  }
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var c = vec3f(0.0);
  if (ex == 0) { c = sunsetView(px); }
  else if (ex == 1) { c = arenaView(px); }
  else { c = signView(px); }
  // filmic-ish shoulder: very bright neon cores roll off to white instead of clipping
  c = vec3f(1.0) - exp(-c * 1.15);
  if (u.crt > 0.5) {
    c *= 0.82 + 0.18 * sin(px.y * 1.6);
    let v = uv * (1.0 - uv);
    c *= pow(clamp(v.x * v.y * 18.0, 0.0, 1.0), 0.25);
  }
  // tiny dither so dark gradients don't band
  c += vec3f((ign(px) - 0.5) / 255.0);
  return vec4f(c, 1.0);
}`;

export default shaderScene({
  interaction: '',
  examples: [
    {
      id: 'sunset',
      label: 'Synthwave sunset',
      kind: 'Abstract',
      hint: 'Move the mouse left/right to steer.',
      note: 'The 80s “outrun” look: a perspective grid racing towards a striped sun, wireframe mountains and a starfield. All of it is math per pixel — the grid is just <code>fract()</code> of a perspective-divided coordinate.',
      params: { glow: 1.0, glowSize: 1.0, colA: '#ff2bd6', colB: '#2be8ff', speed: 1, grid: 10, crt: true },
    },
    {
      id: 'arena',
      label: 'Geometry Wars arena',
      kind: 'In a game',
      hint: 'Move the mouse to fly (the ship bends space). Click for a big blast.',
      note: 'A twin-stick-shooter arena: glowing vector enemies, auto-firing ship and explosions whose shockwaves ripple through a <b>warping background grid</b>. The game logic runs in JS; the shader draws everything from uniform arrays.',
      params: { glow: 1.0, glowSize: 1.0, colA: '#ffe14d', colB: '#3d6bff', speed: 1, grid: 26, warp: 1, crt: false },
    },
    {
      id: 'sign',
      label: 'Neon sign',
      kind: 'Real life',
      hint: '',
      note: 'Glass tubes bent into letters, drawn as SDF strokes: a white-hot core, a colored halo, and the same distance values used to <b>light the brick wall</b>. One tube is dying and flickers.',
      params: { glow: 1.0, glowSize: 1.0, colA: '#ff3b8d', colB: '#24e1ff', speed: 1, flicker: 0.6, crt: false },
    },
  ],
  controls: [
    { type: 'slider', key: 'glow', label: 'Glow intensity', min: 0, max: 2.5, step: 0.01, value: 1 },
    { type: 'slider', key: 'glowSize', label: 'Glow radius', min: 0.3, max: 3, step: 0.01, value: 1, help: 'Light falls off as exp(−distance / radius).' },
    { type: 'color', key: 'colA', label: 'Neon color A', value: '#ff2bd6' },
    { type: 'color', key: 'colB', label: 'Neon color B', value: '#2be8ff' },
    { type: 'slider', key: 'speed', label: 'Speed', min: 0, max: 3, step: 0.01, value: 1 },
    { type: 'slider', key: 'grid', label: 'Grid density', min: 4, max: 40, step: 1, value: 10, showFor: ['sunset', 'arena'] },
    { type: 'slider', key: 'warp', label: 'Grid warp strength', min: 0, max: 2.5, step: 0.01, value: 1, showFor: ['arena'], help: 'How strongly explosions and the ship bend the grid.' },
    { type: 'slider', key: 'flicker', label: 'Flicker', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['sign'], help: 'Random tube drop-outs and mains hum.' },
    { type: 'toggle', key: 'crt', label: 'Retro scanlines', value: true },
  ],
  uniforms: {
    glow: 'f32', glowSize: 'f32', colA: 'vec3f', colB: 'vec3f', speed: 'f32', grid: 'f32', warp: 'f32', flicker: 'f32', crt: 'f32', look: 'f32',
    ship: 'vec4f', en: `array<vec4f, ${N_EN}>`, enRot: `array<vec4f, ${N_EN / 4}>`, bl: `array<vec4f, ${N_BL}>`, ev: `array<vec4f, ${N_EV}>`,
  },
  include: ['math', 'hash', 'noise', 'sdf', 'color'],
  bind(params, ctx) {
    const p = ctx.pointer;
    const target = p.over ? p.nx * 2 - 1 : 0;
    if (!ctx.paused) sim.look += (target - sim.look) * 0.04;
    const out = { look: sim.look };
    if (ctx.example === 'arena') Object.assign(out, stepArena(ctx));
    setTag(ctx, 'info', ctx.example === 'arena' ? `${N_EN} enemies · ${sim.bl.length} bullets · ${sim.ev.length} explosions` : '', 'right:8px;bottom:8px', ctx.example === 'arena');
    return out;
  },
  code: CODE,
  about: {
    summary: 'Neon is light, so in a shader it is additive glow around crisp shapes. Signed distance fields give you both the shape and the glow for free — the basis of synthwave, vector-arcade and neon-sign looks.',
    what: `<p>Three neon scenes drawn entirely in one fragment shader: a synthwave sunset, a Geometry-Wars-style arena where explosions ripple through
      a warping grid, and a flickering neon sign that lights the brick wall it hangs on.</p>`,
    how: `<ol>
      <li><b>Distance fields.</b> Every line, letter and enemy is a function returning the distance <code>d</code> from the pixel to the shape’s
        centre line (segments, arcs, rounded boxes…).</li>
      <li><b>Neon = core + halo.</b> A thin, almost-white core where <code>d &lt; width</code>, plus colored light that falls off with distance:
        <code>exp(-d / r)</code> for a tight glow and a second, wider <code>exp(-d / 4r)</code> for the bloom. Values are <i>added</i> — light adds up.</li>
      <li><b>Roll-off.</b> Adding lights easily exceeds 1.0, so a soft tone curve (<code>1 − exp(−c)</code>) turns overlapping cores white instead
        of clipping to flat color — the hot-core look of real neon.</li>
      <li><b>Perspective grid.</b> For floor pixels, depth = <code>k / (y − horizon)</code>; world x = screen x × depth. Grid lines are where
        <code>fract(world)</code> is near 0; <code>fwidth()</code> anti-aliases them and fades far rows to their average so the horizon doesn’t shimmer.
        Adding time to world z scrolls it.</li>
      <li><b>Warping grid.</b> Each explosion and the ship define a displacement field (an outward-moving ring, a pull toward the ship). The shader draws
        the grid at <code>p − displacement(p)</code>, so lines appear to bend, and brightens them where they’re displaced.</li>
      <li><b>Lighting the wall.</b> The sign reuses its distance values: wall brightness = Σ tube color × <code>exp(−d / 0.16)</code>, plus a bevel on brick
        edges and a contact shadow under the tubes.</li>
    </ol>`,
    uses: [
      { title: 'Vector arcade', text: 'Geometry Wars, Tempest 2000, Polybius, Resogun’s HUD — glow lines on black, warping grids that react to explosions.' },
      { title: 'Synthwave & outrun', text: 'Hotline Miami, Far Cry 3: Blood Dragon, Neon White menus, Cloudpunk signage, Katana ZERO’s neon city.' },
      { title: 'UI & signage', text: 'Glowing buttons, holograms, cyberpunk shop signs and laser grids — cheap, resolution-independent and animatable.' },
    ],
    try: [
      'On <b>Geometry Wars arena</b>, move the mouse around: the ship drags the grid. Click to set off a big shockwave, then raise <i>Grid warp strength</i>.',
      'Set <i>Glow radius</i> to 3 and <i>Glow intensity</i> to 2 — then down to 0: without glow, neon is just thin lines.',
      'On <b>Neon sign</b>, push <i>Flicker</i> to 1 and watch the wall’s lighting flicker with the letters.',
      'Change <i>Neon color A</i> to orange and <i>B</i> to green on the sunset for a whole new mood.',
      'Lower <i>Grid density</i> on the sunset and notice how lines near the horizon still stay smooth (fwidth anti-aliasing).',
    ],
    ask: [
      'neon glow lines drawn with signed distance fields (core + exp falloff halo)',
      'a Geometry Wars style warping background grid that reacts to explosions',
      'synthwave perspective grid floor with a striped sun',
      'a flickering neon sign that lights the wall behind it',
      'additive blending with a soft tone-mapping roll-off for glow',
    ],
    perf: `<p>Cost grows with <i>shapes × pixels</i>: each pixel evaluates every nearby distance function. The arena culls by distance (an enemy is
      only evaluated within a small radius; explosion sparks only inside their blast radius), so ~40 shapes stay cheap. For thousands of glowing shapes
      you would draw them as instanced quads into an HDR buffer and add a bloom pass instead (see <a href="#/s/bloom">Bloom</a>).</p>`,
    api: `<p>A single fragment shader: identical in WebGL2 and WebGPU. The game state travels as uniform arrays (12 enemies, 16 bullets, 8 explosions).
      With WebGPU you would keep that state in storage buffers updated by a compute shader — thousands of enemies and particles with no CPU work.</p>`,
    code: [
      {
        title: 'A neon stroke: white core + two-radius halo',
        lang: 'wgsl',
        src: `fn neonLine(d: f32, col: vec3f, width: f32) -> vec3f {
  let aa = pxSize() * 1.2;
  let core = 1.0 - smoothstep(width, width + aa, d);         // crisp tube
  let r = 0.01 * u.glowSize;
  let halo = exp(-d / r) * 0.55 + exp(-d / (r * 4.0)) * 0.22; // tight + wide glow
  return mix(col, vec3f(1.0), 0.55) * core + col * halo * u.glow;
}
// later: c = vec3f(1.0) - exp(-c * 1.15);   // soft roll-off, hot cores turn white`,
      },
      {
        title: 'Perspective grid floor',
        lang: 'wgsl',
        src: `let dy = max(p.y - hz, 0.0001);                 // distance below the horizon
let z = 0.22 / dy;                                 // depth
let wp = vec2f(p.x * z, z + t * 1.2) * (u.grid * 0.12);  // world position, scrolling
let fw = max(fwidth(wp), vec2f(0.0001));           // size of one pixel in grid units
let dl = abs(fract(wp + 0.5) - 0.5);               // distance to the nearest line
let cov = clamp((vec2f(0.035) - dl) / fw + 0.5, vec2f(0.0), vec2f(1.0));`,
      },
      {
        title: 'Warping grid: draw it where it came from',
        lang: 'wgsl',
        src: `fn warpAt(q: vec2f) -> vec2f {          // displacement from each explosion
  var d = vec2f(0.0);
  for (var i = 0; i < 8; i++) {
    let e = u.ev[i];                                       // xy = centre, z = age
    let v = q - e.xy;  let r = length(v) + 0.0001;
    let amp = 0.045 * exp(-e.z * 1.4) * u.warp;
    d += v / r * amp * exp(-pow((r - e.z * 0.55) / 0.07, 2.0));  // moving ring
  }
  return d;
}
let gp = (p - warpAt(p)) * u.grid;    // grid lines at the un-warped position`,
      },
    ],
  },
});

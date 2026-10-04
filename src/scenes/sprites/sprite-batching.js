// Sprite Batching & Instancing — every sprite lives in ONE storage buffer, a compute shader moves them,
// and ONE instanced draw call renders them all (the vertex shader reads the same buffer).
// Three uses: Bunnymark (the classic renderer benchmark), a bullet-hell pattern generator and a
// y-sorted marching crowd that uses the depth buffer instead of CPU sorting.

import { ShapeBatch, Camera2D } from '../../core/batch.js';
import { atlasTexture, tag, fmt } from './_shared.js';

const SPRITE_WGSL = /* wgsl */ `
struct Sprite { pos: vec2f, vel: vec2f, a: vec4f, b: vec4f };
`;

const PARAMS = {
  res: 'vec2f',
  time: 'f32',
  dt: 'f32',
  count: 'u32',
  spawnFrom: 'u32',
  spawnCount: 'u32',
  mode: 'u32',
  mouse: 'vec4f',
  sim: 'vec4f', // per-mode parameters (see frame())
  sim2: 'vec4f',
  spawnPos: 'vec4f',
  emit: 'array<vec4f, 4>', // bullet emitters: x, y, phase, pattern
  uvs: 'array<vec4f, 8>', // atlas rects: bunny_0, bunny_1, hero_run_0..3, hero_idle_0..1
  seed: 'u32',
  shots: 'u32',
  arms: 'u32',
  size: 'f32', // device pixels per sprite pixel
};

const COMPUTE_WGSL = /* wgsl */ `
fn rnd(i: u32, k: u32) -> f32 {
  return f32(pcg(i * 1664525u + pcg(k + u.seed * 7919u)) >> 8u) / 16777216.0;
}
fn rndS(i: u32, k: u32) -> f32 { return f32(pcg(i * 1664525u + pcg(k)) >> 8u) / 16777216.0; } // seed-independent

// ---------------------------------------------------------------- Bunnymark
@compute @workgroup_size(256) fn cs_bunny(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var s = ps[i];
  let unit = u.sim.z;
  if (i >= u.spawnFrom && i < u.spawnFrom + u.spawnCount) {
    s.pos = u.spawnPos.xy + (vec2f(rnd(i, 1u), rnd(i, 2u)) - 0.5) * u.spawnPos.z;
    // the classic bunnymark launch: speedX = rand*10, speedY = rand*10 - 5 (px per frame at 60 fps)
    // spawnPos.w = 1: launch to the right (auto spawn), 0: spray both ways (mouse spawn)
    let vx = select((rnd(i, 3u) - 0.5) * 16.0, rnd(i, 3u) * 10.0, u.spawnPos.w > 0.5);
    s.vel = vec2f(vx, rnd(i, 4u) * 10.0 - 5.0) * 60.0 * unit;
    s.a = vec4f(0.0);
  }
  s.vel.y += u.sim.x * unit * u.dt;
  s.pos += s.vel * u.dt;
  if (s.pos.x > u.res.x) { s.vel.x = -abs(s.vel.x); s.pos.x = u.res.x; }
  if (s.pos.x < 0.0) { s.vel.x = abs(s.vel.x); s.pos.x = 0.0; }
  if (s.pos.y > u.res.y) {
    s.vel.y = -abs(s.vel.y) * u.sim.y;
    s.pos.y = u.res.y;
    if (rnd(i, 7u) > 0.5) { s.vel.y -= rnd(i, 8u) * 6.0 * 60.0 * unit; }
  }
  if (s.pos.y < 0.0) { s.vel.y = 0.0; s.pos.y = 0.0; }
  ps[i] = s;
}

// ---------------------------------------------------------------- Bullet hell (ring buffer of bullets)
@compute @workgroup_size(256) fn cs_bullet(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let cap = u.count;
  if (i >= cap) { return; }
  var s = ps[i];
  let local = (i + cap - u.spawnFrom) % cap;
  if (local < u.spawnCount) {
    // which emitter / shot / arm does this slot belong to?
    let perE = u.shots * u.arms;
    let e = local / perE;
    let rem = local % perE;
    let sh = rem / u.arms;
    let arm = rem % u.arms;
    let em = u.emit[e];
    let ts = u.time - u.dt + u.dt * (f32(sh) + 1.0) / f32(u.shots); // sub-frame shot time
    let armA = f32(arm) / f32(u.arms) * TAU;
    let speed = u.sim.x * u.size * 0.5;
    var ang = armA + ts * u.sim.y + em.z;
    var turn = 0.0;
    var kind = 1.0;
    let pat = u32(em.w + 0.5);
    if (pat == 0u) { kind = 2.0; }                                         // spiral of rice bullets
    if (pat == 1u) { ang = armA + ts * u.sim.y * select(-1.0, 1.0, (arm % 2u) == 0u) + em.z; } // rosette
    if (pat == 2u) { ang = armA + ts * u.sim.y * 0.35 + em.z; turn = u.sim.z * select(-1.0, 1.0, (arm % 2u) == 0u); kind = 3.0; }
    let dir = vec2f(cos(ang), sin(ang));
    s.pos = em.xy + dir * 6.0 * u.size;
    s.vel = dir * speed;
    s.a = vec4f(0.0, turn, f32(e) + 0.15 * f32(arm % 3u), kind);
  }
  if (s.a.w < 0.5) { return; }           // kind 0 = dead slot
  s.a.x += u.dt;
  let c = cos(s.a.y * u.dt);
  let sn = sin(s.a.y * u.dt);
  s.vel = vec2f(c * s.vel.x - sn * s.vel.y, sn * s.vel.x + c * s.vel.y);
  s.pos += s.vel * u.dt;
  let m = 40.0 * u.size;
  if (s.pos.x < -m || s.pos.y < -m || s.pos.x > u.res.x + m || s.pos.y > u.res.y + m || s.a.x > 30.0) {
    s.a.w = 0.0;
  } else {
    atomicAdd(&alive[0], 1u);
  }
  ps[i] = s;
}

// ---------------------------------------------------------------- Crowd
@compute @workgroup_size(256) fn cs_crowd(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var s = ps[i];
  let unit = u.size * 0.5;
  let margin = vec2f(12.0, 20.0) * u.size;
  if (i >= u.spawnFrom && i < u.spawnFrom + u.spawnCount) {
    s.pos = margin + vec2f(rnd(i, 1u), rnd(i, 2u)) * (u.res - margin * 2.0);
    s.vel = vec2f(0.0);
    let team = floor(rnd(i, 12u) * 3.0);
    s.a = vec4f(rnd(i, 3u) * 4.0, 0.75 + 0.5 * rnd(i, 4u), team, rnd(i, 5u) * 2.5);
    s.b = vec4f(s.pos, 0.0, select(1.0, -1.0, rnd(i, 6u) < 0.5));
  }
  let rally = u.mouse.z > 0.5;
  var goal = s.b.xy;
  if (rally) {
    // run to a stable random spot in an ellipse around the cursor
    let ang = rndS(i, 7u) * TAU;
    let r = sqrt(rndS(i, 8u)) * u.res.y * 0.3;
    goal = u.mouse.xy + vec2f(cos(ang), sin(ang) * 0.55) * r;
  }
  let to = goal - s.pos;
  let d = length(to);
  let spd = u.sim.x * unit * s.a.y * select(1.0, 2.4, rally);
  if (d > 1.0) {
    s.vel = to / d * min(spd, d / max(u.dt, 0.0001));
  } else {
    s.vel = vec2f(0.0);
    if (!rally) {
      s.a.w -= u.dt;                         // idle timer
      if (s.a.w <= 0.0) {                    // pick a new nearby goal
        let ang = rnd(i, 9u) * TAU;
        let r = (20.0 + 110.0 * rnd(i, 10u)) * unit;
        s.b = vec4f(clamp(s.pos + vec2f(cos(ang), sin(ang)) * r, margin, u.res - margin), s.b.zw);
        s.a.w = 0.4 + 3.0 * rnd(i, 11u);
      }
    }
  }
  if (abs(s.vel.x) > 0.5) { s.b.w = sign(s.vel.x); }
  s.pos += s.vel * u.dt;
  ps[i] = s;
}
`;

const DRAW_WGSL = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) local: vec2f,
  @location(2) @interpolate(flat) tint: vec4f,
  @location(3) @interpolate(flat) misc: vec4f,
};
fn toClip(p: vec2f, z: f32) -> vec4f { return vec4f(p.x / u.res.x * 2.0 - 1.0, 1.0 - p.y / u.res.y * 2.0, z, 1.0); }
fn corner(vi: u32) -> vec2f {
  var c = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  return c[vi % 6u];
}

// ---------------------------------------------------------------- bunnies
@vertex fn vs_bunny(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let c = corner(vi);
  let s = ps[ii];
  let r = u.uvs[select(0u, 1u, s.vel.y < 0.0)];          // ears flap while rising
  let p = s.pos + (c - vec2f(0.5, 1.0)) * 16.0 * u.size;
  var o: VOut;
  o.pos = toClip(p, 0.0);
  o.uv = mix(r.xy, r.zw, c);
  o.local = c;
  o.tint = vec4f(1.0);
  if (u.sim.w > 0.5) { o.tint = vec4f(hsv2rgb(vec3f(fract(f32(ii) * 0.6180339), 0.5, 1.0)), 1.0); }
  o.misc = vec4f(0.0);
  return o;
}
@fragment fn fs_bunny(i: VOut) -> @location(0) vec4f {
  let t = textureSampleLevel(atlas, samp, i.uv, 0.0);
  if (t.a < 0.5) { discard; }
  return vec4f(t.rgb * i.tint.rgb, 1.0);
}

// ---------------------------------------------------------------- bullets (premultiplied: core is opaque, halo is additive)
@vertex fn vs_bullet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let s = ps[ii];
  var o: VOut;
  if (s.a.w < 0.5) { o.pos = vec4f(0.0, 0.0, 2.0, 1.0); return o; }   // dead slot: clipped away
  let c = corner(vi) * 2.0 - 1.0;                    // -1..1
  let kind = u32(s.a.w + 0.5);
  var axes = vec2f(4.0, 4.0);                         // half length, half width in sprite px
  if (kind == 2u) { axes = vec2f(6.5, 3.0); }
  if (kind == 3u) { axes = vec2f(3.2, 3.2); }
  let grow = min(1.0, s.a.x * 14.0);                  // pop in
  let dir = normalize(s.vel + vec2f(0.0001, 0.0));
  let nrm = vec2f(-dir.y, dir.x);
  let q = c * 2.2;                                    // quad covers the glow halo
  let w = s.pos + (dir * q.x * axes.x + nrm * q.y * axes.y) * u.size * grow;
  o.pos = toClip(w, 0.0);
  o.uv = vec2f(0.0);
  o.local = q;
  let hue = fract(0.93 + s.a.z * 0.29);
  o.tint = vec4f(hsv2rgb(vec3f(hue, 0.85, 1.0)), 1.0);
  o.misc = vec4f(f32(kind), 0.0, 0.0, 0.0);
  return o;
}
@fragment fn fs_bullet(i: VOut) -> @location(0) vec4f {
  let d = length(i.local);                       // 1.0 = bullet edge
  let aa = fwidth(d) + 0.001;
  let body = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);
  let core = 1.0 - smoothstep(0.5 - aa, 0.55 + aa, d);
  let rim = mix(i.tint.rgb, vec3f(1.0), core);
  let glow = exp(-max(d - 1.0, 0.0) * 2.2) * 0.55 * (1.0 - body);
  // premultiplied output: body is opaque (alpha 1), halo adds light (alpha 0)
  return vec4f(rim * body + i.tint.rgb * glow, body);
}

// ---------------------------------------------------------------- crowd (shadows, then depth-sorted soldiers)
@vertex fn vs_shadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let c = corner(vi);
  let s = ps[ii];
  let p = s.pos + (c - vec2f(0.5, 0.5)) * vec2f(11.0, 4.0) * u.size;
  var o: VOut;
  o.pos = toClip(p, 0.99);
  o.uv = vec2f(0.0);
  o.local = c * 2.0 - 1.0;
  o.tint = vec4f(0.0);
  o.misc = vec4f(0.0);
  return o;
}
@fragment fn fs_shadow(i: VOut) -> @location(0) vec4f {
  let d = length(i.local);
  let a = (1.0 - smoothstep(0.6, 1.0, d)) * 0.32;
  if (a < 0.01) { discard; }
  return vec4f(0.02, 0.06, 0.03, a);
}
@vertex fn vs_soldier(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let c = corner(vi);
  let s = ps[ii];
  let moving = dot(s.vel, s.vel) > 1.0;
  var f = 6u + u32(floor(u.time * 2.5 + s.a.x)) % 2u;                     // idle: 2 frames
  if (moving) { f = 2u + u32(floor(u.time * u.sim.z * s.a.y + s.a.x)) % 4u; } // run: 4 frames
  let r = u.uvs[f];
  let p = s.pos + (c - vec2f(0.5, 1.0)) * 16.0 * u.size;
  var o: VOut;
  // y-sorting via the depth buffer: lower on screen = nearer = smaller depth
  o.pos = toClip(p, 0.9 - 0.8 * clamp(s.pos.y / u.res.y, 0.0, 1.0));
  var cu = c.x;
  if (s.b.w < 0.0) { cu = 1.0 - c.x; }                                     // face left: flip U
  o.uv = mix(r.xy, r.zw, vec2f(cu, c.y));
  o.local = c;
  o.tint = vec4f(1.0);
  o.misc = vec4f(s.a.z, 0.0, 0.0, 0.0);
  return o;
}
fn near3(a: vec3f, b: vec3f) -> bool { let d = abs(a - b); return max(d.x, max(d.y, d.z)) < 0.02; }
@fragment fn fs_soldier(i: VOut) -> @location(0) vec4f {
  let t = textureSampleLevel(atlas, samp, i.uv, 0.0);
  if (t.a < 0.5) { discard; }                  // alpha-test so the depth buffer gets a clean cut-out
  var c = t.rgb;
  let team = u32(i.misc.x + 0.5);
  if (u.sim.y > 0.5 && team > 0u) {
    // palette swap: replace the blue tunic (#3b5dc9) and its shadow (#29366f) with team colors
    var hi = vec3f(0.80, 0.24, 0.27);
    var lo = vec3f(0.45, 0.10, 0.18);
    if (team == 2u) { hi = vec3f(0.30, 0.72, 0.36); lo = vec3f(0.12, 0.40, 0.28); }
    if (near3(c, vec3f(0.231, 0.365, 0.788))) { c = hi; }
    if (near3(c, vec3f(0.161, 0.212, 0.435))) { c = lo; }
  }
  return vec4f(c, 1.0);
}
`;

const BG_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let m = u.mode;
  if (m == 0u) {
    // Bunnymark: calm slate with a soft checkerboard
    let g = floor(px / (32.0 * u.size));
    let chk = fmod(g.x + g.y, 2.0);
    var c = mix(vec3f(0.11, 0.13, 0.19), vec3f(0.07, 0.08, 0.12), uv.y);
    c += vec3f(0.018) * chk;
    return vec4f(c, 1.0);
  }
  if (m == 1u) {
    // shmup arena: deep violet, scrolling grid, vignette
    let p = px / u.size;
    var c = mix(vec3f(0.05, 0.02, 0.10), vec3f(0.01, 0.01, 0.04), uv.y);
    let gp = (p + vec2f(0.0, u.time * 20.0)) / 24.0;
    let gl = abs(fract(gp) - 0.5);
    let line = 1.0 - smoothstep(0.0, 0.035, min(gl.x, gl.y));
    c += vec3f(0.18, 0.08, 0.32) * line * 0.35;
    let st = hash21(floor((p + vec2f(0.0, -u.time * 45.0)) / 2.0));
    c += vec3f(0.55, 0.6, 0.8) * step(0.9975, st);
    let v = length(uv - 0.5);
    c *= 1.0 - v * 0.9;
    return vec4f(c, 1.0);
  }
  // crowd: pixel-art meadow
  let p = floor(px / u.size);
  let n = fbm(p * 0.006, 3) * 0.5 + 0.5;
  var c = mix(vec3f(0.24, 0.50, 0.24), vec3f(0.36, 0.62, 0.27), smoothstep(0.3, 0.7, n));
  let h = hash21(p);
  if (h > 0.985) { c *= 1.18; }
  if (h < 0.012) { c *= 0.82; }
  let f = hash21(floor(p / 3.0) + vec2f(7.0, 3.0));
  if (f > 0.9975 && fract(p.x / 3.0) < 0.4 && fract(p.y / 3.0) < 0.4) { c = vec3f(0.96, 0.9, 0.55); }
  return vec4f(c, 1.0);
}
`;

const MAX_DRAWS = 20000;

export default {
  interaction: 'Hold the mouse button on the canvas — spawn bunnies / pull the bullet emitters / rally the army.',
  examples: [
    {
      id: 'bunny',
      label: 'Bunnymark',
      kind: 'Benchmark',
      note: 'The classic 2D renderer stress test (from PixiJS): every bunny bounces with gravity. <b>Hold the mouse</b> to spawn more and watch the counter — the sprite count climbs into the hundreds of thousands while the draw-call count stays at <b>1</b>.',
      hint: 'Hold the mouse to spawn bunnies',
      params: { size: 2 },
    },
    {
      id: 'bullets',
      label: 'Bullet hell',
      kind: 'In a game',
      note: 'Emitters fire rotating spirals, counter-rotating rosettes and curving galaxy arms. Every bullet is a slot in a GPU ring buffer: the compute shader spawns, moves, curves and kills them; one draw renders all of them.',
      hint: 'Hold the mouse to pull the emitters',
      params: { size: 2 },
    },
    {
      id: 'crowd',
      label: 'Crowd / army',
      kind: 'In a game',
      note: 'Thousands of animated knights wander, idle and turn around. Each picks its animation frame on the GPU from its own phase offset, gets a team color by palette swap, and is y-sorted for free by the <b>depth buffer</b>. <b>Hold the mouse</b> to rally them.',
      hint: 'Hold the mouse to rally the army',
      params: { size: 1 },
    },
  ],
  controls: [
    { type: 'slider', key: 'size', label: 'Sprite scale', min: 1, max: 6, step: 1, value: 2, help: 'Whole-number scaling keeps pixel art crisp. Bigger sprites = more pixels to fill.' },
    { type: 'toggle', key: 'batched', label: 'Batch into one draw call', value: true, showFor: ['bunny', 'crowd'], help: 'Off = one draw call per sprite (capped at 20,000). Watch the CPU time in the readout.' },
    { type: 'heading', label: 'Bunnies', showFor: ['bunny'] },
    { type: 'slider', key: 'spawnRate', label: 'Spawn per frame (hold mouse)', min: 10, max: 10000, step: 10, value: 500, log: true, showFor: ['bunny'], format: (v) => fmt(v) },
    { type: 'slider', key: 'gravity', label: 'Gravity', min: 0, max: 3000, step: 10, value: 1500, showFor: ['bunny'], help: 'Pixels per second² (per sprite pixel).' },
    { type: 'slider', key: 'bounce', label: 'Bounciness', min: 0.3, max: 1, step: 0.01, value: 0.85, showFor: ['bunny'] },
    { type: 'toggle', key: 'tint', label: 'Rainbow tint per instance', value: false, showFor: ['bunny'], help: 'Per-instance data (here: its index) can drive color for free.' },
    { type: 'button', key: 'add10k', label: '+10,000', showFor: ['bunny'] },
    { type: 'button', key: 'add100k', label: '+100,000', showFor: ['bunny'] },
    { type: 'button', key: 'reset', label: 'Clear', showFor: ['bunny'] },
    { type: 'heading', label: 'Emitters', showFor: ['bullets'] },
    {
      type: 'select',
      key: 'pattern',
      label: 'Pattern',
      value: 'mixed',
      showFor: ['bullets'],
      options: [
        { value: 'mixed', label: 'Mixed (one per emitter)' },
        { value: 'spiral', label: 'Spiral (rice bullets)' },
        { value: 'rosette', label: 'Rosette (counter-rotating)' },
        { value: 'galaxy', label: 'Galaxy (curving bullets)' },
      ],
    },
    { type: 'slider', key: 'emitters', label: 'Emitters', min: 1, max: 4, step: 1, value: 3, showFor: ['bullets'] },
    { type: 'slider', key: 'fireRate', label: 'Shots per second', min: 2, max: 120, step: 1, value: 20, showFor: ['bullets'], help: 'Per emitter. Each shot fires one bullet per arm.' },
    { type: 'slider', key: 'arms', label: 'Arms per shot', min: 1, max: 64, step: 1, value: 10, showFor: ['bullets'] },
    { type: 'slider', key: 'speed', label: 'Bullet speed', min: 40, max: 600, step: 1, value: 150, showFor: ['bullets'] },
    { type: 'slider', key: 'spin', label: 'Spin', min: -4, max: 4, step: 0.01, value: 1.2, showFor: ['bullets'], help: 'How fast the pattern rotates (radians per second).' },
    { type: 'slider', key: 'curve', label: 'Bullet curve', min: 0, max: 2, step: 0.01, value: 0.7, showFor: ['bullets'], help: 'Turn rate of galaxy bullets — they steer in flight.' },
    { type: 'heading', label: 'Army', showFor: ['crowd'] },
    { type: 'slider', key: 'count', label: 'Soldiers', min: 500, max: 300000, step: 100, value: 3000, log: true, showFor: ['crowd'], format: (v) => fmt(v) },
    { type: 'slider', key: 'walkSpeed', label: 'Walk speed', min: 5, max: 150, step: 1, value: 40, showFor: ['crowd'] },
    { type: 'slider', key: 'animFps', label: 'Run animation fps', min: 2, max: 24, step: 1, value: 10, showFor: ['crowd'] },
    { type: 'toggle', key: 'ysort', label: 'Y-sort with the depth buffer', value: true, showFor: ['crowd'], help: 'Off = drawn in buffer order: knights further back paint over the ones in front.' },
    { type: 'toggle', key: 'teams', label: 'Team colors (palette swap)', value: true, showFor: ['crowd'] },
    { type: 'toggle', key: 'shadows', label: 'Shadows', value: true, showFor: ['crowd'] },
  ],
  about: {
    summary: 'A draw call is an expensive “please draw this” request from the CPU. Batching and instancing turn 100,000 sprites into ONE draw call — and with compute shaders the CPU doesn’t even touch the sprites.',
    what: `<p>Three scenes that all draw every sprite with a <b>single draw call</b>. The readout shows the sprite count, the number of draw calls
      and how long the CPU spent recording them. <b>Bunnymark</b> is the classic benchmark, <b>Bullet hell</b> shows GPU-spawned bullet patterns,
      and <b>Crowd</b> shows animated, team-colored, depth-sorted characters.</p>`,
    how: `<ol>
      <li><b>Draw call</b>: each time the CPU says “draw this mesh with this texture and shader”, the driver validates state and talks to the GPU.
        That costs microseconds of CPU — fine 100 times per frame, fatal 100,000 times.</li>
      <li><b>Batching</b>: put many sprites into one vertex/instance buffer that share one texture (an <i>atlas</i>) and one shader → one call.
        Anything that changes texture, shader or blend mode “breaks the batch”.</li>
      <li><b>Instancing</b>: the GPU draws the same 6-vertex quad <i>N</i> times. The vertex shader gets <code>instance_index</code> and reads
        that sprite’s position, animation frame and color from a buffer.</li>
      <li><b>GPU-driven</b>: here that buffer is a <b>storage buffer</b> updated by a <b>compute shader</b> (one thread per sprite: physics, spawning,
        AI, animation). The CPU only uploads a few uniforms — the per-sprite data never leaves the GPU.</li>
      <li><b>Y-sorting without sorting</b>: the crowd writes each knight’s feet-y into the depth buffer. With alpha-tested (cut-out) pixel art, the
        depth test hides whatever is behind — order-independent, so no CPU sort of 100k sprites.</li>
      <li><b>Bullets as a ring buffer</b>: new bullets overwrite the oldest slots; the compute shader derives each bullet’s emitter, arm and
        angle from its slot index, so spawning 10,000 bullets costs zero JavaScript loops.</li>
    </ol>`,
    uses: [
      { title: 'Bullet hells & shmups', text: 'Touhou, Enter the Gungeon, Vampire Survivors-style hordes: thousands of projectiles and enemies.' },
      { title: 'Crowds & RTS', text: 'Armies, zombie hordes, city pedestrians, flocks — instanced sprites with per-instance animation.' },
      { title: 'Particles & debris', text: 'Coins, sparks, confetti, grass blades: anything small and numerous.' },
      { title: 'Tile & text rendering', text: 'Every engine batches tilemaps and glyphs the same way (see GPU Tilemaps / Text).' },
    ],
    try: [
      'In <b>Bunnymark</b>, press <i>+100,000</i> a few times and keep an eye on fps and the “draw calls: 1” line.',
      'Turn off <i>Batch into one draw call</i>: the image is identical, but the CPU time to record the frame jumps by orders of magnitude.',
      'In <b>Crowd</b>, turn off <i>Y-sort</i> and hold the mouse: knights in the back paint over the ones in front.',
      'In <b>Bullet hell</b>, set <i>Arms</i> to 64 and <i>Shots per second</i> to 120: tens of thousands of live bullets.',
      'Raise <i>Sprite scale</i> with 500k bunnies: fps drops because of pixels filled (fill-rate), not because of the number of sprites.',
    ],
    ask: [
      'instanced sprite rendering with one draw call per texture atlas',
      'GPU-driven sprites: a compute shader updates a storage buffer the vertex shader reads',
      'a bunnymark benchmark for my renderer',
      'ring-buffer bullet spawning on the GPU for a bullet-hell pattern system',
      'y-sorting via the depth buffer with alpha-tested sprites',
      'per-instance palette swap / team colors',
    ],
    perf: `<p>CPU cost is now ~constant (a few uniforms + 2–3 dispatches/draws). GPU cost = <i>vertex work</i> (6 vertices × sprites, trivial)
      + <i>fill-rate</i> (pixels covered × overdraw) — big overlapping sprites are what eventually slows things down, so the sprite scale matters more than
      the count. The un-batched mode shows the other bottleneck: ~1–5 µs of CPU per draw call.</p>`,
    api: `<p><b>Compute: WebGPU only.</b> WebGL2 can instance (<code>drawArraysInstanced</code>) and batch just as well, but has no compute shaders or
      storage buffers, so the simulation would run on the CPU (uploading the instance buffer every frame) or be faked with
      transform feedback / texture ping-pong. WebGPU lets the GPU spawn, simulate and draw without any round trip.</p>`,
    code: [
      {
        title: 'Vertex shader: one quad per instance, data from the storage buffer',
        lang: 'wgsl',
        src: `@vertex fn vs_bunny(@builtin(vertex_index) vi: u32,
                     @builtin(instance_index) ii: u32) -> VOut {
  let c = corner(vi);                              // 0..1 quad corner
  let s = ps[ii];                                  // this sprite's state
  let r = u.uvs[select(0u, 1u, s.vel.y < 0.0)];    // atlas rect (frame)
  let p = s.pos + (c - vec2f(0.5, 1.0)) * 16.0 * u.size;
  var o: VOut;
  o.pos = toClip(p, 0.0);
  o.uv = mix(r.xy, r.zw, c);
  return o;
}`,
      },
      {
        title: 'Compute shader: spawn + bounce, one thread per bunny',
        lang: 'wgsl',
        src: `@compute @workgroup_size(256) fn cs_bunny(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var s = ps[i];
  if (i >= u.spawnFrom && i < u.spawnFrom + u.spawnCount) { /* init new bunny */ }
  s.vel.y += gravity * u.dt;
  s.pos += s.vel * u.dt;
  if (s.pos.y > u.res.y) { s.vel.y = -abs(s.vel.y) * bounce; }
  ps[i] = s;
}`,
      },
      {
        title: 'JavaScript: the whole army in one call',
        lang: 'js',
        src: `update.dispatch(enc, 'cs_crowd', Math.ceil(count / 256), res);
const pass = enc.beginRenderPass({ colorAttachments: [...], depthStencilAttachment: depth });
pass.setPipeline(soldierPipe);           // depthCompare: 'less-equal'
pass.setBindGroup(0, draw.bind(res));
pass.draw(6, count);                      // 6 vertices × count instances
pass.end();`,
      },
    ],
    links: [
      { title: 'PixiJS Bunnymark', url: 'https://www.goodboydigital.com/pixijs/bunnymark/', note: 'the original benchmark' },
      { title: 'WebGPU Fundamentals', url: 'https://webgpufundamentals.org/', note: 'storage buffers, instancing, compute' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const { atlas, view: atlasView } = await atlasTexture(gpu);
    const MAX = ctx.testMode ? 40000 : 1000000;
    const BULLET_CAP = ctx.testMode ? 16384 : 262144;
    const U = gpu.uniforms(PARAMS, 'Params');
    const buf = gpu.storage(MAX * 48, 'sprites');
    const alive = gpu.storage(16, 'alive');
    const readBuf = gpu.buffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'alive-read' });

    const uvNames = ['bunny_0', 'bunny_1', 'hero_run_0', 'hero_run_1', 'hero_run_2', 'hero_run_3', 'hero_idle_0', 'hero_idle_1'];
    const uvArr = new Float32Array(32);
    uvNames.forEach((n, i) => uvArr.set(atlas.uv(n), i * 4));
    U.set('uvs', uvArr);

    const update = gpu.compute({
      label: 'sprites-update',
      bindings: {
        u: { uniform: U },
        ps: { storage: 'array<Sprite>', access: 'read_write' },
        alive: { storage: 'array<atomic<u32>>', access: 'read_write' },
      },
      include: ['hash'],
      code: SPRITE_WGSL + COMPUTE_WGSL,
    });
    const draw = gpu.program({
      label: 'sprites-draw',
      bindings: {
        u: { uniform: U },
        ps: { storage: 'array<Sprite>', access: 'read' },
        atlas: { texture: true },
        samp: { sampler: true },
      },
      include: ['color'],
      code: SPRITE_WGSL + DRAW_WGSL,
    });
    const bg = gpu.fullscreen({ label: 'sprites-bg', code: BG_WGSL, uniforms: U, include: ['noise'] });

    const depthFmt = 'depth24plus';
    const depthState = (on) => ({ format: depthFmt, depthWriteEnabled: on, depthCompare: on ? 'less-equal' : 'always' });
    const pipes = {
      bunny: draw.renderPipeline({ format: gpu.format, blend: 'alpha', vs: 'vs_bunny', fs: 'fs_bunny' }),
      bullet: draw.renderPipeline({ format: gpu.format, blend: 'premultiplied', vs: 'vs_bullet', fs: 'fs_bullet' }),
      shadow: draw.renderPipeline({ format: gpu.format, blend: 'alpha', vs: 'vs_shadow', fs: 'fs_shadow', depth: depthState(false) }),
      soldier: draw.renderPipeline({ format: gpu.format, vs: 'vs_soldier', fs: 'fs_soldier', depth: depthState(true) }),
      soldierNaive: draw.renderPipeline({ format: gpu.format, vs: 'vs_soldier', fs: 'fs_soldier', depth: depthState(false) }),
    };
    let depth = null;
    const ensureDepth = (w, h) => {
      if (depth && depth.width === w && depth.height === h) return;
      depth?.destroy();
      depth = gpu.texture({ size: [w, h], format: depthFmt, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'crowd-depth' });
    };

    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();

    const readout = tag(ctx, 'right:8px;top:48px;text-align:right;line-height:1.45');

    // ---- simulation state (CPU side only knows counts) ----
    let count = 0; // live bunnies / soldiers
    let spawnFrom = 0;
    let spawnCount = 0;
    let spawnPos = [0, 0, 0, 0];
    let head = 0; // bullet ring head
    let shotAcc = 0;
    let lastShots = 1;
    let needClear = true;
    let readState = 'idle';
    let aliveCount = 0;
    let cpuMs = 0;
    let frameNo = 0;
    let lastMode = null;
    let pendingAdd = 0;
    const emitPos = [];

    const resetMode = (id) => {
      count = 0;
      spawnFrom = 0;
      spawnCount = 0;
      head = 0;
      shotAcc = 0;
      needClear = true;
      aliveCount = 0;
      emitPos.length = 0;
      if (id === 'bunny') pendingAdd = ctx.testMode ? 1500 : 3000;
    };
    resetMode(ctx.example);
    lastMode = ctx.example;

    const doRead = () => {
      if (readState === 'copied') {
        readState = 'mapping';
        readBuf
          .mapAsync(GPUMapMode.READ)
          .then(() => {
            aliveCount = new Uint32Array(readBuf.getMappedRange())[0];
            readBuf.unmap();
            readState = 'idle';
          })
          .catch(() => (readState = 'idle'));
      }
    };

    return {
      resize(w, h) {
        ensureDepth(w, h);
      },
      onExample(id) {
        resetMode(id);
        lastMode = id;
      },
      onAction(key) {
        if (key === 'reset') resetMode(ctx.example);
        if (key === 'add10k') pendingAdd += 10000;
        if (key === 'add100k') pendingAdd += 100000;
      },
      frame(ctx) {
        const p = ctx.params;
        const ex = ctx.example;
        if (ex !== lastMode) {
          resetMode(ex);
          lastMode = ex;
        }
        const W = ctx.width;
        const H = ctx.height;
        const size = Math.max(1, Math.round(p.size)) * Math.max(1, Math.round(ctx.dpr));
        const dt = Math.min(ctx.dt, 1 / 20);
        const mode = ex === 'bunny' ? 0 : ex === 'bullets' ? 1 : 2;
        const ptr = ctx.pointer;
        const holding = ptr.down && ptr.over !== false;
        frameNo++;
        doRead();
        ensureDepth(W, H);
        spawnCount = 0;
        const running = !ctx.paused;

        // ---- spawning decisions (CPU decides HOW MANY, the GPU initialises them) ----
        if (mode === 0 && running) {
          let add = pendingAdd;
          pendingAdd = 0;
          if (holding) add += Math.round(p.spawnRate);
          add = Math.min(add, MAX - count);
          if (add > 0) {
            spawnFrom = count;
            spawnCount = add;
            spawnPos = holding ? [ptr.x, ptr.y, 8 * size, 0] : [W * 0.15, H * 0.1, W * 0.25, 1];
            count += add;
          }
        } else if (mode === 2 && running) {
          const want = Math.min(MAX, Math.round(ctx.testMode ? Math.min(p.count, 1500) : p.count));
          if (want > count) {
            spawnFrom = count;
            spawnCount = want - count;
          }
          count = want;
        } else if (mode === 1) {
          count = BULLET_CAP;
          const E = Math.round(p.emitters);
          const arms = Math.round(p.arms);
          // emitters drift on Lissajous paths, or flock to the cursor while the mouse is held
          for (let e = 0; e < 4; e++) {
            const a = (e / E) * Math.PI * 2;
            const t = ctx.time;
            let tx = W * 0.5 + Math.cos(a + t * 0.21) * W * 0.27 * (E > 1 ? 1 : 0);
            let ty = H * 0.42 + Math.sin(a * 2 + t * 0.33) * H * 0.18 * (E > 1 ? 1 : 0.4);
            if (holding) {
              tx = ptr.x + Math.cos(a + t) * 40 * size * (E > 1 ? 1 : 0);
              ty = ptr.y + Math.sin(a + t) * 40 * size * (E > 1 ? 1 : 0);
            }
            if (!emitPos[e]) emitPos[e] = [tx, ty];
            const k = 1 - Math.exp(-(holding ? 8 : 3) * dt);
            emitPos[e][0] += (tx - emitPos[e][0]) * k;
            emitPos[e][1] += (ty - emitPos[e][1]) * k;
            const pat = p.pattern === 'mixed' ? e % 3 : { spiral: 0, rosette: 1, galaxy: 2 }[p.pattern] ?? 0;
            emitPos[e][2] = e * 1.7;
            emitPos[e][3] = pat;
          }
          const em = new Float32Array(16);
          for (let e = 0; e < 4; e++) em.set(emitPos[e], e * 4);
          U.set('emit', em);
          if (running) {
            shotAcc += p.fireRate * dt;
            let shots = Math.min(16, Math.floor(shotAcc));
            shotAcc -= shots;
            while (shots > 0 && E * shots * arms > BULLET_CAP / 4) shots--;
            if (shots > 0) {
              spawnFrom = head;
              spawnCount = E * shots * arms;
              head = (head + spawnCount) % BULLET_CAP;
              lastShots = shots;
            }
          }
          U.set('shots', Math.max(1, lastShots)).set('arms', arms);
        }

        // ---- uniforms ----
        U.set('res', [W, H])
          .set('time', ctx.time)
          .set('dt', running ? dt : 0)
          .set('count', count)
          .set('spawnFrom', spawnFrom)
          .set('spawnCount', spawnCount)
          .set('mode', mode)
          .set('mouse', [ptr.x, ptr.y, holding ? 1 : 0, ptr.over ? 1 : 0])
          .set('spawnPos', spawnPos)
          .set('seed', frameNo)
          .set('size', size);
        if (mode === 0) U.set('sim', [p.gravity, p.bounce, size / 2, p.tint ? 1 : 0]);
        if (mode === 1) U.set('sim', [p.speed, p.spin, p.curve, 0]);
        if (mode === 2) U.set('sim', [p.walkSpeed, p.teams ? 1 : 0, p.animFps, 0]);
        U.upload();

        const enc = ctx.encoder;
        if (needClear) {
          enc.clearBuffer(buf);
          needClear = false;
        }
        // ---- compute: one dispatch updates every sprite ----
        const res = { u: U, ps: buf, alive, atlas: atlasView, samp: 'nearest' };
        if (running && count > 0) {
          if (mode === 1) enc.clearBuffer(alive);
          update.dispatch(enc, ['cs_bunny', 'cs_bullet', 'cs_crowd'][mode], Math.ceil(count / 256), res);
          if (mode === 1 && readState === 'idle') {
            enc.copyBufferToBuffer(alive, 0, readBuf, 0, 4);
            readState = 'copied';
          }
        }

        // ---- background ----
        const canvas = { view: ctx.target, format: gpu.format };
        bg.draw(enc, canvas);

        // ---- the sprites: ONE draw call (or one per sprite when un-batched) ----
        const t0 = performance.now();
        const batched = mode === 1 || p.batched;
        const n = batched ? count : Math.min(count, MAX_DRAWS);
        let draws = 0;
        const pass = enc.beginRenderPass({
          label: 'sprites',
          colorAttachments: [{ view: ctx.target, loadOp: 'load', storeOp: 'store' }],
          depthStencilAttachment:
            mode === 2 ? { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' } : undefined,
        });
        pass.setBindGroup(0, draw.bind(res));
        const issue = () => {
          if (batched) {
            pass.draw(6, n);
            draws++;
          } else {
            for (let i = 0; i < n; i++) pass.draw(6, 1, 0, i);
            draws += n;
          }
        };
        if (count > 0) {
          if (mode === 0) {
            pass.setPipeline(pipes.bunny);
            issue();
          } else if (mode === 1) {
            pass.setPipeline(pipes.bullet);
            issue();
          } else {
            if (p.shadows) {
              pass.setPipeline(pipes.shadow);
              pass.draw(6, count);
              draws++;
            }
            pass.setPipeline(p.ysort ? pipes.soldier : pipes.soldierNaive);
            issue();
          }
        }
        pass.end();
        const ms = performance.now() - t0;
        cpuMs = cpuMs * 0.85 + ms * 0.15;

        // ---- emitters (a few shapes, drawn separately) ----
        if (mode === 1) {
          cam.setViewport(W, H);
          shapes.begin();
          const E = Math.round(p.emitters);
          for (let e = 0; e < E; e++) {
            const [x, y] = emitPos[e];
            const hue = [[1, 0.35, 0.65], [0.35, 0.8, 1], [1, 0.8, 0.3], [0.6, 1, 0.5]][e];
            shapes.circle(x, y, 7 * size, [hue[0], hue[1], hue[2], 1], { glow: 10 * size, glowStrength: 0.7 });
            shapes.circle(x, y, 4 * size, '#ffffff');
            for (let k = 0; k < 3; k++) {
              const a = ctx.time * 2 + (k / 3) * Math.PI * 2 + e;
              shapes.circle(x + Math.cos(a) * 11 * size, y + Math.sin(a) * 11 * size, 1.6 * size, [hue[0], hue[1], hue[2], 1]);
            }
          }
          shapes.flush(enc, canvas, cam);
        }

        // ---- readout ----
        const what = mode === 0 ? 'Bunnies' : mode === 1 ? 'Live bullets' : 'Knights';
        const shown = mode === 1 ? aliveCount : count;
        const capped = !batched && count > MAX_DRAWS ? ` <span style="color:#fca5a5">(capped at ${fmt(MAX_DRAWS)} draws)</span>` : '';
        readout.innerHTML =
          `<b style="font-size:14px;color:#fcd34d">${what}: ${fmt(shown)}</b><br>` +
          `draw calls: <b style="color:${draws > 2 ? '#fca5a5' : '#86efac'}">${fmt(draws)}</b>${capped}<br>` +
          `CPU record: ${cpuMs.toFixed(cpuMs < 1 ? 3 : 1)} ms` +
          (mode === 1 ? `<br>ring buffer: ${fmt(BULLET_CAP)} slots` : '');
      },
    };
  },
};

// Lightning & Electric Arcs.
//   bolts:      recursive midpoint displacement with branching (CPU, a few hundred segments)
//   rendering:  glowing capsules with MAX blending (no bright beads at the joints) into a bolt buffer
//   afterimage: a feedback buffer, persist = max(persist * decay, bolt), tinted and added back
//   flash:      the backdrop shaders receive the flash intensity and where it came from

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas } from '../../core/assets.js';
import { makeStrip, createBloom, TONEMAP_WGSL, tag, GlowLines, softSprites, cellUV, TEX, prng, clamp, hexLin } from './_shared.js';

/** Midpoint displacement with branches. Returns segments {x1,y1,x2,y2,lvl}. */
export function makeBolt(ax, ay, bx, by, P, rnd) {
  let segs = [{ x1: ax, y1: ay, x2: bx, y2: by, lvl: 0 }];
  let off = Math.hypot(bx - ax, by - ay) * P.jag;
  for (let g = 0; g < P.detail; g++) {
    const next = [];
    for (const s of segs) {
      const dx = s.x2 - s.x1;
      const dy = s.y2 - s.y1;
      const L = Math.hypot(dx, dy) || 1;
      const nx = -dy / L;
      const ny = dx / L;
      // move the midpoint sideways by a random amount (shrinks every generation)
      const o = (rnd() - 0.5) * 2 * off * (s.lvl ? 0.6 : 1);
      const mx = (s.x1 + s.x2) / 2 + nx * o;
      const my = (s.y1 + s.y2) / 2 + ny * o;
      next.push({ x1: s.x1, y1: s.y1, x2: mx, y2: my, lvl: s.lvl }, { x1: mx, y1: my, x2: s.x2, y2: s.y2, lvl: s.lvl });
      // sometimes fork: continue roughly along the first half, shorter and thinner
      if (s.lvl < 3 && next.length < 3000 && rnd() < P.branch * (g === 0 ? 0.4 : 1) / (1 + s.lvl)) {
        const ang = Math.atan2(my - s.y1, mx - s.x1) + (rnd() - 0.5) * 1.3;
        const bl = L * P.branchLen * (0.6 + rnd() * 0.8);
        next.push({ x1: mx, y1: my, x2: mx + Math.cos(ang) * bl, y2: my + Math.sin(ang) * bl, lvl: s.lvl + 1 });
      }
    }
    segs = next;
    off *= 0.5;
  }
  return segs;
}

const BG_WGSL = /* wgsl */ `
fn aa(d: f32) -> f32 { return clamp(0.5 - d * u.resolution.y, 0.0, 1.0); }
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let aspect = res.x / res.y;
  let p = px / res.y;
  let gy = u.groundY / res.y;
  let fl = u.flash;
  let lp = u.lightPos / res.y;
  var c: vec3f;
  if (u.mode == 1u) {
    // chain lightning: dusky field, lit by the bolts
    c = mix(vec3f(0.02, 0.015, 0.05), vec3f(0.06, 0.03, 0.08), p.y);
    let hill = gy - 0.12 - 0.05 * sin(p.x * 4.0) - 0.03 * perlin(vec2f(p.x * 6.0, 1.0));
    c = mix(c, vec3f(0.03, 0.025, 0.05), aa(hill - p.y));
    if (p.y > gy) { c = vec3f(0.035, 0.03, 0.04) * (1.0 - (p.y - gy) * 1.5); }
    let d = length((p - lp) * vec2f(1.0, 1.6));
    c += vec3f(0.5, 0.6, 1.0) * fl * (0.15 + exp(-d * 3.0) * 0.6) * select(0.4, 1.0, p.y > gy);
  } else if (u.mode == 2u) {
    // storm: churning clouds, sheet lightning, silhouettes revealed by flashes
    let cl = fbm(vec2f(p.x * 1.6 + u.time * 0.03, p.y * 3.0 - u.time * 0.01), 5) * 0.5 + 0.5;
    c = mix(vec3f(0.01, 0.012, 0.022), vec3f(0.035, 0.04, 0.06), smoothstep(0.35, 0.75, cl)) * (1.0 - p.y * 0.4);
    // light inside the clouds: from the strike, and from distant sheet lightning (no bolt)
    let dl = length((p - lp) * vec2f(0.6, 1.0));
    let lit = fl * (0.25 + exp(-dl * 2.5)) + u.sheet.z * exp(-length((p - u.sheet.xy / res.y) * vec2f(0.5, 1.4)) * 3.0);
    c += vec3f(0.55, 0.6, 0.85) * lit * (0.3 + 0.9 * smoothstep(0.45, 0.8, cl)) * smoothstep(0.75, 0.2, p.y);
    // far hills + near hill with trees and a house, black against the flashing sky
    let h1 = gy - 0.1 - 0.06 * sin(p.x * 3.0 + 1.0) - 0.04 * perlin(vec2f(p.x * 5.0, 2.0));
    c = mix(c, vec3f(0.012, 0.012, 0.02) + vec3f(0.05, 0.055, 0.08) * fl * 0.5, aa(h1 - p.y));
    var h2 = gy + 0.02 - 0.03 * sin(p.x * 2.0);
    let tx = fmod(p.x, 0.17) - 0.085;
    let tid = floor(p.x / 0.17);
    let th = 0.06 + 0.07 * hash11(tid);
    let tree = sdTriangle(vec2f(tx, p.y), vec2f(-0.035, h2), vec2f(0.035, h2), vec2f(0.0, h2 - th));
    let hx = aspect * 0.7;
    let house = min(sdBox(p - vec2f(hx, h2 - 0.04), vec2f(0.07, 0.04)), sdTriangle(p, vec2f(hx - 0.09, h2 - 0.08), vec2f(hx + 0.09, h2 - 0.08), vec2f(hx, h2 - 0.15)));
    let sil = min(min(h2 - p.y, select(1.0, tree, hash11(tid * 3.0) > 0.35)), house);
    c = mix(c, vec3f(0.004, 0.004, 0.008), aa(sil));
    let win = sdBox(p - vec2f(hx - 0.03, h2 - 0.045), vec2f(0.012, 0.012));
    c = mix(c, vec3f(1.6, 0.9, 0.35), aa(win));
    // rim light on the silhouettes during a flash
    c += vec3f(0.4, 0.45, 0.7) * fl * exp(-abs(sil) * res.y * 0.4) * step(0.0, sil) * 0.6;
  } else if (u.mode == 3u) {
    // tesla lab: tiled wall, floor, purple light pool from the coil
    var q = p * vec2f(10.0, 10.0);
    let f = fract(q);
    let tile = smoothstep(0.0, 0.04, min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)));
    c = vec3f(0.035, 0.04, 0.05) * (0.6 + 0.4 * tile) * (0.85 + 0.3 * hash21(floor(q)));
    if (p.y > gy) { c = vec3f(0.03, 0.03, 0.035) * (1.0 - (p.y - gy) * 1.2); }
    let d = length((p - lp) * vec2f(0.7, 1.0));
    c += vec3f(0.55, 0.4, 1.0) * u.glowAmt * (exp(-d * 4.5) * 0.35 + 0.015) * (1.0 + fl * 0.6);
  } else {
    // generator: dark blue-black with a dot grid
    c = mix(vec3f(0.006, 0.007, 0.018), vec3f(0.014, 0.012, 0.03), uv.y);
    let g = fract(px / 28.0) - 0.5;
    c += vec3f(0.025, 0.03, 0.06) * smoothstep(0.08, 0.0, length(g));
    c += vec3f(0.4, 0.45, 0.8) * fl * 0.06;
  }
  return vec4f(c, 1.0);
}`;

const PERSIST_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  // afterimage: keep the brighter of (old image faded) and (this frame's bolt)
  let prev = TEX(prevT, uv).rgb * u.decay;
  let cur = TEX(boltT, uv).rgb;
  return vec4f(max(prev, cur * 0.55), 1.0);
}`;

const COMBINE_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let b = TEX(boltT, uv).rgb;
  let a = TEX(persistT, uv).rgb * u.afterTint.rgb;
  return vec4f(b + a, 0.0);
}`;

const COMPOSITE_WGSL = /* wgsl */ `
${TONEMAP_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var c = TEX(hdr, uv).rgb + TEX(bloomTex, uv).rgb * u.bloomAmt;
  c += vec3f(0.6, 0.65, 0.9) * u.flash * 0.08;
  let v = uv - 0.5;
  c *= 1.0 - dot(v, v) * 0.8;
  return vec4f(vfxTonemap(c * u.exposure, px), 1.0);
}`;

const SPRITE_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  let c = mix(pow(t.rgb, vec3f(2.2)) * i.color.rgb, vec3f(4.0, 4.2, 5.0), i.extra.w);
  return vec4f(c * i.color.a, i.color.a);
}`;

export default {
  interaction: 'Move the mouse to aim the bolt. Use the sliders to change how it is built.',
  examples: [
    {
      id: 'generator',
      label: 'Bolt generator',
      kind: 'Abstract',
      note: 'Start with one straight line. Split it at the middle and push the midpoint sideways by a random amount; repeat for every new segment, halving the push each time (<b>midpoint displacement</b>). Occasionally fork a branch. Regenerate it many times a second and it crackles.',
      hint: 'Move the mouse: the bolt ends at the cursor. Try Detail = 1, 2, 3…',
      params: { detail: 6, jag: 0.22, branch: 0.35, branchLen: 0.55, width: 2.2, glow: 16, color: '#a9b8ff', flicker: 18, after: 0.25, bloom: 1 },
    },
    {
      id: 'chain',
      label: 'Chain lightning',
      kind: 'In a game',
      note: 'Classic RPG chain lightning: the bolt hits the enemy nearest to the cursor, then jumps to the nearest enemy it hasn’t hit yet, again and again. Each hop is a fresh bolt with a short delay, a white hit-flash and a spark burst.',
      hint: 'Click to cast chain lightning.',
      params: { detail: 5, jag: 0.2, branch: 0.25, branchLen: 0.45, width: 2.4, glow: 14, color: '#9ed0ff', flicker: 24, after: 0.3, jumps: 5, bloom: 1.1 },
    },
    {
      id: 'storm',
      label: 'Storm',
      kind: 'In a game',
      note: 'Strikes from the clouds light the whole sky for a split second — the hills, trees and house become black silhouettes with a rim of light. Between strikes, distant <b>sheet lightning</b> flickers inside the clouds without a visible bolt. Click to strike.',
      hint: 'Click to call a strike there.',
      params: { detail: 7, jag: 0.18, branch: 0.45, branchLen: 0.5, width: 2.5, glow: 18, color: '#c4ccff', flicker: 12, after: 0.45, bloom: 1.2 },
    },
    {
      id: 'tesla',
      label: 'Tesla coil',
      kind: 'Real life',
      note: 'Arcs jump from the coil’s top to whatever is close: your cursor (if it’s in range), the grounded ball, or just into the air. Short, extremely jagged, purple-white, regenerated about 30 times a second.',
      hint: 'Bring the cursor near the coil — the arcs jump to it.',
      params: { detail: 5, jag: 0.3, branch: 0.3, branchLen: 0.4, width: 1.8, glow: 12, color: '#d8a8ff', flicker: 30, after: 0.15, bloom: 1.3 },
    },
  ],
  controls: [
    { type: 'heading', label: 'Shape' },
    { type: 'slider', key: 'detail', label: 'Detail (generations)', min: 0, max: 9, step: 1, value: 6, help: 'How many times every segment is split. Segments = 2^n.' },
    { type: 'slider', key: 'jag', label: 'Jaggedness', min: 0, max: 0.6, step: 0.005, value: 0.22, help: 'First sideways push as a fraction of the length; halves each generation.' },
    { type: 'slider', key: 'branch', label: 'Branching', min: 0, max: 1, step: 0.01, value: 0.35, help: 'Chance that a split also forks a side branch.' },
    { type: 'slider', key: 'branchLen', label: 'Branch length', min: 0.1, max: 1.2, step: 0.01, value: 0.55 },
    { type: 'slider', key: 'jumps', label: 'Chain jumps', min: 1, max: 10, step: 1, value: 5, showFor: ['chain'] },
    { type: 'heading', label: 'Look' },
    { type: 'slider', key: 'width', label: 'Core width (px)', min: 0.5, max: 8, step: 0.1, value: 2.2 },
    { type: 'slider', key: 'glow', label: 'Glow radius (px)', min: 2, max: 50, step: 0.5, value: 16 },
    { type: 'color', key: 'color', label: 'Color', value: '#a9b8ff' },
    { type: 'slider', key: 'flicker', label: 'Flicker (regenerations/s)', min: 0, max: 60, step: 1, value: 18, help: 'Re-randomise the bolt this often. 0 = frozen shape.' },
    { type: 'slider', key: 'after', label: 'Afterimage (s)', min: 0, max: 1.5, step: 0.01, value: 0.25, help: 'A feedback buffer that fades old bolts slowly — your retina does the same.' },
    { type: 'slider', key: 'bloom', label: 'Bloom', min: 0, max: 3, step: 0.01, value: 1 },
  ],
  about: {
    summary: 'Lightning is a fractal line: split a segment, nudge the middle sideways, repeat, and fork now and then. Add a glow, regenerate it constantly and let it leave an afterimage.',
    what: `<p>All bolts are generated on the CPU (a few hundred line segments) and drawn as glowing capsules on the GPU. The scenes show the raw generator,
      a chain-lightning spell, a storm with flashes that light up the world, and a crackling tesla coil.</p>`,
    how: `<ol>
      <li><b>Midpoint displacement</b>: segment A→B becomes A→M→B, where M is the midpoint pushed along the perpendicular by a random amount ≤ <i>offset</i>. Every generation splits all segments and halves the offset: big kinks first, fine crackle last (a 1D fractal, like terrain).</li>
      <li><b>Branches</b>: at a split, with some probability add a new segment from M, roughly continuing the first half’s direction, rotated a little and shorter. Branches keep being subdivided, so they’re jagged too. Thinner and dimmer per branch level.</li>
      <li><b>Glow</b>: each segment is a quad around a capsule; the fragment shader computes the distance to the segment and outputs a sharp core + a Gaussian glow, in HDR (values far above 1).</li>
      <li><b>MAX blending</b>: neighbouring segments overlap at their joints. With additive blending the overlaps would be twice as bright (“beads”); with <code>max</code> they merge seamlessly.</li>
      <li><b>Flicker & afterimage</b>: regenerate the bolt N times per second and jitter its brightness. A feedback buffer keeps <code>max(previous × decay, current)</code>, so old shapes linger as a fading purple ghost.</li>
      <li><b>Flash</b>: the scene shaders get a flash intensity and the strike position: the sky lights up behind silhouettes, the ground near the strike gets light, and bloom does the rest.</li>
    </ol>`,
    uses: [
      { title: 'Spells & abilities', text: 'Chain lightning, lightning arrows, tesla towers in tower-defense games, electric melee hits.' },
      { title: 'Atmosphere', text: 'Storms that reveal the level for a split second (Limbo, Inside, Ori); horror reveals.' },
      { title: 'Sci-fi', text: 'Electric fences, overloaded machinery, plasma arcs, shields being hit.' },
      { title: 'UI', text: 'Crackling borders for “super” states, critical-hit feedback.' },
    ],
    try: [
      'In <b>Bolt generator</b>, set <i>Detail</i> to 0, then step it up one at a time: watch each generation split the segments.',
      'Set <i>Flicker</i> to 0 to freeze one shape; raise <i>Branching</i> to 1 for a tree-like discharge.',
      'Push <i>Afterimage</i> to 1.5 s and move the mouse around.',
      'In <b>Chain lightning</b>, set <i>Chain jumps</i> to 10 and click into the middle of the pack.',
      'In <b>Storm</b>, click repeatedly in different places — the light comes from where the bolt hits.',
    ],
    ask: [
      'procedural lightning bolts with midpoint displacement and branches',
      'chain lightning that jumps to the nearest unhit enemy',
      'lightning flash that lights the scene and reveals silhouettes',
      'glowing lines rendered with max blending to avoid bright joints',
      'afterimage / persistence using a feedback buffer',
      'tesla coil arcs that seek the nearest target',
    ],
    perf: `<p>A bolt is a few hundred segments: generating it in JavaScript is microseconds, and drawing is one instanced call. The glow quads can be large,
      so fill rate (glow radius × total length) is what costs. The afterimage costs two full-screen passes.</p>`,
    api: `<p>Everything here (instanced capsules, max blending, a feedback texture) also exists in WebGL2; this scene is WebGPU-only because it shares the
      WebGPU HDR/bloom pipeline of the other VFX scenes. With WebGPU you could move the generator into a compute shader to make thousands of bolts.</p>`,
    code: [
      {
        title: 'Midpoint displacement with branches (JavaScript)',
        lang: 'js',
        src: `let segs = [{ x1: ax, y1: ay, x2: bx, y2: by, lvl: 0 }];
let off = length * jaggedness;
for (let g = 0; g < detail; g++) {
  const next = [];
  for (const s of segs) {
    const [nx, ny] = perpendicular(s);
    const o = (rnd() - 0.5) * 2 * off;               // push the midpoint sideways
    const mx = (s.x1 + s.x2) / 2 + nx * o, my = (s.y1 + s.y2) / 2 + ny * o;
    next.push({ ...s, x2: mx, y2: my }, { ...s, x1: mx, y1: my });
    if (rnd() < branch / (1 + s.lvl)) {              // fork
      const ang = Math.atan2(my - s.y1, mx - s.x1) + (rnd() - 0.5) * 1.3;
      const bl = segLength * branchLen;
      next.push({ x1: mx, y1: my, x2: mx + Math.cos(ang) * bl, y2: my + Math.sin(ang) * bl, lvl: s.lvl + 1 });
    }
  }
  segs = next;
  off *= 0.5;                                         // finer detail, smaller kinks
}`,
      },
      {
        title: 'Glowing capsule fragment shader (drawn with MAX blending)',
        lang: 'wgsl',
        src: `let pa = i.w - i.seg.xy;
let ba = i.seg.zw - i.seg.xy;
let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
let d = length(pa - ba * h);                       // distance to the segment
let core = clamp((coreWidth * 0.5 - d) / fwidth(d) + 0.5, 0.0, 1.0);
let g = d / glowRadius;
let glow = exp(-g * g);
return vec4f(color * (core * coreI + glow * glowI), 0.0); // HDR, blend: max`,
      },
      {
        title: 'Afterimage: a feedback buffer',
        lang: 'wgsl',
        src: `let prev = TEX(prevT, uv).rgb * u.decay;   // last frame's afterimage, faded
let cur = TEX(boltT, uv).rgb;             // this frame's bolts
return vec4f(max(prev, cur * 0.55), 1.0); // ping-pong: becomes prevT next frame`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasView = gpu.textureFromImage(atlas.canvas, { label: 'atlas' }).createView();
    const strip = makeStrip(gpu).createView();
    const lines = new GlowLines(gpu, { capacity: 8192 });
    const sparks = softSprites(gpu, strip);
    const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest', fragment: SPRITE_FS });
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    const U = gpu.uniforms(
      { resolution: 'vec2f', time: 'f32', mode: 'u32', groundY: 'f32', flash: 'f32', glowAmt: 'f32', decay: 'f32', bloomAmt: 'f32', exposure: 'f32', lightPos: 'vec2f', sheet: 'vec4f', afterTint: 'vec4f' },
      'BoltU',
    );
    const bg = gpu.fullscreen({ label: 'bolt-bg', uniforms: U, include: ['noise', 'sdf'], code: BG_WGSL, format: 'rgba16float' });
    const persistFx = gpu.fullscreen({ label: 'bolt-persist', uniforms: U, textures: ['prevT', 'boltT'], code: PERSIST_WGSL, format: 'rgba16float' });
    const combine = gpu.fullscreen({ label: 'bolt-combine', uniforms: U, textures: ['boltT', 'persistT'], code: COMBINE_WGSL, format: 'rgba16float' });
    const composite = gpu.fullscreen({ label: 'bolt-composite', uniforms: U, textures: ['hdr', 'bloomTex'], include: ['color', 'hash'], code: COMPOSITE_WGSL });
    const bloom = createBloom(gpu);
    let hdr = null;
    let boltBuf = null;
    let persist = null;
    const ensure = (w, h) => {
      if (hdr && hdr.width === w && hdr.height === h) return;
      hdr?.destroy();
      boltBuf?.destroy();
      persist?.destroy();
      hdr = gpu.target(w, h, { format: 'rgba16float', label: 'bolt-hdr' });
      boltBuf = gpu.target(w, h, { format: 'rgba16float', label: 'bolt-buf' });
      persist = gpu.pingPong(w, h, { format: 'rgba16float', label: 'bolt-persist' });
      const e = gpu.device.createCommandEncoder();
      gpu.clear(e, persist.a, [0, 0, 0, 1]);
      gpu.clear(e, persist.b, [0, 0, 0, 1]);
      gpu.queue.submit([e.finish()]);
    };
    const readout = tag(ctx);
    const rand = prng(5);
    let simT = 0;
    let lastClick = -10;
    let flash = 0;
    let lightPos = [0, 0];
    let trauma = 0;
    const particles = [];
    const addSpark = (x, y, vx, vy, life, size, col, tex = TEX.spark, add = 1) => {
      if (particles.length < 2000) particles.push({ x, y, vx, vy, life, age: 0, size, col, tex, add });
    };

    // live bolts: { a: [x,y], b: [x,y], born, life, segs, regenAt, power }
    let bolts = [];
    const strike = (a, b, life, power = 1) => {
      const bolt = { a, b, born: simT, life, power, segs: null, regenAt: 0 };
      bolts.push(bolt);
      return bolt;
    };

    // ---- chain lightning state
    const enemies = [];
    const resetEnemies = () => {
      enemies.length = 0;
      const W = ctx.width;
      const H = ctx.height;
      const r = prng(9);
      for (let i = 0; i < 8; i++) {
        const air = i % 3 === 2;
        enemies.push({
          x: W * (0.38 + 0.56 * r()),
          y: air ? H * (0.25 + 0.25 * r()) : H * 0.78,
          air,
          kind: air ? 'bat' : 'slime',
          phase: r() * 6,
          flash: 0,
          kx: 0,
          dead: 0,
        });
      }
    };
    let chain = null; // { hits: [], next: t }
    const castChain = (tx, ty) => {
      const H = ctx.height;
      const src = [ctx.width * 0.14 + H * 0.05, H * 0.78 - H * 0.2];
      let best = null;
      let bd = 1e9;
      for (const e of enemies) {
        const d = Math.hypot(e.x - tx, e.y - ty);
        if (d < bd) {
          bd = d;
          best = e;
        }
      }
      if (!best) return;
      chain = { from: src, target: best, hit: new Set(), left: Math.round(ctx.params.jumps), next: simT };
    };
    const enemyPos = (e) => [e.x + e.kx, e.y - (e.air ? Math.sin(simT * 3 + e.phase) * ctx.height * 0.02 : ctx.height * 0.04)];

    // ---- storm state
    let nextStrike = 0.4;
    let sheet = { x: 0, y: 0, t: -10 };
    let nextSheet = 1;

    const reset = () => {
      bolts = [];
      particles.length = 0;
      simT = 0;
      flash = 0;
      chain = null;
      nextStrike = 0.3;
      nextSheet = 1;
      resetEnemies();
    };
    reset();

    const params = () => {
      const p = ctx.params;
      return { detail: Math.round(p.detail), jag: p.jag, branch: p.branch, branchLen: p.branchLen };
    };

    const sim = (dt) => {
      const p = ctx.params;
      const ptr = ctx.pointer;
      const ex = ctx.example;
      const W = ctx.width;
      const H = ctx.height;
      simT += dt;
      if (ptr.clicked) lastClick = simT;
      if (ex === 'generator') {
        const tgt = ptr.over ? [ptr.x, ptr.y] : [W * 0.5 + Math.sin(simT * 0.7) * W * 0.25, H * 0.85];
        if (!bolts.length) strike([W * 0.5, H * 0.08], tgt, 1e9);
        bolts[0].b = tgt;
        bolts[0].a = [W * 0.5, H * 0.08];
      } else if (ex === 'chain') {
        if (ptr.clicked) castChain(ptr.x, ptr.y);
        if (!chain && simT - lastClick > 1.0 && simT > nextStrike) {
          castChain(W * (0.5 + 0.4 * rand()), H * 0.6);
          nextStrike = simT + 1.6;
        }
        if (chain && simT >= chain.next) {
          const e = chain.target;
          const to = enemyPos(e);
          strike(chain.from, to, 0.35, 1);
          e.flash = 0.12;
          e.kx += (to[0] > chain.from[0] ? 1 : -1) * H * 0.03;
          e.dead = 0.7; // stunned
          flash = Math.max(flash, 0.5);
          lightPos = to;
          trauma = Math.min(1, trauma + 0.12);
          for (let i = 0; i < 18; i++) {
            const a = rand() * Math.PI * 2;
            const v = H * (0.3 + rand() * 0.6);
            addSpark(to[0], to[1], Math.cos(a) * v, Math.sin(a) * v, 0.2 + rand() * 0.3, 5 + rand() * 4, [1.5, 2.2, 4, 1]);
          }
          chain.hit.add(e);
          chain.left--;
          // next hop: nearest living enemy not yet hit
          let best = null;
          let bd = H * 0.75;
          for (const o of enemies) {
            if (chain.hit.has(o)) continue;
            const d = Math.hypot(o.x - e.x, o.y - e.y);
            if (d < bd) {
              bd = d;
              best = o;
            }
          }
          if (chain.left > 0 && best) {
            chain.from = to;
            chain.target = best;
            chain.next = simT + 0.09;
          } else chain = null;
        }
        for (const e of enemies) {
          e.flash = Math.max(0, e.flash - dt);
          e.kx *= Math.exp(-dt * 6);
          if (e.dead > 0) e.dead -= dt;
        }
      } else if (ex === 'storm') {
        const strikeAt = (x) => {
          const top = [x + (rand() - 0.5) * W * 0.2, -H * 0.02];
          const gy = H * 0.8 - H * 0.03 * Math.sin((x / H) * 2);
          strike(top, [x, gy], 0.45, 1.3);
          flash = 1;
          lightPos = [x, H * 0.3];
          trauma = Math.min(1, trauma + 0.3);
        };
        if (ptr.clicked) strikeAt(ptr.x);
        if (simT > nextStrike) {
          strikeAt(W * (0.1 + 0.8 * rand()));
          nextStrike = simT + 1.6 + rand() * 3;
        }
        if (simT > nextSheet) {
          sheet = { x: W * rand(), y: H * (0.1 + 0.25 * rand()), t: simT };
          nextSheet = simT + 0.6 + rand() * 1.5;
        }
      } else if (ex === 'tesla') {
        const top = [W * 0.32, H * 0.36];
        const ball = [W * 0.72, H * 0.62];
        lightPos = top;
        const near = ptr.over && Math.hypot(ptr.x - top[0], ptr.y - top[1]) < H * 0.55;
        // keep ~4 arcs alive
        bolts = bolts.filter((b) => simT - b.born < b.life);
        while (bolts.length < (near ? 5 : 4)) {
          const ang = -Math.PI * (0.05 + 0.9 * rand());
          let end;
          const r = rand();
          if (near && r < 0.7) end = [ptr.x + (rand() - 0.5) * 8, ptr.y + (rand() - 0.5) * 8];
          else if (r < 0.35) end = [ball[0] + (rand() - 0.5) * H * 0.04, ball[1] - H * 0.03];
          else {
            const L = H * (0.12 + 0.25 * rand());
            end = [top[0] + Math.cos(ang) * L * 1.3, top[1] + Math.sin(ang) * L * 0.9 + H * 0.05];
          }
          const a0 = Math.atan2(end[1] - top[1], end[0] - top[0]);
          const start = [top[0] + Math.cos(a0) * H * 0.07, top[1] + Math.sin(a0) * H * 0.025];
          strike(start, end, 0.06 + rand() * 0.18, 0.9);
        }
        flash = 0.25 + 0.15 * Math.random();
      }
      // regenerate shapes at the flicker rate
      const rate = Math.max(0.0001, ctx.params.flicker);
      for (const b of bolts) {
        if (!b.segs || (ctx.params.flicker > 0 && simT >= b.regenAt)) {
          b.segs = makeBolt(b.a[0], b.a[1], b.b[0], b.b[1], params(), rand);
          b.regenAt = simT + 1 / rate;
        }
      }
      if (ex !== 'tesla') bolts = bolts.filter((b) => simT - b.born < b.life);
      for (let i = particles.length - 1; i >= 0; i--) {
        const q = particles[i];
        q.age += dt;
        if (q.age > q.life) {
          particles[i] = particles[particles.length - 1];
          particles.pop();
          continue;
        }
        q.vy += H * 1.6 * dt;
        q.vx *= Math.exp(-3 * dt);
        q.vy *= Math.exp(-3 * dt);
        q.x += q.vx * dt;
        q.y += q.vy * dt;
      }
      flash *= Math.exp(-dt * (ex === 'storm' ? 6 : 10));
      trauma = Math.max(0, trauma - dt * 1.5);
    };

    return {
      onAction(key) {
        if (key === 'reset') reset();
      },
      onExample() {
        reset();
      },
      onChange() {
        // shape parameters changed: rebuild immediately so frozen bolts update too
        for (const b of bolts) b.segs = null;
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        const W = ctx.width;
        const H = ctx.height;
        ensure(W, H);
        const ex = ctx.example;
        const mode = { generator: 0, chain: 1, storm: 2, tesla: 3 }[ex] ?? 0;
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 0.1);
        if (dt > 0) sim(dt);
        else for (const b of bolts) if (!b.segs) b.segs = makeBolt(b.a[0], b.a[1], b.b[0], b.b[1], params(), rand);
        const shake = trauma * trauma * H * 0.02;
        cam.setViewport(W, H);
        cam.x = W / 2 + Math.sin(ctx.time * 61) * shake;
        cam.y = H / 2 + Math.sin(ctx.time * 47 + 2) * shake;

        const sheetI = ex === 'storm' ? Math.max(0, 1 - (simT - sheet.t) / 0.5) * (0.4 + 0.6 * Math.abs(Math.sin(simT * 40))) * 0.5 : 0;
        U.set('resolution', [W, H])
          .set('time', ctx.time)
          .set('mode', mode)
          .set('groundY', H * 0.78)
          .set('flash', flash)
          .set('glowAmt', 1)
          .set('decay', p.after > 0 ? Math.exp(-(dt || 1 / 60) / Math.max(0.01, p.after)) : 0)
          .set('bloomAmt', p.bloom * 0.4)
          .set('exposure', 1)
          .set('lightPos', lightPos)
          .set('sheet', [sheet.x, sheet.y, sheetI, 0])
          .set('afterTint', [0.75, 0.55, 1.1, 1]);
        bg.draw(enc, hdr, {}, { clear: [0, 0, 0, 1] });

        // props & characters
        shapes.begin();
        sprites.begin();
        sparks.begin();
        if (ex === 'generator') {
          for (const b of bolts) {
            shapes.circle(b.a[0], b.a[1], H * 0.012, [0.6, 0.7, 1.4, 1], { glow: 8, glowStrength: 0.5 });
            shapes.circle(b.b[0], b.b[1], H * 0.012, [0.6, 0.7, 1.4, 1], { stroke: 2, glow: 8, glowStrength: 0.5 });
          }
        } else if (ex === 'chain') {
          const hs = H * 0.18;
          sprites.draw(W * 0.14, H * 0.78, hs, hs, { uv: atlas.uv('hero_idle_' + (Math.floor(ctx.time * 2) % 2)), anchor: [0.5, 1], color: [1, 1, 1, 1] });
          // staff
          shapes.line(W * 0.14 + H * 0.05, H * 0.78, W * 0.14 + H * 0.05, H * 0.78 - H * 0.2, H * 0.012, [0.12, 0.07, 0.04, 1]);
          shapes.circle(W * 0.14 + H * 0.05, H * 0.78 - H * 0.2, H * 0.016, [1.2, 1.6, 3, 1], { glow: H * 0.02, glowStrength: 0.8 });
          for (const e of enemies) {
            const [x, y] = enemyPos(e);
            // stunned enemies twitch and glow blue for a moment
            const st = e.dead > 0 ? e.dead / 0.7 : 0;
            const tw = st > 0 ? (Math.random() - 0.5) * H * 0.008 : 0;
            const frame = e.kind === 'bat' ? 'bat_' + (Math.floor(ctx.time * 8 + e.phase) % 2) : 'slime_' + (Math.floor(ctx.time * 3 + e.phase) % 3);
            const tint = [1 + st * 0.6, 1 + st * 1.2, 1 + st * 3, 1];
            sprites.draw(x + tw, y, H * 0.11, H * 0.11, { uv: atlas.uv(frame), color: tint, user: e.flash > 0 ? 1 : 0 });
          }
        } else if (ex === 'tesla') {
          const tx = W * 0.32;
          const gy = H * 0.78;
          shapes.rect(tx - H * 0.09, gy - H * 0.05, H * 0.18, H * 0.05, [0.05, 0.05, 0.06, 1], { radius: 4 });
          shapes.rect(tx - H * 0.035, H * 0.38, H * 0.07, gy - H * 0.05 - H * 0.38, [0.18, 0.08, 0.03, 1]);
          for (let y = H * 0.4; y < gy - H * 0.06; y += H * 0.012) shapes.line(tx - H * 0.035, y, tx + H * 0.035, y + H * 0.004, H * 0.004, [0.45, 0.22, 0.08, 1]);
          shapes.box(tx, H * 0.36, H * 0.085, H * 0.03, [0.35, 0.36, 0.4, 1], { radius: H * 0.03 });
          shapes.box(tx, H * 0.355, H * 0.07, H * 0.012, [0.7, 0.72, 0.8, 1], { radius: H * 0.012 });
          const bx = W * 0.72;
          shapes.rect(bx - H * 0.008, H * 0.62, H * 0.016, gy - H * 0.62, [0.2, 0.2, 0.22, 1]);
          shapes.circle(bx, H * 0.62, H * 0.035, [0.4, 0.42, 0.48, 1]);
          shapes.circle(bx - H * 0.01, H * 0.61, H * 0.012, [1.2, 1.2, 1.4, 1]);
        }
        for (const q of particles) {
          const sp = Math.hypot(q.vx, q.vy);
          const f = 1 - q.age / q.life;
          sparks.draw(q.x, q.y, q.size + sp * 0.03, q.size * 0.45, { uv: cellUV(q.tex), rotation: Math.atan2(q.vy, q.vx), color: [q.col[0] * f, q.col[1] * f, q.col[2] * f, 1], user: q.add });
        }
        shapes.flush(enc, hdr, cam, {});
        sprites.flush(enc, hdr, cam, {});

        // bolts -> bolt buffer (MAX blend)
        lines.begin();
        const col = hexLin(p.color, 1);
        const hs = H / 700;
        let segCount = 0;
        for (const b of bolts) {
          if (!b.segs) continue;
          const age = simT - b.born;
          const env = b.life > 1e6 ? 1 : Math.exp(-age / (b.life * 0.35));
          const jitter = 0.65 + 0.7 * rand();
          const I = b.power * env * jitter;
          for (const s of b.segs) {
            const k = Math.pow(0.55, s.lvl);
            lines.line(s.x1, s.y1, s.x2, s.y2, p.width * hs * k + 0.6, p.glow * hs * (0.5 + 0.5 * k), [col[0] * 1.2 + 0.3, col[1] * 1.2 + 0.3, col[2] * 1.2 + 0.3], 5 * I * (0.5 + 0.5 * k), 0.9 * I * k, 1);
          }
          segCount += b.segs.length;
        }
        lines.flush(enc, boltBuf, cam, { blend: 'max', clear: [0, 0, 0, 1] });
        // afterimage feedback, then add bolts + afterimage onto the scene
        persistFx.draw(enc, persist.write, { prevT: persist.read, boltT: boltBuf });
        persist.swap();
        combine.draw(enc, hdr, { boltT: boltBuf, persistT: persist.read }, { clear: false, blend: 'add' });
        sparks.flush(enc, hdr, cam, { blend: 'premultiplied' });

        const bl = bloom.render(enc, hdr, { threshold: 1.0, knee: 0.6 });
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { hdr, bloomTex: bl });
        readout.textContent = `${segCount.toLocaleString()} segments`;
      },
    };
  },
};

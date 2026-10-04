// 2D platformer water: a row of springs (1D wave simulation in JavaScript) drives the surface;
// one fullscreen WebGPU pass composites reflection, refraction, tint, caustics and the surface line;
// floating/sinking objects and splash droplets are sprites & shapes on top.

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas, rng } from '../../core/assets.js';
import { createGameScene } from '../../core/gamescene.js';
import { overlayTag } from './_shared.js';

const MAXN = 512;
const OBJECTS = [
  { name: 'crate', size: 34, density: 0.6 },
  { name: 'chest', size: 34, density: 0.85 },
  { name: 'star', size: 28, density: 0.4 },
  { name: 'potion', size: 26, density: 0.7 },
  { name: 'heart', size: 26, density: 0.5 },
  { name: 'mushroom', size: 30, density: 0.55 },
  { name: 'gem', size: 24, density: 2.4 },
  { name: 'bomb', size: 28, density: 1.8 },
];

const COMPOSITE = /* wgsl */ `
fn hAt(i: i32) -> f32 { return hs[clamp(i, 0, i32(u.count) - 1)]; }

// Catmull-Rom through the spring heights -> smooth surface (displacement in px, up = positive)
fn surfH(x: f32) -> f32 {
  let fi = clamp(x / u.res.x, 0.0, 1.0) * (u.count - 1.0);
  let i = i32(floor(fi));
  let t = fract(fi);
  let p0 = hAt(i - 1);
  let p1 = hAt(i);
  let p2 = hAt(i + 1);
  let p3 = hAt(i + 2);
  let t2 = t * t;
  let t3 = t2 * t;
  let h = 0.5 * (2.0 * p1 + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
  // a little ambient chop so the surface is never dead still
  let amb = u.ambient * (sin(x * 0.021 + u.time * 1.7) * 0.6 + sin(x * 0.047 - u.time * 2.3) * 0.4);
  return h + amb;
}

fn objOver(base: vec3f, q: vec2f) -> vec3f {
  let o = TEX(obj, q / u.res);      // premultiplied sprite layer
  return o.rgb + base * (1.0 - o.a);
}

// the world above the water line (game scene + objects)
fn above(q: vec2f) -> vec3f {
  var base = vec3f(0.05, 0.06, 0.1);
  if (u.mode < 0.5) {
    base = mix(vec3f(0.07, 0.08, 0.13), vec3f(0.12, 0.13, 0.2), q.y / u.res.y);
    let g = abs(fract(q / 32.0) - vec2f(0.5));
    base += vec3f(0.035) * smoothstep(0.47, 0.5, max(g.x, g.y));
  } else {
    let gy = mix(u.gameY.x, u.gameY.y, clamp(q.y / u.level, 0.0, 1.0));
    base = TEX(game, vec2f(q.x / u.res.x, gy)).rgb;
  }
  return objOver(base, q);
}

fn caustic(q: vec2f) -> f32 {
  let v1 = voronoi(q / 38.0 + vec2f(u.time * 0.25, u.time * 0.1));
  let v2 = voronoi(q / 27.0 - vec2f(u.time * 0.18, -u.time * 0.21));
  let a = smoothstep(0.35, 0.0, v1.y - v1.x);
  let b = smoothstep(0.35, 0.0, v2.y - v2.x);
  return a * b;
}

// what lies under the water (lake bed or the deep sea), before tint
fn below(q: vec2f) -> vec3f {
  let H = u.res.y;
  var c = vec3f(0.0);
  if (u.mode < 0.5) {
    c = above(q);
    return c;
  }
  let d = (q.y - u.level) / (H - u.level);   // 0 at the surface, 1 at the bottom of the screen
  if (u.mode < 1.5) {
    // lake: soft blue depth, a sandy bed with pebbles and weeds
    c = mix(vec3f(0.16, 0.42, 0.45), vec3f(0.05, 0.16, 0.25), d);
    let bedY = H - H * 0.07 - 10.0 * sin(q.x * 0.01) - 6.0 * sin(q.x * 0.033 + 1.0);
    if (q.y > bedY) {
      let v = voronoi(q / 9.0);
      c = mix(vec3f(0.55, 0.47, 0.33), vec3f(0.42, 0.36, 0.26), smoothstep(0.1, 0.4, v.x));
      c = mix(c, vec3f(0.36, 0.38, 0.4), step(0.8, hash21(v.zw)) * smoothstep(0.35, 0.2, v.x));
    }
    // weeds
    let cell = floor(q.x / 26.0);
    let wx = (cell + 0.5) * 26.0 + (hash11(cell) - 0.5) * 14.0;
    let hgt = H * (0.06 + 0.12 * hash11(cell + 7.0));
    let ty = clamp((bedY - q.y) / hgt, 0.0, 1.0);
    let sway = sin(u.time * 1.3 + cell) * 10.0 * ty * ty;
    let wd = abs(q.x - wx - sway) - 2.5 * (1.0 - ty);
    if (q.y < bedY + 4.0 && q.y > bedY - hgt && hash11(cell + 3.0) > 0.35) {
      c = mix(c, vec3f(0.12, 0.36, 0.18), smoothstep(1.0, 0.0, wd));
    }
  } else {
    // deep sea: depth gradient, distant rock silhouettes, sea floor, kelp
    c = mix(vec3f(0.08, 0.42, 0.58), vec3f(0.01, 0.06, 0.15), pow(d, 0.8));
    let r1 = H * (0.72 - 0.12 * (0.5 + 0.5 * sin(q.x * 0.008 + 1.0)) - 0.05 * valueNoise(vec2f(q.x * 0.02, 1.0)));
    c = mix(c, mix(c, vec3f(0.03, 0.16, 0.26), 0.55), step(r1, q.y));
    let r2 = H * (0.86 - 0.08 * valueNoise(vec2f(q.x * 0.012, 4.0)) - 0.06 * sin(q.x * 0.005));
    c = mix(c, vec3f(0.02, 0.09, 0.15), step(r2, q.y));
    let bedY = H * 0.93 - 8.0 * sin(q.x * 0.012);
    if (q.y > bedY) {
      let ripple = 0.5 + 0.5 * sin(q.x * 0.15 + sin(q.y * 0.3) * 2.0);
      c = mix(vec3f(0.36, 0.33, 0.25), vec3f(0.44, 0.4, 0.3), ripple);
      c *= 0.55;
    }
    // kelp
    let cell = floor(q.x / 34.0);
    let wx = (cell + 0.5) * 34.0 + (hash11(cell) - 0.5) * 18.0;
    let hgt = H * (0.15 + 0.3 * hash11(cell + 7.0));
    let ty = clamp((bedY - q.y) / hgt, 0.0, 1.0);
    let sway = sin(u.time * 0.9 + cell + q.y * 0.01) * 16.0 * ty;
    let wd = abs(q.x - wx - sway) - 3.0 * (1.0 - ty * 0.7);
    if (q.y < bedY + 4.0 && q.y > bedY - hgt && hash11(cell + 3.0) > 0.4) {
      c = mix(c, vec3f(0.08, 0.3, 0.16) * (0.7 + 0.3 * ty), smoothstep(1.0, 0.0, wd));
    }
    // a small school of fish
    for (var i = 0; i < 7; i++) {
      let fi = f32(i);
      let sp = 40.0 + 30.0 * hash11(fi);
      let fx = fmod(u.time * sp + hash11(fi + 2.0) * u.res.x * 2.0, u.res.x + 120.0) - 60.0;
      let fy = H * (0.35 + 0.35 * hash11(fi + 5.0)) + sin(u.time * 1.5 + fi) * 8.0;
      let fq = q - vec2f(fx, fy);
      let body = length(fq / vec2f(14.0, 6.0)) - 1.0;
      let tail = sdTriangle(fq, vec2f(-12.0, 0.0), vec2f(-22.0, -7.0 * (1.0 + 0.2 * sin(u.time * 9.0 + fi))), vec2f(-22.0, 7.0));
      let m = min(body * 6.0, tail);
      c = mix(c, vec3f(0.04, 0.14, 0.22) + vec3f(0.08, 0.06, 0.02) * step(fq.y, -1.0), smoothstep(1.0, -0.5, m));
    }
  }
  return objOver(c, q);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let q = px;
  let H = u.res.y;
  let sy = u.level - surfH(q.x);
  let slope = (surfH(q.x + 2.0) - surfH(q.x - 2.0)) * 0.25;
  let depth = q.y - sy;
  if (depth < -1.0) {
    return vec4f(above(q), 1.0);
  }
  let dpos = max(depth, 0.0);
  // refraction: offset the lookup by the surface slope plus a slow wobble
  let wob = vec2f(sin(q.y * 0.045 + u.time * 2.2) + sin(q.y * 0.11 - u.time * 1.3) * 0.5, 0.0) * 2.5 * u.distort;
  let off = vec2f(slope * 40.0 * u.distort, slope * 10.0) * exp(-dpos / (H * 0.25)) + wob;
  var refr = below(q + off);
  // absorption & tint: deeper = bluer and darker
  let deep = clamp(dpos / (H * 0.45), 0.0, 1.0);
  var tint = mix(vec3f(0.18, 0.62, 0.72), vec3f(0.03, 0.18, 0.36), deep);
  if (u.mode > 1.5) { tint = mix(vec3f(0.2, 0.6, 0.75), vec3f(0.02, 0.1, 0.25), deep); }
  var col = mix(refr * mix(vec3f(0.85, 1.0, 1.0), vec3f(0.35, 0.6, 0.8), deep), tint, mix(0.25, 0.75, deep) * (1.0 - u.clarity * 0.7));
  // caustics: bright dancing net of light, stronger near the surface and on the bed
  let cs = caustic(q + off * 2.0) * u.caustics * (1.0 - deep * 0.6);
  col += vec3f(0.55, 0.85, 0.8) * cs * 0.55;
  // light shafts in the deep sea
  if (u.mode > 1.5) {
    let sx = q.x + q.y * 0.35 - u.time * 12.0;
    let shaft = pow(valueNoise(vec2f(sx * 0.012, u.time * 0.15)), 3.0) * exp(-dpos / (H * 0.45));
    col += vec3f(0.45, 0.75, 0.8) * shaft * 0.6;
  }
  // reflection: mirror the world above around the (local) surface line, fading with depth
  if (u.mode > 0.5 && u.mode < 1.5) {
    let rq = vec2f(q.x + off.x * 1.5, 2.0 * sy - q.y + off.y * 3.0);
    let refl = above(rq);
    let fres = u.reflect * exp(-dpos / (H * 0.18));
    col = mix(col, refl * vec3f(0.85, 0.95, 1.0), fres * 0.8);
  }
  // underside of the surface seen from below: total internal reflection band
  if (u.mode > 1.5) {
    let tir = exp(-dpos / 18.0);
    let mq = vec2f(q.x + off.x, sy + dpos * 2.0 + 30.0);
    col = mix(col, below(mq) * 1.2 + vec3f(0.1, 0.2, 0.22), tir * 0.5);
  }
  // surface line + foam on steep waves
  let line = 1.0 - smoothstep(0.0, 2.2, abs(depth));
  let foam = smoothstep(0.35, 1.0, abs(slope)) * exp(-dpos / 6.0);
  col = mix(col, vec3f(0.85, 0.97, 1.0), clamp(line * 0.85 + foam * 0.6, 0.0, 1.0));
  col += vec3f(0.2, 0.35, 0.4) * exp(-dpos / 5.0) * 0.6;
  // anti-aliased edge against the air
  let a = smoothstep(-1.0, 1.0, depth);
  col = mix(above(q), col, a);
  return vec4f(col, 1.0);
}
`;

export default {
  interaction: 'Click to drop things into the water. Hold and drag to stir.',
  examples: [
    {
      id: 'spring',
      label: 'Spring surface',
      kind: 'Abstract',
      note: 'The surface is a row of vertical <b>springs</b> (dots). Each is pulled back to rest (<i>tension</i>), loses energy (<i>damping</i>), and tugs on its neighbours (<i>spread</i>) — that tug is what makes a splash travel outward as waves.',
      params: { columns: 48, springs: true },
    },
    {
      id: 'lake',
      label: 'Lake with reflections',
      kind: 'In a game',
      note: 'The platformer world is rendered to a texture first. Below the surface the shader samples it <b>mirrored</b> (reflection) and <b>offset by the wave slope</b> (refraction), tints by depth and adds moving caustics. Objects float, bob and splash.',
      params: { columns: 180, springs: false },
    },
    {
      id: 'underwater',
      label: 'Underwater view',
      kind: 'In a game',
      note: 'Same surface, seen from below: deep-water gradient, light shafts, caustics on the sea floor, swaying kelp, bubbles, and the mirror-like underside of the surface (total internal reflection). Heavy objects sink, light ones bob back up.',
      params: { columns: 180, springs: false },
    },
  ],
  controls: [
    { type: 'heading', label: 'Wave simulation' },
    { type: 'slider', key: 'columns', label: 'Springs (columns)', min: 16, max: MAXN, step: 1, value: 180, help: 'More springs = finer waves. The sim is tiny either way (it runs in JavaScript).' },
    { type: 'slider', key: 'tension', label: 'Tension', min: 0.002, max: 0.1, step: 0.001, value: 0.025, help: 'Spring stiffness: higher = faster, choppier bobbing.' },
    { type: 'slider', key: 'damping', label: 'Damping', min: 0, max: 0.1, step: 0.001, value: 0.02, help: 'How fast waves die out. 0 = they never stop.' },
    { type: 'slider', key: 'spread', label: 'Spread', min: 0, max: 0.48, step: 0.01, value: 0.25, help: 'How strongly each spring pulls its neighbours: how far ripples travel.' },
    { type: 'slider', key: 'splash', label: 'Splash strength', min: 0.2, max: 3, step: 0.01, value: 1 },
    { type: 'heading', label: 'Rendering' },
    { type: 'slider', key: 'reflect', label: 'Reflection', min: 0, max: 1, step: 0.01, value: 0.75, showFor: ['lake'] },
    { type: 'slider', key: 'distort', label: 'Refraction wobble', min: 0, max: 3, step: 0.01, value: 1, showFor: ['lake', 'underwater'] },
    { type: 'slider', key: 'clarity', label: 'Water clarity', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['lake', 'underwater'] },
    { type: 'slider', key: 'caustics', label: 'Caustics', min: 0, max: 2, step: 0.01, value: 1, showFor: ['lake', 'underwater'] },
    { type: 'toggle', key: 'rain', label: 'Rain', value: false },
    { type: 'toggle', key: 'springs', label: 'Show the springs', value: true, showFor: ['spring'] },
    { type: 'button', key: 'reset', label: 'Calm the water' },
  ],
  about: {
    summary: 'Platformer water is usually a row of springs: a cheap 1D simulation that makes splashes ripple outward. A shader then fakes reflection, refraction and caustics from a texture of the world.',
    what: `<p>A water surface that reacts to everything that falls in. Behind it, the scene is reflected and refracted, tinted by depth, with
      dancing caustic light. Objects float or sink depending on their density.</p>`,
    how: `<ol>
      <li><b>Surface</b>: N columns, each a spring with a height and a speed. Every step: <code>speed += −tension·height − damping·speed</code>.</li>
      <li><b>Spreading</b>: a few times per step, each column pushes on its neighbours by <code>spread · (myHeight − theirHeight)</code>. This is a discrete wave equation — splashes become travelling ripples.</li>
      <li><b>Splash</b>: an object hitting the water sets the speed of nearby columns and spawns droplet particles; droplets landing make tiny splashes of their own.</li>
      <li><b>Buoyancy</b>: the submerged fraction of an object pushes it up (∝ 1/density); water drag slows it. Floating objects ride the waves.</li>
      <li><b>Rendering</b> (one fragment shader): the column heights are uploaded as a storage buffer and interpolated with a Catmull-Rom spline.
        The world is first rendered to a texture; below the surface we sample it <i>mirrored</i> (reflection) and <i>offset by the surface slope</i> (refraction),
        mix in a depth tint, add a Voronoi-based caustic pattern, and draw a bright line and foam at the surface.</li>
    </ol>`,
    uses: [
      { title: 'Platformers', text: 'Rayman Legends, Ori, Celeste-style pools, Terraria’s water surface — springs + reflection is the standard recipe.' },
      { title: 'Atmosphere', text: 'Lakes that mirror the level, underwater sections with light shafts and caustics, rain hitting puddles.' },
      { title: 'Gameplay feedback', text: 'Jumping in, explosions near water, boats and floating crates that react to waves.' },
    ],
    try: [
      'On <b>Spring surface</b>, set <i>Spread</i> to 0: splashes stay put — each spring only bobs on its own.',
      'Set <i>Damping</i> to 0 and click a few times: the waves never settle (and start interfering).',
      'Drop a gem (they sink) and a star (it floats) — density decides.',
      'On <b>Lake</b>, turn <i>Reflection</i> to 1 and <i>Water clarity</i> to 0 for a dark, mirror-like pond.',
      'Turn on <i>Rain</i> and lower <i>Tension</i>: a lazy, choppy surface.',
    ],
    ask: [
      'spring-based 2D water surface with splashes',
      'water reflection and refraction from a render texture',
      'animated caustics on the lake bed',
      'buoyant objects that float and bob on waves',
      'an underwater section with light shafts and bubbles',
    ],
    perf: `<p>The simulation is a few hundred numbers — negligible on the CPU. Rendering is one fullscreen pass (a few texture reads + two Voronoi lookups for caustics) plus the world texture.
      The extra cost of water is basically “render the world once to a texture”.</p>`,
    api: `<p>WebGPU: the heights go into a <b>storage buffer</b> read directly by the fragment shader. In WebGL2 you would upload them as a 1-pixel-tall
      float texture instead — everything else is identical. For thousands of columns (e.g. a 2D ocean) the spring update could move to a compute shader.</p>`,
    code: [
      {
        title: 'The spring row (JavaScript, from this scene)',
        lang: 'js',
        src: `for (let i = 0; i < n; i++) {               // each column is a damped spring
  vel[i] += -tension * h[i] - damping * vel[i];
  h[i] += vel[i];
}
for (let pass = 0; pass < 8; pass++) {        // neighbours pull on each other -> waves
  for (let i = 0; i < n; i++) {
    if (i > 0)     { L[i] = spread * (h[i] - h[i - 1]); vel[i - 1] += L[i]; }
    if (i < n - 1) { R[i] = spread * (h[i] - h[i + 1]); vel[i + 1] += R[i]; }
  }
  for (let i = 0; i < n; i++) {
    if (i > 0) h[i - 1] += L[i];
    if (i < n - 1) h[i + 1] += R[i];
  }
}`,
      },
      {
        title: 'Reflection & refraction in the fragment shader',
        lang: 'wgsl',
        src: `let sy = u.level - surfH(q.x);                 // surface y at this column
let slope = (surfH(q.x + 2.0) - surfH(q.x - 2.0)) * 0.25;
let off = vec2f(slope * 40.0, slope * 10.0) + wobble;  // refraction offset
var col = below(q + off);                         // what is under the water
col = mix(col * absorb, tint, depthAmount);
let refl = above(vec2f(q.x + off.x, 2.0 * sy - q.y)); // mirror around the surface
col = mix(col, refl, reflect * exp(-depth / falloff));
col += caustic(q + off * 2.0) * 0.5;`,
      },
    ],
    links: [
      { title: 'Make a Splash With Dynamic 2D Water Effects (Michael Hoffman)', url: 'https://gamedevelopment.tutsplus.com/make-a-splash-with-dynamic-2d-water-effects--gamedev-236t', note: 'the spring-column tutorial' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasView = gpu.textureFromImage(atlas.canvas, { label: 'atlas' }).createView();
    const cam = new Camera2D();
    const shapes = new ShapeBatch(gpu);
    const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest' });
    const game = createGameScene(gpu);
    const heights = gpu.storage(MAXN * 4, 'water-heights');
    const U = gpu.uniforms(
      { res: 'vec2f', gameY: 'vec2f', time: 'f32', level: 'f32', count: 'f32', mode: 'f32', reflect: 'f32', clarity: 'f32', caustics: 'f32', distort: 'f32', ambient: 'f32' },
      'Water',
    );
    const comp = gpu.fullscreen({
      label: 'water-composite',
      code: COMPOSITE,
      uniforms: U,
      textures: ['game', 'obj'],
      storage: { hs: 'array<f32>' },
      include: ['hash', 'noise', 'sdf'],
    });
    let objTarget = gpu.target(ctx.width, ctx.height, { label: 'objects' });
    const blank = gpu.target(4, 4, { label: 'blank' });

    const h = new Float32Array(MAXN);
    const vel = new Float32Array(MAXN);
    const L = new Float32Array(MAXN);
    const R = new Float32Array(MAXN);
    let n = 0;
    let acc = 0;
    let objs = [];
    let drops = [];
    let rainAcc = 0;
    const rand = rng(42);
    const hud = overlayTag(ctx, 'right:8px;bottom:8px');

    const levelOf = (ex, H) => (ex === 'underwater' ? H * 0.24 : ex === 'lake' ? H * 0.6 : H * 0.5);
    const colX = (i, W) => (i / (n - 1)) * W;
    const surfaceAt = (x, W) => {
      const fi = Math.max(0, Math.min(n - 1, (x / W) * (n - 1)));
      const i = Math.floor(fi);
      const t = fi - i;
      return h[i] * (1 - t) + h[Math.min(n - 1, i + 1)] * t;
    };
    const splashAt = (x, W, strength, radius) => {
      const r = Math.max(1, Math.round((radius / W) * (n - 1)));
      const c = Math.round((x / W) * (n - 1));
      for (let k = -r; k <= r; k++) {
        const i = c + k;
        if (i < 0 || i >= n) continue;
        const fall = Math.cos(((k / (r + 1)) * Math.PI) / 2);
        vel[i] -= strength * fall;
      }
    };
    const spawnDrops = (x, y, count, speed) => {
      for (let k = 0; k < count && drops.length < 600; k++) {
        const a = -Math.PI / 2 + (rand() - 0.5) * 1.6;
        const s = speed * (0.4 + rand() * 0.8);
        drops.push({ x: x + (rand() - 0.5) * 16, y: y - 2, vx: Math.cos(a) * s, vy: Math.sin(a) * s, r: 1 + rand() * 1.4, life: 1 });
      }
    };
    const resize = (N) => {
      const old = n;
      n = Math.max(4, Math.min(MAXN, Math.round(N)));
      if (old !== n) {
        // resample the current surface so changing resolution doesn't erase it
        const oh = h.slice(0, old || 1);
        const ov = vel.slice(0, old || 1);
        for (let i = 0; i < n; i++) {
          const j = old > 1 ? Math.round((i / (n - 1)) * (old - 1)) : 0;
          h[i] = old ? oh[j] : 0;
          vel[i] = old ? ov[j] : 0;
        }
      }
    };
    const dropObject = (x, y, kind) => {
      const o = OBJECTS[kind ?? Math.floor(rand() * OBJECTS.length)];
      objs.push({ ...o, x, y, vx: (rand() - 0.5) * 60, vy: 0, a: (rand() - 0.5) * 0.6, va: (rand() - 0.5) * 2, wet: false, bubble: 0 });
      if (objs.length > 24) objs.shift();
    };
    const seed = (ex, W, H) => {
      objs = [];
      drops = [];
      h.fill(0);
      vel.fill(0);
      const lv = levelOf(ex, H);
      // things already falling so the very first frames show splashes
      const xs = [0.22, 0.5, 0.78, 0.36, 0.64];
      xs.forEach((fx, k) => dropObject(W * fx, lv - 60 - k * 70, k % OBJECTS.length));
      if (ex === 'underwater') {
        dropObject(W * 0.15, lv + H * 0.3, 6);
        dropObject(W * 0.85, lv + H * 0.2, 2);
      }
      // a pre-existing ripple
      splashAt(W * 0.5, W, 6, 50);
    };
    resize(ctx.params.columns);
    seed(ctx.example, ctx.width, ctx.height);

    const simStep = (p) => {
      const k = p.tension;
      const d = p.damping;
      const sp = p.spread;
      for (let i = 0; i < n; i++) {
        vel[i] += -k * h[i] - d * vel[i];
        h[i] += vel[i];
      }
      for (let pass = 0; pass < 8; pass++) {
        for (let i = 0; i < n; i++) {
          if (i > 0) {
            L[i] = sp * (h[i] - h[i - 1]);
            vel[i - 1] += L[i];
          }
          if (i < n - 1) {
            R[i] = sp * (h[i] - h[i + 1]);
            vel[i + 1] += R[i];
          }
        }
        for (let i = 0; i < n; i++) {
          if (i > 0) h[i - 1] += L[i];
          if (i < n - 1) h[i + 1] += R[i];
        }
      }
      // keep it bounded no matter what the sliders say
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(h[i]) || Math.abs(h[i]) > 400) {
          h[i] = Math.sign(h[i]) * 400 || 0;
          vel[i] *= 0.5;
        }
      }
    };

    let lastEx = ctx.example;
    let dragPrev = null;
    return {
      resize(w, hh) {
        objTarget.destroy();
        objTarget = gpu.target(w, hh, { label: 'objects' });
      },
      onChange(key, v) {
        if (key === 'columns') resize(v);
      },
      onExample(id, c) {
        if (id !== lastEx) seed(id, c.width, c.height);
        lastEx = id;
      },
      onAction(key, c) {
        if (key === 'reset') seed(c.example, c.width, c.height);
      },
      frame(ctx) {
        const p = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const ex = ctx.example;
        const mode = ex === 'spring' ? 0 : ex === 'lake' ? 1 : 2;
        const lv = levelOf(ex, H);
        const dt = Math.min(ctx.dt, 1 / 20);
        cam.setViewport(W, H);
        const ptr = ctx.pointer;

        if (!ctx.paused) {
          // input: click drops an object, dragging across the surface stirs it
          if (ptr.clicked && ptr.button === 0) {
            if (ptr.y < lv - 10) dropObject(ptr.x, ptr.y);
            else dropObject(ptr.x, lv - 80);
          }
          if (ptr.down && dragPrev) {
            const sy = lv - surfaceAt(ptr.x, W);
            if (Math.abs(ptr.y - sy) < 40 && Math.abs(ptr.dx) > 0) splashAt(ptr.x, W, Math.min(6, Math.abs(ptr.dx) * 0.15) * p.splash, 24);
          }
          dragPrev = ptr.down ? [ptr.x, ptr.y] : null;
          // rain
          if (p.rain) {
            rainAcc += dt * 40;
            while (rainAcc > 1) {
              rainAcc -= 1;
              const x = rand() * W;
              splashAt(x, W, 0.8 * p.splash, 6);
              if (rand() < 0.3) spawnDrops(x, lv - surfaceAt(x, W), 2, 120);
            }
          }
          // springs at a fixed 60 Hz
          acc += dt;
          let steps = 0;
          while (acc >= 1 / 60 && steps < 4) {
            simStep(p);
            acc -= 1 / 60;
            steps++;
          }
          // objects: gravity, buoyancy, drag, splashes
          const g = 1400;
          for (const o of objs) {
            const half = o.size / 2;
            const sy = lv - surfaceAt(o.x, W);
            const sub = Math.max(0, Math.min(1, (o.y + half - sy) / o.size));
            const enteringSpeed = o.vy;
            o.vy += g * dt;
            o.vy -= ((g * sub) / o.density) * dt;
            if (sub > 0) {
              const drag = Math.min(1, 3.5 * sub * dt);
              o.vy -= o.vy * drag;
              o.vx -= o.vx * drag * 0.6;
              const slope = (surfaceAt(o.x + 6, W) - surfaceAt(o.x - 6, W)) / 12;
              o.vx -= slope * 400 * dt * sub;
              o.va += (-slope * 1.2 - o.a) * dt * 6 * sub;
              o.va *= 1 - Math.min(1, 2 * dt);
            }
            if (!o.wet && sub > 0.05) {
              o.wet = true;
              const s = Math.min(18, (Math.abs(enteringSpeed) / 60) * (o.size / 30) * p.splash);
              splashAt(o.x, W, s, o.size * 0.9);
              spawnDrops(o.x, sy, Math.round(5 + s * 1.6), Math.min(560, Math.abs(enteringSpeed) * 0.45 + 100));
            }
            if (o.wet && sub < 0.02 && o.vy < -40) o.wet = false;
            // floating objects push the water a little as they bob (two-way coupling)
            if (sub > 0 && sub < 1) splashAt(o.x, W, o.vy * dt * 0.02, half);
            o.x += o.vx * dt;
            o.y += o.vy * dt;
            o.a += o.va * dt;
            if (o.x < half) {
              o.x = half;
              o.vx = Math.abs(o.vx) * 0.5;
            }
            if (o.x > W - half) {
              o.x = W - half;
              o.vx = -Math.abs(o.vx) * 0.5;
            }
            const floor = H - half - (mode === 2 ? H * 0.06 : H * 0.07);
            if (o.y > floor) {
              o.y = floor;
              o.vy = 0;
              o.vx *= 0.9;
            }
            if (mode === 2 && sub >= 1 && rand() < dt * 3) o.bubble++;
          }
          // droplets
          for (const d of drops) {
            if (d.bubble) continue;
            d.vy += 900 * dt;
            d.x += d.vx * dt;
            d.y += d.vy * dt;
            d.life -= dt * 0.6;
            const sy = lv - surfaceAt(d.x, W);
            if (d.vy > 0 && d.y > sy) {
              splashAt(d.x, W, 0.25 * d.r * p.splash, 4);
              d.life = 0;
            }
          }
          drops = drops.filter((d) => d.life > 0 && d.y < H + 20);
          // bubbles in the underwater view
          if (mode === 2) {
            if (rand() < dt * 12) drops.push({ bubble: true, x: rand() * W, y: H * 0.95, vx: 0, vy: -40 - rand() * 40, r: 1.5 + rand() * 3, life: 1 });
            for (const o of objs) {
              while (o.bubble > 0) {
                o.bubble--;
                drops.push({ bubble: true, x: o.x + (rand() - 0.5) * o.size * 0.5, y: o.y - o.size * 0.3, vx: 0, vy: -60 - rand() * 50, r: 1 + rand() * 2.5, life: 1 });
              }
            }
          }
          for (const d of drops) {
            if (!d.bubble) continue;
            d.vy = Math.max(d.vy - 30 * dt, -160);
            d.vx = Math.sin(ctx.time * 3 + d.y * 0.05) * 14;
            d.x += d.vx * dt;
            d.y += d.vy * dt;
            const sy = lv - surfaceAt(d.x, W);
            if (d.y < sy + 2) {
              splashAt(d.x, W, -0.15 * d.r, 3);
              d.life = 0;
            }
          }
        }

        // upload the surface
        gpu.queue.writeBuffer(heights, 0, h, 0, n);

        // world texture (game scene above the waterline)
        let gameTex = blank;
        let gameY = [0, 1];
        if (mode === 1) {
          gameTex = game.render(ctx.encoder, ctx.time, W, Math.max(8, Math.round(lv)));
        } else if (mode === 2) {
          gameTex = game.render(ctx.encoder, ctx.time, W, Math.max(8, Math.round(lv / 0.43)));
          gameY = [0.25, 0.68];
        }
        // object layer (premultiplied, transparent background)
        sprites.begin();
        for (const o of objs) sprites.draw(o.x, o.y, o.size, o.size, { uv: atlas.uv(o.name), rotation: o.a });
        sprites.flush(ctx.encoder, objTarget, cam, { clear: [0, 0, 0, 0] });

        U.set('res', [W, H])
          .set('gameY', gameY)
          .set('time', ctx.time)
          .set('level', lv)
          .set('count', n)
          .set('mode', mode)
          .set('reflect', p.reflect)
          .set('clarity', p.clarity)
          .set('caustics', mode === 0 ? 0 : p.caustics)
          .set('distort', mode === 0 ? 0.3 : p.distort)
          .set('ambient', mode === 0 ? 0 : 1.5);
        const canvas = { view: ctx.target, format: gpu.format };
        comp.draw(ctx.encoder, canvas, { game: gameTex, obj: objTarget, hs: heights });

        // springs visualisation, droplets, bubbles
        shapes.begin();
        if (mode === 0 && p.springs) {
          const anchor = lv + H * 0.32;
          const pts = [];
          for (let i = 0; i < n; i++) {
            const x = colX(i, W);
            const top = lv - h[i];
            const stretch = Math.max(-1, Math.min(1, h[i] / 30));
            const col = stretch > 0 ? [1, 0.55 - stretch * 0.3, 0.35, 0.9] : [0.4 + stretch * 0.1, 0.75, 1, 0.9];
            const coils = 5;
            const amp = Math.min(5, (W / n) * 0.25);
            let px = x;
            let py = anchor;
            for (let c = 1; c <= coils * 2; c++) {
              const t = c / (coils * 2);
              const yy = anchor + (top + 6 - anchor) * t;
              const xx = x + (c === coils * 2 ? 0 : (c % 2 ? -amp : amp));
              shapes.line(px, py, xx, yy, 1.1, [col[0], col[1], col[2], 0.4]);
              px = xx;
              py = yy;
            }
            pts.push([x, top]);
            shapes.rect(x - 4, anchor, 8, 3, [0.5, 0.55, 0.7, 0.8]);
          }
          shapes.polyline(pts, 1.5, [0.9, 0.95, 1, 0.5]);
          for (let i = 0; i < n; i++) {
            const stretch = Math.max(-1, Math.min(1, h[i] / 30));
            shapes.circle(pts[i][0], pts[i][1], Math.max(2, Math.min(5, (W / n) * 0.3)), stretch > 0 ? [1, 0.6 - stretch * 0.3, 0.4, 1] : [0.55, 0.85, 1, 1], { glow: 4, glowStrength: 0.35 });
          }
          shapes.line(0, lv, W, lv, 1, [1, 1, 1, 0.18]);
        }
        for (const d of drops) {
          if (d.bubble) shapes.circle(d.x, d.y, d.r, [0.75, 0.95, 1, 0.65], { stroke: 1.2 });
          else {
            // droplets are stretched along their velocity
            const sp = Math.hypot(d.vx, d.vy) + 1e-3;
            const len = Math.min(6, sp * 0.012);
            shapes.line(d.x - (d.vx / sp) * len, d.y - (d.vy / sp) * len, d.x, d.y, d.r * 2, [0.82, 0.95, 1, 0.8 * Math.min(1, d.life * 2)]);
          }
        }
        if (p.rain) {
          for (let k = 0; k < 90; k++) {
            const x = ((k * 97.13 + ctx.time * 120) % (W + 40)) - 20;
            const y = (k * 53.7 + ctx.time * 900 * (0.8 + (k % 5) * 0.08)) % (lv + 40);
            shapes.line(x, y - 14, x - 3, y, 1.2, [0.75, 0.85, 1, 0.35]);
          }
        }
        shapes.flush(ctx.encoder, canvas, cam);
        hud.textContent = `${n} springs · ${objs.length} objects · ${drops.length} particles`;
        if (ex !== lastEx) lastEx = ex;
      },
    };
  },
};

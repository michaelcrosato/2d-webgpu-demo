// Worms-style destructible terrain. The level is a MASK TEXTURE (R = solid, G = scorch, B = material,
// A = "was solid originally" so dug-out caves show a back wall). Explosions are stamped into it with a
// render pass whose blend state erases (dst · (1 − srcAlpha)). A CPU copy of the same mask answers
// collision queries for projectiles, debris and characters. One fullscreen shader turns the mask into
// dirt, stone, gold, grass edges, outlines, bevels and scorch marks.

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas, rng } from '../../core/assets.js';
import { overlayTag, fbm2, vnoise, clamp } from './_shared.js';

const MAX_STAMPS = 64;

const STAMP = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) @interpolate(flat) s: vec4f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let s = su.stamps[ii];                       // x, y, radius, scorch radius (mask texels)
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let ext = max(s.z, s.w) + 2.0;
  let q = s.xy + corners[vi] * ext;
  var o: VO;
  o.pos = vec4f(q.x / su.size.x * 2.0 - 1.0, 1.0 - q.y / su.size.y * 2.0, 0.0, 1.0);
  o.q = q;
  o.s = s;
  return o;
}
// pass 1 (blend: max) — scorch ring
@fragment fn fs_burn(i: VO) -> @location(0) vec4f {
  let d = length(i.q - i.s.xy);
  let b = (1.0 - smoothstep(i.s.z, i.s.w, d)) * step(i.s.z - 3.0, d) * step(i.s.z + 0.5, i.s.w);
  return vec4f(0.0, b, 0.0, 0.0);
}
// pass 2 (blend: dst * (1 - srcAlpha) on rgb, alpha untouched) — the hole
@fragment fn fs_erase(i: VO) -> @location(0) vec4f {
  let d = length(i.q - i.s.xy);
  return vec4f(0.0, 0.0, 0.0, 1.0 - smoothstep(i.s.z - 1.0, i.s.z, d));
}
`;

const TERRAIN = /* wgsl */ `
fn M(q: vec2f) -> vec4f { return TEX(mask, q / u.msize); }
fn S(q: vec2f) -> f32 { return smoothstep(0.35, 0.65, M(q).r); }

fn sky(p: vec2f) -> vec3f {
  let H = u.res.y;
  let t = p.y / H;
  var c = mix(u.sky0, u.sky1, clamp(t * 1.3, 0.0, 1.0));
  let x = p.x / H;
  let m1 = H * (0.42 + 0.08 * sin(x * 2.1 + 1.0) + 0.05 * sin(x * 5.3));
  c = mix(c, mix(u.sky1, vec3f(0.4, 0.42, 0.6), 0.5), step(m1, p.y) * 0.8);
  let cl = smoothstep(0.55, 0.8, fbm(vec2f(x * 2.0 + u.time * 0.01, t * 7.0), 4) * 0.5 + 0.5) * smoothstep(0.45, 0.1, t);
  c = mix(c, vec3f(1.0, 0.97, 0.94), cl * 0.75);
  return c;
}

fn dirt(p: vec2f, mat: f32) -> vec3f {
  let n = valueNoise(p * 0.06) * 0.5 + valueNoise(p * 0.21) * 0.3 + hash21(floor(p / 2.0)) * 0.2;
  let band = 0.5 + 0.5 * sin(p.y * 0.045 + valueNoise(p * 0.008) * 6.0);
  var c = mix(vec3f(0.5, 0.31, 0.17), vec3f(0.62, 0.42, 0.24), band * 0.6 + n * 0.4);
  let v = voronoi(p / 9.0);
  c = mix(c, vec3f(0.66, 0.56, 0.44), smoothstep(0.28, 0.18, v.x) * step(0.78, hash21(v.zw)));
  if (mat > 0.3) {
    // stone (with cracks) and gold ore
    let stone = smoothstep(0.4, 0.6, mat);
    var sc = vec3f(0.44, 0.45, 0.5) * (0.8 + 0.3 * n);
    let cr = voronoiBorder(p / 34.0, 1.0, 0.0);
    sc *= 0.75 + 0.25 * smoothstep(0.0, 0.08, cr.x);
    c = mix(c, sc, stone);
    let ore = smoothstep(0.85, 0.95, mat);
    let speck = step(0.6, hash21(floor(p / 3.0))) * (0.7 + 0.3 * sin(u.time * 4.0 + hash21(floor(p / 3.0)) * 20.0));
    c = mix(c, mix(vec3f(0.5, 0.45, 0.4), vec3f(1.0, 0.82, 0.25), speck), ore);
  }
  return c;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let p = px - u.shake;
  let q = p / u.scale;
  let m = M(q);
  let s = smoothstep(0.35, 0.65, m.r);

  if (u.showMask > 0.5 && px.x < u.res.x * 0.5) {
    // the raw data: R = solid (white), G = scorch (red), A = originally solid (blue)
    var c = vec3f(m.r) + vec3f(m.g, 0.0, 0.0) * 0.8 + vec3f(0.0, 0.0, 0.25) * m.a * (1.0 - m.r);
    let grid = abs(fract(q / 8.0) - vec2f(0.5));
    c = mix(c, vec3f(0.3, 0.3, 0.4), step(0.47, max(grid.x, grid.y)) * 0.25 * step(4.0, u.scale * 4.0));
    c = mix(c, vec3f(1.0, 0.85, 0.3), 1.0 - smoothstep(0.5, 1.5, abs(px.x - u.res.x * 0.5)));
    return vec4f(c, 1.0);
  }

  // background: sky, or the darker back wall of tunnels we dug
  var col = sky(p);
  let wasSolid = smoothstep(0.3, 0.7, m.a);
  if (wasSolid > 0.0) {
    let wn = valueNoise(p * 0.05) * 0.6 + valueNoise(p * 0.19) * 0.4;
    var wall = mix(vec3f(0.13, 0.08, 0.05), vec3f(0.22, 0.14, 0.09), wn);
    // contact shadow from the remaining terrain
    let ao = (S(q + vec2f(4.0, 0.0)) + S(q - vec2f(4.0, 0.0)) + S(q + vec2f(0.0, 4.0)) + S(q - vec2f(0.0, 4.0))) * 0.25;
    wall *= 1.0 - 0.45 * ao;
    col = mix(col, wall, wasSolid);
  }
  if (s > 0.001) {
    var c = dirt(p, m.b);
    if (u.edges > 0.5) {
      // edge detection: is there air nearby?
      let e1 = min(min(S(q + vec2f(1.5, 0.0)), S(q - vec2f(1.5, 0.0))), min(S(q + vec2f(0.0, 1.5)), S(q - vec2f(0.0, 1.5))));
      // bevel lighting from the gradient of a blurred mask (light from the top-left)
      let gx = S(q + vec2f(3.0, 0.0)) - S(q - vec2f(3.0, 0.0));
      let gy = S(q + vec2f(0.0, 3.0)) - S(q - vec2f(0.0, 3.0));
      let rim = clamp(-gx * 0.6 - gy * 0.8, -1.0, 1.0);
      c *= 1.0 + rim * 0.35;
      let blur = (S(q + vec2f(6.0, 0.0)) + S(q - vec2f(6.0, 0.0)) + S(q + vec2f(0.0, 6.0)) + S(q - vec2f(0.0, 6.0))) * 0.25;
      c *= 0.7 + 0.3 * blur;
      // grass where the air is right above (and not scorched)
      let qa = q - vec2f(0.0, 2.5 + 2.0 * hash11(floor(p.x / 2.0)));
      let airAbove = (1.0 - S(qa)) * (1.0 - smoothstep(0.3, 0.7, M(qa).a));
      let grass = airAbove * (1.0 - smoothstep(0.15, 0.5, m.g)) * (1.0 - smoothstep(0.4, 0.6, m.b)) * u.grassOn;
      let gcol = mix(vec3f(0.28, 0.62, 0.22), vec3f(0.55, 0.85, 0.35), hash11(floor(p.x / 2.0)) * 0.5 + 0.5 * airAbove);
      c = mix(c, gcol, grass);
      // dark outline right at the edge
      c = mix(c * 0.45, c, smoothstep(0.2, 0.8, e1));
    }
    // scorch marks
    if (u.burnOn > 0.5) {
      let b = m.g;
      let ember = smoothstep(0.85, 1.0, b) * (0.5 + 0.5 * sin(u.time * 3.0 + p.x * 0.2));
      c = mix(c, vec3f(0.11, 0.08, 0.07) + vec3f(0.6, 0.18, 0.02) * ember * u.heat, smoothstep(0.05, 0.9, b) * 0.9);
    }
    col = mix(col, c, s);
  }
  // explosion flashes light the scene
  col += u.flash * vec3f(1.0, 0.75, 0.4);
  return vec4f(col, 1.0);
}
`;

const LOOKS = {
  blast: { sky0: '#4a86d8', sky1: '#cfe6ff' },
  mining: { sky0: '#6aa7e8', sky1: '#f2e3c6' },
  artillery: { sky0: '#2b2350', sky1: '#ef9a6a' },
};

function generate(kind, mw, mh, seed) {
  const solid = new Uint8Array(mw * mh);
  const orig = new Uint8Array(mw * mh);
  const mat = new Uint8Array(mw * mh);
  const sc = 540 / mh; // noise scale relative to a 540-texel-tall world
  for (let x = 0; x < mw; x++) {
    const X = x * sc;
    let surf;
    if (kind === 'mining') surf = mh * (0.24 + 0.03 * (fbm2(X / 120, 3.3, 3, seed) - 0.5));
    else if (kind === 'artillery') {
      const u = x / mw;
      const hillL = 0.42 - 0.12 * Math.exp(-((u - 0.14) ** 2) / 0.012);
      const hillR = 0.44 - 0.16 * Math.exp(-((u - 0.8) ** 2) / 0.02);
      surf = mh * (Math.min(hillL, hillR, 0.62 + 0.05 * Math.sin(u * 9)) + 0.04 * (fbm2(X / 70, 1.7, 3, seed) - 0.5));
    } else surf = mh * (0.5 + 0.13 * (fbm2(X / 160, 0.5, 4, seed) - 0.5) * 2 + 0.05 * Math.sin(X / 70));
    for (let y = 0; y < mh; y++) {
      const i = y * mw + x;
      const Y = y * sc;
      let s = y > surf ? 1 : 0;
      const depth = (y - surf) * sc;
      if (kind === 'blast') {
        // floating islands
        const isl = fbm2(X / 90, Y / 45, 3, seed + 5) - Math.abs(Y / 540 - 0.25) * 2.4 - Math.abs(X / (mw * sc) - 0.5) * 0.4;
        if (isl > 0.4) s = 1;
        // caves
        if (s && depth > 30 && fbm2(X / 45, Y / 35, 3, seed + 9) > 0.66) s = 0;
      }
      if (kind === 'artillery' && s && depth > 40 && fbm2(X / 40, Y / 40, 3, seed + 3) > 0.7) s = 0;
      solid[i] = s;
      orig[i] = s || y > surf ? 1 : 0; // caves count as "underground": they show a back wall
      if (s) {
        let m = 0;
        const st = fbm2(X / 70, Y / 60, 3, seed + 2);
        if (depth > 40 && st > 0.67 - Math.min(0.07, depth / 4000)) m = 160;
        if (kind === 'mining' && depth > 40) {
          const vein = Math.abs(fbm2(X / 90, Y / 40, 3, seed + 7) - 0.5);
          if (vein < 0.007 + depth / 80000) m = 255;
        } else if (depth > 60 && vnoise(X / 9, Y / 9, seed + 1) > 0.9) m = 255;
        mat[i] = m;
      }
    }
  }
  return { solid, orig, mat };
}

export default {
  interaction: 'Click to blast a hole · drag to dig · Artillery: aim with the mouse, hold to charge, release to fire.',
  examples: [
    {
      id: 'blast',
      label: 'Blast holes',
      kind: 'Abstract',
      note: 'The whole level is one texture: a pixel is either ground or air. An explosion just <b>draws a circle of “air”</b> into that texture. Turn on <i>Show the mask</i> to see the raw data next to the final look — the grass, outlines, bevels and scorch marks are all computed from it per pixel.',
      params: { radius: 46, showMask: false },
      hint: 'Click to blast a hole',
    },
    {
      id: 'mining',
      label: 'Mining & digging',
      kind: 'In a game',
      note: 'Drag to dig tunnels (Terraria/Dig-Dug style). The dug-out area keeps a dark back wall because the mask remembers what was <i>originally</i> solid. Gold ore pops out as sparkles.',
      params: { radius: 16, showMask: false },
      hint: 'Drag to dig · idle = autodigger',
    },
    {
      id: 'artillery',
      label: 'Artillery',
      kind: 'In a game',
      note: 'A Worms / Scorched Earth duel: shells fly on ballistic arcs (gravity + wind) and test a <b>CPU copy of the mask</b> every step for collisions. Slimes fall when the ground under them is blown away.',
      params: { radius: 40, showMask: false },
      hint: 'Aim with the mouse · hold to charge · release to fire',
    },
  ],
  controls: [
    { type: 'slider', key: 'radius', label: 'Explosion / dig radius', min: 6, max: 120, step: 1, value: 46, help: 'In screen pixels.' },
    { type: 'slider', key: 'debris', label: 'Debris', min: 0, max: 3, step: 0.01, value: 1, help: 'Number of dirt particles thrown out.' },
    { type: 'slider', key: 'shake', label: 'Screen shake', min: 0, max: 2, step: 0.01, value: 1 },
    { type: 'slider', key: 'wind', label: 'Wind', min: -1, max: 1, step: 0.01, value: 0.2, showFor: ['artillery'] },
    { type: 'toggle', key: 'edges', label: 'Edge shading (grass, outline, bevel)', value: true, help: 'Off = the raw solid/air mask with only a texture.' },
    { type: 'toggle', key: 'burn', label: 'Scorch marks', value: true },
    { type: 'toggle', key: 'showMask', label: 'Show the mask (left half)', value: false, showFor: ['blast', 'mining'] },
    { type: 'toggle', key: 'auto', label: 'Autoplay when idle', value: true },
    { type: 'button', key: 'reset', label: 'New terrain', primary: true },
  ],
  about: {
    summary: 'Destructible 2D worlds (Worms, Noita, Terraria) are easiest when the terrain is a picture: a texture where each pixel is ground or air. Destroying terrain is just drawing into that picture.',
    what: `<p>A landscape stored in a single mask texture. Explosions and digging draw circles of “air” into it; the renderer derives grass, outlines,
      shading and scorch marks from the mask every frame, so new holes instantly look hand-made.</p>`,
    how: `<ol>
      <li><b>The mask</b>: an RGBA8 texture (here at half screen resolution). R = solid, G = scorch, B = material (dirt/stone/gold), A = “was solid at the start”.</li>
      <li><b>Carving</b>: a tiny render pass draws one quad per explosion into the mask. The blend state is <code>dst × (1 − srcAlpha)</code>, so the circle erases — no read-back, no copy. A second blend state (<code>max</code>) adds the scorch ring.</li>
      <li><b>Collisions</b>: the same circles are applied to a CPU-side <code>Uint8Array</code> copy. Projectiles, debris and characters just ask “is pixel (x, y) solid?”.</li>
      <li><b>Looks</b> (one fragment shader): sample the mask with bilinear filtering for smooth edges; sample neighbours for an <b>edge-detected</b> dark outline,
        a <b>bevel</b> from the blurred gradient, <b>grass</b> where air is directly above, and a dark back wall where the original terrain was dug out.</li>
    </ol>
    <p><b>Why a texture?</b> Arbitrary shapes for free, O(1) collision queries, constant cost no matter how shredded the level gets, and the GPU renders it in one pass.
      Polygon terrain needs boolean geometry and re-triangulation after every blast.</p>`,
    uses: [
      { title: 'Artillery games', text: 'Worms, Scorched Earth, Liero, Pocket Tanks — the original bitmap-terrain genre.' },
      { title: 'Digging & sandbox', text: 'Terraria and Dig Dug-style tunnels, mining games, Noita (which goes further and simulates every pixel).' },
      { title: 'Juicy destruction', text: 'Bullet holes in walls, crumbling platforms, bosses that chew up the arena.' },
    ],
    try: [
      'Turn on <i>Show the mask</i> and blast: the left half is exactly what the GPU and the collision code see.',
      'Turn off <i>Edge shading</i> — the holes look flat and cheap. Edge detection is what makes them look painted.',
      'In <b>Mining</b>, dig down through the stone to a gold vein.',
      'In <b>Artillery</b>, set <i>Wind</i> to −1 and lob a shell into the wind.',
      'Crank <i>Explosion radius</i> to 120 and <i>Debris</i> to 3 for maximum chaos.',
    ],
    ask: [
      'Worms-style destructible terrain using a mask texture',
      'carve circular holes with a blend-mode erase pass',
      'CPU mirror of the terrain mask for collisions',
      'grass and outlines from edge detection on the terrain mask',
      'artillery projectiles with gravity and wind that destroy terrain',
    ],
    perf: `<p>Carving costs one small instanced draw per frame (any number of holes). Rendering is one fullscreen pass with ~20 mask lookups per pixel.
      The CPU mirror update is proportional to the hole area only. Memory: 4 bytes per mask texel.</p>`,
    api: `<p>WebGPU: the erase is a blend state on a render pass (a compute shader writing a storage texture works too). WebGL2 can do the same with
      <code>blendFuncSeparate</code> into a framebuffer texture — this technique is API-agnostic; here it is WebGPU only because of the surrounding scene code.</p>`,
    code: [
      {
        title: 'Erasing with a blend state (WebGPU pipeline)',
        lang: 'js',
        src: `const erase = stampProg.renderPipeline({ fs: 'fs_erase', targets: [{
  format: 'rgba8unorm',
  blend: {
    color: { srcFactor: 'zero', dstFactor: 'one-minus-src-alpha', operation: 'add' }, // rgb *= 1 - a
    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },                 // keep "original"
  },
}] });`,
      },
      {
        title: 'Terrain look from the mask (WGSL)',
        lang: 'wgsl',
        src: `let s = smoothstep(0.35, 0.65, M(q).r);              // smooth edge from bilinear filtering
let e1 = min(min(S(q + vec2f(1.5, 0.0)), S(q - vec2f(1.5, 0.0))),
             min(S(q + vec2f(0.0, 1.5)), S(q - vec2f(0.0, 1.5))));
let airAbove = 1.0 - S(q - vec2f(0.0, 3.0));            // grass grows where air is above
c = mix(c, grassColor, airAbove);
c = mix(c * 0.45, c, smoothstep(0.2, 0.8, e1));          // dark outline at the edge`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const cam = new Camera2D();
    const shapes = new ShapeBatch(gpu, { capacity: 8192 });
    const glow = new ShapeBatch(gpu, { capacity: 1024 });
    const sprites = new SpriteBatch(gpu, { texture: gpu.textureFromImage(atlas.canvas).createView(), filter: 'nearest' });
    const SU = gpu.uniforms({ size: 'vec2f', stamps: `array<vec4f, ${MAX_STAMPS}>` }, 'Stamp');
    const stampProg = gpu.program({ label: 'stamp', bindings: { su: { uniform: SU } }, code: STAMP });
    const pBurn = stampProg.renderPipeline({
      fs: 'fs_burn',
      targets: [{ format: 'rgba8unorm', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'max' } } }],
    });
    const pErase = stampProg.renderPipeline({
      fs: 'fs_erase',
      targets: [{ format: 'rgba8unorm', blend: { color: { srcFactor: 'zero', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' } } }],
    });
    const TU = gpu.uniforms(
      { res: 'vec2f', msize: 'vec2f', shake: 'vec2f', scale: 'f32', time: 'f32', showMask: 'f32', edges: 'f32', burnOn: 'f32', grassOn: 'f32', flash: 'f32', heat: 'f32', sky0: 'vec3f', sky1: 'vec3f' },
      'Terrain',
    );
    const terrain = gpu.fullscreen({ label: 'terrain', code: TERRAIN, uniforms: TU, textures: ['mask'], include: ['hash', 'noise'] });
    const hud = overlayTag(ctx, 'right:8px;bottom:8px');
    const R = rng(1234);

    let SCALE = 2;
    let mw = 0;
    let mh = 0;
    let maskT = null;
    let world = null;
    let seed = 1;
    let stamps = [];
    let parts = [];
    let fx = [];
    let shake = 0;
    let flash = 0;
    let idle = 99;
    let autoT = 0;
    let gold = 0;
    let craters = 0;
    let dig = null; // autodigger
    // artillery state
    let shells = [];
    let charge = 0;
    let aim = -0.9;
    let slimes = [];
    let lastEx = null;
    const cannon = { x: 0, y: 0 };

    const solidAt = (x, y) => {
      const ix = Math.floor(x / SCALE);
      const iy = Math.floor(y / SCALE);
      if (ix < 0 || ix >= mw) return false;
      if (iy >= mh) return true;
      if (iy < 0) return false;
      return world.solid[iy * mw + ix] === 1;
    };
    const surfaceY = (x) => {
      for (let y = 0; y < mh * SCALE; y += SCALE) if (solidAt(x, y)) return y;
      return mh * SCALE;
    };
    const build = (ex) => {
      const W = ctx.width;
      const H = ctx.height;
      SCALE = ctx.testMode ? 3 : 2;
      mw = Math.ceil(W / SCALE);
      mh = Math.ceil(H / SCALE);
      world = generate(ex, mw, mh, seed);
      const data = new Uint8Array(mw * mh * 4);
      for (let i = 0; i < mw * mh; i++) {
        data[i * 4] = world.solid[i] * 255;
        data[i * 4 + 1] = 0;
        data[i * 4 + 2] = world.mat[i];
        data[i * 4 + 3] = world.orig[i] * 255;
      }
      maskT?.destroy();
      maskT = gpu.target(mw, mh, { label: 'terrain-mask' });
      gpu.queue.writeTexture({ texture: maskT.texture }, data, { bytesPerRow: mw * 4 }, [mw, mh]);
      parts = [];
      fx = [];
      shells = [];
      stamps = [];
      craters = 0;
      gold = 0;
      dig = { x: W * 0.3, y: H * 0.45, a: 0.4, t: 0 };
      slimes = [];
      if (ex === 'artillery') {
        for (let k = 0; k < 4; k++) slimes.push({ x: W * (0.66 + k * 0.08), y: 0, vy: 0, vx: 0, dead: 0, ph: k });
        for (const s of slimes) s.y = surfaceY(s.x);
      }
    };

    /** Carve a circle (screen px) into both the GPU mask and the CPU mirror. */
    const carve = (x, y, r, burn, debris) => {
      const cx = x / SCALE;
      const cy = y / SCALE;
      const rr = r / SCALE;
      if (stamps.length < MAX_STAMPS) stamps.push([cx, cy, rr, burn ? rr * 1.35 + 2 : 0]);
      let removed = 0;
      const x0 = Math.max(0, Math.floor(cx - rr));
      const x1 = Math.min(mw - 1, Math.ceil(cx + rr));
      const y0 = Math.max(0, Math.floor(cy - rr));
      const y1 = Math.min(mh - 1, Math.ceil(cy + rr));
      const p = ctx.params;
      for (let iy = y0; iy <= y1; iy++) {
        for (let ix = x0; ix <= x1; ix++) {
          const dx = ix + 0.5 - cx;
          const dy = iy + 0.5 - cy;
          if (dx * dx + dy * dy > rr * rr) continue;
          const i = iy * mw + ix;
          if (!world.solid[i]) continue;
          world.solid[i] = 0;
          removed++;
          const m = world.mat[i];
          if (m === 255) {
            gold++;
            if (R() < 0.3) parts.push({ x: ix * SCALE, y: iy * SCALE, vx: (R() - 0.5) * 200, vy: -150 - R() * 250, life: 1.5, col: [1, 0.85, 0.3, 1], size: 2.5, spark: true });
          }
          if (debris > 0 && R() < (0.012 * debris * 4) / Math.max(1, rr / 10)) {
            const a = Math.atan2(dy, dx);
            const sp = (burn ? 300 : 120) + R() * (burn ? 420 : 160);
            const col = m === 160 ? [0.45, 0.46, 0.5, 1] : m === 255 ? [0.9, 0.75, 0.3, 1] : [0.5 + R() * 0.15, 0.32 + R() * 0.1, 0.18, 1];
            parts.push({ x: ix * SCALE, y: iy * SCALE, vx: Math.cos(a) * sp * (burn ? 1 : 0.5), vy: Math.sin(a) * sp - (burn ? 200 : 60), life: 2.5 + R() * 2, col, size: 1.5 + R() * 2.5, rot: R() * 6 });
          }
        }
      }
      if (parts.length > 2500) parts.splice(0, parts.length - 2500);
      return removed;
    };
    const explode = (x, y, r) => {
      const p = ctx.params;
      carve(x, y, r, true, p.debris);
      craters++;
      shake = Math.min(1.5, shake + (r / 60) * p.shake);
      flash = Math.min(0.5, flash + r / 300);
      fx.push({ x, y, r, t: 0 });
      for (let k = 0; k < 10; k++) parts.push({ x: x + (R() - 0.5) * r, y: y + (R() - 0.5) * r, vx: (R() - 0.5) * 60, vy: -40 - R() * 60, life: 1.6, col: [0.3, 0.28, 0.27, 0.5], size: r * (0.25 + R() * 0.3), smoke: true });
      // knock slimes around
      for (const s of slimes) {
        const d = Math.hypot(s.x - x, s.y - 10 - y);
        if (d < r * 1.6 && !s.dead) {
          s.dead = 3.5;
          s.vx = Math.sign(s.x - x || 1) * 260;
          s.vy = -520;
        }
      }
    };

    return {
      resize() {
        build(ctx.example);
      },
      onAction(key) {
        if (key === 'reset') {
          seed++;
          build(ctx.example);
        }
      },
      frame(ctx) {
        const p = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const ex = ctx.example;
        if (ex !== lastEx || !world) {
          lastEx = ex;
          build(ex);
        }
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 30);
        const ptr = ctx.pointer;
        const look = LOOKS[ex];
        const rad = p.radius * (H / 700) * 1.4;
        const g = 900 * (H / 700);
        if (ptr.clicked || ptr.down) idle = 0;
        else idle += dt;
        const auto = p.auto && idle > 3;

        // ---------------------------------------------------------------- input & gameplay
        if (dt > 0) {
          if (ex === 'blast') {
            if (ptr.clicked) explode(ptr.x, ptr.y, rad);
            if (auto) {
              autoT -= dt;
              if (autoT <= 0) {
                autoT = 1.1;
                const x = W * (0.1 + R() * 0.8);
                explode(x, surfaceY(x) + rad * (0.2 + R() * 0.6), rad * (0.7 + R() * 0.6));
              }
            }
          } else if (ex === 'mining') {
            const r = rad;
            if (ptr.down) {
              const steps = Math.max(1, Math.ceil(Math.hypot(ptr.dx, ptr.dy) / (r * 0.4)));
              for (let k = 1; k <= steps; k++) carve(ptr.x - (ptr.dx * (steps - k)) / steps, ptr.y - (ptr.dy * (steps - k)) / steps, r, false, p.debris * 0.6);
            }
            if (auto) {
              // an autodigger wanders underground
              dig.t += dt;
              dig.a += (Math.sin(dig.t * 0.7) * 0.9 + Math.sin(dig.t * 1.9) * 0.5) * dt;
              if (dig.y < H * 0.35) dig.a += dt * 2 * Math.sign(Math.cos(dig.a) || 1);
              if (dig.y > H * 0.92) dig.a -= dt * 2 * Math.sign(Math.cos(dig.a) || 1);
              const sp = 90 * (H / 700);
              dig.x += Math.cos(dig.a) * sp * dt;
              dig.y += Math.sin(dig.a) * sp * dt;
              if (dig.x < r || dig.x > W - r) dig.a = Math.PI - dig.a;
              dig.x = clamp(dig.x, r, W - r);
              dig.y = clamp(dig.y, H * 0.3, H * 0.95);
              carve(dig.x, dig.y, r, false, p.debris * 0.6);
            }
          } else {
            // artillery: cannon on the left hill
            const cx = W * 0.14;
            const cy = surfaceY(cx) - 10 * (H / 700);
            const toM = Math.atan2(ptr.y - cy, ptr.x - cx);
            if (!auto) aim = clamp(toM, -Math.PI + 0.1, -0.05);
            if (ptr.down) charge = Math.min(1, charge + dt * 0.8);
            const fire = (pw) => {
              const sp = (300 + 900 * pw) * (H / 700);
              shells.push({ x: cx + Math.cos(aim) * 30, y: cy + Math.sin(aim) * 30, vx: Math.cos(aim) * sp, vy: Math.sin(aim) * sp, trail: [] });
            };
            if (ptr.released && charge > 0) {
              fire(charge);
              charge = 0;
            }
            if (auto) {
              autoT -= dt;
              const alive = slimes.filter((s) => !s.dead);
              const tgt = alive[Math.floor(ctx.time * 0.3) % Math.max(1, alive.length)] || { x: W * 0.8 };
              // aim roughly: a fixed 45° lob with power from the horizontal distance
              const wantA = -0.85 - 0.1 * Math.sin(ctx.time);
              aim += (wantA - aim) * Math.min(1, dt * 3);
              if (autoT <= 0 && shells.length === 0) {
                autoT = 1.6;
                const dist = (tgt.x - cx) / W;
                fire(clamp(0.28 + dist * 0.72 + (R() - 0.5) * 0.08 - p.wind * 0.1, 0.1, 1));
              }
            }
            for (const s of shells) {
              const sub = 4;
              for (let k = 0; k < sub && !s.done; k++) {
                const h = dt / sub;
                s.vx += p.wind * 220 * (H / 700) * h;
                s.vy += g * h;
                s.x += s.vx * h;
                s.y += s.vy * h;
                if (solidAt(s.x, s.y) || s.y > H + 50 || s.x < -50 || s.x > W + 50) {
                  s.done = true;
                  if (s.y < H + 40 && s.x > -40 && s.x < W + 40) explode(s.x, s.y, rad);
                }
                for (const sl of slimes) if (!sl.dead && Math.hypot(sl.x - s.x, sl.y - 12 - s.y) < 16) {
                  s.done = true;
                  explode(s.x, s.y, rad);
                }
              }
              s.trail.push([s.x, s.y]);
              if (s.trail.length > 40) s.trail.shift();
            }
            shells = shells.filter((s) => !s.done);
            // slimes: gravity + ground collision against the CPU mask
            for (const s of slimes) {
              if (s.dead) {
                s.dead -= dt;
                s.vy += g * dt;
                s.x += s.vx * dt;
                s.y += s.vy * dt;
                if (s.dead <= 0) {
                  s.dead = 0;
                  s.x = W * (0.6 + R() * 0.32);
                  s.y = -20;
                  s.vy = 0;
                }
                continue;
              }
              s.vy += g * dt;
              s.y += s.vy * dt;
              if (solidAt(s.x, s.y)) {
                while (solidAt(s.x, s.y) && s.y > 0) s.y -= 1;
                s.vy = 0;
              }
              if (s.y > H + 40) s.dead = 0.01;
            }
            cannon.x = cx;
            cannon.y = cy;
          }
          // particles
          for (const q of parts) {
            q.life -= dt;
            if (q.smoke) {
              q.x += q.vx * dt;
              q.y += q.vy * dt;
              q.size *= 1 + dt * 0.6;
              continue;
            }
            q.vy += g * dt;
            const nx = q.x + q.vx * dt;
            const ny = q.y + q.vy * dt;
            if (!q.spark && solidAt(nx, ny)) {
              if (solidAt(q.x, ny)) {
                q.vy *= -0.25;
                q.vx *= 0.7;
              } else q.vx *= -0.4;
              if (Math.abs(q.vy) < 30) q.vy = 0;
            } else {
              q.x = nx;
              q.y = ny;
            }
            q.rot = (q.rot || 0) + q.vx * dt * 0.05;
          }
          parts = parts.filter((q) => q.life > 0 && q.y < H + 30);
          for (const f of fx) f.t += dt;
          fx = fx.filter((f) => f.t < 0.6);
          shake = Math.max(0, shake - dt * 2.5);
          flash = Math.max(0, flash - dt * 2);
        }

        // ---------------------------------------------------------------- carve on the GPU
        if (stamps.length) {
          const arr = new Float32Array(MAX_STAMPS * 4);
          stamps.forEach((s, i) => arr.set(s, i * 4));
          SU.set('size', [mw, mh]).set('stamps', arr);
          SU.upload();
          const pass = ctx.encoder.beginRenderPass({ colorAttachments: [{ view: maskT.view, loadOp: 'load', storeOp: 'store' }] });
          pass.setBindGroup(0, stampProg.bind({ su: SU }));
          pass.setPipeline(pBurn);
          pass.draw(6, stamps.length);
          pass.setPipeline(pErase);
          pass.draw(6, stamps.length);
          pass.end();
          stamps = [];
        }

        // ---------------------------------------------------------------- draw
        const sx = (R() - 0.5) * shake * 14 * (H / 700);
        const sy = (R() - 0.5) * shake * 14 * (H / 700);
        TU.set('res', [W, H])
          .set('msize', [mw, mh])
          .set('shake', [sx, sy])
          .set('scale', SCALE)
          .set('time', ctx.time)
          .set('showMask', p.showMask && ex !== 'artillery' ? 1 : 0)
          .set('edges', p.edges ? 1 : 0)
          .set('burnOn', p.burn ? 1 : 0)
          .set('grassOn', 1)
          .set('flash', flash * 0.35)
          .set('heat', 1)
          .set('sky0', look.sky0)
          .set('sky1', look.sky1);
        const canvas = { view: ctx.target, format: gpu.format };
        terrain.draw(ctx.encoder, canvas, { mask: maskT });

        cam.setViewport(W, H);
        cam.x = W / 2 - sx;
        cam.y = H / 2 - sy;
        shapes.begin();
        glow.begin();
        sprites.begin();
        for (const q of parts) {
          const a = Math.min(1, q.life * 1.5);
          if (q.smoke) shapes.circle(q.x, q.y, q.size, [q.col[0], q.col[1], q.col[2], q.col[3] * a * 0.6]);
          else if (q.spark) glow.circle(q.x, q.y, q.size, [1, 0.85, 0.3, a], { glow: 6, glowStrength: 0.8 });
          else shapes.rect(q.x - q.size / 2, q.y - q.size / 2, q.size, q.size, [q.col[0], q.col[1], q.col[2], a], { rotation: q.rot });
        }
        for (const f of fx) {
          const k = f.t / 0.6;
          glow.circle(f.x, f.y, f.r * (0.4 + k * 1.4), [1, 0.6, 0.2, 1 - k], { stroke: 4 * (1 - k) + 1, glow: 10, glowStrength: 0.7 });
          if (k < 0.35) glow.circle(f.x, f.y, f.r * 0.8 * (1 - k * 2), [1, 0.9, 0.6, 1 - k / 0.35], { glow: 20, glowStrength: 0.9 });
        }
        if (ex === 'mining') {
          const dp = idle > 3 && p.auto ? dig : ptr;
          if (ptr.over || (idle > 3 && p.auto)) glow.circle(dp.x, dp.y, rad, [1, 0.8, 0.4, 0.5], { stroke: 2, glow: 8, glowStrength: 0.6 });
        }
        if (ex === 'artillery') {
          const cx = cannon.x;
          const cy = cannon.y;
          const s = H / 700;
          // trajectory preview while aiming
          if (!auto) {
            const sp = (300 + 900 * Math.max(charge, 0.35)) * s;
            let x = cx + Math.cos(aim) * 30;
            let y = cy + Math.sin(aim) * 30;
            let vx = Math.cos(aim) * sp;
            let vy = Math.sin(aim) * sp;
            for (let k = 0; k < 40; k++) {
              for (let j = 0; j < 3; j++) {
                const h = 1 / 60;
                vx += p.wind * 220 * s * h;
                vy += g * h;
                x += vx * h;
                y += vy * h;
              }
              if (solidAt(x, y)) break;
              shapes.circle(x, y, 2.2 * s + 1, [1, 1, 1, 0.6 * (1 - k / 40)]);
            }
          }
          // the cannon
          shapes.line(cx, cy, cx + Math.cos(aim) * 34 * s * 1.5, cy + Math.sin(aim) * 34 * s * 1.5, 10 * s * 1.4, '#333c57');
          shapes.circle(cx, cy, 14 * s * 1.4, '#566c86');
          shapes.circle(cx - 12 * s, cy + 10 * s, 8 * s, '#1a1c2c');
          shapes.circle(cx + 12 * s, cy + 10 * s, 8 * s, '#1a1c2c');
          if (charge > 0) {
            shapes.rect(cx - 30 * s, cy - 50 * s, 60 * s, 8 * s, [0, 0, 0, 0.6], { radius: 3 });
            shapes.rect(cx - 29 * s, cy - 49 * s, 58 * s * charge, 6 * s, [1, 0.4 + 0.5 * (1 - charge), 0.2, 1], { radius: 3 });
          }
          for (const sh of shells) {
            sh.trail.forEach(([x, y], i) => shapes.circle(x, y, 2 + i * 0.05, [1, 0.9, 0.7, (i / sh.trail.length) * 0.5]));
            glow.circle(sh.x, sh.y, 4 * s + 2, '#ffe9a0', { glow: 10, glowStrength: 0.8 });
          }
          for (const sl of slimes) {
            const fr = `slime_${Math.floor(ctx.time * 5 + sl.ph) % 3}`;
            const sz = 40 * s * 1.3;
            sprites.draw(sl.x, sl.y + 2, sz, sz, { uv: atlas.uv(fr), anchor: [0.5, 1], rotation: sl.dead ? ctx.time * 8 : 0 });
          }
          // wind sock
          const wx = W * 0.5;
          shapes.line(wx, 30, wx + p.wind * 60, 30, 4, [1, 1, 1, 0.8]);
          shapes.triangle(wx + p.wind * 60, 24, wx + p.wind * 60, 36, wx + p.wind * 72, 30, [1, 1, 1, 0.8]);
        }
        shapes.flush(ctx.encoder, canvas, cam);
        sprites.flush(ctx.encoder, canvas, cam);
        glow.flush(ctx.encoder, canvas, cam, { blend: 'additive' });
        hud.textContent =
          ex === 'mining'
            ? `mask ${mw}×${mh} · gold ${gold} · ${parts.length} particles`
            : ex === 'artillery'
              ? `wind ${p.wind > 0 ? '→' : '←'} ${Math.abs(p.wind * 10).toFixed(0)} · ${slimes.filter((s) => !s.dead).length} slimes left`
              : `mask ${mw}×${mh} texels · ${craters} craters · ${parts.length} particles`;
      },
    };
  },
};


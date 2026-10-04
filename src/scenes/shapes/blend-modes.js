// Blend Modes & Transparency — a custom WebGPU scene using REAL GPU blend states.
//  * Blend lab: the same three soft circles drawn with 8 different pipelines (blend states),
//    one tile each, clipped with scissor rectangles.
//  * Premultiplied alpha: the dark-fringe artifact of straight alpha + linear filtering.
//  * In a game: multiply shadows, screen-blended light and additive fire over the game scene.

import { Camera2D, SpriteBatch } from '../../core/batch.js';
import { createGameScene } from '../../core/gamescene.js';
import { makeCanvas } from '../../core/assets.js';
import { labels } from './_shared.js';

const LBL = 'background:#000b;font-size:11px;padding:2px 7px;color:#e2e8f0';

// Blend states. The blob shader outputs PREMULTIPLIED color (rgb·a, a), or "over white"
// (rgb·a + 1 − a) for the modes that multiply/min with the destination.
const keepAlpha = { srcFactor: 'zero', dstFactor: 'one', operation: 'add' };
const MODES = [
  { id: 'replace', name: 'replace · <b>blending off</b>', blend: undefined, out: 2 },
  { id: 'alpha', name: 'alpha · <b>src + dst·(1−α)</b>', blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: keepAlpha }, out: 0 },
  { id: 'add', name: 'additive · <b>src + dst</b>', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: keepAlpha }, out: 0 },
  { id: 'multiply', name: 'multiply · <b>src × dst</b>', blend: { color: { srcFactor: 'dst', dstFactor: 'zero', operation: 'add' }, alpha: keepAlpha }, out: 1 },
  { id: 'screen', name: 'screen · <b>src + dst·(1−src)</b>', blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src', operation: 'add' }, alpha: keepAlpha }, out: 0 },
  { id: 'subtract', name: 'subtract · <b>dst − src</b>', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'reverse-subtract' }, alpha: keepAlpha }, out: 0 },
  { id: 'max', name: 'max · <b>lighten</b>', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' }, alpha: keepAlpha }, out: 0 },
  { id: 'min', name: 'min · <b>darken</b>', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'min' }, alpha: keepAlpha }, out: 1 },
];

const BLOB_WGSL = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) shape: vec4f,
  @location(2) @interpolate(flat) color: vec4f,
  @location(3) @interpolate(flat) mode: vec4f,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @location(0) shape: vec4f, @location(1) color: vec4f, @location(2) mode: vec4f) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let r = shape.z + 1.5;
  let w = shape.xy + corners[vi] * r;
  var o: VOut;
  o.pos = vec4f(w / u.resolution * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.local = corners[vi] * r;
  o.shape = shape; o.color = color; o.mode = mode;
  return o;
}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  // soft disc: full inside r·(1 − softness), fading to 0 at r (at least 1px wide = anti-aliasing)
  let r = i.shape.z;
  let d = length(i.local);
  var a = clamp((r - d) / max(r * i.shape.w, 1.0), 0.0, 1.0);
  a = a * a * (3.0 - 2.0 * a) * i.color.a;
  let m = i32(i.mode.x);
  if (m == 2) {                       // blending off: opaque or nothing
    if (a < 0.5 * i.color.a) { discard; }
    return vec4f(i.color.rgb, 1.0);
  }
  if (a < 0.002) { discard; }
  let pm = i.color.rgb * a;           // premultiplied color
  if (m == 1) { return vec4f(pm + vec3f(1.0 - a), a); }   // "over white" for multiply / min
  return vec4f(pm, a);
}`;

const BG_WGSL = /* wgsl */ `
fn cov(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }
fn checker(px: vec2f, s: f32) -> f32 { let c = floor(px / s); return fmod(c.x + c.y, 2.0); }
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let mode = i32(u.mode);
  if (mode == 2) { return vec4f(TEX(game, uv).rgb, 1.0); }
  if (mode == 1) {
    // premultiplied tab: bright sky panels
    let pw = res.x / 3.0;
    let cell = floor(px.x / pw);
    let lx = px.x - cell * pw;
    var c = mix(vec3f(0.42, 0.7, 0.98), vec3f(0.85, 0.94, 1.0), uv.y);
    c = mix(c, c * 0.93, checker(px, 24.0) * 0.5);
    c = mix(c, vec3f(0.05, 0.06, 0.09), cov(min(lx, pw - lx) - 2.0, 1.0));
    return vec4f(c, 1.0);
  }
  // blend lab tiles (4 x 2)
  let ts = res / vec2f(4.0, 2.0);
  let cell = floor(px / ts);
  let lp = px - cell * ts;
  let tuv = lp / ts;
  var c: vec3f;
  let bgk = i32(u.bgKind);
  if (bgk == 0) {
    // dark -> light, so every mode is seen over both
    c = mix(vec3f(0.06, 0.07, 0.12), vec3f(0.93, 0.9, 0.82), smoothstep(0.1, 0.9, tuv.x));
    c = mix(c, c * 0.85 + vec3f(0.04), checker(lp, 14.0) * 0.35);
  } else if (bgk == 1) {
    c = vec3f(0.07, 0.08, 0.12) + vec3f(0.03) * checker(lp, 14.0);
  } else if (bgk == 2) {
    c = vec3f(0.9, 0.88, 0.82) - vec3f(0.06) * checker(lp, 14.0);
  } else {
    c = TEX(game, (tuv - vec2f(0.5)) * vec2f(1.0, 1.0) * 0.98 + vec2f(0.5)).rgb;
  }
  // tile frame
  let e = min(min(lp.x, ts.x - lp.x), min(lp.y, ts.y - lp.y));
  c = mix(c, vec3f(0.02, 0.02, 0.035), cov(e - 2.0, 1.0));
  return vec4f(c, 1.0);
}`;

const LAYER_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let fire = u.fire.xy;
  let flick = 1.0 + 0.08 * sin(u.time * 17.0) * sin(u.time * 11.0);
  let d = length((px - fire) / res.y);
  if (i32(u.mode) == 0) {
    // MULTIPLY layer: dusk darkness everywhere except pools of light (output = factor per channel)
    let tint = vec3f(0.28, 0.32, 0.58);
    var light = exp(-d * 5.5) * 1.2 * flick;
    // the game scene's torches repeat every 0.9 screen heights; follow them roughly
    let tx = fmod(px.x / res.y + u.time * 0.12 + 0.3, 0.9) - 0.45;
    light += exp(-length(vec2f(tx, px.y / res.y - 0.69)) * 9.0) * 0.8;
    light = clamp(light, 0.0, 1.0);
    let k = u.strength * (1.0 - light);
    return vec4f(mix(vec3f(1.0), tint, k), 1.0);
  }
  // SCREEN layer: warm glow around the fire + soft sun shafts
  var c = vec3f(1.0, 0.55, 0.2) * exp(-d * 7.0) * 0.55 * flick;
  let shaft = 0.5 + 0.5 * sin((px.x / res.y + px.y / res.y * 0.6) * 18.0 + u.time * 0.4);
  c += vec3f(1.0, 0.85, 0.55) * pow(shaft, 6.0) * 0.18 * smoothstep(0.85, 0.0, uv.y) * u.strength2;
  return vec4f(c, 1.0);
}`;

function starSprite(size) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const s = size / 2;
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const r = i % 2 ? s * 0.42 : s * 0.94;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    g.lineTo(s + r * Math.cos(a), s + 0.5 + r * Math.sin(a));
  }
  g.closePath();
  g.fillStyle = '#ffd43b';
  g.fill();
  g.beginPath();
  g.arc(s - s * 0.18, s - s * 0.12, s * 0.2, 0, Math.PI * 2);
  g.fillStyle = '#fff7d1';
  g.fill();
  return c;
}

export default {
  examples: [
    {
      id: 'lab',
      label: 'Blend lab',
      kind: 'Abstract',
      hint: '',
      note: 'Three identical soft circles, drawn eight times with eight GPU <b>blend states</b>. Blending is fixed-function hardware: it combines the shader’s output (<i>src</i>) with the pixel already in the framebuffer (<i>dst</i>) using a small formula you configure.',
    },
    {
      id: 'premul',
      label: 'Premultiplied alpha',
      kind: 'Comparison',
      hint: '',
      note: 'The same star texture, magnified. With <b>straight alpha</b> and linear filtering, the GPU averages a yellow texel with an invisible <i>black</i> one → a dark, half-transparent ring. With <b>premultiplied alpha</b> (color already multiplied by alpha), the average is “half of yellow” — exactly right.',
    },
    {
      id: 'game',
      label: 'Fire, shadows & light',
      kind: 'In a game',
      hint: 'Move the mouse: magic sparkles follow it.',
      note: 'Layers on top of the game scene: a <b>multiply</b> layer darkens it into dusk (only darkens, never brightens); a <b>screen</b> layer adds warm light (only brightens, never clips harshly); <b>additive</b> fire and sparkles glow where they overlap. Smoke uses normal alpha — real games mix modes per effect.',
    },
  ],
  controls: [
    { type: 'heading', label: 'Shapes', showFor: ['lab'] },
    { type: 'slider', key: 'opacity', label: 'Opacity (α)', min: 0, max: 1, step: 0.01, value: 0.75, showFor: ['lab'], help: 'The alpha written by the shader.' },
    { type: 'slider', key: 'soft', label: 'Edge softness', min: 0, max: 1, step: 0.01, value: 0.35, showFor: ['lab'], help: '0 = hard disc, 1 = soft particle.' },
    { type: 'select', key: 'bgKind', label: 'Background', value: 0, showFor: ['lab'], options: [{ value: 0, label: 'Dark → light ramp' }, { value: 1, label: 'Dark' }, { value: 2, label: 'Light' }, { value: 3, label: 'Game scene' }] },
    { type: 'toggle', key: 'reverse', label: 'Reverse draw order', value: false, showFor: ['lab'], help: 'Alpha blending depends on order; additive, multiply, screen, max and min do not.' },
    { type: 'toggle', key: 'move', label: 'Animate', value: true, showFor: ['lab', 'premul'] },
    { type: 'heading', label: 'Texture', showFor: ['premul'] },
    { type: 'slider', key: 'zoom', label: 'Magnification', min: 1, max: 14, step: 0.1, value: 8, showFor: ['premul'], format: (v) => `${v.toFixed(1)}×` },
    { type: 'select', key: 'matte', label: 'Color of transparent texels', value: '#000000', showFor: ['premul'], options: [{ value: '#000000', label: 'Black (typical export)' }, { value: '#ffffff', label: 'White' }, { value: '#ff00ff', label: 'Magenta (debug)' }], help: 'Invisible pixels still have an RGB value — and filtering blends it in.' },
    { type: 'heading', label: 'Layers', showFor: ['game'] },
    { type: 'select', key: 'fireBlend', label: 'Fire & sparkles blending', value: 'add', showFor: ['game'], options: [{ value: 'add', label: 'Additive (glows)' }, { value: 'alpha', label: 'Alpha (flat)' }] },
    { type: 'toggle', key: 'shadows', label: 'Multiply: dusk shadows', value: true, showFor: ['game'] },
    { type: 'slider', key: 'strength', label: 'Darkness', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['game'], help: 'How dark the multiply layer gets away from the lights.' },
    { type: 'toggle', key: 'lights', label: 'Screen: warm light', value: true, showFor: ['game'] },
  ],
  about: {
    summary: 'Blending decides how a new pixel combines with what is already on screen. Alpha for normal transparency, additive for light and fire, multiply for shadows, screen for glows — plus premultiplied alpha, the fix for dark fringes.',
    what: `<p><b>Blend lab</b>: eight tiles, eight blend pipelines, same shapes. <b>Premultiplied alpha</b>: the classic dark halo around scaled sprites and how to avoid it.
      <b>Fire, shadows &amp; light</b>: multiply, screen, additive and alpha layered on a game scene.</p>`,
    how: `<ol>
      <li>The fragment shader outputs a color <i>src</i> (with alpha). The <b>blend unit</b> computes <code>result = src × srcFactor ⊕ dst × dstFactor</code>, where ⊕ is add, subtract, reverse-subtract, min or max.</li>
      <li><b>Alpha</b>: <code>src·α + dst·(1−α)</code>. Order matters — draw back to front. <b>Additive</b>: <code>src + dst</code>, order-free, never darkens: perfect for light. <b>Multiply</b>: <code>src × dst</code>, only darkens: shadows, tint. <b>Screen</b>: <code>1 − (1−src)(1−dst)</code>, only brightens but saturates gently.</li>
      <li>In WebGPU each combination is baked into a <b>render pipeline</b> (<code>targets: [{ format, blend: {color, alpha} }]</code>); switching modes = switching pipelines.</li>
      <li><b>Premultiplied alpha</b> stores <code>(r·a, g·a, b·a, a)</code>. Filtering, mipmapping and blending then average correctly, and one blend state (<code>one, one−src-alpha</code>) covers both normal and additive (α = 0 with color) sprites.</li>
      <li>Blending happens on the stored values. Into an <code>…-srgb</code> texture the GPU blends in linear light (physically nicer); into a plain 8-bit target, like here, it blends the gamma-encoded numbers.</li>
    </ol>`,
    uses: [
      { title: 'VFX', text: 'Additive for fire, lasers, magic, muzzle flashes; alpha for smoke and dust; premultiplied lets one particle system do both.' },
      { title: 'Lighting', text: 'Multiply a light map over the scene (darkness with holes) — the classic 2D lighting trick in Terraria-style games; screen/additive for glows.' },
      { title: 'UI & art', text: 'Multiply for ink and shadows, screen for highlights, max/min for masks and “keep brightest” effects.' },
      { title: 'Sprites', text: 'Premultiplied textures prevent dark fringes when sprites are scaled, rotated or mipmapped (Unity, Godot and most engines can import this way).' },
    ],
    try: [
      'In the <b>Blend lab</b>, set <i>Opacity</i> to 1 and <i>Edge softness</i> to 0: replace and alpha now look identical — blending only matters at partial alpha.',
      'Toggle <i>Reverse draw order</i>: only the <b>alpha</b> tile changes.',
      'Switch the background to <b>Light</b>: additive saturates to white, while multiply shows the colors clearly.',
      'On <b>Premultiplied alpha</b>, choose <i>Magenta</i> transparent texels — the fringe turns pink, proving where it comes from.',
      'In the game, set <i>Fire &amp; sparkles blending</i> to Alpha: the fire turns into flat orange stickers.',
    ],
    ask: [
      'additive blending for fire and magic particles',
      'a multiply-blended darkness layer with light holes',
      'premultiplied alpha for all sprite textures',
      'screen blend for glow overlays',
      'sort transparent sprites back to front',
    ],
    perf: `<p>Blending itself is essentially free (dedicated hardware) but every blended pixel reads the framebuffer: many large, overlapping
      transparent layers cost <b>fill-rate / bandwidth</b> (overdraw). Additive and multiply don’t need sorting, which can save CPU time and draw calls.</p>`,
    api: `<p>Both APIs have the same blend factors and operations. WebGL2 sets them as global state (<code>gl.blendFuncSeparate</code>) before each draw;
      WebGPU bakes them into the pipeline — more objects, but no state leaks and no driver re-validation. Note: <code>min</code>/<code>max</code> ignore the factors (they must be <code>one</code>).</p>`,
    code: [
      {
        title: 'Blend states used by the tiles (WebGPU)',
        lang: 'js',
        src: `const alpha    = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha };
const additive = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha };
const multiply = { color: { srcFactor: 'dst', dstFactor: 'zero', operation: 'add' }, alpha };
const screen   = { color: { srcFactor: 'one', dstFactor: 'one-minus-src', operation: 'add' }, alpha };
const subtract = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'reverse-subtract' }, alpha };
const lighten  = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' }, alpha };
// one pipeline per blend state:
pipe = prog.renderPipeline({ targets: [{ format: gpu.format, blend: multiply }], buffers });`,
      },
      {
        title: 'Premultiplied output (fragment shader)',
        lang: 'wgsl',
        src: `let pm = color.rgb * a;                               // premultiply
if (mode == 1) { return vec4f(pm + vec3f(1.0 - a), a); } // "over white" for multiply/min
return vec4f(pm, a);                                    // for alpha/add/screen/max`,
      },
    ],
    links: [
      { title: 'Tom Forsyth — Premultiplied alpha', url: 'https://tomforsyth1000.github.io/blog.wiki.html#%5B%5BPremultiplied%20alpha%5D%5D' },
      { title: 'WebGPU spec — GPUBlendState', url: 'https://www.w3.org/TR/webgpu/#dictdef-gpublendstate' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const UB = gpu.uniforms({ resolution: 'vec2f', mode: 'f32', bgKind: 'f32', time: 'f32', strength: 'f32', strength2: 'f32', fire: 'vec2f' }, 'B');
    const UL1 = gpu.uniforms({ resolution: 'vec2f', mode: 'f32', bgKind: 'f32', time: 'f32', strength: 'f32', strength2: 'f32', fire: 'vec2f' }, 'B');
    const UL2 = gpu.uniforms({ resolution: 'vec2f', mode: 'f32', bgKind: 'f32', time: 'f32', strength: 'f32', strength2: 'f32', fire: 'vec2f' }, 'B');

    const blob = gpu.program({ label: 'blob', bindings: { u: { uniform: UB } }, code: BLOB_WGSL });
    const buffers = [
      {
        arrayStride: 48,
        stepMode: 'instance',
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x4' },
          { shaderLocation: 1, offset: 16, format: 'float32x4' },
          { shaderLocation: 2, offset: 32, format: 'float32x4' },
        ],
      },
    ];
    const pipes = MODES.map((m) => blob.renderPipeline({ targets: [{ format: gpu.format, blend: m.blend }], buffers }));
    const pipeAlpha = pipes[1];
    const pipeAdd = pipes[2];

    const bg = gpu.fullscreen({ label: 'blend-bg', code: BG_WGSL, uniforms: UB, textures: ['game'], include: ['math'] });
    const MULT = { color: { srcFactor: 'dst', dstFactor: 'zero', operation: 'add' }, alpha: keepAlpha };
    const SCREEN = MODES[4].blend;
    const layerMul = gpu.fullscreen({ label: 'layer-multiply', code: LAYER_WGSL, uniforms: UL1, blend: MULT });
    const layerScr = gpu.fullscreen({ label: 'layer-screen', code: LAYER_WGSL, uniforms: UL2, blend: SCREEN });
    const game = createGameScene(gpu);
    const blank = gpu.target(1, 1, { label: 'blank' });

    // instance buffer
    let cap = 4096;
    let inst = new Float32Array(cap * 12);
    let ibuf = gpu.buffer({ size: cap * 48, usage: GPUBufferUsage.VERTEX, label: 'blobs' });
    let count = 0;
    const push = (x, y, r, soft, c, a, out) => {
      if (count >= cap) return;
      const o = count++ * 12;
      inst[o] = x;
      inst[o + 1] = y;
      inst[o + 2] = r;
      inst[o + 3] = soft;
      inst[o + 4] = c[0];
      inst[o + 5] = c[1];
      inst[o + 6] = c[2];
      inst[o + 7] = a;
      inst[o + 8] = out;
    };

    // ---- premultiplied-alpha textures (32×32 star, built on the CPU)
    const SZ = 28;
    const src = starSprite(SZ).getContext('2d').getImageData(0, 0, SZ, SZ).data;
    const straightTex = gpu.texture({ size: [SZ, SZ], label: 'star-straight' });
    const premulTex = gpu.texture({ size: [SZ, SZ], label: 'star-premul' });
    let matte = null;
    const buildTextures = (hex) => {
      matte = hex;
      const n = parseInt(hex.slice(1), 16);
      const m = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
      const st = new Uint8Array(src.length);
      const pm = new Uint8Array(src.length);
      for (let i = 0; i < src.length; i += 4) {
        const a = src[i + 3];
        for (let k = 0; k < 3; k++) {
          // straight: fully transparent texels keep the "matte" color (what image editors export)
          st[i + k] = a === 0 ? m[k] : src[i + k];
          pm[i + k] = Math.round((src[i + k] * a) / 255);
        }
        st[i + 3] = a;
        pm[i + 3] = a;
      }
      device.queue.writeTexture({ texture: straightTex }, st, { bytesPerRow: SZ * 4 }, [SZ, SZ]);
      device.queue.writeTexture({ texture: premulTex }, pm, { bytesPerRow: SZ * 4 }, [SZ, SZ]);
    };
    buildTextures('#000000');
    const straightView = straightTex.createView();
    const premulView = premulTex.createView();
    const sprites = new SpriteBatch(gpu, { texture: straightView, filter: 'linear' });
    const cam = new Camera2D();

    // ---- particles for the game tab
    const parts = [];
    const spawn = (o) => parts.push({ age: 0, ...o });
    let acc = { fire: 0, smoke: 0, magic: 0 };
    const rand = (a, b) => a + Math.random() * (b - a);

    return {
      onChange(key, value) {
        if (key === 'matte') buildTextures(value);
      },
      frame(ctx) {
        const P = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const s = Math.max(1, Math.min(W, H) / 700);
        const t = ctx.time;
        const dt = Math.min(ctx.dt, 1 / 30);
        const enc = ctx.encoder;
        const canvas = { view: ctx.target, format: gpu.format };
        const ex = ctx.example;
        count = 0;
        cam.setViewport(W, H);
        if (P.matte !== matte) buildTextures(P.matte);

        if (ex === 'lab') {
          labels(ctx, 'lab', MODES.map((m, i) => ({ text: m.name, x: ((i % 4) + 0.5) / 4, y: (Math.floor(i / 4) + 0.93) / 2, valign: 'middle', style: LBL })));
          const useGame = Number(P.bgKind) === 3;
          const gt = useGame ? game.render(enc, t, Math.round(W / 4), Math.round(H / 2)) : blank;
          UB.set('resolution', [W, H]).set('mode', 0).set('bgKind', Number(P.bgKind)).set('time', t);
          UB.upload();
          // three circles per tile; positions relative to the tile
          const tw = W / 4;
          const th = H / 2;
          const R = Math.min(tw, th) * 0.24;
          const cols = [[1, 0.25, 0.35], [0.25, 0.95, 0.45], [0.3, 0.5, 1]];
          const ph = P.move ? t * 0.8 : 0;
          const ranges = [];
          for (let k = 0; k < 8; k++) {
            const ox = (k % 4) * tw + tw * 0.5;
            const oy = Math.floor(k / 4) * th + th * 0.44;
            const first = count;
            for (let j = 0; j < 3; j++) {
              const jj = P.reverse ? 2 - j : j;
              const a = ph + (jj * Math.PI * 2) / 3 - Math.PI / 2;
              const rr = R * (0.55 + 0.06 * Math.sin(ph * 1.3 + jj));
              push(ox + Math.cos(a) * rr, oy + Math.sin(a) * rr, R, P.soft, cols[jj], P.opacity, MODES[k].out);
            }
            ranges.push([first, count - first, ox - tw / 2, oy - th * 0.44, tw, th]);
          }
          device.queue.writeBuffer(ibuf, 0, inst, 0, count * 12);
          bg.draw(enc, canvas, { game: gt });
          const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.target, loadOp: 'load', storeOp: 'store' }] });
          pass.setBindGroup(0, blob.bind({ u: UB }));
          pass.setVertexBuffer(0, ibuf);
          ranges.forEach(([first, n, x, y, w, h], k) => {
            pass.setScissorRect(Math.max(0, Math.floor(x + 2)), Math.max(0, Math.floor(y + 2)), Math.max(1, Math.floor(w - 4)), Math.max(1, Math.floor(h - 4)));
            pass.setPipeline(pipes[k]);
            pass.draw(6, n, 0, first);
          });
          pass.end();
          return;
        }

        if (ex === 'premul') {
          labels(ctx, 'premul', [
            { text: 'straight α · <b>nearest</b> — no fringe, but blocky', x: 1 / 6, y: 0.95, valign: 'bottom', style: LBL },
            { text: 'straight α · <b>linear</b> — dark fringe', x: 0.5, y: 0.95, valign: 'bottom', style: LBL },
            { text: '<b>premultiplied</b> · linear — clean', x: 5 / 6, y: 0.95, valign: 'bottom', style: LBL },
          ]);
          UB.set('resolution', [W, H]).set('mode', 1);
          UB.upload();
          bg.draw(enc, canvas, { game: blank });
          const pw = W / 3;
          const size = Math.min(SZ * P.zoom * s, pw * 0.8, H * 0.62);
          const rot = P.move ? Math.sin(t * 0.5) * 0.5 : 0;
          const bob = P.move ? Math.sin(t * 0.9) * H * 0.015 : 0;
          // ONE begin() per frame: each flush uploads only what was queued since the previous flush
          // (writeBuffer runs before the frame's commands, so reusing offsets would overwrite panels).
          sprites.begin();
          const mk = (k) => {
            const cx = pw * (k + 0.5);
            sprites.draw(cx, H * 0.44 + bob, size, size, { rotation: rot });
            // a few small ones: the fringe matters when minified too
            for (let j = 0; j < 3; j++) sprites.draw(cx + (j - 1) * pw * 0.26, H * 0.83, SZ * s * (1.2 + j * 0.5), SZ * s * (1.2 + j * 0.5), { rotation: -rot * 2 });
          };
          mk(0);
          sprites.flush(enc, canvas, cam, { texture: straightView, filter: 'nearest', blend: 'alpha' });
          mk(1);
          sprites.flush(enc, canvas, cam, { texture: straightView, filter: 'linear', blend: 'alpha' });
          mk(2);
          sprites.flush(enc, canvas, cam, { texture: premulView, filter: 'linear', blend: 'premultiplied' });
          return;
        }

        // ---------------- game tab
        labels(ctx, 'game', []);
        const gs = ctx.testMode ? 0.5 : 1;
        const gt = game.render(enc, t, Math.round(W * gs), Math.round(H * gs));
        const fire = [W * 0.7, H * 0.79];
        for (const U of [UB, UL1, UL2]) U.set('resolution', [W, H]).set('time', t).set('fire', fire).set('strength', P.strength).set('strength2', 1);
        UB.set('mode', 2);
        UL1.set('mode', 0);
        UL2.set('mode', 1);
        UB.upload();
        bg.draw(enc, canvas, { game: gt });
        if (P.shadows) layerMul.draw(enc, canvas, {}, { clear: false });
        if (P.lights) layerScr.draw(enc, canvas, {}, { clear: false });

        // simulate particles
        const rate = ctx.testMode ? 0.4 : 1;
        if (!ctx.paused) {
          acc.fire += dt * 150 * rate;
          acc.smoke += dt * 14 * rate;
          acc.magic += dt * 70 * rate;
          while (acc.fire >= 1) {
            acc.fire--;
            spawn({ kind: 0, x: fire[0] + rand(-18, 18) * s, y: fire[1] - rand(0, 8) * s, vx: rand(-14, 14) * s, vy: -rand(60, 140) * s, life: rand(0.5, 1.0), r: rand(16, 30) * s });
          }
          while (acc.smoke >= 1) {
            acc.smoke--;
            spawn({ kind: 1, x: fire[0] + rand(-8, 8) * s, y: fire[1] - 70 * s, vx: rand(-6, 14) * s, vy: -rand(25, 45) * s, life: rand(2, 3.2), r: rand(14, 22) * s });
          }
          const p = ctx.pointer;
          const mx = p.over ? p.x : W * 0.42 + Math.cos(t * 1.3) * W * 0.12;
          const my = p.over ? p.y : H * 0.45 + Math.sin(t * 2.1) * H * 0.1;
          while (acc.magic >= 1) {
            acc.magic--;
            const a = rand(0, Math.PI * 2);
            spawn({ kind: 2, x: mx + Math.cos(a) * 22 * s, y: my + Math.sin(a) * 22 * s, vx: Math.cos(a) * 40 * s, vy: Math.sin(a) * 40 * s - 25 * s, life: rand(0.6, 1.1), r: rand(8, 16) * s, hue: Math.random() });
          }
          for (let i = parts.length - 1; i >= 0; i--) {
            const q = parts[i];
            q.age += dt;
            q.x += q.vx * dt;
            q.y += q.vy * dt;
            if (q.kind === 0) q.vx += Math.sin(t * 3 + q.y * 0.05) * 20 * s * dt;
            if (q.kind === 1) q.vx += 8 * s * dt;
            if (q.age > q.life) parts.splice(i, 1);
          }
        }
        // smoke first (alpha), then fire & magic (additive or alpha)
        const fireOut = 0;
        for (const q of parts) {
          if (q.kind !== 1) continue;
          const k = q.age / q.life;
          push(q.x, q.y, q.r * (1 + k * 2.2), 1, [0.22, 0.2, 0.22], 0.32 * Math.sin(Math.PI * k), 0);
        }
        // campfire logs (plain alpha)
        for (const [dx, dy, r] of [[-16, 4, 9], [0, 7, 10], [16, 4, 9], [-6, 1, 7], [8, 1, 7]]) push(fire[0] + dx * s, fire[1] + dy * s, r * s, 0.15, [0.28, 0.15, 0.08], 1, 0);
        const smokeN = count;
        for (const q of parts) {
          if (q.kind === 1) continue;
          const k = q.age / q.life;
          if (q.kind === 0) {
            const c = k < 0.25 ? [1, 0.9, 0.6] : k < 0.6 ? [1, 0.5, 0.15] : [0.8, 0.18, 0.08];
            push(q.x, q.y, q.r * (1 - k * 0.6), 1, c, (1 - k) * 0.55, fireOut);
          } else {
            const c = q.hue < 0.5 ? [0.45, 0.6, 1] : [0.85, 0.45, 1];
            push(q.x, q.y, q.r * (1 - k * 0.5), 1, c, (1 - k) * 0.8, fireOut);
          }
        }
        device.queue.writeBuffer(ibuf, 0, inst, 0, count * 12);
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.target, loadOp: 'load', storeOp: 'store' }] });
        pass.setBindGroup(0, blob.bind({ u: UB }));
        pass.setVertexBuffer(0, ibuf);
        pass.setPipeline(pipeAlpha);
        if (smokeN) pass.draw(6, smokeN, 0, 0);
        pass.setPipeline(P.fireBlend === 'alpha' ? pipeAlpha : pipeAdd);
        if (count > smokeN) pass.draw(6, count - smokeN, 0, smokeN);
        pass.end();
      },
    };
  },
};

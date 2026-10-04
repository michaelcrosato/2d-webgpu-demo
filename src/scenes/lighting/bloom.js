// HDR bloom, the way modern engines do it:
//   scene -> HDR target (rgba16float, values > 1 allowed)
//   prefilter: soft-knee threshold + 13-tap "Karis" downsample (kills fireflies)  -> mip 1
//   downsample chain (13-tap Call of Duty / or dual Kawase)                      -> mips 2..N
//   upsample chain: tent filter, blended back up level by level                  -> bloom
//   composite: scene + bloom × intensity, exposure, tone mapping, sRGB

import { Camera2D, ShapeBatch } from '../../core/batch.js';
import { createGameScene } from '../../core/gamescene.js';
import { rng } from '../../core/assets.js';
import { tag } from './_shared.js';

const MAX_MIPS = 8;

const FILTERS_WGSL = /* wgsl */ `
// Soft-knee threshold: below (threshold - knee) nothing passes, above (threshold + knee) everything
// above the threshold passes, with a smooth quadratic curve in between (no hard "pop").
fn prefilter(c: vec3f) -> vec3f {
  let br = max(c.r, max(c.g, c.b));
  let knee = u.threshold * u.knee + 1e-4;
  var rq = clamp(br - u.threshold + knee, 0.0, 2.0 * knee);
  rq = rq * rq / (4.0 * knee);
  let w = max(rq, br - u.threshold) / max(br, 1e-4);
  return c * w;
}
fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + luma(c)); }

// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare")
fn down13(uv: vec2f, karisAvg: bool) -> vec3f {
  let ts = 1.0 / TEXSIZE(src);
  let a = TEX(src, uv + ts * vec2f(-2.0, -2.0)).rgb;
  let b = TEX(src, uv + ts * vec2f(0.0, -2.0)).rgb;
  let c = TEX(src, uv + ts * vec2f(2.0, -2.0)).rgb;
  let d = TEX(src, uv + ts * vec2f(-1.0, -1.0)).rgb;
  let e = TEX(src, uv + ts * vec2f(1.0, -1.0)).rgb;
  let f = TEX(src, uv + ts * vec2f(-2.0, 0.0)).rgb;
  let g = TEX(src, uv).rgb;
  let h = TEX(src, uv + ts * vec2f(2.0, 0.0)).rgb;
  let i = TEX(src, uv + ts * vec2f(-1.0, 1.0)).rgb;
  let j = TEX(src, uv + ts * vec2f(1.0, 1.0)).rgb;
  let k = TEX(src, uv + ts * vec2f(-2.0, 2.0)).rgb;
  let l = TEX(src, uv + ts * vec2f(0.0, 2.0)).rgb;
  let m = TEX(src, uv + ts * vec2f(2.0, 2.0)).rgb;
  // five overlapping 2×2 boxes: the center one counts 0.5, the four corner ones 0.125 each
  let g0 = (d + e + i + j) * 0.25;
  let g1 = (a + b + f + g) * 0.25;
  let g2 = (b + c + g + h) * 0.25;
  let g3 = (f + g + k + l) * 0.25;
  let g4 = (g + h + l + m) * 0.25;
  if (karisAvg) {
    // weight each box by 1/(1+luma): one super-bright pixel can no longer flicker the whole bloom
    let w0 = karis(g0) * 0.5;
    let w1 = karis(g1) * 0.125;
    let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125;
    let w4 = karis(g4) * 0.125;
    return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  }
  return g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125;
}
// Dual-filter (Kawase) downsample: 5 taps
fn downDual(uv: vec2f) -> vec3f {
  let ts = 1.0 / TEXSIZE(src);
  var s = TEX(src, uv).rgb * 4.0;
  s += TEX(src, uv + ts * vec2f(-1.0, -1.0)).rgb + TEX(src, uv + ts * vec2f(1.0, -1.0)).rgb;
  s += TEX(src, uv + ts * vec2f(-1.0, 1.0)).rgb + TEX(src, uv + ts * vec2f(1.0, 1.0)).rgb;
  return s / 8.0;
}
// 3×3 tent upsample of the smaller level
fn upTent(uv: vec2f) -> vec3f {
  let ts = u.radius / TEXSIZE(src);
  var s = TEX(src, uv).rgb * 4.0;
  s += (TEX(src, uv + vec2f(-ts.x, 0.0)).rgb + TEX(src, uv + vec2f(ts.x, 0.0)).rgb + TEX(src, uv + vec2f(0.0, -ts.y)).rgb + TEX(src, uv + vec2f(0.0, ts.y)).rgb) * 2.0;
  s += TEX(src, uv - ts).rgb + TEX(src, uv + ts).rgb + TEX(src, uv + vec2f(ts.x, -ts.y)).rgb + TEX(src, uv + vec2f(-ts.x, ts.y)).rgb;
  return s / 16.0;
}
// Dual-filter upsample: 8 taps on a diamond
fn upDual(uv: vec2f) -> vec3f {
  let ts = u.radius / TEXSIZE(src);
  var s = TEX(src, uv + vec2f(-2.0 * ts.x, 0.0)).rgb + TEX(src, uv + vec2f(2.0 * ts.x, 0.0)).rgb;
  s += TEX(src, uv + vec2f(0.0, -2.0 * ts.y)).rgb + TEX(src, uv + vec2f(0.0, 2.0 * ts.y)).rgb;
  s += (TEX(src, uv + vec2f(-ts.x, -ts.y)).rgb + TEX(src, uv + vec2f(ts.x, -ts.y)).rgb + TEX(src, uv + vec2f(-ts.x, ts.y)).rgb + TEX(src, uv + vec2f(ts.x, ts.y)).rgb) * 2.0;
  return s / 12.0;
}`;

const UNIFORMS = {
  resolution: 'vec2f',
  threshold: 'f32',
  knee: 'f32',
  intensity: 'f32',
  radius: 'f32',
  scatter: 'f32',
  exposure: 'f32',
  tonemap: 'f32',
  view: 'f32',
  filterMode: 'f32',
  time: 'f32',
  sun: 'vec4f',
};

export default {
  interaction: 'Move the mouse: it carries a very bright light. Watch how brightness, not size, decides how far it glows.',
  examples: [
    {
      id: 'pipeline',
      label: 'Bloom pipeline',
      kind: 'Abstract',
      note: 'Discs of the <b>same color</b> but brightness 0.25 → 16 (HDR: the screen can only show up to 1). Only what passes the <b>threshold</b> blooms, and brighter = wider glow. Turn on the mip strip (top) to see the blur pyramid.',
      params: { threshold: 1, knee: 0.5, intensity: 0.8, scatter: 0.7, levels: 6, tonemap: 'aces', exposure: 1, view: 'final', strip: true },
    },
    {
      id: 'arcade',
      label: 'Neon arcade',
      kind: 'In a game',
      note: 'Vector graphics à la <i>Geometry Wars</i>: every line is drawn with HDR brightness 2–8, and bloom turns them into glowing neon tubes. Without bloom (intensity 0) it looks flat and thin.',
      params: { threshold: 0.9, knee: 0.6, intensity: 1.1, scatter: 0.75, levels: 7, tonemap: 'aces', exposure: 1, view: 'final', strip: false },
    },
    {
      id: 'magic',
      label: 'Magic & sun',
      kind: 'In a game',
      note: 'A normal (LDR) game scene with a few <b>HDR emissive</b> elements on top: the sun’s core, a spell orb following the mouse and its sparks. The threshold keeps the ordinary scene crisp while the magic glows.',
      params: { threshold: 1.1, knee: 0.4, intensity: 0.9, scatter: 0.7, levels: 7, tonemap: 'aces', exposure: 1, view: 'final', strip: false },
    },
  ],
  controls: [
    { type: 'heading', label: 'Bright pass' },
    { type: 'slider', key: 'threshold', label: 'Threshold', min: 0, max: 4, step: 0.01, value: 1, help: 'Brightness where bloom starts. 1.0 = “brighter than white”. 0 = everything glows (dreamy).' },
    { type: 'slider', key: 'knee', label: 'Soft knee', min: 0, max: 1, step: 0.01, value: 0.5, help: 'Softens the cutoff so things fade into bloom instead of popping.' },
    { type: 'heading', label: 'Blur pyramid' },
    { type: 'slider', key: 'levels', label: 'Mip levels', min: 1, max: MAX_MIPS, step: 1, value: 6, help: 'Each level is half the size of the previous: more levels = wider glow, almost free.' },
    { type: 'slider', key: 'scatter', label: 'Scatter (radius)', min: 0, max: 1, step: 0.01, value: 0.7, help: 'How much the wide (small) levels contribute when going back up. Low = tight glow, high = big halo.' },
    {
      type: 'select',
      key: 'filter',
      label: 'Filter',
      value: 'cod',
      options: [
        { value: 'cod', label: '13-tap down + tent up (Call of Duty)' },
        { value: 'dual', label: 'Dual filter (Kawase, cheaper)' },
      ],
    },
    { type: 'heading', label: 'Composite' },
    { type: 'slider', key: 'intensity', label: 'Bloom intensity', min: 0, max: 3, step: 0.01, value: 0.8 },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: 0.1, max: 4, step: 0.01, value: 1 },
    {
      type: 'select',
      key: 'tonemap',
      label: 'Tone mapping',
      value: 'aces',
      options: [
        { value: 'none', label: 'None (clamp)' },
        { value: 'reinhard', label: 'Reinhard' },
        { value: 'aces', label: 'ACES (filmic)' },
      ],
      help: 'How HDR values above 1 are squeezed into the screen range.',
    },
    {
      type: 'select',
      key: 'view',
      label: 'View',
      value: 'final',
      options: [
        { value: 'final', label: 'Final image' },
        { value: 'scene', label: '1. HDR scene (no bloom)' },
        { value: 'bright', label: '2. Bright pass (mip 1)' },
        { value: 'mip', label: '3. Downsample level…' },
        { value: 'bloom', label: '4. Bloom only (after upsampling)' },
      ],
    },
    { type: 'slider', key: 'mip', label: 'Level to show', min: 1, max: MAX_MIPS, step: 1, value: 4, help: 'For View → Downsample level.' },
    { type: 'toggle', key: 'strip', label: 'Show mip strip', value: true, help: 'Thumbnails of every pyramid level along the top.' },
  ],
  about: {
    summary:
      'Bloom makes very bright things bleed light into their surroundings, like a camera lens or your eye. It is the single cheapest way to make a game look “lit”.',
    what: `<p>The scene is rendered in <b>HDR</b> (16-bit float colors that can go far above 1.0). Everything brighter than a threshold
      is extracted, blurred very widely using a pyramid of ever-smaller textures, and added back on top. Then a <b>tone mapper</b> squeezes
      the result into the 0–1 range of the screen.</p>`,
    how: `<ol>
      <li><b>HDR render target</b>: draw the scene into an <code>rgba16float</code> texture so a torch can be 8× brighter than white.</li>
      <li><b>Bright pass</b> with a <b>soft knee</b>: keep only the part above the threshold, with a smooth curve so nothing pops.</li>
      <li><b>Downsample chain</b>: halve the resolution 6–8 times. Each level uses a 13-tap filter (5 overlapping boxes) which avoids
        the blocky, shimmering look of naive 2×2 averaging. The first level uses a <b>Karis average</b> (weights by 1/(1+brightness)) so
        single bright pixels don’t flicker.</li>
      <li><b>Upsample chain</b>: go back up, at each level blurring the smaller level with a 3×3 tent filter and blending it with the
        current level. Small levels = very wide blur at almost no cost (a 1/64-size texture blurred 3×3 covers ~200 screen pixels).</li>
      <li><b>Composite</b>: <code>scene + bloom × intensity</code>, exposure, tone mapping (Reinhard or ACES), then sRGB.</li>
    </ol>`,
    uses: [
      { title: 'Neon & vector looks', text: 'Geometry Wars, Tron-likes and synthwave games are basically “lines + bloom”.' },
      { title: 'Magic, fire, lasers', text: 'Emissive VFX feel hot and energetic once they glow past their edges.' },
      { title: 'Sun & sky', text: 'A bright sun or window that flares into the scene; lens-like softness.' },
      { title: 'Feedback', text: 'Pickups, crits and power-ups flash brighter than everything else and draw the eye.' },
    ],
    try: [
      'On <b>Bloom pipeline</b>, slide <i>Threshold</i> from 0 to 4: discs drop out of the bloom one by one.',
      'Set <i>Mip levels</i> to 1, then 8: the glow radius grows enormously, but the cost barely changes.',
      'Set <i>Tone mapping</i> to None: the bright discs clip to flat white circles. ACES keeps a sense of “very bright”.',
      'Set <i>Knee</i> to 0 and move the mouse light slowly through the threshold — it pops. Knee 1 fades in smoothly.',
      'Step through <i>View</i> 1 → 4 to see each stage; use <i>Level to show</i> to inspect the tiny mips.',
    ],
    ask: [
      'HDR bloom with a soft-knee threshold',
      'Call of Duty style 13-tap downsample / tent upsample bloom',
      'dual Kawase blur bloom for mobile',
      'ACES filmic tone mapping',
      'emissive sprites that glow with bloom',
      'neon vector graphics with glow',
    ],
    perf: `<p>Very cheap: the first downsample reads the full-resolution image once; every later level is 4× smaller, so the whole chain
      costs about 1.33× one full-screen pass, no matter how wide the glow. The dual filter uses fewer taps (5 down, 8 up) — popular on
      mobile. Bloom is usually &lt; 0.5 ms on a desktop GPU.</p>`,
    api: `<p>Works the same in WebGL2 (render-to-texture with half-float color buffers needs <code>EXT_color_buffer_float</code>).
      In WebGPU <code>rgba16float</code> render targets are always available and filterable, so HDR is guaranteed everywhere WebGPU runs.</p>`,
    code: [
      {
        title: 'Soft-knee threshold',
        lang: 'wgsl',
        src: `fn prefilter(c: vec3f) -> vec3f {
  let br = max(c.r, max(c.g, c.b));
  let knee = u.threshold * u.knee + 1e-4;
  var rq = clamp(br - u.threshold + knee, 0.0, 2.0 * knee);
  rq = rq * rq / (4.0 * knee);
  let w = max(rq, br - u.threshold) / max(br, 1e-4);
  return c * w;
}`,
      },
      {
        title: 'Upsample: blend the wider level back in',
        lang: 'wgsl',
        src: `// up[i] = mix(mip[i], tent(up[i+1]), scatter)
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let wide = upTent(uv);                  // 3×3 tent of the smaller level
  let here = TEX(base, uv).rgb;           // this level of the downsample chain
  return vec4f(mix(here, wide, u.scatter), 1.0);
}`,
      },
      {
        title: 'JavaScript: the pass chain',
        lang: 'js',
        src: `prefilter.draw(enc, mips[0], { src: hdr });               // threshold + Karis 13-tap
for (let i = 1; i < levels; i++) down.draw(enc, mips[i], { src: mips[i - 1] });
let wide = mips[levels - 1];
for (let i = levels - 2; i >= 0; i--) {
  up.draw(enc, ups[i], { src: wide, base: mips[i] });
  wide = ups[i];
}
composite.draw(enc, canvas, { hdr, bloom: wide });`,
      },
    ],
    links: [
      { title: 'Jimenez — Next Generation Post Processing in Call of Duty: AW (SIGGRAPH 2014)', url: 'https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/' },
      { title: 'LearnOpenGL — Physically Based Bloom', url: 'https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const U = gpu.uniforms(UNIFORMS, 'BloomU');
    const inc = ['color'];
    const prefilterFx = gpu.fullscreen({
      label: 'bloom-prefilter',
      uniforms: U,
      textures: ['src'],
      include: inc,
      code: `${FILTERS_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(prefilter(down13(uv, true)), 1.0); }`,
    });
    const downFx = gpu.fullscreen({
      label: 'bloom-down',
      uniforms: U,
      textures: ['src'],
      include: inc,
      code: `${FILTERS_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  if (u.filterMode > 0.5) { return vec4f(downDual(uv), 1.0); }
  return vec4f(down13(uv, false), 1.0);
}`,
    });
    const upFx = gpu.fullscreen({
      label: 'bloom-up',
      uniforms: U,
      textures: ['src', 'base'],
      include: inc,
      code: `${FILTERS_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var wide = upTent(uv);
  if (u.filterMode > 0.5) { wide = upDual(uv); }
  let here = TEX(base, uv).rgb;
  return vec4f(mix(here, wide, u.scatter), 1.0);
}`,
    });
    const compositeFx = gpu.fullscreen({
      label: 'bloom-composite',
      uniforms: U,
      textures: ['hdr', 'bloom', 'dbg'],
      include: ['color', 'hash'],
      code: /* wgsl */ `
fn tonemap(c: vec3f) -> vec3f {
  if (u.tonemap < 0.5) { return clamp(c, vec3f(0.0), vec3f(1.0)); }
  if (u.tonemap < 1.5) { return tonemapReinhard(c); }
  return tonemapACES(c);
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let v = i32(u.view + 0.5);
  let scene = TEX(hdr, uv).rgb;
  var c = scene + TEX(bloom, uv).rgb * u.intensity;
  if (v == 1) { c = scene; }
  if (v == 2 || v == 3) { c = TEX(dbg, uv).rgb; }
  if (v == 4) { c = TEX(bloom, uv).rgb * u.intensity; }
  var o = linearToSrgb(tonemap(c * u.exposure));
  o += (ign(px) - 0.5) / 255.0;
  return vec4f(o, 1.0);
}`,
    });
    const thumbFx = gpu.fullscreen({
      label: 'bloom-thumb',
      uniforms: U,
      textures: ['src'],
      include: ['color'],
      code: `fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(linearToSrgb(tonemapACES(TEX(src, uv).rgb * 2.0)), 1.0); }`,
    });
    // LDR game scene -> linear HDR, plus an HDR sun core
    const baseFx = gpu.fullscreen({
      label: 'bloom-base',
      uniforms: U,
      textures: ['game'],
      include: ['color'],
      code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var c = srgbToLinear(TEX(game, uv).rgb);
  let d = length(px - u.sun.xy);
  c += vec3f(9.0, 7.0, 4.0) * (1.0 - smoothstep(u.sun.z * 0.7, u.sun.z, d));
  return vec4f(c, 1.0);
}`,
    });
    const game = createGameScene(gpu);
    const cam = new Camera2D();
    const shapes = new ShapeBatch(gpu, { capacity: 8192 });

    let hdr = null;
    let mips = [];
    let ups = [];
    const alloc = () => {
      hdr?.destroy();
      for (const t of [...mips, ...ups]) t.destroy();
      hdr = gpu.target(ctx.width, ctx.height, { format: 'rgba16float', label: 'hdr' });
      mips = [];
      ups = [];
      let w = ctx.width;
      let h = ctx.height;
      for (let i = 0; i < MAX_MIPS; i++) {
        w = Math.max(2, Math.ceil(w / 2));
        h = Math.max(2, Math.ceil(h / 2));
        mips.push(gpu.target(w, h, { format: 'rgba16float', label: `mip${i + 1}` }));
        ups.push(gpu.target(w, h, { format: 'rgba16float', label: `up${i + 1}` }));
      }
    };
    alloc();

    const readout = tag(ctx, 'right:8px;bottom:8px');
    const labels = [];
    const LEVELS = [0.25, 0.5, 1, 2, 4, 8, 16];
    for (let i = 0; i < LEVELS.length; i++) {
      const el = tag(ctx, 'display:none;transform:translateX(-50%)');
      el.textContent = `${LEVELS[i]}`;
      labels.push(el);
    }

    // arcade state
    const R = rng(3);
    const enemies = Array.from({ length: 14 }, () => ({ x: R(), y: R(), a: R() * 6.28, kind: Math.floor(R() * 3), sp: 0.03 + R() * 0.05, ph: R() * 6 }));
    let bullets = [];
    let sparks = [];
    let ship = { x: 0.5, y: 0.5, a: 0 };
    let fireT = 0;
    let orb = { x: 0, y: 0, init: false };
    let magicSparks = [];

    const line = (x1, y1, x2, y2, w, c) => shapes.line(x1, y1, x2, y2, w, c);
    const poly = (pts, w, c) => shapes.polyline(pts, w, c, { closed: true });

    function drawPipeline(t, W, H, ptr) {
      // dim grid backdrop
      const step = H / 10;
      for (let x = (W / 2) % step; x < W; x += step) line(x, 0, x, H, 1, [0.03, 0.035, 0.06, 1]);
      for (let y = (H / 2) % step; y < H; y += step) line(0, y, W, y, 1, [0.03, 0.035, 0.06, 1]);
      const n = LEVELS.length;
      const gap = Math.min(W / (n + 1), H * 0.2);
      const r = gap * 0.28;
      const y0 = H * 0.42;
      LEVELS.forEach((v, i) => {
        const x = W / 2 + (i - (n - 1) / 2) * gap;
        shapes.circle(x, y0, r, [1.0 * v, 0.45 * v, 0.12 * v, 1]);
        const el = labels[i];
        el.style.display = '';
        el.style.left = `${x / ctx.dpr}px`;
        el.style.top = `${(y0 + r * 1.5) / ctx.dpr}px`;
      });
      // thin HDR lines: same brightness, different widths
      const ly = H * 0.68;
      [1, 2, 4].forEach((w, i) => line(W * 0.18, ly + i * H * 0.05, W * 0.48, ly + i * H * 0.05, w * Math.max(1, H / 500), [0.3, 2.5, 4, 1]));
      // tiny sparkles (single bright pixels -> fireflies without Karis averaging)
      for (let i = 0; i < 40; i++) {
        const a = i * 2.4 + t * 0.3;
        const rr = H * (0.05 + 0.12 * ((i * 0.618) % 1));
        const x = W * 0.72 + Math.cos(a) * rr;
        const y = H * 0.74 + Math.sin(a) * rr * 0.6;
        shapes.circle(x, y, Math.max(1, H / 700), [6, 6, 8, 1]);
      }
      // the mouse light
      const mx = ptr.over ? ptr.x : W / 2 + Math.cos(t * 0.7) * W * 0.3;
      const my = ptr.over ? ptr.y : H * 0.86;
      shapes.circle(mx, my, H * 0.012, [20, 16, 10, 1]);
    }

    function drawArcade(t, dt, W, H, ptr) {
      for (const l of labels) l.style.display = 'none';
      const S = H;
      // warped grid
      const g = S / 16;
      const warp = (x, y) => {
        const dx = x - ship.x * W;
        const dy = y - ship.y * H;
        const d = Math.hypot(dx, dy) + 1;
        const k = (S * 0.06 * Math.exp(-d / (S * 0.18))) / d;
        return [x - dx * k, y - dy * k];
      };
      const gridC = [0.04, 0.07, 0.26, 1];
      for (let x = 0; x <= W + g; x += g) {
        let prev = null;
        for (let y = 0; y <= H + g; y += g / 2) {
          const pnt = warp(x, y);
          if (prev) line(prev[0], prev[1], pnt[0], pnt[1], Math.max(1, S / 700), gridC);
          prev = pnt;
        }
      }
      for (let y = 0; y <= H + g; y += g) {
        let prev = null;
        for (let x = 0; x <= W + g; x += g / 2) {
          const pnt = warp(x, y);
          if (prev) line(prev[0], prev[1], pnt[0], pnt[1], Math.max(1, S / 700), gridC);
          prev = pnt;
        }
      }
      // ship follows the mouse
      const tx = ptr.over ? ptr.x / W : 0.5 + Math.cos(t * 0.6) * 0.25;
      const ty = ptr.over ? ptr.y / H : 0.5 + Math.sin(t * 0.9) * 0.2;
      const dx = tx - ship.x;
      const dy = ty - ship.y;
      if (Math.hypot(dx, dy) > 0.002) ship.a = Math.atan2(dy * H, dx * W);
      ship.x += dx * Math.min(1, dt * 4);
      ship.y += dy * Math.min(1, dt * 4);
      // shoot at the nearest enemy
      fireT -= dt;
      let target = null;
      let best = 1e9;
      for (const e of enemies) {
        const d = Math.hypot((e.x - ship.x) * W, (e.y - ship.y) * H);
        if (d < best) {
          best = d;
          target = e;
        }
      }
      if (fireT <= 0 && target && dt > 0) {
        fireT = 0.09;
        const a = Math.atan2((target.y - ship.y) * H, (target.x - ship.x) * W);
        for (const s of [-0.06, 0.06]) bullets.push({ x: ship.x * W, y: ship.y * H, vx: Math.cos(a + s) * S * 1.6, vy: Math.sin(a + s) * S * 1.6, life: 1 });
      }
      for (const b of bullets) {
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        b.life -= dt;
        for (const e of enemies) {
          if (Math.hypot(b.x - e.x * W, b.y - e.y * H) < S * 0.03) {
            b.life = 0;
            for (let k = 0; k < 18; k++) {
              const a = Math.random() * 6.28;
              const sp = S * (0.2 + Math.random() * 0.6);
              sparks.push({ x: e.x * W, y: e.y * H, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, life: 0.6 + Math.random() * 0.4, c: e.kind });
            }
            e.x = Math.random();
            e.y = Math.random() < 0.5 ? -0.05 : 1.05;
          }
        }
      }
      bullets = bullets.filter((b) => b.life > 0 && b.x > -50 && b.x < W + 50 && b.y > -50 && b.y < H + 50);
      const ecol = [[4, 0.3, 2.6, 1], [0.4, 3.5, 1.0, 1], [0.4, 1.6, 4.5, 1]];
      for (const e of enemies) {
        if (dt > 0) {
          const ax = ship.x - e.x;
          const ay = ship.y - e.y;
          const d = Math.hypot(ax, ay) + 1e-3;
          e.x += (ax / d) * e.sp * dt + Math.sin(t * 2 + e.ph) * 0.0006;
          e.y += (ay / d) * e.sp * dt + Math.cos(t * 2 + e.ph) * 0.0006;
          e.a += dt * 2;
        }
        const x = e.x * W;
        const y = e.y * H;
        const r = S * 0.022;
        const w = Math.max(1.5, S / 350);
        const c = ecol[e.kind];
        if (e.kind === 0) poly([[x, y - r], [x + r, y], [x, y + r], [x - r, y]], w, c);
        else if (e.kind === 1) shapes.box(x, y, r * 0.8, r * 0.8, c, { rotation: e.a, stroke: w });
        else {
          const pts = [];
          for (let k = 0; k < 3; k++) pts.push([x + Math.cos(e.a + k * 2.094) * r, y + Math.sin(e.a + k * 2.094) * r]);
          poly(pts, w, c);
        }
      }
      for (const s of sparks) {
        s.x += s.vx * dt;
        s.y += s.vy * dt;
        s.vx *= 0.96;
        s.vy *= 0.96;
        s.life -= dt;
        const c = ecol[s.c];
        line(s.x, s.y, s.x - s.vx * 0.03, s.y - s.vy * 0.03, Math.max(1, S / 500), [c[0] * 1.5, c[1] * 1.5, c[2] * 1.5, Math.min(1, s.life * 2)]);
      }
      sparks = sparks.filter((s) => s.life > 0);
      for (const b of bullets) line(b.x, b.y, b.x - b.vx * 0.012, b.y - b.vy * 0.012, Math.max(1.5, S / 300), [6, 4.5, 1.2, 1]);
      // ship: a claw shape
      const sx = ship.x * W;
      const sy = ship.y * H;
      const r = S * 0.03;
      const P = (a, rr) => [sx + Math.cos(ship.a + a) * rr, sy + Math.sin(ship.a + a) * rr];
      poly([P(0, r), P(2.4, r), P(Math.PI, r * 0.35), P(-2.4, r)], Math.max(2, S / 260), [5, 5, 6, 1]);
    }

    function drawMagic(t, dt, W, H, ptr) {
      for (const l of labels) l.style.display = 'none';
      if (!orb.init) orb = { x: W * 0.35, y: H * 0.45, init: true };
      const tx = ptr.over ? ptr.x : W * (0.4 + Math.cos(t * 0.5) * 0.2);
      const ty = ptr.over ? ptr.y : H * (0.45 + Math.sin(t * 0.8) * 0.1);
      orb.x += (tx - orb.x) * Math.min(1, dt * 6);
      orb.y += (ty - orb.y) * Math.min(1, dt * 6);
      if (dt > 0)
        for (let k = 0; k < 3; k++) {
          const a = Math.random() * 6.28;
          magicSparks.push({ x: orb.x, y: orb.y, vx: Math.cos(a) * H * 0.08, vy: Math.sin(a) * H * 0.08 - H * 0.05, life: 1 });
        }
      for (const s of magicSparks) {
        s.x += s.vx * dt;
        s.y += s.vy * dt;
        s.vy += H * 0.05 * dt;
        s.life -= dt * 0.9;
        const k = Math.max(0, s.life);
        shapes.circle(s.x, s.y, Math.max(1, H * 0.003 * k + 0.5), [2.5 * k + 0.5, 1.2 * k + 0.6, 6 * k, 1]);
      }
      magicSparks = magicSparks.filter((s) => s.life > 0);
      // orb with rotating runes
      const r = H * 0.03;
      shapes.circle(orb.x, orb.y, r, [3, 2, 9, 1]);
      shapes.circle(orb.x, orb.y, r * 0.55, [12, 11, 16, 1]);
      for (let k = 0; k < 3; k++) {
        const a = t * 2 + (k * Math.PI * 2) / 3;
        shapes.circle(orb.x + Math.cos(a) * r * 2, orb.y + Math.sin(a) * r * 2 * 0.6, r * 0.18, [4, 2, 10, 1]);
      }
    }

    return {
      resize() {
        alloc();
        orb.init = false;
      },
      onExample() {
        bullets = [];
        sparks = [];
        magicSparks = [];
        orb.init = false;
      },
      frame(ctx) {
        const W = ctx.width;
        const H = ctx.height;
        const p = ctx.params;
        const t = ctx.time;
        const dt = ctx.dt;
        const enc = ctx.encoder;
        if (hdr.width !== W || hdr.height !== H) alloc();
        cam.setViewport(W, H);
        cam.x = W / 2;
        cam.y = H / 2;
        const levels = Math.max(1, Math.min(MAX_MIPS, Math.round(p.levels)));
        const viewIdx = { final: 0, scene: 1, bright: 2, mip: 3, bloom: 4 }[p.view] ?? 0;
        U.setAll({
          resolution: [W, H],
          threshold: p.threshold,
          knee: p.knee,
          intensity: p.intensity,
          radius: 1,
          scatter: p.scatter,
          exposure: p.exposure,
          tonemap: { none: 0, reinhard: 1, aces: 2 }[p.tonemap] ?? 2,
          view: viewIdx,
          filterMode: p.filter === 'dual' ? 1 : 0,
          time: t,
          sun: [0.78 * W, 0.25 * H, 0.055 * H, 0],
        });
        U.upload();

        // ---- 1. HDR scene
        shapes.begin();
        if (ctx.example === 'magic') {
          const gs = ctx.testMode ? 0.5 : 1; // the procedural game scene is heavy on the software test GPU
          const g = game.render(enc, t, Math.round(W * gs), Math.round(H * gs));
          baseFx.draw(enc, hdr, { game: g });
          drawMagic(t, dt, W, H, ctx.pointer);
          shapes.flush(enc, hdr, cam, { blend: 'additive' });
        } else {
          if (ctx.example === 'arcade') drawArcade(t, dt, W, H, ctx.pointer);
          else drawPipeline(t, W, H, ctx.pointer);
          shapes.flush(enc, hdr, cam, { clear: [0.004, 0.005, 0.012, 1], blend: ctx.example === 'arcade' ? 'additive' : 'alpha' });
        }

        // ---- 2. bright pass + downsample chain
        prefilterFx.draw(enc, mips[0], { src: hdr });
        for (let i = 1; i < levels; i++) downFx.draw(enc, mips[i], { src: mips[i - 1] });
        // ---- 3. upsample chain
        let wide = mips[levels - 1];
        for (let i = levels - 2; i >= 0; i--) {
          upFx.draw(enc, ups[i], { src: wide, base: mips[i] });
          wide = ups[i];
        }
        // ---- 4. composite
        const dbgLevel = viewIdx === 2 ? 0 : Math.max(0, Math.min(levels - 1, Math.round(p.mip) - 1));
        const canvasT = { view: ctx.target, format: gpu.format };
        compositeFx.draw(enc, canvasT, { hdr, bloom: wide, dbg: mips[dbgLevel] });

        // mip strip
        if (p.strip) {
          const th = Math.round(H * 0.11);
          const tw = Math.round((th * W) / H);
          const gap = Math.round(4 * ctx.dpr);
          const total = levels * (tw + gap);
          const scale = Math.min(1, (W * 0.62) / total);
          const w2 = Math.floor(tw * scale);
          const h2 = Math.floor(th * scale);
          let x = Math.round((W - levels * (w2 + gap)) / 2);
          const y = Math.round(44 * ctx.dpr);
          for (let i = 0; i < levels; i++) {
            thumbFx.draw(enc, canvasT, { src: mips[i] }, { clear: false, viewport: [x, y, w2, h2] });
            x += w2 + gap;
          }
        }
        if (ctx.example !== 'pipeline') for (const l of labels) l.style.display = 'none';
        readout.textContent = `${levels} levels: ${mips[0].width}×${mips[0].height} → ${mips[levels - 1].width}×${mips[levels - 1].height} · ${p.filter === 'dual' ? 'dual filter' : '13-tap + tent'}`;
      },
    };
  },
};

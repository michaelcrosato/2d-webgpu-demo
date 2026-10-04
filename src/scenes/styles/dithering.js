import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import {
  paletteUniforms, PALETTE_UNIFORMS, PALETTE_WGSL, setTag, placeTag, makeSplit, gamePass, GAME_INCLUDES,
  blueNoiseTexture, floydSteinberg,
} from './_shared-a.js';

// Dithering: trade color depth for spatial patterns. Four views:
//   compare – 8 methods side by side on the same grey test card (Floyd–Steinberg computed on the CPU)
//   obra    – 1-bit two-tone look with edge lines (Return of the Obra Dinn / Playdate)
//   retro   – limited palettes (CGA, EGA…) with ordered dithering between the two nearest colors
//   banding – fixing 8-bit gradient banding with noise before quantization (split view)

const CARD_W = 128;
const CARD_H = 96;
const splitPos = makeSplit(0.5);

// Grey test card drawn with Canvas2D. R = grey value, G = the same image Floyd–Steinberg dithered
// on the CPU (sequential, so it is precomputed once), B = unused.
let cardCanvas = null;
function makeCard() {
  if (cardCanvas) return cardCanvas;
  const c = makeCanvas(CARD_W, CARD_H);
  const g = c.getContext('2d');
  // backdrop: dark top → light horizon, darker ground
  const bg = g.createLinearGradient(0, 0, 0, CARD_H);
  bg.addColorStop(0, '#101010');
  bg.addColorStop(0.7, '#a8a8a8');
  bg.addColorStop(0.7, '#444444');
  bg.addColorStop(1, '#1c1c1c');
  g.fillStyle = bg;
  g.fillRect(0, 0, CARD_W, CARD_H);
  // thin rays (fine detail)
  g.save();
  g.translate(CARD_W * 0.84, CARD_H * 0.24);
  for (let i = 0; i < 12; i++) {
    g.rotate(Math.PI / 12);
    g.fillStyle = i % 2 ? '#e0e0e0' : '#808080';
    g.fillRect(0, -0.5, 17, 1);
  }
  g.restore();
  // lit sphere
  const sx = CARD_W * 0.38;
  const sy = CARD_H * 0.42;
  const r = CARD_H * 0.29;
  const sp = g.createRadialGradient(sx - r * 0.38, sy - r * 0.42, r * 0.05, sx, sy, r);
  sp.addColorStop(0, '#ffffff');
  sp.addColorStop(0.35, '#b4b4b4');
  sp.addColorStop(0.8, '#3a3a3a');
  sp.addColorStop(1, '#0c0c0c');
  g.fillStyle = sp;
  g.beginPath();
  g.arc(sx, sy, r, 0, Math.PI * 2);
  g.fill();
  // shadow on the ground
  g.fillStyle = 'rgba(0,0,0,0.5)';
  g.beginPath();
  g.ellipse(sx + 3, CARD_H * 0.75, r * 0.95, 3.5, 0, 0, Math.PI * 2);
  g.fill();
  // grey ramp
  const ramp = g.createLinearGradient(4, 0, CARD_W - 4, 0);
  ramp.addColorStop(0, '#000000');
  ramp.addColorStop(1, '#ffffff');
  g.fillStyle = ramp;
  g.fillRect(4, CARD_H - 13, CARD_W - 8, 9);
  // a small cube (three flat tones)
  const x0 = Math.round(CARD_W * 0.72);
  const y0 = Math.round(CARD_H * 0.52);
  const poly = (pts, col) => {
    g.fillStyle = col;
    g.beginPath();
    pts.forEach(([x, y], i) => (i ? g.lineTo(x0 + x, y0 + y) : g.moveTo(x0 + x, y0 + y)));
    g.closePath();
    g.fill();
  };
  poly([[0, 0], [16, 0], [16, 14], [0, 14]], '#c8c8c8');
  poly([[0, 0], [5, -5], [21, -5], [16, 0]], '#f4f4f4');
  poly([[16, 0], [21, -5], [21, 9], [16, 14]], '#6c6c6c');

  const img = g.getImageData(0, 0, CARD_W, CARD_H);
  const grey = new Float32Array(CARD_W * CARD_H);
  for (let i = 0; i < grey.length; i++) grey[i] = img.data[i * 4] / 255;
  const fs = floydSteinberg(grey.slice(), CARD_W, CARD_H);
  for (let i = 0; i < grey.length; i++) {
    img.data[i * 4 + 1] = fs[i] * 255;
    img.data[i * 4 + 2] = 0;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  cardCanvas = c;
  return c;
}

const TONES = {
  obra: ['#333319', '#e5ffff'],
  playdate: ['#312e28', '#b1aea7'],
  mono: ['#000000', '#ffffff'],
  paper: ['#1d1a2f', '#f4ecd8'],
  amber: ['#1c0f02', '#ffb52e'],
  phosphor: ['#03140a', '#5cff8a'],
};

const METHOD_NAMES = ['Threshold', 'White noise', 'Bayer 2×2', 'Bayer 4×4', 'Bayer 8×8', 'IGN (blue-ish)', 'Blue noise', 'Halftone'];
const PANEL_METHODS = [0, 1, 2, 3, 4, 6, 7, 8];
const PANEL_LABELS = ['Threshold', 'White noise', 'Bayer 2×2', 'Bayer 4×4', 'Bayer 8×8', 'Blue noise', 'Halftone', 'Floyd–Steinberg (CPU)'];

const COMMON = /* wgsl */ `
${PALETTE_WGSL}
// Threshold in [0,1) for dither-space pixel cp. A pixel is "light" when its value > threshold.
fn halftoneThr(cp: vec2f) -> f32 {
  let q = vec2f(cp.x + cp.y, cp.y - cp.x) * (0.70710678 / 6.0);   // 45° screen, 6px cells
  return 0.5 + 0.25 * (cos(TAU * q.x) + cos(TAU * q.y));         // "spot function": 1 at dot centers
}
fn methodThreshold(m: f32, cp: vec2f) -> f32 {
  if (m < 0.5) { return 0.5; }
  if (m < 1.5) { return hash21(cp + vec2f(u.seed * 17.0, u.seed * 31.0)); }
  if (m < 2.5) { return (bayer2(cp) * 4.0 + 0.5) / 4.0; }
  if (m < 3.5) { return (bayer4(cp) * 16.0 + 0.5) / 16.0; }
  if (m < 4.5) { return (bayer8(cp) * 64.0 + 0.5) / 64.0; }
  if (m < 5.5) { return ign(cp + vec2f(5.588238 * u.seed)); }
  if (m < 6.5) { return fract(LOADW(blue, vec2i(cp)).r + u.seed * 0.618034); }
  return halftoneThr(cp);
}
fn lumaAt(cell: vec2f) -> f32 { return luma(TEX(scene, (cell + 0.5) * u.ps / u.resolution).rgb); }
fn tone(g: f32) -> f32 { return clamp((g - 0.5) * u.contrast + 0.5 + u.brightness, 0.0, 1.0); }
fn gameAt(cell: vec2f) -> vec3f {
  let c = (cell + 0.5) * u.ps / u.resolution;
  let o = 0.25 * u.ps / u.resolution;
  return 0.25 * (TEX(scene, c + vec2f(-o.x, -o.y)).rgb + TEX(scene, c + vec2f(o.x, -o.y)).rgb +
                 TEX(scene, c + vec2f(-o.x, o.y)).rgb + TEX(scene, c + vec2f(o.x, o.y)).rgb);
}
`;

export default shaderScene({
  interaction: '',
  examples: [
    {
      id: 'compare',
      label: '8 methods compared',
      kind: 'Comparison',
      hint: '',
      note: 'One grey picture, two colors, eight ways to decide which pixels are ink. Everything except the last panel is computed per pixel on the GPU, independently. Floyd–Steinberg was computed once on the CPU because each pixel depends on the previous one.',
      params: { ps: 2, tones: 'paper', animate: false },
    },
    {
      id: 'obra',
      label: '1-bit adventure',
      kind: 'In a game',
      hint: '',
      note: 'The <i>Return of the Obra Dinn</i> / Playdate look: the game in two colors. Tones are first posterized into a few steps so each region gets one clean pattern, then dithered, and edge lines are drawn on top so shapes stay readable. Try the Playdate tones and blue noise.',
      params: { ps: 2, method: 4, tones: 'obra', steps: 5, contrast: 1.5, brightness: -0.04, edges: 0.6, animate: false },
    },
    {
      id: 'retro',
      label: 'Retro palettes',
      kind: 'Classic',
      hint: '',
      note: 'DOS-era CGA had just 4 colors — artists (and converters) leaned on ordered dithering to suggest more. Each fat pixel picks between its <b>two nearest</b> palette colors using a Bayer threshold.',
      params: { ps: 3, method: 3, palette: 'cga', contrast: 1.3, brightness: -0.08, animate: false },
    },
    {
      id: 'banding',
      label: 'Fixing banding',
      kind: 'Real life',
      hint: 'Drag to move the divider.',
      note: 'A dark, smooth gradient stored with too few levels shows <b>bands</b> (left). Adding a tiny bit of noise <i>before</i> rounding (right) breaks the bands up — the eye averages the noise away. Games do this for skies, fog, vignettes and lighting, even at 8 bits.',
      params: { ps: 1, method: 5, bits: 5, animate: true },
    },
  ],
  controls: [
    {
      type: 'select', key: 'method', label: 'Dither method', value: 6, showFor: ['obra', 'retro', 'banding'],
      options: METHOD_NAMES.map((label, value) => ({ value, label })),
      help: 'Where each pixel’s threshold comes from.',
    },
    { type: 'slider', key: 'ps', label: 'Pixel size', min: 1, max: 6, step: 1, value: 2, help: 'Magnify the pattern: one dither pixel = N×N screen pixels.' },
    { type: 'slider', key: 'contrast', label: 'Contrast', min: 0.5, max: 2.5, step: 0.01, value: 1.2, showFor: ['obra', 'retro'] },
    { type: 'slider', key: 'brightness', label: 'Brightness', min: -0.4, max: 0.4, step: 0.01, value: 0, showFor: ['obra', 'retro'] },
    { type: 'slider', key: 'steps', label: 'Tone steps', min: 2, max: 16, step: 1, value: 5, showFor: ['obra'], help: 'Posterize before dithering: few steps = flat regions with one clean pattern each (16 = off).' },
    { type: 'slider', key: 'edges', label: 'Edge lines', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['obra'], help: 'Sobel edges drawn in the opposite tone keep silhouettes readable in 1-bit.' },
    {
      type: 'select', key: 'tones', label: 'Two-tone palette', value: 'paper', showFor: ['compare', 'obra'],
      options: [
        { value: 'obra', label: 'Obra Dinn (Mac)' },
        { value: 'playdate', label: 'Playdate' },
        { value: 'mono', label: 'Black & white' },
        { value: 'paper', label: 'Ink on paper' },
        { value: 'amber', label: 'Amber monitor' },
        { value: 'phosphor', label: 'Green phosphor' },
      ],
    },
    {
      type: 'select', key: 'palette', label: 'Palette', value: 'cga', showFor: ['retro'],
      options: [
        { value: 'cga', label: 'CGA (4 colors)' },
        { value: 'ega', label: 'EGA (16)' },
        { value: 'gameboy', label: 'Game Boy (4)' },
        { value: 'pico8', label: 'PICO-8 (16)' },
        { value: 'vaporwave', label: 'Vaporwave (6)' },
        { value: 'sepia', label: 'Sepia (5)' },
        { value: 'sweetie16', label: 'Sweetie-16' },
      ],
    },
    { type: 'slider', key: 'bits', label: 'Bits per channel', min: 2, max: 8, step: 1, value: 5, showFor: ['banding'], help: '8 bits = 256 levels (a normal screen). Fewer bits = wider bands.' },
    { type: 'toggle', key: 'animate', label: 'Animate noise (temporal)', value: false, help: 'Re-roll noise every frame. At high frame rates (or with TAA) the eye averages it into smooth tones.' },
  ],
  uniforms: {
    ...PALETTE_UNIFORMS,
    method: 'f32', ps: 'f32', steps: 'f32', contrast: 'f32', brightness: 'f32', edges: 'f32', bits: 'f32', seed: 'f32', split: 'f32',
    ink: 'vec3f', paperCol: 'vec3f', cardScale: 'f32',
  },
  include: ['color', 'dither', ...GAME_INCLUDES],
  passes: [gamePass('scene')],
  textures: {
    card: { source: async () => makeCard(), filter: 'nearest' },
    blue: { source: async () => blueNoiseTexture(64), filter: 'nearest' },
  },
  bind(params, ctx) {
    const ex = ctx.example;
    const tones = TONES[params.tones] || TONES.paper;
    const P = paletteUniforms(params.palette || 'cga');
    const split = splitPos(ctx);
    const seed = params.animate ? (ctx.frame % 4096) + 1 : 0;
    // labels
    for (let i = 0; i < 8; i++) {
      const el = setTag(ctx, `p${i}`, PANEL_LABELS[i], 'font-size:11px', ex === 'compare');
      if (ex === 'compare') placeTag(el, (i % 4) * 25 + 12.5, i < 4 ? 'calc(50% - 30px)' : 'calc(100% - 30px)', '-50%');
    }
    const showSplit = ex === 'banding';
    const a = setTag(ctx, 'before', 'No dither', '', showSplit);
    const b = setTag(ctx, 'after', `${METHOD_NAMES[params.method | 0]} dither`, '', showSplit);
    if (showSplit) {
      placeTag(a, split * 100, '48px', 'calc(-100% - 10px)');
      placeTag(b, split * 100, '48px', '10px');
    }
    const levels = Math.pow(2, params.bits | 0);
    let info = '';
    if (ex === 'banding') info = `${params.bits | 0} bits = ${levels} levels per channel`;
    else if (ex === 'obra') info = `2 colors · ${METHOD_NAMES[params.method | 0]}`;
    else if (ex === 'retro') info = `${P.palN} colors · ${METHOD_NAMES[params.method | 0]}`;
    setTag(ctx, 'info', info, 'right:8px;bottom:8px', !!info);
    // compare view: fit the card into a panel (pixel size = the largest integer zoom that fits, max = slider)
    const fit = Math.floor(Math.min(ctx.width / 4 / CARD_W, ctx.height / 2 / CARD_H));
    const cardScale = Math.max(1, Math.min(params.ps | 0, fit));
    return { ...P, ink: tones[0], paperCol: tones[1], split, seed, cardScale };
  },
  code: /* wgsl */ `
${COMMON}
fn compareView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let panelSize = res / vec2f(4.0, 2.0);
  let pid = floor(px / panelSize);
  let local = px - pid * panelSize;
  // centre the card in the panel (cropped if it doesn't fit)
  let cardPx = vec2f(${CARD_W}.0, ${CARD_H}.0) * u.cardScale;
  let off = floor((panelSize - cardPx) * 0.5);
  let cp = floor((local - off) / u.cardScale);
  var m = 0.0;
  let idx = i32(pid.x + pid.y * 4.0);
  var methods = array<f32, 8>(${PANEL_METHODS.map((v) => v.toFixed(1)).join(', ')});
  m = methods[idx];
  let texel = LOAD(card, vec2i(cp));
  var bit = 0.0;
  if (m > 7.5) { bit = step(0.5, texel.g); } else { bit = step(methodThreshold(m, cp), texel.r); }
  var c = mix(u.ink, u.paperCol, bit);
  // outside the card: dim backdrop
  let inside = step(0.0, cp.x) * step(0.0, cp.y) * step(cp.x, ${CARD_W - 1}.0) * step(cp.y, ${CARD_H - 1}.0);
  c = mix(mix(u.ink, u.paperCol, 0.12), c, inside);
  // panel borders
  let e = min(local, panelSize - local);
  c = mix(c, u.ink * 0.6, 1.0 - smoothstep(0.5, 1.5, min(e.x, e.y)));
  return c;
}

fn obraView(px: vec2f) -> vec3f {
  let cell = floor(px / u.ps);
  var g = tone(luma(gameAt(cell)));
  // posterize into a few tone steps first: each region then gets ONE clean pattern
  if (u.steps < 15.5) { g = floor(g * (u.steps - 1.0) + 0.5) / (u.steps - 1.0); }
  // Sobel on the luminance at dither resolution
  let l00 = lumaAt(cell + vec2f(-1.0, -1.0)); let l10 = lumaAt(cell + vec2f(0.0, -1.0)); let l20 = lumaAt(cell + vec2f(1.0, -1.0));
  let l01 = lumaAt(cell + vec2f(-1.0, 0.0));                                              let l21 = lumaAt(cell + vec2f(1.0, 0.0));
  let l02 = lumaAt(cell + vec2f(-1.0, 1.0)); let l12 = lumaAt(cell + vec2f(0.0, 1.0));  let l22 = lumaAt(cell + vec2f(1.0, 1.0));
  let gx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
  let gy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
  let edge = sqrt(gx * gx + gy * gy);
  var bit = step(methodThreshold(u.method, cell), g);
  if (u.edges > 0.0 && edge > mix(1.4, 0.15, u.edges)) {
    bit = 1.0 - step(0.42, g);   // line in the opposite tone of the area
  }
  return mix(u.ink, u.paperCol, bit);
}

fn retroView(px: vec2f) -> vec3f {
  let cell = floor(px / u.ps);
  let c = gameAt(cell);
  let graded = clamp((c - vec3f(0.5)) * u.contrast + vec3f(0.5 + u.brightness), vec3f(0.0), vec3f(1.0));
  return palDither(graded, methodThreshold(u.method, cell), 1.0);
}

// a smooth, dark night scene: the worst case for banding
fn bandScene(px: vec2f) -> vec3f {
  let res = u.resolution;
  let p = px / res.y;
  let asp = res.x / res.y;
  var c = mix(vec3f(0.02, 0.03, 0.09), vec3f(0.24, 0.13, 0.28), smoothstep(0.0, 0.68, p.y));
  let moon = vec2f(asp * 0.68, 0.26);
  let md = length(p - moon);
  c += vec3f(0.35, 0.32, 0.45) * exp(-md * 5.0) * 0.55;
  c = mix(c, vec3f(0.92, 0.9, 0.82), 1.0 - smoothstep(0.058, 0.062, md));
  // distant hills fading into fog
  let h1 = 0.66 + 0.04 * sin(p.x * 3.1 + 1.0) + 0.02 * sin(p.x * 7.3);
  let fogC = vec3f(0.19, 0.11, 0.24);
  c = mix(c, mix(vec3f(0.05, 0.035, 0.08), fogC, 0.55), smoothstep(h1, h1 + 0.004, p.y));
  let h2 = 0.78 + 0.03 * sin(p.x * 2.3 + 4.0);
  c = mix(c, mix(vec3f(0.02, 0.015, 0.035), fogC * 0.6, (1.0 - smoothstep(h2, 0.95, p.y)) * 0.5), smoothstep(h2, h2 + 0.004, p.y));
  // vignette
  let v = length((px / res - vec2f(0.5)) * vec2f(asp, 1.0));
  c *= 1.0 - smoothstep(0.35, 1.1, v) * 0.8;
  return c;
}

fn bandingView(px: vec2f) -> vec3f {
  let c = bandScene(px);
  let levels = exp2(u.bits) - 1.0;
  let cell = floor(px / u.ps);
  var thr = methodThreshold(u.method, cell);
  if (px.x < u.split * u.resolution.x) { thr = 0.5; }
  let q = floor(c * levels + vec3f(thr)) / levels;
  let line = 1.0 - smoothstep(1.0, 2.5, abs(px.x - u.split * u.resolution.x));
  return mix(q, vec3f(1.0), line * 0.85);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var c = vec3f(0.0);
  if (ex == 0) { c = compareView(px); }
  else if (ex == 1) { c = obraView(px); }
  else if (ex == 2) { c = retroView(px); }
  else { c = bandingView(px); }
  return vec4f(c, 1.0);
}`,
  about: {
    summary: 'Dithering fakes more shades than you have by mixing the few you do have in fine patterns. It is a retro art style, a print technique, and a practical fix for ugly gradient banding.',
    what: `<p>Four views: eight dithering methods side by side; the game rendered in pure <b>1-bit</b>; the game in tiny retro palettes; and a
      dark gradient whose <b>banding</b> disappears once noise is added before rounding.</p>`,
    how: `<ol>
      <li><b>Thresholding.</b> To turn a grey value (0–1) into ink or paper, compare it with a threshold: <code>bit = value &gt; threshold</code>.
        A constant 0.5 loses all midtones — everything becomes black or white blobs.</li>
      <li><b>Vary the threshold per pixel.</b> If the thresholds across an area are evenly spread between 0 and 1, then a 30% grey area turns
        about 30% of its pixels white. Where the thresholds come from is the whole difference between methods:
        <ul>
          <li><b>White noise</b> – random per pixel: correct on average, but clumpy and grainy.</li>
          <li><b>Bayer (ordered)</b> – a tiled matrix (2×2, 4×4, 8×8) that spreads thresholds as evenly as possible. Crisp, regular cross-hatch look — the classic retro style.</li>
          <li><b>Blue noise</b> – random-looking but with no clumps (only high frequencies). Looks organic and fine. Made offline with Ulichney’s <i>void-and-cluster</i> method into a small tiling texture; <b>IGN</b> (interleaved gradient noise) is a one-line formula with similar properties, popular in games for its speed.</li>
          <li><b>Halftone</b> – thresholds arranged as round “spots” on a rotated grid: dots that grow with darkness, like newspaper print (clustered-dot dithering).</li>
        </ul></li>
      <li><b>Error diffusion (Floyd–Steinberg).</b> Process pixels in order; round each one and push its <i>rounding error</i> onto the
        neighbours that come next (7/16 right, 3/16, 5/16 and 1/16 below). Beautiful, adaptive results — but pixel N needs the result of pixel N−1.</li>
      <li><b>Colors and palettes.</b> With a palette, find the two nearest colors and use the threshold to choose between them. For banding,
        add the threshold before rounding to the nearest of 2<sup>bits</sup> levels: <code>floor(c × levels + threshold) / levels</code>.</li>
    </ol>
    <p><b>Why ordered dithering is GPU-friendly:</b> every pixel’s answer depends only on its own value and its own position — perfect for
      millions of independent GPU threads. <b>Error diffusion isn’t:</b> it is a sequential chain from pixel to pixel (a data dependency), so it
      runs on the CPU or with clever, much more complex parallel variants. Blue noise gets close to its quality without the dependency.</p>`,
    uses: [
      { title: '1-bit games', text: 'Return of the Obra Dinn (custom Bayer + blue-noise dithering with stabilization), the Playdate console’s whole library, Minit, World of Horror.' },
      { title: 'Retro looks', text: 'DOS/CGA & EGA homages, Game Boy-style games, Lucas Pope’s Papers Please palette work, and countless pixel-art shaders.' },
      { title: 'Banding fixes', text: 'Modern engines (Unreal, Unity URP/HDRP, Godot) dither gradients, fog, vignettes and HDR→8-bit output with blue noise or IGN to avoid visible bands.' },
      { title: 'Transparency & fades', text: '“Screen-door” transparency and dithered LOD cross-fades (Mario Odyssey, Zelda: BotW) use ordered thresholds instead of alpha blending.' },
    ],
    try: [
      'On <b>8 methods compared</b>, look at the grey ramp at the bottom of each panel: threshold has two values, white noise is grainy, Bayer is a neat cross-hatch, blue noise is fine and even.',
      'Turn on <i>Animate noise</i> and squint: random/blue noise flicker into a smooth grey — temporal dithering.',
      'On <b>Fixing banding</b>, drag <i>Bits per channel</i> from 3 to 8. Even at 8 bits, the dark sky on the left has faint bands.',
      'On <b>1-bit adventure</b>, set <i>Edge lines</i> to 0 — shapes melt into the dither. Then switch to Bayer 8×8 for a crisper, more “Macintosh” look.',
      'On <b>Retro palettes</b>, compare CGA with Threshold vs Bayer 4×4: dithering invents colors CGA never had.',
    ],
    ask: [
      '1-bit dithered rendering like Return of the Obra Dinn',
      'blue-noise dithering texture (void-and-cluster) for gradients',
      'Bayer ordered dithering to a limited palette',
      'add IGN / blue-noise dither before 8-bit output to remove banding',
      'halftone dots post-process',
      'screen-door (dithered) transparency for fades',
    ],
    perf: `<p>Ordered and noise dithering cost one texture read or a few arithmetic operations per pixel — essentially free. Palette dithering
      costs a loop over the palette. Floyd–Steinberg on the CPU is fine for a small static image (this card is 49k pixels) but far too slow
      to run every frame at 1080p in JavaScript, and it isn’t stable frame to frame: one changed pixel ripples through everything after it.</p>`,
    api: `<p>Works the same in WebGL2 and WebGPU — it is pure per-pixel math. WebGPU compute shaders <i>can</i> run parallel error-diffusion
      variants (one workgroup per row band, with wavefront scheduling), but for real-time games blue noise gives similar quality for far less complexity.</p>`,
    code: [
      {
        title: 'Thresholds for every method (one function, all GPU-parallel)',
        lang: 'wgsl',
        src: `fn methodThreshold(m: f32, cp: vec2f) -> f32 {      // cp = integer pixel coordinate
  if (m < 0.5) { return 0.5; }                                   // plain threshold
  if (m < 1.5) { return hash21(cp + vec2f(u.seed * 17.0, u.seed * 31.0)); } // white noise
  if (m < 2.5) { return (bayer2(cp) * 4.0 + 0.5) / 4.0; }        // ordered 2×2
  if (m < 3.5) { return (bayer4(cp) * 16.0 + 0.5) / 16.0; }      // ordered 4×4
  if (m < 4.5) { return (bayer8(cp) * 64.0 + 0.5) / 64.0; }      // ordered 8×8
  if (m < 5.5) { return ign(cp + vec2f(5.588238 * u.seed)); }    // interleaved gradient noise
  if (m < 6.5) { return fract(LOADW(blue, vec2i(cp)).r + u.seed * 0.618034); } // blue noise tex
  return halftoneThr(cp);                                        // clustered dots
}
let bit = step(methodThreshold(u.method, cell), grey);           // 1 = paper, 0 = ink`,
      },
      {
        title: 'Banding fix: noise before quantization',
        lang: 'wgsl',
        src: `let levels = exp2(u.bits) - 1.0;                   // e.g. 255 for 8 bits
let thr = ign(px);                                   // 0..1, different for each pixel
let q = floor(c * levels + vec3f(thr)) / levels;     // instead of floor(c * levels + 0.5)`,
      },
      {
        title: 'Floyd–Steinberg (CPU): why it is sequential',
        lang: 'js',
        src: `for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
  const i = y * w + x;
  const v = grey[i] >= 0.5 ? 1 : 0;
  const err = grey[i] - v;          // the rounding error…
  grey[i + 1]     += err * 7 / 16;  // …is pushed onto pixels not processed yet,
  grey[i + w - 1] += err * 3 / 16;  // so pixel i+1 can't start until pixel i is done
  grey[i + w]     += err * 5 / 16;
  grey[i + w + 1] += err * 1 / 16;
  out[i] = v;
}`,
      },
    ],
    links: [
      { title: 'Lucas Pope — Obra Dinn dithering devlog', url: 'https://forums.tigsource.com/index.php?topic=40832.msg1363742#msg1363742', note: 'how the 1-bit look was stabilized' },
      { title: 'Ulichney’s void-and-cluster method', url: 'https://en.wikipedia.org/wiki/Ordered_dithering', note: 'ordered dithering & blue noise overview' },
    ],
  },
});

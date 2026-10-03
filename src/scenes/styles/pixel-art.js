import { shaderScene } from '../../core/shaderscene.js';
import { paletteUniforms, PALETTE_UNIFORMS, PALETTE_WGSL, setTag, placeTag, makeSplit, gamePass, GAME_INCLUDES } from './_shared-a.js';

// Pixel-art pipeline: (1) render into a LOW-RES target (pass "lo", one texel per virtual pixel),
// (2) quantize every texel to a fixed palette (+ optional ordered dithering, + 1px outline),
// (3) upscale to the screen with NEAREST filtering (+ optional LCD grid and a before/after split).

const splitPos = makeSplit(0.5);

const DITHER_NAMES = ['no dither', 'Bayer 2×2', 'Bayer 4×4', 'Bayer 8×8'];
const PAL_NAMES = { gameboy: 'Game Boy', pico8: 'PICO-8', nes: 'NES', sweetie16: 'Sweetie-16', cga: 'CGA', ega: 'EGA', mono: '1-bit', none: 'full color' };

/** Size of one virtual pixel in screen pixels. */
function pixelSize(params, ctx) {
  const h = Math.max(1, ctx.height);
  let ps = h / Math.max(16, params.vh);
  if (params.intScale) ps = Math.max(1, Math.round(ps));
  return Math.max(1, ps);
}

// The procedural game scene scrolls its ground layer by time * 0.12 screen-heights per second
// (see core/gamescene.js). A pixel-perfect camera samples the world on a grid that moves in
// whole virtual pixels, so we offset the sampling grid by the fractional part of that scroll.
const GAME_SCROLL_SPEED = 0.12;

const SHARED = /* wgsl */ `
${PALETTE_WGSL}
// Abstract test card: a lit sphere, a hue strip, a skin-tone ramp and a grey ramp.
fn testCard(spx: vec2f) -> vec3f {
  let res = u.resolution;
  let p = spx / res.y;
  let asp = res.x / res.y;
  var c = mix(vec3f(0.08, 0.10, 0.26), vec3f(0.42, 0.20, 0.42), smoothstep(0.0, 1.0, p.y));
  c += vec3f(0.9, 0.5, 0.3) * exp(-length(p - vec2f(asp * 0.35, 1.05)) * 3.0) * 0.35;
  // sphere
  let ctr = vec2f(asp * 0.3, 0.46);
  let r = 0.3;
  let q = (p - ctr) / r;
  let d2 = dot(q, q);
  let aa = 1.0 - smoothstep(1.0 - 4.0 / (r * res.y), 1.0, d2);
  if (aa > 0.0) {
    let nrm = vec3f(q.x, q.y, sqrt(max(1.0 - d2, 0.0)));
    let la = u.time * 0.5;
    let L = normalize(vec3f(cos(la) * 0.8, -0.6, 0.55 + 0.3 * sin(la)));
    let diff = max(dot(nrm, L), 0.0);
    let hv = normalize(L + vec3f(0.0, 0.0, 1.0));
    let spec = pow(max(dot(nrm, hv), 0.0), 36.0);
    let base = hsv2rgb(vec3f(fract(0.93 + 0.22 * (q.y * 0.5 + 0.5)), 0.72, 1.0));
    let lit = base * (0.1 + 0.9 * diff) + vec3f(spec * 0.85);
    c = mix(c, lit, aa);
  }
  // swatch bars on the right
  let x0 = asp * 0.62;
  let x1 = asp * 0.95;
  let t = clamp((p.x - x0) / (x1 - x0), 0.0, 1.0);
  let inX = step(x0, p.x) * step(p.x, x1);
  let b1 = inX * step(0.14, p.y) * step(p.y, 0.30);
  let b2 = inX * step(0.36, p.y) * step(p.y, 0.52);
  let b3 = inX * step(0.58, p.y) * step(p.y, 0.74);
  c = mix(c, hsv2rgb(vec3f(t, 0.85, 1.0)), b1);
  c = mix(c, mix(vec3f(0.18, 0.08, 0.05), vec3f(1.0, 0.86, 0.72), t), b2);
  c = mix(c, vec3f(t), b3);
  // a small ball drifting slowly: watch it crawl from pixel to pixel
  let bx = x0 + (x1 - x0) * (0.5 + 0.5 * sin(u.time * 0.35));
  let bd = length(p - vec2f(bx, 0.84)) - 0.05;
  c = mix(c, vec3f(1.0, 0.85, 0.25), 1.0 - smoothstep(-1.0 / res.y, 1.0 / res.y, bd));
  return c;
}
fn srcAt(spx: vec2f) -> vec3f {
  if (u.source > 0.5) { return testCard(spx); }
  return TEX(scene, spx / u.resolution).rgb;
}
`;

export default shaderScene({
  interaction: 'Drag on the canvas to move the before/after divider (turn on “Before/after split”).',
  examples: [
    {
      id: 'gameboy',
      label: 'Game Boy',
      kind: 'In a game',
      note: 'The platformer squeezed into a 160×144-ish screen and the four greens of the 1989 Game Boy. Matching is done by <i>lightness only</i>, and a 4×4 Bayer pattern fakes the in-between shades. The faint grid mimics the LCD.',
      params: { vh: 144, palette: 'gameboy', dither: 2, ditherAmt: 0.55, contrast: 1.2, outline: false, grid: true, source: 0, splitOn: false, downs: 1, snap: true },
    },
    {
      id: 'pico8',
      label: 'PICO-8',
      kind: 'Classic',
      note: 'The PICO-8 fantasy console: 128×128 pixels and a fixed, hand-picked 16-color palette. No dithering — flat colors and a 1px dark outline on the darker side of every strong edge, the way pixel artists separate shapes.',
      params: { vh: 128, palette: 'pico8', dither: 0, ditherAmt: 0.5, contrast: 1.1, outline: true, grid: false, source: 0, splitOn: false, downs: 1, snap: true },
    },
    {
      id: 'nes',
      label: 'NES',
      kind: 'Classic',
      note: 'A 256×240 picture using the NES master palette (54 colors). The real console also limited each 16×16 area to a 4-color sub-palette — palette quantization is the part shaders usually emulate.',
      params: { vh: 240, palette: 'nes', dither: 1, ditherAmt: 0.35, contrast: 1.05, outline: false, grid: false, source: 0, splitOn: false, downs: 1, snap: true },
    },
    {
      id: 'custom',
      label: 'Custom (split view)',
      kind: 'Abstract',
      note: 'A test card — smooth sphere, hue strip, skin-tone and grey ramps — split into <b>original</b> (left) and <b>pixel-art pipeline</b> (right). Drag to move the divider; change resolution, palette and dithering to see what each step does.',
      params: { vh: 96, palette: 'sweetie16', dither: 3, ditherAmt: 0.6, contrast: 1.0, outline: false, grid: false, source: 1, splitOn: true, downs: 1, snap: false },
    },
  ],
  controls: [
    { type: 'heading', label: 'Resolution' },
    { type: 'slider', key: 'vh', label: 'Virtual height (pixels)', min: 48, max: 360, step: 1, value: 144, help: 'How many “fat” pixels tall the screen is. Game Boy 144, PICO-8 128, NES/SNES 224–240.' },
    { type: 'toggle', key: 'intScale', label: 'Integer pixel size', value: true, help: 'Round each fat pixel to a whole number of screen pixels — otherwise some are 1px wider than others.' },
    {
      type: 'select', key: 'downs', label: 'Downsampling', value: 1,
      options: [{ value: 0, label: 'Point sample (crunchy, flickers)' }, { value: 1, label: 'Average 4 taps (smoother)' }],
      help: 'How a fat pixel picks its color from the detailed source image.',
    },
    { type: 'toggle', key: 'snap', label: 'Pixel-perfect camera', value: true, help: 'Snap scrolling to whole fat pixels. Off = sub-pixel camera: edges “crawl” and shimmer as the ground scrolls.' },
    { type: 'heading', label: 'Color' },
    {
      type: 'select', key: 'palette', label: 'Palette', value: 'gameboy',
      options: [
        { value: 'gameboy', label: 'Game Boy (4 greens)' },
        { value: 'pico8', label: 'PICO-8 (16)' },
        { value: 'nes', label: 'NES (54)' },
        { value: 'sweetie16', label: 'Sweetie-16' },
        { value: 'cga', label: 'CGA (4)' },
        { value: 'ega', label: 'EGA (16)' },
        { value: 'mono', label: '1-bit black & white' },
        { value: 'none', label: 'None (full color)' },
      ],
    },
    {
      type: 'select', key: 'dither', label: 'Ordered dithering', value: 2,
      options: [{ value: 0, label: 'Off' }, { value: 1, label: 'Bayer 2×2' }, { value: 2, label: 'Bayer 4×4' }, { value: 3, label: 'Bayer 8×8' }],
      help: 'Checkerboard-like patterns that fake in-between colors.',
    },
    { type: 'slider', key: 'ditherAmt', label: 'Dither strength', min: 0, max: 1, step: 0.01, value: 0.5 },
    { type: 'slider', key: 'contrast', label: 'Pre-contrast', min: 0.5, max: 2, step: 0.01, value: 1.1, help: 'Boost contrast before quantizing so few colors still read well.' },
    { type: 'heading', label: 'Finish' },
    { type: 'toggle', key: 'outline', label: '1px dark outlines', value: false, help: 'Edge pass at low resolution: the darker side of strong edges becomes the darkest palette color.' },
    { type: 'toggle', key: 'grid', label: 'LCD pixel grid', value: false },
    { type: 'toggle', key: 'splitOn', label: 'Before/after split', value: false },
    {
      type: 'select', key: 'source', label: 'Source image', value: 1, showFor: ['custom'],
      options: [{ value: 0, label: 'Game scene' }, { value: 1, label: 'Test card (abstract)' }],
    },
  ],
  uniforms: {
    ...PALETTE_UNIFORMS,
    vh: 'f32', ps: 'f32', downs: 'f32', snapOff: 'f32', dither: 'f32', ditherAmt: 'f32', contrast: 'f32',
    outline: 'f32', grid: 'f32', splitOn: 'f32', split: 'f32', source: 'f32', quant: 'f32',
    dark: 'vec3f', gapCol: 'vec3f',
  },
  include: ['color', 'dither', ...GAME_INCLUDES],
  resetOn: ['vh', 'intScale', 'downs', 'snap', 'palette', 'dither', 'ditherAmt', 'contrast', 'outline', 'source'],
  passes: [
    gamePass('scene'),
    {
      // One texel per virtual ("fat") pixel. This IS the low-resolution render target.
      name: 'lo',
      format: 'rgba8unorm',
      size: (params, ctx) => {
        const ps = pixelSize(params, ctx);
        return [Math.ceil(ctx.width / ps), Math.ceil(ctx.height / ps)];
      },
      code: /* wgsl */ `
${SHARED}
fn cellColor(cell: vec2f) -> vec3f {
  let base = (cell + 0.5) * u.ps - vec2f(u.snapOff, 0.0);
  var c: vec3f;
  if (u.downs > 0.5) {
    let o = u.ps * 0.25;
    c = 0.25 * (srcAt(base + vec2f(-o, -o)) + srcAt(base + vec2f(o, -o)) + srcAt(base + vec2f(-o, o)) + srcAt(base + vec2f(o, o)));
  } else {
    c = srcAt(base);
  }
  return clamp((c - vec3f(0.5)) * u.contrast + vec3f(0.5), vec3f(0.0), vec3f(1.0));
}
fn orderedThreshold(cell: vec2f) -> f32 {
  if (u.dither < 0.5) { return 0.5; }
  if (u.dither < 1.5) { return (bayer2(cell) * 4.0 + 0.5) / 4.0; }
  if (u.dither < 2.5) { return (bayer4(cell) * 16.0 + 0.5) / 16.0; }
  return (bayer8(cell) * 64.0 + 0.5) / 64.0;
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let cell = floor(px);
  var c = cellColor(cell);
  // outline: is a neighbour much brighter than me? then I'm the dark side of an edge
  if (u.outline > 0.5) {
    let l0 = luma(c);
    let lmax = max(max(luma(cellColor(cell + vec2f(1.0, 0.0))), luma(cellColor(cell - vec2f(1.0, 0.0)))),
                   max(luma(cellColor(cell + vec2f(0.0, 1.0))), luma(cellColor(cell - vec2f(0.0, 1.0)))));
    if (lmax - l0 > 0.16) { return vec4f(u.dark, 1.0); }
  }
  if (u.quant < 0.5) { return vec4f(c, 1.0); }
  let thr = orderedThreshold(cell);
  let jitter = (thr - 0.5) * u.ditherAmt * u.spread * 2.0;
  return vec4f(palNearest(c + vec3f(jitter)), 1.0);
}`,
    },
  ],
  bind(params, ctx) {
    const ps = pixelSize(params, ctx);
    const palName = params.palette;
    const P = paletteUniforms(palName === 'none' ? 'pico8' : palName);
    const vw = Math.ceil(ctx.width / ps);
    const vhh = Math.ceil(ctx.height / ps);
    const split = splitPos(ctx);
    const scrollPx = ctx.time * GAME_SCROLL_SPEED * ctx.height; // ground scroll in screen px
    const snapOff = params.snap ? ((scrollPx / ps) % 1) * ps : 0;
    // readouts
    const colors = palName === 'none' ? 'full color' : `${P.palN} colors (${PAL_NAMES[palName] || palName})`;
    setTag(ctx, 'info', `${vw}×${vhh} virtual px · ×${ps.toFixed(ps % 1 ? 2 : 0)} · ${colors} · ${DITHER_NAMES[params.dither | 0]}`, 'right:8px;bottom:8px');
    const before = setTag(ctx, 'before', 'Original', 'top:10px', !!params.splitOn);
    const after = setTag(ctx, 'after', 'Pixel art', 'top:10px', !!params.splitOn);
    if (params.splitOn) {
      placeTag(before, split * 100, 2, 'calc(-100% - 10px)');
      placeTag(after, split * 100, 2, '10px');
    }
    return {
      ...P,
      ps,
      snapOff,
      split,
      quant: palName === 'none' ? 0 : 1,
      dark: P.dark,
      gapCol: palName === 'gameboy' ? P.light : [0, 0, 0],
    };
  },
  code: /* wgsl */ `
${SHARED}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let cell = floor(px / u.ps);
  var c = LOAD(lo, vec2i(cell)).rgb;
  if (u.grid > 0.5 && u.ps >= 3.0) {
    // LCD look: thin gaps between pixels
    let f = fract(px / u.ps) * u.ps;
    let e = min(min(f.x, u.ps - f.x), min(f.y, u.ps - f.y));
    c = mix(c, u.gapCol, (1.0 - smoothstep(0.0, 1.0, e)) * 0.45);
  }
  if (u.splitOn > 0.5) {
    let sx = u.split * u.resolution.x;
    if (px.x < sx) { c = srcAt(px); }
    let line = 1.0 - smoothstep(1.0, 2.5, abs(px.x - sx));
    c = mix(c, vec3f(1.0), line);
  }
  return vec4f(c, 1.0);
}`,
  about: {
    summary: 'Real pixel art is drawn at a tiny resolution and blown up with hard edges. A GPU pipeline can give any scene that look: render small, limit the colors, dither, outline, scale up crisply.',
    what: `<p>The animated platformer (or a test card) pushed through a <b>pixel-art pipeline</b>: it is rendered into a small texture
      (e.g. 160×144), every pixel is snapped to the nearest color of a classic palette, optionally <b>dithered</b> and <b>outlined</b>,
      and the result is scaled up to your screen with <i>nearest-neighbour</i> filtering so each “fat pixel” stays a perfect square.</p>`,
    how: `<ol>
      <li><b>Low-res render target.</b> Instead of drawing to the screen, draw into a texture with one texel per virtual pixel
        (here the pass <code>lo</code>, sized <code>screen ÷ pixel size</code>). Everything after this runs on very few pixels — it is cheap.</li>
      <li><b>Downsampling.</b> Our source is a detailed image, so each fat pixel must pick a color: <i>point sampling</i> takes the
        center (sharp but thin details flicker), <i>averaging</i> a few taps is smoother. A native pixel-art game simply draws its sprites at this size.</li>
      <li><b>Palette quantization.</b> For each texel, loop over the palette (up to 64 colors in a uniform array) and keep the closest one.
        Distance is measured in <b>OKLab</b>, a perceptual color space, so “closest” matches what your eye thinks. Single-hue palettes like
        the Game Boy’s compare <i>lightness only</i>.</li>
      <li><b>Ordered dithering.</b> Before the search, nudge the color up or down by a value from a <b>Bayer matrix</b> (a fixed tiled
        threshold pattern). Neighbouring pixels then round to different palette colors and your eye blends them into in-between shades.</li>
      <li><b>Outline pass.</b> Compare each texel with its 4 neighbours; if one is much brighter, this texel sits on the dark side of an
        edge and becomes the darkest palette color — a 1-pixel “selout” style outline.</li>
      <li><b>Nearest upscale.</b> The final pass reads the low-res texture with <code>LOAD(lo, floor(px / pixelSize))</code>:
        no filtering, so edges stay razor sharp. Use <b>integer</b> scale factors or some fat pixels end up 1px wider than others.</li>
    </ol>
    <p><b>Pixel-perfect vs sub-pixel camera.</b> If the camera can sit <i>between</i> fat pixels, every frame re-samples the world at a
      different offset and edges “crawl” or shimmer. Pixel-perfect games snap the camera (and sprites) to whole virtual pixels — watch the
      ground tiles with <i>Pixel-perfect camera</i> on vs off. Layers moving at other speeds (parallax) need their own snapping, which is why the
      far hills still crawl a little here. Some modern games deliberately render at high resolution and only snap sprites, giving smooth
      sub-pixel scrolling with pixel-art sprites.</p>`,
    uses: [
      { title: 'Retro games', text: 'Shovel Knight (NES palette rules), Celeste (320×180 render target), Stardew Valley, Undertale and Dead Cells all render small and upscale with nearest filtering.' },
      { title: 'Fantasy consoles', text: 'PICO-8, TIC-80 and Pyxel enforce a fixed resolution and palette — limits that give games a coherent look.' },
      { title: 'Pixelated 3D', text: 'A Short Hike and Eastward-style “HD-2D-ish” looks render 3D or high-res scenes into a low-res target and quantize the colors.' },
      { title: 'Retro modes & filters', text: '“Game Boy mode” cheats, photo booths and pixelate transitions are this exact pipeline with a different palette.' },
    ],
    try: [
      'On <b>Custom</b>, set <i>Ordered dithering</i> to Off: the sphere breaks into flat bands. Turn Bayer 8×8 back on and the bands dissolve into patterns.',
      'Turn <i>Pixel-perfect camera</i> off on <b>Game Boy</b> and watch the ground tiles: their edges wobble every frame.',
      'Switch <i>Downsampling</i> to point sampling — the fireflies and grass blades blink in and out.',
      'Turn off <i>Integer pixel size</i> and pick a virtual height like 150: some columns of fat pixels become wider than others.',
      'Try the NES palette with 360 lines vs 64 lines: resolution matters as much as the palette for the “retro” feel.',
    ],
    ask: [
      'render the game into a 320×180 render target and upscale it with nearest filtering',
      'pixel-perfect camera that snaps to whole pixels',
      'palette quantization to the PICO-8 / Game Boy palette in a post-process shader',
      'Bayer ordered dithering between palette colors',
      'a 1-pixel selective outline pass at low resolution',
      'an LCD grid overlay for a Game Boy mode',
    ],
    perf: `<p>Very cheap — and it makes everything else cheaper: the heavy work (quantization with up to 54 palette comparisons per pixel,
      outlines) runs on the <i>low-res</i> texture, e.g. 46,000 texels instead of 2 million screen pixels. The upscale is one texel fetch per
      screen pixel. A native pixel-art renderer also draws its sprites at the small size, so fill-rate is tiny.</p>`,
    api: `<p>Identical in WebGPU and WebGL2: a render-to-texture pass plus a full-screen pass. In WebGPU the palette is an
      <code>array&lt;vec4f, 64&gt;</code> in a uniform buffer; in WebGL2 it is the same data in a <code>std140</code> uniform block.
      With WebGPU you could also run the quantization in a compute shader and build a color histogram to auto-pick a palette.</p>`,
    code: [
      {
        title: 'Palette quantization + ordered dither (low-res pass)',
        lang: 'wgsl',
        src: `fn palNearest(c: vec3f) -> vec3f {
  let q = linearToOklab(srgbToLinear(c));     // perceptual space
  var best = 1.0e9;
  var bestC = vec3f(0.0);
  for (var i = 0; i < 64; i++) {
    if (f32(i) >= u.palN) { break; }
    let e = q - u.lab[i].xyz;                   // palette pre-converted on the CPU
    let dd = e.x * e.x + u.chromaW * (e.y * e.y + e.z * e.z);
    if (dd < best) { best = dd; bestC = u.pal[i].xyz; }
  }
  return bestC;
}
// per low-res texel:
let thr = (bayer4(cell) * 16.0 + 0.5) / 16.0;           // 0..1, tiled 4×4 pattern
let jitter = (thr - 0.5) * u.ditherAmt * u.spread * 2.0;
return vec4f(palNearest(c + vec3f(jitter)), 1.0);`,
      },
      {
        title: 'Nearest-neighbour upscale (final pass)',
        lang: 'wgsl',
        src: `fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let cell = floor(px / u.ps);              // which fat pixel am I in?
  return vec4f(LOAD(lo, vec2i(cell)).rgb, 1.0);  // exact texel, no filtering
}`,
      },
    ],
    links: [
      { title: 'Lospec palette list', url: 'https://lospec.com/palette-list', note: 'thousands of pixel-art palettes' },
      { title: 'PICO-8', url: 'https://www.lexaloffle.com/pico-8.php', note: 'the 128×128, 16-color fantasy console' },
    ],
  },
});

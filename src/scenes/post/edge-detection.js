import { shaderScene } from '../../core/shaderscene.js';
import { overlays, compareSplit, SPLIT_WGSL, gamePass, withGameIncludes } from './_shared.js';

// Convolution: every output pixel is a weighted sum of its 3×3 neighbourhood. Change the 9 weights and
// the same loop blurs, sharpens, embosses or finds edges. Sobel = two such kernels (x & y gradient).

const st = { split: 0.5 };
const ov = overlays();

// [weights (row-major), divisor, bias]
const KERNELS = {
  identity: { w: [0, 0, 0, 0, 1, 0, 0, 0, 0], div: 1, bias: 0, label: 'Identity' },
  box: { w: [1, 1, 1, 1, 1, 1, 1, 1, 1], div: 9, bias: 0, label: 'Box blur' },
  gauss: { w: [1, 2, 1, 2, 4, 2, 1, 2, 1], div: 16, bias: 0, label: 'Gaussian blur' },
  sharpen: { w: [0, -1, 0, -1, 5, -1, 0, -1, 0], div: 1, bias: 0, label: 'Sharpen' },
  emboss: { w: [-2, -1, 0, -1, 1, 1, 0, 1, 2], div: 1, bias: 0, label: 'Emboss' },
  laplacian: { w: [0, -1, 0, -1, 4, -1, 0, -1, 0], div: 1, bias: 0, abs: true, label: 'Laplacian (edges)' },
  outline: { w: [-1, -1, -1, -1, 8, -1, -1, -1, -1], div: 1, bias: 0, abs: true, label: 'Outline' },
  sobelx: { w: [-1, 0, 1, -2, 0, 2, -1, 0, 1], div: 1, bias: 0.5, label: 'Sobel X (vertical edges)' },
  sobely: { w: [-1, -2, -1, 0, 0, 0, 1, 2, 1], div: 1, bias: 0.5, label: 'Sobel Y (horizontal edges)' },
};

function kernelTable(k) {
  const cells = k.w
    .map((v, i) => {
      const col = v > 0 ? '#7dffa0' : v < 0 ? '#ff8a8a' : '#8b93a7';
      return `<span style="display:inline-block;width:2.4em;text-align:right;color:${col}">${v}</span>${i % 3 === 2 ? '<br>' : ''}`;
    })
    .join('');
  const extra = [k.div !== 1 ? `÷ ${k.div}` : '', k.bias ? `+ ${k.bias}` : '', k.abs ? 'abs()' : ''].filter(Boolean).join(' ');
  return `<div style="opacity:.75;margin-bottom:3px">${k.label}</div>${cells}${extra ? `<div style="opacity:.75;margin-top:3px">${extra}</div>` : ''}`;
}

const IMAGE = /* wgsl */ `
${SPLIT_WGSL}

fn px3(p: vec2i, dx: i32, dy: i32) -> vec3f { return LOAD(game, p + vec2i(dx, dy) * i32(u.spread)).rgb; }

// generic 3×3 convolution with the kernel from the uniforms (rows k0, k1, k2)
fn convolve(p: vec2i) -> vec3f {
  var s: vec3f = vec3f(0.0);
  s += px3(p, -1, -1) * u.k0.x + px3(p, 0, -1) * u.k0.y + px3(p, 1, -1) * u.k0.z;
  s += px3(p, -1, 0) * u.k1.x + px3(p, 0, 0) * u.k1.y + px3(p, 1, 0) * u.k1.z;
  s += px3(p, -1, 1) * u.k2.x + px3(p, 0, 1) * u.k2.y + px3(p, 1, 1) * u.k2.z;
  s = s / u.kDiv + vec3f(u.kBias);
  if (u.kAbs > 0.5) { s = abs(s) * 2.0; }
  return clamp(s, vec3f(0.0), vec3f(1.0));
}

// Sobel on luminance at a given sample distance. Returns (gx, gy).
fn sobel(uv: vec2f, d: f32) -> vec2f {
  let o: vec2f = d / u.resolution;
  let tl: f32 = luma(TEX(game, uv + vec2f(-o.x, -o.y)).rgb);
  let tc: f32 = luma(TEX(game, uv + vec2f(0.0, -o.y)).rgb);
  let tr: f32 = luma(TEX(game, uv + vec2f(o.x, -o.y)).rgb);
  let ml: f32 = luma(TEX(game, uv + vec2f(-o.x, 0.0)).rgb);
  let mr: f32 = luma(TEX(game, uv + vec2f(o.x, 0.0)).rgb);
  let bl: f32 = luma(TEX(game, uv + vec2f(-o.x, o.y)).rgb);
  let bc: f32 = luma(TEX(game, uv + vec2f(0.0, o.y)).rgb);
  let br: f32 = luma(TEX(game, uv + vec2f(o.x, o.y)).rgb);
  let gx: f32 = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  let gy: f32 = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
  return vec2f(gx, gy);
}

// Sobel on all three color channels (catches edges between colors of equal brightness)
fn sobelColor(uv: vec2f, d: f32) -> f32 {
  let o: vec2f = d / u.resolution;
  let tl: vec3f = TEX(game, uv + vec2f(-o.x, -o.y)).rgb;
  let tr: vec3f = TEX(game, uv + vec2f(o.x, -o.y)).rgb;
  let bl: vec3f = TEX(game, uv + vec2f(-o.x, o.y)).rgb;
  let br: vec3f = TEX(game, uv + vec2f(o.x, o.y)).rgb;
  let ml: vec3f = TEX(game, uv + vec2f(-o.x, 0.0)).rgb;
  let mr: vec3f = TEX(game, uv + vec2f(o.x, 0.0)).rgb;
  let tc: vec3f = TEX(game, uv + vec2f(0.0, -o.y)).rgb;
  let bc: vec3f = TEX(game, uv + vec2f(0.0, o.y)).rgb;
  let gx: vec3f = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  let gy: vec3f = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
  return length(vec2f(length(gx), length(gy)));
}

fn ironbow(t0: f32) -> vec3f {
  let t: f32 = clamp(t0, 0.0, 1.0);
  var c: vec3f = mix(vec3f(0.02, 0.0, 0.08), vec3f(0.32, 0.02, 0.55), smoothstep(0.0, 0.25, t));
  c = mix(c, vec3f(0.85, 0.1, 0.35), smoothstep(0.22, 0.45, t));
  c = mix(c, vec3f(1.0, 0.5, 0.05), smoothstep(0.42, 0.66, t));
  c = mix(c, vec3f(1.0, 0.9, 0.3), smoothstep(0.62, 0.85, t));
  c = mix(c, vec3f(1.0, 1.0, 0.92), smoothstep(0.85, 1.0, t));
  return c;
}

// magnifier inset: 9×9 input pixels around the mouse; the centre cell shows the OUTPUT of the kernel,
// the 3×3 window that feeds it is outlined.
fn kernelInset(col: vec3f, px: vec2f) -> vec3f {
  let s: f32 = max(1.0, u.resolution.y / 540.0);
  let sz: f32 = min(u.resolution.y * 0.36, 220.0 * s);
  let o: vec2f = vec2f(u.resolution.x - sz - 12.0 * s, u.resolution.y - sz - 12.0 * s);
  let q: vec2f = (px - o) / sz;
  if (q.x < -0.02 || q.x > 1.02 || q.y < -0.02 || q.y > 1.02) { return col; }
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return vec3f(0.05); }
  let cell: vec2f = floor(q * 9.0);
  let f: vec2f = fract(q * 9.0);
  let center: vec2i = vec2i(floor(u.mouse.xy));
  let sp: vec2i = center + (vec2i(cell) - vec2i(4, 4)) * i32(u.spread);
  var c: vec3f = LOAD(game, sp).rgb;
  let isC: bool = cell.x == 4.0 && cell.y == 4.0;
  if (isC) { c = convolve(center); }
  // pixel grid
  let edge: f32 = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * sz / 9.0;
  c = mix(c, vec3f(0.0), (1.0 - smoothstep(0.5, 1.2, edge)) * 0.6);
  // 3x3 window outline
  let w: vec2f = abs(q * 9.0 - 4.5);
  let box: f32 = max(w.x, w.y);
  c = mix(c, vec3f(1.0, 0.85, 0.2), (1.0 - smoothstep(0.0, 0.05 * s, abs(box - 1.5))) * step(box, 1.6));
  if (isC) { c = mix(c, vec3f(1.0), 1.0 - smoothstep(0.03, 0.07, min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)))); }
  return c;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex: i32 = i32(u.example);
  let orig: vec3f = TEX(game, uv).rgb;
  var c: vec3f = orig;
  let t: f32 = u.time;

  if (ex == 0) {
    c = mix(orig, convolve(vec2i(floor(px))), u.strength);
    c = splitView(c, orig, px, u.split);
    if (u.inset > 0.5) { c = kernelInset(c, px); }
    return vec4f(c, 1.0);
  }

  if (ex == 1) {
    let g: vec2f = sobel(uv, u.thickness);
    let mag: f32 = length(g) * u.gain;
    let m: f32 = u.sobelMode;
    if (m < 0.5) { c = vec3f(clamp(mag, 0.0, 1.0)); }
    else if (m < 1.5) {
      // gradient direction as hue, strength as brightness
      let ang: f32 = atan2(g.y, g.x) / TAU + 0.5;
      c = hsv2rgb(vec3f(ang, 0.85, clamp(mag, 0.0, 1.0)));
    } else if (m < 2.5) { c = vec3f(clamp(g.x * u.gain * 0.5 + 0.5, 0.0, 1.0)); }
    else if (m < 3.5) { c = vec3f(clamp(g.y * u.gain * 0.5 + 0.5, 0.0, 1.0)); }
    else { c = vec3f(step(u.threshold, mag)); }
    c = splitView(c, orig, px, u.split);
    return vec4f(c, 1.0);
  }

  if (ex == 2) {
    // TOON: posterized colors + ink lines where the color/brightness changes sharply
    var hsv: vec3f = rgb2hsv(orig);
    hsv.z = floor(hsv.z * u.levels + 0.5) / u.levels;
    hsv.y = min(hsv.y * 1.15, 1.0);
    var toon: vec3f = hsv2rgb(hsv);
    let e: f32 = sobelColor(uv, u.thickness);
    let ink: f32 = smoothstep(u.threshold, u.threshold * 1.6 + 0.02, e);
    toon = mix(toon, u.ink, ink);
    c = splitView(toon, orig, px, u.split);
    return vec4f(c, 1.0);
  }

  // VISION MODES
  let a: f32 = u.resolution.x / u.resolution.y;
  let cp: vec2f = (uv - 0.5) * vec2f(a, 1.0);
  let vm: f32 = u.visionMode;
  let g2: vec2f = sobel(uv, 1.5);
  let edge: f32 = clamp(length(g2) * 2.0, 0.0, 1.0);
  if (vm < 0.5) {
    // NIGHT VISION: luminance boosted, edge-enhanced, green phosphor, noise, scanlines, round eyepiece
    var l: f32 = luma(orig) * 1.6 + edge * 0.6;
    l = pow(clamp(l, 0.0, 1.5), 0.8);
    let n: f32 = hash21(floor(px / 1.5) + vec2f(fract(t * 13.1) * 731.0, fract(t * 7.7) * 173.0)) - 0.5;
    l += n * 0.35 * u.noise;
    c = vec3f(0.12, 1.0, 0.25) * l + vec3f(0.6, 1.0, 0.6) * max(l - 0.9, 0.0);
    c *= 0.88 + 0.12 * sin(px.y * 1.6);
    let r: f32 = length(cp);
    c *= 1.0 - smoothstep(0.3, 0.5, r);
    c = mix(c, vec3f(0.0), smoothstep(0.47, 0.49, r));
  } else if (vm < 1.5) {
    // THERMAL: estimate "heat" from brightness and warm hues, blur slightly, map through an ironbow palette
    let o: vec2f = 2.0 / u.resolution;
    let b: vec3f = (orig * 4.0 + TEX(game, uv + vec2f(o.x, 0.0)).rgb + TEX(game, uv - vec2f(o.x, 0.0)).rgb
                  + TEX(game, uv + vec2f(0.0, o.y)).rgb + TEX(game, uv - vec2f(0.0, o.y)).rgb) / 8.0;
    let warm: f32 = clamp(b.r - b.b * 0.8 - b.g * 0.2, 0.0, 1.0);
    var heat: f32 = luma(b) * 0.45 + warm * 0.9 + pow(max(luma(b) - 0.75, 0.0) * 4.0, 2.0) * 0.3;
    heat = heat * (1.0 - uv.y * 0.15) - edge * 0.08;
    heat += (hash21(floor(px / 2.0) + vec2f(floor(t * 20.0), 0.0)) - 0.5) * 0.04 * u.noise;
    c = ironbow(heat * 1.15);
    // crosshair & readout frame
    let ch: vec2f = abs(px - u.resolution * 0.5);
    let xh: f32 = step(ch.x, 1.0) * step(ch.y, 14.0) * step(5.0, ch.y) + step(ch.y, 1.0) * step(ch.x, 14.0) * step(5.0, ch.x);
    c = mix(c, vec3f(1.0), xh * 0.9);
  } else {
    // DETECTIVE VISION: dark blue world, glowing edges, interactables (gold coins, the hero) in orange, a scan wave
    let l: f32 = luma(orig);
    c = vec3f(0.05, 0.1, 0.18) + vec3f(0.15, 0.3, 0.5) * l;
    c += vec3f(0.25, 0.75, 1.0) * edge * 0.9;
    let hsv: vec3f = rgb2hsv(orig);
    let gold: f32 = smoothstep(0.08, 0.06, abs(hsv.x - 0.13)) * smoothstep(0.55, 0.7, hsv.y) * smoothstep(0.6, 0.8, hsv.z);
    let red: f32 = (smoothstep(0.05, 0.02, min(hsv.x, 1.0 - hsv.x))) * smoothstep(0.5, 0.65, hsv.y) * smoothstep(0.6, 0.75, hsv.z);
    let pulse: f32 = 0.75 + 0.25 * sin(t * 5.0);
    c = mix(c, vec3f(1.0, 0.55, 0.1) * pulse * 1.2, clamp(gold + red, 0.0, 1.0) * 0.9);
    // scan wave sweeping outward from the centre
    let r: f32 = length(cp);
    let wave: f32 = fract(t * 0.35) * 1.4;
    c += vec3f(0.3, 0.8, 1.0) * exp(-pow((r - wave) * 30.0, 2.0)) * 0.5;
    c *= 1.0 - 0.4 * smoothstep(0.5, 1.0, r / (0.5 * a));
  }
  c = splitView(c, orig, px, u.split);
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export default shaderScene({
  interaction: 'Move the mouse to slide the before/after divider.',
  examples: [
    {
      id: 'kernels',
      label: '3×3 kernel playground',
      kind: 'Abstract',
      note: 'Each output pixel = the 9 pixels around it, multiplied by the 9 weights shown top-right, added up. The magnifier (bottom right) shows the pixels under the mouse: the yellow 3×3 window feeds the white-framed centre pixel.',
      params: { kernel: 'sharpen', strength: 1, spread: 1, compare: true, inset: true },
    },
    {
      id: 'sobel',
      label: 'Sobel edges',
      kind: 'Abstract',
      note: 'Sobel uses two kernels: one measures how fast brightness changes left→right (X), the other top→bottom (Y). Together they give the edge <b>strength</b> (length) and <b>direction</b> (angle) at every pixel.',
      params: { sobelMode: 'magnitude', gain: 2, thickness: 1, threshold: 0.35, compare: true },
    },
    {
      id: 'toon',
      label: 'Toon outlines',
      kind: 'In a game',
      note: 'Edge detection as an art style: posterize the colors into a few flat bands (cel shading) and draw ink wherever neighbouring colors differ strongly. Thicker lines come from sampling neighbours further away.',
      params: { threshold: 0.6, thickness: 1.25, levels: 4, ink: '#1a1020', compare: true },
    },
    {
      id: 'vision',
      label: 'Vision modes',
      kind: 'In a game',
      note: 'Goggle and “sense” modes from stealth and detective games: <b>night vision</b> (green, noisy, edge-enhanced), <b>thermal</b> (an estimate of heat through a false-color palette) and <b>detective vision</b> (edges glow, the world goes dark, interactables light up).',
      params: { visionMode: 'detective', noise: 0.6, compare: false },
      hint: 'Pick a vision mode on the right.',
    },
  ],
  controls: [
    {
      type: 'select',
      key: 'kernel',
      label: 'Kernel',
      value: 'sharpen',
      options: Object.entries(KERNELS).map(([value, k]) => ({ value, label: k.label })),
      showFor: ['kernels'],
    },
    { type: 'slider', key: 'strength', label: 'Mix with original', min: 0, max: 1, step: 0.01, value: 1, showFor: ['kernels'] },
    { type: 'slider', key: 'spread', label: 'Kernel spread', min: 1, max: 6, step: 1, value: 1, unit: 'px', showFor: ['kernels'], help: 'Distance between the 9 taps. Bigger = coarser effect with the same 9 reads (and visible artifacts).' },
    { type: 'toggle', key: 'inset', label: 'Pixel magnifier', value: true, showFor: ['kernels'] },
    {
      type: 'select',
      key: 'sobelMode',
      label: 'Show',
      value: 'magnitude',
      options: [
        { value: 'magnitude', label: 'Edge strength (magnitude)' },
        { value: 'direction', label: 'Edge direction (hue = angle)' },
        { value: 'gx', label: 'X gradient only' },
        { value: 'gy', label: 'Y gradient only' },
        { value: 'threshold', label: 'Thresholded (black & white)' },
      ],
      showFor: ['sobel'],
    },
    { type: 'slider', key: 'gain', label: 'Gain', min: 0.5, max: 8, step: 0.1, value: 2, showFor: ['sobel'], help: 'Multiplies the edge strength so faint edges become visible.' },
    { type: 'slider', key: 'thickness', label: 'Sample distance', min: 0.5, max: 4, step: 0.1, value: 1, unit: 'px', showFor: ['sobel', 'toon'], help: 'How far apart the taps are: thicker (and softer) lines.' },
    { type: 'slider', key: 'threshold', label: 'Edge threshold', min: 0.02, max: 1.5, step: 0.01, value: 0.35, showFor: ['sobel', 'toon'], help: 'Only edges stronger than this count. Higher = fewer, cleaner lines.' },
    { type: 'slider', key: 'levels', label: 'Color bands', min: 2, max: 10, step: 1, value: 4, showFor: ['toon'], help: 'Brightness levels for the cel-shaded look.' },
    { type: 'color', key: 'ink', label: 'Ink color', value: '#1a1020', showFor: ['toon'] },
    {
      type: 'select',
      key: 'visionMode',
      label: 'Vision mode',
      value: 'detective',
      options: [
        { value: 'night', label: 'Night vision goggles' },
        { value: 'thermal', label: 'Thermal camera' },
        { value: 'detective', label: 'Detective / sense vision' },
      ],
      showFor: ['vision'],
    },
    { type: 'slider', key: 'noise', label: 'Sensor noise', min: 0, max: 1.5, step: 0.01, value: 0.6, showFor: ['vision'] },
    { type: 'toggle', key: 'compare', label: 'Before / after divider', value: true },
  ],
  uniforms: {
    strength: 'f32', spread: 'f32', inset: 'f32', sobelMode: 'f32', gain: 'f32', thickness: 'f32', threshold: 'f32', levels: 'f32',
    ink: 'vec3f', visionMode: 'f32', noise: 'f32', split: 'f32',
    k0: 'vec4f', k1: 'vec4f', k2: 'vec4f', kDiv: 'f32', kBias: 'f32', kAbs: 'f32',
  },
  include: withGameIncludes(['hash', 'color']),
  passes: [gamePass()],
  bind(p, ctx) {
    ov.begin(ctx);
    const k = KERNELS[p.kernel] || KERNELS.identity;
    if (ctx.example === 'kernels') {
      ov.show('kernel', kernelTable(k), 'right:8px;top:46px;line-height:1.35');
      if (p.inset) {
        const S = Math.max(1, ctx.height / 540);
        const sz = Math.min(ctx.height * 0.36, 220 * S);
        ov.show('mag', '9×9 pixels under the mouse', `right:${(12 * S) / ctx.dpr}px;bottom:${(sz + 22 * S) / ctx.dpr}px`);
      }
    }
    const split = compareSplit(st, ctx, p.compare, ov, ['Original', 'Filtered']);
    ov.end();
    const w = k.w;
    return { k0: [w[0], w[1], w[2], 0], k1: [w[3], w[4], w[5], 0], k2: [w[6], w[7], w[8], 0], kDiv: k.div, kBias: k.bias, kAbs: k.abs ? 1 : 0, split };
  },
  code: IMAGE,
  about: {
    summary:
      'A convolution kernel is a tiny grid of weights slid over the image. The same 9-tap loop blurs, sharpens, embosses or detects edges — and edge detection powers outlines, toon shading and “vision mode” effects.',
    what: `<p>The live game scene filtered by 3×3 kernels. The <b>playground</b> shows the weights and a pixel magnifier; <b>Sobel</b> visualises the
      gradient (how brightness changes) as strength or direction; <b>Toon outlines</b> and <b>Vision modes</b> turn edges into game styles.</p>`,
    how: `<ol>
      <li><b>Convolution</b>: for each pixel, read its 3×3 neighbourhood, multiply each by its weight, sum, then divide (so blur weights add up to 1).</li>
      <li><b>Blur</b> kernels have only positive weights (an average). <b>Sharpen</b> = the pixel minus a bit of its neighbours (adds back the detail a blur removes).
        <b>Edge</b> kernels (Laplacian, outline) sum to 0: flat areas give 0 (black), only changes survive.</li>
      <li><b>Sobel</b>: <code>gx</code> = right column − left column (weighted 1-2-1), <code>gy</code> = bottom row − top row. Edge strength = √(gx² + gy²), direction = atan2(gy, gx).</li>
      <li><b>Toon</b>: posterize brightness into a few bands, then draw ink where the color Sobel exceeds a threshold. Sampling neighbours further away gives thicker lines.</li>
      <li><b>Vision modes</b> are a recipe: luminance or a heat estimate → a palette, edges added or highlighted, plus noise, scanlines and a vignette or eyepiece mask.</li>
    </ol>`,
    uses: [
      { title: 'Outlines', text: 'Toon/comic outlines, selection outlines, “cel-shaded” 2D-HD looks (Borderlands-style ink).' },
      { title: 'Vision modes', text: 'Batman Arkham detective vision, Splinter Cell night vision, Far Cry/Crysis thermal, The Witcher senses.' },
      { title: 'Image polish', text: 'A light sharpen after upscaling; emboss for stamped UI or carved stone; blur for soft shadows.' },
      { title: 'Gameplay & AI', text: 'Edge maps help find walkable surfaces or object boundaries in generated levels.' },
    ],
    try: [
      'In the playground pick <i>Outline</i> and move the magnifier over the hero: flat areas become black, edges light up.',
      'Pick <i>Sharpen</i>, then raise <i>Kernel spread</i> to 4 — a halo appears: that is “over-sharpening”.',
      'On <b>Sobel</b> choose <i>Edge direction</i>: horizontal edges and vertical edges get different colors.',
      'On <b>Toon outlines</b> set Color bands to 2 and Sample distance to 3 for a bold comic look.',
      'On <b>Vision modes</b> switch to Thermal: torches, coins, the sun and the hero glow hot.',
    ],
    ask: [
      'a 3×3 convolution post effect with sharpen/emboss/edge presets',
      'Sobel outlines on the whole screen for a toon look',
      'night vision goggles with noise, scanlines and a round eyepiece',
      'a thermal camera mode with an ironbow palette',
      'detective vision that highlights interactable objects',
    ],
    perf: `<p>A 3×3 kernel is 9 texture reads per pixel (Sobel needs 8). Cheap at any resolution. Bigger kernels grow with the square of the size —
      separate them into two 1D passes when possible (see the Blur scene), or read neighbours from workgroup shared memory with a compute shader.</p>`,
    api: `<p>Identical in WebGPU and WebGL2: neighbour reads use <code>textureLoad</code> / <code>texelFetch</code> (exact pixels) or filtered samples.
      In WebGPU, a compute shader can load a tile plus a 1-pixel border into <i>workgroup memory</i> once and let every thread reuse it — fewer texture reads for big kernels.</p>`,
    code: [
      {
        title: 'Generic 3×3 convolution',
        lang: 'wgsl',
        src: `fn px3(p: vec2i, dx: i32, dy: i32) -> vec3f { return LOAD(game, p + vec2i(dx, dy)).rgb; }
fn convolve(p: vec2i) -> vec3f {
  var s = vec3f(0.0);
  s += px3(p, -1, -1) * u.k0.x + px3(p, 0, -1) * u.k0.y + px3(p, 1, -1) * u.k0.z;
  s += px3(p, -1,  0) * u.k1.x + px3(p, 0,  0) * u.k1.y + px3(p, 1,  0) * u.k1.z;
  s += px3(p, -1,  1) * u.k2.x + px3(p, 0,  1) * u.k2.y + px3(p, 1,  1) * u.k2.z;
  return s / u.kDiv + vec3f(u.kBias);
}`,
      },
      {
        title: 'Sobel gradient',
        lang: 'wgsl',
        src: `let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);   // left -> right change
let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);   // top -> bottom change
let strength = length(vec2f(gx, gy));
let direction = atan2(gy, gx);`,
      },
    ],
    links: [{ title: 'Image kernels explained visually (Setosa)', url: 'https://setosa.io/ev/image-kernels/', note: 'interactive explanation of 3×3 kernels' }],
  },
});

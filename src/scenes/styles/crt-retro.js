import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import { setTag, placeTag, makeSplit, gamePass, GAME_INCLUDES } from './_shared-a.js';

// CRT & VHS simulation. Passes:
//   scene : the platformer at ~2× the target line count (cheap)
//   src   : the "video signal" at the console's resolution (lines × aspect). VHS tape damage happens
//           here, per scan line, in YIQ (luma/chroma) space — like a real tape signal.
//   glow  : a blurred copy of src = phosphor glow / bloom / halation
//   image : the TV: curvature, gaussian scanline beams, RGB convergence error, phosphor mask,
//           glow, vignette, flicker, interlacing, bezel — plus a before/after split and a magnifier.

const splitPos = makeSplit(0.5);

// VCR on-screen display: 4 rows of text in a small atlas (row height 32px).
let osdCanvas = null;
function makeOSD() {
  if (osdCanvas) return osdCanvas;
  const c = makeCanvas(256, 128);
  const g = c.getContext('2d');
  g.clearRect(0, 0, 256, 128);
  g.font = 'bold 25px "Courier New", monospace';
  g.textBaseline = 'middle';
  const rows = ['PLAY ▶', 'OCT. 03 1996', 'PM 11:47', 'SP'];
  rows.forEach((t, i) => {
    g.lineWidth = 5;
    g.strokeStyle = 'rgba(0,0,0,0.85)';
    g.strokeText(t, 6, i * 32 + 17);
    g.fillStyle = '#f2f2f2';
    g.fillText(t, 6, i * 32 + 17);
  });
  osdCanvas = c;
  return c;
}

const MASKS = ['no mask', 'aperture grille', 'slot mask', 'shadow mask'];

const lineCount = (params) => Math.max(64, Math.round(params.lines));
const srcSize = (params, ctx) => {
  const h = lineCount(params);
  return [Math.max(16, Math.round((h * ctx.width) / Math.max(1, ctx.height))), h];
};

export default shaderScene({
  interaction: '',
  examples: [
    {
      id: 'arcade',
      label: 'Arcade CRT',
      kind: 'In a game',
      hint: '',
      note: 'A 224-line arcade monitor: crisp beams with dark gaps between lines, an <b>aperture grille</b> (vertical RGB phosphor stripes), strong glow and only a gentle curve.',
      params: { lines: 224, curvature: 0.18, scan: 0.8, mask: 1, maskStr: 0.55, glow: 0.7, vignette: 0.25, flicker: 0.1, conv: 0.4, interlace: false, tracking: 0, bleed: 0, wobble: 0, osd: false, magnify: false },
    },
    {
      id: 'tv',
      label: 'Old TV',
      kind: 'Real life',
      hint: '',
      note: 'A worn living-room TV: a bulging screen, <b>shadow mask</b> dot triads, misaligned red/blue guns (convergence error), interlaced fields, a rolling hum bar, flicker and a dark vignette.',
      params: { lines: 240, curvature: 0.6, scan: 0.5, mask: 3, maskStr: 0.6, glow: 0.55, vignette: 0.65, flicker: 0.5, conv: 1.6, interlace: true, tracking: 0, bleed: 0, wobble: 0, osd: false, magnify: false },
    },
    {
      id: 'vhs',
      label: 'VHS tape',
      kind: 'Real life',
      hint: '',
      note: 'A worn tape played on that TV: color <b>bleeds</b> sideways (chroma has far less bandwidth than brightness), lines <b>wobble</b>, a <b>tracking</b> noise band rolls through and the bottom lines tear at the <b>head switch</b>.',
      params: { lines: 240, curvature: 0.35, scan: 0.4, mask: 2, maskStr: 0.35, glow: 0.5, vignette: 0.5, flicker: 0.3, conv: 0.8, interlace: false, tracking: 0.6, bleed: 0.75, wobble: 0.5, osd: true, magnify: false },
    },
    {
      id: 'split',
      label: 'Clean vs filtered',
      kind: 'Comparison',
      hint: 'Drag to move the divider. Hover to magnify the phosphors.',
      note: 'Left: the raw pixels, scaled up with nearest filtering. Right: the same pixels through the CRT model. The <b>magnifier</b> under the mouse shows the beams and phosphor stripes up close.',
      params: { lines: 240, curvature: 0.0, scan: 0.7, mask: 1, maskStr: 0.6, glow: 0.55, vignette: 0.2, flicker: 0.05, conv: 0.5, interlace: false, tracking: 0, bleed: 0, wobble: 0, osd: false, magnify: true },
    },
  ],
  controls: [
    { type: 'heading', label: 'Tube' },
    { type: 'slider', key: 'lines', label: 'Scanlines (vertical resolution)', min: 120, max: 480, step: 1, value: 240, help: 'Lines in the video signal. Consoles: 224–240. DVD/480i: 480.' },
    { type: 'slider', key: 'curvature', label: 'Screen curvature', min: 0, max: 1, step: 0.01, value: 0.3 },
    { type: 'slider', key: 'scan', label: 'Scanline strength', min: 0, max: 1, step: 0.01, value: 0.7, help: 'Gaussian beam per line; bright lines bloom wider than dark ones.' },
    {
      type: 'select', key: 'mask', label: 'Phosphor mask', value: 1,
      options: MASKS.map((label, value) => ({ value, label: label[0].toUpperCase() + label.slice(1) })),
      help: 'Grille: Sony Trinitron stripes. Slot: most TVs. Shadow: dot triads of monitors.',
    },
    { type: 'slider', key: 'maskStr', label: 'Mask strength', min: 0, max: 1, step: 0.01, value: 0.5 },
    { type: 'slider', key: 'glow', label: 'Phosphor glow', min: 0, max: 1.5, step: 0.01, value: 0.5, help: 'Bright phosphors bleed light into the glass (halation).' },
    { type: 'slider', key: 'vignette', label: 'Vignette', min: 0, max: 1, step: 0.01, value: 0.4 },
    { type: 'slider', key: 'flicker', label: 'Flicker & noise', min: 0, max: 1, step: 0.01, value: 0.2 },
    { type: 'slider', key: 'conv', label: 'Convergence error (px)', min: 0, max: 4, step: 0.05, value: 0.6, showFor: ['arcade', 'tv', 'split'], help: 'The red, green and blue beams don’t land in exactly the same place.' },
    { type: 'toggle', key: 'interlace', label: 'Interlaced fields', value: false, showFor: ['arcade', 'tv', 'split'], help: 'Odd and even lines are drawn on alternate frames (480i): fine detail shimmers.' },
    { type: 'toggle', key: 'magnify', label: 'Magnifier under the mouse', value: false, showFor: ['split'] },
    { type: 'heading', label: 'VHS tape', showFor: ['vhs'] },
    { type: 'slider', key: 'tracking', label: 'Tracking noise', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['vhs'] },
    { type: 'slider', key: 'bleed', label: 'Chroma bleed', min: 0, max: 1, step: 0.01, value: 0.75, showFor: ['vhs'], help: 'Color is stored at ~1/8 the horizontal resolution of brightness.' },
    { type: 'slider', key: 'wobble', label: 'Tape wobble', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['vhs'] },
    { type: 'toggle', key: 'osd', label: 'VCR on-screen display', value: true, showFor: ['vhs'] },
  ],
  uniforms: {
    lines: 'f32', curvature: 'f32', scan: 'f32', mask: 'f32', maskStr: 'f32', glow: 'f32', vignette: 'f32', flicker: 'f32',
    conv: 'f32', interlace: 'f32', tracking: 'f32', bleed: 'f32', wobble: 'f32', osd: 'f32', magnify: 'f32', split: 'f32', msize: 'f32',
  },
  include: ['color', ...GAME_INCLUDES],
  textures: { osdTex: { source: async () => makeOSD(), filter: 'linear' } },
  resetOn: ['lines', 'tracking', 'bleed', 'wobble', 'osd'],
  passes: [
    gamePass('scene', { scale: (params, ctx) => Math.min(1, (2.2 * lineCount(params)) / Math.max(1, ctx.height)) }),
    {
      name: 'src',
      format: 'rgba8unorm',
      size: srcSize,
      code: /* wgsl */ `
fn rgb2yiq(c: vec3f) -> vec3f {
  return vec3f(dot(c, vec3f(0.299, 0.587, 0.114)), dot(c, vec3f(0.596, -0.274, -0.322)), dot(c, vec3f(0.211, -0.523, 0.312)));
}
fn yiq2rgb(c: vec3f) -> vec3f {
  return vec3f(c.x + 0.956 * c.y + 0.621 * c.z, c.x - 0.272 * c.y - 0.647 * c.z, c.x - 1.106 * c.y + 1.703 * c.z);
}
fn osdText(p: vec2f, origin: vec2f, h: f32, row: f32) -> vec4f {
  // p, origin in src pixels; text box is 8h wide × h tall, mapped to one 256×32 atlas row
  let q = (p - origin) / vec2f(h * 8.0, h);
  if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) { return vec4f(0.0); }
  return TEX(osdTex, vec2f(q.x, (row + q.y) / 4.0));
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let size = TEXSIZE(src);
  let line = floor(px.y);
  let vhs = step(1.5, u.example) * step(u.example, 2.5);
  var x = uv.x;
  var noiseAdd = 0.0;
  if (vhs > 0.5) {
    let t = u.time;
    // tape wobble: slow, smooth per-line horizontal jitter
    let wob = perlin(vec2f(line * 0.035, t * 1.3)) * 2.2 + sin(line * 0.11 + t * 2.0) * 0.6;
    x += wob * u.wobble / size.x;
    // tracking band rolling down the picture
    let bandY = fract(t * 0.06) * 1.4 - 0.2;
    let band = (1.0 - smoothstep(0.0, 0.045, abs(uv.y - bandY))) * u.tracking;
    let jag = (hash11(line * 1.7 + floor(t * 24.0)) - 0.5) * band * 18.0;
    x += jag / size.x;
    // head-switching noise: the bottom lines are torn sideways
    let hs = smoothstep(0.94, 1.0, uv.y);
    x += hs * (6.0 + 5.0 * hash11(line + floor(t * 30.0))) / size.x;
    // white "dropout" streaks inside the band and at the bottom
    let streak = step(1.0 - 0.06 * band - 0.15 * hs, hash21(vec2f(floor(px.x / 6.0), line + floor(t * 30.0) * 7.0)));
    noiseAdd = streak * (0.6 + 0.4 * hash11(px.x + t));
  }
  // luma: 3 taps with a little ringing (sharpening halo) like a VCR's "edge enhance"
  let dx = 1.0 / size.x;
  let y0 = TEX(scene, vec2f(x, uv.y)).rgb;
  var c = y0;
  if (vhs > 0.5) {
    let yl = rgb2yiq(TEX(scene, vec2f(x - dx * 1.5, uv.y)).rgb).x;
    let yr = rgb2yiq(TEX(scene, vec2f(x + dx * 1.5, uv.y)).rgb).x;
    let yc = rgb2yiq(y0).x;
    let lumaY = yc + (yc - 0.5 * (yl + yr)) * 0.6;
    // chroma: wide low-pass, delayed to the right (signal lag)
    var iq = vec2f(0.0);
    var wsum = 0.0;
    for (var i = 0; i < 10; i++) {
      let fi = f32(i);
      let w = exp(-fi * 0.28);
      let s = TEX(scene, vec2f(x - (fi * 1.6 + 1.5) * dx * u.bleed, uv.y)).rgb;
      iq += rgb2yiq(s).yz * w;
      wsum += w;
    }
    iq /= wsum;
    c = yiq2rgb(vec3f(lumaY, iq * 1.15));
    // washed-out tape colors: lifted blacks, slight warm tint
    c = mix(c, vec3f(luma(c)), 0.12) * vec3f(1.02, 0.98, 0.95) * 0.92 + vec3f(0.035, 0.03, 0.045);
    c += vec3f(noiseAdd);
    c += vec3f((hash21(px + vec2f(u.time * 61.0, 0.0)) - 0.5) * 0.06);
    // VCR on-screen display (added by the player AFTER the tape, so it doesn't wobble)
    if (u.osd > 0.5) {
      let h = size.y * 0.06;
      let m = vec2f(size.y * 0.07, size.y * 0.06);
      var o = osdText(px, m, h, 0.0);
      o = max(o, osdText(px, vec2f(size.x - m.x - h * 2.0, m.y), h, 3.0));
      o = max(o, osdText(px, vec2f(m.x, size.y - m.y - h * 2.2), h, 2.0));
      o = max(o, osdText(px, vec2f(m.x, size.y - m.y - h * 1.1), h, 1.0));
      c = mix(c, o.rgb, o.a);
    }
  }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`,
    },
    {
      // phosphor glow: a 5×5-tap blur of the signal at low resolution (linear filtering makes it ~10×10 texels)
      name: 'glow',
      format: 'rgba8unorm',
      size: srcSize,
      code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let texel = 1.0 / TEXSIZE(src);
  var acc = vec3f(0.0);
  var wsum = 0.0;
  for (var j = -2; j <= 2; j++) {
    for (var i = -2; i <= 2; i++) {
      let o = vec2f(f32(i), f32(j)) * 1.7;
      let w = exp(-dot(o, o) * 0.09);
      acc += TEX(src, uv + o * texel).rgb * w;
      wsum += w;
    }
  }
  return vec4f(acc / wsum, 1.0);
}`,
    },
  ],
  bind(params, ctx) {
    const split = splitPos(ctx);
    const ex = ctx.example;
    const [w, h] = srcSize(params, ctx);
    const isSplit = ex === 'split';
    const a = setTag(ctx, 'before', 'Clean pixels', '', isSplit);
    const b = setTag(ctx, 'after', 'CRT model', '', isSplit);
    if (isSplit) {
      placeTag(a, split * 100, '48px', 'calc(-100% - 10px)');
      placeTag(b, split * 100, '48px', '10px');
    }
    setTag(ctx, 'info', `${w}×${h} signal · ${MASKS[params.mask | 0]}`, 'right:8px;bottom:8px');
    // phosphor stripe width: ~1 screen pixel at 1080p, more on bigger screens
    const msize = Math.max(1, Math.round(ctx.height / 900));
    return { split, msize };
  },
  code: /* wgsl */ `
// Barrel distortion: push points outwards more the further they are from the centre.
fn curveUV(uv: vec2f) -> vec2f {
  var p = uv * 2.0 - 1.0;
  let k = u.curvature;
  p *= 1.0 + k * 0.06;                             // zoom out a little: show the bezel
  p *= vec2f(1.0) + k * vec2f(p.y * p.y * 0.12, p.x * p.x * 0.18);
  return p * 0.5 + 0.5;
}
// One electron beam per scan line, gaussian profile, width grows with brightness.
fn beam(d: f32, c: vec3f) -> vec3f {
  let wid = mix(vec3f(0.24), vec3f(0.5), sqrt(clamp(c, vec3f(0.0), vec3f(1.0))));
  let k = vec3f(d) / wid;
  return exp(-k * k);
}
fn scanSample(uvc: vec2f, field: f32) -> vec3f {
  let H = TEXSIZE(src).y;
  let W = TEXSIZE(src).x;
  // horizontally: slightly sharpened linear filtering ("sharp bilinear")
  let fx = uvc.x * W - 0.5;
  let ix = floor(fx);
  let tx = smoothstep(0.2, 0.8, fx - ix);
  let sx = (ix + 0.5 + tx) / W;
  let y = uvc.y * H - 0.5 - field * 0.5;
  let y0 = floor(y);
  let f = y - y0;
  let c0 = TEX(src, vec2f(sx, (y0 + 0.5) / H)).rgb;
  let c1 = TEX(src, vec2f(sx, (y0 + 1.5) / H)).rgb;
  let beams = (c0 * beam(f, c0) + c1 * beam(1.0 - f, c1)) * 1.3;
  let flat = mix(c0, c1, f);
  return mix(flat, beams, u.scan);
}
fn maskColor(p: vec2f) -> vec3f {
  let ms = u.msize;
  let m = u.mask;
  var c = vec3f(1.0);
  if (m > 0.5 && m < 1.5) {
    // aperture grille: continuous vertical R G B stripes
    let i = fmod(floor(p.x / ms), 3.0);
    c = vec3f(step(i, 0.5), step(0.5, i) * step(i, 1.5), step(1.5, i));
  } else if (m > 1.5 && m < 2.5) {
    // slot mask: RGB stripes broken into slots, alternate triads offset by half a slot
    let i = fmod(floor(p.x / ms), 3.0);
    let triad = floor(p.x / (ms * 3.0));
    let yy = fmod(p.y + triad * ms * 2.0, ms * 4.0);
    let gap = step(ms * 3.0, yy);
    c = vec3f(step(i, 0.5), step(0.5, i) * step(i, 1.5), step(1.5, i)) * (1.0 - gap * 0.85);
  } else if (m > 2.5) {
    // shadow mask: dot triads in a delta (triangle) arrangement
    let row = floor(p.y / (ms * 2.0));
    let xx = p.x + row * ms * 1.5;
    let i = fmod(floor(xx / ms), 3.0);
    let fy = fract(p.y / (ms * 2.0));
    let dotShape = smoothstep(0.0, 0.25, fy) * (1.0 - smoothstep(0.75, 1.0, fy));
    c = vec3f(step(i, 0.5), step(0.5, i) * step(i, 1.5), step(1.5, i)) * (0.35 + 0.65 * dotShape);
  }
  return mix(vec3f(1.0), c, u.maskStr);
}
fn crt(pp: vec2f) -> vec3f {
  let res = u.resolution;
  let uv = pp / res;
  let uvc = curveUV(uv);
  let field = u.interlace * fmod(u.frame, 2.0);
  // convergence error: the three guns land slightly apart
  let co = vec2f(u.conv / res.x, u.conv * 0.35 / res.y);
  let r = scanSample(uvc + co, field).r;
  let g = scanSample(uvc, field).g;
  let b = scanSample(uvc - co, field).b;
  var c = vec3f(r, g, b);
  // phosphor mask (average brightness compensated)
  c *= maskColor(pp) * (1.0 + u.maskStr * 0.75);
  // glow / halation from the blurred signal
  let gl = TEX(glow, uvc).rgb;
  c += gl * gl * u.glow * 0.5 + gl * u.glow * 0.08;
  // flicker, rolling hum bar and noise
  let fl = u.flicker;
  c *= 1.0 + fl * 0.06 * (hash11(u.frame) - 0.5);
  c *= 1.0 - fl * 0.12 * smoothstep(0.0, 0.25, 0.5 - abs(fract(uvc.y * 0.6 - u.time * 0.12) - 0.5));
  c += vec3f((hash21(pp + vec2f(fmod(u.frame, 97.0) * 13.0, 0.0)) - 0.5) * fl * 0.07);
  // vignette
  let v = uvc * (1.0 - uvc);
  c *= mix(1.0, pow(clamp(v.x * v.y * 16.0, 0.0, 1.0), 0.35), u.vignette);
  // screen edge with rounded corners, then the bezel
  let q = abs(uvc - 0.5);
  let rad = 0.02 + 0.05 * u.curvature;
  let ed = length(max(q - vec2f(0.5 - rad), vec2f(0.0))) - rad;
  let inside = 1.0 - smoothstep(-0.0015, 0.0015, ed);
  let bez = mix(vec3f(0.035, 0.034, 0.038), vec3f(0.09, 0.088, 0.095), clamp(1.0 - length(uv - 0.5) * 1.2, 0.0, 1.0));
  let refl = TEX(glow, clamp(uvc, vec2f(0.0), vec2f(1.0))).rgb * exp(-max(ed, 0.0) * 60.0) * 0.35;
  return mix(bez + refl, max(c, vec3f(0.0)), inside);
}
fn clean(pp: vec2f) -> vec3f { return TEXN(src, pp / u.resolution).rgb; }

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  var pp = px;
  // magnifier: re-evaluate the whole TV at 5× zoom around the mouse
  let ml = length(px - u.mouse.xy);
  let lr = res.y * 0.2;
  let inLens = u.magnify * u.mouse.w * (1.0 - step(lr, ml));
  if (inLens > 0.5) { pp = u.mouse.xy + (px - u.mouse.xy) / 5.0; }
  let sx = u.split * res.x;
  var c = crt(pp);
  if (u.example > 2.5 && pp.x < sx) { c = clean(pp); }
  if (u.example > 2.5) {
    c = mix(c, vec3f(1.0), (1.0 - smoothstep(1.0, 2.5, abs(px.x - sx))) * (1.0 - inLens));
  }
  // lens rim
  if (u.magnify * u.mouse.w > 0.5) {
    let rim = 1.0 - smoothstep(1.0, 2.5, abs(ml - lr));
    c = mix(c, vec3f(0.85, 0.9, 1.0), rim);
  }
  return vec4f(c, 1.0);
}`,
  about: {
    summary: 'Old games were drawn for glowing glass, not sharp LCD pixels. A CRT shader models the electron beams, phosphors and glass so pixel art looks the way its artists saw it — and VHS adds the charm of worn tape.',
    what: `<p>The platformer as a low-resolution video signal displayed on simulated tubes: an arcade monitor, an old TV, a VHS tape, and a
      side-by-side of raw pixels vs. the CRT model with a magnifier.</p>`,
    how: `<ol>
      <li><b>Signal.</b> Render the game at console resolution (e.g. 427×240) — pass <code>src</code>. One texel per original pixel.</li>
      <li><b>Curvature.</b> Bend the screen coordinates with a barrel distortion (<code>p *= 1 + k·(p.y², p.x²)</code>) before sampling;
        outside the curved rectangle draw the plastic bezel (with a faint reflection of the glow).</li>
      <li><b>Scanlines.</b> For each screen pixel find the two nearest video lines and add their colors weighted by a <b>gaussian beam
        profile</b>. Bright beams are wider, so highlights fill the gaps while dark areas show strong lines — the authentic look, not just darkened rows.</li>
      <li><b>Convergence.</b> Sample red, green and blue at slightly different positions: the three electron guns never align perfectly.</li>
      <li><b>Phosphor mask.</b> Multiply by a repeating pattern of red, green and blue sub-pixels in screen space: vertical stripes (aperture grille),
        broken stripes (slot mask) or dot triads (shadow mask). Then brighten to compensate for the light the mask blocks.</li>
      <li><b>Glow.</b> A blurred copy of the signal (pass <code>glow</code>) is added back — phosphors and glass scatter light (halation/bloom).</li>
      <li><b>Imperfections.</b> Vignette, per-frame flicker, a slowly rolling “hum bar”, noise, and interlacing (odd/even lines on alternate frames).</li>
      <li><b>VHS</b> happens in the <i>signal</i> pass, per scan line, in YIQ space (brightness Y + color I/Q, like the analog signal):
        color is low-pass filtered and delayed sideways (<b>chroma bleed</b>), brightness gets a sharpening halo, lines are shifted by
        slow noise (<b>wobble</b>), a band of torn lines and white dropouts rolls through (<b>tracking</b>), and the last lines are shoved
        sideways where the video heads switch (<b>head-switching noise</b>). The VCR’s on-screen text is added after the tape, so it stays steady.</li>
    </ol>`,
    uses: [
      { title: 'Retro games & collections', text: 'Sonic Mania, Shovel Knight, Blasphemous and most Nintendo/SEGA/Capcom retro collections ship CRT filters; emulators (RetroArch’s CRT-Royale, CRT-Geom, crt-lottes) made them famous.' },
      { title: 'Horror & found footage', text: 'VHS distortion sells analog horror: Iron Lung, Signalis menus, Inscryption’s tape segments, “VHS” indie horror.' },
      { title: 'Menus & diegetic screens', text: 'In-game monitors, security cameras and computer terminals (Alien: Isolation, Observation) use scanlines and curvature.' },
      { title: 'Music videos & synthwave', text: 'VHS tracking bands and chroma bleed are shorthand for the 80s/90s.' },
    ],
    try: [
      'On <b>Clean vs filtered</b>, hover the right side: the magnifier shows each line as a soft beam and the RGB phosphor stripes.',
      'Set <i>Scanline strength</i> to 1 and look at the dark trees vs the bright sky: bright lines swell, dark lines thin out.',
      'Switch the <i>Phosphor mask</i> between grille, slot and shadow mask while magnifying.',
      'On <b>VHS tape</b>, push <i>Chroma bleed</i> to 1: the hero’s red smears to the right of its outline.',
      'Raise <i>Scanlines</i> to 480 — the signal becomes DVD-like and the effect gets subtler.',
    ],
    ask: [
      'a CRT shader with gaussian scanlines, aperture-grille mask and bloom',
      'barrel distortion with a curved bezel',
      'VHS effect: chroma bleed, tracking noise and head-switching noise',
      'interlacing and a rolling hum bar',
      'a before/after split to compare raw pixels and the CRT filter',
    ],
    perf: `<p>Cheap. The signal and glow passes run at console resolution (~100k texels). The final pass does ~7 texture reads per screen
      pixel (two lines × three colors + glow). Phosphor masks need roughly 1 screen pixel per stripe to look right, so CRT shaders look best at
      1080p and above — at low resolution the mask causes moiré; lower <i>Mask strength</i>.</p>`,
    api: `<p>Pure fragment-shader work: identical on WebGL2 and WebGPU. The multi-pass structure (signal → glow → screen) is the same render-to-texture
      chain in both. Famous emulator shaders (crt-royale) use 10+ passes; this one keeps the essentials in 3.</p>`,
    code: [
      {
        title: 'Gaussian scanline beams',
        lang: 'wgsl',
        src: `fn beam(d: f32, c: vec3f) -> vec3f {            // d = distance to the line centre (in lines)
  let wid = mix(vec3f(0.24), vec3f(0.5), sqrt(c));  // brighter beam = wider
  let k = vec3f(d) / wid;
  return exp(-k * k);
}
let y = uvc.y * H - 0.5;  let y0 = floor(y);  let f = y - y0;
let c0 = TEX(src, vec2f(sx, (y0 + 0.5) / H)).rgb;   // line above
let c1 = TEX(src, vec2f(sx, (y0 + 1.5) / H)).rgb;   // line below
let beams = (c0 * beam(f, c0) + c1 * beam(1.0 - f, c1)) * 1.3;`,
      },
      {
        title: 'Aperture grille + barrel distortion',
        lang: 'wgsl',
        src: `fn curveUV(uv: vec2f) -> vec2f {
  var p = uv * 2.0 - 1.0;
  p *= vec2f(1.0) + u.curvature * vec2f(p.y * p.y * 0.12, p.x * p.x * 0.18);
  return p * 0.5 + 0.5;
}
let i = fmod(floor(px.x / u.msize), 3.0);          // 0 = R, 1 = G, 2 = B stripe
let stripe = vec3f(step(i, 0.5), step(0.5, i) * step(i, 1.5), step(1.5, i));
c *= mix(vec3f(1.0), stripe, u.maskStr) * (1.0 + u.maskStr * 0.75);`,
      },
      {
        title: 'VHS chroma bleed (in YIQ)',
        lang: 'wgsl',
        src: `var iq = vec2f(0.0); var wsum = 0.0;
for (var i = 0; i < 10; i++) {                       // wide, one-sided low-pass on color only
  let w = exp(-f32(i) * 0.28);
  let s = TEX(scene, vec2f(x - (f32(i) * 1.6 + 1.5) * dx * u.bleed, uv.y)).rgb;
  iq += rgb2yiq(s).yz * w;  wsum += w;
}
c = yiq2rgb(vec3f(lumaY, iq / wsum * 1.15));        // sharp brightness, smeared color`,
      },
    ],
    links: [
      { title: 'Libretro CRT shaders', url: 'https://github.com/libretro/glsl-shaders/tree/master/crt', note: 'crt-royale, crt-geom, crt-lottes…' },
    ],
  },
});

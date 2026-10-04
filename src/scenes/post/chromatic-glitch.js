import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import { overlays, compareSplit, SPLIT_WGSL, gamePass, withGameIncludes } from './_shared.js';

// Chromatic aberration (radial / directional / spectral), a damage-hit pulse, and a cyberpunk glitch
// transition with block displacement, scanline jitter, RGB split and a datamosh-style feedback smear.

const cur = { ex: 'aberration' };
const st = { split: 0.5, hitT: -10, shake: [0, 0], shakeT: -1 };
const ov = overlays();

// ------------------------------------------------------------------------------------------ WGSL
const NEON_WGSL = /* wgsl */ `
// A second, procedural "level": a synthwave city at night (the glitch transition target).
fn neonCity(uv: vec2f) -> vec3f {
  let a: f32 = u.resolution.x / u.resolution.y;
  let p: vec2f = vec2f(uv.x * a, uv.y);
  let t: f32 = u.time;
  let hor: f32 = 0.64;
  var c: vec3f = mix(vec3f(0.04, 0.02, 0.12), vec3f(0.5, 0.08, 0.38), smoothstep(0.0, hor, uv.y));
  // stars
  let sc: vec2f = floor(p * 90.0);
  c += vec3f(0.8, 0.7, 1.0) * step(0.985, hash21(sc)) * (1.0 - smoothstep(0.1, 0.45, uv.y)) * (0.6 + 0.4 * sin(t * 3.0 + hash21(sc) * 40.0));
  // striped sun
  let sp: vec2f = p - vec2f(a * 0.5, 0.45);
  let sd: f32 = length(sp) - 0.2;
  let cutY: f32 = (uv.y - 0.43) * 30.0;
  let cut: f32 = step(0.43, uv.y) * step(fract(cutY), clamp((uv.y - 0.43) * 6.0, 0.0, 0.8));
  let sunCol: vec3f = mix(vec3f(1.0, 0.85, 0.3), vec3f(1.0, 0.2, 0.55), smoothstep(0.25, 0.62, uv.y));
  c += vec3f(1.0, 0.3, 0.6) * exp(-max(sd, 0.0) * 9.0) * 0.45;
  c = mix(c, sunCol, (1.0 - smoothstep(-0.002, 0.002, sd)) * (1.0 - cut));
  // skyline with lit windows
  let bw: f32 = 0.055;
  let bx: f32 = floor(p.x / bw);
  let hb: f32 = hash21(vec2f(bx, 3.0));
  let top: f32 = hor - 0.05 - 0.22 * hb * hb;
  if (uv.y > top && uv.y < hor) {
    c = vec3f(0.06, 0.02, 0.1);
    let wp: vec2f = vec2f(fract(p.x / bw) * 5.0, (uv.y - top) / 0.018);
    let wid: vec2f = floor(wp);
    let lit: f32 = step(0.6, hash21(vec2f(bx * 7.0 + wid.x, wid.y))) * step(0.6, fract(wp.x)) * step(0.45, fract(wp.y)) * step(0.5, wid.x) * step(wid.x, 3.5);
    c += mix(vec3f(0.2, 0.9, 1.0), vec3f(1.0, 0.3, 0.8), hash21(vec2f(bx, wid.y))) * lit * 0.9;
  }
  // perspective grid floor
  if (uv.y >= hor) {
    let z: f32 = 0.5 / (uv.y - hor + 0.004);
    let gx: f32 = (p.x - a * 0.5) * z * 0.9;
    let gz: f32 = z + t * 2.0;
    let lx: f32 = abs(fract(gx) - 0.5);
    let lz: f32 = abs(fract(gz) - 0.5);
    let wx: f32 = fwidth(gx);
    let wz: f32 = fwidth(gz);
    let line: f32 = max(1.0 - smoothstep(0.0, wx * 1.5, 0.5 - lx), 1.0 - smoothstep(0.0, wz * 1.5, 0.5 - lz));
    c = mix(vec3f(0.05, 0.01, 0.1), vec3f(0.1, 0.02, 0.18), smoothstep(hor, 1.0, uv.y));
    c += vec3f(1.0, 0.2, 0.85) * line * smoothstep(hor, hor + 0.1, uv.y);
    c += vec3f(1.0, 0.2, 0.7) * exp(-(uv.y - hor) * 40.0) * 0.5;
  }
  // title text (Canvas2D texture)
  let tp: vec2f = vec2f((p.x - a * 0.5) / 0.95 + 0.5, (uv.y - 0.1) / 0.2375);
  if (tp.x > 0.0 && tp.x < 1.0 && tp.y > 0.0 && tp.y < 1.0) {
    let tx: vec4f = TEX(neonText, tp);
    c = mix(c, tx.rgb, tx.a);
  }
  return c;
}`;

const GLITCH_WGSL = /* wgsl */ `
// Per-block random numbers that change "tick" times per second: the jumpy look of digital glitches.
fn blockRand(cell: vec2f, tick: f32) -> vec3f { return hash33(vec3f(cell, tick)); }

// Displace uv: random horizontal slices, rectangular blocks and per-scanline jitter.
// Returns (uv.x, uv.y, block random) — the third value is reused to pick per-block color tricks.
fn glitchUV(uv: vec2f, px: vec2f, g: f32, tick: f32) -> vec3f {
  var q: vec2f = uv;
  // 1) horizontal slices (rows of random height)
  let rows: f32 = 9.0 + 14.0 * hash11(tick * 1.7);
  let row: f32 = floor(uv.y * rows);
  let hr: vec3f = blockRand(vec2f(row, 11.0), tick);
  if (hr.x < g * 0.35) { q.x += (hr.y - 0.5) * 0.22 * g; }
  // 2) rectangular blocks
  let bs: vec2f = vec2f(u.blockSize * 1.6, u.blockSize) * 0.06;
  let cell: vec2f = floor(uv / bs);
  let hb: vec3f = blockRand(cell, tick);
  if (hb.x < g * 0.22) { q += (hb.yz - 0.5) * vec2f(0.16, 0.035) * g; }
  // 3) scanline jitter: every pixel row gets a tiny random offset each frame
  let jr: f32 = hash21(vec2f(floor(px.y / 2.0), floor(u.time * 60.0)));
  q.x += (jr - 0.5) * 0.012 * u.jitter * g;
  return vec3f(q, hb.x);
}`;

const SOURCE_WGSL = /* wgsl */ `
// which image are we showing? per block during a transition, so A and B tear into each other
fn sceneAt(uv: vec2f, pickB: f32) -> vec3f {
  if (pickB > 0.5) { return TEX(neon, uv).rgb; }
  return TEX(game, uv).rgb;
}
fn pickB(uv: vec2f, tick: f32) -> f32 {
  let cell: vec2f = floor(uv / vec2f(0.1, 0.06));
  let h: f32 = hash33(vec3f(cell, tick + 91.0)).x;
  let b: f32 = step(h, u.progress);
  return select(b, 1.0 - b, u.fromB > 0.5);
}`;

const NEON_PASS = /* wgsl */ `
${NEON_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(neonCity(uv), 1.0); }`;

// datamosh: some blocks keep the PREVIOUS frame, pushed along a random "motion vector" — smearing old
// pixels across the new picture, like a video codec that lost its key frames.
const MOSH_PASS = /* wgsl */ `
${SOURCE_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let tick: f32 = floor(u.time * 8.0);
  let cell: vec2f = floor(uv / vec2f(0.0625, 0.0625 * u.resolution.x / u.resolution.y));
  let h: vec3f = hash33(vec3f(cell, floor(u.time * 2.0)));
  let mv: vec2f = (hash33(vec3f(floor(cell / 3.0), tick)).xy - 0.5) * vec2f(0.012, 0.006);
  let fresh: vec3f = sceneAt(uv, pickB(uv, floor(u.time * 12.0)));
  let stale: vec3f = TEX(mosh, uv - mv).rgb;
  let keep: f32 = step(h.x, u.mosh * u.glitch * 0.5) * step(1.5, u.frame);
  return vec4f(mix(fresh, stale, keep), 1.0);
}`;

const IMAGE = /* wgsl */ `
${SPLIT_WGSL}
${GLITCH_WGSL}
${SOURCE_WGSL}

fn spectrum(t: f32) -> vec3f {
  return clamp(vec3f(1.5 - abs(t * 3.0 - 0.5), 1.5 - abs(t * 3.0 - 1.5), 1.5 - abs(t * 3.0 - 2.5)), vec3f(0.0), vec3f(1.0));
}

// chromatic aberration of texture t at uv. mode: 0 radial (lens), 1 directional, 2 spectral (radial, many samples)
fn aberrate(uv: vec2f, amount: f32, mode: f32, px: vec2f) -> vec3f {
  let a: f32 = u.resolution.x / u.resolution.y;
  let d: vec2f = (uv - 0.5) * vec2f(a, 1.0);
  let r: f32 = length(d) / (0.5 * sqrt(a * a + 1.0));
  var off: vec2f = (uv - 0.5) * amount * 0.06 * pow(r, u.falloff);
  if (mode > 0.5 && mode < 1.5) { off = vec2f(cos(u.angle), sin(u.angle)) * amount * 0.012; }
  if (mode < 1.5) {
    return vec3f(TEX(game, uv + off).r, TEX(game, uv).g, TEX(game, uv - off).b);
  }
  var acc: vec3f = vec3f(0.0);
  var wsum: vec3f = vec3f(0.0);
  let n: f32 = u.samples;
  let j: f32 = ign(px);
  for (var i = 0; i < 32; i++) {
    if (f32(i) >= n) { break; }
    let t: f32 = (f32(i) + j) / n;
    let w: vec3f = spectrum(t);
    acc += TEX(game, uv + off * (1.0 - 2.0 * t)).rgb * w;
    wsum += w;
  }
  return acc / max(wsum, vec3f(0.0001));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex: i32 = i32(u.example);
  let orig: vec3f = TEX(game, uv).rgb;
  var c: vec3f = orig;

  if (ex == 0) {
    c = aberrate(uv, u.caAmount, u.caMode, px);
    c = splitView(c, orig, px, u.split);
    return vec4f(c, 1.0);
  }

  let tick: f32 = floor(u.time * 12.0);
  let g: f32 = u.glitch;
  if (ex == 1) {
    // DAMAGE HIT: shake + aberration spike + a few glitch blocks + red flash & vignette
    let q0: vec2f = uv + u.shakeOff;
    let gq: vec3f = glitchUV(q0, px, g * u.glitchAmt, tick);
    let ca: f32 = 0.15 + g * u.hitStrength * 2.5;
    let off: vec2f = (gq.xy - 0.5) * ca * 0.05 + vec2f(ca * 0.004, 0.0);
    c = vec3f(TEX(game, gq.xy + off).r, TEX(game, gq.xy).g, TEX(game, gq.xy - off).b);
    // desaturate & tint red, red vignette, white flash on the first frames
    let l: f32 = luma(c);
    c = mix(c, vec3f(l * 1.1, l * 0.35, l * 0.3), g * 0.45);
    let a: f32 = u.resolution.x / u.resolution.y;
    let vr: f32 = length((uv - 0.5) * vec2f(a, 1.0)) / (0.5 * sqrt(a * a + 1.0));
    c = mix(c, vec3f(0.55, 0.0, 0.02), smoothstep(0.35, 1.1, vr) * g * 0.9);
    c = mix(c, vec3f(1.0, 0.9, 0.9), u.flash * 0.6);
    return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
  }

  // CYBERPUNK GLITCH TRANSITION
  let gq: vec3f = glitchUV(uv, px, g, tick);
  let pb: f32 = pickB(gq.xy, tick);
  // RGB split that grows with the glitch, per block in random directions
  let dir: vec2f = (hash33(vec3f(floor(uv / vec2f(0.2, 0.1)), tick)).xy - 0.5);
  let off: vec2f = (vec2f(0.006, 0.0) + dir * 0.02) * g * u.rgbSplit;
  var src: vec3f;
  if (u.mosh > 0.001) {
    src = vec3f(TEX(mosh, gq.xy + off).r, TEX(mosh, gq.xy).g, TEX(mosh, gq.xy - off).b);
  } else {
    src = vec3f(sceneAt(gq.xy + off, pb).r, sceneAt(gq.xy, pb).g, sceneAt(gq.xy - off, pb).b);
  }
  c = src;
  // per-block color corruption: inverted, posterized or channel-swapped blocks
  if (gq.z < g * 0.035) { c = vec3f(1.0) - c; }
  else if (gq.z < g * 0.08) { c = floor(c * 3.0) / 3.0 * vec3f(0.3, 1.0, 0.9); }
  else if (gq.z < g * 0.11) { c = c.gbr; }
  // noise bars & scanlines
  let bar: f32 = step(0.97 - g * 0.08, hash21(vec2f(floor(uv.y * 80.0), tick)));
  c = mix(c, vec3f(hash21(px + vec2f(tick, 0.0))), bar * g * 0.8);
  c *= 1.0 - 0.12 * g * step(0.5, fract(px.y / 3.0));
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

function neonTextCanvas() {
  const W = 1024;
  const H = 256;
  const cv = makeCanvas(W, H);
  const g = cv.getContext('2d');
  g.clearRect(0, 0, W, H);
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.font = 'italic 900 132px system-ui, "Segoe UI", Roboto, sans-serif';
  g.shadowColor = '#ff2bd6';
  g.shadowBlur = 28;
  g.lineWidth = 6;
  g.strokeStyle = '#ff5ce1';
  g.strokeText('NEON//CITY', W / 2, H * 0.43);
  g.shadowBlur = 0;
  const grad = g.createLinearGradient(0, H * 0.2, 0, H * 0.65);
  grad.addColorStop(0, '#e9fdff');
  grad.addColorStop(1, '#41e8ff');
  g.fillStyle = grad;
  g.fillText('NEON//CITY', W / 2, H * 0.43);
  g.font = '700 34px ui-monospace, Menlo, Consolas, monospace';
  g.fillStyle = '#ffd2f5';
  g.fillText('SECTOR 7  ·  JACKING IN…', W / 2, H * 0.86);
  return cv;
}

export default shaderScene({
  interaction: 'Move the mouse to slide the before/after divider.',
  examples: [
    {
      id: 'aberration',
      label: 'Chromatic aberration',
      kind: 'Real life',
      note: 'Real lenses bend red and blue light by slightly different amounts, so colors separate towards the edges of the frame. Games add it on purpose for a camera / VHS / sci-fi feel (and turn it up when you get hit).',
      params: { caMode: 'radial', caAmount: 0.6, falloff: 1.5, compare: true },
      hint: 'Move the mouse to slide the before/after divider.',
    },
    {
      id: 'damage',
      label: 'Damage hit',
      kind: 'In a game',
      note: 'Taking damage, in one short pulse: screen shake, an aberration spike, a few glitch blocks, a red vignette and a one-frame flash. Everything decays back to normal in a fraction of a second. Click to get hit!',
      params: { hitStrength: 0.8, decay: 4, shake: 0.6, glitchAmt: 0.6, auto: true },
      hint: 'Click to take a hit.',
    },
    {
      id: 'glitch',
      label: 'Cyberpunk glitch transition',
      kind: 'In a game',
      note: 'A level change disguised as signal corruption: the picture tears into displaced slices and blocks, colors split and invert, and stale pixels smear (datamosh) while the new scene breaks through block by block.',
      params: { autoCycle: true, period: 3.5, blockSize: 1, jitter: 1, rgbSplit: 1, mosh: 0.6, amount: 0.5, progress: 0.5 },
      hint: '',
    },
  ],
  controls: [
    {
      type: 'select',
      key: 'caMode',
      label: 'Type',
      value: 'radial',
      options: [
        { value: 'radial', label: 'Radial (lens): grows towards the edges' },
        { value: 'directional', label: 'Directional (RGB split)' },
        { value: 'spectral', label: 'Spectral (smooth rainbow, many samples)' },
      ],
      showFor: ['aberration'],
    },
    { type: 'slider', key: 'caAmount', label: 'Amount', min: 0, max: 2, step: 0.01, value: 0.6, showFor: ['aberration'] },
    { type: 'slider', key: 'falloff', label: 'Edge falloff', min: 0, max: 4, step: 0.05, value: 1.5, showFor: ['aberration'], help: 'Radial / spectral: 0 = same everywhere, higher = only near the edges (like a real lens).' },
    { type: 'slider', key: 'angle', label: 'Direction', min: 0, max: 6.283, step: 0.01, value: 0, format: (v) => `${Math.round((v * 180) / Math.PI)}°`, showFor: ['aberration'], help: 'Directional only.' },
    { type: 'slider', key: 'samples', label: 'Spectral samples', min: 3, max: 32, step: 1, value: 12, showFor: ['aberration'], help: 'Spectral only. 3 = plain RGB split; more = smooth rainbow fringes.' },
    { type: 'toggle', key: 'compare', label: 'Before / after divider', value: true, showFor: ['aberration'] },
    { type: 'slider', key: 'hitStrength', label: 'Aberration spike', min: 0, max: 2, step: 0.01, value: 0.8, showFor: ['damage'] },
    { type: 'slider', key: 'decay', label: 'Recovery speed', min: 1, max: 12, step: 0.1, value: 4, showFor: ['damage'], help: 'How fast the hit fades (per second). Short = punchy.' },
    { type: 'slider', key: 'shake', label: 'Screen shake', min: 0, max: 2, step: 0.01, value: 0.6, showFor: ['damage'] },
    { type: 'slider', key: 'glitchAmt', label: 'Glitch blocks', min: 0, max: 1.5, step: 0.01, value: 0.6, showFor: ['damage'] },
    { type: 'toggle', key: 'auto', label: 'Auto-hit every 2.5 s', value: true, showFor: ['damage'] },
    { type: 'toggle', key: 'autoCycle', label: 'Auto-play transition', value: true, showFor: ['glitch'], help: 'Off = a steady glitch at the intensity below.' },
    { type: 'slider', key: 'period', label: 'Seconds per scene', min: 1.5, max: 8, step: 0.1, value: 3.5, showFor: ['glitch'] },
    { type: 'slider', key: 'amount', label: 'Manual intensity', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['glitch'], help: 'Used when auto-play is off.' },
    { type: 'slider', key: 'progress', label: 'Manual progress', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['glitch'], help: 'Share of blocks already showing the new scene (auto-play off).' },
    { type: 'slider', key: 'blockSize', label: 'Block size', min: 0.3, max: 3, step: 0.01, value: 1, showFor: ['glitch'] },
    { type: 'slider', key: 'jitter', label: 'Scanline jitter', min: 0, max: 3, step: 0.01, value: 1, showFor: ['glitch'] },
    { type: 'slider', key: 'rgbSplit', label: 'RGB split', min: 0, max: 3, step: 0.01, value: 1, showFor: ['glitch'] },
    { type: 'slider', key: 'mosh', label: 'Datamosh smear', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['glitch'], help: 'Blocks that keep last frame’s pixels and drift — feedback through the previous frame.' },
  ],
  uniforms: {
    caMode: 'f32', caAmount: 'f32', falloff: 'f32', angle: 'f32', samples: 'f32',
    hitStrength: 'f32', glitchAmt: 'f32', blockSize: 'f32', jitter: 'f32', rgbSplit: 'f32', mosh: 'f32',
    split: 'f32', glitch: 'f32', flash: 'f32', shakeOff: 'vec2f', progress: 'f32', fromB: 'f32',
  },
  include: withGameIncludes(['hash', 'color']),
  textures: { neonText: { source: async () => neonTextCanvas() } },
  passes: [
    gamePass(),
    { name: 'neon', format: 'rgba8unorm', iterations: () => (cur.ex === 'glitch' ? 1 : 0), code: NEON_PASS },
    { name: 'mosh', iterations: () => (cur.ex === 'glitch' ? 1 : 0), code: MOSH_PASS },
  ],
  resetOnExample: false,
  bind(p, ctx) {
    cur.ex = ctx.example;
    ov.begin(ctx);
    const t = ctx.time;
    const out = { split: -1, glitch: 0, flash: 0, shakeOff: [0, 0], progress: 0, fromB: 0 };
    if (ctx.example === 'aberration') out.split = compareSplit(st, ctx, p.compare, ov, ['Original', 'Aberration']);
    if (ctx.example === 'damage') {
      if (ctx.pointer.clicked) st.hitT = t;
      if (p.auto && t - st.hitT > 2.5) st.hitT = t;
      const age = Math.max(0, t - st.hitT);
      let env = Math.exp(-age * p.decay);
      if (ctx.testMode) env = 0.75; // the headless test renders ~2 fps: hold the pulse so the screenshot shows it
      out.glitch = env;
      out.flash = age < 0.06 ? 1 : 0;
      const tick = Math.floor(t * 40);
      if (tick !== st.shakeT) {
        st.shakeT = tick;
        const s = p.shake * env * 0.025;
        st.shake = [(Math.random() - 0.5) * s * (ctx.height / ctx.width), (Math.random() - 0.5) * s];
      }
      out.shakeOff = st.shake;
      ov.show('hp', `HP ${'█'.repeat(Math.max(1, 8 - (Math.floor(t / 2.5) % 8)))}`, `right:8px;bottom:8px;color:${env > 0.3 ? '#ff6b6b' : '#fff'}`);
    }
    if (ctx.example === 'glitch') {
      if (p.autoCycle) {
        const T = p.period;
        const k = Math.floor(t / T);
        const tau = t - k * T;
        const w = 0.55; // transition half-width in seconds
        const mid = T * 0.5;
        const env = Math.max(0, 1 - Math.abs(tau - mid) / w);
        // tiny random micro-glitches while idle
        const micro = (Math.sin(Math.floor(t * 5) * 91.7) * 43758.5) % 1 > 0.82 ? 0.12 : 0;
        out.glitch = Math.max(Math.pow(env, 0.6), micro);
        out.progress = Math.min(1, Math.max(0, (tau - (mid - w * 0.55)) / (w * 1.1)));
        out.fromB = k % 2;
        if (ctx.testMode) {
          out.glitch = 0.6;
          out.progress = 0.5;
        }
      } else {
        out.glitch = p.amount;
        out.progress = p.progress;
      }
    }
    ov.end();
    return out;
  },
  code: IMAGE,
  about: {
    summary:
      'Chromatic aberration splits the color channels apart; glitch effects deliberately break the image the way damaged video and corrupted data do. Both are cheap and instantly read as “camera”, “damage” or “cyberpunk”.',
    what: `<p><b>Chromatic aberration</b> shows the three common ways to split colors. <b>Damage hit</b> is a short “you got hurt” pulse built from
      several tiny effects at once. <b>Cyberpunk glitch transition</b> uses glitches to tear from the platformer into a second scene.</p>`,
    how: `<ol>
      <li><b>RGB split</b>: read the red channel at <code>uv + offset</code>, green at <code>uv</code>, blue at <code>uv − offset</code>.
        For a lens-like (radial) look the offset points away from the centre and grows towards the edges.</li>
      <li><b>Spectral</b> aberration takes many samples along the offset and weights each by a rainbow color — smooth fringes instead of three ghost copies.</li>
      <li><b>Glitch blocks</b>: divide the screen into rows and rectangles; hash each one together with a <i>tick</i> (time rounded down, 12×/s) to get
        random numbers that jump instead of flowing. Blocks whose number is below the glitch intensity get shifted, inverted, posterized or channel-swapped.</li>
      <li><b>Scanline jitter</b>: every row of pixels gets its own tiny horizontal offset, re-rolled every frame.</li>
      <li><b>Datamosh</b>: a feedback pass. Most blocks take the new frame, but some keep <i>last frame’s</i> pixels, nudged by a random “motion vector”, so old images smear across new ones — just like a video file that lost its key frames.</li>
      <li><b>Transition</b>: during the glitch peak each block independently picks scene A or scene B, with the share of B blocks going from 0 to 1.</li>
    </ol>`,
    uses: [
      { title: 'Feedback', text: 'Damage, low health, EMP, hacking, being spotted by a camera.' },
      { title: 'Style', text: 'Cyberpunk, sci-fi HUDs, found-footage horror, VHS / analog horror, “the simulation is breaking”.' },
      { title: 'Transitions', text: 'Level loads, teleports, fast travel and menu swaps disguised as signal loss.' },
      { title: 'Camera realism', text: 'A touch of lens aberration at the frame edges makes a clean render feel photographed.' },
    ],
    try: [
      'On <b>Chromatic aberration</b> set Type to Spectral and Amount to 2, then drop Samples to 3 — the smooth rainbow becomes three hard copies.',
      'Set Edge falloff to 0: the whole image splits equally, which looks like a broken display rather than a lens.',
      'On <b>Damage hit</b> raise Recovery speed to 12 for a snappy arcade hit, or 1.5 for a heavy, groggy one.',
      'On the <b>glitch transition</b> set Datamosh to 1 and RGB split to 0: pure smearing, no color tricks.',
    ],
    ask: [
      'a damage hit effect: screen shake, chromatic aberration spike, red vignette',
      'subtle radial chromatic aberration at the screen edges',
      'a glitch transition between levels with block displacement and RGB split',
      'a datamosh smear using the previous frame',
      'scanline jitter and noise bars when the signal is lost',
    ],
    perf: `<p>RGB split costs 3 texture reads instead of 1; spectral costs N. Glitch logic is a few hashes per pixel. Datamosh adds one full-screen feedback pass.
      All of it is negligible next to rendering the game itself.</p>`,
    api: `<p>Works the same on WebGPU and WebGL2: everything is fragment-shader math plus one double-buffered feedback texture for the datamosh.</p>`,
    code: [
      {
        title: 'Radial chromatic aberration',
        lang: 'wgsl',
        src: `let r = length((uv - 0.5) * vec2f(aspect, 1.0)) / halfDiagonal;   // 0 centre .. 1 corner
let off = (uv - 0.5) * amount * 0.06 * pow(r, falloff);          // points outward, grows to the edges
col = vec3f(TEX(game, uv + off).r, TEX(game, uv).g, TEX(game, uv - off).b);`,
      },
      {
        title: 'Glitch blocks that jump 12 times per second',
        lang: 'wgsl',
        src: `let tick = floor(u.time * 12.0);                       // time in steps, not smooth
let cell = floor(uv / blockSize);
let h = hash33(vec3f(cell, tick));                       // new random numbers every tick
if (h.x < g * 0.3) { q += (h.yz - 0.5) * vec2f(0.18, 0.04) * g; }   // shift some blocks
q.x += (hash21(vec2f(floor(px.y / 2.0), floor(u.time * 60.0))) - 0.5) * 0.012 * g;  // scanline jitter`,
      },
      {
        title: 'Datamosh feedback pass',
        lang: 'wgsl',
        src: `let fresh = sceneAt(uv, pickB(uv, tick));      // this frame
let stale = TEX(mosh, uv - motionVector).rgb;     // last frame, nudged
let keep = step(h.x, u.mosh * u.glitch);          // some blocks refuse to update
return vec4f(mix(fresh, stale, keep), 1.0);`,
      },
    ],
  },
});

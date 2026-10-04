import { shaderScene } from '../../core/shaderscene.js';
import { setTag, placeTag, makeSplit, gamePass, GAME_INCLUDES, noiseTexture, NOISE_TEX_WGSL } from './_shared-a.js';

// Painterly filters built on the (anisotropic) Kuwahara filter:
//   scene : the platformer (or, on the "How it works" tab, a noisy test card)
//   sst   : structure tensor — how strongly and in which direction the image changes (Sobel)
//   tfm   : the tensor blurred, then its eigenvectors: stroke direction t + anisotropy A
//   paint : Kuwahara: for each pixel look at several overlapping sectors of a window (an ellipse
//           stretched along t), and output the mean color of the sectors with the lowest variance.
//           Flat areas get smoothed, edges are preserved → blobs of paint.
//   image : oil (impasto relief + brush streaks + canvas), watercolor (bleed, pooled edges,
//           granulation, paper), pencil (hatching by tone, outlines, grain, smudge), or the explainer.

const st = { testMode: false };
// Unrolled code generation: every array index becomes a constant (no dynamically indexed local arrays).
const UNROLL8 = (f) => Array.from({ length: 8 }, (_, k) => f(k)).join('\n');
const splitPos = makeSplit(0.5);
const RES_SCALE = [1, 0.5, 0.34];
const KUWA = ['Classic (4 boxes)', 'Generalized (8 sectors)', 'Anisotropic (follows edges)'];

const paintScale = (params, ctx) => (ctx?.testMode ? 0.25 : RES_SCALE[params.res | 0] ?? 0.5);
const paintSize = (params, ctx) => {
  const s = paintScale(params, ctx);
  return [Math.max(8, Math.round(ctx.width * s)), Math.max(8, Math.round(ctx.height * s))];
};

const SRC = /* wgsl */ `
// Abstract test card: bold shapes + heavy per-pixel noise (shows edge-preserving smoothing)
fn card(p: vec2f) -> vec3f {
  let res = u.resolution;
  let q = (p - 0.5 * res) / res.y;
  var c = mix(vec3f(0.95, 0.88, 0.7), vec3f(0.55, 0.75, 0.9), smoothstep(-0.5, 0.5, q.y));
  let stripes = step(0.5, fract((q.x + q.y) * 6.0));
  c = mix(c, mix(vec3f(0.2, 0.35, 0.75), vec3f(0.92, 0.94, 1.0), stripes), step(0.25, q.y) * step(q.x, -0.05));
  c = mix(c, vec3f(0.95, 0.45, 0.15), step(length(q - vec2f(-0.35, -0.08)), 0.22));
  let r = rot2(0.5) * (q - vec2f(0.32, 0.02));
  c = mix(c, vec3f(0.1, 0.6, 0.55), step(max(abs(r.x), abs(r.y)), 0.17));
  c = mix(c, vec3f(0.85, 0.2, 0.5), step(sdEquilateralTriangle(q - vec2f(0.0, -0.2), 0.12), 0.0));
  c = mix(c, vec3f(0.15, 0.12, 0.2), step(abs(length(q - vec2f(0.32, 0.02)) - 0.3), 0.012));
  let n = hash23(floor(p) + vec2f(0.5)) - vec3f(0.5);
  return clamp(c + n * 0.38, vec3f(0.0), vec3f(1.0));
}
fn srcAt(uv: vec2f) -> vec3f {
  if (u.example > 2.5) { return card(uv * u.resolution); }
  return TEX(scene, uv).rgb;
}
`;

export default shaderScene({
  interaction: '',
  examples: [
    {
      id: 'oil',
      label: 'Oil painting',
      kind: 'In a game',
      hint: '',
      note: 'An <b>anisotropic Kuwahara</b> filter turns the game into blobs and strokes of paint that follow the shapes, then lighting on the paint’s height adds impasto ridges, brush streaks and canvas weave.',
      params: { kuwa: 2, radius: 5, res: 1, relief: 1.0, strokes: 0.6 },
    },
    {
      id: 'watercolor',
      label: 'Watercolor',
      kind: 'Real life',
      hint: '',
      note: 'Simplified washes (Kuwahara) sampled through a noise warp so colors <b>bleed</b> past their edges, pigment <b>pooling</b> darker at the rim of each wash, <b>granulation</b> in the paper’s valleys, and white paper showing through.',
      params: { kuwa: 2, radius: 4, res: 1, bleed: 0.6, edgeDark: 0.7, gran: 0.6 },
    },
    {
      id: 'sketch',
      label: 'Pencil sketch',
      kind: 'Classic',
      hint: '',
      note: 'Brightness decides how many layers of <b>cross-hatching</b> are drawn (light: one direction, dark: four), outlines come from edges, and paper grain breaks up the graphite. Smudging adds soft tone.',
      params: { kuwa: 2, radius: 3, res: 1, hatch: 6, smudge: 0.5, gran: 0.6 },
    },
    {
      id: 'explain',
      label: 'How Kuwahara works',
      kind: 'Abstract',
      hint: 'Drag to move the divider. Hover to see the filter window.',
      note: 'A noisy test card: raw on the left, filtered on the right. Hover to see the <b>window</b> (3× enlarged): it is split into sectors, and the result is the average of the <i>calmest</i> sectors. Near an edge only the sectors on one side are calm, so the edge stays sharp.',
      params: { kuwa: 2, radius: 5, res: 1 },
    },
  ],
  controls: [
    {
      type: 'select', key: 'kuwa', label: 'Kuwahara variant', value: 2, showFor: ['oil', 'explain'],
      options: KUWA.map((label, value) => ({ value, label })),
      help: 'Classic = blocky; generalized = round, smoother; anisotropic = stretched along edges like brush strokes.',
    },
    { type: 'slider', key: 'radius', label: 'Brush radius', min: 2, max: 8, step: 1, value: 5, help: 'Window radius in filter pixels. Cost grows with radius².' },
    {
      type: 'select', key: 'res', label: 'Filter resolution', value: 1,
      options: [{ value: 0, label: 'Full (most expensive)' }, { value: 1, label: 'Half (4× cheaper)' }, { value: 2, label: 'Third (9× cheaper)' }],
      help: 'Run the filter on a smaller image: bigger strokes for less work.',
    },
    { type: 'slider', key: 'relief', label: 'Impasto relief', min: 0, max: 2.5, step: 0.01, value: 1, showFor: ['oil'], help: 'Light the paint as if it had thickness.' },
    { type: 'slider', key: 'strokes', label: 'Brush streaks', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['oil'], help: 'Bristle grooves smeared along the stroke direction.' },
    { type: 'slider', key: 'bleed', label: 'Color bleed', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['watercolor'], help: 'Noise-warped sampling: colors run past their edges.' },
    { type: 'slider', key: 'edgeDark', label: 'Edge darkening', min: 0, max: 1, step: 0.01, value: 0.7, showFor: ['watercolor'], help: 'Pigment pools at the drying rim of each wash.' },
    { type: 'slider', key: 'gran', label: 'Granulation & paper', min: 0, max: 1, step: 0.01, value: 0.6, showFor: ['watercolor', 'sketch'] },
    { type: 'slider', key: 'hatch', label: 'Hatching spacing (px)', min: 3, max: 12, step: 0.5, value: 6, showFor: ['sketch'] },
    { type: 'slider', key: 'smudge', label: 'Smudge', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['sketch'] },
  ],
  uniforms: {
    kuwa: 'f32', radius: 'f32', res: 'f32', relief: 'f32', strokes: 'f32', bleed: 'f32', edgeDark: 'f32', gran: 'f32',
    hatch: 'f32', smudge: 'f32', split: 'f32',
  },
  include: ['color', ...GAME_INCLUDES],
  resetOn: ['kuwa', 'radius', 'res', 'relief', 'strokes', 'bleed', 'edgeDark', 'gran', 'hatch', 'smudge'],
  renderScale: () => (st.testMode ? 0.5 : 1),
  textures: { noiseTex: { source: async () => noiseTexture(), filter: 'linear', wrap: 'repeat' } },
  passes: [
    gamePass('scene'),
    {
      // structure tensor: products of the image gradient (summed over R, G, B)
      name: 'sst',
      size: paintSize,
      code: /* wgsl */ `
${SRC}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let d = 1.0 / TEXSIZE(sst);
  let tl = srcAt(uv + vec2f(-d.x, -d.y)); let tt = srcAt(uv + vec2f(0.0, -d.y)); let tr = srcAt(uv + vec2f(d.x, -d.y));
  let ml = srcAt(uv + vec2f(-d.x, 0.0));                                       let mr = srcAt(uv + vec2f(d.x, 0.0));
  let bl = srcAt(uv + vec2f(-d.x, d.y));  let bb = srcAt(uv + vec2f(0.0, d.y));  let br = srcAt(uv + vec2f(d.x, d.y));
  let gx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * 0.25;
  let gy = ((bl + 2.0 * bb + br) - (tl + 2.0 * tt + tr)) * 0.25;
  return vec4f(dot(gx, gx), dot(gy, gy), dot(gx, gy), 1.0);
}`,
    },
    {
      // smooth the tensor, then eigen-decompose: t = stroke (edge tangent) direction, A = anisotropy
      name: 'tfm',
      size: paintSize,
      code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let d = 1.0 / TEXSIZE(tfm);
  var g = vec3f(0.0);
  var ws = 0.0;
  for (var j = -2; j <= 2; j++) {
    for (var i = -2; i <= 2; i++) {
      let o = vec2f(f32(i), f32(j)) * 1.5;
      let w = exp(-dot(o, o) / 8.0);
      g += TEX(sst, uv + o * d).xyz * w;
      ws += w;
    }
  }
  g /= ws;
  let E = g.x;
  let G = g.y;
  let F = g.z;
  let root = sqrt((E - G) * (E - G) + 4.0 * F * F);
  let l1 = 0.5 * (E + G + root);
  let l2 = 0.5 * (E + G - root);
  var t = vec2f(l1 - E, -F);
  if (length(t) > 0.000001) { t = normalize(t); } else { t = vec2f(0.0, 1.0); }
  var A = 0.0;
  if (l1 + l2 > 0.0000001) { A = (l1 - l2) / (l1 + l2); }
  return vec4f(t, A, 1.0);
}`,
    },
    {
      name: 'paint',
      size: paintSize,
      code: /* wgsl */ `
${SRC}
// Classic Kuwahara: 4 square quadrants, pick the mean of the one with the lowest variance.
fn classicK(uv: vec2f, texel: vec2f, r: i32) -> vec4f {
  var m = array<vec3f, 4>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var s = array<vec3f, 4>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var n = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var j = -r; j <= r; j++) {
    for (var i = -r; i <= r; i++) {
      let c = srcAt(uv + vec2f(f32(i), f32(j)) * texel);
      if (i <= 0 && j <= 0) { m[0] += c; s[0] += c * c; n[0] += 1.0; }
      if (i >= 0 && j <= 0) { m[1] += c; s[1] += c * c; n[1] += 1.0; }
      if (i <= 0 && j >= 0) { m[2] += c; s[2] += c * c; n[2] += 1.0; }
      if (i >= 0 && j >= 0) { m[3] += c; s[3] += c * c; n[3] += 1.0; }
    }
  }
  var best = 1.0e9;
  var outc = vec3f(0.0);
  for (var k = 0; k < 4; k++) {
    let mean = m[k] / n[k];
    let v = s[k] / n[k] - mean * mean;
    let sv = v.r + v.g + v.b;
    if (sv < best) { best = sv; outc = mean; }
  }
  return vec4f(outc, 1.0);
}
// mean & variance of one sector → weighted contribution (calm sectors dominate)
fn sectorResult(mk: vec4f, sk: vec3f) -> vec4f {
  let mw = max(mk.w, 0.000001);
  let mean = mk.rgb / mw;
  let v3 = abs(sk / mw - mean * mean);
  let sigma2 = v3.r + v3.g + v3.b;
  let wk = 1.0 / (1.0 + pow(8000.0 * sigma2, 4.0));
  return vec4f(mean * wk, wk);
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let texel = 1.0 / TEXSIZE(paint);
  let r = u.radius;
  if (u.kuwa < 0.5) { return classicK(uv, texel, i32(r)); }
  // Generalized / anisotropic Kuwahara with polynomial sector weights (Kyprianidis et al.)
  let tf = TEX(tfm, uv);
  var A = tf.z;
  if (u.kuwa < 1.5) { A = 0.0; }
  let a = r * clamp(1.0 + A, 0.1, 2.0);            // ellipse: long along the edge…
  let b = r * clamp(1.0 / (1.0 + A), 0.1, 2.0);    // …short across it
  let t = tf.xy;
  let tp = vec2f(-t.y, t.x);
  let maxX = i32(sqrt(a * a * t.x * t.x + b * b * t.y * t.y));
  let maxY = i32(sqrt(a * a * t.y * t.y + b * b * t.x * t.x));
  let zeta = 2.0 / r;
  let zc = 0.58;
  let eta = (zeta + cos(zc)) / (sin(zc) * sin(zc));
  var m: array<vec4f, 8>;
  var s: array<vec3f, 8>;
${UNROLL8((k) => `  m[${k}] = vec4f(0.0); s[${k}] = vec3f(0.0);`)}
  var w = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var j = -maxY; j <= maxY; j++) {
    for (var i = -maxX; i <= maxX; i++) {
      let o = vec2f(f32(i), f32(j));
      var v = vec2f(dot(o, t) * 0.5 / a, dot(o, tp) * 0.5 / b);   // offset in the unit-circle frame
      if (dot(v, v) <= 0.25) {
        let c = clamp(srcAt(uv + o * texel), vec3f(0.0), vec3f(1.0));
        var sum = 0.0;
        var vxx = zeta - eta * v.x * v.x;
        var vyy = zeta - eta * v.y * v.y;
        var z = max(0.0, v.y + vxx); w[0] = z * z; sum += w[0];
        z = max(0.0, -v.x + vyy); w[2] = z * z; sum += w[2];
        z = max(0.0, -v.y + vxx); w[4] = z * z; sum += w[4];
        z = max(0.0, v.x + vyy); w[6] = z * z; sum += w[6];
        v = 0.70710678 * vec2f(v.x - v.y, v.x + v.y);
        vxx = zeta - eta * v.x * v.x;
        vyy = zeta - eta * v.y * v.y;
        z = max(0.0, v.y + vxx); w[1] = z * z; sum += w[1];
        z = max(0.0, -v.x + vyy); w[3] = z * z; sum += w[3];
        z = max(0.0, -v.y + vxx); w[5] = z * z; sum += w[5];
        z = max(0.0, v.x + vyy); w[7] = z * z; sum += w[7];
        let g = exp(-3.125 * dot(v, v)) / max(sum, 0.000001);
        let cc = c * c;
${UNROLL8((k) => `        m[${k}] += vec4f(c, 1.0) * (w[${k}] * g); s[${k}] += cc * (w[${k}] * g);`)}
      }
    }
  }
  var outc = vec4f(0.0);
${UNROLL8((k) => `  outc += sectorResult(m[${k}], s[${k}]);`)}
  return vec4f(clamp(outc.rgb / max(outc.w, 1.0e-20), vec3f(0.0), vec3f(1.0)), 1.0);
}`,
    },
  ],
  bind(params, ctx) {
    st.testMode = !!ctx.testMode;
    const out = { split: splitPos(ctx) };
    if (ctx.testMode) out.radius = Math.min(params.radius, 3);
    const [w, h] = paintSize(params, ctx);
    const isEx = ctx.example === 'explain';
    const a = setTag(ctx, 'before', 'Noisy original', '', isEx);
    const b = setTag(ctx, 'after', KUWA[params.kuwa | 0], '', isEx);
    if (isEx) {
      placeTag(a, out.split * 100, '48px', 'calc(-100% - 10px)');
      placeTag(b, out.split * 100, '48px', '10px');
    }
    const r = out.radius ?? params.radius;
    const taps = (params.kuwa | 0) === 0 ? (2 * r + 1) ** 2 : Math.round(Math.PI * r * r); // ellipse area = π·a·b = π·r²
    setTag(ctx, 'info', `filter ${w}×${h} · ~${taps} taps/pixel`, 'right:8px;bottom:8px');
    return out;
  },
  code: /* wgsl */ `
${SRC}
${NOISE_TEX_WGSL}
fn paintAt(uv: vec2f) -> vec3f { return TEX(paint, uv).rgb; }

// ------------------------------------------------------------------ oil
fn oilView(uv: vec2f, p: vec2f) -> vec3f {
  let res = u.resolution;
  var c = paintAt(uv);
  let e = 1.0 / TEXSIZE(paint);
  // stroke direction from the orientation field
  let t = TEX(tfm, uv).xy;
  // bristle streaks: noise averaged along the stroke direction (a tiny line-integral convolution)
  var streak = 0.0;
  for (var k = -3; k <= 3; k++) {
    let q = p + t * f32(k) * 2.5;
    streak += tnoise(q * 0.45) + 0.5 * tnoise(q * 1.1 + vec2f(13.0, 7.0));
  }
  streak = streak / 10.5;
  // height = paint lightness + streaks → bumps → lighting (impasto)
  let px1 = vec2f(e.x, 0.0);
  let py1 = vec2f(0.0, e.y);
  let hx = luma(paintAt(uv + px1)) - luma(paintAt(uv - px1));
  let hy = luma(paintAt(uv + py1)) - luma(paintAt(uv - py1));
  var sx = 0.0;
  var sy = 0.0;
  for (var k = -3; k <= 3; k++) {
    let q = p + t * f32(k) * 2.5;
    sx += tnoise((q + vec2f(1.0, 0.0)) * 0.45) - tnoise((q - vec2f(1.0, 0.0)) * 0.45);
    sy += tnoise((q + vec2f(0.0, 1.0)) * 0.45) - tnoise((q - vec2f(0.0, 1.0)) * 0.45);
  }
  let grad = vec2f(hx, hy) * 1.5 + vec2f(sx, sy) * 0.08 * u.strokes;
  let n = normalize(vec3f(-grad * u.relief * 3.0, 1.0));
  let L = normalize(vec3f(-0.55, -0.65, 0.55));
  let diff = dot(n, L) - L.z;
  let spec = pow(max(dot(n, normalize(L + vec3f(0.0, 0.0, 1.0))), 0.0), 24.0);
  c = c * (1.0 + diff * 0.9) + vec3f(spec) * 0.12 * u.relief;
  c *= 0.94 + 0.08 * streak * u.strokes;
  // canvas weave
  let weave = sin(p.x * 1.9) * sin(p.y * 1.9);
  c *= 1.0 - 0.05 * (weave * 0.5 + 0.5) * (1.0 - smoothstep(0.0, 2.0, u.relief) * 0.3);
  c = adjustSaturation(c, 1.12);
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

// ------------------------------------------------------------------ watercolor
fn paperBump(p: vec2f) -> f32 {
  return tnoise(p * 0.35) * 0.5 + tnoise(p * 0.9) * 0.3 + tnoise(p * 2.3) * 0.2;
}
fn waterView(uv: vec2f, p: vec2f) -> vec3f {
  let res = u.resolution;
  // bleeding: sample the simplified image through a smooth noise warp
  let wq = p * 0.012;
  let warp = vec2f(tfbm(wq), tfbm(wq + vec2f(5.2, 1.3))) * u.bleed * 14.0 / res;
  let c = paintAt(uv + warp);
  // pooled edges: compare with the average around → where colors change, pigment collects
  var avg = vec3f(0.0);
  for (var k = 0; k < 6; k++) {
    let a = f32(k) * 1.0472;
    avg += paintAt(uv + warp + vec2f(cos(a), sin(a)) * 4.0 / res);
  }
  avg /= 6.0;
  let edge = clamp(length(c - avg) * 4.0, 0.0, 1.0);
  // paper & granulation: pigment settles into the paper's valleys
  let bump = paperBump(p);
  let gran = (tnoise(p * 0.6) - 0.5) * 0.8 + (bump - 0.5);
  // transparent color: treat the image as transmittance and vary pigment density
  let wash = mix(vec3f(1.0), c, 0.82);               // lighter than the original: transparent paint
  // uneven drying: low-frequency blotches ("blooms") in every wash
  let bloom = tfbm(p * 0.006 + vec2f(2.7, 9.1));
  let density = 0.8 + bloom * 0.45 + edge * 1.6 * u.edgeDark + gran * 0.5 * u.gran;
  let paper = vec3f(0.98, 0.965, 0.93) * (0.95 + 0.05 * bump * u.gran + 0.0);
  var col = paper * pow(clamp(wash, vec3f(0.001), vec3f(1.0)), vec3f(max(density, 0.2)));
  // highlights: let the white paper show where the scene is very light
  col = mix(col, paper, smoothstep(0.88, 0.97, luma(c)) * 0.7);
  // faint pencil under-drawing
  let ex = abs(luma(paintAt(uv + vec2f(1.5, 0.0) / res)) - luma(paintAt(uv - vec2f(1.5, 0.0) / res)));
  let ey = abs(luma(paintAt(uv + vec2f(0.0, 1.5) / res)) - luma(paintAt(uv - vec2f(0.0, 1.5) / res)));
  col *= 1.0 - smoothstep(0.08, 0.3, ex + ey) * 0.25;
  return col;
}

// ------------------------------------------------------------------ pencil sketch
fn hatchLayer(p: vec2f, ang: f32, period: f32, seed: f32) -> f32 {
  var q = rot2(ang) * p;
  q.y += (tnoise(vec2f(q.x * 0.02, seed)) - 0.5) * period * 0.8;   // hand wobble
  let row = floor(q.y / period);
  let d = abs(fract(q.y / period) - 0.5) * period;
  let line = 1.0 - smoothstep(0.35, 1.1, d);
  // strokes have ends: break the line into dashes of varying length
  let dash = smoothstep(0.3, 0.55, tnoise(vec2f(q.x * 0.07 + row * 7.3, row * 1.7 + seed)));
  return line * dash;
}
fn sketchView(uv: vec2f, p: vec2f) -> vec3f {
  let res = u.resolution;
  let c = paintAt(uv);
  let l = clamp((luma(c) - 0.5) * 1.4 + 0.64, 0.0, 1.0);
  let per = u.hatch;
  var g = 0.0;
  g = max(g, hatchLayer(p, 0.785, per, 1.0) * (1.0 - smoothstep(0.7, 0.86, l)));
  g = max(g, hatchLayer(p, -0.785, per, 2.0) * (1.0 - smoothstep(0.5, 0.66, l)));
  g = max(g, hatchLayer(p, 0.15, per * 0.85, 3.0) * (1.0 - smoothstep(0.3, 0.46, l)));
  g = max(g, hatchLayer(p, 1.4, per * 0.7, 4.0) * (1.0 - smoothstep(0.12, 0.28, l)));
  // outlines: edges of the simplified image, with a slight wobble
  let jit = (vec2f(tnoise(p * 0.05), tnoise(p * 0.05 + 9.0)) - 0.5) * 2.0 / res;
  let e = 1.2 / res;
  let gx = luma(paintAt(uv + jit + vec2f(e.x, 0.0))) - luma(paintAt(uv + jit - vec2f(e.x, 0.0)));
  let gy = luma(paintAt(uv + jit + vec2f(0.0, e.y))) - luma(paintAt(uv + jit - vec2f(0.0, e.y)));
  let outline = smoothstep(0.06, 0.2, length(vec2f(gx, gy)));
  // paper grain: graphite only catches on the paper's peaks
  let grain = paperBump(p * 1.6);
  let tooth = mix(1.0, 0.45 + 0.75 * grain, u.gran);
  var ink = max(g * 0.75, outline * 0.9) * tooth;
  // smudge: soft graphite tone in the darker areas, blotchy
  let blot = tnoise(p * 0.015) * 0.6 + tnoise(p * 0.04) * 0.4;
  let smear = (1.0 - l) * u.smudge * (0.35 + 0.4 * blot);
  ink = 1.0 - (1.0 - ink) * (1.0 - smear * 0.55);
  let paper = vec3f(0.96, 0.94, 0.89) * (0.96 + 0.06 * grain);
  let graphite = vec3f(0.2, 0.21, 0.25);
  return mix(paper, graphite, clamp(ink, 0.0, 1.0));
}

// ------------------------------------------------------------------ explainer
fn explainView(uv: vec2f, p: vec2f) -> vec3f {
  let res = u.resolution;
  let sx = u.split * res.x;
  var c = paintAt(uv);
  if (p.x < sx) { c = srcAt(uv); }
  c = mix(c, vec3f(1.0), 1.0 - smoothstep(1.0, 2.5, abs(p.x - sx)));
  // the filter window at the mouse, 3× enlarged
  if (u.mouse.w > 0.5) {
    let m = u.mouse.xy;
    let toScreen = res.y / TEXSIZE(paint).y * 3.0;
    let d = p - m;
    let r = u.radius;
    var inside = 0.0;
    var line = 0.0;
    if (u.kuwa < 0.5) {
      let hw = (r + 0.5) * toScreen;
      let bd = max(abs(d.x), abs(d.y));
      inside = step(bd, hw);
      line = max(1.0 - smoothstep(1.0, 2.0, abs(bd - hw)), (1.0 - smoothstep(1.0, 2.0, min(abs(d.x), abs(d.y)))) * inside);
    } else {
      let tf = TEX(tfm, m / res);
      var A = tf.z;
      if (u.kuwa < 1.5) { A = 0.0; }
      let a = r * clamp(1.0 + A, 0.1, 2.0) * toScreen;
      let b = r * clamp(1.0 / (1.0 + A), 0.1, 2.0) * toScreen;
      let t = tf.xy;
      let q = vec2f(dot(d, t) / a, dot(d, vec2f(-t.y, t.x)) / b);
      let ql = length(q);
      inside = step(ql, 1.0);
      line = 1.0 - smoothstep(1.0, 2.0, abs(ql - 1.0) * min(a, b));
      // 8 sector boundaries
      let ang = atan2(q.y, q.x) / (TAU / 8.0) + 0.5;
      let sd = abs(fract(ang) - 0.5) * (TAU / 8.0) * ql * min(a, b);
      line = max(line, (1.0 - smoothstep(0.5, 1.5, sd)) * inside * 0.7);
    }
    c = mix(c, c * 0.55 + vec3f(0.0, 0.0, 0.08), inside * 0.35);
    c = mix(c, vec3f(1.0), line);
  }
  return c;
}

fn shade(uv: vec2f, px0: vec2f) -> vec4f {
  let p = uv * u.resolution;
  let ex = i32(u.example);
  var c = vec3f(0.0);
  if (ex == 0) { c = oilView(uv, p); }
  else if (ex == 1) { c = waterView(uv, p); }
  else if (ex == 2) { c = sketchView(uv, p); }
  else { c = explainView(uv, p); }
  return vec4f(c, 1.0);
}`,
  about: {
    summary: 'Painterly filters turn any image into paint, washes or pencil. The workhorse is the Kuwahara filter: a smoothing filter that keeps edges sharp, so detail melts into blobs and strokes of color.',
    what: `<p>The platformer as an oil painting, a watercolor and a pencil sketch — plus a noisy test card that shows what the Kuwahara filter does, with its
      window drawn under the mouse.</p>`,
    how: `<ol>
      <li><b>Kuwahara, the idea.</b> Around each pixel, look at a window split into regions (4 squares in the classic version, 8 overlapping pie slices in the
        generalized one). Compute each region’s average color and how much it varies. Output the average of the <i>calmest</i> region(s). In flat areas all
        regions agree → smoothing. At an edge, regions straddling it are “noisy”, the ones on one side are calm → the edge is kept.</li>
      <li><b>Anisotropic version</b> (Kyprianidis et al.): first measure the local direction of the image with the <b>structure tensor</b> (products of the
        Sobel gradient, blurred) and its eigenvectors: the edge tangent <code>t</code> and how directional the area is (<code>A</code>). Then squash the circular
        window into an <b>ellipse along t</b>. Regions follow the shapes, producing brush-stroke-like smears.</li>
      <li><b>Smooth weights.</b> Instead of hard pie slices, each sector uses a cheap polynomial weight that fades out at its borders, and sectors are mixed with
        weights <code>1 / (1 + (k·variance)⁴)</code> — calm sectors dominate, but nothing pops.</li>
      <li><b>Oil:</b> treat paint lightness plus streak noise (noise averaged <i>along</i> the stroke direction — a mini line-integral convolution) as height,
        compute a normal, light it: impasto ridges. Add canvas weave.</li>
      <li><b>Watercolor:</b> sample the simplified image through a noise warp (<b>bleed</b>), darken where the color differs from its surroundings (<b>pooling at wash
        edges</b>), vary pigment density with paper noise (<b>granulation</b>), multiply onto paper as transparent color, let highlights go to white paper.</li>
      <li><b>Pencil:</b> each brightness range adds another layer of hatching at a new angle (light → one direction, dark → four), lines broken into dashes and
        wobbled by noise; outlines from edges; graphite only sticks to the paper’s peaks; dark areas get a soft smudge.</li>
    </ol>`,
    uses: [
      { title: 'Painted games', text: 'Okami (sumi-e), Gris and Ori (painted looks), Valiant Hearts, Sable (line art), Return of the Obra Dinn’s cousin Mars After Midnight; “painterly” post filters in Disco Elysium-style menus.' },
      { title: 'Watercolor worlds', text: 'Child of Light, Gris, Dordogne and Wanderer’s Tale use watercolor washes, edge darkening and paper texture.' },
      { title: 'Sketch & storyboard', text: 'Hand-drawn intro/flashback scenes, Sketch-mode photo filters, Max Payne-style comic panels, level-editor “blueprint” views.' },
      { title: 'Tools', text: 'Photo apps (Prisma-style filters), video stylization and non-photorealistic rendering research all build on Kuwahara and structure tensors.' },
    ],
    try: [
      'On <b>How Kuwahara works</b>, switch between the three variants and hover an edge: the classic box ignores direction, the anisotropic ellipse lines up with it.',
      'Raise <i>Brush radius</i> to 8 at <i>Full</i> resolution and watch the fps — then switch to <i>Half</i>: nearly the same look for a quarter of the work.',
      'On <b>Oil painting</b>, set <i>Impasto relief</i> to 0 for a flat “posterized” paint look, then to 2.5 for thick paint.',
      'On <b>Watercolor</b>, turn <i>Edge darkening</i> to 1 and <i>Color bleed</i> to 0, then the reverse.',
      'On <b>Pencil sketch</b>, set <i>Hatching spacing</i> to 3 for dense graphite, 12 for loose strokes.',
    ],
    ask: [
      'anisotropic Kuwahara filter for an oil-paint look',
      'structure tensor to get stroke directions for painterly effects',
      'watercolor post-process with edge darkening, granulation and paper texture',
      'pencil cross-hatching shader based on luminance',
      'run the paint filter at half resolution to save performance',
    ],
    perf: `<p><b>Expensive.</b> Kuwahara reads every pixel in its window: radius 5 is ~80–120 texture reads per pixel (classic: (2r+1)² = 121). At 1080p that is
      ~250 million reads per frame, so this scene runs the filter at <i>half resolution</i> by default (4× cheaper) and keeps radius ≤ 8. The structure tensor adds
      two cheap passes. Fast implementations precompute sector weights, exploit symmetry, or use separable/box approximations; WebGPU compute can share loaded texels
      between neighbouring pixels through workgroup memory.</p>`,
    api: `<p>Runs on WebGL2 and WebGPU from the same source as a chain of render-to-texture passes (tensor → orientation → paint → composite). With WebGPU,
      a compute-shader version can load a tile of the image into <code>var&lt;workgroup&gt;</code> memory once and let all 256 threads of a workgroup read it —
      a big win for wide-window filters like this.</p>`,
    code: [
      {
        title: 'Orientation from the structure tensor',
        lang: 'wgsl',
        src: `// g = blurred (gx·gx, gy·gy, gx·gy)
let root = sqrt((E - G) * (E - G) + 4.0 * F * F);
let l1 = 0.5 * (E + G + root);              // strongest change
let l2 = 0.5 * (E + G - root);              // weakest change
var t = vec2f(l1 - E, -F);                  // eigenvector of l2 = edge tangent
if (length(t) > 0.000001) { t = normalize(t); } else { t = vec2f(0.0, 1.0); }
let A = (l1 - l2) / (l1 + l2);              // 0 = no direction, 1 = strong edge`,
      },
      {
        title: 'Anisotropic Kuwahara: elliptical window, 8 polynomial sectors',
        lang: 'wgsl',
        src: `let a = r * clamp(1.0 + A, 0.1, 2.0);        // long along the edge
let b = r * clamp(1.0 / (1.0 + A), 0.1, 2.0);  // short across it
for (var j = -maxY; j <= maxY; j++) { for (var i = -maxX; i <= maxX; i++) {
  let o = vec2f(f32(i), f32(j));
  var v = vec2f(dot(o, t) * 0.5 / a, dot(o, tp) * 0.5 / b);   // into the unit circle
  if (dot(v, v) <= 0.25) {
    let c = srcAt(uv + o * texel);
    // w[0..7] = squared polynomial sector weights (see source), then for each sector k:
    m[k] += vec4f(c, 1.0) * (w[k] * g);  s[k] += c * c * (w[k] * g);
  }
}}
for (var k = 0; k < 8; k++) {
  let mean = m[k].rgb / m[k].w;
  let v3 = abs(s[k] / m[k].w - mean * mean);                     // variance
  let wk = 1.0 / (1.0 + pow(8000.0 * (v3.r + v3.g + v3.b), 4.0)); // calm sectors win
  outc += vec4f(mean * wk, wk);
}`,
      },
    ],
    links: [
      { title: 'Kyprianidis et al. — Anisotropic Kuwahara Filtering (2009)', url: 'https://www.kyprianidis.com/p/pg2009/', note: 'the paper behind the oil look' },
      { title: 'Kuwahara filter (Wikipedia)', url: 'https://en.wikipedia.org/wiki/Kuwahara_filter' },
    ],
  },
});

// Trails, Ribbons & Slashes — triangle-strip meshes built every frame from position history.
//   ribbon:  history of the cursor (sampled at a fixed rate), Catmull-Rom smoothing, width taper,
//            UV = (age along, side across) + arc length for scrolling textures
//   slash:   a crescent mesh swept along an arc, plus the same mesh rendered into a distortion buffer
//   comets:  history tails vs. tails procedurally aimed away from the sun, shedding sparks
//   dash:    afterimages (silhouette sprites), speed lines and a motion streak
// Everything is drawn additively into an HDR target, then bloom + tone mapping.

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas } from '../../core/assets.js';
import { makeStrip, createBloom, TONEMAP_WGSL, tag, MeshBatch, GlowLines, softSprites, cellUV, TEX, prng, hsv, clamp, lerp, hexLin } from './_shared.js';

const MESH_FS = /* wgsl */ `
fn blackbody(h: f32) -> vec3f {
  return vec3f(smoothstep(0.0, 0.35, h) * 3.0, smoothstep(0.25, 0.75, h) * 2.0, smoothstep(0.6, 1.0, h) * 1.6);
}
fn mesh_fs(i: MOut) -> vec4f {
  let style = i.data.w;
  let along = i.uv.x;        // ribbons: 0 = head (newest) .. 1 = tail;  slash: 0 = tail .. 1 = head
  let ac = i.uv.y;           // -1 .. 1 across the strip
  let t = cam.viewport.w;
  let scroll = cam.extra.y;
  if (cam.extra.x > 0.5) {
    // distortion pass: write a screen-space offset (rg) instead of a color
    let k = smoothstep(-1.0, 0.7, ac) * (1.0 - smoothstep(0.85, 1.0, ac)) * pow(along, 1.2);
    return vec4f(i.data.yz * k * i.color.a, 0.0, 0.0);
  }
  let e = 1.0 - abs(ac);
  var c = i.color.rgb;
  var a = i.color.a;
  if (style < 0.5) {
    // gradient ribbon: soft edges + a hot core
    a *= smoothstep(0.0, 0.7, e);
    c += vec3f(1.2) * pow(e, 8.0) * (1.0 - along);
  } else if (style < 1.5) {
    // energy: noise scrolled along the arc length -> crawling filaments
    let n = fbm(vec2f(i.data.x * 0.011 - t * scroll * 2.0, ac * 1.4 + t * 0.3), 4);
    let fil = pow(clamp(1.0 - abs(n * 3.0 + ac * 0.4), 0.0, 1.0), 5.0);
    a *= smoothstep(0.0, 0.5, e);
    c = c * (0.15 + fil * 3.0) + vec3f(1.5) * pow(fil, 10.0);
  } else if (style < 2.5) {
    // fire: noise heat that cools toward the tail
    let n = fbm(vec2f(i.data.x * 0.014 - t * scroll * 2.5, ac * 1.3 - t * 0.8), 4) * 0.5 + 0.5;
    let heat = clamp((1.0 - along) * (n * 1.5 - 0.1) * smoothstep(0.0, 0.8, e) + pow(e, 6.0) * (1.0 - along) * 0.6, 0.0, 1.0);
    c = blackbody(heat) * 1.3;
    a = smoothstep(0.02, 0.2, heat) * i.color.a;
  } else if (style < 3.5) {
    // rainbow by arc length
    c = hsv2rgb(vec3f(fract(i.data.x * 0.0015 - t * scroll * 0.4), 0.8, 1.0)) * 2.2;
    a *= smoothstep(0.0, 0.7, e);
    c += vec3f(1.0) * pow(e, 10.0) * (1.0 - along);
  } else if (style < 4.5) {
    // slash crescent: bright, sharp outer edge, streaky body, fades toward the tail
    let body = smoothstep(-1.0, 0.7, ac) * (1.0 - smoothstep(0.94, 1.0, ac));
    let core = smoothstep(0.5, 0.9, ac) * (1.0 - smoothstep(0.95, 1.0, ac));
    let streaks = 0.7 + 0.3 * sin(ac * 34.0 + along * 3.0);
    c = mix(c, vec3f(2.0), core * 0.85) * (body * streaks + core * 1.5);
    a *= pow(along, 1.3);
  } else if (style < 5.5) {
    // comet tail: gaussian across, fading to the end, white-hot near the head
    let g = exp(-ac * ac * 2.5);
    a *= g * pow(1.0 - along, 1.7);
    c = mix(c, vec3f(2.0), pow(1.0 - along, 10.0) * g);
  } else {
    // speed line: thin, tapered at both ends
    a *= pow(e, 1.5) * smoothstep(0.0, 0.3, along) * (1.0 - smoothstep(0.55, 1.0, along));
  }
  return vec4f(c * a, 0.0);
}`;

const SPRITE_LIN_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  // pixel art is sRGB: convert to linear for the HDR pipeline; user = white hit-flash amount
  let c = mix(pow(t.rgb, vec3f(2.2)) * i.color.rgb * 1.3, vec3f(3.0), i.extra.w);
  return vec4f(c, 1.0);
}`;

const GHOST_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  // afterimage: a flat glowing silhouette, added on top
  return vec4f(i.color.rgb * i.color.a, 0.0);
}`;

const BG_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let p = px / res.y;                     // x 0..aspect, y 0..1 (down)
  let aspect = res.x / res.y;
  let gy = u.groundY / res.y;
  var c: vec3f;
  if (u.mode == 1u) {
    // night courtyard: sky, moon, rooftops, wooden floor
    c = mix(vec3f(0.01, 0.012, 0.035), vec3f(0.05, 0.03, 0.07), p.y);
    let mp = vec2f(aspect * 0.82, 0.22);
    let md = length(p - mp);
    c += vec3f(0.6, 0.65, 0.8) * (smoothstep(0.062, 0.058, md) * 1.2 + exp(-md * 9.0) * 0.12);
    let roof = gy - 0.2 - 0.05 * abs(sin(p.x * 3.0 + 1.0)) - 0.12 * step(0.5, fract(p.x * 0.9 + 0.2)) * smoothstep(0.0, 0.08, abs(fract(p.x * 0.9 + 0.2) - 0.75));
    if (p.y > roof) { c = vec3f(0.012, 0.01, 0.018); }
    if (p.y > gy) {
      let plank = fract(p.y * 18.0 / (0.3 + (p.y - gy) * 3.0));
      c = vec3f(0.06, 0.035, 0.022) * (0.75 + 0.25 * hash11(floor(p.y * 30.0))) * (0.7 + 0.3 * smoothstep(0.0, 0.1, plank));
      c *= 1.0 - (p.y - gy) * 1.2;
    }
  } else if (u.mode == 2u) {
    // deep space: nebula, stars, sun glow, planet
    let n = fbm(p * 2.2 + vec2f(u.time * 0.01, 0.0), 5) * 0.5 + 0.5;
    let n2 = fbm(p * 3.1 + vec2f(5.0, 2.0), 4) * 0.5 + 0.5;
    c = vec3f(0.004, 0.004, 0.012) + vec3f(0.06, 0.015, 0.08) * smoothstep(0.45, 0.9, n) + vec3f(0.0, 0.03, 0.06) * smoothstep(0.5, 0.95, n2);
    let cell = floor(px / 2.5);
    let h = hash21(cell);
    if (h > 0.996) { c += vec3f(0.8, 0.85, 1.0) * (0.5 + 0.5 * sin(u.time * 3.0 + h * 500.0)) * (h - 0.996) * 300.0; }
    let sd = length(p - u.sun / res.y);
    c += vec3f(1.6, 1.2, 0.8) * (exp(-sd * 30.0) * 1.6 + exp(-sd * 5.0) * 0.12);
    let pc = u.planet.xy / res.y;
    let pr = u.planet.z / res.y;
    let pd = length(p - pc);
    let atm = exp(-max(pd - pr, 0.0) * 40.0) * 0.4;
    c += vec3f(0.2, 0.5, 1.0) * atm;
    if (pd < pr) {
      let q = (p - pc) / pr;
      let z = sqrt(max(1.0 - dot(q, q), 0.0));
      let ld = normalize(vec3f(-0.7, -0.5, 0.5));
      let lit = max(dot(vec3f(q, z), ld), 0.0);
      let bands = fbm(vec2f(q.y * 6.0 + fbm(q * 3.0, 3), q.x * 1.5 + u.time * 0.02), 4) * 0.5 + 0.5;
      c = mix(vec3f(0.15, 0.25, 0.6), vec3f(0.5, 0.75, 0.95), bands) * (lit * 0.9 + 0.02) * 0.6;
      c += vec3f(0.3, 0.6, 1.0) * pow(1.0 - z, 3.0) * 0.5 * lit;
    }
  } else if (u.mode == 3u) {
    // dusk city street
    c = mix(vec3f(0.06, 0.03, 0.12), vec3f(0.9, 0.35, 0.3), smoothstep(0.1, 0.75, p.y));
    let sd = length(p - vec2f(aspect * 0.3, gy - 0.12));
    c += vec3f(1.4, 0.7, 0.3) * (smoothstep(0.1, 0.095, sd) * 1.2 + exp(-sd * 6.0) * 0.25);
    let bx = floor(p.x * 9.0);
    let bh = gy - 0.12 - 0.3 * hash11(bx * 2.3) * hash11(bx + 9.0);
    if (p.y > bh) {
      c = vec3f(0.05, 0.02, 0.06);
      let wc = floor(vec2f(p.x * 70.0, p.y * 50.0));
      c += vec3f(1.0, 0.6, 0.3) * step(0.8, hash21(wc)) * 0.3 * step(0.25, fract(p.x * 70.0)) * step(0.3, fract(p.y * 50.0));
    }
    if (p.y > gy) {
      c = mix(vec3f(0.06, 0.03, 0.06), vec3f(0.02, 0.01, 0.025), smoothstep(gy, 1.0, p.y));
      c += vec3f(1.0, 0.4, 0.3) * exp(-(p.y - gy) * 60.0) * 0.4;
    }
  } else {
    // dark gradient with a faint dot grid
    c = mix(vec3f(0.006, 0.008, 0.02), vec3f(0.02, 0.012, 0.03), uv.y);
    let g = fract(px / 32.0) - 0.5;
    c += vec3f(0.03, 0.04, 0.07) * smoothstep(0.08, 0.0, length(g));
  }
  return vec4f(c, 1.0);
}`;

const COMPOSITE_WGSL = /* wgsl */ `
${TONEMAP_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let off = TEX(distTex, uv).rg * u.distort / u.resolution;
  var c = TEX(hdr, uv + off).rgb + TEX(bloomTex, uv + off * 0.5).rgb * u.bloomAmt;
  c += vec3f(1.0) * u.flash;
  let v = uv - 0.5;
  c *= 1.0 - dot(v, v) * 0.8;
  return vec4f(vfxTonemap(c * u.exposure, px), 1.0);
}`;

// ------------------------------------------------------------------------------- geometry helpers

/** Catmull-Rom subdivision: sub new points per segment (the curve passes through the originals). */
function smoothPath(pts, sub) {
  if (sub <= 1 || pts.length < 3) return pts;
  const out = [];
  const n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(n - 1, i + 2)];
    for (let k = 0; k < sub; k++) {
      const t = k / sub;
      const t2 = t * t;
      const t3 = t2 * t;
      const cr = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push({ x: cr(p0.x, p1.x, p2.x, p3.x), y: cr(p0.y, p1.y, p2.y, p3.y), u: lerp(p1.u, p2.u, t) });
    }
  }
  out.push(pts[n - 1]);
  return out;
}

/**
 * Emit a ribbon (triangle strip as quads) along pts (head first, each {x,y,u}).
 * width(u) -> half width, color(u) -> [r,g,b,a]. Returns edge segments for the wireframe view.
 */
function ribbon(mesh, pts, width, color, style, wire = null) {
  const n = pts.length;
  if (n < 2) return;
  let dist = 0;
  let pL = null;
  let pR = null;
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    if (i > 0) dist += Math.hypot(p.x - pts[i - 1].x, p.y - pts[i - 1].y);
    const a = pts[Math.max(0, i - 1)];
    const b = pts[Math.min(n - 1, i + 1)];
    let tx = b.x - a.x;
    let ty = b.y - a.y;
    const tl = Math.hypot(tx, ty) || 1;
    tx /= tl;
    ty /= tl;
    const w = width(p.u);
    const c = color(p.u);
    const L = [p.x - ty * w, p.y + tx * w, p.u, -1, c[0], c[1], c[2], c[3], dist, 0, 0, style];
    const R = [p.x + ty * w, p.y - tx * w, p.u, 1, c[0], c[1], c[2], c[3], dist, 0, 0, style];
    if (pL) {
      mesh.quad(pL, L, pR, R);
      if (wire) wire.push([pL[0], pL[1], L[0], L[1]], [pR[0], pR[1], R[0], R[1]], [L[0], L[1], R[0], R[1]], [pR[0], pR[1], L[0], L[1]]);
    } else if (wire) wire.push([L[0], L[1], R[0], R[1]]);
    pL = L;
    pR = R;
  }
}

const ease = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);

// combo: start angle, end angle (screen radians, y down), radius scale, duration, color, sparks, shake, hit-stop
const COMBO = [
  { a0: -1.9, a1: 1.05, r: 1.0, dur: 0.13, col: [0.4, 1.0, 2.6], sparks: 26, shake: 0.25, stop: 0.05 },
  { a0: 1.25, a1: -1.6, r: 1.0, dur: 0.13, col: [2.6, 0.5, 1.6], sparks: 26, shake: 0.25, stop: 0.05 },
  { a0: -2.9, a1: 1.6, r: 1.3, dur: 0.2, col: [2.8, 1.6, 0.35], sparks: 60, shake: 0.6, stop: 0.12 },
];

export default {
  interaction: 'Move the mouse — it leaves a ribbon. Click in the other tabs.',
  examples: [
    {
      id: 'ribbon',
      label: 'Mouse ribbon',
      kind: 'Abstract',
      note: 'The cursor position is recorded 120 times a second. Each frame those points are smoothed (Catmull-Rom), turned into a strip of quads whose width tapers with age, and shaded with a gradient or a texture scrolled along the strip’s length. Turn on <b>Show triangles</b>.',
      hint: 'Move the mouse (when idle, a ghost cursor draws for you).',
      params: { trailLen: 1.0, width: 44, taper: 1.1, smooth: 4, style: 'energy', scroll: 1, count: 3, colA: '#5ce1ff', colB: '#b04dff', glow: 1.2 },
    },
    {
      id: 'slash',
      label: 'Sword slash combo',
      kind: 'In a game',
      note: 'Each slash is a crescent mesh swept along an arc: the head races ahead, the tail catches up, and the shape is thickest in the middle. It is also drawn into a <b>distortion buffer</b> that bends the background behind it. Hits add sparks, a white flash, hit-stop and screen shake. Click to attack — three hits make a combo.',
      hint: 'Click to slash (click again quickly for the combo).',
      params: { width: 60, smooth: 4, distortion: true, glow: 1 },
    },
    {
      id: 'comets',
      label: 'Comets & meteors',
      kind: 'In a game',
      note: 'Two kinds of tail. The yellow <b>dust tail</b> is a history ribbon that follows the path. The blue <b>ion tail</b> isn’t history at all: it’s a straight strip aimed away from the sun every frame. Meteors use a scrolling fire texture and shed sparks. Click to call a meteor.',
      hint: 'Click to drop a meteor at that spot.',
      params: { trailLen: 1.3, width: 16, taper: 0.8, smooth: 3, glow: 1.2 },
    },
    {
      id: 'dash',
      label: 'Dash & speed lines',
      kind: 'In a game',
      note: 'A dash sells speed with three cheap tricks: <b>afterimages</b> (copies of the sprite left behind, drawn as fading glowing silhouettes), a short <b>motion streak</b>, and anime <b>speed lines</b> rushing past in the background. Click to dash toward the cursor.',
      hint: 'Click to dash toward the cursor.',
      params: { trailLen: 0.3, width: 26, ghosts: 7, glow: 1 },
    },
  ],
  controls: [
    { type: 'slider', key: 'trailLen', label: 'Trail length (s)', min: 0.05, max: 2.5, step: 0.01, value: 1.0, help: 'How long a recorded point lives before it drops off the tail.', showFor: ['ribbon', 'comets', 'dash'] },
    { type: 'slider', key: 'width', label: 'Width (px)', min: 2, max: 120, step: 1, value: 44 },
    { type: 'slider', key: 'taper', label: 'Taper', min: 0, max: 4, step: 0.01, value: 1.2, help: 'width × (1 − age)^taper. 0 = constant width.', showFor: ['ribbon', 'comets'] },
    { type: 'slider', key: 'smooth', label: 'Smoothing (subdivisions)', min: 1, max: 8, step: 1, value: 4, help: 'Catmull-Rom points inserted between samples. 1 = raw polyline.', showFor: ['ribbon', 'comets', 'slash'] },
    {
      type: 'select',
      key: 'style',
      label: 'Style',
      value: 'energy',
      options: [
        { value: 'gradient', label: 'Gradient' },
        { value: 'energy', label: 'Energy (scrolling noise)' },
        { value: 'fire', label: 'Fire' },
        { value: 'rainbow', label: 'Rainbow' },
      ],
      showFor: ['ribbon'],
    },
    { type: 'slider', key: 'scroll', label: 'UV scroll speed', min: -3, max: 3, step: 0.01, value: 1, help: 'Texture slides along the ribbon’s length.', showFor: ['ribbon'] },
    { type: 'slider', key: 'count', label: 'Ribbons', min: 1, max: 6, step: 1, value: 3, help: 'Followers chase the leader on springs.', showFor: ['ribbon'] },
    { type: 'color', key: 'colA', label: 'Head color', value: '#5ce1ff', showFor: ['ribbon'] },
    { type: 'color', key: 'colB', label: 'Tail color', value: '#b04dff', showFor: ['ribbon'] },
    { type: 'toggle', key: 'distortion', label: 'Distortion', value: true, help: 'Render the slash into an offset buffer that warps the background.', showFor: ['slash'] },
    { type: 'slider', key: 'ghosts', label: 'Afterimages', min: 0, max: 16, step: 1, value: 7, showFor: ['dash'] },
    { type: 'toggle', key: 'wire', label: 'Show triangles', value: false, help: 'Wireframe of the generated strip.' },
    { type: 'slider', key: 'glow', label: 'Glow (bloom)', min: 0, max: 3, step: 0.01, value: 1 },
  ],
  about: {
    summary: 'Trails are meshes rebuilt every frame from a history of positions: smooth them, give them width that tapers with age, and shade them with gradients or scrolling textures.',
    what: `<p>Every trail here is a <b>triangle strip</b> generated on the CPU each frame — a handful of vertices — then shaded on the GPU.
      The interesting part is <i>what</i> you build the strip from: cursor history, an animated arc, a direction away from the sun, or copies of a sprite.</p>`,
    how: `<ol>
      <li><b>Record</b> positions at a fixed rate (here 120 Hz, interpolated between frames so the trail is smooth even when fps drops). Drop points older than the trail length.</li>
      <li><b>Smooth</b> the polyline with Catmull-Rom splines: new points inserted between samples, and the curve still passes through every sample.</li>
      <li><b>Extrude</b>: at each point take the tangent, rotate it 90° for the normal, and place a left and a right vertex at ± width. Width = base × (1 − age)<sup>taper</sup>.</li>
      <li><b>UVs</b>: u = age along the trail (0 head → 1 tail), v = −1…1 across. Also store the <i>arc length</i>; scrolling a noise texture by <code>arcLength − time·speed</code> makes energy crawl along the ribbon without stretching.</li>
      <li><b>Shade additively</b> into an HDR buffer: soft edges from |v|, a hot core, then bloom.</li>
      <li><b>Slashes</b> are a strip along an arc with a crescent width profile. A second render of the same mesh writes screen-space offsets into a <b>distortion buffer</b>, and the final pass samples the scene at <code>uv + offset</code>.</li>
    </ol>`,
    uses: [
      { title: 'Melee combat', text: 'Weapon trails and slash arcs are the core of action-game feel (Hades, Dead Cells, Hollow Knight).' },
      { title: 'Movement', text: 'Dash afterimages, motion streaks and speed lines (Celeste’s dash, Sonic, Katana ZERO).' },
      { title: 'Projectiles', text: 'Comet/meteor tails, bullet tracers, homing missiles, magic bolts with ribbons.' },
      { title: 'UI & cursors', text: 'Fancy cursor trails, swipe effects in mobile games (Fruit Ninja), drawing gestures.' },
    ],
    try: [
      'On <b>Mouse ribbon</b> set <i>Smoothing</i> to 1 and move the mouse fast — corners appear. Raise it back to 4.',
      'Turn on <b>Show triangles</b> and change <i>Taper</i>: the strip is just quads getting thinner.',
      'Set <i>UV scroll speed</i> to 0 and then to −3 with the Energy style: the texture flows backwards along the trail.',
      'On <b>Sword slash combo</b>, toggle <i>Distortion</i> while clicking — the background bends behind the blade.',
      'On <b>Dash</b>, set <i>Afterimages</i> to 16 and click left/right repeatedly.',
    ],
    ask: [
      'a smooth mouse trail ribbon with Catmull-Rom smoothing and width taper',
      'weapon slash arcs with a crescent mesh and screen distortion',
      'UV-scrolled energy texture along a trail',
      'dash afterimages and anime speed lines',
      'comet tails that point away from the sun',
      'a trail that samples position at a fixed rate independent of fps',
    ],
    perf: `<p>Trails are tiny: a few hundred vertices rebuilt and uploaded each frame. The cost is fill rate — wide, overlapping, additive strips —
      plus whatever the fragment shader does (the fBm “energy” style is the priciest here). Hundreds of trails are fine; for thousands (bullet hell) build them in a compute shader.</p>`,
    api: `<p>Nothing here needs WebGPU specifically — dynamic vertex buffers, additive blending and a distortion render target all exist in WebGL2.
      This scene is written against WebGPU only (custom pipelines, HDR float targets and the shared bloom pass).</p>`,
    code: [
      {
        title: 'Extruding the strip (JavaScript)',
        lang: 'js',
        src: `for (let i = 0; i < n; i++) {
  const p = pts[i];                                   // head first, p.u = age 0..1
  if (i > 0) dist += Math.hypot(p.x - pts[i - 1].x, p.y - pts[i - 1].y);
  const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
  let tx = b.x - a.x, ty = b.y - a.y;                 // tangent
  const tl = Math.hypot(tx, ty) || 1; tx /= tl; ty /= tl;
  const w = base * Math.pow(1 - p.u, taper);          // width tapers with age
  const L = [p.x - ty * w, p.y + tx * w, p.u, -1, ...color, dist, 0, 0, style];
  const R = [p.x + ty * w, p.y - tx * w, p.u,  1, ...color, dist, 0, 0, style];
  if (prevL) mesh.quad(prevL, L, prevR, R);           // two triangles
  prevL = L; prevR = R;
}`,
      },
      {
        title: 'Energy style: noise scrolled along the arc length',
        lang: 'wgsl',
        src: `let n = fbm(vec2f(i.data.x * 0.011 - t * scroll * 2.0, ac * 1.4 + t * 0.3), 4);
let fil = pow(clamp(1.0 - abs(n * 3.0 + ac * 0.4), 0.0, 1.0), 5.0); // thin filaments
a *= smoothstep(0.0, 0.5, e);                                        // soft edges
c = c * (0.15 + fil * 3.0) + vec3f(1.5) * pow(fil, 10.0);            // HDR core`,
      },
      {
        title: 'Distortion: render offsets, then sample the scene shifted',
        lang: 'wgsl',
        src: `// pass 2 of the slash mesh (cam.extra.x = 1): write an offset instead of a color
let k = smoothstep(-1.0, 0.7, ac) * (1.0 - smoothstep(0.85, 1.0, ac)) * pow(along, 1.2);
return vec4f(i.data.yz * k, 0.0, 0.0);      // data.yz = radial direction * strength
// composite:
let off = TEX(distTex, uv).rg * u.distort / u.resolution;
var c = TEX(hdr, uv + off).rgb;`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasView = gpu.textureFromImage(atlas.canvas, { label: 'atlas' }).createView();
    const strip = makeStrip(gpu).createView();
    const mesh = new MeshBatch(gpu, { fragment: MESH_FS, include: ['noise', 'color'], label: 'trail-mesh' });
    const lines = new GlowLines(gpu);
    const sparks = softSprites(gpu, strip);
    const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest', fragment: SPRITE_LIN_FS });
    const ghosts = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest', fragment: GHOST_FS });
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    const U = gpu.uniforms(
      { resolution: 'vec2f', time: 'f32', mode: 'u32', groundY: 'f32', bloomAmt: 'f32', exposure: 'f32', distort: 'f32', flash: 'f32', sun: 'vec2f', planet: 'vec4f' },
      'TrailU',
    );
    const bg = gpu.fullscreen({ label: 'trail-bg', uniforms: U, include: ['noise'], code: BG_WGSL, format: 'rgba16float' });
    const composite = gpu.fullscreen({ label: 'trail-composite', uniforms: U, textures: ['hdr', 'bloomTex', 'distTex'], include: ['color', 'hash'], code: COMPOSITE_WGSL });
    const bloom = createBloom(gpu);
    let hdr = null;
    let dist = null;
    const ensure = (w, h) => {
      if (hdr && hdr.width === w && hdr.height === h) return;
      hdr?.destroy();
      dist?.destroy();
      hdr = gpu.target(w, h, { format: 'rgba16float', label: 'trail-hdr' });
      dist = gpu.target(w, h, { format: 'rgba16float', label: 'trail-dist' });
    };
    const readout = tag(ctx);
    const rand = prng(11);
    const SAMPLE = 1 / 120;

    // ---------------------------------------------------------------- shared state
    let simT = 0;
    let lastInput = -10;
    let prevPtr = null;
    let trauma = 0;
    let stopT = 0;
    let flash = 0;
    const particles = [];
    const addSpark = (x, y, vx, vy, life, size, col, tex = TEX.spark, add = 1, drag = 2, grav = 900) => {
      if (particles.length < 3000) particles.push({ x, y, vx, vy, life, age: 0, size, col, tex, add, drag, grav });
    };
    const stepParticles = (dt) => {
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.age += dt;
        if (p.age >= p.life) {
          particles[i] = particles[particles.length - 1];
          particles.pop();
          continue;
        }
        p.vy += p.grav * dt;
        const k = Math.exp(-p.drag * dt);
        p.vx *= k;
        p.vy *= k;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
      }
    };
    const drawParticles = () => {
      for (const p of particles) {
        const t = p.age / p.life;
        const sp = Math.hypot(p.vx, p.vy);
        const fade = 1 - t;
        if (p.tex === TEX.spark) {
          sparks.draw(p.x, p.y, p.size + sp * 0.035, p.size * 0.45, { uv: cellUV(TEX.spark), rotation: Math.atan2(p.vy, p.vx), color: [p.col[0] * fade, p.col[1] * fade, p.col[2] * fade, 1], user: p.add });
        } else {
          const s = p.size * (p.tex === TEX.smoke ? 1 + t * 1.5 : 1 - t * 0.5);
          sparks.draw(p.x, p.y, s, s, { uv: cellUV(p.tex), rotation: p.age * 0.7 + p.size, color: [p.col[0], p.col[1], p.col[2], fade * (p.col[3] ?? 1)], user: p.add });
        }
      }
    };

    // ---------------------------------------------------------------- ribbon example
    const ribbons = [];
    const resetRibbons = () => {
      ribbons.length = 0;
      for (let i = 0; i < 6; i++) ribbons.push({ x: ctx.width / 2, y: ctx.height / 2, vx: 0, vy: 0, hist: [] });
    };
    resetRibbons();
    const ghostCursor = (t) => {
      const W = ctx.width;
      const H = ctx.height;
      return [W * 0.5 + Math.sin(t * 1.3) * W * 0.32 + Math.sin(t * 3.1) * W * 0.05, H * 0.5 + Math.sin(t * 2.1 + 0.6) * H * 0.3];
    };

    // ---------------------------------------------------------------- slash example
    let slash = null; // {k, t}
    let comboQueued = 0;
    let comboIdx = 0;
    let dummy = { rot: 0, vrot: 0, flash: 0 };
    let nextAuto = 0.3;

    // ---------------------------------------------------------------- comets example
    const comets = [];
    const meteors = [];
    const newComet = (i, t0) => {
      const W = ctx.width;
      const H = ctx.height;
      const fromLeft = i % 2 === 0;
      return {
        t0,
        dur: 7 + rand() * 5,
        a: [fromLeft ? -0.15 * W : 1.15 * W, H * (0.1 + 0.5 * rand())],
        b: [W * (0.3 + 0.4 * rand()), H * (0.2 + 0.6 * rand())],
        c: [fromLeft ? 1.2 * W : -0.2 * W, H * (0.2 + 0.6 * rand())],
        hist: [],
        col: i % 3 === 0 ? [2.2, 1.7, 0.9, 1] : i % 3 === 1 ? [1.4, 1.9, 2.4, 1] : [2.0, 1.3, 2.2, 1],
      };
    };
    const cometPos = (c, t) => {
      const s = clamp((t - c.t0) / c.dur, 0, 1);
      const i = 1 - s;
      return [i * i * c.a[0] + 2 * i * s * c.b[0] + s * s * c.c[0], i * i * c.a[1] + 2 * i * s * c.b[1] + s * s * c.c[1], s];
    };
    let nextMeteor = 0.5;
    const launchMeteor = (tx, ty) => {
      const W = ctx.width;
      const fromLeft = tx > W * 0.5 ? false : true;
      const sx = fromLeft ? tx - ctx.height * 0.9 : tx + ctx.height * 0.9;
      const sy = -ctx.height * 0.15;
      meteors.push({ x: sx, y: sy, tx, ty, speed: ctx.height * 1.4, hist: [], alive: true, done: 0 });
    };

    // ---------------------------------------------------------------- dash example
    const hero = { x: 0, dir: 1, dashT: -1, from: 0, to: 0, hist: [], ghosts: [], lastGhost: 0, idle: 0.6 };
    const speedLines = [];

    const resetAll = () => {
      simT = 0;
      particles.length = 0;
      resetRibbons();
      slash = null;
      comboQueued = 0;
      comboIdx = 0;
      nextAuto = 0.3;
      comets.length = 0;
      for (let i = 0; i < 3; i++) {
        const c = newComet(i, -i * 2.6 - 1.5);
        comets.push(c);
      }
      meteors.length = 0;
      nextMeteor = 0.5;
      hero.x = ctx.width * 0.3;
      hero.dashT = -1;
      hero.hist = [];
      hero.ghosts = [];
      hero.idle = 0.25;
      speedLines.length = 0;
      trauma = 0;
      stopT = 0;
    };
    resetAll();

    // ---------------------------------------------------------------- per-example simulation at a fixed sample rate
    const sampleRibbon = (p, h, ptr, k, n) => {
      const auto = !ptr.over || simT - lastInput > 1.5;
      let tx;
      let ty;
      if (auto) [tx, ty] = ghostCursor(simT);
      else {
        const f = (k + 1) / n;
        tx = lerp(prevPtr ? prevPtr[0] : ptr.x, ptr.x, f);
        ty = lerp(prevPtr ? prevPtr[1] : ptr.y, ptr.y, f);
      }
      const count = Math.round(p.count);
      for (let r = 0; r < count; r++) {
        const R = ribbons[r];
        if (r === 0) {
          const kf = 1 - Math.exp(-h * 30);
          R.x += (tx - R.x) * kf;
          R.y += (ty - R.y) * kf;
        } else {
          // followers: a damped spring toward the ribbon in front
          const L = ribbons[r - 1];
          const stiff = 90 - r * 8;
          R.vx += ((L.x - R.x) * stiff - R.vx * 9) * h;
          R.vy += ((L.y - R.y) * stiff - R.vy * 9) * h;
          R.x += R.vx * h;
          R.y += R.vy * h;
        }
        R.hist.unshift({ x: R.x, y: R.y, t: simT });
        while (R.hist.length && simT - R.hist[R.hist.length - 1].t > p.trailLen) R.hist.pop();
      }
    };

    const startSlash = () => {
      slash = { k: comboIdx, t: 0, hit: false };
      comboIdx = (comboIdx + 1) % COMBO.length;
    };
    const heroPivot = () => {
      const H = ctx.height;
      return [ctx.width * 0.4, H * 0.78 - H * 0.12];
    };

    const sim = (dt, p) => {
      const ptr = ctx.pointer;
      const ex = ctx.example;
      // the ribbon reacts to movement; the click-driven tabs only count clicks as input
      if (ptr.clicked || (ex === 'ribbon' && (ptr.dx || ptr.dy))) lastInput = simT;
      const n = Math.max(1, Math.min(60, Math.ceil(dt / SAMPLE)));
      const h = dt / n;
      for (let k = 0; k < n; k++) {
        simT += h;
        if (ex === 'ribbon') sampleRibbon(p, h, ptr, k, n);
        else if (ex === 'comets') {
          for (const c of comets) {
            const [x, y, s] = cometPos(c, simT);
            c.hist.unshift({ x, y, t: simT });
            while (c.hist.length && simT - c.hist[c.hist.length - 1].t > p.trailLen) c.hist.pop();
            if (s > 0 && s < 1 && rand() < h * 25) addSpark(x, y, (rand() - 0.5) * 30, (rand() - 0.5) * 30, 0.8 + rand(), 3 + rand() * 3, [c.col[0] * 0.8, c.col[1] * 0.8, c.col[2] * 0.8, 1], TEX.soft, 1, 1.5, 0);
          }
          for (const m of meteors) {
            if (!m.alive) continue;
            const dx = m.tx - m.x;
            const dy = m.ty - m.y;
            const d = Math.hypot(dx, dy);
            const st = m.speed * h;
            if (d <= st) {
              m.alive = false;
              m.x = m.tx;
              m.y = m.ty;
              flash = Math.max(flash, 0.25);
              trauma = Math.min(1, trauma + 0.35);
              for (let i = 0; i < 70; i++) {
                const a = rand() * Math.PI * 2;
                const v = (0.2 + rand()) * ctx.height * 0.9;
                addSpark(m.x, m.y, Math.cos(a) * v, Math.sin(a) * v - ctx.height * 0.3, 0.5 + rand() * 0.7, 6 + rand() * 6, [3, 1.4 + rand(), 0.4, 1], TEX.spark, 1, 2.2, ctx.height * 1.2);
              }
              for (let i = 0; i < 12; i++) addSpark(m.x, m.y, (rand() - 0.5) * 200, -rand() * 120, 1.2 + rand(), 40 + rand() * 40, [0.25, 0.22, 0.25, 0.8], TEX.smoke, 0, 1.2, -60);
            } else {
              m.x += (dx / d) * st;
              m.y += (dy / d) * st;
              if (rand() < h * 60) addSpark(m.x, m.y, (rand() - 0.5) * 120 - (dx / d) * 120, (rand() - 0.5) * 120 - (dy / d) * 120, 0.3 + rand() * 0.4, 5 + rand() * 4, [3, 1.3, 0.3, 1], TEX.spark, 1, 3, 300);
            }
            m.hist.unshift({ x: m.x, y: m.y, t: simT });
            while (m.hist.length && simT - m.hist[m.hist.length - 1].t > 0.45) m.hist.pop();
          }
        } else if (ex === 'dash') {
          const H = ctx.height;
          if (hero.dashT >= 0) {
            hero.dashT += h;
            const s = ease(hero.dashT / 0.22);
            hero.x = lerp(hero.from, hero.to, s);
            if (simT - hero.lastGhost > 0.025 && hero.dashT < 0.22) {
              hero.lastGhost = simT;
              hero.ghosts.unshift({ x: hero.x, t: simT, dir: hero.dir });
            }
            if (hero.dashT >= 0.22) {
              hero.dashT = -1;
              hero.idle = 0.45;
              for (let i = 0; i < 8; i++) addSpark(hero.x - hero.dir * 20, H * 0.78, -hero.dir * (40 + rand() * 160), -rand() * 60, 0.6 + rand() * 0.5, 30 + rand() * 30, [0.5, 0.45, 0.5, 0.55], TEX.smoke, 0, 2.5, -40);
            }
          } else {
            hero.idle -= h;
            if (hero.idle <= 0 && simT - lastInput > 0.7) {
              const tgt = hero.x < ctx.width * 0.5 ? ctx.width * 0.78 : ctx.width * 0.22;
              dashTo(tgt);
            }
          }
          hero.hist.unshift({ x: hero.x, y: H * 0.78 - H * 0.08, t: simT });
          while (hero.hist.length && simT - hero.hist[hero.hist.length - 1].t > p.trailLen) hero.hist.pop();
          // speed lines rush past while dashing
          const fast = hero.dashT >= 0 || hero.idle > 0.25;
          if (fast && rand() < h * 160) {
            speedLines.push({ x: hero.dir > 0 ? ctx.width + 50 : -50, y: H * (0.1 + rand() * 0.8), len: H * (0.15 + rand() * 0.4), w: 1 + rand() * 2.5, v: -hero.dir * H * (5 + rand() * 4), age: 0 });
          }
          for (let i = speedLines.length - 1; i >= 0; i--) {
            const l = speedLines[i];
            l.x += l.v * h;
            l.age += h;
            if (l.age > 0.6) speedLines.splice(i, 1);
          }
        }
      }
      if (ex === 'slash') {
        if (ptr.clicked) {
          if (!slash || slash.t > COMBO[slash.k].dur * 0.6) startSlash();
          else comboQueued = 1;
        }
        if (simT - lastInput > 0.7 && simT > nextAuto && !slash) {
          startSlash();
          nextAuto = simT + (comboIdx === 0 ? 0.6 : 0.05);
        }
        if (slash) {
          const C = COMBO[slash.k];
          slash.t += dt;
          const py = heroPivot()[1];
          if (!slash.hit && slash.t > C.dur * 0.55) {
            slash.hit = true;
            const dxp = ctx.width * 0.4 + ctx.height * 0.26;
            const hx = dxp;
            const hy = py;
            dummy.flash = 0.09;
            dummy.vrot += (C.a1 > C.a0 ? 1 : -1) * 4 * C.r;
            trauma = Math.min(1, trauma + C.shake);
            stopT = C.stop;
            for (let i = 0; i < C.sparks; i++) {
              const a = (C.a1 > C.a0 ? 0.3 : -0.3) + (rand() - 0.5) * 2.2;
              const v = ctx.height * (0.5 + rand() * 1.1);
              addSpark(hx, hy, Math.cos(a) * v, Math.sin(a) * v, 0.25 + rand() * 0.35, 5 + rand() * 5, [C.col[0] * 0.6 + 1.5, C.col[1] * 0.6 + 1.2, C.col[2] * 0.6 + 0.8, 1], TEX.spark, 1, 4, ctx.height * 1.5);
            }
          }
          if (slash.t > C.dur + 0.3) {
            slash = null;
            if (comboQueued) {
              comboQueued = 0;
              startSlash();
            } else if (simT - lastInput < 0.7) comboIdx = 0;
          } else if (comboQueued && slash.t > C.dur * 0.8) {
            comboQueued = 0;
            startSlash();
          }
        }
        // dummy wobble spring
        dummy.vrot += (-dummy.rot * 60 - dummy.vrot * 6) * dt;
        dummy.rot += dummy.vrot * dt;
        dummy.flash = Math.max(0, dummy.flash - dt);
      }
      if (ex === 'comets') {
        for (let i = 0; i < comets.length; i++) if (simT - comets[i].t0 > comets[i].dur + p.trailLen) comets[i] = newComet(i + Math.floor(simT), simT);
        if (ptr.clicked) launchMeteor(ptr.x, ptr.y);
        if (simT > nextMeteor && simT - lastInput > 1) {
          launchMeteor(ctx.width * (0.25 + 0.5 * rand()), ctx.height * (0.55 + 0.25 * rand()));
          nextMeteor = simT + 1.6 + rand() * 1.5;
        }
        for (let i = meteors.length - 1; i >= 0; i--) if (!meteors[i].alive && (meteors[i].done += dt) > 0.5) meteors.splice(i, 1);
      }
      if (ex === 'dash' && ptr.clicked) dashTo(clamp(ptr.x, ctx.width * 0.1, ctx.width * 0.9));
      stepParticles(dt);
      for (let i = hero.ghosts.length - 1; i >= 0; i--) if (simT - hero.ghosts[i].t > 0.6) hero.ghosts.splice(i, 1);
      prevPtr = [ptr.x, ptr.y];
    };
    function dashTo(x) {
      hero.from = hero.x;
      hero.to = x;
      hero.dir = x >= hero.x ? 1 : -1;
      hero.dashT = 0;
      hero.lastGhost = -1;
      trauma = Math.min(1, trauma + 0.15);
      for (let i = 0; i < 6; i++) addSpark(hero.x, ctx.height * 0.78, -hero.dir * (60 + rand() * 120), -rand() * 50, 0.5 + rand() * 0.4, 26 + rand() * 24, [0.5, 0.45, 0.5, 0.5], TEX.smoke, 0, 2.5, -40);
    }

    // ---------------------------------------------------------------- drawing
    const wire = [];
    const drawSlash = (distPass) => {
      if (!slash) return;
      const C = COMBO[slash.k];
      const [px, py] = heroPivot();
      const R = ctx.height * 0.3 * C.r;
      const ph = ease(slash.t / C.dur);
      const pt = ease((slash.t - C.dur * 0.45) / (C.dur * 0.55 + 0.28));
      if (ph - pt < 0.002) return;
      const N = 10 * Math.round(ctx.params.smooth || 4);
      const Wmax = (ctx.params.width / 60) * ctx.height * 0.11 * C.r;
      let prevO = null;
      let prevI = null;
      for (let i = 0; i <= N; i++) {
        const s = i / N; // 0 tail .. 1 head
        const a = lerp(C.a0, C.a1, lerp(pt, ph, s));
        const w = Wmax * Math.pow(Math.sin(Math.PI * Math.min(1, s * 0.85 + 0.08)), 0.8) * (0.35 + 0.65 * s);
        const dx = Math.cos(a);
        const dy = Math.sin(a);
        const st = 18;
        const O = [px + dx * R, py + dy * R, s, 1, C.col[0], C.col[1], C.col[2], 1, 0, dx * st, dy * st, 4];
        const I = [px + dx * (R - w), py + dy * (R - w), s, -1, C.col[0], C.col[1], C.col[2], 1, 0, dx * st, dy * st, 4];
        if (prevO) {
          mesh.quad(prevI, I, prevO, O);
          if (!distPass && ctx.params.wire) wire.push([prevO[0], prevO[1], O[0], O[1]], [prevI[0], prevI[1], I[0], I[1]], [I[0], I[1], O[0], O[1]], [prevO[0], prevO[1], I[0], I[1]]);
        }
        prevO = O;
        prevI = I;
      }
    };

    return {
      resize() {},
      onAction(key) {
        if (key === 'reset') resetAll();
      },
      onExample() {
        resetAll();
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        const W = ctx.width;
        const H = ctx.height;
        ensure(W, H);
        const ex = ctx.example;
        const mode = { ribbon: 0, slash: 1, comets: 2, dash: 3 }[ex] ?? 0;
        let dt = Math.min(ctx.dt, 0.1);
        if (ctx.paused) dt = 0;
        // hit-stop: freeze the simulation for a few frames on impact
        if (stopT > 0) {
          stopT -= dt;
          dt = 0;
        }
        if (dt > 0) sim(dt, p);
        trauma = Math.max(0, trauma - ctx.dt * 1.6);
        flash = Math.max(0, flash - ctx.dt * 3);
        const shake = trauma * trauma * H * 0.025;
        cam.setViewport(W, H);
        cam.x = W / 2 + (Math.sin(ctx.time * 53) + Math.sin(ctx.time * 31)) * 0.5 * shake;
        cam.y = H / 2 + (Math.sin(ctx.time * 47 + 1) + Math.sin(ctx.time * 37)) * 0.5 * shake;
        cam.zoom = ex === 'dash' && hero.dashT >= 0 ? 1.02 : 1;

        U.set('resolution', [W, H])
          .set('time', ctx.time)
          .set('mode', mode)
          .set('groundY', H * 0.78)
          .set('bloomAmt', p.glow * 0.4)
          .set('exposure', 1.0)
          .set('distort', p.distortion && ex === 'slash' ? 1 : 0)
          .set('flash', flash * 0.3)
          .set('sun', [W * 0.12, H * 0.15])
          .set('planet', [W * 0.82, H * 0.82, H * 0.26, 0]);
        bg.draw(enc, hdr, {}, { clear: [0, 0, 0, 1] });

        mesh.begin();
        lines.begin();
        sparks.begin();
        sprites.begin();
        ghosts.begin();
        shapes.begin();
        wire.length = 0;
        const time = ctx.time;
        const scroll = p.scroll ?? 1;
        const smooth = Math.round(p.smooth ?? 4);
        const taper = p.taper ?? 1;
        const hk = H / 700;

        if (ex === 'ribbon') {
          const cA = hexLin(p.colA, 3.2);
          const cB = hexLin(p.colB, 2.6);
          const styleId = { gradient: 0, energy: 1, fire: 2, rainbow: 3 }[p.style] ?? 0;
          for (let r = 0; r < Math.round(p.count); r++) {
            const R = ribbons[r];
            const hist = R.hist;
            if (hist.length < 2) continue;
            // decimate to ~40 Hz before smoothing, so smoothing has visible work to do
            const raw = [];
            for (let i = 0; i < hist.length; i += 3) raw.push({ x: hist[i].x, y: hist[i].y, u: (simT - hist[i].t) / p.trailLen });
            const pts = smoothPath(raw, smooth);
            const wk = 1 - r * 0.13;
            ribbon(
              mesh,
              pts,
              (u) => p.width * hk * 0.5 * wk * Math.pow(clamp(1 - u, 0, 1), taper),
              (u) => [lerp(cA[0], cB[0], u), lerp(cA[1], cB[1], u), lerp(cA[2], cB[2], u), 1],
              styleId,
              p.wire ? wire : null,
            );
          }
        } else if (ex === 'slash') {
          // dummy, hero, sword
          const gy = H * 0.78;
          const dxp = W * 0.4 + H * 0.26;
          const fl = dummy.flash > 0 ? 1 : 0;
          shapes.rect(dxp - H * 0.012, gy - H * 0.22, H * 0.024, H * 0.22, [0.09, 0.05, 0.03, 1]);
          const sackCol = fl ? [3, 3, 3, 1] : [0.32, 0.22, 0.1, 1];
          shapes.box(dxp, gy - H * 0.16, H * 0.055, H * 0.085, sackCol, { radius: H * 0.04, rotation: dummy.rot * 0.25 });
          shapes.box(dxp, gy - H * 0.27, H * 0.035, H * 0.035, fl ? [3, 3, 3, 1] : [0.36, 0.26, 0.12, 1], { radius: H * 0.035, rotation: dummy.rot * 0.3 });
          shapes.line(dxp - H * 0.07, gy - H * 0.19, dxp + H * 0.07, gy - H * 0.19, H * 0.012, fl ? [3, 3, 3, 1] : [0.12, 0.07, 0.04, 1]);
          const hs = H * 0.2;
          const heroX = W * 0.4 - H * 0.03;
          sprites.draw(heroX, gy, hs, hs, { uv: atlas.uv(slash ? 'hero_run_1' : 'hero_idle_' + (Math.floor(time * 2) % 2)), anchor: [0.5, 1] });
          const [px, py] = heroPivot();
          let swordA = -0.6;
          if (slash) {
            const C = COMBO[slash.k];
            swordA = lerp(C.a0, C.a1, ease(slash.t / C.dur));
          }
          const sl = H * 0.3 * (slash ? COMBO[slash.k].r : 1) * 0.62;
          sprites.draw(px + Math.cos(swordA) * sl * 0.5, py + Math.sin(swordA) * sl * 0.5, sl * 1.15, sl * 1.15, { uv: atlas.uv('sword'), rotation: swordA + Math.PI / 4 });
          drawSlash(false);
        } else if (ex === 'comets') {
          const sun = [W * 0.12, H * 0.15];
          for (const c of comets) {
            if (c.hist.length < 2) continue;
            const raw = [];
            for (let i = 0; i < c.hist.length; i += 4) raw.push({ x: c.hist[i].x, y: c.hist[i].y, u: (simT - c.hist[i].t) / p.trailLen });
            const pts = smoothPath(raw, smooth);
            // dust tail: follows the path
            ribbon(mesh, pts, (u) => p.width * hk * 0.5 * (0.4 + u * 1.6) * Math.pow(1 - clamp(u, 0, 1), taper * 0.3), () => [c.col[0], c.col[1], c.col[2] * 0.7, 0.9], 5, p.wire ? wire : null);
            // ion tail: a straight strip pointing away from the sun, rebuilt every frame
            const hx = c.hist[0].x;
            const hy = c.hist[0].y;
            let ax = hx - sun[0];
            let ay = hy - sun[1];
            const al = Math.hypot(ax, ay) || 1;
            ax /= al;
            ay /= al;
            const L = H * 0.5 * p.trailLen;
            const ion = [];
            for (let i = 0; i <= 12; i++) {
              const s = i / 12;
              const wob = Math.sin(time * 3 + s * 6) * s * s * H * 0.01;
              ion.push({ x: hx + ax * L * s - ay * wob, y: hy + ay * L * s + ax * wob, u: s });
            }
            ribbon(mesh, ion, (u) => p.width * hk * 0.25 * (0.3 + u * 1.2), () => [0.35, 0.8, 2.6, 0.8], 5, p.wire ? wire : null);
            sparks.draw(hx, hy, H * 0.05, H * 0.05, { uv: cellUV(TEX.soft), color: [3, 3, 3, 1], user: 1 });
            sparks.draw(hx, hy, H * 0.12, H * 0.12, { uv: cellUV(TEX.star), color: [c.col[0] * 0.6, c.col[1] * 0.6, c.col[2] * 0.6, 1], user: 1, rotation: time * 0.3 });
          }
          for (const m of meteors) {
            if (m.hist.length >= 2) {
              const raw = [];
              for (let i = 0; i < m.hist.length; i += 3) raw.push({ x: m.hist[i].x, y: m.hist[i].y, u: (simT - m.hist[i].t) / 0.45 });
              ribbon(mesh, smoothPath(raw, 2), (u) => H * 0.035 * (1 - u * 0.6), () => [1, 1, 1, 1], 2, p.wire ? wire : null);
            }
            if (m.alive) sparks.draw(m.x, m.y, H * 0.06, H * 0.06, { uv: cellUV(TEX.soft), color: [4, 2.5, 1, 1], user: 1 });
          }
        } else if (ex === 'dash') {
          const gy = H * 0.78;
          const hs = H * 0.2;
          // speed lines (background)
          for (const l of speedLines) {
            const a = 1 - l.age / 0.6;
            const x0 = l.x;
            const x1 = l.x + Math.sign(l.v) * -l.len;
            const pts = [
              { x: x0, y: l.y, u: 0 },
              { x: x1, y: l.y, u: 1 },
            ];
            ribbon(mesh, pts, () => l.w * hk, () => [1.5 * a, 1.6 * a, 2.0 * a, 1], 6, p.wire ? wire : null);
          }
          // motion streak at body height
          if (hero.hist.length > 2) {
            const raw = hero.hist.filter((_, i) => i % 3 === 0).map((h) => ({ x: h.x, y: h.y, u: (simT - h.t) / p.trailLen }));
            ribbon(mesh, raw, (u) => p.width * hk * 0.5 * (1 - u), (u) => [0.4 + 1.6 * (1 - u), 1.2, 2.4, 1], 0, p.wire ? wire : null);
          }
          // afterimages: silhouettes left along the dash path
          const maxG = Math.round(p.ghosts);
          hero.ghosts.slice(0, maxG).forEach((g, i) => {
            const a = clamp(1 - (simT - g.t) / 0.6, 0, 1) * (1 - i / (maxG + 1));
            const col = hsv(0.5 + i * 0.05, 0.8, 2.2);
            ghosts.draw(g.x, gy, hs, hs, { uv: atlas.uv('hero_run_2'), anchor: [0.5, 1], flipX: g.dir < 0, color: [col[0], col[1], col[2], a] });
          });
          const frame = hero.dashT >= 0 ? 'hero_run_2' : 'hero_idle_' + (Math.floor(time * 2) % 2);
          sprites.draw(hero.x, gy, hs, hs, { uv: atlas.uv(frame), anchor: [0.5, 1], flipX: hero.dir < 0 });
        }
        drawParticles();

        // world -> HDR
        shapes.flush(enc, hdr, cam, {});
        mesh.flush(enc, hdr, cam, { blend: 'add', time, extra: [0, scroll, 0, 0] });
        ghosts.flush(enc, hdr, cam, { blend: 'add' });
        sprites.flush(enc, hdr, cam, {});
        sparks.flush(enc, hdr, cam, { blend: 'premultiplied' });
        if (wire.length) {
          for (const s of wire) lines.line(s[0], s[1], s[2], s[3], 1.2, 0, [0.4, 1.4, 0.6], 1, 0);
          lines.flush(enc, hdr, cam, { blend: 'max' });
        }
        const verts = mesh.b.n;
        // distortion buffer: the slash mesh again, writing offsets
        mesh.begin();
        if (ex === 'slash' && p.distortion) drawSlash(true);
        mesh.flush(enc, dist, cam, { blend: 'add', time, extra: [1, 0, 0, 0], clear: [0, 0, 0, 0] });

        const bl = bloom.render(enc, hdr, { threshold: 0.9, knee: 0.6 });
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { hdr, bloomTex: bl, distTex: dist });
        readout.textContent = ex === 'slash' ? `combo ${slash ? slash.k + 1 : comboIdx || 0}/3${stopT > 0 ? ' · hit-stop' : ''}` : `${verts.toLocaleString()} vertices`;
      },
    };
  },
};

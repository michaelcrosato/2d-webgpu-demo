// Jump Flooding Algorithm (JFA): turn any painted mask into a nearest-seed map — and therefore a
// distance field and a Voronoi diagram — in log2(N) compute passes. Each pass, every pixel looks at
// 9 pixels `step` away and keeps the nearest seed it has heard of; step halves every pass.
// We flood two things at once: the nearest INSIDE pixel and the nearest OUTSIDE pixel, which gives
// a signed distance field (outlines, glows, inner shadows) from a single run.

import { FULLSCREEN_VS } from '../../core/webgpu.js';
import { getAtlas, makeCanvas } from '../../core/assets.js';
import { overlayTag } from './_shared-b.js';

const MAXP = 13;

const COMMON = /* wgsl */ `
const NONE: u32 = 0xffffffffu;
fn pk(p: vec2i) -> u32 { return u32(p.x) | (u32(p.y) << 16u); }
fn upk(v: u32) -> vec2f { return vec2f(f32(v & 0xffffu), f32(v >> 16u)) + 0.5; }   // seed pixel center
fn cellIndex(p: vec2i) -> u32 { return u32(p.y) * u.size.x + u32(p.x); }
fn landValue(p: vec2f) -> f32 {
  // continent shape (the same formula is used in JavaScript to place the first cities)
  return 0.55 * sin(p.x * 1.7 + 1.3) + 0.45 * sin(p.y * 2.1 + p.x * 0.9) + 0.25 * sin((p.x - p.y) * 3.3 + 0.5) - 0.12 * dot(p, p) + 0.25;
}
`;

const SIM_WGSL = /* wgsl */ `
${COMMON}
fn rot(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }
fn spriteAt(p: vec2f) -> u32 {
  // hero (animated atlas frame), stretched over heroDst; nearest-neighbour lookup = pixel art
  let a = (p - u.heroDst.xy) / u.heroDst.zw;
  if (a.x >= 0.0 && a.y >= 0.0 && a.x < 1.0 && a.y < 1.0) {
    if (textureLoad(atlas, vec2i(u.heroSrc.xy + floor(a * u.heroSrc.zw)), 0).a > 0.5) { return 1u; }
  }
  let b = (p - u.textDst.xy) / u.textDst.zw;
  if (b.x >= 0.0 && b.y >= 0.0 && b.x < 1.0 && b.y < 1.0) {
    if (textureLoad(textTex, vec2i(b * vec2f(textureDimensions(textTex))), 0).a > 0.5) { return 2u; }
  }
  let c = rot(-u.starRot) * (p - u.starDst.xy) / u.starDst.z + 0.5;
  if (c.x >= 0.0 && c.y >= 0.0 && c.x < 1.0 && c.y < 1.0) {
    if (textureLoad(atlas, vec2i(u.starSrc.xy + floor(c * u.starSrc.zw)), 0).a > 0.5) { return 3u; }
  }
  return 0u;
}

// paint (or erase) a stroke from the previous to the current mouse position
@compute @workgroup_size(8, 8) fn paintStroke(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (p.x >= i32(u.size.x) || p.y >= i32(u.size.y)) { return; }
  let pf = vec2f(p) + 0.5;
  let ba = u.mouse - u.pmouse;
  let h = clamp(dot(pf - u.pmouse, ba) / max(dot(ba, ba), 0.0001), 0.0, 1.0);
  if (length(pf - u.pmouse - ba * h) < u.brush) {
    paint[cellIndex(p)] = select(0u, u.paintId, u.painting < 1.5);
  }
}

// mask = painted strokes + animated sprites (outline example)
@compute @workgroup_size(8, 8) fn compose(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (p.x >= i32(u.size.x) || p.y >= i32(u.size.y)) { return; }
  let i = cellIndex(p);
  var id = paint[i];
  if (id == 0u && u.sprites > 0.5) { id = spriteAt(vec2f(p) + 0.5); }
  mask[i] = id;
  // JFA initialisation: x = nearest INSIDE pixel (itself if inside), y = nearest OUTSIDE pixel
  seedOut[i] = vec2u(select(NONE, pk(p), id != 0u), select(NONE, pk(p), id == 0u));
}

// one jump-flooding pass with step size ps.step
@compute @workgroup_size(8, 8) fn jfa(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (p.x >= i32(u.size.x) || p.y >= i32(u.size.y)) { return; }
  let pf = vec2f(p) + 0.5;
  var bestA = NONE;
  var bestB = NONE;
  var dA = 1e30;
  var dB = 1e30;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = p + vec2i(x, y) * ps.step;
      if (q.x < 0 || q.y < 0 || q.x >= i32(u.size.x) || q.y >= i32(u.size.y)) { continue; }
      let s = seedIn[cellIndex(q)];
      if (s.x != NONE) {
        let d = pf - upk(s.x);
        let dd = dot(d, d);
        if (dd < dA) { dA = dd; bestA = s.x; }
      }
      if (s.y != NONE) {
        let d = pf - upk(s.y);
        let dd = dot(d, d);
        if (dd < dB) { dB = dd; bestB = s.y; }
      }
    }
  }
  seedOut[cellIndex(p)] = vec2u(bestA, bestB);
}
`;

const DRAW_WGSL = /* wgsl */ `
${COMMON}
${FULLSCREEN_VS}
fn pal(id: u32) -> vec3f {
  let h = fract(f32(id) * 0.618034 + 0.11);
  return 0.55 + 0.42 * cos(6.2831853 * (h + vec3f(0.0, 0.33, 0.67)));
}
fn hashf(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn seedsAt(pj: vec2f) -> vec2u {
  let c = clamp(vec2i(floor(pj)), vec2i(0), vec2i(u.size) - vec2i(1));
  return seeds[cellIndex(c)];
}
fn idOfSeed(s: u32) -> u32 {
  if (s == NONE) { return 0u; }
  return mask[cellIndex(vec2i(upk(s)))];
}
fn ring(p: vec2f, c: vec2f, r: f32, w: f32) -> f32 { return 1.0 - smoothstep(w * 0.5, w * 0.5 + 1.0, abs(length(p - c) - r)); }

@fragment fn fs_main(i: VSOut) -> @location(0) vec4f {
  let px = i.pos.xy;                        // canvas pixel
  let pj = px * u.scale;                    // JFA-grid coordinates
  let s = seedsAt(pj);
  let ex = i32(u.example + 0.5);
  // distance (in canvas px) to the nearest inside / outside pixel center
  var dIn = 1e5;
  var dOut = 1e5;
  if (s.x != NONE) { dIn = length(pj - upk(s.x)) / u.scale; }
  if (s.y != NONE) { dOut = length(pj - upk(s.y)) / u.scale; }
  let sd = dIn - dOut;                      // signed distance: < 0 inside the shapes
  var col = vec3f(0.0);

  if (ex == 0) {
    // ------------------------------------------------ step-by-step Voronoi / distance
    if (s.x == NONE) {
      let ck = step(0.5, fract((floor(px.x / 12.0) + floor(px.y / 12.0)) * 0.5));
      col = mix(vec3f(0.09, 0.09, 0.11), vec3f(0.12, 0.12, 0.15), ck);           // no seed found yet
    } else {
      let id = idOfSeed(s.x);
      col = pal(id) * (0.62 + 0.38 * (0.5 + 0.5 * cos(dIn * 0.21)));            // distance rings
      col *= 1.0 - 0.45 * smoothstep(0.0, 420.0, dIn);
      // cell borders: does a neighbour cell belong to another seed?
      let e = 1.0;
      let n1 = idOfSeed(seedsAt(pj + vec2f(e, 0.0)).x);
      let n2 = idOfSeed(seedsAt(pj + vec2f(0.0, e)).x);
      if (n1 != id || n2 != id) { col = mix(col, vec3f(0.05), 0.75); }
    }
    if (u.showSeeds > 0.5 && dIn < 1.5 / u.scale + 1.0) { col = vec3f(1.0); }
    // the 3x3 sample pattern of the NEXT pass, around the mouse
    if (u.nextStep > 0.0 && u.mouseOver > 0.5) {
      let m = u.mouse / u.scale;
      let st = u.nextStep / u.scale;
      for (var k = -1; k <= 1; k++) {
        let o = f32(k) * st;
        if (abs(px.x - m.x) <= st) { col = mix(col, vec3f(1.0, 0.9, 0.3), (1.0 - smoothstep(0.5, 1.5, abs(px.y - m.y - o))) * 0.4); }
        if (abs(px.y - m.y) <= st) { col = mix(col, vec3f(1.0, 0.9, 0.3), (1.0 - smoothstep(0.5, 1.5, abs(px.x - m.x - o))) * 0.4); }
        for (var j = -1; j <= 1; j++) {
          col = mix(col, vec3f(1.0, 0.9, 0.3), ring(px, m + vec2f(o, f32(j) * st), 6.0, 2.0));
        }
      }
    }
    return vec4f(col, 1.0);
  }

  if (ex == 1) {
    // ------------------------------------------------ outline & glow any shape
    let uv = px / u.canvas;
    col = mix(vec3f(0.06, 0.06, 0.1), vec3f(0.1, 0.07, 0.15), uv.y);
    let g = abs(fract(px / 40.0) - 0.5);
    col += vec3f(0.025) * (1.0 - smoothstep(0.0, 0.03, min(g.x, g.y)));
    let id = idOfSeed(s.x);
    var tint = u.tint;
    if (id == 2u) { tint = u.tint2; }
    if (id >= 10u) { tint = pal(id); }
    let fx = i32(u.effect + 0.5);
    let w = u.width;
    // object fill
    var fill = vec3f(0.0);
    let pjf = pj;
    let a = (pjf - u.heroDst.xy) / u.heroDst.zw;
    if (id == 1u) {
      fill = textureLoad(atlas, vec2i(u.heroSrc.xy + floor(clamp(a, vec2f(0.0), vec2f(0.999)) * u.heroSrc.zw)), 0).rgb;
    } else if (id == 2u) {
      let b = (pjf - u.textDst.xy) / u.textDst.zw;
      fill = mix(vec3f(1.0, 0.98, 0.92), vec3f(0.55, 0.85, 1.0), clamp(b.y, 0.0, 1.0));
    } else if (id == 3u) {
      let c = mat2x2f(cos(u.starRot), -sin(u.starRot), sin(u.starRot), cos(u.starRot)) * (pjf - u.starDst.xy) / u.starDst.z + 0.5;
      fill = textureLoad(atlas, vec2i(u.starSrc.xy + floor(clamp(c, vec2f(0.0), vec2f(0.999)) * u.starSrc.zw)), 0).rgb;
    } else {
      fill = pal(id) * 0.9;
    }
    let inside = 1.0 - smoothstep(-0.5, 0.5, sd);
    if (fx == 5) {
      // raw signed distance field
      var c = mix(vec3f(0.25, 0.55, 1.0), vec3f(1.0, 0.45, 0.25), step(sd, 0.0));
      c *= 0.35 + 0.65 * exp(-abs(sd) / 120.0);
      c *= 0.8 + 0.2 * cos(sd * 0.4);
      c = mix(c, vec3f(1.0), 1.0 - smoothstep(0.0, 1.5, abs(sd)));
      return vec4f(c, 1.0);
    }
    if (fx == 0 || fx == 2) {
      col += tint * exp(-max(sd - (select(0.0, w, fx == 0)), 0.0) / max(u.glow, 0.5)) * 0.9 * step(0.0, sd);  // soft glow
    }
    if (fx == 0 || fx == 1) {
      let o = 1.0 - smoothstep(w - 0.5, w + 0.5, sd);                                                 // thick outline
      var oc = tint;
      if (fx == 1) { oc = mix(tint * 0.6, tint, smoothstep(0.0, w, sd)); }
      col = mix(col, oc, o * step(-0.5, sd));
      col = mix(col, vec3f(0.02), (1.0 - smoothstep(0.0, 1.2, abs(sd - w))) * 0.5);
    }
    if (fx == 3) {
      // neon: hollow shapes with two glowing tubes
      let t1 = 1.0 - smoothstep(1.0, 2.5, abs(sd));
      let t2 = 1.0 - smoothstep(1.0, 2.5, abs(sd - w * 1.6));
      col += tint * (exp(-abs(sd) / max(u.glow * 0.4, 0.5)) * 0.7 + exp(-abs(sd - w * 1.6) / max(u.glow * 0.3, 0.5)) * 0.45);
      col = mix(col, mix(tint, vec3f(1.0), 0.7), max(t1, t2 * 0.8));
      return vec4f(col, 1.0);
    }
    var inner = fill;
    if (fx == 4) {
      // bevel & inner shadow: light comes from the top-left; the edge normal is the direction to the nearest OUTSIDE pixel
      let dir = normalize(pj - upk(s.y) + vec2f(0.0001, 0.0));
      let edge = 1.0 - smoothstep(0.0, max(w, 1.0), -sd);
      inner *= 1.0 - 0.6 * edge * clamp(dot(dir, vec2f(-0.7, -0.7)) * 0.5 + 0.5, 0.0, 1.0);
      inner += vec3f(0.35) * edge * clamp(dot(dir, vec2f(0.7, 0.7)), 0.0, 1.0);
      inner *= 0.55 + 0.45 * smoothstep(0.0, u.glow + 1.0, -sd);
      // drop shadow from the shifted field
      let sh = seedsAt(pj - vec2f(8.0, 10.0) * u.scale);
      if (sh.x != NONE) {
        let dsh = length(pj - vec2f(8.0, 10.0) * u.scale - upk(sh.x)) / u.scale;
        col *= 1.0 - 0.6 * (1.0 - smoothstep(0.0, 14.0, dsh));
      }
    }
    col = mix(col, inner, inside);
    return vec4f(col, 1.0);
  }

  // ------------------------------------------------ Voronoi territories on a map
  let uv = px / u.canvas;
  let mp = (uv - 0.5) * vec2f(u.canvas.x / u.canvas.y, 1.0) * 3.0;
  let lv = landValue(mp) + 0.05 * sin(mp.x * 13.0 + sin(mp.y * 9.0)) + 0.03 * sin(mp.y * 21.0 + mp.x * 4.0);
  let paper = 0.92 + 0.08 * hashf(floor(px / 2.0)) * 0.6;
  let land = smoothstep(-0.01, 0.01, lv);
  var sea = vec3f(0.55, 0.7, 0.75) * paper;
  sea *= 0.9 + 0.1 * smoothstep(0.3, 0.5, abs(fract(lv * 7.0) - 0.5));           // contour "waves" off the coast
  var ground = vec3f(0.93, 0.86, 0.7) * paper;
  ground *= 0.92 + 0.08 * sin(lv * 40.0);
  if (s.x != NONE) {
    let id = idOfSeed(s.x);
    let claim = 1.0 - smoothstep(u.influence * 0.75, u.influence, dIn);
    let terr = mix(pal(id), vec3f(dot(pal(id), vec3f(0.33))), 0.35);              // muted "ink wash" colors
    ground = mix(ground, ground * mix(vec3f(1.0), terr * 1.25, 0.6), claim);
    // borders between territories
    let e = 1.0;
    let n1 = idOfSeed(seedsAt(pj + vec2f(e, 0.0)).x);
    let n2 = idOfSeed(seedsAt(pj + vec2f(0.0, e)).x);
    let n3 = idOfSeed(seedsAt(pj - vec2f(e, 0.0)).x);
    if ((n1 != id || n2 != id || n3 != id) && claim > 0.3 && u.borders > 0.5) {
      let dash = step(0.35, fract((px.x + px.y) / 14.0));
      ground = mix(ground, vec3f(0.45, 0.1, 0.08), 0.8 * dash);
    }
    // influence edge
    ground = mix(ground, pal(id) * 0.5, (1.0 - smoothstep(0.0, 1.5, abs(dIn - u.influence * 0.9))) * 0.6 * u.borders);
  }
  col = mix(sea, ground, land);
  col = mix(col, vec3f(0.3, 0.22, 0.15), (1.0 - smoothstep(0.0, 1.5, abs(lv) / max(fwidth(lv), 0.00001))) * 0.8);   // coastline
  // cities: a dot with a ring
  if (s.x != NONE && dIn < 9.0) {
    let id = idOfSeed(s.x);
    col = mix(col, vec3f(0.12, 0.08, 0.06), 1.0 - smoothstep(7.0, 8.5, dIn));
    col = mix(col, pal(id) * 1.1, 1.0 - smoothstep(4.5, 6.0, dIn));
  }
  col *= 1.0 - 0.35 * dot(uv - 0.5, uv - 0.5);
  return vec4f(col, 1.0);
}`;

const examples = [
  {
    id: 'steps',
    label: 'JFA step by step',
    kind: 'Abstract',
    note: 'Seeds are painted pixels. Each <b>pass</b>, every pixel looks at 9 pixels <code>step</code> away and keeps the closest seed it has heard of; then the step halves. Grey = no seed found yet. The yellow rings around the mouse show where the next pass will look. Paint new seeds with the mouse.',
    params: { animate: true, scaleMode: '0.5' },
  },
  {
    id: 'outline',
    label: 'Outline & glow any shape',
    kind: 'In a game',
    note: 'The mask (an animated sprite, text and a spinning star) is flooded every frame — the resulting <b>signed distance field</b> gives thick outlines, glows, neon tubes and bevels for <i>any</i> shape, at any width, with no extra cost per pixel of width. Paint to add your own shapes.',
    params: { scaleMode: '0.5' },
  },
  {
    id: 'voronoi',
    label: 'Voronoi territories',
    kind: 'In a game',
    note: 'Each city claims the land closest to it — a Voronoi diagram, straight from the nearest-seed map. Distance from the city fades its influence. <b>Click</b> to found a new city (each click is a new seed), right-drag to erase.',
    params: { scaleMode: '0.5' },
  },
];

const controls = [
  { type: 'heading', label: 'Jump flooding' },
  { type: 'toggle', key: 'animate', label: 'Animate the passes', value: true, showFor: ['steps'], help: 'Replays the passes one by one. Turn off to pick a pass with the slider.' },
  { type: 'slider', key: 'passes', label: 'Passes to run', min: 0, max: MAXP, step: 1, value: MAXP, showFor: ['steps'], help: 'log₂(resolution) passes complete the flood.' },
  { type: 'toggle', key: 'showSeeds', label: 'Show seeds', value: true, showFor: ['steps'] },
  {
    type: 'select',
    key: 'scaleMode',
    label: 'JFA resolution',
    value: '0.5',
    options: [
      { value: '1', label: 'Full' },
      { value: '0.5', label: 'Half' },
      { value: '0.25', label: 'Quarter' },
    ],
    help: 'The flood runs on this grid; distances are still evaluated per screen pixel.',
  },
  { type: 'slider', key: 'brush', label: 'Brush size', min: 1, max: 40, step: 1, value: 6, help: 'Paint radius (px).' },
  { type: 'heading', label: 'Effect', showFor: ['outline'] },
  {
    type: 'select',
    key: 'effect',
    label: 'Effect',
    value: 'combo',
    showFor: ['outline'],
    options: [
      { value: 'combo', label: 'Outline + glow' },
      { value: 'outline', label: 'Thick outline' },
      { value: 'glow', label: 'Soft glow' },
      { value: 'neon', label: 'Neon tubes' },
      { value: 'bevel', label: 'Bevel, inner & drop shadow' },
      { value: 'raw', label: 'Raw signed distance field' },
    ],
  },
  { type: 'slider', key: 'width', label: 'Outline width', min: 1, max: 40, step: 0.5, value: 7, showFor: ['outline'] },
  { type: 'slider', key: 'glow', label: 'Glow radius', min: 1, max: 80, step: 1, value: 22, showFor: ['outline'] },
  { type: 'color', key: 'tint', label: 'Outline color', value: '#ffc94d', showFor: ['outline'] },
  { type: 'color', key: 'tint2', label: 'Text outline color', value: '#4de1ff', showFor: ['outline'] },
  { type: 'slider', key: 'influence', label: 'City influence (px)', min: 40, max: 600, step: 5, value: 230, showFor: ['voronoi'] },
  { type: 'toggle', key: 'borders', label: 'Draw borders', value: true, showFor: ['voronoi'] },
  { type: 'button', key: 'reset', label: 'Clear & reseed', primary: true },
];

function makeTextCanvas() {
  const c = makeCanvas(1024, 360);
  const g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = '#fff';
  g.font = 'bold 300px "Arial Black", Impact, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('JFA', c.width / 2, c.height / 2 + 10);
  return c;
}

const landJS = (x, y) => 0.55 * Math.sin(x * 1.7 + 1.3) + 0.45 * Math.sin(y * 2.1 + x * 0.9) + 0.25 * Math.sin((x - y) * 3.3 + 0.5) - 0.12 * (x * x + y * y) + 0.25;

export default {
  interaction: 'Paint with the mouse (left), erase with right-drag.',
  examples,
  controls,
  reinitOnExample: true,
  about: {
    summary:
      'The Jump Flooding Algorithm computes, for <i>every</i> pixel, the nearest “seed” pixel — in only log₂(N) full-screen passes. That gives you distance fields, Voronoi diagrams, outlines and glows for any shape, every frame.',
    what: `<p>A mask (painted strokes, a sprite, text) is turned into a <b>nearest-seed map</b> by ~10 compute passes. From it we read, per pixel,
      the distance to the nearest shape pixel and which shape it is. Everything you see — rings, territories, outlines, glows, bevels — is a
      simple function of that distance.</p>`,
    how: `<ol>
      <li><b>Init</b>: every pixel inside the mask stores its own coordinate as its seed; others store “none”.</li>
      <li><b>Pass k</b> with step <code>s = N/2, N/4, … 1</code>: each pixel reads the seeds stored at its 8 neighbours <code>s</code> pixels away (and itself),
        and keeps the one closest to <i>itself</i>. Information jumps across the screen in big strides first, then gets refined.</li>
      <li>After <code>log₂ N</code> passes (10 for a 1024 grid) nearly every pixel knows its true nearest seed. JFA is approximate — rare pixels can be
        wrong — an extra pass of step 1 (“JFA+1”) fixes most of them.</li>
      <li><b>Distance</b> = length(pixel − seed). <b>Voronoi</b> = the id stored at the seed.</li>
      <li><b>Signed distance</b>: flood the inside and the outside at the same time (two seeds per pixel). Distance to the nearest inside pixel minus
        distance to the nearest outside pixel is negative inside, positive outside, and smooth across the edge.</li>
      <li><b>Effects</b>: outline = <code>sd &lt; width</code>; glow = <code>exp(−sd / radius)</code>; bevel = lighting from the direction to the nearest edge;
        drop shadow = the field sampled at an offset.</li>
    </ol>
    <p>Compare: a brute-force distance field checks every pixel against every seed (N² work); a blur-based glow costs more the wider it gets.
      JFA is ~10 × 9 reads per pixel, regardless of width.</p>`,
    uses: [
      { title: 'Outlines & selection glows', text: 'Highlight units, interactables and characters with thick, crisp outlines — any width, any shape, animated sprites included.' },
      { title: 'Strategy maps', text: 'Territories, borders and zones of control (Civilization-style influence maps), nearest-base lookups.' },
      { title: 'SDF generation', text: 'Generate distance-field fonts and shapes at runtime; soft shadows and ambient occlusion for 2D lighting (see 2D Global Illumination).' },
      { title: 'Procedural art', text: 'Stained glass, cell patterns, “grow” animations, fluid boundaries.' },
    ],
    try: [
      'In <b>JFA step by step</b> turn off <i>Animate</i> and move the <i>Passes</i> slider from 0 up: watch the huge first jumps, then the refinement.',
      'Hover the mouse to see the 9 sample positions of the next pass shrink.',
      'In <b>Outline & glow</b> push <i>Outline width</i> to 40: the cost does not change at all.',
      'Switch the effect to <b>Raw signed distance field</b> to see the data behind every effect.',
      'Set <i>JFA resolution</i> to Quarter: still smooth, since distances are recomputed per screen pixel.',
    ],
    ask: [
      'jump flooding algorithm to generate a distance field every frame',
      'thick outline and glow around any sprite using JFA',
      'Voronoi territories / influence map on the GPU',
      'signed distance field from a mask for inner shadows and bevels',
      'runtime SDF generation for text and shapes',
    ],
    perf: `<p>Per pass each pixel reads 9 seeds: at 960×540 that is ~4.7 M reads × 10 passes ≈ 47 M reads per frame — about a millisecond on a
      mid-range GPU. Halving the grid resolution divides the cost by 4 with little visible loss. The cost is independent of how many seeds
      there are and of the outline width.</p>`,
    api: `<p>Written as <b>WebGPU compute</b> passes over storage buffers (seed coordinates packed as <code>x | y&lt;&lt;16</code>, two per pixel).
      JFA also works in WebGL2 with fragment shaders ping-ponging between float/int textures — it was invented that way — but WebGPU makes
      the bookkeeping (integer storage, many small passes, per-pass uniforms) simpler.</p>`,
    code: [
      {
        title: 'One jump-flooding pass',
        lang: 'wgsl',
        src: `let pf = vec2f(p) + 0.5;
var best = NONE;  var bestD = 1e30;
for (var y = -1; y <= 1; y++) {
  for (var x = -1; x <= 1; x++) {
    let q = p + vec2i(x, y) * ps.step;          // look 'step' pixels away
    if (outOfBounds(q)) { continue; }
    let s = seedIn[cellIndex(q)].x;              // the nearest seed THAT pixel knows
    if (s != NONE) {
      let d = pf - upk(s);
      if (dot(d, d) < bestD) { bestD = dot(d, d); best = s; }
    }
  }
}
seedOut[cellIndex(p)].x = best;                  // step halves next pass`,
      },
      {
        title: 'Outline + glow from the signed distance',
        lang: 'wgsl',
        src: `let sd = dIn - dOut;                         // < 0 inside the shape
col += tint * exp(-max(sd - w, 0.0) / u.glow) * 0.9 * step(0.0, sd);   // glow
col  = mix(col, tint, 1.0 - smoothstep(w - 0.5, w + 0.5, sd));          // outline
col  = mix(col, fill, 1.0 - smoothstep(-0.5, 0.5, sd));                 // the shape`,
      },
    ],
    links: [
      { title: 'Rong & Tan — Jump Flooding in GPU (2006)', url: 'https://www.comp.nus.edu.sg/~tants/jfa.html', note: 'the original paper' },
      { title: 'Ben Golus — The Quest for Very Wide Outlines', url: 'https://bgolus.medium.com/the-quest-for-very-wide-outlines-ba82ed442cd9', note: 'JFA outlines in practice' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasTex = gpu.textureFromImage(atlas.canvas, { label: 'jfa-atlas' });
    const textTex = gpu.textureFromImage(makeTextCanvas(), { label: 'jfa-text' });
    const U = gpu.uniforms(
      {
        size: 'vec2u', canvas: 'vec2f',
        mouse: 'vec2f', pmouse: 'vec2f',
        scale: 'f32', example: 'f32', brush: 'f32', painting: 'f32',
        paintId: 'u32', sprites: 'f32', starRot: 'f32', nextStep: 'f32',
        heroDst: 'vec4f', heroSrc: 'vec4f', textDst: 'vec4f', starDst: 'vec4f', starSrc: 'vec4f',
        tint: 'vec3f', effect: 'f32',
        tint2: 'vec3f', width: 'f32',
        glow: 'f32', influence: 'f32', borders: 'f32', showSeeds: 'f32',
        mouseOver: 'f32',
      },
      'U',
    );
    const steps = Array.from({ length: 16 }, (_, k) => {
      const b = gpu.uniforms({ step: 'i32' }, 'Pass');
      b.set('step', 1 << k).upload();
      return b;
    });
    const sim = gpu.compute({
      label: 'jfa',
      bindings: {
        u: { uniform: U },
        ps: { uniform: steps[0] },
        paint: { storage: 'array<u32>', access: 'read_write' },
        mask: { storage: 'array<u32>', access: 'read_write' },
        seedIn: { storage: 'array<vec2u>', access: 'read' },
        seedOut: { storage: 'array<vec2u>', access: 'read_write' },
        atlas: { texture: true },
        textTex: { texture: true },
      },
      code: SIM_WGSL,
    });
    const draw = gpu.program({
      label: 'jfa-draw',
      bindings: {
        u: { uniform: U },
        seeds: { storage: 'array<vec2u>', access: 'read' },
        mask: { storage: 'array<u32>', access: 'read' },
        atlas: { texture: true },
      },
      code: DRAW_WGSL,
    });
    const pipe = draw.renderPipeline({ format: gpu.format });
    const readout = overlayTag(ctx);

    const exIndex = Math.max(0, examples.findIndex((e) => e.id === ctx.example));
    let G = null; // grid state
    let nextId = 100;
    let lastM = null;

    const fr = (name) => {
      const f = atlas.frames[name];
      return [f.x, f.y, f.w, f.h];
    };

    const seedInitial = (paint, w, h, scale) => {
      const put = (x, y, r, id) => {
        for (let j = -r; j <= r; j++)
          for (let i = -r; i <= r; i++) {
            const X = Math.round(x + i);
            const Y = Math.round(y + j);
            if (i * i + j * j <= r * r && X >= 0 && Y >= 0 && X < w && Y < h) paint[Y * w + X] = id;
          }
      };
      if (exIndex === 0) {
        for (let k = 0; k < 22; k++) put(Math.random() * w, Math.random() * h, Math.max(1, Math.round(3 * scale)), k + 1);
      } else if (exIndex === 2) {
        let placed = 0;
        for (let tries = 0; tries < 2000 && placed < 16; tries++) {
          const x = Math.random();
          const y = Math.random();
          const mx = (x - 0.5) * (w / h) * 3;
          const my = (y - 0.5) * 3;
          if (landJS(mx, my) > 0.12) put(x * w, y * h, Math.max(1, Math.round(4 * scale)), ++placed);
        }
      }
    };

    const build = () => {
      const scale = parseFloat(ctx.params.scaleMode) || 0.5;
      const sc = ctx.testMode ? Math.min(scale, 0.5) : scale;
      const w = Math.max(8, Math.round(ctx.width * sc));
      const h = Math.max(8, Math.round(ctx.height * sc));
      const n = w * h;
      if (G) for (const b of [G.paint, G.mask, G.a, G.b]) b.destroy();
      const paint = new Uint32Array(n);
      seedInitial(paint, w, h, sc);
      G = {
        w, h, scale: sc,
        paint: gpu.storage(paint, 'jfa-paint'),
        mask: gpu.storage(n * 4, 'jfa-mask'),
        a: gpu.storage(n * 8, 'jfa-seeds-a'),
        b: gpu.storage(n * 8, 'jfa-seeds-b'),
        K: Math.ceil(Math.log2(Math.max(w, h))),
      };
      const res = (stepBlock, seedIn, seedOut) => ({ u: U, ps: stepBlock, paint: G.paint, mask: G.mask, seedIn, seedOut, atlas: atlasTex, textTex });
      G.bgInit = sim.bind(res(steps[0], G.b, G.a));
      // pass k reads A (even k) or B (odd k)
      G.bgPass = Array.from({ length: G.K }, (_, k) => {
        const st = steps[G.K - 1 - k];
        return k % 2 === 0 ? sim.bind(res(st, G.a, G.b)) : sim.bind(res(st, G.b, G.a));
      });
      G.drawA = draw.bind({ u: U, seeds: G.a, mask: G.mask, atlas: atlasTex });
      G.drawB = draw.bind({ u: U, seeds: G.b, mask: G.mask, atlas: atlasTex });
    };
    build();

    return {
      resize() {
        build();
      },
      onChange(key) {
        if (key === 'scaleMode') build();
      },
      onAction(key) {
        if (key === 'reset') build();
      },
      frame(ctx) {
        const p = ctx.params;
        const ptr = ctx.pointer;
        const sc = G.scale;
        const m = [ptr.x * sc, ptr.y * sc];
        const pm = lastM && ptr.down ? lastM : m;
        lastM = m;
        const erase = ptr.down && ptr.button === 2;
        let painting = 0;
        if (ptr.down) {
          painting = erase ? 2 : 1;
          if (ptr.clicked && !erase) nextId++;
        }
        // animated shapes for the outline example (positions in JFA-grid pixels)
        const t = ctx.time;
        const hero = fr(`hero_run_${Math.floor(t * 8) % 4}`);
        const H = G.h;
        const Wd = G.w;
        const heroH = H * 0.5;
        const heroW = (heroH * hero[2]) / hero[3];
        const heroDst = [Wd * 0.27 - heroW / 2, H * 0.5 - heroH / 2 + Math.sin(t * 4) * H * 0.015, heroW, heroH];
        const textW = Math.min(Wd * 0.42, H * 0.95);
        const textH = (textW * 360) / 1024;
        const textDst = [Wd * 0.68 - textW / 2, H * 0.36 - textH / 2, textW, textH];
        const star = fr('star');
        const starS = H * 0.26;
        const starDst = [Wd * 0.68, H * 0.74, starS, 0];

        // which pass are we showing?
        let nPass = G.K;
        if (exIndex === 0) {
          nPass = p.animate ? Math.min(G.K, Math.floor((t * 1.4) % (G.K + 3))) : Math.min(G.K, Math.round(p.passes));
        }
        const nextStep = nPass < G.K ? 1 << (G.K - 1 - nPass) : 0;

        U.set('size', [G.w, G.h])
          .set('canvas', [ctx.width, ctx.height])
          .set('mouse', m)
          .set('pmouse', pm)
          .set('scale', sc)
          .set('example', exIndex)
          .set('brush', Math.max(0.75, p.brush * sc))
          .set('painting', painting)
          .set('paintId', exIndex === 0 ? nextId % 1000 : exIndex === 1 ? 10 + (nextId % 50) : nextId % 1000)
          .set('sprites', exIndex === 1 ? 1 : 0)
          .set('starRot', t * 0.8)
          .set('nextStep', nextStep)
          .set('heroDst', heroDst)
          .set('heroSrc', hero)
          .set('textDst', textDst)
          .set('starDst', starDst)
          .set('starSrc', star)
          .set('tint', p.tint)
          .set('tint2', p.tint2)
          .set('effect', ['combo', 'outline', 'glow', 'neon', 'bevel', 'raw'].indexOf(p.effect))
          .set('width', p.width)
          .set('glow', p.glow)
          .set('influence', p.influence)
          .set('borders', p.borders ? 1 : 0)
          .set('showSeeds', p.showSeeds ? 1 : 0)
          .set('mouseOver', ptr.over ? 1 : 0)
          .upload();

        const enc = ctx.encoder;
        const gx = Math.ceil(G.w / 8);
        const gy = Math.ceil(G.h / 8);
        const pass = enc.beginComputePass({ label: 'jfa' });
        if (painting && !ctx.paused) {
          pass.setPipeline(sim.computePipeline('paintStroke'));
          pass.setBindGroup(0, G.bgInit);
          pass.dispatchWorkgroups(gx, gy);
        }
        pass.setPipeline(sim.computePipeline('compose'));
        pass.setBindGroup(0, G.bgInit); // writes the initial seeds into A
        pass.dispatchWorkgroups(gx, gy);
        pass.setPipeline(sim.computePipeline('jfa'));
        for (let k = 0; k < nPass; k++) {
          pass.setBindGroup(0, G.bgPass[k]);
          pass.dispatchWorkgroups(gx, gy);
        }
        pass.end();
        const finalBG = nPass % 2 === 0 ? G.drawA : G.drawB;
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.target, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
        rp.setPipeline(pipe);
        rp.setBindGroup(0, finalBG);
        rp.draw(3);
        rp.end();
        readout.textContent =
          exIndex === 0
            ? nPass < G.K
              ? `pass ${nPass} / ${G.K} done · next step ${nextStep} px`
              : `all ${G.K} passes done (grid ${G.w}×${G.h})`
            : `${G.K} passes × 9 reads per pixel · grid ${G.w}×${G.h}`;
      },
    };
  },
};

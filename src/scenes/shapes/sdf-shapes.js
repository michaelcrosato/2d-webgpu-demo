import { shaderScene } from '../../core/shaderscene.js';
import { labels, place } from './_shared.js';

// Signed Distance Fields: a shape zoo (with IQ-style distance visualisation), boolean & smooth
// blending with a mouse-driven circle, and a game HUD of SDF icons inspected with a magnifier.
// Everything is one portable-WGSL fragment shader, so it runs on WebGPU and WebGL2.

const ZOO = [
  'circle', 'box', 'triangle', 'hexagon', 'star',
  'heart', 'rhombus', 'cross', 'moon', 'pie',
  'arc', 'vesica', 'ellipse', 'capsule', 'quadratic bézier',
];
const OPS = [
  'union · <b>min(a, b)</b>', 'subtract · <b>max(a, −b)</b>', 'intersect · <b>max(a, b)</b>',
  'smooth union · <b>k</b>', 'smooth subtract · <b>k</b>', 'smooth intersect · <b>k</b>',
];
const LABEL_STYLE = 'background:#0009;border-color:#ffffff14;font-size:11px;padding:2px 7px;color:#dbe4f3';

function lensState(ctx) {
  const p = ctx.pointer;
  const R = ctx.height * 0.2;
  let x;
  let y;
  if (p.over) {
    x = p.x;
    y = p.y;
  } else {
    const t = ctx.time;
    x = ctx.width * (0.5 + 0.2 * Math.sin(t * 0.37));
    y = ctx.height * (0.84 - 0.06 * Math.cos(t * 0.53));
  }
  return [x, y, R];
}

export default shaderScene({
  interaction: 'Zoo: turn on “Show distance field” and hover a shape. Booleans: the mouse moves the circle. HUD: the mouse is a magnifying glass.',
  examples: [
    {
      id: 'zoo',
      label: 'Shape zoo',
      kind: 'Abstract',
      note: 'Fifteen shapes, each a few lines of math returning the <b>signed distance</b> to its edge: negative inside, positive outside, zero on the border. Rounding, onion rings, outlines and glow are all one-line tweaks of that number.',
    },
    {
      id: 'boolean',
      label: 'Booleans & smooth blending',
      kind: 'Abstract',
      note: 'Combine two distance fields with <code>min</code>/<code>max</code> to get union, subtraction and intersection. The bottom row uses <b>smooth</b> versions that melt the shapes together over a radius <i>k</i>. Move the mouse to push the circle around.',
      params: { outline: 0, glow: 0.25 },
    },
    {
      id: 'hud',
      label: 'Crisp vector HUD',
      kind: 'In a game',
      note: 'Every icon, pin, badge and reticle here is an SDF evaluated per pixel — no textures. The magnifier proves it: the left half re-evaluates the math at the zoomed position (always sharp), the right half shows what a bitmap of the same HUD would look like (blurry texels).',
      params: { glow: 0.5 },
    },
  ],
  controls: [
    { type: 'heading', label: 'Distance tweaks' },
    { type: 'toggle', key: 'fieldView', label: 'Show distance field', value: false, showFor: ['zoo', 'boolean'], help: 'Color = distance: blue inside, orange outside, iso-lines every few pixels, white = the edge (d = 0).' },
    { type: 'slider', key: 'rounding', label: 'Rounding  (d − r)', min: 0, max: 0.12, step: 0.001, value: 0, showFor: ['zoo'], help: 'Subtracting a constant grows the shape and rounds every corner.' },
    { type: 'slider', key: 'onion', label: 'Onion / annular  (|d| − t)', min: 0, max: 0.06, step: 0.001, value: 0, showFor: ['zoo'], help: 'abs(d) turns any filled shape into a hollow ring of thickness 2t.' },
    { type: 'slider', key: 'outline', label: 'Outline width', min: 0, max: 0.05, step: 0.001, value: 0.012, showFor: ['zoo', 'boolean'], help: 'An outer stroke: pixels with 0 < d < width.' },
    { type: 'slider', key: 'glow', label: 'Glow', min: 0, max: 1.5, step: 0.01, value: 0.35, help: 'exp(−d / radius) outside the shape.' },
    { type: 'slider', key: 'k', label: 'Smooth blend radius k', min: 0.005, max: 0.25, step: 0.001, value: 0.08, showFor: ['boolean'], help: 'How far apart two shapes start to melt together.' },
    { type: 'toggle', key: 'spin', label: 'Rotate shapes', value: true, showFor: ['zoo'] },
    { type: 'color', key: 'tint', label: 'Fill color', value: '#f472b6', showFor: ['zoo', 'boolean'] },
    { type: 'heading', label: 'HUD', showFor: ['hud'] },
    { type: 'slider', key: 'uiScale', label: 'UI scale', min: 0.5, max: 2.5, step: 0.01, value: 1, showFor: ['hud'], help: 'Like a game’s “HUD size” option. SDF icons are re-rendered at the new size, not resampled.' },
    { type: 'slider', key: 'lensZoom', label: 'Magnifier zoom', min: 1, max: 16, step: 0.1, value: 6, log: true, showFor: ['hud'], format: (v) => `${v.toFixed(1)}×` },
    { type: 'toggle', key: 'compare', label: 'Split: SDF | bitmap', value: true, showFor: ['hud'], help: 'Right half of the lens: the same HUD as a fixed-resolution image, scaled up with bilinear filtering.' },
  ],
  uniforms: {
    fieldView: 'f32', rounding: 'f32', onion: 'f32', outline: 'f32', glow: 'f32', k: 'f32', spin: 'f32', tint: 'vec3f',
    uiScale: 'f32', lensZoom: 'f32', compare: 'f32', lens: 'vec4f',
  },
  include: ['sdf', 'math', 'noise'],
  bind(params, ctx) {
    const ex = ctx.example;
    if (ex === 'zoo') {
      labels(ctx, 'zoo', ZOO.map((n, i) => ({ text: n, x: ((i % 5) + 0.5) / 5, y: (Math.floor(i / 5) + 0.88) / 3, valign: 'middle', style: LABEL_STYLE })));
    } else if (ex === 'boolean') {
      labels(ctx, 'boolean', OPS.map((n, i) => ({ text: n, x: ((i % 3) + 0.5) / 3, y: (Math.floor(i / 3) + 0.9) / 2, valign: 'middle', style: LABEL_STYLE })));
    }
    const [lx, ly, lr] = lensState(ctx);
    if (ex === 'hud') {
      const els = labels(ctx, 'hud', [
        { text: 'SDF: math re-evaluated', style: LABEL_STYLE },
        { text: 'bitmap: texels stretched', style: LABEL_STYLE },
      ]);
      const W = ctx.width;
      const H = ctx.height;
      const showB = !!params.compare;
      els[1].style.display = showB ? '' : 'none';
      const below = ly - lr - 30 < 0;
      const yy = (below ? ly + lr + 8 : ly - lr - 8) / H;
      place(els[0], (showB ? lx - 6 : lx) / W, yy, showB ? 'right' : 'center', below ? 'top' : 'bottom');
      place(els[1], (lx + 6) / W, yy, 'left', below ? 'top' : 'bottom');
    }
    return { lens: [lx, ly, lr, params.lensZoom] };
  },
  code: /* wgsl */ `
fn cov(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }

// ------------------------------------------------------------------ the zoo
fn zooShape(id: i32, q: vec2f) -> f32 {
  if (id == 0) { return sdCircle(q, 0.29); }
  if (id == 1) { return sdBox(q, vec2f(0.3, 0.19)); }
  if (id == 2) { return sdEquilateralTriangle(vec2f(q.x, -q.y + 0.05), 0.3); }
  if (id == 3) { return sdHexagon(q, 0.26); }
  if (id == 4) { return sdStar5(q, 0.33, 0.48); }
  if (id == 5) { return sdHeart(q / 0.5 + vec2f(0.0, 0.05)) * 0.5; }
  if (id == 6) { return sdRhombus(q, vec2f(0.32, 0.21)); }
  if (id == 7) { return sdCross(q, vec2f(0.3, 0.1), 0.0); }
  if (id == 8) { return sdMoon(q + vec2f(0.05, 0.0), 0.16, 0.29, 0.24); }
  if (id == 9) { return sdPie(q + vec2f(0.0, -0.08), vec2f(sin(1.0), cos(1.0)), 0.32); }
  if (id == 10) { return sdArc(q, vec2f(sin(2.2), cos(2.2)), 0.25, 0.045); }
  if (id == 11) { return sdVesica(q.yx, 0.32, 0.13); }
  if (id == 12) { return sdEllipseApprox(q, vec2f(0.33, 0.19)); }
  if (id == 13) { return sdSegment(q, vec2f(-0.22, 0.12), vec2f(0.22, -0.12)) - 0.08; }
  return sdBezier(q, vec2f(-0.3, 0.16), vec2f(0.0, -0.5), vec2f(0.3, 0.16)) - 0.045;
}

// Inigo Quilez's classic distance-field coloring
fn fieldColor(d: f32, pw: f32) -> vec3f {
  var c = mix(vec3f(0.93, 0.62, 0.33), vec3f(0.42, 0.7, 0.98), step(d, 0.0));
  c *= 1.0 - exp(-7.0 * abs(d));
  c *= 0.78 + 0.22 * cos(140.0 * d);
  return mix(c, vec3f(1.0), clamp(1.6 - abs(d) / pw, 0.0, 1.0));
}

// fill + outer outline + glow, all from one distance d (local units)
fn styleShape(bg: vec3f, d: f32, pw: f32, fillCol: vec3f) -> vec3f {
  var c = bg + fillCol * u.glow * 0.85 * exp(-max(d, 0.0) / 0.07) * (1.0 - cov(d, pw));
  if (u.outline > 0.0005) {
    let od = abs(d - u.outline * 0.5) - u.outline * 0.5;
    c = mix(c, vec3f(0.97, 0.97, 1.0), cov(od, pw));
  }
  // "pillow" shading: brighter towards the inside, using the same distance
  let inner = clamp(-d * 7.0, 0.0, 1.0);
  let fc = mix(fillCol * 0.72, min(fillCol * 1.18 + vec3f(0.05), vec3f(1.0)), inner);
  return mix(c, fc, cov(d, pw));
}

fn zooView(px: vec2f) -> vec3f {
  let cols = 5.0;
  let rows = 3.0;
  let cs = u.resolution / vec2f(cols, rows);
  let cell = min(floor(px / cs), vec2f(cols - 1.0, rows - 1.0));
  let unit = min(cs.x, cs.y);
  let pw = 1.0 / unit;
  let center = (cell + vec2f(0.5, 0.44)) * cs;
  let ang = u.time * 0.35 * u.spin;
  let id = i32(cell.y * cols + cell.x);
  let rot = rot2(-(ang + f32(id) * 0.7) * step(0.5, f32(id)));
  let q = rot * ((px - center) / unit);

  var d = zooShape(id, q) - u.rounding;
  if (u.onion > 0.0005) { d = abs(d) - u.onion; }

  // mouse: which cell is it in?
  let mcell = min(floor(u.mouse.xy / cs), vec2f(cols - 1.0, rows - 1.0));
  let hover = u.mouse.w * step(abs(mcell.x - cell.x) + abs(mcell.y - cell.y), 0.5);

  var c: vec3f;
  if (u.fieldView > 0.5) {
    c = fieldColor(d, pw);
    if (hover > 0.5) {
      let mq = rot * ((u.mouse.xy - center) / unit);
      var md = zooShape(id, mq) - u.rounding;
      if (u.onion > 0.0005) { md = abs(md) - u.onion; }
      let r = abs(md);
      let ringD = abs(length(q - mq) - r) - pw * 0.8;
      c = mix(c, vec3f(1.0, 0.86, 0.25), cov(ringD, pw));
      c = mix(c, vec3f(1.0, 0.86, 0.25), cov(length(q - mq) - 0.012, pw));
    }
    let edge = abs(fract(px / cs) - vec2f(0.5));
    c *= 0.55 + 0.45 * smoothstep(0.497, 0.49, max(edge.x, edge.y));
  } else {
    let bg = mix(vec3f(0.045, 0.05, 0.075), vec3f(0.07, 0.06, 0.1), px.y / u.resolution.y);
    let card = sdRoundBox(px - (cell + vec2f(0.5)) * cs, cs * 0.5 - vec2f(5.0), 14.0);
    var base = mix(bg, vec3f(0.085, 0.09, 0.13) + hover * vec3f(0.03, 0.025, 0.045), cov(card, 1.0));
    base = mix(base, vec3f(0.95, 0.5, 0.75), cov(abs(card) - 0.6, 1.0) * hover * 0.6);
    c = styleShape(base, d, pw, u.tint);
  }
  return c;
}

// ------------------------------------------------------------------ booleans
fn boolView(px: vec2f) -> vec3f {
  let cols = 3.0;
  let rows = 2.0;
  let cs = u.resolution / vec2f(cols, rows);
  let cell = min(floor(px / cs), vec2f(cols - 1.0, rows - 1.0));
  let unit = min(cs.x, cs.y) * 0.9;
  let pw = 1.0 / unit;
  let center = (cell + vec2f(0.5, 0.45)) * cs;
  let q = (px - center) / unit;
  let id = i32(cell.y * cols + cell.x);

  // the circle follows the mouse (same relative position in every tile)
  var bpos = vec2f(0.27 * cos(u.time * 0.7), 0.15 * sin(u.time * 1.3));
  if (u.mouse.w > 0.5) {
    let mcell = min(floor(u.mouse.xy / cs), vec2f(cols - 1.0, rows - 1.0));
    let mq = (u.mouse.xy - (mcell + vec2f(0.5, 0.45)) * cs) / unit;
    bpos = clamp(mq, vec2f(-0.42, -0.3), vec2f(0.42, 0.3));
  }
  let a = sdRoundBox(q, vec2f(0.27, 0.16), 0.04);
  let b = sdCircle(q - bpos, 0.17);
  let k = max(u.k, 0.0001);
  var d: f32;
  if (id == 0) { d = min(a, b); }
  else if (id == 1) { d = max(a, -b); }
  else if (id == 2) { d = max(a, b); }
  else if (id == 3) { d = opSmoothUnion(a, b, k); }
  else if (id == 4) { d = opSmoothSubtract(a, b, k); }
  else { d = opSmoothIntersect(a, b, k); }

  var c: vec3f;
  if (u.fieldView > 0.5) {
    c = fieldColor(d, pw);
    let edge = abs(fract(px / cs) - vec2f(0.5));
    c *= 0.55 + 0.45 * smoothstep(0.497, 0.49, max(edge.x, edge.y));
  } else {
    let bg = mix(vec3f(0.045, 0.05, 0.075), vec3f(0.07, 0.06, 0.1), px.y / u.resolution.y);
    let card = sdRoundBox(px - (cell + vec2f(0.5)) * cs, cs * 0.5 - vec2f(5.0), 14.0);
    var base = mix(bg, vec3f(0.085, 0.09, 0.13), cov(card, 1.0));
    // blue tint behind the smooth row
    base += vec3f(0.0, 0.01, 0.03) * cell.y * cov(card, 1.0);
    c = styleShape(base, d, pw, u.tint);
  }
  // ghost outlines of the two input shapes
  let ghost = max(cov(abs(a) - pw * 0.5, pw), cov(abs(b) - pw * 0.5, pw));
  c = mix(c, vec3f(0.75, 0.85, 1.0), ghost * 0.35);
  return c;
}

// ------------------------------------------------------------------ the HUD (units: 1080 px tall)
fn icoExcl(q: vec2f) -> f32 { return min(sdSegment(q, vec2f(0.0, -0.55), vec2f(0.0, 0.1)) - 0.13, sdCircle(q - vec2f(0.0, 0.5), 0.14)); }
fn icoHouse(q: vec2f) -> f32 {
  let body = sdBox(q - vec2f(0.0, 0.22), vec2f(0.44, 0.38));
  let roof = sdTriangle(q, vec2f(-0.72, -0.06), vec2f(0.72, -0.06), vec2f(0.0, -0.72));
  let door = sdBox(q - vec2f(0.0, 0.42), vec2f(0.12, 0.2));
  return max(min(body, roof), -door);
}
fn icoSwords(q: vec2f) -> f32 {
  let a = sdSegment(q, vec2f(-0.5, -0.5), vec2f(0.5, 0.5)) - 0.11;
  let b = sdSegment(q, vec2f(0.5, -0.5), vec2f(-0.5, 0.5)) - 0.11;
  let ga = sdSegment(q, vec2f(-0.62, 0.22), vec2f(-0.22, 0.62)) - 0.07;
  let gb = sdSegment(q, vec2f(0.62, 0.22), vec2f(0.22, 0.62)) - 0.07;
  return min(min(a, b), min(ga, gb));
}
fn icoSword(q0: vec2f) -> f32 {
  let q = rot2(-0.785398) * q0;
  let blade = min(sdBox(q - vec2f(0.0, -0.2), vec2f(0.12, 0.48)), sdTriangle(q, vec2f(-0.12, -0.67), vec2f(0.12, -0.67), vec2f(0.0, -0.95)));
  let guard = sdRoundBox(q - vec2f(0.0, 0.33), vec2f(0.38, 0.08), 0.06);
  let grip = sdBox(q - vec2f(0.0, 0.56), vec2f(0.065, 0.18));
  let pommel = sdCircle(q - vec2f(0.0, 0.8), 0.11);
  return min(min(blade, guard), min(grip, pommel));
}
fn icoShield(q: vec2f) -> f32 {
  let top = sdRoundBox(q - vec2f(0.0, -0.28), vec2f(0.62, 0.45), 0.12);
  let bottom = sdTriangle(q, vec2f(-0.62, -0.12), vec2f(0.62, -0.12), vec2f(0.0, 0.9));
  return opSmoothUnion(top, bottom, 0.1);
}
fn icoHeart(q: vec2f) -> f32 { return sdHeart(q / 0.8 + vec2f(0.0, 0.04)) * 0.8; }
fn icoBolt(q: vec2f) -> f32 {
  let t1 = sdTriangle(q, vec2f(0.34, -0.95), vec2f(-0.5, 0.16), vec2f(0.18, 0.02));
  let t2 = sdTriangle(q, vec2f(-0.34, 0.95), vec2f(0.5, -0.16), vec2f(-0.18, -0.02));
  return min(t1, t2);
}
fn pinShape(q: vec2f) -> f32 {
  let head = sdCircle(q - vec2f(0.0, -0.32), 0.62);
  let tip = sdTriangle(q, vec2f(-0.48, 0.0), vec2f(0.48, 0.0), vec2f(0.0, 1.0));
  return opSmoothUnion(head, tip, 0.12);
}

// one map pin; returns (rgb, alpha) composited by caller
fn drawPin(c0: vec3f, p: vec2f, at: vec2f, sz: f32, col: vec3f, icon: i32, pw: f32) -> vec3f {
  var c = c0;
  let q = (p - at) / sz;
  let lpw = pw / sz;
  if (abs(q.x) > 1.6 || q.y < -1.6 || q.y > 1.6) { return c; }
  // ground shadow & ring
  let sh = sdEllipseApprox(q - vec2f(0.0, 1.02), vec2f(0.42, 0.13));
  c = mix(c, c * 0.35, cov(sh, lpw) * 0.8);
  let d = pinShape(q);
  c = mix(c, vec3f(0.02, 0.02, 0.04), cov(d - 0.1, lpw) * 0.9);
  c = mix(c, col * mix(0.75, 1.15, smoothstep(0.4, -0.9, q.y)), cov(d, lpw));
  let inner = sdCircle(q - vec2f(0.0, -0.32), 0.42);
  c = mix(c, vec3f(0.97, 0.96, 0.92), cov(inner, lpw));
  let iq = (q - vec2f(0.0, -0.32)) / 0.3;
  var id: f32;
  if (icon == 0) { id = icoExcl(iq); }
  else if (icon == 1) { id = icoHouse(iq); }
  else if (icon == 2) { id = sdStar5(iq, 0.85, 0.45); }
  else { id = icoSwords(iq); }
  c = mix(c, col * 0.55, cov(id * 0.3, lpw));
  return c;
}

fn mapBg(p: vec2f, pw: f32, W: f32) -> vec3f {
  let wp = p / 380.0 + vec2f(3.1, 1.7) + vec2f(u.time * 0.006, 0.0);
  let h = fbm(wp, 5) + 0.06;
  let hw = pw * 0.006;
  let land = clamp(0.5 + h / max(hw, 0.0005), 0.0, 1.0);
  var water = mix(vec3f(0.03, 0.09, 0.16), vec3f(0.06, 0.2, 0.28), smoothstep(-0.25, 0.0, h));
  water += vec3f(0.05, 0.1, 0.12) * smoothstep(-0.03, 0.0, h);
  let ground = mix(vec3f(0.12, 0.2, 0.13), vec3f(0.3, 0.27, 0.17), smoothstep(0.05, 0.4, h));
  var c = mix(water, ground, land);
  // contour lines
  let iso = abs(fract(h * 9.0) - 0.5) / 9.0;
  c = mix(c, c * 1.35 + vec3f(0.02), clamp(1.0 - iso / max(hw * 1.2, 0.0003), 0.0, 1.0) * 0.5 * land);
  // map grid
  let g = abs(fract(p / 135.0) - vec2f(0.5)) * 135.0;
  let gd = 67.5 - max(g.x, g.y);
  c = mix(c, vec3f(0.35, 0.6, 0.7), cov(gd - pw * 0.5, pw) * 0.12);
  // vignette
  let v = length((p - vec2f(W * 0.5, 540.0)) / vec2f(W, 1080.0));
  return c * (1.05 - 0.7 * v * v);
}

fn hud(p: vec2f, pw: f32, W: f32) -> vec3f {
  let S = u.uiScale;
  let t = u.time;
  var c = mapBg(p, pw, W);

  // ---- map pins (world space: they don't scale with the UI)
  let bob = 6.0 * sin(t * 2.2);
  c = drawPin(c, p, vec2f(W * 0.3, 380.0 + bob), 34.0, vec3f(0.98, 0.72, 0.18), 0, pw);
  c = drawPin(c, p, vec2f(W * 0.72, 300.0 - bob), 30.0, vec3f(0.3, 0.6, 0.98), 1, pw);
  c = drawPin(c, p, vec2f(W * 0.2, 700.0 + bob * 0.6), 30.0, vec3f(0.72, 0.42, 0.98), 2, pw);

  // ---- player arrow + radar ping at the center
  let pc = vec2f(W * 0.5, 560.0);
  let pq = rot2(-0.4 * sin(t * 0.5)) * (p - pc);
  let ping = fract(t * 0.45);
  let pr = abs(length(p - pc) - ping * 150.0) - 1.5;
  c = mix(c, vec3f(0.5, 1.0, 0.85), cov(pr, pw) * (1.0 - ping) * 0.8);
  let arrow = max(sdTriangle(pq, vec2f(0.0, -26.0), vec2f(19.0, 20.0), vec2f(-19.0, 20.0)), -sdTriangle(pq, vec2f(0.0, 6.0), vec2f(14.0, 26.0), vec2f(-14.0, 26.0)));
  c = mix(c, vec3f(0.0), cov(arrow - 3.0, pw) * 0.8);
  c = mix(c, vec3f(0.55, 1.0, 0.85), cov(arrow, pw));

  // ---- enemy marker with a lock-on reticle (a crosshair)
  let ec = vec2f(W * 0.66, 600.0);
  let eq = p - ec;
  let rh = sdRhombus(eq, vec2f(22.0, 28.0));
  c = mix(c, vec3f(0.0), cov(rh - 3.0, pw) * 0.8);
  c = mix(c, vec3f(0.92, 0.22, 0.25), cov(rh, pw));
  c = mix(c, vec3f(1.0, 0.92, 0.9), cov(icoSwords(eq / 13.0) * 13.0, pw));
  let rq = rot2(t * 0.8) * eq;
  let ring = abs(length(eq) - 52.0) - 1.5;
  let tk = min(sdBox(vec2f(abs(rq.x) - 62.0, rq.y), vec2f(10.0, 1.8)), sdBox(vec2f(rq.x, abs(rq.y) - 62.0), vec2f(1.8, 10.0)));
  let reticle = min(max(ring, -min(sdBox(rq, vec2f(9.0, 70.0)), sdBox(rq, vec2f(70.0, 9.0)))), tk);
  let pulse = 0.75 + 0.25 * sin(t * 6.0);
  c += vec3f(1.0, 0.3, 0.3) * u.glow * 0.5 * exp(-max(reticle, 0.0) / 6.0) * pulse;
  c = mix(c, vec3f(1.0, 0.45, 0.42), cov(reticle, pw) * pulse);

  // ---- top-left: rank badge + hearts
  let bc = vec2f(70.0 * S, 72.0 * S);
  let bq = (p - bc) / S;
  let hex = sdHexagon(vec2f(bq.y, bq.x), 44.0);
  c += vec3f(1.0, 0.75, 0.3) * u.glow * 0.4 * exp(-max(hex, 0.0) / (10.0 * S));
  c = mix(c, vec3f(0.0), cov((hex - 4.0) * S, pw) * 0.8);
  c = mix(c, vec3f(0.95, 0.7, 0.25), cov(hex * S, pw));
  c = mix(c, vec3f(0.12, 0.1, 0.2), cov((hex + 6.0) * S, pw));
  let st = sdStar5(bq + vec2f(0.0, 2.0), 26.0, 0.48);
  c = mix(c, mix(vec3f(1.0, 0.86, 0.4), vec3f(1.0, 0.6, 0.2), smoothstep(-20.0, 20.0, bq.y)), cov(st * S, pw));
  for (var i = 0; i < 3; i++) {
    let hc = vec2f((150.0 + f32(i) * 58.0) * S, 60.0 * S);
    let hq = (p - hc) / (24.0 * S);
    let hd = icoHeart(hq) * 24.0 * S;
    c = mix(c, vec3f(0.02), cov(hd - 3.5 * S, pw) * 0.85);
    var fillA = cov(hd, pw);
    if (i == 2) { fillA *= cov(hq.x * 24.0 * S, pw); }
    c = mix(c, vec3f(0.3, 0.1, 0.14), cov(hd, pw));
    c = mix(c, mix(vec3f(1.0, 0.35, 0.4), vec3f(0.85, 0.12, 0.25), smoothstep(-0.5, 0.6, hq.y)), fillA);
    c = mix(c, vec3f(1.0, 0.85, 0.85), cov((length(hq - vec2f(-0.32, -0.3)) - 0.13) * 24.0 * S, pw) * fillA);
  }

  // ---- top-right: compass
  let cc = vec2f(W - 90.0 * S, 90.0 * S);
  let cq = (p - cc) / S;
  let cr = length(cq);
  c = mix(c, vec3f(0.03, 0.05, 0.08), cov((cr - 64.0) * S, pw) * 0.75);
  c = mix(c, vec3f(0.7, 0.85, 0.95), cov((abs(cr - 62.0) - 2.0) * S, pw));
  let ca = atan2(cq.y, cq.x) + t * 0.2;
  let sector = TAU / 24.0;
  let la = (fract(ca / sector + 0.5) - 0.5) * sector * cr;
  let tickD = max(abs(la) - 1.2, abs(cr - 52.0) - 5.0);
  c = mix(c, vec3f(0.7, 0.85, 0.95), cov(tickD * S, pw) * 0.8);
  let nq = rot2(t * 0.2) * cq;
  let needleN = sdTriangle(nq, vec2f(-9.0, 0.0), vec2f(9.0, 0.0), vec2f(0.0, -42.0));
  let needleS = sdTriangle(nq, vec2f(-9.0, 0.0), vec2f(9.0, 0.0), vec2f(0.0, 42.0));
  c = mix(c, vec3f(0.95, 0.3, 0.3), cov(needleN * S, pw));
  c = mix(c, vec3f(0.85, 0.88, 0.95), cov(needleS * S, pw));
  c = mix(c, vec3f(0.1), cov((cr - 5.0) * S, pw));

  // ---- bottom: ability bar
  let slotHalf = 46.0;
  let gapX = 112.0;
  let barY = 1080.0 - 78.0 * S;
  let barQ = (p - vec2f(W * 0.5, barY)) / S;
  if (abs(barQ.x) < 260.0 && abs(barQ.y) < 80.0) {
    let tray = sdRoundBox(barQ, vec2f(236.0, 62.0), 20.0);
    c = mix(c, vec3f(0.02, 0.02, 0.04), cov(tray * S, pw) * 0.7);
    for (var i = 0; i < 4; i++) {
      let sq = barQ - vec2f((f32(i) - 1.5) * gapX, 0.0);
      if (abs(sq.x) < 60.0) {
        let sd = sdRoundBox(sq, vec2f(slotHalf), 14.0);
        var accent = vec3f(0.95, 0.75, 0.3);
        var icon: f32;
        let iq = sq / 30.0;
        if (i == 0) { icon = icoSword(iq); accent = vec3f(1.0, 0.55, 0.3); }
        else if (i == 1) { icon = icoShield(iq); accent = vec3f(0.4, 0.7, 1.0); }
        else if (i == 2) { icon = icoHeart(iq); accent = vec3f(1.0, 0.4, 0.5); }
        else { icon = icoBolt(iq); accent = vec3f(1.0, 0.9, 0.35); }
        c = mix(c, mix(vec3f(0.14, 0.15, 0.22), vec3f(0.06, 0.06, 0.1), smoothstep(-46.0, 46.0, sq.y)), cov(sd * S, pw));
        c += accent * u.glow * 0.35 * exp(-max(icon * 30.0, 0.0) / 9.0) * cov(sd * S, pw);
        let idd = icon * 30.0;
        c = mix(c, vec3f(0.0), cov((idd - 3.0) * S, pw) * 0.7);
        c = mix(c, mix(accent, vec3f(1.0), smoothstep(10.0, -30.0, sq.y) * 0.45), cov(idd * S, pw));
        // radial cooldown wipe on the shield slot
        if (i == 1) {
          let prog = fract(t * 0.18);
          let a = fract(atan2(sq.x, -sq.y) / TAU + 1.0);
          let edgeD = (prog - a) * TAU * length(sq);
          let shade = cov(-edgeD * S, pw) * cov(sd * S, pw);
          c = mix(c, c * 0.25, shade * 0.85);
        }
        c = mix(c, accent * 0.9, cov((abs(sd) - 2.0) * S, pw));
        c = mix(c, vec3f(1.0), cov((abs(sd + 5.0) - 0.8) * S, pw) * 0.08);
      }
    }
  }
  return c;
}

fn hudView(px: vec2f) -> vec3f {
  let k = 1080.0 / u.resolution.y;
  let W = u.resolution.x * k;
  let pw = k;
  let lp = px - u.lens.xy;
  let ld = length(lp);
  let R = u.lens.z;
  if (ld > R + 4.0) { return hud(px * k, pw, W); }

  let zoom = max(u.lensZoom, 1.0);
  let src = u.lens.xy + lp / zoom;
  var c: vec3f;
  if (u.compare > 0.5 && lp.x > 0.0) {
    // bitmap: the HUD rendered at native resolution, magnified with bilinear filtering
    let tp = src - vec2f(0.5);
    let i0 = floor(tp);
    let f = tp - i0;
    let c00 = hud((i0 + vec2f(0.5, 0.5)) * k, pw, W);
    let c10 = hud((i0 + vec2f(1.5, 0.5)) * k, pw, W);
    let c01 = hud((i0 + vec2f(0.5, 1.5)) * k, pw, W);
    let c11 = hud((i0 + vec2f(1.5, 1.5)) * k, pw, W);
    c = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
  } else {
    c = hud(src * k, pw / zoom, W);
  }
  // lens rim, divider and outside shadow
  let rimD = ld - R;
  c = mix(c, vec3f(0.95, 0.97, 1.0), cov(abs(rimD + 1.5) - 1.5, 1.0));
  if (u.compare > 0.5) { c = mix(c, vec3f(0.95, 0.97, 1.0), cov(abs(lp.x) - 1.0, 1.0) * cov(rimD, 1.0)); }
  let outside = hud(px * k, pw, W) * (0.55 + 0.45 * smoothstep(0.0, 4.0, rimD));
  return mix(outside, c, cov(rimD, 1.0));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var c: vec3f;
  if (ex == 0) { c = zooView(px); }
  else if (ex == 1) { c = boolView(px); }
  else { c = hudView(px); }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`,
  about: {
    summary: 'A signed distance field (SDF) is a function that tells every pixel how far it is from a shape’s edge. From that one number you get perfectly anti-aliased fills, outlines, glows, rounding and smooth blends — at any resolution.',
    what: `<p><b>Shape zoo</b>: fifteen classic 2D SDFs, rotating. Turn on <i>Show distance field</i> to see the number itself: blue inside, orange outside,
      one iso-line every few pixels, white where the distance is exactly zero. Hover a shape: the yellow circle around the cursor has radius = distance, so it always just touches the edge.</p>
      <p><b>Booleans</b>: a rounded box (<i>a</i>) and a circle (<i>b</i>, follows the mouse) combined six ways. <b>Crisp vector HUD</b>: a whole game overlay built only from SDFs, under a magnifier.</p>`,
    how: `<ol>
      <li>Each shape is a function <code>sd(p) → distance</code>. A circle is <code>length(p) − r</code>; a box, hexagon or star is a few more lines of geometry (most from Inigo Quilez).</li>
      <li><b>Anti-aliasing for free</b>: coverage = <code>clamp(0.5 − d / pixelSize, 0, 1)</code>. Pixels half-way across the edge get half the color — no MSAA, no textures.</li>
      <li><b>Modifiers</b> are arithmetic on <i>d</i>: <code>d − r</code> rounds and grows, <code>abs(d) − t</code> makes a ring (“onion”), <code>0 &lt; d &lt; w</code> is an outline, <code>exp(−d / R)</code> is a glow.</li>
      <li><b>Booleans</b>: union = <code>min(a, b)</code>, intersection = <code>max(a, b)</code>, subtraction = <code>max(a, −b)</code>. The smooth versions blend the two distances within radius <i>k</i>, producing goo-like fillets.</li>
      <li>Because the shape is math rather than pixels, you can evaluate it at <i>any</i> position and scale — which is why the magnifier’s SDF half stays sharp while the bitmap half blurs.</li>
    </ol>`,
    uses: [
      { title: 'UI & HUD', text: 'Icons, reticles, cooldown pies, rounded panels and outlines that stay crisp at any resolution or UI scale.' },
      { title: 'Text', text: 'SDF fonts (Valve’s 2007 technique, used in TextMeshPro, Godot, Unity UI) — see the Text Rendering scene.' },
      { title: 'Gameplay shapes', text: 'Collision and proximity tests (“how far am I from the wall?”), soft shadows and 2D global illumination by ray-marching distance fields.' },
      { title: 'Procedural art', text: 'Goo, metaballs, liquid UI and morphing logos via smooth unions (e.g. Media Molecule’s Dreams is built on SDF sculpting).' },
    ],
    try: [
      'In the zoo, turn on <i>Show distance field</i> and hover different shapes — notice the iso-lines stay evenly spaced (it is a true distance).',
      'Raise <i>Rounding</i>: every corner rounds and the shapes grow. Then raise <i>Onion</i> to hollow them out.',
      'On <b>Booleans</b>, enable the distance field and compare hard union (sharp crease in the lines) with smooth union (rounded).',
      'Drag <i>Smooth blend radius k</i> to the maximum and bring the circle close to the box — they reach for each other like liquid.',
      'On <b>Crisp vector HUD</b>, set the magnifier to 16× and hover the heart icons; then change <i>UI scale</i>.',
    ],
    ask: [
      'resolution-independent SDF icons for the HUD',
      'a smooth-union “goo” effect between two shapes',
      'SDF outlines and glow around UI elements',
      'a radial cooldown wipe drawn with an SDF',
      'rounded rectangles with anti-aliased borders in a shader',
    ],
    perf: `<p>Each pixel evaluates the SDF of the shapes near it. Simple shapes cost a handful of instructions, so thousands of icons are cheap.
      Complex scenes (like the HUD) stay fast by only evaluating shapes whose bounding box contains the pixel. Smooth booleans cost the same as hard ones.</p>`,
    api: `<p>Pure fragment-shader math: written once in WGSL and auto-translated to GLSL, so it runs identically on WebGPU and WebGL2.
      For many independent shapes, a batch renderer (instanced quads, one SDF each — like this project’s <code>ShapeBatch</code>) is faster than one giant full-screen shader.</p>`,
    code: [
      {
        title: 'Fill, outline and glow from one distance',
        lang: 'wgsl',
        src: `fn cov(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }

var d = zooShape(id, q) - u.rounding;          // rounding: d - r
if (u.onion > 0.0005) { d = abs(d) - u.onion; } // onion:    |d| - t
// glow outside, outline between 0 and width, then the fill
var c = bg + fillCol * u.glow * exp(-max(d, 0.0) / 0.07);
let od = abs(d - u.outline * 0.5) - u.outline * 0.5;
c = mix(c, vec3f(0.97), cov(od, pw));
c = mix(c, fillCol, cov(d, pw));`,
      },
      {
        title: 'Booleans: min / max and the smooth union',
        lang: 'wgsl',
        src: `let a = sdRoundBox(q, vec2f(0.27, 0.16), 0.04);
let b = sdCircle(q - bpos, 0.17);
let uni = min(a, b);      let sub = max(a, -b);     let inter = max(a, b);

fn opSmoothUnion(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}`,
      },
    ],
    links: [
      { title: 'Inigo Quilez — 2D distance functions', url: 'https://iquilezles.org/articles/distfunctions2d/', note: 'the reference list of 2D SDFs' },
      { title: 'Inigo Quilez — smooth minimum', url: 'https://iquilezles.org/articles/smin/' },
    ],
  },
});

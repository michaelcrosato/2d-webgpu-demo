import { shaderScene } from '../../core/shaderscene.js';
import { getAtlas } from '../../core/assets.js';
import { labels } from './_shared.js';

// Game UI on the GPU: animated health/mana/boss bars, radial cooldowns with seven-segment
// countdowns, nine-slice panels from the atlas vs naive stretching, and "juice" (springy buttons,
// low-health heartbeat vignette, hit shake). One portable-WGSL shader → WebGPU and WebGL2.
// Values that need memory (damage chip trail, springs) are simulated in JS in bind().

const LBL = 'background:#000a;font-size:11px;padding:2px 7px;color:#dbe4f3';
const BAR_NAMES = ['plain fill', 'gradient + damage chip', 'segmented', 'glass + shine sweep', 'boss bar · 3 layers'];
let lowRes = false;
let atlasUV = null;
let atlasRequested = false;

// --------------------------------------------------------------------------- JS simulation
const sim = {
  ctx: null,
  hp: 0.82,
  target: 0.82,
  chip: 0.82,
  chipHold: 0,
  lastHit: -10,
  next: 1.2,
  mana: 0.7,
  boss: 2.55,
  coins: 37,
  pop: 0,
  popV: 0,
  btn: [1, 1, 1],
  btnV: [0, 0, 0],
};
function resetSim(ctx) {
  Object.assign(sim, { ctx, hp: 0.82, target: 0.82, chip: 0.82, chipHold: 0, lastHit: -10, next: 1.2, mana: 0.7, boss: 2.55, coins: 37, pop: 0, popV: 0, btn: [1, 1, 1], btnV: [0, 0, 0], coinNext: 1.5 });
}
const rnd = (a, b) => a + Math.random() * (b - a);

function step(params, ctx) {
  if (sim.ctx !== ctx) resetSim(ctx);
  const t = ctx.time;
  const dt = Math.min(ctx.dt, 1 / 20);
  if (params.autoplay) {
    if (t > sim.next) {
      const lowHp = sim.target < 0.28;
      if (lowHp && Math.random() < 0.55) sim.target = Math.min(1, sim.target + rnd(0.35, 0.6));
      else {
        sim.target = Math.max(0.04, sim.target - rnd(0.07, 0.2));
        sim.lastHit = t;
        sim.chipHold = t + 0.55;
      }
      if (Math.random() < 0.4) sim.mana = Math.max(0, sim.mana - rnd(0.15, 0.35));
      sim.next = t + rnd(0.7, 1.5);
    }
  } else if (Math.abs(params.hp - sim.target) > 1e-4) {
    if (params.hp < sim.target) {
      sim.lastHit = t;
      sim.chipHold = t + 0.55;
    }
    sim.target = params.hp;
  }
  // damage is instant, healing fills up smoothly
  if (sim.target < sim.hp) sim.hp = sim.target;
  else sim.hp = Math.min(sim.target, sim.hp + dt * 0.45);
  // the chip waits, then drains
  if (sim.chip < sim.hp) sim.chip = sim.hp;
  else if (t > sim.chipHold) sim.chip = Math.max(sim.hp, sim.chip - dt * 0.6);
  sim.mana = Math.min(1, sim.mana + dt * 0.06);
  sim.boss -= dt * 0.08;
  if (sim.boss < 0) sim.boss = 3;
  // coin counter with a springy "pop"
  if (t > (sim.coinNext || 0)) {
    sim.coins = (sim.coins + 1 + Math.floor(Math.random() * 3)) % 1000;
    sim.popV += 9;
    sim.coinNext = t + rnd(0.9, 1.8);
  }
  const springK = 380;
  const damp = 2 * Math.sqrt(springK) * (1 - params.bounce * 0.88);
  sim.popV += (-springK * sim.pop - damp * sim.popV) * dt;
  sim.pop += sim.popV * dt;
  // buttons: hover / press targets, springs
  const W = ctx.width;
  const H = ctx.height;
  const lay = buttonLayout(W, H);
  const p = ctx.pointer;
  for (let i = 0; i < 3; i++) {
    const cx = W * 0.5 + (i - 1) * lay.spacing;
    const over = p.over && Math.abs(p.x - cx) < lay.hw && Math.abs(p.y - lay.y) < lay.hh;
    if (over && p.clicked) sim.btnV[i] -= 5;
    const tgt = over ? (p.down ? 0.9 : 1.14) : 1;
    sim.btnV[i] += (springK * (tgt - sim.btn[i]) - damp * sim.btnV[i]) * dt;
    sim.btn[i] += sim.btnV[i] * dt;
  }
  return lay;
}
function buttonLayout(W, H) {
  const hh = Math.min(H * 0.075, W * 0.045);
  return { y: H * 0.85, hw: hh * 1.9, hh, spacing: hh * 4.6 };
}

export default shaderScene({
  examples: [
    {
      id: 'bars',
      label: 'Health & mana bars',
      kind: 'In a game',
      hint: '',
      note: 'Five common bar designs driven by a simulated fight. The <b>damage chip</b> (white) shows how much you just lost, then drains — a tiny delay that makes hits readable (Street Fighter, Dark Souls). Segments, glass highlights, shine sweeps and multi-layer boss bars are all a few lines of shader math.',
    },
    {
      id: 'cooldowns',
      label: 'Radial cooldowns',
      kind: 'In a game',
      hint: '',
      note: 'Ability icons with cooldowns. The “clock wipe” is just <code>atan2</code>: compare each pixel’s angle around the icon center with the progress. Countdown digits are seven-segment SDFs; abilities flash and ring when ready; the dash shows <b>charges</b>; the big orb is an ultimate meter.',
    },
    {
      id: 'nineslice',
      label: 'Nine-slice panels',
      kind: 'Comparison',
      hint: '',
      note: 'The same 24×24 pixel-art panel from the atlas. <b>Left</b>: stretched as one image — corners smear and borders change thickness. <b>Right</b>: <b>nine-slice</b> — the 8-pixel corners stay fixed, edges stretch in one direction, the center fills the rest. One small texture → panels of any size.',
    },
    {
      id: 'juice',
      label: 'Juice: springy UI & low HP',
      kind: 'In a game',
      hint: 'Hover and click the buttons.',
      note: '“Juice” = feedback that makes UI feel alive. Buttons use <b>springs</b> (overshoot and settle), the coin counter <b>pops</b>, the HP bar <b>shakes</b> on hits, and at low health the screen desaturates and a red vignette pulses like a heartbeat.',
    },
  ],
  controls: [
    { type: 'heading', label: 'Simulation', showFor: ['bars', 'juice'] },
    { type: 'toggle', key: 'autoplay', label: 'Simulated fight', value: true, showFor: ['bars', 'juice'], help: 'Random hits and heals. Turn off to drive HP yourself.' },
    { type: 'slider', key: 'hp', label: 'HP (when not simulated)', min: 0, max: 1, step: 0.005, value: 0.6, showFor: ['bars', 'juice'], format: (v) => `${Math.round(v * 100)}%` },
    { type: 'slider', key: 'segments', label: 'Segments', min: 2, max: 20, step: 1, value: 10, showFor: ['bars'] },
    { type: 'toggle', key: 'shine', label: 'Shine sweep', value: true, showFor: ['bars'] },
    { type: 'heading', label: 'Cooldowns', showFor: ['cooldowns'] },
    { type: 'select', key: 'cdStyle', label: 'Cooldown style', value: 'wipe', showFor: ['cooldowns'], options: [{ value: 'wipe', label: 'Clock wipe (darken)' }, { value: 'fill', label: 'Fill from bottom' }, { value: 'ring', label: 'Progress ring' }] },
    { type: 'slider', key: 'cdSpeed', label: 'Time speed', min: 0.1, max: 3, step: 0.05, value: 1, showFor: ['cooldowns'] },
    { type: 'toggle', key: 'numbers', label: 'Countdown digits', value: true, showFor: ['cooldowns'] },
    { type: 'heading', label: 'Panel', showFor: ['nineslice'] },
    { type: 'toggle', key: 'animPanel', label: 'Animate size', value: true, showFor: ['nineslice'] },
    { type: 'slider', key: 'panelW', label: 'Width', min: 0.2, max: 1, step: 0.01, value: 0.8, showFor: ['nineslice'] },
    { type: 'slider', key: 'panelH', label: 'Height', min: 0.15, max: 1, step: 0.01, value: 0.55, showFor: ['nineslice'] },
    { type: 'slider', key: 'pixelScale', label: 'Pixel scale', min: 1, max: 8, step: 1, value: 4, showFor: ['nineslice'], help: 'Screen pixels per texture pixel (keep it an integer for crisp pixel art).' },
    { type: 'select', key: 'centerMode', label: 'Edges & center', value: 'stretch', showFor: ['nineslice'], options: [{ value: 'stretch', label: 'Stretch' }, { value: 'tile', label: 'Tile (repeat)' }] },
    { type: 'toggle', key: 'guides', label: 'Show slice guides', value: true, showFor: ['nineslice'] },
    { type: 'heading', label: 'Juice', showFor: ['juice'] },
    { type: 'slider', key: 'bounce', label: 'Bounciness', min: 0, max: 1, step: 0.01, value: 0.7, showFor: ['juice'], help: 'Spring damping: 0 = no overshoot, 1 = jelly.' },
    { type: 'slider', key: 'lowHp', label: 'Low-HP threshold', min: 0, max: 0.6, step: 0.01, value: 0.35, showFor: ['juice'], format: (v) => `${Math.round(v * 100)}%` },
    { type: 'slider', key: 'vignette', label: 'Vignette strength', min: 0, max: 1, step: 0.01, value: 0.75, showFor: ['juice'] },
    { type: 'toggle', key: 'desat', label: 'Desaturate at low HP', value: true, showFor: ['juice'] },
  ],
  uniforms: {
    autoplay: 'f32', segments: 'f32', shine: 'f32', cdStyle: 'f32', cdSpeed: 'f32', numbers: 'f32',
    animPanel: 'f32', panelW: 'f32', panelH: 'f32', pixelScale: 'f32', centerMode: 'f32', guides: 'f32',
    lowHp: 'f32', vignette: 'f32', desat: 'f32',
    sim0: 'vec4f', sim1: 'vec4f', btn: 'vec4f', btnLay: 'vec4f',
    uvPanel: 'vec4f', uvHero: 'vec4f', uvSlime: 'vec4f', uvTile: 'vec4f', uvCoin: 'vec4f',
  },
  textures: { atlas: { source: async () => (await getAtlas()).canvas, filter: 'nearest' } },
  include: ['sdf', 'math', 'hash'],
  renderScale: () => (lowRes ? 0.5 : 1),
  bind(params, ctx) {
    lowRes = !!ctx.testMode;
    if (!atlasRequested) {
      atlasRequested = true;
      getAtlas().then((a) => (atlasUV = a));
    }
    const ex = ctx.example;
    if (ex === 'bars') labels(ctx, 'bars', BAR_NAMES.map((n, i) => ({ text: n, x: 0.235, y: 0.15 + i * 0.17, align: 'right', valign: 'middle', style: LBL })));
    else if (ex === 'nineslice') {
      labels(ctx, 'nine', [
        { text: 'naive stretch', x: 0.25, y: 0.94, valign: 'bottom', style: LBL },
        { text: 'nine-slice', x: 0.75, y: 0.94, valign: 'bottom', style: LBL },
      ]);
    } else if (ex === 'cooldowns') {
      labels(ctx, 'cd', [{ text: 'ultimate meter', x: 0.5, y: 0.95, valign: 'bottom', style: LBL }]);
    } else labels(ctx, ex, []);
    const lay = step(params, ctx);
    const t = ctx.time;
    const out = {
      sim0: [sim.hp, sim.chip, sim.target, t - sim.lastHit],
      sim1: [sim.mana, sim.boss, sim.coins, sim.pop],
      btn: [sim.btn[0], sim.btn[1], sim.btn[2], 0],
      btnLay: [lay.y, lay.hw, lay.hh, lay.spacing],
    };
    if (atlasUV) {
      const run = atlasUV.anim('hero_run');
      out.uvPanel = atlasUV.uv('ui_panel');
      out.uvHero = atlasUV.uv(run[Math.floor(t * 8) % run.length]);
      out.uvSlime = atlasUV.uv(`slime_${Math.floor(t * 3) % 3}`);
      out.uvTile = atlasUV.uv('tile_stone');
      out.uvCoin = atlasUV.uv(`coin_${Math.floor(t * 10) % 8}`);
    }
    return out;
  },
  code: /* wgsl */ `
fn cov(d: f32, pw: f32) -> f32 { return clamp(0.5 - d / pw, 0.0, 1.0); }
fn bgPanel(px: vec2f) -> vec3f {
  let uv = px / u.resolution;
  var c = mix(vec3f(0.07, 0.075, 0.11), vec3f(0.035, 0.04, 0.06), uv.y);
  let v = length(uv - vec2f(0.5));
  return c * (1.1 - 0.6 * v * v) + vec3f((hash21(floor(px)) - 0.5) / 255.0);
}

// atlas sprite: color (premultiplied by alpha) of frame rect r [u0,v0,u1,v1] drawn in a box
fn sprite(px: vec2f, center: vec2f, size: vec2f, r: vec4f) -> vec4f {
  let q = (px - center) / size + vec2f(0.5);
  if (q.x < 0.0 || q.y < 0.0 || q.x >= 1.0 || q.y >= 1.0) { return vec4f(0.0); }
  let ts = TEXSIZE(atlas);
  let tex = floor(mix(r.xy, r.zw, q) * ts);
  let c = LOAD(atlas, vec2i(tex));
  return vec4f(c.rgb * c.a, c.a);
}

// seven-segment digit; p in units where the digit is 1 tall and 0.55 wide
fn digit7(p: vec2f, dgt: i32) -> f32 {
  var masks = array<i32, 10>(63, 6, 91, 79, 102, 109, 125, 7, 127, 111);
  var pa = array<vec2f, 7>(vec2f(-0.22, -0.45), vec2f(0.25, -0.42), vec2f(0.25, 0.03), vec2f(-0.22, 0.45), vec2f(-0.25, 0.03), vec2f(-0.25, -0.42), vec2f(-0.22, 0.0));
  var pb = array<vec2f, 7>(vec2f(0.22, -0.45), vec2f(0.25, -0.03), vec2f(0.25, 0.42), vec2f(0.22, 0.45), vec2f(-0.25, 0.42), vec2f(-0.25, -0.03), vec2f(0.22, 0.0));
  let m = masks[clamp(dgt, 0, 9)];
  var d = 1e5;
  for (var i = 0; i < 7; i++) {
    if (((m >> u32(i)) & 1) == 1) { d = min(d, sdSegment(p, pa[i], pb[i]) - 0.065); }
  }
  return d;
}
// up to 3 digits right-aligned around center (one digit7 call site)
fn number(p: vec2f, value: i32, ndig: i32) -> f32 {
  var d = 1e5;
  let w = 0.72;
  for (var k = 0; k < 3; k++) {
    if (k >= ndig) { break; }
    var v = value;
    for (var j = 0; j < 3; j++) { if (j < ndig - 1 - k) { v = v / 10; } }
    let x = (f32(k) - f32(ndig - 1) * 0.5) * w;
    d = min(d, digit7(p - vec2f(x, 0.0), v % 10));
  }
  return d;
}
fn ndigits(v: i32) -> i32 { if (v >= 100) { return 3; } if (v >= 10) { return 2; } return 1; }

// ------------------------------------------------------------------ bars
fn barsView(px: vec2f) -> vec3f {
  let res = u.resolution;
  var c = bgPanel(px);
  let t = u.time;
  let hp = u.sim0.x;
  let chip = u.sim0.y;
  let heal = u.sim0.z;
  let sinceHit = u.sim0.w;
  let i = i32(clamp(floor((px.y / res.y - 0.15 + 0.085) / 0.17), 0.0, 4.0));
  let cy = res.y * (0.15 + f32(i) * 0.17);
  let bh = res.y * 0.033;
  if (abs(px.y - cy) > bh * 2.4) { return c; }
  var shake = 0.0;
  if (i < 3) { shake = sin(t * 70.0) * 5.0 * exp(-sinceHit * 7.0) * res.y / 800.0; }
  let x0 = res.x * 0.26 + shake;
  let x1 = res.x * 0.92 + shake;
  let bw = x1 - x0;
  let p = px - vec2f(x0 + bw * 0.5, cy);
  var radius = bh;
  if (i == 0) { radius = 3.0; }
  if (i == 4) { radius = 4.0; }
  let box = sdRoundBox(p, vec2f(bw * 0.5, bh), radius);
  let inside = cov(box, 1.0);
  // frame + drop shadow
  c = mix(c, vec3f(0.0), cov(box - 4.0, 4.0) * 0.6);
  c = mix(c, vec3f(0.02, 0.02, 0.03), cov(box - 2.5, 1.0));
  c = mix(c, vec3f(0.14, 0.12, 0.16), inside);
  let fx = px.x - x0;   // distance along the bar in px
  let ny = (px.y - cy) / bh;  // -1 (top) .. 1 (bottom)
  var value = hp;
  if (i == 3) { value = u.sim1.x; }
  var fillC = vec3f(0.3, 0.85, 0.35);
  if (i == 0) {
    fillC = vec3f(0.3, 0.78, 0.36);
  } else if (i == 1) {
    // damage chip (bright, blinking slightly) and heal ghost
    let chipA = cov(fx - bw * chip, 1.0) * (1.0 - cov(fx - bw * hp, 1.0));
    c = mix(c, mix(vec3f(1.0, 0.95, 0.8), vec3f(1.0, 0.55, 0.3), 0.4 + 0.4 * sin(t * 25.0)), chipA * inside);
    let ghost = cov(fx - bw * heal, 1.0) * (1.0 - cov(fx - bw * hp, 1.0));
    c = mix(c, vec3f(0.5, 1.0, 0.6), ghost * inside * (0.35 + 0.2 * sin(t * 8.0)));
    fillC = mix(vec3f(0.62, 0.05, 0.1), vec3f(0.95, 0.25, 0.25), fx / bw) * mix(1.25, 0.8, ny * 0.5 + 0.5);
  } else if (i == 2) {
    fillC = mix(vec3f(0.95, 0.75, 0.2), vec3f(0.35, 0.85, 0.35), smoothstep(0.2, 0.6, hp)) * mix(1.15, 0.8, ny * 0.5 + 0.5);
  } else if (i == 3) {
    fillC = mix(vec3f(0.12, 0.3, 0.85), vec3f(0.35, 0.7, 1.0), smoothstep(0.8, -0.8, ny));
  } else {
    // boss: three stacked layers of HP in different colors
    let layer = floor(u.sim1.y);
    value = fract(u.sim1.y);
    var lc = array<vec3f, 4>(vec3f(0.85, 0.15, 0.2), vec3f(0.95, 0.5, 0.12), vec3f(0.6, 0.25, 0.9), vec3f(0.6, 0.25, 0.9));
    let li = i32(clamp(layer, 0.0, 3.0));
    fillC = lc[li] * mix(1.2, 0.8, ny * 0.5 + 0.5);
    if (li > 0) { c = mix(c, lc[li - 1] * 0.55, inside); }   // the next layer waits underneath
    // layer pips at the right end
    for (var k = 0; k < 3; k++) {
      let pc = vec2f(x1 + bh * (1.2 + f32(k) * 1.1), cy);
      let pd = sdRhombus(px - pc, vec2f(bh * 0.45, bh * 0.7));
      var pcol = vec3f(0.2);
      if (f32(k) <= layer) { pcol = lc[k]; }
      c = mix(c, vec3f(0.0), cov(pd - 1.5, 1.0));
      c = mix(c, pcol, cov(pd, 1.0));
    }
    // ornamental end caps
    let capL = sdRhombus(px - vec2f(x0, cy), vec2f(bh * 0.8, bh * 1.4));
    c = mix(c, vec3f(0.85, 0.7, 0.4), cov(abs(capL) - 1.5, 1.0));
  }
  // the fill itself (mana gets a wobbly liquid edge)
  var edge = bw * value;
  if (i == 3) { edge += sin(px.y * 0.25 + t * 7.0) * 2.0 * step(0.01, value); }
  var fillA = cov(fx - edge, 1.0) * cov(box + 2.0, 1.0);
  if (i == 2) {
    // segment gaps
    let segW = bw / u.segments;
    let sx = fmod(fx, segW);
    let gap = min(sx, segW - sx);
    fillA *= cov(1.5 - gap, 1.0);
    // the partially filled segment is dimmer
    let segId = floor(fx / segW);
    let full = floor(value * u.segments);
    if (segId >= full) { fillC *= 0.6; }
  }
  // low HP blink
  if (i < 3) { fillC *= 1.0 + 0.4 * (1.0 - smoothstep(0.15, 0.3, hp)) * (0.5 + 0.5 * sin(t * 12.0)); }
  c = mix(c, fillC, fillA);
  // glassy highlight on the top side
  let gloss = smoothstep(-0.2, -0.95, ny) * 0.35;
  if (i >= 1) { c += vec3f(gloss) * inside; }
  // shine sweep: a diagonal band crossing the filled part every few seconds
  if (u.shine > 0.5 && (i == 3 || i == 1)) {
    let sp = fract(t * 0.35 + f32(i) * 0.3) * (bw + bh * 8.0) - bh * 4.0;
    let band = 1.0 - smoothstep(0.0, bh * 1.2, abs(fx - sp + ny * bh * 0.8));
    c += vec3f(0.6, 0.8, 1.0) * band * fillA * 0.55;
  }
  // mana bubbles
  if (i == 3) {
    let bq = vec2f(fx, px.y) / bh;
    let cell = floor(bq * vec2f(1.0, 1.0) + vec2f(0.0, t * 1.5));
    let h = hash21(cell);
    let bp = fract(bq + vec2f(0.0, t * 1.5)) - vec2f(0.5) - (vec2f(hash21(cell + vec2f(3.0)), h) - vec2f(0.5)) * 0.4;
    c += vec3f(0.6, 0.8, 1.0) * cov((length(bp) - 0.12) * bh, 1.0) * step(0.75, h) * fillA * 0.5;
  }
  // inner border
  c = mix(c, vec3f(1.0), cov(abs(box + 1.0) - 0.6, 1.0) * 0.12);
  return c;
}

// ------------------------------------------------------------------ cooldowns
fn icon(id: i32, q: vec2f) -> f32 {
  if (id == 0) {
    let r = rot2(-0.785398) * q;
    let blade = min(sdBox(r - vec2f(0.0, -0.2), vec2f(0.12, 0.48)), sdTriangle(r, vec2f(-0.12, -0.67), vec2f(0.12, -0.67), vec2f(0.0, -0.95)));
    let guard = sdRoundBox(r - vec2f(0.0, 0.33), vec2f(0.38, 0.08), 0.06);
    return min(min(blade, guard), min(sdBox(r - vec2f(0.0, 0.56), vec2f(0.065, 0.18)), sdCircle(r - vec2f(0.0, 0.8), 0.11)));
  }
  if (id == 1) {
    let top = sdRoundBox(q - vec2f(0.0, -0.28), vec2f(0.62, 0.45), 0.12);
    return opSmoothUnion(top, sdTriangle(q, vec2f(-0.62, -0.12), vec2f(0.62, -0.12), vec2f(0.0, 0.9)), 0.1);
  }
  if (id == 2) { return sdHeart(q / 0.8 + vec2f(0.0, 0.04)) * 0.8; }
  if (id == 3) {
    // dash: three speed chevrons
    var d = 1e5;
    for (var k = 0; k < 3; k++) {
      let o = vec2f(f32(k) * 0.42 - 0.42, 0.0);
      let a = sdSegment(q - o, vec2f(-0.18, -0.5), vec2f(0.18, 0.0)) - 0.1;
      let b = sdSegment(q - o, vec2f(-0.18, 0.5), vec2f(0.18, 0.0)) - 0.1;
      d = min(d, min(a, b));
    }
    return d;
  }
  return sdStar5(q, 0.85, 0.45);
}

fn cooldownView(px: vec2f) -> vec3f {
  let res = u.resolution;
  var c = bgPanel(px);
  let t = u.time * u.cdSpeed;
  let S = min(res.x / 7.2, res.y * 0.24);
  let spacing = S * 1.3;
  let rowY = res.y * 0.33;
  let fi = clamp(floor((px.x - res.x * 0.5) / spacing + 2.5), 0.0, 4.0);
  let i = i32(fi);
  let center = vec2f(res.x * 0.5 + (fi - 2.0) * spacing, rowY);
  let q = (px - center) / (S * 0.5);
  let pw = 2.0 / S;
  var periods = array<f32, 5>(3.0, 6.0, 9.0, 2.5, 14.0);
  var cols = array<vec3f, 5>(vec3f(1.0, 0.5, 0.3), vec3f(0.4, 0.7, 1.0), vec3f(1.0, 0.4, 0.55), vec3f(0.45, 1.0, 0.6), vec3f(1.0, 0.85, 0.3));
  let period = periods[i];
  let hold = 1.6;
  let ph = fmod(t + f32(i) * 1.7, period + hold);
  var prog = ph / period;            // 0..1 while cooling down
  let ready = step(1.0, prog);
  prog = min(prog, 1.0);
  let since = (ph - period) * ready;  // seconds since ready
  let accent = cols[i];

  if (abs(q.x) < 1.6 && abs(q.y) < 1.9) {
    let sd = sdRoundBox(q, vec2f(0.92), 0.22);
    c = mix(c, vec3f(0.0), cov(sd - 0.1, pw * 3.0) * 0.6);
    var slot = mix(vec3f(0.15, 0.16, 0.23), vec3f(0.07, 0.07, 0.11), q.y * 0.5 + 0.5);
    c = mix(c, slot, cov(sd, pw));
    let id = icon(i, q / 0.62) * 0.62;
    var ic = mix(accent, vec3f(1.0), smoothstep(0.2, -0.6, q.y) * 0.4);
    c += accent * exp(-max(id, 0.0) / 0.12) * 0.25 * cov(sd, pw);
    c = mix(c, vec3f(0.0), cov(id - 0.07, pw) * 0.7);
    c = mix(c, ic, cov(id, pw));
    // ----- cooldown overlay
    let ang = fract(atan2(q.x, -q.y) / TAU + 1.0);       // 0 at 12 o'clock, clockwise
    let cooling = 1.0 - ready;
    let style = i32(u.cdStyle);
    if (style == 0) {
      // clock wipe: darken the part that is still on cooldown
      let edgeD = (ang - prog) * TAU * max(length(q), 0.15);
      let dark = cov(-edgeD, pw) * cov(sd, pw) * cooling;
      c = mix(c, c * 0.22, dark);
      // bright sweep line
      let lineD = abs(edgeD);
      c += accent * cov(lineD - pw, pw) * cooling * cov(sd, pw) * step(0.02, length(q)) * 0.8;
    } else if (style == 1) {
      // fill from the bottom: unfilled part is greyed out
      let level = mix(0.92, -0.92, prog);
      let grey = cov(level - q.y, pw) * cooling * cov(sd, pw);
      let g = vec3f(dot(c, vec3f(0.3, 0.59, 0.11))) * 0.45;
      c = mix(c, g, grey);
      c = mix(c, accent, cov(abs(q.y - level) - pw, pw) * cooling * cov(sd, pw));
    } else {
      // progress ring around the slot
      let rd = abs(length(q) - 1.18) - 0.06;
      c = mix(c, vec3f(0.18), cov(rd, pw) * cooling);
      c = mix(c, accent, cov(rd, pw) * cov((ang - prog) * TAU * 1.18, pw) * cooling);
      c = mix(c, c * 0.5, cooling * cov(sd, pw));
    }
    // countdown number
    if (u.numbers > 0.5 && cooling > 0.5) {
      let remain = i32(ceil(period * (1.0 - prog)));
      let dq = q / 0.62;
      let nd = number(dq, remain, ndigits(remain)) * 0.62;
      c = mix(c, vec3f(0.0), cov(nd - 0.06, pw) * 0.8);
      c = mix(c, vec3f(1.0), cov(nd, pw));
    }
    // ready: flash + expanding ring
    let fl = exp(-since * 4.0) * ready;
    c = mix(c, vec3f(1.0), fl * 0.5 * cov(sd, pw));
    let ringR = 0.95 + since * 1.4;
    c += accent * cov(abs(length(q) - ringR) - 0.04, pw) * exp(-since * 2.5) * ready;
    // border glow when ready
    c = mix(c, mix(accent * 0.55, accent, ready * (0.6 + 0.4 * sin(u.time * 5.0))), cov(abs(sd) - 0.035, pw));
    // charges (dash): three pips
    if (i == 3) {
      let cyc = fmod(t, 7.5) / 2.5;
      for (var k = 0; k < 3; k++) {
        let pq = q - vec2f(f32(k - 1) * 0.42, 1.32);
        let pd = length(pq) - 0.14;
        let fillk = clamp(cyc - f32(k), 0.0, 1.0);
        c = mix(c, vec3f(0.05), cov(pd - 0.04, pw));
        let pa = fract(atan2(pq.x, -pq.y) / TAU + 1.0);
        let filled = max(step(1.0, fillk), step(pa, fillk));
        c = mix(c, mix(vec3f(0.22), accent, filled), cov(pd, pw));
      }
    }
  }
  // ---- ultimate orb
  let oc = vec2f(res.x * 0.5, res.y * 0.72);
  let R = res.y * 0.14;
  let oq = (px - oc) / R;
  let opw = 1.0 / R;
  let ol = length(oq);
  if (ol < 1.6) {
    let charge = clamp(fmod(u.time * u.cdSpeed, 16.0) / 12.0, 0.0, 1.0);
    let full = step(1.0, charge);
    let a = fract(atan2(oq.x, -oq.y) / TAU + 1.0);
    // liquid inside the orb
    let level = mix(0.95, -0.95, charge) + 0.04 * sin(oq.x * 9.0 + u.time * 4.0);
    var orb = mix(vec3f(0.06, 0.05, 0.1), vec3f(0.08, 0.06, 0.14), oq.y);
    let liquid = cov(level - oq.y, opw) * cov(ol - 0.8, opw);
    orb = mix(orb, mix(vec3f(0.55, 0.2, 0.95), vec3f(0.95, 0.5, 1.0), smoothstep(0.9, -0.9, oq.y)), liquid);
    c = mix(c, orb, cov(ol - 0.8, opw));
    c += vec3f(0.8, 0.4, 1.0) * exp(-max(ol - 0.8, 0.0) * 5.0) * (0.15 + 0.45 * full * (0.6 + 0.4 * sin(u.time * 6.0)));
    // segmented arc meter around it
    let seg = abs(fract(a * 12.0) - 0.5) * 2.0;
    let arc = max(abs(ol - 0.98) - 0.06, -(seg - 0.12) * 0.2);
    c = mix(c, vec3f(0.2, 0.18, 0.28), cov(arc, opw));
    c = mix(c, vec3f(1.0, 0.75, 1.0), cov(arc, opw) * step(a, charge));
    let st = sdStar5(oq * vec2f(1.0, 1.0), 0.45 * (1.0 + 0.12 * full * sin(u.time * 8.0)), 0.45);
    c = mix(c, vec3f(1.0, 0.95, 1.0), cov(st, opw) * (0.25 + 0.75 * full));
    c = mix(c, vec3f(0.85, 0.75, 1.0), cov(abs(ol - 0.8) - 0.012, opw) * 0.6);
  }
  return c;
}

// ------------------------------------------------------------------ nine-slice
fn panelTexel(lp: vec2f, size: vec2f, nine: bool) -> vec2f {
  // returns a texel coordinate (0..24) inside the 24x24 panel for local pixel lp in a panel of 'size'
  if (!nine) { return lp / size * 24.0; }
  let k = u.pixelScale;
  let b = 8.0 * k;
  var tx = vec2f(0.0);
  for (var a = 0; a < 2; a++) {
    var x = lp.x;
    var w = size.x;
    if (a == 1) { x = lp.y; w = size.y; }
    var r: f32;
    if (x < b) { r = x / k; }
    else if (x > w - b) { r = 24.0 - (w - x) / k; }
    else if (u.centerMode > 0.5) { r = 8.0 + fmod((x - b) / k, 8.0); }
    else { r = 8.0 + (x - b) / max(w - 2.0 * b, 1.0) * 8.0; }
    if (a == 0) { tx.x = r; } else { tx.y = r; }
  }
  return tx;
}

fn nineView(px: vec2f) -> vec3f {
  let res = u.resolution;
  var c = bgPanel(px);
  // checker so the panel edges read clearly
  let ch = fmod(floor(px.x / 16.0) + floor(px.y / 16.0), 2.0);
  c += vec3f(0.012) * ch;
  let side = floor(px.x / (res.x * 0.5));
  let hc = vec2f(res.x * (0.25 + 0.5 * side), res.y * 0.46);
  var wf = u.panelW;
  var hf = u.panelH;
  if (u.animPanel > 0.5) {
    wf = 0.35 + 0.6 * (0.5 + 0.5 * sin(u.time * 0.7));
    hf = 0.3 + 0.55 * (0.5 + 0.5 * sin(u.time * 0.53 + 1.0));
  }
  let k = u.pixelScale;
  let maxSize = vec2f(res.x * 0.45, res.y * 0.78);
  let minSize = vec2f(16.0 * k + 2.0);
  let size = floor(max(maxSize * vec2f(wf, hf), minSize) / k) * k;
  let origin = floor(hc - size * 0.5);
  let lp = px - origin;
  if (lp.x >= 0.0 && lp.y >= 0.0 && lp.x < size.x && lp.y < size.y) {
    let tx = clamp(panelTexel(lp, size, side > 0.5), vec2f(0.0), vec2f(23.999));
    let ts = TEXSIZE(atlas);
    let o = floor(u.uvPanel.xy * ts);
    let tc = LOAD(atlas, vec2i(o + floor(tx)));
    c = mix(c, tc.rgb, tc.a);
    // dialog content: a portrait and "text" lines (same on both sides, scaled by the panel)
    let inner = size - vec2f(24.0 * k);
    if (inner.x > 40.0 && inner.y > 30.0) {
      let pc = origin + vec2f(12.0 * k) + vec2f(min(inner.y, inner.x * 0.4) * 0.5);
      let psz = min(inner.y, inner.x * 0.4) * 0.9;
      let spr = sprite(px, pc, vec2f(psz), u.uvHero);
      c = c * (1.0 - spr.a) + spr.rgb;
      let tx0 = pc.x + psz * 0.62;
      let tx1 = origin.x + size.x - 12.0 * k;
      for (var l = 0; l < 3; l++) {
        let ly = origin.y + 12.0 * k + (f32(l) + 0.6) * min(inner.y / 3.4, 14.0 * k);
        let len = (tx1 - tx0) * (0.9 - f32(l) * 0.22);
        let ld = sdRoundBox(px - vec2f(tx0 + len * 0.5, ly), vec2f(max(len * 0.5, 0.0), k * 1.5), k * 1.5);
        c = mix(c, vec3f(0.75, 0.82, 1.0), cov(ld, 1.0) * step(4.0, len) * 0.8);
      }
    }
    // slice guides
    if (u.guides > 0.5) {
      var gx = vec2f(size.x / 3.0, size.x * 2.0 / 3.0);
      var gy = vec2f(size.y / 3.0, size.y * 2.0 / 3.0);
      if (side > 0.5) { gx = vec2f(8.0 * k, size.x - 8.0 * k); gy = vec2f(8.0 * k, size.y - 8.0 * k); }
      else { gx = vec2f(size.x / 3.0, size.x * 2.0 / 3.0); gy = vec2f(size.y / 3.0, size.y * 2.0 / 3.0); }
      let gd = min(min(abs(lp.x - gx.x), abs(lp.x - gx.y)), min(abs(lp.y - gy.x), abs(lp.y - gy.y)));
      let dash = step(0.5, fract((lp.x + lp.y) / 8.0));
      c = mix(c, vec3f(0.3, 1.0, 0.85), cov(gd - 0.5, 1.0) * dash * 0.8);
    }
  }
  // the source texture, shown small in the corner of each side
  let srcSize = 24.0 * max(floor(res.y / 180.0), 2.0);
  let so = vec2f(res.x * 0.5 * side + 16.0, res.y - srcSize - 16.0);
  let sl = px - so;
  if (sl.x >= 0.0 && sl.y >= 0.0 && sl.x < srcSize && sl.y < srcSize) {
    let ts = TEXSIZE(atlas);
    let tc = LOAD(atlas, vec2i(floor(u.uvPanel.xy * ts) + floor(sl / srcSize * 24.0)));
    c = mix(c, tc.rgb, tc.a);
  }
  // divider
  c = mix(c, vec3f(0.25, 0.28, 0.38), cov(abs(px.x - res.x * 0.5) - 1.0, 1.0));
  return c;
}

// ------------------------------------------------------------------ juice
fn juiceView(px: vec2f) -> vec3f {
  let res = u.resolution;
  let t = u.time;
  let hp = u.sim0.x;
  let sinceHit = u.sim0.w;
  let k = max(floor(res.y / 160.0), 2.0);
  // tiled dungeon floor from the atlas (nearest, k screen px per texel)
  let ts = TEXSIZE(atlas);
  let tileO = floor(u.uvTile.xy * ts);
  let tp = fmod2(floor(px / k), vec2f(16.0));
  var c = LOAD(atlas, vec2i(tileO + tp)).rgb * 0.6;
  let vv = px / res - vec2f(0.5);
  c *= 1.0 - 0.9 * dot(vv, vv);
  // sprites: hero (flashes white when hit) and two slimes
  let heroC = vec2f(res.x * 0.42, res.y * 0.5);
  var hs = sprite(px, heroC, vec2f(16.0 * k * 2.0), u.uvHero);
  let flash = exp(-sinceHit * 10.0);
  hs = vec4f(mix(hs.rgb, vec3f(hs.a), flash), hs.a);
  c = c * (1.0 - hs.a) + hs.rgb;
  for (var s = 0; s < 2; s++) {
    let sc = vec2f(res.x * (0.62 + f32(s) * 0.14), res.y * (0.47 + f32(s) * 0.1) - abs(sin(t * 3.0 + f32(s) * 1.5)) * k * 4.0);
    let ss = sprite(px, sc, vec2f(16.0 * k * 1.6), u.uvSlime);
    c = c * (1.0 - ss.a) + ss.rgb;
  }
  // low health: desaturate the world + heartbeat vignette
  let low = clamp(1.0 - hp / max(u.lowHp, 0.001), 0.0, 1.0);
  if (u.desat > 0.5) { c = mix(c, vec3f(dot(c, vec3f(0.3, 0.59, 0.11))), low * 0.85); }
  let beatT = fract(t * mix(1.0, 2.0, low));
  let beat = exp(-beatT * 9.0) + 0.6 * exp(-abs(beatT - 0.22) * 14.0);
  let uv = px / res;
  let vg = smoothstep(0.25, 0.85, length((uv - vec2f(0.5)) * vec2f(res.x / res.y, 1.0) * 0.9));
  c = mix(c, vec3f(0.55, 0.0, 0.05), vg * u.vignette * low * (0.55 + 0.45 * beat));
  // damage flash at the screen edges
  c = mix(c, vec3f(1.0, 0.1, 0.1), vg * exp(-sinceHit * 6.0) * 0.6);

  // ---- HP bar (top-left), shakes when hit
  let shake = vec2f(sin(t * 80.0), cos(t * 67.0)) * 6.0 * exp(-sinceHit * 7.0) * res.y / 800.0;
  let bh = res.y * 0.03;
  let bx0 = res.x * 0.05 + shake.x;
  let bw = res.x * 0.3;
  let by = res.y * 0.12 + shake.y;
  let bp = px - vec2f(bx0 + bw * 0.5, by);
  let box = sdRoundBox(bp, vec2f(bw * 0.5, bh), bh);
  c = mix(c, vec3f(0.0), cov(box - 3.0, 1.0) * 0.85);
  c = mix(c, vec3f(0.15, 0.08, 0.1), cov(box, 1.0));
  let fx = px.x - bx0;
  let chipA = cov(fx - bw * u.sim0.y, 1.0) * (1.0 - cov(fx - bw * hp, 1.0)) * cov(box + 2.0, 1.0);
  c = mix(c, vec3f(1.0, 0.95, 0.85), chipA);
  var fillC = mix(vec3f(0.95, 0.2, 0.25), vec3f(0.3, 0.9, 0.4), smoothstep(0.2, 0.7, hp));
  fillC *= 1.0 + 0.5 * low * beat;
  let ny = (px.y - by) / bh;
  c = mix(c, fillC * mix(1.2, 0.8, ny * 0.5 + 0.5), cov(fx - bw * hp, 1.0) * cov(box + 2.0, 1.0));
  c += vec3f(smoothstep(-0.2, -0.95, ny) * 0.3) * cov(box, 1.0);
  // heart icon that beats
  let hq = (px - vec2f(bx0 - bh * 0.2, by)) / (bh * 1.5 * (1.0 + 0.15 * beat * low));
  let hd = sdHeart(hq / 0.8 + vec2f(0.0, 0.04)) * 0.8;
  c = mix(c, vec3f(0.0), cov(hd - 0.12, 2.0 / (bh * 1.5)));
  c = mix(c, vec3f(1.0, 0.3, 0.38), cov(hd, 2.0 / (bh * 1.5)));

  // ---- coin counter (top-right) with a springy pop
  let pop = 1.0 + u.sim1.w * 0.06;
  let cc = vec2f(res.x * 0.86, res.y * 0.14);
  let csz = bh * 2.2 * pop;
  let cs = sprite(px, cc - vec2f(csz * 1.4, 0.0), vec2f(csz * 1.3), u.uvCoin);
  c = c * (1.0 - cs.a) + cs.rgb;
  let coins = i32(u.sim1.z);
  let nq = (px - cc - vec2f(csz * 0.6, 0.0)) / csz;
  let nd = number(nq, coins, ndigits(coins));
  c = mix(c, vec3f(0.0), cov(nd - 0.1, 2.0 / csz));
  c = mix(c, vec3f(1.0, 0.87, 0.35), cov(nd, 2.0 / csz));

  // ---- springy buttons
  let lay = u.btnLay;
  let fi = clamp(floor((px.x - res.x * 0.5) / lay.w + 1.5), 0.0, 2.0);
  let bc = vec2f(res.x * 0.5 + (fi - 1.0) * lay.w, lay.x);
  var scl = u.btn.x;
  if (fi > 0.5) { scl = u.btn.y; }
  if (fi > 1.5) { scl = u.btn.z; }
  let q = (px - bc) / scl;
  let bd = sdRoundBox(q, vec2f(lay.y, lay.z), lay.z * 0.45);
  let hover = smoothstep(1.0, 1.1, scl);
  c = mix(c, vec3f(0.0), cov(bd - 5.0, 6.0) * 0.5);
  var bcol = mix(vec3f(0.98, 0.62, 0.2), vec3f(0.85, 0.32, 0.1), q.y / lay.z * 0.5 + 0.5);
  if (fi > 0.5) { bcol = mix(vec3f(0.35, 0.65, 1.0), vec3f(0.15, 0.3, 0.8), q.y / lay.z * 0.5 + 0.5); }
  if (fi > 1.5) { bcol = mix(vec3f(0.45, 0.9, 0.5), vec3f(0.15, 0.55, 0.3), q.y / lay.z * 0.5 + 0.5); }
  bcol *= 1.0 + 0.2 * hover;
  c = mix(c, vec3f(0.05, 0.04, 0.07), cov(bd - 3.0, 1.0));
  c = mix(c, bcol, cov(bd, 1.0));
  c += vec3f(0.35) * smoothstep(-0.2, -0.9, q.y / lay.z) * cov(bd + 4.0, 1.0);
  // icon: play / gear / home
  let iq = q / (lay.z * 0.6);
  var id: f32;
  if (fi < 0.5) { id = sdTriangle(iq, vec2f(-0.45, -0.6), vec2f(-0.45, 0.6), vec2f(0.6, 0.0)); }
  else if (fi < 1.5) {
    let r = length(iq);
    let a = atan2(iq.y, iq.x) + t * 0.5;
    let teeth = smoothstep(0.1, 0.0, abs(fract(a / TAU * 8.0) - 0.5) - 0.2) * 0.18;
    id = max(r - 0.55 - teeth, -(r - 0.22));
  } else {
    let body = sdBox(iq - vec2f(0.0, 0.2), vec2f(0.42, 0.36));
    let roof = sdTriangle(iq, vec2f(-0.7, -0.05), vec2f(0.7, -0.05), vec2f(0.0, -0.7));
    id = max(min(body, roof), -sdBox(iq - vec2f(0.0, 0.38), vec2f(0.11, 0.2)));
  }
  let ipw = 1.0 / (lay.z * 0.6);
  c = mix(c, vec3f(0.0), cov(id - 0.1, ipw) * cov(bd, 1.0) * 0.6);
  c = mix(c, vec3f(1.0), cov(id, ipw) * cov(bd, 1.0));
  return c;
}

fn shade(uv: vec2f, px0: vec2f) -> vec4f {
  let px = uv * u.resolution;   // canvas pixels, even when rendering at reduced resolution
  let ex = i32(u.example);
  var c: vec3f;
  if (ex == 0) { c = barsView(px); }
  else if (ex == 1) { c = cooldownView(px); }
  else if (ex == 2) { c = nineView(px); }
  else { c = juiceView(px); }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`,
  about: {
    summary: 'Game UI is mostly rectangles, circles and a few textures — but the details (trailing damage, sweeps, springs, nine-slicing) are what make it feel good. All of it maps naturally to shaders.',
    what: `<p><b>Bars</b>: five health/mana/boss bar styles reacting to a simulated fight. <b>Cooldowns</b>: clock-wipe, fill and ring styles with digits, charges and an ultimate orb.
      <b>Nine-slice</b>: one 24×24 pixel-art panel scaled two ways. <b>Juice</b>: springy buttons, a popping coin counter, hit shake and a low-health heartbeat.</p>`,
    how: `<ol>
      <li><b>Bars</b> are a rounded-box SDF (the frame) plus a comparison <code>x &lt; width × value</code> (the fill). Gradients, gloss and shine are functions of the bar’s local x/y.</li>
      <li>The <b>damage chip</b> needs memory: JavaScript keeps a second value that waits ~0.5 s after a hit, then drains toward the real HP. Both values are uniforms.</li>
      <li><b>Radial cooldown</b>: <code>angle = atan2(x, −y)</code> gives 0 at 12 o’clock going clockwise; darken pixels whose angle (as a 0..1 fraction) is beyond the progress. Multiply the angle difference by the radius to get a pixel distance for a crisp anti-aliased edge.</li>
      <li><b>Nine-slice</b>: map each pixel to a texel. Inside the 8-px border → copy the border texels 1:1 (scaled by the pixel scale); in the middle → stretch or repeat the middle 8 texels.</li>
      <li><b>Juice</b>: springs (<code>a = k·(target − x) − c·v</code>) for scale, exponential decays (<code>exp(−t·k)</code>) for flashes and shakes, a heartbeat curve for the vignette.</li>
    </ol>`,
    uses: [
      { title: 'Action & RPG HUDs', text: 'Damage chips (fighting games, Souls-likes), segmented stamina, layered boss bars (MMOs), radial cooldowns (MOBAs, WoW).' },
      { title: 'Menus & dialogs', text: 'Nine-slice panels and buttons from a tiny texture: dialogue boxes, inventory slots, tooltips — any size, any resolution.' },
      { title: 'Game feel', text: 'Springy buttons and popping counters (mobile & casual games), low-HP vignettes and desaturation (shooters).' },
    ],
    try: [
      'On <b>Bars</b>, turn off the simulated fight and drag HP down quickly — watch the white chip wait, then drain.',
      'Set <i>Segments</i> to 4: the segmented bar becomes a “hearts”-style meter.',
      'On <b>Cooldowns</b>, switch styles and raise <i>Time speed</i>.',
      'On <b>Nine-slice</b>, stop the animation, make the panel very wide and short, and compare the corners.',
      'On <b>Juice</b>, set <i>Bounciness</i> to 1 and hover the buttons; then drag the HP below the threshold.',
    ],
    ask: [
      'a health bar with a delayed damage chip',
      'radial cooldown overlay with a countdown',
      'nine-slice panels for all dialog boxes',
      'springy hover/press animation for buttons',
      'a low-health heartbeat vignette with desaturation',
      'a multi-layer boss health bar',
    ],
    perf: `<p>Everything here is one full-screen fragment shader; a real game draws each widget as a small quad (only the pixels it covers).
      UI is rarely a performance problem — the usual cost is CPU-side layout and too many draw calls, solved by batching widgets into one instanced draw.</p>`,
    api: `<p>Identical on WebGPU and WebGL2 (one WGSL source, translated to GLSL). The atlas is read with exact texel loads (<code>LOAD</code>) so the pixel art stays crisp.</p>`,
    code: [
      {
        title: 'Radial cooldown (clock wipe)',
        lang: 'wgsl',
        src: `let ang = fract(atan2(q.x, -q.y) / TAU + 1.0);   // 0 at 12 o'clock, clockwise
let edgeD = (ang - prog) * TAU * length(q);       // angular distance -> pixel distance
let dark = cov(-edgeD, pw) * insideSlot * cooling;
c = mix(c, c * 0.22, dark);`,
      },
      {
        title: 'Nine-slice texel mapping (one axis)',
        lang: 'wgsl',
        src: `let b = 8.0 * k;                                   // border in screen px
if (x < b) { r = x / k; }                           // left border: 1:1
else if (x > w - b) { r = 24.0 - (w - x) / k; }     // right border: 1:1
else { r = 8.0 + (x - b) / (w - 2.0 * b) * 8.0; }   // middle: stretched`,
      },
      {
        title: 'Damage chip & springs (JavaScript, per frame)',
        lang: 'js',
        src: `if (target < hp) hp = target;                       // damage is instant
if (t > chipHold) chip = Math.max(hp, chip - dt * 0.6); // chip waits, then drains
v += (k * (goal - scale) - damping * v) * dt;          // spring
scale += v * dt;`,
      },
    ],
  },
});

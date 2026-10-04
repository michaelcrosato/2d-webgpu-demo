import { shaderScene } from '../../core/shaderscene.js';
import { readout, stepper } from './_shared-a.js';

// Cellular automata: Conway's Life and friends, plus roguelike cave generation.
// One feedback pass ("cells") holds the grid: r = state, g = age, b = death trail, a = cave step.

// Life-like rules as bit masks: bit n set = "n neighbours" triggers birth / survival.
const bits = (...ns) => ns.reduce((m, n) => m | (1 << n), 0);
const RULES = {
  conway: { label: 'Conway’s Life (B3/S23)', b: bits(3), s: bits(2, 3), kind: 0, density: 0.3 },
  highlife: { label: 'HighLife (B36/S23) — replicators', b: bits(3, 6), s: bits(2, 3), kind: 0, density: 0.3 },
  daynight: { label: 'Day & Night (B3678/S34678)', b: bits(3, 6, 7, 8), s: bits(3, 4, 6, 7, 8), kind: 0, density: 0.5 },
  seeds: { label: 'Seeds (B2/S) — explosive', b: bits(2), s: 0, kind: 0, density: 0.04 },
  maze: { label: 'Maze (B3/S12345)', b: bits(3), s: bits(1, 2, 3, 4, 5), kind: 0, density: 0.1 },
  brain: { label: 'Brian’s Brain (3 states)', b: 0, s: 0, kind: 1, density: 0.3 },
  cyclic: { label: 'Cyclic CA (spirals)', b: 0, s: 0, kind: 2, density: 1 },
};
const RULE_KEYS = Object.keys(RULES);

// Pattern atlas for the "start" options (RLE → pixels on a tiny canvas)
const RLE = {
  gun: '24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!',
  acorn: 'bo5b$3bo3b$2o2b3o!',
};
function patternCanvas() {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 16;
  const g = c.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, 64, 16);
  g.fillStyle = '#fff';
  const draw = (rle, ox, oy) => {
    let x = 0;
    let y = 0;
    let n = '';
    for (const ch of rle) {
      if (ch >= '0' && ch <= '9') n += ch;
      else {
        const k = n ? +n : 1;
        n = '';
        if (ch === 'b') x += k;
        else if (ch === 'o') {
          g.fillRect(ox + x, oy + y, k, 1);
          x += k;
        } else if (ch === '$') {
          y += k;
          x = 0;
        }
      }
    }
  };
  draw(RLE.gun, 0, 0); // 36×9 at (0,0)
  draw(RLE.acorn, 40, 0); // 7×3 at (40,0)
  return c;
}

let isTest = false;
let ticker = stepper();
let stepsNow = 1;
let tag = null;
let tagCtx = null;
let caveDone = 0;
let caveKey = '';

export default shaderScene({
  interaction: 'Draw living cells with the mouse (right-drag erases). On “Cave generation” drag to carve tunnels.',
  examples: [
    {
      id: 'life',
      label: 'Conway’s Life',
      kind: 'Classic',
      note: 'Every cell looks at its 8 neighbours. A dead cell with exactly <b>3</b> live neighbours is <b>born</b>; a live cell with <b>2 or 3</b> <b>survives</b>; everything else dies. Bright = newborn, blue = old, pink = recently died. Try the <i>Glider guns</i> start.',
      params: { rule: 'conway', start: 'soup', speed: 15, cell: 4, trails: 0.85 },
    },
    {
      id: 'rules',
      label: 'Rule explorer',
      kind: 'Abstract',
      note: 'Change the birth/survival numbers and you get different universes: HighLife has self-copying replicators, Day & Night treats live and dead symmetrically, Brian’s Brain adds a “refractory” state (always moving), cyclic automata grow spirals.',
      params: { rule: 'brain', speed: 20, cell: 3, trails: 0.6 },
    },
    {
      id: 'cave',
      label: 'Cave generation',
      kind: 'In a game',
      note: 'The classic roguelike trick: fill the map with ~45% random walls, then repeat “become a wall if 5 or more of the 3×3 block are walls”. Noise clumps into natural caves in 4–5 steps. Drag <i>Smoothing steps</i> to watch it converge; move the mouse to carry a lantern.',
      params: { speed: 4, cell: 9, fill: 46, caveSteps: 5, seed: 7 },
    },
  ],
  controls: [
    { type: 'heading', label: 'Rules' },
    {
      type: 'select',
      key: 'rule',
      label: 'Rule',
      value: 'conway',
      options: RULE_KEYS.map((k) => ({ value: k, label: RULES[k].label })),
      showFor: ['rules'],
    },
    {
      type: 'select',
      key: 'start',
      label: 'Start pattern',
      value: 'soup',
      options: [
        { value: 'soup', label: 'Random soup (30%)' },
        { value: 'guns', label: 'Glider guns' },
        { value: 'acorn', label: 'Acorn (a methuselah)' },
        { value: 'empty', label: 'Empty — draw your own' },
      ],
      showFor: ['life'],
    },
    { type: 'slider', key: 'speed', label: 'Generations / second', min: 1, max: 240, step: 1, value: 15, log: true, help: 'Above 60 several generations run per frame.' },
    { type: 'slider', key: 'cell', label: 'Cell size', min: 2, max: 16, step: 1, value: 4, unit: 'px' },
    { type: 'slider', key: 'trails', label: 'Death trails', min: 0, max: 0.97, step: 0.01, value: 0.85, help: 'How long recently-dead cells keep glowing.', showFor: ['life', 'rules'] },
    { type: 'slider', key: 'brush', label: 'Brush size', min: 1, max: 30, step: 1, value: 6, unit: 'px' },
    { type: 'heading', label: 'Cave generator', showFor: ['cave'] },
    { type: 'slider', key: 'fill', label: 'Initial wall fill', min: 30, max: 65, step: 1, value: 46, unit: '%', help: 'Below ~40% everything opens up; above ~52% it closes into rock.', showFor: ['cave'] },
    { type: 'slider', key: 'caveSteps', label: 'Smoothing steps', min: 0, max: 12, step: 1, value: 5, help: 'How many times the 4-5 rule is applied. 0 = raw noise.', showFor: ['cave'] },
    { type: 'slider', key: 'seed', label: 'Seed', min: 1, max: 99, step: 1, value: 7, help: 'Same seed + same settings = same map (deterministic).', showFor: ['cave'] },
    { type: 'button', key: 'reset', label: 'Restart', primary: true },
  ],
  uniforms: {
    advance: 'f32',
    rule: 'f32',
    kindR: 'f32',
    birthMask: 'f32',
    surviveMask: 'f32',
    density: 'f32',
    start: 'f32',
    trails: 'f32',
    brush: 'f32',
    erase: 'f32',
    cellPx: 'f32',
    fill: 'f32',
    caveSteps: 'f32',
    seed: 'f32',
  },
  include: ['math', 'hash', 'noise', 'sdf', 'color'],
  textures: { patterns: { source: async () => patternCanvas(), filter: 'nearest' } },
  resetOn: ['rule', 'start', 'fill', 'seed', 'caveSteps', 'cell'],
  onAction(key) {
    if (key === 'reset') caveDone = 0;
  },
  bind(params, ctx) {
    isTest = ctx.testMode;
    if (tagCtx !== ctx) {
      tagCtx = ctx;
      tag = readout(ctx);
      ticker = stepper();
    }
    const ex = ctx.example;
    const ruleKey = ex === 'life' ? 'conway' : ex === 'rules' ? params.rule : 'conway';
    const R = RULES[ruleKey] || RULES.conway;
    // whole generations to run this frame (fixed rate, independent of the display's fps)
    const rate = params.speed * (isTest ? 4 : 1);
    stepsNow = ctx.paused ? 0 : ticker.take(ctx.dt, rate, 8);
    // cave progress readout (the shader keeps the real counter per cell)
    const key = `${ex}|${params.fill}|${params.seed}|${params.caveSteps}|${params.cell}|${ctx.width}x${ctx.height}`;
    if (key !== caveKey) {
      caveKey = key;
      caveDone = 0;
    }
    if (ex === 'cave') {
      caveDone = Math.min(params.caveSteps, caveDone + stepsNow);
      tag.textContent = `smoothing step ${caveDone} / ${params.caveSteps} · ${Math.round(params.fill)}% walls at start`;
    } else {
      const cpx = isTest ? Math.max(4, params.cell) : params.cell;
      const gw = Math.floor(ctx.width / cpx);
      const gh = Math.floor(ctx.height / cpx);
      tag.textContent = `${R.label} · ${gw}×${gh} = ${(gw * gh).toLocaleString()} cells`;
    }
    return {
      advance: stepsNow > 0 ? 1 : 0,
      rule: RULE_KEYS.indexOf(ruleKey),
      kindR: R.kind,
      birthMask: R.b,
      surviveMask: R.s,
      density: R.density,
      start: ex === 'life' ? ['soup', 'guns', 'acorn', 'empty'].indexOf(params.start) : 0,
      erase: ctx.pointer.down && ctx.pointer.button === 2 ? 1 : 0,
      cellPx: isTest ? Math.max(4, params.cell) : params.cell,
      fill: params.fill / 100,
    };
  },
  passes: [
    {
      name: 'cells',
      format: 'rgba16float',
      filter: 'linear',
      size: (p, ctx) => {
        const c = ctx.testMode ? Math.max(4, p.cell || 4) : p.cell || 4;
        return [Math.max(8, Math.floor(ctx.width / c)), Math.max(8, Math.floor(ctx.height / c))];
      },
      iterations: () => Math.max(1, stepsNow),
      code: /* wgsl */ `
fn isOn(p: vec2i) -> f32 {
  let c = LOADW(cells, p);
  return step(0.5, c.r) * step(c.r, 1.5);   // state exactly 1
}
fn isWall(p: vec2i) -> f32 {
  let s = vec2i(TEXSIZE(cells));
  if (p.x < 0 || p.y < 0 || p.x >= s.x || p.y >= s.y) { return 1.0; }   // outside the map = rock
  return step(0.5, LOAD(cells, p).r);
}
fn bitSet(mask: f32, n: i32) -> bool { return ((i32(mask + 0.5) >> u32(n)) & 1) == 1; }

fn initCell(px: vec2f, size: vec2f, ex: i32) -> vec4f {
  let h = hash21(px + vec2f(17.0, 3.0));
  if (ex == 2) {
    // cave: random walls with the given fill (deterministic from the seed); border is always rock
    let hs = hash21(px * 1.37 + vec2f(u.seed * 13.17, u.seed * 7.31));
    var w = step(hs, u.fill);
    if (px.x < 1.0 || px.y < 1.0 || px.x > size.x - 1.0 || px.y > size.y - 1.0) { w = 1.0; }
    return vec4f(w, 0.0, 0.0, 0.0);
  }
  if (ex == 0) {
    let st = i32(u.start + 0.5);
    if (st == 1) {
      // four Gosper glider guns, two of them mirrored so the streams collide
      var on = 0.0;
      var corners = array<vec2f, 4>(vec2f(4.0, 4.0), vec2f(size.x - 40.0, 4.0), vec2f(4.0, size.y - 13.0), vec2f(size.x - 40.0, size.y - 13.0));
      for (var i = 0; i < 4; i++) {
        var q = vec2i(floor(px - corners[i]));
        if (i == 1 || i == 3) { q.x = 35 - q.x; }
        if (i >= 2) { q.y = 8 - q.y; }
        if (q.x >= 0 && q.y >= 0 && q.x < 36 && q.y < 9) { on = max(on, LOAD(patterns, q).r); }
      }
      return vec4f(step(0.5, on), 0.0, 0.0, 0.0);
    }
    if (st == 2) {
      let q = vec2i(floor(px - floor(size * 0.5) + vec2f(3.0, 1.0)));
      var on = 0.0;
      if (q.x >= 0 && q.y >= 0 && q.x < 7 && q.y < 3) { on = LOAD(patterns, q + vec2i(40, 0)).r; }
      return vec4f(step(0.5, on), 0.0, 0.0, 0.0);
    }
    if (st == 3) { return vec4f(0.0); }
    return vec4f(step(h, 0.3), 0.0, 0.0, 0.0);
  }
  // rule explorer
  let kind = i32(u.kindR + 0.5);
  if (kind == 2) { return vec4f(floor(h * 3.0), 0.0, 0.0, 0.0); }
  var d = u.density;
  let r = length((px - size * 0.5) / size.y);
  if (d < 0.2) { d = d * 8.0 * step(r, 0.12); }   // sparse rules start from a central patch
  return vec4f(step(h, d), 0.0, 0.0, 0.0);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let size = TEXSIZE(cells);
  let ex = i32(u.example + 0.5);
  let p = vec2i(floor(px));
  if (u.frame < 0.5) { return initCell(floor(px), size, ex); }
  var c = LOADW(cells, p);

  if (u.advance > 0.5) {
    if (ex == 2) {
      // cave smoothing: wall if >= 5 walls in the 3×3 block (including itself)
      if (c.a < u.caveSteps - 0.5) {
        var n = 0.0;
        for (var y = -1; y <= 1; y++) {
          for (var x = -1; x <= 1; x++) { n += isWall(p + vec2i(x, y)); }
        }
        c = vec4f(step(4.5, n), 0.0, 0.0, c.a + 1.0);
      }
    } else {
      let kind = i32(u.kindR + 0.5);
      if (kind == 2) {
        // cyclic CA (313 rule): advance to the next state if >= 3 neighbours already have it
        let nextS = fmod(c.r + 1.0, 3.0);
        var n = 0.0;
        for (var y = -1; y <= 1; y++) {
          for (var x = -1; x <= 1; x++) {
            if (x != 0 || y != 0) { n += step(abs(LOADW(cells, p + vec2i(x, y)).r - nextS), 0.1); }
          }
        }
        if (n >= 3.0) { c = vec4f(nextS, 0.0, 1.0, 0.0); } else { c = vec4f(c.r, min(c.g + 1.0, 500.0), c.b * u.trails, 0.0); }
      } else {
        var n = 0;
        for (var y = -1; y <= 1; y++) {
          for (var x = -1; x <= 1; x++) {
            if (x != 0 || y != 0) { n += i32(isOn(p + vec2i(x, y))); }
          }
        }
        let on = c.r > 0.5 && c.r < 1.5;
        if (kind == 1) {
          // Brian's Brain: on -> dying -> off; off -> on with exactly 2 "on" neighbours
          if (on) { c = vec4f(2.0, 0.0, 1.0, 0.0); }
          else if (c.r > 1.5) { c = vec4f(0.0, 0.0, 1.0, 0.0); }
          else if (n == 2) { c = vec4f(1.0, 0.0, c.b * u.trails, 0.0); }
          else { c = vec4f(0.0, 0.0, c.b * u.trails, 0.0); }
        } else {
          if (on) {
            if (bitSet(u.surviveMask, n)) { c = vec4f(1.0, min(c.g + 1.0, 500.0), c.b * u.trails, 0.0); }
            else { c = vec4f(0.0, 0.0, 1.0, 0.0); }
          } else {
            if (bitSet(u.birthMask, n)) { c = vec4f(1.0, 0.0, c.b * u.trails, 0.0); }
            else { c = vec4f(0.0, 0.0, c.b * u.trails, 0.0); }
          }
        }
      }
    }
  }

  // mouse brush (segment between last and current pointer position, in cells)
  if (u.mouse.z > 0.5) {
    let s = size / u.resolution;
    let d = sdSegment(px, u.pmouse * s, u.mouse.xy * s);
    if (d < max(0.7, u.brush / u.cellPx)) {
      if (ex == 2) {
        c = vec4f(u.erase, 0.0, 0.0, c.a);           // carve floor (right button: add rock)
      } else if (u.erase > 0.5) {
        c = vec4f(0.0, 0.0, c.b, 0.0);
      } else if (i32(u.kindR + 0.5) == 2) {
        c = vec4f(floor(hash21(px + u.time) * 3.0), 0.0, 1.0, 0.0);
      } else if (hash21(px + floor(u.time * 30.0)) < 0.55) {
        c = vec4f(1.0, 0.0, 0.0, 0.0);
      }
    }
  }
  return c;
}`,
    },
  ],
  code: /* wgsl */ `
fn lifeColor(age: f32) -> vec3f {
  let t = clamp(age / 40.0, 0.0, 1.0);
  let young = vec3f(1.0, 0.97, 0.75);
  let mid = vec3f(0.3, 0.95, 0.85);
  let old = vec3f(0.25, 0.35, 1.0);
  return mix(mix(young, mid, smoothstep(0.0, 0.15, t)), old, smoothstep(0.15, 1.0, t));
}

fn caveView(uv: vec2f, px: vec2f) -> vec3f {
  let size = TEXSIZE(cells);
  let cp = uv * size;
  let ci = vec2i(floor(cp));
  let f = fract(cp);
  let cs = u.resolution.x / size.x;
  let wall = step(0.5, LOAD(cells, ci).r);
  var below = 1.0;
  if (ci.y + 1 < i32(size.y)) { below = step(0.5, LOAD(cells, ci + vec2i(0, 1)).r); }
  // ambient occlusion: blurred wall density
  let ao = TEX(cells, uv).r * 0.4 + 0.15 * (TEX(cells, uv + vec2f(1.5, 0.0) / size).r + TEX(cells, uv - vec2f(1.5, 0.0) / size).r + TEX(cells, uv + vec2f(0.0, 1.5) / size).r + TEX(cells, uv - vec2f(0.0, 1.5) / size).r);
  let tileN = hash21(vec2f(ci));
  var col = vec3f(0.0);
  if (wall < 0.5) {
    // floor: worn flagstones
    let seam = (1.0 - smoothstep(0.0, 1.2 / cs, min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)))) * smoothstep(5.0, 8.0, cs);
    col = vec3f(0.5, 0.42, 0.33) * (0.8 + 0.25 * tileN) * (0.9 + 0.1 * valueNoise(px * 0.25));
    col = mix(col, vec3f(0.24, 0.2, 0.16), seam * 0.6);
    col *= 1.0 - 0.75 * clamp(ao, 0.0, 1.0);
    // sparse loot / mushrooms on open floor
    if (tileN > 0.985 && ao < 0.05) {
      let gem = 1.0 - smoothstep(0.12, 0.2, length(f - 0.5));
      col = mix(col, vec3f(0.3, 1.0, 0.7), gem);
    }
  } else if (below < 0.5 && f.y > 0.35) {
    // 2.5D wall face: bricks above a floor tile
    let by = (f.y - 0.35) / 0.65;
    let row = floor(by * 3.0);
    let bx = fract(f.x * 2.0 + row * 0.5);
    let mortar = max(1.0 - smoothstep(0.0, 0.06, min(bx, 1.0 - bx)), 1.0 - smoothstep(0.0, 0.08, abs(fract(by * 3.0) - 0.0)));
    col = vec3f(0.62, 0.52, 0.44) * (0.85 + 0.2 * hash21(vec2f(ci) + row));
    col = mix(col, vec3f(0.2, 0.16, 0.14), mortar * 0.85);
    col *= 0.55 + 0.45 * (1.0 - by);
  } else {
    // wall top: rough rock, darker deep inside
    let edge = 1.0 - clamp(ao * 1.15 - 0.35, 0.0, 1.0);
    col = vec3f(0.3, 0.29, 0.34) * (0.7 + 0.5 * valueNoise(px * 0.12)) * (0.35 + 0.9 * edge);
  }
  // lantern light (follows the mouse) + ambient
  var lp = vec2f(0.5, 0.5) * u.resolution;
  if (u.mouse.w > 0.5) { lp = u.mouse.xy; }
  let ld = length(px - lp) / u.resolution.y;
  let flick = 0.92 + 0.08 * sin(u.time * 11.0) * sin(u.time * 7.0);
  let light = 0.5 + 1.0 * exp(-ld * 3.2) * flick;
  col *= light * vec3f(1.05, 0.95, 0.8);
  col += vec3f(1.0, 0.6, 0.25) * exp(-ld * 40.0) * 0.6 * u.mouse.w;
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example + 0.5);
  if (ex == 2) { return vec4f(caveView(uv, px), 1.0); }
  let size = TEXSIZE(cells);
  let cp = uv * size;
  let ci = vec2i(floor(cp));
  let f = fract(cp);
  let cs = u.resolution.x / size.x;
  let c = LOAD(cells, ci);
  let kind = i32(u.kindR + 0.5);
  var bg = mix(vec3f(0.018, 0.02, 0.035), vec3f(0.03, 0.025, 0.05), uv.y);
  if (cs >= 4.0) {
    let g = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * cs;
    bg += vec3f(0.025, 0.03, 0.05) * (1.0 - smoothstep(0.0, 1.0, g));
  }
  // cells drawn as rounded squares
  let d = sdRoundBox((f - 0.5) * cs, vec2f(cs * 0.5 - max(0.6, cs * 0.08)), cs * 0.22);
  let fillC = clamp(0.5 - d, 0.0, 1.0);
  var col = bg;
  if (kind == 2) {
    let st = c.r;
    let hue = st / 3.0 + 0.55;
    let cc = hsv2rgb(vec3f(fract(hue), 0.75, 0.95 - 0.5 * clamp(c.g / 60.0, 0.0, 1.0)));
    col = mix(bg, cc, max(fillC, 0.85));
    return vec4f(col, 1.0);
  }
  // trail glow of recently dead cells
  let tr = TEX(cells, uv).b;
  col += vec3f(0.85, 0.25, 0.75) * tr * tr * 0.55;
  // soft bloom from live cells
  let e = 1.6 / size;
  let glow = TEX(cells, uv + vec2f(e.x, 0.0)).r + TEX(cells, uv - vec2f(e.x, 0.0)).r + TEX(cells, uv + vec2f(0.0, e.y)).r + TEX(cells, uv - vec2f(0.0, e.y)).r;
  if (kind == 1) {
    col += vec3f(0.2, 0.4, 1.0) * clamp(glow, 0.0, 4.0) * 0.06;
    if (c.r > 0.5 && c.r < 1.5) { col = mix(col, vec3f(0.75, 0.9, 1.0), fillC); }
    if (c.r > 1.5) { col = mix(col, vec3f(0.9, 0.2, 0.35), fillC * 0.9); }
    return vec4f(col, 1.0);
  }
  col += vec3f(0.3, 0.8, 0.9) * clamp(glow, 0.0, 4.0) * 0.05;
  if (c.r > 0.5) { col = mix(col, lifeColor(c.g), fillC); }
  return vec4f(col, 1.0);
}`,
  about: {
    summary:
      'Cellular automata: a grid of cells, each updated by the same tiny rule that only looks at its neighbours. Gliders, guns, spirals — and practical roguelike cave maps. Runs on WebGPU and WebGL2.',
    what: `<p>A grid stored in a texture. Every generation a fragment shader computes the next state of <i>all</i> cells at once from the previous texture
      (double buffering, so every cell sees the same “old” world). Colors show extra bookkeeping: how long a cell has lived, and a fading trail after it dies.</p>`,
    how: `<ol>
      <li><b>Neighbourhood</b>: each cell counts the live cells among its 8 neighbours (the “Moore neighbourhood”). The edges wrap around, so the world is a torus.</li>
      <li><b>Rule</b> as two lists, written <code>B3/S23</code>: <b>B</b>irth if the count is in the first list, <b>S</b>urvive if it is in the second. In the shader those lists are bit masks, so any “Life-like” rule is just two numbers.</li>
      <li><b>Double buffering</b> is essential: if you updated cells in place, cells processed later would see a half-updated world. The GPU naturally reads texture A and writes texture B.</li>
      <li><b>Speed</b>: the simulation runs at a fixed number of generations per second, independent of your display’s refresh rate; above 60 it runs several passes per frame.</li>
      <li><b>Caves</b>: start from random noise (≈45% walls), then repeatedly apply the “4-5 rule” (a cell becomes rock if at least 5 cells of its 3×3 block are rock). Isolated specks vanish, gaps fill, and smooth organic caverns appear after 4–5 iterations.</li>
    </ol>`,
    uses: [
      { title: 'Procedural caves & maps', text: 'The 4-5 rule is a staple of roguelikes and Terraria-likes; follow it with a flood fill to keep the largest connected cave and place stairs/loot in open areas.' },
      { title: 'Spreading simulations', text: 'Fire spreading through grass, infection, crowd panic, forest growth — CA rules are easy to design and run on the whole map at once.' },
      { title: 'Falling sand & liquids', text: 'Games like Noita are cellular automata with more states and smarter update orders (see the Falling Sand scene).' },
      { title: 'Puzzles & toys', text: 'Life-like rules make great generative backgrounds, screensavers and puzzle mechanics.' },
    ],
    try: [
      'On <b>Conway’s Life</b> pick <i>Glider guns</i>: four guns fire streams of gliders that collide in the middle.',
      'Pick <i>Acorn</i>: 7 cells that take over 5,000 generations to settle. Raise the speed to 240.',
      'On <b>Rule explorer</b>, try <i>Seeds</i> (everything explodes) vs <i>Maze</i> (grows corridors) vs <i>Cyclic</i> (spirals self-organise from noise).',
      'On <b>Cave generation</b>, set <i>Smoothing steps</i> to 0, then step it up one at a time. Change <i>Initial wall fill</i> by ±5% to see open caverns vs isolated pockets.',
      'Pause (⏸) and draw a pattern, then unpause to see what it does.',
    ],
    ask: [
      'procedural cave generation with cellular automata (4-5 rule)',
      'Conway’s Game of Life on the GPU with ping-pong textures',
      'a fire/infection spread simulation on a grid',
      'deterministic seeded map generation',
      'a cellular automaton background for my menu',
    ],
    perf: `<p>One full-screen pass per generation with 9 texture reads per cell — 500k cells cost well under a millisecond. Even at 240 generations per second it’s cheap.
      On the CPU in JavaScript a 1000×1000 Life grid runs at maybe 30–100 generations per second; the GPU does thousands.</p>`,
    api: `<p><b>Runs on both WebGPU and WebGL2.</b> This is the textbook “GPGPU with fragment shaders” technique that predates compute shaders: render a full-screen
      quad into a texture whose pixels are the cells. A compute shader version could pack 32 cells into one integer and count neighbours with bit tricks for even more speed.</p>`,
    code: [
      {
        title: 'Life-like rule with bit masks',
        lang: 'wgsl',
        src: `fn bitSet(mask: f32, n: i32) -> bool { return ((i32(mask + 0.5) >> u32(n)) & 1) == 1; }

var n = 0;
for (var y = -1; y <= 1; y++) {
  for (var x = -1; x <= 1; x++) {
    if (x != 0 || y != 0) { n += i32(isOn(p + vec2i(x, y))); }
  }
}
// Conway B3/S23: birthMask = 1<<3, surviveMask = (1<<2) | (1<<3)
if (on) { alive = bitSet(u.surviveMask, n); } else { alive = bitSet(u.birthMask, n); }`,
      },
      {
        title: 'Cave smoothing (the 4-5 rule)',
        lang: 'wgsl',
        src: `var n = 0.0;
for (var y = -1; y <= 1; y++) {
  for (var x = -1; x <= 1; x++) { n += isWall(p + vec2i(x, y)); }  // outside the map counts as wall
}
let wall = step(4.5, n);   // rock if 5+ of the 9 cells are rock`,
      },
    ],
    links: [
      { title: 'LifeWiki', url: 'https://conwaylife.com/wiki/', note: 'every pattern and rule' },
      { title: 'RogueBasin — Cellular Automata Method for Generating Random Cave-Like Levels', url: 'https://www.roguebasin.com/index.php/Cellular_Automata_Method_for_Generating_Random_Cave-Like_Levels' },
    ],
  },
});

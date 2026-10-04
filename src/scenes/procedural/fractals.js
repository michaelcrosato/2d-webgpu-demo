import { shaderScene } from '../../core/shaderscene.js';
import { setLabels, PanZoom } from './_shared.js';

// Fractals: Mandelbrot (smooth coloring, palettes, orbit display, pan/zoom/fly-to and an honest
// float32 precision limit), Julia sets with c from the mouse, the Burning Ship, and Newton's
// method basins. The view (center, zoom) lives in JavaScript as doubles and goes to the shader
// as a uniform — which is exactly where float32 precision runs out.

// world: x = Re(c), y = −Im(c) (screen y points down)
const views = {
  mandelbrot: new PanZoom({ x: -0.62, y: 0, zoom: 0.4, minZoom: 0.2, maxZoom: 5e6 }),
  julia: new PanZoom({ x: 0, y: 0, zoom: 0.36, minZoom: 0.2, maxZoom: 5e6 }),
  ship: new PanZoom({ x: -0.45, y: -0.5, zoom: 0.38, minZoom: 0.2, maxZoom: 5e6 }),
  newton: new PanZoom({ x: 0, y: 0, zoom: 0.33, minZoom: 0.05, maxZoom: 5e6 }),
};
const julia = { c: [-0.75, 0.12], target: [-0.75, 0.12], locked: false };

const fmtE = (v) => (v >= 1000 || v < 0.01 ? v.toExponential(1).replace('e+', 'e') : v.toFixed(v < 10 ? 2 : 0));

function precisionLabel(pz, ctx) {
  const px = 1 / (pz.zoom * ctx.height); // world units per pixel
  const mag = Math.max(Math.abs(pz.x), Math.abs(pz.y), 0.5);
  const ulp = mag * 1.19e-7; // spacing of float32 numbers near the center
  const ratio = px / ulp;
  const warn = ratio < 4;
  return {
    text:
      `zoom <b>${fmtE(pz.zoom / pz.home.zoom)}×</b> · pixel = ${px.toExponential(1)} · float32 step ≈ ${ulp.toExponential(1)} ` +
      (warn ? '<span style="color:#fca5a5">⚠ past float32 precision: pixels become blocks</span>' : '<span style="color:#86efac">✓ precise</span>'),
    style: 'right:8px;bottom:8px;font-size:11px',
  };
}

const CODE = /* wgsl */ `
// world (u.view: center.xy, world units per pixel, zoom) -> complex number (y up)
fn toC(px: vec2f) -> vec2f {
  let w = u.view.xy + (px - 0.5 * u.resolution) * u.view.z;
  return vec2f(w.x, -w.y);
}
fn toPx(c: vec2f) -> vec2f { return (vec2f(c.x, -c.y) - u.view.xy) / u.view.z + 0.5 * u.resolution; }

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / max(dot(b, b), 1e-20); }

// "Ultra Fractal"-style gradient (deep blue -> white -> orange -> black), cyclic
fn ultra(t0: f32) -> vec3f {
  let t = fract(t0);
  let c0 = vec3f(0.0, 0.03, 0.39);
  let c1 = vec3f(0.13, 0.42, 0.8);
  let c2 = vec3f(0.93, 1.0, 1.0);
  let c3 = vec3f(1.0, 0.67, 0.0);
  let c4 = vec3f(0.0, 0.01, 0.0);
  var c = mix(c0, c1, smoothstep(0.0, 0.16, t));
  c = mix(c, c2, smoothstep(0.16, 0.42, t));
  c = mix(c, c3, smoothstep(0.42, 0.64, t));
  c = mix(c, c4, smoothstep(0.64, 0.86, t));
  c = mix(c, c0, smoothstep(0.86, 1.0, t));
  return c;
}

fn fpal(t: f32) -> vec3f {
  let k = i32(u.pal);
  if (k == 0) { return ultra(t); }
  if (k == 1) { return palette(t, vec3f(0.5, 0.5, 0.5), vec3f(0.5, 0.5, 0.5), vec3f(1.0, 0.7, 0.4), vec3f(0.0, 0.15, 0.2)); }
  if (k == 2) { return palette(t, vec3f(0.5, 0.5, 0.5), vec3f(0.5, 0.5, 0.5), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.33, 0.67)); }
  if (k == 3) {
    let x = fract(t);
    return mix(mix(vec3f(0.05, 0.0, 0.0), vec3f(0.9, 0.2, 0.02), smoothstep(0.0, 0.4, x)), vec3f(1.0, 0.95, 0.6), smoothstep(0.4, 0.9, x)) * (1.0 - 0.6 * smoothstep(0.9, 1.0, x));
  }
  return vec3f(0.5 + 0.5 * cos(t * TAU));
}

// escape-time iteration. kind 0: Mandelbrot, 1: Julia (z0 = p, c = u.jc), 2: Burning Ship
// returns (smooth iteration count or -1 inside, orbit trap = min |z|^2)
fn escape(p: vec2f, kind: i32) -> vec2f {
  var z = vec2f(0.0);
  var c = p;
  if (kind == 1) { z = p; c = u.jc; }
  if (kind == 0) {
    // skip the main cardioid and the period-2 bulb: they never escape (big speed-up)
    let q = (c.x - 0.25) * (c.x - 0.25) + c.y * c.y;
    if (q * (q + (c.x - 0.25)) < 0.25 * c.y * c.y) { return vec2f(-1.0, 0.0); }
    if ((c.x + 1.0) * (c.x + 1.0) + c.y * c.y < 0.0625) { return vec2f(-1.0, 0.0); }
  }
  var trap = 1e9;
  for (var i = 0; i < 4000; i++) {
    if (f32(i) >= u.maxIter) { break; }
    if (kind == 2) { z = abs(z); }
    z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    let m2 = dot(z, z);
    trap = min(trap, m2);
    if (m2 > 65536.0) {
      // continuous ("smooth") iteration count: removes the color bands
      return vec2f(f32(i) + 1.0 - log2(0.5 * log(m2)), trap);
    }
  }
  return vec2f(-1.0, trap);
}

fn colorEscape(r: vec2f, kind: i32) -> vec3f {
  if (r.x < 0.0) {
    // inside: dark, with a faint glow from the orbit trap
    let g = clamp(1.0 - sqrt(r.y), 0.0, 1.0);
    return vec3f(0.01, 0.012, 0.03) + fpal(0.6 + r.y * 0.4) * 0.12 * g;
  }
  var t = sqrt(max(r.x, 0.0)) * u.density + u.time * u.cycle * 0.1;
  if (kind == 2) { t = t + 0.5; }
  // points that escape immediately are far from the set: fade them out so the boundary glows
  let glow = 1.0 - exp(-max(r.x, 0.0) * 0.11);
  return fpal(t) * (0.06 + 0.94 * glow);
}

// Newton's method on z^n - 1: which root does each starting point converge to, and how fast?
fn newton(p: vec2f) -> vec3f {
  var z = p;
  let n = i32(u.degree);
  var it = 0.0;
  let maxN = min(u.maxIter, 120.0);
  for (var i = 0; i < 120; i++) {
    if (f32(i) >= maxN) { break; }
    var zn1 = vec2f(1.0, 0.0);
    for (var k = 0; k < 7; k++) {
      if (k >= n - 1) { break; }
      zn1 = cmul(zn1, z);
    }
    let zn = cmul(zn1, z);
    let dz = cdiv(zn - vec2f(1.0, 0.0), zn1 * f32(n)) * u.relax;
    z -= dz;
    it = f32(i);
    if (dot(dz, dz) < 1e-10) { break; }
  }
  let seg = TAU / f32(n);
  let root = fmod(floor(atan2(z.y, z.x) / seg + 0.5), f32(n));
  return vec3f(root, it, length(z - vec2f(cos(root * seg), sin(root * seg))));
}

fn colorNewton(r: vec3f) -> vec3f {
  let hue = r.x / u.degree;
  var c = palette(hue + 0.05, vec3f(0.55, 0.5, 0.5), vec3f(0.4, 0.4, 0.45), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.33, 0.67));
  // brighter = converged in fewer steps
  let k = exp(-r.y * 0.09);
  c *= 0.2 + 0.95 * k;
  if (r.z > 0.01) { c *= 0.25; }
  return c;
}

fn fractalColor(px: vec2f) -> vec3f {
  let ex = i32(u.example);
  let c = toC(px);
  if (ex == 3) { return colorNewton(newton(c)); }
  var kind = 0;
  var cc = c;
  if (ex == 1) { kind = 1; }
  if (ex == 2) { kind = 2; cc = vec2f(c.x, -c.y); }
  return colorEscape(escape(cc, kind), kind);
}

// polyline of an orbit z0 -> z1 -> ... (Mandelbrot: z0 = 0; Newton: the steps toward a root)
fn orbitOverlay(col0: vec3f, px: vec2f, c: vec2f, ex: i32) -> vec3f {
  var col = col0;
  var z = vec2f(0.0);
  if (ex == 3) { z = c; }
  var prev = toPx(z);
  var dmin = 1e9;
  var dots = 1e9;
  let n = i32(u.degree);
  for (var i = 0; i < 60; i++) {
    if (ex == 3) {
      var zn1 = vec2f(1.0, 0.0);
      for (var k = 0; k < 7; k++) {
        if (k >= n - 1) { break; }
        zn1 = cmul(zn1, z);
      }
      z -= cdiv(cmul(zn1, z) - vec2f(1.0, 0.0), zn1 * f32(n)) * u.relax;
    } else {
      z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    }
    if (dot(z, z) > 16.0) { break; }
    let cur = toPx(z);
    dmin = min(dmin, sdSegment(px, prev, cur));
    dots = min(dots, length(px - cur));
    prev = cur;
  }
  col = mix(col, vec3f(0.0), (1.0 - smoothstep(1.0, 3.0, dmin)) * 0.5);
  col = mix(col, vec3f(1.0, 0.95, 0.6), (1.0 - smoothstep(0.4, 1.3, dmin)) * 0.85);
  col = mix(col, vec3f(1.0), 1.0 - smoothstep(2.0, 3.0, dots));
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (u.aa > 0.5) {
    // 4 samples per pixel (rotated grid) = 4x the cost
    col = fractalColor(px + vec2f(0.125, 0.375)) + fractalColor(px + vec2f(-0.375, 0.125))
        + fractalColor(px + vec2f(0.375, -0.125)) + fractalColor(px + vec2f(-0.125, -0.375));
    col *= 0.25;
  } else {
    col = fractalColor(px);
  }
  // overlays
  if (ex == 3) {
    // the n roots of unity
    for (var k = 0; k < 8; k++) {
      if (f32(k) >= u.degree) { break; }
      let a = f32(k) * TAU / u.degree;
      let rp = toPx(vec2f(cos(a), sin(a)));
      let d = length(px - rp);
      col = mix(col, vec3f(0.0), 1.0 - smoothstep(6.0, 7.5, d));
      col = mix(col, vec3f(1.0), 1.0 - smoothstep(4.0, 5.0, d));
    }
  }
  if (u.orbit > 0.5 && u.mouse.w > 0.5 && u.mouse.z < 0.5 && (ex == 0 || ex == 3)) {
    col = orbitOverlay(col, px, toC(u.mouse.xy), ex);
  }
  if (ex == 1) {
    // inset: the Mandelbrot set as a map of c; the dot is the current c
    let ih = floor(u.resolution.y * 0.26);
    let iw = floor(ih * 1.3);
    let io = u.resolution - vec2f(iw + 12.0, ih + 12.0);
    let lq = (px - io) / vec2f(iw, ih);
    if (lq.x > 0.0 && lq.x < 1.0 && lq.y > 0.0 && lq.y < 1.0) {
      let mc = vec2f(mix(-2.1, 0.75, lq.x), mix(1.1, -1.1, lq.y));
      var z = vec2f(0.0);
      var esc = 0.0;
      for (var i = 0; i < 60; i++) {
        z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + mc;
        if (dot(z, z) > 4.0) { esc = f32(i); break; }
      }
      var ic = vec3f(0.02, 0.025, 0.05);
      if (esc > 0.0) { ic = fpal(sqrt(esc) * 0.25) * 0.75; }
      let cp = io + vec2f((u.jc.x + 2.1) / 2.85, (1.1 - u.jc.y) / 2.2) * vec2f(iw, ih);
      let dc = length(px - cp);
      ic = mix(ic, vec3f(0.0), 1.0 - smoothstep(5.0, 6.5, dc));
      ic = mix(ic, vec3f(1.0, 0.3, 0.4), 1.0 - smoothstep(3.0, 4.2, dc));
      col = ic;
    }
    col = mix(col, vec3f(0.85), (1.0 - smoothstep(0.5, 1.5, abs(max(abs(lq.x - 0.5) * iw - iw * 0.5, abs(lq.y - 0.5) * ih - ih * 0.5)))) * 0.8);
  }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: 'Drag: pan · wheel: zoom · click: zoom in 2.5× · right-click: zoom out.',
  wheel: true,
  examples: [
    {
      id: 'mandelbrot',
      label: 'Mandelbrot set',
      kind: 'Classic',
      note: 'For every pixel’s complex number <b>c</b>, iterate <code>z = z² + c</code> from z = 0. If |z| escapes past a radius, color the pixel by <i>how many steps</i> it took (smoothed so there are no bands); if it never escapes, it’s inside the set (black). Hover to see the <b>orbit</b> of z for the point under the cursor. Use the fly-to buttons — and visit the precision limit.',
    },
    {
      id: 'julia',
      label: 'Julia sets',
      kind: 'In a game',
      note: 'Same formula, but now <b>c is fixed</b> (taken from your mouse — see the dot on the Mandelbrot map in the corner) and the pixel is the starting z. Every point of the Mandelbrot set gives a connected Julia set; outside it, the Julia set shatters into dust. Animated c makes a mesmerising backdrop for a title screen, a warp portal or a boss arena. <b>Click</b> to lock/unlock c.',
      params: { maxIter: 200, pal: 'neon' },
      hint: 'Hover: choose c · click: lock c · drag: pan · wheel: zoom.',
    },
    {
      id: 'ship',
      label: 'Burning Ship',
      kind: 'Classic',
      note: 'One tiny change — take the absolute value of z’s real and imaginary parts before squaring — and the symmetric Mandelbrot turns into a jagged, burning ship (look on the left, around −1.76). Small formula changes create entirely new worlds.',
      params: { pal: 'fire', maxIter: 300 },
    },
    {
      id: 'newton',
      label: 'Newton’s method',
      kind: 'Real life',
      note: 'Newton’s method finds roots of equations — used in physics solvers, inverse kinematics, ray–surface intersection and finance. Start anywhere, repeatedly jump along the tangent: <code>z −= f(z) / f′(z)</code>. Coloring each starting point by the root (white dots) it reaches reveals fractal borders where the method is chaotic. Hover to see the jumps.',
      params: { maxIter: 60 },
    },
  ],
  controls: [
    { type: 'slider', key: 'maxIter', label: 'Max iterations', min: 16, max: 4000, step: 1, value: 300, log: true, format: (v) => Math.round(v).toString(), help: 'More = finer detail near the boundary when zoomed in (and slower: cost ≈ pixels × iterations).' },
    {
      type: 'select', key: 'pal', label: 'Palette', value: 'ultra', showFor: ['mandelbrot', 'julia', 'ship'],
      options: [{ value: 'ultra', label: 'Ultra (blue–white–gold)' }, { value: 'sunset', label: 'Sunset' }, { value: 'neon', label: 'Neon rainbow' }, { value: 'fire', label: 'Fire' }, { value: 'mono', label: 'Grayscale bands' }],
    },
    { type: 'slider', key: 'density', label: 'Color density', min: 0.02, max: 1, step: 0.005, value: 0.2, showFor: ['mandelbrot', 'julia', 'ship'], help: 'How fast the palette repeats with iteration count.' },
    { type: 'slider', key: 'cycle', label: 'Color cycling', min: 0, max: 3, step: 0.01, value: 0.4, showFor: ['mandelbrot', 'julia', 'ship'], help: 'Shift the palette over time — the classic demoscene trick.' },
    { type: 'toggle', key: 'orbit', label: 'Show orbit under the mouse', value: true, showFor: ['mandelbrot', 'newton'], help: 'The sequence of z values for the hovered point.' },
    { type: 'slider', key: 'degree', label: 'Polynomial zⁿ − 1, n =', min: 3, max: 8, step: 1, value: 3, showFor: ['newton'] },
    { type: 'slider', key: 'relax', label: 'Step size (relaxation)', min: 0.3, max: 1.9, step: 0.01, value: 1, showFor: ['newton'], help: '1 = classic Newton. Other values overshoot or undershoot each jump.' },
    { type: 'toggle', key: 'aa', label: 'Anti-aliasing (4 samples)', value: false, help: 'Smooths the noisy edges — at 4× the cost.' },
    { type: 'button', key: 'resetView', label: 'Reset view' },
    { type: 'button', key: 'flySeahorse', label: 'Fly: Seahorse valley', showFor: ['mandelbrot'] },
    { type: 'button', key: 'flyMini', label: 'Fly: mini-Mandelbrot', showFor: ['mandelbrot'] },
    { type: 'button', key: 'flyDeep', label: 'Fly: precision limit', showFor: ['mandelbrot'] },
    { type: 'button', key: 'flyShip', label: 'Fly: the little ship', showFor: ['ship'] },
  ],
  uniforms: {
    maxIter: 'f32', pal: 'f32', density: 'f32', cycle: 'f32', orbit: 'f32', degree: 'f32', relax: 'f32', aa: 'f32',
    view: 'vec4f', jc: 'vec2f',
  },
  include: ['hash', 'sdf', 'color'],
  onAction(key, ctx) {
    const pz = views[ctx.example];
    if (!pz) return;
    if (key === 'resetView' || key === 'reset') pz.flyTo(pz.home.x, pz.home.y, pz.home.zoom);
    // targets in world space: x = Re, y = −Im
    if (key === 'flySeahorse') pz.flyTo(-0.7453, -0.1127, 120);
    if (key === 'flyMini') pz.flyTo(-1.7548, 0, 22);
    if (key === 'flyDeep') pz.flyTo(-0.743643887, -0.131825904, 120000);
    if (key === 'flyShip') pz.flyTo(-1.762, -0.035, 12);
  },
  bind(params, ctx) {
    const ex = ctx.example;
    const pz = views[ex] || views.mandelbrot;
    const p = ctx.pointer;
    const clicked = pz.update(ctx, { smooth: 0.12 });
    if (clicked) {
      if (ex === 'julia') julia.locked = !julia.locked;
      else {
        const [wx, wy] = pz.toWorld(ctx, p.x, p.y);
        pz.flyTo(wx, wy, p.button === 2 ? pz.tzoom / 2.5 : pz.tzoom * 2.5);
      }
    }
    if (ex === 'julia') {
      if (!julia.locked) {
        if (p.over && !p.down) julia.target = [(p.nx - 0.5) * 2.85 - 0.675, (0.5 - p.ny) * 2.2];
        else if (!p.over) {
          const a = ctx.time * 0.25;
          julia.target = [0.7885 * Math.cos(a), 0.7885 * Math.sin(a)];
        }
      }
      julia.c[0] += (julia.target[0] - julia.c[0]) * 0.08;
      julia.c[1] += (julia.target[1] - julia.c[1]) * 0.08;
      setLabels(ctx, [
        { text: `c = ${julia.c[0].toFixed(3)} ${julia.c[1] < 0 ? '−' : '+'} ${Math.abs(julia.c[1]).toFixed(3)}i${julia.locked ? ' 🔒' : ''}`, style: 'left:8px;top:44px' },
        { text: 'c on the Mandelbrot map', style: 'right:12px;bottom:calc(26% + 18px);font-size:11px;opacity:.85' },
      ]);
    } else {
      setLabels(ctx, [precisionLabel(pz, ctx)]);
    }
    return { view: [pz.x, pz.y, 1 / (pz.zoom * ctx.height), pz.zoom], jc: julia.c };
  },
  code: CODE,
  about: {
    summary: 'Fractals have detail at every scale. The Mandelbrot and Julia sets come from one line — z = z² + c — repeated per pixel, which is exactly the kind of massively parallel work GPUs love.',
    what: `<p><b>Mandelbrot</b>: the map of all c whose orbit stays bounded, with smooth coloring, orbit display, fly-to locations and a live float32 precision readout.
      <b>Julia</b>: c chosen with the mouse. <b>Burning Ship</b>: a variant formula. <b>Newton’s method</b>: basins of attraction of a root-finding algorithm.</p>`,
    how: `<ol>
      <li><b>Complex numbers as vec2</b>: z = (x, y); squaring is <code>(x² − y², 2xy)</code>. Every pixel maps to a complex number through the camera (center + zoom).</li>
      <li><b>Escape time</b>: iterate <code>z = z² + c</code>. Once |z| &gt; 2 it will fly off to infinity — count the steps. Never escapes (up to <i>Max iterations</i>) = inside.</li>
      <li><b>Smooth coloring</b>: instead of the integer step count use <code>n + 1 − log₂(log|z|)</code> (with a large escape radius). The fractional part removes the color bands.</li>
      <li><b>Palette</b>: map the smooth count through a cyclic gradient; adding time to the lookup makes colors flow (color cycling).</li>
      <li><b>Speed-ups</b>: points in the big cardioid and the period-2 circle are tested with a closed formula and skipped — they would otherwise run all iterations.</li>
      <li><b>The precision limit</b>: GPUs compute in 32-bit floats with ~7 significant digits. When neighbouring pixels differ by less than the spacing between
        representable floats (≈ 1.2×10⁻⁷ near 1), they get the <i>same</i> c, and the image turns into blocks. Deep-zoom renderers use double precision on the CPU for one
        reference orbit and compute the others as tiny differences from it in floats (<b>perturbation theory</b>), or emulate doubles with pairs of floats.</li>
    </ol>`,
    uses: [
      { title: 'Backgrounds & visualizers', text: 'Animated Julia sets for menus, music visualizers, warp tunnels and psychedelic boss stages.' },
      { title: 'Procedural art', text: 'Alien planets, magic sigils, “infinite zoom” transitions and screensavers (fractal zoomers were classic demoscene showpieces).' },
      { title: 'Math & engineering', text: 'Newton’s method is everywhere: physics constraint solvers, inverse kinematics, finding where a ray hits a surface.' },
      { title: 'Benchmarks', text: 'Escape-time fractals are a classic GPU stress test: pure ALU work, perfectly parallel.' },
    ],
    try: [
      'Press <b>Fly: Seahorse valley</b>, then raise <i>Max iterations</i> — dark blobs at the edges fill with detail.',
      'Press <b>Fly: precision limit</b> and watch the readout turn red as the image becomes blocky: that is float32 running out of digits.',
      'Hover inside the black main body with the orbit on: the orbit spirals into a fixed point. Near the edge it wanders for a long time before escaping.',
      'On <b>Julia sets</b>, move the mouse along the edge of the Mandelbrot map (the dot in the inset): that is where the most intricate Julia sets live.',
      'On <b>Newton’s method</b>, set n = 5 and move <i>Step size</i> away from 1: the basins twist and grow new chaotic regions.',
    ],
    ask: [
      'GPU Mandelbrot with smooth coloring and palette cycling',
      'animated Julia set background',
      'click-to-zoom fractal explorer with drag to pan',
      'perturbation theory / double-float emulation for deep fractal zooms',
      'Newton fractal (basins of attraction) shader',
      'orbit trap coloring',
    ],
    perf: `<p>Cost ≈ <b>pixels × iterations</b>. Points outside escape quickly; points inside run every iteration, so views full of black are the slowest
      (the cardioid/bulb check removes the biggest black areas for free). 4× anti-aliasing costs 4×. At 1080p, 300 iterations is easy for any discrete or modern integrated GPU;
      thousands of iterations when deeply zoomed is where it gets heavy — real explorers render progressively over several frames.</p>`,
    api: `<p>Same shader in WebGL2 and WebGPU. Both are limited to 32-bit floats in shaders (WebGPU has an optional 16-bit <code>f16</code> extension, but no
      64-bit doubles on the web), so the precision limit is identical. WebGPU compute shaders make <i>progressive</i> rendering (keep iterating unfinished pixels across
      frames in a storage buffer) and perturbation-based deep zoom much more convenient.</p>`,
    code: [
      {
        title: 'Escape time with smooth coloring',
        lang: 'wgsl',
        src: `var z = vec2f(0.0);
for (var i = 0; i < 4000; i++) {
  if (f32(i) >= u.maxIter) { break; }
  if (kind == 2) { z = abs(z); }                         // Burning Ship
  z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;   // z = z² + c
  let m2 = dot(z, z);
  if (m2 > 65536.0) {
    return f32(i) + 1.0 - log2(0.5 * log(m2));            // smooth count
  }
}
return -1.0;                                             // inside the set`,
      },
      {
        title: 'Newton’s method on zⁿ − 1',
        lang: 'wgsl',
        src: `for (var i = 0; i < 120; i++) {
  var zn1 = vec2f(1.0, 0.0);                    // z^(n-1)
  for (var k = 0; k < 7; k++) { if (k >= n - 1) { break; } zn1 = cmul(zn1, z); }
  let dz = cdiv(cmul(zn1, z) - vec2f(1.0, 0.0), zn1 * f32(n)) * u.relax;   // f / f'
  z -= dz;
  if (dot(dz, dz) < 1e-10) { break; }           // converged
}
let root = fmod(floor(atan2(z.y, z.x) / (TAU / f32(n)) + 0.5), f32(n));`,
      },
      {
        title: 'JavaScript: the camera in double precision',
        lang: 'js',
        src: `// pan/zoom state is a JS double — but the shader only gets float32:
return { view: [pz.x, pz.y, 1 / (pz.zoom * ctx.height), pz.zoom] };
// pixel size vs float32 spacing near the center:
const ulp = Math.max(Math.abs(pz.x), Math.abs(pz.y), 0.5) * 1.19e-7;
if (1 / (pz.zoom * ctx.height) < 4 * ulp) warn('blocky!');`,
      },
    ],
    links: [
      { title: 'Inigo Quilez — Smooth iteration count', url: 'https://iquilezles.org/articles/msetsmooth/' },
      { title: 'Wikipedia — Plotting algorithms for the Mandelbrot set', url: 'https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set', note: 'smooth coloring, perturbation, distance estimation' },
      { title: '3Blue1Brown — Newton’s fractal', url: 'https://www.youtube.com/watch?v=-RdOwhmqP5s' },
    ],
  },
});

// 2D shadows by ray-marching a signed distance field (SDF).
//   1. Occluders (boxes, circles, capsules) live in a storage buffer and are animated on the CPU.
//   2. A compute shader "bakes" the distance to the nearest occluder into a half-resolution texture.
//   3. A fullscreen pass walks from every pixel toward every light, using the SDF to take big safe
//      steps (sphere tracing). The closest miss along the way gives a soft penumbra for free.
//   Stealth tab: the guards' line of sight is ALSO tested on the CPU (segment vs shapes) for gameplay.

import { Camera2D, ShapeBatch } from '../../core/batch.js';
import { tag, hexLinear, damp } from './_shared.js';

const MAX_SHAPES = 96;
const MAX_LIGHTS = 8;

const SHAPE_WGSL = /* wgsl */ `
// a = (x, y, halfW | radius, halfH)   b = (rotation, type 0 box / 1 circle / 2 capsule, x2, y2)   c = color
struct Shape { a: vec4f, b: vec4f, c: vec4f };
fn sdShape(p: vec2f, s: Shape) -> f32 {
  if (s.b.y < 0.5) {
    let q = p - s.a.xy;
    let cs = cos(s.b.x);
    let sn = sin(s.b.x);
    let l = vec2f(cs * q.x + sn * q.y, -sn * q.x + cs * q.y);
    return sdBox(l, s.a.zw);
  }
  if (s.b.y < 1.5) { return length(p - s.a.xy) - s.a.z; }
  return sdSegment(p, s.a.xy, s.b.zw) - s.a.z;
}`;

const BAKE_WGSL = /* wgsl */ `
${SHAPE_WGSL}
// one thread per SDF texel: distance (in screen pixels) to the nearest occluder + its index
@compute @workgroup_size(8, 8)
fn bake(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(sdfOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = (vec2f(id.xy) + 0.5) * u.scale;
  var d = 1e5;
  var idx = 0.0;
  for (var i: u32 = 0u; i < u.count; i++) {
    let di = sdShape(p, shapes[i]);
    if (di < d) { d = di; idx = f32(i); }
  }
  textureStore(sdfOut, vec2i(id.xy), vec4f(d, idx, 0.0, 1.0));
}`;

const MARCH_WGSL = /* wgsl */ `
// Sphere-traced soft shadow from p toward a light at lp with radius 'size' (0 = point light = hard shadow).
fn shadowRay(p: vec2f, lp: vec2f, size: f32, jitter: f32) -> vec2f {
  let toL = lp - p;
  let dist = length(toL);
  if (dist < 1.0) { return vec2f(1.0, 0.0); }
  let dir = toL / dist;
  var t = 1.0 + jitter * 2.0;
  var vis = 1.0;
  var steps = 0.0;
  for (var i = 0; i < 160; i++) {
    if (i >= i32(u.steps) || t >= dist) { break; }
    let h = sdfAt(p + dir * t);                      // distance to the nearest occluder from here
    let w = max(size * t / dist, 0.6);               // width of the light's "cone" at this point
    vis = min(vis, clamp(0.5 + 0.5 * h / w, 0.0, 1.0)); // closest miss -> penumbra
    steps += 1.0;
    if (vis < 0.003) { break; }
    t += max(abs(h), 1.0);                           // safe step: nothing is closer than h
  }
  return vec2f(vis * vis * (3.0 - 2.0 * vis), steps);
}`;

const LIGHT_WGSL = /* wgsl */ `
${SHAPE_WGSL}
fn sdfAt(p: vec2f) -> f32 { return TEX(sdf, p / u.resolution).r; }
${MARCH_WGSL}

fn floorAlbedo(px: vec2f) -> vec3f {
  let ex = i32(u.example + 0.5);
  let k = u.resolution.y / 700.0;
  if (ex == 0) {
    // drafting paper
    let g = fract(px / (32.0 * k));
    let g2 = fract(px / (160.0 * k));
    let line = max(step(min(g.x, g.y), 0.04), 2.0 * step(min(g2.x, g2.y), 0.008));
    return vec3f(0.42, 0.44, 0.48) * (1.0 - 0.18 * clamp(line, 0.0, 1.0));
  }
  if (ex == 1) {
    // warehouse concrete with seams and painted lines
    let n = valueNoise(px / (14.0 * k)) * 0.5 + valueNoise(px / (60.0 * k)) * 0.5;
    var c = vec3f(0.40, 0.41, 0.44) * (0.8 + 0.35 * n);
    let s = fract(px / (128.0 * k));
    c *= 1.0 - 0.25 * step(min(s.x, s.y), 0.012);
    return c;
  }
  // marble checker
  let q = floor(px / (72.0 * k));
  let chk = fmod(q.x + q.y, 2.0);
  let n = fbm(px / (90.0 * k), 3) * 0.5 + 0.5;
  return mix(vec3f(0.5, 0.48, 0.46), vec3f(0.3, 0.3, 0.33), chk) * (0.85 + 0.25 * n);
}

fn heat(t: f32) -> vec3f {
  return clamp(vec3f(1.5 * t - 0.2, 1.6 * t * (1.4 - t), 1.0 - 1.8 * t), vec3f(0.0), vec3f(1.0));
}

fn shade(uv: vec2f, fragPx: vec2f) -> vec4f {
  let px = fragPx * u.pxScale;     // render target may be smaller than the canvas (test mode)
  let d0 = sdfAt(px);
  let view = i32(u.view + 0.5);
  let jitter = ign(px + vec2f(fract(u.time * 7.0) * 61.0));
  if (view == 1) {
    // distance field: blue outside, orange inside, iso-lines every 24px
    var c = select(vec3f(0.35, 0.6, 0.95), vec3f(0.95, 0.55, 0.25), d0 < 0.0);
    c *= 1.0 - exp(-abs(d0) / (u.resolution.y * 0.08));
    c *= 0.8 + 0.2 * cos(d0 * 0.26);
    c = mix(c, vec3f(1.0), 1.0 - smoothstep(0.0, 2.0, abs(d0)));
    return vec4f(c, 1.0);
  }
  var lightF = u.ambient;             // floor: shadowed
  var lightR = u.ambient;             // occluder roofs: lit but never shadowed
  var cost = 0.0;
  let cover = clamp(0.5 - d0 / u.pxScale, 0.0, 1.0);   // anti-aliased occluder edge
  for (var k = 0; k < ${MAX_LIGHTS}; k++) {
    if (k >= i32(u.nLights)) { break; }
    let A = u.L[k * 3];       // xy = position, z = range, w = light size (radius)
    let B = u.L[k * 3 + 1];   // rgb = color (linear), a = intensity
    let C = u.L[k * 3 + 2];   // xy = cone direction, z = cos(outer), w = cos(inner)
    let toL = A.xy - px;
    let dist = length(toL);
    if (dist >= A.z) { continue; }
    let x = dist / A.z;
    var att = (1.0 - x * x) * (1.0 - x * x) * B.a;
    if (C.z > -1.5) {
      let cd = dot(-toL / max(dist, 0.001), C.xy);
      att *= smoothstep(C.z, C.w, cd);
    }
    if (att < 0.002) { continue; }
    var size = A.w;
    if (u.split > 0.5 && px.x < u.splitX) { size = 0.0; }
    lightR += B.rgb * att * 0.4;
    if (cover < 0.999) {
      let r = shadowRay(px, A.xy, size, jitter);
      lightF += B.rgb * att * r.x;
      if (k == 0) { cost = r.y; }
    }
  }
  if (view == 3) {
    return vec4f(mix(heat(cost / 48.0), vec3f(0.0), cover * 0.7), 1.0);
  }
  var floorC = floorAlbedo(px) * (1.0 - 0.35 * exp(-max(d0, 0.0) / (u.resolution.y * 0.012)));  // contact darkening
  let idx = i32(LOAD(sdf, vec2i(px / u.scale)).g + 0.5);
  var roofC = shapes[idx].c.rgb * (0.75 + 0.25 * smoothstep(-u.resolution.y * 0.03, 0.0, d0));
  roofC += vec3f(0.12) * (1.0 - smoothstep(0.0, 2.5, -d0));  // bevel rim
  if (view == 2) { floorC = vec3f(0.6); roofC = vec3f(0.6); }
  let hdr = mix(floorC * lightF, roofC * lightR, cover);
  var col = linearToSrgb(tonemapACES(hdr * 1.1));
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}`;

// ------------------------------------------------------------------------------- CPU geometry

function segHitsShape(ax, ay, bx, by, s) {
  if (s.type === 1 || s.type === 2) {
    // circle / capsule: distance from segment to center(s)
    const r = s.r;
    if (s.type === 1) return segPointDist(ax, ay, bx, by, s.x, s.y) < r;
    return segSegDist(ax, ay, bx, by, s.x, s.y, s.x2, s.y2) < r;
  }
  // box: transform segment into box space, slab test
  const c = Math.cos(s.rot);
  const sn = Math.sin(s.rot);
  const tx = (x, y) => [c * (x - s.x) + sn * (y - s.y), -sn * (x - s.x) + c * (y - s.y)];
  const [px, py] = tx(ax, ay);
  const [qx, qy] = tx(bx, by);
  let t0 = 0;
  let t1 = 1;
  const dx = qx - px;
  const dy = qy - py;
  for (const [p, d, h] of [[px, dx, s.hw], [py, dy, s.hh]]) {
    if (Math.abs(d) < 1e-9) {
      if (p < -h || p > h) return false;
    } else {
      let a = (-h - p) / d;
      let b = (h - p) / d;
      if (a > b) [a, b] = [b, a];
      t0 = Math.max(t0, a);
      t1 = Math.min(t1, b);
      if (t0 > t1) return false;
    }
  }
  return true;
}
function segPointDist(ax, ay, bx, by, px, py) {
  const dx = bx - ax;
  const dy = by - ay;
  const h = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(px - ax - dx * h, py - ay - dy * h);
}
function segSegDist(ax, ay, bx, by, cx, cy, dx, dy) {
  const cross = (x1, y1, x2, y2) => x1 * y2 - y1 * x2;
  const d1 = cross(bx - ax, by - ay, cx - ax, cy - ay);
  const d2 = cross(bx - ax, by - ay, dx - ax, dy - ay);
  const d3 = cross(dx - cx, dy - cy, ax - cx, ay - cy);
  const d4 = cross(dx - cx, dy - cy, bx - cx, by - cy);
  if (d1 * d2 < 0 && d3 * d4 < 0) return 0;
  return Math.min(segPointDist(ax, ay, bx, by, cx, cy), segPointDist(ax, ay, bx, by, dx, dy), segPointDist(cx, cy, dx, dy, ax, ay), segPointDist(cx, cy, dx, dy, bx, by));
}

export default {
  interaction: 'Move the mouse: it is the light. On Stealth, you are the thief — avoid the cones.',
  examples: [
    {
      id: 'hardsoft',
      label: 'Hard vs soft shadows',
      kind: 'Abstract',
      note: 'Left of the line: a <b>point light</b> (size 0) casts razor-sharp shadows. Right: a light with a <b>size</b> casts a <b>penumbra</b> — soft near the tips, sharp where the shadow touches its occluder, just like real contact shadows.',
      params: { size: 0.35, range: 1.2, ambient: 0.14, split: true, view: 'final' },
    },
    {
      id: 'stealth',
      label: 'Stealth vision cones',
      kind: 'In a game',
      note: 'Guards see with a cone-shaped light that walls block. The same rule runs twice: on the GPU to <i>draw</i> the cones, and on the CPU (a segment-vs-shapes test) to decide if a guard <b>spots you</b>. Hide in the shadows!',
      params: { size: 0.08, range: 1, ambient: 0.16, split: false, view: 'final' },
    },
    {
      id: 'colored',
      label: 'Multiple colored lights',
      kind: 'Real life',
      note: 'Three colored lights orbit a hall of pillars. Where one light is blocked but the others are not, the shadow takes the <b>remaining</b> colors — the “colored shadows” you see on a theater stage. Each light marches its own rays.',
      params: { size: 0.25, range: 1.0, ambient: 0.04, split: false, view: 'final' },
    },
  ],
  controls: [
    { type: 'heading', label: 'Shadows' },
    { type: 'slider', key: 'size', label: 'Light size (softness)', min: 0, max: 1, step: 0.01, value: 0.35, help: 'Radius of the light source. 0 = hard shadows; bigger = wider penumbra.' },
    { type: 'slider', key: 'range', label: 'Light range', min: 0.3, max: 2.5, step: 0.01, value: 1.4, help: 'How far light reaches. Also bounds how far each shadow ray walks.' },
    { type: 'slider', key: 'steps', label: 'March steps (quality)', min: 4, max: 128, step: 1, value: 48, help: 'Max steps per ray. Too few = light leaks through thin walls far away.' },
    { type: 'slider', key: 'ambient', label: 'Ambient light', min: 0, max: 0.4, step: 0.005, value: 0.05, help: 'Light that is everywhere, even in shadow.' },
    { type: 'toggle', key: 'split', label: 'Split view: hard | soft', value: true, showFor: ['hardsoft'], help: 'Left half ignores the light size.' },
    { type: 'slider', key: 'cone', label: 'Guard field of view', min: 20, max: 150, step: 1, value: 70, showFor: ['stealth'], format: (v) => `${Math.round(v)}°`, help: 'Width of the vision cones.' },
    { type: 'toggle', key: 'animate', label: 'Animate occluders', value: true, showFor: ['hardsoft', 'colored'] },
    {
      type: 'select',
      key: 'view',
      label: 'View',
      value: 'final',
      options: [
        { value: 'final', label: 'Final image' },
        { value: 'sdf', label: 'Distance field (baked)' },
        { value: 'mask', label: 'Light & shadow only' },
        { value: 'cost', label: 'Ray-march steps (cost)' },
      ],
      help: 'See the distance field the rays walk through, and how many steps they take.',
    },
  ],
  about: {
    summary:
      'Shadows in 2D are a visibility question: “can this pixel see the light?”. Here the GPU answers it for every pixel and every light by walking through a distance field.',
    what: `<p>Moving occluders lit by one or more lights. Every pixel shoots a ray toward each light; if anything is in the way it is in shadow.
      Lights with a <b>size</b> produce soft edges (penumbrae). The stealth tab reuses the same idea for guards’ line of sight.</p>`,
    how: `<ol>
      <li><b>Bake a distance field</b>: a compute shader computes, for every texel of a half-resolution texture, the distance to the nearest occluder (negative inside). Shapes are analytic SDFs (boxes, circles, capsules); for arbitrary sprites you would build it with <a href="#/s/jump-flood">jump flooding</a>.</li>
      <li><b>Sphere tracing</b>: from a pixel, step toward the light. The SDF value <code>h</code> says “nothing is closer than h”, so you can safely jump h pixels. Open space is crossed in a few big steps; you only slow down near edges.</li>
      <li><b>Hit test</b>: if <code>h</code> drops to ~0 before reaching the light, the pixel is in shadow.</li>
      <li><b>Soft shadows for free</b>: a light of radius R seen from the pixel is a cone. At distance <code>t</code> along the ray that cone is <code>w = R·t/dist</code> wide. The smallest ratio <code>h / w</code> along the ray tells how much of the light disk is covered → a smooth 0..1 visibility. Occluders near the receiver give sharp edges, far ones blur — physically right.</li>
      <li><b>Several lights</b> = several rays per pixel, results added together (colored shadows appear naturally).</li>
      <li><b>Gameplay</b> (stealth): the GPU result is just pixels, so the guard’s “can I see the player?” check is done again on the CPU with one segment-vs-shape test per occluder.</li>
    </ol>
    <p>Other common 2D shadow techniques: <b>shadow geometry</b> (extrude each occluder edge away from the light into a quad and draw it black — exact hard shadows, cheap), <b>1D polar shadow maps</b> (render occluder distance per angle into a 1-pixel-tall texture, like a 3D shadow map), and <b>visibility polygons</b> (CPU ray casts to wall corners — great for line-of-sight).</p>`,
    uses: [
      { title: 'Stealth & horror', text: 'Mark of the Ninja, Monaco and Teleglitch hide what the player can’t see and let shadows become gameplay.' },
      { title: 'Top-down atmosphere', text: 'Flashlights and lamps that cast dynamic shadows from crates, pillars and characters.' },
      { title: 'Puzzle mechanics', text: 'Light beams that must reach a sensor; shadows as platforms (Shadow puppeteer-style games).' },
      { title: 'Soft lighting', text: 'Big area lights (windows, sky, magic orbs) with realistic penumbrae.' },
    ],
    try: [
      'On <b>Hard vs soft</b>, put the light close to a box: the penumbra is narrow at the box and widens with distance.',
      'Raise <i>Light size</i> to 1 — huge area light, very soft shadows. Drop it to 0: hard shadows, identical on both halves.',
      'Set <i>March steps</i> to 6 and watch light leak through walls far from the light: the rays give up early.',
      'View → <i>Ray-march steps</i>: open space is cheap (blue), rays that skim edges are expensive (red).',
      'On <b>Stealth</b>, sneak behind crates: the guard only reacts when the CPU line-of-sight test agrees with what you see.',
    ],
    ask: [
      'soft 2D shadows by ray marching a distance field',
      'hard 2D shadows from a mouse-controlled light',
      'guard vision cones blocked by walls, with detection',
      'colored lights with colored shadows in a top-down game',
      'a line-of-sight check that matches the rendered shadows',
    ],
    perf: `<p>Cost ≈ <i>pixels × lights × steps</i>. Sphere tracing keeps steps low in open space (often &lt; 10), but rays that graze edges use the
      full budget. Baking the SDF is cheap (pixels/4 × shapes) and only needs redoing when occluders move. To scale up, compute
      lighting at half resolution, limit light ranges, or cache per-light shadow maps (1D polar maps cost only 1 row per light).</p>`,
    api: `<p>The ray-march itself is a plain fragment shader (works in WebGL2). WebGPU makes the <b>SDF bake</b> a natural compute shader
      writing a storage texture from a storage buffer of shapes; in WebGL2 you would render it with a fragment pass reading shapes from a
      uniform array or a data texture.</p>`,
    code: [
      { title: 'Soft shadow by sphere tracing (fragment)', lang: 'wgsl', src: MARCH_WGSL.trim() },
      {
        title: 'Bake the distance field (compute)',
        lang: 'wgsl',
        src: `@compute @workgroup_size(8, 8)
fn bake(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(sdfOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = (vec2f(id.xy) + 0.5) * u.scale;      // texel -> screen pixels
  var d = 1e5;
  for (var i: u32 = 0u; i < u.count; i++) {
    d = min(d, sdShape(p, shapes[i]));         // box / circle / capsule SDFs
  }
  textureStore(sdfOut, vec2i(id.xy), vec4f(d, 0.0, 0.0, 1.0));
}`,
      },
      {
        title: 'Guard line of sight (CPU, for gameplay)',
        lang: 'js',
        src: `const toPlayer = Math.hypot(px - g.x, py - g.y);
const inCone = Math.cos(angleBetween(g.dir, player)) > Math.cos(fov / 2);
const blocked = shapes.some((s) => segHitsShape(g.x, g.y, px, py, s));
g.alert = toPlayer < range && inCone && !blocked;`,
      },
    ],
    links: [
      { title: 'Inigo Quilez — soft shadows in raymarched SDFs', url: 'https://iquilezles.org/articles/rmshadows/' },
      { title: 'Red Blob Games — 2D visibility', url: 'https://www.redblobgames.com/articles/visibility/' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const cam = new Camera2D();
    const shapesOverlay = new ShapeBatch(gpu);
    const U = gpu.uniforms(
      {
        resolution: 'vec2f',
        scale: 'f32',
        count: 'u32',
        ambient: 'vec3f',
        nLights: 'f32',
        steps: 'f32',
        view: 'f32',
        split: 'f32',
        splitX: 'f32',
        example: 'f32',
        time: 'f32',
        pxScale: 'f32',
        L: `array<vec4f, ${MAX_LIGHTS * 3}>`,
      },
      'ShadowU',
    );
    const shapeData = new Float32Array(MAX_SHAPES * 12);
    const shapeBuf = gpu.storage(shapeData.byteLength, 'occluders');
    const bake = gpu.compute({
      label: 'sdf-bake',
      bindings: {
        u: { uniform: U },
        shapes: { storage: 'array<Shape>', access: 'read' },
        sdfOut: { storageTexture: 'rgba16float', access: 'write' },
      },
      include: ['sdf'],
      code: BAKE_WGSL,
    });
    const light = gpu.fullscreen({
      label: 'shadow-lighting',
      uniforms: U,
      textures: ['sdf'],
      storage: { shapes: 'array<Shape>' },
      include: ['sdf', 'color', 'noise'],
      code: LIGHT_WGSL,
    });

    let sdfT = null;
    let sdfScale = 2;
    let lowT = null; // test mode: light at half resolution, then upscale
    const alloc = () => {
      sdfScale = ctx.testMode ? 3 : 2;
      sdfT?.destroy();
      sdfT = gpu.target(Math.ceil(ctx.width / sdfScale), Math.ceil(ctx.height / sdfScale), { format: 'rgba16float', label: 'sdf' });
      lowT?.destroy();
      lowT = ctx.testMode ? gpu.target(Math.ceil(ctx.width / 2), Math.ceil(ctx.height / 2), { format: gpu.format, label: 'lowres' }) : null;
    };
    alloc();

    const readout = tag(ctx, 'right:8px;bottom:8px');
    const labelL = tag(ctx, 'left:12px;top:40px;display:none');
    const labelR = tag(ctx, 'right:12px;top:40px;display:none');
    labelL.textContent = 'HARD · point light';
    labelR.textContent = 'SOFT · area light';

    // ------------------------------------------------------------------ world
    let shapes = [];
    let lightPos = [ctx.width * 0.5, ctx.height * 0.5];
    let player = { x: 0, y: 0, init: false };
    let guards = [];
    let builtFor = '';

    const box = (x, y, hw, hh, rot, color) => ({ type: 0, x, y, hw, hh, rot, color });
    const circ = (x, y, r, color) => ({ type: 1, x, y, r, color });
    const caps = (x, y, x2, y2, r, color) => ({ type: 2, x, y, x2, y2, r, color });

    function build(ex, W, H) {
      builtFor = `${ex}|${W}x${H}`;
      shapes = [];
      guards = [];
      if (ex === 'hardsoft') {
        const c1 = hexLinear('#5b7bd6');
        const c2 = hexLinear('#d6705b');
        const c3 = hexLinear('#e0c060');
        shapes.push({ ...box(W * 0.22, H * 0.3, H * 0.07, H * 0.07, 0.3, c1), spin: 0.25 });
        shapes.push({ ...box(W * 0.78, H * 0.68, H * 0.09, H * 0.045, -0.2, c2), spin: -0.18 });
        shapes.push(circ(W * 0.3, H * 0.72, H * 0.06, c3));
        shapes.push(circ(W * 0.72, H * 0.27, H * 0.045, c1));
        shapes.push({ ...caps(0, 0, 0, 0, H * 0.018, c2), bar: [W * 0.5, H * 0.2, H * 0.13] });
        for (let i = 0; i < 5; i++) shapes.push(circ(W * (0.36 + i * 0.07), H * 0.84, H * 0.016, c3));
        for (let i = 0; i < 4; i++) shapes.push(box(W * 0.9, H * (0.18 + i * 0.1), H * 0.012, H * 0.03, 0, c1));
      } else if (ex === 'stealth') {
        const wall = hexLinear('#8a93aa');
        const crate = hexLinear('#b08850');
        const t = H * 0.016; // wall half-thickness
        // outer walls
        shapes.push(box(W / 2, t, W / 2, t, 0, wall), box(W / 2, H - t, W / 2, t, 0, wall), box(t, H / 2, t, H / 2, 0, wall), box(W - t, H / 2, t, H / 2, 0, wall));
        // interior walls with doorways
        shapes.push(box(W * 0.35, H * 0.22, t, H * 0.22, 0, wall));
        shapes.push(box(W * 0.35, H * 0.82, t, H * 0.18, 0, wall));
        shapes.push(box(W * 0.68, H * 0.5, W * 0.1, t, 0, wall));
        shapes.push(box(W * 0.68, H * 0.32, t, H * 0.18, 0, wall));
        // crates & pillars
        const cs = H * 0.04;
        for (const [x, y, r] of [[0.15, 0.3, 0.1], [0.19, 0.38, -0.05], [0.12, 0.72, 0.4], [0.5, 0.35, 0.0], [0.52, 0.65, 0.2], [0.85, 0.75, -0.3], [0.88, 0.2, 0.15], [0.6, 0.85, 0.0]])
          shapes.push(box(W * x, H * y, cs, cs, r, crate));
        shapes.push(circ(W * 0.5, H * 0.5, H * 0.035, wall), circ(W * 0.85, H * 0.45, H * 0.03, wall));
        const path = (pts) => pts.map(([x, y]) => [x * W, y * H]);
        guards = [
          { path: path([[0.12, 0.12], [0.28, 0.12], [0.28, 0.55], [0.12, 0.55]]), speed: 0.09, seg: 0, k: 0 },
          { path: path([[0.45, 0.15], [0.6, 0.15], [0.6, 0.6], [0.45, 0.6]]), speed: 0.07, seg: 1, k: 0.3 },
          { path: path([[0.75, 0.88], [0.92, 0.88], [0.92, 0.6], [0.75, 0.6]]), speed: 0.08, seg: 2, k: 0.5 },
        ].map((g) => ({ ...g, x: g.path[g.seg][0], y: g.path[g.seg][1], dir: 0, alert: 0, look: 0 }));
      } else {
        const stone = hexLinear('#8c8c96');
        for (let i = 0; i < 4; i++) {
          shapes.push(circ(W * (0.2 + i * 0.2), H * 0.28, H * 0.04, stone));
          shapes.push(circ(W * (0.2 + i * 0.2), H * 0.72, H * 0.04, stone));
        }
        shapes.push({ ...box(W * 0.5, H * 0.5, H * 0.06, H * 0.06, 0.785, hexLinear('#c8b070')), spin: 0.15 });
        shapes.push(box(W * 0.06, H * 0.5, H * 0.02, H * 0.12, 0, stone), box(W * 0.94, H * 0.5, H * 0.02, H * 0.12, 0, stone));
      }
    }

    function uploadShapes() {
      const n = Math.min(shapes.length, MAX_SHAPES);
      for (let i = 0; i < n; i++) {
        const s = shapes[i];
        const o = i * 12;
        if (s.type === 0) shapeData.set([s.x, s.y, s.hw, s.hh, s.rot, 0, 0, 0], o);
        else if (s.type === 1) shapeData.set([s.x, s.y, s.r, 0, 0, 1, 0, 0], o);
        else shapeData.set([s.x, s.y, s.r, 0, 0, 2, s.x2, s.y2], o);
        shapeData.set([s.color[0], s.color[1], s.color[2], 1], o + 8);
      }
      gpu.queue.writeBuffer(shapeBuf, 0, shapeData, 0, Math.max(12, n * 12));
      return n;
    }

    const visible = (ax, ay, bx, by) => !shapes.some((s) => segHitsShape(ax, ay, bx, by, s));

    return {
      resize() {
        alloc();
        builtFor = '';
      },
      onExample() {
        builtFor = '';
        player.init = false;
      },
      onAction(key) {
        if (key === 'reset') builtFor = '';
      },
      frame(ctx) {
        const W = ctx.width;
        const H = ctx.height;
        const p = ctx.params;
        const ex = ctx.example;
        const t = ctx.time;
        const dt = ctx.dt;
        if (builtFor !== `${ex}|${W}x${H}`) build(ex, W, H);
        cam.setViewport(W, H);
        cam.x = W / 2;
        cam.y = H / 2;

        // animate occluders
        const anim = p.animate !== false && ex !== 'stealth';
        for (const s of shapes) {
          if (s.spin && anim) s.rot += s.spin * dt;
          if (s.bar) {
            const a = anim ? t * 0.35 : 0.6;
            const [bx, by, len] = s.bar;
            s.x = bx - Math.cos(a) * len;
            s.y = by - Math.sin(a) * len * 0.5;
            s.x2 = bx + Math.cos(a) * len;
            s.y2 = by + Math.sin(a) * len * 0.5;
          }
        }

        // lights
        const lights = [];
        const ptr = ctx.pointer;
        const sizePx = p.size * H * 0.15;
        const omni = [0, 0, -2, -2];
        if (ex === 'hardsoft') {
          const target = ptr.over ? [ptr.x, ptr.y] : [W * 0.5 + Math.cos(t * 0.5) * W * 0.25, H * 0.5 + Math.sin(t * 0.7) * H * 0.2];
          const k = damp(14, dt || 0.016);
          lightPos = [lightPos[0] + (target[0] - lightPos[0]) * k, lightPos[1] + (target[1] - lightPos[1]) * k];
          lights.push([lightPos[0], lightPos[1], H * p.range, sizePx, ...hexLinear('#fff1d6'), 1.1, ...omni]);
        } else if (ex === 'colored') {
          const cols = ['#ff3b3b', '#3bff5a', '#3b6bff'];
          cols.forEach((c, i) => {
            const a = t * 0.3 + (i * Math.PI * 2) / 3;
            lights.push([W / 2 + Math.cos(a) * W * 0.3, H / 2 + Math.sin(a) * H * 0.3, H * p.range, sizePx, ...hexLinear(c), 1.0, ...omni]);
          });
          if (ptr.over) lights.push([ptr.x, ptr.y, H * p.range * 0.6, sizePx, ...hexLinear('#fff4e8'), 0.6, ...omni]);
        } else {
          // stealth: move the player, patrol the guards, test line of sight
          if (!player.init) {
            player = { x: W * 0.12, y: H * 0.9, init: true };
          }
          if (ptr.over) {
            const k = damp(8, dt || 0.016);
            player.x += (ptr.x - player.x) * k;
            player.y += (ptr.y - player.y) * k;
          }
          const fov = (p.cone * Math.PI) / 180;
          const range = H * 0.42 * p.range;
          for (const g of guards) {
            if (dt > 0 && g.alert < 0.5) {
              const [tx, ty] = g.path[(g.seg + 1) % g.path.length];
              const dx = tx - g.x;
              const dy = ty - g.y;
              const d = Math.hypot(dx, dy);
              const step = g.speed * H * dt;
              if (d <= step) {
                g.seg = (g.seg + 1) % g.path.length;
                g.x = tx;
                g.y = ty;
              } else {
                g.x += (dx / d) * step;
                g.y += (dy / d) * step;
              }
              let want = Math.atan2(dy, dx) + Math.sin(t * 1.3 + g.k * 9) * 0.35;
              let diff = ((want - g.dir + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
              g.dir += diff * damp(4, dt);
            }
            const dx = player.x - g.x;
            const dy = player.y - g.y;
            const dist = Math.hypot(dx, dy);
            const ang = Math.acos(Math.max(-1, Math.min(1, (dx * Math.cos(g.dir) + dy * Math.sin(g.dir)) / (dist || 1))));
            const sees = dist < range && ang < fov / 2 && visible(g.x, g.y, player.x, player.y);
            if (sees && dt > 0) {
              const want = Math.atan2(dy, dx);
              let diff = ((want - g.dir + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
              g.dir += diff * damp(6, dt);
            }
            g.alert += ((sees ? 1 : 0) - g.alert) * damp(sees ? 10 : 2, dt || 0.016);
            const col = g.alert > 0.5 ? hexLinear('#ff4040') : hexLinear('#ffe7a0');
            const outer = Math.cos(fov / 2);
            const inner = Math.cos(Math.max(0, fov / 2 - 0.12));
            lights.push([g.x, g.y, range, sizePx, ...col, 2.2, Math.cos(g.dir), Math.sin(g.dir), outer, inner]);
          }
          // a dim ceiling lamp for mood
          lights.push([W * 0.5, H * 0.5, H * 0.7 * p.range, H * 0.03, ...hexLinear('#7088ff'), 0.5, ...omni]);
        }

        const n = uploadShapes();
        const L = new Float32Array(MAX_LIGHTS * 12);
        lights.slice(0, MAX_LIGHTS).forEach((l, i) => L.set(l, i * 12));
        const amb = p.ambient * p.ambient * 2.5;
        const ambTint = ex === 'stealth' ? [0.55, 0.65, 1] : ex === 'colored' ? [0.8, 0.8, 1] : [1, 1, 1];
        U.setAll({
          resolution: [W, H],
          scale: sdfScale,
          count: n,
          ambient: ambTint.map((v) => v * amb),
          nLights: Math.min(lights.length, MAX_LIGHTS),
          steps: ctx.testMode ? Math.min(p.steps, 32) : p.steps,
          view: { final: 0, sdf: 1, mask: 2, cost: 3 }[p.view] ?? 0,
          split: ex === 'hardsoft' && p.split ? 1 : 0,
          splitX: W / 2,
          example: ctx.exampleIndex,
          time: t,
          pxScale: lowT ? ctx.width / lowT.width : 1,
          L,
        });
        U.upload();
        const enc = ctx.encoder;
        bake.dispatch(enc, 'bake', [Math.ceil(sdfT.width / 8), Math.ceil(sdfT.height / 8)], { u: U, shapes: shapeBuf, sdfOut: sdfT });
        const canvasT = { view: ctx.target, format: gpu.format };
        if (lowT) {
          light.draw(enc, lowT, { sdf: sdfT, shapes: shapeBuf });
          gpu.blit(enc, lowT, canvasT);
        } else light.draw(enc, canvasT, { sdf: sdfT, shapes: shapeBuf });

        // overlay: light markers, guards, player, split line
        const o = shapesOverlay.begin();
        const lw = Math.max(1, H * 0.003);
        if (ex === 'hardsoft' && p.split) o.line(W / 2, 0, W / 2, H, lw, [1, 1, 1, 0.35]);
        if (ex !== 'stealth') {
          for (const l of lights) {
            const c = [Math.pow(l[4], 1 / 2.2), Math.pow(l[5], 1 / 2.2), Math.pow(l[6], 1 / 2.2), 1];
            o.circle(l[0], l[1], Math.max(H * 0.008, l[3]), c, { glow: H * 0.02, glowStrength: 0.7 });
            o.circle(l[0], l[1], Math.max(H * 0.004, l[3] * 0.6), [1, 1, 1, 0.9]);
          }
        } else {
          let seen = false;
          for (const g of guards) {
            const al = g.alert;
            seen = seen || al > 0.5;
            const r = H * 0.022;
            o.circle(g.x, g.y, r, al > 0.5 ? '#d83a3a' : '#3a4a6a', { stroke: 0 });
            o.circle(g.x, g.y, r, '#0b0e16', { stroke: lw * 1.2 });
            o.circle(g.x + Math.cos(g.dir) * r * 0.55, g.y + Math.sin(g.dir) * r * 0.55, r * 0.32, '#f2e3c2');
            if (al > 0.5) {
              o.rect(g.x - r * 0.18, g.y - r * 3.1, r * 0.36, r * 1.2, '#ff4848', { radius: r * 0.1 });
              o.circle(g.x, g.y - r * 1.6, r * 0.2, '#ff4848');
            }
          }
          const pr = H * 0.018;
          o.circle(player.x, player.y, pr * 1.6, seen ? [1, 0.2, 0.2, 0.35] : [0.3, 1, 0.5, 0.25], { glow: pr, glowStrength: 0.5 });
          o.circle(player.x, player.y, pr, seen ? '#ff6a6a' : '#5dff8a');
          readout.innerHTML = seen ? '<span style="color:#ff6a6a">⚠ SPOTTED!</span> line of sight confirmed on the CPU' : '<span style="color:#5dff8a">hidden</span> · no guard has line of sight';
        }
        shapesOverlay.flush(enc, canvasT, cam, { blend: 'alpha' });

        labelL.style.display = labelR.style.display = ex === 'hardsoft' && p.split ? '' : 'none';
        if (ex !== 'stealth')
          readout.textContent = `${n} occluders · ${lights.length} light${lights.length > 1 ? 's' : ''} · SDF ${sdfT.width}×${sdfT.height} · ≤${Math.round(p.steps)} steps/ray`;
      },
    };
  },
};

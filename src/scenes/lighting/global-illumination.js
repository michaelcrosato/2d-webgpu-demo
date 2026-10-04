// 2D global illumination: emissive & solid materials -> jump-flood distance field -> per-pixel ray marching
// (N rays, rotated by noise every frame) -> temporal accumulation (+ infinite bounces through the history).
//
//   compose (compute)  : preset scene + your paint -> material texture + JFA seeds        (GI resolution)
//   jfa ×log2(n) (cs)  : every texel learns the coordinates of its nearest solid texel
//   gi (compute)       : march N rays per texel through the distance field, average what they hit
//   composite (frag)   : full-res materials, radiance upsampled, walls lit from the gathered light

import { tag } from './_shared.js';

const SCENE_WGSL = /* wgsl */ `
// material: rgb = emission (lights, HDR) or albedo (walls); a = 0 empty, 1 light, 2 wall
fn mLight(c: vec3f) -> vec4f { return vec4f(c, 1.0); }
fn mWall(c: vec3f) -> vec4f { return vec4f(c, 2.0); }

fn presetMat(px: vec2f) -> vec4f {
  let H = u.resolution.y;
  let A = u.resolution.x / H;
  let q = px / H;                         // x in 0..aspect, y in 0..1 (down)
  let t = u.time;
  let ex = i32(u.example + 0.5);
  var m = vec4f(0.0);
  if (ex == 0) {
    // a simple room to start painting in
    if (sdBox(q - vec2f(0.5 * A, 0.16), vec2f(0.24, 0.014)) < 0.0) { m = mWall(vec3f(0.85)); }
    if (sdBox(q - vec2f(0.3 * A, 0.62), vec2f(0.016, 0.2)) < 0.0) { m = mWall(vec3f(0.95, 0.12, 0.08)); }
    if (sdBox(q - vec2f(0.68 * A, 0.64), vec2f(0.13, 0.016)) < 0.0) { m = mWall(vec3f(0.1, 0.35, 0.95)); }
    if (length(q - vec2f(0.8 * A, 0.32)) < 0.06) { m = mWall(vec3f(0.85)); }
    if (sdBox(q - vec2f(0.86 * A, 0.8), vec2f(0.03, 0.03)) < 0.0) { m = mWall(vec3f(0.2, 0.9, 0.3)); }
    if (length(q - vec2f(0.13 * A, 0.3)) < 0.04) { m = mLight(vec3f(7.0, 4.2, 1.8)); }
    if (sdBox(q - vec2f(0.55 * A, 0.86), vec2f(0.08, 0.014)) < 0.0) { m = mLight(vec3f(0.5, 2.6, 3.6)); }
    return m;
  }
  if (ex == 1) {
    // lava cave: noisy rock, a glowing lava river, crystals hanging from the ceiling
    let n = fbm(q * vec2f(2.4, 3.0) + vec2f(1.7, 4.2), 4) * 0.5 + 0.5;
    let ceiling = smoothstep(0.42, 0.08, q.y);
    let sides = pow(abs(q.x / A - 0.5) * 2.0, 5.0) * 0.6;
    let pillars = smoothstep(0.55, 0.9, valueNoise(vec2f(q.x * 5.0, 0.5))) * smoothstep(0.35, 0.75, q.y) * 0.5;
    let rock = n * 0.55 + ceiling * 0.7 + sides + pillars;
    var solid = rock > 0.72;
    // stalactites, stalagmites and a rock ledge: occluders that throw long shadows up from the lava
    for (var j = 0; j < 4; j++) {
      let fj = f32(j);
      let sx = (0.2 + 0.21 * fj) * A;
      let top = sdTriangle(q, vec2f(sx - 0.035, 0.2), vec2f(sx + 0.035, 0.2), vec2f(sx + 0.01, 0.42 + 0.05 * sin(fj * 3.1)));
      solid = solid || top < 0.0;
    }
    solid = solid || sdTriangle(q, vec2f(0.33 * A - 0.05, 0.9), vec2f(0.33 * A + 0.05, 0.9), vec2f(0.34 * A, 0.66)) < 0.0;
    solid = solid || sdTriangle(q, vec2f(0.78 * A - 0.06, 0.9), vec2f(0.78 * A + 0.04, 0.9), vec2f(0.77 * A, 0.7)) < 0.0;
    solid = solid || sdRoundBox(q - vec2f(0.55 * A, 0.6), vec2f(0.09, 0.022), 0.015) < 0.0;
    if (solid) { m = mWall(vec3f(0.42, 0.35, 0.31) * (0.75 + 0.5 * valueNoise(q * 50.0))); }
    for (var i = 0; i < 5; i++) {
      let fi = f32(i);
      let c = vec2f((0.1 + 0.21 * fi) * A, 0.27 + 0.04 * sin(fi * 2.3));
      let d = sdRhombus(rot2(0.35 * sin(fi * 1.7)) * (q - c), vec2f(0.016, 0.04));
      if (d < 0.0) {
        let hue = mix(vec3f(0.3, 2.0, 3.2), vec3f(2.4, 0.5, 3.2), fract(fi * 0.37 + 0.2));
        m = mLight(hue * (0.8 + 0.2 * sin(t * 1.5 + fi * 2.0)));
      }
    }
    let lavaY = 0.86 + 0.018 * sin(q.x * 9.0 + t * 0.7) + 0.01 * sin(q.x * 23.0 - t * 1.3);
    if (q.y > lavaY) {
      let f = fbm(vec2f(q.x * 7.0 - t * 0.35, q.y * 16.0 + t * 0.1), 3) * 0.5 + 0.5;
      let crust = smoothstep(0.45, 0.6, f);
      m = mLight(mix(vec3f(3.2, 0.9, 0.12), vec3f(0.5, 0.08, 0.02), crust) * (0.9 + 0.15 * sin(t * 2.0 + q.x * 4.0)));
    }
    return m;
  }
  // neon signs in a dark alley
  if (q.x < 0.06 * A || q.x > 0.94 * A || q.y < 0.05 || q.y > 0.95) { m = mWall(vec3f(0.22, 0.22, 0.27)); }
  if (sdBox(q - vec2f(0.5 * A, 0.42), vec2f(0.05, 0.09)) < 0.0) { m = mWall(vec3f(0.85, 0.25, 0.6)); }
  if (sdBox(q - vec2f(0.16 * A, 0.75), vec2f(0.04, 0.04)) < 0.0) { m = mWall(vec3f(0.6, 0.45, 0.3)); }
  if (sdBox(q - vec2f(0.2 * A, 0.8), vec2f(0.03, 0.03)) < 0.0) { m = mWall(vec3f(0.6, 0.45, 0.3)); }
  // someone walking through the alley (a moving occluder -> moving shadows)
  let wx = (0.5 + 0.33 * sin(t * 0.35)) * A;
  if (sdSegment(q, vec2f(wx, 0.6), vec2f(wx, 0.7)) < 0.022) { m = mWall(vec3f(0.12, 0.12, 0.14)); }
  let fl1 = step(0.12, valueNoise(vec2f(t * 4.0, 1.0)));
  let fl2 = 0.85 + 0.15 * sin(t * 30.0);
  let th = 0.0065;
  // open heart (gap at the top so its inside is not a closed box of light)
  let hp = (q - vec2f(0.27 * A, 0.22)) / 0.11;
  if (abs(sdHeart(hp) * 0.11) < th && abs(hp.x) > 0.12) { m = mLight(vec3f(3.2, 0.45, 1.7) * fl1); }
  // crescent moon (an arc)
  let mp = rot2(-0.6) * (q - vec2f(0.76 * A, 0.27));
  if (sdArc(mp, vec2f(sin(2.2), cos(2.2)), 0.075, th) < 0.0) { m = mLight(vec3f(0.35, 2.0, 2.8) * fl2); }
  // lightning bolt
  let b0 = vec2f(0.66 * A, 0.58);
  let b1 = b0 + vec2f(0.06, 0.08);
  let b2 = b1 + vec2f(-0.05, 0.02);
  let b3 = b2 + vec2f(0.07, 0.11);
  let bd = min(sdSegment(q, b0, b1), min(sdSegment(q, b1, b2), sdSegment(q, b2, b3)));
  if (bd < th) { m = mLight(vec3f(2.8, 2.0, 0.3)); }
  return m;
}

fn material(px: vec2f) -> vec4f {
  var m = presetMat(px);
  let pt = LOADP(vec2i(px));
  if (pt.a > 0.5) {
    if (pt.a > 2.5) { m = vec4f(0.0); } else { m = pt; }
  }
  return m;
}`;

const COMPOSE_WGSL = /* wgsl */ `
fn LOADP(p: vec2i) -> vec4f { return textureLoad(paint, clamp(p, vec2i(0), vec2i(textureDimensions(paint)) - vec2i(1)), 0); }
${SCENE_WGSL}
@compute @workgroup_size(8, 8)
fn compose(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2u(u.giSize);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let px = (vec2f(id.xy) + 0.5) / u.giScale;   // GI texel -> canvas pixel
  let m = material(px);
  textureStore(sceneOut, vec2i(id.xy), m);
  // jump-flood seeds: solid texels point at themselves, empty ones at "nothing"
  var seed = vec4f(-1.0, -1.0, 0.0, 0.0);
  if (m.a > 0.5) { seed = vec4f(vec2f(id.xy), 1.0, 0.0); }
  textureStore(seedOut, vec2i(id.xy), seed);
}`;

const JFA_WGSL = /* wgsl */ `
// one jump-flood pass: look at 9 texels 'jump' apart and keep the nearest seed any of them knows about
@compute @workgroup_size(8, 8)
fn jfa(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(src));
  let p = vec2i(id.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let k = i32(js.jump);
  var best = vec4f(-1.0, -1.0, 0.0, 0.0);
  var bestD = 1e9;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = p + vec2i(dx, dy) * k;
      if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) { continue; }
      let s = textureLoad(src, q, 0);
      if (s.z < 0.5) { continue; }
      let d = distance(s.xy, vec2f(p));
      if (d < bestD) { bestD = d; best = s; }
    }
  }
  textureStore(dst, p, best);
}`;

const GI_WGSL = /* wgsl */ `
fn hashU(p: vec2u, f: u32) -> f32 { return f32(pcg3d(vec3u(p, f)).x) / 4294967296.0; }

// march one ray through the jump-flooded distance field; returns the light it brings back
fn traceRay(p: vec2f, dir: vec2f, size: vec2f) -> vec3f {
  var t = 0.0;
  for (var i = 0; i < 96; i++) {
    if (i >= i32(u.steps)) { break; }
    let q = p + dir * t;
    if (q.x < 0.0 || q.y < 0.0 || q.x >= size.x || q.y >= size.y) { return vec3f(u.ambient); }
    let s = textureLoad(jfa, vec2i(q), 0);
    if (s.z < 0.5) { return vec3f(u.ambient); }        // no solid anywhere
    let d = length(s.xy + 0.5 - q);                    // distance to the nearest solid texel
    if (d < 1.0) {
      let m = textureLoad(scene, vec2i(s.xy), 0);      // what did we hit?
      if (m.a < 1.5) { return m.rgb; }                 // a light: its emission
      // a wall: reflect (albedo ×) the light that was gathered in front of it LAST frame -> bounces
      let back = textureSampleLevel(prev, samp, (q - dir * 1.5) / size, 0.0).rgb;
      return m.rgb * back * u.bounce;
    }
    t += max(d - 0.75, 0.75);                          // sphere-tracing step
  }
  return vec3f(0.0);
}

@compute @workgroup_size(8, 8)
fn gi(@builtin(global_invocation_id) id: vec3u) {
  let size = u.giSize;
  if (f32(id.x) >= size.x || f32(id.y) >= size.y) { return; }
  let p = vec2f(id.xy) + 0.5;
  let here = textureLoad(scene, vec2i(id.xy), 0);
  var raw = vec4f(0.0);
  if (here.a < 0.5) {
    // N rays evenly spread around the circle, rotated by per-pixel, per-frame noise (stratified)
    let n = i32(u.rays);
    let jitter = fract(hashU(id.xy, 0u) + u.frame * 0.61803398875);
    var sum = vec3f(0.0);
    for (var r = 0; r < 64; r++) {
      if (r >= n) { break; }
      let a = (f32(r) + jitter) / f32(n) * TAU;
      sum += traceRay(p, vec2f(cos(a), sin(a)), size);
    }
    raw = vec4f(sum / f32(n), 1.0);
  }
  // temporal accumulation: exponential moving average with last frame
  let old = textureLoad(prev, vec2i(id.xy), 0);
  let acc = mix(old, raw, u.alpha);
  textureStore(accOut, vec2i(id.xy), acc);
  textureStore(rawOut, vec2i(id.xy), raw);
}`;

const COMPOSITE_WGSL = /* wgsl */ `
fn LOADP(p: vec2i) -> vec4f { return LOAD(paint, p); }
${SCENE_WGSL}

fn irradianceNear(uv: vec2f) -> vec3f {
  // normalized blur: average only the EMPTY texels around a wall pixel (acc.a = 1 there, 0 in walls)
  let ts = 1.0 / u.giSize;
  var s = vec4f(0.0);
  for (var i = 0; i < 24; i++) {
    let ring = f32(i / 8);
    let a = f32(i) * 0.785398 + ring * 0.39;
    let r = 1.5 + ring * ring * 3.0 + ring * 1.5;     // 1.5, 6, 16.5 texels
    s += TEX(acc, uv + vec2f(cos(a), sin(a)) * ts * r) / (1.0 + ring);
  }
  return s.rgb / max(s.a, 0.02);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let view = i32(u.view + 0.5);
  let m = material(px);
  if (view == 1) {
    var c = vec3f(0.05);
    if (m.a > 1.5) { c = m.rgb * 0.8; }
    if (m.a > 0.5 && m.a < 1.5) { c = m.rgb / (vec3f(1.0) + m.rgb) * 1.4; }
    return vec4f(c, 1.0);
  }
  if (view == 2 || view == 3) {
    let s = LOAD(jfa, vec2i(uv * u.giSize));
    let d = length(s.xy + 0.5 - uv * u.giSize);
    if (view == 2) {
      let h = hash22(s.xy);
      var c = 0.25 + 0.6 * vec3f(h, fract(h.x + h.y));
      if (s.z < 0.5) { c = vec3f(0.0); }
      return vec4f(c * (0.75 + 0.25 * step(1.0, d)), 1.0);
    }
    let k = 1.0 - exp(-d / (u.giSize.y * 0.15));
    var c = vec3f(0.3, 0.55, 0.95) * k * (0.8 + 0.2 * cos(d * 0.8));
    c = mix(c, vec3f(1.0, 0.6, 0.3), step(d, 1.0));
    return vec4f(c, 1.0);
  }
  if (view == 4) { return vec4f(linearToSrgb(tonemapACES(TEX(raw, uv).rgb * u.exposure)), 1.0); }
  let rad = TEX(acc, uv);
  if (view == 5) { return vec4f(linearToSrgb(tonemapACES(rad.rgb / max(rad.a, 0.05) * u.exposure)), 1.0); }
  var hdr = vec3f(0.0);
  if (m.a > 0.5 && m.a < 1.5) {
    hdr = m.rgb;                                       // emitters show their own light
  } else if (m.a > 1.5) {
    hdr = m.rgb * (irradianceNear(uv) * 1.2 + vec3f(0.01));
  } else {
    let grain = 0.94 + 0.06 * valueNoise(px / (u.resolution.y * 0.006));
    hdr = rad.rgb / max(rad.a, 0.05) * grain;          // the floor shows the gathered light
  }
  var col = linearToSrgb(tonemapACES(hdr * u.exposure));
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}`;

const BRUSH_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let old = LOAD(paintPrev, vec2i(px));
  if (sdSegment(px, u.brushA, u.brushB) < u.brushR) { return vec4f(u.brushColor * u.brushGain, u.brushType); }
  return old;
}`;

export default {
  interaction: 'Paint with the mouse (pick Light, Wall or Eraser in the panel). Right-drag erases.',
  examples: [
    {
      id: 'paint',
      label: 'Paint with light',
      kind: 'Abstract',
      note: 'Every lit pixel here is lit only by the glowing shapes — there are no “light objects”. Paint a light and watch the red and blue walls <b>bleed their color</b> onto the floor (light bouncing). Paint walls to cast soft shadows.',
      params: { material: 'light', rays: 16, bounce: 0.8, history: 0.9, exposure: 1.0, view: 'final' },
    },
    {
      id: 'lava',
      label: 'Lava cave',
      kind: 'In a game',
      note: 'A river of lava and a few crystals are the only light sources. The cave walls catch orange from below and cyan from above — the mix happens automatically because every surface gathers light from every direction.',
      params: { material: 'wall', brushColor: '#9a8070', rays: 16, bounce: 0.8, history: 0.85, exposure: 0.65, view: 'final' },
    },
    {
      id: 'neon',
      label: 'Neon signs in the dark',
      kind: 'In a game',
      note: 'Flickering neon tubes light a dark alley. A passer-by blocks the light and casts soft moving shadows; the magenta panel spills color onto the ground. Raise “Rays per pixel” for less noise.',
      params: { material: 'light', brushColor: '#60f0ff', rays: 16, bounce: 0.7, history: 0.85, exposure: 0.5, view: 'final' },
    },
  ],
  controls: [
    { type: 'heading', label: 'Brush' },
    {
      type: 'select',
      key: 'material',
      label: 'Paint material',
      value: 'light',
      options: [
        { value: 'light', label: 'Light (emissive)' },
        { value: 'wall', label: 'Wall (solid, colored)' },
        { value: 'erase', label: 'Eraser' },
      ],
      help: 'Lights emit their color; walls block and reflect light (tinted by their color).',
    },
    { type: 'color', key: 'brushColor', label: 'Brush color', value: '#ffb060' },
    { type: 'slider', key: 'brushSize', label: 'Brush size', min: 0.005, max: 0.08, step: 0.001, value: 0.014 },
    { type: 'slider', key: 'emission', label: 'Light strength', min: 0.5, max: 12, step: 0.1, value: 3, help: 'Emission of painted lights (HDR: can exceed 1).' },
    { type: 'button', key: 'reset', label: 'Clear painting' },
    { type: 'heading', label: 'Global illumination' },
    { type: 'slider', key: 'rays', label: 'Rays per pixel', min: 2, max: 64, step: 1, value: 16, help: 'Directions sampled per pixel per frame. More = less noise, more cost.' },
    { type: 'slider', key: 'bounce', label: 'Bounce light', min: 0, max: 1, step: 0.01, value: 0.8, help: 'How much light walls reflect. 0 = direct light only (no color bleeding).' },
    { type: 'slider', key: 'history', label: 'Temporal accumulation', min: 0, max: 0.98, step: 0.01, value: 0.9, help: 'Weight of last frame’s result. 0 = raw noisy frame; high = smooth but slow to react.' },
    { type: 'slider', key: 'giScale', label: 'GI resolution', min: 0.2, max: 1, step: 0.05, value: 0.5, format: (v) => `${Math.round(v * 100)}%`, help: 'Lighting is computed at a fraction of the screen resolution and upscaled.' },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: 0.2, max: 4, step: 0.01, value: 1.2 },
    {
      type: 'select',
      key: 'view',
      label: 'View (pipeline stage)',
      value: 'final',
      options: [
        { value: 'final', label: 'Final image' },
        { value: 'material', label: '1. Materials (lights & walls)' },
        { value: 'seeds', label: '2. Jump flood: nearest wall (Voronoi)' },
        { value: 'dist', label: '3. Distance field' },
        { value: 'raw', label: '4. One frame of rays (noisy)' },
        { value: 'acc', label: '5. Accumulated light' },
      ],
    },
  ],
  about: {
    summary:
      'Global illumination (GI) means light comes from everything that glows and bounces off everything it hits. In 2D it can be done in real time by shooting rays through a distance field.',
    what: `<p>There are no light objects. Lights are just <b>pixels that emit</b>; walls are pixels that <b>block and reflect</b>.
      Every pixel on screen looks around in many directions to see what light reaches it — so you get soft shadows, light from big
      glowing shapes, and <b>color bleeding</b> (a red wall tints the floor red) for free.</p>`,
    how: `<ol>
      <li><b>Materials</b>: the scene (preset + your painting) is written into a texture: emission color for lights, albedo for walls.</li>
      <li><b>Jump flooding (JFA)</b>: to march rays fast we need, for every pixel, the distance to the nearest wall. JFA computes it in
        log₂(size) compute passes: each texel looks at 9 texels <i>k</i> apart (k = 512, 256 … 1) and keeps the nearest wall any of them
        knows about.</li>
      <li><b>Ray marching</b>: each pixel shoots N rays in evenly spaced directions. Along a ray we jump by the distance field value
        (sphere tracing), so empty space costs only a few steps. When a ray hits a light it brings back its color.</li>
      <li><b>Noise + temporal accumulation</b>: the ray directions are rotated randomly per pixel and per frame, and each frame is blended
        into the previous result. A few rays per frame add up to hundreds over time.</li>
      <li><b>Bounces</b>: when a ray hits a wall, it brings back <i>wall color × the light gathered in front of that wall last frame</i>.
        Each frame adds one more bounce, so light propagates around corners.</li>
      <li><b>Composite</b>: the empty floor shows the gathered light; walls show their color times the light next to them; lights show their emission. Then tone mapping.</li>
    </ol>
    <p>The state of the art for this is <b>Radiance Cascades</b> (used in Path of Exile 2), which removes the noise entirely by tracing short rays at many resolutions and merging them.</p>`,
    uses: [
      { title: 'Atmospheric games', text: 'Lava, neon, magic and bioluminescence that light the world without placing hundreds of point lights by hand.' },
      { title: 'Emissive art', text: 'Any sprite can be a light — glowing runes, screens, windows — with correct soft shadows.' },
      { title: 'Indirect light', text: 'Light bouncing into rooms and around corners, colored by the walls: Noita-like caves, Path of Exile 2 (radiance cascades).' },
      { title: 'Level design tools', text: 'Paint lights directly into the level and see the final lighting instantly.' },
    ],
    try: [
      'Set <i>Rays per pixel</i> to 2 and <i>Temporal accumulation</i> to 0 — that is the raw noise. Then raise accumulation to 0.95 and watch it settle.',
      'On <b>Paint with light</b>, paint a light right next to the red wall and look at the floor near it: it turns red. Set <i>Bounce light</i> to 0 and the tint disappears.',
      'Draw a closed box of walls with a light inside, leaving a small gap: light leaks out of the gap as a soft beam.',
      'Step through the <i>View</i> menu 1 → 5 to see each stage of the pipeline.',
      'Lower <i>GI resolution</i> to 20%: much faster, blurrier light, sharp materials still on top.',
    ],
    ask: [
      '2D global illumination with emissive pixels',
      'jump flooding to build a distance field every frame',
      'ray-marched soft shadows and color bleeding in 2D',
      'radiance cascades for noise-free 2D lighting',
      'temporal accumulation to denoise ray-traced lighting',
      'paint lights directly into the level',
    ],
    perf: `<p>Cost ≈ <i>GI pixels × rays × steps</i>. At half resolution on 1080p (≈ 500k pixels) with 16 rays and ~16 steps that is
      ~130 million distance-field reads per frame — fine for a desktop GPU, heavy for phones. The JFA costs ~11 cheap passes.
      Levers: GI resolution, ray count, temporal accumulation (fewer rays per frame), or radiance cascades (fixed cost, no noise).</p>`,
    api: `<p><b>WebGPU-friendly</b>: the passes are compute shaders writing storage textures, chained without any CPU round-trip,
      and JFA needs ~11 passes per frame that each reuse the previous output. All of it can be done with WebGL2 fragment passes and
      ping-pong framebuffers too (many WebGL demos exist) — compute just makes it simpler and lets you skip the rasterizer.</p>`,
    code: [
      {
        title: 'March one ray through the distance field',
        lang: 'wgsl',
        src: `fn traceRay(p: vec2f, dir: vec2f, size: vec2f) -> vec3f {
  var t = 0.0;
  for (var i = 0; i < 96; i++) {
    let q = p + dir * t;
    let s = textureLoad(jfa, vec2i(q), 0);             // nearest solid texel (from JFA)
    let d = length(s.xy + 0.5 - q);
    if (d < 1.0) {
      let m = textureLoad(scene, vec2i(s.xy), 0);      // what did we hit?
      if (m.a < 1.5) { return m.rgb; }                 // a light: its emission
      let back = textureSampleLevel(prev, samp, (q - dir * 1.5) / size, 0.0).rgb;
      return m.rgb * back * u.bounce;                  // a wall: albedo × last frame's light
    }
    t += max(d - 0.75, 0.75);                          // sphere tracing
  }
  return vec3f(0.0);
}`,
      },
      { title: 'One jump-flood pass (compute)', lang: 'wgsl', src: JFA_WGSL.trim() },
      {
        title: 'N stratified rays + temporal blend',
        lang: 'wgsl',
        src: `let jitter = fract(hashU(id.xy, 0u) + u.frame * 0.61803398875);
for (var r = 0; r < n; r++) {
  let a = (f32(r) + jitter) / f32(n) * TAU;
  sum += traceRay(p, vec2f(cos(a), sin(a)), size);
}
let acc = mix(old, vec4f(sum / f32(n), 1.0), u.alpha);   // alpha = 1 - history`,
      },
    ],
    links: [
      { title: 'Jason McGhee — Building Real-Time Global Illumination', url: 'https://jason.today/gi', note: 'excellent 2D GI walkthrough (JFA, ray marching, radiance cascades)' },
      { title: 'Rong & Tan — Jump Flooding (2006)', url: 'https://www.comp.nus.edu.sg/~tants/jfa.html' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const U = gpu.uniforms(
      {
        resolution: 'vec2f',
        giSize: 'vec2f',
        brushA: 'vec2f',
        brushB: 'vec2f',
        brushColor: 'vec3f',
        brushR: 'f32',
        giScale: 'f32',
        time: 'f32',
        example: 'f32',
        frame: 'f32',
        rays: 'f32',
        steps: 'f32',
        bounce: 'f32',
        alpha: 'f32',
        view: 'f32',
        exposure: 'f32',
        ambient: 'f32',
        brushType: 'f32',
        brushGain: 'f32',
      },
      'GIU',
    );
    const jumps = [];
    for (let i = 0; i < 16; i++) jumps.push(gpu.uniforms({ jump: 'f32' }, 'JStep'));

    const compose = gpu.compute({
      label: 'gi-compose',
      bindings: {
        u: { uniform: U },
        paint: { texture: 'float' },
        sceneOut: { storageTexture: 'rgba16float', access: 'write' },
        seedOut: { storageTexture: 'rgba16float', access: 'write' },
      },
      include: ['sdf', 'noise', 'math'],
      code: COMPOSE_WGSL,
    });
    const jfa = gpu.compute({
      label: 'gi-jfa',
      bindings: { js: { uniform: jumps[0] }, src: { texture: 'float' }, dst: { storageTexture: 'rgba16float', access: 'write' } },
      code: JFA_WGSL,
    });
    const giProg = gpu.compute({
      label: 'gi-trace',
      bindings: {
        u: { uniform: U },
        scene: { texture: 'float' },
        jfa: { texture: 'float' },
        prev: { texture: 'float' },
        samp: { sampler: true },
        accOut: { storageTexture: 'rgba16float', access: 'write' },
        rawOut: { storageTexture: 'rgba16float', access: 'write' },
      },
      include: ['hash'],
      code: GI_WGSL,
    });
    const composite = gpu.fullscreen({
      label: 'gi-composite',
      uniforms: U,
      textures: ['paint', 'acc', 'raw', 'jfa'],
      include: ['sdf', 'noise', 'math', 'color'],
      code: COMPOSITE_WGSL,
    });
    const brush = gpu.fullscreen({ label: 'gi-brush', uniforms: U, textures: ['paintPrev'], include: ['sdf'], code: BRUSH_WGSL });

    let paint = null;
    let gw = 0;
    let gh = 0;
    let giScale = 0;
    let sceneT = null;
    let jfaA = null;
    let jfaB = null;
    let accA = null;
    let accB = null;
    let rawT = null;
    let fresh = 0; // frames since the lighting was reset

    const allocPaint = () => {
      paint?.destroy();
      paint = gpu.pingPong(ctx.width, ctx.height, { format: 'rgba16float', label: 'paint' });
      const enc = gpu.device.createCommandEncoder();
      gpu.clear(enc, paint.a, [0, 0, 0, 0]);
      gpu.clear(enc, paint.b, [0, 0, 0, 0]);
      gpu.queue.submit([enc.finish()]);
    };
    const allocGI = (scale) => {
      giScale = scale;
      gw = Math.max(16, Math.round(ctx.width * scale));
      gh = Math.max(16, Math.round(ctx.height * scale));
      for (const t of [sceneT, jfaA, jfaB, accA, accB, rawT]) t?.destroy();
      const mk = (label) => gpu.target(gw, gh, { format: 'rgba16float', label });
      sceneT = mk('gi-scene');
      jfaA = mk('jfa-a');
      jfaB = mk('jfa-b');
      accA = mk('acc-a');
      accB = mk('acc-b');
      rawT = mk('gi-raw');
      const enc = gpu.device.createCommandEncoder();
      gpu.clear(enc, accA, [0, 0, 0, 0]);
      gpu.clear(enc, accB, [0, 0, 0, 0]);
      gpu.queue.submit([enc.finish()]);
      fresh = 0;
    };
    allocPaint();

    const readout = tag(ctx, 'right:8px;bottom:8px');
    let last = null; // last brush point
    let frameNo = 0;

    return {
      resize() {
        allocPaint();
        giScale = 0;
      },
      onAction(key) {
        if (key === 'reset') {
          allocPaint();
          fresh = 0;
        }
      },
      onExample() {
        allocPaint();
        fresh = 0;
      },
      onChange(key) {
        if (key === 'bounce' || key === 'giScale') fresh = 0;
      },
      frame(ctx) {
        const p = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const wantScale = ctx.testMode ? Math.min(p.giScale, 0.3) : p.giScale;
        if (wantScale !== giScale || Math.round(W * wantScale) !== gw || Math.round(H * wantScale) !== gh) allocGI(wantScale);
        const enc = ctx.encoder;
        frameNo++;

        // ---- brush
        const ptr = ctx.pointer;
        let painting = false;
        if (ptr.down && ptr.over) {
          const cur = [ptr.x, ptr.y];
          const from = last || cur;
          const erase = ptr.button === 2 || p.material === 'erase';
          const isLight = !erase && p.material === 'light';
          const hex = p.brushColor || '#ffffff';
          const n = parseInt(hex.slice(1), 16);
          const lin = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255].map((v) => Math.pow(v, 2.2));
          U.setAll({ brushA: from, brushB: cur, brushR: p.brushSize * H, brushType: erase ? 3 : isLight ? 1 : 2, brushColor: lin, brushGain: isLight ? p.emission : 1 });
          painting = true;
          last = cur;
        } else last = null;

        const animated = ctx.example !== 'paint';
        if (painting) fresh = Math.min(fresh, 2);
        fresh++;
        const minAlpha = 1 - p.history;
        const alpha = Math.max(minAlpha, 1 / fresh);
        U.setAll({
          resolution: [W, H],
          giSize: [gw, gh],
          giScale: gw / W,
          time: ctx.time,
          example: ctx.exampleIndex,
          frame: frameNo % 4096,
          rays: ctx.testMode ? Math.min(p.rays, 8) : p.rays,
          steps: ctx.testMode ? 40 : 72,
          bounce: p.bounce,
          alpha: animated ? Math.max(alpha, 0.04) : alpha,
          view: { final: 0, material: 1, seeds: 2, dist: 3, raw: 4, acc: 5 }[p.view] ?? 0,
          exposure: p.exposure,
          ambient: 0.0,
        });
        U.upload();
        if (painting) {
          brush.draw(enc, paint.write, { paintPrev: paint.read });
          paint.swap();
        }

        // ---- 1. materials + JFA seeds
        const g8 = [Math.ceil(gw / 8), Math.ceil(gh / 8)];
        compose.dispatch(enc, 'compose', g8, { u: U, paint: paint.read, sceneOut: sceneT, seedOut: jfaA });
        // ---- 2. jump flooding: jumps of n/2, n/4 … 1, plus one extra pass of 1 (JFA+1) for accuracy
        let src = jfaA;
        let dst = jfaB;
        let k = 1 << Math.ceil(Math.log2(Math.max(gw, gh)));
        let pass = 0;
        const ks = [];
        while ((k >>= 1) >= 1) ks.push(k);
        ks.push(1);
        for (const kk of ks) {
          const js = jumps[pass++];
          js.set('jump', kk);
          js.upload();
          jfa.dispatch(enc, 'jfa', g8, { js, src, dst });
          [src, dst] = [dst, src];
        }
        const jfaFinal = src;
        // ---- 3. trace rays, accumulate
        giProg.dispatch(enc, 'gi', g8, { u: U, scene: sceneT, jfa: jfaFinal, prev: accA, samp: 'linear', accOut: accB, rawOut: rawT });
        [accA, accB] = [accB, accA];
        // ---- 4. composite
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { paint: paint.read, acc: accA, raw: rawT, jfa: jfaFinal });

        readout.textContent = `GI ${gw}×${gh} · ${Math.round(ctx.testMode ? Math.min(p.rays, 8) : p.rays)} rays/px · ${ks.length} JFA passes · ${painting ? 'painting…' : `blend ${(Math.max(minAlpha, 1 / fresh) * 100).toFixed(0)}% new`}`;
      },
    };
  },
};

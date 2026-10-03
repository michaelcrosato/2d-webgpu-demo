// Dynamic 2D lights & normal maps — a small deferred 2D renderer:
//   1. sprites are drawn twice: colors -> "albedo" target, normal-map atlas -> "normals" target (+ emissive in alpha)
//   2. a compute shader bins every light into 16×16-pixel screen tiles (tiled light culling)
//   3. a fullscreen pass lights every pixel with only the lights of its tile (storage buffers), tone-maps to the screen
//   4. the light sources themselves are drawn as additive glows on top.

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas, rng } from '../../core/assets.js';
import { tag, hexLinear, noise1, clamp } from './_shared.js';

const MAX_LIGHTS = 1024;
const TILE = 16;
const STRIDE = 128; // per tile: [count, idx0, idx1, … idx126]

const LIGHT_WGSL = /* wgsl */ `
struct Light { pos: vec2f, radius: f32, height: f32, color: vec3f, intensity: f32 };
const TILE: u32 = ${TILE}u;
const STRIDE: u32 = ${STRIDE}u;
`;

// Sprites draw into two targets. cam.extra.x = 0: albedo pass, 1: normal pass.
// user value: emissive (0..1, or 2 = "only hot/orange pixels glow") + 10 if flipped horizontally.
const SPRITE_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  var flags = i.extra.w;
  var flip = 0.0;
  if (flags >= 5.0) { flip = 1.0; flags = flags - 10.0; }
  if (cam.extra.x < 0.5) { return vec4f(t.rgb * i.color.rgb, 1.0); }
  var em = flags;
  if (flags > 1.5) { em = clamp((t.r - t.b) * 2.0 - 0.6, 0.0, 1.0) * step(0.6, t.r); }
  // normal map texel (tangent space, +y up) -> screen space (+y down), honouring flip & rotation
  var n = t.xyz * 2.0 - 1.0;
  if (flip > 0.5) { n.x = -n.x; }
  let s = vec2f(n.x, -n.y);
  let c = cos(i.extra.x);
  let sn = sin(i.extra.x);
  let r = vec2f(c * s.x - sn * s.y, sn * s.x + c * s.y);
  return vec4f(normalize(vec3f(r, n.z)) * 0.5 + 0.5, em);
}`;

const FALLOFF_WGSL = /* wgsl */ `
// d = distance to the light, r = radius where the light reaches zero
fn falloff(d: f32, r: f32) -> f32 {
  let x = clamp(d / r, 0.0, 1.0);
  if (u.curve < 0.5) { return 1.0 - x; }                 // linear: a cone, visible hard rim
  if (u.curve < 1.5) {                                    // inverse-square, windowed to reach 0 at r
    let w = clamp(1.0 - x * x * x * x, 0.0, 1.0);
    let k = d / max(r * 0.2, 1.0);
    return w * w / (1.0 + k * k);
  }
  let s = 1.0 - x * x;                                    // smooth: (1 - x²)²
  return s * s;
}`;

const LIGHTING_WGSL = /* wgsl */ `
${LIGHT_WGSL}
${FALLOFF_WGSL}
struct Contrib { d: vec3f, s: vec3f };

fn lightContrib(L: Light, px: vec2f, N: vec3f) -> Contrib {
  var c: Contrib;
  let delta = L.pos - px;
  let dist = length(delta);
  if (dist >= L.radius) { return c; }
  let ldir = normalize(vec3f(delta, L.height));         // light hangs L.height pixels above the floor
  let att = falloff(dist, L.radius) * L.intensity;
  // N·L relative to a flat surface, so normal maps add relief without changing overall brightness
  let relief = clamp(dot(N, ldir) / max(ldir.z, 0.06), 0.0, 2.2);
  let ndl = mix(1.0, relief, u.normalsOn);
  let hv = normalize(ldir + vec3f(0.0, 0.0, 1.0));      // Blinn-Phong half vector, viewer looks straight down
  let sp = pow(max(dot(N, hv), 0.0), u.shininess) * u.specular;
  c.d = L.color * att * ndl;
  c.s = L.color * att * sp;
  return c;
}

fn heat(t: f32) -> vec3f {
  return clamp(vec3f(1.5 * t - 0.2, 1.6 * t * (1.4 - t), 1.0 - 1.8 * t), vec3f(0.0), vec3f(1.0));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let a = LOAD(albedo, vec2i(px));
  let nt = LOAD(normals, vec2i(px));
  let alb = srgbToLinear(a.rgb);
  let N = normalize(nt.xyz * 2.0 - 1.0);
  let emis = nt.a;

  // which lights touch this pixel? (tiled: only the ones binned into our 16×16 tile)
  let tp = vec2u(px) / TILE;
  let base = (tp.y * u.tilesX + tp.x) * STRIDE;
  var n = u.count;
  if (u.tiled > 0.5) { n = min(tiles[base], STRIDE - 1u); }
  var diff = vec3f(0.0);
  var spec = vec3f(0.0);
  for (var k: u32 = 0u; k < n; k++) {
    var li = k;
    if (u.tiled > 0.5) { li = tiles[base + 1u + k]; }
    let c = lightContrib(lights[li], px, N);
    diff += c.d;
    spec += c.s;
  }

  let v = i32(u.view + 0.5);
  var hdr = alb * (u.ambient + diff) + spec * (0.35 + 0.65 * luma(alb)) + alb * emis * 4.0;
  if (v == 1) { return vec4f(a.rgb, 1.0); }
  if (v == 2) { return vec4f(nt.rgb, 1.0); }
  if (v == 3) { hdr = vec3f(0.5) * (u.ambient + diff) + spec; }
  var col = linearToSrgb(tonemapACES(hdr * u.exposure));
  if (v == 4) {
    let cnt = f32(tiles[base]);
    let fx = fract(px / f32(TILE));
    let grid = step(min(fx.x, fx.y), 1.0 / f32(TILE));
    var h = heat(clamp(cnt / 24.0, 0.0, 1.0)) * step(0.5, cnt);
    if (cnt > f32(STRIDE - 1u)) { h = vec3f(1.0, 0.0, 0.6); }
    col = mix(col * 0.35 + h * 0.65, vec3f(0.0), grid * 0.5);
  }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}`;

const CULL_WGSL = /* wgsl */ `
${LIGHT_WGSL}
var<workgroup> wgCount: atomic<u32>;

// one workgroup (64 threads) per 16×16 tile; threads split the light list between them
@compute @workgroup_size(64)
fn cull(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wgCount, 0u); }
  workgroupBarrier();
  let lo = vec2f(f32(wg.x * TILE), f32(wg.y * TILE));
  let hi = lo + vec2f(f32(TILE));
  let base = (wg.y * u.tilesX + wg.x) * STRIDE;
  for (var i: u32 = li; i < u.count; i += 64u) {
    let L = lights[i];
    let nearest = clamp(L.pos, lo, hi);                  // closest point of the tile to the light
    if (L.intensity > 0.0 && distance(nearest, L.pos) < L.radius) {
      let slot = atomicAdd(&wgCount, 1u);
      if (slot < STRIDE - 1u) { tiles[base + 1u + slot] = i; }
    }
  }
  workgroupBarrier();
  if (li == 0u) { tiles[base] = atomicLoad(&wgCount); }
}`;

// ------------------------------------------------------------------------------------- layouts

const COLORS = {
  red: '#ff4b4b',
  green: '#4bff6e',
  blue: '#4b7bff',
  torch: '#ff9a3c',
  lantern: '#ffe2b0',
};
const NEON = ['#ff3fa4', '#38e8ff', '#ffd23f', '#9b5cff', '#4dff9a', '#ff6a3d'];

function falloffJS(curve, x) {
  x = clamp(x, 0, 1);
  if (curve === 0) return 1 - x;
  if (curve === 1) {
    const w = clamp(1 - x ** 4, 0, 1);
    const k = x / 0.2;
    return (w * w) / (1 + k * k);
  }
  const s = 1 - x * x;
  return s * s;
}

export default {
  interaction: 'Move the mouse to carry a light · click to drop colored lights (Falloff tab).',
  examples: [
    {
      id: 'falloff',
      label: 'Falloff & color',
      kind: 'Abstract',
      note: 'Red, green and blue lights <b>add up</b>: where they overlap you get yellow, cyan, magenta and white. The graph (bottom-right) shows how brightness drops with distance for the selected <b>falloff curve</b>. Toggle normal maps to see the sprites gain relief.',
      params: { curve: 'smooth', radius: 1, height: 0.12, ambient: 0.06, normals: true, specular: 0.5, view: 'final' },
    },
    {
      id: 'dungeon',
      label: 'Dungeon torches',
      kind: 'In a game',
      note: 'A tile room lit only by flickering torches and the lantern you carry. Every brick and stone has a <b>normal map</b>, so light rakes across the bumps exactly like a real wall. Turn “Normal maps” off to see the flat version.',
      params: { curve: 'invsq', radius: 1, height: 0.08, ambient: 0.03, normals: true, specular: 0.35, view: 'final' },
    },
    {
      id: 'neon',
      label: 'Neon night street',
      kind: 'In a game',
      note: 'Hundreds of lights: neon signs, car head- and tail-lights, lantern strings and a sky-lantern festival. All lights live in a GPU <b>storage buffer</b>; a compute shader sorts them into 16×16-pixel tiles so each pixel only evaluates the few lights that reach it (<b>tiled lighting</b>). Try View → “Lights per tile”.',
      params: { curve: 'smooth', radius: 1, height: 0.1, ambient: 0.05, normals: true, specular: 0.6, count: 320, tiled: true, view: 'final' },
    },
  ],
  controls: [
    { type: 'heading', label: 'Lights' },
    {
      type: 'select',
      key: 'curve',
      label: 'Falloff curve',
      value: 'smooth',
      options: [
        { value: 'linear', label: 'Linear (1 − d/r)' },
        { value: 'invsq', label: 'Inverse-square (windowed)' },
        { value: 'smooth', label: 'Smooth (1 − (d/r)²)²' },
      ],
      help: 'How brightness drops with distance. Inverse-square is physical (bright core, long tail); linear shows a visible rim.',
    },
    { type: 'slider', key: 'radius', label: 'Light radius', min: 0.3, max: 2.5, step: 0.01, value: 1, help: 'Multiplies every light’s reach. Bigger = more overlap = more cost per pixel.' },
    { type: 'slider', key: 'height', label: 'Light height', min: 0.01, max: 0.5, step: 0.005, value: 0.12, help: 'How far above the floor the lights hang. Low = grazing light that rakes across bumps.' },
    { type: 'slider', key: 'ambient', label: 'Ambient light', min: 0, max: 0.4, step: 0.005, value: 0.06, help: 'Light that reaches everywhere (sky, bounce). 0 = pitch black outside the lights.' },
    { type: 'heading', label: 'Surfaces' },
    { type: 'toggle', key: 'normals', label: 'Normal maps', value: true, help: 'Per-pixel surface direction from the normal-map atlas. Off = everything is a flat plane.' },
    { type: 'slider', key: 'specular', label: 'Specular (shine)', min: 0, max: 2, step: 0.01, value: 0.5, help: 'Blinn-Phong highlight: wet stone, metal, polished floors.' },
    { type: 'heading', label: 'Many lights', showFor: ['neon'] },
    { type: 'slider', key: 'count', label: 'Light count', min: 16, max: MAX_LIGHTS, step: 1, value: 320, log: true, showFor: ['neon'], format: (v) => Math.round(v).toLocaleString(), help: 'Signs, cars, lantern strings, then floating sky lanterns.' },
    { type: 'toggle', key: 'tiled', label: 'Tiled light culling (compute)', value: true, showFor: ['neon'], help: 'Off = every pixel loops over every light (brute force). Watch the fps at 1,024 lights.' },
    {
      type: 'select',
      key: 'view',
      label: 'View',
      value: 'final',
      options: [
        { value: 'final', label: 'Final image' },
        { value: 'albedo', label: 'Albedo (colors only)' },
        { value: 'normals', label: 'Normal buffer' },
        { value: 'light', label: 'Light only' },
        { value: 'heat', label: 'Lights per tile' },
      ],
      help: 'Inspect the intermediate buffers of the deferred pipeline.',
    },
    { type: 'button', key: 'reset', label: 'Remove dropped lights' },
  ],
  about: {
    summary:
      'Real-time 2D lights: each pixel adds up the light reaching it from every nearby lamp, shaped by a falloff curve, colored, and bent by a normal map so flat pixel art looks sculpted.',
    what: `<p>A <b>deferred 2D lighting</b> renderer. Sprites and tiles are drawn once for their colors and once for their
      <b>normals</b> (which way each pixel “faces”). A single full-screen pass then lights every pixel from the light list.
      Light sources are drawn on top as additive glows.</p>`,
    how: `<ol>
      <li><b>G-buffer</b>: draw all sprites into an <i>albedo</i> texture (plain colors) and, with the matching normal-map atlas, into a <i>normal</i> texture. The alpha channel of the normal texture marks emissive pixels (torch flames).</li>
      <li><b>Falloff</b>: brightness = <code>intensity × falloff(distance / radius)</code>. Physical light follows inverse-square (1/d²), which never reaches zero, so games multiply it by a <i>window</i> that fades to 0 at the radius.</li>
      <li><b>Normal mapping</b>: each light is a point <i>height</i> pixels above the floor. The surface normal N (from the texture) dotted with the direction to the light L gives Lambert shading: slopes facing the light brighten, the rest darken.</li>
      <li><b>Specular</b>: Blinn-Phong — the half-vector between light and viewer dotted with N, raised to a power → small sharp highlights.</li>
      <li><b>Color mixing</b> happens in <i>linear light</i>: colors are converted from sRGB, added, then tone-mapped (ACES) and converted back, so overlapping red + green really makes yellow, not mud.</li>
      <li><b>Tiled culling</b> (Neon tab): a compute shader runs one workgroup per 16×16-pixel tile, tests every light’s circle against the tile and writes a short list of indices into a storage buffer. The lighting pass loops only over its tile’s list.</li>
    </ol>`,
    uses: [
      { title: 'Atmosphere', text: 'Torch-lit dungeons and caves (Dead Cells, Blasphemous, Children of Morta) — darkness is half the mood.' },
      { title: 'Gameplay', text: 'Light radius as a resource (Don’t Starve), lanterns that reveal secrets, flashlights in horror games.' },
      { title: 'Pixel art that pops', text: 'Normal-mapped sprites (Sprite Lamp, Godot / Unity 2D lights) react to moving lights like 3D models.' },
      { title: 'Cities & neon', text: 'Hundreds of small lights — signs, cars, windows — made affordable by tiled or clustered lighting.' },
    ],
    try: [
      'On <b>Falloff & color</b>, switch the curve to <i>Linear</i> — notice the visible ring where the light ends. <i>Inverse-square</i> has a hot core and a long soft tail.',
      'Drop the <i>Light height</i> to 0.02 on <b>Dungeon torches</b>: the light skims across the bricks and every bump casts a highlight. Raise it to 0.5 and the walls go flat.',
      'Turn <i>Normal maps</i> off and on while moving the lantern around a crate or the chest.',
      'On <b>Neon night street</b>, set View to <i>Lights per tile</i>, then push <i>Light radius</i> up: tiles get hot (red) as more lights overlap.',
      'Max the light count, then toggle <i>Tiled light culling</i> off and watch the fps counter.',
    ],
    ask: [
      'normal-mapped 2D lighting for pixel-art sprites',
      'dynamic point lights with flicker for torches',
      'a lantern light that follows the player',
      'tiled (or clustered) light culling for hundreds of 2D lights',
      'inverse-square falloff with a radius cutoff',
      'emissive sprites that glow in the dark',
    ],
    perf: `<p>Cost ≈ <i>pixels × lights touching each pixel</i>. Brute force is <code>pixels × all lights</code>: 2 million pixels × 1,000 lights
      = 2 billion light evaluations per frame — too slow. Tiled culling costs one tiny compute pass and turns that into “a dozen lights per
      pixel”. Large radii defeat it (every tile sees every light), so keep radii tight. The G-buffer costs two sprite passes but makes
      the light count independent of the sprite count.</p>`,
    api: `<p>The lighting itself is a fragment shader and would work in WebGL2 too (with a fixed-size uniform array of ~64–256 lights).
      What WebGPU adds: <b>storage buffers</b> (arbitrarily many lights, read directly in the shader) and <b>compute shaders with workgroup
      atomics</b> for building the per-tile light lists on the GPU. In WebGL2 you would cull lights on the CPU or render light volumes
      with additive blending instead.</p>`,
    code: [
      {
        title: 'Per-light shading (fragment)',
        lang: 'wgsl',
        src: `fn lightContrib(L: Light, px: vec2f, N: vec3f) -> Contrib {
  var c: Contrib;
  let delta = L.pos - px;
  let dist = length(delta);
  if (dist >= L.radius) { return c; }
  let ldir = normalize(vec3f(delta, L.height));     // light floats above the floor
  let att = falloff(dist, L.radius) * L.intensity;
  let relief = clamp(dot(N, ldir) / max(ldir.z, 0.06), 0.0, 2.2);
  let ndl = mix(1.0, relief, u.normalsOn);          // normal maps on/off
  let hv = normalize(ldir + vec3f(0.0, 0.0, 1.0));  // Blinn-Phong
  let sp = pow(max(dot(N, hv), 0.0), u.shininess) * u.specular;
  c.d = L.color * att * ndl;
  c.s = L.color * att * sp;
  return c;
}`,
      },
      {
        title: 'Tiled light culling (compute)',
        lang: 'wgsl',
        src: `var<workgroup> wgCount: atomic<u32>;
@compute @workgroup_size(64)
fn cull(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wgCount, 0u); }
  workgroupBarrier();
  let lo = vec2f(f32(wg.x * TILE), f32(wg.y * TILE));
  let hi = lo + vec2f(f32(TILE));
  let base = (wg.y * u.tilesX + wg.x) * STRIDE;
  for (var i: u32 = li; i < u.count; i += 64u) {      // 64 threads share the list
    let L = lights[i];
    if (distance(clamp(L.pos, lo, hi), L.pos) < L.radius) {
      let slot = atomicAdd(&wgCount, 1u);
      if (slot < STRIDE - 1u) { tiles[base + 1u + slot] = i; }
    }
  }
  workgroupBarrier();
  if (li == 0u) { tiles[base] = atomicLoad(&wgCount); }
}`,
      },
      {
        title: 'Falloff curves',
        lang: 'wgsl',
        src: FALLOFF_WGSL.trim(),
      },
    ],
    links: [
      { title: 'LearnOpenGL — Normal Mapping', url: 'https://learnopengl.com/Advanced-Lighting/Normal-Mapping' },
      { title: 'LearnOpenGL — Light casters (attenuation)', url: 'https://learnopengl.com/Lighting/Light-casters' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasTex = gpu.textureFromImage(atlas.canvas, { label: 'atlas' });
    const normTex = gpu.textureFromImage(atlas.normalCanvas, { label: 'atlas-normals' });
    const atlasView = atlasTex.createView();
    const normView = normTex.createView();

    const cam = new Camera2D();
    const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest', fragment: SPRITE_FS, capacity: 8192 });
    const shapes = new ShapeBatch(gpu); // albedo-pass shapes (cars)
    const glows = new ShapeBatch(gpu); // light sources, after lighting
    const ui = new ShapeBatch(gpu); // falloff graph

    const U = gpu.uniforms(
      {
        resolution: 'vec2f',
        tilesX: 'u32',
        tilesY: 'u32',
        ambient: 'vec3f',
        count: 'u32',
        curve: 'f32',
        normalsOn: 'f32',
        specular: 'f32',
        shininess: 'f32',
        view: 'f32',
        tiled: 'f32',
        exposure: 'f32',
        time: 'f32',
      },
      'LightU',
    );
    const lightData = new Float32Array(MAX_LIGHTS * 8);
    const lightBuf = gpu.storage(lightData.byteLength, 'lights');

    const cull = gpu.compute({
      label: 'light-cull',
      bindings: {
        u: { uniform: U },
        lights: { storage: 'array<Light>', access: 'read' },
        tiles: { storage: 'array<u32>', access: 'read_write' },
      },
      code: CULL_WGSL,
    });
    const lighting = gpu.fullscreen({
      label: 'lighting',
      uniforms: U,
      textures: ['albedo', 'normals'],
      storage: { lights: 'array<Light>', tiles: 'array<u32>' },
      include: ['color', 'hash'],
      code: LIGHTING_WGSL,
    });

    let W = 0;
    let H = 0;
    let albedoT = null;
    let normalT = null;
    let tileBuf = null;
    let tilesX = 1;
    let tilesY = 1;
    const alloc = (w, h) => {
      W = w;
      H = h;
      albedoT?.destroy();
      normalT?.destroy();
      tileBuf?.destroy();
      albedoT = gpu.target(w, h, { label: 'albedo' });
      normalT = gpu.target(w, h, { label: 'normals' });
      tilesX = Math.ceil(w / TILE);
      tilesY = Math.ceil(h / TILE);
      tileBuf = gpu.storage(tilesX * tilesY * STRIDE * 4, 'tile-lists');
    };
    alloc(ctx.width, ctx.height);

    const readout = tag(ctx, 'right:8px;bottom:8px');
    const legend = tag(ctx, 'right:8px;bottom:40px;display:none');

    // ---------------------------------------------------------------- per-frame draw lists
    let cmds = [];
    let lights = []; // {x,y,r,h,c:[r,g,b] linear,i, glow?: {size, col}}
    const spr = (name, x, y, w, h, o = {}) => cmds.push([atlas.uv(name), x, y, w, h, o]);
    const submit = () => {
      for (const [uv, x, y, w, h, o] of cmds)
        sprites.draw(x, y, w, h, { uv, rotation: o.rot || 0, flipX: !!o.flip, color: o.color, anchor: o.anchor || [0, 0], user: (o.emis || 0) + (o.flip ? 10 : 0) });
    };
    const addLight = (x, y, r, h, hex, intensity = 1, glow = null) => lights.push({ x, y, r, h, c: hexLinear(hex), i: intensity, glow, hex });

    let dropped = [];
    let mouse = { x: ctx.width * 0.5, y: ctx.height * 0.5 };

    // ---------------------------------------------------------------- scene builders
    const tileScale = (rowsWanted) => Math.max(1, Math.round(H / (16 * rowsWanted)));

    function sceneFalloff(t, p) {
      const S = tileScale(9);
      const T = 16 * S;
      const cols = Math.ceil(W / T) + 1;
      const rows = Math.ceil(H / T) + 1;
      const ox = (W - cols * T) / 2;
      const oy = (H - rows * T) / 2;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) spr('tile_metal', ox + c * T, oy + r * T, T, T);
      // a shelf of props, big, to show normal-mapped relief
      const props = ['shield', 'gem', 'chest', 'sword', 'potion', 'heart', 'crate', 'star', 'bomb', 'mushroom', 'key'];
      const ps = Math.max(2, Math.round(S * 2.2)) * 16;
      const gap = ps * 1.15;
      const n = Math.min(props.length, Math.floor((W * 0.86) / gap));
      for (let i = 0; i < n; i++) {
        const x = W / 2 + (i - (n - 1) / 2) * gap;
        spr(props[i], x, H * 0.5, ps, ps, { anchor: [0.5, 0.5], rot: Math.sin(t * 0.3 + i) * 0.0 });
      }
      // RGB Venn: three lights circling
      const R = Math.min(W, H) * 0.42 * p.radius;
      const hgt = p.height * H;
      ['red', 'green', 'blue'].forEach((k, i) => {
        const a = t * 0.25 + (i * Math.PI * 2) / 3 - Math.PI / 2;
        const x = W / 2 + Math.cos(a) * H * 0.17;
        const y = H / 2 + Math.sin(a) * H * 0.17;
        addLight(x, y, R, hgt, COLORS[k], 1.6, { size: H * 0.012 });
      });
      if (ctx.pointer.over) addLight(mouse.x, mouse.y, R * 0.8, hgt, '#ffffff', 1.3, { size: H * 0.01 });
      for (const d of dropped) addLight(d.x, d.y, R * 0.8, hgt, d.hex, 1.5, { size: H * 0.01 });
    }

    function sceneDungeon(t, p) {
      const S = tileScale(10);
      const T = 16 * S;
      const cols = Math.ceil(W / T);
      const rows = Math.ceil(H / T);
      const ox = (W - cols * T) / 2;
      const oy = (H - rows * T) / 2;
      const cx = Math.floor(cols / 2);
      const cy = Math.floor(rows / 2) + 1;
      const pil = [
        [Math.round(cols * 0.22), Math.round(rows * 0.55)],
        [cols - 1 - Math.round(cols * 0.22), Math.round(rows * 0.55)],
      ];
      const isWall = (c, r) => r <= 1 || r >= rows - 1 || c <= 0 || c >= cols - 1 || pil.some(([pc, pr]) => pc === c && pr === r);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          let name = 'tile_stone';
          if (isWall(c, r)) name = 'tile_brick';
          else if (Math.abs(c - cx) <= 2 && Math.abs(r - cy) <= 1) name = 'tile_wood';
          spr(name, ox + c * T, oy + r * T, T, T);
        }
      const at = (c, r) => [ox + c * T, oy + r * T];
      // torches on the back wall and on the pillars
      const torchFrame = (k) => `torch_${Math.floor(t * 8 + k * 1.7) % 3}`;
      const torchSpots = [];
      for (let c = 2; c < cols - 1; c += 4) torchSpots.push([c, 1]);
      for (const [pc, pr] of pil) torchSpots.push([pc, pr]);
      const R = H * 0.42 * p.radius;
      const hgt = p.height * H;
      torchSpots.forEach(([c, r], k) => {
        const [x, y] = at(c, r);
        spr(torchFrame(k), x, y, T, T, { emis: 2 });
        const fl = 0.75 + 0.35 * noise1(t * 9 + k * 13, 3) + 0.1 * Math.sin(t * 23 + k);
        const jx = (noise1(t * 5 + k * 7, 9) - 0.5) * T * 0.06;
        addLight(x + T / 2 + jx, y + T * 0.32, R * (0.92 + 0.08 * fl), hgt, COLORS.torch, 1.5 * fl, { size: T * 0.09 });
      });
      // props
      const put = (name, c, r, o) => {
        const [x, y] = at(c, r);
        spr(name, x, y, T, T, o);
      };
      put('chest', cx, 2);
      put('crate', 1, rows - 2);
      put('crate', 2, rows - 2);
      put('crate', 1, rows - 3);
      put('potion', cols - 3, rows - 2);
      put('key', cols - 2, rows - 2);
      put('mushroom', 3, 2);
      put('bomb', cols - 4, 2);
      put('sword', cx - 3, cy + 2, { rot: 0.0 });
      put('shield', cx + 3, cy + 2);
      put('gem', cx + 1, 2);
      for (let i = 0; i < 3; i++) put(`coin_${Math.floor(t * 10 + i * 2) % 8}`, cx - 1 + i, cy - 2);
      // hero, slime and bat
      const heroX = ox + (cx - 0.5) * T;
      const heroY = oy + cy * T;
      spr(`hero_idle_${Math.floor(t * 2) % 2}`, heroX, heroY, T, T, { flip: mouse.x < heroX + T / 2 });
      const sx = ox + (cols * 0.5 + Math.sin(t * 0.6) * cols * 0.3) * T;
      spr(`slime_${Math.floor(t * 4) % 3}`, sx, oy + (rows - 3) * T, T, T, { flip: Math.cos(t * 0.6) < 0 });
      const bx = W * 0.5 + Math.sin(t * 0.45) * W * 0.32;
      const by = oy + (3.2 + Math.sin(t * 1.3) * 0.6) * T;
      spr(`bat_${Math.floor(t * 8) % 2}`, bx, by, T, T, { flip: Math.cos(t * 0.45) < 0 });
      // magic gem glow
      const [gx, gy] = at(cx + 1, 2);
      addLight(gx + T / 2, gy + T / 2, R * 0.35, hgt, '#40d0ff', 0.6 + 0.3 * Math.sin(t * 2.2), null);
      // the lantern: mouse, or carried by the hero
      const lx = ctx.pointer.over ? mouse.x : heroX + T * 0.9;
      const ly = ctx.pointer.over ? mouse.y : heroY + T * 0.4;
      addLight(lx, ly, R * 1.1, hgt * 1.3, COLORS.lantern, 1.3, { size: T * 0.07 });
    }

    // neon street: a fixed pool of potential lights, first `count` are used
    let neonCache = null;
    function neonLayout(T, cols, rows) {
      const key = `${W}x${H}`;
      if (neonCache && neonCache.key === key) return neonCache;
      const R = rng(7);
      const top = 3 * T; // first road-side row (sidewalk) y
      const bot = (rows - 3) * T;
      const signs = [];
      for (let x = T * 1.5; x < W - T * 2; x += T * (4 + Math.floor(R() * 3))) {
        signs.push({ x, y: top - T * 0.3, w: T * (2 + R() * 1.5), hex: NEON[Math.floor(R() * NEON.length)], up: true });
      }
      for (let x = T * 3; x < W - T * 2; x += T * (4 + Math.floor(R() * 3))) {
        signs.push({ x, y: bot + T * 1.3, w: T * (2 + R() * 1.5), hex: NEON[Math.floor(R() * NEON.length)], up: false });
      }
      const strings = [];
      for (let x = T * 2; x < W; x += T * 5.5) {
        const pts = [];
        const n = 9;
        const x0 = x;
        const x1 = x + T * 2.5;
        for (let i = 0; i < n; i++) {
          const k = (i + 0.5) / n;
          pts.push({ x: x0 + (x1 - x0) * k, y: top + T + (bot - top - T) * k, hex: NEON[(i + strings.length) % NEON.length] });
        }
        strings.push({ x0, y0: top + T * 0.5, x1, y1: bot + T * 0.5, pts });
      }
      const floaters = [];
      for (let i = 0; i < MAX_LIGHTS; i++) {
        floaters.push({ x: R() * W, y: R() * H, sp: 0.2 + R() * 0.6, ph: R() * 10, hex: ['#ffb347', '#ffd27a', '#ff8a5c', '#ff6fae'][Math.floor(R() * 4)] });
      }
      neonCache = { key, top, bot, signs, strings, floaters };
      return neonCache;
    }

    function sceneNeon(t, p) {
      const S = tileScale(13);
      const T = 16 * S;
      const cols = Math.ceil(W / T);
      const rows = Math.ceil(H / T);
      const L = neonLayout(T, cols, rows);
      const roofs = ['tile_wood', 'tile_metal', 'tile_brick', 'tile_stone'];
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          let name = 'tile_stone';
          let color = '#7c86a8';
          if (r < 2 || r >= rows - 2) {
            name = roofs[Math.floor((c + (r < 2 ? 0 : 3)) / 5) % roofs.length];
            color = '#8a8098';
          } else if (r === 2 || r === rows - 3) {
            name = 'tile_metal';
            color = '#9aa2b4';
          }
          spr(name, c * T, r * T, T, T, { color });
        }
      // lane dashes
      const midY = (L.top + T + L.bot) / 2;
      for (let x = (-(t * 0) % (T * 2)); x < W; x += T * 2) spr('tile_ice', x, midY - T * 0.05, T, T * 0.1, { color: '#e8d27a' });
      const R = H * 0.3 * p.radius;
      const hgt = p.height * H;
      // 1. neon signs (3 lights each)
      for (const s of L.signs) {
        const flick = noise1(t * 3 + s.x, 5) > 0.08 ? 1 : 0.25;
        spr('tile_ice', s.x, s.y - T * 0.08, s.w, T * 0.16, { color: s.hex, emis: flick });
        for (let k = 0; k < 3; k++) addLight(s.x + s.w * (k + 0.5) / 3, s.y + (s.up ? T * 0.35 : -T * 0.35), R * 0.7, hgt * 0.6, s.hex, 1.2 * flick, null);
      }
      // 2. cars: body in albedo pass, head- and tail-lights
      shapes.begin();
      const lanes = [L.top + T * 1.6, L.bot - T * 0.6];
      for (let i = 0; i < 4; i++) {
        const dir = i % 2 ? -1 : 1;
        const lane = lanes[i % 2];
        const span = W + T * 6;
        const x = ((((t * T * (2.2 + i * 0.35) + i * span * 0.27) % span) + span) % span) - T * 3;
        const cxp = dir > 0 ? x : W - x;
        const body = ['#c23b4e', '#2f6fd6', '#e0b13a', '#e8e8f0'][i];
        shapes.box(cxp, lane, T * 0.9, T * 0.42, body, { radius: T * 0.18 });
        shapes.box(cxp + dir * T * 0.25, lane, T * 0.22, T * 0.34, '#1b2133', { radius: T * 0.06 });
        shapes.box(cxp - dir * T * 0.55, lane, T * 0.14, T * 0.32, '#1b2133', { radius: T * 0.05 });
        for (const side of [-1, 1]) {
          addLight(cxp + dir * T * 1.6, lane + side * T * 0.28, R * 0.55, hgt * 0.4, '#fff1cc', 1.6, { size: T * 0.07, at: [cxp + dir * T * 0.88, lane + side * T * 0.28] });
          addLight(cxp - dir * T * 1.0, lane + side * T * 0.3, R * 0.25, hgt * 0.3, '#ff2030', 1.2, { size: T * 0.06, at: [cxp - dir * T * 0.9, lane + side * T * 0.3] });
        }
      }
      // 3. lantern strings across the street
      for (const s of L.strings) {
        for (const q of s.pts) {
          const sway = Math.sin(t * 1.3 + q.x * 0.01) * T * 0.05;
          addLight(q.x + sway, q.y, R * 0.45, hgt * 1.4, q.hex, 0.9, { size: T * 0.11, string: s });
        }
      }
      // 4. floating sky lanterns fill the rest of the budget
      const fixed = lights.length;
      const want = Math.round(p.count);
      for (let i = 0; fixed + i < want && i < L.floaters.length; i++) {
        const f = L.floaters[i];
        const y = ((((f.y - t * T * f.sp * 0.6) % (H + T * 2)) + H + T * 2) % (H + T * 2)) - T;
        const x = f.x + Math.sin(t * 0.5 + f.ph) * T * 0.6;
        addLight(x, y, R * 0.4, hgt * 2, f.hex, 0.75 + 0.25 * Math.sin(t * 3 + f.ph), { size: T * 0.08 });
      }
      lights.length = Math.min(lights.length, want);
      return L;
    }

    // ---------------------------------------------------------------- frame
    return {
      resize(w, h) {
        alloc(w, h);
        neonCache = null;
      },
      onAction(key) {
        if (key === 'reset') dropped = [];
      },
      onExample() {
        dropped = [];
      },
      frame(ctx) {
        if (ctx.width !== W || ctx.height !== H) alloc(ctx.width, ctx.height);
        const p = ctx.params;
        const t = ctx.time;
        const enc = ctx.encoder;
        cam.setViewport(W, H);
        cam.x = W / 2;
        cam.y = H / 2;
        cam.zoom = 1;
        mouse = { x: ctx.pointer.x, y: ctx.pointer.y };
        if (ctx.example === 'falloff' && ctx.pointer.clicked && dropped.length < 24) {
          dropped.push({ x: mouse.x, y: mouse.y, hex: ['#ff4b4b', '#4bff6e', '#4b7bff', '#ffd23f', '#ff3fa4', '#38e8ff'][dropped.length % 6] });
        }

        cmds = [];
        lights = [];
        shapes.begin();
        let neon = null;
        if (ctx.example === 'dungeon') sceneDungeon(t, p);
        else if (ctx.example === 'neon') neon = sceneNeon(t, p);
        else sceneFalloff(t, p);

        // upload lights
        const n = Math.min(lights.length, MAX_LIGHTS);
        for (let i = 0; i < n; i++) {
          const L = lights[i];
          lightData.set([L.x, L.y, L.r, L.h, L.c[0], L.c[1], L.c[2], L.i], i * 8);
        }
        gpu.queue.writeBuffer(lightBuf, 0, lightData, 0, Math.max(8, n * 8));

        // 1. G-buffer: albedo + normals (same sprite list drawn twice)
        sprites.begin();
        submit();
        sprites.flush(enc, albedoT, cam, { clear: [0, 0, 0, 1], extra: [0, 0, 0, 0] });
        shapes.flush(enc, albedoT, cam);
        submit();
        sprites.flush(enc, normalT, cam, { clear: [0.5, 0.5, 1, 0], extra: [1, 0, 0, 0], texture: normView, blend: 'none' });

        // 2. tiled light culling, 3. lighting
        const tiled = ctx.example !== 'neon' || p.tiled;
        const ambBase = ctx.example === 'dungeon' ? [0.35, 0.45, 1.0] : ctx.example === 'neon' ? [0.45, 0.4, 1.0] : [1, 1, 1];
        const amb = p.ambient * p.ambient * 2.2;
        const viewIdx = { final: 0, albedo: 1, normals: 2, light: 3, heat: 4 }[p.view] ?? 0;
        U.setAll({
          resolution: [W, H],
          tilesX,
          tilesY,
          ambient: ambBase.map((v) => v * amb),
          count: n,
          curve: { linear: 0, invsq: 1, smooth: 2 }[p.curve] ?? 2,
          normalsOn: p.normals ? 1 : 0,
          specular: p.specular,
          shininess: 28,
          view: viewIdx,
          tiled: tiled ? 1 : 0,
          exposure: ctx.example === 'neon' ? 1.25 : 1.1,
          time: t,
        });
        U.upload();
        if (tiled || viewIdx === 4) cull.dispatch(enc, 'cull', [tilesX, tilesY], { u: U, lights: lightBuf, tiles: tileBuf });
        const canvasT = { view: ctx.target, format: gpu.format };
        lighting.draw(enc, canvasT, { albedo: albedoT, normals: normalT, lights: lightBuf, tiles: tileBuf });

        // 4. light sources as glows (+ lantern strings, unlit, hanging above the street)
        glows.begin();
        if (viewIdx === 0 || viewIdx === 3) {
          if (neon) {
            for (const s of neon.strings) glows.line(s.x0, s.y0, s.x1, s.y1, Math.max(1, H * 0.0025), [0.05, 0.05, 0.08, 0.9]);
            glows.flush(enc, canvasT, cam, { blend: 'alpha' });
          }
          for (const L of lights) {
            if (!L.glow) continue;
            const [gx, gy] = L.glow.at || [L.x, L.y];
            const col = [Math.pow(L.c[0], 1 / 2.2), Math.pow(L.c[1], 1 / 2.2), Math.pow(L.c[2], 1 / 2.2), 0.9];
            glows.circle(gx, gy, L.glow.size, col, { glow: L.glow.size * 2.2, glowStrength: 0.55 * Math.min(1.2, L.i) });
            glows.circle(gx, gy, L.glow.size * 0.5, [1, 1, 1, 0.8]);
          }
          glows.flush(enc, canvasT, cam, { blend: 'additive' });
        }

        // falloff graph (abstract tab)
        if (ctx.example === 'falloff') {
          ui.begin();
          const gw = Math.min(W * 0.3, H * 0.5);
          const gh = gw * 0.55;
          const m = Math.round(8 * ctx.dpr);
          const gx = W - gw - m;
          const gy = H - gh - m - Math.round(36 * ctx.dpr);
          ui.rect(gx, gy, gw, gh, [0.02, 0.03, 0.06, 0.78], { radius: 6 * ctx.dpr });
          const px0 = gx + gw * 0.08;
          const py0 = gy + gh * 0.88;
          const pw = gw * 0.86;
          const ph = gh * 0.74;
          ui.line(px0, py0, px0 + pw, py0, ctx.dpr, [1, 1, 1, 0.35]);
          ui.line(px0, py0, px0, py0 - ph, ctx.dpr, [1, 1, 1, 0.35]);
          const sel = { linear: 0, invsq: 1, smooth: 2 }[p.curve] ?? 2;
          const cols = [[0.55, 0.75, 1, 1], [1, 0.55, 0.35, 1], [0.5, 1, 0.6, 1]];
          for (let c = 0; c < 3; c++) {
            const pts = [];
            for (let i = 0; i <= 48; i++) {
              const x = i / 48;
              pts.push([px0 + x * pw, py0 - falloffJS(c, x) * ph]);
            }
            const col = c === sel ? cols[c] : [cols[c][0], cols[c][1], cols[c][2], 0.28];
            ui.polyline(pts, (c === sel ? 2.2 : 1.2) * ctx.dpr, col);
          }
          ui.flush(enc, canvasT, cam, { blend: 'alpha' });
          legend.style.display = '';
          legend.style.bottom = `${Math.round((H - gy) / ctx.dpr) + 4}px`;
          legend.innerHTML = `brightness ↑ vs distance → &nbsp;<span style="color:${['#8cbfff', '#ff8c5a', '#80ff99'][sel]}">${['linear', 'inverse-square', 'smooth'][sel]}</span>`;
        } else legend.style.display = 'none';

        readout.textContent =
          ctx.example === 'neon'
            ? `${n.toLocaleString()} lights · ${tiled ? `tiled: ${tilesX}×${tilesY} tiles of ${TILE}px` : 'brute force: every light × every pixel'}`
            : `${n} lights · ${['linear', 'inverse-square', 'smooth'][{ linear: 0, invsq: 1, smooth: 2 }[p.curve] ?? 2]} falloff · normal maps ${p.normals ? 'on' : 'off'}`;
      },
    };
  },
};

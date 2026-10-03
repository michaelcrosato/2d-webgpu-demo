# Authoring scenes for the 2D GPU Showcase

This is the contract every scene follows. Read it fully before writing a scene.

- Scenes live in `src/scenes/<category>/<id>.js` and are listed in `src/scenes/registry.js`.
- A scene module `export default`s a **scene object**: metadata, controls, examples, explanation,
  plus `init(ctx)` (WebGPU) and optionally `initGL(ctx)` (WebGL2).
- The fastest way to write a fragment-shader-only scene is **`shaderScene()`**. It runs on **both**
  WebGPU and WebGL2 from one portable-WGSL source.
- Anything with compute shaders, storage buffers, instancing or custom pipelines is a **custom WebGPU
  scene** (`init(ctx)` only).

Good references: `src/scenes/start/shader-basics.js` (shaderScene with examples), `src/scenes/start/gpu-pipeline.js`
(shaderScene with mouse dragging via `bind()` and an atlas texture), `src/scenes/start/compute-power.js`
(custom WebGPU scene: compute + storage buffers + HDR + overlay readout).

---

## 1. The scene object

```js
export default {
  // shown as a hint chip in the bottom-left corner of the canvas
  interaction: 'Click to spawn an explosion. Drag to aim.',

  // 2–4 example tabs. Give at least one abstract view AND one in-game / real-life use.
  // kind is a small label: 'Abstract' | 'In a game' | 'Real life' | 'Classic' | 'Comparison' | 'Step 1' …
  examples: [
    { id: 'abstract', label: 'Raw noise', kind: 'Abstract', note: 'HTML shown above the explanation when selected.' },
    { id: 'terrain', label: 'Terrain map', kind: 'In a game', note: '…', params: { scale: 3 } /* preset values */ },
  ],

  // sliders etc. `key` becomes ctx.params[key]. `help` is shown under the control (keep it short).
  controls: [
    { type: 'heading', label: 'Shape' },
    { type: 'slider', key: 'scale', label: 'Scale', min: 1, max: 20, step: 0.1, value: 4, help: 'Zoom of the noise.' },
    { type: 'slider', key: 'count', label: 'Count', min: 100, max: 1e6, step: 100, value: 1e4, log: true,
      format: (v) => Math.round(v).toLocaleString() },
    { type: 'toggle', key: 'animate', label: 'Animate', value: true },
    { type: 'select', key: 'kind', label: 'Noise type', value: 'perlin',
      options: [{ value: 'value', label: 'Value' }, { value: 'perlin', label: 'Perlin' }] },
    { type: 'color', key: 'tint', label: 'Tint', value: '#ff8800' },
    { type: 'button', key: 'reset', label: 'Reset simulation', primary: true },   // -> instance.onAction('reset')
    { type: 'info', label: 'Free <b>HTML</b> note in the panel.' },
    { type: 'slider', key: 'octaves', label: 'Octaves', min: 1, max: 8, step: 1, value: 5, showFor: ['terrain'] },
  ],

  about: { /* see §6 — the educational content */ },

  // optional flags
  keys: true,             // prevent arrow/space/WASD from scrolling the page while the canvas is hovered
  wheel: true,            // capture the mouse wheel (ctx.pointer.wheel), e.g. for zooming
  reinitOnExample: false, // true = destroy & re-init the scene whenever the example tab changes
  glAntialias: false,     // WebGL2 only: request an MSAA default framebuffer

  async init(ctx) { /* WebGPU */ return instance; },
  async initGL(ctx) { /* optional WebGL2 */ return instance; },
};
```

The **instance** returned by `init` may implement:

| method | when |
|---|---|
| `frame(ctx)` | every animation frame (required) |
| `resize(w, h, ctx)` | canvas size changed (recreate size-dependent targets) |
| `onChange(key, value, ctx)` | a control changed (rebuild buffers if e.g. a count changed) |
| `onAction(key, ctx)` | a button was pressed (`reset` is also wired to the ↻ toolbar button) |
| `onExample(id, ctx)` | the example tab changed (preset params are applied *before* this) |
| `destroy()` | scene unloaded (GPU objects created via `ctx.gpu` are freed automatically) |

### `ctx` (the same object every frame, fields updated in place)

| field | meaning |
|---|---|
| `backend` | `'webgpu'` or `'webgl2'` |
| `gpu` | `SceneGPU` helper (WebGPU) — see §3 |
| `gl`, `glkit` | WebGL2 context and `GLKit` helper (WebGL2 only) |
| `canvas`, `width`, `height`, `dpr` | canvas & its size in **device pixels** |
| `params` | live control values (`ctx.params.scale`) |
| `example`, `exampleIndex` | current example id / index |
| `pointer` | `{x, y}` in canvas px (top-left origin), `nx, ny` (0..1), `down`, `clicked` (this frame), `released`, `button` (0 left, 2 right), `over`, `dx, dy` (movement this frame), `wheel` |
| `keys` / `keysPressed` | `Set` of held keys / keys pressed this frame (`'a'`, `'ArrowLeft'`, `'Space'`, `'KeyA'`…) |
| `time`, `dt`, `frame` | seconds (pausable, speed-scaled), delta seconds, frame count |
| `paused` | true when paused — **do not advance simulations**; `frame()` is still called when params change so the view updates |
| `testMode` | true under the headless test harness — use it to shrink heavy workloads (e.g. 50k particles instead of 2M) |
| `encoder`, `target`, `targetTexture` | WebGPU per-frame: the command encoder (submitted for you) and the canvas texture view |
| `overlay` | a `<div>` over the canvas for live readouts. Add children with class `tag` and absolute position. Keep clear of the bottom-left (hint chip) and top-right (toolbar): e.g. `style.cssText = 'right:8px;bottom:8px'` |
| `status(text)` | show a toast |

Coordinates everywhere: **origin top-left, y down**, in device pixels unless noted.

---

## 2. `shaderScene()` — fragment-shader scenes that run on WebGPU *and* WebGL2

```js
import { shaderScene } from '../../core/shaderscene.js';

export default shaderScene({
  examples, controls, about, interaction,
  uniforms: { scale: 'f32', tint: 'vec3f', mode: 'f32' }, // auto-filled from controls with the SAME key
  include: ['noise', 'sdf', 'color'],                       // shader libraries (§5)
  input: 'game',            // optional: provides texture `game` = the procedural platformer scene, re-rendered each frame
  textures: {               // optional extra textures, loaded once
    atlas: { source: async () => (await getAtlas()).canvas, filter: 'nearest' },
  },
  passes: [                 // optional Shadertoy-style buffers (see below)
    { name: 'state', format: 'rgba16float', size: [256, 256], iterations: 'steps', code: `fn shade(uv: vec2f, px: vec2f) -> vec4f { … }` },
  ],
  bind(params, ctx) { return { mode: params.kind === 'x' ? 1 : 0 }; }, // optional: extra/derived uniform values, JS state
  resetOn: ['seed'],        // optional: param keys that clear the pass buffers
  renderScale: 'pixelSize', // optional: number | param key | (params) => number. Final image rendered smaller and nearest-upscaled
  gl: false,                // optional: set if the scene can't be translated (WebGPU only)
  code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let n = fbm(uv * u.scale + vec2f(u.time * 0.1, 0.0), 5);
  return vec4f(u.tint * (n * 0.5 + 0.5), 1.0);
}`,
});
```

- Your code defines `fn shade(uv: vec2f, px: vec2f) -> vec4f`. `uv` is 0..1 (top-left origin), `px` is the
  pixel coordinate in the **current render target** (for the final image = canvas pixels).
- **Uniforms** are read as `u.<name>`. Built-ins always available:
  `u.resolution` (canvas px), `u.time`, `u.frame` (frames since start/reset), `u.dt`,
  `u.mouse` (xy = px, z = 1 if pressed, w = 1 if over canvas), `u.pmouse` (previous xy), `u.example` (index of the example tab, as f32).
- **Auto-binding** of controls to uniforms with the same key: slider → value, toggle → 0/1,
  select → option *index* (or the value itself if it’s a number), color → vec3f (sRGB 0..1).
  Use `bind()` for anything derived (and for JS-side state such as dragging handles — see `gpu-pipeline.js`).
- **Textures** are module-level variables named after the pass / `game` / your `textures` keys.
  Read them only through the helpers `TEX(t, uv)` (linear, clamp), `TEXR(t, uv)` (repeat), `TEXN(t, uv)` (nearest),
  `LOAD(t, vec2i(p))` (exact texel, clamped), `LOADW(t, vec2i(p))` (wrapping), `TEXSIZE(t)` (vec2f size).
- **Passes** (multi-pass / feedback): every pass is double-buffered. Inside any pass, reading texture
  `state` returns the **previous** output of pass `state` (its own previous frame if it is the current pass,
  or this frame's result for passes that already ran). Passes run in order, then the final `code` image.
  - `size: [w, h]` fixed, or `scale: 0.5` relative to the canvas (default 1), or `size: (params, ctx) => [w, h]`.
  - `format`: `'rgba16float'` (default, filterable), `'rgba8unorm'`, or `'rgba32float'` (**LOAD only**, no TEX).
  - `iterations`: number, param key, or `(params) => n` — run the pass several times per frame (sim speed).
  - Initialise state when `u.frame < 0.5` (first frame after start/reset/resize). The ↻ button and `reset` buttons clear buffers.
  - Passes do not run while paused (the final image still does).
- `input: 'game'` gives you `game`, an animated 2D platformer scene (sky, parallax mountains, trees, tiles, coins, torches, a hopping hero).
  Perfect raw material for post-processing and style filters.

### Portable WGSL rules (so the GLSL translation works)

The WGSL is translated to GLSL ES 3.00 by `src/core/wgsl2glsl.js`. Stay inside this subset:

1. **Float literals need a dot**: write `x * 2.0`, `pow(x, 2.0)`, `vec2f(1.0, 0.0)`. WGSL accepts `x * 2`; GLSL rejects it. Convert ints explicitly: `f32(i)`.
2. `if (…)` / `while (…)` / `for (…)` with parentheses. **No** `switch`, `loop`, `break if`, pointers, `%` on floats
   (use `fmod(x, y)`, `fmod2`, `fmod3`). `%` on integers is fine.
3. `select(falseVal, trueVal, cond)` only with a **scalar** bool. For vectors use `mix(a, b, step(edge, x))`.
4. Textures only via the `TEX/TEXR/TEXN/LOAD/LOADW/TEXSIZE` helpers.
5. Don't name variables/functions after GLSL keywords or built-ins: `input output sample filter texture mod mix step
   smooth active common buffer shared half fixed cross dot length distance normalize reflect sign round floor fract
   target set get type` … (also WGSL reserved words like `target`, `filter`, `mod`, `set`, `get`, `type`, `match`, `self`, `pass`, `ref`, `of`).
6. `let x = expr;` types are inferred in most cases; if the translator can't, the overlay shows
   `cannot infer the type of "x"` — add an explicit type: `let x: vec2f = …`.
7. Arrays: `var a = array<f32, 4>(1.0, 2.0, 3.0, 4.0);` (indexing with a variable requires `var`, not `let`).
8. No vector comparisons (`a < b` with vectors) — compare components. No `@attributes`, no `var<storage>` etc.
9. WGSL is strict about matching types: `clamp(v, vec3f(0.0), vec3f(1.0))`, `max(v, vec2f(0.0))`, `mix(a, b, t)` is fine with scalar `t`.
10. `PI`, `TAU`, `fmod*` are predefined; `saturate()` exists in both.
11. Integer loops: `for (var i = 0; i < 8; i++) { … f32(i) … }`. Bounded loops are best (GLSL drivers like constant bounds); break early with `if (i >= n) { break; }`.

The headless check (§8) compiles **both** versions; GLSL errors show the offending translated line.

---

## 3. Custom WebGPU scenes — `ctx.gpu` (SceneGPU, `src/core/webgpu.js`)

Everything created through `ctx.gpu` is destroyed automatically when the scene unloads.

```js
async init(ctx) {
  const gpu = ctx.gpu;                                         // gpu.device, gpu.queue, gpu.format (canvas format)
  const U = gpu.uniforms({ time: 'f32', count: 'u32', color: 'vec3f' }, 'Params'); // UniformBlock + buffer
  const buf = gpu.storage(N * 16, 'particles');                // storage buffer (or pass a TypedArray)
  const prog = gpu.program({                                   // generates @binding declarations for you
    label: 'update',
    bindings: {
      u: { uniform: U },                                       // struct Params is generated, bound as `u`
      ps: { storage: 'array<Particle>', access: 'read_write' },
      src: { texture: true },                                  // texture_2d<f32>
      samp: { sampler: true },
      outTex: { storageTexture: 'rgba8unorm', access: 'write' },
    },
    include: ['noise'],
    code: `struct Particle { pos: vec2f, vel: vec2f };
           @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) { … }`,
  });
  return {
    frame(ctx) {
      U.set('time', ctx.time).set('count', N); U.upload();
      prog.dispatch(ctx.encoder, 'main', Math.ceil(N / 64), { u: U, ps: buf, src: someTarget, samp: 'linear', outTex: otherTarget });
    },
  };
}
```

Helpers:

| helper | notes |
|---|---|
| `gpu.uniforms(fields, StructName)` | `UniformBlock`: `.set(name, v)` (chainable; numbers, arrays, `'#hex'`), `.setAll(obj)`, `.upload()`. Types: f32 i32 u32 vec2f vec3f vec4f vec2i vec4i vec2u vec4u mat4x4f `array<vec4f, N>`. Field names must not be reserved words (it throws). |
| `gpu.storage(sizeOrTypedArray, label)` | STORAGE \| COPY_SRC \| COPY_DST buffer |
| `gpu.buffer({size, usage, data, label})` / `gpu.vertexBuffer(data)` | generic buffers |
| `gpu.target(w, h, {format, mips})` | `{texture, view, width, height, format}` — render attachment + sampled + (if format allows) storage |
| `gpu.pingPong(w, h, opts)` | `{read, write, swap()}` of two targets |
| `gpu.sampler(name)` | `'linear' 'nearest' 'linear-repeat' 'nearest-repeat' 'linear-mirror' 'linear-mip' 'linear-mip-repeat' 'aniso'` |
| `gpu.textureFromImage(canvasOrBitmap, {mips})` | upload an image; returns GPUTexture (`.createView()`) |
| `gpu.textureFromData(w, h, data, {format})` | raw pixels |
| `gpu.program({label, bindings, include, code})` | → `Program`: `.bind(resources)` (cached bind groups), `.renderPipeline({format, blend, vs, fs, topology, buffers, targets})`, `.computePipeline(entry)`, `.dispatch(encoder, entry, groups, resources)` |
| `gpu.fullscreen({label, code, uniforms, textures:['src'], include, blend, unfilterable:[]})` | fullscreen pass; `code` defines `fn shade(uv, px) -> vec4f` (or its own `@fragment fn fs_main(in: VSOut)`); `.draw(encoder, target, {src: viewOrTarget}, {clear, blend})`. `u`, `samp`, `sampR` and `TEX…` helpers available. |
| `gpu.blit(encoder, src, dst, {filter, clear, blend})` | copy/scale a texture |
| `gpu.clear(encoder, target, color)` | |
| `gpu.shader(code, label)` | raw module with readable error reporting |

Drawing to the canvas: the canvas view is `ctx.target`; its format is `gpu.format` (usually `bgra8unorm`), so pass
`{ view: ctx.target, format: gpu.format }` to helpers that need to know the format.

Blend presets (`blend:` option): `'alpha' 'premultiplied' 'additive' (src-alpha, one) 'add' (one, one) 'multiply' 'screen' 'subtract' 'max' 'min'`.

### WebGPU gotchas (read these!)

1. **`queue.writeBuffer` runs before the frame's commands.** Writing a uniform buffer twice in one frame means every pass sees the
   last value. Use one UniformBlock per pass that needs different values (or `ShapeBatch`/`SpriteBatch`, which handle this).
2. A texture can't be sampled **and** rendered/written in the same pass → ping-pong.
3. Storage buffers with `read_write` can't be bound in a **vertex** shader. Make a second `program` for drawing that declares the same buffer with `access: 'read'`.
   Atomics: declare `array<atomic<u32>>` in the compute program and `array<u32>` in the read-only draw program.
4. `rgba32float` / `r32float` textures are **not filterable** (no linear sampling) — use `rgba16float`, or `textureLoad`. Declare them `{ texture: 'unfilterable-float' }`.
5. Pipeline target format must match the attachment: canvas = `gpu.format`; `gpu.target()` default = `rgba8unorm`.
6. Max 256 threads per workgroup; dispatch `Math.ceil(n / 64)` groups for `@workgroup_size(64)`; guard `if (i >= count) { return; }`.
7. WGSL `let` arrays can't be indexed with a runtime value — use `var`.
8. WGSL reserved words bite as names: `target filter sample mod set get type match self pass ref of meta from use do` …
9. `PI`, `TAU`, `fmod`, `fmod2`, `fmod3` are predefined in every `program()`/`fullscreen()` (don't redefine them; pass `prelude: false` to `program()` to opt out).
10. Every generated module starts with `diagnostic(off, derivative_uniformity);` so `fwidth`/`textureSample` in branches compile.
11. Keep heavy work proportional to `ctx.testMode` (the test harness uses a slow software GPU).
12. When `ctx.paused`, skip simulation steps but still draw.

---

## 4. 2D drawing helpers (`src/core/batch.js`, WebGPU)

```js
import { Camera2D, ShapeBatch, SpriteBatch, rgba } from '../../core/batch.js';
const cam = new Camera2D();               // world units = pixels by default; .x/.y = world point at screen center, .zoom, .rotation
const shapes = new ShapeBatch(gpu);       // anti-aliased SDF shapes with optional glow
const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest', fragment: optionalWGSL });

frame(ctx) {
  cam.setViewport(ctx.width, ctx.height);            // first call centers the camera so world == screen px
  shapes.begin(); sprites.begin();                    // once per frame
  shapes.circle(x, y, r, '#ff8800', { glow: 12, glowStrength: 0.6, stroke: 0 });
  shapes.line(x1, y1, x2, y2, width, [r, g, b, a]);
  shapes.rect(x, y, w, h, color, { radius: 8, rotation: 0.3, stroke: 2 });   // top-left x,y
  shapes.box(cx, cy, halfW, halfH, color, opts);       // center-based
  shapes.triangle(x1, y1, x2, y2, x3, y3, color);
  shapes.polyline(points, width, color, { closed: true });
  shapes.flush(ctx.encoder, { view: ctx.target, format: gpu.format }, cam, { clear: [0, 0, 0, 1], blend: 'alpha' });
  sprites.draw(x, y, w, h, { uv: atlas.uv('hero_run_0'), rotation, anchor: [0.5, 1], color: '#ffffff', flipX, alpha, user });
  sprites.flush(ctx.encoder, target, cam, { blend: 'alpha', texture: otherView, extra: [a, b, c, d] });
}
```

- Multiple flushes per frame are fine (e.g. a second flush with `blend: 'additive'` for glows). Each `flush` is its own render pass, drawing on top (`clear` only if you pass it).
- Custom sprite effects: pass `fragment` WGSL defining `fn sprite_fs(t: vec4f, i: VOut) -> vec4f` where `i.uv` (atlas uv), `i.local` (0..1 within the sprite),
  `i.color` (tint), `i.extra` (x = rotation, y/z = anchor, **w = your `user` value**), `i.world` (world position); `cam.extra` = the flush's `extra` vec4,
  `cam.viewport.w` = seconds. `tex`/`samp` are available for extra lookups (`textureSample(tex, samp, uv)`).
- `cam.screenToWorld(px, py)` / `cam.worldToScreen(x, y)`.
- For 100k+ objects, don't call `draw()` per object per frame from JS — write instance data with a compute shader and draw it with your own pipeline (see `compute-power.js`).

---

## 5. Shader libraries (`src/core/shaderlib.js`) — `include: [...]`

All written in portable WGSL (usable from both `program()`/`fullscreen()` and `shaderScene()`).

- **math**: `rot2(a) -> mat2x2f` (`rot2(a) * p`), `remap(x,a,b,c,d)`, `remap01(x,a,b)`, `centerUV(px, res)` (centered, aspect-correct, y in −0.5..0.5, y down), `easeInOut`, `easeOutBack`, `easeOutElastic`.
- **hash**: `hash11 hash12 hash13 hash21 hash22 hash23 hash31 hash33` (hashNM: N floats in → M out, [0,1)), `pcg(u32)`, `pcg3d(vec3u)`, `ign(px)` (interleaved gradient noise for dithering).
- **noise** (needs hash, math — auto): `valueNoise(p)` [0,1], `perlin(p)` / `simplex(p)` / `perlin3(p3)` ~[−1,1], `fbm(p, octaves)`, `fbmEx(p, oct, lacunarity, gain)`, `fbm3(p3, oct)`, `ridged(p, oct)`,
  `voronoi(p) -> vec4f(F1, F2, cellId.xy)`, `voronoiEx(p, jitter, t)`, `voronoiBorder(p, jitter, t) -> vec3f(borderDist, cellId)`, `curl(p) -> vec2f`.
- **sdf**: `sdCircle sdBox sdRoundBox sdSegment sdTriangle sdEquilateralTriangle sdHexagon sdStar5 sdHeart sdRhombus sdArc sdPie sdVesica sdMoon sdCross sdEllipseApprox sdBezier`,
  `opSmoothUnion opSmoothSubtract opSmoothIntersect opOnion`, `sdfFill(d)` (AA coverage), `sdfStroke(d, width)`, `sdfGlow(d, radius)`.
- **color**: `luma hsv2rgb rgb2hsv srgbToLinear linearToSrgb linearToOklab oklabToLinear mixOklab palette(t,a,b,c,d) rainbow tonemapACES tonemapReinhard adjustSaturation adjustContrast hueRotate blendScreen blendOverlay blendSoftLight`.
- **dither**: `bayer2 bayer4 bayer8 bayer16` (pass pixel coords, returns threshold in [0,1)).

---

## 6. Assets & the game scene

- `import { getAtlas, PALETTES, PAL, makeCanvas, rng, textures } from '../../core/assets.js'`
  - `const atlas = await getAtlas()` → `atlas.canvas` (pixel art, 512×N), `atlas.normalCanvas` (matching normal map),
    `atlas.uv(name)` → `[u0, v0, u1, v1]`, `atlas.frames[name]` → `{x, y, w, h}` px, `atlas.anim('hero_run')` → frame names, `atlas.names()`.
  - Frames: `hero_idle_0..1 hero_run_0..3 hero_jump_0 hero_hurt_0 slime_0..2 bunny_0..1 bat_0..1 coin_0..7 gem heart potion key chest crate mushroom star torch_0..2 bomb sword shield`,
    tiles (16×16, seamless) `tile_grass tile_dirt tile_sand tile_leaves tile_stone tile_brick tile_wood tile_ice tile_metal tile_water_0..3 tile_lava_0..3`,
    scenery `tree (32×48) bush flower cloud (32×16) rock`, UI `ui_panel (24×24, 8px border for nine-slice) ui_button_0..1`,
    soft particles (32×32, white, smooth) `p_soft p_spark p_star p_smoke p_ring`.
  - `PALETTES`: `sweetie16 gameboy pico8 nes cga mono obra vaporwave sepia` (arrays of `'#rrggbb'`).
  - `rng(seed)` deterministic random; `makeCanvas(w, h)`; `textures.softCircle(size)` etc.
- Procedural game scene: in `shaderScene` use `input: 'game'`; in custom WebGPU code:
  `import { createGameScene } from '../../core/gamescene.js'; const game = createGameScene(gpu); const tgt = game.render(ctx.encoder, ctx.time, w, h);`
- Make your own illustrations with Canvas2D (`makeCanvas`) when you need a "photo" or specific art — no external images.

---

## 7. The educational content (`about`) — this matters as much as the visuals

The audience is a game-curious person who wants to know **what is possible** and **what to ask for**.
Plain English, short paragraphs, explain jargon the first time. HTML strings are allowed.

```js
about: {
  summary: 'One or two sentences shown under the title.',
  what: '<p>What the viewer is looking at right now.</p>',
  how: '<ol><li>Step-by-step: the actual technique, the passes, the math intuition.</li></ol>',
  uses: [{ title: 'Platformers', text: 'Where/how games use it — name real games when well known.' }, …],
  try: ['Concrete slider experiments: “Drag Octaves to 1 and see …”', …],
  ask: ['short phrases a person can paste into a request, e.g. “normal-mapped 2D lighting with soft shadows”', …],
  perf: '<p>What it costs and what scales the cost.</p>',
  api: '<p>WebGPU vs WebGL2 notes for this technique (why compute helps, or that it works the same).</p>',
  code: [{ title: 'Core idea', lang: 'wgsl', src: `…the key 5–25 lines, real code from the scene…` }],
  links: [{ title: 'Well-known reference', url: 'https://…', note: 'optional' }],   // optional; only stable, famous URLs
}
```

Quality bar: each scene should have 2–4 example tabs (abstract + in-game/real-life), 4–10 meaningful
controls with `help`, 3–5 "try this" items, 3–6 "ask for it" phrases, and code taken from the real implementation.
It should look **good** — this is a showcase. Use pleasing palettes, anti-aliasing, motion, and sensible defaults.

---

## 8. Testing

```bash
node tools/check.mjs <scene-id> [more ids] --shots      # every example × every API, screenshots → tools/out/
node tools/check.mjs --category=post --first             # first example only
```

The harness runs headless Chromium with a **software** GPU (SwiftShader): slow (expect 2–20 fps), but it catches WGSL/GLSL
compile errors, WebGPU validation errors, JS exceptions and blank canvases. It drags the mouse across the canvas once.
Look at the screenshots in `tools/out/` (they are PNGs you can open) to check that it actually looks right.
A scene is done when every example passes on every API it claims, and the screenshots look intentional.

Local dev: `node tools/serve.mjs` → http://localhost:8080 (a real browser shows real performance).

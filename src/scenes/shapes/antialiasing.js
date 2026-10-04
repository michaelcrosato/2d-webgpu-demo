// Anti-aliasing & Pixel Scaling — a custom WebGPU scene.
//  * the same wagon wheel rendered 4 ways: no AA, 4× MSAA (sampleCount 4 + resolve),
//    analytic SDF coverage, and supersampling (render k× larger, box-filter down) + magnifiers
//  * pixel-art sprites at non-integer zoom: nearest vs linear vs "sharp bilinear"
//  * mipmaps vs no mipmaps vs anisotropic filtering on a receding textured floor

import { Camera2D, SpriteBatch } from '../../core/batch.js';
import { getAtlas, makeCanvas } from '../../core/assets.js';
import { labels } from './_shared.js';

const LBL = 'background:#000b;font-size:11px;padding:2px 7px;color:#e2e8f0';

// ---------------------------------------------------------------- segment rendering
// instance: a.xy b.xy | halfWidth, 0, 0, 0 | color rgba
const SEG_WGSL = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) w: vec2f,
  @location(1) @interpolate(flat) ab: vec4f,
  @location(2) @interpolate(flat) hw: f32,
  @location(3) @interpolate(flat) color: vec4f,
};
fn toClip(p: vec2f) -> vec4f { return vec4f(p / u.resolution * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0); }
fn corner(vi: u32) -> vec2f {
  var c = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  return c[vi];
}
// exact rectangle: what a plain triangle rasterizer sees (butt ends)
@vertex fn vs_tri(@builtin(vertex_index) vi: u32, @location(0) ab: vec4f, @location(1) prm: vec4f, @location(2) color: vec4f) -> VOut {
  let d = normalize(ab.zw - ab.xy);
  let n = vec2f(-d.y, d.x);
  let c = corner(vi);
  let w = mix(ab.xy, ab.zw, c.x) + n * c.y * prm.x;
  var o: VOut;
  o.pos = toClip(w); o.w = w; o.ab = ab; o.hw = prm.x; o.color = color;
  return o;
}
@fragment fn fs_tri(i: VOut) -> @location(0) vec4f { return vec4f(i.color.rgb, 1.0); }

// expanded quad + analytic coverage from the distance to the segment
@vertex fn vs_sdf(@builtin(vertex_index) vi: u32, @location(0) ab: vec4f, @location(1) prm: vec4f, @location(2) color: vec4f) -> VOut {
  let d = normalize(ab.zw - ab.xy);
  let n = vec2f(-d.y, d.x);
  let c = corner(vi);
  let e = max(prm.x, 0.5) + 1.5;
  let w = mix(ab.xy - d * 1.5, ab.zw + d * 1.5, c.x) + n * c.y * e;
  var o: VOut;
  o.pos = toClip(w); o.w = w; o.ab = ab; o.hw = prm.x; o.color = color;
  return o;
}
fn sdSeg(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}
@fragment fn fs_sdf(i: VOut) -> @location(0) vec4f {
  let dist = sdSeg(i.w, i.ab.xy, i.ab.zw);
  // lines thinner than a pixel: draw 1px wide but fainter (keeps the same "ink")
  let hwE = max(i.hw, 0.5);
  let cov = clamp(hwE + 0.5 - dist, 0.0, 1.0) * min(i.hw / 0.5, 1.0);
  if (cov <= 0.002) { discard; }
  return vec4f(i.color.rgb, cov);
}`;

// ---------------------------------------------------------------- compose (panels + magnifiers)
const COMPOSE_WGSL = /* wgsl */ `
fn fetch(p: vec2f, k: i32, origin: vec2f) -> vec3f {
  let ip = vec2i(floor(p));
  if (i32(u.mode) == 0) {
    if (k == 1) { return LOAD(B, ip).rgb; }
    if (k == 3) {
      // supersampled panel: average f x f texels of the big target (a box filter)
      let f = i32(u.ssaa);
      let base = vec2i(floor(p - origin)) * f;
      var acc = vec3f(0.0);
      for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
          if (i < f && j < f) { acc += LOAD(S, base + vec2i(i, j)).rgb; }
        }
      }
      return acc / f32(f * f);
    }
  }
  return LOAD(A, ip).rgb;
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ps = u.resolution / u.grid;
  let cell = min(floor(px / ps), u.grid - vec2f(1.0));
  let k = i32(cell.y * u.grid.x + cell.x);
  let origin = cell * ps;
  let lp = px - origin;
  let ins = u.inset;
  let rel = lp - (ins.xy + ins.zw * 0.5);
  let inInset = abs(rel.x) < ins.z * 0.5 && abs(rel.y) < ins.w * 0.5;
  var sp = px;
  if (inInset) { sp = origin + u.src + rel / u.zoom; }
  var c = fetch(sp, k, origin);
  // magnifier frame and the outline of the magnified region
  let fr = max(abs(rel.x) - ins.z * 0.5, abs(rel.y) - ins.w * 0.5);
  c = mix(c, vec3f(0.95), clamp(1.5 - abs(fr + 1.0), 0.0, 1.0));
  let srel = lp - u.src;
  let hs = ins.zw * 0.5 / u.zoom;
  let sf = max(abs(srel.x) - hs.x, abs(srel.y) - hs.y);
  if (!inInset) { c = mix(c, vec3f(1.0, 0.85, 0.3), clamp(1.2 - abs(sf), 0.0, 1.0) * 0.9); }
  // panel separators
  let e = min(min(lp.x, ps.x - lp.x), min(lp.y, ps.y - lp.y));
  c = mix(c, vec3f(0.01, 0.01, 0.02), clamp(2.0 - e, 0.0, 1.0));
  return vec4f(c, 1.0);
}`;

// ---------------------------------------------------------------- mipmaps floor
const FLOOR_WGSL = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
  return o;
}
@fragment fn fs_main(i: VSOut) -> @location(0) vec4f {
  let px = i.pos.xy;
  let res = u.resolution;
  let pw = res.x / 3.0;
  let k = min(floor(px.x / pw), 2.0);
  let lx = px.x - k * pw - pw * 0.5;
  let horizon = res.y * u.horizon;
  // perspective floor: depth from the screen row, x scaled by depth (Mode-7 style)
  let dy = max(px.y - horizon, 0.5);
  let z = res.y * 0.5 / dy;
  var tuv = vec2f(lx / res.y * z * 2.0, z + u.time * u.speed) * u.scale;
  tuv = rot2(u.angle) * tuv;
  var c: vec4f;
  if (k < 0.5) { c = textureSampleLevel(tex, sNoMip, tuv, 0.0); }   // level 0 only: no mipmaps
  else if (k < 1.5) { c = textureSample(tex, sMip, tuv); }          // trilinear mipmapping
  else { c = textureSample(tex, sAniso, tuv); }                     // + 16x anisotropic
  var col = c.rgb;
  if (u.showMips > 0.5 && k > 0.5) {
    let tsz = vec2f(textureDimensions(tex));
    let dx = dpdx(tuv * tsz);
    let dy2 = dpdy(tuv * tsz);
    let lod = 0.5 * log2(max(dot(dx, dx), dot(dy2, dy2)));
    let lc = vec3f(0.5 + 0.5 * cos(lod * 1.1 + vec3f(0.0, 2.1, 4.2)));
    col = mix(col, lc, 0.45 * clamp(lod, 0.0, 1.0));
  }
  // fog toward the horizon + sky
  let fog = clamp(1.0 - dy / (res.y * 0.6), 0.0, 1.0);
  let sky = mix(vec3f(0.55, 0.72, 0.95), vec3f(0.2, 0.32, 0.6), clamp((horizon - px.y) / max(horizon, 1.0), 0.0, 1.0));
  col = mix(col, vec3f(0.62, 0.74, 0.92), pow(fog, 6.0) * 0.6);
  if (px.y < horizon) { col = sky; }
  // separators
  let e = min(px.x - k * pw, (k + 1.0) * pw - px.x);
  col = mix(col, vec3f(0.02), clamp(2.0 - e, 0.0, 1.0));
  return vec4f(col, 1.0);
}`;

// pixel-art sprite fragment: 0 nearest/linear (plain), 2 = sharp bilinear ("fat pixel" AA)
const PIXEL_FS = /* wgsl */ `
fn sprite_fs(t0: vec4f, i: VOut) -> vec4f {
  var t = t0;
  if (cam.extra.x > 1.5) {
    let ts = vec2f(textureDimensions(tex));
    let texel = i.uv * ts;
    let seam = floor(texel + vec2f(0.5));
    let dudv = max(fwidth(texel), vec2f(1e-5));
    let st = clamp((texel - seam) / dudv, vec2f(-0.5), vec2f(0.5)) + seam;
    t = textureSample(tex, samp, st / ts);
  }
  let c = t * i.color;
  if (c.a < 0.004) { discard; }
  return c;
}`;

function floorTexture() {
  const N = 512;
  const c = makeCanvas(N, N);
  const g = c.getContext('2d');
  const cell = 32;
  for (let y = 0; y < N / cell; y++)
    for (let x = 0; x < N / cell; x++) {
      g.fillStyle = (x + y) % 2 ? '#f1ece2' : '#2b2f3a';
      g.fillRect(x * cell, y * cell, cell, cell);
    }
  // fine detail: thin lines and colored tiles every 4 cells
  g.strokeStyle = '#e1435f';
  g.lineWidth = 3;
  for (let i = 0; i <= N; i += cell * 4) {
    g.beginPath();
    g.moveTo(i, 0);
    g.lineTo(i, N);
    g.moveTo(0, i);
    g.lineTo(N, i);
    g.stroke();
  }
  g.fillStyle = '#3b82f6';
  for (let y = 0; y < N; y += cell * 4) for (let x = 0; x < N; x += cell * 4) g.fillRect(x + cell * 1.5 + 6, y + cell * 1.5 + 6, cell - 12, cell - 12);
  return c;
}

export default {
  interaction: 'Watch the magnifiers: they show real screen pixels enlarged.',
  examples: [
    {
      id: 'methods',
      label: 'Jaggies & the fixes',
      kind: 'Comparison',
      hint: 'The boxes on the right magnify the outlined spot on each wheel.',
      note: 'One rotating wheel, four renderers. <b>No AA</b>: each pixel is either in or out → stair-steps and flickering spokes. <b>MSAA</b>: the GPU tests 4 sample points per pixel at triangle edges. <b>SDF</b>: the shader computes exact coverage from the distance to the line. <b>SSAA</b>: render at k× resolution and average down.',
    },
    {
      id: 'pixelart',
      label: 'Pixel art scaling',
      kind: 'In a game',
      hint: '',
      note: 'Pixel art at a <b>non-integer</b> zoom and with rotation. <b>Nearest</b> keeps hard texels but some become wider than others and they shimmer when moving. <b>Linear</b> blurs everything. <b>Sharp bilinear</b> keeps texels crisp and only blends the 1-pixel seam between them — the standard trick for smooth-scrolling, freely scaled pixel art.',
      params: { zoom: 5.37 },
    },
    {
      id: 'mipmaps',
      label: 'Mipmaps & moiré',
      kind: 'Real life',
      hint: '',
      note: 'A checkerboard floor stretching to the horizon. Far away, dozens of texels fall into each pixel. Without <b>mipmaps</b> the GPU picks one of them more or less at random → sparkling moiré. Mipmaps pre-shrink the texture; <b>anisotropic</b> filtering also handles surfaces seen at a grazing angle, keeping them sharp.',
    },
  ],
  controls: [
    { type: 'heading', label: 'Wheel', showFor: ['methods'] },
    { type: 'slider', key: 'width', label: 'Line width', min: 0.3, max: 4, step: 0.05, value: 1.3, unit: 'px', showFor: ['methods'], help: 'Below ~1 px, un-antialiased lines break up into dots.' },
    { type: 'slider', key: 'spokes', label: 'Spokes', min: 4, max: 96, step: 1, value: 36, showFor: ['methods'], help: 'Many thin spokes near the hub = moiré.' },
    { type: 'slider', key: 'spin', label: 'Rotation speed', min: 0, max: 1.5, step: 0.01, value: 0.12, showFor: ['methods'], help: 'Slow motion makes aliased edges “crawl”.' },
    { type: 'select', key: 'ssaa', label: 'Supersampling factor', value: 3, showFor: ['methods'], options: [{ value: 2, label: '2×2 = 4 samples' }, { value: 3, label: '3×3 = 9 samples' }, { value: 4, label: '4×4 = 16 samples' }], help: 'SSAA renders the panel this many times larger.' },
    { type: 'heading', label: 'Sprite', showFor: ['pixelart'] },
    { type: 'slider', key: 'zoom', label: 'Zoom', min: 1, max: 9, step: 0.01, value: 5.37, showFor: ['pixelart'], format: (v) => `${v.toFixed(2)}×`, help: 'Try exact integers (4.00) vs fractions (4.37).' },
    { type: 'slider', key: 'rot', label: 'Rotation', min: -45, max: 45, step: 0.5, value: 12, unit: '°', showFor: ['pixelart'] },
    { type: 'toggle', key: 'drift', label: 'Sub-pixel drift', value: true, showFor: ['pixelart'], help: 'Slow movement reveals shimmering texel sizes with nearest filtering.' },
    { type: 'heading', label: 'Magnifier', showFor: ['methods', 'pixelart'] },
    { type: 'slider', key: 'mag', label: 'Magnification', min: 2, max: 16, step: 0.5, value: 6, showFor: ['methods', 'pixelart'], format: (v) => `${v}×` },
    { type: 'heading', label: 'Floor', showFor: ['mipmaps'] },
    { type: 'slider', key: 'scale', label: 'Texture frequency', min: 0.2, max: 4, step: 0.01, value: 1.2, showFor: ['mipmaps'], help: 'More repeats = more texels per pixel in the distance.' },
    { type: 'slider', key: 'speed', label: 'Scroll speed', min: 0, max: 2, step: 0.01, value: 0.3, showFor: ['mipmaps'] },
    { type: 'slider', key: 'angle', label: 'Floor rotation', min: -45, max: 45, step: 0.5, value: 12, unit: '°', showFor: ['mipmaps'] },
    { type: 'toggle', key: 'showMips', label: 'Tint by mip level', value: false, showFor: ['mipmaps'], help: 'Colors show which pre-shrunk level the GPU picked.' },
  ],
  about: {
    summary: 'Pixels are squares on a grid; shapes are not. Aliasing is the jagged, crawling, sparkling mess that happens when you sample a shape only once per pixel — and there are several ways to fix it, each with a different cost.',
    what: `<p><b>Jaggies &amp; the fixes</b>: the same wheel through four anti-aliasing methods, with magnifiers showing real pixels.
      <b>Pixel art scaling</b>: nearest vs linear vs sharp-bilinear filtering of atlas sprites at odd zoom levels.
      <b>Mipmaps</b>: texture minification on a receding floor, with and without mipmaps and anisotropic filtering.</p>`,
    how: `<ol>
      <li><b>No AA</b>: the rasterizer tests one point (the pixel center) per pixel. Edges become stair-steps; anything thinner than a pixel can vanish between samples.</li>
      <li><b>MSAA</b> (multisample AA): a render target with <code>sampleCount: 4</code> stores 4 coverage samples per pixel but runs the fragment shader only once per pixel. At the end it is <b>resolved</b> (averaged) into a normal texture. Cheap, but only smooths geometry edges, not texture or shader detail.</li>
      <li><b>SDF / analytic AA</b>: the fragment shader knows the distance to the shape’s edge, so it outputs <i>exact</i> partial coverage as alpha. Excellent quality, needs a distance function (lines, circles, fonts, UI).</li>
      <li><b>SSAA</b> (supersampling): render everything k× larger and average k×k pixels. Fixes everything — at k² times the cost.</li>
      <li><b>Textures</b> alias too: magnified pixel art needs <i>sharp bilinear</i> (blend only across the texel seam, width = one screen pixel via <code>fwidth</code>); minified textures need <b>mipmaps</b> (a chain of half-size copies; the GPU picks the level whose texels match the pixel size).</li>
    </ol>`,
    uses: [
      { title: 'Vector & UI', text: 'SDF anti-aliasing for HUDs, text, lines and shapes — crisp at any resolution.' },
      { title: 'Geometry', text: 'MSAA for polygonal games (meshes, tessellated lines, low-poly). Post-process AA (FXAA/SMAA/TAA) is the alternative when MSAA is too expensive.' },
      { title: 'Pixel art', text: 'Smooth camera zoom and rotation in games like Celeste-style platformers or Terraria-like zoom: sharp bilinear avoids both blur and shimmer.' },
      { title: 'Textures', text: 'Mipmaps for every texture that can be shown smaller than its size: tilemaps when zoomed out, floors, distant parallax layers.' },
    ],
    try: [
      'Set <i>Line width</i> to 0.4 px: the no-AA spokes break into dots; SDF and SSAA keep faint continuous lines.',
      'Raise <i>Spokes</i> to 96 and watch the moiré near the hubs.',
      'On <b>Pixel art scaling</b>, set <i>Zoom</i> to exactly 5.00 and rotation to 0°: nearest becomes perfect. Then move it to 5.37.',
      'On <b>Mipmaps</b>, enable <i>Tint by mip level</i> to see the GPU switch to smaller levels in the distance.',
      'Lower <i>Texture frequency</i> to 0.3 — with fewer texels per pixel the left panel improves.',
    ],
    ask: [
      'enable 4× MSAA for the sprite/geometry renderer',
      'analytic (SDF) anti-aliasing for lines and shapes',
      'sharp bilinear filtering for pixel art at non-integer zoom',
      'generate mipmaps for tilemaps and floor textures',
      'anisotropic filtering for the Mode-7 floor',
    ],
    perf: `<p>No AA: 1× cost. <b>MSAA 4×</b>: ~4× memory and bandwidth on the render target, shader cost unchanged — usually 10–30% slower.
      <b>SDF</b>: a few extra instructions per pixel of each shape. <b>SSAA k×k</b>: k² times everything (here 9× for 3×3). <b>Mipmaps</b> are
      <i>faster</i> than no mipmaps (smaller levels are cache-friendly) and cost +33% memory. Anisotropic filtering costs a few extra texture reads.</p>`,
    api: `<p>WebGPU: create a texture with <code>sampleCount: 4</code>, render into it with <code>resolveTarget</code> set, and pipelines need <code>multisample: { count: 4 }</code>
      (only 1 and 4 are guaranteed). Mipmaps must be generated yourself (a chain of downsample passes, as in this project’s <code>generateMips</code>).
      WebGL2 has <code>antialias: true</code> for the canvas, multisampled renderbuffers + <code>blitFramebuffer</code>, and <code>gl.generateMipmap()</code> built in.</p>`,
    code: [
      {
        title: 'MSAA in WebGPU: 4-sample texture, resolved into a normal one',
        lang: 'js',
        src: `const msaa = device.createTexture({ size: [w, h], format: 'rgba8unorm', sampleCount: 4,
                                   usage: GPUTextureUsage.RENDER_ATTACHMENT });
const pipe = prog.renderPipeline({ format: 'rgba8unorm', sampleCount: 4, buffers });
const pass = encoder.beginRenderPass({ colorAttachments: [{
  view: msaa.createView(), resolveTarget: resolved.view,   // averaged here
  loadOp: 'clear', storeOp: 'discard' }] });`,
      },
      {
        title: 'Analytic coverage for a line, and sharp bilinear for pixel art',
        lang: 'wgsl',
        src: `let dist = sdSeg(pixelPos, a, b);
let cov = clamp(halfWidth + 0.5 - dist, 0.0, 1.0);        // fraction of the pixel covered

let texel = uv * texSize;
let seam = floor(texel + 0.5);
let st = clamp((texel - seam) / fwidth(texel), vec2f(-0.5), vec2f(0.5)) + seam;
color = textureSample(tex, linearSampler, st / texSize); // blends only at the seam`,
      },
    ],
    links: [
      { title: 'WebGPU Fundamentals — Multisampling / MSAA', url: 'https://webgpufundamentals.org/webgpu/lessons/webgpu-multisampling.html' },
      { title: 'WebGPU Fundamentals — Textures & mipmaps', url: 'https://webgpufundamentals.org/webgpu/lessons/webgpu-textures.html' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const device = gpu.device;
    const atlas = await getAtlas();
    const UA = gpu.uniforms({ resolution: 'vec2f' }, 'SU');
    const US = gpu.uniforms({ resolution: 'vec2f' }, 'SU');
    const segProg = gpu.program({ label: 'aa-segments', bindings: { u: { uniform: UA } }, code: SEG_WGSL });
    const buffers = [
      {
        arrayStride: 48,
        stepMode: 'instance',
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x4' },
          { shaderLocation: 1, offset: 16, format: 'float32x4' },
          { shaderLocation: 2, offset: 32, format: 'float32x4' },
        ],
      },
    ];
    const triPipe = segProg.renderPipeline({ format: 'rgba8unorm', vs: 'vs_tri', fs: 'fs_tri', buffers });
    const triPipeMS = segProg.renderPipeline({ format: 'rgba8unorm', vs: 'vs_tri', fs: 'fs_tri', buffers, sampleCount: 4 });
    const sdfPipe = segProg.renderPipeline({ format: 'rgba8unorm', vs: 'vs_sdf', fs: 'fs_sdf', buffers, blend: 'alpha' });

    const UC = gpu.uniforms({ resolution: 'vec2f', grid: 'vec2f', inset: 'vec4f', src: 'vec2f', zoom: 'f32', ssaa: 'f32', mode: 'f32' }, 'C');
    const compose = gpu.fullscreen({ label: 'aa-compose', code: COMPOSE_WGSL, uniforms: UC, textures: ['A', 'B', 'S'] });

    const UF = gpu.uniforms({ resolution: 'vec2f', time: 'f32', speed: 'f32', scale: 'f32', angle: 'f32', horizon: 'f32', showMips: 'f32' }, 'F');
    const floorProg = gpu.program({
      label: 'aa-floor',
      bindings: { u: { uniform: UF }, tex: { texture: true }, sNoMip: { sampler: true }, sMip: { sampler: true }, sAniso: { sampler: true } },
      include: ['math'],
      code: FLOOR_WGSL,
    });
    const floorPipe = floorProg.renderPipeline({ format: gpu.format });
    const floorTex = gpu.textureFromImage(floorTexture(), { mips: true, label: 'floor' });
    const floorView = floorTex.createView();
    const sNoMip = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });

    // pixel art: premultiplied atlas so linear filtering doesn't create dark fringes
    const atlasTex = gpu.textureFromImage(atlas.canvas, { premultiply: true, label: 'atlas-premul' });
    const sprites = new SpriteBatch(gpu, { texture: atlasTex.createView(), filter: 'nearest', fragment: PIXEL_FS });
    const cam = new Camera2D();

    // instance data
    let cap = 2048;
    let inst = new Float32Array(cap * 12);
    let ibuf = gpu.buffer({ size: cap * 48, usage: GPUBufferUsage.VERTEX, label: 'aa-segments' });
    let count = 0;
    const seg = (ax, ay, bx, by, hw, c) => {
      if (count >= cap) {
        cap *= 2;
        const n = new Float32Array(cap * 12);
        n.set(inst);
        inst = n;
        ibuf.destroy();
        ibuf = gpu.buffer({ size: cap * 48, usage: GPUBufferUsage.VERTEX, label: 'aa-segments' });
      }
      const o = count++ * 12;
      inst[o] = ax;
      inst[o + 1] = ay;
      inst[o + 2] = bx;
      inst[o + 3] = by;
      inst[o + 4] = hw;
      inst[o + 8] = c[0];
      inst[o + 9] = c[1];
      inst[o + 10] = c[2];
      inst[o + 11] = 1;
    };
    const wheel = (cx, cy, R, k, hw, ang, spokes) => {
      const rimCol = [1, 0.62, 0.25];
      const spokeCol = [0.95, 0.95, 0.9];
      const N = 120;
      for (let i = 0; i < N; i++) {
        const a0 = ang + (i / N) * Math.PI * 2;
        const a1 = ang + ((i + 1) / N) * Math.PI * 2;
        seg(cx + Math.cos(a0) * R * k, cy + Math.sin(a0) * R * k, cx + Math.cos(a1) * R * k, cy + Math.sin(a1) * R * k, hw * k * 1.4, rimCol);
      }
      for (let i = 0; i < spokes; i++) {
        const a = ang + (i / spokes) * Math.PI * 2;
        const r0 = R * 0.12;
        seg(cx + Math.cos(a) * r0 * k, cy + Math.sin(a) * r0 * k, cx + Math.cos(a) * R * k, cy + Math.sin(a) * R * k, hw * k, spokeCol);
      }
      // a slowly tilting near-horizontal bar through the hub (stair-steps!)
      const tilt = 0.06 * Math.sin(ang * 3) + 0.04;
      seg(cx - Math.cos(tilt) * R * 1.05 * k, cy - Math.sin(tilt) * R * 1.05 * k, cx + Math.cos(tilt) * R * 1.05 * k, cy + Math.sin(tilt) * R * 1.05 * k, hw * k * 1.6, [0.45, 0.8, 1]);
    };

    // targets
    let A = null;
    let B = null;
    let MS = null;
    let MSV = null;
    let S = null;
    let sizeKey = '';
    const ensureTargets = (W, H, ssaaW, ssaaH) => {
      const key = `${W}x${H}:${ssaaW}x${ssaaH}`;
      if (key === sizeKey) return;
      sizeKey = key;
      A?.destroy();
      B?.destroy();
      MS?.destroy();
      S?.destroy();
      A = gpu.target(W, H, { label: 'aa-A' });
      B = gpu.target(W, H, { label: 'aa-B' });
      MS = gpu.texture({ size: [W, H], format: 'rgba8unorm', sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'aa-msaa' });
      MSV = MS.createView();
      S = gpu.target(ssaaW, ssaaH, { label: 'aa-ssaa' });
    };
    const BG = [0.06, 0.07, 0.11, 1];
    let angle = 0;
    let drift = 0;

    return {
      frame(ctx) {
        const P = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const t = ctx.time;
        const enc = ctx.encoder;
        const ex = ctx.example;
        const canvas = { view: ctx.target, format: gpu.format };

        if (ex === 'mipmaps') {
          labels(ctx, 'mipmaps', [
            { text: 'no mipmaps', x: 1 / 6, y: 0.1, style: LBL },
            { text: 'mipmaps (trilinear)', x: 0.5, y: 0.1, style: LBL },
            { text: 'mipmaps + 16× anisotropic', x: 5 / 6, y: 0.1, style: LBL },
          ]);
          UF.set('resolution', [W, H]).set('time', t).set('speed', P.speed).set('scale', P.scale).set('angle', (P.angle * Math.PI) / 180).set('horizon', 0.3).set('showMips', P.showMips ? 1 : 0);
          UF.upload();
          const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.target, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
          pass.setPipeline(floorPipe);
          pass.setBindGroup(0, floorProg.bind({ u: UF, tex: floorView, sNoMip, sMip: 'linear-mip-repeat', sAniso: 'aniso' }));
          pass.draw(3);
          pass.end();
          return;
        }

        const grid = ex === 'methods' ? [2, 2] : [3, 1];
        const pw = W / grid[0];
        const ph = H / grid[1];
        const f = ex === 'methods' ? Math.max(1, Math.min(Number(P.ssaa), Math.floor(8192 / Math.max(pw, ph)), ctx.testMode ? 2 : 4)) : 1;
        ensureTargets(W, H, Math.ceil(pw * f), Math.ceil(ph * f));
        let inset;
        let src;

        if (ex === 'methods') {
          labels(ctx, 'methods', [
            { text: 'no AA · 1 sample / pixel', x: 0.25, y: 0.47, valign: 'bottom', style: LBL },
            { text: 'MSAA 4× · hardware samples', x: 0.75, y: 0.47, valign: 'bottom', style: LBL },
            { text: 'SDF · analytic coverage', x: 0.25, y: 0.97, valign: 'bottom', style: LBL },
            { text: `SSAA ${f}×${f} · render big, shrink`, x: 0.75, y: 0.97, valign: 'bottom', style: LBL },
          ]);
          if (!ctx.paused) angle += ctx.dt * P.spin;
          const R = Math.min(pw * 0.27, ph * 0.38);
          const cx = pw * 0.3;
          const cy = ph * 0.47;
          const hw = P.width / 2;
          const spokes = Math.round(P.spokes);
          count = 0;
          wheel(cx, cy, R, 1, hw, angle, spokes); // panel 0 (no AA) in A
          const n0 = count;
          wheel(cx, ph + cy, R, 1, hw, angle, spokes); // panel 2 (SDF) in A
          const n2 = count - n0;
          wheel(pw + cx, cy, R, 1, hw, angle, spokes); // panel 1 (MSAA) in MS -> B
          const n1 = count - n0 - n2;
          wheel(cx * f, cy * f, R, f, hw, angle, spokes); // panel 3 (SSAA) in S, f× larger
          const n3 = count - n0 - n2 - n1;
          device.queue.writeBuffer(ibuf, 0, inst, 0, count * 12);
          UA.set('resolution', [W, H]).upload();
          US.set('resolution', [S.width, S.height]).upload();
          // A: no-AA panel + SDF panel
          let pass = enc.beginRenderPass({ colorAttachments: [{ view: A.view, loadOp: 'clear', clearValue: BG, storeOp: 'store' }] });
          pass.setVertexBuffer(0, ibuf);
          pass.setBindGroup(0, segProg.bind({ u: UA }));
          pass.setPipeline(triPipe);
          pass.draw(6, n0, 0, 0);
          pass.setPipeline(sdfPipe);
          pass.draw(6, n2, 0, n0);
          pass.end();
          // MSAA: 4 samples per pixel, resolved (averaged) into B
          pass = enc.beginRenderPass({ colorAttachments: [{ view: MSV, resolveTarget: B.view, loadOp: 'clear', clearValue: BG, storeOp: 'discard' }] });
          pass.setVertexBuffer(0, ibuf);
          pass.setBindGroup(0, segProg.bind({ u: UA }));
          pass.setPipeline(triPipeMS);
          pass.draw(6, n1, 0, n0 + n2);
          pass.end();
          // SSAA: the same un-antialiased triangles in a target f× larger
          pass = enc.beginRenderPass({ colorAttachments: [{ view: S.view, loadOp: 'clear', clearValue: BG, storeOp: 'store' }] });
          pass.setVertexBuffer(0, ibuf);
          pass.setBindGroup(0, segProg.bind({ u: US }));
          pass.setPipeline(triPipe);
          pass.draw(6, n3, 0, n0 + n2 + n1);
          pass.end();
          const isz = Math.min(pw * 0.36, ph * 0.72);
          inset = [pw * 0.6, ph * 0.08, isz, isz];
          src = [cx + Math.cos(-0.8) * R * 0.9, cy + Math.sin(-0.8) * R * 0.9];
        } else {
          // ---- pixel art
          labels(ctx, 'pixelart', [
            { text: 'nearest · shimmers', x: 1 / 6, y: 0.1, style: LBL },
            { text: 'linear · blurry', x: 0.5, y: 0.1, style: LBL },
            { text: 'sharp bilinear · crisp & smooth', x: 5 / 6, y: 0.1, style: LBL },
          ]);
          if (!ctx.paused && P.drift) drift += ctx.dt;
          cam.setViewport(W, H);
          const z = P.zoom * Math.max(1, H / 600) * 1.5;
          const rot = (P.rot * Math.PI) / 180;
          const dx = P.drift ? Math.sin(drift * 0.6) * 6 : 0;
          const dy = P.drift ? Math.cos(drift * 0.45) * 3 : 0;
          const run = atlas.anim('hero_run');
          const frame = run[Math.floor(t * 6) % run.length];
          const hs = Math.min(16 * z, pw * 0.7, ph * 0.42);
          const hx = pw * 0.5 + dx;
          const hy = ph * 0.38 + dy;
          sprites.begin();
          const modes = [
            { filter: 'nearest', extra: [0, 0, 0, 0] },
            { filter: 'linear', extra: [0, 0, 0, 0] },
            { filter: 'linear', extra: [2, 0, 0, 0] },
          ];
          const small = hs * 0.3;
          modes.forEach((m, k) => {
            const ox = k * pw;
            sprites.draw(ox + hx, hy, hs, hs, { uv: atlas.uv(frame), rotation: rot });
            sprites.draw(ox + pw * 0.14 + dx * 0.5, ph * 0.2, small, small, { uv: atlas.uv(`coin_${Math.floor(t * 10) % 8}`), rotation: -rot * 0.5 });
            sprites.draw(ox + pw * 0.86 - dx * 0.5, ph * 0.22, small, small, { uv: atlas.uv('gem'), rotation: rot * 0.3 });
            for (let i = 0; i < 5; i++) sprites.draw(ox + pw * (0.1 + i * 0.2) + dx * 0.3, ph * 0.66, small, small, { uv: atlas.uv('tile_grass'), rotation: rot * 0.15 });
            sprites.flush(enc, A, cam, { clear: k === 0 ? BG : null, filter: m.filter, extra: m.extra, blend: 'premultiplied', format: 'rgba8unorm' });
          });
          const iw = Math.min(pw * 0.82, ph * 0.5);
          const ih = Math.min(ph * 0.24, iw);
          inset = [pw * 0.5 - iw / 2, ph * 0.74, iw, ih];
          // look at the hero's back edge (rotated -> diagonal staircase)
          const ex2 = -0.22 * hs;
          const ey2 = 0.05 * hs;
          src = [hx + Math.cos(rot) * ex2 - Math.sin(rot) * ey2, hy + Math.sin(rot) * ex2 + Math.cos(rot) * ey2];
        }
        UC.set('resolution', [W, H]).set('grid', grid).set('inset', inset).set('src', src).set('zoom', P.mag).set('ssaa', f).set('mode', ex === 'methods' ? 0 : 1);
        compose.draw(enc, canvas, { A, B, S });
      },
    };
  },
};

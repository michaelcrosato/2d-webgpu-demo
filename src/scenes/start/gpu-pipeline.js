import { shaderScene } from '../../core/shaderscene.js';
import { getAtlas } from '../../core/assets.js';

// "How a GPU draws 2D": rasterization, interpolation and textured quads, visualised with a
// fragment shader. Vertex dragging is handled in JS (bind) and passed in as uniforms.

const verts = [
  [0.3, 0.2],
  [0.72, 0.38],
  [0.42, 0.82],
  [0.18, 0.62],
];
const quadVerts = [
  [0.33, 0.18],
  [0.7, 0.24],
  [0.66, 0.8],
  [0.3, 0.74],
];
let dragging = -1;
let tileRect = [0, 0, 1, 1];
let atlasRequested = false;

function handleDrag(ctx, list, count) {
  const p = ctx.pointer;
  const px = p.x / ctx.width;
  const py = p.y / ctx.height;
  if (p.down) {
    if (dragging < 0 && p.clicked) {
      let best = -1;
      let bestD = 40 / Math.max(1, ctx.height);
      for (let i = 0; i < count; i++) {
        const dx = (list[i][0] - px) * (ctx.width / ctx.height);
        const dy = list[i][1] - py;
        const d = Math.hypot(dx, dy);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      dragging = best;
    }
    if (dragging >= 0) list[dragging] = [Math.min(0.98, Math.max(0.02, px)), Math.min(0.98, Math.max(0.02, py))];
  } else dragging = -1;
}

export default shaderScene({
  interaction: 'Drag the white vertex handles. Everything updates live.',
  examples: [
    {
      id: 'raster',
      label: 'Rasterization',
      kind: 'Step 1',
      note: 'The GPU tests the <b>center</b> of every pixel against the triangle’s three edges. Pixels whose center is inside become <b>fragments</b> (filled). Notice the stair-steps: that is aliasing.',
    },
    {
      id: 'interp',
      label: 'Interpolation',
      kind: 'Step 2',
      note: 'Each vertex carries data (here a color). The rasterizer blends it across the triangle using <b>barycentric weights</b> — that is how UVs, colors and normals reach the fragment shader.',
    },
    {
      id: 'sprite',
      label: 'A sprite = 2 triangles',
      kind: 'In a game',
      note: 'Every sprite is a quad made of two triangles with texture coordinates (UVs) at the corners. Drag a corner: each triangle maps the texture <i>affinely</i>, creating the seam/warping famous from PS1 games.',
    },
  ],
  controls: [
    { type: 'slider', key: 'cell', label: 'Pixel size (zoom)', min: 4, max: 64, step: 1, value: 24, help: 'Make pixels huge to see what the rasterizer does.' },
    { type: 'toggle', key: 'samples', label: 'Show pixel centers', value: true, showFor: ['raster'] },
    { type: 'toggle', key: 'wire', label: 'Show triangle edges', value: true },
    { type: 'toggle', key: 'pixelate', label: 'Shade per (big) pixel', value: true, showFor: ['interp', 'sprite'], help: 'Off = evaluate at full screen resolution.' },
    { type: 'slider', key: 'repeat', label: 'Texture repeat', min: 1, max: 8, step: 1, value: 3, showFor: ['sprite'], help: 'UVs above 1 tile the texture.' },
  ],
  uniforms: { cell: 'f32', samples: 'f32', wire: 'f32', pixelate: 'f32', repeat: 'f32', v0: 'vec2f', v1: 'vec2f', v2: 'vec2f', v3: 'vec2f', tile: 'vec4f' },
  textures: { atlas: { source: async () => (await getAtlas()).canvas, filter: 'nearest' } },
  include: ['sdf'],
  bind(params, ctx) {
    if (!atlasRequested) {
      atlasRequested = true;
      getAtlas().then((a) => (tileRect = a.uv('tile_brick')));
    }
    const isQuad = ctx.example === 'sprite';
    const list = isQuad ? quadVerts : verts;
    handleDrag(ctx, list, isQuad ? 4 : 3);
    const toPx = (v) => [v[0] * ctx.width, v[1] * ctx.height];
    return { v0: toPx(list[0]), v1: toPx(list[1]), v2: toPx(list[2]), v3: toPx(list[3]), tile: tileRect };
  },
  code: /* wgsl */ `
fn edgeFn(a: vec2f, b: vec2f, p: vec2f) -> f32 { return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x); }

// barycentric weights of p in triangle (a,b,c); all >= 0 means inside
fn bary(a: vec2f, b: vec2f, c: vec2f, p: vec2f) -> vec3f {
  let area = edgeFn(a, b, c);
  return vec3f(edgeFn(b, c, p), edgeFn(c, a, p), edgeFn(a, b, p)) / area;
}

fn insideW(w: vec3f) -> f32 { return step(0.0, min(w.x, min(w.y, w.z))); }

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  let cs = u.cell;
  let cellId = floor(px / cs);
  let center = (cellId + 0.5) * cs;
  let f = fract(px / cs);
  let gridLine = 1.0 - step(1.0 / cs, min(f.x, f.y));
  var sp = px;
  if (ex == 0 || u.pixelate > 0.5) { sp = center; }

  var col = vec3f(0.07, 0.08, 0.11);
  col = mix(col, vec3f(0.13, 0.15, 0.2), gridLine * step(5.0, cs));

  if (ex == 0) {
    let w = bary(u.v0, u.v1, u.v2, center);
    let ins = insideW(w);
    col = mix(col, vec3f(0.36, 0.5, 1.0), ins * 0.85);
    // partially covered pixels (true coverage) shown faintly: what anti-aliasing would use
    let wp = bary(u.v0, u.v1, u.v2, px);
    col = mix(col, vec3f(0.4, 0.45, 0.6), insideW(wp) * (1.0 - ins) * 0.25);
    if (u.samples > 0.5 && cs >= 6.0) {
      let dd = length(px - center);
      let dotc = mix(vec3f(0.45), vec3f(1.0, 0.9, 0.3), ins);
      col = mix(col, dotc, 1.0 - smoothstep(1.5, 2.6, dd));
    }
  } else if (ex == 1) {
    let w = bary(u.v0, u.v1, u.v2, sp);
    let ins = insideW(w);
    let c = w.x * vec3f(1.0, 0.25, 0.3) + w.y * vec3f(0.25, 1.0, 0.4) + w.z * vec3f(0.3, 0.45, 1.0);
    col = mix(col, c, ins);
  } else {
    // quad = triangle (v0, v1, v2) + triangle (v0, v2, v3), uv corners (0,0) (1,0) (1,1) (0,1)
    let wa = bary(u.v0, u.v1, u.v2, sp);
    let wb = bary(u.v0, u.v2, u.v3, sp);
    var tuv = vec2f(-1.0);
    var tri = 0.0;
    if (insideW(wa) > 0.5) { tuv = wa.x * vec2f(0.0, 0.0) + wa.y * vec2f(1.0, 0.0) + wa.z * vec2f(1.0, 1.0); tri = 1.0; }
    else if (insideW(wb) > 0.5) { tuv = wb.x * vec2f(0.0, 0.0) + wb.y * vec2f(1.0, 1.0) + wb.z * vec2f(0.0, 1.0); tri = 2.0; }
    if (tri > 0.5) {
      let rep = fract(tuv * u.repeat);
      let auv = mix(u.tile.xy, u.tile.zw, rep);
      var tc = TEXN(atlas, auv).rgb;
      tc = mix(tc, tc * vec3f(0.85, 0.9, 1.15), step(1.5, tri) * 0.35);
      col = tc;
    }
  }

  // wireframe edges
  if (u.wire > 0.5) {
    var d = min(sdSegment(px, u.v0, u.v1), min(sdSegment(px, u.v1, u.v2), sdSegment(px, u.v2, u.v0)));
    if (ex == 2) {
      d = min(min(sdSegment(px, u.v0, u.v1), sdSegment(px, u.v1, u.v2)), min(sdSegment(px, u.v2, u.v3), sdSegment(px, u.v3, u.v0)));
      d = min(d, sdSegment(px, u.v0, u.v2) + 0.0);
    }
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(0.6, 1.6, d));
  }
  // vertex handles
  var hv = min(length(px - u.v0), min(length(px - u.v1), length(px - u.v2)));
  if (ex == 2) { hv = min(hv, length(px - u.v3)); }
  col = mix(col, vec3f(0.0), 1.0 - smoothstep(8.0, 9.5, hv));
  col = mix(col, vec3f(1.0), 1.0 - smoothstep(5.5, 7.0, hv));
  return vec4f(col, 1.0);
}`,
  about: {
    summary: 'Every 2D game frame — sprites, tiles, text, particles — is made of triangles. Here is what happens between “draw this quad” and colored pixels.',
    what: `<p>A triangle drawn on a grid of giant pixels. You can see which pixels the GPU decides to color (<b>rasterization</b>),
      how per-vertex data is blended across the surface (<b>interpolation</b>), and how a sprite is really just two textured triangles.</p>`,
    how: `<ol>
      <li><b>Vertex shader</b>: runs once per corner and outputs its screen position (and extra data such as UVs or colors).</li>
      <li><b>Rasterizer</b> (fixed hardware): for each pixel, tests whether the pixel <i>center</i> is inside the triangle using three “edge functions”.</li>
      <li><b>Interpolation</b>: the rasterizer computes barycentric weights (how close the pixel is to each corner) and blends the vertex outputs.</li>
      <li><b>Fragment shader</b>: runs for every covered pixel with the interpolated data, and returns a color (e.g. by reading a texture at the interpolated UV).</li>
      <li><b>Blending</b>: the color is combined with what is already in the framebuffer (see <a href="#/s/blend-modes">Blend Modes</a>).</li>
    </ol>`,
    uses: [
      { title: 'Sprites & tiles', text: 'Each one is a quad (2 triangles) with UVs into a texture atlas.' },
      { title: 'Retro looks', text: 'PS1-style texture warping comes from affine (non-perspective) interpolation across triangles.' },
      { title: 'Debugging', text: 'Wireframe views reveal overdraw and wasted transparent pixels in large sprites.' },
    ],
    try: [
      'Drag a vertex so the triangle becomes a thin sliver — pixels flicker on/off: aliasing.',
      'On <b>Interpolation</b> turn off “Shade per pixel” to see the smooth full-resolution blend.',
      'On <b>A sprite = 2 triangles</b> drag one corner far out — the bricks bend along the diagonal seam.',
      'Shrink “Pixel size” to 4 to see how fine the grid really is at screen resolution.',
    ],
    ask: [
      'draw all sprites as instanced quads (two triangles each)',
      'PS1-style affine texture warping',
      'per-vertex colors interpolated across the shape',
      'a wireframe/debug view of the sprite quads',
    ],
    perf: `<p>Triangles are cheap; <b>pixels are expensive</b>. Big transparent sprites still run the fragment shader on every covered pixel
      (overdraw). Trimming quads to the visible part of a sprite can measurably speed up particle-heavy scenes.</p>`,
    api: `<p>The pipeline stages are identical in WebGL2 and WebGPU. WebGPU just makes them explicit: you build a <i>render pipeline</i>
      object up front (shaders + vertex layout + blending), which lets the driver validate once instead of every draw.</p>`,
    code: {
      title: 'Edge functions & barycentric weights — what the rasterizer computes',
      lang: 'wgsl',
      src: `fn edgeFn(a: vec2f, b: vec2f, p: vec2f) -> f32 {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}
fn bary(a: vec2f, b: vec2f, c: vec2f, p: vec2f) -> vec3f {
  let area = edgeFn(a, b, c);
  return vec3f(edgeFn(b, c, p), edgeFn(c, a, p), edgeFn(a, b, p)) / area;
}
// inside if all three weights are >= 0
// interpolated value = w.x * valueA + w.y * valueB + w.z * valueC`,
    },
  },
});

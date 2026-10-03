// Immediate-mode 2D drawing on WebGPU, with a canvas-like API but GPU instancing underneath:
//
//   const cam = new Camera2D();            // world units = pixels by default, y DOWN
//   const shapes = new ShapeBatch(gpu);    // circles, lines, rects, triangles (SDF, anti-aliased, optional glow)
//   const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest' });
//
//   frame(ctx) {
//     cam.setViewport(ctx.width, ctx.height);
//     shapes.begin();
//     shapes.circle(100, 100, 40, [1, 0.5, 0, 1], { glow: 10 });
//     shapes.flush(ctx.encoder, ctx.target, cam, { clear: [0, 0, 0, 1] });
//   }
//
// Several flushes per frame are fine (e.g. one with 'alpha' blending, one 'additive').
// Call begin() once per frame before the first draw.

import { BLEND } from './webgpu.js';
import { hexToRgb } from './uniforms.js';

const SLOT = 256; // uniform ring slot size (minUniformBufferOffsetAlignment)
const SLOTS = 64;

const colorCache = new Map();
/** Normalise a color: [r,g,b,a] | [r,g,b] | '#rrggbb' | '#rrggbbaa' -> [r,g,b,a] */
export function rgba(c, alpha = 1) {
  if (Array.isArray(c) || ArrayBuffer.isView(c)) return c.length === 4 ? c : [c[0], c[1], c[2], alpha];
  if (typeof c === 'string') {
    let v = colorCache.get(c);
    if (!v) {
      const h = c.replace('#', '');
      const rgb = hexToRgb(h.slice(0, h.length >= 6 ? 6 : 3));
      const a = h.length === 8 ? parseInt(h.slice(6), 16) / 255 : 1;
      v = [rgb[0], rgb[1], rgb[2], a];
      colorCache.set(c, v);
    }
    return alpha === 1 ? v : [v[0], v[1], v[2], v[3] * alpha];
  }
  return [1, 1, 1, alpha];
}

export class Camera2D {
  constructor() {
    this.x = 0; // world point at the CENTER of the screen
    this.y = 0;
    this.zoom = 1;
    this.rotation = 0; // radians, positive = clockwise on screen
    this.width = 1;
    this.height = 1;
    this._init = false;
    this.m = new Float32Array(16);
  }
  /** Set the viewport size (device pixels). First call centers the camera so world == screen pixels. */
  setViewport(w, h) {
    if (!this._init) {
      this.x = w / 2;
      this.y = h / 2;
      this._init = true;
    }
    this.width = w;
    this.height = h;
    return this;
  }
  reset() {
    this.x = this.width / 2;
    this.y = this.height / 2;
    this.zoom = 1;
    this.rotation = 0;
    return this;
  }
  /** World -> clip-space matrix (column-major mat4). */
  matrix() {
    const sx = (this.zoom * 2) / this.width;
    const sy = (this.zoom * 2) / this.height;
    const c = Math.cos(this.rotation);
    const s = Math.sin(this.rotation);
    const a = sx * c;
    const b = sx * s;
    const cc = sy * s;
    const d = -sy * c;
    const m = this.m;
    m.fill(0);
    m[0] = a;
    m[1] = cc;
    m[4] = b;
    m[5] = d;
    m[10] = 1;
    m[12] = -(a * this.x + b * this.y);
    m[13] = -(cc * this.x + d * this.y);
    m[15] = 1;
    return m;
  }
  /** Screen pixel (top-left origin) -> world. */
  screenToWorld(sx, sy) {
    const dx = (sx - this.width / 2) / this.zoom;
    const dy = (sy - this.height / 2) / this.zoom;
    const c = Math.cos(this.rotation);
    const s = Math.sin(this.rotation);
    return [this.x + c * dx - s * dy, this.y + s * dx + c * dy];
  }
  worldToScreen(wx, wy) {
    const dx = wx - this.x;
    const dy = wy - this.y;
    const c = Math.cos(this.rotation);
    const s = Math.sin(this.rotation);
    return [(c * dx + s * dy) * this.zoom + this.width / 2, (-s * dx + c * dy) * this.zoom + this.height / 2];
  }
}

/** Shared machinery: growable instance buffer + uniform ring for per-flush camera data. */
class InstanceBatch {
  constructor(gpu, floatsPerInstance, capacity) {
    this.gpu = gpu;
    this.fpi = floatsPerInstance;
    this.capacity = capacity;
    this.data = new Float32Array(capacity * floatsPerInstance);
    this.count = 0; // instances written this frame
    this.flushed = 0; // instances already submitted this frame
    this.slot = 0;
    this.retired = [];
    this.buffer = this._makeBuffer(capacity);
    this.ubo = gpu.track(
      gpu.device.createBuffer({ size: SLOT * SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'batch-ubo' }),
    );
    this.uboData = new Float32Array(SLOT / 4);
  }
  _makeBuffer(cap) {
    return this.gpu.track(
      this.gpu.device.createBuffer({ size: cap * this.fpi * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: 'batch-instances' }),
    );
  }
  begin() {
    this.count = 0;
    this.flushed = 0;
    this.slot = 0;
    for (const b of this.retired) b.destroy();
    this.retired.length = 0;
  }
  /** Reserve one instance; returns float offset into this.data. */
  alloc() {
    if (this.count >= this.capacity) {
      const cap = this.capacity * 2;
      const nd = new Float32Array(cap * this.fpi);
      nd.set(this.data);
      this.data = nd;
      // the old buffer may still be referenced by an earlier flush this frame
      this.retired.push(this.buffer);
      this.buffer = this._makeBuffer(cap);
      // re-upload everything already flushed so offsets stay valid in the new buffer
      if (this.flushed) this.gpu.queue.writeBuffer(this.buffer, 0, this.data, 0, this.flushed * this.fpi);
      this.capacity = cap;
    }
    return this.count++ * this.fpi;
  }
  /** Write camera uniforms into the next ring slot, returns dynamic offset. */
  writeCamera(cam, extra = [0, 0, 0, 0]) {
    const off = (this.slot++ % SLOTS) * SLOT;
    this.uboData.set(cam.matrix(), 0);
    this.uboData[16] = cam.width;
    this.uboData[17] = cam.height;
    this.uboData[18] = cam.zoom;
    this.uboData[19] = performance.now() / 1000;
    this.uboData.set(extra, 20);
    this.gpu.queue.writeBuffer(this.ubo, off, this.uboData, 0, 24);
    return off;
  }
  /** Upload pending instances; returns [first, count]. */
  uploadPending() {
    const first = this.flushed;
    const n = this.count - first;
    if (n > 0) this.gpu.queue.writeBuffer(this.buffer, first * this.fpi * 4, this.data, first * this.fpi, n * this.fpi);
    this.flushed = this.count;
    return [first, n];
  }
}

const CAMERA_WGSL = /* wgsl */ `
struct Cam { m: mat4x4f, viewport: vec4f, extra: vec4f };
@group(0) @binding(0) var<uniform> cam: Cam;
`;

// ------------------------------------------------------------------------------------- shapes

const SHAPE_WGSL = /* wgsl */ `
diagnostic(off, derivative_uniformity);
${CAMERA_WGSL}
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) w: vec2f,
  @location(1) @interpolate(flat) a: vec4f,
  @location(2) @interpolate(flat) b: vec4f,
  @location(3) @interpolate(flat) color: vec4f,
  @location(4) @interpolate(flat) style: vec4f,
};
fn rot2(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }
fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(pa - ba * h);
}
fn sdBox(p: vec2f, b: vec2f) -> f32 { let d = abs(p) - b; return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0); }
fn sdTriangle(p: vec2f, p0: vec2f, p1: vec2f, p2: vec2f) -> f32 {
  let e0 = p1 - p0; let e1 = p2 - p1; let e2 = p0 - p2;
  let v0 = p - p0; let v1 = p - p1; let v2 = p - p2;
  let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  let s = sign(e0.x * e2.y - e0.y * e2.x);
  let d = min(min(vec2f(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                  vec2f(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                  vec2f(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}
@vertex fn vs_main(@builtin(vertex_index) vi: u32,
                   @location(0) a: vec4f, @location(1) b: vec4f,
                   @location(2) color: vec4f, @location(3) style: vec4f) -> VOut {
  let kind = u32(style.x);
  let margin = style.y * 0.5 + style.z * 5.0 + 2.0 / max(cam.viewport.z, 1e-3);
  var lo: vec2f; var hi: vec2f;
  if (kind == 0u) { lo = a.xy - vec2f(a.z); hi = a.xy + vec2f(a.z); }
  else if (kind == 1u) { lo = min(a.xy, a.zw) - vec2f(b.x); hi = max(a.xy, a.zw) + vec2f(b.x); }
  else if (kind == 2u) { let r = length(a.zw); lo = a.xy - vec2f(r); hi = a.xy + vec2f(r); }
  else { lo = min(min(a.xy, a.zw), b.xy); hi = max(max(a.xy, a.zw), b.xy); }
  lo -= vec2f(margin); hi += vec2f(margin);
  var corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let w = mix(lo, hi, corners[vi]);
  var o: VOut;
  o.pos = cam.m * vec4f(w, 0.0, 1.0);
  o.w = w; o.a = a; o.b = b; o.color = color; o.style = style;
  return o;
}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let kind = u32(i.style.x);
  var d: f32;
  if (kind == 0u) { d = length(i.w - i.a.xy) - i.a.z; }
  else if (kind == 1u) { d = sdSegment(i.w, i.a.xy, i.a.zw) - i.b.x; }
  else if (kind == 2u) { let q = rot2(-i.b.x) * (i.w - i.a.xy); d = sdBox(q, i.a.zw - vec2f(i.b.y)) - i.b.y; }
  else { d = sdTriangle(i.w, i.a.xy, i.a.zw, i.b.xy); }
  if (i.style.y > 0.0) { d = abs(d) - i.style.y * 0.5; }
  let aa = max(fwidth(d), 1e-4);
  var cov = clamp(0.5 - d / aa, 0.0, 1.0);
  if (i.style.z > 0.0) { cov = max(cov, exp(-max(d, 0.0) / i.style.z) * i.style.w); }
  if (cov <= 0.001) { discard; }
  return vec4f(i.color.rgb, i.color.a * cov);
}
`;

export class ShapeBatch {
  constructor(gpu, { capacity = 4096 } = {}) {
    this.gpu = gpu;
    this.b = new InstanceBatch(gpu, 16, capacity);
    const device = gpu.device;
    this.module = gpu.shader(SHAPE_WGSL, 'shapes');
    this.bgl = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } }],
    });
    this.layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.bindGroup = device.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: { buffer: this.b.ubo, size: 96 } }] });
    this.pipelines = new Map();
  }
  _pipe(format, blend) {
    const key = `${format}|${blend}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const attrs = [0, 1, 2, 3].map((i) => ({ shaderLocation: i, offset: i * 16, format: 'float32x4' }));
      p = this.gpu.device.createRenderPipeline({
        label: 'shapes',
        layout: this.layout,
        vertex: { module: this.module, entryPoint: 'vs_main', buffers: [{ arrayStride: 64, stepMode: 'instance', attributes: attrs }] },
        fragment: { module: this.module, entryPoint: 'fs_main', targets: [{ format, blend: BLEND[blend] }] },
        primitive: { topology: 'triangle-list' },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }
  begin() {
    this.b.begin();
    return this;
  }
  _push(kind, a0, a1, a2, a3, b0, b1, b2, b3, color, o) {
    const off = this.b.alloc();
    const d = this.b.data;
    const c = rgba(color, o.alpha ?? 1);
    d[off] = a0;
    d[off + 1] = a1;
    d[off + 2] = a2;
    d[off + 3] = a3;
    d[off + 4] = b0;
    d[off + 5] = b1;
    d[off + 6] = b2;
    d[off + 7] = b3;
    d[off + 8] = c[0];
    d[off + 9] = c[1];
    d[off + 10] = c[2];
    d[off + 11] = c[3] ?? 1;
    d[off + 12] = kind;
    d[off + 13] = o.stroke || 0;
    d[off + 14] = o.glow || 0;
    d[off + 15] = o.glowStrength ?? 0.6;
    return this;
  }
  /** Filled (or stroked with opts.stroke) circle. opts: { stroke, glow, glowStrength, alpha } */
  circle(x, y, r, color, opts = {}) {
    return this._push(0, x, y, r, 0, 0, 0, 0, 0, color, opts);
  }
  /** Line segment with round caps. */
  line(x1, y1, x2, y2, width, color, opts = {}) {
    return this._push(1, x1, y1, x2, y2, width / 2, 0, 0, 0, color, opts);
  }
  /** Rectangle with TOP-LEFT corner (x,y). opts: { rotation (about center), radius (corner), stroke, glow } */
  rect(x, y, w, h, color, opts = {}) {
    return this._push(2, x + w / 2, y + h / 2, w / 2, h / 2, opts.rotation || 0, Math.min(opts.radius || 0, w / 2, h / 2), 0, 0, color, opts);
  }
  /** Rectangle by CENTER and half-size. */
  box(cx, cy, hw, hh, color, opts = {}) {
    return this._push(2, cx, cy, hw, hh, opts.rotation || 0, Math.min(opts.radius || 0, hw, hh), 0, 0, color, opts);
  }
  triangle(x1, y1, x2, y2, x3, y3, color, opts = {}) {
    return this._push(3, x1, y1, x2, y2, x3, y3, 0, 0, color, opts);
  }
  /** Connected line segments through [[x,y], ...] or flat [x0,y0,x1,y1,...]. */
  polyline(points, width, color, opts = {}) {
    const flat = typeof points[0] === 'number';
    const n = flat ? points.length / 2 : points.length;
    for (let i = 0; i < n - 1 + (opts.closed ? 1 : 0); i++) {
      const j = (i + 1) % n;
      const [ax, ay] = flat ? [points[i * 2], points[i * 2 + 1]] : points[i];
      const [bx, by] = flat ? [points[j * 2], points[j * 2 + 1]] : points[j];
      this.line(ax, ay, bx, by, width, color, opts);
    }
    return this;
  }
  /**
   * Draw everything queued since the last flush.
   * opts: { clear: [r,g,b,a] | false, blend: 'alpha' | 'additive' | ..., format }
   */
  flush(encoder, target, camera, opts = {}) {
    const [first, n] = this.b.uploadPending();
    const clear = opts.clear || null;
    if (n === 0 && !clear) return;
    const off = this.b.writeCamera(camera);
    const pass = encoder.beginRenderPass({
      label: 'shapes',
      colorAttachments: [{ view: target.view || target, loadOp: clear ? 'clear' : 'load', clearValue: clear || undefined, storeOp: 'store' }],
    });
    if (n > 0) {
      pass.setPipeline(this._pipe(opts.format || target.format || this.gpu.format, opts.blend || 'alpha'));
      pass.setBindGroup(0, this.bindGroup, [off]);
      pass.setVertexBuffer(0, this.b.buffer);
      pass.draw(6, n, 0, first);
    }
    pass.end();
  }
}

// ------------------------------------------------------------------------------------ sprites

const SPRITE_WGSL = (fragment) => /* wgsl */ `
diagnostic(off, derivative_uniformity);
${CAMERA_WGSL}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) local: vec2f,
  @location(2) @interpolate(flat) color: vec4f,
  @location(3) @interpolate(flat) extra: vec4f,
  @location(4) world: vec2f,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32,
                   @location(0) posSize: vec4f, @location(1) uvRect: vec4f,
                   @location(2) color: vec4f, @location(3) xform: vec4f) -> VOut {
  // xform: x = rotation, y = anchorX, z = anchorY, w = user value
  var corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let local = (c - xform.yz) * posSize.zw;
  let cs = cos(xform.x); let sn = sin(xform.x);
  let w = posSize.xy + vec2f(cs * local.x - sn * local.y, sn * local.x + cs * local.y);
  var o: VOut;
  o.pos = cam.m * vec4f(w, 0.0, 1.0);
  o.uv = mix(uvRect.xy, uvRect.zw, c);
  o.local = c;
  o.color = color;
  o.extra = xform;
  o.world = w;
  return o;
}
${fragment}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let t = textureSample(tex, samp, i.uv);
  return sprite_fs(t, i);
}
`;

const DEFAULT_SPRITE_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  let c = t * i.color;
  if (c.a < 0.004) { discard; }
  return c;
}`;

export class SpriteBatch {
  /**
   * opts: { texture: GPUTextureView | target, filter: 'nearest' | 'linear', capacity,
   *         fragment: WGSL defining fn sprite_fs(t: vec4f, i: VOut) -> vec4f  (custom per-pixel effect;
   *                   VOut has uv, local (0..1 in sprite), color, extra (rotation, anchorX, anchorY, user), world)
   *         }
   * Inside a custom fragment you can read cam.extra (vec4f) set per flush via opts.extra,
   * and cam.viewport.w = seconds (performance.now).
   */
  constructor(gpu, { texture, filter = 'nearest', capacity = 4096, fragment = DEFAULT_SPRITE_FS } = {}) {
    this.gpu = gpu;
    this.texture = texture;
    this.filter = filter;
    this.b = new InstanceBatch(gpu, 16, capacity);
    const device = gpu.device;
    this.module = gpu.shader(SPRITE_WGSL(fragment), 'sprites');
    this.bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    this.layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.pipelines = new Map();
    this.groups = new Map();
  }
  _group(view, filter) {
    const key = view;
    let byFilter = this.groups.get(key);
    if (!byFilter) this.groups.set(key, (byFilter = {}));
    if (!byFilter[filter]) {
      byFilter[filter] = this.gpu.device.createBindGroup({
        layout: this.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.b.ubo, size: 96 } },
          { binding: 1, resource: this.gpu.sampler(filter) },
          { binding: 2, resource: view },
        ],
      });
    }
    return byFilter[filter];
  }
  _pipe(format, blend) {
    const key = `${format}|${blend}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const attrs = [0, 1, 2, 3].map((i) => ({ shaderLocation: i, offset: i * 16, format: 'float32x4' }));
      p = this.gpu.device.createRenderPipeline({
        label: 'sprites',
        layout: this.layout,
        vertex: { module: this.module, entryPoint: 'vs_main', buffers: [{ arrayStride: 64, stepMode: 'instance', attributes: attrs }] },
        fragment: { module: this.module, entryPoint: 'fs_main', targets: [{ format, blend: BLEND[blend] }] },
        primitive: { topology: 'triangle-list' },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }
  begin() {
    this.b.begin();
    return this;
  }
  /**
   * Queue a sprite. (x, y) is the anchor position in world units, w/h its size.
   * opts: { uv: [u0,v0,u1,v1] (default full texture), rotation, anchor: [ax, ay] (default [0.5,0.5]),
   *         color: tint (multiplied), flipX, flipY, user: number (available as i.extra.w) }
   */
  draw(x, y, w, h, opts = {}) {
    const off = this.b.alloc();
    const d = this.b.data;
    let uv = opts.uv || [0, 0, 1, 1];
    if (opts.flipX) uv = [uv[2], uv[1], uv[0], uv[3]];
    if (opts.flipY) uv = [uv[0], uv[3], uv[2], uv[1]];
    const c = opts.color ? rgba(opts.color, opts.alpha ?? 1) : [1, 1, 1, opts.alpha ?? 1];
    const an = opts.anchor || [0.5, 0.5];
    d[off] = x;
    d[off + 1] = y;
    d[off + 2] = w;
    d[off + 3] = h;
    d[off + 4] = uv[0];
    d[off + 5] = uv[1];
    d[off + 6] = uv[2];
    d[off + 7] = uv[3];
    d[off + 8] = c[0];
    d[off + 9] = c[1];
    d[off + 10] = c[2];
    d[off + 11] = c[3] ?? 1;
    d[off + 12] = opts.rotation || 0;
    d[off + 13] = an[0];
    d[off + 14] = an[1];
    d[off + 15] = opts.user || 0;
    return this;
  }
  /**
   * opts: { clear, blend ('alpha' default; use 'premultiplied' for premultiplied textures), texture, filter, format, extra: [4 floats] }
   */
  flush(encoder, target, camera, opts = {}) {
    const [first, n] = this.b.uploadPending();
    const clear = opts.clear || null;
    if (n === 0 && !clear) return;
    const off = this.b.writeCamera(camera, opts.extra || [0, 0, 0, 0]);
    const pass = encoder.beginRenderPass({
      label: 'sprites',
      colorAttachments: [{ view: target.view || target, loadOp: clear ? 'clear' : 'load', clearValue: clear || undefined, storeOp: 'store' }],
    });
    if (n > 0) {
      const tex = opts.texture || this.texture;
      const view = tex.view || tex;
      pass.setPipeline(this._pipe(opts.format || target.format || this.gpu.format, opts.blend || 'alpha'));
      pass.setBindGroup(0, this._group(view, opts.filter || this.filter), [off]);
      pass.setVertexBuffer(0, this.b.buffer);
      pass.draw(6, n, 0, first);
    }
    pass.end();
  }
}

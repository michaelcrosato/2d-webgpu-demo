// Shared helpers for the Particles & VFX scenes (src/scenes/vfx/*). WebGPU only.
//
//   makeStrip(gpu)          -> procedurally painted strip of soft particle textures (with mips)
//   createBloom(gpu)        -> HDR bloom: soft-threshold prefilter + downsample pyramid + tent upsample
//   GlowLines               -> instanced glowing capsules with HDR intensity ('max' or 'add' blending)
//   MeshBatch               -> dynamic triangle meshes with your own fragment shader (ribbons, slashes)
//   SoftSprites             -> SpriteBatch over the strip, premultiplied so additive & alpha mix in one batch
//   tag(ctx)                -> overlay readout
//   TONEMAP_WGSL            -> ACES + sRGB encode + dither, shared by the composites

import { SpriteBatch } from '../../core/batch.js';
import { BLEND } from '../../core/webgpu.js';
import { resolveIncludes, WGSL_MATH_PRELUDE } from '../../core/shaderlib.js';

// ------------------------------------------------------------------------------ overlay readout

export function tag(ctx, css = 'right:8px;bottom:8px') {
  const d = document.createElement('div');
  d.className = 'tag';
  d.style.cssText = css;
  ctx.overlay.append(d);
  return d;
}

// ------------------------------------------------------------------------------ texture strip

export const CELL = 64;
export const TEX = { soft: 0, spark: 1, star: 2, smoke: 3, ring: 4, leaf: 5, flake: 6, puff: 7 };
export const TEX_COUNT = 8;
/** UV rect of a strip cell for SpriteBatch. */
export const cellUV = (i) => [i / TEX_COUNT, 0, (i + 1) / TEX_COUNT, 1];

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const sstep = (a, b, x) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
function hash2(x, y, s) {
  let h = (x * 374761393 + y * 668265263 + s * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x, y, s) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = x - xi;
  const fy = y - yi;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(xi, yi, s);
  const b = hash2(xi + 1, yi, s);
  const c = hash2(xi, yi + 1, s);
  const d = hash2(xi + 1, yi + 1, s);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}
const fbm = (x, y, s) => vnoise(x, y, s) * 0.5 + vnoise(x * 2.1, y * 2.1, s + 1) * 0.3 + vnoise(x * 4.3, y * 4.3, s + 2) * 0.2;

// Each painter returns [gray, alpha] for u, v in [-1, 1].
const PAINTERS = [
  // 0 soft: gaussian glow
  (u, v) => {
    const r = Math.hypot(u, v);
    return [1, Math.exp(-r * r * 5.0) * (1 - sstep(0.7, 1.0, r))];
  },
  // 1 spark: horizontal streak with a hot core (stretch it along the velocity)
  (u, v) => {
    const r = Math.hypot(u, v);
    const streak = Math.exp(-v * v * 36) * Math.pow(Math.max(0, 1 - u * u), 1.6);
    return [1, clamp01(streak + 0.5 * Math.exp(-r * r * 22)) * (1 - sstep(0.85, 1.0, r * 0.98))];
  },
  // 2 star: four-point twinkle
  (u, v) => {
    const r = Math.hypot(u, v);
    const arms = Math.exp(-Math.abs(v) * 22) * Math.pow(Math.max(0, 1 - Math.abs(u)), 2) + Math.exp(-Math.abs(u) * 22) * Math.pow(Math.max(0, 1 - Math.abs(v)), 2);
    const du = (u + v) * 0.7071;
    const dv = (u - v) * 0.7071;
    const diag = Math.exp(-Math.abs(dv) * 30) * Math.pow(Math.max(0, 1 - Math.abs(du) * 1.6), 2) + Math.exp(-Math.abs(du) * 30) * Math.pow(Math.max(0, 1 - Math.abs(dv) * 1.6), 2);
    return [1, clamp01(Math.exp(-r * r * 30) + 0.85 * arms + 0.3 * diag + 0.25 * Math.exp(-r * r * 6)) * (1 - sstep(0.9, 1.0, r))];
  },
  // 3 smoke: noisy puff
  (u, v) => {
    const r = Math.hypot(u, v);
    const n = fbm(u * 2.4 + 5, v * 2.4 + 3, 11);
    const body = 1 - sstep(0.25, 0.95, r + (n - 0.5) * 0.45);
    return [0.82 + 0.18 * n, clamp01(body * (0.55 + 0.6 * n))];
  },
  // 4 ring: thin soft ring (shockwaves, ripples)
  (u, v) => {
    const r = Math.hypot(u, v);
    const x = (r - 0.8) * 10;
    return [1, Math.exp(-x * x) * (1 - sstep(0.94, 1.0, r))];
  },
  // 5 leaf: pointed leaf along +x with a center vein (white, shaded) — tint it
  (u, v) => {
    const x = u * 1.08;
    const t = (x + 0.95) / 1.7; // 0..1 along the blade
    if (t < 0 || t > 1.15) return [1, 0];
    const half = t <= 1 ? 0.46 * Math.pow(Math.sin(Math.PI * Math.min(t, 1)), 0.8) * (1 - 0.25 * t) : 0;
    const edge = half - Math.abs(v);
    let a = clamp01(edge * 28 + 0.5);
    // stem
    if (t > 0.95) a = Math.max(a, clamp01((0.035 - Math.abs(v)) * 30) * (t < 1.13 ? 1 : 0));
    const vein = Math.exp(-v * v * 900) * 0.35;
    const side = Math.exp(-Math.pow(((Math.abs(v) * 1.6 + x) * 6) % 1 - 0.5, 2) * 60) * 0.12 * (half > 0 ? 1 : 0);
    const shade = 0.78 + 0.22 * (v < 0 ? 1 : 0.6) - vein - side;
    return [clamp01(shade), a];
  },
  // 6 flake: six-armed snow crystal with a soft halo
  (u, v) => {
    const r = Math.hypot(u, v);
    let arm = 0;
    for (let k = 0; k < 3; k++) {
      const a = (k * Math.PI) / 3;
      const nx = -Math.sin(a);
      const ny = Math.cos(a);
      const d = Math.abs(u * nx + v * ny);
      arm = Math.max(arm, Math.exp(-d * d * 900) * (1 - sstep(0.62, 0.8, r)));
      // little side branches
      const along = Math.abs(u * Math.cos(a) + v * Math.sin(a));
      for (const b of [0.38, 0.55]) {
        const db = Math.abs(along - b - Math.abs(u * nx + v * ny) * 0.9);
        arm = Math.max(arm, Math.exp(-db * db * 1400) * (Math.abs(u * nx + v * ny) < 0.16 ? 1 : 0));
      }
    }
    return [1, clamp01(arm * 0.95 + Math.exp(-r * r * 22) * 0.9 + 0.18 * Math.exp(-r * r * 5)) * (1 - sstep(0.85, 1.0, r))];
  },
  // 7 puff: billowy cloud puff, top-lit (fireballs, dust, smoke columns)
  (u, v) => {
    let a = 0;
    const blobs = [
      [0, 0.05, 0.62],
      [-0.35, -0.12, 0.42],
      [0.33, -0.15, 0.45],
      [0.05, -0.38, 0.4],
      [-0.25, 0.3, 0.38],
      [0.3, 0.28, 0.36],
    ];
    for (const [bx, by, br] of blobs) {
      const d = Math.hypot(u - bx, v - by) / br;
      a = Math.max(a, 1 - sstep(0.55, 1.0, d));
    }
    const n = fbm(u * 3 + 1, v * 3 + 7, 5);
    a = clamp01(a * (0.75 + 0.5 * n) - 0.08) * (1 - sstep(0.85, 1.0, Math.hypot(u, v)));
    const light = clamp01(0.55 + 0.45 * (-u * 0.45 - v * 0.9) + (n - 0.5) * 0.4);
    return [0.55 + 0.45 * light, a];
  },
];

/** Raw RGBA8 pixels of the strip (gray in rgb, coverage in alpha). */
export function stripPixels() {
  const W = CELL * TEX_COUNT;
  const H = CELL;
  const data = new Uint8Array(W * H * 4);
  for (let c = 0; c < TEX_COUNT; c++) {
    const fn = PAINTERS[c];
    for (let j = 0; j < CELL; j++) {
      for (let i = 0; i < CELL; i++) {
        const u = ((i + 0.5) / CELL) * 2 - 1;
        const v = ((j + 0.5) / CELL) * 2 - 1;
        const [g, a] = fn(u, v);
        const k = (j * W + c * CELL + i) * 4;
        const gv = Math.round(clamp01(g) * 255);
        data[k] = gv;
        data[k + 1] = gv;
        data[k + 2] = gv;
        data[k + 3] = Math.round(clamp01(a) * 255);
      }
    }
  }
  return { data, width: W, height: H };
}

let stripCache = null;
/** GPU texture of the particle strip (mipmapped, sample with gpu.sampler('linear-mip')). */
export function makeStrip(gpu) {
  if (!stripCache) stripCache = stripPixels();
  const { data, width, height } = stripCache;
  const mips = Math.floor(Math.log2(CELL)) + 1;
  const tex = gpu.texture({ size: [width, height], format: 'rgba8unorm', mips, label: 'vfx-strip' });
  gpu.queue.writeTexture({ texture: tex }, data, { bytesPerRow: width * 4 }, [width, height]);
  gpu.generateMips(tex);
  return tex;
}

// ------------------------------------------------------------------------------ tone mapping

/** Composite helpers. Needs include 'color' (tonemapACES) and 'hash' (ign). */
export const TONEMAP_WGSL = /* wgsl */ `
fn vfxTonemap(c: vec3f, px: vec2f) -> vec3f {
  let t = tonemapACES(max(c, vec3f(0.0)));
  // encode for the (non-sRGB) canvas and add 1/255 of noise so dark glows don't band
  return pow(t, vec3f(1.0 / 2.2)) + vec3f((ign(px) - 0.5) / 255.0);
}
`;

// ------------------------------------------------------------------------------ bloom

/**
 * HDR bloom (the "Call of Duty / Unreal" pyramid):
 *   prefilter (soft threshold + Karis average) -> 1/2 -> 1/4 -> ... -> 1/64
 *   then upsample with a 3x3 tent and ADD into the next larger level.
 * render() returns the 1/2-resolution result to add in your composite.
 */
export function createBloom(gpu, { levels = 6 } = {}) {
  const U = gpu.uniforms({ threshold: 'f32', knee: 'f32', clampMax: 'f32', spread: 'f32' }, 'BloomU');
  const pre = gpu.fullscreen({
    label: 'bloom-prefilter',
    textures: ['src'],
    uniforms: U,
    format: 'rgba16float',
    code: /* wgsl */ `
fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + max(c.r, max(c.g, c.b))); }
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ts = 1.0 / TEXSIZE(src);
  let a = TEX(src, uv + ts * vec2f(-1.0, -1.0)).rgb;
  let b = TEX(src, uv + ts * vec2f(1.0, -1.0)).rgb;
  let c = TEX(src, uv + ts * vec2f(-1.0, 1.0)).rgb;
  let d = TEX(src, uv + ts * vec2f(1.0, 1.0)).rgb;
  let wa = karis(a); let wb = karis(b); let wc = karis(c); let wd = karis(d);
  var col = (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
  let br = max(col.r, max(col.g, col.b));
  let k = max(u.knee, 0.0001);
  var soft = clamp(br - u.threshold + k, 0.0, 2.0 * k);
  soft = soft * soft / (4.0 * k);
  let w = max(soft, br - u.threshold) / max(br, 0.0001);
  return vec4f(min(col * w, vec3f(u.clampMax)), 1.0);
}`,
  });
  const down = gpu.fullscreen({
    label: 'bloom-down',
    textures: ['src'],
    format: 'rgba16float',
    code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ts = 1.0 / TEXSIZE(src);
  let c = TEX(src, uv + ts * vec2f(-1.0, -1.0)) + TEX(src, uv + ts * vec2f(1.0, -1.0))
        + TEX(src, uv + ts * vec2f(-1.0, 1.0)) + TEX(src, uv + ts * vec2f(1.0, 1.0));
  return vec4f(c.rgb * 0.25, 1.0);
}`,
  });
  const up = gpu.fullscreen({
    label: 'bloom-up',
    textures: ['src'],
    uniforms: U,
    format: 'rgba16float',
    code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let o = u.spread / TEXSIZE(src);
  var s = TEX(src, uv).rgb * 4.0;
  s += (TEX(src, uv + vec2f(o.x, 0.0)).rgb + TEX(src, uv - vec2f(o.x, 0.0)).rgb
      + TEX(src, uv + vec2f(0.0, o.y)).rgb + TEX(src, uv - vec2f(0.0, o.y)).rgb) * 2.0;
  s += TEX(src, uv + o).rgb + TEX(src, uv - o).rgb + TEX(src, uv + vec2f(o.x, -o.y)).rgb + TEX(src, uv + vec2f(-o.x, o.y)).rgb;
  return vec4f(s / 16.0, 1.0);
}`,
  });
  let chain = [];
  let size = '';
  const ensure = (w, h) => {
    const key = `${w}x${h}`;
    if (key === size) return;
    size = key;
    for (const t of chain) t.destroy();
    chain = [];
    let cw = w;
    let ch = h;
    for (let i = 0; i < levels; i++) {
      cw = Math.max(1, cw >> 1);
      ch = Math.max(1, ch >> 1);
      chain.push(gpu.target(cw, ch, { format: 'rgba16float', label: `bloom${i}` }));
    }
  };
  return {
    levels,
    get chain() {
      return chain;
    },
    /** src: full-res HDR target. Returns the half-res bloom target. */
    render(encoder, src, { threshold = 1.0, knee = 0.5, clampMax = 64, spread = 1.0 } = {}) {
      ensure(src.width, src.height);
      U.set('threshold', threshold).set('knee', knee).set('clampMax', clampMax).set('spread', spread);
      pre.draw(encoder, chain[0], { src });
      for (let i = 1; i < chain.length; i++) down.draw(encoder, chain[i], { src: chain[i - 1] });
      for (let i = chain.length - 1; i > 0; i--) up.draw(encoder, chain[i - 1], { src: chain[i] }, { clear: false, blend: 'add' });
      return chain[0];
    },
    destroy() {
      for (const t of chain) t.destroy();
    },
  };
}

// ------------------------------------------------------------------------------ uniform ring

const SLOT = 256;
class UniformRing {
  constructor(gpu, slots = 48) {
    this.gpu = gpu;
    this.slots = slots;
    this.i = 0;
    this.buf = gpu.track(gpu.device.createBuffer({ size: SLOT * slots, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'vfx-ring' }));
    this.tmp = new Float32Array(24);
    this.frameStamp = -1;
  }
  /** camera matrix + vec4 (viewport w,h, zoom, time) + vec4 extra -> dynamic offset */
  write(cam, time, extra) {
    const off = (this.i++ % this.slots) * SLOT;
    this.tmp.set(cam.matrix(), 0);
    this.tmp[16] = cam.width;
    this.tmp[17] = cam.height;
    this.tmp[18] = cam.zoom;
    this.tmp[19] = time;
    this.tmp.set(extra || [0, 0, 0, 0], 20);
    this.gpu.queue.writeBuffer(this.buf, off, this.tmp);
    return off;
  }
}

const RING_WGSL = /* wgsl */ `
struct Cam { m: mat4x4f, viewport: vec4f, extra: vec4f };
@group(0) @binding(0) var<uniform> cam: Cam;
`;

/** Growable CPU float array + GPU vertex buffer that is re-uploaded each flush. */
class DynBuffer {
  constructor(gpu, floatsPer, capacity, label) {
    this.gpu = gpu;
    this.fp = floatsPer;
    this.cap = capacity;
    this.data = new Float32Array(capacity * floatsPer);
    this.n = 0;
    this.flushed = 0;
    this.label = label;
    this.retired = [];
    this.buf = this._make(capacity);
  }
  _make(cap) {
    return this.gpu.track(this.gpu.device.createBuffer({ size: cap * this.fp * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: this.label }));
  }
  begin() {
    this.n = 0;
    this.flushed = 0;
    for (const b of this.retired) b.destroy();
    this.retired.length = 0;
  }
  alloc(k = 1) {
    if (this.n + k > this.cap) {
      let cap = this.cap * 2;
      while (this.n + k > cap) cap *= 2;
      const nd = new Float32Array(cap * this.fp);
      nd.set(this.data);
      this.data = nd;
      this.retired.push(this.buf);
      this.buf = this._make(cap);
      if (this.flushed) this.gpu.queue.writeBuffer(this.buf, 0, this.data, 0, this.flushed * this.fp);
      this.cap = cap;
    }
    const off = this.n * this.fp;
    this.n += k;
    return off;
  }
  upload() {
    const first = this.flushed;
    const n = this.n - first;
    if (n > 0) this.gpu.queue.writeBuffer(this.buf, first * this.fp * 4, this.data, first * this.fp, n * this.fp);
    this.flushed = this.n;
    return [first, n];
  }
}

function pipelineFactory(gpu, module, layout, buffers, label) {
  const cache = new Map();
  return (format, blend, topology = 'triangle-list') => {
    const key = `${format}|${blend}|${topology}`;
    let p = cache.get(key);
    if (!p) {
      p = gpu.device.createRenderPipeline({
        label,
        layout,
        vertex: { module, entryPoint: 'vs_main', buffers },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format, blend: typeof blend === 'string' ? BLEND[blend] : blend }] },
        primitive: { topology },
      });
      cache.set(key, p);
    }
    return p;
  };
}

function ringLayout(gpu) {
  const bgl = gpu.device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } }],
  });
  return { bgl, layout: gpu.device.createPipelineLayout({ bindGroupLayouts: [bgl] }) };
}

// ------------------------------------------------------------------------------ glow lines

const GLOW_WGSL = /* wgsl */ `
diagnostic(off, derivative_uniformity);
${RING_WGSL}
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) w: vec2f,
  @location(1) @interpolate(flat) seg: vec4f,
  @location(2) @interpolate(flat) prm: vec4f,
  @location(3) @interpolate(flat) col: vec4f,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @location(0) seg: vec4f, @location(1) prm: vec4f, @location(2) col: vec4f) -> VOut {
  let a = seg.xy;
  let b = seg.zw;
  let ext = max(prm.x * 0.5, prm.y * 2.8) + 2.0 / max(cam.viewport.z, 0.001);
  let dv = b - a;
  let len = length(dv);
  let dir = select(vec2f(1.0, 0.0), dv / max(len, 0.00001), len > 0.00001);
  let nrm = vec2f(-dir.y, dir.x);
  var corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[vi];
  let w = mix(a - dir * ext, b + dir * ext, c.x) + nrm * ext * c.y;
  var o: VOut;
  o.pos = cam.m * vec4f(w, 0.0, 1.0);
  o.w = w; o.seg = seg; o.prm = prm; o.col = col;
  return o;
}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f {
  let pa = i.w - i.seg.xy;
  let ba = i.seg.zw - i.seg.xy;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.000001), 0.0, 1.0);
  let d = length(pa - ba * h);
  let aa = max(fwidth(d), 0.0001);
  let core = clamp((i.prm.x * 0.5 - d) / aa + 0.5, 0.0, 1.0);
  let g = d / max(i.prm.y, 0.0001);
  let glow = exp(-g * g);
  let v = i.col.rgb * (core * i.prm.z + glow * i.prm.w) * i.col.a;
  return vec4f(v, 0.0);
}
`;

/**
 * Glowing line segments with HDR intensity. Use blend 'max' so overlapping segments of one
 * polyline don't create bright "beads" at the joints, or 'add' to accumulate.
 *   lines.line(x1, y1, x2, y2, coreWidth, glowRadius, [r,g,b], coreIntensity, glowIntensity, alpha)
 */
export class GlowLines {
  constructor(gpu, { capacity = 4096 } = {}) {
    this.gpu = gpu;
    this.b = new DynBuffer(gpu, 12, capacity, 'glowlines');
    this.ring = new UniformRing(gpu);
    const { bgl, layout } = ringLayout(gpu);
    this.bindGroup = gpu.device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: this.ring.buf, size: 96 } }] });
    const attrs = [0, 1, 2].map((i) => ({ shaderLocation: i, offset: i * 16, format: 'float32x4' }));
    this.pipe = pipelineFactory(gpu, gpu.shader(GLOW_WGSL, 'glowlines'), layout, [{ arrayStride: 48, stepMode: 'instance', attributes: attrs }], 'glowlines');
  }
  begin() {
    this.b.begin();
    return this;
  }
  line(x1, y1, x2, y2, core, glow, color, coreI = 1, glowI = 0.5, alpha = 1) {
    const o = this.b.alloc();
    const d = this.b.data;
    d[o] = x1;
    d[o + 1] = y1;
    d[o + 2] = x2;
    d[o + 3] = y2;
    d[o + 4] = core;
    d[o + 5] = glow;
    d[o + 6] = coreI;
    d[o + 7] = glowI;
    d[o + 8] = color[0];
    d[o + 9] = color[1];
    d[o + 10] = color[2];
    d[o + 11] = alpha;
    return this;
  }
  /** points: [[x,y],...]; widths may be a number or a function(i) */
  polyline(pts, core, glow, color, coreI = 1, glowI = 0.5, alpha = 1) {
    for (let i = 0; i < pts.length - 1; i++) this.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], core, glow, color, coreI, glowI, alpha);
    return this;
  }
  flush(encoder, target, cam, { blend = 'max', clear = null, time = 0 } = {}) {
    const [first, n] = this.b.upload();
    if (!n && !clear) return;
    const off = this.ring.write(cam, time, null);
    const pass = encoder.beginRenderPass({
      label: 'glowlines',
      colorAttachments: [{ view: target.view || target, loadOp: clear ? 'clear' : 'load', clearValue: clear || undefined, storeOp: 'store' }],
    });
    if (n) {
      pass.setPipeline(this.pipe(target.format || this.gpu.format, blend));
      pass.setBindGroup(0, this.bindGroup, [off]);
      pass.setVertexBuffer(0, this.b.buf);
      pass.draw(6, n, 0, first);
    }
    pass.end();
  }
}

// ------------------------------------------------------------------------------ mesh batch

const MESH_WGSL = (fragment, include) => /* wgsl */ `
diagnostic(off, derivative_uniformity);
${RING_WGSL}
${WGSL_MATH_PRELUDE}
${include.length ? resolveIncludes(include) : ''}
struct MOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
  @location(2) data: vec4f,
  @location(3) world: vec2f,
};
@vertex fn vs_main(@location(0) pos: vec2f, @location(1) uv: vec2f, @location(2) color: vec4f, @location(3) data: vec4f) -> MOut {
  var o: MOut;
  o.pos = cam.m * vec4f(pos, 0.0, 1.0);
  o.uv = uv; o.color = color; o.data = data; o.world = pos;
  return o;
}
${fragment}
@fragment fn fs_main(i: MOut) -> @location(0) vec4f { return mesh_fs(i); }
`;

/**
 * Immediate-mode triangles with a custom fragment shader:
 *   fragment WGSL must define fn mesh_fs(i: MOut) -> vec4f   (i.uv, i.color, i.data, i.world;
 *   cam.viewport.w = time passed to flush, cam.extra = flush extra)
 *   mesh.vertex(x, y, u, v, color[4], data[4]) ; three vertices per triangle.
 *   mesh.quad(a, b, c, d) with vertex arrays [x,y,u,v,r,g,b,a,d0,d1,d2,d3]
 */
export class MeshBatch {
  constructor(gpu, { fragment, include = [], capacity = 8192, label = 'mesh' }) {
    this.gpu = gpu;
    this.b = new DynBuffer(gpu, 12, capacity, label);
    this.ring = new UniformRing(gpu);
    const { bgl, layout } = ringLayout(gpu);
    this.bindGroup = gpu.device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: this.ring.buf, size: 96 } }] });
    const attrs = [
      { shaderLocation: 0, offset: 0, format: 'float32x2' },
      { shaderLocation: 1, offset: 8, format: 'float32x2' },
      { shaderLocation: 2, offset: 16, format: 'float32x4' },
      { shaderLocation: 3, offset: 32, format: 'float32x4' },
    ];
    this.pipe = pipelineFactory(gpu, gpu.shader(MESH_WGSL(fragment, include), label), layout, [{ arrayStride: 48, attributes: attrs }], label);
  }
  begin() {
    this.b.begin();
    return this;
  }
  vertex(x, y, u, v, c, dd) {
    const o = this.b.alloc();
    const d = this.b.data;
    d[o] = x;
    d[o + 1] = y;
    d[o + 2] = u;
    d[o + 3] = v;
    d[o + 4] = c[0];
    d[o + 5] = c[1];
    d[o + 6] = c[2];
    d[o + 7] = c[3] ?? 1;
    d[o + 8] = dd ? dd[0] : 0;
    d[o + 9] = dd ? dd[1] : 0;
    d[o + 10] = dd ? dd[2] : 0;
    d[o + 11] = dd ? dd[3] : 0;
    return this;
  }
  /** Raw vertex array [x,y,u,v,r,g,b,a,d0,d1,d2,d3]. */
  v(a) {
    const o = this.b.alloc();
    this.b.data.set(a, o);
    return this;
  }
  tri(a, b, c) {
    return this.v(a).v(b).v(c);
  }
  /** a b on one side, c d on the other (a-c and b-d across). */
  quad(a, b, c, d) {
    return this.v(a).v(b).v(c).v(c).v(b).v(d);
  }
  /**
   * A ribbon along points: pts[i] = {x, y, w (half width), u (along coord), color [4], data [4]}.
   * Builds left/right edges from smoothed normals (miter-free, clamped) and emits quads.
   */
  ribbon(pts) {
    const n = pts.length;
    if (n < 2) return this;
    let prevL = null;
    let prevR = null;
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const a = pts[Math.max(0, i - 1)];
      const b = pts[Math.min(n - 1, i + 1)];
      let tx = b.x - a.x;
      let ty = b.y - a.y;
      const tl = Math.hypot(tx, ty) || 1;
      tx /= tl;
      ty /= tl;
      const nx = -ty;
      const ny = tx;
      const c = p.color;
      const dd = p.data || [0, 0, 0, 0];
      const L = [p.x + nx * p.w, p.y + ny * p.w, p.u, -1, c[0], c[1], c[2], c[3] ?? 1, dd[0], dd[1], dd[2], dd[3]];
      const R = [p.x - nx * p.w, p.y - ny * p.w, p.u, 1, c[0], c[1], c[2], c[3] ?? 1, dd[0], dd[1], dd[2], dd[3]];
      if (prevL) this.quad(prevL, L, prevR, R);
      prevL = L;
      prevR = R;
    }
    return this;
  }
  flush(encoder, target, cam, { blend = 'premultiplied', clear = null, time = 0, extra = null } = {}) {
    const [first, n] = this.b.upload();
    if (!n && !clear) return;
    const off = this.ring.write(cam, time, extra);
    const pass = encoder.beginRenderPass({
      label: 'mesh',
      colorAttachments: [{ view: target.view || target, loadOp: clear ? 'clear' : 'load', clearValue: clear || undefined, storeOp: 'store' }],
    });
    if (n) {
      pass.setPipeline(this.pipe(target.format || this.gpu.format, blend));
      pass.setBindGroup(0, this.bindGroup, [off]);
      pass.setVertexBuffer(0, this.b.buf);
      pass.draw(n, 1, first, 0);
    }
    pass.end();
  }
}

// ------------------------------------------------------------------------------ soft sprites

/**
 * SpriteBatch over the particle strip with a premultiplied-alpha fragment:
 *   user = 1 -> purely additive (glows, sparks), user = 0 -> normal alpha (smoke, debris).
 * Flush with blend 'premultiplied'. Colors may exceed 1 (HDR).
 */
export const SOFT_SPRITE_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  let a = t.a * i.color.a;
  if (a < 0.002) { discard; }
  return vec4f(t.rgb * i.color.rgb * a, a * (1.0 - i.extra.w));
}`;

export function softSprites(gpu, stripView, capacity = 4096) {
  return new SpriteBatch(gpu, { texture: stripView, filter: 'linear-mip', capacity, fragment: SOFT_SPRITE_FS });
}

// ------------------------------------------------------------------------------ misc

/** Small deterministic PRNG (mulberry32). */
export function prng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const lerp = (a, b, t) => a + (b - a) * t;
export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export function hsv(h, s, v) {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
}
/** '#rrggbb' -> linear-ish [r,g,b] (squared as a cheap sRGB->linear). */
export function hexLin(hex, k = 1) {
  const h = hex.replace('#', '');
  const n = parseInt(h.slice(0, 6), 16);
  const c = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  return [Math.pow(c[0], 2.2) * k, Math.pow(c[1], 2.2) * k, Math.pow(c[2], 2.2) * k];
}

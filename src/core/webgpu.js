// WebGPU helpers. One shared GPUDevice; every scene gets its own SceneGPU which tracks
// the resources it creates so they are destroyed automatically when you switch scenes.

import { UniformBlock } from './uniforms.js';
import { resolveIncludes, WGSL_MATH_PRELUDE, WGSL_TEXTURE_PRELUDE } from './shaderlib.js';

let sharedPromise = null;
let errorSink = (msg) => console.error(msg);

/** Route device errors (validation, OOM, shader compile) to the stage overlay. */
export function setGPUErrorSink(fn) {
  errorSink = fn;
}

/** Get (or create) the shared device. Resolves to {adapter, device, format, info} or {error}. */
export async function getWebGPU() {
  if (sharedPromise) {
    const s = await sharedPromise;
    if (!s.lost) return s;
  }
  sharedPromise = (async () => {
    if (!('gpu' in navigator)) return { error: 'This browser does not expose navigator.gpu (WebGPU is unavailable).' };
    let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
    if (!adapter) adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (!adapter) return { error: 'navigator.gpu exists but no GPU adapter was found (blocklisted driver?).' };
    const wanted = ['float32-filterable', 'float32-blendable', 'timestamp-query', 'rg11b10ufloat-renderable', 'bgra8unorm-storage'];
    const requiredFeatures = wanted.filter((f) => adapter.features.has(f));
    const requiredLimits = {};
    for (const k of [
      'maxStorageBufferBindingSize',
      'maxBufferSize',
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupSizeX',
      'maxComputeWorkgroupSizeY',
      'maxStorageBuffersPerShaderStage',
      'maxStorageTexturesPerShaderStage',
      'maxSampledTexturesPerShaderStage',
      'maxTextureDimension2D',
      'maxColorAttachmentBytesPerSample',
      'maxComputeWorkgroupsPerDimension',
    ]) {
      if (adapter.limits[k] !== undefined) requiredLimits[k] = adapter.limits[k];
    }
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    const state = {
      adapter,
      device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      features: new Set(requiredFeatures),
      info: adapter.info || {},
      lost: false,
    };
    device.addEventListener('uncapturederror', (ev) => errorSink(`[WebGPU] ${ev.error.message}`));
    device.lost.then((info) => {
      state.lost = true;
      errorSink(`[WebGPU] device lost: ${info.message || info.reason}`);
    });
    return state;
  })();
  return sharedPromise;
}

const U = () => GPUBufferUsage;
const T = () => GPUTextureUsage;
const STORAGE_FORMATS = new Set([
  'rgba8unorm', 'rgba8snorm', 'rgba8uint', 'rgba8sint', 'rgba16uint', 'rgba16sint', 'rgba16float',
  'r32uint', 'r32sint', 'r32float', 'rg32uint', 'rg32sint', 'rg32float', 'rgba32uint', 'rgba32sint', 'rgba32float',
]);

export const BLEND = {
  none: undefined,
  replace: undefined,
  alpha: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  premultiplied: {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  additive: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  },
  add: {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  },
  multiply: {
    color: { srcFactor: 'dst', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  screen: {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  subtract: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'reverse-subtract' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  },
  max: {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'max' },
  },
  min: {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'min' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'min' },
  },
};

function resolveBlend(b) {
  if (b === undefined || b === null || b === false) return undefined;
  if (typeof b === 'string') {
    if (!(b in BLEND)) throw new Error(`Unknown blend preset "${b}". Use one of: ${Object.keys(BLEND).join(', ')}`);
    return BLEND[b];
  }
  return b;
}

// Give every GPU object a stable id so bind groups can be cached.
const ids = new WeakMap();
let nextId = 1;
const idOf = (o) => {
  if (o === null || o === undefined) return 0;
  if (typeof o !== 'object') return String(o);
  let id = ids.get(o);
  if (!id) ids.set(o, (id = nextId++));
  return id;
};

/** Format a WGSL compile/validation message, adding the offending source lines. */
function formatShaderError(message, code, label) {
  const lines = code.split('\n');
  const seen = new Set();
  let extra = '';
  for (const m of message.matchAll(/:(\d+):(\d+)/g)) {
    const n = +m[1];
    if (seen.has(n) || !lines[n - 1]) continue;
    seen.add(n);
    extra += `\n    line ${n}: ${lines[n - 1].trim()}`;
    if (seen.size >= 3) break;
  }
  return `[WGSL] ${label}: ${message.trim()}${extra}`;
}

/** The fullscreen-triangle vertex stage shared by all fullscreen passes. uv origin is TOP-LEFT. */
export const FULLSCREEN_VS = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
  return o;
}
`;

export class SceneGPU {
  constructor(state, context = null) {
    this.state = state;
    this.device = state.device;
    this.queue = state.device.queue;
    this.format = state.format; // canvas format
    this.features = state.features;
    this.context = context;
    this._res = [];
    this._samplers = new Map();
    this._blit = new Map();
    this._mip = new Map();
  }

  /** Track an object with .destroy() so it is freed when the scene unloads. */
  track(r) {
    this._res.push(r);
    return r;
  }

  destroy() {
    for (const r of this._res) {
      try {
        r.destroy?.();
      } catch {
        /* ignore */
      }
    }
    this._res.length = 0;
  }

  // ---------------------------------------------------------------- resources

  /** Create a buffer. `data` (TypedArray) is uploaded if given. usage: GPUBufferUsage flags or string list. */
  buffer({ size, usage, data = null, label = 'buffer' }) {
    const u = U();
    let flags = usage;
    if (typeof usage === 'string') {
      flags = 0;
      for (const k of usage.split(/[\s|,]+/)) flags |= u[k.toUpperCase().replace(/-/g, '_')];
    }
    const byteSize = Math.max(16, Math.ceil((size ?? data.byteLength) / 4) * 4);
    const buf = this.track(
      this.device.createBuffer({ size: byteSize, usage: flags | u.COPY_DST, label, mappedAtCreation: false }),
    );
    if (data) this.queue.writeBuffer(buf, 0, data.buffer ?? data, data.byteOffset ?? 0, data.byteLength);
    return buf;
  }

  /** Storage buffer (read/write from compute, readable from vertex/fragment). */
  storage(sizeOrData, label = 'storage', extraUsage = 0) {
    const u = U();
    const isData = typeof sizeOrData !== 'number';
    return this.buffer({
      size: isData ? sizeOrData.byteLength : sizeOrData,
      data: isData ? sizeOrData : null,
      usage: u.STORAGE | u.COPY_SRC | u.COPY_DST | extraUsage,
      label,
    });
  }

  /** Vertex buffer. */
  vertexBuffer(data, label = 'vertices') {
    const u = U();
    return this.buffer({ data, usage: u.VERTEX | u.COPY_DST, label });
  }

  /** A UniformBlock (see uniforms.js) bound to a GPU buffer. Call .upload() after setting values. */
  uniforms(fields, structName = 'U') {
    const block = fields instanceof UniformBlock ? fields : new UniformBlock(fields, structName);
    block.device = this.device;
    block.gpuBuffer = this.track(
      this.device.createBuffer({
        size: block.byteSize,
        usage: U().UNIFORM | U().COPY_DST,
        label: `uniforms:${block.structName}`,
      }),
    );
    block.buffer = block.gpuBuffer;
    return block;
  }

  /** Create a texture. size: [w,h] (or [w,h,layers]). */
  texture({ size, format = 'rgba8unorm', usage, mips = 1, label = 'texture', sampleCount = 1, dimension = '2d' }) {
    const t = T();
    const flags =
      usage ??
      (t.TEXTURE_BINDING | t.COPY_DST | t.COPY_SRC | t.RENDER_ATTACHMENT | (STORAGE_FORMATS.has(format) && sampleCount === 1 ? t.STORAGE_BINDING : 0));
    return this.track(
      this.device.createTexture({ size, format, usage: flags, mipLevelCount: mips, label, sampleCount, dimension }),
    );
  }

  /**
   * A render target: { texture, view, width, height, format }.
   * Usable as color attachment, sampled texture and (for storage-capable formats) storage texture.
   */
  target(width, height, { format = 'rgba8unorm', label = 'target', mips = 1, sampleCount = 1 } = {}) {
    width = Math.max(1, Math.floor(width));
    height = Math.max(1, Math.floor(height));
    const texture = this.texture({ size: [width, height], format, label, mips, sampleCount });
    const view = texture.createView(mips > 1 ? { baseMipLevel: 0, mipLevelCount: mips } : undefined);
    return { texture, view, width, height, format, label, destroy: () => texture.destroy() };
  }

  /** Two targets you can swap: read from `.read`, render into `.write`, then `.swap()`. */
  pingPong(width, height, opts = {}) {
    const a = this.target(width, height, { ...opts, label: (opts.label || 'pp') + ':a' });
    const b = this.target(width, height, { ...opts, label: (opts.label || 'pp') + ':b' });
    return {
      a,
      b,
      read: a,
      write: b,
      width: a.width,
      height: a.height,
      swap() {
        const t = this.read;
        this.read = this.write;
        this.write = t;
      },
      destroy() {
        a.destroy();
        b.destroy();
      },
    };
  }

  /**
   * Samplers by name: 'linear' | 'nearest' | 'linear-repeat' | 'nearest-repeat' | 'linear-mirror'
   * | 'nearest-mirror' | 'linear-mip' | 'linear-mip-repeat' | 'aniso'.
   */
  sampler(name = 'linear') {
    if (typeof name !== 'string') return name;
    let s = this._samplers.get(name);
    if (s) return s;
    const filter = name.startsWith('nearest') ? 'nearest' : 'linear';
    const mode = name.includes('repeat') ? 'repeat' : name.includes('mirror') ? 'mirror-repeat' : 'clamp-to-edge';
    const desc = {
      magFilter: filter,
      minFilter: filter,
      mipmapFilter: name.includes('mip') || name === 'aniso' ? 'linear' : 'nearest',
      addressModeU: mode,
      addressModeV: mode,
    };
    if (name === 'aniso') {
      desc.maxAnisotropy = 16;
      desc.addressModeU = desc.addressModeV = 'repeat';
    }
    s = this.device.createSampler(desc);
    this._samplers.set(name, s);
    return s;
  }

  /** Upload an image/canvas/ImageBitmap/OffscreenCanvas/video frame to a new texture. */
  textureFromImage(source, { mips = false, format = 'rgba8unorm', flipY = false, label = 'image', premultiply = false } = {}) {
    const w = source.width || source.videoWidth;
    const h = source.height || source.videoHeight;
    const levels = mips ? Math.floor(Math.log2(Math.max(w, h))) + 1 : 1;
    const t = T();
    const texture = this.texture({
      size: [w, h],
      format,
      mips: levels,
      label,
      usage: t.TEXTURE_BINDING | t.COPY_DST | t.RENDER_ATTACHMENT | t.COPY_SRC,
    });
    this.queue.copyExternalImageToTexture(
      { source, flipY },
      { texture, premultipliedAlpha: premultiply },
      [w, h],
    );
    if (levels > 1) this.generateMips(texture);
    return texture;
  }

  /** Create a texture from raw pixel data (Uint8Array rgba8 by default, Float32Array for float formats). */
  textureFromData(width, height, data, { format = 'rgba8unorm', label = 'data-texture', bytesPerPixel } = {}) {
    const texture = this.texture({ size: [width, height], format, label });
    const bpp =
      bytesPerPixel ?? { rgba8unorm: 4, r8unorm: 1, rg8unorm: 2, rgba16float: 8, rgba32float: 16, r32float: 4, rg32float: 8, r32uint: 4 }[format] ?? 4;
    this.queue.writeTexture({ texture }, data, { bytesPerRow: width * bpp }, [width, height]);
    return texture;
  }

  /** Regenerate the mip chain of a texture by repeated linear downsampling. */
  generateMips(texture) {
    const format = texture.format;
    let pipe = this._mip.get(format);
    if (!pipe) {
      const module = this.device.createShaderModule({
        code:
          FULLSCREEN_VS +
          `@group(0) @binding(0) var s: sampler; @group(0) @binding(1) var t: texture_2d<f32>;
           @fragment fn fs_main(i: VSOut) -> @location(0) vec4f { return textureSample(t, s, i.uv); }`,
      });
      pipe = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
      });
      this._mip.set(format, pipe);
    }
    const enc = this.device.createCommandEncoder();
    const s = this.sampler('linear');
    for (let i = 1; i < texture.mipLevelCount; i++) {
      const src = texture.createView({ baseMipLevel: i - 1, mipLevelCount: 1 });
      const dst = texture.createView({ baseMipLevel: i, mipLevelCount: 1 });
      const bg = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: s },
          { binding: 1, resource: src },
        ],
      });
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: dst, loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }
    this.queue.submit([enc.finish()]);
  }

  // ---------------------------------------------------------------- shaders

  /** Create a shader module and report compile errors (with source lines) to the overlay. */
  shader(code, label = 'shader') {
    // Note: we avoid module.getCompilationInfo() (it can spuriously lose the device on some
    // software adapters); the validation error message already carries line:col information.
    this.device.pushErrorScope('validation');
    const module = this.device.createShaderModule({ code, label });
    this.device.popErrorScope().then((err) => {
      if (err) errorSink(formatShaderError(err.message, code, label));
    });
    return module;
  }

  /**
   * Declarative program: you describe bindings, it generates the WGSL `@binding` declarations,
   * the bind group layout and caches bind groups.
   *
   * bindings: { name: descriptor } in order. Descriptors:
   *   { uniform: UniformBlock }            -> var<uniform> name: <StructName>  (struct generated)
   *   { uniform: 'MyStruct' }              -> var<uniform> name: MyStruct      (you write the struct)
   *   { storage: 'array<Particle>', access: 'read' | 'read_write' }
   *   { texture: true | 'float' | 'unfilterable-float' | 'uint' | 'sint' | 'depth', dim?: '2d' | '2d-array' }
   *   { sampler: true | 'filtering' | 'non-filtering' | 'comparison' }
   *   { storageTexture: 'rgba8unorm', access: 'write' | 'read' | 'read_write' }
   * include: ['noise', 'sdf', ...] portable library code to prepend.
   */
  program({ label = 'program', code, bindings = {}, include = [], prelude = true }) {
    const S = GPUShaderStage;
    const entries = [];
    const decls = [];
    const structs = new Set();
    const names = Object.keys(bindings);
    names.forEach((name, binding) => {
      const b = bindings[name];
      if (b.uniform) {
        let type = b.uniform;
        if (b.uniform instanceof UniformBlock) {
          type = b.uniform.structName;
          if (!structs.has(type)) {
            structs.add(type);
            decls.push(b.uniform.wgsl());
          }
        }
        decls.push(`@group(0) @binding(${binding}) var<uniform> ${name}: ${type};`);
        entries.push({ binding, visibility: S.VERTEX | S.FRAGMENT | S.COMPUTE, buffer: { type: 'uniform' } });
      } else if (b.storage) {
        const rw = b.access === 'read_write' || b.access === 'read-write';
        decls.push(`@group(0) @binding(${binding}) var<storage, ${rw ? 'read_write' : 'read'}> ${name}: ${b.storage};`);
        entries.push({
          binding,
          visibility: rw ? S.FRAGMENT | S.COMPUTE : S.VERTEX | S.FRAGMENT | S.COMPUTE,
          buffer: { type: rw ? 'storage' : 'read-only-storage' },
        });
      } else if (b.texture) {
        const st = b.texture === true ? 'float' : b.texture;
        const dim = b.dim || '2d';
        const wgslDim = dim.replace('-', '_');
        const type =
          st === 'depth'
            ? `texture_depth_${wgslDim}`
            : `texture_${wgslDim}<${st === 'uint' ? 'u32' : st === 'sint' ? 'i32' : 'f32'}>`;
        decls.push(`@group(0) @binding(${binding}) var ${name}: ${type};`);
        entries.push({ binding, visibility: S.VERTEX | S.FRAGMENT | S.COMPUTE, texture: { sampleType: st, viewDimension: dim } });
      } else if (b.sampler) {
        const type = b.sampler === true ? 'filtering' : b.sampler;
        decls.push(`@group(0) @binding(${binding}) var ${name}: ${type === 'comparison' ? 'sampler_comparison' : 'sampler'};`);
        entries.push({ binding, visibility: S.VERTEX | S.FRAGMENT | S.COMPUTE, sampler: { type } });
      } else if (b.storageTexture) {
        const access = (b.access || 'write').replace('-only', '').replace('-', '_');
        const apiAccess = { write: 'write-only', read: 'read-only', read_write: 'read-write' }[access];
        decls.push(`@group(0) @binding(${binding}) var ${name}: texture_storage_2d<${b.storageTexture}, ${access}>;`);
        entries.push({
          binding,
          visibility: access === 'read' ? S.VERTEX | S.FRAGMENT | S.COMPUTE : S.FRAGMENT | S.COMPUTE,
          storageTexture: { access: apiAccess, format: b.storageTexture, viewDimension: '2d' },
        });
      } else {
        throw new Error(`program(${label}): binding "${name}" has no recognised type`);
      }
    });
    const libs = include.length ? resolveIncludes(include) : '';
    const fullCode = [
      'diagnostic(off, derivative_uniformity);',
      decls.join('\n'),
      prelude ? WGSL_MATH_PRELUDE : '',
      libs,
      code,
    ].join('\n');
    const module = this.shader(fullCode, label);
    const bgl = this.device.createBindGroupLayout({ entries, label: `${label}:bgl` });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bgl], label: `${label}:layout` });
    return new Program(this, { label, module, bgl, layout, names, bindings, code: fullCode });
  }

  /**
   * A fullscreen pass. Your code must define:
   *   fn shade(uv: vec2f, px: vec2f) -> vec4f
   * (or its own `@fragment fn fs_main(in: VSOut) -> @location(0) vec4f`).
   * uniforms: UniformBlock (bound as `u`), textures: ['src', ...] (bound by name),
   * samplers `samp` (linear clamp) and `sampR` (linear repeat) are always available, as are
   * TEX(), TEXR(), TEXN(), LOAD(), TEXSIZE() helpers.
   */
  fullscreen({ label = 'fullscreen', code, uniforms = null, textures = [], storage = {}, include = [], format, blend, unfilterable = [] }) {
    const bindings = {};
    if (uniforms) bindings.u = { uniform: uniforms };
    bindings.samp = { sampler: 'filtering' };
    bindings.sampR = { sampler: 'filtering' };
    for (const t of textures) bindings[t] = { texture: unfilterable.includes(t) ? 'unfilterable-float' : 'float' };
    for (const [k, v] of Object.entries(storage)) bindings[k] = { storage: v, access: 'read' };
    const hasFs = /@fragment/.test(code);
    const full =
      WGSL_TEXTURE_PRELUDE +
      FULLSCREEN_VS +
      code +
      (hasFs ? '' : `\n@fragment fn fs_main(in: VSOut) -> @location(0) vec4f { return shade(in.uv, in.pos.xy); }\n`);
    const prog = this.program({ label, code: full, bindings, include });
    const pipelines = new Map();
    const getPipe = (fmt, bl) => {
      const key = `${fmt}|${typeof bl === 'string' ? bl : JSON.stringify(bl)}`;
      let p = pipelines.get(key);
      if (!p) pipelines.set(key, (p = prog.renderPipeline({ format: fmt, blend: bl })));
      return p;
    };
    const self = this;
    return {
      program: prog,
      uniforms,
      /**
       * draw(encoder, target, resources?, opts?)
       *   target: GPUTextureView | {view, format} (a target()) | 'canvas'-like object {view, format}
       *   resources: { textureName: view|target, storageName: buffer }
       *   opts: { clear: [r,g,b,a] | false, blend, format, viewport: [x,y,w,h] }
       */
      draw(encoder, target, resources = {}, opts = {}) {
        if (uniforms) uniforms.upload();
        const view = target.view || target;
        const fmt = opts.format || target.format || format || self.format;
        const pipe = getPipe(fmt, opts.blend !== undefined ? opts.blend : blend);
        const res = { samp: self.sampler('linear'), sampR: self.sampler('linear-repeat'), ...resources };
        if (uniforms) res.u = uniforms;
        const clear = opts.clear === undefined ? [0, 0, 0, 1] : opts.clear;
        const pass = encoder.beginRenderPass({
          label,
          colorAttachments: [
            {
              view,
              loadOp: clear ? 'clear' : 'load',
              clearValue: clear || undefined,
              storeOp: 'store',
            },
          ],
        });
        pass.setPipeline(pipe);
        pass.setBindGroup(0, prog.bind(res));
        if (opts.viewport) pass.setViewport(...opts.viewport, 0, 1);
        pass.draw(3);
        pass.end();
      },
    };
  }

  /** Compute program convenience: gpu.compute({code, bindings, include}).dispatch(encoder, 'main', [x,y], resources) */
  compute(opts) {
    return this.program(opts);
  }

  /**
   * Copy/scale one texture into another with a draw (works across sizes and formats).
   * filter: 'linear' | 'nearest'.
   */
  blit(encoder, src, dst, { filter = 'linear', clear = [0, 0, 0, 1], blend } = {}) {
    const fmt = dst.format || this.format;
    const key = `${fmt}|${filter}|${blend || ''}`;
    let entry = this._blit.get(key);
    if (!entry) {
      const prog = this.program({
        label: 'blit',
        prelude: false,
        bindings: { s: { sampler: 'filtering' }, t: { texture: 'float' } },
        code:
          FULLSCREEN_VS +
          `@fragment fn fs_main(i: VSOut) -> @location(0) vec4f { return textureSampleLevel(t, s, i.uv, 0.0); }`,
      });
      entry = { prog, pipe: prog.renderPipeline({ format: fmt, blend }) };
      this._blit.set(key, entry);
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: dst.view || dst, loadOp: clear ? 'clear' : 'load', clearValue: clear || undefined, storeOp: 'store' }],
    });
    pass.setPipeline(entry.pipe);
    pass.setBindGroup(0, entry.prog.bind({ s: this.sampler(filter), t: src.view || src }));
    pass.draw(3);
    pass.end();
  }

  /** Clear a view to a color. */
  clear(encoder, target, color = [0, 0, 0, 1]) {
    encoder
      .beginRenderPass({ colorAttachments: [{ view: target.view || target, loadOp: 'clear', clearValue: color, storeOp: 'store' }] })
      .end();
  }
}

export class Program {
  constructor(gpu, { label, module, bgl, layout, names, bindings, code }) {
    this.gpu = gpu;
    this.label = label;
    this.module = module;
    this.bindGroupLayout = bgl;
    this.layout = layout;
    this.names = names;
    this.bindings = bindings;
    this.code = code;
    this._bgCache = new Map();
    this._compute = new Map();
  }

  /** Create (or fetch cached) bind group from a { bindingName: resource } map. */
  bind(resources) {
    const gpu = this.gpu;
    let key = '';
    const entries = [];
    for (let i = 0; i < this.names.length; i++) {
      const name = this.names[i];
      let r = resources[name];
      const desc = this.bindings[name];
      if (r === undefined) {
        if (desc.sampler) r = gpu.sampler(desc.sampler === 'non-filtering' ? 'nearest' : 'linear');
        else throw new Error(`${this.label}: missing resource for binding "${name}"`);
      }
      if (typeof r === 'string' && desc.sampler) r = gpu.sampler(r);
      let resource;
      if (r instanceof UniformBlock) resource = { buffer: r.gpuBuffer };
      else if (r instanceof GPUBuffer) resource = { buffer: r };
      else if (r instanceof GPUTexture) resource = r.createView();
      else if (r instanceof GPUTextureView || r instanceof GPUSampler) resource = r;
      else if (r && r.view) resource = r.view;
      else if (r && r.buffer) resource = r;
      else throw new Error(`${this.label}: bad resource for "${name}"`);
      const rid = resource.buffer ? `${idOf(resource.buffer)}:${resource.offset || 0}:${resource.size || ''}` : idOf(resource);
      key += rid + ',';
      entries.push({ binding: i, resource });
    }
    let bg = this._bgCache.get(key);
    if (!bg) {
      if (this._bgCache.size > 64) this._bgCache.clear();
      bg = gpu.device.createBindGroup({ layout: this.bindGroupLayout, entries, label: `${this.label}:bg` });
      this._bgCache.set(key, bg);
    }
    return bg;
  }

  /**
   * Create a render pipeline.
   * opts: { format, blend, vs='vs_main', fs='fs_main', topology='triangle-list', buffers=[], targets, depth, sampleCount }
   */
  renderPipeline(opts = {}) {
    const fmt = opts.format || this.gpu.format;
    const targets = opts.targets || [{ format: fmt, blend: resolveBlend(opts.blend) }];
    return this.gpu.device.createRenderPipeline({
      label: `${this.label}:render`,
      layout: this.layout,
      vertex: { module: this.module, entryPoint: opts.vs || 'vs_main', buffers: opts.buffers || [] },
      fragment: { module: this.module, entryPoint: opts.fs || 'fs_main', targets },
      primitive: { topology: opts.topology || 'triangle-list', stripIndexFormat: opts.stripIndexFormat },
      depthStencil: opts.depth,
      multisample: opts.sampleCount ? { count: opts.sampleCount } : undefined,
    });
  }

  /** Get (cached) compute pipeline for an entry point. */
  computePipeline(entry = 'main') {
    let p = this._compute.get(entry);
    if (!p) {
      p = this.gpu.device.createComputePipeline({
        label: `${this.label}:${entry}`,
        layout: this.layout,
        compute: { module: this.module, entryPoint: entry },
      });
      this._compute.set(entry, p);
    }
    return p;
  }

  /** Dispatch a compute entry point. groups: number | [x, y, z]. */
  dispatch(encoder, entry, groups, resources) {
    const pass = encoder.beginComputePass({ label: `${this.label}:${entry}` });
    pass.setPipeline(this.computePipeline(entry));
    pass.setBindGroup(0, this.bind(resources));
    const g = Array.isArray(groups) ? groups : [groups];
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(g[0])), Math.max(1, Math.ceil(g[1] || 1)), Math.max(1, Math.ceil(g[2] || 1)));
    pass.end();
  }
}

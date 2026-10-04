// shaderScene(): build a scene from fragment shaders only — one portable-WGSL source that runs
// on BOTH WebGPU and WebGL2 (translated automatically). Supports Shadertoy-style multi-pass
// "buffers" with feedback (each pass can read every pass's previous output, including its own),
// a built-in procedural game scene as an input image, and extra textures.
//
// See AUTHORING.md for the full contract. Minimal example:
//
//   export default shaderScene({
//     controls: [{ type: 'slider', key: 'scale', label: 'Scale', min: 1, max: 20, value: 5 }],
//     uniforms: { scale: 'f32' },
//     include: ['noise'],
//     code: `fn shade(uv: vec2f, px: vec2f) -> vec4f {
//              let n = fbm(uv * u.scale + u.time * 0.1, 5);
//              return vec4f(vec3f(n * 0.5 + 0.5), 1.0);
//            }`,
//   });

import { UniformBlock, hexToRgb } from './uniforms.js';
import { GAME_SCENE_CODE, GAME_SCENE_INCLUDES } from './gamescene.js';

export const BUILTIN_UNIFORMS = {
  resolution: 'vec2f', // canvas size in pixels
  time: 'f32', // seconds since start (pausable)
  frame: 'f32', // frames since start/reset
  mouse: 'vec4f', // xy: pointer px (top-left origin), z: 1 if pressed, w: 1 if over canvas
  pmouse: 'vec2f', // pointer px last frame
  dt: 'f32', // seconds since last frame
  example: 'f32', // index of the selected example tab
};

const RESERVED_PASS_NAMES = new Set(['u', 'samp', 'sampR', 'texture', 'input', 'output', 'sample', 'filter']);

function makeUniformBlock(spec) {
  const fields = { ...BUILTIN_UNIFORMS };
  for (const [k, v] of Object.entries(spec.uniforms || {})) {
    if (fields[k]) throw new Error(`shaderScene: uniform "${k}" collides with a built-in`);
    fields[k] = v;
  }
  return new UniformBlock(fields, 'U');
}

/** Map control values to uniform values (select -> option index unless numeric, toggle -> 0/1, color -> rgb). */
function autoBind(spec, params, block) {
  const controls = spec.controls || [];
  for (const c of controls) {
    if (!c.key || !block.has(c.key)) continue;
    let v = params[c.key];
    if (c.type === 'select') {
      const opts = c.options.map((o) => (typeof o === 'object' ? o.value : o));
      v = typeof v === 'number' ? v : Math.max(0, opts.indexOf(v));
    } else if (c.type === 'toggle') v = v ? 1 : 0;
    else if (c.type === 'color') v = hexToRgb(v);
    block.set(c.key, v);
  }
  if (spec.bind) {
    const extra = spec.bind(params, block.ctx) || {};
    for (const [k, v] of Object.entries(extra)) if (block.has(k)) block.set(k, v);
  }
}

function writeBuiltins(block, ctx) {
  const p = ctx.pointer;
  block.set('resolution', [ctx.width, ctx.height]);
  block.set('time', ctx.time);
  block.set('frame', block._frame);
  block.set('mouse', [p.x, p.y, p.down ? 1 : 0, p.over ? 1 : 0]);
  block.set('pmouse', [block._pm ? block._pm[0] : p.x, block._pm ? block._pm[1] : p.y]);
  block.set('dt', ctx.dt);
  block.set('example', ctx.exampleIndex || 0);
  block._pm = [p.x, p.y];
}

function resolveSize(pass, ctx) {
  if (typeof pass.size === 'function') return pass.size(ctx.params, ctx);
  if (Array.isArray(pass.size)) return pass.size;
  const s = typeof pass.scale === 'function' ? pass.scale(ctx.params) : pass.scale ?? 1;
  return [Math.max(1, Math.round(ctx.width * s)), Math.max(1, Math.round(ctx.height * s))];
}

function resolveIterations(pass, params) {
  const it = pass.iterations;
  if (it === undefined) return 1;
  if (typeof it === 'number') return it;
  if (typeof it === 'function') return it(params);
  return Math.round(params[it] ?? 1);
}

function normalizePasses(spec) {
  const passes = (spec.passes || []).map((p) => ({ format: 'rgba16float', ...p }));
  for (const p of passes) {
    if (!/^[a-zA-Z]\w*$/.test(p.name) || RESERVED_PASS_NAMES.has(p.name)) throw new Error(`shaderScene: bad pass name "${p.name}"`);
  }
  return passes;
}

function textureNames(spec, passes) {
  const names = passes.map((p) => p.name);
  if (spec.input === 'game') names.push('game');
  for (const k of Object.keys(spec.textures || {})) names.push(k);
  return names;
}

async function loadExtraTextures(spec) {
  const out = {};
  for (const [name, t] of Object.entries(spec.textures || {})) {
    const src = typeof t === 'function' ? await t() : typeof t.source === 'function' ? await t.source() : t.source;
    out[name] = { source: src, filter: t.filter || 'linear', wrap: t.wrap || 'clamp' };
  }
  return out;
}

function renderScaleOf(spec, params, ctx) {
  const r = spec.renderScale;
  if (r === undefined) return 1;
  if (typeof r === 'number') return r;
  if (typeof r === 'function') return r(params, ctx);
  return params[r] ?? 1;
}

// ---------------------------------------------------------------------------------------- WebGPU

async function initGPU(spec, ctx) {
  const gpu = ctx.gpu;
  const passes = normalizePasses(spec);
  const texNames = textureNames(spec, passes);
  const block = gpu.uniforms(makeUniformBlock(spec));
  block.ctx = ctx;
  block._frame = 0;
  const include = spec.include || [];
  const unfilterable = passes.filter((p) => /32f|32float/.test(p.format)).map((p) => p.name);
  const fmt = (f) => (f === 'rgba8' ? 'rgba8unorm' : f === 'rgba16f' ? 'rgba16float' : f === 'rgba32f' ? 'rgba32float' : f);

  const extras = await loadExtraTextures(spec);
  const extraTex = {};
  for (const [name, e] of Object.entries(extras)) {
    extraTex[name] = gpu.textureFromImage(e.source, { label: name }).createView();
  }

  const mk = (label, code, format, inc = include, tex = texNames) =>
    gpu.fullscreen({ label, code, uniforms: block, textures: tex, include: inc, format, unfilterable });
  const gameInclude = [...new Set([...include, ...GAME_SCENE_INCLUDES])];

  const passFx = passes.map((p) => mk(`pass:${p.name}`, p.code, fmt(p.format)));
  const imageFx = mk('image', spec.code, null);
  const gameFx = spec.input === 'game' ? mk('game', GAME_SCENE_CODE, 'rgba8unorm', gameInclude, []) : null;

  let targets = {}; // name -> pingPong
  let gameTarget = null;
  let lowres = null;
  let sizes = {};

  const blank = gpu.target(1, 1, { label: 'blank' });

  const allocate = () => {
    for (const p of passes) {
      const [w, h] = resolveSize(p, ctx);
      const key = `${w}x${h}`;
      if (sizes[p.name] === key) continue;
      targets[p.name]?.destroy();
      targets[p.name] = gpu.pingPong(w, h, { format: fmt(p.format), label: p.name });
      sizes[p.name] = key;
      block._frame = 0;
    }
    if (gameFx) {
      const [w, h] = [ctx.width, ctx.height];
      if (!gameTarget || gameTarget.width !== w || gameTarget.height !== h) {
        gameTarget?.destroy();
        gameTarget = gpu.target(w, h, { label: 'game' });
      }
    }
  };
  allocate();

  const resources = () => {
    const r = { ...extraTex };
    for (const p of passes) r[p.name] = targets[p.name].read;
    if (gameFx) r.game = gameTarget;
    for (const n of texNames) if (!r[n]) r[n] = blank;
    return r;
  };

  const clearAll = (encoder) => {
    for (const p of passes) {
      const pp = targets[p.name];
      gpu.clear(encoder, pp.a, [0, 0, 0, 0]);
      gpu.clear(encoder, pp.b, [0, 0, 0, 0]);
    }
  };

  let needClear = true;

  return {
    resize() {
      allocate();
      needClear = true;
    },
    reset() {
      block._frame = 0;
      needClear = true;
    },
    onAction(key) {
      if (key === 'reset') this.reset();
      spec.onAction?.(key, ctx);
    },
    onChange(key, value) {
      // pass sizes may depend on params
      if (passes.some((p) => typeof p.size === 'function' || typeof p.scale === 'function')) {
        allocate();
      }
      if (spec.resetOn?.includes(key)) this.reset();
    },
    onExample() {
      if (spec.resetOnExample !== false && passes.length) this.reset();
    },
    frame(ctx) {
      const enc = ctx.encoder;
      allocate();
      if (needClear) {
        clearAll(enc);
        needClear = false;
      }
      writeBuiltins(block, ctx);
      autoBind(spec, ctx.params, block);
      block.upload();
      if (gameFx) gameFx.draw(enc, gameTarget, resources());
      if (!ctx.paused || block._frame === 0) {
        for (let i = 0; i < passes.length; i++) {
          const p = passes[i];
          const n = resolveIterations(p, ctx.params);
          for (let k = 0; k < n; k++) {
            const pp = targets[p.name];
            passFx[i].draw(enc, pp.write, resources(), { clear: false });
            pp.swap();
          }
        }
        block._frame++;
      }
      const rs = renderScaleOf(spec, ctx.params, ctx);
      if (rs < 0.999) {
        const w = Math.max(1, Math.round(ctx.width * rs));
        const h = Math.max(1, Math.round(ctx.height * rs));
        if (!lowres || lowres.width !== w || lowres.height !== h) {
          lowres?.destroy();
          lowres = gpu.target(w, h, { format: ctx.gpu.format, label: 'lowres' });
        }
        imageFx.draw(enc, lowres, resources());
        gpu.blit(enc, lowres, { view: ctx.target, format: gpu.format }, { filter: 'nearest' });
      } else {
        imageFx.draw(enc, { view: ctx.target, format: gpu.format }, resources());
      }
    },
    destroy() {},
  };
}

// ---------------------------------------------------------------------------------------- WebGL2

async function initGL(spec, ctx) {
  const kit = ctx.glkit;
  const passes = normalizePasses(spec);
  const texNames = textureNames(spec, passes);
  const block = kit.uniforms(makeUniformBlock(spec));
  block.ctx = ctx;
  block._frame = 0;
  const include = spec.include || [];
  const fmt = (f) => (/32/.test(f) ? 'rgba32f' : /16/.test(f) ? 'rgba16f' : 'rgba8');

  const extras = await loadExtraTextures(spec);
  const extraTex = {};
  for (const [name, e] of Object.entries(extras)) {
    extraTex[name] = kit.textureFromImage(e.source, { filter: e.filter, wrap: e.wrap });
  }

  const mk = (label, code, inc = include, tex = texNames) => kit.fullscreen({ label, code, uniforms: block, textures: tex, include: inc });
  const gameInclude = [...new Set([...include, ...GAME_SCENE_INCLUDES])];
  const passFx = passes.map((p) => mk(`pass:${p.name}`, p.code));
  const imageFx = mk('image', spec.code);
  const gameFx = spec.input === 'game' ? mk('game', GAME_SCENE_CODE, gameInclude, []) : null;

  const targets = {};
  const sizes = {};
  let gameTarget = null;
  let lowres = null;
  const blank = kit.target(1, 1);

  const allocate = () => {
    for (const p of passes) {
      const [w, h] = resolveSize(p, ctx);
      const key = `${w}x${h}`;
      if (sizes[p.name] === key) continue;
      targets[p.name]?.destroy();
      targets[p.name] = kit.pingPong(w, h, { format: fmt(p.format), filter: p.filter || 'linear' });
      sizes[p.name] = key;
      block._frame = 0;
    }
    if (gameFx && (!gameTarget || gameTarget.width !== ctx.width || gameTarget.height !== ctx.height)) {
      gameTarget?.destroy();
      gameTarget = kit.target(ctx.width, ctx.height);
    }
  };
  allocate();

  const resources = () => {
    const r = { ...extraTex };
    for (const p of passes) r[p.name] = targets[p.name].read;
    if (gameFx) r.game = gameTarget;
    for (const n of texNames) if (!r[n]) r[n] = blank;
    return r;
  };

  const clearAll = () => {
    const gl = ctx.gl;
    for (const p of passes) {
      for (const t of [targets[p.name].a, targets[p.name].b]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };
  let needClear = true;

  return {
    resize() {
      allocate();
      needClear = true;
    },
    reset() {
      block._frame = 0;
      needClear = true;
    },
    onAction(key) {
      if (key === 'reset') this.reset();
      spec.onAction?.(key, ctx);
    },
    onChange(key) {
      if (passes.some((p) => typeof p.size === 'function' || typeof p.scale === 'function')) allocate();
      if (spec.resetOn?.includes(key)) this.reset();
    },
    onExample() {
      if (spec.resetOnExample !== false && passes.length) this.reset();
    },
    frame(ctx) {
      allocate();
      if (needClear) {
        clearAll();
        needClear = false;
      }
      writeBuiltins(block, ctx);
      autoBind(spec, ctx.params, block);
      if (gameFx) gameFx.draw(gameTarget, resources());
      if (!ctx.paused || block._frame === 0) {
        for (let i = 0; i < passes.length; i++) {
          const p = passes[i];
          const n = resolveIterations(p, ctx.params);
          for (let k = 0; k < n; k++) {
            const pp = targets[p.name];
            passFx[i].draw(pp.write, resources(), { clear: false });
            pp.swap();
          }
        }
        block._frame++;
      }
      const rs = renderScaleOf(spec, ctx.params, ctx);
      if (rs < 0.999) {
        const w = Math.max(1, Math.round(ctx.width * rs));
        const h = Math.max(1, Math.round(ctx.height * rs));
        if (!lowres || lowres.width !== w || lowres.height !== h) {
          lowres?.destroy();
          lowres = kit.target(w, h, { filter: 'nearest' });
        }
        imageFx.draw(lowres, resources());
        kit.blit(lowres, null, { filter: 'nearest', canvasSize: [ctx.width, ctx.height] });
      } else {
        imageFx.draw(null, resources(), { canvasSize: [ctx.width, ctx.height] });
      }
    },
    destroy() {},
  };
}

/** Define a fragment-shader-only scene that runs on WebGPU and WebGL2. */
export function shaderScene(spec) {
  const backends = spec.gl === false ? ['webgpu'] : ['webgpu', 'webgl2'];
  return {
    ...spec,
    kind: 'shader',
    backends,
    init: (ctx) => initGPU(spec, ctx),
    initGL: spec.gl === false ? undefined : (ctx) => initGL(spec, ctx),
  };
}

// WebGL2 helpers mirroring the WebGPU helpers (subset).
//
// Convention trick: when rendering into an offscreen framebuffer we flip clip-space Y, so that
// texel row 0 holds the TOP of the image — exactly like WebGPU. That keeps uv (top-left origin)
// identical across both backends, and lets portable shaders sample textures the same way.

import { UniformBlock } from './uniforms.js';
import { wgslToGlsl } from './wgsl2glsl.js';
import { resolveIncludes, GLSL_MATH_PRELUDE, GLSL_TEXTURE_PRELUDE } from './shaderlib.js';

let errorSink = (msg) => console.error(msg);
export function setGLErrorSink(fn) {
  errorSink = fn;
}

export const GL_FULLSCREEN_VS = /* glsl */ `#version 300 es
uniform float u_flipY;
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  v_uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  gl_Position = vec4(p.x, p.y * u_flipY, 0.0, 1.0);
}`;

export const GLSL_HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;

const PRELUDE_FUNCS = { fmod: 'f32', fmod2: 'vec2f', fmod3: 'vec3f' };

function formatGLError(log, src, label) {
  const lines = src.split('\n');
  return log
    .split('\n')
    .filter(Boolean)
    .map((l) => {
      const m = /ERROR:\s*\d+:(\d+):/.exec(l);
      if (m) {
        const n = +m[1];
        return `${label}: ${l}\n    ${(lines[n - 1] || '').trim()}`;
      }
      return `${label}: ${l}`;
    })
    .join('\n');
}

const GL_BLEND = {
  alpha: (gl) => gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA),
  premultiplied: (gl) => gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA),
  additive: (gl) => gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE),
  add: (gl) => gl.blendFunc(gl.ONE, gl.ONE),
  multiply: (gl) => gl.blendFuncSeparate(gl.DST_COLOR, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA),
  screen: (gl) => gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_COLOR, gl.ONE, gl.ONE_MINUS_SRC_ALPHA),
};

export class GLKit {
  constructor(gl) {
    this.gl = gl;
    this._res = [];
    this.floatRT = !!gl.getExtension('EXT_color_buffer_float');
    this.floatLinear = !!gl.getExtension('OES_texture_float_linear');
    gl.getExtension('EXT_color_buffer_half_float');
    this.vao = gl.createVertexArray();
    this._res.push(() => gl.deleteVertexArray(this.vao));
    this._ubBinding = 0;
  }

  track(deleter) {
    this._res.push(deleter);
  }

  destroy() {
    for (const d of this._res) {
      try {
        d();
      } catch {
        /* ignore */
      }
    }
    this._res.length = 0;
  }

  /** Compile & link. Throws a readable error on failure. */
  compile(vsSrc, fsSrc, label = 'program') {
    const gl = this.gl;
    const sh = (type, src, which) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const msg = formatGLError(gl.getShaderInfoLog(s) || 'compile failed', src, `${label} (${which})`);
        gl.deleteShader(s);
        throw new Error(`[GLSL] ${msg}`);
      }
      return s;
    };
    const vs = sh(gl.VERTEX_SHADER, vsSrc, 'vertex');
    const fs = sh(gl.FRAGMENT_SHADER, fsSrc, 'fragment');
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      throw new Error(`[GLSL] ${label} link error: ${log}`);
    }
    this.track(() => gl.deleteProgram(p));
    return p;
  }

  /** UniformBlock backed by a UBO. */
  uniforms(fields, structName = 'U') {
    const gl = this.gl;
    const block = fields instanceof UniformBlock ? fields : new UniformBlock(fields, structName);
    block.gl = gl;
    block.glBuffer = gl.createBuffer();
    block.glBinding = this._ubBinding++;
    gl.bindBuffer(gl.UNIFORM_BUFFER, block.glBuffer);
    gl.bufferData(gl.UNIFORM_BUFFER, block.byteSize, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    this.track(() => gl.deleteBuffer(block.glBuffer));
    return block;
  }

  /** Attach a UniformBlock to a program (by its struct/block name) and bind its UBO. */
  bindUniforms(program, block) {
    const gl = this.gl;
    const idx = gl.getUniformBlockIndex(program, block.structName);
    if (idx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, idx, block.glBinding);
    block.upload();
    gl.bindBufferBase(gl.UNIFORM_BUFFER, block.glBinding, block.glBuffer);
  }

  _fmt(format) {
    const gl = this.gl;
    if ((format === 'rgba16f' || format === 'rgba16float' || format === 'rgba32f' || format === 'rgba32float') && !this.floatRT) {
      console.warn('EXT_color_buffer_float unavailable: falling back to RGBA8 render target');
      format = 'rgba8';
    }
    switch (format) {
      case 'rgba16f':
      case 'rgba16float':
        return { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT, float: true, filterable: true };
      case 'rgba32f':
      case 'rgba32float':
        return { internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT, float: true, filterable: this.floatLinear };
      default:
        return { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, float: false, filterable: true };
    }
  }

  /** Offscreen render target: { tex, fbo, width, height }. */
  target(width, height, { format = 'rgba8', filter = 'linear', wrap = 'clamp' } = {}) {
    const gl = this.gl;
    width = Math.max(1, Math.floor(width));
    height = Math.max(1, Math.floor(height));
    const f = this._fmt(format);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, f.internal, width, height, 0, f.format, f.type, null);
    const flt = filter === 'nearest' || !f.filterable ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, flt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, flt);
    const w = wrap === 'repeat' ? gl.REPEAT : gl.CLAMP_TO_EDGE;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, w);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, w);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    const t = {
      tex,
      fbo,
      width,
      height,
      format,
      destroy: () => {
        gl.deleteTexture(tex);
        gl.deleteFramebuffer(fbo);
      },
    };
    this.track(t.destroy);
    return t;
  }

  pingPong(width, height, opts = {}) {
    const a = this.target(width, height, opts);
    const b = this.target(width, height, opts);
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

  /** Upload an image-like source. Row 0 = top of the image (same as WebGPU). */
  textureFromImage(source, { filter = 'linear', wrap = 'clamp', mips = false } = {}) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
    const lin = filter !== 'nearest';
    if (mips) gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mips ? (lin ? gl.LINEAR_MIPMAP_LINEAR : gl.NEAREST_MIPMAP_NEAREST) : lin ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, lin ? gl.LINEAR : gl.NEAREST);
    const w = wrap === 'repeat' ? gl.REPEAT : gl.CLAMP_TO_EDGE;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, w);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, w);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.track(() => gl.deleteTexture(tex));
    return { tex, width: source.width, height: source.height };
  }

  /**
   * Translate portable WGSL (with a `fn shade(uv, px) -> vec4f`) to a GLSL fragment shader.
   * Returns { source, errors }.
   */
  translate({ code, uniforms = null, textures = [], include = [] }) {
    const libs = include.length ? resolveIncludes(include) : '';
    const structs = {};
    const globals = { PI: 'f32', TAU: 'f32' };
    if (uniforms) {
      structs[uniforms.structName] = Object.fromEntries(uniforms.fields.map((f) => [f.name, f.type.wgsl.replace(/\s+/g, '')]));
      globals.u = uniforms.structName;
    }
    for (const t of textures) globals[t] = 'texture_2d<f32>';
    const { code: body, errors } = wgslToGlsl(libs + '\n' + code, { structs, globals, funcs: PRELUDE_FUNCS });
    const source = [
      GLSL_HEADER,
      uniforms ? uniforms.glsl('u') : '',
      ...textures.map((t) => `uniform sampler2D ${t};`),
      'uniform vec2 u_targetSize;',
      'uniform float u_flipY;',
      'in vec2 v_uv;',
      'out vec4 fragColor;',
      GLSL_MATH_PRELUDE,
      GLSL_TEXTURE_PRELUDE,
      body,
      'void main() {',
      '  vec2 px = vec2(gl_FragCoord.x, u_flipY > 0.0 ? u_targetSize.y - gl_FragCoord.y : gl_FragCoord.y);',
      '  fragColor = shade(v_uv, px);',
      '}',
    ].join('\n');
    return { source, errors };
  }

  /**
   * Fullscreen pass from portable WGSL. Same contract as SceneGPU.fullscreen():
   * code defines `fn shade(uv: vec2f, px: vec2f) -> vec4f`.
   */
  fullscreen({ label = 'fullscreen', code, uniforms = null, textures = [], include = [], blend }) {
    const gl = this.gl;
    const { source, errors } = this.translate({ code, uniforms, textures, include });
    if (errors.length) errorSink(`[WGSL→GLSL] ${label}:\n  ${errors.join('\n  ')}`);
    const program = this.compile(GL_FULLSCREEN_VS, source, label);
    const loc = {
      flip: gl.getUniformLocation(program, 'u_flipY'),
      size: gl.getUniformLocation(program, 'u_targetSize'),
      tex: textures.map((t) => gl.getUniformLocation(program, t)),
    };
    if (uniforms) {
      const idx = gl.getUniformBlockIndex(program, uniforms.structName);
      if (idx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, idx, uniforms.glBinding);
    }
    const kit = this;
    return {
      program,
      source,
      /**
       * draw(target|null, resources, opts)
       *   target: a GLKit.target() or null for the canvas
       *   resources: { textureName: target | {tex} | WebGLTexture }
       *   opts: { clear: [r,g,b,a] | false, blend, canvasSize: [w,h] }
       */
      draw(target, resources = {}, opts = {}) {
        const w = target ? target.width : opts.canvasSize?.[0] ?? gl.drawingBufferWidth;
        const h = target ? target.height : opts.canvasSize?.[1] ?? gl.drawingBufferHeight;
        gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
        gl.viewport(0, 0, w, h);
        gl.useProgram(program);
        gl.bindVertexArray(kit.vao);
        gl.uniform1f(loc.flip, target ? -1 : 1);
        gl.uniform2f(loc.size, w, h);
        if (uniforms) {
          uniforms.upload();
          gl.bindBufferBase(gl.UNIFORM_BUFFER, uniforms.glBinding, uniforms.glBuffer);
        }
        textures.forEach((name, i) => {
          const r = resources[name];
          const tex = r ? r.tex || r.read?.tex || r : null;
          gl.activeTexture(gl.TEXTURE0 + i);
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.uniform1i(loc.tex[i], i);
        });
        const clear = opts.clear === undefined ? [0, 0, 0, 1] : opts.clear;
        if (clear) {
          gl.clearColor(clear[0], clear[1], clear[2], clear[3]);
          gl.clear(gl.COLOR_BUFFER_BIT);
        }
        const b = opts.blend !== undefined ? opts.blend : blend;
        if (b && GL_BLEND[b]) {
          gl.enable(gl.BLEND);
          GL_BLEND[b](gl);
        } else gl.disable(gl.BLEND);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.disable(gl.BLEND);
        for (let i = 0; i < textures.length; i++) {
          gl.activeTexture(gl.TEXTURE0 + i);
          gl.bindTexture(gl.TEXTURE_2D, null);
        }
      },
    };
  }

  /** Copy a target to the canvas (or another target), scaling with linear/nearest filtering. */
  blit(src, dst = null, { filter = 'linear', canvasSize } = {}) {
    if (!this._blitPass) {
      this._blitPass = this.fullscreen({
        label: 'blit',
        code: 'fn shade(uv: vec2f, px: vec2f) -> vec4f { return TEX(src, uv); }',
        textures: ['src'],
      });
      this._blitN = this.fullscreen({
        label: 'blitN',
        code: 'fn shade(uv: vec2f, px: vec2f) -> vec4f { return TEXN(src, uv); }',
        textures: ['src'],
      });
    }
    (filter === 'nearest' ? this._blitN : this._blitPass).draw(dst, { src }, { canvasSize });
  }
}

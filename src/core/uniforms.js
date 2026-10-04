// UniformBlock: declare uniform fields ONCE and get
//   - a correctly padded byte layout (WGSL uniform rules == GLSL std140 for these types)
//   - the WGSL `struct` text
//   - the GLSL `layout(std140) uniform` block text
//   - setters that write into a single ArrayBuffer that both backends upload as-is.
//
// Supported field types: f32 i32 u32 vec2f vec3f vec4f vec2i vec4i vec2u vec4u mat4x4f
// and arrays of 16-byte types: 'array<vec4f, N>' / 'array<vec4i, N>' / 'array<vec4u, N>'.

const SCALAR = {
  f32: { align: 4, size: 4, glsl: 'float', kind: 'f' },
  i32: { align: 4, size: 4, glsl: 'int', kind: 'i' },
  u32: { align: 4, size: 4, glsl: 'uint', kind: 'u' },
  vec2f: { align: 8, size: 8, glsl: 'vec2', kind: 'f' },
  vec3f: { align: 16, size: 12, glsl: 'vec3', kind: 'f' },
  vec4f: { align: 16, size: 16, glsl: 'vec4', kind: 'f' },
  vec2i: { align: 8, size: 8, glsl: 'ivec2', kind: 'i' },
  vec4i: { align: 16, size: 16, glsl: 'ivec4', kind: 'i' },
  vec2u: { align: 8, size: 8, glsl: 'uvec2', kind: 'u' },
  vec4u: { align: 16, size: 16, glsl: 'uvec4', kind: 'u' },
  mat4x4f: { align: 16, size: 64, glsl: 'mat4', kind: 'f' },
};

const roundUp = (a, n) => Math.ceil(n / a) * a;

// Words that are reserved in WGSL and/or GLSL and therefore can't be field names.
export const RESERVED_NAMES = new Set(
  (
    'target filter sample input output active common smooth flat buffer shared half fixed long short double layout ' +
    'uniform in out inout attribute varying precision lowp mediump highp static const var let fn struct enum union ' +
    'class namespace template this new delete module import export default switch case break continue loop return ' +
    'true false discard override mod set get type match move self pass meta from use ref do of package precise ' +
    'texture sampler alias private function workgroup storage handle ' +
    'auto demote extern final friend impl macro mutable operator protected public register super throw try typedef ' +
    'typename unless unsized virtual where with yield async await become catch crate debugger decltype explicit ' +
    'extends finally goto inline interface nil null nullptr typeof unsafe using volatile'
  ).split(' '),
);

function parseType(type) {
  const t = type.replace(/\s+/g, '');
  if (SCALAR[t]) return { ...SCALAR[t], wgsl: t, count: 0 };
  const m = /^array<(vec4f|vec4i|vec4u),(\d+)>$/.exec(t);
  if (m) {
    const el = SCALAR[m[1]];
    const n = parseInt(m[2], 10);
    return { align: 16, size: 16 * n, glsl: el.glsl, kind: el.kind, wgsl: `array<${m[1]}, ${n}>`, count: n };
  }
  throw new Error(`UniformBlock: unsupported type "${type}". Use f32/i32/u32/vec2f/vec3f/vec4f/mat4x4f or array<vec4f,N>.`);
}

/** Parse "#rrggbb" / "#rgb" into [r,g,b] floats in 0..1 (sRGB values, untouched). */
export function hexToRgb(hex) {
  if (Array.isArray(hex)) return hex;
  let h = String(hex).replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h.slice(0, 6), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export class UniformBlock {
  /**
   * @param {Record<string,string>|Array<[string,string]>} fields  name -> type, in order
   * @param {string} structName  name of the generated struct (WGSL) / block (GLSL)
   */
  constructor(fields, structName = 'U') {
    this.structName = structName;
    this.fields = [];
    this.map = Object.create(null);
    const list = Array.isArray(fields) ? fields : Object.entries(fields);
    let offset = 0;
    let maxAlign = 16;
    for (const [name, type] of list) {
      if (this.map[name]) throw new Error(`UniformBlock: duplicate field "${name}"`);
      if (RESERVED_NAMES.has(name) || name.includes('__'))
        throw new Error(`UniformBlock: "${name}" is a reserved word in WGSL or GLSL — pick another name (e.g. "${name}Value").`);
      const t = parseType(type);
      offset = roundUp(t.align, offset);
      const f = { name, type: t, offset };
      this.fields.push(f);
      this.map[name] = f;
      offset += t.size;
      maxAlign = Math.max(maxAlign, t.align);
    }
    this.byteSize = Math.max(16, roundUp(maxAlign, offset));
    this.data = new ArrayBuffer(this.byteSize);
    this.f32 = new Float32Array(this.data);
    this.i32 = new Int32Array(this.data);
    this.u32 = new Uint32Array(this.data);
    this.dirty = true;
    this.gpuBuffer = null; // set by SceneGPU.uniforms()
    this.device = null;
    this.glBuffer = null; // set by GLKit.uniforms()
    this.gl = null;
  }

  has(name) {
    return !!this.map[name];
  }

  /** Set a field. Accepts number, boolean, array, typed array, or '#hex' for vec3f/vec4f. */
  set(name, value) {
    const f = this.map[name];
    if (!f) throw new Error(`UniformBlock "${this.structName}": no field "${name}"`);
    const view = f.type.kind === 'f' ? this.f32 : f.type.kind === 'i' ? this.i32 : this.u32;
    const base = f.offset >> 2;
    if (typeof value === 'number' || typeof value === 'boolean') {
      view[base] = +value;
    } else if (typeof value === 'string') {
      const c = hexToRgb(value);
      view[base] = c[0];
      view[base + 1] = c[1];
      view[base + 2] = c[2];
      if (f.type.size === 16) view[base + 3] = 1;
    } else if (value && value.length !== undefined) {
      const n = Math.min(value.length, f.type.size >> 2);
      for (let i = 0; i < n; i++) view[base + i] = value[i];
    } else {
      throw new Error(`UniformBlock: bad value for "${name}": ${value}`);
    }
    this.dirty = true;
    return this;
  }

  /** Set many fields from an object. Unknown keys are ignored. */
  setAll(obj) {
    for (const k in obj) if (this.map[k]) this.set(k, obj[k]);
    return this;
  }

  /** WGSL struct declaration. */
  wgsl() {
    const lines = this.fields.map((f) => `  ${f.name}: ${f.type.wgsl},`);
    return `struct ${this.structName} {\n${lines.join('\n')}\n};\n`;
  }

  /** GLSL ES 3.00 std140 uniform block declaration with an instance name. */
  glsl(instance = 'u') {
    const lines = this.fields.map((f) =>
      f.type.count ? `  ${f.type.glsl} ${f.name}[${f.type.count}];` : `  ${f.type.glsl} ${f.name};`,
    );
    return `layout(std140) uniform ${this.structName} {\n${lines.join('\n')}\n} ${instance};\n`;
  }

  /** Upload to whichever backend buffer exists (only if changed). */
  upload(force = false) {
    if (!this.dirty && !force) return;
    if (this.gpuBuffer && this.device) this.device.queue.writeBuffer(this.gpuBuffer, 0, this.data);
    if (this.glBuffer && this.gl) {
      const gl = this.gl;
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.glBuffer);
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, new Uint8Array(this.data));
      gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    }
    this.dirty = false;
  }
}

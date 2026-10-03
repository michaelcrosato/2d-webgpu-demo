// A small translator from "portable WGSL" to GLSL ES 3.00.
//
// It is NOT a full WGSL compiler. It handles the subset documented in AUTHORING.md:
//   * structs, consts, functions (any order: we hoist + emit prototypes)
//   * let/var/const with or without explicit types (simple type inference)
//   * select(f, t, cond) with a scalar condition
//   * the usual math builtins, bitcast<T>(), array<T, N>(...) constructors
//   * texture access ONLY through the TEX/TEXR/TEXN/LOAD/LOADW/TEXSIZE helpers
// Unsupported (reported as errors): switch, loop/continuing, pointers, raw texture builtins.
//
// Rule of thumb for authors: write float literals with a dot (2.0, not 2) whenever the value
// is used as a float. WGSL allows `x * 2`; GLSL does not.

const SCALAR_GLSL = { f32: 'float', i32: 'int', u32: 'uint', bool: 'bool' };
const VEC_PREFIX = { f: 'vec', i: 'ivec', u: 'uvec', b: 'bvec' };

const GLSL_RESERVED = new Set(
  (
    'input output filter sample texture common partition active smooth flat noise buffer shared half fixed long ' +
    'short double unsigned superp external interface namespace using union enum typedef template this packed goto ' +
    'inline public static extern volatile cast sizeof class asm resource coherent restrict readonly writeonly precise ' +
    'invariant centroid patch subroutine lowp mediump highp precision attribute varying layout uniform in out inout ' +
    'lerp mod atan texture2D sampler sampler2D isampler2D usampler2D gl_FragCoord gl_Position'
  ).split(' '),
);

// Builtins that return the same type as their first argument.
const GEN_FIRST = new Set(
  (
    'abs sign floor ceil fract round trunc sqrt inverseSqrt exp exp2 log log2 sin cos tan asin acos atan sinh cosh tanh ' +
    'asinh acosh atanh atan2 pow min max clamp saturate mix normalize reflect refract faceForward fwidth dpdx dpdy ' +
    'dpdxFine dpdyFine dpdxCoarse dpdyCoarse fwidthFine fwidthCoarse degrees radians fmod fmod2 fmod3 select ' +
    'countOneBits reverseBits firstLeadingBit firstTrailingBit extractBits insertBits transpose'
  ).split(' '),
);
const RET_F32 = new Set('length distance dot determinant'.split(' '));
const RET_FIXED = {
  cross: 'vec3f',
  TEX: 'vec4f',
  TEXR: 'vec4f',
  TEXN: 'vec4f',
  LOAD: 'vec4f',
  LOADW: 'vec4f',
  TEXSIZE: 'vec2f',
  any: 'bool',
  all: 'bool',
  pack4x8unorm: 'u32',
  unpack4x8unorm: 'vec4f',
  pack2x16float: 'u32',
  unpack2x16float: 'vec2f',
};

const BUILTIN_RENAME = {
  atan2: 'atan',
  inverseSqrt: 'inversesqrt',
  dpdx: 'dFdx',
  dpdy: 'dFdy',
  dpdxFine: 'dFdx',
  dpdyFine: 'dFdy',
  dpdxCoarse: 'dFdx',
  dpdyCoarse: 'dFdy',
  fwidthFine: 'fwidth',
  fwidthCoarse: 'fwidth',
  countOneBits: 'bitCount',
  reverseBits: 'bitfieldReverse',
  firstLeadingBit: 'findMSB',
  firstTrailingBit: 'findLSB',
  extractBits: 'bitfieldExtract',
  insertBits: 'bitfieldInsert',
  pack4x8unorm: 'packUnorm4x8',
  unpack4x8unorm: 'unpackUnorm4x8',
  pack2x16float: 'packHalf2x16',
  unpack2x16float: 'unpackHalf2x16',
  faceForward: 'faceforward',
};

class TranspileError extends Error {}

// ---------------------------------------------------------------------------
// Type helpers. Internal type strings are normalized WGSL: f32 i32 u32 bool vecNk matNxNf
// array<T,N> texture_2d<f32> sampler or a struct name.
// ---------------------------------------------------------------------------

function normType(t) {
  if (!t) return t;
  t = t.replace(/\s+/g, '');
  t = t.replace(/\bvec([234])<f32>/g, 'vec$1f').replace(/\bvec([234])<i32>/g, 'vec$1i');
  t = t.replace(/\bvec([234])<u32>/g, 'vec$1u').replace(/\bvec([234])<bool>/g, 'vec$1b');
  t = t.replace(/\bmat([234])x([234])<f32>/g, 'mat$1x$2f');
  t = t.replace(/\bmat([234])x([234])\b(?!f)/g, 'mat$1x$2f');
  return t;
}

function vecInfo(t) {
  const m = /^vec([234])([fiub])$/.exec(t);
  if (m) return { n: +m[1], k: m[2] };
  if (t === 'f32') return { n: 1, k: 'f' };
  if (t === 'i32') return { n: 1, k: 'i' };
  if (t === 'u32') return { n: 1, k: 'u' };
  if (t === 'bool') return { n: 1, k: 'b' };
  return null;
}
function makeVec(n, k) {
  if (n === 1) return { f: 'f32', i: 'i32', u: 'u32', b: 'bool' }[k];
  return `vec${n}${k}`;
}
function matInfo(t) {
  const m = /^mat([234])x([234])f$/.exec(t);
  return m ? { c: +m[1], r: +m[2] } : null;
}
function arrayInfo(t) {
  const m = /^array<(.+),(\d+)>$/.exec(t);
  return m ? { el: m[1], n: +m[2] } : null;
}

export function glslType(t) {
  t = normType(t);
  if (SCALAR_GLSL[t]) return SCALAR_GLSL[t];
  const v = /^vec([234])([fiub])$/.exec(t);
  if (v) return VEC_PREFIX[v[2]] + v[1];
  const m = matInfo(t);
  if (m) return m.c === m.r ? `mat${m.c}` : `mat${m.c}x${m.r}`;
  const a = arrayInfo(t);
  if (a) return `${glslType(a.el)}[${a.n}]`;
  if (t === 'texture_2d<f32>') return 'sampler2D';
  if (/^[A-Za-z_]\w*$/.test(t)) return t; // struct
  throw new TranspileError(`unsupported type "${t}"`);
}

// ---------------------------------------------------------------------------
// Low level scanning helpers
// ---------------------------------------------------------------------------

function stripComments(src) {
  let out = '';
  let i = 0;
  while (i < src.length) {
    if (src[i] === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') i++;
    } else if (src[i] === '/' && src[i + 1] === '*') {
      let depth = 1;
      i += 2;
      while (i < src.length && depth > 0) {
        if (src[i] === '/' && src[i + 1] === '*') {
          depth++;
          i += 2;
        } else if (src[i] === '*' && src[i + 1] === '/') {
          depth--;
          i += 2;
        } else {
          if (src[i] === '\n') out += '\n';
          i++;
        }
      }
    } else {
      out += src[i++];
    }
  }
  return out;
}

/** Find the index of the bracket matching the opener at `start`. */
function matchBracket(s, start) {
  const open = s[start];
  const close = { '(': ')', '[': ']', '{': '}', '<': '>' }[open];
  let depth = 0;
  for (let i = start; i < s.length; i++) {
    if (s[i] === open) depth++;
    else if (s[i] === close) {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Split by `sep` at nesting depth 0 (parens, brackets, braces and generics). */
function splitTop(s, sep = ',') {
  const parts = [];
  let depth = 0;
  let angle = 0;
  let cur = '';
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '(' || c === '[' || c === '{') depth++;
    else if (c === ')' || c === ']' || c === '}') depth--;
    else if (c === '<' && /\w/.test(s[i - 1] || '') && /^(array|vec\d|mat\dx\d|texture_2d|bitcast)$/.test(lastWord(s, i))) angle++;
    else if (c === '>' && angle > 0) angle--;
    if (c === sep && depth === 0 && angle === 0) {
      parts.push(cur);
      cur = '';
    } else cur += c;
  }
  if (cur.trim() !== '' || parts.length) parts.push(cur);
  return parts;
}
function lastWord(s, i) {
  let j = i;
  while (j > 0 && /\w/.test(s[j - 1])) j--;
  return s.slice(j, i);
}

/** Parse a type starting at s[i] (identifier with optional balanced <...>). Returns [type, endIndex]. */
function readType(s, i) {
  while (/\s/.test(s[i])) i++;
  const m = /^[A-Za-z_]\w*/.exec(s.slice(i));
  if (!m) return [null, i];
  let j = i + m[0].length;
  let k = j;
  while (/\s/.test(s[k])) k++;
  if (s[k] === '<') {
    const end = matchBracket(s, k);
    if (end < 0) throw new TranspileError('unbalanced <> in type');
    j = end + 1;
  }
  return [normType(s.slice(i, j)), j];
}

// ---------------------------------------------------------------------------
// Expression type inference (best effort)
// ---------------------------------------------------------------------------

class Infer {
  constructor(ctx) {
    this.ctx = ctx; // { structs, funcs, globals }
    this.locals = Object.create(null);
  }
  lookup(name) {
    return this.locals[name] || this.ctx.globals[name] || null;
  }
  type(expr) {
    try {
      return this._type(expr.trim());
    } catch (e) {
      return null;
    }
  }
  _type(e) {
    e = (e || '').trim();
    if (!e) return null;
    // strip wrapping parens
    while (e[0] === '(' && matchBracket(e, 0) === e.length - 1) e = e.slice(1, -1).trim();

    // binary operators, lowest precedence first
    for (const ops of [['||'], ['&&'], ['|'], ['^'], ['&'], ['==', '!='], ['<=', '>=', '<', '>'], ['<<', '>>'], ['+', '-'], ['*', '/', '%']]) {
      const parts = this.splitBinary(e, ops);
      if (parts) {
        const [lhs, op, rhs] = parts;
        if (['||', '&&', '==', '!=', '<=', '>=', '<', '>'].includes(op)) {
          const lt = this._type(lhs);
          const lv = lt && vecInfo(lt);
          return lv && lv.n > 1 ? makeVec(lv.n, 'b') : 'bool';
        }
        const a = this._type(lhs);
        const b = this._type(rhs);
        if (['<<', '>>', '|', '^', '&'].includes(op)) return a;
        return combine(a, b, op);
      }
    }
    // unary
    if (e[0] === '-' || e[0] === '+' || e[0] === '~') return this._type(e.slice(1));
    if (e[0] === '!') return 'bool';
    return this.primary(e);
  }
  // Split at the LAST top-level occurrence of one of `ops` (left associative).
  splitBinary(e, ops) {
    let depth = 0;
    for (let i = e.length - 1; i > 0; i--) {
      const c = e[i];
      if (c === ')' || c === ']') depth++;
      else if (c === '(' || c === '[') depth--;
      if (depth !== 0) continue;
      for (const op of ops) {
        if (e.substr(i - op.length + 1, op.length) !== op) continue;
        const start = i - op.length + 1;
        if (start <= 0) continue;
        const before = e[start - 1];
        const after = e[i + 1];
        // avoid splitting multi-char ops wrongly
        if (op === '<' || op === '>') {
          if (before === '<' || before === '>' || after === '<' || after === '>' || after === '=' || before === '-') continue;
          // generic type brackets like array<f32,4>
          if (op === '<' && /(array|vec\d|mat\dx\d|bitcast)$/.test(e.slice(0, start))) continue;
          if (op === '>') {
            const open = e.lastIndexOf('<', start);
            if (open >= 0 && /(array|vec\d|mat\dx\d|bitcast)$/.test(e.slice(0, open))) continue;
          }
        }
        if (op === '&' && (before === '&' || after === '&')) continue;
        if (op === '|' && (before === '|' || after === '|')) continue;
        if ((op === '+' || op === '-')) {
          // unary if preceded by an operator or '(' or ','
          let k = start - 1;
          while (k >= 0 && /\s/.test(e[k])) k--;
          if (k < 0 || /[-+*/%(,<>=!&|^~]/.test(e[k])) continue;
          // exponent of a float literal: 1e-3
          if ((e[k] === 'e' || e[k] === 'E') && /(^|[^\w.])\d+\.?\d*[eE]$/.test(e.slice(0, k + 1))) continue;
        }
        if ((op === '*' || op === '/') && (before === '*' || before === '/' || after === '=')) continue;
        if (op === '=' || after === '=' && op.length === 1 && op !== '<' && op !== '>') continue;
        return [e.slice(0, start), op, e.slice(i + 1)];
      }
    }
    return null;
  }
  primary(e) {
    let m;
    // literals
    if ((m = /^(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?([fhiu]?)$/.exec(e))) {
      if (m[3] === 'u') return 'u32';
      if (m[3] === 'i') return 'i32';
      if (m[3] === 'f' || m[3] === 'h' || m[1].includes('.') || m[2]) return 'f32';
      return 'i32';
    }
    if (/^0x[0-9a-fA-F]+u$/.test(e)) return 'u32';
    if (/^0x[0-9a-fA-F]+i?$/.test(e)) return 'i32';
    if (e === 'true' || e === 'false') return 'bool';

    let t = null;
    let i = 0;
    if (e[0] === '(') {
      // parenthesized head followed by a postfix, e.g. (a + b).xy
      const end = matchBracket(e, 0);
      t = this._type(e.slice(1, end));
      i = end + 1;
    } else {
      // primary head: identifier possibly followed by <generic> and (call)
      m = /^[A-Za-z_]\w*/.exec(e);
      if (!m) return null;
      const name = m[0];
      i = name.length;
      let generic = null;
      if (e[i] === '<' && /^(array|vec\d|mat\dx\d|bitcast)$/.test(name)) {
        const end = matchBracket(e, i);
        generic = e.slice(i + 1, end);
        i = end + 1;
      }
      if (e[i] === '(') {
        const end = matchBracket(e, i);
        const args = splitTop(e.slice(i + 1, end));
        t = this.callType(name, generic, args);
        i = end + 1;
      } else {
        t = this.lookup(name);
      }
    }
    // postfix chain
    while (i < e.length && t) {
      if (e[i] === '.') {
        const mm = /^\.([A-Za-z_]\w*)/.exec(e.slice(i));
        const field = mm[1];
        i += mm[0].length;
        const v = vecInfo(t);
        if (v && v.n > 1 && /^[xyzwrgba]+$/.test(field)) t = makeVec(field.length, v.k);
        else if (this.ctx.structs[t]) t = this.ctx.structs[t][field] || null;
        else return null;
      } else if (e[i] === '[') {
        const end = matchBracket(e, i);
        i = end + 1;
        const v = vecInfo(t);
        const mt = matInfo(t);
        const a = arrayInfo(t);
        if (a) t = a.el;
        else if (mt) t = makeVec(mt.r, 'f');
        else if (v && v.n > 1) t = makeVec(1, v.k);
        else return null;
      } else if (/\s/.test(e[i])) i++;
      else return null;
    }
    return t;
  }
  callType(name, generic, args) {
    if (generic !== null) {
      const full = normType(`${name}<${generic}>`);
      if (name === 'bitcast') return normType(generic);
      return full; // array<T,N>(...) or vec3<f32>(...)
    }
    if (/^vec[234][fiu]$/.test(name) || /^mat[234]x[234]f$/.test(name)) return name;
    if (/^vec[234]$/.test(name)) return name + 'f';
    if (/^mat([234])x([234])$/.test(name)) return name + 'f';
    if (name === 'f32' || name === 'i32' || name === 'u32' || name === 'bool') return name;
    if (this.ctx.structs[name]) return name;
    if (this.ctx.funcs[name]) return this.ctx.funcs[name].ret;
    if (RET_F32.has(name)) return 'f32';
    if (RET_FIXED[name]) return RET_FIXED[name];
    if (name === 'step') return this._type(args[1]) || this._type(args[0]);
    if (name === 'smoothstep') return this._type(args[2]) || this._type(args[0]);
    if (GEN_FIRST.has(name)) {
      for (const a of args) {
        const t = this._type(a);
        if (t) {
          // max(vec, vec) etc: first known arg type
          return t;
        }
      }
    }
    return null;
  }
}

function combine(a, b, op) {
  if (!a || !b) {
    return null;
  }
  if (a === b) return a;
  const ma = matInfo(a);
  const mb = matInfo(b);
  const va = vecInfo(a);
  const vb = vecInfo(b);
  if (ma && vb && vb.n > 1) return b; // mat * vec
  if (va && va.n > 1 && mb) return a; // vec * mat
  if (ma && vb) return a; // mat * scalar
  if (mb && va) return b;
  if (va && vb) {
    if (va.n > 1 && vb.n === 1) return a;
    if (vb.n > 1 && va.n === 1) return b;
    // abstract int literal mixing with float: prefer float
    if (va.n === vb.n) {
      if (va.k === 'f' || vb.k === 'f') return makeVec(va.n, 'f');
      return a;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Main translation
// ---------------------------------------------------------------------------

/**
 * @param {string} src portable WGSL source
 * @param {object} opts
 *   opts.structs: { StructName: { field: 'wgslType' } } extra known structs (e.g. uniform block)
 *   opts.globals: { name: 'wgslType' } extra known globals (e.g. u: 'U', state: 'texture_2d<f32>')
 *   opts.funcs:   { name: 'retType' } extra known functions (e.g. prelude)
 * @returns {{code: string, errors: string[]}}
 */
export function wgslToGlsl(src, opts = {}) {
  const errors = [];
  let s = stripComments(src.replace(/\r/g, ''));
  s = s.replace(/^\s*(diagnostic\s*\([^)]*\)|enable\s+[^;]*|requires\s+[^;]*)\s*;/gm, '');
  s = s.replace(/@(must_use|diagnostic\s*\([^)]*\))/g, '');

  const ctx = {
    structs: Object.assign(Object.create(null), opts.structs || {}),
    funcs: Object.create(null),
    globals: Object.assign(Object.create(null), opts.globals || {}),
  };
  for (const [k, v] of Object.entries(opts.funcs || {})) ctx.funcs[k] = { ret: v };

  // ---- split into top-level items ----
  const items = [];
  let i = 0;
  const lineAt = (idx) => s.slice(0, idx).split('\n').length;
  while (i < s.length) {
    while (i < s.length && /[\s;]/.test(s[i])) i++;
    if (i >= s.length) break;
    const rest = s.slice(i);
    let m;
    if ((m = /^struct\s+(\w+)\s*\{/.exec(rest))) {
      const open = i + m[0].length - 1;
      const close = matchBracket(s, open);
      items.push({ kind: 'struct', name: m[1], body: s.slice(open + 1, close), line: lineAt(i) });
      i = close + 1;
    } else if ((m = /^fn\s+(\w+)\s*\(/.exec(rest))) {
      const po = i + m[0].length - 1;
      const pc = matchBracket(s, po);
      const params = s.slice(po + 1, pc);
      let j = pc + 1;
      const head = /^\s*(->\s*)?/.exec(s.slice(j));
      let ret = null;
      j += head[0].length;
      if (head[1]) {
        // skip attributes on return type
        const am = /^(@\w+(\([^)]*\))?\s*)*/.exec(s.slice(j));
        j += am[0].length;
        [ret, j] = readType(s, j);
      }
      while (/\s/.test(s[j])) j++;
      if (s[j] !== '{') throw new TranspileError(`expected { after fn ${m[1]}`);
      const close = matchBracket(s, j);
      items.push({ kind: 'fn', name: m[1], params, ret, body: s.slice(j + 1, close), line: lineAt(i) });
      i = close + 1;
    } else if ((m = /^(const|var(\s*<\s*private\s*>)?|override|alias|const_assert)\b/.exec(rest))) {
      const end = s.indexOf(';', i);
      const text = s.slice(i, end);
      if (/^(override|alias|const_assert)/.test(m[1])) {
        if (!/^const_assert/.test(m[1])) errors.push(`line ${lineAt(i)}: "${m[1]}" is not supported in portable WGSL`);
      } else if (/^var\s*<\s*(workgroup|storage|uniform)/.test(rest)) {
        errors.push(`line ${lineAt(i)}: module-scope ${text.slice(0, 20)}... is not portable`);
      } else items.push({ kind: 'global', text, line: lineAt(i) });
      i = end + 1;
    } else if (/^@/.test(rest)) {
      errors.push(`line ${lineAt(i)}: attributes / entry points are not allowed in portable code`);
      const nl = s.indexOf('\n', i);
      i = nl < 0 ? s.length : nl + 1;
    } else {
      errors.push(`line ${lineAt(i)}: unexpected top-level text "${rest.slice(0, 30).split('\n')[0]}"`);
      const nl = s.indexOf('\n', i);
      i = nl < 0 ? s.length : nl + 1;
    }
  }

  // ---- first pass: register struct fields and function signatures ----
  for (const it of items) {
    if (it.kind === 'struct') {
      const fields = Object.create(null);
      it.fields = [];
      for (const raw of splitTop(it.body, ',')) {
        const f = raw.trim().replace(/^@\w+(\([^)]*\))?\s*/g, '');
        if (!f) continue;
        const c = f.indexOf(':');
        const name = f.slice(0, c).trim();
        const type = normType(f.slice(c + 1));
        fields[name] = type;
        it.fields.push([name, type]);
      }
      ctx.structs[it.name] = fields;
    } else if (it.kind === 'fn') {
      it.paramList = splitTop(it.params, ',')
        .map((p) => p.trim())
        .filter(Boolean)
        .map((p) => {
          const c = p.indexOf(':');
          return [p.slice(0, c).trim(), normType(p.slice(c + 1))];
        });
      ctx.funcs[it.name] = { ret: it.ret ? normType(it.ret) : null };
    }
  }

  const renames = new Map();
  const safeName = (n) => {
    if (GLSL_RESERVED.has(n) || n.startsWith('gl_') || /__/.test(n)) {
      const r = n.replace(/__+/g, '_') + '_x';
      renames.set(n, r);
      return r;
    }
    return n;
  };

  const out = { structs: [], globals: [], protos: [], fns: [] };
  const inferGlobal = new Infer(ctx);

  const convertBody = (body, infer, baseLine) => {
    try {
      return convertStatements(body, infer, errors, baseLine, safeName);
    } catch (e) {
      errors.push(`line ~${baseLine}: ${e.message}`);
      return body;
    }
  };

  for (const it of items) {
    try {
      if (it.kind === 'struct') {
        const lines = it.fields.map(([n, t]) => {
          const a = arrayInfo(t);
          return a ? `  ${glslType(a.el)} ${safeName(n)}[${a.n}];` : `  ${glslType(t)} ${safeName(n)};`;
        });
        out.structs.push(`struct ${it.name} {\n${lines.join('\n')}\n};`);
      } else if (it.kind === 'global') {
        const converted = convertBody(it.text + ';', inferGlobal, it.line);
        // register global type
        const gm = /^(?:const|var(?:\s*<[^>]*>)?)\s+(\w+)/.exec(it.text);
        if (gm) ctx.globals[gm[1]] = inferGlobal.locals[gm[1]] || null;
        out.globals.push(converted.trim());
      } else if (it.kind === 'fn') {
        const infer = new Infer(ctx);
        for (const [n, t] of it.paramList) infer.locals[n] = t;
        const params = it.paramList
          .map(([n, t]) => {
            if (/^ptr</.test(t)) throw new TranspileError(`pointer parameter "${n}" is not portable`);
            return `${glslType(t)} ${safeName(n)}`;
          })
          .join(', ');
        const ret = it.ret ? glslType(it.ret) : 'void';
        const body = convertBody(it.body, infer, it.line);
        out.protos.push(`${ret} ${it.name}(${params});`);
        out.fns.push(`${ret} ${it.name}(${params}) {${body}}`);
      }
    } catch (e) {
      errors.push(`line ${it.line}: ${e.message}`);
    }
  }

  let code = [...out.structs, ...out.globals, ...out.protos, ...out.fns].join('\n');
  code = finalRewrite(code, ctx, errors);
  for (const [from, to] of renames) code = code.replace(new RegExp(`\\b${from}\\b(?!\\s*\\()`, 'g'), to);
  return { code, errors };
}

// Convert declarations inside a block of statements.
function convertStatements(body, infer, errors, baseLine, safeName) {
  const lineOf = (idx) => baseLine + body.slice(0, idx).split('\n').length - 1;
  // structural checks
  const bad = [
    [/\bswitch\b/, 'switch is not portable (use if/else)'],
    [/\bloop\s*\{/, 'loop {} is not portable (use for/while)'],
    [/\bcontinuing\b/, 'continuing is not portable'],
    [/\bbreak\s+if\b/, '"break if" is not portable'],
    [/\bptr\s*</, 'pointers are not portable'],
    [/&\s*[A-Za-z_]\w*\s*[,)]/, null],
    [/\btexture(Sample|SampleLevel|Load|Dimensions|Store|Gather)\w*\s*\(/, 'use TEX()/TEXN()/LOAD()/TEXSIZE() instead of raw texture builtins'],
    [/\b(if|while)\s+[^\s(]/, 'if/while conditions must be wrapped in parentheses'],
    [/\barray\s*\(/, 'array(...) needs an explicit type: array<f32, 4>(...)'],
  ];
  for (const [re, msg] of bad) {
    if (!msg) continue;
    const m = re.exec(body);
    if (m) errors.push(`line ${lineOf(m.index)}: ${msg}`);
  }

  let res = '';
  let i = 0;
  const declRe = /\b(let|var|const)\b/g;
  let m;
  while ((m = declRe.exec(body))) {
    const start = m.index;
    // make sure this isn't part of an identifier like "my_let"
    if (start > 0 && /[\w.]/.test(body[start - 1])) continue;
    res += body.slice(i, start);
    let j = start + m[0].length;
    let isConst = m[1] === 'const';
    // optional <private> etc
    while (/\s/.test(body[j])) j++;
    if (body[j] === '<') {
      const e = matchBracket(body, j);
      const space = body.slice(j + 1, e).trim();
      if (space !== 'private' && space !== 'function') throw new TranspileError(`var<${space}> is not portable`);
      j = e + 1;
    }
    const nm = /^\s*([A-Za-z_]\w*)\s*/.exec(body.slice(j));
    if (!nm) throw new TranspileError(`bad declaration near "${body.slice(start, start + 30)}"`);
    const name = nm[1];
    j += nm[0].length;
    let type = null;
    if (body[j] === ':') {
      [type, j] = readType(body, j + 1);
    }
    while (/\s/.test(body[j])) j++;
    let rhs = null;
    let end;
    if (body[j] === '=') {
      // RHS up to ; at depth 0
      let depth = 0;
      let k = j + 1;
      for (; k < body.length; k++) {
        const c = body[k];
        if (c === '(' || c === '[' || c === '{') depth++;
        else if (c === ')' || c === ']' || c === '}') depth--;
        else if (c === ';' && depth === 0) break;
      }
      rhs = body.slice(j + 1, k);
      end = k; // position of ';'
    } else if (body[j] === ';') {
      end = j;
    } else {
      throw new TranspileError(`expected = or ; after declaration of "${name}"`);
    }
    if (!type) {
      if (rhs === null) throw new TranspileError(`"${name}" needs a type`);
      type = infer.type(rhs);
      if (!type) {
        errors.push(
          `line ${lineOf(start)}: cannot infer the type of "${name}" — add an explicit type, e.g. "let ${name}: f32 = ..."`,
        );
        type = 'f32';
      }
    }
    type = normType(type);
    if (/^(abstract|vec\db)/.test(type) && type !== 'bool') {
      // boolean vectors are allowed as GLSL bvec
    }
    infer.locals[name] = type;
    const g = glslType(type);
    const decl = `${isConst ? 'const ' : ''}${g} ${safeName(name)}`;
    res += rhs !== null ? `${decl} =${rhs}` : decl;
    i = end;
    declRe.lastIndex = end;
  }
  res += body.slice(i);
  return res;
}

// Rewrites that apply to the whole translated program.
function finalRewrite(code, ctx, errors) {
  // select(f, t, c) -> ((c) ? (t) : (f))  (innermost-first by repeated scanning)
  let guard = 0;
  for (;;) {
    const idx = findCall(code, 'select');
    if (idx < 0 || guard++ > 500) break;
    const open = code.indexOf('(', idx);
    const close = matchBracket(code, open);
    const args = splitTop(code.slice(open + 1, close));
    if (args.length !== 3) {
      errors.push('select() needs 3 arguments');
      break;
    }
    code = code.slice(0, idx) + `((${args[2].trim()}) ? (${args[1].trim()}) : (${args[0].trim()}))` + code.slice(close + 1);
  }

  // bitcast<T>(x)
  code = code.replace(/\bbitcast\s*<\s*([^>]+?)\s*>\s*\(/g, (_, t) => {
    const k = vecInfo(normType(t))?.k;
    if (k === 'u') return 'floatBitsToUint(';
    if (k === 'i') return 'floatBitsToInt(';
    return 'uintBitsToFloat(';
  });

  // array<T, N>(...) constructors -> T[N](...)
  code = code.replace(/\barray\s*<\s*([^,<>]+(?:<[^<>]*>)?)\s*,\s*(\d+)\s*>\s*\(/g, (_, t, n) => `${glslType(t)}[${n}](`);

  // generic vector/matrix syntax in expressions
  code = code.replace(/\bvec([234])\s*<\s*f32\s*>/g, 'vec$1').replace(/\bvec([234])\s*<\s*i32\s*>/g, 'ivec$1');
  code = code.replace(/\bvec([234])\s*<\s*u32\s*>/g, 'uvec$1').replace(/\bvec([234])\s*<\s*bool\s*>/g, 'bvec$1');
  code = code.replace(/\bmat([234])x\1\s*<\s*f32\s*>/g, 'mat$1');

  // type names used as constructors/casts
  code = code
    .replace(/\bvec([234])f\b/g, 'vec$1')
    .replace(/\bvec([234])i\b/g, 'ivec$1')
    .replace(/\bvec([234])u\b/g, 'uvec$1')
    .replace(/\bmat([234])x\1f?\b/g, 'mat$1')
    .replace(/\bf32\b/g, 'float')
    .replace(/\bi32\b/g, 'int')
    .replace(/\bu32\b/g, 'uint');

  // builtin renames
  for (const [from, to] of Object.entries(BUILTIN_RENAME)) {
    code = code.replace(new RegExp(`\\b${from}\\s*\\(`, 'g'), `${to}(`);
  }

  // numeric literal suffixes: 1.0f -> 1.0, 2f -> 2.0, 3i -> 3
  code = code.replace(/\b(\d+\.\d*|\.\d+)([eE][-+]?\d+)?[fh]\b/g, '$1$2');
  code = code.replace(/\b(\d+)([eE][-+]?\d+)[fh]\b/g, '$1.0$2');
  code = code.replace(/(^|[^\w.])(\d+)[fh]\b/g, '$1$2.0');
  code = code.replace(/(^|[^\w.])(\d+)i\b/g, '$1$2');

  // trailing commas in calls/constructors
  code = code.replace(/,(\s*)\)/g, '$1)');

  // `_ = expr;` phony assignment
  code = code.replace(/(^|[;{}\s])_\s*=[^;]*;/g, '$1');
  return code;
}

function findCall(code, name) {
  const re = new RegExp(`\\b${name}\\s*\\(`, 'g');
  let m;
  let last = -1;
  // innermost: choose the last occurrence whose args contain no further select(
  while ((m = re.exec(code))) last = m.index;
  return last;
}

export { TranspileError };

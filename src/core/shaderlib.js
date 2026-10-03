// Shared shader library, written ONCE in "portable WGSL" (see AUTHORING.md):
// valid WGSL that our tiny translator (wgsl2glsl.js) can turn into GLSL ES 3.00.
//
// Include by name: include: ['noise', 'sdf', 'color', ...]. Dependencies resolve automatically.
//
// Coordinate conventions used everywhere in this project:
//   uv  : 0..1, origin TOP-LEFT, y points DOWN (same as texture coordinates)
//   px  : pixel coordinates, origin TOP-LEFT, y DOWN (fragment position)

// ---------------------------------------------------------------------------
// Backend preludes (hand written per language, NOT translated)
// ---------------------------------------------------------------------------

// Requires module-scope `samp` (linear/clamp) and `sampR` (linear/repeat) samplers.
export const WGSL_TEXTURE_PRELUDE = /* wgsl */ `
fn TEX(t: texture_2d<f32>, uv: vec2f) -> vec4f { return textureSampleLevel(t, samp, uv, 0.0); }
fn TEXR(t: texture_2d<f32>, uv: vec2f) -> vec4f { return textureSampleLevel(t, sampR, uv, 0.0); }
fn LOAD(t: texture_2d<f32>, p: vec2i) -> vec4f {
  let s = vec2i(textureDimensions(t));
  return textureLoad(t, clamp(p, vec2i(0), s - vec2i(1)), 0);
}
fn LOADW(t: texture_2d<f32>, p: vec2i) -> vec4f {
  let s = vec2i(textureDimensions(t));
  return textureLoad(t, ((p % s) + s) % s, 0);
}
fn TEXN(t: texture_2d<f32>, uv: vec2f) -> vec4f {
  let s = vec2i(textureDimensions(t));
  return textureLoad(t, clamp(vec2i(floor(uv * vec2f(s))), vec2i(0), s - vec2i(1)), 0);
}
fn TEXSIZE(t: texture_2d<f32>) -> vec2f { return vec2f(textureDimensions(t)); }
`;

export const GLSL_TEXTURE_PRELUDE = /* glsl */ `
vec4 TEX(sampler2D t, vec2 uv) { return textureLod(t, uv, 0.0); }
vec4 TEXR(sampler2D t, vec2 uv) { return textureLod(t, fract(uv), 0.0); }
vec4 LOAD(sampler2D t, ivec2 p) { ivec2 s = textureSize(t, 0); return texelFetch(t, clamp(p, ivec2(0), s - ivec2(1)), 0); }
vec4 LOADW(sampler2D t, ivec2 p) { ivec2 s = textureSize(t, 0); return texelFetch(t, ((p % s) + s) % s, 0); }
vec4 TEXN(sampler2D t, vec2 uv) { ivec2 s = textureSize(t, 0); return texelFetch(t, clamp(ivec2(floor(uv * vec2(s))), ivec2(0), s - ivec2(1)), 0); }
vec2 TEXSIZE(sampler2D t) { return vec2(textureSize(t, 0)); }
`;

// Math helpers that exist natively in one language but not the other.
export const WGSL_MATH_PRELUDE = /* wgsl */ `
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
fn fmod(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }
fn fmod2(x: vec2f, y: vec2f) -> vec2f { return x - y * floor(x / y); }
fn fmod3(x: vec3f, y: vec3f) -> vec3f { return x - y * floor(x / y); }
`;

export const GLSL_MATH_PRELUDE = /* glsl */ `
const float PI = 3.14159265359;
const float TAU = 6.28318530718;
float fmod(float x, float y) { return x - y * floor(x / y); }
vec2 fmod2(vec2 x, vec2 y) { return x - y * floor(x / y); }
vec3 fmod3(vec3 x, vec3 y) { return x - y * floor(x / y); }
float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, 0.0, 1.0); }
vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }
vec4 saturate(vec4 x) { return clamp(x, 0.0, 1.0); }
`;

// ---------------------------------------------------------------------------
// Portable libraries
// ---------------------------------------------------------------------------

const math = /* wgsl */ `
// Rotation matrix: rot2(a) * p rotates p by angle a (radians).
fn rot2(a: f32) -> mat2x2f { let c: f32 = cos(a); let s: f32 = sin(a); return mat2x2f(c, s, -s, c); }
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 { return c + (x - a) * (d - c) / (b - a); }
fn remap01(x: f32, a: f32, b: f32) -> f32 { return clamp((x - a) / (b - a), 0.0, 1.0); }
// Centered, aspect-correct coordinates: (0,0) at screen center, y in [-0.5, 0.5], y DOWN.
fn centerUV(px: vec2f, res: vec2f) -> vec2f { return (px - 0.5 * res) / res.y; }
fn easeInOut(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn easeOutBack(t: f32) -> f32 { let c1: f32 = 1.70158; let c3: f32 = c1 + 1.0; let k: f32 = t - 1.0; return 1.0 + c3 * k * k * k + c1 * k * k; }
fn easeOutElastic(t: f32) -> f32 {
  if (t <= 0.0) { return 0.0; }
  if (t >= 1.0) { return 1.0; }
  return pow(2.0, -10.0 * t) * sin((t * 10.0 - 0.75) * (TAU / 3.0)) + 1.0;
}
`;

const hash = /* wgsl */ `
// PCG-style 3D integer hash (Jarzynski & Olano 2020). Great quality, no sin() artifacts.
fn pcg3d(v0: vec3u) -> vec3u {
  var v: vec3u = v0 * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn pcg(v: u32) -> u32 {
  let state: u32 = v * 747796405u + 2891336453u;
  let word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
// hashNM: N inputs -> M outputs, all in [0,1). Works for ANY float input (uses the raw bits).
fn hash33(p: vec3f) -> vec3f { return vec3f(pcg3d(bitcast<vec3u>(p))) * (1.0 / 4294967296.0); }
fn hash31(p: vec3f) -> f32 { return hash33(p).x; }
fn hash23(p: vec2f) -> vec3f { return hash33(vec3f(p, 7.0)); }
fn hash22(p: vec2f) -> vec2f { return hash33(vec3f(p, 7.0)).xy; }
fn hash21(p: vec2f) -> f32 { return hash33(vec3f(p, 7.0)).x; }
fn hash11(p: f32) -> f32 { return hash33(vec3f(p, 3.0, 7.0)).x; }
fn hash12(p: f32) -> vec2f { return hash33(vec3f(p, 3.0, 7.0)).xy; }
fn hash13(p: f32) -> vec3f { return hash33(vec3f(p, 3.0, 7.0)); }
// Cheap per-pixel noise, great for dithering away banding.
fn ign(px: vec2f) -> f32 { return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715)))); }
`;

const noise = /* wgsl */ `
const FBM_ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);
fn grad2(i: vec2f) -> vec2f { let a: f32 = hash21(i) * TAU; return vec2f(cos(a), sin(a)); }
// Value noise in [0,1]: random values at grid corners, smoothly interpolated.
fn valueNoise(p: vec2f) -> f32 {
  let i: vec2f = floor(p);
  let f: vec2f = fract(p);
  let u: vec2f = f * f * (3.0 - 2.0 * f);
  let a: f32 = hash21(i);
  let b: f32 = hash21(i + vec2f(1.0, 0.0));
  let c: f32 = hash21(i + vec2f(0.0, 1.0));
  let d: f32 = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// Perlin (gradient) noise, roughly [-1,1]: random gradient directions at grid corners.
fn perlin(p: vec2f) -> f32 {
  let i: vec2f = floor(p);
  let f: vec2f = fract(p);
  let u: vec2f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let va: f32 = dot(grad2(i), f);
  let vb: f32 = dot(grad2(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0));
  let vc: f32 = dot(grad2(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0));
  let vd: f32 = dot(grad2(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0));
  return mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y) * 1.4142;
}
// Simplex noise, roughly [-1,1]: triangular grid, fewer artifacts, cheaper in higher dimensions.
fn simplex(p: vec2f) -> f32 {
  let K1: f32 = 0.366025404;
  let K2: f32 = 0.211324865;
  let i: vec2f = floor(p + (p.x + p.y) * K1);
  let a: vec2f = p - i + (i.x + i.y) * K2;
  let m: f32 = step(a.y, a.x);
  let o: vec2f = vec2f(m, 1.0 - m);
  let b: vec2f = a - o + K2;
  let c: vec2f = a - 1.0 + 2.0 * K2;
  let h: vec3f = max(vec3f(0.5) - vec3f(dot(a, a), dot(b, b), dot(c, c)), vec3f(0.0));
  let n: vec3f = h * h * h * h * vec3f(dot(a, grad2(i)), dot(b, grad2(i + o)), dot(c, grad2(i + vec2f(1.0))));
  return dot(n, vec3f(70.0));
}
fn grad3(i: vec3f) -> vec3f { return hash33(i) * 2.0 - 1.0; }
// 3D Perlin noise, roughly [-1,1]. Use z = time for smoothly evolving 2D noise.
fn perlin3(p: vec3f) -> f32 {
  let i: vec3f = floor(p);
  let f: vec3f = fract(p);
  let u: vec3f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n000: f32 = dot(grad3(i), f);
  let n100: f32 = dot(grad3(i + vec3f(1.0, 0.0, 0.0)), f - vec3f(1.0, 0.0, 0.0));
  let n010: f32 = dot(grad3(i + vec3f(0.0, 1.0, 0.0)), f - vec3f(0.0, 1.0, 0.0));
  let n110: f32 = dot(grad3(i + vec3f(1.0, 1.0, 0.0)), f - vec3f(1.0, 1.0, 0.0));
  let n001: f32 = dot(grad3(i + vec3f(0.0, 0.0, 1.0)), f - vec3f(0.0, 0.0, 1.0));
  let n101: f32 = dot(grad3(i + vec3f(1.0, 0.0, 1.0)), f - vec3f(1.0, 0.0, 1.0));
  let n011: f32 = dot(grad3(i + vec3f(0.0, 1.0, 1.0)), f - vec3f(0.0, 1.0, 1.0));
  let n111: f32 = dot(grad3(i + vec3f(1.0, 1.0, 1.0)), f - vec3f(1.0, 1.0, 1.0));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z) * 1.1547;
}
// Fractal Brownian Motion: sum octaves of noise, each smaller & fainter. Result roughly [-1,1].
fn fbmEx(p: vec2f, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
  var sum: f32 = 0.0;
  var amp: f32 = 0.5;
  var q: vec2f = p;
  for (var i: i32 = 0; i < 12; i++) {
    if (i >= octaves) { break; }
    sum += amp * perlin(q);
    q = FBM_ROT * q * lacunarity + vec2f(17.13, 3.71);
    amp *= gain;
  }
  return sum;
}
fn fbm(p: vec2f, octaves: i32) -> f32 { return fbmEx(p, octaves, 2.0, 0.5); }
fn fbm3(p: vec3f, octaves: i32) -> f32 {
  var sum: f32 = 0.0;
  var amp: f32 = 0.5;
  var q: vec3f = p;
  for (var i: i32 = 0; i < 12; i++) {
    if (i >= octaves) { break; }
    sum += amp * perlin3(q);
    q = q * 2.0 + vec3f(17.13, 3.71, 9.17);
    amp *= 0.5;
  }
  return sum;
}
// Ridged multifractal: sharp creases, great for mountains & lightning-like veins. Result ~[0,1].
fn ridged(p: vec2f, octaves: i32) -> f32 {
  var sum: f32 = 0.0;
  var amp: f32 = 0.5;
  var q: vec2f = p;
  var prev: f32 = 1.0;
  for (var i: i32 = 0; i < 12; i++) {
    if (i >= octaves) { break; }
    var n: f32 = 1.0 - abs(perlin(q));
    n = n * n;
    sum += n * amp * prev;
    prev = n;
    q = FBM_ROT * q * 2.0 + vec2f(17.13, 3.71);
    amp *= 0.5;
  }
  return sum;
}
// Worley / cellular noise. Returns (F1, F2, cellX, cellY): distances to nearest & 2nd nearest
// feature point, and the integer id of the nearest cell (hash it for a per-cell random value).
fn voronoiEx(p: vec2f, jitter: f32, t: f32) -> vec4f {
  let n: vec2f = floor(p);
  let f: vec2f = fract(p);
  var d1: f32 = 8.0;
  var d2: f32 = 8.0;
  var id: vec2f = vec2f(0.0);
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let g: vec2f = vec2f(f32(i), f32(j));
      let h: vec2f = hash22(n + g);
      let o: vec2f = 0.5 + jitter * 0.5 * sin(t + TAU * h);
      let r: vec2f = g + o - f;
      let d: f32 = dot(r, r);
      if (d < d1) { d2 = d1; d1 = d; id = n + g; } else if (d < d2) { d2 = d; }
    }
  }
  return vec4f(sqrt(d1), sqrt(d2), id);
}
fn voronoi(p: vec2f) -> vec4f { return voronoiEx(p, 1.0, 0.0); }
// Exact distance to the voronoi cell BORDER (IQ). Returns (borderDist, cellX, cellY).
fn voronoiBorder(p: vec2f, jitter: f32, t: f32) -> vec3f {
  let n: vec2f = floor(p);
  let f: vec2f = fract(p);
  var mg: vec2f = vec2f(0.0);
  var mr: vec2f = vec2f(0.0);
  var md: f32 = 8.0;
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let g: vec2f = vec2f(f32(i), f32(j));
      let o: vec2f = 0.5 + jitter * 0.5 * sin(t + TAU * hash22(n + g));
      let r: vec2f = g + o - f;
      let d: f32 = dot(r, r);
      if (d < md) { md = d; mr = r; mg = g; }
    }
  }
  md = 8.0;
  for (var j: i32 = -2; j <= 2; j++) {
    for (var i: i32 = -2; i <= 2; i++) {
      let g: vec2f = mg + vec2f(f32(i), f32(j));
      let o: vec2f = 0.5 + jitter * 0.5 * sin(t + TAU * hash22(n + g));
      let r: vec2f = g + o - f;
      if (dot(mr - r, mr - r) > 0.00001) {
        md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
      }
    }
  }
  return vec3f(md, n + mg);
}
// Curl of a noise field: a divergence-free flow (swirly, never "sinks"). Great for smoke & particles.
fn curl(p: vec2f) -> vec2f {
  let e: f32 = 0.01;
  let a: f32 = perlin(p + vec2f(0.0, e));
  let b: f32 = perlin(p - vec2f(0.0, e));
  let c: f32 = perlin(p + vec2f(e, 0.0));
  let d: f32 = perlin(p - vec2f(e, 0.0));
  return vec2f(a - b, -(c - d)) / (2.0 * e);
}
`;

const sdf = /* wgsl */ `
// 2D signed distance functions (most by Inigo Quilez, iquilezles.org). Negative = inside.
fn sdCircle(p: vec2f, r: f32) -> f32 { return length(p) - r; }
fn sdBox(p: vec2f, b: vec2f) -> f32 {
  let d: vec2f = abs(p) - b;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}
fn sdRoundBox(p: vec2f, b: vec2f, r: f32) -> f32 { return sdBox(p, b - vec2f(r)) - r; }
fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa: vec2f = p - a;
  let ba: vec2f = b - a;
  let h: f32 = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}
fn sdTriangle(p: vec2f, p0: vec2f, p1: vec2f, p2: vec2f) -> f32 {
  let e0: vec2f = p1 - p0; let e1: vec2f = p2 - p1; let e2: vec2f = p0 - p2;
  let v0: vec2f = p - p0; let v1: vec2f = p - p1; let v2: vec2f = p - p2;
  let pq0: vec2f = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  let pq1: vec2f = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  let pq2: vec2f = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  let s: f32 = sign(e0.x * e2.y - e0.y * e2.x);
  let d: vec2f = min(min(vec2f(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                         vec2f(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                         vec2f(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}
fn sdEquilateralTriangle(p0: vec2f, r: f32) -> f32 {
  let k: f32 = sqrt(3.0);
  var p: vec2f = p0;
  p.x = abs(p.x) - r;
  p.y = p.y + r / k;
  if (p.x + k * p.y > 0.0) { p = vec2f(p.x - k * p.y, -k * p.x - p.y) / 2.0; }
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
fn sdHexagon(p0: vec2f, r: f32) -> f32 {
  let k: vec3f = vec3f(-0.866025404, 0.5, 0.577350269);
  var p: vec2f = abs(p0);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2f(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}
// 5-pointed star. r = outer radius, rf = inner ratio (0.3..0.6 looks good).
fn sdStar5(p0: vec2f, r: f32, rf: f32) -> f32 {
  let k1: vec2f = vec2f(0.809016994375, -0.587785252292);
  let k2: vec2f = vec2f(-k1.x, k1.y);
  var p: vec2f = vec2f(abs(p0.x), -p0.y);
  p -= 2.0 * max(dot(k1, p), 0.0) * k1;
  p -= 2.0 * max(dot(k2, p), 0.0) * k2;
  p.x = abs(p.x);
  p.y -= r;
  let ba: vec2f = rf * vec2f(-k1.y, k1.x) - vec2f(0.0, 1.0);
  let h: f32 = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
  return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}
// Heart, roughly unit size, centered near (0,-0.5) in y-down coords.
fn sdHeart(p0: vec2f) -> f32 {
  var p: vec2f = vec2f(abs(p0.x), -p0.y + 0.5);
  if (p.y + p.x > 1.0) { return sqrt(dot(p - vec2f(0.25, 0.75), p - vec2f(0.25, 0.75))) - sqrt(2.0) / 4.0; }
  let a: vec2f = p - vec2f(0.0, 1.0);
  let b: vec2f = p - 0.5 * max(p.x + p.y, 0.0);
  return sqrt(min(dot(a, a), dot(b, b))) * sign(p.x - p.y);
}
fn sdRhombus(p0: vec2f, b: vec2f) -> f32 {
  let p: vec2f = abs(p0);
  let ndot: f32 = b.x * (b.x - 2.0 * p.x) - b.y * (b.y - 2.0 * p.y);
  let h: f32 = clamp(ndot / dot(b, b), -1.0, 1.0);
  let d: f32 = length(p - 0.5 * b * vec2f(1.0 - h, 1.0 + h));
  return d * sign(p.x * b.y + p.y * b.x - b.x * b.y);
}
// Arc / ring segment: sc = (sin, cos) of half-aperture, ra = radius, rb = thickness.
fn sdArc(p0: vec2f, sc: vec2f, ra: f32, rb: f32) -> f32 {
  let p: vec2f = vec2f(abs(p0.x), -p0.y);
  if (sc.y * p.x > sc.x * p.y) { return length(p - sc * ra) - rb; }
  return abs(length(p) - ra) - rb;
}
fn sdPie(p0: vec2f, c: vec2f, r: f32) -> f32 {
  let p: vec2f = vec2f(abs(p0.x), -p0.y);
  let l: f32 = length(p) - r;
  let m: f32 = length(p - c * clamp(dot(p, c), 0.0, r));
  return max(l, m * sign(c.y * p.x - c.x * p.y));
}
fn sdVesica(p0: vec2f, w: f32, h: f32) -> f32 {
  let d: f32 = 0.5 * (w * w - h * h) / h;
  let p: vec2f = abs(p0);
  var c: vec3f = vec3f(-d, 0.0, d + h);
  if (w * p.y < d * (p.x - w)) { c = vec3f(0.0, w, 0.0); }
  return length(p - c.yx) - c.z;
}
fn sdMoon(p0: vec2f, d: f32, ra: f32, rb: f32) -> f32 {
  let p: vec2f = vec2f(p0.x, abs(p0.y));
  let a: f32 = (ra * ra - rb * rb + d * d) / (2.0 * d);
  let b: f32 = sqrt(max(ra * ra - a * a, 0.0));
  if (d * (p.x * b - p.y * a) > d * d * max(b - p.y, 0.0)) { return length(p - vec2f(a, b)); }
  return max(length(p) - ra, -(length(p - vec2f(d, 0.0)) - rb));
}
fn sdCross(p0: vec2f, b: vec2f, r: f32) -> f32 {
  var p: vec2f = abs(p0);
  if (p.y > p.x) { p = p.yx; }
  let q: vec2f = p - b;
  let k: f32 = max(q.y, q.x);
  var w: vec2f = vec2f(b.y - p.x, -k);
  if (k > 0.0) { w = q; }
  return sign(k) * length(max(w, vec2f(0.0))) + r;
}
fn sdEllipseApprox(p: vec2f, r: vec2f) -> f32 {
  let k0: f32 = length(p / r);
  let k1: f32 = length(p / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 0.00001);
}
// Quadratic bezier (exact distance, IQ).
fn sdBezier(pos: vec2f, A: vec2f, B: vec2f, C: vec2f) -> f32 {
  let a: vec2f = B - A;
  let b: vec2f = A - 2.0 * B + C;
  let c: vec2f = a * 2.0;
  let d: vec2f = A - pos;
  let kk: f32 = 1.0 / max(dot(b, b), 0.000001);
  let kx: f32 = kk * dot(a, b);
  let ky: f32 = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
  let kz: f32 = kk * dot(d, a);
  var res: f32 = 0.0;
  let p: f32 = ky - kx * kx;
  let p3: f32 = p * p * p;
  let q: f32 = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
  var h: f32 = q * q + 4.0 * p3;
  if (h >= 0.0) {
    h = sqrt(h);
    let x: vec2f = (vec2f(h, -h) - q) / 2.0;
    let uv: vec2f = sign(x) * pow(abs(x), vec2f(1.0 / 3.0));
    let t: f32 = clamp(uv.x + uv.y - kx, 0.0, 1.0);
    let w: vec2f = d + (c + b * t) * t;
    res = dot(w, w);
  } else {
    let z: f32 = sqrt(-p);
    let v: f32 = acos(q / (p * z * 2.0)) / 3.0;
    let m: f32 = cos(v);
    let n: f32 = sin(v) * 1.732050808;
    let t: vec3f = clamp(vec3f(m + m, -n - m, n - m) * z - kx, vec3f(0.0), vec3f(1.0));
    let w1: vec2f = d + (c + b * t.x) * t.x;
    let w2: vec2f = d + (c + b * t.y) * t.y;
    res = min(dot(w1, w1), dot(w2, w2));
  }
  return sqrt(res);
}
// Boolean operations. k = blend radius for the smooth versions.
fn opSmoothUnion(a: f32, b: f32, k: f32) -> f32 {
  let h: f32 = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
fn opSmoothSubtract(a: f32, cut: f32, k: f32) -> f32 {
  let h: f32 = clamp(0.5 - 0.5 * (a + cut) / k, 0.0, 1.0);
  return mix(a, -cut, h) + k * h * (1.0 - h);
}
fn opSmoothIntersect(a: f32, b: f32, k: f32) -> f32 {
  let h: f32 = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) + k * h * (1.0 - h);
}
fn opOnion(d: f32, r: f32) -> f32 { return abs(d) - r; }
// Anti-aliased coverage from a distance (in the SAME units as the screen derivative).
fn sdfFill(d: f32) -> f32 { let w: f32 = max(fwidth(d), 0.00001); return clamp(0.5 - d / w, 0.0, 1.0); }
fn sdfStroke(d: f32, width: f32) -> f32 { return sdfFill(abs(d) - width * 0.5); }
// Soft glow that falls off with distance outside the shape.
fn sdfGlow(d: f32, radius: f32) -> f32 { return exp(-max(d, 0.0) / max(radius, 0.00001)); }
`;

const color = /* wgsl */ `
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn hsv2rgb(c: vec3f) -> vec3f {
  let k: vec3f = vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0);
  let p: vec3f = abs(fract(c.xxx + k) * 6.0 - vec3f(3.0));
  return c.z * mix(vec3f(1.0), clamp(p - vec3f(1.0), vec3f(0.0), vec3f(1.0)), c.y);
}
fn rgb2hsv(c: vec3f) -> vec3f {
  let K: vec4f = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p: vec4f = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
  let q: vec4f = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
  let d: f32 = q.x - min(q.w, q.y);
  let e: f32 = 1.0e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
fn srgbToLinear(c: vec3f) -> vec3f {
  return mix(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, step(c, vec3f(0.04045)));
}
fn linearToSrgb(c: vec3f) -> vec3f {
  let x: vec3f = max(c, vec3f(0.0));
  return mix(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, step(x, vec3f(0.0031308)));
}
// OKLab: a perceptual color space. Mixing colors here avoids muddy/dark midpoints.
fn linearToOklab(c: vec3f) -> vec3f {
  let l: f32 = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  let m: f32 = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  let s: f32 = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  let l_: f32 = pow(max(l, 0.0), 1.0 / 3.0);
  let m_: f32 = pow(max(m, 0.0), 1.0 / 3.0);
  let s_: f32 = pow(max(s, 0.0), 1.0 / 3.0);
  return vec3f(0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
               1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
               0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}
fn oklabToLinear(c: vec3f) -> vec3f {
  let l_: f32 = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  let m_: f32 = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  let s_: f32 = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  let l: f32 = l_ * l_ * l_;
  let m: f32 = m_ * m_ * m_;
  let s: f32 = s_ * s_ * s_;
  return vec3f(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
               -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
               -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}
// Mix two sRGB colors perceptually (through OKLab).
fn mixOklab(a: vec3f, b: vec3f, t: f32) -> vec3f {
  let la: vec3f = linearToOklab(srgbToLinear(a));
  let lb: vec3f = linearToOklab(srgbToLinear(b));
  return linearToSrgb(oklabToLinear(mix(la, lb, t)));
}
// IQ cosine palette: palette(t, a, b, c, d) = a + b*cos(2pi(c*t+d)).
fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f { return a + b * cos(TAU * (c * t + d)); }
fn rainbow(t: f32) -> vec3f { return palette(t, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67)); }
fn tonemapACES(x: vec3f) -> vec3f {
  let a: f32 = 2.51; let b: f32 = 0.03; let c: f32 = 2.43; let d: f32 = 0.59; let e: f32 = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}
fn tonemapReinhard(x: vec3f) -> vec3f { return x / (vec3f(1.0) + x); }
fn adjustSaturation(c: vec3f, s: f32) -> vec3f { return mix(vec3f(luma(c)), c, s); }
fn adjustContrast(c: vec3f, k: f32) -> vec3f { return (c - vec3f(0.5)) * k + vec3f(0.5); }
// Rotate hue by angle a (radians), preserving luminance (Rodrigues rotation around grey axis).
fn hueRotate(c: vec3f, a: f32) -> vec3f {
  let k: vec3f = vec3f(0.57735);
  let ca: f32 = cos(a);
  return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}
fn blendScreen(a: vec3f, b: vec3f) -> vec3f { return vec3f(1.0) - (vec3f(1.0) - a) * (vec3f(1.0) - b); }
fn blendOverlay(a: vec3f, b: vec3f) -> vec3f {
  let lo: vec3f = 2.0 * a * b;
  let hi: vec3f = vec3f(1.0) - 2.0 * (vec3f(1.0) - a) * (vec3f(1.0) - b);
  return mix(lo, hi, step(vec3f(0.5), a));
}
fn blendSoftLight(a: vec3f, b: vec3f) -> vec3f {
  return (vec3f(1.0) - 2.0 * b) * a * a + 2.0 * b * a;
}
`;

const dither = /* wgsl */ `
// Ordered (Bayer) dithering thresholds in [0,1). Pass pixel coordinates.
fn bayer2(a: vec2f) -> f32 { let f: vec2f = floor(a); return fract(dot(f, vec2f(0.5, f.y * 0.75))); }
fn bayer4(a: vec2f) -> f32 { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
fn bayer8(a: vec2f) -> f32 { return bayer4(0.5 * a) * 0.25 + bayer2(a); }
fn bayer16(a: vec2f) -> f32 { return bayer8(0.5 * a) * 0.25 + bayer2(a); }
`;

export const LIBS = {
  math: { deps: [], code: math },
  hash: { deps: [], code: hash },
  noise: { deps: ['hash', 'math'], code: noise },
  sdf: { deps: [], code: sdf },
  color: { deps: [], code: color },
  dither: { deps: [], code: dither },
};

/** Resolve library names (with dependencies) into concatenated portable-WGSL source. */
export function resolveIncludes(names = []) {
  const out = [];
  const seen = new Set();
  const visit = (n) => {
    if (seen.has(n)) return;
    const lib = LIBS[n];
    if (!lib) throw new Error(`Unknown shader include "${n}". Available: ${Object.keys(LIBS).join(', ')}`);
    seen.add(n);
    lib.deps.forEach(visit);
    out.push(`// ---- lib:${n} ----\n${lib.code}`);
  };
  names.forEach(visit);
  return out.join('\n');
}

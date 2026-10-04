import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import { overlays, GAME_POS_WGSL, gamePass, withGameIncludes } from './_shared.js';

// Screen transitions between two live scenes: the platformer (`game`) and a procedural overworld map
// (`world`). Every transition is a function  (uv, progress) -> mix of A and B.

const ov = overlays();
const cur = { ex: 'classic' };

const CLASSIC = [
  ['dissolve', 'Cross-dissolve'],
  ['black', 'Fade through black'],
  ['white', 'Fade through white (flash)'],
  ['wipe', 'Directional wipe (soft edge)'],
  ['clock', 'Radial clock wipe'],
  ['iris', 'Iris (closes on the hero)'],
  ['push', 'Slide / push'],
];
const PIXEL = [
  ['diamond', 'Diamond tile wipe'],
  ['tiles', 'Random tiles'],
  ['pixelate', 'Pixelate out → in'],
  ['burn', 'Noise dissolve (burning edge)'],
];
const FANCY = [
  ['shatter', 'Voronoi shatter'],
  ['ripple', 'Ripple'],
  ['swirl', 'Swirl & zoom'],
];
const ALL = [...CLASSIC, ...PIXEL, ...FANCY].map((x) => x[0]);

const NOTES = {
  dissolve: 'mix(A, B, t) — the simplest transition.',
  black: 'A → black → B. The safe classic; hides loading.',
  white: 'A blows out to white, then B fades in. Flashbacks, teleports, explosions.',
  wipe: 'B is revealed behind a moving line with a soft edge.',
  clock: 'B is revealed by an angle sweeping round, like a clock hand.',
  iris: 'A circle closes on the hero, then opens on the map. Mario & cartoons.',
  push: 'B pushes A off the screen — great for menus and rooms.',
  diamond: 'Tiles grow diamonds that sweep across the screen (Zelda / FF style).',
  tiles: 'Each tile has a random delay, then pops to the new scene.',
  pixelate: 'Blocks grow huge, the scene swaps, blocks shrink again.',
  burn: 'A noise texture decides when each pixel switches; a glowing band marks the edge.',
  shatter: 'Voronoi shards shrink towards their centres, starting at the hero.',
  ripple: 'A water ripple carries the new scene outward from the centre.',
  swirl: 'Twist and zoom into the centre, swap, untwist.',
};

// ------------------------------------------------------------------------------------------ WGSL
const WORLD_PASS = /* wgsl */ `
fn node(i: i32) -> vec2f {
  let a: f32 = u.resolution.x / u.resolution.y;
  var xs = array<f32, 5>(0.14, 0.31, 0.5, 0.68, 0.86);
  var ys = array<f32, 5>(0.66, 0.47, 0.6, 0.4, 0.55);
  return vec2f(xs[i] * a, ys[i]);
}

fn worldMap(uv: vec2f) -> vec3f {
  let a: f32 = u.resolution.x / u.resolution.y;
  let p: vec2f = vec2f(uv.x * a, uv.y);
  let t: f32 = u.time;
  // distance to the path through the 5 level nodes (and where along it we are, for the dashes)
  var dp: f32 = 1000.0;
  var along: f32 = 0.0;
  var acc: f32 = 0.0;
  for (var i = 0; i < 4; i++) {
    let n0: vec2f = node(i);
    let n1: vec2f = node(i + 1);
    let ba: vec2f = n1 - n0;
    let h: f32 = clamp(dot(p - n0, ba) / dot(ba, ba), 0.0, 1.0);
    let d: f32 = length(p - n0 - ba * h);
    if (d < dp) { dp = d; along = acc + h * length(ba); }
    acc += length(ba);
  }
  // terrain height: fractal noise, pushed up along the path so the levels are on land
  var h: f32 = fbm(p * 2.4 + vec2f(4.1, 1.7), 5) * 0.9 + 0.42 * exp(-dp * dp / 0.008) - 0.16;
  h -= 0.25 * smoothstep(0.35, 0.55, abs(uv.y - 0.52));
  // ocean
  var c: vec3f = mix(vec3f(0.07, 0.25, 0.5), vec3f(0.2, 0.55, 0.72), smoothstep(-0.25, 0.0, h));
  let wv: f32 = sin(p.y * 70.0 + sin(p.x * 9.0 + t * 0.7) * 2.5 + t * 1.5);
  c += vec3f(0.6, 0.8, 0.9) * smoothstep(0.9, 1.0, wv) * smoothstep(0.1, 0.7, valueNoise(p * 6.0 + vec2f(t * 0.1, 0.0))) * 0.35 * smoothstep(0.0, -0.08, h);
  c = mix(c, vec3f(0.85, 0.95, 1.0), (1.0 - smoothstep(0.0, 0.012, abs(h + 0.03))) * 0.35 * (0.6 + 0.4 * sin(t * 2.0 + p.x * 10.0)));
  if (h > 0.0) {
    var land: vec3f = mix(vec3f(0.93, 0.84, 0.58), vec3f(0.45, 0.72, 0.32), smoothstep(0.02, 0.06, h));
    land = mix(land, vec3f(0.3, 0.55, 0.25), smoothstep(0.12, 0.25, h));
    // forests: little dots
    let fc: vec2f = floor(p * 55.0);
    let fh: f32 = hash21(fc);
    let fd: f32 = length(fract(p * 55.0) - 0.5);
    land = mix(land, vec3f(0.12, 0.35, 0.18), step(fd, 0.3) * step(0.55, fh) * smoothstep(0.1, 0.14, h) * (1.0 - smoothstep(0.28, 0.3, h)));
    // mountains & snow
    land = mix(land, vec3f(0.55, 0.47, 0.4), smoothstep(0.3, 0.34, h));
    land = mix(land, vec3f(0.95, 0.96, 1.0), smoothstep(0.44, 0.47, h));
    land *= 0.9 + 0.1 * perlin(p * 30.0);
    c = mix(c, land, smoothstep(0.0, 0.004, h));
    c = mix(c, vec3f(0.25, 0.2, 0.12), (1.0 - smoothstep(0.0, 0.006, h)) * 0.6);
  }
  // dotted path
  let dash: f32 = step(0.5, fract(along * 38.0));
  c = mix(c, vec3f(0.98, 0.94, 0.8), (1.0 - smoothstep(0.004, 0.007, dp)) * dash);
  // level nodes: cleared (gold), current (pulsing red), locked (white)
  for (var i = 0; i < 5; i++) {
    let d: f32 = length(p - node(i));
    var nc: vec3f = vec3f(0.95, 0.95, 0.98);
    if (i < 2) { nc = vec3f(1.0, 0.78, 0.2); }
    if (i == 2) {
      nc = vec3f(0.95, 0.25, 0.25);
      let pr: f32 = d - (0.03 + 0.012 * fract(t * 1.2) * 3.0);
      c = mix(c, vec3f(1.0, 0.5, 0.4), (1.0 - smoothstep(0.0, 0.004, abs(pr))) * (1.0 - fract(t * 1.2)));
    }
    c = mix(c, vec3f(0.15, 0.1, 0.08), 1.0 - smoothstep(0.024, 0.027, d));
    c = mix(c, nc, 1.0 - smoothstep(0.019, 0.022, d));
  }
  // banner text
  let tp: vec2f = vec2f((p.x - a * 0.5) / 0.8 + 0.5, (uv.y - 0.05) / 0.2);
  if (tp.x > 0.0 && tp.x < 1.0 && tp.y > 0.0 && tp.y < 1.0) {
    let tx: vec4f = TEX(mapText, tp);
    c = mix(c, tx.rgb, tx.a);
  }
  // vignette
  c *= 1.0 - 0.35 * smoothstep(0.45, 0.95, length((uv - 0.5) * vec2f(a, 1.0)) / (0.5 * a));
  return c;
}
fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(worldMap(uv), 1.0); }`;

const IMAGE = /* wgsl */ `
${GAME_POS_WGSL}

// A = the scene we leave, B = the scene we go to (they swap every other cycle)
fn sceneA(uv: vec2f) -> vec3f {
  if (u.fromB > 0.5) { return TEX(world, uv).rgb; }
  return TEX(game, uv).rgb;
}
fn sceneB(uv: vec2f) -> vec3f {
  if (u.fromB > 0.5) { return TEX(game, uv).rgb; }
  return TEX(world, uv).rgb;
}
// focus points for the iris: the hero in the game, the current level node on the map
fn focusA() -> vec2f {
  let a: f32 = u.resolution.x / u.resolution.y;
  if (u.fromB > 0.5) { return vec2f(0.5 * a, 0.6); }
  return heroP() + vec2f(0.0, -0.03);
}
fn focusB() -> vec2f {
  let a: f32 = u.resolution.x / u.resolution.y;
  if (u.fromB > 0.5) { return heroP() + vec2f(0.0, -0.03); }
  return vec2f(0.5 * a, 0.6);
}

fn ease(x: f32) -> f32 { let t: f32 = clamp(x, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }

// voronoi cell lookup returning the seed point (xy), a per-cell random (z) and the distance to the border (w)
fn vcell(q: vec2f) -> vec4f {
  let n: vec2f = floor(q);
  var best: f32 = 100.0;
  var second: f32 = 100.0;
  var seed: vec2f = vec2f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let g: vec2f = n + vec2f(f32(i), f32(j));
      let s: vec2f = g + 0.1 + 0.8 * hash22(g);
      let d: f32 = length(q - s);
      if (d < best) { second = best; best = d; seed = s; } else if (d < second) { second = d; }
    }
  }
  return vec4f(seed, hash21(seed * 1.31), (second - best) * 0.5);
}

fn transition(uv: vec2f, px: vec2f, t: f32) -> vec3f {
  let k: f32 = u.kind;
  let a: f32 = u.resolution.x / u.resolution.y;
  let p: vec2f = vec2f(uv.x * a, uv.y);
  let s: f32 = max(u.softness, 0.001);

  if (k < 0.5) { return mix(sceneA(uv), sceneB(uv), ease(t)); }                      // cross-dissolve
  if (k < 1.5) {                                                                     // through black
    if (t < 0.5) { return sceneA(uv) * (1.0 - ease(t * 2.0)); }
    return sceneB(uv) * ease(t * 2.0 - 1.0);
  }
  if (k < 2.5) {                                                                     // through white
    if (t < 0.5) { let e: f32 = ease(t * 2.0); return mix(sceneA(uv) * (1.0 + e * 1.5), vec3f(1.0), e); }
    return mix(vec3f(1.0), sceneB(uv), ease(t * 2.0 - 1.0));
  }
  if (k < 3.5) {                                                                     // directional wipe
    let dir: vec2f = vec2f(cos(u.angle), sin(u.angle));
    // project onto the direction, normalised so 0..1 spans the whole screen
    let c: vec2f = (uv - 0.5) * vec2f(a, 1.0);
    let ext: f32 = 0.5 * (abs(dir.x) * a + abs(dir.y));
    let d: f32 = dot(c, dir) / (2.0 * ext) + 0.5;
    let edge: f32 = t * (1.0 + s) - s;
    return mix(sceneA(uv), sceneB(uv), 1.0 - smoothstep(edge, edge + s, d));
  }
  if (k < 4.5) {                                                                     // clock wipe
    let c: vec2f = p - vec2f(0.5 * a, 0.5);
    let ang: f32 = fract(atan2(c.x, -c.y) / TAU + 1.0);                              // 0 at 12 o'clock, clockwise
    let ss: f32 = s * 0.25;
    let m: f32 = 1.0 - smoothstep(t * (1.0 + ss) - ss, t * (1.0 + ss), ang);
    return mix(sceneA(uv), sceneB(uv), m);
  }
  if (k < 5.5) {                                                                     // iris
    let rmax: f32 = length(vec2f(a, 1.0)) * 1.05;
    let aa: f32 = 1.5 / u.resolution.y + s * 0.02;
    if (t < 0.5) {
      let r: f32 = rmax * (1.0 - ease(t * 2.0)) * (1.0 - ease(t * 2.0)) ;
      let m: f32 = smoothstep(r, r + aa, length(p - focusA()));
      return mix(sceneA(uv), vec3f(0.0), m);
    }
    let r2: f32 = rmax * ease(t * 2.0 - 1.0) * ease(t * 2.0 - 1.0);
    let m2: f32 = smoothstep(r2, r2 + aa, length(p - focusB()));
    return mix(sceneB(uv), vec3f(0.0), m2);
  }
  if (k < 6.5) {                                                                     // slide / push
    let e: f32 = ease(t);
    let x: f32 = uv.x + e;
    var c: vec3f = sceneB(vec2f(x - 1.0, uv.y));
    if (x < 1.0) { c = sceneA(vec2f(x, uv.y)); }
    let seam: f32 = abs(x - 1.0) * a;
    c *= 1.0 - 0.5 * exp(-seam * 40.0) * step(0.001, e) * step(e, 0.999);
    return c;
  }
  let tile: f32 = u.tileSize / u.resolution.y;
  if (k < 7.5) {                                                                     // diamond wipe
    let cell: vec2f = floor(p / tile);
    let f: vec2f = fract(p / tile);
    let sweep: f32 = (cell.x * tile) / a * 0.75 + cell.y * tile * 0.25;
    let local: f32 = clamp(t * 2.0 - sweep, 0.0, 1.0);
    let d: f32 = abs(f.x - 0.5) + abs(f.y - 0.5);
    return mix(sceneA(uv), sceneB(uv), 1.0 - step(local * 1.001, d));
  }
  if (k < 8.5) {                                                                     // random tiles
    let cell: vec2f = floor(p / tile);
    let f: vec2f = fract(p / tile);
    let h: f32 = hash21(cell);
    let local: f32 = clamp((t - h * 0.75) / 0.25, 0.0, 1.0);
    let d: f32 = max(abs(f.x - 0.5), abs(f.y - 0.5)) * 2.0;
    var c: vec3f = mix(sceneA(uv), sceneB(uv), 1.0 - step(local, d));
    c *= 1.0 - 0.25 * (1.0 - smoothstep(0.0, 0.08, 1.0 - d)) * step(0.01, local) * step(local, 0.99);
    return c;
  }
  if (k < 9.5) {                                                                     // pixelate out -> in
    let tri: f32 = 1.0 - abs(t * 2.0 - 1.0);
    let bs: f32 = max(1.0, floor(pow(tri, 2.0) * u.tileSize * 1.5));
    let q: vec2f = (floor(px / bs) + 0.5) * bs / u.resolution;
    return mix(sceneA(q), sceneB(q), smoothstep(0.45, 0.55, t));
  }
  if (k < 10.5) {                                                                    // burn / noise dissolve
    let n: f32 = clamp(0.5 + 0.75 * fbm(p * 3.2 + vec2f(1.7, 9.2), 5), 0.0, 1.0);
    let th: f32 = t * (1.0 + s * 0.4) - s * 0.2;
    let w: f32 = 0.02 + s * 0.08;
    var c: vec3f = sceneA(uv);
    let dd: f32 = n - th;
    c *= mix(1.0, 0.25, smoothstep(w * 3.0, 0.0, dd));                               // char before the edge
    c = mix(c, sceneB(uv), step(dd, 0.0));
    let glow: f32 = exp(-abs(dd) / w * 2.0);
    c += u.glowColor * glow * 1.6 + vec3f(1.0, 0.9, 0.6) * pow(glow, 6.0) * 0.8;
    return c;
  }
  if (k < 11.5) {                                                                    // voronoi shatter
    let dens: f32 = 7.0 / max(u.tileSize / 40.0, 0.25);
    let q: vec2f = p * dens;
    let v: vec4f = vcell(q);
    let impact: vec2f = focusA() * dens;
    let delay: f32 = length(v.xy - impact) / (dens * length(vec2f(a, 1.0))) * 0.7 + v.z * 0.15;
    let lt: f32 = clamp((t * 1.35 - delay) / 0.4, 0.0, 1.0);
    // cracks appear first
    var c: vec3f = sceneB(uv);
    if (lt <= 0.0) {
      c = sceneA(uv);
      let crack: f32 = (1.0 - smoothstep(0.0, 0.03, v.w)) * smoothstep(0.0, 0.12, t * 1.35 - delay + 0.12);
      return mix(c, vec3f(1.0), crack * 0.8);
    }
    // shards shrink towards their seed point; read the shard's ORIGINAL position
    let sc: f32 = pow(1.0 - lt, 1.5);
    if (sc > 0.001) {
      let q0: vec2f = v.xy + (q - v.xy) / sc;
      let v0: vec4f = vcell(q0);
      if (length(v0.xy - v.xy) < 0.0001) {
        let src: vec2f = vec2f(q0.x / dens / a, q0.y / dens);
        let edgeL: f32 = 1.0 - smoothstep(0.0, 0.05, v0.w);
        c = sceneA(src) * (0.75 + 0.5 * v.z) + vec3f(edgeL * 0.6);
      }
    }
    return c;
  }
  if (k < 12.5) {                                                                    // ripple
    let ctr: vec2f = vec2f(0.5 * a, 0.5);
    let d: vec2f = p - ctr;
    let r: f32 = length(d);
    let amp: f32 = sin(t * PI) * 0.03;
    let off: vec2f = d / max(r, 0.0001) * sin(r * 45.0 - t * 30.0) * amp;
    let q: vec2f = vec2f((p.x + off.x) / a, p.y + off.y);
    let front: f32 = t * 1.6 * length(vec2f(a, 1.0)) * 0.5 - 0.1;
    let m: f32 = 1.0 - smoothstep(front - 0.1 - s * 0.2, front + 0.1, r);
    return mix(sceneA(q), sceneB(q), m);
  }
  // swirl & zoom
  let ctr: vec2f = vec2f(0.5 * a, 0.5);
  let tri: f32 = 1.0 - abs(t * 2.0 - 1.0);
  let d: vec2f = p - ctr;
  let r: f32 = length(d);
  let ang: f32 = ease(tri) * 9.0 * exp(-r * 2.5);
  let z: f32 = 1.0 - 0.5 * ease(tri);
  let dd: vec2f = rot2(ang) * d * z;
  let q: vec2f = vec2f((ctr.x + dd.x) / a, ctr.y + dd.y);
  return mix(sceneA(q), sceneB(q), smoothstep(0.4, 0.6, t));
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  var c: vec3f = transition(uv, px, clamp(u.progress, 0.0, 1.0));
  // progress bar
  let barY: f32 = u.resolution.y - 3.0;
  if (px.y > barY) { c = mix(c * 0.4, vec3f(0.55, 0.65, 1.0), step(px.x / u.resolution.x, u.progress)); }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

function mapTextCanvas() {
  const W = 1024;
  const H = 256;
  const cv = makeCanvas(W, H);
  const g = cv.getContext('2d');
  g.clearRect(0, 0, W, H);
  // ribbon banner
  g.fillStyle = '#7a2e22';
  g.beginPath();
  g.moveTo(170, 60);
  g.lineTo(854, 60);
  g.lineTo(820, 128);
  g.lineTo(854, 196);
  g.lineTo(170, 196);
  g.lineTo(204, 128);
  g.closePath();
  g.fill();
  g.fillStyle = '#c4473a';
  g.fillRect(230, 52, 564, 136);
  g.strokeStyle = '#f6d9a8';
  g.lineWidth = 4;
  g.strokeRect(242, 64, 540, 112);
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = '#fff3d9';
  g.font = '900 64px system-ui, "Segoe UI", Roboto, sans-serif';
  g.fillText('WORLD 2', W / 2, 108);
  g.font = '600 28px system-ui, sans-serif';
  g.fillStyle = '#ffe2b0';
  g.fillText('The Misty Isles', W / 2, 152);
  return cv;
}

const pick = (list, v) => Math.max(0, ALL.indexOf(v));

export default shaderScene({
  interaction: 'Watch it loop, or turn off auto-play and scrub the progress slider.',
  examples: [
    {
      id: 'classic',
      label: 'Classic',
      kind: 'In a game',
      note: 'The bread-and-butter transitions: dissolves, fades through a color, wipes, the clock wipe, the cartoon iris and the push. Each is a tiny formula that decides, per pixel, “old scene or new scene?”.',
      params: { tClassic: 'iris', auto: true, duration: 1.6, softness: 0.3, angle: 0 },
    },
    {
      id: 'pixel',
      label: 'Pixel & dissolve',
      kind: 'In a game',
      note: 'Transitions built from tiles and noise: classic RPG diamond wipes, random tile pops, the pixelate-out/in used by countless retro games, and a burning noise dissolve with a glowing edge.',
      params: { tPixel: 'burn', auto: true, duration: 2, softness: 0.4, tileSize: 48 },
    },
    {
      id: 'fancy',
      label: 'Fancy',
      kind: 'Abstract',
      note: 'Show-off transitions that warp the image: glass-like Voronoi shards, a ripple that carries the new scene outward, and a swirl that twists into the centre.',
      params: { tFancy: 'shatter', auto: true, duration: 2.2, softness: 0.3, tileSize: 48 },
    },
  ],
  controls: [
    { type: 'select', key: 'tClassic', label: 'Transition', value: 'iris', options: CLASSIC.map(([value, label]) => ({ value, label })), showFor: ['classic'] },
    { type: 'select', key: 'tPixel', label: 'Transition', value: 'burn', options: PIXEL.map(([value, label]) => ({ value, label })), showFor: ['pixel'] },
    { type: 'select', key: 'tFancy', label: 'Transition', value: 'shatter', options: FANCY.map(([value, label]) => ({ value, label })), showFor: ['fancy'] },
    { type: 'toggle', key: 'auto', label: 'Auto-play', value: true, help: 'Loops game → map → game. Turn off to scrub by hand.' },
    { type: 'slider', key: 'progress', label: 'Progress (manual)', min: 0, max: 1, step: 0.001, value: 0.5, help: 'Used when auto-play is off.' },
    { type: 'slider', key: 'duration', label: 'Duration', min: 0.3, max: 5, step: 0.05, value: 1.6, unit: 's' },
    { type: 'slider', key: 'softness', label: 'Edge softness', min: 0, max: 1, step: 0.01, value: 0.3, help: 'Wipes, clock, iris, burn and ripple edges.' },
    { type: 'slider', key: 'angle', label: 'Wipe direction', min: 0, max: 6.283, step: 0.01, value: 0, format: (v) => `${Math.round((v * 180) / Math.PI)}°`, showFor: ['classic'], help: 'Directional wipe only.' },
    { type: 'slider', key: 'tileSize', label: 'Tile / block size', min: 12, max: 120, step: 1, value: 48, unit: 'px', showFor: ['pixel', 'fancy'], help: 'Tiles, pixelate blocks, shard size.' },
    { type: 'color', key: 'glowColor', label: 'Burn edge color', value: '#ff7a1a', showFor: ['pixel'] },
  ],
  uniforms: { kind: 'f32', progress: 'f32', fromB: 'f32', softness: 'f32', angle: 'f32', tileSize: 'f32', glowColor: 'vec3f' },
  include: withGameIncludes(['math', 'hash', 'noise']),
  textures: { mapText: { source: async () => mapTextCanvas() } },
  passes: [gamePass(), { name: 'world', format: 'rgba8unorm', code: WORLD_PASS }],
  resetOnExample: false,
  bind(p, ctx) {
    cur.ex = ctx.example;
    ov.begin(ctx);
    const name = ctx.example === 'classic' ? p.tClassic : ctx.example === 'pixel' ? p.tPixel : p.tFancy;
    let progress = p.progress;
    let fromB = 0;
    if (p.auto) {
      const hold = 0.9;
      const T = p.duration + hold;
      const k = Math.floor(ctx.time / T);
      const tau = ctx.time - k * T;
      progress = Math.min(1, Math.max(0, (tau - hold) / p.duration));
      fromB = k % 2;
    }
    if (ctx.testMode) progress = name === 'iris' ? 0.35 : 0.5; // the headless test renders ~2 fps: show mid-transition
    const dpr = ctx.dpr || 1;
    ov.show('name', `${(([...CLASSIC, ...PIXEL, ...FANCY].find((x) => x[0] === name) || [, name])[1])} — <span style="opacity:.75">${NOTES[name] || ''}</span>`, `left:8px;top:46px;max-width:min(42%, 340px);white-space:normal`);
    ov.show('pct', `${fromB ? 'map → game' : 'game → map'} · ${Math.round(progress * 100)}%`, `right:8px;bottom:${12 / dpr + 8}px`);
    ov.end();
    return { kind: pick(ALL, name), progress, fromB };
  },
  code: IMAGE,
  about: {
    summary:
      'A transition is a per-pixel decision between the old scene and the new one, driven by a single progress value from 0 to 1. Change the formula and you get fades, wipes, irises, pixelation, burning paper or shattering glass.',
    what: `<p>Two live scenes — the platformer and an overworld map — rendered into two textures. A final pass reads both and blends them
      according to the chosen transition and <i>progress</i>. Auto-play loops game → map → game; turn it off to scrub.</p>`,
    how: `<ol>
      <li>Render scene A and scene B into textures (both keep animating during the transition).</li>
      <li>For each pixel compute a <b>mask</b> m from its position and the progress t, then output <code>mix(A, B, m)</code>.</li>
      <li><b>Wipes</b>: m = “is my distance along a direction (or my angle, for the clock) less than t?”, with a <code>smoothstep</code> for a soft edge.</li>
      <li><b>Iris</b>: a circle around the hero shrinks to black, then a circle around the target grows. <b>Tiles</b>: the same test per tile, with a delay per tile (position for a sweep, random for pops).</li>
      <li><b>Noise dissolve</b>: m = noise(uv) &lt; t; pixels just about to switch get an emissive edge color.</li>
      <li><b>Warps</b> (push, pixelate, ripple, swirl, shatter) also change <i>where</i> A and B are read from, not just how they’re mixed.</li>
    </ol>`,
    uses: [
      { title: 'Level & room changes', text: 'Iris and fades in platformers, push/slide between rooms (Zelda), tile wipes into RPG battles (Final Fantasy, Pokémon).' },
      { title: 'Menus & UI', text: 'Push and wipe between menu pages, pixelate in/out for retro titles.' },
      { title: 'Story moments', text: 'Flash to white for flashbacks, burn away for dramatic reveals, ripple for dreams and memories.' },
      { title: 'Hiding loading', text: 'Any “through black” transition gives you a frame or two to swap assets.' },
    ],
    try: [
      'Turn off <b>Auto-play</b> and scrub <i>Progress</i> slowly for the iris — notice it focuses on the hero, then reopens on the map’s red node.',
      'On <b>Pixel & dissolve</b> set Edge softness to 1 on the burn: the glowing edge widens into a fire front.',
      'Try the Directional wipe with a 45° direction and softness 0 vs 1.',
      'On <b>Fancy</b> shrink Tile size to 16 for tiny glass shards.',
    ],
    ask: [
      'an iris transition that closes on the player',
      'a diamond tile wipe into battle like old RPGs',
      'a noise dissolve transition with a burning glowing edge',
      'pixelate out and back in between levels',
      'a shatter transition where the screen breaks into shards',
    ],
    perf: `<p>A transition is one full-screen pass reading each scene once or a few times — trivial. The real cost is rendering <b>both</b>
      scenes during the transition; many games freeze scene A into a texture (render once) to halve that.</p>`,
    api: `<p>Identical in WebGPU and WebGL2: two render targets and one blending pass. Nothing here needs compute.</p>`,
    code: [
      {
        title: 'Soft directional wipe',
        lang: 'wgsl',
        src: `let d = dot(centered, dir) / (2.0 * ext) + 0.5;   // 0..1 along the wipe direction
let edge = t * (1.0 + s) - s;                       // moves from -s to 1
return mix(sceneA(uv), sceneB(uv), 1.0 - smoothstep(edge, edge + s, d));`,
      },
      {
        title: 'Noise dissolve with a glowing edge',
        lang: 'wgsl',
        src: `let n = 0.5 + 0.75 * fbm(p * 3.2, 5);   // when does this pixel switch?
let dd = n - t;
var c = sceneA(uv) * mix(1.0, 0.25, smoothstep(w * 3.0, 0.0, dd));  // char just before
c = mix(c, sceneB(uv), step(dd, 0.0));                              // switched
c += glowColor * exp(-abs(dd) / w * 2.0) * 1.6;                     // the burning edge`,
      },
      {
        title: 'Iris closing on the hero',
        lang: 'wgsl',
        src: `let r = rmax * (1.0 - ease(t * 2.0)) * (1.0 - ease(t * 2.0));
let m = smoothstep(r, r + aa, length(p - heroPosition));
return mix(sceneA(uv), vec3f(0.0), m);`,
      },
    ],
  },
});

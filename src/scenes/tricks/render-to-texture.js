// Render targets: draw the same world several times per frame, from different cameras, into textures —
// then use those textures as minimaps, portals (with recursion), a reflecting pool, split-screen
// views and a CRT security-camera feed.

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas, rng } from '../../core/assets.js';
import { overlayTag, keyAxis, KEYS, anyKey, clamp } from './_shared.js';

const TS = 32; // world units per tile
const MW = 64;
const MH = 40;
// tile types
const GRASS = 0, SAND = 2, WATER = 3, STONE = 4, WALL = 5, WOOD = 6, TREE = 7, POOLT = 8;
const BLOCK = new Set([WATER, WALL, TREE, POOLT]);
const TILE_NAMES = ['tile_grass', 'tile_sand', 'tile_water_0', 'tile_water_1', 'tile_water_2', 'tile_water_3', 'tile_stone', 'tile_brick', 'tile_wood', 'tile_leaves'];

const PORTAL_A = [20.5 * TS, 19.5 * TS];
const PORTAL_B = [44.5 * TS, 35 * TS];
const PORTAL_ROT = Math.PI / 2; // what you see through A is the area around B, turned 90°
const POOL = [24 * TS, 13 * TS, 30 * TS, 16 * TS]; // reflecting pool (x0, y0, x1, y1)

function buildMap() {
  const t = new Uint8Array(MW * MH);
  const v = new Uint8Array(MW * MH);
  const R = rng(77);
  const set = (x, y, k) => {
    if (x >= 0 && y >= 0 && x < MW && y < MH) t[y * MW + x] = k;
  };
  for (let i = 0; i < MW * MH; i++) v[i] = R() < 0.12 ? 1 + Math.floor(R() * 3) : 0;
  // lake with sand shore and a bridge
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      const d = Math.hypot((x - 12) / 8.5, (y - 27) / 6);
      if (d < 1) set(x, y, WATER);
      else if (d < 1.25) set(x, y, SAND);
    }
  for (let x = 2; x <= 22; x++) set(x, 27, WOOD);
  // stone roads and a plaza
  for (let x = 4; x < 60; x++) {
    set(x, 9, STONE);
    set(x, 10, STONE);
  }
  for (let y = 9; y < 38; y++) {
    set(32, y, STONE);
    set(33, y, STONE);
  }
  for (let y = 15; y < 26; y++) for (let x = 17; x < 29; x++) if (Math.hypot(x - 22.5, y - 20) < 5) set(x, y, STONE);
  for (let x = 33; x < 46; x++) set(x, 35, STONE);
  // reflecting pool (north of the plaza road)
  for (let y = POOL[1] / TS; y < POOL[3] / TS; y++) for (let x = POOL[0] / TS; x < POOL[2] / TS; x++) set(x, y, POOLT);
  for (let x = POOL[0] / TS - 1; x <= POOL[2] / TS; x++) {
    set(x, POOL[1] / TS - 1, STONE);
    set(x, POOL[3] / TS, STONE);
  }
  // a house with a wooden floor (the security camera watches it)
  for (let y = 18; y <= 29; y++)
    for (let x = 44; x <= 57; x++) {
      const edge = y === 18 || y === 29 || x === 44 || x === 57;
      set(x, y, edge ? WALL : WOOD);
    }
  set(50, 29, WOOD);
  set(51, 29, WOOD);
  for (let y = 30; y < 35; y++) {
    set(50, y, STONE);
    set(51, y, STONE);
  }
  // trees: border + scattered
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      const i = y * MW + x;
      const border = x < 2 || y < 2 || x >= MW - 2 || y >= MH - 2;
      if (border && R() < 0.85) t[i] = TREE;
      else if (t[i] === GRASS && R() < 0.07) {
        // keep portal surroundings clear
        const dA = Math.hypot(x * TS - PORTAL_A[0], y * TS - PORTAL_A[1]);
        const dB = Math.hypot(x * TS - PORTAL_B[0], y * TS - PORTAL_B[1]);
        if (dA > 90 && dB > 90) t[i] = TREE;
      }
    }
  return { t, v };
}

const VIEW_FIELDS = {
  center: 'vec2f', size: 'vec2f', zoom: 'f32', rot: 'f32', time: 'f32', style: 'f32',
  portalA: 'vec4f', portalB: 'vec4f', pool: 'vec4f', srcA: 'f32', srcB: 'f32', mirrorOn: 'f32', portalRot: 'f32',
  tiles: 'array<vec4f, 10>',
};

const WORLD = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}
fn tileAt(t: vec2f) -> i32 {
  if (t.x < 0.0 || t.y < 0.0 || t.x >= ${MW}.0 || t.y >= ${MH}.0) { return ${TREE}; }
  return i32(textureLoad(mapTex, vec2i(t), 0).r * 255.0 + 0.5);
}
fn varAt(t: vec2f) -> f32 { return textureLoad(mapTex, vec2i(clamp(t, vec2f(0.0), vec2f(${MW - 1}.0, ${MH - 1}.0))), 0).g * 255.0; }
// crisp pixel-art lookup of a 16x16 atlas tile
fn tileTex(k: i32, local: vec2f) -> vec3f {
  let r = v.tiles[k];
  let sz = vec2f(textureDimensions(atlas));
  let p = r.xy * sz + floor(fract(local) * 16.0);
  return textureLoad(atlas, vec2i(p), 0).rgb;
}
fn toWorld(s: vec2f) -> vec2f { return v.center + rot2(v.rot) * (s - v.size * 0.5) / v.zoom; }

// top-down grass: chunky 2-unit "texels", soft color patches, blade specks
fn grassTex(w: vec2f) -> vec3f {
  let q = floor(w / 2.0);
  let n = valueNoise(w * 0.025) * 0.7 + valueNoise(w * 0.09) * 0.3;
  var c = mix(vec3f(0.3, 0.56, 0.24), vec3f(0.45, 0.72, 0.31), n);
  c *= 0.92 + 0.16 * hash21(q);
  c = mix(c, vec3f(0.2, 0.42, 0.19), step(0.94, hash21(q + 7.0)));
  c = mix(c, vec3f(0.6, 0.84, 0.42), step(0.96, hash21(q + 3.0)));
  return c;
}

// tree canopies may overhang into neighbouring tiles: returns (coverage, shading)
fn canopy(w: vec2f) -> vec2f {
  let t = floor(w / ${TS}.0);
  var best = vec2f(0.0, 0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let tt = t + vec2f(f32(i), f32(j));
      if (tileAt(tt) == ${TREE}) {
        let c = (tt + vec2f(0.5)) * ${TS}.0 + (hash22(tt) - vec2f(0.5)) * 8.0;
        let r = ${TS}.0 * (0.55 + 0.15 * hash21(tt + 3.0));
        let d = length(w - c) / r;
        let lump = 0.08 * sin(atan2(w.y - c.y, w.x - c.x) * 7.0 + hash21(tt) * 6.0);
        if (d < 1.0 + lump) {
          let sh = clamp(1.0 - length(w - c - vec2f(-0.3, -0.35) * r) / (r * 1.3), 0.0, 1.0);
          best = vec2f(1.0, max(best.y, sh));
        }
      }
    }
  }
  return best;
}

fn flatColor(k: i32) -> vec3f {
  if (k == ${WATER} || k == ${POOLT}) { return vec3f(0.25, 0.5, 0.85); }
  if (k == ${SAND}) { return vec3f(0.85, 0.78, 0.55); }
  if (k == ${STONE}) { return vec3f(0.7, 0.7, 0.68); }
  if (k == ${WALL}) { return vec3f(0.35, 0.25, 0.25); }
  if (k == ${WOOD}) { return vec3f(0.62, 0.45, 0.28); }
  if (k == ${TREE}) { return vec3f(0.12, 0.38, 0.18); }
  return vec3f(0.36, 0.62, 0.3);
}

@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  let w = toWorld(i.pos.xy);
  let tf = w / ${TS}.0;
  let t = floor(tf);
  let k = tileAt(t);
  if (v.style > 0.5) {
    // minimap style: flat colors, no textures — a different "look" of the same world
    var c = flatColor(k);
    let g = fract(tf);
    c *= 0.92 + 0.08 * step(0.06, min(g.x, g.y));
    return vec4f(c, 1.0);
  }
  var c = grassTex(w);
  if (k == ${SAND}) { c = tileTex(1, tf); }
  if (k == ${STONE}) { c = tileTex(6, tf); }
  if (k == ${WOOD}) { c = tileTex(8, tf); }
  if (k == ${WATER}) {
    let fr = i32(fmod(floor(v.time * 4.0), 4.0));
    c = tileTex(2 + fr, tf + vec2f(v.time * 0.05, 0.0));
    // foam along the shore
    let g = fract(tf);
    var shore = 0.0;
    if (tileAt(t + vec2f(1.0, 0.0)) != ${WATER} && tileAt(t + vec2f(1.0, 0.0)) != ${WOOD}) { shore = max(shore, smoothstep(0.8, 1.0, g.x)); }
    if (tileAt(t - vec2f(1.0, 0.0)) != ${WATER} && tileAt(t - vec2f(1.0, 0.0)) != ${WOOD}) { shore = max(shore, smoothstep(0.2, 0.0, g.x)); }
    if (tileAt(t + vec2f(0.0, 1.0)) != ${WATER} && tileAt(t + vec2f(0.0, 1.0)) != ${WOOD}) { shore = max(shore, smoothstep(0.8, 1.0, g.y)); }
    if (tileAt(t - vec2f(0.0, 1.0)) != ${WATER} && tileAt(t - vec2f(0.0, 1.0)) != ${WOOD}) { shore = max(shore, smoothstep(0.2, 0.0, g.y)); }
    c = mix(c, vec3f(0.85, 0.95, 1.0), shore * (0.6 + 0.4 * sin(v.time * 3.0 + w.x * 0.1)));
  }
  if (k == ${POOLT}) {
    c = mix(vec3f(0.08, 0.2, 0.35), vec3f(0.12, 0.3, 0.45), fract(tf.y));
  }
  if (k == ${WALL}) {
    c = tileTex(7, tf);
    // fake height: the south face of the wall is darker
    let below = tileAt(t + vec2f(0.0, 1.0));
    if (below != ${WALL}) { c *= mix(1.0, 0.55, step(0.55, fract(tf.y))); }
    c *= mix(1.15, 1.0, step(0.12, fract(tf.y)));
  }
  if (k == ${GRASS} || k == ${TREE}) {
    let fv = varAt(t);
    if (fv > 0.5) {
      // a few flowers
      let fp = fract(tf * 3.0) - vec2f(0.5);
      let fid = floor(tf * 3.0);
      let on = step(0.6, hash21(fid));
      let fc = mix(vec3f(1.0, 0.85, 0.3), vec3f(1.0, 0.5, 0.7), step(1.5, fv));
      c = mix(c, fc, on * (1.0 - smoothstep(0.1, 0.16, length(fp - (hash22(fid) - vec2f(0.5)) * 0.4))));
    }
  }
  // shadows cast to the south-east by walls and tree canopies
  if (k != ${WALL}) {
    let sw = w - vec2f(7.0, 10.0);
    var sh = 0.0;
    if (tileAt(floor(sw / ${TS}.0)) == ${WALL}) { sh = 1.0; }
    sh = max(sh, canopy(sw).x * 0.8);
    c *= 1.0 - 0.35 * sh;
  }
  let cp = canopy(w);
  if (cp.x > 0.5) {
    let leaf = tileTex(9, w / 20.0);
    c = mix(vec3f(0.08, 0.25, 0.13), vec3f(0.35, 0.65, 0.28), cp.y) * (0.75 + 0.5 * leaf.g);
  }
  return vec4f(c, 1.0);
}
`;

// portals, reflecting pool: drawn on top of a view, sampling other render targets
const OVERLAY = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}
fn toWorld(s: vec2f) -> vec2f { return v.center + rot2(v.rot) * (s - v.size * 0.5) / v.zoom; }
fn portalTex(src: f32, uv: vec2f) -> vec3f {
  if (src < 0.5) { return textureSampleLevel(texA, samp, uv, 0.0).rgb; }
  if (src < 1.5) { return textureSampleLevel(texB, samp, uv, 0.0).rgb; }
  return textureSampleLevel(texM, samp, uv, 0.0).rgb;
}
fn drawPortal(col: vec4f, w: vec2f, uv: vec2f, P: vec4f, src: f32, tint: vec3f) -> vec4f {
  // P = (x, y, rx, ry) in world units
  let q = (w - P.xy) / P.zw;
  let d = length(q);
  let px = 1.5 / (v.zoom * P.z);
  var c = col;
  if (d < 1.0) {
    var inner = portalTex(src, uv);
    if (src > 2.5) { inner = vec3f(0.05); }
    // swirl at the edge, darkening toward the rim
    inner *= mix(1.0, 0.45, smoothstep(0.6, 1.0, d));
    c = vec4f(mix(c.rgb, inner, 1.0 - smoothstep(1.0 - px, 1.0, d)), 1.0);
  }
  let ang = atan2(q.y, q.x);
  let swirl = 0.5 + 0.5 * sin(ang * 6.0 - v.time * 5.0 + d * 8.0);
  let rim = exp(-abs(d - 1.0) * 14.0) * (0.75 + 0.5 * swirl);
  c = vec4f(c.rgb + tint * rim * 1.4, max(c.a, clamp(rim * 1.2, 0.0, 1.0)));
  return c;
}
@fragment fn fs_main(i: VO) -> @location(0) vec4f {
  let s = i.pos.xy;
  let uv = s / v.size;
  let w = toWorld(s);
  var col = vec4f(0.0);
  // reflecting pool: the world rendered from a camera mirrored about the pool's north edge, flipped back
  if (v.mirrorOn > 0.5 && w.x > v.pool.x && w.x < v.pool.z && w.y > v.pool.y && w.y < v.pool.w) {
    let ripple = vec2f(sin(w.y * 0.25 + v.time * 3.0), cos(w.x * 0.2 + v.time * 2.0)) * 0.0025;
    let r = textureSampleLevel(texR, samp, vec2f(uv.x, 1.0 - uv.y) + ripple, 0.0).rgb;
    let edge = min(min(w.x - v.pool.x, v.pool.z - w.x), min(w.y - v.pool.y, v.pool.w - w.y));
    col = vec4f(mix(r * vec3f(0.6, 0.8, 1.0), vec3f(0.1, 0.25, 0.4), 0.3), smoothstep(0.0, 3.0, edge));
  }
  if (v.srcA >= 0.0) { col = drawPortal(col, w, uv, v.portalA, v.srcA, vec3f(0.3, 0.7, 1.0)); }
  if (v.srcB >= 0.0) { col = drawPortal(col, w, uv, v.portalB, v.srcB, vec3f(1.0, 0.55, 0.15)); }
  return vec4f(col.rgb * col.a, col.a);
}
`;

const CRT = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  // barrel distortion
  let cc = uv - vec2f(0.5);
  let q = vec2f(0.5) + cc * (1.0 + dot(cc, cc) * 0.22);
  if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) { return vec4f(0.02, 0.03, 0.02, 1.0); }
  let jitter = (hash11(floor(u.time * 20.0)) - 0.5) * 0.004;
  var c = TEX(src, q + vec2f(jitter, 0.0)).rgb;
  let l = dot(c, vec3f(0.3, 0.59, 0.11));
  c = mix(vec3f(l), c, 0.25) * vec3f(0.95, 1.4, 1.0);                // greenish monochrome
  c *= 0.75 + 0.25 * sin(q.y * u.lines * 3.14159);                     // scanlines
  c += (hash21(floor(px) + vec2f(u.time * 60.0, 0.0)) - 0.5) * 0.08;   // static
  let bar = smoothstep(0.0, 0.08, abs(fract(q.y - u.time * 0.15) - 0.5)); // rolling bar
  c *= 0.85 + 0.15 * bar;
  c *= 1.0 - dot(cc, cc) * 1.6;                                        // vignette
  let rec = step(length((q - vec2f(0.08, 0.1)) * vec2f(1.0, u.aspect)), 0.025) * step(0.5, fract(u.time));
  c = mix(c, vec3f(1.0, 0.15, 0.1), rec);
  return vec4f(c, 1.0);
}
`;

export default {
  interaction: 'WASD / arrows move the hero (idle = autopilot). Split-screen: WASD = player 1, arrows = player 2.',
  keys: true,
  examples: [
    {
      id: 'minimap',
      label: 'Live minimap',
      kind: 'In a game',
      note: 'The world is drawn <b>twice</b> per frame: once around the hero for the main view, once in full into a small 256-pixel texture with a flat “map” shader. The minimap is just that texture drawn as a sprite, plus dots and the main camera’s rectangle.',
      hint: 'WASD / arrows: walk (idle = autopilot)',
    },
    {
      id: 'portals',
      label: 'Portals & mirror',
      kind: 'In a game',
      note: 'Each portal shows a render of the world from the <i>other</i> portal’s point of view (rotated 90°). Inside a portal view the partner portal shows <b>last frame’s</b> main image — feedback that builds an infinite tunnel. The pool is a mirror: the world rendered from a reflected camera, flipped. Walk through a portal!',
      hint: 'WASD / arrows: walk into the portals',
    },
    {
      id: 'split',
      label: 'Split-screen & CCTV',
      kind: 'In a game',
      note: 'Two players, two cameras, two render targets placed side by side. The picture-in-picture is a fixed security camera rendered into a tiny texture and then run through a CRT shader (scanlines, static, green tint, rolling bar).',
      hint: 'P1: WASD · P2: arrows',
    },
  ],
  controls: [
    { type: 'slider', key: 'zoom', label: 'Main camera zoom', min: 0.75, max: 4, step: 0.01, value: 2, help: 'World pixels per screen pixel.' },
    { type: 'slider', key: 'mapRes', label: 'Minimap resolution', min: 64, max: 512, step: 1, value: 256, showFor: ['minimap'], help: 'Size of the minimap render target in pixels.' },
    { type: 'toggle', key: 'mapStyle', label: 'Flat “map” shader for the minimap', value: true, showFor: ['minimap'], help: 'Off = the minimap uses the full textured shader, just smaller.' },
    { type: 'toggle', key: 'recursion', label: 'Recursive portals', value: true, showFor: ['portals'], help: 'Feed last frame back into the portals (the infinite tunnel).' },
    { type: 'toggle', key: 'mirror', label: 'Reflecting pool', value: true, showFor: ['portals'] },
    { type: 'slider', key: 'portalRes', label: 'Portal texture resolution', min: 0.1, max: 1, step: 0.01, value: 1, showFor: ['portals'], help: 'Fraction of the screen resolution used for the portal & mirror render targets.' },
    { type: 'slider', key: 'portalSize', label: 'Portal size', min: 0.5, max: 2, step: 0.01, value: 1, showFor: ['portals'] },
    { type: 'slider', key: 'pipRes', label: 'CCTV resolution', min: 40, max: 400, step: 1, value: 160, showFor: ['split'], help: 'Vertical resolution of the security camera render target.' },
    { type: 'toggle', key: 'pip', label: 'Show security camera', value: true, showFor: ['split'] },
    { type: 'toggle', key: 'auto', label: 'Autopilot when idle', value: true },
  ],
  about: {
    summary: 'A render target is a texture you draw into instead of the screen. Render the world into one and you can show it anywhere: as a minimap, through a portal, in a mirror, on a TV, or as one half of a split screen.',
    what: `<p>A small top-down world drawn up to five times per frame, each time from a different camera and into a different texture.
      Those textures are then used like images: scaled down (minimap), cut into an oval (portals), flipped (mirror), side by side (split-screen), and post-processed (CCTV).</p>`,
    how: `<ol>
      <li>Wrap your world drawing in one function: <code>renderWorld(target, camera)</code>. Here that is a tile shader + a sprite batch, both driven by a camera uniform.</li>
      <li><b>Minimap</b>: call it with a camera that sees the whole map and a small target; draw that target as a sprite. Optionally use a cheaper “map style” shader.</li>
      <li><b>Portals</b>: for portal A, use the main camera <i>transformed by the portal</i> (moved to B, rotated by B’s angle). Draw the result inside A’s oval at the <b>same screen coordinates</b>.
        Because the two portals are linked, a portal seen inside a portal view shows exactly the previous main frame — so feeding last frame back gives free recursion.</li>
      <li><b>Mirror</b>: render with the camera reflected about the mirror line, then sample the texture flipped.</li>
      <li><b>Split-screen</b>: one camera and one target per player (or one target with two viewports). <b>Picture-in-picture</b>: a fixed camera into a low-res target, then a CRT post-process when compositing.</li>
    </ol>
    <p>Rule of thumb: you can’t sample a texture while rendering into it, so feedback effects keep two textures and swap them (“ping-pong”).</p>`,
    uses: [
      { title: 'Portals', text: 'Portal (2007) is 3D, but 2D games do the same: Teleglitch, Antichamber-style tricks, “Portal 2D” fan games.' },
      { title: 'Minimaps & radars', text: 'Almost every RPG, strategy and racing game; a live render or a cached one updated only when the world changes.' },
      { title: 'Split-screen co-op', text: 'Couch co-op (Lego games, Overcooked-style versus modes, Micro Machines).' },
      { title: 'In-world screens', text: 'Security cameras (Five Nights at Freddy’s), mirrors, TVs, rear-view mirrors in racers, scopes.' },
    ],
    try: [
      'In <b>Live minimap</b>, drag <i>Minimap resolution</i> to 64 — the texture is tiny but still useful. Turn off the flat shader to see the full textured world shrunk.',
      'In <b>Portals</b>, walk into the blue portal and come out of the orange one, turned 90°.',
      'Turn off <i>Recursive portals</i>: the tunnel disappears and the inner portal goes dark.',
      'In <b>Split-screen</b>, drag <i>CCTV resolution</i> down to 40 for a crunchy, old-camera look.',
    ],
    ask: [
      'render the world into a texture and use it as a live minimap',
      '2D portals that show the view from the linked portal, with recursion',
      'a mirror / reflecting pool using a reflected camera render target',
      'split-screen co-op with one render target per player',
      'a security camera feed with a CRT shader (picture-in-picture)',
    ],
    perf: `<p>Every extra view re-runs the world rendering, so cost scales with <i>number of views × pixels per view</i>. Keep secondary targets small
      (the minimap and CCTV here are a few hundred pixels), update minimaps less often if the world is static, and reuse last frame for recursion instead of drawing N nested levels.</p>`,
    api: `<p>WebGPU: every target is a texture with <code>RENDER_ATTACHMENT | TEXTURE_BINDING</code>; each view gets its own uniform buffer because all
      <code>writeBuffer</code> calls happen before the frame’s commands run. WebGL2 does the same with framebuffer objects (FBOs).</p>`,
    code: [
      {
        title: 'Portal camera = main camera moved through the portal',
        lang: 'js',
        src: `// T_A maps points near portal A to points near portal B (rotated by PORTAL_ROT)
const throughA = (x, y) => {
  const dx = x - A[0], dy = y - A[1];
  return [B[0] + dx * cos(rot) - dy * sin(rot), B[1] + dx * sin(rot) + dy * cos(rot)];
};
viewA.center = throughA(main.x, main.y);
viewA.rot = main.rot + rot;
renderWorld(targetA, viewA);   // then draw targetA inside A's oval at the same screen uv`,
      },
      {
        title: 'Inside the oval: sample at the same screen position',
        lang: 'wgsl',
        src: `let uv = s / v.size;                 // this pixel's screen uv
let q = (w - P.xy) / P.zw;           // position relative to the portal oval
if (length(q) < 1.0) {
  col = textureSampleLevel(texA, samp, uv, 0.0).rgb;   // the other side
}`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasTex = gpu.textureFromImage(atlas.canvas, { label: 'atlas' });
    const map = buildMap();
    const mapData = new Uint8Array(MW * MH * 4);
    for (let i = 0; i < MW * MH; i++) {
      mapData[i * 4] = map.t[i];
      mapData[i * 4 + 1] = map.v[i];
      mapData[i * 4 + 3] = 255;
    }
    const mapTex = gpu.textureFromData(MW, MH, mapData);
    const tileUV = new Float32Array(40);
    TILE_NAMES.forEach((n, i) => tileUV.set(atlas.uv(n), i * 4));
    const fmt = gpu.format;

    const tmpl = gpu.uniforms(VIEW_FIELDS, 'View'); // every view gets its own block with this layout
    const worldProg = gpu.program({ label: 'rtt-world', bindings: { v: { uniform: tmpl }, mapTex: { texture: true }, atlas: { texture: true } }, include: ['math', 'hash', 'noise'], code: WORLD });
    const worldPipe = worldProg.renderPipeline({ format: fmt });
    const overProg = gpu.program({
      label: 'rtt-overlay',
      bindings: { v: { uniform: tmpl }, texA: { texture: true }, texB: { texture: true }, texM: { texture: true }, texR: { texture: true }, samp: { sampler: true } },
      include: ['math'],
      code: OVERLAY,
    });
    const overPipe = overProg.renderPipeline({ format: fmt, blend: 'premultiplied' });
    const CU = gpu.uniforms({ time: 'f32', lines: 'f32', aspect: 'f32' }, 'Crt');
    const crt = gpu.fullscreen({ label: 'cctv', code: CRT, uniforms: CU, textures: ['src'], include: ['hash'] });

    const sprites = new SpriteBatch(gpu, { texture: atlasTex.createView(), filter: 'nearest' });
    const panels = new SpriteBatch(gpu, { texture: atlasTex.createView(), filter: 'linear' });
    const shapes = new ShapeBatch(gpu);
    const blank = gpu.target(4, 4, { format: fmt });
    const views = {}; // name -> { block, cam }
    const viewOf = (name) => {
      if (!views[name]) {
        const block = gpu.uniforms(VIEW_FIELDS, 'View');
        views[name] = { block, cam: new Camera2D() };
      }
      return views[name];
    };
    const targets = {};
    const targetOf = (name, w, h) => {
      w = Math.max(4, Math.round(w));
      h = Math.max(4, Math.round(h));
      const t = targets[name];
      if (t && t.width === w && t.height === h) return t;
      t?.destroy();
      return (targets[name] = gpu.target(w, h, { format: fmt, label: name }));
    };
    let mainPP = null;

    const blocked = (x, y) => {
      const tx = Math.floor(x / TS);
      const ty = Math.floor(y / TS);
      if (tx < 0 || ty < 0 || tx >= MW || ty >= MH) return true;
      const k = map.t[ty * MW + tx];
      if (k === WOOD) return false;
      return BLOCK.has(k);
    };
    const R = rng(5);
    const mkEnt = (x, y, kind) => ({ x, y, kind, tx: x, ty: y, dir: 1, walk: 0, cool: 0, moving: false });
    const hero = mkEnt(20.5 * TS, 14.8 * TS, 'hero');
    const p2 = mkEnt(36 * TS, 12 * TS, 'bunny');
    const mobs = [];
    for (let i = 0; i < 9; i++) {
      let x;
      let y;
      do {
        x = (3 + R() * (MW - 6)) * TS;
        y = (3 + R() * (MH - 6)) * TS;
      } while (blocked(x, y));
      mobs.push(mkEnt(x, y, i % 3 === 2 ? 'bat' : 'slime'));
    }
    mobs.push(mkEnt(48 * TS, 23 * TS, 'slime')); // a thief in the house
    const coins = [];
    for (let i = 0; i < 14; i++) {
      let x;
      let y;
      do {
        x = (3 + R() * (MW - 6)) * TS;
        y = (3 + R() * (MH - 6)) * TS;
      } while (blocked(x, y));
      coins.push({ x, y });
    }
    const chest = { x: 50.5 * TS, y: 21.5 * TS };
    let idle = 99;
    let heroPath = 0;
    const camMain = { x: hero.x, y: hero.y };
    const camP2 = { x: p2.x, y: p2.y };
    const hud = overlayTag(ctx, 'right:8px;bottom:8px');

    const throughA = (x, y) => {
      const dx = x - PORTAL_A[0];
      const dy = y - PORTAL_A[1];
      const c = Math.cos(PORTAL_ROT);
      const s = Math.sin(PORTAL_ROT);
      return [PORTAL_B[0] + dx * c - dy * s, PORTAL_B[1] + dx * s + dy * c];
    };
    const throughB = (x, y) => {
      const dx = x - PORTAL_B[0];
      const dy = y - PORTAL_B[1];
      const c = Math.cos(-PORTAL_ROT);
      const s = Math.sin(-PORTAL_ROT);
      return [PORTAL_A[0] + dx * c - dy * s, PORTAL_A[1] + dx * s + dy * c];
    };

    const moveEnt = (e, dx, dy, dt, speed, portals, portalR) => {
      const len = Math.hypot(dx, dy);
      e.moving = len > 0.01;
      if (!e.moving) return;
      const vx = (dx / len) * speed * dt;
      const vy = (dy / len) * speed * dt;
      if (!blocked(e.x + vx + Math.sign(vx) * 8, e.y)) e.x += vx;
      if (!blocked(e.x, e.y + vy + Math.sign(vy) * 6)) e.y += vy;
      if (Math.abs(vx) > 0.01) e.dir = Math.sign(vx);
      e.walk += dt * 10;
      e.cool = Math.max(0, e.cool - dt);
      if (portals && e.cool <= 0) {
        for (const [P, fn] of [[PORTAL_A, throughA], [PORTAL_B, throughB]]) {
          const q = Math.hypot((e.x - P[0]) / (34 * portalR), (e.y - P[1]) / (48 * portalR));
          if (q < 0.55) {
            const [nx, ny] = fn(e.x, e.y);
            const ox = e.x;
            const oy = e.y;
            e.x = nx;
            e.y = ny;
            e.cool = 1.2;
            // retarget wanderers so they walk out of the portal
            const [tx, ty] = fn(e.tx, e.ty);
            e.tx = tx;
            e.ty = ty;
            e.jumped = [nx - ox, ny - oy];
            break;
          }
        }
      }
    };
    const wander = (e, dt, speed, portals, portalR) => {
      if (Math.hypot(e.tx - e.x, e.ty - e.y) < 8 || R() < dt * 0.15) {
        const a = R() * Math.PI * 2;
        const d = 60 + R() * 160;
        const tx = e.x + Math.cos(a) * d;
        const ty = e.y + Math.sin(a) * d;
        if (!blocked(tx, ty)) {
          e.tx = tx;
          e.ty = ty;
        }
      }
      const ox = e.x;
      const oy = e.y;
      moveEnt(e, e.tx - e.x, e.ty - e.y, dt, speed, portals, portalR);
      if (Math.hypot(e.x - ox, e.y - oy) < speed * dt * 0.2 && e.moving) {
        e.tx = e.x;
        e.ty = e.y;
      }
    };

    /** Draw the whole world (tiles + sprites + overlays) for one camera into one target. */
    const renderWorld = (enc, target, name, cam, opts) => {
      const view = viewOf(name);
      const W = target.width;
      const H = target.height;
      view.block
        .set('center', [cam.x, cam.y])
        .set('size', [W, H])
        .set('zoom', cam.zoom)
        .set('rot', cam.rot || 0)
        .set('time', opts.time)
        .set('style', opts.style || 0)
        .set('portalA', opts.portalA || [0, 0, 1, 1])
        .set('portalB', opts.portalB || [0, 0, 1, 1])
        .set('pool', POOL)
        .set('srcA', opts.srcA ?? -1)
        .set('srcB', opts.srcB ?? -1)
        .set('mirrorOn', opts.mirror ? 1 : 0)
        .set('tiles', tileUV);
      view.block.upload();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: target.view, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
      pass.setPipeline(worldPipe);
      pass.setBindGroup(0, worldProg.bind({ v: view.block, mapTex, atlas: atlasTex }));
      pass.draw(3);
      pass.end();
      if (opts.sprites !== false) {
        const c = view.cam;
        c.setViewport(W, H);
        c.x = cam.x;
        c.y = cam.y;
        c.zoom = cam.zoom;
        c.rotation = cam.rot || 0;
        const t = opts.time;
        for (const co of coins) sprites.draw(co.x, co.y, 14, 14, { uv: atlas.uv(`coin_${Math.floor(t * 10 + co.x) % 8}`) });
        sprites.draw(chest.x, chest.y, 26, 26, { uv: atlas.uv('chest') });
        const ents = [...mobs, hero];
        if (opts.p2) ents.push(p2);
        ents.sort((a, b) => a.y - b.y);
        for (const e of ents) {
          let frame;
          let size = 26;
          if (e.kind === 'hero') frame = e.moving ? `hero_run_${Math.floor(e.walk) % 4}` : `hero_idle_${Math.floor(t * 2) % 2}`;
          else if (e.kind === 'bunny') frame = `bunny_${Math.floor(e.walk * 0.5) % 2}`;
          else if (e.kind === 'bat') {
            frame = `bat_${Math.floor(t * 8 + e.x) % 2}`;
            size = 22;
          } else frame = `slime_${Math.floor(t * 5 + e.x) % 3}`;
          const fr = atlas.frames[frame];
          sprites.draw(e.x, e.y + 6, (size * fr.w) / fr.h, size, { uv: atlas.uv(frame), anchor: [0.5, 1], flipX: e.dir < 0 });
        }
        sprites.flush(enc, target, c);
      }
      if (opts.overlay) {
        const pass2 = enc.beginRenderPass({ colorAttachments: [{ view: target.view, loadOp: 'load', storeOp: 'store' }] });
        pass2.setPipeline(overPipe);
        pass2.setBindGroup(0, overProg.bind({ v: view.block, texA: opts.texA || blank, texB: opts.texB || blank, texM: opts.texM || blank, texR: opts.texR || blank, samp: 'linear' }));
        pass2.draw(3);
        pass2.end();
      }
    };

    return {
      frame(ctx) {
        const p = ctx.params;
        const W = ctx.width;
        const H = ctx.height;
        const ex = ctx.example;
        const enc = ctx.encoder;
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 30);
        const keys = ctx.keys;
        const t = ctx.time;
        const zoom = p.zoom * Math.max(0.5, H / 700);
        const portalR = p.portalSize;
        const usePortals = ex === 'portals';
        const canvas = { view: ctx.target, format: fmt, width: W, height: H };
        // batches: begin() exactly once per frame (each flush gets its own camera slot)
        sprites.begin();
        panels.begin();
        shapes.begin();

        // ------------------------------------------------------------ movement
        if (dt > 0) {
          const split = ex === 'split';
          const ax = split ? keyAxis(keys, ['a', 'KeyA'], ['d', 'KeyD']) : keyAxis(keys, KEYS.left, KEYS.right);
          const ay = split ? keyAxis(keys, ['w', 'KeyW'], ['s', 'KeyS']) : keyAxis(keys, KEYS.up, KEYS.down);
          const manual = anyKey(keys, [KEYS.up, KEYS.down, KEYS.left, KEYS.right]);
          idle = manual ? 0 : idle + dt;
          if (ax || ay) moveEnt(hero, ax, ay, dt, 140, usePortals, portalR);
          else if (p.auto && idle > 2.5) {
            if (usePortals) {
              // stroll: into portal A, out of B, around, into B, out of A ...
              const pts = [PORTAL_A, [PORTAL_B[0], PORTAL_B[1] + 110], PORTAL_B, [PORTAL_A[0] + 120, PORTAL_A[1] - 40]];
              const tgt = pts[heroPath % 4];
              if (Math.hypot(tgt[0] - hero.x, tgt[1] - hero.y) < 14) heroPath++;
              hero.jumped = null;
              moveEnt(hero, tgt[0] - hero.x, tgt[1] - hero.y, dt, 120, true, portalR);
              if (hero.jumped) heroPath++;
            } else wander(hero, dt, 110, false, portalR);
          } else hero.moving = false;
          if (hero.jumped) {
            // keep the camera glued to the hero through the teleport
            camMain.x += hero.jumped[0];
            camMain.y += hero.jumped[1];
            hero.jumped = null;
          }
          if (split) {
            const bx = keyAxis(keys, ['ArrowLeft'], ['ArrowRight']);
            const by = keyAxis(keys, ['ArrowUp'], ['ArrowDown']);
            if (bx || by) moveEnt(p2, bx, by, dt, 140, false, 1);
            else if (p.auto) wander(p2, dt, 100, false, 1);
          }
          for (const m of mobs) {
            wander(m, dt, m.kind === 'bat' ? 70 : 40, usePortals, portalR);
            m.jumped = null;
          }
          const k = 1 - Math.exp(-dt * 6);
          camMain.x += (hero.x - camMain.x) * k;
          camMain.y += (hero.y - camMain.y) * k;
          camP2.x += (p2.x - camP2.x) * k;
          camP2.y += (p2.y - camP2.y) * k;
        }

        const pA = [PORTAL_A[0], PORTAL_A[1], 34 * portalR, 48 * portalR];
        const pB = [PORTAL_B[0], PORTAL_B[1], 34 * portalR, 48 * portalR];
        let info = '';
        // ------------------------------------------------------------ examples
        if (ex === 'minimap') {
          const main = { x: camMain.x, y: camMain.y, zoom };
          renderWorld(enc, canvas, 'main', main, { time: t });
          const mw = Math.round(p.mapRes);
          const mh = Math.round((mw * MH) / MW);
          const mt = targetOf('minimap', mw, mh);
          renderWorld(enc, mt, 'mini', { x: (MW * TS) / 2, y: (MH * TS) / 2, zoom: mw / (MW * TS) }, { time: t, style: p.mapStyle ? 1 : 0, sprites: !p.mapStyle });
          // composite the minimap
          const scr = new Camera2D();
          scr.setViewport(W, H);
          const dw = Math.min(W * 0.3, 300 * (H / 700) + 120);
          const dh = (dw * MH) / MW;
          const x0 = W - dw - 12;
          const y0 = 52;
          shapes.rect(x0 - 4, y0 - 4, dw + 8, dh + 8, [0.05, 0.06, 0.1, 0.85], { radius: 8 });
          shapes.flush(enc, canvas, scr);
          panels.draw(x0, y0, dw, dh, { anchor: [0, 0] });
          panels.flush(enc, canvas, scr, { texture: mt });
          const toMap = (x, y) => [x0 + (x / (MW * TS)) * dw, y0 + (y / (MH * TS)) * dh];
          // the main camera's visible rectangle
          const vw = W / zoom;
          const vh = H / zoom;
          const [rx, ry] = toMap(camMain.x - vw / 2, camMain.y - vh / 2);
          shapes.rect(rx, ry, (vw / (MW * TS)) * dw, (vh / (MH * TS)) * dh, [1, 1, 1, 0.9], { stroke: 1.5 });
          for (const m of mobs) {
            const [mx, my] = toMap(m.x, m.y);
            shapes.circle(mx, my, 2.5, '#ff5a5f');
          }
          for (const c of coins) {
            const [cx, cy] = toMap(c.x, c.y);
            shapes.circle(cx, cy, 1.6, '#ffcd75');
          }
          const [hx, hy] = toMap(hero.x, hero.y);
          shapes.circle(hx, hy, 4.5, '#ffffff', { glow: 6, glowStrength: 0.8 });
          shapes.circle(hx, hy, 3, '#41a6f6');
          shapes.rect(x0 - 4, y0 - 4, dw + 8, dh + 8, [0.6, 0.65, 0.8, 1], { radius: 8, stroke: 2 });
          shapes.flush(enc, canvas, scr);
          info = `2 renders of the world · minimap target ${mw}×${mh}`;
        } else if (ex === 'portals') {
          const main = { x: camMain.x, y: camMain.y, zoom, rot: 0 };
          if (!mainPP || mainPP.width !== W || mainPP.height !== H) {
            mainPP?.destroy();
            mainPP = gpu.pingPong(W, H, { format: fmt, label: 'main-pp' });
          }
          // portal/mirror views are sampled by screen uv, so they can be any resolution
          const rs = ctx.testMode ? 0.35 : p.portalRes;
          const tA = targetOf('portalA', W * rs, H * rs);
          const tB = targetOf('portalB', W * rs, H * rs);
          const tR = targetOf('mirror', W * rs, H * rs);
          const rec = p.recursion ? 2 : 3; // 2 = previous main frame, 3 = dark
          // portal A shows the world around B: the main camera carried through A
          const [ax, ay] = throughA(main.x, main.y);
          renderWorld(enc, tA, 'viewA', { x: ax, y: ay, zoom: zoom * (tA.width / W), rot: PORTAL_ROT }, { time: t, overlay: true, portalA: pA, portalB: pB, srcA: rec, srcB: rec, texM: mainPP.read });
          const [bx, by] = throughB(main.x, main.y);
          renderWorld(enc, tB, 'viewB', { x: bx, y: by, zoom: zoom * (tB.width / W), rot: -PORTAL_ROT }, { time: t, overlay: true, portalA: pA, portalB: pB, srcA: rec, srcB: rec, texM: mainPP.read });
          // the reflecting pool: camera mirrored about the pool's north edge
          if (p.mirror) renderWorld(enc, tR, 'mirror', { x: main.x, y: 2 * POOL[1] - main.y, zoom: zoom * (tR.width / W) }, { time: t });
          renderWorld(enc, mainPP.write, 'main', main, { time: t, overlay: true, portalA: pA, portalB: pB, srcA: 0, srcB: 1, texA: tA, texB: tB, texR: tR, mirror: p.mirror });
          gpu.blit(enc, mainPP.write, canvas, { filter: 'nearest' });
          mainPP.swap();
          info = `${p.mirror ? 4 : 3} world renders / frame · recursion ${p.recursion ? 'on (feedback)' : 'off'}`;
        } else {
          const half = Math.floor(W / 2);
          const tL = targetOf('splitL', half, H);
          const tR2 = targetOf('splitR', W - half, H);
          renderWorld(enc, tL, 'left', { x: camMain.x, y: camMain.y, zoom }, { time: t, p2: true });
          renderWorld(enc, tR2, 'right', { x: camP2.x, y: camP2.y, zoom }, { time: t, p2: true });
          const scr = new Camera2D();
          scr.setViewport(W, H);
          panels.draw(0, 0, half, H, { anchor: [0, 0] });
          panels.flush(enc, canvas, scr, { texture: tL, clear: [0, 0, 0, 1], filter: 'nearest' });
          panels.draw(half, 0, W - half, H, { anchor: [0, 0] });
          panels.flush(enc, canvas, scr, { texture: tR2, filter: 'nearest' });
          shapes.rect(half - 2, 0, 4, H, [0.05, 0.05, 0.08, 1]);
          shapes.flush(enc, canvas, scr);
          info = `3 cameras → 3 render targets`;
          if (p.pip) {
            const ph = Math.round(p.pipRes);
            const pw = Math.round((ph * 16) / 9);
            const tc = targetOf('cctv', pw, ph);
            const pan = Math.sin(t * 0.25) * 4 * TS;
            renderWorld(enc, tc, 'cctv', { x: 50.5 * TS + pan, y: 23.5 * TS, zoom: ph / (9 * TS) }, { time: t, p2: true });
            const dw = Math.min(W * 0.3, 360);
            const dh = (dw * 9) / 16;
            const x0 = Math.round(W / 2 - dw / 2);
            const y0 = 46;
            shapes.rect(x0 - 6, y0 - 6, dw + 12, dh + 12, [0.12, 0.12, 0.14, 1], { radius: 6 });
            shapes.flush(enc, canvas, scr);
            CU.set('time', t).set('lines', Math.min(ph, dh / 2.5)).set('aspect', dh / dw);
            crt.draw(enc, canvas, { src: tc }, { clear: false, viewport: [x0, y0, dw, dh] });
            info += ` · CCTV ${pw}×${ph}`;
          }
        }
        hud.textContent = info;
      },
    };
  },
};

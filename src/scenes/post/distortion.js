import { shaderScene } from '../../core/shaderscene.js';
import { overlays, GAME_POS_WGSL, gamePass, withGameIncludes } from './_shared.js';

// Distortion = "read the image from somewhere else". Every effect here only computes an offset for the
// texture coordinate (plus a little shading); the picture itself is untouched.

const MAX_WAVES = 8;
const st = { waves: [], lastSpawn: -10, rnd: 1 };
const ov = overlays();
const rand = () => {
  st.rnd = (st.rnd * 16807) % 2147483647;
  return (st.rnd - 1) / 2147483646;
};

const HEAT_MODES = ['heat', 'underwater', 'drunk', 'rain'];
const LENS_MODES = ['magnify', 'fisheye', 'pinch', 'swirl', 'barrel', 'pincushion'];

const IMAGE = /* wgsl */ `
${GAME_POS_WGSL}

// p units: p = px / resolution.y (aspect-correct, y down). Convert a p-space point to texture uv.
fn toUV(p: vec2f) -> vec2f { return vec2f(p.x * u.resolution.y / u.resolution.x, p.y); }

// ---------------------------------------------------------------- shockwaves
// returns xy = offset (p units), z = brightness of the ring front
fn shockwaves(p: vec2f) -> vec3f {
  var off: vec2f = vec2f(0.0);
  var glow: f32 = 0.0;
  for (var i = 0; i < ${MAX_WAVES}; i++) {
    let w: vec4f = u.waves[i];
    if (w.w > 0.0) {
      let c: vec2f = w.xy / u.resolution.y;
      let age: f32 = u.time - w.z;
      let R: f32 = age * u.speed;
      let d: vec2f = p - c;
      let r: f32 = length(d);
      let x: f32 = (r - R) / u.width;
      let fade: f32 = w.w * clamp(1.0 - R / 1.4, 0.0, 1.0) * smoothstep(0.0, 0.04, age);
      // derivative-of-a-bump profile: pushes pixels one way in front of the ring and the other way behind it
      let prof: f32 = x * exp(-x * x * 1.6);
      off += d / max(r, 0.0001) * prof * u.amp * fade;
      glow += exp(-x * x * 2.5) * fade;
    }
  }
  return vec3f(off, glow);
}

// ---------------------------------------------------------------- heat & water
fn heatOffset(p: vec2f) -> vec2f {
  let t: f32 = u.time * u.speed;
  let k: f32 = u.scale;
  // hot columns rising above each torch flame, plus a faint shimmer over the warm ground
  let dx: f32 = torchDX(p);
  let column: f32 = exp(-dx * dx / (0.05 * 0.05)) * smoothstep(0.705, 0.66, p.y) * smoothstep(0.28, 0.62, p.y);
  let ground: f32 = smoothstep(0.66, 0.8, p.y) * smoothstep(0.98, 0.84, p.y) * 0.45;
  let n: vec2f = vec2f(perlin(vec2f(p.x * 34.0 * k, p.y * 20.0 * k + t * 2.6)),
                       perlin(vec2f(p.x * 29.0 * k + 7.1, p.y * 25.0 * k + t * 3.1)));
  return n * (column + ground) * u.strength * 0.012;
}

fn underwaterOffset(p: vec2f) -> vec2f {
  let t: f32 = u.time * u.speed;
  let k: f32 = u.scale;
  return vec2f(sin(p.y * 11.0 * k + t * 1.7) + 0.45 * sin(p.y * 29.0 * k - t * 2.6 + p.x * 3.0),
               0.7 * cos(p.x * 8.0 * k + t * 1.3) + 0.3 * sin(p.x * 23.0 * k + t * 2.1)) * u.strength * 0.006;
}

// raindrops on a window: xy = where to look inside the drop (p units), z = drop mask
fn raindrops(p: vec2f) -> vec3f {
  var best: vec3f = vec3f(0.0, 0.0, 0.0);
  let t: f32 = u.time * u.speed;
  let k: f32 = u.scale;
  // static beads
  let g: f32 = 0.075 / k;
  let cell: vec2f = floor(p / g);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let id: vec2f = cell + vec2f(f32(i), f32(j));
      let h: vec3f = hash23(id);
      let c: vec2f = (id + 0.2 + 0.6 * h.xy) * g;
      let rad: f32 = g * (0.12 + 0.22 * h.z * h.z);
      let d: vec2f = (p - c) * vec2f(1.0, 0.85);
      let r: f32 = length(d) / rad;
      if (r < 1.0 && h.z > 0.25) {
        // a drop is a tiny fish-eye lens: it shows a flipped, shrunken view of what is behind it
        best = vec3f(c - d * 2.2 + vec2f(0.0, -rad * 0.6), smoothstep(1.0, 0.8, r));
      }
    }
  }
  // sliding drops with a wet trail
  let colW: f32 = 0.11 / k;
  let col: f32 = floor(p.x / colW);
  let hc: vec3f = hash23(vec2f(col, 17.0));
  let x0: f32 = (col + 0.3 + 0.4 * hc.x) * colW;
  let y0: f32 = fract(t * (0.12 + 0.25 * hc.y) + hc.z) * 1.4 - 0.2;
  let wob: f32 = 0.006 * sin(p.y * 40.0 + col);
  let rad2: f32 = 0.018 / k;
  let dd: vec2f = (p - vec2f(x0 + wob, y0)) * vec2f(1.0, 0.75);
  let r2: f32 = length(dd) / rad2;
  if (r2 < 1.0 && hc.z > 0.35) { best = vec3f(vec2f(x0, y0) - dd * 2.4, smoothstep(1.0, 0.8, r2)); }
  return best;
}

// ---------------------------------------------------------------- lenses
fn lens(p: vec2f, m: vec2f, mode: f32) -> vec2f {
  let d: vec2f = p - m;
  let r: f32 = length(d);
  let R: f32 = u.radius;
  let k: f32 = u.strength;
  if (mode > 3.5) {
    // whole-screen barrel (+) / pincushion (-) distortion, like a cheap camera lens or a CRT
    let a: f32 = u.resolution.x / u.resolution.y;
    let c: vec2f = p - vec2f(a * 0.5, 0.5);
    let sgn: f32 = select(-1.0, 1.0, mode < 4.5);
    let f: f32 = 1.0 - sgn * k * 0.35 * dot(c, c) / (0.25 * (a * a + 1.0)) * 1.6;
    return vec2f(a * 0.5, 0.5) + c * f;
  }
  if (r >= R) { return p; }
  let nr: f32 = r / R;
  if (mode < 0.5) {
    // magnifying glass: zoom inside, smooth bulge near the rim
    let z: f32 = 1.0 + 2.5 * k;
    let e: f32 = smoothstep(1.0, 0.82, nr);
    return m + d / mix(1.0, z, e);
  }
  if (mode < 1.5) { return m + d * pow(nr, 0.6 * k) ; }          // fisheye bulge (centre magnified)
  if (mode < 2.5) { return m + d * pow(nr, -0.6 * k) ; }         // pinch (centre shrunk)
  // swirl / twirl: rotate more the closer to the centre
  let a: f32 = k * 7.0 * (1.0 - nr) * (1.0 - nr);
  return m + rot2(a) * d;
}

// ---------------------------------------------------------------- black hole
fn blackHole(p: vec2f, m: vec2f) -> vec4f {
  let d: vec2f = p - m;
  let r: f32 = max(length(d), 0.00001);
  let rs: f32 = u.radius * 0.35;            // event horizon radius
  // light passing at distance r is bent by ~ rs^2 / r: everything inside the Einstein ring
  // comes from the OTHER side of the hole (that is why the image looks flipped there)
  let bend: f32 = rs * rs * 1.7 / r;
  var q: vec2f = p - d / r * bend;
  // a spinning hole drags space around with it
  q = m + rot2(u.strength * 2.0 * rs / max(r, rs)) * (q - m);
  return vec4f(q, r, rs);
}

fn uvGrid(q: vec2f) -> f32 {
  let g: vec2f = abs(fract(q * 14.0) - 0.5);
  let w: vec2f = fwidth(q * 14.0);
  let lines: vec2f = vec2f(1.0) - smoothstep(vec2f(0.0), w * 1.2, vec2f(0.5) - g);
  return max(lines.x, lines.y);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex: i32 = i32(u.example);
  let p: vec2f = px / u.resolution.y;
  let m: vec2f = u.mouse.xy / u.resolution.y;
  let t: f32 = u.time;
  var q: vec2f = p;                 // where to read the image from (p units)
  var col: vec3f = vec3f(0.0);
  var extra: vec3f = vec3f(0.0);    // additive light
  var shadeK: f32 = 1.0;

  if (ex == 0) {
    let s: vec3f = shockwaves(p);
    q = p - s.xy;
    let o: vec2f = s.xy * u.chroma;
    col = vec3f(TEX(game, toUV(q - o)).r, TEX(game, toUV(q)).g, TEX(game, toUV(q + o)).b);
    extra = vec3f(0.85, 0.9, 1.0) * s.z * 0.12;
  } else if (ex == 1) {
    let mode: f32 = u.heatMode;
    if (mode < 0.5) {
      q = p + heatOffset(p);
      col = TEX(game, toUV(q)).rgb;
    } else if (mode < 1.5) {
      q = p + underwaterOffset(p);
      col = TEX(game, toUV(q)).rgb;
      col = col * vec3f(0.55, 0.85, 1.05) + vec3f(0.0, 0.04, 0.09);
      let rays: f32 = pow(max(0.0, sin(p.x * 9.0 + p.y * 3.0 + t * 0.5) * sin(p.x * 4.0 - t * 0.3)), 3.0);
      extra = vec3f(0.4, 0.8, 0.9) * rays * (1.0 - p.y) * 0.25;
    } else if (mode < 2.5) {
      // drunk / dizzy: the whole view sways and splits into two drifting copies
      let a: f32 = u.resolution.x / u.resolution.y;
      let c: vec2f = vec2f(a * 0.5, 0.5);
      let sway: f32 = 0.05 * sin(t * u.speed * 0.7) * u.strength;
      q = c + rot2(sway) * (p - c) * (1.0 - 0.03 * u.strength * (0.5 + 0.5 * sin(t * u.speed * 1.1)));
      let o: vec2f = vec2f(sin(t * u.speed * 0.9), 0.5 * cos(t * u.speed * 0.63)) * 0.022 * u.strength * u.scale;
      col = 0.5 * (TEX(game, toUV(q + o)).rgb + TEX(game, toUV(q - o)).rgb);
      let vr: f32 = length(p - c) / (0.5 * a);
      col *= 1.0 - 0.45 * smoothstep(0.4, 1.1, vr) * u.strength;
    } else {
      let rd: vec3f = raindrops(p);
      q = mix(p, rd.xy, rd.z);
      col = TEX(game, toUV(q)).rgb;
      // fogged glass between the drops; drops get a dark rim and a highlight
      let fogged: vec3f = mix(col, vec3f(luma(col)) * 0.9 + vec3f(0.05, 0.06, 0.08), 0.35);
      col = mix(fogged, col * 1.08, rd.z);
      shadeK = 1.0 - 0.35 * rd.z * (1.0 - rd.z) * 4.0;
    }
  } else if (ex == 2) {
    q = lens(p, m, u.lensMode);
    col = TEX(game, toUV(q)).rgb;
    if (u.lensMode < 3.5) {
      // draw the lens rim
      let r: f32 = length(p - m);
      let rim: f32 = abs(r - u.radius);
      let px1: f32 = 1.0 / u.resolution.y;
      if (u.lensMode < 0.5) {
        col = mix(col, vec3f(0.08, 0.06, 0.05), 1.0 - smoothstep(3.0 * px1, 5.0 * px1, rim));
        col = mix(col, vec3f(0.95, 0.8, 0.45), (1.0 - smoothstep(0.5 * px1, 1.8 * px1, abs(r - u.radius - 2.5 * px1))) * 0.8);
        // glass highlight
        let hl: vec2f = (p - m) / u.radius - vec2f(-0.35, -0.4);
        extra = vec3f(1.0) * (1.0 - smoothstep(0.0, 0.35, length(hl * vec2f(1.0, 1.8)))) * 0.18 * step(r, u.radius);
      } else {
        col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.5 * px1, 1.5 * px1, rim)) * 0.35);
      }
    }
  } else {
    let bh: vec4f = blackHole(p, m);
    q = bh.xy;
    let r: f32 = bh.z;
    let rs: f32 = bh.w;
    col = TEX(game, toUV(q)).rgb;
    // gravitational redshift-ish darkening towards the horizon, then the black disc itself
    col *= smoothstep(rs * 0.98, rs * 1.6, r) * 0.6 + 0.4 * smoothstep(rs, rs * 1.04, r);
    col *= smoothstep(rs, rs * 1.03, r);
    // photon ring: light orbiting just outside the horizon
    let pr: f32 = (r - rs * 1.12) / (rs * 0.05);
    extra = vec3f(1.0, 0.82, 0.55) * exp(-pr * pr) * 0.9;
    if (u.disk > 0.5) {
      // accretion disk: hot swirling gas, brighter on the side moving towards us (Doppler beaming)
      let d: vec2f = p - m;
      let a: f32 = atan2(d.y, d.x);
      let swirl: f32 = a + 3.0 / (r / rs) - t * 1.2;
      let gas: f32 = 0.5 + 0.5 * fbm(vec2f(swirl * 1.6, r / rs * 2.5), 3);
      let band: f32 = smoothstep(rs * 1.35, rs * 1.7, r) * smoothstep(rs * 4.2, rs * 2.0, r);
      let doppler: f32 = 0.55 + 0.45 * sin(a + 0.6);
      let hot: vec3f = mix(vec3f(1.0, 0.35, 0.08), vec3f(1.0, 0.85, 0.55), gas);
      extra += hot * band * gas * doppler * 1.3;
    }
  }

  col = col * shadeK + extra;
  if (u.grid > 0.5) { col = mix(col, vec3f(1.0, 0.95, 0.4), uvGrid(toUV(q)) * 0.55); }
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export default shaderScene({
  interaction: 'Click to fire a shockwave.',
  examples: [
    {
      id: 'shockwave',
      label: 'Shockwaves',
      kind: 'In a game',
      note: 'Explosions, landings, boss roars: a ring that pushes the image outward as it expands. Click anywhere — several rings can overlap. The color fringes come from offsetting red and blue slightly differently.',
      params: { amp: 0.035, width: 0.045, speed: 0.75, chroma: 0.35, auto: true },
      hint: 'Click anywhere to fire a shockwave.',
    },
    {
      id: 'heat',
      label: 'Heat & water',
      kind: 'In a game',
      note: 'Scrolling noise used as an offset: rising heat shimmer above the torches, an underwater wobble, a dizzy/drunk double vision, or raindrops on a window, each drop a tiny upside-down lens.',
      params: { heatMode: 'heat', strength: 1, scale: 1, speed: 1 },
      hint: 'Pick a mode on the right. Turn on the UV grid to see the warp.',
    },
    {
      id: 'lens',
      label: 'Lenses & swirl',
      kind: 'Abstract',
      note: 'A lens is just a function from “where I am” to “where I look”. Move the mouse around: magnifier, fisheye bulge, pinch, swirl — or whole-screen barrel / pincushion distortion like a camera lens or CRT.',
      params: { lensMode: 'magnify', radius: 0.2, strength: 0.6 },
      hint: 'Move the mouse: the lens follows it.',
    },
    {
      id: 'blackhole',
      label: 'Black hole',
      kind: 'Abstract',
      note: 'Gravitational lensing: light is bent towards the hole, so the background around it smears into an <b>Einstein ring</b> and the image inside the ring comes from the far side (flipped). Great for a gravity-gun, a boss or a portal.',
      params: { radius: 0.2, strength: 0.4, disk: true },
      hint: 'Move the mouse to drag the black hole around.',
    },
  ],
  controls: [
    { type: 'slider', key: 'amp', label: 'Strength', min: 0, max: 0.12, step: 0.001, value: 0.035, showFor: ['shockwave'], help: 'How far pixels get pushed by the ring.' },
    { type: 'slider', key: 'width', label: 'Ring width', min: 0.01, max: 0.15, step: 0.001, value: 0.045, showFor: ['shockwave'] },
    { type: 'slider', key: 'speed', label: 'Speed', min: 0.1, max: 2, step: 0.01, value: 0.75, showFor: ['shockwave', 'heat'], help: 'Expansion speed / animation speed.' },
    { type: 'slider', key: 'chroma', label: 'Color fringing', min: 0, max: 1.5, step: 0.01, value: 0.35, showFor: ['shockwave'], help: 'Offset red and blue by different amounts (chromatic aberration).' },
    { type: 'toggle', key: 'auto', label: 'Auto-fire when idle', value: true, showFor: ['shockwave'] },
    {
      type: 'select',
      key: 'heatMode',
      label: 'Mode',
      value: 'heat',
      options: [
        { value: 'heat', label: 'Heat haze (torches)' },
        { value: 'underwater', label: 'Underwater wobble' },
        { value: 'drunk', label: 'Dizzy / drunk' },
        { value: 'rain', label: 'Raindrops on glass' },
      ],
      showFor: ['heat'],
    },
    { type: 'slider', key: 'scale', label: 'Noise scale', min: 0.3, max: 3, step: 0.01, value: 1, showFor: ['heat'], help: 'Size of the ripples / drops (bigger number = finer).' },
    {
      type: 'select',
      key: 'lensMode',
      label: 'Lens',
      value: 'magnify',
      options: [
        { value: 'magnify', label: 'Magnifying glass' },
        { value: 'fisheye', label: 'Fisheye bulge' },
        { value: 'pinch', label: 'Pinch' },
        { value: 'swirl', label: 'Swirl / twirl' },
        { value: 'barrel', label: 'Barrel (whole screen)' },
        { value: 'pincushion', label: 'Pincushion (whole screen)' },
      ],
      showFor: ['lens'],
    },
    { type: 'slider', key: 'radius', label: 'Radius', min: 0.05, max: 0.5, step: 0.005, value: 0.2, showFor: ['lens', 'blackhole'], help: 'Size of the lens / black hole (fraction of the screen height).' },
    { type: 'slider', key: 'strength', label: 'Strength', min: 0, max: 2, step: 0.01, value: 1, showFor: ['heat', 'lens', 'blackhole'], help: 'Lens: zoom / bend amount. Black hole: spin (frame dragging).' },
    { type: 'toggle', key: 'disk', label: 'Accretion disk', value: true, showFor: ['blackhole'] },
    { type: 'toggle', key: 'grid', label: 'Show UV grid', value: false, help: 'Draw a grid in the coordinates the image is read from — the warp becomes visible.' },
  ],
  uniforms: {
    amp: 'f32', width: 'f32', speed: 'f32', chroma: 'f32', heatMode: 'f32', scale: 'f32', lensMode: 'f32', radius: 'f32',
    strength: 'f32', disk: 'f32', grid: 'f32', waves: `array<vec4f, ${MAX_WAVES}>`,
  },
  include: withGameIncludes(['math', 'hash', 'noise', 'color']),
  passes: [gamePass()],
  resetOnExample: false,
  bind(p, ctx) {
    ov.begin(ctx);
    const arr = new Float32Array(MAX_WAVES * 4);
    if (ctx.example === 'shockwave') {
      const t = ctx.time;
      const spawn = (x, y, s) => {
        st.waves.push({ x, y, t0: t, s });
        if (st.waves.length > MAX_WAVES) st.waves.shift();
        st.lastSpawn = t;
      };
      if (ctx.pointer.clicked) spawn(ctx.pointer.x, ctx.pointer.y, 1);
      if (p.auto && t - st.lastSpawn > 1.5) {
        const h = ctx.height;
        // aim near the hero or somewhere on the ground
        if (rand() < 0.5) spawn(0.45 * h, 0.74 * h, 0.9);
        else spawn((0.15 + 0.7 * rand()) * ctx.width, (0.35 + 0.45 * rand()) * h, 0.8);
      }
      const life = 1.4 / Math.max(0.05, p.speed);
      st.waves = st.waves.filter((w) => t - w.t0 < life && t >= w.t0);
      st.waves.forEach((w, i) => arr.set([w.x, w.y, w.t0, w.s], i * 4));
      ov.show('n', `${st.waves.length} active wave${st.waves.length === 1 ? '' : 's'} (max ${MAX_WAVES})`, 'right:8px;bottom:8px');
    }
    ov.end();
    return { waves: arr };
  },
  code: IMAGE,
  about: {
    summary:
      'Distortion effects never move pixels — they change where each pixel reads the image from. One offset per pixel gives you shockwaves, heat haze, underwater wobble, lenses, swirls and even black holes.',
    what: `<p>The live game scene, read through different “lenses”. Turn on <b>Show UV grid</b>: the grid is drawn in the coordinates each
      pixel samples from, so you can literally see the space being bent.</p>`,
    how: `<ol>
      <li>Render the game normally into a texture.</li>
      <li>In a full-screen pass, compute an <b>offset</b> for every pixel and read the texture at <code>uv + offset</code>.</li>
      <li><b>Shockwave</b>: the offset points away from the impact and is only non-zero near a ring of radius <i>age × speed</i>.
        Its profile <code>x·e<sup>−x²</sup></code> pushes one way in front of the ring and the other way behind it, like a lens. Each click stores
        <i>(x, y, start time)</i> in a small uniform array — up to 8 rings at once.</li>
      <li><b>Heat haze / underwater</b>: the offset is scrolling noise (or a few sine waves), masked to where the heat is.</li>
      <li><b>Lenses</b>: radial functions of the distance to the mouse — divide it to magnify, raise it to a power for fisheye/pinch, rotate by an angle that grows towards the centre for a swirl.</li>
      <li><b>Black hole</b>: bend each ray by <i>r<sub>s</sub>²/r</i> towards the hole. Inside the Einstein ring the ray comes from the far side, which flips the image; add a black disc, a thin photon ring and a swirling disk.</li>
    </ol>`,
    uses: [
      { title: 'Impacts', text: 'Explosion and landing shockwaves, boss roars, ground pounds — often with a little chromatic aberration.' },
      { title: 'Environment', text: 'Heat over lava and deserts, underwater levels, rain on the camera, fogged glass.' },
      { title: 'Status effects', text: 'Dizzy, drunk, poisoned, confused: sway + double vision.' },
      { title: 'Magic & sci-fi', text: 'Portals, gravity wells, black holes, invisibility “predator” shimmer, teleport warps.' },
      { title: 'UI', text: 'Magnifier loupes on maps, fisheye minimaps, barrel distortion for a CRT or camera look.' },
    ],
    try: [
      'Turn on <b>Show UV grid</b> and click a few shockwaves — watch the grid ripple outward.',
      'On <b>Shockwaves</b> raise <i>Color fringing</i> to 1.5 for a sci-fi energy blast.',
      'On <b>Heat & water</b> pick <i>Raindrops on glass</i>: each drop shows a tiny upside-down world, like real drops.',
      'On <b>Lenses & swirl</b> choose <i>Swirl</i>, set Strength to 2 and Radius to 0.5 — instant vortex.',
      'Park the <b>Black hole</b> over the hero and look inside the bright ring: the image is mirrored.',
    ],
    ask: [
      'a shockwave distortion ring when something explodes',
      'heat haze above fire and lava using scrolling noise',
      'an underwater wobble screen effect',
      'a magnifying-glass lens that follows the mouse',
      'a black hole with gravitational lensing and an accretion disk',
      'raindrops on the camera lens',
    ],
    perf: `<p>Usually one to three texture reads per pixel plus some math — very cheap. The shockwave loop is over at most 8 rings. A
      displacement can also be drawn as sprites into a separate “distortion buffer” (rendered like particles) and applied in one pass:
      that is how engines handle hundreds of heat sources.</p>`,
    api: `<p>Identical in WebGPU and WebGL2. The shockwave list is a uniform array (<code>array&lt;vec4f, 8&gt;</code>) in both.
      With WebGPU you could keep thousands of distortion sources in a storage buffer and bin them into screen tiles with a compute shader.</p>`,
    code: [
      {
        title: 'One shockwave ring (looped over the active rings)',
        lang: 'wgsl',
        src: `let c = w.xy / u.resolution.y;          // ring centre
let age = u.time - w.z;                    // seconds since the click
let R = age * u.speed;                     // current radius
let d = p - c;
let r = length(d);
let x = (r - R) / u.width;                 // signed distance to the ring, in ring widths
let fade = clamp(1.0 - R / 1.4, 0.0, 1.0);
let prof = x * exp(-x * x * 1.6);          // push / pull profile
off += d / r * prof * u.amp * fade;
// ...then read the scene at  p - off
col = TEX(game, toUV(p - off)).rgb;`,
      },
      {
        title: 'Black hole lensing',
        lang: 'wgsl',
        src: `let d = p - m;
let r = length(d);
let bend = rs * rs * 1.7 / r;              // stronger near the hole
var q = p - d / r * bend;                  // look "around" the hole
q = m + rot2(spin * rs / max(r, rs)) * (q - m);  // frame dragging
col = TEX(game, toUV(q)).rgb * smoothstep(rs, rs * 1.03, r);`,
      },
    ],
    links: [{ title: 'Inigo Quilez — domain warping', url: 'https://iquilezles.org/articles/warp/', note: 'the same "offset the lookup" idea applied to noise' }],
  },
});

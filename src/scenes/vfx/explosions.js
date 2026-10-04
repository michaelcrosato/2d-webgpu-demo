// Game Feel: Hits & Explosions — "juice".
// One explosion recipe built from independent layers you can switch off one by one:
// flash frame + light, shockwave ring + screen-space distortion, debris, sparks, fire→smoke
// (premultiplied alpha: the same particle slides from additive to alpha-blended), screen shake
// (trauma), hit-stop (freeze frames), chromatic flash and camera kick / zoom punch.

import { Camera2D, ShapeBatch, SpriteBatch } from '../../core/batch.js';
import { getAtlas } from '../../core/assets.js';
import { makeStrip, createBloom, TONEMAP_WGSL, tag, softSprites, cellUV, TEX, prng, clamp, lerp } from './_shared.js';

const MAX_RINGS = 8;
const MAX_LIGHTS = 4;

const BG_WGSL = /* wgsl */ `
fn lightAt(p: vec2f) -> vec3f {
  var l = vec3f(0.0);
  for (var k = 0; k < ${MAX_LIGHTS}; k++) {
    let L = u.lights[k];
    if (L.z > 0.0) {
      let d = p - L.xy;
      l += vec3f(1.0, 0.62, 0.3) * L.z * L.w * L.w / (dot(d, d) + L.w * L.w);
    }
  }
  return l;
}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  // world position of this pixel (the camera shakes, so the background must too)
  let w = (px - res * 0.5) / u.cam.z + u.cam.xy;
  let p = w / res.y;
  let gy = u.groundY / res.y;
  var alb: vec3f;
  var amb = vec3f(0.05, 0.055, 0.075);
  if (u.mode == 1u) {
    // brick wall filling the screen
    var b = p * vec2f(7.0, 14.0);
    b.x += 0.5 * floor(b.y);
    let f = fract(b);
    let mortar = smoothstep(0.0, 0.06, min(min(f.x, 1.0 - f.x) * 0.5, min(f.y, 1.0 - f.y)));
    let tone = 0.75 + 0.3 * hash21(floor(b)) + 0.12 * valueNoise(p * 60.0);
    alb = mix(vec3f(0.18, 0.17, 0.16), vec3f(0.42, 0.18, 0.12) * tone, mortar);
    amb = vec3f(0.3, 0.27, 0.26) * (1.1 - p.y * 0.6);
  } else if (u.mode == 2u) {
    // dusk desert with mesas
    let sky = mix(vec3f(0.1, 0.06, 0.18), vec3f(0.9, 0.45, 0.25), smoothstep(0.0, gy, p.y));
    let m = gy - 0.12 - 0.12 * step(0.5, fract(p.x * 0.6 + 0.3)) * smoothstep(0.0, 0.03, 0.5 - abs(fract(p.x * 0.6 + 0.3) - 0.75) * 2.0) - 0.02 * sin(p.x * 9.0);
    var c = sky;
    if (p.y > m) { c = vec3f(0.18, 0.09, 0.1); }
    if (p.y > gy) {
      c = mix(vec3f(0.5, 0.3, 0.18), vec3f(0.28, 0.16, 0.1), smoothstep(gy, 1.0, p.y)) * (0.9 + 0.1 * valueNoise(p * vec2f(40.0, 120.0)));
    }
    return vec4f(c + c * lightAt(w) * 0.6, 1.0);
  } else {
    // warehouse: concrete panels and floor
    let pp = fract(p * vec2f(3.0, 3.0));
    let seam = smoothstep(0.0, 0.015, min(min(pp.x, 1.0 - pp.x), min(pp.y, 1.0 - pp.y)));
    alb = vec3f(0.2, 0.21, 0.24) * (0.75 + 0.25 * seam) * (0.85 + 0.15 * valueNoise(p * 30.0));
    if (p.y > gy) {
      alb = vec3f(0.14, 0.14, 0.15) * (0.8 + 0.2 * valueNoise(p * vec2f(20.0, 60.0)));
      let g = abs(fract(p.x * 6.0) - 0.5);
      alb *= 0.85 + 0.15 * smoothstep(0.0, 0.02, g);
    }
  }
  return vec4f(alb * (amb + lightAt(w)), 1.0);
}`;

const COMPOSITE_WGSL = /* wgsl */ `
${TONEMAP_WGSL}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  // shockwave distortion: each ring pushes pixels outward in a thin band (derivative of a gaussian)
  var off = vec2f(0.0);
  for (var k = 0; k < ${MAX_RINGS}; k++) {
    let r = u.rings[k];
    if (r.w > 0.0) {
      let d = px - r.xy;
      let dist = max(length(d), 0.001);
      let th = res.y * 0.025 + r.z * 0.12;
      let x = (dist - r.z) / th;
      off += (d / dist) * r.w * x * exp(-x * x) * th * 0.9;
    }
  }
  let suv = (px - off) / res;
  // chromatic flash: sample R, G and B at slightly different radial offsets
  let ca = (uv - 0.5) * u.chroma * 0.02;
  var c = vec3f(TEX(hdr, suv + ca).r, TEX(hdr, suv).g, TEX(hdr, suv - ca).b);
  c += TEX(bloomTex, suv).rgb * u.bloomAmt;
  c += vec3f(1.0, 0.95, 0.85) * u.flash * 1.5;
  let v = uv - 0.5;
  c *= 1.0 - dot(v, v) * 0.9;
  return vec4f(vfxTonemap(c * u.exposure, px), 1.0);
}`;

const SPRITE_FS = /* wgsl */ `
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  let c = mix(pow(t.rgb, vec3f(2.2)) * i.color.rgb, vec3f(4.0), i.extra.w);
  return vec4f(c, 1.0);
}`;

const ALL_ON = { flash: true, shock: true, distort: true, debris: true, sparks: true, smoke: true, shake: true, hitstop: true, chroma: true, kick: true };
const ALL_OFF = Object.fromEntries(Object.keys(ALL_ON).map((k) => [k, false]));

export default {
  interaction: 'Click to explode. Switch the juice toggles off one at a time and feel the difference.',
  keys: false,
  examples: [
    {
      id: 'juice',
      label: 'Juice toggles',
      kind: 'Comparison',
      note: 'One explosion, ten independent layers. Each toggle removes one: try them one by one. The magic is that none of them is complicated — a white frame, a ring, a shake — but together they make a click feel like an impact.',
      hint: 'Click anywhere to detonate. Toggle the layers on the right.',
      params: { ...ALL_ON, power: 1 },
    },
    {
      id: 'nojuice',
      label: 'Same, no juice',
      kind: 'Comparison',
      note: 'The identical explosion with every layer switched off except a bare fireball. Click a few times, then go back to <b>Juice toggles</b>. Turn layers on here one at a time to find the ones that matter most to you.',
      hint: 'Click to detonate — then compare with “Juice toggles”.',
      params: { ...ALL_OFF, smoke: true, power: 1 },
    },
    {
      id: 'impacts',
      label: 'Bullet impacts on a wall',
      kind: 'In a game',
      note: 'Small hits need small juice: a spark burst that sprays off the surface, chips falling, a dust puff, a bullet-hole decal that stays, a muzzle flash and a tiny camera kick per shot. Hold the mouse button for automatic fire.',
      hint: 'Click or hold to shoot.',
      params: { ...ALL_ON, hitstop: false, power: 0.5 },
    },
    {
      id: 'bigboom',
      label: 'Big boom',
      kind: 'In a game',
      note: 'Everything at full strength: a fireball whose particles slide from additive (fire) to alpha-blended (smoke), two shockwaves, flaming debris with smoke trails, embers, a scorch mark, a lingering smoke column, heavy shake and a 120 ms hit-stop.',
      hint: 'Click on the ground for a big explosion.',
      params: { ...ALL_ON, power: 1.8 },
    },
  ],
  controls: [
    { type: 'slider', key: 'power', label: 'Power', min: 0.2, max: 2.5, step: 0.01, value: 1, help: 'Scales every layer (counts, sizes, shake).' },
    { type: 'heading', label: 'Juice layers' },
    { type: 'toggle', key: 'flash', label: 'Flash frame & light', value: true, help: 'A white frame, white-flashed targets and a light that briefly lights the scene.' },
    { type: 'toggle', key: 'shock', label: 'Shockwave ring', value: true },
    { type: 'toggle', key: 'distort', label: 'Screen distortion', value: true, help: 'The shockwave bends the image (offsets in the final pass).' },
    { type: 'toggle', key: 'debris', label: 'Debris', value: true },
    { type: 'toggle', key: 'sparks', label: 'Sparks', value: true },
    { type: 'toggle', key: 'smoke', label: 'Fire & smoke', value: true },
    { type: 'toggle', key: 'shake', label: 'Screen shake (trauma)', value: true, help: 'shake = trauma². Trauma adds up and decays.' },
    { type: 'toggle', key: 'hitstop', label: 'Hit-stop (freeze frames)', value: true, help: 'Pause the simulation for ~60–120 ms on impact.' },
    { type: 'toggle', key: 'chroma', label: 'Chromatic flash', value: true },
    { type: 'toggle', key: 'kick', label: 'Camera kick & zoom punch', value: true },
    { type: 'button', key: 'reset', label: 'Clear decals' },
  ],
  about: {
    summary: '“Juice” is the stack of small, cheap effects that make an action feel powerful. Here every layer of an explosion can be switched off so you can feel what each one adds.',
    what: `<p>Click to set off explosions (or shoot a wall). Each explosion is a <i>recipe</i>: it emits particles, pushes a shockwave into the post-processing,
      adds camera trauma, freezes time for a few frames and flashes the screen. The toggles remove ingredients.</p>`,
    how: `<ol>
      <li><b>Flash frame</b>: for one or two frames, add white to the whole screen and draw nearby objects pure white. Also spawn a short-lived point light that lights the background.</li>
      <li><b>Shockwave</b>: an expanding ring sprite plus a ring passed to the final full-screen pass, which offsets the lookup position along the radius inside a thin band — the image bends as the wave passes.</li>
      <li><b>Fire → smoke</b>: puff particles use premultiplied alpha: output <code>(rgb·a, a·(1 − additive))</code>. The same particle starts additive (glowing fire) and slides to alpha-blended (dark smoke that hides what’s behind) as it ages.</li>
      <li><b>Debris & sparks</b>: ballistic particles with gravity, drag and ground bounce; sparks are stretched along their velocity; debris spins and trails smoke.</li>
      <li><b>Screen shake (trauma)</b>: each hit adds <i>trauma</i> (0–1) which decays linearly. The camera offset and roll are <code>maxShake × trauma² × noise(t)</code> — squaring makes small hits subtle and big ones violent.</li>
      <li><b>Hit-stop</b>: skip simulation updates for 50–120 ms while still rendering. The impact frame lingers, so the brain registers it.</li>
      <li><b>Chromatic flash & camera kick</b>: split R/G/B radially for a moment; push the camera away from the blast and zoom in a few percent, then spring back.</li>
    </ol>`,
    uses: [
      { title: 'Every action game', text: 'Vlambeer’s “The Art of Screenshake” and Nuclear Throne made this toolbox famous; Celeste, Hades and Dead Cells use all of it.' },
      { title: 'Shooters', text: 'Impact sparks, decals, muzzle flashes and tiny recoil kicks make guns feel good (Enter the Gungeon, Broforce).' },
      { title: 'Fighting & melee', text: 'Hit-stop is the signature of fighting games (Street Fighter) and Smash-style knockback.' },
      { title: 'Restraint', text: 'Accessibility options should let players reduce shake and flashes — design them as separate, tunable layers like here.' },
    ],
    try: [
      'On <b>Juice toggles</b>, click a few times, then turn off <i>Screen shake</i> and <i>Hit-stop</i> together. It suddenly feels weak.',
      'Turn everything off except <i>Hit-stop</i> and <i>Flash frame</i>: two invisible-seeming tricks, big effect.',
      'Compare the <b>Same, no juice</b> tab with <b>Juice toggles</b> — identical code path, different toggles.',
      'On <b>Bullet impacts</b>, hold the mouse and sweep across the wall; then turn off <i>Camera kick</i>.',
      'Set <i>Power</i> to 2.5 on <b>Big boom</b> and click repeatedly: trauma accumulates.',
    ],
    ask: [
      'trauma-based screen shake (trauma squared, decaying)',
      'hit-stop / freeze frames on impact',
      'shockwave ring with screen-space distortion',
      'explosion where fire particles turn into smoke (additive to alpha)',
      'white flash frame and chromatic aberration on big hits',
      'bullet impact sparks, chips, dust puff and decals',
    ],
    perf: `<p>All of this is cheap: a few hundred CPU particles drawn as instanced sprites, one extra full-screen pass for distortion + chromatic aberration,
      and a handful of uniforms. Large smoke puffs are the main fill-rate cost. The bigger risk is <i>design</i>: too much shake and flashing is tiring,
      so give players a slider.</p>`,
    api: `<p>Nothing here requires WebGPU — the same layers work in WebGL2 or Canvas2D. This scene uses the shared WebGPU HDR pipeline (float targets, bloom, a full-screen
      composite) for convenience.</p>`,
    code: [
      {
        title: 'The explosion recipe (JavaScript, simplified)',
        lang: 'js',
        src: `function explode(x, y, power) {
  if (P.hitstop) freeze = 0.05 + 0.04 * power;          // freeze frames
  if (P.flash)   { flash = 1; lights.push({ x, y, I: 8 * power }); }
  if (P.shock)   rings.push({ x, y, t: 0, maxR: H * 0.45 * power });
  if (P.shake)   trauma = Math.min(1, trauma + 0.35 * power);
  if (P.kick)    { kick = awayFrom(x, y) * 0.03 * H * power; zoom += 0.05 * power; }
  if (P.chroma)  chroma = power;
  if (P.sparks)  burst(40 * power, 'spark');
  if (P.smoke)   burst(14 * power, 'puff');               // fire that turns into smoke
  if (P.debris)  burst(12 * power, 'chunk');
}
// every frame:
trauma = Math.max(0, trauma - dt * 1.4);
const s = trauma * trauma;                               // squared: small hits stay subtle
cam.x = baseX + maxShake * s * noise(t * 25, 0);
cam.rotation = maxRoll * s * noise(t * 25, 9);`,
      },
      {
        title: 'Shockwave distortion + chromatic aberration (final pass)',
        lang: 'wgsl',
        src: `for (var k = 0; k < 8; k++) {
  let r = u.rings[k];                        // xy = center (px), z = radius, w = strength
  if (r.w > 0.0) {
    let d = px - r.xy;
    let dist = length(d);
    let x = (dist - r.z) / th;               // position inside the band
    off += (d / dist) * r.w * x * exp(-x * x) * th;
  }
}
let suv = (px - off) / res;
let ca = (uv - 0.5) * u.chroma * 0.02;     // R, G, B sampled at different offsets
var c = vec3f(TEX(hdr, suv + ca).r, TEX(hdr, suv).g, TEX(hdr, suv - ca).b);`,
      },
      {
        title: 'Fire that becomes smoke: one particle, premultiplied alpha',
        lang: 'wgsl',
        src: `// user (extra.w) = 1 - additive amount, animated by the CPU from 0 (fire) to 1 (smoke)
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  let a = t.a * i.color.a;
  return vec4f(t.rgb * i.color.rgb * a, a * (1.0 - i.extra.w));
}
// blend state: src * 1 + dst * (1 - srcAlpha)`,
      },
    ],
    links: [
      { title: 'Jan Willem Nijman — The Art of Screenshake (talk)', url: 'https://www.youtube.com/watch?v=AJdEqssNZ-U' },
      { title: 'Squirrel Eiserloh — Math for Game Programmers: Juicing Your Cameras With Math (GDC)', url: 'https://www.youtube.com/watch?v=tu-Qe66AvtY', note: 'trauma-based shake' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const atlasView = gpu.textureFromImage(atlas.canvas, { label: 'atlas' }).createView();
    const strip = makeStrip(gpu).createView();
    const soft = softSprites(gpu, strip, 8192);
    const sprites = new SpriteBatch(gpu, { texture: atlasView, filter: 'nearest', fragment: SPRITE_FS });
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    const U = gpu.uniforms(
      {
        resolution: 'vec2f',
        time: 'f32',
        mode: 'u32',
        cam: 'vec4f',
        groundY: 'f32',
        flash: 'f32',
        chroma: 'f32',
        bloomAmt: 'f32',
        exposure: 'f32',
        rings: `array<vec4f, ${MAX_RINGS}>`,
        lights: `array<vec4f, ${MAX_LIGHTS}>`,
      },
      'JuiceU',
    );
    const bg = gpu.fullscreen({ label: 'juice-bg', uniforms: U, include: ['noise'], code: BG_WGSL, format: 'rgba16float' });
    const composite = gpu.fullscreen({ label: 'juice-composite', uniforms: U, textures: ['hdr', 'bloomTex'], include: ['color', 'hash'], code: COMPOSITE_WGSL });
    const bloom = createBloom(gpu);
    let hdr = null;
    const ensure = (w, h) => {
      if (hdr && hdr.width === w && hdr.height === h) return;
      hdr?.destroy();
      hdr = gpu.target(w, h, { format: 'rgba16float', label: 'juice-hdr' });
    };
    const readout = tag(ctx);
    const rand = prng(21);
    const noise1 = (t, s) => Math.sin(t * 1.0 + s) * 0.5 + Math.sin(t * 2.3 + s * 1.7) * 0.3 + Math.sin(t * 4.1 + s * 3.1) * 0.2;

    let simT = 0;
    let stopT = 0;
    let trauma = 0;
    let flash = 0;
    let chroma = 0;
    let kick = [0, 0];
    let kickV = [0, 0];
    let zoomK = 0;
    let nextAuto = 0.25;
    let lastClick = -10;
    let fireCD = 0;
    let muzzle = 0;
    const parts = [];
    const debris = [];
    const rings = [];
    const lights = [];
    const decals = [];
    const crates = [];

    const H = () => ctx.height;
    const groundY = () => ctx.height * (ctx.example === 'bigboom' ? 0.8 : 0.82);

    const resetCrates = () => {
      crates.length = 0;
      const W = ctx.width;
      const h = H();
      const s = h * 0.1;
      const xs = [0.18, 0.3, 0.3, 0.72, 0.84, 0.84, 0.84];
      const lv = [0, 0, 1, 0, 0, 1, 2];
      xs.forEach((x, i) => crates.push({ x: W * x, y: groundY() - s * (lv[i] + 0.5), vx: 0, vy: 0, rot: 0, vr: 0, s, flash: 0, home: [W * x, groundY() - s * (lv[i] + 0.5)] }));
    };
    const reset = () => {
      parts.length = 0;
      debris.length = 0;
      rings.length = 0;
      lights.length = 0;
      decals.length = 0;
      simT = 0;
      trauma = 0;
      flash = 0;
      chroma = 0;
      stopT = 0;
      nextAuto = ctx.example === 'bigboom' ? 0.08 : 0.2;
      resetCrates();
    };
    reset();

    const P = () => ctx.params;
    const part = (o) => {
      if (parts.length < 4000) parts.push({ age: 0, rot: rand() * 6.28, spin: (rand() - 0.5) * 2, drag: 1, grav: 0, add0: 1, add1: 1, size1: o.size, bounce: 0, stretch: 0, tex: TEX.soft, ...o });
    };

    // ------------------------------------------------------------------ recipes
    function explode(x, y, power) {
      const p = P();
      const h = H();
      const big = ctx.example === 'bigboom';
      if (p.hitstop) stopT = Math.max(stopT, big ? 0.12 : 0.05 + 0.03 * power);
      if (p.flash) {
        flash = Math.min(1.2, 0.7 + 0.3 * power);
        lights.push({ x, y, I: 4 * power, r: h * 0.3 * Math.sqrt(power), t: 0, life: big ? 0.8 : 0.45 });
      }
      if (p.shock) {
        rings.push({ x, y, t: 0, life: 0.5 + 0.15 * power, maxR: h * 0.42 * power, str: 0.5 + 0.3 * power });
        if (big) rings.push({ x, y, t: -0.12, life: 0.7, maxR: h * 0.75, str: 0.35 });
      }
      if (p.shake) trauma = Math.min(1, trauma + 0.3 * power);
      if (p.kick) {
        const dx = ctx.width / 2 - x;
        const dy = h / 2 - y;
        const d = Math.hypot(dx, dy) || 1;
        kickV[0] += (dx / d) * h * 0.9 * power;
        kickV[1] += (dy / d) * h * 0.9 * power;
        zoomK = Math.min(0.15, zoomK + 0.045 * power);
      }
      if (p.chroma) chroma = Math.min(2, chroma + 0.8 * power);
      // crates: knockback (and white flash if the flash layer is on)
      for (const c of crates) {
        const dx = c.x - x;
        const dy = c.y - y;
        const d = Math.hypot(dx, dy);
        const R = h * 0.45 * power;
        if (d < R) {
          const f = (1 - d / R) * h * 2.2 * power;
          c.vx += (dx / (d || 1)) * f;
          c.vy += (dy / (d || 1)) * f - f * 0.35;
          c.vr += (rand() - 0.5) * 14 * (1 - d / R);
          if (p.flash) c.flash = 0.08;
        }
      }
      if (p.sparks) {
        const n = Math.round(40 * Math.min(power, 1.4));
        for (let i = 0; i < n; i++) {
          const a = -Math.PI * rand();
          const v = h * (0.6 + rand() * 1.6) * Math.sqrt(power);
          part({ type: 'spark', x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0.35 + rand() * 0.6, size: h * 0.012, col: [4, 2.4 + rand(), 0.7, 1], col1: [2, 0.4, 0.05, 1], drag: 2.5, grav: h * 2, tex: TEX.spark, stretch: 0.03, bounce: 0.4 });
        }
        for (let i = 0; i < Math.round(20 * power); i++) {
          const a = rand() * Math.PI * 2;
          const v = h * rand() * 0.4;
          part({ type: 'ember', x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - h * 0.2, life: 1 + rand() * 1.5, size: h * 0.008, col: [3, 1.2, 0.3, 1], col1: [1.5, 0.2, 0.02, 1], drag: 1.5, grav: -h * 0.15 });
        }
      }
      if (p.smoke) {
        // fireball: puffs that start additive (fire) and end alpha-blended (smoke)
        const n = Math.round((big ? 16 : 12) * power);
        for (let i = 0; i < n; i++) {
          const a = rand() * Math.PI * 2;
          const v = h * (0.15 + rand() * 0.55) * Math.sqrt(power);
          const s = h * (0.07 + rand() * 0.07) * Math.sqrt(power);
          part({ type: 'puff', x: x + Math.cos(a) * s * 0.3, y: y + Math.sin(a) * s * 0.3, vx: Math.cos(a) * v, vy: Math.sin(a) * v * 0.7 - h * 0.1, life: 0.9 + rand() * 0.9, size: s, size1: s * 2.4, col: [2.6, 1.15, 0.32, 0.85], col1: [0.045, 0.04, 0.045, 0.85], add0: 1, add1: 0, drag: 3.2, grav: -h * 0.25, tex: TEX.puff });
        }
        // lingering smoke
        for (let i = 0; i < Math.round((big ? 14 : 5) * power); i++) {
          part({ type: 'smoke', x: x + (rand() - 0.5) * h * 0.1, y: y - rand() * h * 0.05, vx: (rand() - 0.5) * h * 0.1, vy: -h * (0.08 + rand() * 0.12), life: 2 + rand() * 2.5, size: h * 0.12, size1: h * 0.3 * power, col: [0.09, 0.085, 0.09, 0.0], col1: [0.06, 0.06, 0.065, 0.6], add0: 0, add1: 0, drag: 0.8, grav: -h * 0.03, tex: TEX.smoke });
        }
        // ground dust ring
        if (Math.abs(y - groundY()) < h * 0.1) {
          for (let i = 0; i < Math.round(12 * power); i++) {
            const dir = rand() < 0.5 ? -1 : 1;
            part({ type: 'dust', x, y: groundY() - h * 0.01, vx: dir * h * (0.4 + rand() * 0.9) * power, vy: -h * rand() * 0.1, life: 0.8 + rand() * 0.8, size: h * 0.05, size1: h * 0.14, col: [0.3, 0.24, 0.2, 0.5], col1: [0.25, 0.21, 0.18, 0], add0: 0, add1: 0, drag: 2.5, grav: 0, tex: TEX.smoke });
          }
        }
      }
      if (p.debris) {
        const n = Math.round((big ? 18 : 10) * power);
        for (let i = 0; i < n; i++) {
          const a = -Math.PI * (0.1 + 0.8 * rand());
          const v = h * (0.6 + rand() * 1.1) * Math.sqrt(power);
          debris.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, rot: rand() * 6, vr: (rand() - 0.5) * 20, s: h * (0.01 + rand() * 0.018), age: 0, life: 2.5 + rand(), hot: big || rand() < 0.4, trailT: 0 });
        }
      }
      if (big) decals.push({ type: 'scorch', x, y: groundY(), s: h * 0.22 * power });
    }

    function impact(x, y) {
      const p = P();
      const h = H();
      const pw = p.power;
      decals.push({ type: 'hole', x, y, s: h * (0.008 + 0.006 * rand()), rot: rand() * 6 });
      if (decals.length > 80) decals.shift();
      if (p.flash) {
        lights.push({ x, y, I: 3 * pw, r: h * 0.1, t: 0, life: 0.12 });
        part({ type: 'flash', x, y, vx: 0, vy: 0, life: 0.06, size: h * 0.09 * pw, col: [6, 5, 3.5, 1], col1: [3, 2, 1, 1], tex: TEX.star });
      }
      if (p.shock) rings.push({ x, y, t: 0, life: 0.18, maxR: h * 0.08 * pw, str: 0.25 });
      if (p.sparks) {
        for (let i = 0; i < Math.round(14 * pw * 2); i++) {
          const a = rand() * Math.PI * 2;
          const v = h * (0.4 + rand() * 1.2);
          part({ type: 'spark', x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - h * 0.3, life: 0.12 + rand() * 0.3, size: h * 0.009, col: [5, 3.5, 1.5, 1], col1: [2, 0.6, 0.1, 1], drag: 3, grav: h * 2.4, tex: TEX.spark, stretch: 0.025 });
        }
      }
      if (p.debris) {
        for (let i = 0; i < Math.round(4 * pw * 2); i++) {
          debris.push({ x, y, vx: (rand() - 0.5) * h * 0.6, vy: -h * rand() * 0.5, rot: rand() * 6, vr: (rand() - 0.5) * 30, s: h * (0.004 + rand() * 0.006), age: 0, life: 1.2, hot: false, trailT: 0, chip: true });
        }
      }
      if (p.smoke) {
        for (let i = 0; i < 3; i++) part({ type: 'smoke', x: x + (rand() - 0.5) * h * 0.02, y, vx: (rand() - 0.5) * h * 0.08, vy: -h * rand() * 0.05, life: 0.7 + rand() * 0.6, size: h * 0.03, size1: h * 0.09, col: [0.5, 0.42, 0.36, 0.7], col1: [0.45, 0.4, 0.36, 0], add0: 0, add1: 0, drag: 3, grav: -h * 0.02, tex: TEX.smoke });
      }
      if (p.shake) trauma = Math.min(1, trauma + 0.06 * pw * 2);
      if (p.kick) {
        // recoil: kick the camera down-right (the gun's side), zoom a hair
        kickV[0] += h * 0.35;
        kickV[1] += h * 0.25;
        zoomK = Math.min(0.06, zoomK + 0.008);
      }
      if (p.chroma) chroma = Math.min(1, chroma + 0.15);
      muzzle = p.flash ? 0.05 : 0;
    }

    // ------------------------------------------------------------------ simulation
    const step = (dt) => {
      const p = P();
      const ex = ctx.example;
      const h = H();
      const W = ctx.width;
      const ptr = ctx.pointer;
      simT += dt;
      const world = cam.screenToWorld(ptr.x, ptr.y);
      if (ex === 'impacts') {
        fireCD -= dt;
        if ((ptr.clicked || ptr.down) && fireCD <= 0) {
          impact(world[0] + (rand() - 0.5) * h * 0.01, world[1] + (rand() - 0.5) * h * 0.01);
          fireCD = 0.085;
          lastClick = simT;
        }
        if (simT - lastClick > 1 && simT > nextAuto) {
          const bx = W * (0.2 + 0.6 * rand());
          const by = h * (0.2 + 0.5 * rand());
          impact(bx, by);
          nextAuto = simT + (rand() < 0.7 ? 0.09 : 0.9);
        }
      } else {
        if (ptr.clicked) {
          lastClick = simT;
          const gy = groundY();
          explode(world[0], ex === 'bigboom' ? Math.min(world[1], gy) : world[1], p.power);
        }
        if (simT - lastClick > 1.2 && simT > nextAuto) {
          const gy = groundY();
          const x = W * (0.3 + 0.4 * rand());
          const y = ex === 'bigboom' ? gy : gy - h * (0.05 + rand() * 0.3);
          explode(x, y, p.power);
          nextAuto = simT + (ex === 'bigboom' ? 3.5 : 1.6);
        }
      }
      const gy = groundY();
      for (let i = parts.length - 1; i >= 0; i--) {
        const q = parts[i];
        q.age += dt;
        if (q.age >= q.life) {
          parts[i] = parts[parts.length - 1];
          parts.pop();
          continue;
        }
        q.vy += q.grav * dt;
        const k = Math.exp(-q.drag * dt);
        q.vx *= k;
        q.vy *= k;
        q.x += q.vx * dt;
        q.y += q.vy * dt;
        q.rot += q.spin * dt;
        if (q.bounce && q.y > gy && q.vy > 0 && ex !== 'impacts') {
          q.y = gy;
          q.vy *= -q.bounce;
          q.vx *= 0.7;
        }
      }
      for (let i = debris.length - 1; i >= 0; i--) {
        const d = debris[i];
        d.age += dt;
        if (d.age > d.life) {
          debris.splice(i, 1);
          continue;
        }
        d.vy += h * 2.2 * dt;
        d.x += d.vx * dt;
        d.y += d.vy * dt;
        d.rot += d.vr * dt;
        const floor = ex === 'impacts' ? h * 2 : gy;
        if (d.y > floor) {
          d.y = floor;
          d.vy *= -0.35;
          d.vx *= 0.6;
          d.vr *= 0.5;
        }
        // flaming debris leaves smoke trails
        if (d.hot && p.smoke && d.age < 1.4) {
          d.trailT -= dt;
          if (d.trailT <= 0) {
            d.trailT = 0.03;
            part({ type: 'trail', x: d.x, y: d.y, vx: 0, vy: -h * 0.03, life: 0.6 + rand() * 0.4, size: h * 0.025, size1: h * 0.07, col: [2.2, 0.9, 0.25, 0.8], col1: [0.06, 0.055, 0.06, 0.5], add0: 1, add1: 0, drag: 1, grav: -h * 0.05, tex: TEX.puff });
          }
        }
      }
      for (const c of crates) {
        c.vy += h * 2.5 * dt;
        c.x += c.vx * dt;
        c.y += c.vy * dt;
        c.rot += c.vr * dt;
        const floor = gy - c.s / 2;
        if (c.y > floor) {
          c.y = floor;
          c.vy *= -0.25;
          c.vx *= 0.5;
          c.vr *= 0.4;
          c.rot *= 0.8;
        }
        c.vx *= Math.exp(-dt * 1.5);
        c.flash = Math.max(0, c.flash - dt);
        // drift slowly back home so the arena never empties
        c.x += (c.home[0] - c.x) * dt * 0.3;
        if (c.y >= floor - 0.5) c.y += (Math.min(c.home[1], floor) - c.y) * dt * 0.5;
        c.rot *= Math.exp(-dt * 0.8);
      }
      for (let i = rings.length - 1; i >= 0; i--) if ((rings[i].t += dt) > rings[i].life) rings.splice(i, 1);
      for (let i = lights.length - 1; i >= 0; i--) if ((lights[i].t += dt) > lights[i].life) lights.splice(i, 1);
    };

    return {
      onAction(key) {
        if (key === 'reset') reset();
      },
      onExample() {
        reset();
      },
      frame(ctx) {
        const p = ctx.params;
        const enc = ctx.encoder;
        const W = ctx.width;
        const h = ctx.height;
        ensure(W, h);
        if (crates.length && Math.abs(crates[0].home[0] - W * 0.18) > 2) resetCrates();
        const ex = ctx.example;
        const mode = { juice: 0, nojuice: 0, impacts: 1, bigboom: 2 }[ex] ?? 0;
        let dt = ctx.paused ? 0 : Math.min(ctx.dt, 0.05);
        // hit-stop: the world freezes, but we keep rendering (and the flash keeps fading)
        const frozen = stopT > 0;
        if (frozen) {
          stopT -= ctx.dt;
          dt = 0;
        }
        if (dt > 0) step(dt);
        const rdt = ctx.paused ? 0 : Math.min(ctx.dt, 0.05);
        if (!frozen) {
          trauma = Math.max(0, trauma - rdt * 1.4);
          chroma = Math.max(0, chroma - rdt * 4);
          zoomK *= Math.exp(-rdt * 8);
          // camera kick: a damped spring back to rest
          kickV[0] += (-kick[0] * 120 - kickV[0] * 14) * rdt;
          kickV[1] += (-kick[1] * 120 - kickV[1] * 14) * rdt;
          kick[0] += kickV[0] * rdt;
          kick[1] += kickV[1] * rdt;
        }
        flash = Math.max(0, flash - rdt * 9);
        muzzle = Math.max(0, muzzle - rdt);

        const s = trauma * trauma;
        const t = ctx.time * 25;
        cam.setViewport(W, h);
        cam.x = W / 2 + h * 0.05 * s * noise1(t, 0) + kick[0];
        cam.y = h / 2 + h * 0.05 * s * noise1(t, 9) + kick[1];
        cam.rotation = 0.05 * s * noise1(t, 17);
        cam.zoom = 1 + zoomK;

        const ringArr = new Float32Array(MAX_RINGS * 4);
        rings.slice(-MAX_RINGS).forEach((r, i) => {
          if (r.t < 0) return;
          const k = r.t / r.life;
          const [sx, sy] = cam.worldToScreen(r.x, r.y);
          ringArr.set([sx, sy, r.maxR * (1 - Math.pow(1 - k, 2.5)), p.distort ? r.str * (1 - k) : 0], i * 4);
        });
        const lightArr = new Float32Array(MAX_LIGHTS * 4);
        lights.slice(-MAX_LIGHTS).forEach((L, i) => {
          const k = L.t / L.life;
          lightArr.set([L.x, L.y, L.I * Math.pow(1 - k, 2), L.r], i * 4);
        });
        if (ex === 'impacts' && muzzle > 0) lightArr.set([W * 0.95, h * 0.95, 6, h * 0.4], 0);
        U.set('resolution', [W, h])
          .set('time', ctx.time)
          .set('mode', mode)
          .set('cam', [cam.x, cam.y, cam.zoom, 0])
          .set('groundY', groundY())
          .set('flash', p.flash ? flash * 0.5 : 0)
          .set('chroma', chroma)
          .set('bloomAmt', 0.45)
          .set('exposure', 1)
          .set('rings', ringArr)
          .set('lights', lightArr);
        bg.draw(enc, hdr, {}, { clear: [0, 0, 0, 1] });

        shapes.begin();
        sprites.begin();
        soft.begin();
        // decals
        for (const d of decals) {
          if (d.type === 'scorch') {
            shapes.box(d.x, d.y + h * 0.005, d.s, d.s * 0.12, [0.02, 0.015, 0.012, 0.85], { radius: d.s * 0.12 });
            shapes.box(d.x, d.y + h * 0.005, d.s * 0.6, d.s * 0.07, [0.01, 0.008, 0.006, 0.9], { radius: d.s * 0.07 });
          } else {
            shapes.circle(d.x, d.y, d.s * 2.2, [0.05, 0.04, 0.035, 0.5]);
            shapes.circle(d.x, d.y, d.s, [0.008, 0.006, 0.006, 1]);
            for (let k = 0; k < 4; k++) {
              const a = d.rot + k * 1.6;
              shapes.line(d.x, d.y, d.x + Math.cos(a) * d.s * 3, d.y + Math.sin(a) * d.s * 3, d.s * 0.35, [0.02, 0.018, 0.016, 0.8]);
            }
          }
        }
        // crates
        if (ex !== 'impacts') {
          for (const c of crates) sprites.draw(c.x, c.y, c.s, c.s, { uv: atlas.uv('crate'), rotation: c.rot, user: c.flash > 0 ? 1 : 0 });
        }
        // debris
        for (const d of debris) {
          const hot = d.hot ? Math.max(0, 1 - d.age / 1.2) : 0;
          const col = d.chip ? [0.32, 0.14, 0.09, 1] : [0.05 + hot * 3, 0.04 + hot * 1.1, 0.035 + hot * 0.25, 1];
          shapes.box(d.x, d.y, d.s, d.s * 0.65, col, { rotation: d.rot, radius: d.s * 0.15 });
        }
        // gun (impacts)
        if (ex === 'impacts') {
          const [gx, gy2] = [W * 0.96, h * 1.02];
          const aim = Math.atan2(ctx.pointer.y - gy2, ctx.pointer.x - gx);
          const len = h * 0.3;
          shapes.line(gx, gy2, gx + Math.cos(aim) * len, gy2 + Math.sin(aim) * len, h * 0.045, [0.03, 0.03, 0.035, 1]);
          if (muzzle > 0) soft.draw(gx + Math.cos(aim) * len * 1.12, gy2 + Math.sin(aim) * len * 1.12, h * 0.16, h * 0.16, { uv: cellUV(TEX.star), rotation: rand() * 6, color: [8, 5, 2, 1], user: 1 });
          // crosshair
          const [cx, cy] = cam.screenToWorld(ctx.pointer.x, ctx.pointer.y);
          shapes.circle(cx, cy, h * 0.022, [1.5, 1.5, 1.5, 0.8], { stroke: 2 });
          shapes.circle(cx, cy, h * 0.003, [2, 2, 2, 1]);
        }
        // particles: alpha-ish first, then additive on top
        const drawPart = (q) => {
          const k = q.age / q.life;
          const c0 = q.col;
          const c1 = q.col1 || q.col;
          const kc = q.type === 'puff' || q.type === 'trail' ? clamp(k * 2.2, 0, 1) : k;
          const col = [lerp(c0[0], c1[0], kc), lerp(c0[1], c1[1], kc), lerp(c0[2], c1[2], kc), lerp(c0[3], c1[3], kc)];
          const add = lerp(q.add0, q.add1, clamp(k * 2.4, 0, 1));
          if (q.type === 'spark') {
            const sp = Math.hypot(q.vx, q.vy);
            soft.draw(q.x, q.y, q.size + sp * q.stretch, q.size * 0.5, { uv: cellUV(TEX.spark), rotation: Math.atan2(q.vy, q.vx), color: [col[0], col[1], col[2], 1 - k * k], user: 1 });
          } else {
            const sz = lerp(q.size, q.size1, Math.sqrt(k));
            const fadeIn = q.type === 'smoke' ? clamp(k * 6, 0, 1) : 1;
            const a = col[3] * fadeIn * (q.type === 'puff' || q.type === 'trail' ? 1 - Math.pow(k, 3) : 1);
            soft.draw(q.x, q.y, sz, sz, { uv: cellUV(q.tex), rotation: q.rot, color: [col[0], col[1], col[2], a], user: add });
          }
        };
        for (const q of parts) if (q.add0 < 0.5 && q.add1 < 0.5) drawPart(q);
        for (const q of parts) if (q.type === 'puff' || q.type === 'trail') drawPart(q);
        for (const q of parts) if (q.add0 >= 0.5 && q.type !== 'puff' && q.type !== 'trail') drawPart(q);
        // shockwave ring sprites
        if (p.shock) {
          for (const r of rings) {
            if (r.t < 0) continue;
            const k = r.t / r.life;
            const R = r.maxR * (1 - Math.pow(1 - k, 2.5));
            const a = Math.pow(1 - k, 1.5);
            soft.draw(r.x, r.y, R * 2.5, R * 2.5, { uv: cellUV(TEX.ring), color: [2.2 * a, 2.0 * a, 1.8 * a, 1], user: 1 });
          }
        }
        shapes.flush(enc, hdr, cam, {});
        sprites.flush(enc, hdr, cam, {});
        soft.flush(enc, hdr, cam, { blend: 'premultiplied' });

        const bl = bloom.render(enc, hdr, { threshold: 1.0, knee: 0.6 });
        composite.draw(enc, { view: ctx.target, format: gpu.format }, { hdr, bloomTex: bl });
        readout.textContent = `trauma ${trauma.toFixed(2)}${frozen ? ' · HIT-STOP' : ''} · ${parts.length + debris.length} particles`;
      },
    };
  },
};

// Sprite Shader Effects — one custom SpriteBatch fragment shader with many per-sprite effects.
// Each sprite carries `user = effectId + amount` (amount in the fraction); the flush passes time & settings.
// Effects: hit flash, outline, dissolve, hologram, freeze, petrify, cloak, x-ray silhouette, drop shadow.

import { ShapeBatch, SpriteBatch, Camera2D } from '../../core/batch.js';
import { atlasTexture, tag, pixelText, wgslLib, clamp } from './_shared.js';

const FX = { NORMAL: 0, FLASH: 1, OUTLINE: 2, DISSOLVE: 3, HOLO: 4, FREEZE: 5, PETRIFY: 6, CLOAK: 7, XRAY: 8, SHADOW: 9 };

const FX_WGSL =
  wgslLib('noise', 'color') +
  /* wgsl */ `
fn texel(uv: vec2f) -> vec4f { return textureSampleLevel(tex, samp, uv, 0.0); }
// 4-neighbour alpha (in atlas texels) — used for outlines, rims and edges
fn nbAlpha(uv: vec2f, ts: vec2f) -> f32 {
  return texel(uv + vec2f(ts.x, 0.0)).a + texel(uv - vec2f(ts.x, 0.0)).a + texel(uv + vec2f(0.0, ts.y)).a + texel(uv - vec2f(0.0, ts.y)).a;
}
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  let fx = i32(floor(i.extra.w + 0.0005));
  let amt = clamp(fract(i.extra.w + 0.0005) / 0.998, 0.0, 1.0);   // per-sprite amount 0..1
  let time = cam.extra.x;
  let ts = 1.0 / vec2f(textureDimensions(tex));                   // one atlas texel in uv
  let tc = floor(i.uv / ts);                                        // integer atlas texel
  let cell = floor(i.local * 18.0);                                 // texel inside the (16+2 px) quad

  if (fx == 1) {                                     // HIT FLASH: blend toward a solid colour
    if (t.a < 0.5) { discard; }
    return vec4f(mix(t.rgb, i.color.rgb, amt), 1.0);
  }
  if (fx == 2) {                                     // OUTLINE: transparent texel next to an opaque one
    if (t.a > 0.5) { return vec4f(t.rgb, 1.0); }
    if (nbAlpha(i.uv, ts) < 0.5) { discard; }
    return vec4f(i.color.rgb, 1.0);
  }
  if (fx == 3) {                                     // DISSOLVE: noise threshold + glowing burn edge
    if (t.a < 0.5) { discard; }
    var q = i.local * 18.0;
    if (cam.extra.w > 0.5) { q = cell + 0.5; }       // pixel-art: one noise value per texel
    let n = valueNoise(q * cam.extra.z) * 0.65 + valueNoise(q * cam.extra.z * 2.7 + vec2f(7.0, 3.0)) * 0.35;
    let edge = cam.extra.y;
    let th = mix(-edge - 0.02, 1.02, amt);
    if (n < th) { discard; }
    if (n < th + edge) {
      let k = (n - th) / max(edge, 0.0001);
      return vec4f(mix(vec3f(1.0, 1.0, 0.9), i.color.rgb, smoothstep(0.0, 0.6, k)) * 1.3, 1.0);
    }
    return vec4f(t.rgb, 1.0);
  }
  if (fx == 4) {                                     // HOLOGRAM: chromatic split, scanlines, flicker, glitch rows
    let row = floor(i.local.y * 18.0);
    let glitch = step(0.94, hash21(vec2f(row, floor(time * 7.0))));
    let off = vec2f(ts.x * (1.0 + 2.0 * glitch), 0.0);
    let uv = i.uv + vec2f(ts.x * 2.0 * glitch * sign(sin(time * 31.0)), 0.0);
    let cr = texel(uv + off);
    let cg = texel(uv);
    let cb = texel(uv - off);
    let a = max(cr.a, max(cg.a, cb.a));
    if (a < 0.5) { discard; }
    let l = luma(vec3f(cr.r, cg.g, cb.b));
    var c = vec3f(0.25, 0.85, 1.0) * (0.35 + l * 1.3) + vec3f(cr.r * cr.a, 0.0, cb.b * cb.a) * 0.35;
    let scan = 0.55 + 0.45 * step(0.5, fract(i.world.y * 0.33 - time * 1.5));
    let flick = 0.8 + 0.2 * sin(time * 50.0) * step(0.7, hash11(floor(time * 9.0)));
    c *= scan * flick;
    return vec4f(c, a * (0.45 + 0.4 * amt));
  }
  if (fx == 5) {                                     // FREEZE: desaturate, ice tint, rim light, sparkles
    if (t.a < 0.5) { discard; }
    let l = luma(t.rgb);
    var ice = mix(vec3f(0.42, 0.7, 0.98) * (0.55 + 0.8 * l), vec3f(0.92, 0.98, 1.0), smoothstep(0.55, 0.85, l));
    if (texel(i.uv - vec2f(0.0, ts.y)).a < 0.5 || texel(i.uv - vec2f(ts.x, 0.0)).a < 0.5) { ice = vec3f(0.95, 1.0, 1.0); }
    let h = hash21(tc);
    let tw = pow(max(0.0, sin(time * 3.0 + h * 40.0)), 24.0) * step(0.8, h);
    return vec4f(mix(t.rgb, ice, amt) + vec3f(tw * amt), 1.0);
  }
  if (fx == 6) {                                     // PETRIFY: grey stone with speckles and cracks
    if (t.a < 0.5) { discard; }
    let l = luma(t.rgb);
    let speck = hash21(tc) * 0.12 - 0.06;
    let crack = step(0.82, valueNoise(cell * 0.7 + vec2f(3.0, 1.0))) * step(0.5, hash21(tc + vec2f(1.0)));
    let stone = vec3f(0.5 + l * 0.45 + speck) * vec3f(0.98, 0.95, 0.9) * (1.0 - crack * 0.45);
    return vec4f(mix(t.rgb, stone, amt), 1.0);
  }
  if (fx == 7) {                                     // CLOAK: nearly invisible body, shimmering edge
    let nb = nbAlpha(i.uv, ts);
    let edge = t.a > 0.5 && nb < 3.5;
    if (t.a < 0.5) { discard; }
    let sh = valueNoise(i.local * 7.0 + vec2f(time * 1.7, -time * 1.1));
    var col = t.rgb;
    var a = 1.0;
    let cloakCol = vec3f(0.6, 0.9, 1.0);
    if (edge) { col = mix(col, cloakCol * (0.8 + 0.6 * sh), amt); a = mix(1.0, 0.75, amt); }
    else { col = mix(col, cloakCol * sh * 0.6, amt); a = mix(1.0, 0.06 + 0.16 * sh, amt); }
    return vec4f(col, a);
  }
  if (fx == 8) {                                     // X-RAY SILHOUETTE: only where the wall (cam.extra rect) covers it
    if (t.a < 0.5) { discard; }
    let r = cam.extra;
    if (i.world.x < r.x || i.world.x > r.z || i.world.y < r.y || i.world.y > r.w) { discard; }
    let rim = nbAlpha(i.uv, ts) < 3.5;
    let d = fract((i.world.x + i.world.y) * 0.08);
    var a = 0.5 + 0.2 * step(0.5, d);
    if (rim) { a = 0.95; }
    return vec4f(i.color.rgb * select(0.8, 1.2, rim), a);
  }
  if (fx == 9) {                                     // SHADOW: the sprite's shape in a solid colour
    if (t.a < 0.5) { discard; }
    return i.color;
  }
  let c = t * i.color;
  if (c.a < 0.01) { discard; }
  return c;
}`;

const LABELS = ['DROP SHADOW', 'HIT FLASH', 'OUTLINE', 'DISSOLVE', 'HOLOGRAM', 'FREEZE', 'PETRIFY', 'CLOAK', 'X-RAY'];
const hex = (h, a = 1) => {
  const n = parseInt(String(h).slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, a];
};

export default {
  interaction: 'See the hint for each tab.',
  examples: [
    {
      id: 'gallery',
      label: 'Effect gallery',
      kind: 'Abstract',
      note: 'Nine effects, one shader. Every sprite tells the shader which effect it wants (and how strongly) through a single per-sprite number, so all of them still draw in one batch. Turn off <i>Animate</i> and drag <i>Effect amount</i> to scrub them.',
      hint: '',
    },
    {
      id: 'combat',
      label: 'Combat feedback',
      kind: 'In a game',
      note: 'Click a slime to hit it: a one-frame <b>white flash</b>, <b>knockback</b>, a short <b>hit-stop</b>, a damage number and <b>blinking</b> invulnerability. Hover shows a target <b>outline</b>; defeated slimes <b>burn away</b> and re-form; right-click casts a <b>freeze</b> spell. Every bit of it is a per-sprite shader parameter.',
      hint: 'Click a slime · right-click = freeze',
    },
    {
      id: 'dissolve',
      label: 'Dissolve & teleport',
      kind: 'In a game',
      note: 'A noise value per texel is compared with a threshold that sweeps from 0 to 1: texels below it disappear, texels just above it glow — the burning edge. Play it backwards to materialise. Here a knight teleports between pads and a slime burns and re-forms.',
      hint: 'Turn off “Animate” to scrub the dissolve',
    },
    {
      id: 'stealth',
      label: 'Hologram, stealth & x-ray',
      kind: 'In a game',
      note: 'A hologram briefing on the left, a cloaked agent sneaking along the front, and guards patrolling <b>behind the wall</b> — their x-ray silhouettes are drawn on top of the wall, but only where the wall hides them.',
      hint: 'Toggle “X-ray silhouettes” in the panel',
    },
  ],
  controls: [
    { type: 'toggle', key: 'animate', label: 'Animate', value: true, showFor: ['gallery', 'dissolve'], help: 'Off = every effect uses the amount slider below.' },
    { type: 'slider', key: 'amount', label: 'Effect amount', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['gallery', 'dissolve'] },
    { type: 'slider', key: 'edge', label: 'Dissolve edge width', min: 0, max: 0.3, step: 0.005, value: 0.08, showFor: ['gallery', 'dissolve', 'combat'] },
    { type: 'slider', key: 'noiseScale', label: 'Dissolve noise scale', min: 0.05, max: 1, step: 0.01, value: 0.3, showFor: ['gallery', 'dissolve'], help: 'Small = big blobs, large = fine grain.' },
    { type: 'toggle', key: 'pixelNoise', label: 'Pixel-art noise (one value per texel)', value: true, showFor: ['gallery', 'dissolve', 'combat'] },
    { type: 'color', key: 'edgeColor', label: 'Burn edge colour', value: '#ff7a1a', showFor: ['gallery', 'dissolve', 'combat'] },
    { type: 'color', key: 'outlineColor', label: 'Outline colour', value: '#ffe14d', showFor: ['gallery', 'combat'] },
    { type: 'toggle', key: 'hitstop', label: 'Hit-stop (freeze 70 ms on hit)', value: true, showFor: ['combat'] },
    { type: 'toggle', key: 'knockback', label: 'Knockback & blink', value: true, showFor: ['combat'] },
    { type: 'toggle', key: 'xray', label: 'X-ray silhouettes', value: true, showFor: ['stealth', 'gallery'] },
    { type: 'color', key: 'xrayColor', label: 'Silhouette colour', value: '#5ee7ff', showFor: ['stealth', 'gallery'] },
  ],
  about: {
    summary: 'A custom fragment shader on your sprites unlocks flashes, outlines, dissolves, holograms, status effects and x-ray silhouettes — all without new art.',
    what: `<p>Pixel-art sprites drawn through one custom sprite shader. A per-sprite number picks the effect and its strength; per-batch values carry time,
      colours and the x-ray wall rectangle. The gallery shows them side by side; the other tabs use them the way games do.</p>`,
    how: `<ol>
      <li><b>Hit flash</b>: <code>mix(texel, white, amount)</code> for ~1–3 frames. The classic “I got hit” signal.</li>
      <li><b>Outline</b>: for a transparent texel, look at its 4 neighbours; if any is opaque, output the outline colour. The quad is drawn 1 texel bigger so the outline has room.</li>
      <li><b>Dissolve</b>: <code>n = noise(texel)</code>; <code>discard</code> if <code>n &lt; threshold</code>; if <code>n &lt; threshold + edge</code> output a hot colour. Animate the threshold.</li>
      <li><b>Hologram</b>: sample R, G and B at slightly different offsets (chromatic split), tint cyan, multiply by scanlines from the world y, flicker, and shift random rows (glitch).</li>
      <li><b>Freeze / petrify</b>: recolour by luminance (keeps shading) toward ice blue or grey stone; add hashed per-texel sparkles or cracks.</li>
      <li><b>Cloak</b>: make the body nearly transparent but keep a shimmering edge (texels with a transparent neighbour).</li>
      <li><b>X-ray</b>: draw characters, draw the wall, then draw the characters again as a flat colour — but only where they overlap the wall.
        Here the wall is a rectangle passed to the shader; engines usually use the <b>stencil</b> or depth buffer for arbitrary occluders.</li>
    </ol>`,
    uses: [
      { title: 'Combat', text: 'Hit flash + knockback + hit-stop is in nearly every action game (Hollow Knight, Dead Cells, Hades).' },
      { title: 'Death & spawn', text: 'Dissolves for enemy deaths, teleports, summons and loot appearing.' },
      { title: 'Status effects', text: 'Frozen, petrified, poisoned, burning, invisible — recolour instead of drawing new frames.' },
      { title: 'Readability', text: 'Outlines for selection/hover and x-ray silhouettes so the player never loses their character behind scenery.' },
    ],
    try: [
      'In <b>Combat</b>, turn off <i>Hit-stop</i> and <i>Knockback</i> and hit a slime: the same damage feels weak.',
      'In <b>Dissolve</b>, turn off <i>Pixel-art noise</i>: smooth noise on a pixel sprite looks out of place.',
      'Set <i>Dissolve edge width</i> to 0 — no burning edge, just holes. Then try 0.3.',
      'In <b>Hologram, stealth & x-ray</b>, toggle <i>X-ray silhouettes</i> off: the guards vanish behind the wall.',
      'In the gallery, turn off <i>Animate</i> and drag <i>Effect amount</i> slowly from 0 to 1.',
    ],
    ask: [
      'a white hit flash with knockback and hit-stop',
      'pixel-perfect sprite outline shader for hover/selection',
      'dissolve shader with a glowing burn edge',
      'hologram shader with scanlines and chromatic aberration',
      'frozen / petrified status effect shader',
      'x-ray silhouette when the player is behind a wall',
    ],
    perf: `<p>Each effect is a few texture reads and math per sprite pixel — negligible. Keeping all effects in <b>one shader</b> (switching on a per-sprite id) lets
      every sprite still share one batch; separate shaders per effect would split the draw calls. Outline/cloak need 4 extra texture reads per pixel.</p>`,
    api: `<p>Works the same in WebGL2 (GLSL) and WebGPU (WGSL). The per-sprite id/amount rides along in the instance data; the shader branches on it.
      (This scene is implemented with the WebGPU SpriteBatch’s custom-fragment hook.)</p>`,
    code: [
      {
        title: 'Outline and dissolve (excerpt of the sprite fragment shader)',
        lang: 'wgsl',
        src: `if (fx == 2) {                                   // OUTLINE
  if (t.a > 0.5) { return vec4f(t.rgb, 1.0); }
  if (nbAlpha(i.uv, ts) < 0.5) { discard; }      // no opaque neighbour
  return vec4f(i.color.rgb, 1.0);
}
if (fx == 3) {                                   // DISSOLVE
  if (t.a < 0.5) { discard; }
  let n = valueNoise((cell + 0.5) * cam.extra.z);  // one value per texel
  let th = mix(-edge - 0.02, 1.02, amt);
  if (n < th) { discard; }                       // burned away
  if (n < th + edge) { return vec4f(i.color.rgb * 1.3, 1.0); } // glowing rim
  return vec4f(t.rgb, 1.0);
}`,
      },
      {
        title: 'JavaScript: pick the effect per sprite',
        lang: 'js',
        src: `// user = effect id + amount (0..1) in the fraction
sprites.draw(x, y, w, h, { uv, anchor: [0.5, 1], user: FX.FLASH + flash * 0.998, color: '#ffffff' });
sprites.draw(x, y, w, h, { uv, user: FX.DISSOLVE + t * 0.998, color: edgeColor });
sprites.flush(encoder, target, cam, { extra: [time, edgeWidth, noiseScale, pixelNoise] });`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const { atlas, view } = await atlasTexture(gpu);
    const sprites = new SpriteBatch(gpu, { texture: view, filter: 'nearest', fragment: FX_WGSL });
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    const info = tag(ctx, 'right:8px;top:48px;text-align:right;line-height:1.5');
    const canvasOf = () => ({ view: ctx.target, format: gpu.format });
    const AW = atlas.width;
    const AH = atlas.height;

    // draw a sprite 1 texel bigger on every side (room for outlines / rims); (x, y) = feet
    const fx = (name, x, y, s, effect, amount, opts = {}) => {
      const f = atlas.frames[name];
      sprites.draw(x, y + s, (f.w + 2) * s * (opts.sx || 1), (f.h + 2) * s * (opts.sy || 1), {
        uv: [(f.x - 1) / AW, (f.y - 1) / AH, (f.x + f.w + 1) / AW, (f.y + f.h + 1) / AH],
        anchor: [0.5, 1],
        flipX: opts.flipX,
        color: opts.color,
        alpha: opts.alpha,
        user: effect + clamp(amount, 0, 1) * 0.998,
      });
    };
    const extraOf = (p, t) => [t, p.edge, p.noiseScale, p.pixelNoise ? 1 : 0];

    // ---------------------------------------------------------------- gallery
    function gallery(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const t = ctx.time;
      const top = H * 0.1;
      const cw = W / 3;
      const ch = (H - top - H * 0.03) / 3;
      const s = Math.max(2, Math.floor((ch * 0.62) / 18));
      const fs = Math.max(1, Math.round(H / 330));
      const amtOf = (k) => (p.animate ? k : p.amount);
      const tri = (x) => 1 - Math.abs(((x % 2) + 2) % 2 - 1);
      shapes.rect(0, 0, W, H, '#11141f');
      const cells = [];
      for (let i = 0; i < 9; i++) {
        const cx = cw * (i % 3) + cw / 2;
        const cy = top + ch * Math.floor(i / 3);
        cells.push([cx, cy]);
        shapes.rect(cw * (i % 3) + 6, cy + 4, cw - 12, ch - 8, i % 2 ? '#1b2132' : '#1e2537', { radius: 8 });
        shapes.box(cx, cy + ch * 0.8, 7 * s, 1.6 * s, '#00000055', { radius: 2 * s });
      }
      shapes.flush(enc, canvasOf(), cam, { clear: [0.06, 0.07, 0.1, 1] });
      const hero = `hero_run_${Math.floor(t * 8) % 4}`;
      const foot = (i) => cells[i][1] + ch * 0.8;
      // 0 drop shadow: the same sprite in black, offset — drawn first
      fx(hero, cells[0][0] + 3 * s, foot(0) + 2 * s, s, FX.SHADOW, 0, { color: [0, 0, 0, 0.45] });
      fx(hero, cells[0][0], foot(0), s, FX.NORMAL, 0);
      fx(hero, cells[1][0], foot(1), s, FX.FLASH, amtOf(t % 1.2 < 0.12 ? 1 : 0), { color: '#ffffff' });
      fx('slime_0', cells[2][0], foot(2), s, FX.OUTLINE, 1, { color: p.outlineColor });
      fx('hero_idle_0', cells[3][0], foot(3), s, FX.DISSOLVE, amtOf(tri(t * 0.45)), { color: p.edgeColor });
      fx(`hero_idle_${Math.floor(t * 2) % 2}`, cells[4][0], foot(4) - 3 * s, s, FX.HOLO, amtOf(0.8));
      fx('slime_0', cells[5][0], foot(5), s, FX.FREEZE, amtOf(0.5 + 0.5 * Math.sin(t)));
      fx('hero_idle_0', cells[6][0], foot(6), s, FX.PETRIFY, amtOf(tri(t * 0.3)));
      fx(hero, cells[7][0] + Math.sin(t) * cw * 0.15, foot(7), s, FX.CLOAK, amtOf(0.85), { flipX: Math.cos(t) < 0 });
      // 8 x-ray: knight walks behind a brick wall
      const wx = cells[8][0] + Math.sin(t * 0.8) * cw * 0.22;
      fx(hero, wx, foot(8), s, FX.NORMAL, 0, { flipX: Math.cos(t * 0.8) < 0 });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      const wall = [cells[8][0] - cw * 0.12, cells[8][1] + ch * 0.18, cells[8][0] + cw * 0.12, foot(8) + 2];
      for (let y = wall[1]; y < wall[3] - 1; y += 16 * Math.max(1, s / 2)) {
        const ts = 16 * Math.max(1, s / 2);
        for (let x = wall[0]; x < wall[2] - 1; x += ts) sprites.draw(x, y, Math.min(ts, wall[2] - x), Math.min(ts, wall[3] - y), { uv: atlas.uv('tile_brick'), anchor: [0, 0] });
      }
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      if (p.xray) {
        fx(hero, wx, foot(8), s, FX.XRAY, 1, { color: p.xrayColor, flipX: Math.cos(t * 0.8) < 0 });
        sprites.flush(enc, canvasOf(), cam, { extra: wall });
      }
      for (let i = 0; i < 9; i++) pixelText(shapes, LABELS[i], cells[i][0], cells[i][1] + 10 * fs, fs * 2, '#cbd5e1', { align: 'center' });
      shapes.flush(enc, canvasOf(), cam);
      info.innerHTML = '';
    }

    // ---------------------------------------------------------------- combat
    const slimes = [];
    const nums = [];
    const sparks = [];
    let hitstop = 0;
    let freeze = 0;
    let lastInput = -99;
    let autoT = 1;
    let heroLunge = 0;
    let heroFace = 1;
    let slash = null;
    let clock = 0;
    const spawnSlime = (i, W, H) => ({ i, x: W * (0.5 + 0.13 * i), y: H * (0.58 + 0.12 * ((i + 1) % 3)), bx: 0, vx: 0, hp: 40, flash: 0, blink: 0, state: 'alive', st: 0, hop: Math.random() * 6 });
    const hit = (sl, W) => {
      const crit = Math.random() < 0.25;
      const dmg = Math.round((crit ? 2 : 1) * (8 + Math.random() * 7));
      sl.hp -= dmg;
      sl.flash = 1;
      const p = ctx.params;
      if (p.knockback) {
        sl.vx = 520 * (H0.k / 5) * Math.sign(sl.x - W * 0.22);
        sl.blink = 0.6;
      }
      if (p.hitstop) hitstop = 0.07;
      nums.push({ x: sl.x + sl.bx, y: sl.y - 18 * H0.k, v: dmg, crit, t: 0 });
      for (let k = 0; k < 10; k++) {
        const a = -Math.PI / 2 + (Math.random() - 0.5) * 2.4;
        sparks.push({ x: sl.x + sl.bx, y: sl.y - 8 * H0.k, vx: Math.cos(a) * 300, vy: Math.sin(a) * 300, t: 0 });
      }
      heroLunge = 0.18;
      slash = { x: sl.x + sl.bx, y: sl.y - 8 * H0.k, t: 0 };
      if (sl.hp <= 0) {
        sl.state = 'dying';
        sl.st = 0;
      }
    };
    const H0 = { k: 5 };

    function combat(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const k = Math.max(2, Math.floor(H / 64));
      H0.k = k;
      let dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 20);
      if (slimes.length === 0) for (let i = 0; i < 3; i++) slimes.push(spawnSlime(i, W, H));
      const ptr = ctx.pointer;
      // hover / click
      let hover = null;
      for (const sl of slimes) {
        if (sl.state !== 'alive') continue;
        if (Math.abs(ptr.x - (sl.x + sl.bx)) < 8 * k && ptr.y < sl.y + k && ptr.y > sl.y - 14 * k) hover = sl;
      }
      if (ptr.clicked) {
        lastInput = clock;
        if (ptr.button === 2) freeze = 2.5;
        else if (hover) hit(hover, W);
      }
      // autopilot: attack a random slime every ~1.1 s, freeze now and then
      if (dt > 0 && clock - lastInput > 3) {
        autoT -= dt;
        if (autoT < 0) {
          const alive = slimes.filter((s) => s.state === 'alive');
          if (alive.length) hit(alive[Math.floor(Math.random() * alive.length)], W);
          autoT = 1.0 + Math.random() * 0.4;
          if (Math.random() < 0.12 && freeze <= 0) freeze = 2.5;
        }
      }
      // hit-stop: freeze the simulation for a few frames (rendering continues)
      if (hitstop > 0) {
        hitstop -= dt;
        dt = 0;
      }
      clock += ctx.paused ? 0 : Math.min(ctx.dt, 1 / 20);
      freeze = Math.max(0, freeze - dt);
      heroLunge = Math.max(0, heroLunge - dt);
      for (const sl of slimes) {
        sl.flash = Math.max(0, sl.flash - dt / 0.1);
        sl.blink = Math.max(0, sl.blink - dt);
        sl.bx += sl.vx * dt;
        sl.vx *= Math.exp(-10 * dt);
        sl.bx *= Math.exp(-2.5 * dt);
        if (freeze <= 0) sl.hop += dt * 4;
        sl.st += dt;
        if (sl.state === 'dying' && sl.st > 0.9) {
          sl.state = 'dead';
          sl.st = 0;
        }
        if (sl.state === 'dead' && sl.st > 1.2) {
          Object.assign(sl, spawnSlime(sl.i, W, H), { state: 'spawning', st: 0 });
        }
        if (sl.state === 'spawning' && sl.st > 0.8) sl.state = 'alive';
      }
      cam.setViewport(W, H).reset();
      // arena
      const ts = 16 * Math.max(1, Math.round(k / 2));
      for (let y = Math.floor(H * 0.42 / ts) * ts; y < H; y += ts) for (let x = 0; x < W; x += ts) sprites.draw(x, y, ts, ts, { uv: atlas.uv('tile_stone'), anchor: [0, 0], color: [0.75, 0.75, 0.85, 1] });
      for (let x = 0; x < W; x += ts) sprites.draw(x, Math.floor(H * 0.42 / ts) * ts - ts, ts, ts, { uv: atlas.uv('tile_brick'), anchor: [0, 0], color: [0.6, 0.55, 0.65, 1] });
      shapes.rect(0, 0, W, Math.floor(H * 0.42 / ts) * ts - ts, '#1a1626');
      shapes.flush(enc, canvasOf(), cam, { clear: [0.08, 0.07, 0.12, 1] });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, clock) });
      // shadows
      const hx = W * 0.22 + heroLunge * 260 * (k / 5);
      const hy = H * 0.72;
      shapes.box(hx, hy, 6 * k, 1.5 * k, '#00000066', { radius: k * 2 });
      for (const sl of slimes) if (sl.state !== 'dead') shapes.box(sl.x + sl.bx, sl.y, 6 * k, 1.5 * k, '#00000066', { radius: k * 2 });
      shapes.flush(enc, canvasOf(), cam);
      // characters, sorted by y
      const list = [{ y: hy, draw: () => fx(heroLunge > 0 ? 'hero_run_1' : `hero_idle_${Math.floor(clock * 2.5) % 2}`, hx, hy, k, FX.NORMAL, 0, { flipX: heroFace < 0 }) }];
      for (const sl of slimes) {
        if (sl.state === 'dead') continue;
        const x = sl.x + sl.bx;
        const hop = freeze > 0 || sl.state !== 'alive' ? 0 : Math.max(0, Math.sin(sl.hop * Math.PI)) * 3 * k;
        const sq = 1 + 0.12 * Math.sin(sl.hop * Math.PI * 2) * (freeze > 0 ? 0 : 1);
        list.push({
          y: sl.y,
          draw: () => {
            const blinkHide = p.knockback && sl.blink > 0 && Math.floor(sl.blink * 20) % 2 === 0 && sl.flash <= 0;
            if (sl.state === 'dying') fx('slime_0', x, sl.y, k, FX.DISSOLVE, sl.st / 0.9, { color: p.edgeColor });
            else if (sl.state === 'spawning') fx('slime_0', x, sl.y, k, FX.DISSOLVE, 1 - sl.st / 0.8, { color: '#5ee7ff' });
            else if (sl.flash > 0) fx('slime_0', x, sl.y - hop, k, FX.FLASH, 1, { color: '#ffffff', sx: 1 / sq, sy: sq });
            else if (freeze > 0) fx('slime_0', x, sl.y, k, FX.FREEZE, Math.min(1, freeze * 3, (2.5 - freeze) * 6));
            else if (!blinkHide) fx(hover === sl ? 'slime_1' : 'slime_0', x, sl.y - hop, k, hover === sl ? FX.OUTLINE : FX.NORMAL, 1, { color: hover === sl ? p.outlineColor : undefined, sx: 1 / sq, sy: sq });
          },
        });
      }
      list.sort((a, b) => a.y - b.y).forEach((o) => o.draw());
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, clock) });
      // hp bars, slash, sparks, numbers
      for (const sl of slimes) {
        if (sl.state !== 'alive') continue;
        const x = sl.x + sl.bx;
        shapes.rect(x - 7 * k, sl.y - 19 * k, 14 * k, 2 * k, '#000000aa', { radius: k });
        shapes.rect(x - 7 * k, sl.y - 19 * k, 14 * k * Math.max(0, sl.hp / 40), 2 * k, sl.hp > 15 ? '#4ade80' : '#f87171', { radius: k });
      }
      shapes.flush(enc, canvasOf(), cam);
      if (slash) {
        slash.t += dt + (hitstop > 0 ? 0.004 : 0);
        const a = slash.t / 0.2;
        if (a < 1) {
          const pts = [];
          for (let i = 0; i <= 12; i++) {
            const ang = -2.2 + (i / 12) * 2.4 * Math.min(1, a * 2.2);
            pts.push([slash.x - 4 * k + Math.cos(ang) * 11 * k, slash.y + Math.sin(ang) * 11 * k]);
          }
          shapes.polyline(pts, 2.2 * k * (1 - a), [1, 1, 0.85, 1 - a], { glow: 3 * k, glowStrength: 0.7 });
        } else slash = null;
      }
      for (const q of sparks) {
        q.t += dt;
        q.x += q.vx * dt;
        q.y += q.vy * dt;
        q.vy += 900 * dt;
      }
      for (let i = sparks.length - 1; i >= 0; i--) if (sparks[i].t > 0.35) sparks.splice(i, 1);
      for (const q of sparks) shapes.line(q.x, q.y, q.x - q.vx * 0.025, q.y - q.vy * 0.025, 0.6 * k, [1, 0.9, 0.5, 1 - q.t / 0.35], { glow: k, glowStrength: 0.6 });
      shapes.flush(enc, canvasOf(), cam, { blend: 'additive' });
      for (const n of nums) n.t += ctx.paused ? 0 : Math.min(ctx.dt, 1 / 20);
      for (let i = nums.length - 1; i >= 0; i--) if (nums[i].t > 0.9) nums.splice(i, 1);
      for (const n of nums) {
        const pop = n.t < 0.1 ? 1 + (0.1 - n.t) * 6 : 1;
        const sz = Math.max(1, Math.round((n.crit ? 1.2 : 0.8) * k * pop));
        pixelText(shapes, n.crit ? `${n.v}!` : `${n.v}`, n.x, n.y - n.t * 22 * k, sz, n.crit ? '#fde047' : '#ffffff', { align: 'center', alpha: Math.min(1, (0.9 - n.t) * 4) });
      }
      if (freeze > 0) pixelText(shapes, 'FREEZE!', W / 2, H * 0.14, Math.max(2, Math.round(k * 0.6)), '#93e5ff', { align: 'center' });
      shapes.flush(enc, canvasOf(), cam);
      info.innerHTML = `hit flash · knockback · hit-stop ${hitstop > 0 ? '<b style="color:#fde047">ACTIVE</b>' : ''}<br><span style="color:#94a3b8">${clock - lastInput > 3 ? 'autopilot attacking — click a slime!' : 'right-click = freeze spell'}</span>`;
    }

    // ---------------------------------------------------------------- dissolve & teleport
    function dissolve(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const t = ctx.time;
      const k = Math.max(2, Math.floor(H / 58));
      cam.setViewport(W, H).reset();
      shapes.rect(0, 0, W, H, '#140f22');
      for (let i = 0; i < 18; i++) shapes.rect(0, H * (0.62 + i * 0.03), W, H * 0.03, i % 2 ? '#1f1834' : '#1c1530');
      const padA = [W * 0.2, H * 0.74];
      const padB = [W * 0.52, H * 0.74];
      // teleport cycle: idle 1.2 s, out 0.7 s, in 0.7 s
      const cyc = 2.6;
      const ph = t % (cyc * 2);
      const atB = ph >= cyc;
      const lp = ph % cyc;
      let amt = 0;
      let where = atB ? padB : padA;
      if (lp > 1.2 && lp < 1.9) amt = (lp - 1.2) / 0.7;
      else if (lp >= 1.9) {
        amt = 1 - (lp - 1.9) / 0.7;
        where = atB ? padA : padB;
      }
      if (!p.animate) {
        amt = p.amount;
        where = padA;
      }
      for (const pd of [padA, padB]) {
        shapes.box(pd[0], pd[1], 13 * k, 3.2 * k, '#2a2350', { radius: 3 * k });
        shapes.box(pd[0], pd[1], 11 * k, 2.4 * k, '#5ee7ff55', { radius: 2.4 * k, stroke: k * 0.6 });
      }
      shapes.flush(enc, canvasOf(), cam, { clear: [0.07, 0.05, 0.12, 1] });
      // beam (additive)
      const beam = p.animate && amt > 0 ? Math.sin(Math.min(1, amt) * Math.PI) : 0;
      for (const pd of [padA, padB]) {
        const b = beam * (pd === where ? 1 : 0.4);
        if (b > 0.01) {
          shapes.rect(pd[0] - 6 * k, 0, 12 * k, pd[1], [0.35, 0.9, 1, 0.18 * b], { glow: 10 * k, glowStrength: 0.3 });
          shapes.rect(pd[0] - 1.5 * k, 0, 3 * k, pd[1], [0.8, 1, 1, 0.5 * b]);
        }
        shapes.box(pd[0], pd[1], 10 * k, 2 * k, [0.35, 0.9, 1, 0.25 + 0.5 * b], { radius: 2 * k, glow: 4 * k, glowStrength: 0.5 });
      }
      shapes.flush(enc, canvasOf(), cam, { blend: 'additive' });
      fx(`hero_idle_${Math.floor(t * 2.5) % 2}`, where[0], where[1], k, FX.DISSOLVE, amt, { color: '#5ee7ff' });
      // the burning slime on the right
      const sp = t % 3.2;
      let samt = sp < 1 ? 0 : sp < 2 ? sp - 1 : sp < 2.4 ? 1 : 1 - (sp - 2.4) / 0.8;
      if (!p.animate) samt = p.amount;
      fx('slime_0', W * 0.82, H * 0.74, k, FX.DISSOLVE, samt, { color: p.edgeColor });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      // embers from the burn edge
      if (samt > 0.05 && samt < 0.98) {
        for (let i = 0; i < 10; i++) {
          const e = (t * 0.9 + i * 0.137) % 1;
          const x = W * 0.82 + Math.sin(i * 12.9) * 7 * k + Math.sin(t * 3 + i) * k;
          shapes.circle(x, H * 0.74 - 4 * k - e * 22 * k, 0.6 * k * (1 - e), [...hex(p.edgeColor).slice(0, 3), 1 - e], { glow: k, glowStrength: 0.8 });
        }
      }
      const fs = Math.max(2, Math.round(k * 0.45));
      pixelText(shapes, 'TELEPORT', (padA[0] + padB[0]) / 2, H * 0.2, fs, '#93e5ff', { align: 'center' });
      pixelText(shapes, 'BURN', W * 0.82, H * 0.2, fs, '#fdba74', { align: 'center' });
      pixelText(shapes, `T=${amt.toFixed(2)}`, (padA[0] + padB[0]) / 2, H * 0.2 + 8 * fs, Math.max(1, fs - 1), '#94a3b8', { align: 'center' });
      pixelText(shapes, `T=${samt.toFixed(2)}`, W * 0.82, H * 0.2 + 8 * fs, Math.max(1, fs - 1), '#94a3b8', { align: 'center' });
      shapes.flush(enc, canvasOf(), cam, { blend: 'additive' });
      info.innerHTML = '';
    }

    // ---------------------------------------------------------------- hologram, stealth & x-ray
    function stealth(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const t = ctx.time;
      const k = Math.max(2, Math.floor(H / 70));
      cam.setViewport(W, H).reset();
      const ts = 16 * Math.max(1, Math.round(k / 2));
      for (let y = Math.floor((H * 0.25) / ts) * ts; y < H; y += ts) for (let x = 0; x < W; x += ts) sprites.draw(x, y, ts, ts, { uv: atlas.uv('tile_metal'), anchor: [0, 0], color: [0.45, 0.5, 0.6, 1] });
      shapes.rect(0, 0, W, H, '#0b0f1a');
      shapes.flush(enc, canvasOf(), cam, { clear: [0.04, 0.05, 0.08, 1] });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      // guards patrolling behind the wall
      const wall = [W * 0.38, H * 0.36, W * 0.97, H * 0.6];
      const guards = [0, 1].map((i) => {
        const ph = t * (0.35 + i * 0.12) + i * 2;
        return { x: W * 0.43 + (W * 0.5) * (0.5 + 0.5 * Math.sin(ph)), y: H * (0.47 + i * 0.07), flip: Math.cos(ph) < 0, f: `hero_run_${Math.floor(t * 8 + i) % 4}` };
      });
      for (const g of guards) fx(g.f, g.x, g.y, k, FX.NORMAL, 0, { flipX: g.flip, color: '#ffb4b4' });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      // the wall (drawn over the guards)
      const wt = 16 * Math.max(1, Math.round(k / 2));
      for (let y = wall[1]; y < wall[3] - 0.5; y += wt) for (let x = wall[0]; x < wall[2] - 0.5; x += wt) sprites.draw(x, y, Math.min(wt, wall[2] - x), Math.min(wt, wall[3] - y), { uv: atlas.uv('tile_brick'), anchor: [0, 0], color: [0.55, 0.6, 0.75, 1] });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      shapes.rect(wall[0], wall[1] - 2 * k, wall[2] - wall[0], 2 * k, '#94a3b8');
      shapes.rect(wall[0], wall[3], wall[2] - wall[0], 1.5 * k, '#00000088');
      shapes.flush(enc, canvasOf(), cam);
      // x-ray pass: the same guards as flat silhouettes, clipped to the wall rectangle
      if (p.xray) {
        for (const g of guards) fx(g.f, g.x, g.y, k, FX.XRAY, 1, { flipX: g.flip, color: p.xrayColor });
        sprites.flush(enc, canvasOf(), cam, { extra: wall });
      }
      // hologram projector
      const hx = W * 0.18;
      const hy = H * 0.62;
      shapes.box(hx, hy + 2 * k, 10 * k, 2.5 * k, '#334155', { radius: 2 * k });
      shapes.flush(enc, canvasOf(), cam);
      shapes.triangle(hx - 9 * k, hy - 22 * k, hx + 9 * k, hy - 22 * k, hx, hy + k, [0.3, 0.85, 1, 0.08]);
      shapes.box(hx, hy + 1.5 * k, 8 * k, 1.6 * k, [0.3, 0.85, 1, 0.6], { radius: 1.6 * k, glow: 4 * k, glowStrength: 0.6 });
      shapes.flush(enc, canvasOf(), cam, { blend: 'additive' });
      fx(`hero_idle_${Math.floor(t * 2) % 2}`, hx, hy - 3 * k + Math.sin(t * 2) * k * 0.6, k, FX.HOLO, 0.85, { flipX: Math.sin(t * 0.5) < 0 });
      // cloaked agent in front, briefly de-cloaking
      const ax = W * (0.3 + 0.55 * (0.5 + 0.5 * Math.sin(t * 0.4)));
      const cloak = clamp(0.5 + 2 * Math.sin(t * 0.7), 0, 1);
      fx(`hero_run_${Math.floor(t * 9) % 4}`, ax, H * 0.8, k, FX.CLOAK, cloak, { flipX: Math.cos(t * 0.4) < 0 });
      sprites.flush(enc, canvasOf(), cam, { extra: extraOf(p, t) });
      const fs = Math.max(2, Math.round(k * 0.4));
      pixelText(shapes, 'HOLOGRAM', hx, H * 0.2, fs, '#93e5ff', { align: 'center' });
      pixelText(shapes, p.xray ? 'GUARDS (X-RAY)' : 'GUARDS (HIDDEN)', (wall[0] + wall[2]) / 2, H * 0.2, fs, '#fca5a5', { align: 'center' });
      pixelText(shapes, `CLOAK ${Math.round(cloak * 100)}%`, ax, H * 0.8 - 22 * k, Math.max(1, fs - 1), '#cbd5e1', { align: 'center' });
      shapes.flush(enc, canvasOf(), cam);
      info.innerHTML = '';
    }

    return {
      onExample() {
        slimes.length = 0;
      },
      frame(ctx) {
        shapes.begin(); // once per frame; each flush draws what was added since the previous one
        sprites.begin();
        if (ctx.example === 'gallery') {
          cam.setViewport(ctx.width, ctx.height).reset();
          gallery(ctx);
        } else if (ctx.example === 'combat') combat(ctx);
        else if (ctx.example === 'dissolve') dissolve(ctx);
        else stealth(ctx);
        info.style.display = info.innerHTML ? '' : 'none';
      },
    };
  },
};

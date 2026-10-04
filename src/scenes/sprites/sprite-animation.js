// Sprite Animation & Atlases — flipbooks, the texture atlas they come from, squash & stretch,
// palette swaps in a custom sprite fragment shader, and a playable hero with an animation state machine.

import { ShapeBatch, SpriteBatch, Camera2D } from '../../core/batch.js';
import { atlasTexture, tag, pixelText, clamp, damp, fmt } from './_shared.js';

// Palette swap: the hero is painted with a handful of exact colors. The shader looks for those KEY colors
// and replaces them with the team's colors. Everything else (skin, outline, visor) is left alone.
const KEYS = ['#3b5dc9', '#29366f', '#b13e53', '#94b0c2', '#566c86']; // tunic, tunic shade, plume, armor, armor shade
const TEAMS = [
  { name: 'ORIGINAL', cols: KEYS },
  { name: 'CRIMSON', cols: ['#c8424a', '#6e1f33', '#f4f4f4', '#94b0c2', '#566c86'] },
  { name: 'FOREST', cols: ['#38b764', '#1f5e52', '#ffcd75', '#a5b49a', '#5d6b55'] },
  { name: 'ROYAL', cols: ['#7a3c9e', '#3f1d5a', '#ffcd75', '#f7c64b', '#c28b1f'] },
  { name: 'CUSTOM', cols: null },
];
const hex3 = (h) => {
  const n = parseInt(h.slice(1), 16);
  return `vec3f(${(((n >> 16) & 255) / 255).toFixed(4)}, ${(((n >> 8) & 255) / 255).toFixed(4)}, ${((n & 255) / 255).toFixed(4)})`;
};

const PALETTE_FS = /* wgsl */ `
fn near3(a: vec3f, b: vec3f) -> bool { let d = abs(a - b); return max(d.x, max(d.y, d.z)) < 0.03; }
fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  // user value = team index + flash amount (0..0.95) in the fraction
  let team = i32(floor(i.extra.w + 0.0001));
  let flash = fract(i.extra.w + 0.0001);
  var keys = array<vec3f, 5>(${KEYS.map(hex3).join(', ')});
  var pal = array<vec3f, 20>(${TEAMS.slice(0, 4).flatMap((t) => t.cols.map(hex3)).join(', ')});
  let custom = cam.extra.rgb;
  var c = t.rgb;
  if (team > 0) {
    if (cam.extra.w > 0.5) {
      // the naive way: multiply-tint the whole sprite (skin, outline and armor change too)
      var tint = pal[team * 5];
      if (team == 4) { tint = custom; }
      c = t.rgb * tint * 1.7;
    } else {
      for (var k = 0; k < 5; k++) {
        if (near3(t.rgb, keys[k])) {
          if (team == 4) {
            var cc = custom;
            if (k == 1) { cc = custom * 0.5; }
            if (k == 2) { cc = vec3f(1.0) - custom * 0.6; }
            if (k >= 3) { cc = t.rgb; }
            c = cc;
          } else {
            c = pal[team * 5 + k];
          }
        }
      }
    }
  }
  // damage flash: red tint, then a pure white frame at the peak
  let white = smoothstep(0.6, 0.85, flash);
  let red = smoothstep(0.0, 0.4, flash) * (1.0 - white);
  c = mix(c, vec3f(1.0, 0.25, 0.3) * (0.35 + dot(c, vec3f(0.3, 0.6, 0.1))), red * 0.85);
  c = mix(c, vec3f(1.0), white);
  return vec4f(c * i.color.rgb, i.color.a);
}`;

const SKY_WGSL = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  // sky gradient + two parallax hill bands driven by the camera x (u.camX in world px)
  var c = mix(vec3f(0.35, 0.62, 0.95), vec3f(0.80, 0.90, 0.98), smoothstep(0.0, 0.8, uv.y));
  let s = u.scale;
  let wx = (px.x - u.res.x * 0.5) / s;
  let x1 = (wx + u.camX * 0.25) * 0.012;
  let h1 = u.res.y * (0.52 + 0.06 * sin(x1 * 2.0) + 0.04 * sin(x1 * 5.3 + 1.0));
  if (px.y > h1) { c = mix(c, vec3f(0.55, 0.72, 0.88), 0.85); }
  let x2 = (wx + u.camX * 0.5) * 0.02;
  let h2 = u.res.y * (0.62 + 0.05 * sin(x2 * 1.7 + 2.0) + 0.03 * sin(x2 * 4.1));
  if (px.y > h2) { c = mix(c, vec3f(0.36, 0.62, 0.52), 0.9); }
  let q = floor(px / s);
  let cl = sin(q.x * 0.03 + u.camX * 0.004) * 0.5 + 0.5;
  return vec4f(c + vec3f(0.03) * cl * step(px.y, h1), 1.0);
}`;

const ANIMS = [
  { value: 'hero_run', label: 'Hero — run (4 frames)' },
  { value: 'hero_idle', label: 'Hero — idle (2 frames)' },
  { value: 'coin', label: 'Coin — spin (8 frames)' },
  { value: 'bat', label: 'Bat — flap (2 frames)' },
  { value: 'slime', label: 'Slime — wobble (3 frames)' },
  { value: 'torch', label: 'Torch — flicker (3 frames)' },
  { value: 'tile_water', label: 'Water tile (4 frames)' },
  { value: 'tile_lava', label: 'Lava tile (4 frames)' },
];

// 60×12 tile platformer level. G grass, D dirt, B brick, W wood, ~ water, c coin
const LEVEL = [
  '                                                            ',
  '                                                            ',
  '                                                            ',
  '                         c c c                              ',
  '                        BBBBBBB          c c                ',
  '            c c                         WWWWWW              ',
  '           WWWWW                                     c c c  ',
  '                       c                            BBBBBBB ',
  '    c                 BBB          c c                      ',
  'GGGGGGGGGGGGG    GGGGGGGGGGGGGGGGGGGGGGGG    GGGGGGGGGGGGGGGG',
  'DDDDDDDDDDDDD~~~~DDDDDDDDDDDDDDDDDDDDDDDD~~~~DDDDDDDDDDDDDDDD',
  'DDDDDDDDDDDDD~~~~DDDDDDDDDDDDDDDDDDDDDDDD~~~~DDDDDDDDDDDDDDDD',
];
const TILE = 16;
const LW = LEVEL[0].length;
const LH = LEVEL.length;
const tileAt = (tx, ty) => (ty < 0 || ty >= LH ? ' ' : tx < 0 || tx >= LW ? 'D' : LEVEL[ty][tx]);
const solid = (tx, ty) => 'GDBW'.includes(tileAt(tx, ty));

const STATES = ['IDLE', 'RUN', 'JUMP', 'FALL', 'LAND'];

export default {
  keys: true,
  interaction: 'Interact with the canvas — see the hint for this tab.',
  examples: [
    {
      id: 'flipbook',
      label: 'Flipbook & atlas',
      kind: 'Abstract',
      note: 'An animation is just a list of frames shown one after another. Left: the playing sprite, the filmstrip and onion skin. Right: the <b>texture atlas</b> — every sprite in the game packed into one image; the highlighted rectangle is the frame being shown. Click any sprite in the atlas to play its animation.',
      hint: 'Hover the atlas · click a sprite to play it.',
    },
    {
      id: 'squash',
      label: 'Squash & stretch',
      kind: 'In a game',
      note: 'Same jump, two treatments. Left: linear motion, rigid body. Right: <b>anticipation</b> (crouch before the jump), <b>stretch</b> while moving fast, <b>squash</b> on impact with a springy recovery, a gravity arc and dust. Volume is preserved: scaleX = 1 / scaleY. Toggle each principle to feel what it adds.',
      hint: 'Click to jump now.',
    },
    {
      id: 'palette',
      label: 'Palette swap',
      kind: 'In a game',
      note: 'One sprite, many teams. The fragment shader swaps a few <b>exact key colors</b> (tunic, plume, armor) for each team’s palette — skin, outline and visor stay untouched. Switch the method to “multiply tint” to see why plain tinting looks muddy. The flashing knight shows a damage flash done in the same shader.',
      hint: 'Click a knight to hit it.',
    },
    {
      id: 'play',
      label: 'Playable hero',
      kind: 'In a game',
      note: 'An animation <b>state machine</b>: the game state (on ground? moving? rising? falling?) picks which animation plays — IDLE, RUN, JUMP, FALL, LAND — and with which speed and squash. Run with ←/→ or A/D, jump with ↑/W/Space (hold for higher jumps).',
      hint: '←/→ A/D run · ↑/W/Space jump',
    },
  ],
  controls: [
    { type: 'select', key: 'anim', label: 'Animation', value: 'hero_run', options: ANIMS, showFor: ['flipbook'] },
    { type: 'slider', key: 'fps', label: 'Frames per second', min: 1, max: 30, step: 1, value: 8, showFor: ['flipbook', 'palette', 'play'], help: 'How long each frame stays on screen (1/fps seconds). Pixel-art games typically use 6–12.' },
    {
      type: 'select',
      key: 'loop',
      label: 'Playback',
      value: 'loop',
      showFor: ['flipbook'],
      options: [
        { value: 'loop', label: 'Loop 0 1 2 3 0 1 …' },
        { value: 'pingpong', label: 'Ping-pong 0 1 2 3 2 1 …' },
      ],
    },
    { type: 'toggle', key: 'onion', label: 'Onion skin', value: true, showFor: ['flipbook'], help: 'Ghosts of the previous (red) and next (blue) frame, like in animation software.' },
    { type: 'toggle', key: 'rects', label: 'Show all frame rectangles', value: true, showFor: ['flipbook'] },
    { type: 'slider', key: 'squash', label: 'Squash & stretch amount', min: 0, max: 0.6, step: 0.01, value: 0.35, showFor: ['squash'] },
    { type: 'slider', key: 'antic', label: 'Anticipation time', min: 0, max: 0.4, step: 0.01, value: 0.16, unit: 's', showFor: ['squash'], help: 'The crouch before take-off that tells the player “a jump is coming”.' },
    { type: 'slider', key: 'jumpH', label: 'Jump height', min: 10, max: 48, step: 1, value: 32, showFor: ['squash'], help: 'In sprite pixels.' },
    { type: 'toggle', key: 'useAntic', label: 'Anticipation', value: true, showFor: ['squash'] },
    { type: 'toggle', key: 'useSquash', label: 'Squash & stretch', value: true, showFor: ['squash'] },
    { type: 'toggle', key: 'useArc', label: 'Gravity arc (ease in/out)', value: true, showFor: ['squash'] },
    { type: 'toggle', key: 'useWobble', label: 'Landing wobble (spring)', value: true, showFor: ['squash'] },
    { type: 'toggle', key: 'useDust', label: 'Dust & shadow', value: true, showFor: ['squash'] },
    {
      type: 'select',
      key: 'method',
      label: 'Recolor method',
      value: 'palette',
      showFor: ['palette'],
      options: [
        { value: 'palette', label: 'Palette swap (exact key colors)' },
        { value: 'multiply', label: 'Multiply tint (naive)' },
      ],
    },
    { type: 'color', key: 'teamColor', label: 'Custom team color', value: '#e07a2f', showFor: ['palette'] },
    { type: 'toggle', key: 'damage', label: 'Damage flash loop', value: true, showFor: ['palette'], help: 'Red tint then a pure white frame — the classic “I got hit” feedback.' },
    { type: 'slider', key: 'runSpeed', label: 'Run speed', min: 30, max: 160, step: 1, value: 85, showFor: ['play'], help: 'The run animation speeds up with the run speed so feet don’t slide.' },
    { type: 'toggle', key: 'syncAnim', label: 'Sync animation speed to velocity', value: true, showFor: ['play'], help: 'Off = fixed fps: at high speeds the feet “skate”.' },
    { type: 'toggle', key: 'landSquash', label: 'Land squash & dust', value: true, showFor: ['play'] },
  ],
  about: {
    summary: 'Sprites come to life by swapping small images quickly (flipbooks) cut from one big texture (the atlas), and by bending them with scale, timing and color tricks.',
    what: `<p>Four views of 2D animation: a <b>flipbook</b> player with the atlas it reads from, the <b>squash & stretch</b> principle on a jumping slime,
      <b>palette swapping</b> in a fragment shader, and a playable knight whose animations are chosen by a <b>state machine</b>.</p>`,
    how: `<ol>
      <li><b>Flipbook</b>: <code>frame = floor(time × fps) % frameCount</code>. Each frame is a rectangle in the atlas; the sprite quad’s UVs
        point at it. Changing the frame = changing 4 numbers. Ping-pong plays 0→3→0 to reuse frames.</li>
      <li><b>Texture atlas</b>: all sprites packed into one texture (with 1–2 px padding so filtering doesn’t bleed). One texture means
        everything can be drawn in one batch — see <a href="#/s/sprite-batching">Sprite Batching</a>.</li>
      <li><b>Squash & stretch</b>: scale the sprite around its feet. scaleY &lt; 1 squashes, &gt; 1 stretches, and scaleX = 1/scaleY keeps the area
        (the “volume”) constant. Drive it from velocity and from timers after events (land, jump).</li>
      <li><b>Easing</b>: a jump with gravity is a parabola — fast at take-off, slow at the top (hang time). Linear motion feels robotic.</li>
      <li><b>Palette swap</b>: in the fragment shader, compare the texel with a few key colors and output the team’s color instead.
        Classic hardware (NES, SNES) did this with indexed color; we emulate it per pixel.</li>
      <li><b>State machine</b>: physics decides the state (grounded, vx, vy); the state selects the clip, its fps and extra squash. Transitions
        like RUN→JUMP happen on events (jump pressed) or conditions (vy &gt; 0 → FALL).</li>
    </ol>`,
    uses: [
      { title: 'Characters', text: 'Every pixel-art game: Celeste, Hollow Knight, Shovel Knight, Stardew Valley — flipbooks + state machines.' },
      { title: 'Team colors & variants', text: 'Recolored enemies (the classic “red slime / blue slime”), player 2 colors, skins, rarity tiers — no new art.' },
      { title: 'Game feel', text: 'Squash on landing and stretch on jumps make Super Meat Boy / Celeste-style characters feel elastic and responsive.' },
      { title: 'Environment', text: 'Animated water, lava, torches and coins are just flipbooks — even tiles (see GPU Tilemaps).' },
    ],
    try: [
      'In <b>Flipbook</b>, drop <i>Frames per second</i> to 2 and watch the filmstrip; turn on onion skin to see motion between frames.',
      'Click the coin or the torch in the atlas to play their animations.',
      'In <b>Squash & stretch</b>, turn the toggles off one by one — anticipation and squash matter most.',
      'In <b>Palette swap</b>, switch to “Multiply tint”: the skin and armor get stained too.',
      'In <b>Playable hero</b>, turn off <i>Sync animation speed</i> and raise <i>Run speed</i>: the feet start skating.',
    ],
    ask: [
      'sprite sheet animation with a texture atlas and per-animation fps',
      'an animation state machine for idle/run/jump/fall/land',
      'squash and stretch with anticipation on jumps and landings',
      'palette swap shader for team colors',
      'a damage flash (white frame) when the player is hit',
      'onion-skin preview for my sprite animations',
    ],
    perf: `<p>Practically free. Changing a frame only changes the UV rectangle; palette swapping is a handful of compares per pixel.
      Keeping every frame in <b>one atlas</b> is what matters — it lets the renderer draw hundreds of animated sprites in a single batch.</p>`,
    api: `<p>Works identically on WebGL2 and WebGPU — it’s plain textured quads plus a small fragment shader. (This scene is implemented with the
      WebGPU SpriteBatch helper.)</p>`,
    code: [
      {
        title: 'Flipbook: pick the frame',
        lang: 'js',
        src: `const frames = atlas.anim('hero_run');           // ['hero_run_0', … 'hero_run_3']
let k = Math.floor(time * fps);
k = pingPong ? (k % (2*n - 2) < n ? k % (2*n - 2) : 2*n - 2 - k % (2*n - 2))
             : k % n;
sprites.draw(x, y, 16 * scale, 16 * scale, { uv: atlas.uv(frames[k]), anchor: [0.5, 1] });`,
      },
      {
        title: 'Palette swap in the sprite fragment shader',
        lang: 'wgsl',
        src: `fn sprite_fs(t: vec4f, i: VOut) -> vec4f {
  if (t.a < 0.5) { discard; }
  let team = i32(floor(i.extra.w));          // per-sprite "user" value
  var c = t.rgb;
  for (var k = 0; k < 5; k++) {
    if (near3(t.rgb, keys[k])) {            // tunic, plume, armor…
      c = pal[team * 5 + k];                // …become the team's colors
    }
  }
  return vec4f(c, 1.0);
}`,
      },
      {
        title: 'Squash & stretch (volume preserving)',
        lang: 'js',
        src: `let sy = 1;
if (crouching) sy = 1 - amount * easeInOut(t / anticipation);
else if (inAir) sy = 1 + amount * 0.6 * Math.min(1, Math.abs(vy) / vMax);
else if (landed) sy = 1 - amount * Math.exp(-tLand * 9) * Math.cos(tLand * 22);
const sx = 1 / sy;                    // keep the area constant
sprites.draw(x, groundY, 16 * s * sx, 16 * s * sy, { anchor: [0.5, 1] });`,
      },
    ],
    links: [
      { title: 'The 12 principles of animation', url: 'https://en.wikipedia.org/wiki/Twelve_basic_principles_of_animation', note: 'squash & stretch, anticipation, timing' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const { atlas, view } = await atlasTexture(gpu);
    const shapes = new ShapeBatch(gpu);
    const sprites = new SpriteBatch(gpu, { texture: view, filter: 'nearest', fragment: PALETTE_FS });
    const cam = new Camera2D();
    const SU = gpu.uniforms({ res: 'vec2f', camX: 'f32', scale: 'f32' }, 'Sky');
    const sky = gpu.fullscreen({ label: 'sky', code: SKY_WGSL, uniforms: SU });
    const info = tag(ctx, 'right:8px;top:48px;text-align:right;line-height:1.5');
    const hover = tag(ctx, 'left:0;top:0;display:none;pointer-events:none');
    const canvasOf = () => ({ view: ctx.target, format: gpu.format });
    const css = (v) => v / ctx.dpr;

    // ---------------------------------------------------------------- flipbook state
    let clickedAnim = null;
    const frameIndex = (t, fps, n, ping) => {
      let k = Math.floor(t * fps);
      if (!ping || n < 3) return k % n;
      const m = 2 * n - 2;
      k %= m;
      return k < n ? k : m - k;
    };
    const animFrames = (prefix) => {
      const f = atlas.anim(prefix);
      return f.length ? f : [prefix];
    };

    function flipbook(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const prefix = clickedAnim || p.anim;
      const frames = animFrames(prefix);
      const n = frames.length;
      const k = frameIndex(ctx.time, p.fps, n, p.loop === 'pingpong');
      const leftW = W * 0.42;
      const f0 = atlas.frames[frames[0]];
      const big = Math.max(2, Math.floor(Math.min(leftW * 0.55, H * 0.34) / Math.max(f0.w, f0.h)));
      const cx = leftW / 2 + W * 0.01;
      const by = H * 0.47;

      // atlas panel geometry
      const ax0 = leftW + W * 0.02;
      const aw = W - ax0 - W * 0.03;
      let as = Math.min(aw / atlas.width, (H * 0.8) / atlas.height);
      if (as >= 1) as = Math.floor(as);
      const AW = atlas.width * as;
      const AH = atlas.height * as;
      const ax = ax0 + (aw - AW) / 2;
      const ay = (H - AH) / 2 + H * 0.03;

      cam.setViewport(W, H).reset();
      // backgrounds
      shapes.rect(0, 0, W, H, '#151a26');
      shapes.rect(W * 0.02, H * 0.1, leftW - W * 0.01, H * 0.74, '#1c2233', { radius: 10 * ctx.dpr });
      // checkerboard behind the atlas = transparency
      const ck = 8 * as;
      for (let y = 0; y < atlas.height / 8; y++)
        for (let x = 0; x < atlas.width / 8; x++) shapes.rect(ax + x * ck, ay + y * ck, ck, ck, (x + y) & 1 ? '#262c3c' : '#20263a');
      // ground line + shadow under the big sprite
      shapes.box(cx, by, big * 7, big * 1.4, '#00000055', { radius: big * 1.4 });
      shapes.flush(enc, canvasOf(), cam, { clear: [0.08, 0.1, 0.15, 1] });

      // onion skin, big sprite
      const drawFrame = (name, x, y, s, opts = {}) => {
        const f = atlas.frames[name];
        sprites.draw(x, y, f.w * s, f.h * s, { uv: atlas.uv(name), anchor: [0.5, 1], ...opts });
      };
      if (p.onion && n > 1) {
        drawFrame(frames[(k + n - 1) % n], cx, by, big, { color: [1, 0.35, 0.35, 0.35] });
        drawFrame(frames[(k + 1) % n], cx, by, big, { color: [0.35, 0.6, 1, 0.35] });
      }
      drawFrame(frames[k], cx, by, big);
      // filmstrip
      const stripY = H * 0.58;
      const cell = Math.min((leftW * 0.86) / n, H * 0.17);
      const fs = Math.max(1, Math.floor((cell - 8 * ctx.dpr) / Math.max(f0.w, f0.h)));
      const sx0 = cx - (cell * n) / 2;
      for (let i = 0; i < n; i++) drawFrame(frames[i], sx0 + cell * (i + 0.5), stripY + cell * 0.5 + (f0.h * fs) / 2, fs);
      sprites.flush(enc, canvasOf(), cam, { extra: [1, 1, 1, 0] });
      // the atlas itself (smooth filtering only when it has to be shrunk)
      sprites.draw(ax, ay, AW, AH, { uv: [0, 0, 1, 1], anchor: [0, 0] });
      sprites.flush(enc, canvasOf(), cam, { extra: [1, 1, 1, 0], filter: as < 1 ? 'linear' : 'nearest' });

      // overlays: frame rectangles, highlights, filmstrip boxes
      const lw = Math.max(1, ctx.dpr);
      const fr = (name) => {
        const f = atlas.frames[name];
        return [ax + f.x * as, ay + f.y * as, f.w * as, f.h * as];
      };
      if (p.rects) {
        for (const name of atlas.names()) {
          const [x, y, w, h] = fr(name);
          shapes.rect(x - 0.5, y - 0.5, w + 1, h + 1, '#ffffff30', { stroke: lw });
        }
      }
      for (let i = 0; i < n; i++) {
        const [x, y, w, h] = fr(frames[i]);
        shapes.rect(x - 1, y - 1, w + 2, h + 2, i === k ? '#fcd34d' : '#7dd3fc99', { stroke: (i === k ? 2 : 1) * lw });
      }
      // hover
      const ptr = ctx.pointer;
      const hx = (ptr.x - ax) / as;
      const hy = (ptr.y - ay) / as;
      let hit = null;
      if (hx >= 0 && hy >= 0 && hx < atlas.width && hy < atlas.height) {
        for (const name of atlas.names()) {
          const f = atlas.frames[name];
          if (hx >= f.x - 1 && hy >= f.y - 1 && hx < f.x + f.w + 1 && hy < f.y + f.h + 1) hit = name;
        }
      }
      if (hit) {
        const [x, y, w, h] = fr(hit);
        shapes.rect(x - 2, y - 2, w + 4, h + 4, '#f9a8d4', { stroke: 2 * lw, glow: 6 * lw, glowStrength: 0.5 });
        const f = atlas.frames[hit];
        hover.style.display = '';
        hover.style.left = `${css(x + w) + 8}px`;
        hover.style.top = `${css(y) - 4}px`;
        hover.innerHTML = `<b style="color:#f9a8d4">${hit}</b> · ${f.w}×${f.h}px @ (${f.x}, ${f.y})`;
        if (ptr.x > W * 0.75) hover.style.left = `${css(x) - 8 - hover.offsetWidth}px`;
        if (ptr.clicked) {
          const m = /^(.*)_\d+$/.exec(hit);
          clickedAnim = m ? m[1] : hit;
        }
      } else hover.style.display = 'none';
      // filmstrip boxes + numbers + playhead
      for (let i = 0; i < n; i++) {
        const x = sx0 + cell * i + 2 * ctx.dpr;
        const on = i === k;
        shapes.rect(x, stripY, cell - 4 * ctx.dpr, cell, on ? '#fcd34d' : '#ffffff40', { stroke: (on ? 2 : 1) * lw, radius: 4 * ctx.dpr });
        pixelText(shapes, String(i), x + 5 * ctx.dpr, stripY + 5 * ctx.dpr, Math.max(1, Math.round(1.5 * ctx.dpr)), on ? '#fcd34d' : '#94a3b8');
      }
      const prog = ((ctx.time * p.fps) % n) / n;
      shapes.rect(sx0, stripY + cell + 6 * ctx.dpr, cell * n, 3 * ctx.dpr, '#ffffff22', { radius: 2 });
      shapes.rect(sx0, stripY + cell + 6 * ctx.dpr, cell * n * (p.loop === 'pingpong' ? (k + 0.5) / n : prog), 3 * ctx.dpr, '#fcd34d', { radius: 2 });
      // atlas border + labels
      shapes.rect(ax - 1, ay - 1, AW + 2, AH + 2, '#ffffff55', { stroke: lw });
      const ts = Math.max(1, Math.round(1.5 * ctx.dpr));
      pixelText(shapes, `ATLAS ${atlas.width}X${atlas.height}`, ax, ay - 9 * ts, ts, '#cbd5e1');
      pixelText(shapes, 'FILMSTRIP', sx0, stripY - 9 * ts, ts, '#cbd5e1');
      shapes.flush(enc, canvasOf(), cam);

      const fname = frames[k];
      const f = atlas.frames[fname];
      const uv = atlas.uv(fname);
      info.innerHTML =
        `<b style="color:#fcd34d">${fname}</b> — frame ${k + 1}/${n}${clickedAnim ? ' <span style="color:#f9a8d4">(clicked in atlas)</span>' : ''}<br>` +
        `rect x ${f.x} y ${f.y} · ${f.w}×${f.h} px<br>uv (${uv[0].toFixed(3)}, ${uv[1].toFixed(3)}) → (${uv[2].toFixed(3)}, ${uv[3].toFixed(3)})`;
    }

    // ---------------------------------------------------------------- squash & stretch
    const sq = { t0: 0, kick: false, dust: [] };
    const easeInOut = (t) => t * t * (3 - 2 * t);
    // Returns { y (height, px), sy (scale y), phase } for a slime at cycle time t.
    const jumpState = (t, p, juicy) => {
      const A = juicy && p.useAntic ? p.antic : 0;
      const D = 0.62; // air time
      const L = 0.55; // landing recovery
      const R = 0.45; // rest
      const amount = juicy && p.useSquash ? p.squash : 0;
      let y = 0;
      let sy = 1;
      let phase = 'rest';
      if (t < A) {
        phase = 'anticipation';
        sy = 1 - amount * 0.75 * easeInOut(t / A);
      } else if (t < A + D) {
        phase = 'air';
        const u = (t - A) / D;
        if (juicy && p.useArc) {
          y = 4 * p.jumpH * u * (1 - u); // gravity parabola
          const v = Math.abs(1 - 2 * u); // |velocity| normalised
          sy = 1 + amount * 0.55 * v;
        } else {
          y = p.jumpH * (1 - Math.abs(1 - 2 * u)); // linear up, linear down
          sy = 1 + amount * 0.4;
        }
      } else if (t < A + D + L) {
        phase = 'land';
        const u = t - A - D;
        if (amount > 0) {
          sy = juicy && p.useWobble ? 1 - amount * Math.exp(-u * 9) * Math.cos(u * 22) : 1 - amount * Math.max(0, 1 - u / 0.12);
        }
      }
      return { y, sy, phase, cycle: A + D + L + R, A, D };
    };

    function squash(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const s = (H * 0.5) / 64; // fractional is fine: squash & stretch rescales anyway
      const groundY = H * 0.64;
      const probe = jumpState(0, p, true);
      const cycle = probe.cycle;
      if (ctx.pointer.clicked) sq.t0 = ctx.time;
      const t = (ctx.time - sq.t0) % cycle;
      const pos = [W * 0.27, W * 0.68];
      cam.setViewport(W, H).reset();
      // backdrop: the same parallax sky as the playable level, slowly drifting
      SU.set('res', [W, H]).set('camX', ctx.time * 12).set('scale', s);
      sky.draw(enc, canvasOf());
      const ts = 16 * s;
      for (let x = 0; x < W + ts; x += ts) {
        sprites.draw(x, groundY, ts, ts, { uv: atlas.uv('tile_grass'), anchor: [0, 0] });
        for (let y = groundY + ts; y < H; y += ts) sprites.draw(x, y, ts, ts, { uv: atlas.uv('tile_dirt'), anchor: [0, 0], color: [0.75, 0.75, 0.8, 1] });
      }
      sprites.flush(enc, canvasOf(), cam, { extra: [1, 1, 1, 0] });

      const states = [];
      for (let side = 0; side < 2; side++) {
        const juicy = side === 1;
        const st = jumpState(t, p, juicy);
        states.push(st);
        const x = pos[side];
        // shadow shrinks with height
        if (!juicy || p.useDust) {
          const k = 1 - Math.min(1, st.y / (p.jumpH * 1.4));
          shapes.box(x, groundY + s, 7 * s * (0.5 + 0.5 * k) * (juicy ? 1 / st.sy : 1), 1.6 * s, '#00000066', { radius: 2 * s });
        }
      }
      shapes.flush(enc, canvasOf(), cam);
      for (let side = 0; side < 2; side++) {
        const st = states[side];
        const x = pos[side];
        const sx = 1 / st.sy;
        const frame = st.phase === 'anticipation' && side === 1 ? 'slime_1' : 'slime_0';
        sprites.draw(x, groundY - st.y * s + s, 16 * s * sx, 16 * s * st.sy, { uv: atlas.uv(frame), anchor: [0.5, 1] });
      }
      sprites.flush(enc, canvasOf(), cam, { extra: [1, 1, 1, 0] });

      // dust puffs on landing (juicy only)
      const st = states[1];
      if (p.useDust && st.phase === 'land') {
        const u = t - st.A - st.D;
        for (let i = 0; i < 6; i++) {
          const dir = i < 3 ? -1 : 1;
          const k = (i % 3) / 3;
          const r = (2 + 3 * (1 - u / 0.55)) * s * (1 - k * 0.3);
          const dx = dir * (8 + 26 * Math.sqrt(u) * (1 + k)) * s * 0.5;
          const dy = -(3 + 10 * u * (1 - k)) * s * 0.4;
          shapes.circle(pos[1] + dx, groundY + dy, Math.max(0, r), [0.9, 0.85, 0.75, Math.max(0, 0.75 - u * 1.4)]);
        }
        shapes.flush(enc, canvasOf(), cam);
      }

      // graph of height and scale over one cycle
      const gx = W * 0.06;
      const gw = W * 0.88;
      const gy = H * 0.77;
      const gh = H * 0.1;
      const fsz = Math.max(2, Math.round(1.4 * ctx.dpr));
      shapes.rect(gx, gy - gh * 0.1 - 9 * fsz, gw, gh * 1.25 + 9 * fsz, '#0b0f19cc', { radius: 6 * ctx.dpr });
      const ptsH = [];
      const ptsS = [];
      const ptsL = [];
      const N = 90;
      for (let i = 0; i <= N; i++) {
        const tt = (i / N) * cycle;
        const a = jumpState(tt, p, true);
        const b = jumpState(tt, p, false);
        const X = gx + (i / N) * gw;
        ptsH.push([X, gy + gh - (a.y / p.jumpH) * gh * 0.95]);
        ptsL.push([X, gy + gh - (b.y / p.jumpH) * gh * 0.95]);
        ptsS.push([X, gy + gh * 0.5 - (a.sy - 1) * gh * 1.2]);
      }
      const lw = Math.max(1.5, 1.5 * ctx.dpr);
      shapes.polyline(ptsL, lw, '#94a3b8aa');
      shapes.polyline(ptsH, lw * 1.3, '#fcd34d');
      shapes.polyline(ptsS, lw, '#7dd3fc');
      const px = gx + (t / cycle) * gw;
      shapes.line(px, gy - gh * 0.08, px, gy + gh * 1.12, lw, '#f9a8d4');
      pixelText(shapes, 'HEIGHT', gx + 6 * fsz, gy - 7 * fsz, fsz, '#fcd34d');
      pixelText(shapes, 'SCALE Y', gx + 34 * fsz, gy - 7 * fsz, fsz, '#7dd3fc');
      pixelText(shapes, 'LINEAR', gx + 66 * fsz, gy - 7 * fsz, fsz, '#94a3b8');
      // labels above the slimes
      const ls = Math.max(2, Math.round(s * 0.45));
      pixelText(shapes, 'PLAIN', pos[0], H * 0.2, ls, '#ffffff', { align: 'center' });
      pixelText(shapes, 'JUICY', pos[1], H * 0.2, ls, '#fcd34d', { align: 'center' });
      pixelText(shapes, st.phase.toUpperCase(), pos[1], H * 0.2 + 8 * ls, Math.max(1, Math.round(ls * 0.75)), '#f9a8d4', { align: 'center' });
      shapes.flush(enc, canvasOf(), cam);
      pixelText(shapes, `SY ${st.sy.toFixed(2)}  SX ${(1 / st.sy).toFixed(2)}`, gx + gw - 4 * fsz, gy - 7 * fsz, fsz, '#e2e8f0', { align: 'right' });
      shapes.flush(enc, canvasOf(), cam);
      info.innerHTML = '';
    }

    // ---------------------------------------------------------------- palette swap
    const hits = new Map();
    function palette(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const n = TEAMS.length;
      const s = Math.max(2, Math.min(Math.floor(H / 66), Math.floor((W * 0.9) / n / 22)));
      const gap = W / n;
      const baseY = H * 0.8;
      const frames = atlas.anim('hero_run');
      cam.setViewport(W, H).reset();
      shapes.rect(0, 0, W, H, '#161b29');
      shapes.rect(0, baseY, W, H - baseY, '#202638');
      shapes.flush(enc, canvasOf(), cam, { clear: [0.08, 0.1, 0.15, 1] });
      const ts = 16 * Math.max(1, Math.round(s / 2));
      for (let x = 0; x < W + ts; x += ts) sprites.draw(x, baseY, ts, ts, { uv: atlas.uv('tile_stone'), anchor: [0, 0] });
      const custom = (() => {
        const n2 = parseInt(String(p.teamColor).slice(1), 16);
        return [((n2 >> 16) & 255) / 255, ((n2 >> 8) & 255) / 255, (n2 & 255) / 255];
      })();
      for (let i = 0; i < n; i++) {
        const x = gap * (i + 0.5);
        const k = Math.floor(ctx.time * p.fps + i * 0.7) % frames.length;
        let flash = 0;
        const hitT = hits.get(i);
        if (hitT !== undefined && ctx.time - hitT < 0.45) flash = 1 - (ctx.time - hitT) / 0.45;
        if (p.damage && i === 2) {
          const ph = (ctx.time % 1.6) / 0.45;
          if (ph < 1) flash = Math.max(flash, 1 - ph);
        }
        const kb = flash > 0 ? Math.sin(flash * Math.PI) * 3 * s : 0;
        sprites.draw(x - kb, baseY + 1, 16 * s, 16 * s, { uv: atlas.uv(frames[k]), anchor: [0.5, 1], user: i + Math.min(0.95, flash * 0.95) });
        if (ctx.pointer.clicked && Math.abs(ctx.pointer.x - x) < 8 * s && ctx.pointer.y > baseY - 16 * s && ctx.pointer.y < baseY) hits.set(i, ctx.time);
      }
      sprites.flush(enc, canvasOf(), cam, { extra: [...custom, p.method === 'multiply' ? 1 : 0] });

      // swatches: key colors (top row) -> team colors (bottom row)
      const fsz = Math.max(1, Math.round(H / 220));
      const sw = Math.max(5, Math.min(Math.floor((gap * 0.78) / 5.6), Math.round(H * 0.045)));
      const sg = Math.max(2, Math.round(sw * 0.15));
      const y0 = H * 0.15;
      const nameSize = Math.max(1, Math.min(fsz * 2, Math.floor((gap * 0.9) / 32)));
      for (let i = 0; i < n; i++) {
        const x = gap * (i + 0.5);
        const cols = TEAMS[i].cols || [p.teamColor, '#' + custom.map((v) => Math.round(v * 127).toString(16).padStart(2, '0')).join(''), '#' + custom.map((v) => Math.round((1 - v * 0.6) * 255).toString(16).padStart(2, '0')).join(''), KEYS[3], KEYS[4]];
        pixelText(shapes, TEAMS[i].name, x, y0, nameSize, i === 0 ? '#cbd5e1' : '#fcd34d', { align: 'center' });
        const rowW = 5 * sw + 4 * sg;
        const ys = y0 + 14 * fsz;
        for (let k = 0; k < 5; k++) {
          const xx = x - rowW / 2 + k * (sw + sg);
          shapes.rect(xx, ys, sw, sw, KEYS[k], { radius: 2 });
          shapes.triangle(xx + sw / 2 - sw * 0.2, ys + sw + sg, xx + sw / 2 + sw * 0.2, ys + sw + sg, xx + sw / 2, ys + sw + sg + sw * 0.3, '#ffffff55');
          shapes.rect(xx, ys + sw * 1.45 + sg * 2, sw, sw, cols[k], { radius: 2 });
        }
      }
      shapes.flush(enc, canvasOf(), cam);
      info.innerHTML = '';
    }

    // ---------------------------------------------------------------- playable hero
    const hero = { x: 3 * TILE, y: 9 * TILE, vx: 0, vy: 0, ground: true, state: 'IDLE', stateT: 0, face: 1, coyote: 0, buffer: 0, landT: 9, prev: 'IDLE', why: '' };
    const coins = [];
    LEVEL.forEach((row, ty) => [...row].forEach((ch, tx) => ch === 'c' && coins.push({ x: tx * TILE + 8, y: ty * TILE + 8, taken: -99 })));
    const dust = [];
    let manual = false;
    let auto = { dir: 1, jumpCd: 1 };
    let camX = hero.x;
    let camY = hero.y;
    let score = 0;
    let animClock = 0;
    const setState = (s, why) => {
      if (s === hero.state) return;
      hero.prev = hero.state;
      hero.state = s;
      hero.stateT = 0;
      hero.why = why;
    };

    function step(dt, p, keys, pressed) {
      const left = keys.has('ArrowLeft') || keys.has('a');
      const right = keys.has('ArrowRight') || keys.has('d');
      const jumpHeld = keys.has('ArrowUp') || keys.has('w') || keys.has('Space') || keys.has(' ');
      const jumpPressed = pressed.has('ArrowUp') || pressed.has('w') || pressed.has('Space') || pressed.has(' ');
      if (left || right || jumpPressed) manual = true;
      let ix = (right ? 1 : 0) - (left ? 1 : 0);
      let jp = jumpPressed;
      let jh = jumpHeld;
      if (!manual) {
        // autopilot: run, jump over gaps / walls, hop onto platforms now and then
        const ahead = Math.floor((hero.x + auto.dir * 14) / TILE);
        const feet = Math.floor((hero.y + 2) / TILE);
        auto.jumpCd -= dt;
        if (hero.x > (LW - 3) * TILE) auto.dir = -1;
        if (hero.x < 2 * TILE) auto.dir = 1;
        ix = auto.dir;
        const gapAhead = !solid(ahead, feet);
        const wallAhead = solid(ahead, feet - 1);
        if (hero.ground && (gapAhead || wallAhead || auto.jumpCd < 0)) {
          jp = true;
          auto.jumpCd = 1.2 + Math.random() * 2.5;
        }
        jh = hero.vy < 0;
      }
      const speed = p.runSpeed;
      const acc = hero.ground ? 900 : 600;
      const target = ix * speed;
      if (Math.abs(target - hero.vx) < acc * dt) hero.vx = target;
      else hero.vx += Math.sign(target - hero.vx) * acc * dt;
      if (ix) hero.face = ix;
      hero.buffer = jp ? 0.12 : hero.buffer - dt;
      hero.coyote = hero.ground ? 0.1 : hero.coyote - dt;
      if (hero.buffer > 0 && hero.coyote > 0) {
        hero.vy = -300;
        hero.ground = false;
        hero.coyote = 0;
        hero.buffer = 0;
        setState('JUMP', 'jump pressed');
      }
      if (!jh && hero.vy < -80) hero.vy *= 0.6; // variable jump height
      hero.vy = Math.min(hero.vy + 980 * dt, 520);
      // move X, resolve against tiles
      const hw = 4;
      const hh = 13;
      hero.x += hero.vx * dt;
      for (const yy of [hero.y - 1, hero.y - hh / 2, hero.y - hh + 1]) {
        const ty = Math.floor(yy / TILE);
        if (hero.vx > 0 && solid(Math.floor((hero.x + hw) / TILE), ty)) {
          hero.x = Math.floor((hero.x + hw) / TILE) * TILE - hw - 0.01;
          hero.vx = 0;
        }
        if (hero.vx < 0 && solid(Math.floor((hero.x - hw) / TILE), ty)) {
          hero.x = (Math.floor((hero.x - hw) / TILE) + 1) * TILE + hw + 0.01;
          hero.vx = 0;
        }
      }
      // move Y
      const wasGround = hero.ground;
      hero.y += hero.vy * dt;
      hero.ground = false;
      for (const xx of [hero.x - hw + 1, hero.x + hw - 1]) {
        const tx = Math.floor(xx / TILE);
        if (hero.vy >= 0 && solid(tx, Math.floor(hero.y / TILE))) {
          hero.y = Math.floor(hero.y / TILE) * TILE;
          hero.vy = 0;
          hero.ground = true;
        }
        if (hero.vy < 0 && solid(tx, Math.floor((hero.y - hh) / TILE))) {
          hero.y = (Math.floor((hero.y - hh) / TILE) + 1) * TILE + hh;
          hero.vy = 0;
        }
      }
      // water: splash & respawn
      if (tileAt(Math.floor(hero.x / TILE), Math.floor((hero.y - 4) / TILE)) === '~' || hero.y > LH * TILE + 40) {
        for (let i = 0; i < 12; i++) dust.push({ x: hero.x, y: 10 * TILE, vx: (Math.random() - 0.5) * 120, vy: -60 - Math.random() * 140, t: 0, life: 0.7, c: [0.45, 0.75, 1] });
        hero.x = 3 * TILE;
        hero.y = 9 * TILE;
        hero.vx = hero.vy = 0;
        setState('IDLE', 'splash! respawn');
      }
      // states
      hero.stateT += dt;
      if (hero.ground && !wasGround) {
        hero.landT = 0;
        setState('LAND', 'touched ground');
        if (p.landSquash) for (let i = 0; i < 6; i++) dust.push({ x: hero.x + (i - 2.5) * 2, y: hero.y, vx: (i - 2.5) * 22, vy: -12 - Math.random() * 18, t: 0, life: 0.45, c: [0.92, 0.86, 0.74] });
      }
      hero.landT += dt;
      if (!hero.ground) {
        if (hero.vy >= 0) setState('FALL', hero.state === 'JUMP' ? 'vy ≥ 0 (apex)' : 'left the ledge');
        else if (hero.state !== 'JUMP') setState('JUMP', 'rising');
      } else if (hero.state === 'LAND' && hero.stateT < 0.12) {
        /* hold the landing pose briefly */
      } else if (Math.abs(hero.vx) > 5) setState('RUN', ix ? 'input ← / →' : 'still sliding');
      else setState('IDLE', 'no input');
      // running dust
      if (hero.state === 'RUN' && hero.ground && p.landSquash && Math.random() < dt * 8) dust.push({ x: hero.x - hero.face * 4, y: hero.y, vx: -hero.face * 15, vy: -10, t: 0, life: 0.35, c: [0.92, 0.86, 0.74] });
      // coins
      for (const c of coins) {
        if (animClock - c.taken < 6) continue;
        if (Math.abs(c.x - hero.x) < 9 && Math.abs(c.y - (hero.y - 7)) < 11) {
          c.taken = animClock;
          score++;
          for (let i = 0; i < 8; i++) {
            const a = (i / 8) * Math.PI * 2;
            dust.push({ x: c.x, y: c.y, vx: Math.cos(a) * 70, vy: Math.sin(a) * 70, t: 0, life: 0.35, c: [1, 0.85, 0.3] });
          }
        }
      }
      for (const d of dust) {
        d.t += dt;
        d.x += d.vx * dt;
        d.y += d.vy * dt;
        d.vy += 200 * dt;
      }
      for (let i = dust.length - 1; i >= 0; i--) if (dust[i].t > dust[i].life) dust.splice(i, 1);
    }

    const stateTag = tag(ctx, 'right:8px;top:48px;text-align:right;line-height:1.7');
    function play(ctx) {
      const p = ctx.params;
      const W = ctx.width;
      const H = ctx.height;
      const enc = ctx.encoder;
      const dt = Math.min(ctx.dt, 1 / 30);
      if (!ctx.paused) {
        animClock += dt;
        const sub = 2;
        for (let i = 0; i < sub; i++) step(dt / sub, p, ctx.keys, i === 0 ? ctx.keysPressed : new Set());
      }
      const k = Math.max(1, Math.round(H / 210));
      // camera: smooth follow, clamped to the level
      camX += (hero.x - camX) * damp(6, dt);
      camY += (hero.y - 30 - camY) * damp(4, dt);
      const halfW = W / 2 / k;
      const halfH = H / 2 / k;
      cam.setViewport(W, H);
      cam.zoom = k;
      cam.x = Math.round(clamp(camX, halfW, Math.max(halfW, LW * TILE - halfW)) * k) / k;
      cam.y = Math.round(clamp(camY, Math.min(LH * TILE - halfH, halfH), LH * TILE - halfH + 8) * k) / k;
      SU.set('res', [W, H]).set('camX', cam.x).set('scale', k);
      sky.draw(enc, canvasOf());
      // level tiles (only the visible ones)
      const x0 = Math.max(0, Math.floor((cam.x - halfW) / TILE));
      const x1 = Math.min(LW - 1, Math.ceil((cam.x + halfW) / TILE));
      const wf = Math.floor(animClock * 5) % 4;
      for (let ty = 0; ty < LH; ty++) {
        for (let tx = x0; tx <= x1; tx++) {
          const ch = LEVEL[ty][tx];
          const name = { G: 'tile_grass', D: 'tile_dirt', B: 'tile_brick', W: 'tile_wood', '~': `tile_water_${wf}` }[ch];
          if (name) sprites.draw(tx * TILE, ty * TILE, TILE, TILE, { uv: atlas.uv(name), anchor: [0, 0], color: ch === '~' ? [1, 1, 1, 0.9] : undefined });
        }
      }
      // decorations
      for (const [tx, name] of [[6, 'bush'], [20, 'flower'], [29, 'rock'], [35, 'flower'], [52, 'bush'], [57, 'flower']]) {
        sprites.draw(tx * TILE + 8, 9 * TILE, 16, 16, { uv: atlas.uv(name), anchor: [0.5, 1] });
      }
      sprites.draw(9 * TILE, 9 * TILE, 32, 48, { uv: atlas.uv('tree'), anchor: [0.5, 1] });
      sprites.draw(47 * TILE, 9 * TILE, 32, 48, { uv: atlas.uv('tree'), anchor: [0.5, 1] });
      // coins
      for (const c of coins) {
        if (animClock - c.taken < 6) continue;
        sprites.draw(c.x, c.y + Math.sin(animClock * 3 + c.x) * 1.5, 16, 16, { uv: atlas.uv(`coin_${Math.floor(animClock * 12 + c.x) % 8}`) });
      }
      // hero: state -> clip, fps, squash
      let frame = 'hero_idle_0';
      let sy = 1;
      const st = hero.state;
      if (st === 'IDLE') frame = `hero_idle_${Math.floor(animClock * 2.5) % 2}`;
      if (st === 'RUN') {
        const fps = p.syncAnim ? p.fps * (Math.abs(hero.vx) / 85) : p.fps;
        hero.runPhase = (hero.runPhase || 0) + (ctx.paused ? 0 : dt * fps);
        frame = `hero_run_${Math.floor(hero.runPhase) % 4}`;
      }
      if (st === 'JUMP') {
        frame = 'hero_jump_0';
        sy = 1 + 0.18 * Math.min(1, -hero.vy / 300);
      }
      if (st === 'FALL') {
        frame = 'hero_jump_0';
        sy = 1 + 0.08 * Math.min(1, hero.vy / 400);
      }
      if (p.landSquash && hero.landT < 0.3) sy = 1 - 0.28 * Math.exp(-hero.landT * 14) * Math.cos(hero.landT * 18);
      if (st === 'LAND') frame = 'hero_idle_1';
      sprites.draw(hero.x, hero.y, 16 / sy, 16 * sy, { uv: atlas.uv(frame), anchor: [0.5, 1], flipX: hero.face < 0 });
      sprites.flush(enc, canvasOf(), cam, { extra: [1, 1, 1, 0] });
      // dust & sparkles
      for (const d of dust) {
        const a = 1 - d.t / d.life;
        shapes.circle(d.x, d.y, 1 + 2.5 * (1 - a * 0.5), [...d.c, a * 0.85]);
      }
      shapes.flush(enc, canvasOf(), cam);
      // HUD in screen space
      const hud = new Camera2D().setViewport(W, H);
      const fs = Math.max(2, k);
      pixelText(shapes, `COINS ${score}`, W / 2, 12 * ctx.dpr, fs, '#fcd34d', { align: 'center' });
      if (!manual) pixelText(shapes, 'AUTOPILOT - PRESS A KEY', W / 2, 12 * ctx.dpr + 8 * fs, fs, '#ffffff', { align: 'center' });
      shapes.flush(enc, canvasOf(), hud);
      const chip = (s) =>
        `<span style="display:inline-block;margin-left:4px;padding:1px 7px;border-radius:5px;${s === st ? 'background:#fcd34d;color:#111;text-shadow:none' : 'background:#ffffff14;color:#cbd5e1'}">${s}</span>`;
      stateTag.innerHTML =
        `${STATES.map(chip).join('')}<br><span style="color:#94a3b8">${hero.prev} → <b style="color:#fcd34d">${st}</b>: ${hero.why}</span>` +
        `<br><span style="color:#94a3b8">vx ${hero.vx.toFixed(0)} · vy ${hero.vy.toFixed(0)} · clip <b style="color:#e2e8f0">${frame}</b></span>`;
      info.innerHTML = '';
    }

    return {
      onExample(id) {
        hover.style.display = 'none';
        stateTag.style.display = id === 'play' ? '' : 'none';
        info.innerHTML = '';
      },
      onChange(key) {
        if (key === 'anim') clickedAnim = null;
      },
      onAction(key) {
        if (key !== 'reset') return;
        clickedAnim = null;
        sq.t0 = ctx.time;
        hits.clear();
        Object.assign(hero, { x: 3 * TILE, y: 9 * TILE, vx: 0, vy: 0, state: 'IDLE', prev: 'IDLE', why: '' });
        manual = false;
        score = 0;
      },
      frame(ctx) {
        shapes.begin(); // once per frame: every flush draws only what was added since the previous flush
        sprites.begin();
        stateTag.style.display = ctx.example === 'play' ? '' : 'none';
        if (ctx.example !== 'flipbook') hover.style.display = 'none';
        if (ctx.example === 'flipbook') flipbook(ctx);
        else if (ctx.example === 'squash') squash(ctx);
        else if (ctx.example === 'palette') palette(ctx);
        else play(ctx);
        info.style.display = info.innerHTML ? '' : 'none';
      },
    };
  },
};

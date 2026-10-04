// 2D Cameras — a small playable world to feel the difference between camera follow modes
// (locked, smoothed, dead zone, look-ahead), trauma-based noise screen shake, zoom & rotation,
// and framing several targets at once. Debug overlay + minimap show what the camera is doing.

import { ShapeBatch, SpriteBatch, Camera2D } from '../../core/batch.js';
import { makeCanvas, rng } from '../../core/assets.js';
import { atlasTexture, tag, pixelText, clamp, damp } from './_shared.js';

const TS = 16;
const MW = 96;
const MH = 64;

// smooth 1D noise in [-1, 1] (value noise with a smoothstep) — the heart of good screen shake
function noise1(seed, x) {
  const h = (i) => {
    let v = Math.imul((i | 0) ^ Math.imul(seed, 0x9e3779b1), 0x85ebca6b);
    v ^= v >>> 13;
    v = Math.imul(v, 0xc2b2ae35);
    v ^= v >>> 16;
    return ((v >>> 0) / 4294967296) * 2 - 1;
  };
  const i = Math.floor(x);
  const f = x - i;
  const s = f * f * (3 - 2 * f);
  return h(i) + (h(i + 1) - h(i)) * s;
}

function buildWorld() {
  const r = rng(42);
  const ground = new Uint8Array(MW * MH); // 0 grass, 1 sand path, 2 water, 3 stone plaza, 4 flowers
  const cx = MW / 2;
  const cy = MH / 2;
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      let g = 0;
      const pond1 = Math.hypot((x - 70) / 11, (y - 18) / 7) + 0.15 * Math.sin(x * 0.7 + y);
      const pond2 = Math.hypot((x - 22) / 8, (y - 46) / 6) + 0.15 * Math.sin(y * 0.9 + x * 0.3);
      if (Math.abs(y - cy - Math.round(3 * Math.sin(x * 0.12))) <= 1 || Math.abs(x - cx - Math.round(3 * Math.sin(y * 0.15))) <= 1) g = 1;
      if (pond1 < 1 || pond2 < 1) g = 2;
      if (Math.abs(x - cx) <= 3 && Math.abs(y - cy) <= 2) g = 3;
      if (g === 0 && r() < 0.03) g = 4;
      ground[y * MW + x] = g;
    }
  const objects = [];
  const free = (x, y) => ground[y * MW + x] === 0 || ground[y * MW + x] === 4;
  // ground tileset: grass, sand, water ×4, stone (grass painted here: calmer than the atlas leaves)
  const tiles = makeCanvas(7 * TS, TS);
  const tg = tiles.getContext('2d');
  for (let y = 0; y < TS; y++)
    for (let x = 0; x < TS; x++) {
      const q = r();
      tg.fillStyle = q < 0.12 ? '#43a457' : q > 0.95 ? '#62c56f' : '#4cb05c';
      if ((x * 7 + y * 3) % 23 === 0) tg.fillStyle = '#7fd06a';
      tg.fillRect(x, y, 1, 1);
    }
  for (let i = 0; i < 520; i++) {
    const x = Math.floor(r() * MW);
    const y = Math.floor(r() * MH);
    if (!free(x, y)) continue;
    if (Math.abs(x - cx) < 9 && Math.abs(y - cy) < 7) continue;
    const k = r();
    const name = k < 0.55 ? 'tree' : k < 0.72 ? 'bush' : k < 0.84 ? 'rock' : k < 0.93 ? 'mushroom' : 'crate';
    objects.push({ name, x: x * TS + 8 + (r() - 0.5) * 6, y: y * TS + 14 });
  }
  objects.push({ name: 'chest', x: cx * TS, y: cy * TS + 6 });
  const coins = [];
  for (let i = 0; i < 400 && coins.length < 45; i++) {
    const x = Math.floor(r() * MW);
    const y = Math.floor(r() * MH);
    if (ground[y * MW + x] === 1 || (ground[y * MW + x] === 0 && r() < 0.3)) coins.push({ x: x * TS + 8, y: y * TS + 8, taken: -99 });
  }
  // minimap texture: one pixel per tile
  const mini = makeCanvas(MW, MH);
  const g = mini.getContext('2d');
  const cols = ['#4cb05c', '#d8b073', '#3b6fd0', '#8a96a8', '#5cc46a'];
  for (let y = 0; y < MH; y++)
    for (let x = 0; x < MW; x++) {
      g.fillStyle = cols[ground[y * MW + x]];
      g.fillRect(x, y, 1, 1);
    }
  g.fillStyle = '#1f6b45';
  for (const o of objects) if (o.name === 'tree') g.fillRect(Math.floor(o.x / TS), Math.floor(o.y / TS) - 1, 1, 2);
  return { ground, objects, coins, mini, tiles };
}

export default {
  keys: true,
  wheel: true,
  interaction: 'WASD / arrows move · click = explosion · wheel = zoom · Q/E rotate',
  examples: [
    {
      id: 'follow',
      label: 'Follow modes',
      kind: 'In a game',
      note: 'The camera chases the knight. <b>Locked</b> is rigid and every tiny step jerks the screen. <b>Smooth</b> eases toward the player. A <b>dead zone</b> lets the player move freely inside a box before the camera reacts. <b>Look-ahead</b> shifts the view toward where you are heading so you can see what’s coming. The overlay shows the box, the target and the camera centre.',
      hint: 'WASD / arrows move · wheel zoom · Q/E rotate',
    },
    {
      id: 'shake',
      label: 'Trauma screen shake',
      kind: 'In a game',
      note: 'Click anywhere to set off an explosion. Each one adds <b>trauma</b> (0–1) that decays over time; the shake is <b>trauma²</b> × max offset × <b>smooth noise</b> — so small hits barely move the screen, big ones rattle it, and the motion is organic instead of random jitter (the approach from Squirrel Eiserloh’s GDC talk “Juicing Your Cameras With Math”).',
      hint: 'Click = explosion (adds trauma)',
      params: { mode: 'lerp' },
    },
    {
      id: 'multi',
      label: 'Framing multiple targets',
      kind: 'In a game',
      note: 'Co-op and party games keep everyone on screen: the camera centres on the <b>bounding box</b> of all targets and zooms out until the box (plus padding) fits — smoothly, and within zoom limits. Watch the slimes wander off and the view widen.',
      hint: 'WASD moves the knight · slimes wander off',
    },
  ],
  controls: [
    {
      type: 'select',
      key: 'mode',
      label: 'Follow mode',
      value: 'deadzone',
      showFor: ['follow', 'shake'],
      options: [
        { value: 'locked', label: 'Locked (camera = player)' },
        { value: 'lerp', label: 'Smooth (lerp / exponential)' },
        { value: 'deadzone', label: 'Dead zone (window)' },
        { value: 'lookahead', label: 'Look-ahead' },
      ],
    },
    { type: 'slider', key: 'smooth', label: 'Smoothing speed', min: 0.5, max: 20, step: 0.1, value: 5, showFor: ['follow', 'shake', 'multi'], help: 'Exponential follow: higher = snappier. Frame-rate independent: k = 1 − e^(−speed·dt).' },
    { type: 'slider', key: 'deadzone', label: 'Dead-zone size', min: 0.05, max: 0.6, step: 0.01, value: 0.28, showFor: ['follow'], help: 'Fraction of the screen the player can roam before the camera moves.' },
    { type: 'slider', key: 'lookahead', label: 'Look-ahead distance', min: 0, max: 120, step: 1, value: 60, showFor: ['follow'], help: 'In world pixels, in the direction of movement.' },
    { type: 'slider', key: 'shakeMax', label: 'Max shake offset', min: 0, max: 40, step: 0.5, value: 14, showFor: ['shake'], help: 'World pixels at trauma = 1.' },
    { type: 'slider', key: 'shakeAngle', label: 'Max shake rotation', min: 0, max: 15, step: 0.1, value: 4, unit: '°', showFor: ['shake'] },
    { type: 'slider', key: 'shakeFreq', label: 'Shake frequency', min: 2, max: 40, step: 0.5, value: 18, showFor: ['shake'], help: 'How fast the noise is sampled.' },
    { type: 'slider', key: 'decay', label: 'Trauma decay / second', min: 0.2, max: 3, step: 0.05, value: 0.9, showFor: ['shake'] },
    { type: 'toggle', key: 'smoothNoise', label: 'Smooth noise (vs. random jitter)', value: true, showFor: ['shake'], help: 'Off = a new random offset every frame: harsh and frame-rate dependent.' },
    { type: 'slider', key: 'zoom', label: 'Zoom', min: 0.4, max: 3, step: 0.01, value: 1, log: true, showFor: ['follow', 'shake'], help: 'The mouse wheel zooms too.' },
    { type: 'slider', key: 'rotation', label: 'Rotation', min: -45, max: 45, step: 1, value: 0, unit: '°', showFor: ['follow', 'shake'], help: 'Q / E also rotate.' },
    { type: 'slider', key: 'padding', label: 'Framing padding', min: 10, max: 200, step: 1, value: 60, showFor: ['multi'], help: 'World pixels of margin around the targets’ bounding box.' },
    { type: 'toggle', key: 'debug', label: 'Debug overlay', value: true },
    { type: 'toggle', key: 'minimap', label: 'Minimap with camera rect', value: true },
  ],
  about: {
    summary: 'The camera is a transform from world to screen. How it follows, shakes, zooms and frames the action is a huge part of how a game feels.',
    what: `<p>A little top-down world with a knight (WASD/arrows, or autopilot). The yellow overlay shows what the camera does: its centre, its target,
      the dead-zone box and the look-ahead point. The minimap shows the camera’s view rectangle in the world.</p>`,
    how: `<ol>
      <li>A 2D camera is a <b>position, zoom and rotation</b>. Rendering applies <code>screen = (world − camera) × zoom</code> (rotated) — here as one 4×4 matrix in the vertex shader.</li>
      <li><b>Smoothing</b>: each frame move a fraction of the way: <code>cam += (target − cam) × (1 − e^(−speed·dt))</code>. Using <code>e^(−speed·dt)</code>
        instead of a fixed 0.1 keeps it identical at 30, 60 or 144 fps.</li>
      <li><b>Dead zone</b>: only when the player leaves a box around the camera centre does the camera move — by exactly the overshoot. Great for platformers
        where small hops shouldn’t move the screen.</li>
      <li><b>Look-ahead</b>: the target is the player plus a smoothed velocity direction × distance — the player sees more of where they are going.</li>
      <li><b>Trauma shake</b>: events add trauma (clamped to 1), which decays linearly. Shake = trauma² × max × noise(t·freq) for x, y and angle. Squaring makes
        the falloff feel natural; smooth noise avoids the frame-rate-dependent jitter of <code>Math.random()</code>. The shake is applied <i>on top of</i> the
        follow position so it never drifts the camera.</li>
      <li><b>Framing</b>: bounding box of all targets → centre; zoom = min(viewW / boxW, viewH / boxH), clamped, then smoothed.</li>
    </ol>`,
    uses: [
      { title: 'Platformers', text: 'Dead zones and look-ahead (Super Mario World, Celeste, Hollow Knight) — often with separate horizontal/vertical rules.' },
      { title: 'Action & shooters', text: 'Trauma shake on hits, explosions and heavy landings (Nuclear Throne, Enter the Gungeon, Vlambeer’s “juice”).' },
      { title: 'Co-op & fighting games', text: 'Framing all players: Smash Bros., Overcooked, Cuphead co-op.' },
      { title: 'Strategy & sims', text: 'Smooth zoom to cursor, edge panning and rotation in city builders and RTS games.' },
    ],
    try: [
      'In <b>Follow modes</b>, choose <i>Locked</i> and walk in small steps: every pixel of movement moves the whole screen. Then try <i>Dead zone</i>.',
      'Pick <i>Look-ahead</i> and run in one direction, then turn around: the camera swings ahead of the knight.',
      'In <b>Trauma shake</b>, click once, then click five times quickly: trauma adds up and the squared curve makes the difference huge.',
      'Turn off <i>Smooth noise</i>: the shake becomes harsh static.',
      'Zoom out with the wheel and rotate with Q/E — watch the camera rectangle on the minimap.',
    ],
    ask: [
      'a smooth follow camera with a dead zone and look-ahead',
      'trauma-based screen shake using Perlin noise (trauma squared)',
      'frame-rate independent camera smoothing (exponential decay)',
      'a camera that frames all players with zoom limits',
      'zoom to cursor with the mouse wheel',
      'a minimap showing the camera view rectangle',
    ],
    perf: `<p>Free — a camera is a matrix uniform. What matters is <b>culling</b>: only tiles and objects inside the (rotated, zoomed) view rectangle are sent to the
      GPU. Zooming out raises the number of visible tiles quadratically; the readout shows how many are drawn.</p>`,
    api: `<p>Identical on WebGL2 and WebGPU: the camera is a 4×4 matrix in a uniform buffer used by the vertex shader. (This scene is implemented with the WebGPU
      sprite batcher.)</p>`,
    code: [
      {
        title: 'Follow modes (JavaScript, per frame)',
        lang: 'js',
        src: `let target = [player.x, player.y];
if (mode === 'lookahead') {
  look = lerp(look, normalize(player.vel), damp(3, dt));
  target = [player.x + look.x * dist, player.y + look.y * dist];
}
if (mode === 'deadzone') {                // move only by the overshoot
  const hw = viewW * size / 2, hh = viewH * size / 2;
  if (player.x > cam.x + hw) cam.x = player.x - hw;
  if (player.x < cam.x - hw) cam.x = player.x + hw;
  /* same for y */
} else if (mode === 'locked') cam.x = target[0];
else cam.x += (target[0] - cam.x) * (1 - Math.exp(-speed * dt));`,
      },
      {
        title: 'Trauma shake',
        lang: 'js',
        src: `trauma = Math.min(1, trauma + 0.45);          // on explosion
trauma = Math.max(0, trauma - decay * dt);     // every frame
const s = trauma * trauma;                     // squared: gentle → violent
const t = time * freq;
view.x = cam.x + maxOffset * s * noise(1, t);  // smooth noise in [-1, 1]
view.y = cam.y + maxOffset * s * noise(2, t);
view.rotation = rotation + maxAngle * s * noise(3, t);`,
      },
    ],
    links: [
      { title: 'Squirrel Eiserloh — Juicing Your Cameras With Math (GDC)', url: 'https://www.youtube.com/watch?v=tu-Qe66AvtY', note: 'trauma, noise shake, smoothing' },
      { title: 'Itay Keren — Scroll Back: The Theory and Practice of Cameras in Side-Scrollers', url: 'https://www.gamedeveloper.com/design/scroll-back-the-theory-and-practice-of-cameras-in-side-scrollers', note: 'dead zones, look-ahead and more' },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const { atlas, view } = await atlasTexture(gpu);
    const world = buildWorld();
    const miniTex = gpu.textureFromImage(world.mini, { label: 'minimap' }).createView();
    const sprites = new SpriteBatch(gpu, { texture: view, filter: 'nearest' });
    const shapes = new ShapeBatch(gpu);
    const cam = new Camera2D();
    const screen = new Camera2D();
    const readout = tag(ctx, 'right:8px;top:48px;text-align:right;line-height:1.5');
    const canvasOf = () => ({ view: ctx.target, format: gpu.format });

    const at = world.tiles.getContext('2d');
    atlas.anim('tile_water').concat(['tile_sand', 'tile_stone']).forEach((n, i) => {
      const f = atlas.frames[n];
      const slot = n === 'tile_sand' ? 1 : n === 'tile_stone' ? 6 : 2 + i;
      at.drawImage(atlas.canvas, f.x, f.y, TS, TS, slot * TS, 0, TS, TS);
    });
    const groundTex = gpu.textureFromImage(world.tiles, { label: 'ground' }).createView();
    const tileUV = (slot) => [(slot * TS) / world.tiles.width, 0, ((slot + 1) * TS) / world.tiles.width, 1];
    const tileSlot = [0, 1, 2, 6, 0];
    const water = (x, y) => {
      const tx = Math.floor(x / TS);
      const ty = Math.floor(y / TS);
      return tx < 0 || ty < 0 || tx >= MW || ty >= MH || world.ground[ty * MW + tx] === 2;
    };

    const player = { x: (MW / 2 - 12) * TS, y: (MH / 2) * TS + 8, vx: 0, vy: 0, face: 1, phase: 0 };
    let manual = false;
    let wp = null;
    let wpT = 0;
    const r = rng(7);
    const camPos = { x: player.x, y: player.y };
    const look = { x: 0, y: 0 };
    let zoomMul = 1;
    let rotKeys = 0;
    let trauma = 0;
    let shakeView = { x: 0, y: 0, a: 0 };
    const traumaHist = [];
    const booms = [];
    const parts = [];
    let fitZoom = 1;
    const slimes = [0, 1, 2].map((i) => ({ x: player.x + (i - 1) * 40, y: player.y + 30, vx: 0, vy: 0, goal: null, t: i, hop: 0 }));
    let score = 0;
    let clock = 0;
    let lastZoomParam = ctx.params.zoom;
    let lastClick = -99;
    let lastAuto = 0;

    const explode = (x, y) => {
      trauma = Math.min(1, trauma + 0.45);
      booms.push({ x, y, t: 0 });
      for (let i = 0; i < 26; i++) {
        const a = Math.random() * Math.PI * 2;
        const sp = 40 + Math.random() * 160;
        parts.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, t: 0, life: 0.4 + Math.random() * 0.5, smoke: i % 3 === 0 });
      }
    };

    const movePlayer = (dt, p) => {
      const k = ctx.keys;
      let ix = (k.has('d') || k.has('ArrowRight') ? 1 : 0) - (k.has('a') || k.has('ArrowLeft') ? 1 : 0);
      let iy = (k.has('s') || k.has('ArrowDown') ? 1 : 0) - (k.has('w') || k.has('ArrowUp') ? 1 : 0);
      if (ix || iy) manual = true;
      if (!manual) {
        wpT -= dt;
        if (!wp || wpT < 0 || Math.hypot(wp[0] - player.x, wp[1] - player.y) < 10) {
          // autopilot: wander to random land points nearby (with pauses → shows dead zone & smoothing)
          for (let tries = 0; tries < 20; tries++) {
            const a = r() * Math.PI * 2;
            const d = 60 + r() * 220;
            const nx = clamp(player.x + Math.cos(a) * d, 40, MW * TS - 40);
            const ny = clamp(player.y + Math.sin(a) * d, 40, MH * TS - 40);
            if (!water(nx, ny)) {
              wp = [nx, ny];
              break;
            }
          }
          wpT = 2.5 + r() * 2;
        }
        const dx = wp[0] - player.x;
        const dy = wp[1] - player.y;
        const d = Math.hypot(dx, dy);
        if (d > 6 && wpT < 2.2) {
          ix = dx / d;
          iy = dy / d;
        }
      }
      const len = Math.hypot(ix, iy) || 1;
      const speed = 95;
      const acc = 1 - Math.exp(-14 * dt);
      player.vx += ((ix / len) * speed - player.vx) * acc;
      player.vy += ((iy / len) * speed - player.vy) * acc;
      const nx = player.x + player.vx * dt;
      const ny = player.y + player.vy * dt;
      if (!water(nx, player.y)) player.x = nx;
      else player.vx = 0;
      if (!water(player.x, ny)) player.y = ny;
      else player.vy = 0;
      if (Math.abs(player.vx) > 5) player.face = Math.sign(player.vx);
      player.phase += dt * 10 * (Math.hypot(player.vx, player.vy) / speed);
    };

    const moveSlimes = (dt, spread) => {
      for (const s of slimes) {
        s.t -= dt;
        if (!s.goal || s.t < 0) {
          for (let tries = 0; tries < 20; tries++) {
            const a = r() * Math.PI * 2;
            const d = 30 + r() * spread;
            const nx = clamp(player.x + Math.cos(a) * d, 30, MW * TS - 30);
            const ny = clamp(player.y + Math.sin(a) * d, 30, MH * TS - 30);
            if (!water(nx, ny)) {
              s.goal = [nx, ny];
              break;
            }
          }
          s.t = 1.5 + r() * 3;
        }
        if (!s.goal) continue;
        const dx = s.goal[0] - s.x;
        const dy = s.goal[1] - s.y;
        const d = Math.hypot(dx, dy);
        if (d > 4) {
          s.hop += dt * 3;
          const sp = 55 * (0.4 + 0.6 * Math.abs(Math.sin(s.hop * Math.PI)));
          s.x += (dx / d) * sp * dt;
          s.y += (dy / d) * sp * dt;
        }
      }
    };

    return {
      onChange(key, v) {
        if (key === 'zoom') {
          zoomMul = 1;
          lastZoomParam = v;
        }
      },
      onAction(key) {
        if (key !== 'reset') return;
        Object.assign(player, { x: (MW / 2 - 12) * TS, y: (MH / 2) * TS + 8, vx: 0, vy: 0 });
        camPos.x = player.x;
        camPos.y = player.y;
        manual = false;
        trauma = 0;
        zoomMul = 1;
        rotKeys = 0;
        traumaHist.length = 0;
      },
      frame(ctx) {
        shapes.begin(); // once per frame; each flush draws what was added since the previous flush
        sprites.begin();
        const p = ctx.params;
        const ex = ctx.example;
        const W = ctx.width;
        const H = ctx.height;
        const dt = ctx.paused ? 0 : Math.min(ctx.dt, 1 / 20);
        const ptr = ctx.pointer;
        clock += dt;
        const base = Math.max(1, Math.round(H / 230));
        if (ptr.wheel) zoomMul = clamp(zoomMul * Math.exp(-ptr.wheel * 0.0015), 0.3 / lastZoomParam, 4 / lastZoomParam);
        if (ctx.keys.has('q')) rotKeys -= dt * 40;
        if (ctx.keys.has('e')) rotKeys += dt * 40;
        if (dt > 0) {
          movePlayer(dt, p);
          moveSlimes(dt, ex === 'multi' ? 340 : 90);
        }

        // ---------------- follow
        const userZoom = ex === 'multi' ? 1 : p.zoom * zoomMul;
        let zoom = base * userZoom;
        const viewW = W / zoom;
        const viewH = H / zoom;
        const mode = ex === 'multi' ? 'multi' : p.mode;
        const kS = damp(p.smooth, dt);
        let target = [player.x, player.y];
        let box = null;
        if (mode === 'lookahead') {
          const sp = Math.hypot(player.vx, player.vy);
          const dir = sp > 10 ? [player.vx / sp, player.vy / sp] : [look.x * 0.98, look.y * 0.98];
          look.x += (dir[0] - look.x) * damp(3, dt);
          look.y += (dir[1] - look.y) * damp(3, dt);
          target = [player.x + look.x * p.lookahead, player.y + look.y * p.lookahead];
        }
        if (mode === 'locked') {
          camPos.x = target[0];
          camPos.y = target[1];
        } else if (mode === 'deadzone') {
          const hw = (viewW * p.deadzone) / 2;
          const hh = (viewH * p.deadzone) / 2;
          if (player.x > camPos.x + hw) camPos.x = player.x - hw;
          if (player.x < camPos.x - hw) camPos.x = player.x + hw;
          if (player.y > camPos.y + hh) camPos.y = player.y - hh;
          if (player.y < camPos.y - hh) camPos.y = player.y + hh;
        } else if (mode === 'multi') {
          const xs = [player.x, ...slimes.map((s) => s.x)];
          const ys = [player.y - 8, ...slimes.map((s) => s.y - 6)];
          box = [Math.min(...xs) - p.padding, Math.min(...ys) - p.padding, Math.max(...xs) + p.padding, Math.max(...ys) + p.padding];
          const want = clamp(Math.min(W / (box[2] - box[0]), H / (box[3] - box[1])), base * 0.35, base * 2.5);
          fitZoom += (want - fitZoom) * damp(p.smooth * 0.6, dt);
          zoom = fitZoom;
          camPos.x += ((box[0] + box[2]) / 2 - camPos.x) * kS;
          camPos.y += ((box[1] + box[3]) / 2 - camPos.y) * kS;
        } else {
          camPos.x += (target[0] - camPos.x) * kS;
          camPos.y += (target[1] - camPos.y) * kS;
        }

        // ---------------- shake (applied on top of the follow position)
        if (ptr.clicked && ptr.button === 0) {
          const [wx, wy] = cam.screenToWorld(ptr.x, ptr.y);
          explode(wx, wy);
          lastClick = clock;
        }
        // demo mode: random explosions near the knight until you click yourself
        if (ex === 'shake' && dt > 0 && clock - lastClick > 4 && clock - lastAuto > 1.6 + (Math.sin(clock * 1.3) + 1) * 0.8) {
          lastAuto = clock;
          const a = r() * Math.PI * 2;
          explode(player.x + Math.cos(a) * 50, player.y + Math.sin(a) * 35);
        }
        trauma = Math.max(0, trauma - p.decay * dt);
        const s2 = trauma * trauma;
        const tt = clock * p.shakeFreq;
        if (p.smoothNoise) shakeView = { x: noise1(1, tt), y: noise1(2, tt), a: noise1(3, tt) };
        else if (dt > 0) shakeView = { x: Math.random() * 2 - 1, y: Math.random() * 2 - 1, a: Math.random() * 2 - 1 };
        const shakeOn = ex === 'shake' || trauma > 0;
        const rot = ((ex === 'multi' ? 0 : p.rotation + rotKeys) * Math.PI) / 180;
        cam.setViewport(W, H);
        cam.zoom = zoom;
        cam.x = camPos.x + (shakeOn ? p.shakeMax * s2 * shakeView.x : 0);
        cam.y = camPos.y + (shakeOn ? p.shakeMax * s2 * shakeView.y : 0);
        cam.rotation = rot + (shakeOn ? ((p.shakeAngle * Math.PI) / 180) * s2 * shakeView.a : 0);
        if (dt > 0) {
          traumaHist.push(trauma);
          if (traumaHist.length > 150) traumaHist.shift();
        }

        // ---------------- visible rect (rotated view → world AABB) for culling
        const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([x, y]) => cam.screenToWorld(x, y));
        const minX = Math.min(...corners.map((c) => c[0]));
        const maxX = Math.max(...corners.map((c) => c[0]));
        const minY = Math.min(...corners.map((c) => c[1]));
        const maxY = Math.max(...corners.map((c) => c[1]));

        // ---------------- world
        const enc = ctx.encoder;
        const tx0 = clamp(Math.floor(minX / TS), 0, MW - 1);
        const tx1 = clamp(Math.floor(maxX / TS), 0, MW - 1);
        const ty0 = clamp(Math.floor(minY / TS), 0, MH - 1);
        const ty1 = clamp(Math.floor(maxY / TS), 0, MH - 1);
        const wf = Math.floor(clock * 4) % 4;
        let tiles = 0;
        for (let y = ty0; y <= ty1; y++)
          for (let x = tx0; x <= tx1; x++) {
            const g = world.ground[y * MW + x];
            sprites.draw(x * TS, y * TS, TS + 0.02, TS + 0.02, { uv: tileUV(g === 2 ? 2 + wf : tileSlot[g]), anchor: [0, 0] });
            tiles++;
          }
        sprites.flush(enc, canvasOf(), cam, { clear: [0.06, 0.08, 0.12, 1], texture: groundTex });
        for (let y = ty0; y <= ty1; y++)
          for (let x = tx0; x <= tx1; x++) if (world.ground[y * MW + x] === 4) sprites.draw(x * TS + 8, y * TS + 12, 12, 12, { uv: atlas.uv('flower'), anchor: [0.5, 1] });
        sprites.flush(enc, canvasOf(), cam);

        // y-sorted objects + characters
        const drawList = [];
        for (const o of world.objects) if (o.x > minX - 32 && o.x < maxX + 32 && o.y > minY - 8 && o.y < maxY + 56) drawList.push(o);
        for (const c of world.coins) {
          if (clock - c.taken < 8 || c.x < minX - 16 || c.x > maxX + 16 || c.y < minY - 16 || c.y > maxY + 16) continue;
          drawList.push({ name: `coin_${Math.floor(clock * 12 + c.x) % 8}`, x: c.x, y: c.y + 4 + Math.sin(clock * 3 + c.x) * 1.5, coin: true });
          if (Math.hypot(c.x - player.x, c.y - (player.y - 6)) < 12) {
            c.taken = clock;
            score++;
            for (let i = 0; i < 8; i++) parts.push({ x: c.x, y: c.y, vx: Math.cos(i) * 60, vy: Math.sin(i) * 60, t: 0, life: 0.35, coin: true });
          }
        }
        const moving = Math.hypot(player.vx, player.vy) > 8;
        drawList.push({ name: moving ? `hero_run_${Math.floor(player.phase) % 4}` : `hero_idle_${Math.floor(clock * 2.5) % 2}`, x: player.x, y: player.y, flip: player.face < 0, hero: true });
        if (ex === 'multi') for (const s of slimes) drawList.push({ name: `slime_${Math.floor(s.hop * 2) % 2 ? 1 : 0}`, x: s.x, y: s.y, slime: true });
        drawList.sort((a, b) => a.y - b.y);
        // shadows first
        for (const o of drawList) if (o.hero || o.slime || o.name === 'tree') shapes.box(o.x, o.y - 1, o.name === 'tree' ? 9 : 6, 2.2, '#00000040', { radius: 2 });
        shapes.flush(enc, canvasOf(), cam);
        for (const o of drawList) {
          const f = atlas.frames[o.name];
          sprites.draw(o.x, o.y, f.w, f.h, { uv: atlas.uv(o.name), anchor: [0.5, 1], flipX: o.flip });
        }
        sprites.flush(enc, canvasOf(), cam);

        // explosions & particles
        for (const b of booms) b.t += dt;
        for (let i = booms.length - 1; i >= 0; i--) if (booms[i].t > 0.6) booms.splice(i, 1);
        for (const q of parts) {
          q.t += dt;
          q.x += q.vx * dt;
          q.y += q.vy * dt;
          q.vx *= Math.exp(-3 * dt);
          q.vy *= Math.exp(-3 * dt);
        }
        for (let i = parts.length - 1; i >= 0; i--) if (parts[i].t > parts[i].life) parts.splice(i, 1);
        for (const q of parts) if (q.smoke) shapes.circle(q.x, q.y, 3 + q.t * 14, [0.25, 0.22, 0.25, 0.5 * (1 - q.t / q.life)]);
        shapes.flush(enc, canvasOf(), cam);
        for (const b of booms) {
          const k = b.t / 0.6;
          shapes.circle(b.x, b.y, 6 + k * 46, [1, 0.8, 0.4, 0.9 * (1 - k)], { stroke: 3 * (1 - k) + 0.5, glow: 6, glowStrength: 0.5 });
          if (k < 0.3) shapes.circle(b.x, b.y, 14 * (1 - k), [1, 0.9, 0.6, 1 - k / 0.3], { glow: 14, glowStrength: 0.9 });
        }
        for (const q of parts) {
          if (q.smoke) continue;
          const a = 1 - q.t / q.life;
          shapes.line(q.x, q.y, q.x - q.vx * 0.03, q.y - q.vy * 0.03, 1.6, q.coin ? [1, 0.85, 0.3, a] : [1, 0.6 + 0.3 * a, 0.2, a], { glow: 3, glowStrength: 0.6 });
        }
        shapes.flush(enc, canvasOf(), cam, { blend: 'additive' });

        // ---------------- debug overlay (world space)
        const lw = 1.5 / zoom;
        if (p.debug) {
          if (mode === 'deadzone') {
            const hw = (viewW * p.deadzone) / 2;
            const hh = (viewH * p.deadzone) / 2;
            shapes.rect(camPos.x - hw, camPos.y - hh, hw * 2, hh * 2, '#fcd34dcc', { stroke: lw * 1.5, rotation: 0 });
          }
          if (mode === 'lookahead') {
            shapes.line(player.x, player.y - 8, target[0], target[1] - 8, lw, '#7dd3fccc');
            shapes.circle(target[0], target[1] - 8, 3 / zoom + 2, '#7dd3fc', { stroke: lw * 1.5 });
          }
          if (mode === 'lerp' || mode === 'lookahead') shapes.line(camPos.x, camPos.y, target[0], target[1] - (mode === 'lookahead' ? 8 : 0), lw, '#f9a8d488');
          if (box) {
            const pd = p.padding;
            shapes.rect(box[0] + pd, box[1] + pd, box[2] - box[0] - 2 * pd, box[3] - box[1] - 2 * pd, '#86efacdd', { stroke: lw * 1.5 });
            shapes.rect(box[0], box[1], box[2] - box[0], box[3] - box[1], '#86efac55', { stroke: lw });
          }
          if (box) for (const s of slimes) shapes.line(s.x, s.y - 6, player.x, player.y - 8, lw * 0.7, '#86efac55');
          // camera centre (unshaken) crosshair
          const c = 6 / zoom + 3;
          shapes.line(camPos.x - c, camPos.y, camPos.x + c, camPos.y, lw, '#f9a8d4');
          shapes.line(camPos.x, camPos.y - c, camPos.x, camPos.y + c, lw, '#f9a8d4');
        }
        shapes.flush(enc, canvasOf(), cam);

        // ---------------- HUD (screen space): minimap + trauma meter
        screen.setViewport(W, H).reset();
        const u = Math.max(1, Math.round(ctx.dpr));
        if (p.minimap) {
          const mw = Math.round(Math.min(W * 0.22, 200 * u));
          const mh = Math.round((mw * MH) / MW);
          const mx = W - mw - 10 * u;
          const my = H - mh - 10 * u;
          shapes.rect(mx - 3 * u, my - 3 * u, mw + 6 * u, mh + 6 * u, '#000000aa', { radius: 4 * u });
          shapes.flush(enc, canvasOf(), screen);
          sprites.draw(mx, my, mw, mh, { uv: [0, 0, 1, 1], anchor: [0, 0] });
          sprites.flush(enc, canvasOf(), screen, { texture: miniTex });
          const toMini = ([x, y]) => [mx + (x / (MW * TS)) * mw, my + (y / (MH * TS)) * mh];
          shapes.polyline(corners.map(toMini), 1.5 * u, '#fcd34d', { closed: true });
          const [px, py] = toMini([player.x, player.y]);
          shapes.circle(px, py, 2.5 * u, '#ffffff');
          for (const b of booms) {
            const [bx, by] = toMini([b.x, b.y]);
            shapes.circle(bx, by, 3 * u, '#fb923c');
          }
          if (ex === 'multi')
            for (const s of slimes) {
              const [sx, sy] = toMini([s.x, s.y]);
              shapes.circle(sx, sy, 2 * u, '#86efac');
            }
        }
        if (ex === 'shake') {
          const gw = Math.round(Math.min(W * 0.28, 240 * u));
          const gh = 46 * u;
          const gx = p.minimap ? W - Math.round(Math.min(W * 0.22, 200 * u)) - gw - 26 * u : W - gw - 10 * u;
          const gy = H - gh - 10 * u;
          shapes.rect(gx, gy - 12 * u, gw, gh + 12 * u, '#000000aa', { radius: 4 * u });
          const pts = traumaHist.map((v, i) => [gx + 4 * u + (i / 149) * (gw - 8 * u), gy + gh - 4 * u - v * (gh - 10 * u)]);
          const pts2 = traumaHist.map((v, i) => [gx + 4 * u + (i / 149) * (gw - 8 * u), gy + gh - 4 * u - v * v * (gh - 10 * u)]);
          if (pts.length > 1) {
            shapes.polyline(pts, 1.5 * u, '#94a3b8');
            shapes.polyline(pts2, 2 * u, '#fb923c');
          }
          pixelText(shapes, 'TRAUMA', gx + 4 * u, gy - 9 * u, 2 * u, '#94a3b8');
          pixelText(shapes, 'SHAKE=T²', gx + gw - 4 * u, gy - 9 * u, 2 * u, '#fb923c', { align: 'right' });
        }
        if (!manual) pixelText(shapes, 'AUTOPILOT - PRESS WASD', W / 2, 12 * u, 2 * u, '#ffffff', { align: 'center' });
        pixelText(shapes, `COINS ${score}`, W / 2, (manual ? 12 : 26) * u, 2 * u, '#fcd34d', { align: 'center' });
        shapes.flush(enc, canvasOf(), screen);

        const modeName = { locked: 'locked', lerp: 'smooth', deadzone: 'dead zone', lookahead: 'look-ahead', multi: 'framing' }[mode];
        readout.innerHTML =
          `camera: <b style="color:#fcd34d">${modeName}</b> · zoom ${(zoom / base).toFixed(2)}× · rot ${((cam.rotation * 180) / Math.PI).toFixed(1)}°<br>` +
          (ex === 'shake' ? `trauma <b style="color:#fb923c">${trauma.toFixed(2)}</b> → shake ${(s2 * 100).toFixed(0)}%<br>` : '') +
          `<span style="color:#94a3b8">culling: ${tiles} tiles + ${drawList.length} objects drawn</span>`;
      },
    };
  },
};

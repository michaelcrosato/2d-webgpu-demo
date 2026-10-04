import { shaderScene } from '../../core/shaderscene.js';
import { getAtlas } from '../../core/assets.js';
import { overlays, gamePass, withGameIncludes } from './_shared.js';

// Feedback: a pass that reads its OWN previous output. Fade it a little, transform it a little, draw
// something new on top — and you get motion trails, smudges, infinite tunnels and dream effects.

const ov = overlays();
const st = { atlas: null, requested: false, rects: {}, lastStamp: -1, stamp: 0, brush: [0, 0], last: null, autoT: 0 };
const ZERO = [0, 0, 0, 0];

const FB_COMMON = /* wgsl */ `
// premultiplied sprite from the atlas: rect = atlas uv rect, c = centre (p units), h = height (p units)
fn sprite(p: vec2f, c: vec2f, h: f32, rect: vec4f, flip: f32) -> vec4f {
  var l: vec2f = (p - c) / h + 0.5;
  if (l.x < 0.0 || l.x > 1.0 || l.y < 0.0 || l.y > 1.0 || rect.z <= rect.x) { return vec4f(0.0); }
  if (flip > 0.5) { l.x = 1.0 - l.x; }
  let s: vec4f = TEXN(atlas, mix(rect.xy, rect.zw, l));
  return vec4f(s.rgb * s.a, s.a);
}
fn over(top: vec4f, bottom: vec4f) -> vec4f { return top + bottom * (1.0 - top.a); }

// the moving things of the "Motion trails" example (premultiplied color, alpha)
fn spritesLayer(p: vec2f, t: f32) -> vec4f {
  let a: f32 = u.resolution.x / u.resolution.y;
  var acc: vec4f = vec4f(0.0);
  // fireball orbiting in the sky
  let fc: vec2f = vec2f(a * 0.5 + 0.42 * cos(t * 2.1), 0.33 + 0.16 * sin(t * 2.1 * 1.5));
  let fd: f32 = length(p - fc);
  let core: f32 = exp(-fd / 0.012);
  let fire: vec3f = mix(vec3f(1.0, 0.35, 0.05), vec3f(1.0, 0.95, 0.7), core);
  acc = over(vec4f(fire * min(1.0, core * 2.5 + exp(-fd / 0.03) * 0.6), min(1.0, core * 2.5 + exp(-fd / 0.03) * 0.5)), acc);
  // three bats on looping paths
  for (var i = 0; i < 3; i++) {
    let fi: f32 = f32(i);
    let bc: vec2f = vec2f(a * (0.5 + 0.36 * sin(t * (0.8 + fi * 0.17) + fi * 2.1)), 0.24 + 0.1 * sin(t * (1.9 + fi * 0.3) + fi));
    let vx: f32 = cos(t * (0.8 + fi * 0.17) + fi * 2.1);
    var rect: vec4f = u.batA;
    if (fract(t * 6.0 + fi * 0.33) > 0.5) { rect = u.batB; }
    acc = over(sprite(p, bc, 0.075, rect, step(vx, 0.0)), acc);
  }
  // the dashing hero
  let hx: f32 = a * (0.5 + 0.4 * sin(t * 1.4));
  let hv: f32 = cos(t * 1.4);
  acc = over(sprite(p, vec2f(hx, 0.757), 0.09, u.heroRect, step(hv, 0.0)), acc);
  return acc;
}`;

const FB_PASS = /* wgsl */ `
${FB_COMMON}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex: i32 = i32(u.example);
  let t: f32 = u.time;
  let a: f32 = u.resolution.x / u.resolution.y;
  let p: vec2f = vec2f(uv.x * a, uv.y);
  let fade: f32 = pow(u.decay, u.dt * 60.0);          // frame-rate independent fading
  let prev: vec4f = TEX(fb, uv);

  if (ex == 0) {
    // MOTION TRAILS: fade what is there, then add this frame's sprites
    let spr: vec4f = spritesLayer(p, t);
    var keep: vec4f = prev * fade;
    if (u.trailMode < 0.5) { return over(spr, keep); }                      // smooth smear
    if (u.trailMode < 1.5) {                                                   // afterimages
      if (u.stamp > 0.5) { return over(vec4f(u.tint * spr.a, spr.a) * 0.85, keep); }
      return keep;
    }
    return vec4f(keep.rgb + spr.rgb * 0.5, max(keep.a, spr.a));               // additive glow
  }

  if (ex == 1) {
    // PAINT SMEAR: pull pixels along the brush motion, slowly heal back to the live game
    let live: vec3f = TEX(game, uv).rgb;
    if (u.frame < 1.5) { return vec4f(live, 1.0); }
    let d: f32 = length(px - u.brush.xy) / max(u.brushR, 1.0);
    let w: f32 = exp(-d * d) * u.brushOn;
    let src: vec2f = uv - u.brush.zw / u.resolution * w * u.smear;
    var c: vec3f = TEX(fb, src).rgb;
    c = mix(c, live, 1.0 - pow(1.0 - u.heal, u.dt * 60.0));
    let ink: vec3f = hsv2rgb(vec3f(fract(t * 0.15 + uv.x * 0.2), 0.85, 1.0));
    c = mix(c, ink, w * u.paint * 0.6);
    return vec4f(c, 1.0);
  }

  if (ex == 2) {
    // INFINITE TUNNEL: draw last frame zoomed + rotated + shifted + hue-shifted, faded, then a new seed
    var d: vec2f = (uv - 0.5) * vec2f(a, 1.0);
    d = rot2(-u.rotation) * d / u.zoom;
    let suv: vec2f = 0.5 + d / vec2f(a, 1.0) - vec2f(u.offX, u.offY) * 0.01;
    var c: vec3f = vec3f(0.0);
    if (suv.x > 0.0 && suv.x < 1.0 && suv.y > 0.0 && suv.y < 1.0) { c = TEX(fb, suv).rgb; }
    c = hueRotate(c, u.hueDrift) * fade;
    // seed
    let hue: vec3f = hsv2rgb(vec3f(fract(t * 0.12), 0.8, 1.0));
    let cp: vec2f = p - vec2f(a * 0.5, 0.5);
    if (u.seed < 0.5) {
      let sd: f32 = abs(sdStar5(rot2(t * 0.8) * cp, 0.12, 0.45)) - 0.004;
      c = max(c, hue * (1.0 - smoothstep(0.0, 0.006, sd)));
    } else if (u.seed < 1.5) {
      let rd: f32 = abs(length(cp) - 0.08 - 0.02 * sin(t * 3.0)) - 0.004;
      c = max(c, hue * (1.0 - smoothstep(0.0, 0.006, rd)));
    } else {
      let q: vec2f = cp / vec2f(a, 1.0) / 0.28 + 0.5;
      if (q.x > 0.0 && q.x < 1.0 && q.y > 0.0 && q.y < 1.0) {
        c = TEX(game, q).rgb;
        let e: f32 = min(min(q.x, 1.0 - q.x), min(q.y, 1.0 - q.y));
        c = mix(vec3f(1.0), c, smoothstep(0.0, 0.015, e));
      }
    }
    // the mouse is a brush too
    let md: f32 = length(px - u.mouse.xy) / u.resolution.y;
    c = max(c, hsv2rgb(vec3f(fract(t * 0.12 + 0.5), 0.7, 1.0)) * exp(-md / 0.008) * u.mouse.w);
    return vec4f(c, 1.0);
  }

  // DREAM: the live game blended with a warped, zoomed, hue-drifting copy of the previous frame
  let live: vec3f = TEX(game, uv).rgb;
  if (u.frame < 1.5) { return vec4f(live, 1.0); }
  var d: vec2f = (uv - 0.5) * vec2f(a, 1.0);
  d = rot2(-u.rotation) * d / u.zoom;
  let warp: vec2f = vec2f(perlin(p * 3.0 + vec2f(t * 0.2, 0.0)), perlin(p * 3.0 + vec2f(5.2, t * 0.2))) * u.warp * 0.004;
  let suv: vec2f = 0.5 + d / vec2f(a, 1.0) + warp;
  let old: vec3f = hueRotate(TEX(fb, suv).rgb, u.hueDrift);
  return vec4f(mix(live, old, pow(u.blend, u.dt * 60.0)), 1.0);
}`;

const IMAGE = /* wgsl */ `
${FB_COMMON}
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex: i32 = i32(u.example);
  let t: f32 = u.time;
  let a: f32 = u.resolution.x / u.resolution.y;
  let p: vec2f = vec2f(uv.x * a, uv.y);
  let fb0: vec4f = TEX(fb, uv);
  var c: vec3f = TEX(game, uv).rgb;
  if (ex == 0) {
    if (u.trailMode > 1.5) { c += fb0.rgb; }
    else { c = c * (1.0 - fb0.a * 0.85) + fb0.rgb * 0.85; }
    let spr: vec4f = spritesLayer(p, t);
    c = c * (1.0 - spr.a) + spr.rgb;
  } else if (ex == 1) {
    c = fb0.rgb;
    // brush cursor
    let d: f32 = abs(length(px - u.brush.xy) - u.brushR) ;
    c = mix(c, vec3f(1.0), (1.0 - smoothstep(0.5, 1.8, d)) * 0.6 * u.brushVis);
  } else if (ex == 2) {
    c = fb0.rgb;
    c *= 1.0 - 0.3 * smoothstep(0.4, 1.0, length((uv - 0.5) * vec2f(a, 1.0)) / (0.5 * a));
  } else {
    c = fb0.rgb;
    c = mix(vec3f(luma(c)), c, 1.2);
    c *= 1.0 - 0.35 * smoothstep(0.45, 1.0, length((uv - 0.5) * vec2f(a, 1.0)) / (0.5 * a));
  }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export default shaderScene({
  interaction: 'Feedback: every frame is drawn on top of the previous one.',
  examples: [
    {
      id: 'trails',
      label: 'Motion trails',
      kind: 'In a game',
      note: 'A transparent “trail layer” that is never cleared, only faded a little each frame. Sprites are drawn into it, so they leave smears behind — or, if we only stamp every few frames, the classic dash <b>afterimages</b>.',
      params: { trailMode: 'after', decay: 0.93, spacing: 0.06, tint: '#44e0ff' },
      hint: 'Switch the trail style on the right.',
    },
    {
      id: 'paint',
      label: 'Paint smear',
      kind: 'Interactive',
      note: 'Drag on the canvas to smudge the world like wet paint: each frame, pixels under the brush are re-read from slightly behind the brush’s motion, and everything slowly heals back towards the live game.',
      params: { brushSize: 0.08, smear: 1.2, heal: 0.01, paint: 0, autoBrush: true },
      hint: 'Drag on the canvas to smudge the scene.',
    },
    {
      id: 'tunnel',
      label: 'Infinite tunnel',
      kind: 'Abstract',
      note: 'Point a camera at its own monitor: each frame redraws the previous one slightly zoomed and rotated, so a single shape in the middle repeats into an endless tunnel. Move the mouse to paint into it.',
      params: { seed: 'star', zoom: 1.03, rotation: 0.035, offX: 0, offY: 0, decay: 0.965, hueDrift: 0.02 },
      hint: 'Move the mouse to paint into the feedback loop.',
    },
    {
      id: 'dream',
      label: 'Dream sequence',
      kind: 'In a game',
      note: 'The whole game blended with a slowly zooming, warping, hue-shifting copy of its own past frames. Ghostly trails follow everything that moves — dream sequences, drugs, hallucinations, time powers.',
      params: { blend: 0.88, zoom: 1.006, rotation: 0.003, hueDrift: 0.03, warp: 1 },
      hint: '',
    },
  ],
  controls: [
    {
      type: 'select',
      key: 'trailMode',
      label: 'Trail style',
      value: 'after',
      options: [
        { value: 'smear', label: 'Smooth smear (every frame)' },
        { value: 'after', label: 'Afterimages (snapshots)' },
        { value: 'glow', label: 'Additive glow' },
      ],
      showFor: ['trails'],
    },
    { type: 'slider', key: 'decay', label: 'Persistence', min: 0.5, max: 0.995, step: 0.001, value: 0.93, showFor: ['trails', 'tunnel'], help: 'How much of the previous frame survives (per 1/60 s). Higher = longer trails.' },
    { type: 'slider', key: 'spacing', label: 'Afterimage interval', min: 0.02, max: 0.25, step: 0.005, value: 0.06, unit: 's', showFor: ['trails'], help: 'Afterimages only: how often a snapshot is stamped.' },
    { type: 'color', key: 'tint', label: 'Afterimage tint', value: '#44e0ff', showFor: ['trails'] },
    { type: 'slider', key: 'brushSize', label: 'Brush size', min: 0.02, max: 0.25, step: 0.005, value: 0.08, showFor: ['paint'] },
    { type: 'slider', key: 'smear', label: 'Smear strength', min: 0, max: 3, step: 0.01, value: 1.2, showFor: ['paint'] },
    { type: 'slider', key: 'heal', label: 'Heal speed', min: 0, max: 0.1, step: 0.001, value: 0.01, showFor: ['paint'], help: 'How fast the smudge relaxes back to the live picture. 0 = permanent.' },
    { type: 'slider', key: 'paint', label: 'Rainbow paint', min: 0, max: 1, step: 0.01, value: 0, showFor: ['paint'], help: 'Also deposit color along the stroke.' },
    { type: 'toggle', key: 'autoBrush', label: 'Ghost brush when idle', value: true, showFor: ['paint'] },
    {
      type: 'select',
      key: 'seed',
      label: 'Seed shape',
      value: 'star',
      options: [
        { value: 'star', label: 'Spinning star' },
        { value: 'ring', label: 'Pulsing ring' },
        { value: 'game', label: 'Game picture (infinite mirror)' },
      ],
      showFor: ['tunnel'],
    },
    { type: 'slider', key: 'zoom', label: 'Zoom per frame', min: 0.95, max: 1.08, step: 0.001, value: 1.03, showFor: ['tunnel', 'dream'], help: '>1: the past grows outward (flying in). <1: it shrinks away.' },
    { type: 'slider', key: 'rotation', label: 'Rotation per frame', min: -0.1, max: 0.1, step: 0.001, value: 0.035, unit: 'rad', showFor: ['tunnel', 'dream'] },
    { type: 'slider', key: 'offX', label: 'Drift X', min: -1, max: 1, step: 0.01, value: 0, showFor: ['tunnel'] },
    { type: 'slider', key: 'offY', label: 'Drift Y', min: -1, max: 1, step: 0.01, value: 0, showFor: ['tunnel'] },
    { type: 'slider', key: 'hueDrift', label: 'Hue drift', min: -0.2, max: 0.2, step: 0.001, value: 0.02, showFor: ['tunnel', 'dream'], help: 'Rotate the colors of the old frame a bit each time — rainbow echoes.' },
    { type: 'slider', key: 'blend', label: 'Echo amount', min: 0, max: 0.97, step: 0.005, value: 0.88, showFor: ['dream'], help: 'How much of the warped past is mixed with the live game.' },
    { type: 'slider', key: 'warp', label: 'Warp', min: 0, max: 4, step: 0.01, value: 1, showFor: ['dream'] },
    { type: 'button', key: 'reset', label: 'Clear feedback buffer' },
  ],
  uniforms: {
    trailMode: 'f32', decay: 'f32', tint: 'vec3f', smear: 'f32', heal: 'f32', paint: 'f32', seed: 'f32', zoom: 'f32', rotation: 'f32',
    offX: 'f32', offY: 'f32', hueDrift: 'f32', blend: 'f32', warp: 'f32',
    stamp: 'f32', batA: 'vec4f', batB: 'vec4f', heroRect: 'vec4f', brush: 'vec4f', brushR: 'f32', brushOn: 'f32', brushVis: 'f32',
  },
  include: withGameIncludes(['math', 'hash', 'noise', 'sdf', 'color']),
  textures: { atlas: { source: async () => (await getAtlas()).canvas, filter: 'nearest' } },
  passes: [gamePass(), { name: 'fb', code: FB_PASS }],
  bind(p, ctx) {
    ov.begin(ctx);
    if (!st.requested) {
      st.requested = true;
      getAtlas().then((a) => {
        st.rects = { batA: a.uv('bat_0'), batB: a.uv('bat_1'), hero: ['hero_run_0', 'hero_run_1', 'hero_run_2', 'hero_run_3'].map((n) => a.uv(n)) };
      });
    }
    const t = ctx.time;
    const out = { stamp: 0, batA: st.rects.batA || ZERO, batB: st.rects.batB || ZERO, heroRect: st.rects.hero ? st.rects.hero[Math.floor(t * 12) % 4] : ZERO, brushOn: 0, brushVis: 0 };
    if (ctx.example === 'trails') {
      // afterimages: stamp a snapshot every `spacing` seconds
      const k = Math.floor(t / p.spacing);
      if (k !== st.lastStamp) {
        st.lastStamp = k;
        out.stamp = 1;
      }
    }
    if (ctx.example === 'paint') {
      const ptr = ctx.pointer;
      let pos;
      let on = 0;
      if (ptr.down && ptr.over) {
        pos = [ptr.x, ptr.y];
        on = 1;
      } else if (p.autoBrush && !ptr.over) {
        st.autoT += ctx.dt;
        const T = st.autoT;
        pos = [ctx.width * (0.5 + 0.38 * Math.sin(T * 0.9)), ctx.height * (0.55 + 0.25 * Math.sin(T * 1.7))];
        on = 1;
      } else pos = [ptr.x, ptr.y];
      const last = st.last || pos;
      const vel = [pos[0] - last[0], pos[1] - last[1]];
      st.last = pos;
      out.brush = [pos[0], pos[1], vel[0], vel[1]];
      out.brushOn = on;
      out.brushVis = ptr.over || on ? 1 : 0;
      out.brushR = p.brushSize * ctx.height;
    }
    ov.end();
    return out;
  },
  code: IMAGE,
  about: {
    summary:
      'Feedback means feeding a pass its own previous output. Fade it, nudge it, draw something new on top, repeat — the simplest way to get trails, smears, echoes and infinite tunnels, with no history of positions stored anywhere.',
    what: `<p>A full-screen buffer that is <b>never cleared</b>. <b>Motion trails</b> keeps a fading layer of sprites; <b>Paint smear</b> keeps a smudgeable
      copy of the game; <b>Infinite tunnel</b> redraws the last frame zoomed and rotated; <b>Dream sequence</b> mixes the live game with its own warped past.</p>`,
    how: `<ol>
      <li>Keep two textures and swap them every frame (<b>ping-pong</b>): read last frame from one, write this frame into the other. A texture can’t be read and written in the same pass.</li>
      <li><b>Fade</b>: multiply what you read by a decay factor (0.9 = trails vanish in ~0.5 s). Use <code>pow(decay, dt·60)</code> so trails look the same at 30 or 144 fps.</li>
      <li><b>Transform</b> the lookup: read last frame at a slightly zoomed / rotated / shifted position and the whole history flows — that is the tunnel.</li>
      <li><b>Add</b> new content on top: sprites, a brush stroke, a seed shape, or the live game.</li>
      <li><b>Afterimages</b> are the same layer, but the sprite is only stamped every N milliseconds and tinted, giving distinct ghost copies (Celeste, Mega Man X dash).</li>
    </ol>`,
    uses: [
      { title: 'Movement', text: 'Dash afterimages, speed trails, sword arcs, comet tails, cursor trails.' },
      { title: 'Atmosphere', text: 'Dream/flashback sequences, drug and poison effects, time-slow “echoes”, ghosts.' },
      { title: 'Psychedelic & music', text: 'Visualizers, menu backgrounds and tunnels — the classic Winamp/demoscene look.' },
      { title: 'Painting & persistence', text: 'Footprints, tire marks, blood splats or paint that stays — a buffer you only ever add to.' },
    ],
    try: [
      'On <b>Motion trails</b> switch between Smooth smear, Afterimages and Additive glow, and push Persistence to 0.99.',
      'On <b>Infinite tunnel</b> pick “Game picture” as the seed: an infinite mirror of the level.',
      'Set Zoom below 1 on the tunnel: the history shrinks away instead of flying at you.',
      'On <b>Paint smear</b> set Heal speed to 0 and draw a permanent swirl; add Rainbow paint for color.',
      'Pause (top right) on the <b>Dream sequence</b>: feedback stops because the passes stop running, but the last frame stays.',
    ],
    ask: [
      'dash afterimages that fade out behind the player',
      'motion trails using a feedback buffer instead of storing old positions',
      'a dreamy echo effect where the screen leaves fading ghost copies',
      'an infinite zooming feedback tunnel for a menu background',
      'let the player smudge the screen with the mouse',
    ],
    perf: `<p>One extra full-screen pass and one extra full-screen texture (two for ping-pong). Cost doesn’t depend on how many things leave trails —
      that’s the big advantage over storing a history of positions per object. Use a half-resolution buffer if the trails are soft anyway.</p>`,
    api: `<p>Identical in WebGPU and WebGL2: two render targets swapped each frame. A 16-bit float buffer (used here) avoids the banding and “never quite
      fades to zero” problem that 8-bit buffers have when you multiply by 0.97 every frame.</p>`,
    code: [
      {
        title: 'Trail layer: fade the past, draw the present on top',
        lang: 'wgsl',
        src: `let fade = pow(u.decay, u.dt * 60.0);     // same look at any frame rate
let prev = TEX(fb, uv) * fade;              // fb = THIS pass's previous output
let spr = spritesLayer(p, u.time);          // premultiplied sprites for this frame
return spr + prev * (1.0 - spr.a);          // "over" blend`,
      },
      {
        title: 'Infinite tunnel',
        lang: 'wgsl',
        src: `var d = (uv - 0.5) * vec2f(aspect, 1.0);
d = rot2(-u.rotation) * d / u.zoom;         // read the past slightly zoomed & rotated
let suv = 0.5 + d / vec2f(aspect, 1.0);
var c = hueRotate(TEX(fb, suv).rgb, u.hueDrift) * fade;
c = max(c, seedShape(p));                   // something new in the middle`,
      },
    ],
  },
});

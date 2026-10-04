import { shaderScene } from '../../core/shaderscene.js';

// God rays / light shafts (GPU Gems 3, ch. 13 "Volumetric Light Scattering as a Post-Process"):
//   pass occ  : occlusion mask — the light source (sun, window, water surface) bright, everything that blocks it black
//   pass rays : radial blur of that mask toward the light's screen position (with decay per sample)
//   image     : the colored scene + rays, plus dust motes that sparkle only inside the beams
// Written in portable WGSL, so it runs on WebGPU and WebGL2.

const SCENE = /* wgsl */ `
fn gr_aa(d: f32, w: f32) -> f32 { return clamp(0.5 - d / w, 0.0, 1.0); }

// ---------------------------------------------------------------- forest
fn forestLeaves(q: vec2f, t: f32) -> f32 {
  let sway: vec2f = vec2f(sin(t * 0.7 + q.y * 3.0) * 0.012, 0.0);
  let n: f32 = fbm((q + sway) * vec2f(4.0, 6.0), 4) * 0.5 + 0.5;
  let band: f32 = smoothstep(0.95, 0.1, q.y);
  return smoothstep(0.5, 0.56, n * 0.6 + band * 0.56);
}
fn forestTrunks(q: vec2f, A: f32, pw: f32) -> f32 {
  var m: f32 = 0.0;
  for (var i = 0; i < 8; i++) {
    let fi: f32 = f32(i);
    let x: f32 = (0.04 + 0.13 * fi + 0.04 * sin(fi * 7.3)) * A;
    let w: f32 = 0.01 + 0.014 * fract(fi * 0.618);
    let lean: f32 = 0.03 * sin(fi * 2.1);
    let d: f32 = sdBox(q - vec2f(x + (q.y - 1.0) * lean, 0.55), vec2f(w + q.y * 0.012, 0.55));
    m = max(m, gr_aa(d, pw));
  }
  return m;
}
fn forestGroundY(x: f32) -> f32 { return 0.86 + 0.03 * sin(x * 5.0) + 0.02 * perlin(vec2f(x * 6.0, 2.0)); }
fn forestOcc(q: vec2f, A: f32, pw: f32, t: f32) -> f32 {
  return max(max(forestLeaves(q, t), forestTrunks(q, A, pw)), gr_aa(forestGroundY(q.x) - q.y, pw));
}

// ---------------------------------------------------------------- window
fn windowPanes(q: vec2f, A: f32, pw: f32) -> f32 {
  let c: vec2f = vec2f(0.3 * A, 0.33);
  let hs: vec2f = vec2f(0.13, 0.17);
  let inner: f32 = sdBox(q - c, hs - vec2f(0.012));
  // mullions: a cross dividing the window into 4 panes
  let bars: f32 = min(abs(q.x - c.x), abs(q.y - c.y)) - 0.008;
  return gr_aa(max(inner, -bars), pw);
}

// ---------------------------------------------------------------- underwater
fn surfaceY(x: f32, t: f32) -> f32 { return 0.07 + 0.01 * sin(x * 9.0 + t * 1.3) + 0.006 * sin(x * 23.0 - t * 2.1); }
fn kelp(q: vec2f, A: f32, pw: f32, t: f32) -> f32 {
  var m: f32 = 0.0;
  for (var i = 0; i < 7; i++) {
    let fi: f32 = f32(i);
    let x0: f32 = (0.08 + 0.14 * fi + 0.03 * sin(fi * 5.1)) * A;
    let top: f32 = 0.35 + 0.2 * fract(fi * 0.37);
    let k: f32 = clamp((1.0 - q.y) / (1.0 - top), 0.0, 1.0);
    let x: f32 = x0 + sin(q.y * 9.0 + t * 1.1 + fi) * 0.025 * k;
    let w: f32 = 0.008 + 0.006 * sin(q.y * 40.0 + fi);
    let d: f32 = max(abs(q.x - x) - w, top - q.y);
    m = max(m, gr_aa(d, pw));
  }
  return m;
}
fn fishes(q: vec2f, A: f32, pw: f32, t: f32) -> f32 {
  var m: f32 = 0.0;
  for (var i = 0; i < 9; i++) {
    let fi: f32 = f32(i);
    let span: f32 = A + 0.3;
    let x: f32 = fmod(fi * 0.21 * A + t * (0.05 + 0.02 * fract(fi * 0.7)), span) - 0.15;
    let y: f32 = 0.3 + 0.12 * sin(fi * 1.7) + 0.02 * sin(t + fi);
    let p: vec2f = q - vec2f(x, y);
    let body: f32 = sdEllipseApprox(p, vec2f(0.03, 0.011));
    let tail: f32 = sdTriangle(p, vec2f(-0.026, 0.0), vec2f(-0.045, -0.012), vec2f(-0.045, 0.012));
    m = max(m, gr_aa(min(body, tail), pw));
  }
  return m;
}
fn boat(q: vec2f, A: f32, pw: f32, t: f32) -> f32 {
  let c: vec2f = vec2f(0.62 * A + 0.02 * sin(t * 0.4), surfaceY(0.62 * A, t) - 0.005);
  let hull: f32 = max(sdEllipseApprox(q - c, vec2f(0.17, 0.05)), c.y - q.y);
  return gr_aa(hull, pw);
}
fn waterOcc(q: vec2f, A: f32, pw: f32, t: f32) -> f32 {
  let rocks: f32 = gr_aa(0.9 + 0.05 * perlin(vec2f(q.x * 4.0, 1.0)) - q.y, pw);
  return max(max(kelp(q, A, pw, t), fishes(q, A, pw, t)), max(boat(q, A, pw, t), rocks));
}
fn caustics(q: vec2f, t: f32) -> f32 {
  let v: vec4f = voronoiEx(q * vec2f(7.0, 9.0), 1.0, t * 0.8);
  return smoothstep(0.18, 0.0, v.y - v.x);
}
`;

const OCC = /* wgsl */ `
${SCENE}
// The occlusion mask: bright = light source, black = anything that blocks it.
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let H: f32 = u.resolution.y;
  let A: f32 = u.resolution.x / H;
  let q: vec2f = uv * u.resolution / H;
  let lq: vec2f = u.light / H;
  let pw: f32 = 3.0 / H;
  let t: f32 = u.time;
  let ex: i32 = i32(u.example + 0.5);
  let dl: f32 = length(q - lq);
  if (ex == 0) {
    let sun: f32 = smoothstep(0.065, 0.055, dl) + 0.8 * exp(-dl * 6.0) + 0.04;
    return vec4f(vec3f(1.0, 0.9, 0.7) * sun * (1.0 - forestOcc(q, A, pw, t)), 1.0);
  }
  if (ex == 1) {
    let glow: f32 = 0.55 + 0.6 * exp(-dl * 5.0);
    return vec4f(vec3f(1.0, 0.93, 0.8) * glow * windowPanes(q, A, pw), 1.0);
  }
  // underwater: the bright surface is the source
  let s: f32 = surfaceY(q.x, t);
  let surf: f32 = smoothstep(s + 0.03, s - 0.01, q.y);
  let rip: f32 = 0.65 + 0.35 * caustics(vec2f(q.x, 0.3), t);
  let src: f32 = surf * rip * (0.7 + 0.6 * exp(-abs(q.x - lq.x) * 3.0));
  return vec4f(vec3f(src) * (1.0 - waterOcc(q, A, pw, t)), 1.0);
}`;

const RAYS = /* wgsl */ `
// Radial blur toward the light: march from this pixel to the light, summing the mask with a decaying weight.
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let lp: vec2f = u.light / u.resolution;           // light position in uv
  let n: f32 = floor(u.samples);
  let delta: vec2f = (uv - lp) * u.density / n;      // step toward the light
  let decay: f32 = pow(u.decay, 64.0 / n);           // same look for any sample count
  let w: f32 = u.weight * 64.0 / n;
  var tc: vec2f = uv - delta * ign(px);               // jitter the start: noise instead of banding
  var illum: f32 = 1.0;
  var sum: vec3f = vec3f(0.0);
  for (var i = 0; i < 128; i++) {
    if (f32(i) >= n) { break; }
    tc = tc - delta;
    sum = sum + TEX(occ, tc).rgb * illum * w;
    illum = illum * decay;
  }
  return vec4f(sum * u.exposure, 1.0);
}`;

const IMAGE = /* wgsl */ `
${SCENE}
fn dust(q: vec2f, t: f32, scale: f32) -> f32 {
  let g: vec2f = q * scale + vec2f(t * 0.05, t * 0.02);
  let cell: vec2f = floor(g);
  let f: vec2f = fract(g);
  let h: vec3f = hash23(cell);
  let pos: vec2f = vec2f(0.5) + 0.32 * vec2f(sin(t * (0.3 + h.x) + h.y * 6.0), cos(t * (0.25 + h.y) + h.x * 6.0));
  let d: f32 = length(f - pos);
  return smoothstep(0.09, 0.0, d) * step(0.55, h.z);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let H: f32 = u.resolution.y;
  let A: f32 = u.resolution.x / H;
  let q: vec2f = px / H;
  let lq: vec2f = u.light / H;
  let pw: f32 = 1.5 / H;
  let t: f32 = u.time;
  let ex: i32 = i32(u.example + 0.5);
  let view: i32 = i32(u.view + 0.5);
  let shafts: vec3f = TEX(rays, uv).rgb;
  if (view == 1) { return vec4f(TEX(occ, uv).rgb, 1.0); }
  if (view == 2) { return vec4f(shafts * u.tint, 1.0); }
  let dl: f32 = length(q - lq);
  var col: vec3f = vec3f(0.0);
  if (ex == 0) {
    // sky, sun, layered forest
    col = mix(vec3f(0.98, 0.78, 0.5), vec3f(0.42, 0.58, 0.82), smoothstep(0.65, 0.0, q.y));
    col = col + vec3f(1.0, 0.8, 0.5) * exp(-dl * 6.0) * 0.5;
    col = mix(col, vec3f(1.0, 0.97, 0.88), smoothstep(0.065, 0.055, dl));
    // distant tree line in haze
    let far: f32 = gr_aa(0.62 + 0.04 * fbm(vec2f(q.x * 6.0, 0.0), 3) - q.y, pw);
    col = mix(col, vec3f(0.55, 0.6, 0.55), far * 0.7);
    let leaves: f32 = forestLeaves(q, t);
    let lv: f32 = fbm(q * 18.0, 2) * 0.5 + 0.5;
    col = mix(col, mix(vec3f(0.05, 0.12, 0.06), vec3f(0.16, 0.3, 0.1), lv), leaves);
    col = mix(col, vec3f(0.1, 0.07, 0.05), forestTrunks(q, A, pw));
    let gy: f32 = forestGroundY(q.x);
    col = mix(col, mix(vec3f(0.12, 0.16, 0.06), vec3f(0.04, 0.06, 0.03), smoothstep(gy, 1.0, q.y)), gr_aa(gy - q.y, pw));
    col = col + vec3f(0.5, 0.45, 0.35) * smoothstep(0.55, 0.95, q.y) * 0.12;   // ground mist
  } else if (ex == 1) {
    // a dark room with a window
    let plaster: f32 = 0.85 + 0.15 * fbm(q * 9.0, 3);
    col = vec3f(0.16, 0.11, 0.08) * plaster;
    let floorY: f32 = 0.8;
    if (q.y > floorY) {
      let plank: f32 = fract(q.x * 7.0 + floor(q.y * 30.0) * 0.37);
      col = mix(vec3f(0.14, 0.08, 0.05), vec3f(0.2, 0.12, 0.07), smoothstep(0.0, 0.05, plank) * (0.7 + 0.3 * fbm(q * vec2f(3.0, 40.0), 2)));
      col = col * (1.0 - 0.6 * smoothstep(floorY + 0.03, floorY, q.y));
    }
    let wc: vec2f = vec2f(0.3 * A, 0.33);
    let frame: f32 = gr_aa(sdBox(q - wc, vec2f(0.145, 0.185)), pw);
    col = mix(col, vec3f(0.24, 0.16, 0.1), frame);
    let panes: f32 = windowPanes(q, A, pw);
    var sky: vec3f = mix(vec3f(1.0, 0.92, 0.75), vec3f(0.6, 0.78, 1.0), smoothstep(0.5, 0.15, q.y));
    sky = sky + vec3f(1.0, 0.9, 0.7) * exp(-dl * 7.0);
    col = mix(col, sky, panes);
    // a table and a chair, in silhouette
    let table: f32 = min(sdBox(q - vec2f(0.72 * A, 0.66), vec2f(0.13, 0.012)), min(sdBox(q - vec2f(0.72 * A - 0.11, 0.73), vec2f(0.008, 0.07)), sdBox(q - vec2f(0.72 * A + 0.11, 0.73), vec2f(0.008, 0.07))));
    let chair: f32 = min(min(sdBox(q - vec2f(0.52 * A, 0.7), vec2f(0.045, 0.008)), sdBox(q - vec2f(0.52 * A - 0.04, 0.68), vec2f(0.007, 0.12))), sdBox(q - vec2f(0.52 * A + 0.04, 0.75), vec2f(0.006, 0.05)));
    col = mix(col, vec3f(0.06, 0.04, 0.03), gr_aa(min(table, chair), pw));
    // dust motes: only visible where a beam passes
    let beam: f32 = luma(shafts);
    let motes: f32 = dust(q, t, 55.0) + 0.6 * dust(q + vec2f(3.1, 1.7), t * 0.7, 90.0);
    col = col + vec3f(1.0, 0.9, 0.75) * motes * beam * 2.5 * u.dust;
  } else {
    // underwater
    col = mix(vec3f(0.08, 0.42, 0.55), vec3f(0.0, 0.05, 0.11), smoothstep(0.05, 1.0, q.y));
    let bed: f32 = gr_aa(0.9 + 0.05 * perlin(vec2f(q.x * 4.0, 1.0)) - q.y, pw);
    let sand: vec3f = vec3f(0.12, 0.17, 0.16) + vec3f(0.25, 0.4, 0.38) * caustics(q * 1.5, t) * 0.6;
    col = mix(col, sand, bed);
    col = mix(col, vec3f(0.02, 0.12, 0.08), kelp(q, A, pw, t));
    col = mix(col, vec3f(0.03, 0.08, 0.12), fishes(q, A, pw, t));
    let s: f32 = surfaceY(q.x, t);
    let surf: f32 = smoothstep(s + 0.004, s - 0.004, q.y);
    col = mix(col, vec3f(0.55, 0.85, 0.95) * (0.8 + 0.3 * caustics(vec2f(q.x, 0.3), t)), surf);
    col = mix(col, vec3f(0.05, 0.05, 0.07), boat(q, A, pw, t));
    // bubbles
    for (var i = 0; i < 6; i++) {
      let fi: f32 = f32(i);
      let bx: f32 = (0.15 + 0.13 * fi) * A + 0.01 * sin(t * 3.0 + fi);
      let by: f32 = 1.0 - fract(t * (0.08 + 0.03 * fract(fi * 0.7)) + fi * 0.37);
      let bd: f32 = abs(length(q - vec2f(bx, by)) - 0.006) - 0.0015;
      col = col + vec3f(0.6, 0.9, 1.0) * gr_aa(bd, pw) * 0.6;
    }
  }
  col = col + shafts * u.tint;
  col = col + (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}`;

let lightState = null;

export default shaderScene({
  interaction: 'Move the mouse: the sun follows it. In the room, the beams point at the mouse.',
  examples: [
    {
      id: 'forest',
      label: 'Forest sunbeams',
      kind: 'In a game',
      note: 'The sun behind a swaying canopy. The mask contains only the sun and the sky seen <i>between</i> the leaves; blurring it toward the sun turns every gap into a beam.',
      params: { samples: 64, density: 0.9, decay: 0.965, weight: 0.06, exposure: 1.1, tint: '#ffd49a', view: 'final' },
    },
    {
      id: 'window',
      label: 'Window light in a dusty room',
      kind: 'Real life',
      note: 'Light pours through the window panes; the window cross splits it into four shafts. Dust motes are drawn everywhere but only <b>light up inside the beams</b> (they read the ray texture).',
      params: { samples: 72, density: 1.2, decay: 0.975, weight: 0.05, exposure: 1.4, tint: '#ffdcae', view: 'final' },
    },
    {
      id: 'underwater',
      label: 'Underwater',
      kind: 'In a game',
      note: 'The bright, rippling water surface is the light source; kelp, fish and a boat hull cut it into shafts that wave with the surface. The light sits above the screen — radial blur works with off-screen lights too.',
      params: { samples: 64, density: 1.0, decay: 0.975, weight: 0.08, exposure: 1.3, tint: '#a6ecff', view: 'final' },
    },
    {
      id: 'mask',
      label: 'How it works: the mask',
      kind: 'Step 1',
      note: 'This is the <b>occlusion mask</b> of the forest: light source white, blockers black, rendered at half resolution. Everything you see as “rays” is just this image smeared toward the light.',
      params: { samples: 64, density: 0.9, decay: 0.965, weight: 0.045, exposure: 1, tint: '#ffd49a', view: 'mask' },
    },
  ],
  controls: [
    { type: 'slider', key: 'samples', label: 'Samples', min: 8, max: 128, step: 1, value: 64, help: 'Taps along each ray. Fewer = cheaper but grainier (the start is jittered to trade banding for noise).' },
    { type: 'slider', key: 'density', label: 'Density (ray length)', min: 0.1, max: 1.5, step: 0.01, value: 0.9, help: 'How far toward the light each pixel looks. 1 = all the way.' },
    { type: 'slider', key: 'decay', label: 'Decay', min: 0.85, max: 1, step: 0.001, value: 0.965, help: 'Each step counts a bit less: shorter, softer shafts.' },
    { type: 'slider', key: 'weight', label: 'Weight', min: 0.005, max: 0.15, step: 0.001, value: 0.045, help: 'Contribution of each sample.' },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: 0, max: 3, step: 0.01, value: 1, help: 'Overall brightness of the shafts.' },
    { type: 'color', key: 'tint', label: 'Light color', value: '#ffd49a' },
    { type: 'toggle', key: 'dust', label: 'Dust motes', value: true, showFor: ['window'] },
    {
      type: 'select',
      key: 'view',
      label: 'View',
      value: 'final',
      options: [
        { value: 'final', label: 'Final image' },
        { value: 'mask', label: '1. Occlusion mask' },
        { value: 'rays', label: '2. Rays only (radial blur)' },
      ],
    },
  ],
  uniforms: { samples: 'f32', density: 'f32', decay: 'f32', weight: 'f32', exposure: 'f32', tint: 'vec3f', dust: 'f32', view: 'f32', light: 'vec2f' },
  include: ['noise', 'sdf', 'color'],
  passes: [
    { name: 'occ', scale: 0.5, format: 'rgba8unorm', code: OCC },
    { name: 'rays', scale: 0.5, code: RAYS },
  ],
  resetOnExample: false,
  bind(params, ctx) {
    const ex = ctx.example === 'mask' ? 'forest' : ctx.example;
    const def = { forest: [0.62, 0.18], window: [0.1, 0.06], underwater: [0.45, -0.12] }[ex] || [0.5, 0.2];
    let target = ctx.pointer.over ? [ctx.pointer.x, ctx.pointer.y] : [def[0] * ctx.width, def[1] * ctx.height];
    if (ex === 'window' && ctx.pointer.over) {
      // the mouse aims the beams: put the sun on the opposite side of the window
      const wx = 0.3 * ctx.height;
      const wy = 0.33 * ctx.height;
      target = [wx - (target[0] - wx) * 0.8, wy - (target[1] - wy) * 0.8];
    }
    if (ex === 'underwater') target = [target[0], -0.12 * ctx.height]; // the sun stays above the surface
    if (!lightState || lightState.ex !== ex) lightState = { x: target[0], y: target[1], ex };
    const k = 1 - Math.exp(-8 * Math.max(ctx.dt, 1 / 60));
    lightState.x += (target[0] - lightState.x) * k;
    lightState.y += (target[1] - lightState.y) * k;
    return { light: [lightState.x, lightState.y], example: ctx.example === 'mask' ? 0 : ctx.exampleIndex };
  },
  code: IMAGE,
  about: {
    summary:
      'God rays are beams of light visible in dusty or misty air. In 2D (and in many 3D games) they are faked with one cheap trick: blur an image of the light source toward the light.',
    what: `<p>Bright shafts radiating from the sun through leaves, from a window into a room, and down from the water surface.
      Move the light: the beams swing around because they are recomputed every frame from the light’s screen position.</p>`,
    how: `<ol>
      <li><b>Occlusion mask</b> (half resolution): draw the light source and the bright sky in white/color and every occluder in black.</li>
      <li><b>Radial blur</b>: for every pixel, walk in a straight line toward the light’s screen position, sampling the mask
        (<code>samples</code> times, <code>density</code> = how far), each sample weighted a little less (<code>decay</code>).
        A pixel behind a gap in the leaves picks up the bright gap many times → a beam. This is GPU Gems 3, chapter 13.</li>
      <li><b>Composite</b>: add the rays on top of the normal scene, tinted with the light color. Dust motes multiply by the rays so they sparkle only inside beams.</li>
      <li><b>Jitter</b>: each pixel starts its walk at a random offset, which turns ugly stair-step banding into fine noise.</li>
    </ol>`,
    uses: [
      { title: 'Forests & ruins', text: 'Ori and the Blind Forest, Rayman Legends, Hollow Knight: shafts through canopies and broken roofs.' },
      { title: 'Interiors', text: 'Churches, dusty attics, prison cells — light through windows and bars.' },
      { title: 'Underwater', text: 'Shimmering beams from the surface (Abzû, Subnautica-like 2D games).' },
      { title: 'Drama', text: 'Boss reveals, holy light, a door opening into a bright room.' },
    ],
    try: [
      'Move the light behind the thickest part of the canopy: the rays get thin and sparse. Move it into a gap: they flood in.',
      'Set <i>Samples</i> to 8: grainy. 128: smooth. The start jitter keeps it from banding.',
      'Lower <i>Decay</i> to 0.9: beams become short glows around the gaps.',
      'Switch View to <i>Occlusion mask</i> and move the mouse — that tiny image is all the effect needs.',
      'On <b>Window light</b>, toggle Dust motes and watch them appear only inside the beams.',
    ],
    ask: [
      'screen-space god rays (GPU Gems 3 radial blur)',
      'light shafts through trees that follow the sun',
      'dust particles that only show inside light beams',
      'underwater light rays from the surface',
      'volumetric-looking window light in a dark room',
    ],
    perf: `<p>Cost = <i>half-resolution pixels × samples</i>: at 1080p with 64 samples ≈ 33 million texture reads — about 1 ms on a desktop
      GPU. Rendering the mask and rays at half (or quarter) resolution is the main trick; the blur hides the low resolution.</p>`,
    api: `<p>A pure fragment-shader effect: identical on WebGL2 and WebGPU (this scene is one WGSL source auto-translated to GLSL).
      Multiple render passes at reduced resolution are all it needs.</p>`,
    code: [
      { title: 'Radial blur toward the light (GPU Gems 3 style)', lang: 'wgsl', src: RAYS.trim() },
      {
        title: 'Occlusion mask (forest)',
        lang: 'wgsl',
        src: `let sun = smoothstep(0.065, 0.055, dl) + 0.5 * exp(-dl * 9.0) + 0.1;   // disc + halo + sky
return vec4f(vec3f(1.0, 0.9, 0.7) * sun * (1.0 - forestOcc(q, A, pw, t)), 1.0);`,
      },
    ],
    links: [{ title: 'GPU Gems 3 — Volumetric Light Scattering as a Post-Process', url: 'https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process' }],
  },
});

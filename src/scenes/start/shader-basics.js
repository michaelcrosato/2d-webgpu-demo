import { shaderScene } from '../../core/shaderscene.js';

// Shaders 101 — the exemplar for fragment-only scenes (runs on WebGPU and WebGL2).

export default shaderScene({
  interaction: 'Move the mouse over the canvas — the shader receives its position as a uniform.',
  examples: [
    {
      id: 'uv',
      label: 'UV & gradients',
      kind: 'Abstract',
      note: 'Every pixel gets its own <code>uv</code> (0→1 across, 0→1 down). Color = position: red grows to the right, green grows downwards.',
    },
    {
      id: 'shapes',
      label: 'step, smoothstep & circles',
      kind: 'Abstract',
      note: 'A circle is just “is my distance to the center less than r?”. <code>step</code> gives a hard edge; <code>smoothstep</code> a soft, anti-aliased one.',
    },
    {
      id: 'waves',
      label: 'sin() & time',
      kind: 'Abstract',
      note: 'Feed position and time into <code>sin()</code> and you get moving waves — the basis of water, plasma and countless effects.',
    },
    {
      id: 'game',
      label: 'Power-up orb',
      kind: 'In a game',
      note: 'The same three ideas — distance, smoothstep and sin(time) — combined into a pulsing pickup with a glow and a sparkle ring.',
    },
  ],
  controls: [
    { type: 'slider', key: 'speed', label: 'Animation speed', min: 0, max: 4, step: 0.01, value: 1, help: 'Multiplies the time uniform.' },
    { type: 'slider', key: 'radius', label: 'Radius', min: 0.05, max: 0.45, step: 0.005, value: 0.25, showFor: ['shapes', 'game'] },
    { type: 'slider', key: 'softness', label: 'Edge softness', min: 0, max: 0.2, step: 0.001, value: 0.01, showFor: ['shapes', 'game'], help: '0 = hard step() edge (jaggy). Higher = smoothstep() blur.' },
    { type: 'slider', key: 'frequency', label: 'Wave frequency', min: 1, max: 60, step: 0.5, value: 12, showFor: ['waves', 'uv'] },
    { type: 'toggle', key: 'grid', label: 'Show pixel-cell grid', value: false, showFor: ['uv'], help: 'Visualise that the shader runs separately for each cell (here: blocks of 16×16 pixels).' },
    { type: 'color', key: 'tint', label: 'Tint', value: '#7c9cff', showFor: ['shapes', 'waves', 'game'] },
  ],
  uniforms: { speed: 'f32', radius: 'f32', softness: 'f32', frequency: 'f32', grid: 'f32', tint: 'vec3f' },
  include: ['math', 'color'],
  code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let t = u.time * u.speed;
  let ex = i32(u.example);
  // aspect-correct coordinates centered on the screen (y down), height = 1
  let p = centerUV(px, u.resolution);
  let m = centerUV(u.mouse.xy, u.resolution);

  if (ex == 0) {
    var c = vec3f(uv.x, uv.y, 0.5 + 0.5 * sin(t));
    let stripes = 0.5 + 0.5 * sin(uv.x * u.frequency * TAU);
    c = mix(c, c * stripes, 0.25);
    if (u.grid > 0.5) {
      let cell = fract(px / 16.0);
      let line = step(cell.x, 0.06) + step(cell.y, 0.06);
      c = mix(c, vec3f(0.0), clamp(line, 0.0, 1.0) * 0.6);
    }
    // mouse crosshair
    let d = abs(px - u.mouse.xy);
    let xhair = (1.0 - smoothstep(0.0, 1.5, min(d.x, d.y))) * u.mouse.w;
    return vec4f(mix(c, vec3f(1.0), xhair * 0.8), 1.0);
  }

  if (ex == 1) {
    let bg = vec3f(0.06, 0.07, 0.1) + 0.04 * step(0.0, sin(p.x * 40.0) * sin(p.y * 40.0));
    // left: hard step() edge, right: smoothstep() edge
    let left = vec2f(-0.33 * u.resolution.x / u.resolution.y, 0.0);
    let right = vec2f(0.33 * u.resolution.x / u.resolution.y, 0.0);
    let d1 = length(p - left);
    let d2 = length(p - right);
    let hard = 1.0 - step(u.radius, d1);
    let soft = 1.0 - smoothstep(u.radius - u.softness - 0.002, u.radius + u.softness + 0.002, d2);
    // the circle under the mouse
    let dm = length(p - m);
    let follow = (1.0 - smoothstep(u.radius * 0.4 - u.softness, u.radius * 0.4 + u.softness + 0.002, dm)) * u.mouse.w;
    var c = bg;
    c = mix(c, u.tint, hard);
    c = mix(c, u.tint, soft);
    c = mix(c, vec3f(1.0, 0.8, 0.3), follow);
    return vec4f(c, 1.0);
  }

  if (ex == 2) {
    let w1 = sin(p.x * u.frequency + t * 2.0);
    let w2 = sin((p.x * 0.6 + p.y) * u.frequency * 0.8 - t * 1.6);
    let w3 = sin(length(p - m) * u.frequency * 1.5 - t * 4.0);
    let v = (w1 + w2 + w3) / 3.0;
    let c = mix(vec3f(0.03, 0.04, 0.12), u.tint, 0.5 + 0.5 * v);
    let hi = smoothstep(0.85, 1.0, v);
    return vec4f(c + vec3f(hi), 1.0);
  }

  // ex == 3: power-up orb
  let bob = 0.03 * sin(t * 2.0);
  let q = p - vec2f(0.0, bob);
  let d = length(q);
  let pulse = 1.0 + 0.06 * sin(t * 6.0);
  let r = u.radius * 0.6 * pulse;
  var c = mix(vec3f(0.02, 0.03, 0.07), vec3f(0.07, 0.05, 0.12), uv.y);
  // glow: brightness falls off with distance outside the orb
  c += u.tint * exp(-max(d - r, 0.0) * 9.0) * 0.6;
  // orb body with a fake 3D highlight
  let body = 1.0 - smoothstep(r - u.softness - 0.003, r + u.softness + 0.003, d);
  let shade_ = 0.55 + 0.45 * clamp(1.0 - length(q - vec2f(-r * 0.35, -r * 0.35)) / r, 0.0, 1.0);
  c = mix(c, u.tint * shade_ + vec3f(0.15), body);
  let spec = 1.0 - smoothstep(0.0, r * 0.22, length(q - vec2f(-r * 0.38, -r * 0.42)));
  c += vec3f(spec) * body;
  // rotating sparkle ring
  let a = atan2(q.y, q.x);
  let ring = abs(d - r * 1.55) - 0.004;
  let dots = smoothstep(0.75, 1.0, sin(a * 8.0 + t * 3.0));
  c += u.tint * (1.0 - smoothstep(0.0, 0.01, ring)) * dots * 1.2;
  return vec4f(c, 1.0);
}`,
  about: {
    summary: 'A fragment shader is a tiny function that runs once for every pixel, in parallel, every frame. Everything else builds on this.',
    what: `<p>Four tiny programs. Each one receives a pixel’s coordinate and returns a color — that’s all a <b>fragment shader</b> does.
      The GPU runs it for <i>every</i> pixel at the same time (about two million times per frame at 1080p).</p>`,
    how: `<ol>
      <li><b>uv</b> is the pixel’s position from 0 to 1. Using it directly as a color gives the red/green gradient.</li>
      <li><b>Distance</b>: <code>length(p - center)</code> tells each pixel how far it is from a point. Compare with a radius → circle.</li>
      <li><b>step(edge, x)</b> returns 0 or 1 — a hard, jaggy edge. <b>smoothstep(a, b, x)</b> fades between them — anti-aliasing and glows.</li>
      <li><b>Uniforms</b> like <code>time</code> and <code>mouse</code> are the same for all pixels; they’re how JavaScript talks to the shader. Sliders on the right are uniforms too.</li>
      <li><b>sin(x)</b> turns a steadily increasing number into a repeating wave. Add <code>time</code> and it moves.</li>
    </ol>`,
    uses: [
      { title: 'Pickups & UI', text: 'Pulsing orbs, glowing buttons, selection rings — no textures needed.' },
      { title: 'Backgrounds', text: 'Animated gradients, plasma and waves for menus and title screens.' },
      { title: 'Everything else', text: 'Every effect in this showcase (lighting, water, CRT, bloom…) is these building blocks combined.' },
    ],
    try: [
      'On <b>step, smoothstep & circles</b>, drag <i>Edge softness</i> to 0 and look closely at the right circle’s edge — then increase it.',
      'On <b>UV & gradients</b>, enable the grid: each cell is computed independently and in parallel.',
      'Set <i>Animation speed</i> to 0 to freeze time — the shader is a pure function of its inputs.',
      'Switch the API toggle (top bar) to WebGL2: the same shader, auto-translated from WGSL to GLSL.',
    ],
    ask: [
      'a pulsing glowing pickup made in a shader',
      'an animated gradient background shader',
      'anti-aliased circles with smoothstep instead of textures',
      'pass the mouse position into the shader as a uniform',
    ],
    perf: `<p>Cost ≈ <i>pixels × work per pixel</i>. These shaders do a handful of math operations per pixel — trivial for any GPU.
      Watch the fps counter while lowering <i>Render resolution</i>: fewer pixels, less work.</p>`,
    api: `<p>This scene is written once in WGSL and automatically translated to GLSL for WebGL2. Fragment shaders work the same in both APIs —
      the differences show up with compute shaders (see <a href="#/s/compute-power">Compute Shaders</a>).</p>`,
    code: {
      title: 'A complete fragment shader (WGSL)',
      lang: 'wgsl',
      src: `fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let p = centerUV(px, u.resolution);        // -0.5..0.5, aspect-correct
  let d = length(p);                          // distance to the center
  let circle = 1.0 - smoothstep(0.24, 0.26, d); // soft edge = anti-aliasing
  let glow = exp(-max(d - 0.25, 0.0) * 9.0);  // light falling off outside
  let pulse = 0.5 + 0.5 * sin(u.time * 6.0);  // animate with time
  let color = vec3f(0.5, 0.6, 1.0) * (circle + glow * pulse);
  return vec4f(color, 1.0);
}`,
    },
    links: [
      { title: 'The Book of Shaders', url: 'https://thebookofshaders.com/', note: 'the classic gentle introduction' },
      { title: 'Inigo Quilez — articles', url: 'https://iquilezles.org/articles/', note: 'distance functions, noise, and much more' },
    ],
  },
});

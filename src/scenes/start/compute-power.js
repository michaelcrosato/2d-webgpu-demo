// Compute Shaders: WebGPU's Superpower — the exemplar for custom WebGPU scenes.
// Shows: storage buffers, a compute pass, drawing straight from a storage buffer (instancing),
// HDR additive accumulation + tone mapping, and a CPU (JavaScript) comparison mode.

const PARTICLE_WGSL = /* wgsl */ `
struct Particle { pos: vec2f, vel: vec2f };
`;

export default {
  interaction: 'Hold the mouse button to pull particles toward the cursor. Watch the fps and the update-time readout.',
  examples: [
    { id: 'gpu', label: 'GPU compute shader', kind: 'WebGPU', note: 'Each particle is updated by its own GPU thread — thousands run at the same time. The data never leaves GPU memory.' },
    {
      id: 'cpu',
      label: 'CPU JavaScript loop',
      kind: 'Comparison',
      note: 'The exact same physics, but done one particle at a time in JavaScript, then the whole array is uploaded to the GPU every frame. Push the count up and watch the frame time explode.',
      params: { count: 100000 },
    },
    {
      id: 'vortex',
      label: 'Magic vortex',
      kind: 'In a game',
      note: 'The same compute particles styled as a spell: a swirling vortex that implodes when you click. Games use this for portals, auras, loot explosions and weather.',
      params: { swirl: 1.6, pull: 0.55, damping: 0.985, size: 2, exposure: 1.4, hue: 0.78 },
    },
  ],
  controls: [
    { type: 'slider', key: 'count', label: 'Particles', min: 1000, max: 2000000, step: 1000, value: 500000, log: true, format: (v) => Math.round(v).toLocaleString(), help: 'Number of particles simulated & drawn every frame.' },
    { type: 'slider', key: 'pull', label: 'Gravity to center', min: 0, max: 2, step: 0.01, value: 0.35 },
    { type: 'slider', key: 'swirl', label: 'Swirl', min: 0, max: 3, step: 0.01, value: 0.8, help: 'Sideways force that makes orbits instead of collapse.' },
    { type: 'slider', key: 'damping', label: 'Damping', min: 0.95, max: 1, step: 0.0005, value: 0.995, help: 'Velocity kept each step (1 = no friction).' },
    { type: 'slider', key: 'size', label: 'Point size (px)', min: 1, max: 6, step: 0.5, value: 1.5 },
    { type: 'slider', key: 'exposure', label: 'Exposure', min: 0.1, max: 4, step: 0.01, value: 1, help: 'Particles add up in an HDR buffer; exposure maps that brightness to the screen.' },
    { type: 'slider', key: 'hue', label: 'Color', min: 0, max: 1, step: 0.01, value: 0.6 },
    { type: 'button', key: 'reset', label: 'Respawn particles', primary: true },
  ],
  about: {
    summary: 'A compute shader runs a function on thousands of GPU threads at once. WebGPU has them; WebGL2 does not. This is the single biggest reason WebGPU is “the future”.',
    what: `<p>Up to two million particles orbiting under gravity and swirl forces. In <b>GPU</b> mode a compute shader moves every particle in parallel;
      in <b>CPU</b> mode JavaScript loops over them one by one and re-uploads the array each frame.</p>`,
    how: `<ol>
      <li>All particles live in a <b>storage buffer</b> on the GPU: an array of <code>{pos, vel}</code> structs.</li>
      <li>Each frame a <b>compute pass</b> is dispatched with one thread per particle (in workgroups of 256). Each thread reads its particle, applies forces, writes it back.</li>
      <li>A <b>render pass</b> then draws one tiny quad per particle. The vertex shader reads the <i>same</i> storage buffer, so nothing is copied back to the CPU.</li>
      <li>Quads are blended <b>additively</b> into a 16-bit float (HDR) texture: dense areas get brighter than 1.0. A final pass <b>tone-maps</b> that to the screen.</li>
    </ol>
    <p>In WebGL2 you would have to fake this by storing positions in textures and “drawing” a full-screen quad to update them (or use transform feedback) — possible, but awkward and limited.</p>`,
    uses: [
      { title: 'Particles', text: 'Explosions, magic, sparks, smoke, weather, bullet hell — hundreds of thousands at once.' },
      { title: 'Simulations', text: 'Boids, fluids, cloth, sand, crowds, GPU physics.' },
      { title: 'GPU-driven rendering', text: 'Culling, sorting, building draw lists, tile lighting without a CPU round trip.' },
      { title: 'Non-graphics', text: 'Pathfinding fields, image processing, procedural generation, even machine learning.' },
    ],
    try: [
      'Switch to <b>CPU JavaScript loop</b> and raise the particle count to 1,000,000. Compare the update time readout with GPU mode.',
      'Set <i>Damping</i> to 1.0 and <i>Swirl</i> high: particles never lose energy and form stable rings.',
      'Hold the mouse to pull everything to the cursor, release to watch it rebound.',
      'Lower <i>Exposure</i> to see the raw density — the HDR buffer stores far more than the screen can show.',
    ],
    ask: [
      'move the particle simulation to a compute shader',
      'GPU particles with storage buffers and instanced rendering',
      'additive HDR particle accumulation with tone mapping',
      'a swirling vortex / black-hole particle effect',
    ],
    perf: `<p>GPU update cost is ~constant per particle and parallel: millions per millisecond on a desktop GPU. The CPU path is limited by
      JavaScript speed (a few ns per particle) <i>plus</i> uploading 16 bytes × N to the GPU every frame. Drawing is usually the real
      bottleneck: overdraw from large, overlapping additive quads costs fill-rate — try raising <i>Point size</i>.</p>`,
    api: `<p><b>WebGPU only.</b> Compute shaders and read/write storage buffers don’t exist in WebGL2. That is why the WebGL2 toggle is disabled here.</p>`,
    code: [
      {
        title: 'Compute shader: one thread per particle',
        lang: 'wgsl',
        src: `struct Particle { pos: vec2f, vel: vec2f };
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var p = particles[i];
  let toCenter = u.attractor - p.pos;
  let d2 = dot(toCenter, toCenter) + 0.02;
  let swirl = vec2f(-toCenter.y, toCenter.x);          // perpendicular
  p.vel += (toCenter * u.pull + swirl * u.swirl) / d2 * u.dt;
  p.vel *= u.damping;
  p.pos += p.vel * u.dt;
  particles[i] = p;
}`,
      },
      {
        title: 'JavaScript: dispatch it',
        lang: 'js',
        src: `const pass = encoder.beginComputePass();
pass.setPipeline(updatePipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(count / 256));   // e.g. 3,907 groups for 1M particles
pass.end();`,
      },
    ],
  },

  async init(ctx) {
    const gpu = ctx.gpu;
    const MAX = ctx.testMode ? 50000 : 2000000;
    const U = gpu.uniforms(
      {
        attractor: 'vec2f',
        aspect: 'f32',
        dt: 'f32',
        pull: 'f32',
        swirl: 'f32',
        damping: 'f32',
        count: 'u32',
        size: 'vec2f',
        hue: 'f32',
        mode: 'f32',
        exposure: 'f32',
        time: 'f32',
      },
      'Sim',
    );
    const particles = gpu.storage(MAX * 16, 'particles');
    const cpuData = new Float32Array(MAX * 4);

    const respawn = () => {
      for (let i = 0; i < MAX; i++) {
        const a = Math.random() * Math.PI * 2;
        const r = 0.15 + Math.pow(Math.random(), 0.6) * 0.85;
        const x = Math.cos(a) * r * 1.4;
        const y = Math.sin(a) * r * 0.9;
        const v = 0.55 / Math.sqrt(r);
        cpuData[i * 4] = x;
        cpuData[i * 4 + 1] = y;
        cpuData[i * 4 + 2] = -Math.sin(a) * v;
        cpuData[i * 4 + 3] = Math.cos(a) * v;
      }
      gpu.queue.writeBuffer(particles, 0, cpuData);
    };
    respawn();

    const update = gpu.compute({
      label: 'particles-update',
      bindings: { u: { uniform: U }, ps: { storage: 'array<Particle>', access: 'read_write' } },
      code:
        PARTICLE_WGSL +
        /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var p = ps[i];
  let to = u.attractor - p.pos;
  let d2 = dot(to, to) + 0.02;
  let side = vec2f(-to.y, to.x);
  p.vel += (to * u.pull + side * u.swirl) / d2 * u.dt;
  p.vel *= pow(u.damping, u.dt * 60.0);
  p.pos += p.vel * u.dt;
  ps[i] = p;
}`,
    });

    const draw = gpu.program({
      label: 'particles-draw',
      bindings: { u: { uniform: U }, ps: { storage: 'array<Particle>', access: 'read' } },
      include: ['color'],
      code:
        PARTICLE_WGSL +
        /* wgsl */ `
struct VOut { @builtin(position) pos: vec4f, @location(0) color: vec3f };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  let i = vi / 6u;
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let p = ps[i];
  let clip = vec2f(p.pos.x / u.aspect, -p.pos.y) + corners[vi % 6u] * u.size;
  var o: VOut;
  o.pos = vec4f(clip, 0.0, 1.0);
  let speed = length(p.vel);
  var h = u.hue + speed * 0.12;
  if (u.mode > 1.5) { h = u.hue + 0.15 * sin(atan2(p.pos.y, p.pos.x) * 3.0 + u.time); }
  o.color = hsv2rgb(vec3f(fract(h), 0.75, 1.0)) * (0.06 + speed * 0.05);
  return o;
}
@fragment fn fs_main(i: VOut) -> @location(0) vec4f { return vec4f(i.color, 1.0); }`,
    });
    const drawPipe = draw.renderPipeline({ format: 'rgba16float', blend: 'add' });

    const tonemap = gpu.fullscreen({
      label: 'tonemap',
      textures: ['hdr'],
      uniforms: U,
      include: ['color'],
      code: /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let c = TEX(hdr, uv).rgb * u.exposure;
  let bg = mix(vec3f(0.01, 0.012, 0.03), vec3f(0.03, 0.02, 0.06), uv.y);
  return vec4f(tonemapACES(c) + bg, 1.0);
}`,
    });

    let hdr = gpu.target(ctx.width, ctx.height, { format: 'rgba16float', label: 'hdr' });
    let cpuMs = 0;
    let target = [0, 0];

    const overlay = document.createElement('div');
    overlay.className = 'tag';
    overlay.style.cssText = 'right:8px;bottom:8px';
    ctx.overlay.append(overlay);

    const cpuStep = (n, dt, p) => {
      const t0 = performance.now();
      const d = cpuData;
      const damp = Math.pow(p.damping, dt * 60);
      for (let i = 0; i < n; i++) {
        const k = i * 4;
        const tx = target[0] - d[k];
        const ty = target[1] - d[k + 1];
        const d2 = tx * tx + ty * ty + 0.02;
        d[k + 2] = (d[k + 2] + ((tx * p.pull - ty * p.swirl) / d2) * dt) * damp;
        d[k + 3] = (d[k + 3] + ((ty * p.pull + tx * p.swirl) / d2) * dt) * damp;
        d[k] += d[k + 2] * dt;
        d[k + 1] += d[k + 3] * dt;
      }
      gpu.queue.writeBuffer(particles, 0, d, 0, n * 4);
      cpuMs = cpuMs * 0.9 + (performance.now() - t0) * 0.1;
    };

    let lastMode = ctx.example;
    return {
      resize(w, h) {
        hdr.destroy();
        hdr = gpu.target(w, h, { format: 'rgba16float', label: 'hdr' });
      },
      onAction(key) {
        if (key === 'reset') respawn();
      },
      onExample(id) {
        // CPU mode works on cpuData — sync it from scratch so both modes start the same
        if (id !== lastMode) respawn();
        lastMode = id;
      },
      frame(ctx) {
        const p = ctx.params;
        const n = Math.min(MAX, Math.round(p.count));
        const aspect = ctx.width / ctx.height;
        const dt = Math.min(ctx.dt, 1 / 30);
        // pointer -> simulation space (x in [-aspect, aspect], y in [-1, 1], y up)
        const mx = (ctx.pointer.nx * 2 - 1) * aspect;
        const my = -(ctx.pointer.ny * 2 - 1);
        const tgt = ctx.pointer.down ? [mx, my] : [0, 0];
        target = tgt;
        const pull = ctx.pointer.down ? p.pull * 3 + 0.5 : p.pull;
        U.set('attractor', tgt)
          .set('aspect', aspect)
          .set('dt', dt)
          .set('pull', pull)
          .set('swirl', p.swirl)
          .set('damping', p.damping)
          .set('count', n)
          .set('size', [p.size / ctx.width, p.size / ctx.height])
          .set('hue', p.hue)
          .set('mode', ctx.example === 'vortex' ? 2 : ctx.example === 'cpu' ? 1 : 0)
          .set('exposure', p.exposure)
          .set('time', ctx.time);
        U.upload();
        const enc = ctx.encoder;
        if (dt > 0) {
          if (ctx.example === 'cpu') cpuStep(n, dt, { ...p, pull });
          else update.dispatch(enc, 'main', Math.ceil(n / 256), { u: U, ps: particles });
        }
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: hdr.view, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }],
        });
        pass.setPipeline(drawPipe);
        pass.setBindGroup(0, draw.bind({ u: U, ps: particles }));
        pass.draw(n * 6);
        pass.end();
        tonemap.draw(enc, { view: ctx.target, format: gpu.format }, { hdr });
        overlay.textContent =
          ctx.example === 'cpu'
            ? `CPU update: ${cpuMs.toFixed(1)} ms / frame · ${n.toLocaleString()} particles`
            : `GPU update: ${Math.ceil(n / 256).toLocaleString()} workgroups × 256 threads · ${n.toLocaleString()} particles`;
      },
    };
  },
};

// Home page and Field Guide (glossary + "ask for it" index).

import { CATEGORIES, SCENES } from '../scenes/registry.js';
import { el } from './controls.js';
import { getWebGPU } from '../core/webgpu.js';

async function envInfo() {
  const out = { gpu: null, gpuErr: null, gl: null };
  try {
    const s = await getWebGPU();
    if (s.error) throw new Error(s.error);
    const i = s.info || {};
    out.gpu = [i.vendor, i.architecture, i.description].filter(Boolean).join(' · ') || 'available';
  } catch (e) {
    out.gpuErr = e.message;
  }
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      out.gl = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch {
    /* ignore */
  }
  return out;
}

/** A gallery tile: thumbnail (assets/thumbs/<id>.webp) + title + badges. */
function sceneTile(s) {
  const a = el('a', { class: 'tile', href: `#/s/${s.id}`, title: s.blurb });
  const img = el('img', { src: `assets/thumbs/${s.id}.webp`, alt: '', loading: 'lazy', decoding: 'async' });
  img.addEventListener('error', () => img.replaceWith(el('div', { class: 'tile-ph' })));
  const badges = el('span', { class: 'tile-badges' });
  if (s.badges.includes('compute')) badges.append(el('span', { class: 'mini compute' }, 'compute'));
  if (s.badges.includes('gl2')) badges.append(el('span', { class: 'mini gl2' }, 'gl2'));
  a.append(el('div', { class: 'tile-img' }, img, badges), el('div', { class: 'tile-t' }, s.title), el('div', { class: 'tile-b' }, s.blurb));
  return a;
}

export function renderHome(main) {
  const page = el('div', { class: 'page' });
  const hero = el('section', { class: 'hero' });
  hero.innerHTML = `
    <h1>The <span>2D GPU Showcase</span></h1>
    <p>A hands-on catalogue of what modern GPUs can do for 2D games and graphics — ${SCENES.length} live scenes across
    ${CATEGORIES.length} categories. Every scene explains <b>what</b> you are seeing, <b>how</b> it works, <b>where</b> you'd use it,
    and gives you the <b>vocabulary</b> to ask for it. Drag the sliders and break things — that’s how it clicks.</p>
    <p style="color:var(--muted);font-size:14px">Built on <b>WebGPU</b> (the modern API with compute shaders), with <b>WebGL2</b>
    versions of every scene that can run on it — flip the API switch in the top bar to compare.</p>`;
  const envs = el('div', { class: 'envs' });
  const gpuBox = el('div', { class: 'env' }, el('b', {}, 'WebGPU'), el('small', {}, 'checking…'));
  const glBox = el('div', { class: 'env' }, el('b', {}, 'WebGL2'), el('small', {}, 'checking…'));
  const startBox = el('div', { class: 'env' }, el('b', {}, 'New here?'), el('small', {}, ''));
  startBox.querySelector('small').innerHTML = 'Start with <a href="#/s/gpu-pipeline">How a GPU draws 2D</a> or jump to <a href="#/s/gpu-particles">a million particles</a>.';
  envs.append(gpuBox, glBox, startBox);
  hero.append(envs);
  envInfo().then((e) => {
    gpuBox.querySelector('small').innerHTML = e.gpu ? `<span class="ok">✓ available</span> — ${e.gpu}` : `<span class="no">✗ unavailable</span> — ${e.gpuErr}. WebGL2 versions will be used where possible.`;
    glBox.querySelector('small').innerHTML = e.gl ? `<span class="ok">✓ available</span> — ${e.gl}` : '<span class="no">✗ unavailable</span>';
  });
  const mosaic = el('div', { class: 'hero-mosaic', 'aria-hidden': 'true' });
  for (const id of ['holo-foil', 'gpu-particles', 'crt-retro', 'fluid-sim', 'mode7', 'global-illumination', 'neon-synthwave', 'falling-sand', 'bloom']) {
    const img = el('img', { src: `assets/thumbs/${id}.webp`, alt: '', loading: 'lazy' });
    img.addEventListener('error', () => img.remove());
    mosaic.append(img);
  }
  hero.prepend(mosaic);
  page.append(hero);

  page.append(el('h2', { class: 'section-title' }, 'How to use this showcase'));
  const how = el('div', { class: 'cols3' });
  how.innerHTML = `
    <div class="card"><h2><span class="ic">①</span>Pick a scene</h2><p>Browse the categories on the left or search (try “glow”, “water”, “pixel”). Most scenes have several <b>example tabs</b>: an abstract view of the technique plus real-world / in-game uses.</p></div>
    <div class="card"><h2><span class="ic">②</span>Play with it</h2><p>Every scene has <b>sliders</b> on the right. Double-click a slider to reset it. Many scenes react to the <b>mouse</b> — hints appear in the corner of the canvas.</p></div>
    <div class="card"><h2><span class="ic">③</span>Learn the words</h2><p>Below each canvas: what you’re seeing, how it works, where games use it, the key shader code, and <b>“Ask for it like…”</b> phrases you can paste into a request.</p></div>`;
  page.append(how);

  page.append(el('h2', { class: 'section-title' }, 'WebGPU vs WebGL2 in one minute'));
  const cmp = el('div', { class: 'card' });
  cmp.innerHTML = `
  <table class="compare">
    <tr><th></th><th>WebGL2 (2017)</th><th>WebGPU (2023+)</th></tr>
    <tr><td><b>Based on</b></td><td>OpenGL ES 3.0 (a 2012 mobile API)</td><td>Modern native APIs: Vulkan, Metal, Direct3D 12</td></tr>
    <tr><td><b>Shader language</b></td><td>GLSL ES 3.00</td><td>WGSL</td></tr>
    <tr><td><b>Compute shaders</b></td><td>No — simulations must be faked by drawing into textures</td><td>Yes — run arbitrary parallel programs on the GPU</td></tr>
    <tr><td><b>Storage buffers &amp; atomics</b></td><td>No — data must be packed into textures</td><td>Yes — read/write big arrays of structs, count, sort, scatter</td></tr>
    <tr><td><b>API style</b></td><td>Global state machine, lots of driver validation per call</td><td>Pre-validated pipelines &amp; bind groups → far less CPU overhead</td></tr>
    <tr><td><b>Support</b></td><td>Practically every browser and device</td><td>Chrome/Edge (2023), Safari 26 and Firefox 141+ (2025); still rolling out on some Linux/Android setups</td></tr>
    <tr><td><b>Use it for</b></td><td>Maximum compatibility; anything that is “draw sprites + fragment effects”</td><td>Huge particle counts, physics &amp; fluid sims, GPU-driven rendering, anything compute-heavy</td></tr>
  </table>
  <p style="margin-top:10px;color:var(--muted);font-size:14px">The good news: for most 2D effects the <i>ideas</i> are identical — a fragment shader is a fragment shader. Here, every pure-shader scene is written once in WGSL and automatically translated to GLSL for WebGL2 (look for the <span class="mini gl2">gl2</span> badge). Scenes marked <span class="mini compute">compute</span> need WebGPU.</p>`;
  page.append(cmp);

  page.append(el('h2', { class: 'section-title' }, 'Everything in the showcase'));
  for (const cat of CATEGORIES) {
    const items = SCENES.filter((s) => s.category === cat.id);
    const sec = el('section', { class: 'gallery-cat' });
    sec.innerHTML = `<h3><span class="dot" style="background:${cat.color}"></span>${cat.title}<span class="count">${items.length} scenes</span></h3><p>${cat.blurb}</p>`;
    const tiles = el('div', { class: 'tiles' });
    for (const s of items) tiles.append(sceneTile(s));
    sec.append(tiles);
    page.append(sec);
  }
  page.append(
    el('p', { style: 'margin-top:28px;color:var(--muted)' }, 'Looking for words to describe what you want? The ', el('a', { href: '#/guide' }, 'Field Guide'), ' has a glossary and every “ask for it” phrase in one place.'),
  );
  main.replaceChildren(page);
}

const GLOSSARY = [
  ['GPU', 'Graphics Processing Unit: thousands of small cores that run the same small program on lots of data at once. Perfect for “do this for every pixel / particle”.'],
  ['Shader', 'A small program that runs on the GPU. 2D games mostly use vertex, fragment and compute shaders.'],
  ['Vertex shader', 'Runs once per vertex (corner). Positions shapes on screen: moving, rotating, scaling sprites, wobbling grass.'],
  ['Fragment (pixel) shader', 'Runs once per covered pixel and returns its color. Almost every “effect” in this showcase is a fragment shader.'],
  ['Compute shader', 'A general-purpose GPU program with no fixed drawing job: update a million particles, simulate fluid, sort, count. WebGPU only.'],
  ['WGSL / GLSL', 'The shading languages of WebGPU and WebGL2. Same ideas, different syntax.'],
  ['Uniform', 'A value that is the same for every pixel/vertex in a draw — time, mouse position, a slider value.'],
  ['Texture', 'An image the GPU can read (sample) from. Also used to store data such as simulation state.'],
  ['UV coordinates', 'Position inside a texture or the screen from 0 to 1. Shaders think in UVs instead of pixels to stay resolution-independent.'],
  ['Sampler / filtering', 'How a texture is read between pixels: “nearest” keeps hard pixels (pixel art), “linear” blends smoothly.'],
  ['Mipmaps', 'Pre-shrunk copies of a texture so it does not shimmer when drawn small.'],
  ['Render target', 'An off-screen texture you draw into instead of the screen. The basis of post-processing, minimaps, portals.'],
  ['Ping-pong buffers', 'Two render targets that swap roles each step (read A → write B, then read B → write A). How simulations evolve over time.'],
  ['Draw call', 'One command to the GPU to draw something. Few big draw calls are fast; thousands of tiny ones are slow.'],
  ['Batching & instancing', 'Drawing many sprites with one draw call by putting them all in one buffer. The #1 2D performance trick.'],
  ['Blend mode', 'How a new color combines with what is already on screen: alpha, additive (glow), multiply (shadow), screen…'],
  ['Premultiplied alpha', 'Storing color already multiplied by its opacity. Avoids dark fringes and makes additive & normal blending mix.'],
  ['SDF (signed distance field)', 'A function giving the distance to a shape’s edge (negative inside). Gives perfectly crisp shapes, outlines and glows at any size.'],
  ['Noise', 'Smooth pseudo-random values (Perlin, simplex, value, Worley). The raw material for clouds, terrain, fire, water and wobble.'],
  ['fBm', 'Fractal Brownian motion: layering several octaves of noise at increasing frequency for natural detail.'],
  ['Normal map', 'A texture storing which way each pixel “faces”, so flat 2D sprites can be lit as if they had bumps.'],
  ['HDR & tone mapping', 'Rendering with brightness above 1.0 (sun, magic) then compressing it to the screen’s range. Enables realistic bloom.'],
  ['Bloom', 'Bright areas bleed light into their surroundings, made by blurring only the bright parts and adding them back.'],
  ['Post-processing', 'Effects applied to the finished image: color grading, blur, CRT, distortion, outlines.'],
  ['Kernel / convolution', 'A small grid of weights applied around each pixel. Blur, sharpen, emboss and edge detection are all kernels.'],
  ['LUT', 'Look-Up Table: a precomputed color mapping used for fast color grading (“make it look like a sunset”).'],
  ['Dithering', 'Using patterns of few colors to fake more colors or smooth gradients. Iconic in retro and 1-bit styles.'],
  ['Palette quantization', 'Snapping every color to the nearest color in a fixed palette (e.g. Game Boy’s 4 greens).'],
  ['Storage buffer', 'A big read/write array on the GPU (WebGPU). Holds particles, agents, grids, anything.'],
  ['Workgroup', 'A group of compute shader threads that run together and can share fast memory.'],
  ['Atomics', 'Operations that many GPU threads can safely do on the same memory at once (e.g. counting). WebGPU only.'],
  ['Delta time (dt)', 'Seconds since the last frame. Multiply motion by dt so games run the same speed at 30 or 144 fps.'],
  ['Feedback buffer', 'Feeding the previous frame back in as input — trails, smears, simulations and infinite tunnels.'],
  ['Atlas', 'One big texture holding many sprites/frames so they can be drawn in one batch.'],
  ['Tilemap', 'A grid of small reusable tiles (grass, wall…) used to build levels efficiently.'],
  ['Parallax', 'Layers that move at different speeds as the camera moves, creating an illusion of depth.'],
  ['Jump flooding (JFA)', 'A GPU algorithm that computes nearest-seed maps (Voronoi / distance fields) in log₂(size) passes.'],
];

export async function renderGuide(main) {
  const page = el('div', { class: 'page' });
  page.innerHTML = `<div class="hero"><h1>Field <span>Guide</span></h1><p>The vocabulary of 2D GPU graphics, plus every “ask for it” phrase from every scene — so you can describe exactly the look and feel you want in your game.</p></div>`;
  const glossary = el('div', { class: 'card', style: 'margin-top:18px' });
  glossary.innerHTML = '<h2><span class="ic">📖</span>Glossary</h2>';
  const dl = el('dl', { class: 'glossary' });
  for (const [t, d] of GLOSSARY) dl.append(el('dt', {}, t), el('dd', {}, d));
  glossary.append(dl);
  page.append(glossary);
  page.append(el('h2', { class: 'section-title' }, 'Ask for it — by scene'));
  const holder = el('div', {}, el('p', { style: 'color:var(--muted)' }, 'Loading scenes…'));
  page.append(holder);
  main.replaceChildren(page);

  const mods = await Promise.allSettled(SCENES.map((s) => s.load()));
  holder.replaceChildren();
  for (const cat of CATEGORIES) {
    const c = el('div', { class: 'card', style: 'margin-bottom:14px' });
    c.innerHTML = `<h2><span class="dot" style="background:${cat.color}"></span>${cat.title}</h2>`;
    SCENES.forEach((s, i) => {
      if (s.category !== cat.id) return;
      const r = mods[i];
      const scene = r.status === 'fulfilled' ? r.value.default : null;
      const ask = scene?.about?.ask || [];
      const row = el('div', { class: 'guide-scene' });
      row.innerHTML = `<h4><a href="#/s/${s.id}">${s.title}</a></h4><p style="margin:0 0 6px;color:var(--muted);font-size:14px">${s.blurb}</p>`;
      if (ask.length) {
        const l = el('div', { class: 'ask-list' });
        for (const a of ask) l.append(el('span', { class: 'ask' }, `“${a}”`));
        row.append(l);
      }
      c.append(row);
    });
    holder.append(c);
  }
}

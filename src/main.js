// App shell: layout, routing, sidebar, scene view (stage + controls + explanation).

import { CATEGORIES, SCENES, sceneById } from './scenes/registry.js';
import { Stage, defaultParams } from './core/stage.js';
import { buildControls, el } from './ui/controls.js';
import { renderExplain } from './ui/explain.js';
import { renderHome, renderGuide } from './ui/pages.js';
import { getWebGPU } from './core/webgpu.js';

const app = document.getElementById('app');
const store = {
  get(k, d) {
    try {
      const v = localStorage.getItem(`gpu2d:${k}`);
      return v === null ? d : JSON.parse(v);
    } catch {
      return d;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(`gpu2d:${k}`, JSON.stringify(v));
    } catch {
      /* ignore */
    }
  },
};

let preferredApi = store.get('api', 'webgpu');
let stage = null;
let current = null; // { meta, scene, backend }

// ------------------------------------------------------------------ layout
const top = el('header', { class: 'topbar' });
const menuBtn = el('button', { class: 'menu-btn', 'aria-label': 'Menu', onclick: () => document.body.classList.toggle('nav-open') }, '☰');
const brand = el('a', { class: 'brand', href: '#/' }, el('span', { class: 'brand-logo' }), el('span', {}, '2D GPU Showcase ', el('small', {}, 'WebGPU · WebGL2')));
const searchInput = el('input', { type: 'search', placeholder: `Search ${SCENES.length} scenes…  (try “glow”, “water”, “pixel”)`, 'aria-label': 'Search scenes' });
const search = el('div', { class: 'search' }, searchInput);
const apiSeg = el('div', { class: 'seg', title: 'Which graphics API runs the scene' });
const btnGPU = el('button', { onclick: () => switchApi('webgpu') }, 'WebGPU');
const btnGL = el('button', { onclick: () => switchApi('webgl2') }, 'WebGL2');
apiSeg.append(btnGPU, btnGL);
const gpuChip = el('span', { class: 'chip gpu-chip' }, el('span', { class: 'dot', style: 'background:#666' }), 'GPU…');
top.append(menuBtn, brand, search, el('div', { class: 'top-right' }, apiSeg, gpuChip));

const sidebar = el('aside', { class: 'sidebar' });
const main = el('main', { class: 'main' });
app.replaceChildren(top, sidebar, main);

// GPU availability chip
(async () => {
  let ok = false;
  let label = 'WebGPU unavailable';
  try {
    const s = await getWebGPU();
    if (!s.error) {
      ok = true;
      label = `WebGPU ✓ ${s.info?.vendor || ''}`.trim();
    }
  } catch {
    /* ignore */
  }
  gpuChip.replaceChildren(el('span', { class: 'dot', style: `background:${ok ? '#6ee7b7' : '#fca5a5'}` }), label);
  gpuChip.title = ok ? 'WebGPU is available in this browser' : 'WebGPU is not available: scenes with a WebGL2 version will use it';
  if (!ok && preferredApi === 'webgpu') preferredApi = 'webgl2';
})();

// ------------------------------------------------------------------ sidebar
const collapsed = new Set(store.get('collapsed', []));
function buildSidebar(filter = '') {
  sidebar.replaceChildren();
  const f = filter.trim().toLowerCase();
  const navTop = el(
    'div',
    { class: 'nav-top' },
    el('a', { href: '#/', 'data-route': 'home' }, '🏠  Home'),
    el('a', { href: '#/guide', 'data-route': 'guide' }, '📖  Field Guide & Glossary'),
  );
  sidebar.append(navTop);
  for (const cat of CATEGORIES) {
    const items = SCENES.filter(
      (s) => s.category === cat.id && (!f || `${s.title} ${s.blurb} ${cat.title} ${s.id}`.toLowerCase().includes(f)),
    );
    if (!items.length) continue;
    const wrap = el('div', { class: 'cat' + (collapsed.has(cat.id) && !f ? ' collapsed' : '') });
    const title = el(
      'div',
      { class: 'cat-title', title: cat.blurb },
      el('span', { class: 'dot', style: `background:${cat.color}` }),
      cat.title,
      el('span', { class: 'count' }, String(items.length)),
    );
    title.addEventListener('click', () => {
      wrap.classList.toggle('collapsed');
      if (wrap.classList.contains('collapsed')) collapsed.add(cat.id);
      else collapsed.delete(cat.id);
      store.set('collapsed', [...collapsed]);
    });
    const list = el('div', { class: 'cat-list' });
    for (const s of items) {
      const a = el('a', { href: `#/s/${s.id}`, 'data-id': s.id, title: s.blurb }, el('span', { class: 't' }, s.title));
      if (s.badges.includes('compute')) a.append(el('span', { class: 'mini compute' }, 'compute'));
      else if (s.badges.includes('gl2')) a.append(el('span', { class: 'mini gl2' }, 'gl2'));
      list.append(a);
    }
    wrap.append(title, list);
    sidebar.append(wrap);
  }
  markActive();
}
function markActive() {
  const route = parseRoute();
  sidebar.querySelectorAll('a').forEach((a) => {
    a.classList.toggle('active', (route.view === 'scene' && a.dataset.id === route.id) || a.dataset.route === route.view);
  });
  const act = sidebar.querySelector('a.active[data-id]');
  if (act) act.scrollIntoView({ block: 'nearest' });
}
searchInput.addEventListener('input', () => buildSidebar(searchInput.value));
searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const first = sidebar.querySelector('a[data-id]');
    if (first) location.hash = first.getAttribute('href');
  }
});

// ------------------------------------------------------------------ routing
function parseRoute() {
  const h = location.hash.replace(/^#\/?/, '');
  const [path, query = ''] = h.split('?');
  const q = Object.fromEntries(new URLSearchParams(query));
  const parts = path.split('/').filter(Boolean);
  if (parts[0] === 's' && parts[1]) return { view: 'scene', id: parts[1], q };
  if (parts[0] === 'guide') return { view: 'guide', q };
  return { view: 'home', q };
}

async function route() {
  const r = parseRoute();
  document.body.classList.remove('nav-open');
  markActive();
  if (r.view !== 'scene') {
    stage?.unload();
    current = null;
    updateApiButtons(null);
    if (r.view === 'guide') {
      document.title = 'Field Guide — 2D GPU Showcase';
      await renderGuide(main);
    } else {
      document.title = '2D GPU Showcase — WebGPU & WebGL2 effects explained';
      renderHome(main);
    }
    main.scrollTop = 0;
    return;
  }
  await openScene(r.id, r.q);
}
window.addEventListener('hashchange', route);

// ------------------------------------------------------------------ scene view
let view = null;
function buildSceneView() {
  const v = {};
  v.root = el('section', { class: 'scene-view' });
  v.crumbs = el('div', { class: 'crumbs' });
  v.title = el('h1');
  v.summary = el('p', { class: 'summary' });
  v.head = el('div', { class: 'scene-head' }, el('div', {}, v.crumbs, v.title), v.summary);
  v.tabs = el('div', { class: 'tabs' });
  v.stageHost = el('div', { class: 'stage' });
  v.fps = el('span', {}, '— fps');
  v.api = el('span', {}, '');
  v.res = el('span', {}, '');
  v.hud = el('div', { class: 'hud' }, v.fps, v.api, v.res);
  v.hint = el('div', { class: 'hint' });
  v.pause = el('button', { title: 'Pause / resume (simulation time stops)', onclick: () => togglePause() }, '⏸');
  v.restart = el('button', { title: 'Restart / reset the simulation', onclick: () => stage?.action('reset') }, '↻');
  v.full = el('button', { title: 'Fullscreen', onclick: () => toggleFullscreen() }, '⛶');
  v.shot = el('button', { title: 'Save a screenshot (PNG)', onclick: () => screenshot() }, '📷');
  v.tools = el('div', { class: 'stage-tools' }, v.pause, v.restart, v.full, v.shot);
  v.stageWrap = el('div', { class: 'stage-wrap' }, v.stageHost, v.hud, v.hint, v.tools);
  v.note = el('div', { class: 'note' });
  v.explain = el('div', { class: 'explain' });
  v.stageCol = el('div', { class: 'stage-col' }, v.tabs, v.stageWrap, v.note, v.explain);
  v.controls = el('div', { class: 'panel' });
  v.global = el('div', { class: 'panel' });
  v.controlsCol = el('aside', { class: 'controls-col' }, v.controls, v.global);
  v.body = el('div', { class: 'scene-body' }, v.stageCol, v.controlsCol);
  v.root.append(v.head, v.body);
  buildGlobalPanel(v.global);
  return v;
}

let timeScale = store.get('timeScale', 1);
let renderScale = store.get('renderScale', 1);
function buildGlobalPanel(p) {
  p.replaceChildren(el('h3', {}, 'Playback'));
  const mk = (label, key, value, min, max, step, fmt, onInput, help) => {
    const out = el('output', {}, fmt(value));
    const input = el('input', { type: 'range', min, max, step, value });
    input.style.setProperty('--fill', `${((value - min) / (max - min)) * 100}%`);
    input.addEventListener('input', () => {
      const v = +input.value;
      out.textContent = fmt(v);
      input.style.setProperty('--fill', `${((v - min) / (max - min)) * 100}%`);
      onInput(v);
      store.set(key, v);
    });
    return el('div', { class: 'ctl' }, el('div', { class: 'ctl-head' }, el('label', {}, label), out), input, el('div', { class: 'ctl-help' }, help));
  };
  p.append(
    mk('Time speed', 'timeScale', timeScale, 0, 3, 0.05, (v) => `${v.toFixed(2)}×`, (v) => {
      timeScale = v;
      if (stage) stage.timeScale = v;
    }, 'Slow motion helps you see how an effect evolves.'),
    mk('Render resolution', 'renderScale', renderScale, 0.25, 1, 0.05, (v) => `${Math.round(v * 100)}%`, (v) => {
      renderScale = v;
      stage?.setRenderScale(v);
    }, 'Lower = fewer pixels to shade. Watch the fps counter change.'),
  );
}

function togglePause() {
  if (!stage) return;
  stage.setPaused(!stage.paused);
  view.pause.textContent = stage.paused ? '▶' : '⏸';
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else view.stageWrap.requestFullscreen?.();
}
document.addEventListener('fullscreenchange', () => view?.stageWrap.classList.toggle('fs', !!document.fullscreenElement));
async function screenshot() {
  if (!stage) return;
  const blob = await stage.screenshot();
  if (!blob) return;
  const a = el('a', { href: URL.createObjectURL(blob), download: `${current?.meta.id || 'scene'}.png` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function toast(msg) {
  const t = el('div', { class: 'toast' }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 3200);
}

function updateApiButtons(scene) {
  const backends = scene ? scene.backends || (scene.initGL ? ['webgpu', 'webgl2'] : ['webgpu']) : ['webgpu', 'webgl2'];
  btnGPU.disabled = !backends.includes('webgpu');
  btnGL.disabled = !backends.includes('webgl2');
  btnGL.title = backends.includes('webgl2') ? 'Run this scene on WebGL2' : 'This scene needs WebGPU features (compute shaders / storage buffers) that WebGL2 lacks';
  const active = current?.backend || preferredApi;
  btnGPU.classList.toggle('on', active === 'webgpu');
  btnGL.classList.toggle('on', active === 'webgl2');
}

function setUrlQuery(updates) {
  const r = parseRoute();
  const q = { ...r.q, ...updates };
  for (const k of Object.keys(q)) if (q[k] === null || q[k] === undefined) delete q[k];
  const qs = new URLSearchParams(q).toString();
  history.replaceState(null, '', `#/s/${r.id}${qs ? '?' + qs : ''}`);
}

async function switchApi(api) {
  preferredApi = api;
  store.set('api', api);
  if (!current) return updateApiButtons(null);
  setUrlQuery({ api });
  await loadIntoStage({ keepParams: true });
}

async function openScene(id, q) {
  const meta = sceneById(id);
  if (!view) view = buildSceneView();
  if (!main.contains(view.root)) main.replaceChildren(view.root);
  if (!meta) {
    view.title.textContent = 'Scene not found';
    return;
  }
  const cat = CATEGORIES.find((c) => c.id === meta.category);
  document.title = `${meta.title} — 2D GPU Showcase`;
  view.crumbs.replaceChildren(el('span', { class: 'dot', style: `background:${cat.color}` }), cat.title);
  view.title.textContent = meta.title;
  view.summary.textContent = meta.blurb;
  view.explain.replaceChildren(el('div', { class: 'card' }, 'Loading…'));
  view.tabs.replaceChildren();
  view.controls.replaceChildren();
  view.note.textContent = '';
  view.hint.textContent = '';
  let mod;
  try {
    mod = await meta.load();
  } catch (e) {
    console.error(e);
    view.explain.replaceChildren(
      el('div', { class: 'card wide' }, `This scene could not be loaded (${e.message}). It may still be under construction.`),
    );
    stage?.unload();
    return;
  }
  if (parseRoute().id !== id) return; // navigated away meanwhile
  const scene = mod.default;
  current = { meta, scene, backend: null, example: null };
  view.summary.textContent = scene.about?.summary || meta.blurb;
  if (!stage) {
    stage = new Stage(view.stageHost, {
      onStatus: (t) => toast(t),
      onFps: (fps, ms) => {
        view.fps.textContent = `${fps.toFixed(0)} fps`;
        view.fps.title = `CPU time to record the frame: ${ms.toFixed(2)} ms`;
        if (stage?.ctx) view.res.textContent = `${stage.ctx.width}×${stage.ctx.height}`;
      },
    });
  }
  stage.timeScale = timeScale;
  stage.setRenderScale(renderScale);
  const examples = scene.examples || [];
  current.example = examples.find((e) => e.id === q.ex)?.id || examples[0]?.id || null;
  const idx = SCENES.indexOf(meta);
  renderExplain(view.explain, scene, { prev: SCENES[idx - 1] || null, next: SCENES[idx + 1] || null });
  view.hint.textContent = scene.interaction || '';
  buildTabs();
  if (q.api) preferredApi = q.api;
  await loadIntoStage({ keepParams: false });
  main.scrollTop = 0;
}

function buildTabs() {
  const scene = current.scene;
  const examples = scene.examples || [];
  view.tabs.replaceChildren();
  if (examples.length > 1) {
    for (const ex of examples) {
      const b = el('button', { class: ex.id === current.example ? 'on' : '' });
      if (ex.kind) b.append(el('span', { class: 'k' }, ex.kind));
      b.append(ex.label);
      b.addEventListener('click', () => selectExample(ex.id));
      view.tabs.append(b);
    }
  }
  const ex = examples.find((e) => e.id === current.example);
  view.note.innerHTML = ex?.note ? `<b>${ex.label}:</b> ${ex.note}` : '';
}

function selectExample(id) {
  if (!current || id === current.example) return;
  current.example = id;
  setUrlQuery({ ex: id });
  buildTabs();
  const ex = (current.scene.examples || []).find((e) => e.id === id);
  if (current.scene.reinitOnExample) {
    loadIntoStage({ keepParams: false });
    return;
  }
  stage.setExample(id);
  if (ex?.hint !== undefined) view.hint.textContent = ex.hint;
  else view.hint.textContent = current.scene.interaction || '';
  rebuildControls();
}

function rebuildControls() {
  if (!stage?.ctx) {
    buildControls(view.controls, current.scene, defaultParams(current.scene, current.example), current.example, { onChange() {}, onAction() {}, onReset() {} });
    return;
  }
  buildControls(view.controls, current.scene, stage.ctx.params, current.example, {
    onChange: (k, v) => stage.setParam(k, v),
    onAction: (k) => stage.action(k),
    onReset: () => {
      const d = defaultParams(current.scene, current.example);
      for (const [k, v] of Object.entries(d)) stage.setParam(k, v);
      rebuildControls();
    },
  });
}

async function loadIntoStage({ keepParams }) {
  const params = keepParams && stage?.ctx ? { ...stage.ctx.params } : undefined;
  const scene = current.scene;
  const res = await stage.load(scene, { backend: preferredApi, example: current.example, params });
  if (!res) return;
  current.backend = res.backend;
  view.api.textContent = res.backend === 'webgpu' ? 'WebGPU' : res.backend === 'webgl2' ? 'WebGL2' : 'no GPU';
  view.pause.textContent = stage.paused ? '▶' : '⏸';
  const ex = (scene.examples || []).find((e) => e.id === current.example);
  if (ex?.hint !== undefined) view.hint.textContent = ex.hint;
  updateApiButtons(scene);
  rebuildControls();
}

buildSidebar();
route();

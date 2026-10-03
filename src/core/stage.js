// Stage: owns the canvas, the graphics context, the frame loop, input and errors for ONE scene
// at a time. Scenes are plain objects (see AUTHORING.md) with init(ctx) / initGL(ctx).

import { getWebGPU, SceneGPU, setGPUErrorSink } from './webgpu.js';
import { GLKit, setGLErrorSink } from './webgl2.js';

const MAX_PIXELS = 2560 * 1600;

export function defaultParams(scene, exampleId) {
  const p = {};
  for (const c of scene.controls || []) if (c.key && c.value !== undefined) p[c.key] = c.value;
  const ex = (scene.examples || []).find((e) => e.id === exampleId);
  if (ex?.params) Object.assign(p, ex.params);
  return p;
}

export class Stage {
  constructor(host, { onStatus = () => {}, onFps = () => {} } = {}) {
    this.host = host;
    this.onStatus = onStatus;
    this.onFps = onFps;
    this.canvas = null;
    this.instance = null;
    this.scene = null;
    this.ctx = null;
    this.paused = false;
    this.timeScale = 1;
    this.renderScale = 1;
    this.errors = [];
    this._token = 0;
    this._raf = 0;
    this._dirty = true;
    this._fps = { frames: 0, last: performance.now(), ms: 0 };
    this.overlay = document.createElement('div');
    this.overlay.className = 'stage-errors';
    this.overlay.hidden = true;
    this.message = document.createElement('div');
    this.message.className = 'stage-message';
    this.message.hidden = true;
    this.sceneOverlay = document.createElement('div');
    this.sceneOverlay.className = 'scene-overlay';
    host.append(this.sceneOverlay, this.overlay, this.message);

    const sink = (msg) => this.reportError(msg);
    setGPUErrorSink(sink);
    setGLErrorSink(sink);

    this._ro = new ResizeObserver(() => {
      this._dirty = true;
    });
    this._ro.observe(host);
    this._keydown = (e) => this._onKey(e, true);
    this._keyup = (e) => this._onKey(e, false);
    window.addEventListener('keydown', this._keydown);
    window.addEventListener('keyup', this._keyup);
    window.__showcase = window.__showcase || {};
    window.__showcase.stage = this;
  }

  reportError(msg) {
    const text = String(msg);
    if (this.errors.includes(text) || this.errors.length > 30) return;
    this.errors.push(text);
    console.error(text);
    this.overlay.hidden = false;
    const pre = document.createElement('pre');
    pre.textContent = text;
    this.overlay.append(pre);
    window.__showcase.errors = this.errors;
  }

  showMessage(html) {
    this.message.innerHTML = html;
    this.message.hidden = !html;
  }

  _clearErrors() {
    this.errors = [];
    window.__showcase.errors = this.errors;
    this.overlay.replaceChildren();
    this.overlay.hidden = true;
  }

  _onKey(e, down) {
    if (!this.ctx) return;
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (down) {
      this.ctx.keys.add(k);
      this.ctx.keys.add(e.code);
      this.ctx.keysPressed.add(k);
      this.ctx.keysPressed.add(e.code);
    } else {
      this.ctx.keys.delete(k);
      this.ctx.keys.delete(e.code);
    }
    const focused = document.activeElement === this.canvas || this.ctx.pointer.over;
    if (focused && this.scene?.keys && /^(Arrow|Space| |w|a|s|d)/i.test(e.key === ' ' ? 'Space' : e.key)) e.preventDefault();
    this._dirty = true;
  }

  _makeCanvas() {
    this.canvas?.remove();
    const c = document.createElement('canvas');
    c.className = 'stage-canvas';
    c.tabIndex = 0;
    this.host.prepend(c);
    this.canvas = c;
    const pointer = this.ctx.pointer;
    const toPx = (e) => {
      const r = c.getBoundingClientRect();
      const sx = c.width / Math.max(1, r.width);
      const sy = c.height / Math.max(1, r.height);
      return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
    };
    const move = (e) => {
      const [x, y] = toPx(e);
      pointer.dx += x - pointer.x;
      pointer.dy += y - pointer.y;
      pointer.x = x;
      pointer.y = y;
      pointer.nx = x / Math.max(1, c.width);
      pointer.ny = y / Math.max(1, c.height);
      this._dirty = true;
    };
    c.addEventListener('pointermove', move);
    c.addEventListener('pointerdown', (e) => {
      move(e);
      pointer.dx = pointer.dy = 0;
      pointer.down = true;
      pointer.button = e.button;
      pointer.clicked = true;
      c.setPointerCapture(e.pointerId);
      c.focus({ preventScroll: true });
    });
    const up = (e) => {
      if (!pointer.down) return;
      move(e);
      pointer.down = false;
      pointer.released = true;
    };
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    c.addEventListener('pointerenter', () => (pointer.over = true));
    c.addEventListener('pointerleave', () => (pointer.over = false));
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener(
      'wheel',
      (e) => {
        if (!this.scene?.wheel) return;
        e.preventDefault();
        pointer.wheel += e.deltaY;
        this._dirty = true;
      },
      { passive: false },
    );
    return c;
  }

  _size() {
    const r = this.host.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * this.renderScale;
    let w = Math.max(16, Math.round(r.width * dpr));
    let h = Math.max(16, Math.round(r.height * dpr));
    if (w * h > MAX_PIXELS) {
      const k = Math.sqrt(MAX_PIXELS / (w * h));
      w = Math.round(w * k);
      h = Math.round(h * k);
    }
    return [w, h, dpr];
  }

  /**
   * Load a scene module. opts: { backend: 'webgpu'|'webgl2', example, params }
   * Returns { backend } actually used.
   */
  async load(scene, opts = {}) {
    const token = ++this._token;
    this.unload();
    this._clearErrors();
    this.showMessage('');
    this.scene = scene;
    const backends = scene.backends || (scene.initGL ? ['webgpu', 'webgl2'] : ['webgpu']);
    let backend = opts.backend && backends.includes(opts.backend) ? opts.backend : backends[0];

    let gpuState = null;
    if (backend === 'webgpu') {
      gpuState = await getWebGPU();
      if (token !== this._token) return null;
      if (gpuState.error) {
        if (backends.includes('webgl2')) {
          backend = 'webgl2';
          this.onStatus(`WebGPU unavailable (${gpuState.error}) — running the WebGL2 version.`);
        } else {
          this.showMessage(
            `<h3>This scene needs WebGPU</h3><p>${gpuState.error}</p><p>It relies on features WebGL2 does not have (compute shaders / storage buffers). ` +
              `Try a recent Chrome, Edge, Safari 26+ or Firefox 141+ — the explanation below still applies.</p>`,
          );
          return { backend: null };
        }
      }
    }

    const examples = scene.examples || [];
    const example = examples.find((e) => e.id === opts.example) ? opts.example : examples[0]?.id || null;
    this.ctx = {
      backend,
      scene,
      canvas: null,
      gpu: null,
      gl: null,
      glkit: null,
      width: 1,
      height: 1,
      dpr: 1,
      params: opts.params || defaultParams(scene, example),
      example,
      exampleIndex: Math.max(0, examples.findIndex((e) => e.id === example)),
      pointer: { x: 0, y: 0, nx: 0.5, ny: 0.5, dx: 0, dy: 0, down: false, over: false, clicked: false, released: false, button: 0, wheel: 0 },
      keys: new Set(),
      keysPressed: new Set(),
      time: 0,
      dt: 0,
      frame: 0,
      paused: this.paused,
      testMode: /[?&]test=1/.test(location.href),
      encoder: null,
      target: null,
      targetTexture: null,
      status: (text) => this.onStatus(text),
      overlay: this.sceneOverlay,
      reset: () => this.action('reset'),
    };
    const canvas = this._makeCanvas();
    this.ctx.canvas = canvas;
    const [w, h, dpr] = this._size();
    canvas.width = w;
    canvas.height = h;
    Object.assign(this.ctx, { width: w, height: h, dpr });
    this.ctx.pointer.x = w / 2;
    this.ctx.pointer.y = h / 2;

    try {
      if (backend === 'webgpu') {
        const context = canvas.getContext('webgpu');
        context.configure({ device: gpuState.device, format: gpuState.format, alphaMode: 'opaque' });
        this.ctx.gpu = new SceneGPU(gpuState, context);
        this.ctx.context = context;
        this.instance = (await scene.init(this.ctx)) || {};
      } else {
        const gl = canvas.getContext('webgl2', {
          antialias: !!scene.glAntialias,
          alpha: false,
          premultipliedAlpha: false,
          preserveDrawingBuffer: false,
        });
        if (!gl) throw new Error('WebGL2 is not available in this browser.');
        this.ctx.gl = gl;
        this.ctx.glkit = new GLKit(gl);
        this.instance = (await scene.initGL(this.ctx)) || {};
      }
    } catch (e) {
      this.reportError(e && e.stack ? `${e.message}\n${e.stack.split('\n').slice(1, 4).join('\n')}` : String(e));
      this.instance = null;
    }
    if (token !== this._token) {
      this.instance?.destroy?.();
      return null;
    }
    this._last = performance.now();
    this._dirty = true;
    window.__showcase.frames = 0;
    window.__showcase.ready = true;
    window.__showcase.backend = backend;
    this._raf = requestAnimationFrame((t) => this._loop(t));
    return { backend };
  }

  unload() {
    cancelAnimationFrame(this._raf);
    this._raf = 0;
    if (window.__showcase) window.__showcase.ready = false;
    try {
      this.instance?.destroy?.();
    } catch (e) {
      console.warn(e);
    }
    this.instance = null;
    this.ctx?.gpu?.destroy();
    this.ctx?.glkit?.destroy();
    if (this.ctx?.gl) this.ctx.gl.getExtension('WEBGL_lose_context')?.loseContext();
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
    this.scene = null;
    this.sceneOverlay.replaceChildren();
    this.sceneOverlay.removeAttribute('style');
  }

  setParam(key, value) {
    if (!this.ctx) return;
    this.ctx.params[key] = value;
    this._dirty = true;
    try {
      this.instance?.onChange?.(key, value, this.ctx);
    } catch (e) {
      this.reportError(e.message);
    }
  }

  setExample(id) {
    if (!this.ctx) return;
    const examples = this.scene.examples || [];
    const ex = examples.find((e) => e.id === id);
    if (!ex) return;
    this.ctx.example = id;
    this.ctx.exampleIndex = examples.indexOf(ex);
    if (ex.params) for (const [k, v] of Object.entries(ex.params)) this.ctx.params[k] = v;
    this._dirty = true;
    try {
      this.instance?.onExample?.(id, this.ctx);
      if (ex.params) for (const [k, v] of Object.entries(ex.params)) this.instance?.onChange?.(k, v, this.ctx);
    } catch (e) {
      this.reportError(e.message);
    }
  }

  action(key) {
    if (!this.ctx) return;
    this._dirty = true;
    try {
      if (key === 'reset' && !this.instance?.onAction && this.instance?.reset) this.instance.reset();
      else this.instance?.onAction?.(key, this.ctx);
    } catch (e) {
      this.reportError(e.message);
    }
  }

  setPaused(p) {
    this.paused = p;
    if (this.ctx) this.ctx.paused = p;
    this._dirty = true;
  }

  setRenderScale(s) {
    this.renderScale = s;
    this._dirty = true;
  }

  /** Capture the next rendered frame as a PNG blob. */
  screenshot() {
    return new Promise((resolve) => {
      this._shot = resolve;
      this._dirty = true;
    });
  }

  _loop(now) {
    this._raf = requestAnimationFrame((t) => this._loop(t));
    const ctx = this.ctx;
    if (!ctx || !this.instance) return;
    const real = Math.min(0.1, Math.max(0, (now - this._last) / 1000));
    this._last = now;

    // fps meter
    const f = this._fps;
    f.frames++;
    if (now - f.last > 500) {
      this.onFps((f.frames * 1000) / (now - f.last), f.ms);
      f.frames = 0;
      f.last = now;
    }

    if (this.paused && !this._dirty && !this._shot) return;

    // resize
    const [w, h, dpr] = this._size();
    if (w !== this.canvas.width || h !== this.canvas.height) {
      this.canvas.width = w;
      this.canvas.height = h;
      ctx.width = w;
      ctx.height = h;
      ctx.dpr = dpr;
      try {
        this.instance.resize?.(w, h, ctx);
      } catch (e) {
        this.reportError(e.message);
      }
    }

    ctx.paused = this.paused;
    ctx.dt = this.paused ? 0 : real * this.timeScale;
    ctx.time += ctx.dt;
    const t0 = performance.now();
    try {
      if (ctx.backend === 'webgpu') {
        const device = ctx.gpu.device;
        ctx.encoder = device.createCommandEncoder({ label: 'frame' });
        ctx.targetTexture = ctx.context.getCurrentTexture();
        ctx.target = ctx.targetTexture.createView();
        this.instance.frame?.(ctx);
        device.queue.submit([ctx.encoder.finish()]);
        ctx.encoder = null;
      } else {
        this.instance.frame?.(ctx);
      }
    } catch (e) {
      this.reportError(e && e.stack ? `${e.message}\n${e.stack.split('\n').slice(1, 4).join('\n')}` : String(e));
      this.instance = null;
      return;
    }
    f.ms = f.ms * 0.9 + (performance.now() - t0) * 0.1;
    if (this._shot) {
      const done = this._shot;
      this._shot = null;
      this.canvas.toBlob((b) => done(b), 'image/png');
    }
    if (!this.paused) ctx.frame++;
    const p = ctx.pointer;
    p.dx = p.dy = 0;
    p.wheel = 0;
    p.clicked = false;
    p.released = false;
    ctx.keysPressed.clear();
    this._dirty = false;
    window.__showcase.frames = (window.__showcase.frames || 0) + 1;
  }
}

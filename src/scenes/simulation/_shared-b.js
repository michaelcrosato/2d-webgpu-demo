// Small helpers shared by the "GPU Simulations (part 2)" scenes:
// water-ripples, verlet-physics, nbody, metaballs, jump-flood.

/**
 * Workaround for a shaderScene() bug (core/shaderscene.js, WebGPU path): with `input: 'game'`, the pass that
 * RENDERS the game scene also binds the `game` texture as a sampled input (every pass gets every texture),
 * so WebGPU rejects the command buffer ("includes writable usage and another usage in the same
 * synchronization scope"). We wrap gpu.fullscreen()/glkit.fullscreen() while the scene initialises and,
 * for the pass labelled 'game', swap the self-reference for a 1×1 dummy texture.
 */
export function withGameInputFix(scene, { staticGame = false } = {}) {
  const init = scene.init;
  const initGL = scene.initGL;
  const patch = (owner, makeDummy, wrapDraw) => {
    const orig = owner.fullscreen;
    let dummy = null;
    owner.fullscreen = function (opts) {
      const fx = orig.call(owner, opts);
      if (opts && opts.label === 'game') {
        const draw = fx.draw.bind(fx);
        const wrapped = wrapDraw(draw, () => (dummy ??= makeDummy()));
        // staticGame (used under the slow software-GPU test harness): render the game scene only a few times
        let n = 0;
        fx.draw = staticGame ? (...args) => (n++ < 3 ? wrapped(...args) : undefined) : wrapped;
      }
      return fx;
    };
    return () => (owner.fullscreen = orig);
  };
  return {
    ...scene,
    init: async (ctx) => {
      const gpu = ctx.gpu;
      const restore = patch(
        gpu,
        () => gpu.target(1, 1, { label: 'game-dummy' }),
        (draw, dummy) => (enc, target, res = {}, o) => draw(enc, target, { ...res, game: dummy() }, o),
      );
      try {
        return await init(ctx);
      } finally {
        restore();
      }
    },
    initGL: initGL
      ? async (ctx) => {
          const kit = ctx.glkit;
          const restore = patch(
            kit,
            () => kit.target(1, 1),
            (draw, dummy) => (target, res = {}, o) => draw(target, { ...res, game: dummy() }, o),
          );
          try {
            return await initGL(ctx);
          } finally {
            restore();
          }
        }
      : undefined,
  };
}

/** Create an overlay readout tag (bottom-right by default). */
export function overlayTag(ctx, css = 'right:8px;bottom:8px') {
  const el = document.createElement('div');
  el.className = 'tag';
  el.style.cssText = css;
  ctx.overlay.append(el);
  return el;
}

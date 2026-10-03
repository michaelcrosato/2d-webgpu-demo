// Small helpers shared by the Lighting & Shadows scenes.

/** Add a readout chip to the scene overlay. css: absolute position, e.g. 'right:8px;bottom:8px'. */
export function tag(ctx, css) {
  const el = document.createElement('div');
  el.className = 'tag';
  el.style.cssText = css;
  ctx.overlay.append(el);
  return el;
}

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
/** Frame-rate independent exponential smoothing factor. */
export const damp = (k, dt) => 1 - Math.exp(-k * dt);

/** '#rrggbb' -> linear-light [r, g, b] (so lights mix physically). */
export function hexLinear(hex, mul = 1) {
  const n = parseInt(hex.replace('#', ''), 16);
  const f = (v) => Math.pow(v / 255, 2.2) * mul;
  return [f((n >> 16) & 255), f((n >> 8) & 255), f(n & 255)];
}

/** '#rrggbb' -> sRGB [r, g, b] in 0..1. */
export function hexRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** Smooth 1D value noise in [0,1] (for flicker etc.). */
export function noise1(x, seed = 0) {
  const h = (i) => {
    let t = Math.imul((i + seed * 7919) | 0, 0x27d4eb2d) ^ 0x165667b1;
    t = Math.imul(t ^ (t >>> 15), 0x85ebca6b);
    t ^= t >>> 13;
    return ((t >>> 0) % 10007) / 10007;
  };
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * (3 - 2 * f);
  return h(i) * (1 - u) + h(i + 1) * u;
}

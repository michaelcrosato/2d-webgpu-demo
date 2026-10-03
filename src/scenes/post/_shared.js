// Shared helpers for the Post-Processing scenes (src/scenes/post/*).
//
//  - overlays(): tiny manager for `.tag` labels in ctx.overlay (created lazily from shaderScene's bind()).
//  - compareSplit(): the draggable-by-hover before/after divider used by most post effects.
//  - SPLIT_WGSL: draws that divider in the final image (portable WGSL; needs `u.resolution`).
//  - GAME_POS_WGSL: where things are in the procedural `game` image (hero, torches) so effects can target them.

/**
 * Overlay label manager. Call begin(ctx) at the top of bind(), show(...) for every label that should be
 * visible this frame, end() at the bottom: labels not shown this frame are hidden.
 */
export function overlays() {
  const els = new Map();
  let host = null;
  let used = new Set();
  return {
    begin(ctx) {
      host = ctx.overlay;
      used = new Set();
    },
    /** id: stable key, html: content, css: inline style (position!). Returns the element. */
    show(id, html, css) {
      let el = els.get(id);
      if (!el || el.parentNode !== host) {
        el = document.createElement('div');
        el.className = 'tag';
        host.append(el);
        els.set(id, el);
        el._html = el._css = null;
      }
      if (el._html !== html) {
        el.innerHTML = html;
        el._html = html;
      }
      if (el._css !== css) {
        el.style.cssText = css;
        el._css = css;
      }
      el.hidden = false;
      used.add(id);
      return el;
    },
    end() {
      for (const [id, el] of els) if (!used.has(id)) el.hidden = true;
    },
  };
}

/**
 * Before/after split position in device pixels (or -1 when disabled). Follows the mouse while it hovers
 * the canvas; eases back to the middle when the mouse leaves (e.g. while you drag a slider).
 * Also draws the two "ORIGINAL | EFFECT" labels through the overlay manager `ov`.
 */
export function compareSplit(state, ctx, enabled, ov, labels = ['Original', 'Effect']) {
  if (!enabled) return -1;
  const p = ctx.pointer;
  if (state.split === undefined) state.split = 0.5;
  if (p.over) state.split = p.nx;
  else state.split += (0.5 - state.split) * Math.min(1, (ctx.dt || 1 / 60) * 5 + (ctx.paused ? 1 : 0));
  const pct = (state.split * 100).toFixed(2);
  if (ov) {
    ov.show('cmpL', `◀ ${labels[0]}`, `top:46px;left:calc(${pct}% - 10px);transform:translateX(-100%);opacity:.92`);
    ov.show('cmpR', `${labels[1]} ▶`, `top:46px;left:calc(${pct}% + 10px);opacity:.92`);
  }
  return state.split * ctx.width;
}

/** WGSL: blend `after` / `before` around a vertical divider at x = split (px). split < 0 → just `after`. */
export const SPLIT_WGSL = /* wgsl */ `
fn splitView(after: vec3f, before: vec3f, px: vec2f, split: f32) -> vec3f {
  if (split < 0.0) { return after; }
  let s: f32 = max(1.0, u.resolution.y / 720.0);
  var c: vec3f = mix(after, before, step(px.x, split));
  let d: f32 = abs(px.x - split);
  c = mix(c, c * 0.3, (1.0 - smoothstep(0.0, 7.0 * s, d)) * 0.55);
  c = mix(c, vec3f(1.0), 1.0 - smoothstep(0.7 * s, 1.7 * s, d));
  // round handle in the middle with two little arrows
  let k: vec2f = vec2f(px.x - split, px.y - u.resolution.y * 0.5) / s;
  let r: f32 = length(k);
  c = mix(c, vec3f(0.07, 0.08, 0.12), 1.0 - smoothstep(13.0, 14.5, r));
  c = mix(c, vec3f(1.0), (1.0 - smoothstep(0.8, 1.8, abs(r - 13.5))));
  let ax: f32 = abs(k.x);
  let arrow: f32 = step(3.0, ax) * step(ax, 9.0) * step(abs(k.y), (9.0 - ax) * 0.75);
  c = mix(c, vec3f(1.0), arrow);
  return c;
}
`;

/**
 * WGSL: screen positions of things inside the procedural game image (see src/core/gamescene.js), so
 * effects can be aimed at the hero or the torches. Coordinates are in "p units": p = px / resolution.y.
 */
export const GAME_POS_WGSL = /* wgsl */ `
fn heroP() -> vec2f {
  let hop: f32 = abs(sin(u.time * 2.6));
  return vec2f(0.45, 0.8 - 0.042 - hop * 0.13);
}
// signed x distance (p units) from p to the nearest torch post; torches repeat every 0.9 and scroll
fn torchDX(p: vec2f) -> f32 {
  let gx: f32 = p.x + u.time * 0.12;
  return fmod(gx + 0.3, 0.9) - 0.45;
}
`;

export const clamp01 = (x) => Math.min(1, Math.max(0, x));
export const smooth = (x) => {
  const t = clamp01(x);
  return t * t * (3 - 2 * t);
};

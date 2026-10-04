// Small helpers shared by the Procedural Generation scenes.

/**
 * Show a set of label "tags" over the canvas (HTML in ctx.overlay). Call it every frame from
 * bind(); the DOM is only touched when the items change. items: [{ text, style }] (style = CSS
 * positioning, use % so it follows canvas resizes). Pass [] to hide everything.
 */
export function setLabels(ctx, items) {
  const ov = ctx && ctx.overlay;
  if (!ov) return;
  let box = ov.querySelector(':scope > .proc-labels');
  if (!box) {
    box = document.createElement('div');
    box.className = 'proc-labels';
    box.style.cssText = 'position:absolute;inset:0;pointer-events:none';
    ov.append(box);
  }
  const sig = JSON.stringify(items);
  if (box._sig === sig) return;
  box._sig = sig;
  box.replaceChildren(
    ...items.map((it) => {
      const d = document.createElement('div');
      d.className = 'tag';
      d.style.cssText = it.style;
      d.innerHTML = it.text;
      return d;
    }),
  );
}

// The page puts a HUD (fps/API) in the top-left corner and tool buttons in the top-right corner,
// so labels in the top row start below them (TOP px).
const TOP = 44;

/** Labels at the top-left corner of the four quadrants (order: TL, TR, BL, BR). */
export const quadLabels = (texts) =>
  texts.map((t, i) => ({ text: t, style: `left:calc(${(i % 2) * 50}% + 8px);top:${i < 2 ? `${TOP}px` : 'calc(50% + 8px)'}` }));

/** Labels at the top-left of N equal vertical columns. */
export const columnLabels = (texts, top = TOP) =>
  texts.map((t, i) => ({ text: t, style: `left:calc(${(i * 100) / texts.length}% + 8px);top:${top}px` }));

/**
 * Pan & zoom camera state driven by the pointer (drag = pan, wheel = zoom at cursor,
 * optional click-without-drag callback). Units: world units per canvas HEIGHT = 1 / zoom.
 * World y grows downward like the screen.
 */
export class PanZoom {
  constructor({ x = 0, y = 0, zoom = 1, minZoom = 0.05, maxZoom = 1e9 } = {}) {
    this.home = { x, y, zoom };
    this.minZoom = minZoom;
    this.maxZoom = maxZoom;
    this.reset();
  }
  reset() {
    this.x = this.home.x;
    this.y = this.home.y;
    this.zoom = this.home.zoom;
    this.tx = this.x;
    this.ty = this.y;
    this.tzoom = this.zoom;
    this.dragDist = 0;
  }
  /** world coordinates of a canvas pixel */
  toWorld(ctx, px, py) {
    const s = 1 / (this.zoom * ctx.height);
    return [this.x + (px - ctx.width / 2) * s, this.y + (py - ctx.height / 2) * s];
  }
  /** animate toward a target view */
  flyTo(x, y, zoom) {
    this.tx = x;
    this.ty = y;
    this.tzoom = Math.min(this.maxZoom, Math.max(this.minZoom, zoom));
  }
  /**
   * Update from ctx.pointer. Returns true if a "click" (press + release without dragging) happened.
   * smooth: 0..1 easing factor per frame for wheel/fly-to zoom (1 = instant).
   */
  update(ctx, { smooth = 0.25, clickThreshold = 6 } = {}) {
    const p = ctx.pointer;
    let clicked = false;
    if (p.clicked) this.dragDist = 0;
    if (p.down && (p.dx || p.dy)) {
      this.dragDist += Math.hypot(p.dx, p.dy);
      const s = 1 / (this.zoom * ctx.height);
      this.x -= p.dx * s;
      this.y -= p.dy * s;
      this.tx = this.x;
      this.ty = this.y;
    }
    if (p.released && this.dragDist < clickThreshold) clicked = true;
    if (p.wheel) {
      // zoom toward the cursor: keep the world point under the cursor fixed
      const f = Math.exp(-p.wheel * 0.0015);
      const nz = Math.min(this.maxZoom, Math.max(this.minZoom, this.tzoom * f));
      const [wx, wy] = this.toWorldAt(ctx, p.x, p.y, this.tx, this.ty, this.tzoom);
      const s2 = 1 / (nz * ctx.height);
      this.tx = wx - (p.x - ctx.width / 2) * s2;
      this.ty = wy - (p.y - ctx.height / 2) * s2;
      this.tzoom = nz;
    }
    // ease toward the target: zoom in log space, and the target's SCREEN-space offset (not the
    // world-space one) so that deep fly-tos keep the destination on screen the whole way
    const k = smooth;
    const lz = Math.log(this.zoom);
    const ltz = Math.log(this.tzoom);
    const ox = (this.tx - this.x) * this.zoom;
    const oy = (this.ty - this.y) * this.zoom;
    if (Math.abs(ltz - lz) > 1e-5 || Math.abs(ox) + Math.abs(oy) > 1e-6) {
      const nz = Math.exp(lz + (ltz - lz) * k);
      this.zoom = nz;
      this.x = this.tx - (ox * (1 - k)) / nz;
      this.y = this.ty - (oy * (1 - k)) / nz;
    } else {
      this.zoom = this.tzoom;
      this.x = this.tx;
      this.y = this.ty;
    }
    return clicked;
  }
  toWorldAt(ctx, px, py, x, y, zoom) {
    const s = 1 / (zoom * ctx.height);
    return [x + (px - ctx.width / 2) * s, y + (py - ctx.height / 2) * s];
  }
}

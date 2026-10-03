// Small helpers shared by the "Shapes, Lines & Color" scenes.

/**
 * HTML labels over the canvas (via ctx.overlay). Rebuilt only when `key` changes.
 * items: [{ text, x, y, align?: 'center'|'left'|'right', valign?: 'top'|'middle'|'bottom', style? }]
 *   x, y are fractions (0..1) of the canvas size.
 * Returns the array of label elements so callers can move them every frame with place().
 */
export function labels(ctx, key, items) {
  let st = ctx.__shapeLabels;
  if (!st || !st.root.isConnected || st.root.parentNode !== ctx.overlay) {
    const root = document.createElement('div');
    root.style.cssText = 'position:absolute;inset:0;pointer-events:none';
    ctx.overlay.append(root);
    st = ctx.__shapeLabels = { root, key: null, els: [] };
  }
  if (st.key !== key) {
    st.key = key;
    st.root.replaceChildren();
    st.els = (items || []).map((it) => {
      const e = document.createElement('div');
      e.className = 'tag';
      e.innerHTML = it.text;
      if (it.style) e.style.cssText = it.style;
      st.root.append(e);
      place(e, it.x, it.y, it.align, it.valign);
      return e;
    });
  }
  return st.els;
}

/** Position a label element at fractional canvas coords with an alignment. */
export function place(e, x, y, align = 'center', valign = 'top') {
  e.style.left = `${(x * 100).toFixed(3)}%`;
  e.style.top = `${(y * 100).toFixed(3)}%`;
  const tx = align === 'left' ? '0' : align === 'right' ? '-100%' : '-50%';
  const ty = valign === 'top' ? '0' : valign === 'bottom' ? '-100%' : '-50%';
  e.style.transform = `translate(${tx}, ${ty})`;
}

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;

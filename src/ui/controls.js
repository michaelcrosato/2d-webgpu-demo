// Builds the slider/toggle/select/color/button panel from a scene's `controls` array.

const el = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};
export { el };

function decimals(step) {
  if (!step || step >= 1) return 0;
  const s = String(step);
  return s.includes('e-') ? +s.split('e-')[1] : (s.split('.')[1] || '').length;
}

export function formatValue(c, v) {
  if (c.format) return c.format(v);
  if (typeof v !== 'number') return String(v);
  const d = c.decimals ?? decimals(c.step ?? (c.max - c.min) / 100);
  let s = v.toFixed(Math.min(d, 4));
  if (Math.abs(v) >= 10000) s = Math.round(v).toLocaleString();
  return s + (c.unit ? ` ${c.unit}` : '');
}

const visible = (c, example) => !c.showFor || c.showFor.includes(example);

/**
 * @param container element to fill
 * @param scene scene module
 * @param params live params object
 * @param example current example id
 * @param handlers { onChange(key, value), onAction(key), onReset() }
 */
export function buildControls(container, scene, params, example, handlers) {
  container.replaceChildren();
  const controls = (scene.controls || []).filter((c) => visible(c, example));
  const head = el('h3', {}, 'Controls', el('button', { title: 'Reset all controls to their defaults', onclick: handlers.onReset }, 'Reset'));
  container.append(head);
  if (!controls.length) {
    container.append(el('div', { class: 'ctl-help' }, 'This scene has no adjustable parameters — interact with the canvas instead.'));
    return;
  }
  let btnRow = null;
  for (const c of controls) {
    if (c.type !== 'button') btnRow = null;
    if (c.type === 'heading') {
      container.append(el('div', { class: 'ctl-heading' }, c.label));
      continue;
    }
    if (c.type === 'info') {
      const d = el('div', { class: 'ctl-help', style: 'margin-bottom:10px' });
      d.innerHTML = c.label;
      container.append(d);
      continue;
    }
    if (c.type === 'button') {
      if (!btnRow) {
        btnRow = el('div', { class: 'btnrow ctl' });
        container.append(btnRow);
      }
      btnRow.append(el('button', { class: 'btn' + (c.primary ? ' primary' : ''), title: c.help || '', onclick: () => handlers.onAction(c.key) }, c.label));
      continue;
    }
    const wrap = el('div', { class: 'ctl' });
    const id = `ctl-${c.key}`;
    if (c.type === 'slider') {
      const out = el('output', { for: id }, formatValue(c, params[c.key]));
      const log = c.log && c.min > 0;
      const toPos = (v) => (log ? Math.log(v / c.min) / Math.log(c.max / c.min) : (v - c.min) / (c.max - c.min));
      const fromPos = (p) => {
        let v = log ? c.min * Math.pow(c.max / c.min, p) : c.min + p * (c.max - c.min);
        const step = c.step ?? 0;
        if (step) v = Math.round(v / step) * step;
        return +Math.min(c.max, Math.max(c.min, v)).toFixed(6);
      };
      const input = el('input', { id, type: 'range', min: 0, max: 1000, step: 1, value: Math.round(toPos(params[c.key]) * 1000) });
      const fill = () => input.style.setProperty('--fill', `${(input.value / 10).toFixed(1)}%`);
      fill();
      input.addEventListener('input', () => {
        const v = fromPos(input.value / 1000);
        out.textContent = formatValue(c, v);
        fill();
        handlers.onChange(c.key, v);
      });
      input.addEventListener('dblclick', () => {
        const v = c.value;
        input.value = Math.round(toPos(v) * 1000);
        out.textContent = formatValue(c, v);
        fill();
        handlers.onChange(c.key, v);
      });
      wrap.append(el('div', { class: 'ctl-head' }, el('label', { for: id }, c.label), out), input);
    } else if (c.type === 'toggle') {
      const input = el('input', { id, type: 'checkbox' });
      input.checked = !!params[c.key];
      input.addEventListener('change', () => handlers.onChange(c.key, input.checked));
      wrap.append(el('label', { class: 'switch', for: id }, el('span', {}, c.label), input, el('span', { class: 'knob' })));
    } else if (c.type === 'select') {
      const sel = el('select', { id });
      for (const o of c.options) {
        const value = typeof o === 'object' ? o.value : o;
        const label = typeof o === 'object' ? o.label : o;
        const opt = el('option', { value: String(value) }, label);
        if (String(value) === String(params[c.key])) opt.selected = true;
        sel.append(opt);
      }
      sel.addEventListener('change', () => {
        const o = c.options.find((o) => String(typeof o === 'object' ? o.value : o) === sel.value);
        const v = typeof o === 'object' ? o.value : o;
        handlers.onChange(c.key, v);
      });
      wrap.append(el('div', { class: 'ctl-head' }, el('label', { for: id }, c.label)), sel);
    } else if (c.type === 'color') {
      const input = el('input', { id, type: 'color', value: params[c.key] });
      const out = el('output', {}, params[c.key]);
      input.addEventListener('input', () => {
        out.textContent = input.value;
        handlers.onChange(c.key, input.value);
      });
      wrap.append(el('div', { class: 'ctl-head' }, el('label', { for: id }, c.label), el('span', { class: 'colorrow' }, out, input)));
    } else {
      continue;
    }
    if (c.help) wrap.append(el('div', { class: 'ctl-help' }, c.help));
    container.append(wrap);
  }
}

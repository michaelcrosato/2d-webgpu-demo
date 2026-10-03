// Renders a scene's educational `about` content into cards.

import { el } from './controls.js';

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

/** Tiny highlighter for WGSL / GLSL / JS snippets. */
export function highlight(code) {
  const re =
    /(\/\/[^\n]*)|(\b\d+\.?\d*(?:e[-+]?\d+)?[fu]?\b)|(\b(?:fn|let|var|const|return|if|else|for|while|struct|break|continue|discard|uniform|in|out|inout|void|true|false|function|new|of|this|layout|precision|highp)\b|@\w+)|(\b(?:f32|i32|u32|bool|vec[234][fiu]?|ivec[234]|uvec[234]|mat[234]x[234]f?|mat[234]|float|int|uint|sampler2D|sampler|texture_2d|texture_storage_2d|array|atomic)\b)|(\b[a-zA-Z_]\w*(?=\s*\())/g;
  let out = '';
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const cls = m[1] ? 'tok-c' : m[2] ? 'tok-n' : m[3] ? 'tok-k' : m[4] ? 'tok-t' : 'tok-f';
    out += `<span class="${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + esc(code.slice(last));
}

function html(content) {
  const d = document.createElement('div');
  d.innerHTML = Array.isArray(content) ? content.map((x) => `<p>${x}</p>`).join('') : content;
  return d;
}

function list(items, ordered = false) {
  const l = document.createElement(ordered ? 'ol' : 'ul');
  for (const it of items) {
    const li = document.createElement('li');
    if (typeof it === 'string') li.innerHTML = it;
    else li.innerHTML = `<b>${it.title}</b> — ${it.text}`;
    l.append(li);
  }
  return l;
}

function card(icon, title, body, wide = false) {
  return el('section', { class: 'card' + (wide ? ' wide' : '') }, el('h2', {}, el('span', { class: 'ic' }, icon), title), body);
}

function codeBox(snippet) {
  const pre = document.createElement('pre');
  pre.innerHTML = highlight(snippet.src.replace(/^\n+|\s+$/g, ''));
  const copy = el('button', { class: 'copy' }, 'Copy');
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(snippet.src.trim());
      copy.textContent = 'Copied!';
    } catch {
      copy.textContent = 'Select & copy';
    }
    setTimeout(() => (copy.textContent = 'Copy'), 1400);
  });
  const box = el('div', { class: 'codebox' });
  if (snippet.title) box.append(el('div', { class: 'lang' }, `${snippet.title}${snippet.lang ? ` · ${snippet.lang}` : ''}`));
  box.append(pre, copy);
  return box;
}

/**
 * @param container element
 * @param scene module
 * @param nav { prev: {id,title}|null, next: {id,title}|null }
 */
export function renderExplain(container, scene, nav) {
  container.replaceChildren();
  const a = scene.about || {};
  if (a.what) container.append(card('👁', 'What you’re seeing', html(a.what)));
  if (a.how) container.append(card('⚙', 'How it works', html(a.how)));
  if (a.uses?.length) container.append(card('🎮', 'Where you’d use it', list(a.uses)));
  if (a.try?.length) container.append(card('🧪', 'Try this', list(a.try)));
  if (a.ask?.length) {
    const wrap = el('div', { class: 'ask-list' });
    for (const phrase of a.ask) {
      const b = el('button', { class: 'ask', title: 'Click to copy' }, `“${phrase}”`);
      b.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(phrase);
        } catch {
          /* ignore */
        }
        b.classList.add('copied');
        setTimeout(() => b.classList.remove('copied'), 1200);
      });
      wrap.append(b);
    }
    const body = el('div', {}, el('p', { style: 'color:var(--muted);font-size:13.5px;margin:0 0 10px' }, 'Vocabulary to use when you request this in your game (click to copy):'), wrap);
    container.append(card('💬', 'Ask for it like…', body));
  }
  if (a.perf) container.append(card('⏱', 'Performance', html(a.perf)));
  if (a.api) container.append(card('🧩', 'WebGPU vs WebGL2', html(a.api)));
  const snippets = a.code ? (Array.isArray(a.code) ? a.code : [a.code]) : [];
  if (snippets.length) {
    const body = el('div', { style: 'display:grid;gap:12px' }, ...snippets.map(codeBox));
    container.append(card('⌨', 'Key shader code', body, true));
  }
  if (a.links?.length) container.append(card('📚', 'Learn more', list(a.links.map((l) => `<a href="${l.url}" target="_blank" rel="noopener">${l.title}</a>${l.note ? ` — ${l.note}` : ''}`))));
  if (nav) {
    const pn = el('div', { class: 'prevnext' });
    pn.append(
      nav.prev ? el('a', { href: `#/s/${nav.prev.id}` }, el('small', {}, '← Previous'), nav.prev.title) : el('span'),
      nav.next ? el('a', { href: `#/s/${nav.next.id}`, class: 'next' }, el('small', {}, 'Next →'), nav.next.title) : el('span'),
    );
    container.append(pn);
  }
}

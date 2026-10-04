import { shaderScene } from '../../core/shaderscene.js';
import { getAtlas, makeCanvas } from '../../core/assets.js';

// Holographic foil, Balatro-style editions and shiny metals.
//  * Card faces (art, text, frame) are painted once with Canvas2D (+ pixel-art sprites from the atlas),
//    together with a MASK texture: R = art-window background, G = the creature, B = the frame.
//  * The card is a plane rotated in 3D by the mouse ("tilt"). For each pixel we intersect the view ray with
//    that plane (cheap perspective), which also gives the VIEW DIRECTION in card space.
//  * Foil = color that depends on the view direction: rainbow hue = f(position + view angle), glitter flakes
//    with random orientations that only flash when the angle lines up, and a moving shine band.
//  * Metal text/badges: normals from a blurred mask (text) or from an SDF bevel (badges) reflect a
//    procedural studio environment.

const CW = 400; // card face size in the texture
const CH = 560;
const CARDS = [
  { name: 'Slime King', cost: 3, type: 'Creature — Goo', text: ['When played, split into', 'two smaller slimes.'], flavor: '“Bow before the wobble.”', stats: '2 / 4', sprite: 'slime_0', sky: ['#41a6f6', '#a7f070'], frame: ['#29366f', '#3b5dc9'] },
  { name: 'Wandering Hero', cost: 2, type: 'Creature — Human', text: ['Double jump. Gains +1', 'attack for each coin.'], flavor: '“One more level…”', stats: '3 / 2', sprite: 'hero_idle_0', sky: ['#ffcd75', '#ef7d57'], frame: ['#5d275d', '#b13e53'] },
  { name: 'Mimic Chest', cost: 4, type: 'Creature — Box', text: ['Looks like treasure.', 'Bites anyone who opens it.'], flavor: '“It was NOT loot.”', stats: '4 / 3', sprite: 'chest', sky: ['#73eff7', '#257179'], frame: ['#333c57', '#566c86'] },
  { name: 'Night Bat', cost: 1, type: 'Creature — Beast', text: ['Flying. Can’t be blocked', 'during the night phase.'], flavor: '“Squeak.”', stats: '1 / 1', sprite: 'bat_0', sky: ['#5d275d', '#29366f'], frame: ['#1a1c2c', '#333c57'] },
];

let facesPromise = null;
function buildFaces() {
  if (facesPromise) return facesPromise;
  facesPromise = getAtlas().then((atlas) => {
    const faces = makeCanvas(CW * CARDS.length, CH);
    const masks = makeCanvas(CW * CARDS.length, CH);
    const g = faces.getContext('2d');
    const m = masks.getContext('2d');
    const rr = (ctx, x, y, w, h, r) => {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    };
    const sans = '"Segoe UI", "Helvetica Neue", Arial, sans-serif';
    m.fillStyle = '#000';
    m.fillRect(0, 0, masks.width, masks.height);
    CARDS.forEach((c, i) => {
      const x0 = i * CW;
      g.save();
      g.translate(x0, 0);
      // frame
      let gr = g.createLinearGradient(0, 0, CW, CH);
      gr.addColorStop(0, c.frame[1]);
      gr.addColorStop(1, c.frame[0]);
      g.fillStyle = gr;
      g.fillRect(0, 0, CW, CH);
      g.strokeStyle = 'rgba(255,255,255,0.35)';
      g.lineWidth = 3;
      rr(g, 9, 9, CW - 18, CH - 18, 14);
      g.stroke();
      // name banner + cost
      g.fillStyle = '#f4ead5';
      rr(g, 24, 22, CW - 48, 44, 10);
      g.fill();
      g.fillStyle = '#1a1c2c';
      g.font = `bold 25px ${sans}`;
      g.textBaseline = 'middle';
      g.fillText(c.name, 38, 45);
      g.fillStyle = '#3b5dc9';
      g.beginPath();
      g.arc(CW - 50, 44, 18, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = '#f4ead5';
      g.lineWidth = 3;
      g.stroke();
      g.fillStyle = '#fff';
      g.font = `bold 22px ${sans}`;
      g.textAlign = 'center';
      g.fillText(String(c.cost), CW - 50, 45);
      g.textAlign = 'left';
      // art window: sky gradient, sun rays, ground, then the sprite (pixel art, nearest)
      const ax = 24;
      const ay = 76;
      const aw = CW - 48;
      const ah = 250;
      gr = g.createLinearGradient(0, ay, 0, ay + ah);
      gr.addColorStop(0, c.sky[0]);
      gr.addColorStop(1, c.sky[1]);
      g.fillStyle = gr;
      g.fillRect(ax, ay, aw, ah);
      g.save();
      g.beginPath();
      g.rect(ax, ay, aw, ah);
      g.clip();
      g.fillStyle = 'rgba(255,255,255,0.18)';
      for (let k = 0; k < 12; k++) {
        const a0 = (k / 12) * Math.PI * 2;
        g.beginPath();
        g.moveTo(ax + aw / 2, ay + ah * 0.55);
        g.lineTo(ax + aw / 2 + Math.cos(a0) * 400, ay + ah * 0.55 + Math.sin(a0) * 400);
        g.lineTo(ax + aw / 2 + Math.cos(a0 + 0.22) * 400, ay + ah * 0.55 + Math.sin(a0 + 0.22) * 400);
        g.fill();
      }
      g.fillStyle = 'rgba(26,28,44,0.25)';
      g.beginPath();
      g.ellipse(ax + aw / 2, ay + ah - 30, 110, 16, 0, 0, Math.PI * 2);
      g.fill();
      g.restore();
      const f = atlas.frames[c.sprite];
      const s = Math.floor(Math.min(190 / f.h, 220 / f.w));
      const sx = ax + aw / 2 - (f.w * s) / 2;
      const sy = ay + ah - 34 - f.h * s;
      g.imageSmoothingEnabled = false;
      g.drawImage(atlas.canvas, f.x, f.y, f.w, f.h, sx, sy, f.w * s, f.h * s);
      g.strokeStyle = 'rgba(0,0,0,0.5)';
      g.lineWidth = 3;
      g.strokeRect(ax, ay, aw, ah);
      // type line, rules text, stats
      g.fillStyle = '#e8dcc0';
      rr(g, 24, 334, CW - 48, 32, 8);
      g.fill();
      g.fillStyle = '#333c57';
      g.font = `italic 600 17px ${sans}`;
      g.fillText(c.type, 38, 351);
      g.fillStyle = '#f6efe0';
      rr(g, 24, 374, CW - 48, 142, 8);
      g.fill();
      g.fillStyle = '#1a1c2c';
      g.font = `18px ${sans}`;
      c.text.forEach((line, k) => g.fillText(line, 40, 402 + k * 25));
      g.fillStyle = '#566c86';
      g.font = `italic 16px ${sans}`;
      g.fillText(c.flavor, 40, 470);
      g.fillStyle = c.frame[0];
      rr(g, CW - 112, 494, 88, 44, 10);
      g.fill();
      g.strokeStyle = '#f4ead5';
      g.lineWidth = 3;
      g.stroke();
      g.fillStyle = '#fff';
      g.font = `bold 23px ${sans}`;
      g.textAlign = 'center';
      g.fillText(c.stats, CW - 68, 517);
      g.textAlign = 'left';
      g.fillStyle = '#ffcd75';
      g.font = `bold 20px ${sans}`;
      g.fillText('★', 36, 519);
      g.restore();

      // ---- mask: B = frame, R = art background, G = creature
      m.save();
      m.translate(x0, 0);
      m.fillStyle = 'rgb(0,0,255)';
      m.fillRect(0, 0, CW, CH);
      m.fillStyle = '#000';
      rr(m, 24, 22, CW - 48, 44, 10);
      m.fill();
      rr(m, 24, 334, CW - 48, 32, 8);
      m.fill();
      rr(m, 24, 374, CW - 48, 142, 8);
      m.fill();
      rr(m, CW - 112, 494, 88, 44, 10);
      m.fill();
      m.fillStyle = 'rgb(255,0,0)';
      m.fillRect(ax, ay, aw, ah);
      // creature silhouette in green
      const tmp = makeCanvas(f.w, f.h);
      const tg = tmp.getContext('2d');
      tg.drawImage(atlas.canvas, f.x, f.y, f.w, f.h, 0, 0, f.w, f.h);
      tg.globalCompositeOperation = 'source-in';
      tg.fillStyle = 'rgb(0,255,0)';
      tg.fillRect(0, 0, f.w, f.h);
      m.imageSmoothingEnabled = false;
      m.drawImage(tmp, 0, 0, f.w, f.h, sx, sy, f.w * s, f.h * s);
      m.restore();
    });
    return { faces, masks };
  });
  return facesPromise;
}

// "VICTORY!" for the metal example: R = sharp mask, G = wide blur (bevel), B = narrow blur (rim)
function buildMetalText() {
  const W = 1024;
  const H = 256;
  const layer = (blur) => {
    const c = makeCanvas(W, H);
    const g = c.getContext('2d');
    g.fillStyle = '#000';
    g.fillRect(0, 0, W, H);
    g.filter = blur ? `blur(${blur}px)` : 'none';
    g.fillStyle = '#fff';
    g.font = '900 168px "Arial Black", "Helvetica Neue", Arial, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText('VICTORY!', W / 2, H / 2 + 6);
    return g.getImageData(0, 0, W, H).data;
  };
  const a = layer(0);
  const b = layer(9);
  const c2 = layer(3);
  const out = makeCanvas(W, H);
  const og = out.getContext('2d');
  const img = og.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    img.data[i * 4] = a[i * 4];
    img.data[i * 4 + 1] = b[i * 4];
    img.data[i * 4 + 2] = c2[i * 4];
    img.data[i * 4 + 3] = 255;
  }
  og.putImageData(img, 0, 0);
  return out;
}

const CODE = /* wgsl */ `
const CARD_W: f32 = 0.7142857;   // card width / height (400 / 560)

fn rotX3(v: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(v.x, c * v.y - s * v.z, s * v.y + c * v.z); }
fn rotY3(v: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z); }

struct Hit { uv: vec2f, cover: f32, view: vec3f, shadow: f32 };

// Ray / plane intersection for a card of height 1 (in units of hpx pixels) rotated by (rx, ry).
fn hitCard(px: vec2f, center: vec2f, hpx: f32, aspectW: f32, rx: f32, ry: f32) -> Hit {
  let p = (px - center) / hpx;
  let dist = 2.4;
  let o = vec3f(0.0, 0.0, -dist);
  let dir = normalize(vec3f(p, dist));
  let n = rotY3(rotX3(vec3f(0.0, 0.0, -1.0), rx), ry);
  let t = -dot(o, n) / dot(dir, n);
  let X = o + dir * t;
  let L = rotX3(rotY3(X, -ry), -rx);                       // into card space
  var h: Hit;
  h.uv = vec2f(L.x / aspectW + 0.5, L.y + 0.5);
  let r = 0.045;
  let q = abs(L.xy) - vec2f(aspectW * 0.5, 0.5) + vec2f(r);
  let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
  h.cover = clamp(0.5 - d * hpx, 0.0, 1.0);
  h.view = rotX3(rotY3(-dir, -ry), -rx);                   // direction to the eye, in card space
  // soft drop shadow (screen space, unrotated, offset down-right)
  let qs = abs(p - vec2f(0.03, 0.05)) - vec2f(aspectW * 0.5, 0.5) + vec2f(r);
  let ds = length(max(qs, vec2f(0.0))) + min(max(qs.x, qs.y), 0.0) - r;
  h.shadow = 1.0 - smoothstep(-0.02, 0.07, ds);
  return h;
}

// --------------------------------------------------------------- Balatro-like swirling paint background
fn swirlBg(px: vec2f, t: f32) -> vec3f {
  let pix = max(2.0, floor(u.resolution.y / 280.0));        // chunky pixels, like the game
  let q = (floor(px / pix) * pix - 0.5 * u.resolution) / length(u.resolution);
  let r = length(q);
  let a = atan2(q.y, q.x) + t * 0.2 * u.swirl - 6.0 * (0.6 * r + 0.4);   // spiral twist
  var w = vec2f(cos(a), sin(a)) * r * 26.0;
  var w2 = vec2f(w.x + w.y);
  for (var i = 0; i < 5; i++) {
    w2 += sin(max(w.x, w.y)) + w;
    w += 0.5 * vec2f(cos(4.3 + 0.37 * w2.y + t * 0.13 * u.swirl), sin(w2.x - 0.11 * t * u.swirl));
    w -= cos(w.x + w.y) - sin(w.x * 0.7 - w.y);
  }
  let k = clamp(length(w) * 0.04, 0.0, 2.0);
  let c1 = max(0.0, 1.0 - 1.6 * abs(1.0 - k));
  let c2 = max(0.0, 1.0 - 1.6 * abs(k));
  let c3 = 1.0 - min(1.0, c1 + c2);
  let red = vec3f(0.85, 0.27, 0.24);
  let blue = vec3f(0.0, 0.42, 0.7);
  let dark = vec3f(0.09, 0.14, 0.15);
  return red * 0.2 + 0.8 * (red * c1 + blue * c2 + dark * c3);
}

// --------------------------------------------------------------- foil building blocks
// the angle a point of the sheet "sees": perspective view direction + overall tilt
fn foilAngle(view: vec3f) -> vec2f { return view.xy * 2.5 + vec2f(u.tilt.y, -u.tilt.x) * 1.3; }

fn rainbowFoil(uv: vec2f, ang: vec2f) -> vec3f {
  let uvc = uv - 0.5;
  let hue = fract(dot(uvc, vec2f(0.8, 1.2)) * 1.1 + ang.x * 0.9 - ang.y * 0.7);
  return hsv2rgb(vec3f(hue, 0.7, 1.0));
}

// how strongly the foil "catches" light at this angle (diffraction bands sweep across as you tilt)
fn foilBands(uv: vec2f, ang: vec2f) -> f32 {
  let s = dot(uv - 0.5, vec2f(1.0, 0.7)) * 2.4 + ang.x * 1.6 + ang.y * 1.1;
  return 0.35 + 0.65 * pow(0.5 + 0.5 * cos(s * TAU), 2.0);
}

fn patternMask(uv: vec2f, kind: i32) -> f32 {
  let p = vec2f(uv.x * CARD_W, uv.y);
  if (kind == 1) {
    // stars on a jittered grid
    let g = p * 13.0;
    let id = floor(g);
    let h = hash23(id);
    let c = fract(g) - 0.5 - (h.xy - 0.5) * 0.4;
    let d = sdStar5(rot2(h.z * 6.0) * c, 0.18 + 0.2 * h.z, 0.45);
    return 0.18 + 0.82 * (1.0 - smoothstep(-0.02, 0.02, d));
  }
  if (kind == 2) {
    let s = fract(dot(p, vec2f(1.0, -0.6)) * 28.0);
    return 0.15 + 0.85 * smoothstep(0.35, 0.5, s) * (1.0 - smoothstep(0.5, 0.65, s));
  }
  if (kind == 3) {
    // "cosmos": scattered circles & rings at three sizes
    var m = 0.15;
    for (var k = 0; k < 3; k++) {
      let sc = 5.0 + 6.0 * f32(k);
      let g = p * sc + vec2f(f32(k) * 3.7);
      let id = floor(g);
      let h = hash23(id + vec2f(f32(k) * 17.0));
      let c = fract(g) - 0.5 - (h.xy - 0.5) * 0.5;
      let r = 0.12 + 0.2 * h.z;
      let d = length(c) - r;
      let ring = abs(d + 0.02) - 0.02;
      m = max(m, select(1.0 - smoothstep(0.0, 0.02, d), 1.0 - smoothstep(0.0, 0.02, ring), h.z > 0.5));
    }
    return m;
  }
  if (kind == 4) { return 0.3; }
  return 1.0;
}

// glitter: tiny flakes with random orientations; each one flashes only when the view angle matches it
fn glitter(uv: vec2f, ang: vec2f, density: f32) -> f32 {
  let g = vec2f(uv.x * CARD_W, uv.y) * 150.0;
  let id = floor(g);
  let h = hash33(vec3f(id, 5.0));
  let flake = (h.xy - 0.5) * 2.6;
  let align = 1.0 - smoothstep(0.0, 0.3, length(flake - ang));
  let shape = 1.0 - smoothstep(0.1, 0.42, length(fract(g) - 0.5));
  return align * align * shape * step(1.0 - density, h.z) * 2.5;
}

fn shineBand(uv: vec2f, ang: vec2f, phase: f32) -> f32 {
  let s = dot(uv - 0.5, normalize(vec2f(1.0, 0.75))) - (ang.x * 0.55 + ang.y * 0.45) - phase;
  return exp(-s * s * 70.0);
}

fn latticeLines(uv: vec2f, sc: f32) -> f32 {
  let p = vec2f(uv.x * CARD_W, uv.y) * sc;
  let a = abs(fract(p.y) - 0.5);
  let b = abs(fract(dot(p, vec2f(0.866, 0.5))) - 0.5);
  let c = abs(fract(dot(p, vec2f(-0.866, 0.5))) - 0.5);
  return 1.0 - smoothstep(0.0, 0.07, min(a, min(b, c)));
}

// Balatro-style editions applied to the printed card color
fn applyEdition(c0: vec3f, uv: vec2f, ang: vec2f, ed: i32, strength: f32) -> vec3f {
  var c = c0;
  let uvc = uv - 0.5;
  if (ed == 1) {
    // Foil: cool, desaturated silver-blue with flowing streaks
    let fl = 0.5 + 0.5 * sin(dot(uvc, vec2f(13.0, 8.0)) + ang.x * 6.0 + 3.0 * sin(uv.y * 7.0 + ang.y * 4.0));
    c = mix(c, vec3f(luma(c)) * vec3f(0.78, 0.9, 1.2), 0.45 * strength);
    c += vec3f(0.3, 0.5, 1.0) * pow(fl, 3.0) * 0.55 * strength;
  } else if (ed == 2) {
    // Holographic: red/cyan shifting sheen through a lattice
    let gl = latticeLines(uv, 22.0);
    let hc = mix(vec3f(1.0, 0.25, 0.32), vec3f(0.2, 0.95, 1.0), 0.5 + 0.5 * sin(ang.x * 5.0 + uv.y * 6.0 - ang.y * 3.0));
    c = c * (1.0 - 0.25 * strength) + hc * (0.18 + 0.4 * gl) * strength;
  } else if (ed == 3) {
    // Polychrome: rotate the hue of everything by an angle that varies over the card and with the tilt
    let a = (uv.x + uv.y) * 3.0 + ang.x * 3.5 - ang.y * 2.5 + u.time * 0.3;
    c = adjustSaturation(hueRotate(c, a * strength), 1.0 + 0.5 * strength);
    c += rainbowFoil(uv, ang) * 0.12 * strength;
  } else if (ed == 4) {
    // Negative: inverted, pushed toward violet, with a sheen
    c = mix(c, (vec3f(1.0) - c) * vec3f(0.85, 0.8, 1.1), strength);
  }
  return max(c, vec3f(0.0));
}

// --------------------------------------------------------------- the full holo card
fn faceAt(uv: vec2f, k: f32) -> vec3f { return TEX(faces, vec2f((k + clamp(uv.x, 0.0, 1.0)) / 4.0, uv.y)).rgb; }
fn maskAt(uv: vec2f, k: f32) -> vec3f { return TEX(masks, vec2f((k + clamp(uv.x, 0.0, 1.0)) / 4.0, uv.y)).rgb; }

fn cardColor(h: Hit, k: f32, ed: i32, pattern: i32, coverage: i32, shinePhase: f32) -> vec3f {
  let ang = foilAngle(h.view);
  var c = faceAt(h.uv, k);
  let msk = maskAt(h.uv, k);
  var area = msk.r;                                             // art window background
  if (coverage == 1) { area = clamp(1.0 - msk.r - msk.g, 0.0, 1.0) * 0.8 + msk.b * 0.2; }   // reverse holo
  if (coverage == 2) { area = 1.0; }
  if (ed == 0) {
    // trading-card holo: rainbow by angle x pattern x diffraction bands, blended like "color dodge"
    let rb = rainbowFoil(h.uv, ang);
    let pm = patternMask(h.uv, pattern);
    let fb = foilBands(h.uv, ang);
    let amt = area * pm * fb * u.foil;
    c = mix(c, c * 0.55 + rb * 0.75, clamp(amt, 0.0, 1.0));
    c += rb * glitter(h.uv, ang, select(0.35, 0.9, pattern == 4)) * u.sparkle * max(area, 0.15);
  } else {
    c = applyEdition(c, h.uv, ang, ed - 0, u.foil);
    c += vec3f(1.0) * glitter(h.uv, ang, 0.25) * u.sparkle * 0.5;
  }
  // lamination: the shine band slides across the whole card
  c += vec3f(1.0, 0.98, 0.95) * shineBand(h.uv, ang, shinePhase) * u.shine * (0.25 + 0.5 * area);
  // a subtle edge highlight
  let e = min(min(h.uv.x, 1.0 - h.uv.x) * CARD_W, min(h.uv.y, 1.0 - h.uv.y));
  c += vec3f(0.25) * (1.0 - smoothstep(0.0, 0.012, e)) * (0.5 + 0.5 * ang.x);
  return c;
}

fn singleCard(px: vec2f) -> vec3f {
  let t = u.time;
  var col = swirlBg(px, t);
  let hpx = u.resolution.y * 0.78;
  let center = u.resolution * 0.5;
  let h = hitCard(px, center, hpx, CARD_W, u.tilt.x, u.tilt.y);
  col *= 1.0 - 0.55 * h.shadow;
  let shinePhase = fract(t * 0.12) * 3.0 - 1.5;
  let ed = i32(u.edition);
  let c = cardColor(h, 0.0, ed, i32(u.pattern), i32(u.coverage), shinePhase);
  return mix(col, c, h.cover);
}

fn editionsRow(px: vec2f) -> vec3f {
  let t = u.time;
  var col = swirlBg(px, t);
  let hpx = min(u.resolution.y * 0.58, u.resolution.x / (5.0 * CARD_W * 1.22));
  let gap = hpx * CARD_W * 1.18;
  let ci = clamp(floor((px.x - u.resolution.x * 0.5) / gap + 2.5), 0.0, 4.0);
  let center = vec2f(u.resolution.x * 0.5 + (ci - 2.0) * gap, u.resolution.y * 0.46);
  // each card leans toward the mouse a little and sways on its own
  let mrel = (u.mouse.xy - center) / hpx;
  let lean = vec2f(clamp(mrel.y, -1.0, 1.0), -clamp(mrel.x, -1.0, 1.0)) * u.tiltAmt * PI / 180.0 * 0.8 * u.mouse.w;
  let sway = vec2f(sin(t * 0.9 + ci * 1.3), sin(t * 0.7 + ci * 2.1)) * 0.12;
  let tl = lean + sway;
  let bob = sin(t * 1.4 + ci) * hpx * 0.012;
  let h = hitCard(px, center + vec2f(0.0, bob), hpx, CARD_W, tl.x, tl.y);
  col *= 1.0 - 0.5 * h.shadow;
  let ang = h.view.xy * 2.5 + vec2f(tl.y, -tl.x) * 1.3;
  var c = faceAt(h.uv, fmod(ci, 4.0));
  c = applyEdition(c, h.uv, ang, i32(ci), u.foil);
  c += vec3f(1.0) * shineBand(h.uv, ang, fract(t * 0.12 + ci * 0.13) * 3.0 - 1.5) * u.shine * 0.35;
  c += vec3f(1.0) * glitter(h.uv, ang, 0.2) * 0.4 * step(0.5, ci);
  return mix(col, c, h.cover);
}

// four panels showing the ingredients separately (same tilt everywhere)
fn howPanels(px: vec2f) -> vec3f {
  var col = mix(vec3f(0.07, 0.07, 0.1), vec3f(0.12, 0.1, 0.16), px.y / u.resolution.y);
  let hpx = min(u.resolution.y * 0.5, u.resolution.x / 4.0 / 1.25);
  let gap = hpx * 1.15;
  let ci = clamp(floor((px.x - u.resolution.x * 0.5) / gap + 2.0), 0.0, 3.0);
  let center = vec2f(u.resolution.x * 0.5 + (ci - 1.5) * gap, u.resolution.y * 0.45);
  let h = hitCard(px, center, hpx, 1.0, u.tilt.x, u.tilt.y);
  col *= 1.0 - 0.4 * h.shadow;
  let ang = foilAngle(h.view);
  let uvq = vec2f(h.uv.x, h.uv.y);
  let k = i32(ci);
  let pat = i32(u.pattern);
  var c = vec3f(0.18, 0.18, 0.22);
  if (k == 0) {
    c = rainbowFoil(uvq, ang) * foilBands(uvq, ang);
  } else if (k == 1) {
    c = vec3f(patternMask(uvq, pat)) * 0.85;
  } else if (k == 2) {
    c = vec3f(0.08) + vec3f(glitter(uvq, ang, 0.6) * u.sparkle);
  } else {
    let base = vec3f(0.2, 0.22, 0.3);
    let rb = rainbowFoil(uvq, ang);
    c = mix(base, base * 0.5 + rb * 0.8, clamp(patternMask(uvq, pat) * foilBands(uvq, ang) * u.foil, 0.0, 1.0));
    c += rb * glitter(uvq, ang, 0.6) * u.sparkle;
    c += vec3f(shineBand(uvq, ang, fract(u.time * 0.12) * 3.0 - 1.5) * u.shine * 0.6);
  }
  return mix(col, c, h.cover);
}

// --------------------------------------------------------------- shiny metal & gold
fn metalTint(k: i32) -> vec3f {
  var c = vec3f(1.0, 0.76, 0.33);                    // gold
  if (k == 1) { c = vec3f(0.9, 0.92, 0.95); }        // silver
  if (k == 2) { c = vec3f(1.0, 1.0, 1.0); }          // chrome
  if (k == 3) { c = vec3f(1.0, 0.66, 0.58); }        // rose gold
  if (k == 4) { c = vec3f(0.85, 0.52, 0.28); }       // bronze
  return c;
}

// procedural "studio" environment: bright sky, two soft-box lights, a dark floor and a horizon line
fn studioEnv(r0: vec3f, rot: f32) -> f32 {
  let r = rotY3(r0, rot);
  let el = r.y;                                      // + = down
  var e = mix(1.0, 0.06, smoothstep(-0.25, 0.25, el));
  e += 2.2 * (1.0 - smoothstep(0.0, 0.07, abs(el + 0.42)));            // horizontal soft box
  let az = atan2(r.x, -r.z);
  let s1 = (az - 0.9) * 5.0;
  let s2 = (az + 1.3) * 7.0;
  e += 1.6 * exp(-s1 * s1) * (1.0 - smoothstep(-0.1, 0.2, el));   // vertical strip light
  e += 0.8 * exp(-s2 * s2) * (1.0 - smoothstep(-0.3, 0.1, el));
  e += 0.5 * (1.0 - smoothstep(0.0, 0.03, abs(el - 0.02)));          // horizon glint
  return e;
}

fn shadeMetal(n: vec3f, tint: vec3f, chrome: f32, rot: f32) -> vec3f {
  let iv = vec3f(0.0, 0.0, 1.0);                    // view ray, going into the screen
  let r = reflect(iv, n);
  let e = studioEnv(r, rot);
  let fres = pow(1.0 - clamp(-dot(n, iv), 0.0, 1.0), 3.0);
  var c = tint * (0.12 + 0.85 * e) + vec3f(fres * 0.25);
  if (chrome > 0.5) { c = vec3f(0.05) + vec3f(0.95) * pow(e * 0.6, 1.4); }
  return c;
}

fn badgeSdf(p: vec2f, k: i32) -> vec2f {   // x = outer shape, y = embossed inner symbol
  if (k == 0) { return vec2f(length(p) - 0.13, sdStar5(p, 0.075, 0.45)); }
  if (k == 1) {
    var q = p;
    let shield = max(sdRoundBox(q - vec2f(0.0, -0.03), vec2f(0.11, 0.08), 0.02), length(q * vec2f(0.75, 1.0) - vec2f(0.0, -0.06)) - 0.15);
    return vec2f(shield, sdCross(p - vec2f(0.0, -0.01), vec2f(0.055, 0.016), 0.0));
  }
  return vec2f(sdHexagon(p, 0.12), sdHexagon(p, 0.06));
}

fn badgeHeight(p: vec2f, k: i32) -> f32 {
  let d = badgeSdf(p, k);
  let bev = u.bevel * 0.03;
  var h = smoothstep(0.0, bev, -d.x);                // outer bevel
  h -= 0.35 * (1.0 - smoothstep(0.0, 0.008, abs(d.x + bev * 1.6) - 0.004));   // engraved ring
  h += 0.6 * smoothstep(0.0, bev * 0.6, -d.y);       // raised symbol
  return h;
}

fn metalScene(px: vec2f) -> vec3f {
  let res = u.resolution;
  var col = mix(vec3f(0.1, 0.07, 0.14), vec3f(0.03, 0.02, 0.05), length((px - res * vec2f(0.5, 0.35)) / res.y));
  let rot = (u.mouse.x / res.x - 0.5) * 1.6 * u.mouse.w + 0.25 * sin(u.time * 0.4);
  let glintPhase = fract(u.time * 0.18) * 3.0 - 1.0;
  // ---- the text
  let tw = min(res.x * 0.88, res.y * 1.55);
  let th = tw * 0.25;
  let to = vec2f((res.x - tw) * 0.5, res.y * 0.34 - th * 0.5);
  let tuv = (px - to) / vec2f(tw, th);
  if (tuv.x > -0.05 && tuv.x < 1.05 && tuv.y > -0.2 && tuv.y < 1.25) {
    let shadow = TEX(metalText, tuv - vec2f(0.006, 0.035)).g;
    col *= 1.0 - 0.75 * shadow;
    let m = TEX(metalText, tuv);
    let e = vec2f(1.5) / vec2f(1024.0, 256.0);
    let hx = TEX(metalText, tuv + vec2f(e.x, 0.0)).g - TEX(metalText, tuv - vec2f(e.x, 0.0)).g;
    let hy = TEX(metalText, tuv + vec2f(0.0, e.y)).g - TEX(metalText, tuv - vec2f(0.0, e.y)).g;
    let n = normalize(vec3f(-hx * 6.0 * u.bevel, -hy * 6.0 * u.bevel, -1.0));
    var mc = shadeMetal(n, metalTint(i32(u.metal)), select(0.0, 1.0, i32(u.metal) == 2), rot);
    mc *= 0.55 + 0.45 * smoothstep(0.15, 0.6, m.b);                     // darker rim at the edges
    let gq = (tuv.x - tuv.y * 0.35) - glintPhase;
    let g = exp(-gq * gq * 120.0);
    mc += vec3f(1.0, 0.95, 0.85) * g * u.shine * 0.8;
    col = mix(col, mc, smoothstep(0.35, 0.65, m.r));
  }
  // ---- three badges: gold medal, silver shield, chrome hexagon
  let bs = min(res.y, res.x / 2.4);
  for (var k = 0; k < 3; k++) {
    let bc = vec2f(res.x * 0.5 + (f32(k) - 1.0) * bs * 0.42, res.y * 0.72);
    let p = (px - bc) / bs;
    let d = badgeSdf(p, k);
    if (d.x < 0.03) {
      col = mix(col, col * 0.3, (1.0 - smoothstep(-0.01, 0.03, badgeSdf(p - vec2f(0.01, 0.02), k).x)) * 0.8);
      let ee = 0.002;
      let gx = badgeHeight(p + vec2f(ee, 0.0), k) - badgeHeight(p - vec2f(ee, 0.0), k);
      let gy = badgeHeight(p + vec2f(0.0, ee), k) - badgeHeight(p - vec2f(0.0, ee), k);
      let n = normalize(vec3f(-gx, -gy, -2.0 * ee * 18.0));
      var tint = vec3f(1.0, 0.76, 0.33);
      if (k == 1) { tint = vec3f(0.88, 0.9, 0.95); }
      var mc = shadeMetal(n, tint, select(0.0, 1.0, k == 2), rot + f32(k) * 0.4);
      let gq = (p.x - p.y * 0.5) * 3.0 - glintPhase * 1.5;
      let g = exp(-gq * gq * 40.0);
      mc += vec3f(1.0) * g * u.shine * 0.6;
      col = mix(col, mc, clamp(0.5 - d.x * bs, 0.0, 1.0));
    }
  }
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = singleCard(px); }
  else if (ex == 1) { col = editionsRow(px); }
  else if (ex == 2) { col = howPanels(px); }
  else { col = metalScene(px); }
  let v = uv - 0.5;
  col *= 1.0 - dot(v, v) * 0.45;
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

// --- JS-side state: smoothed card tilt and the overlay labels
const tilt = { x: 0, y: 0 };
const labels = { els: [], host: null };
function showLabels(ctx, items) {
  if (labels.host !== ctx.overlay) {
    labels.els.forEach((e) => e.remove());
    labels.els = [];
    labels.host = ctx.overlay;
  }
  while (labels.els.length < 5) {
    const e = document.createElement('div');
    e.className = 'tag';
    ctx.overlay.append(e);
    labels.els.push(e);
  }
  labels.els.forEach((e, i) => {
    const it = items[i];
    e.hidden = !it;
    if (!it) return;
    if (e.textContent !== it.text) e.textContent = it.text;
    const css = `left:${it.x.toFixed(2)}%;top:${it.y.toFixed(2)}%;transform:translateX(-50%)`;
    if (e._css !== css) {
      e.style.cssText = css;
      e._css = css;
    }
  });
}

export default shaderScene({
  interaction: 'Move the mouse over the card to tilt it — the foil reacts to the viewing angle.',
  examples: [
    {
      id: 'card',
      label: 'Holo trading card',
      kind: 'In a game',
      note: 'Move the mouse: the card tilts in 3D, the rainbow shifts with the viewing angle, glitter flakes flash when they face you and a shine band sweeps across. Try the patterns (stars, cosmos…), reverse-holo coverage and the Balatro editions.',
    },
    {
      id: 'editions',
      label: 'Balatro editions',
      kind: 'In a game',
      note: 'Base, Foil, Holographic, Polychrome and Negative — the five card editions from Balatro, re-created as small color functions of <i>(card position, view angle)</i>, over a swirling pixel-paint background.',
      params: { foil: 1.0 },
    },
    {
      id: 'how',
      label: 'How foil works',
      kind: 'Abstract',
      note: 'The ingredients, separated: ① hue from position + viewing angle, ② the printed foil pattern (mask), ③ glitter flakes that only flash at their own angle, ④ everything combined. All panels share the same tilt.',
      params: { pattern: 'stars' },
    },
    {
      id: 'metal',
      label: 'Shiny metal & gold',
      kind: 'Real life',
      note: 'Gold UI text and badges: a bevel gives every pixel a surface normal, the normal reflects a made-up “photo studio” (bright sky, soft-box lights, dark floor). Move the mouse to rotate the studio and watch the reflections slide.',
    },
  ],
  controls: [
    {
      type: 'select', key: 'edition', label: 'Edition / finish', value: 'holo', showFor: ['card'],
      options: [
        { value: 'holo', label: 'Holo foil (trading card)' },
        { value: 'foil', label: 'Foil (Balatro)' },
        { value: 'holographic', label: 'Holographic (Balatro)' },
        { value: 'polychrome', label: 'Polychrome (Balatro)' },
        { value: 'negative', label: 'Negative (Balatro)' },
      ],
    },
    {
      type: 'select', key: 'pattern', label: 'Foil pattern', value: 'stars', showFor: ['card', 'how'],
      options: [
        { value: 'smooth', label: 'Smooth rainbow' },
        { value: 'stars', label: 'Stars' },
        { value: 'stripes', label: 'Linear stripes' },
        { value: 'cosmos', label: 'Cosmos' },
        { value: 'glitter', label: 'Glitter / sequins' },
      ],
      help: 'The printed mask that decides where the foil shows.',
    },
    {
      type: 'select', key: 'coverage', label: 'Foil coverage', value: 'art', showFor: ['card'],
      options: [
        { value: 'art', label: 'Art window (classic holo)' },
        { value: 'reverse', label: 'Reverse holo (everything else)' },
        { value: 'full', label: 'Full card' },
      ],
    },
    { type: 'slider', key: 'foil', label: 'Foil strength', min: 0, max: 1.5, step: 0.01, value: 0.9, showFor: ['card', 'editions', 'how'] },
    { type: 'slider', key: 'sparkle', label: 'Glitter', min: 0, max: 1.5, step: 0.01, value: 0.7, showFor: ['card', 'how'], help: 'Tiny flakes with random orientations.' },
    { type: 'slider', key: 'shine', label: 'Shine sweep', min: 0, max: 1.5, step: 0.01, value: 0.8, help: 'A glossy highlight band that moves with the tilt (and on its own).' },
    { type: 'slider', key: 'tiltAmt', label: 'Max tilt', min: 0, max: 40, step: 1, value: 24, unit: '°', showFor: ['card', 'editions', 'how'] },
    { type: 'slider', key: 'swirl', label: 'Background swirl', min: 0, max: 3, step: 0.01, value: 1, showFor: ['card', 'editions'], help: 'Speed of the Balatro-style paint swirl.' },
    {
      type: 'select', key: 'metal', label: 'Text metal', value: 'gold', showFor: ['metal'],
      options: [
        { value: 'gold', label: 'Gold' },
        { value: 'silver', label: 'Silver' },
        { value: 'chrome', label: 'Chrome' },
        { value: 'rose', label: 'Rose gold' },
        { value: 'bronze', label: 'Bronze' },
      ],
    },
    { type: 'slider', key: 'bevel', label: 'Bevel depth', min: 0.2, max: 3, step: 0.01, value: 1.2, showFor: ['metal'], help: 'How steep the beveled edges are — steeper = more reflections.' },
  ],
  uniforms: {
    edition: 'f32', pattern: 'f32', coverage: 'f32', foil: 'f32', sparkle: 'f32', shine: 'f32', tiltAmt: 'f32', swirl: 'f32',
    metal: 'f32', bevel: 'f32', tilt: 'vec2f',
  },
  include: ['hash', 'noise', 'sdf', 'color', 'math'],
  textures: {
    faces: { source: async () => (await buildFaces()).faces },
    masks: { source: async () => (await buildFaces()).masks },
    metalText: { source: async () => buildMetalText() },
  },
  bind(p, ctx) {
    // smoothed 3D tilt: follow the mouse while it is over the canvas, otherwise sway gently
    const max = (p.tiltAmt * Math.PI) / 180;
    let tx;
    let ty;
    if (ctx.pointer.over) {
      const nx = Math.max(-1, Math.min(1, (ctx.pointer.x / ctx.width - 0.5) * 2.2));
      const ny = Math.max(-1, Math.min(1, (ctx.pointer.y / ctx.height - 0.5) * 2.2));
      tx = ny * max; // rotate around X: the half under the cursor is pushed away
      ty = -nx * max;
    } else {
      tx = Math.sin(ctx.time * 0.7) * max * 0.5;
      ty = Math.sin(ctx.time * 0.45) * max * 0.7;
    }
    const k = 1 - Math.exp(-Math.min(ctx.dt || 0.016, 0.1) * 8);
    tilt.x += (tx - tilt.x) * k;
    tilt.y += (ty - tilt.y) * k;
    // labels under the cards
    if (ctx.example === 'editions') {
      const hpx = Math.min(ctx.height * 0.58, ctx.width / (5 * 0.7142857 * 1.22));
      const gap = hpx * 0.7142857 * 1.18;
      const names = ['Base', 'Foil', 'Holographic', 'Polychrome', 'Negative'];
      showLabels(ctx, names.map((text, i) => ({ text, x: ((ctx.width * 0.5 + (i - 2) * gap) / ctx.width) * 100, y: ((ctx.height * 0.46 + hpx * 0.56) / ctx.height) * 100 })));
    } else if (ctx.example === 'how') {
      const hpx = Math.min(ctx.height * 0.5, ctx.width / 4 / 1.25);
      const gap = hpx * 1.15;
      const names = ['① angle → rainbow', '② pattern mask', '③ glitter flakes', '④ combined'];
      showLabels(ctx, names.map((text, i) => ({ text, x: ((ctx.width * 0.5 + (i - 1.5) * gap) / ctx.width) * 100, y: ((ctx.height * 0.45 + hpx * 0.58) / ctx.height) * 100 })));
    } else showLabels(ctx, []);
    return { tilt: [tilt.x, tilt.y] };
  },
  code: CODE,
  about: {
    summary: 'Foil is just color that depends on the viewing angle. Tilt a card with the mouse and watch rainbows, glitter and shine move — the effect behind Balatro editions, Pokémon holos and golden cards.',
    what: `<p>A trading card rendered as a tilted 3D plane, with holographic foil printed on parts of it, the five Balatro card editions,
      the separate ingredients of a foil shader, and beveled gold text & badges reflecting a fake studio environment.</p>`,
    how: `<ol>
      <li><b>Tilt.</b> The mouse sets two rotation angles (smoothed in JavaScript). Per pixel, a ray from a virtual camera hits the rotated card plane →
        card UV <i>and</i> the direction toward the eye in card space. That direction is the “view angle”.</li>
      <li><b>Rainbow.</b> <code>hue = fract(position·k + viewAngle·m)</code>: neighbouring points have shifted hues and tilting slides the whole rainbow — like real diffraction foil.</li>
      <li><b>Diffraction bands.</b> A cosine of the same mix brightens stripes that sweep across as the card moves.</li>
      <li><b>Pattern mask.</b> Stars, stripes or “cosmos” circles decide where the foil is printed; the mask texture decides <i>which part</i> of the card (art window, everything else, whole card).</li>
      <li><b>Glitter.</b> Split the card into tiny cells; each flake gets a random preferred angle and only lights up when the current view angle is close to it.</li>
      <li><b>Shine sweep.</b> A narrow Gaussian band along a diagonal, offset by the tilt and by time.</li>
      <li><b>Editions.</b> Balatro-style: Foil = desaturated blue streaks, Holographic = red/cyan through a lattice, Polychrome = hue rotation that varies over the card, Negative = inverted colors.</li>
      <li><b>Metal.</b> Text mask blurred → height; its gradient → normal; <code>reflect()</code> the view ray and look up a procedural environment (bright top, soft boxes, dark floor). Tint it gold/silver/bronze.</li>
    </ol>`,
    uses: [
      { title: 'Card games', text: 'Balatro (Foil, Holographic, Polychrome, Negative editions), Pokémon TCG Pocket & Live, Hearthstone golden cards, Marvel Snap foil/prism variants, MTG Arena foils.' },
      { title: 'Rarity & rewards', text: 'Make rare loot, legendary items and premium skins feel special: a shiny sweep and glitter instantly say “valuable”.' },
      { title: 'UI', text: 'Gold “VICTORY” banners, achievement badges, rank emblems, metallic buttons and logos in menus.' },
      { title: 'Collectibles & stickers', text: 'Holo stickers, sparkly badges and foil stamps in collection screens.' },
    ],
    try: [
      'On the card, move the mouse slowly from corner to corner — every flake of glitter has its own “sweet spot”.',
      'Switch <i>Foil coverage</i> to Reverse holo: the art stays matte and the frame and text box shimmer.',
      'Set <i>Edition</i> to Polychrome, then Negative — the same card art, totally different rarity feel.',
      'On <b>How foil works</b>, set <i>Glitter</i> to 0 and <i>Foil strength</i> to 0 one at a time to see what each ingredient adds.',
      'On <b>Shiny metal</b>, choose Chrome and move the mouse left and right: reflections of the “studio lights” slide over the letters.',
    ],
    ask: [
      'a holographic foil card shader that reacts to mouse tilt',
      'Balatro-style card editions: foil, holographic, polychrome, negative',
      'view-angle-dependent rainbow with glitter sparkles',
      'a shine sweep across rare items',
      'gold beveled UI text with environment reflections',
      'Balatro-like swirling paint background',
    ],
    perf: `<p>Cheap: one ray–plane intersection per pixel, two texture reads for the card, a few hashes for glitter and patterns.
      The background swirl does 5 warp iterations per (chunky) pixel. In a game you would run the foil only on the card quads (sprites),
      so the cost is proportional to the cards’ on-screen area.</p>`,
    api: `<p>All fragment-shader math, identical on WebGPU and WebGL2. In a real engine you would draw each card as a quad and pass its tilt as a per-instance value —
      on WebGPU, hundreds of animated cards fit in one instanced draw call.</p>`,
    code: [
      {
        title: 'Card tilt: ray / plane intersection gives UV and view angle',
        lang: 'wgsl',
        src: `let p = (px - center) / hpx;                         // card height = 1
let o = vec3f(0.0, 0.0, -dist);
let dir = normalize(vec3f(p, dist));
let n = rotY3(rotX3(vec3f(0.0, 0.0, -1.0), rx), ry);  // tilted card normal
let t = -dot(o, n) / dot(dir, n);
let L = rotX3(rotY3(o + dir * t, -ry), -rx);          // hit point in card space
uv = vec2f(L.x / CARD_W + 0.5, L.y + 0.5);
view = rotX3(rotY3(-dir, -ry), -rx);                  // toward the eye, in card space`,
      },
      {
        title: 'Foil = f(position, angle)',
        lang: 'wgsl',
        src: `let ang = view.xy * 2.5 + vec2f(u.tilt.y, -u.tilt.x) * 1.3;
let hue = fract(dot(uv - 0.5, vec2f(0.8, 1.2)) * 1.1 + ang.x * 0.9 - ang.y * 0.7);
let rb = hsv2rgb(vec3f(hue, 0.7, 1.0));
let amt = area * patternMask(uv, pattern) * foilBands(uv, ang) * u.foil;
c = mix(c, c * 0.55 + rb * 0.75, clamp(amt, 0.0, 1.0));
// glitter: each flake has its own preferred angle
let flake = (hash33(vec3f(id, 5.0)).xy - 0.5) * 2.6;
let align = 1.0 - smoothstep(0.0, 0.3, length(flake - ang));`,
      },
      {
        title: 'Metal: bevel normal → reflection',
        lang: 'wgsl',
        src: `let hx = TEX(metalText, tuv + vec2f(e.x, 0.0)).g - TEX(metalText, tuv - vec2f(e.x, 0.0)).g;
let hy = TEX(metalText, tuv + vec2f(0.0, e.y)).g - TEX(metalText, tuv - vec2f(0.0, e.y)).g;
let n = normalize(vec3f(-hx * 6.0 * u.bevel, -hy * 6.0 * u.bevel, -1.0));
let r = reflect(vec3f(0.0, 0.0, 1.0), n);
let c = gold * (0.12 + 0.85 * studioEnv(r, rotation));`,
      },
    ],
  },
});

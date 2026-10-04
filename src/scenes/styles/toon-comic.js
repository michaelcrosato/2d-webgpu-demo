import { shaderScene } from '../../core/shaderscene.js';
import { makeCanvas } from '../../core/assets.js';
import { setTag, placeTag, gamePass, GAME_INCLUDES } from './_shared-a.js';

// Comic, manga, cel shading and pop art — all as post-processing of the game image (plus an
// abstract "toon ramp" demo on lit spheres). Building blocks:
//   ink lines   : Sobel edge detection on the image (8 taps at the chosen line thickness)
//   posterize   : quantize lightness (OKLab L) into N bands, keep and boost the hue/chroma
//   halftone    : a "spot function" on a rotated grid; ink where spot > 1 - amount
//   screentones : dot / line / cross-hatch patterns chosen by tone level (manga)

const st = { testMode: false, light: [0.5, -0.6, 0.62] };

// "POW!" lettering for the action burst
let powCanvas = null;
function makePow() {
  if (powCanvas) return powCanvas;
  const c = makeCanvas(256, 128);
  const g = c.getContext('2d');
  g.clearRect(0, 0, 256, 128);
  g.font = 'italic 900 84px Impact, "Arial Black", sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.lineJoin = 'round';
  g.lineWidth = 14;
  g.strokeStyle = '#111111';
  g.strokeText('POW!', 128, 68);
  g.fillStyle = '#e8231f';
  g.fillText('POW!', 128, 68);
  g.fillStyle = 'rgba(255,255,255,0.55)';
  g.fillText('POW!', 125, 64);
  g.fillStyle = '#e8231f';
  g.fillText('POW!', 127, 67);
  powCanvas = c;
  return c;
}

const CODE = /* wgsl */ `
fn img(p: vec2f) -> vec3f { return TEX(scene, p / u.resolution).rgb; }

// Sobel edge strength on RGB, taps w pixels apart (= line thickness)
fn edgeAt(p: vec2f, w: f32) -> f32 {
  let tl = img(p + vec2f(-w, -w)); let tt = img(p + vec2f(0.0, -w)); let tr = img(p + vec2f(w, -w));
  let ml = img(p + vec2f(-w, 0.0));                                   let mr = img(p + vec2f(w, 0.0));
  let bl = img(p + vec2f(-w, w));  let bb = img(p + vec2f(0.0, w));  let br = img(p + vec2f(w, w));
  let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  let gy = (bl + 2.0 * bb + br) - (tl + 2.0 * tt + tr);
  return sqrt(dot(gx, gx) + dot(gy, gy));
}
fn inkMask(p: vec2f) -> f32 {
  let e = edgeAt(p, u.inkWidth);
  let thr = mix(2.2, 0.35, u.ink);
  return smoothstep(thr, thr * 1.35, e) * step(0.01, u.ink);
}
// Posterize: N lightness bands in OKLab, hue kept, chroma scaled by sat
fn posterize(c: vec3f, n: f32, sat: f32) -> vec3f {
  let lab = linearToOklab(srgbToLinear(clamp(c, vec3f(0.0), vec3f(1.0))));
  let q = (floor(clamp(lab.x, 0.0, 0.999) * n) + 0.6) / n;
  return clamp(linearToSrgb(oklabToLinear(vec3f(q, lab.y * sat, lab.z * sat))), vec3f(0.0), vec3f(1.0));
}
// Halftone screen: spot function is 1 at dot centres, 0 between; ink where spot > 1 - amount.
fn spotFn(p: vec2f, ang: f32, cell: f32) -> f32 {
  let q = rot2(ang) * p / cell;
  return 0.5 + 0.25 * (cos(TAU * q.x) + cos(TAU * q.y));
}
fn screen(p: vec2f, ang: f32, cell: f32, amount: f32) -> f32 {
  let s = spotFn(p, ang, cell);
  let aa = max(fwidth(s), 0.002) * 0.8;
  return smoothstep(1.0 - amount - aa, 1.0 - amount + aa, s) * step(0.004, amount);
}
// line screen (manga): lines at angle ang, period in px, coverage 0..1
fn lineScreen(p: vec2f, ang: f32, period: f32, cover: f32) -> f32 {
  let d = abs(fract(dot(p, vec2f(cos(ang), sin(ang))) / period) - 0.5) * period;
  let w = cover * period * 0.5;
  return 1.0 - smoothstep(w - 0.6, w + 0.6, d);
}
fn paperTex(p: vec2f) -> vec3f {
  let fib = valueNoise(p * vec2f(0.9, 0.25)) * 0.5 + valueNoise(p * 0.08) * 0.5;
  return vec3f(0.975, 0.955, 0.9) - vec3f(0.06, 0.06, 0.07) * fib * u.paper;
}
// hero position in the game scene (mirrors core/gamescene.js) — focus of speed lines
fn heroPx() -> vec2f {
  let hop = abs(sin(u.time * 2.6));
  return vec2f(0.45, 0.8 - 0.042 - hop * 0.13) * u.resolution.y;
}
// manga "focus lines": thin black wedges pointing at the hero, crowding the frame edges
fn speedLines(p: vec2f) -> f32 {
  let res = u.resolution;
  let v = p - heroPx();
  let a = atan2(v.y, v.x) / TAU + 0.5;
  let n = 150.0;
  let cell = floor(a * n);
  let h = hash11(cell * 1.31 + floor(u.time * 6.0) * 0.17);
  // how close to the frame edge (0 = on the edge), with a random reach per line
  let edge = min(min(p.x, res.x - p.x), min(p.y, res.y - p.y)) / res.y;
  let reach = 0.08 + 0.2 * h;
  let k = clamp(1.0 - edge / reach, 0.0, 1.0);
  let f = abs(fract(a * n) - 0.5) * 2.0;            // 0 at the wedge centre
  let wid = k * k * 0.55 * step(0.35, h);
  let aa = min(n / (TAU * max(length(v), 1.0)) * 1.5, 0.3);
  return (1.0 - smoothstep(wid - aa, wid + aa, f)) * step(0.02, wid);
}
// "POW!" starburst
fn burst(p: vec2f) -> vec4f {
  let res = u.resolution;
  let c = vec2f(0.79 * res.x, 0.27 * res.y);
  let R = 0.15 * res.y * (1.0 + 0.04 * sin(u.time * 9.0));
  let v = (p - c) / R;
  let a = atan2(v.y, v.x);
  let rays = 13.0;
  let k = a / TAU * rays;
  let tri = abs(fract(k) - 0.5) * 2.0;
  let jag = 0.68 + 0.32 * (1.0 - tri) * (0.75 + 0.25 * hash11(floor(k) + 3.0));
  let d = (length(v) - jag) * R;                   // approximate distance in px
  let fill = 1.0 - smoothstep(-0.8, 0.8, d);
  let outline = 1.0 - smoothstep(-0.8, 0.8, d - 5.0);
  var col = mix(vec3f(0.1), vec3f(1.0, 0.86, 0.12), fill);
  // a second, red inner ring
  col = mix(col, vec3f(0.92, 0.22, 0.12), (1.0 - smoothstep(-0.8, 0.8, abs(d + 9.0) - 2.5)) * fill);
  // lettering from a Canvas2D texture
  let tq = (p - c) / vec2f(R * 0.9, R * 0.45) * 0.5 + 0.5;
  if (tq.x > 0.0 && tq.y > 0.0 && tq.x < 1.0 && tq.y < 1.0) {
    let t = TEX(powTex, vec2f(tq.x, tq.y));
    col = mix(col, t.rgb, t.a);
  }
  return vec4f(col, outline);
}

// ---------------------------------------------------------------- comic book (CMYK halftone)
fn comicView(p: vec2f) -> vec3f {
  let base = posterize(img(p), u.bands, u.sat);
  // RGB → CMYK with some under-color removal
  let k = 1.0 - max(max(base.r, base.g), base.b);
  let inv = 1.0 / max(1.0 - k, 0.001);
  // like old comics, only a few tint levels per ink (0, 33, 66, 100%) → flat, regular dot fields
  let cmy = floor((vec3f(1.0) - base - vec3f(k)) * inv * 3.0 + 0.5) / 3.0;
  let mis = u.paper * 1.5;                          // plate misregistration in px
  let cell = u.dot;
  let cC = screen(p + vec2f(mis, 0.0), 0.2618, cell, cmy.x);
  let cM = screen(p + vec2f(0.0, mis * 0.7), 1.309, cell, cmy.y);
  let cY = screen(p, 0.0, cell, cmy.z);
  let cK = screen(p - vec2f(mis * 0.4, 0.0), 0.7854, cell, k * 0.85);
  var col = paperTex(p);
  col *= mix(vec3f(1.0), vec3f(0.05, 0.72, 0.95), cC);
  col *= mix(vec3f(1.0), vec3f(0.95, 0.12, 0.55), cM);
  col *= mix(vec3f(1.0), vec3f(1.0, 0.92, 0.08), cY);
  col *= mix(vec3f(1.0), vec3f(0.12, 0.11, 0.13), cK);
  return col;
}

// ---------------------------------------------------------------- manga screentones
fn mangaView(p: vec2f) -> vec3f {
  let l = clamp((luma(img(p)) - 0.5) * 1.35 + 0.5, 0.0, 1.0);
  let cell = u.dot;
  var inkAmt = 0.0;
  if (l < 0.2) { inkAmt = 1.0; }
  else if (l < 0.38) { inkAmt = max(lineScreen(p, 0.785, cell * 0.6, 0.42), lineScreen(p, -0.785, cell * 0.6, 0.42)); }
  else if (l < 0.55) { inkAmt = lineScreen(p, 0.785, cell * 0.55, 0.32); }
  else if (l < 0.74) { inkAmt = screen(p, 0.785, cell, 0.2); }
  else if (l < 0.86) { inkAmt = screen(p, 0.785, cell * 0.8, 0.07); }
  let paper = paperTex(p) * vec3f(1.0, 1.0, 1.02);
  return mix(paper, vec3f(0.07), inkAmt);
}

// ---------------------------------------------------------------- pop art (Lichtenstein)
fn popView(p: vec2f) -> vec3f {
  let c = clamp(adjustSaturation(img(p), 1.3), vec3f(0.0), vec3f(1.0));
  let q = linearToOklab(srgbToLinear(c));
  // bold palette; "skin" and "sky" are not inks but Ben-Day dot patterns (red/blue dots on white)
  var pal = array<vec3f, 8>(vec3f(0.07, 0.07, 0.08), vec3f(0.98, 0.96, 0.9), vec3f(0.86, 0.13, 0.15), vec3f(0.98, 0.83, 0.1),
                            vec3f(0.12, 0.33, 0.68), vec3f(0.15, 0.6, 0.32), vec3f(0.95, 0.72, 0.66), vec3f(0.6, 0.74, 0.92));
  var best = 1.0e9;
  var bi = 0;
  for (var i = 0; i < 8; i++) {
    let e = linearToOklab(srgbToLinear(pal[i])) - q;
    let dd = dot(e, e);
    if (dd < best) { best = dd; bi = i; }
  }
  var col = pal[bi];
  let cell = u.dot * 1.5;
  let g = rot2(0.785) * p / cell;
  let dotD = length(fract(g) - 0.5);
  let dotC = 1.0 - smoothstep(0.29, 0.29 + 1.5 / cell, dotD);
  if (bi == 6) { col = mix(pal[1], pal[2], dotC); }
  if (bi == 7) { col = mix(pal[1], pal[4], dotC); }
  return col * mix(vec3f(1.0), paperTex(p) / vec3f(0.975, 0.955, 0.9), 0.6);
}

// ---------------------------------------------------------------- cel shading demo
fn celSpheres(p0: vec2f) -> vec3f {
  let res = u.resolution;
  let asp = res.x / res.y;
  let p = (p0 - 0.5 * res) / res.y;
  let L = normalize(u.light);
  let H = normalize(L + vec3f(0.0, 0.0, 1.0));
  var col = mix(vec3f(0.93, 0.9, 0.84), vec3f(0.8, 0.84, 0.9), p0.y / res.y);
  let spacing = min(0.4, asp * 0.31);
  let r = min(0.17, spacing * 0.42);
  for (var i = 0; i < 3; i++) {
    let fi = f32(i);
    let c = vec2f((fi - 1.0) * spacing, -0.03);
    // shadow on the floor (soft for the realistic one, hard for the toon ones)
    let sv = (p - c - vec2f(-L.x * 0.08, r * 1.12)) / vec2f(r * 1.05, r * 0.22);
    let sd = length(sv) - 1.0;
    var sh = 1.0 - smoothstep(-0.6, 0.4, sd);
    if (i > 0) { sh = 1.0 - smoothstep(-0.03, 0.03, sd); }
    col = mix(col, col * 0.72, sh * 0.8);
    let q = (p - c) / r;
    let d2 = dot(q, q);
    let edgeD = (sqrt(d2) - 1.0) * r;              // signed distance to the silhouette
    let aa = 1.5 / res.y;
    let cover = 1.0 - smoothstep(-aa, 0.0, edgeD);
    if (d2 < 1.0) {
      let n = vec3f(q.x, q.y, sqrt(max(1.0 - d2, 0.0)));
      let lam = max(dot(n, L), 0.0);
      let base = vec3f(0.96, 0.42, 0.26);
      var s = vec3f(0.0);
      if (i == 0) {
        s = base * (0.12 + 0.88 * lam) + vec3f(pow(max(dot(n, H), 0.0), 40.0) * 0.6);
      } else {
        // toon ramp: lighting quantized into bands; shadows shift towards a cool purple
        let band = clamp(floor(lam * u.bands) / max(u.bands - 1.0, 1.0), 0.0, 1.0);
        let shadow = vec3f(0.42, 0.16, 0.34);
        s = mix(shadow, base, band);
        if (i == 2) {
          s = mix(s, vec3f(1.0, 0.97, 0.9), step(0.94, dot(n, H)));                 // hard specular
          s = mix(s, vec3f(0.75, 0.85, 1.0), step(0.78, 1.0 - n.z) * step(lam, 0.05)); // rim light
        }
      }
      col = mix(col, s, cover);
    }
    if (i == 2) {
      let w = 0.0035 * u.inkWidth;
      col = mix(col, vec3f(0.08, 0.06, 0.1), (1.0 - smoothstep(w - aa, w + aa, abs(edgeD + w * 0.5))) * step(0.01, u.ink));
    }
  }
  return col;
}
fn celGame(p: vec2f) -> vec3f { return posterize(img(p), u.bands, u.sat); }

fn shade(uv: vec2f, px0: vec2f) -> vec4f {
  let p = uv * u.resolution;      // canvas pixels (independent of render scale)
  let ex = i32(u.example);
  var col = vec3f(0.0);
  var inkLines = 1.0;
  if (ex == 0) { col = comicView(p); }
  else if (ex == 1) { col = mangaView(p); inkLines = 1.3; }
  else if (ex == 2) {
    if (u.source < 0.5) { col = celSpheres(p); inkLines = 0.0; } else { col = celGame(p); }
  } else { col = popView(p); inkLines = 1.2; }
  // ink outlines
  if (inkLines > 0.0) { col = mix(col, vec3f(0.06, 0.05, 0.08), clamp(inkMask(p) * inkLines, 0.0, 1.0)); }
  // action FX
  if (u.fx > 0.5 && (ex == 0 || ex == 1)) {
    let sl = speedLines(p);
    col = mix(col, vec3f(0.06), sl);
    if (ex == 0) {
      let b = burst(p);
      col = mix(col, b.rgb, b.a);
    }
  }
  return vec4f(col, 1.0);
}`;

export default shaderScene({
  interaction: '',
  examples: [
    {
      id: 'comic',
      label: 'Comic book',
      kind: 'In a game',
      hint: '',
      note: 'Printed-comic look: colors flattened into a few bands, black <b>ink outlines</b> from edge detection, and four rotated <b>CMYK halftone</b> dot screens (cyan, magenta, yellow, black) slightly misregistered on cheap paper. Plus speed lines and a POW!.',
      params: { bands: 4, ink: 0.65, inkWidth: 1.5, dot: 7, sat: 1.25, fx: true, paper: 0.6 },
    },
    {
      id: 'manga',
      label: 'Manga screentone',
      kind: 'Classic',
      hint: '',
      note: 'Black and white only. Each tone range gets its own <b>screentone</b> — the adhesive dot and line sheets manga artists cut out: light dots, diagonal lines, cross-hatching, solid black — with bold ink lines and focus lines aimed at the hero.',
      params: { ink: 0.6, inkWidth: 1.6, dot: 5, fx: true, paper: 0.4 },
    },
    {
      id: 'cel',
      label: 'Cel shading',
      kind: 'Abstract',
      hint: 'Move the mouse to move the light.',
      note: 'Left: smooth shading. Middle: the same lighting <b>quantized into bands</b>, shadows shifted to a cool color. Right: plus a hard specular highlight, rim light and an ink outline — the anime / Wind Waker recipe. Switch the source to apply it to the game.',
      params: { bands: 3, ink: 0.7, inkWidth: 1.4, sat: 1.3, source: 0 },
    },
    {
      id: 'popart',
      label: 'Pop art',
      kind: 'Classic',
      hint: '',
      note: 'Roy Lichtenstein’s style: a handful of bold flat inks, thick black outlines, and light tones (skin, sky) printed as big <b>Ben-Day dots</b> — red or blue dots on white.',
      params: { ink: 0.7, inkWidth: 2.2, dot: 7, paper: 0.5 },
    },
  ],
  controls: [
    { type: 'slider', key: 'bands', label: 'Color bands', min: 2, max: 8, step: 1, value: 4, showFor: ['comic', 'cel'], help: 'Lightness levels after posterizing (cel shading = 2–4).' },
    { type: 'slider', key: 'ink', label: 'Ink lines', min: 0, max: 1, step: 0.01, value: 0.65, help: 'Edge sensitivity: higher = more (and weaker) edges get inked.' },
    { type: 'slider', key: 'inkWidth', label: 'Line thickness', min: 0.5, max: 4, step: 0.1, value: 1.5, help: 'Distance between the edge-detection taps, in pixels.' },
    { type: 'slider', key: 'dot', label: 'Dot / screen size', min: 3, max: 16, step: 0.5, value: 6, showFor: ['comic', 'manga', 'popart'], help: 'Halftone cell size in pixels.' },
    { type: 'slider', key: 'sat', label: 'Saturation', min: 0, max: 2, step: 0.01, value: 1.3, showFor: ['comic', 'cel'] },
    { type: 'slider', key: 'paper', label: 'Paper & print wear', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['comic', 'manga', 'popart'], help: 'Paper fibre texture and CMYK plate misregistration.' },
    { type: 'toggle', key: 'fx', label: 'Speed lines & POW!', value: true, showFor: ['comic', 'manga'] },
    {
      type: 'select', key: 'source', label: 'Source', value: 0, showFor: ['cel'],
      options: [{ value: 0, label: 'Lit spheres (abstract)' }, { value: 1, label: 'Game scene' }],
    },
  ],
  uniforms: { bands: 'f32', ink: 'f32', inkWidth: 'f32', dot: 'f32', sat: 'f32', paper: 'f32', fx: 'f32', source: 'f32', light: 'vec3f' },
  include: ['color', ...GAME_INCLUDES],
  passes: [gamePass('scene')],
  textures: { powTex: { source: async () => makePow(), filter: 'linear' } },
  renderScale: () => (st.testMode ? 0.5 : 1),
  bind(params, ctx) {
    st.testMode = !!ctx.testMode;
    const p = ctx.pointer;
    // light direction: follows the mouse, otherwise slowly orbits
    let target;
    if (p.over) target = [(p.nx - 0.5) * 2.2, (p.ny - 0.5) * 2.2, 0.8];
    else target = [Math.cos(ctx.time * 0.5) * 0.8, -0.55 + 0.25 * Math.sin(ctx.time * 0.5), 0.6];
    if (!ctx.paused || p.over) for (let i = 0; i < 3; i++) st.light[i] += (target[i] - st.light[i]) * 0.08;
    const spheres = ctx.example === 'cel' && !params.source;
    const asp = ctx.width / Math.max(1, ctx.height);
    const spacing = Math.min(0.4, asp * 0.31);
    const labels = ['Smooth shading', `${params.bands | 0} cel bands`, '+ spec, rim & ink'];
    for (let i = 0; i < 3; i++) {
      const el = setTag(ctx, `s${i}`, labels[i], '', spheres);
      if (spheres) placeTag(el, 50 + (((i - 1) * spacing) / asp) * 100, 50 + (-0.03 + Math.min(0.17, spacing * 0.42) + 0.075) * 100, '-50%');
    }
    return { light: st.light };
  },
  code: CODE,
  about: {
    summary: 'Comic and cartoon looks come from a few cheap image operations: find edges and ink them, flatten colors into bands, and replace smooth tones with printed patterns (halftone dots, screentones).',
    what: `<p>The platformer re-printed as a color comic (CMYK halftone), a black-and-white manga page (screentones), and a Lichtenstein pop-art panel —
      plus an abstract demo of <b>cel shading</b> on lit spheres.</p>`,
    how: `<ol>
      <li><b>Ink lines = edge detection.</b> A Sobel filter compares the 8 neighbours of each pixel (left vs right, top vs bottom). A big difference means an
        edge → paint it black. Spreading the taps further apart gives thicker lines.</li>
      <li><b>Posterize / cel bands.</b> Convert to OKLab, round the lightness <code>L</code> to N levels, keep (and boost) the color. Large areas become flat
        fills, like an inked-and-colored cartoon. In 3D cel shading the same rounding is applied to the <i>lighting</i> (<code>floor(N·dot(n, l)) / N</code>),
        usually with shadow colors shifted cooler instead of just darker.</li>
      <li><b>Halftone.</b> Printers fake shades with dots of one ink. For each ink, take a grid rotated to its own angle (C 15°, M 75°, Y 0°, K 45° to avoid moiré),
        evaluate a smooth “spot function” that is 1 at each dot centre, and ink the pixel where <code>spot &gt; 1 − amount</code>. More ink → bigger dots.
        The plates are multiplied over the paper (subtractive color) and slightly offset, like a cheap print.</li>
      <li><b>Screentones.</b> For black-and-white manga, pick a pattern per tone range: white, sparse dots, diagonal lines, cross-hatching, black.</li>
      <li><b>Ben-Day dots.</b> Pop art maps every color to a tiny bold palette; light tones become a fixed pattern of colored dots on white.</li>
      <li><b>Speed lines</b> are thin wedges around a focus point (here the hero), only drawn near the frame edge.</li>
    </ol>`,
    uses: [
      { title: 'Cel-shaded games', text: 'Zelda: The Wind Waker, Borderlands (ink outlines), Jet Set Radio, Okami, Hi-Fi Rush, Genshin Impact and most anime-style games.' },
      { title: 'Comic presentation', text: 'Comix Zone, XIII, Sin City-style cutscenes, Spider-Man: Into the Spider-Verse (halftone & misregistration), Marvel’s Midnight Suns panels.' },
      { title: 'Manga & B/W', text: 'Screentone looks in visual novels, Gravity-Rush style panels, MadWorld’s black & white with color accents.' },
      { title: 'Juice', text: 'Speed lines, impact bursts and “POW!” lettering for hits, supers and dramatic moments.' },
    ],
    try: [
      'On <b>Cel shading</b>, move the mouse to swing the light and watch the hard band edges slide across the spheres. Then set <i>Color bands</i> to 2.',
      'On <b>Comic book</b>, increase <i>Dot / screen size</i> to 14 to see the four rotated dot grids and the rosettes they form.',
      'Set <i>Ink lines</i> to 0, then to 1: too low misses outlines, too high inks every texture detail.',
      'On <b>Manga screentone</b>, raise <i>Line thickness</i> for a bolder, brush-pen look.',
      'Switch the cel demo’s <i>Source</i> to the game scene: a few bands and ink lines already read as “cartoon”.',
    ],
    ask: [
      'cel shading with 3 bands and cool-tinted shadows',
      'ink outlines from Sobel edge detection as a post-process',
      'CMYK halftone dots with rotated screens',
      'manga screentone filter (dots, lines, cross-hatching)',
      'speed lines / focus lines around the player on a big hit',
      'Lichtenstein-style Ben-Day dots and bold palette',
    ],
    perf: `<p>Cheap: one full-screen pass with ~9 texture reads (the Sobel taps) plus arithmetic for the dot patterns. Edge detection on the final image is
      simple but can catch texture noise; games often detect edges on <i>depth and normals</i> (or draw back-face “inverted hull” outlines) for cleaner lines.</p>`,
    api: `<p>Pure fragment-shader work: identical in WebGL2 and WebGPU. The halftone uses <code>fwidth()</code> for anti-aliased dot edges, available in both.</p>`,
    code: [
      {
        title: 'Ink lines: Sobel edge detection',
        lang: 'wgsl',
        src: `fn edgeAt(p: vec2f, w: f32) -> f32 {           // w = line thickness in px
  let tl = img(p + vec2f(-w, -w)); let tt = img(p + vec2f(0.0, -w)); let tr = img(p + vec2f(w, -w));
  let ml = img(p + vec2f(-w, 0.0));                                   let mr = img(p + vec2f(w, 0.0));
  let bl = img(p + vec2f(-w, w));  let bb = img(p + vec2f(0.0, w));  let br = img(p + vec2f(w, w));
  let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  let gy = (bl + 2.0 * bb + br) - (tl + 2.0 * tt + tr);
  return sqrt(dot(gx, gx) + dot(gy, gy));
}
col = mix(col, inkColor, smoothstep(thr, thr * 1.35, edgeAt(p, u.inkWidth)));`,
      },
      {
        title: 'Halftone screen + CMYK',
        lang: 'wgsl',
        src: `fn spotFn(p: vec2f, ang: f32, cell: f32) -> f32 {
  let q = rot2(ang) * p / cell;                       // rotated dot grid
  return 0.5 + 0.25 * (cos(TAU * q.x) + cos(TAU * q.y));  // 1 at dot centres
}
fn screen(p: vec2f, ang: f32, cell: f32, amount: f32) -> f32 {
  let s = spotFn(p, ang, cell);
  let aa = max(fwidth(s), 0.002) * 0.8;
  return smoothstep(1.0 - amount - aa, 1.0 - amount + aa, s);   // ink coverage
}
let k = 1.0 - max(max(base.r, base.g), base.b);
let cmy = (vec3f(1.0) - base - vec3f(k)) / (1.0 - k);
col *= mix(vec3f(1.0), cyanInk, screen(p, 0.2618, cell, cmy.x));   // 15°, then M 75°, Y 0°, K 45°`,
      },
      {
        title: 'Cel shading: quantized lighting',
        lang: 'wgsl',
        src: `let lam = max(dot(n, L), 0.0);
let band = floor(lam * u.bands) / (u.bands - 1.0);   // 3 bands: 0, 0.5, 1
let s = mix(coolShadow, baseColor, band);            // shadows shift hue, not just darken
s = mix(s, white, step(0.94, dot(n, H)));             // hard specular blob`,
      },
    ],
  },
});

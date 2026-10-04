import { shaderScene } from '../../core/shaderscene.js';
import { tag } from './_shared.js';

// Day/night cycle: palette keyframes (sky top, horizon, ambient, sun color) interpolated on the CPU from the
// time of day and season, passed as uniforms. The shader draws sky, sun & moon arcs, twinkling stars,
// a town whose windows and street lamps switch on as the ambient light fades, and long shadows at low sun.
// Portable WGSL: runs on WebGPU and WebGL2.

const SEASONS = {
  spring: { sr: 6, ss: 19.5, foliage: [0.42, 0.72, 0.32], ground: [0.38, 0.62, 0.3], snow: 0, blossom: 1 },
  summer: { sr: 5, ss: 21, foliage: [0.22, 0.55, 0.2], ground: [0.33, 0.58, 0.24], snow: 0, blossom: 0 },
  autumn: { sr: 6.5, ss: 18.5, foliage: [0.86, 0.42, 0.13], ground: [0.55, 0.5, 0.25], snow: 0, blossom: 0 },
  winter: { sr: 7.5, ss: 16.5, foliage: [0.3, 0.24, 0.2], ground: [0.86, 0.9, 0.96], snow: 1, blossom: 0 },
};

// colors per phase: sky top, horizon, ambient light, sun color
const PH = {
  night: ['#04060f', '#0e1530', '#232a4a', '#000000'],
  predawn: ['#0b1030', '#3b2c58', '#3a3456', '#402040'],
  dawn: ['#27407a', '#f2875a', '#a87a78', '#ff9a5a'],
  morning: ['#3c78cc', '#f3c48e', '#e0cfbf', '#ffd8a0'],
  noon: ['#2f6fd8', '#a9d6f7', '#ffffff', '#fff4e0'],
  afternoon: ['#3a72c8', '#f0cf98', '#eadcc6', '#ffe0a8'],
  sunset: ['#3a3c80', '#ff7040', '#b87a68', '#ff7a3a'],
  dusk: ['#171c48', '#7a3f70', '#4e3e66', '#301830'],
};
const hex = (h) => {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};
const PHC = Object.fromEntries(Object.entries(PH).map(([k, v]) => [k, v.map(hex)]));

/** Interpolate the palette keyframes for an hour (0..24) given sunrise/sunset. */
function palette(hour, sr, ss, nightLevel) {
  const keys = [
    [ss + 1.8 - 24, 'night'],
    [sr - 1.0, 'predawn'],
    [sr, 'dawn'],
    [sr + 1.3, 'morning'],
    [(sr + ss) / 2, 'noon'],
    [ss - 1.3, 'afternoon'],
    [ss, 'sunset'],
    [ss + 0.8, 'dusk'],
    [ss + 1.8, 'night'],
    [sr - 1.0 + 24, 'predawn'],
  ];
  let h = hour;
  if (h < keys[0][0]) h += 24;
  let i = 0;
  while (i < keys.length - 2 && h > keys[i + 1][0]) i++;
  const [h0, a] = keys[i];
  const [h1, b] = keys[i + 1];
  let f = Math.min(1, Math.max(0, (h - h0) / (h1 - h0)));
  f = f * f * (3 - 2 * f);
  const out = PHC[a].map((c, k) => c.map((v, j) => v + (PHC[b][k][j] - v) * f));
  // night ambient level is adjustable
  const nightness = (n) => (n === 'night' || n === 'predawn' ? 1 : 0);
  const nk = nightness(a) * (1 - f) + nightness(b) * f;
  out[2] = out[2].map((v) => v * (1 - nk + nk * nightLevel * 2));
  return out;
}

let state = null;
let clock = null;

export default shaderScene({
  interaction: 'Drag “Time of day” to scrub the day, or set an auto-cycle speed.',
  examples: [
    {
      id: 'scrub',
      label: 'Scrub the day',
      kind: 'Abstract',
      note: 'Drag <b>Time of day</b>: the sky gradient, sun color and ambient light all come from a handful of <b>palette keyframes</b> (night, dawn, morning, noon, sunset, dusk) blended by the clock. Watch the shadows stretch at sunrise.',
      params: { hour: 7.2, speed: 0, season: 'summer', shadows: 0.7, night: 0.3 },
    },
    {
      id: 'town',
      label: 'Town lights',
      kind: 'In a game',
      note: 'Auto-cycling evening: as the ambient light drops below each window’s own random threshold, it switches on — so the town lights up gradually, not all at once. Street lamps cast pools of light; late at night windows go dark again.',
      params: { hour: 18.2, speed: 0.35, season: 'autumn', shadows: 0.7, night: 0.3 },
    },
    {
      id: 'seasons',
      label: 'Moods & seasons',
      kind: 'Real life',
      note: 'Same scene, different keyframe inputs: a <b>winter</b> afternoon (short day: sunrise 7:30, sunset 16:30, snow, bare trees). Switch the season to see day length, foliage and mood change.',
      params: { hour: 15.6, speed: 0, season: 'winter', shadows: 0.8, night: 0.3 },
    },
  ],
  controls: [
    { type: 'slider', key: 'hour', label: 'Time of day', min: 0, max: 24, step: 0.01, value: 7.2, format: (v) => `${String(Math.floor(v) % 24).padStart(2, '0')}:${String(Math.floor((v % 1) * 60)).padStart(2, '0')}`, help: 'Starting time; auto-cycle continues from here.' },
    { type: 'slider', key: 'speed', label: 'Auto-cycle speed', min: 0, max: 3, step: 0.01, value: 0, format: (v) => `${v.toFixed(2)} h/s`, help: 'Game hours per real second. 0 = frozen at the slider time.' },
    {
      type: 'select',
      key: 'season',
      label: 'Season',
      value: 'summer',
      options: [
        { value: 'spring', label: 'Spring (6:00–19:30)' },
        { value: 'summer', label: 'Summer (5:00–21:00)' },
        { value: 'autumn', label: 'Autumn (6:30–18:30)' },
        { value: 'winter', label: 'Winter (7:30–16:30, snow)' },
      ],
      help: 'Changes sunrise/sunset times, foliage and ground.',
    },
    { type: 'slider', key: 'shadows', label: 'Shadow strength', min: 0, max: 1, step: 0.01, value: 0.7, help: 'Shadow length follows the sun height: long at sunrise/sunset, short at noon.' },
    { type: 'slider', key: 'night', label: 'Night brightness', min: 0, max: 1, step: 0.01, value: 0.3, help: 'Moonlight / how readable the night is — a gameplay choice.' },
    { type: 'color', key: 'lampColor', label: 'Window & lamp color', value: '#ffc46b' },
  ],
  uniforms: {
    lampColor: 'vec3f',
    shadows: 'f32',
    skyTop: 'vec3f',
    skyHor: 'vec3f',
    ambient: 'vec3f',
    sunCol: 'vec3f',
    foliage: 'vec3f',
    groundCol: 'vec3f',
    sunPos: 'vec2f',
    moonPos: 'vec2f',
    sunElev: 'f32',
    darkness: 'f32',
    late: 'f32',
    snow: 'f32',
    blossom: 'f32',
    hourNow: 'f32',
  },
  include: ['noise', 'sdf', 'color'],
  // the software GPU of the test harness is slow: render at half resolution there only
  renderScale: () => (/[?&]test=1/.test(location.href) ? 0.35 : 1),
  bind(params, ctx) {
    const se = SEASONS[params.season] || SEASONS.summer;
    if (!state || state.base !== params.hour) state = { base: params.hour, elapsed: 0 };
    if (!ctx.paused) state.elapsed += ctx.dt * params.speed;
    const hour = (((params.hour + state.elapsed) % 24) + 24) % 24;
    const [top, hor, amb, sun] = palette(hour, se.sr, se.ss, params.night);
    // sun: arc from the left horizon (sunrise) to the right (sunset); moon: the night arc
    const A = ctx.width / ctx.height;
    const dayT = (hour - se.sr) / (se.ss - se.sr);
    const elev = Math.sin(Math.PI * Math.min(1.08, Math.max(-0.08, dayT)));
    const sunPos = [(0.08 + 0.84 * dayT) * A, 0.66 - elev * 0.52];
    const nightLen = 24 - (se.ss - se.sr);
    const nT = ((((hour - se.ss) % 24) + 24) % 24) / nightLen;
    const mElev = Math.sin(Math.PI * Math.min(1, Math.max(0, nT)));
    const moonPos = [(0.08 + 0.84 * nT) * A, 0.66 - mElev * 0.45];
    const lum = 0.2126 * amb[0] + 0.7152 * amb[1] + 0.0722 * amb[2];
    const lateness = hour >= 23 || hour < 4 ? Math.min(1, (hour >= 23 ? hour - 23 : hour + 1) / 3) : hour < 6 ? Math.max(0, 1 - (hour - 4) / 2) : 0;
    if (!clock || !clock.isConnected) clock = tag(ctx, 'right:8px;bottom:8px');
    const hh = Math.floor(hour);
    const mm = Math.floor((hour % 1) * 60);
    const phase = hour < se.sr - 1 || hour > se.ss + 1.8 ? 'night' : hour < se.sr + 0.5 ? 'dawn' : hour > se.ss - 0.5 ? (hour > se.ss + 0.5 ? 'dusk' : 'sunset') : 'day';
    clock.textContent = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')} · ${phase} · sun ${dayT > 0 && dayT < 1 ? `${Math.round(Math.asin(Math.max(0, elev)) * 57.3 * 0.9)}° up` : 'below horizon'}`;
    return {
      skyTop: top,
      skyHor: hor,
      ambient: amb,
      sunCol: sun,
      foliage: se.foliage,
      groundCol: se.ground,
      sunPos,
      moonPos,
      sunElev: dayT > -0.05 && dayT < 1.05 ? elev : -0.2,
      darkness: 1 - Math.min(1, lum * 1.15),
      late: lateness,
      snow: se.snow,
      blossom: se.blossom,
      hourNow: hour,
    };
  },
  code: /* wgsl */ `
fn dn_aa(d: f32, w: f32) -> f32 { return clamp(0.5 - d / w, 0.0, 1.0); }
fn mountainY(x: f32) -> f32 { return 0.6 - 0.12 * ridged(vec2f(x * 1.3, 4.1), 3); }
fn hillY(x: f32) -> f32 { return 0.72 + 0.025 * sin(x * 3.1) + 0.015 * perlin(vec2f(x * 4.0, 7.0)); }

// houses: one per cell along x, standing on the town baseline
const BASE: f32 = 0.8;
fn houseCell(x: f32) -> f32 { return floor(x / 0.17); }
// returns (facade sdf, roof sdf, window id hash or -1, window local coverage)
fn house(q: vec2f, pw: f32) -> vec4f {
  let id: f32 = houseCell(q.x);
  let h1: vec3f = hash13(id + 3.0);
  let cx: f32 = (id + 0.5) * 0.17 + (h1.x - 0.5) * 0.02;
  let hw: f32 = 0.05 + 0.025 * h1.y;
  let hh: f32 = 0.035 + 0.04 * h1.z;
  let p: vec2f = q - vec2f(cx, BASE - hh);
  let facade: f32 = sdBox(p, vec2f(hw, hh));
  let roofH: f32 = 0.03 + 0.025 * h1.x;
  let roof: f32 = sdTriangle(q, vec2f(cx - hw - 0.012, BASE - 2.0 * hh), vec2f(cx + hw + 0.012, BASE - 2.0 * hh), vec2f(cx, BASE - 2.0 * hh - roofH));
  let chim: f32 = sdBox(q - vec2f(cx + hw * 0.55, BASE - 2.0 * hh - roofH * 0.6), vec2f(0.007, roofH * 0.45));
  // windows on a grid inside the facade
  let g: vec2f = vec2f(0.032, 0.038);
  let lp: vec2f = q - vec2f(cx - hw, BASE - 2.0 * hh);
  let cell: vec2f = floor(lp / g);
  let inC: vec2f = lp - (cell + 0.5) * g;
  let win: f32 = sdBox(inC, vec2f(0.009, 0.012));
  var wid: f32 = -1.0;
  let cols: f32 = floor(hw * 2.0 / g.x);
  let rows: f32 = floor(hh * 2.0 / g.y);
  if (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < cols && cell.y < rows - 0.5 && lp.x < cols * g.x) { wid = hash12(id * 17.0 + cell.x * 5.0 + cell.y).x; }
  return vec4f(facade, min(roof, chim), wid, dn_aa(win, pw));
}

fn treeAt(i: f32) -> vec3f {
  // x, base y, height
  let h: vec3f = hash13(i * 7.7 + 1.0);
  return vec3f((i + 0.35 + 0.3 * h.x) * 0.31, 0.915 + 0.02 * h.y, 0.11 + 0.06 * h.z);
}

fn shade(uv: vec2f, tpx: vec2f) -> vec4f {
  let px: vec2f = uv * u.resolution;               // canvas pixels (the render target may be smaller)
  let H: f32 = u.resolution.y;
  let A: f32 = u.resolution.x / H;
  let q: vec2f = px / H;
  let pw: f32 = max(1.5 / H, length(fwidth(q)));
  let t: f32 = u.time;
  let amb: vec3f = u.ambient;
  let sunUp: f32 = smoothstep(-0.03, 0.12, u.sunElev);
  let nightK: f32 = smoothstep(0.55, 0.9, u.darkness);

  // ---- sky
  var col: vec3f = mix(u.skyHor, u.skyTop, smoothstep(0.72, 0.0, q.y));
  // stars, twinkling, only at night
  let sc: f32 = H * 0.011;
  let cellS: vec2f = floor(px / sc);
  let hs: vec3f = hash23(cellS);
  let sp: vec2f = (cellS + 0.2 + 0.6 * hs.xy) * sc;
  let tw: f32 = 0.55 + 0.45 * sin(t * (2.0 + 3.0 * hs.z) + hs.x * 40.0);
  let star: f32 = step(0.86, hs.z) * smoothstep(1.6, 0.0, length(px - sp)) * tw;
  col += vec3f(0.9, 0.92, 1.0) * star * nightK * smoothstep(0.7, 0.2, q.y);
  // sun glow & disc
  let ds: f32 = length(q - u.sunPos);
  col += u.sunCol * exp(-ds * 7.0) * 0.55 * sunUp;
  col = mix(col, mix(u.sunCol, vec3f(1.0, 0.98, 0.92), 0.6), dn_aa(ds - 0.035, pw) * smoothstep(-0.06, 0.02, u.sunElev));
  // moon with a crescent
  let dm: f32 = length(q - u.moonPos);
  let moonVis: f32 = nightK;
  let moon: f32 = sdMoon(q - u.moonPos, 0.018, 0.026, 0.022);
  col += vec3f(0.6, 0.7, 1.0) * exp(-dm * 14.0) * 0.18 * moonVis;
  col = mix(col, vec3f(0.92, 0.93, 0.85), dn_aa(moon, pw) * moonVis);
  // drifting clouds lit by the sun color / ambient
  if (q.y > 0.04 && q.y < 0.52) {
    let cn: f32 = fbm(vec2f(q.x * 2.0 + t * 0.01, q.y * 6.0), 3) * 0.5 + 0.5;
    let cl: f32 = smoothstep(0.55, 0.75, cn) * smoothstep(0.05, 0.2, q.y) * smoothstep(0.5, 0.3, q.y);
    col = mix(col, amb * 0.85 + u.sunCol * 0.35 * sunUp, cl * 0.75);
  }

  // ---- far mountains (hazy, tinted by the horizon)
  if (q.y > 0.45) {
    let my: f32 = mountainY(q.x);
    let mtn: vec3f = mix(u.skyHor * 0.7, vec3f(0.3, 0.33, 0.45) * amb, 0.55);
    col = mix(col, mtn, dn_aa(my - q.y, pw));
    if (u.snow > 0.5) { col = mix(col, amb * 0.95, dn_aa(my - q.y, pw) * smoothstep(my + 0.04, my, q.y) * 0.8); }
  }
  // ---- hills
  if (q.y > 0.68) {
    let hy: f32 = hillY(q.x);
    let hillC: vec3f = mix(u.groundCol * 0.75, vec3f(0.2, 0.25, 0.3), 0.3) * amb;
    col = mix(col, hillC, dn_aa(hy - q.y, pw));
  }

  // ---- town
  let hs4: vec4f = house(q, pw);
  let hcell: f32 = houseCell(q.x);
  let wallC: vec3f = mix(vec3f(0.75, 0.62, 0.5), vec3f(0.6, 0.55, 0.62), hash11(hcell * 3.1));
  let roofC: vec3f = mix(vec3f(0.55, 0.22, 0.18), vec3f(0.3, 0.32, 0.45), hash11(hcell * 5.7));
  // walls catch the sun on the side facing it
  let sunSide: f32 = select(0.0, 1.0, (u.sunPos.x / A - 0.5) * (fract(q.x / 0.17) - 0.5) > 0.0);
  let wallLit: vec3f = wallC * (amb + u.sunCol * 0.25 * sunUp * sunSide);
  col = mix(col, wallLit, dn_aa(hs4.x, pw));
  var roofLit: vec3f = roofC * amb;
  if (u.snow > 0.5) { roofLit = mix(roofLit, vec3f(0.95, 0.97, 1.0) * amb, 0.85); }
  col = mix(col, roofLit, dn_aa(hs4.y, pw));
  // windows: each has its own random "switch-on" darkness, some go dark late at night
  if (hs4.z >= 0.0) {
    let on: f32 = smoothstep(0.32 + 0.4 * hs4.z, 0.36 + 0.4 * hs4.z, u.darkness) * (1.0 - step(1.0 - u.late * 0.75, hash11(hs4.z * 91.0)));
    let flick: f32 = 0.92 + 0.08 * sin(t * 7.0 + hs4.z * 50.0);
    let glass: vec3f = mix(vec3f(0.12, 0.16, 0.24) * amb + u.skyHor * 0.15, u.lampColor * 1.25 * flick, on);
    col = mix(col, glass, hs4.w * dn_aa(hs4.x, pw));
  }

  // ---- ground band: grass/snow + street
  let gy: f32 = 0.86;
  if (q.y > BASE - 0.002) {
    var g: vec3f = u.groundCol * (0.85 + 0.15 * valueNoise(q * vec2f(60.0, 120.0)));
    let street: f32 = smoothstep(gy, gy + 0.004, q.y) * smoothstep(0.985, 0.98, q.y);
    let cob: f32 = valueNoise(q * vec2f(90.0, 160.0));
    var streetC: vec3f = vec3f(0.42, 0.4, 0.4) * (0.8 + 0.25 * cob);
    if (u.snow > 0.5) { streetC = mix(streetC, vec3f(0.85, 0.88, 0.95), 0.5); }
    g = mix(g, streetC, street);
    var lit: vec3f = g * amb;

    // long shadows: every house, lamp and tree throws a shadow away from the sun on the ground,
    // its length ~ height / tan(sun elevation)
    let sdir: f32 = sign(u.sunPos.x / A - 0.5 + 0.0001);
    let len: f32 = min(0.55, 0.05 / max(u.sunElev, 0.09));
    let vsh: vec2f = vec2f(-sdir * len, len * 0.22);
    var shadow: f32 = 0.0;
    // houses: the shadow of the facade's footprint, sheared
    let hc: f32 = houseCell(q.x - vsh.x * 0.5);
    for (var k = -2; k <= 2; k++) {
      let id: f32 = hc + f32(k);
      let h1: vec3f = hash13(id + 3.0);
      let cx: f32 = (id + 0.5) * 0.17 + (h1.x - 0.5) * 0.02;
      let hw: f32 = 0.05 + 0.025 * h1.y;
      let hh: f32 = 0.035 + 0.04 * h1.z;
      let tip: vec2f = vsh * (hh * 2.0 / 0.12);
      let a: vec2f = vec2f(cx - hw, BASE);
      let b: vec2f = vec2f(cx + hw, BASE);
      let s1: f32 = sdTriangle(q, a, b, b + tip);
      let s2: f32 = sdTriangle(q, a, b + tip, a + tip);
      shadow = max(shadow, dn_aa(min(s1, s2), pw * 4.0));
    }
    // lamp posts and trees in the foreground
    for (var i = 0; i < 8; i++) {
      let fi: f32 = f32(i);
      let lx: f32 = (fi + 0.5) * 0.31 + 0.12;
      let ltop: vec2f = vec2f(lx, 0.93) + vsh * (0.13 / 0.12);
      if (q.x > min(lx, ltop.x) - 0.02 && q.x < max(lx, ltop.x) + 0.02) {
        shadow = max(shadow, dn_aa(sdSegment(q, vec2f(lx, 0.93), ltop) - 0.003, pw * 3.0));
      }
      let tr: vec3f = treeAt(fi);
      let tb: vec2f = tr.xy;
      let tt: vec2f = tb + vsh * (tr.z / 0.12);
      if (q.x > min(tb.x, tt.x) - 0.08 && q.x < max(tb.x, tt.x) + 0.08) {
        shadow = max(shadow, dn_aa(sdSegment(q, tb, tt) - 0.004, pw * 3.0));
        shadow = max(shadow, dn_aa(sdEllipseApprox(q - mix(tb, tt, 0.75), vec2f(0.035 + length(vsh) * 0.15, 0.012)), pw * 6.0));
      }
    }
    lit = mix(lit, lit * (1.0 - 0.65 * u.shadows) * vec3f(0.85, 0.9, 1.1), shadow * sunUp * u.shadows);
    // pools of lamp light at night
    if (nightK > 0.0) {
      for (var i = 0; i < 8; i++) {
        let lx: f32 = (f32(i) + 0.5) * 0.31 + 0.12;
        if (abs(q.x - lx) > 0.3) { continue; }
        let d: f32 = length((q - vec2f(lx, 0.93)) * vec2f(1.0, 3.2));
        lit += g * u.lampColor * exp(-d * 18.0) * 1.4 * nightK;
      }
    }
    col = mix(col, lit, dn_aa(BASE - q.y, pw));
  }

  // ---- foreground trees (seasonal) and lamp posts
  for (var i = 0; i < 8; i++) {
    let fi: f32 = f32(i);
    let tr: vec3f = treeAt(fi);
    if (abs(q.x - tr.x) > tr.z * 0.6 || q.y > tr.y + 0.01 || q.y < tr.y - tr.z * 1.3) { continue; }
    let trunk: f32 = sdSegment(q, tr.xy, tr.xy - vec2f(0.0, tr.z * 0.6)) - 0.005;
    col = mix(col, vec3f(0.25, 0.17, 0.12) * amb, dn_aa(trunk, pw));
    let cc: vec2f = tr.xy - vec2f(0.0, tr.z * 0.75);
    var crown: f32 = length((q - cc) * vec2f(1.0, 1.15)) - tr.z * 0.42 + 0.012 * perlin(q * 60.0);
    var fol: vec3f = u.foliage * (0.75 + 0.35 * fbm(q * 40.0, 2));
    if (u.blossom > 0.5) { fol = mix(fol, vec3f(1.0, 0.72, 0.82), step(0.62, valueNoise(q * 140.0))); }
    if (u.snow > 0.5) {
      // winter: bare branches instead of a leafy crown
      var bd: f32 = 1.0;
      for (var k = 0; k < 5; k++) {
        let fk: f32 = f32(k);
        let ang: f32 = -1.1 + fk * 0.55 + 0.15 * sin(fi * 3.0 + fk);
        let st: vec2f = tr.xy - vec2f(0.0, tr.z * (0.45 + 0.05 * fk));
        let en: vec2f = st + vec2f(sin(ang), -cos(ang)) * tr.z * (0.45 - 0.05 * abs(fk - 2.0));
        bd = min(bd, sdSegment(q, st, en) - 0.0025);
      }
      crown = bd;
      fol = vec3f(0.32, 0.26, 0.22);
    }
    let lightDir: f32 = clamp((q.x - cc.x) * sign(u.sunPos.x / A - cc.x) / (tr.z * 0.42) * 0.5 + 0.5, 0.0, 1.0);
    col = mix(col, fol * (amb + u.sunCol * 0.3 * sunUp * lightDir), dn_aa(crown, pw));
  }
  for (var i = 0; i < 8; i++) {
    let lx: f32 = (f32(i) + 0.5) * 0.31 + 0.12;
    if (abs(q.x - lx) > 0.1 || q.y < 0.7 || q.y > 0.95) { continue; }
    let pole: f32 = sdSegment(q, vec2f(lx, 0.93), vec2f(lx, 0.8)) - 0.0025;
    col = mix(col, vec3f(0.12, 0.13, 0.16) * (amb + 0.2), dn_aa(pole, pw));
    let head: f32 = sdBox(q - vec2f(lx, 0.795), vec2f(0.007, 0.01));
    col = mix(col, mix(vec3f(0.2, 0.2, 0.22) * amb, u.lampColor * 1.6, nightK), dn_aa(head, pw));
    col += u.lampColor * exp(-length(q - vec2f(lx, 0.797)) * 45.0) * 0.6 * nightK;
  }
  // a touch of night blue / day warmth grading
  col = col * mix(vec3f(1.0), vec3f(0.85, 0.92, 1.12), nightK * 0.6);
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}`,
  about: {
    summary:
      'A day/night cycle is mostly a well-chosen set of colors over time. Interpolate a few palette keyframes from the clock, apply them as sky and ambient light, and let lights react to the darkness.',
    what: `<p>A little town through a whole day: the sky gradient, sun and moon arcs, stars, an ambient light that tints everything,
      windows and street lamps that switch on at dusk, and shadows that stretch long when the sun is low. Seasons change the day length,
      foliage and snow.</p>`,
    how: `<ol>
      <li><b>Keyframes</b> (CPU): night, pre-dawn, dawn, morning, noon, afternoon, sunset, dusk — each with a sky-top color, a
        horizon color, an ambient light color and a sun color. Their times are placed relative to the season’s sunrise and sunset.</li>
      <li>Every frame JavaScript finds the two keyframes around the current hour and blends them (smoothstep), then sends the results as uniforms.</li>
      <li><b>Sun & moon</b>: positions on half-circle arcs from sunrise to sunset (and sunset to sunrise for the moon). The sun’s
        height (elevation) drives shadow length (<code>≈ height / tan(elevation)</code>) and how much direct light walls get.</li>
      <li><b>Ambient tint</b>: every surface color is multiplied by the ambient color — the single most important part of the look.</li>
      <li><b>Lights at dusk</b>: darkness = 1 − brightness(ambient). Each window has a random threshold, so they switch on one by one; after 23:00 a random subset turns off.</li>
      <li><b>Stars</b> fade in with darkness and twinkle via <code>sin(time × random speed)</code>.</li>
    </ol>`,
    uses: [
      { title: 'Life sims & farming', text: 'Stardew Valley, Animal Crossing: time drives shops, NPC schedules, crops — and the palette sets the mood.' },
      { title: 'Survival', text: 'Don’t Starve, Terraria, Minecraft: night is dangerous; light sources become gameplay.' },
      { title: 'Atmosphere', text: 'Golden-hour menus, dawn after a boss fight, a snowy evening level.' },
      { title: 'Readability', text: '“Night brightness” is a design knob: realistic nights are unplayably dark, so games cheat with blue moonlight.' },
    ],
    try: [
      'On <b>Scrub the day</b>, drag the time from 4:00 to 9:00 slowly: pre-dawn purple, orange sunrise, long shadows shrinking.',
      'Set <i>Auto-cycle speed</i> to 2 and watch a full day in 12 seconds.',
      'On <b>Town lights</b>, look at individual windows around 18:00–19:30: they switch on at different times.',
      'Change the season to Winter at 16:00 — it is already sunset. In Summer, still bright day.',
      'Raise <i>Night brightness</i> to 1: “movie night” — readable, bluish moonlight.',
    ],
    ask: [
      'a day/night cycle with palette keyframes for sky and ambient light',
      'windows and street lamps that turn on at dusk one by one',
      'sun and moon moving along arcs with long shadows at sunrise',
      'twinkling stars that fade in at night',
      'seasons that change day length, foliage and snow',
    ],
    perf: `<p>Practically free: a few uniform values per frame from JavaScript, and the shader already runs per pixel. In a real game the
      ambient tint is one multiply in your sprite shader (or a full-screen color grade), plus switching light sources on and off.</p>`,
    api: `<p>Pure fragment shader + CPU-side keyframes: identical on WebGL2 and WebGPU (this scene is auto-translated from WGSL to GLSL).</p>`,
    code: [
      {
        title: 'Keyframe interpolation (JavaScript)',
        lang: 'js',
        src: `const keys = [[sr - 1, 'predawn'], [sr, 'dawn'], [sr + 1.3, 'morning'], [(sr + ss) / 2, 'noon'],
              [ss - 1.3, 'afternoon'], [ss, 'sunset'], [ss + 0.8, 'dusk'], [ss + 1.8, 'night'] /* … */];
let i = 0;
while (i < keys.length - 2 && hour > keys[i + 1][0]) i++;
let f = (hour - keys[i][0]) / (keys[i + 1][0] - keys[i][0]);
f = f * f * (3 - 2 * f);                                   // smoothstep easing
const [skyTop, skyHorizon, ambient, sunColor] = PALETTE[keys[i][1]].map((c, k) => mixRGB(c, PALETTE[keys[i + 1][1]][k], f));`,
      },
      {
        title: 'Windows switching on (WGSL)',
        lang: 'wgsl',
        src: `// each window has a random id in [0,1): its own darkness threshold
let on = smoothstep(0.32 + 0.4 * wid, 0.36 + 0.4 * wid, u.darkness)
       * (1.0 - step(1.0 - u.late * 0.75, hash11(wid * 91.0)));   // some go dark late at night
let glass = mix(vec3f(0.12, 0.16, 0.24) * amb, u.lampColor * 1.25, on);`,
      },
      {
        title: 'Shadow length from sun elevation',
        lang: 'wgsl',
        src: `let len = min(0.55, 0.05 / max(u.sunElev, 0.09));       // ~ 1 / tan(elevation)
let vsh = vec2f(-sunSide * len, len * 0.22);               // away from the sun, onto the ground
let tip = base + vsh * (objectHeight / 0.12);
shadow = max(shadow, coverage(sdSegment(q, base, tip) - width));`,
      },
    ],
  },
});

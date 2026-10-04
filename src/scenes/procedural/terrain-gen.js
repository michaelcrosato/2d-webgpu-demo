import { shaderScene } from '../../core/shaderscene.js';
import { setLabels, PanZoom } from './_shared.js';

// World & terrain generation from noise: an island with biomes (height × moisture), a
// Terraria-style side-view cave world, and a shaded-relief topographic map. Pan by dragging,
// zoom with the wheel. The camera lives in JavaScript and reaches the shader as one vec4 uniform.

const views = {
  island: new PanZoom({ x: 0, y: 0, zoom: 0.42, minZoom: 0.15, maxZoom: 400 }),
  caves: new PanZoom({ x: 0, y: 0.32, zoom: 1.05, minZoom: 0.25, maxZoom: 30 }),
  relief: new PanZoom({ x: 0.3, y: 0.1, zoom: 0.55, minZoom: 0.1, maxZoom: 200 }),
};

const CODE = /* wgsl */ `
// world position of a pixel: u.view = (center.x, center.y, world units per pixel, zoom)
fn toWorld(px: vec2f) -> vec2f { return u.view.xy + (px - 0.5 * u.resolution) * u.view.z; }

// ================================================================== island
fn islandOct() -> i32 { return clamp(i32(u.detail + log2(max(u.view.w / 0.42, 1.0))), 2, 11); }

fn islandHeight(w: vec2f, oc: i32) -> f32 {
  let q = w * 1.5 + vec2f(u.seed * 13.7, u.seed * 7.3);
  let n = fbm(q, oc);
  let d = length(w * vec2f(0.72, 1.0));
  // fBm + a radial falloff: high in the middle, below sea level at the edges
  return n * 0.95 + 0.4 - 0.72 * d * d - u.sea;
}

fn islandMoist(w: vec2f) -> f32 {
  return clamp(0.5 + fbm(w * 1.1 + vec2f(50.0 + u.seed, 9.0), 4) * 1.15, 0.0, 1.0);
}

// Whittaker-style lookup: elevation × moisture -> biome color
fn biome(h: f32, m: f32) -> vec3f {
  if (h < 0.0) {
    return mix(vec3f(0.24, 0.6, 0.68), vec3f(0.03, 0.12, 0.3), smoothstep(0.0, 0.3, -h));
  }
  let desert = vec3f(0.84, 0.74, 0.5);
  let grass = vec3f(0.5, 0.7, 0.32);
  let forest = vec3f(0.24, 0.5, 0.22);
  let rain = vec3f(0.12, 0.38, 0.24);
  let shrub = vec3f(0.62, 0.62, 0.44);
  let taiga = vec3f(0.3, 0.47, 0.38);
  let rock = vec3f(0.52, 0.48, 0.44);
  let snow = vec3f(0.96, 0.97, 0.99);
  var low = mix(desert, grass, smoothstep(0.22, 0.34, m));
  low = mix(low, forest, smoothstep(0.52, 0.62, m));
  low = mix(low, rain, smoothstep(0.78, 0.88, m));
  let mid = mix(shrub, taiga, smoothstep(0.38, 0.52, m));
  var c = mix(low, mid, smoothstep(0.26, 0.32, h));
  c = mix(c, rock, smoothstep(0.42, 0.47, h));
  c = mix(c, snow, smoothstep(0.55, 0.59, h));
  c = mix(vec3f(0.9, 0.83, 0.6), c, smoothstep(0.012, 0.03, h));
  return c;
}

fn forestAmount(h: f32, m: f32) -> f32 {
  return smoothstep(0.55, 0.65, m) * smoothstep(0.03, 0.06, h) * (1.0 - smoothstep(0.4, 0.45, h));
}

fn exIsland(px: vec2f) -> vec3f {
  let w = toWorld(px);
  let oc = islandOct();
  let h = islandHeight(w, oc);
  let m = islandMoist(w);
  let t = u.time;
  var col = biome(h, m);
  if (h >= 0.0) {
    // hillshading from the height gradient (finite differences, 2 extra samples)
    let e = u.view.z * 1.5;
    let hx = islandHeight(w + vec2f(e, 0.0), oc);
    let hy = islandHeight(w + vec2f(0.0, e), oc);
    let n = normalize(vec3f((h - hx) / e * 0.35, (h - hy) / e * 0.35, 1.0));
    let az = radians(u.azimuth);
    let L = normalize(vec3f(cos(az), sin(az), 1.1));
    col *= clamp(0.45 + 0.75 * dot(n, L), 0.3, 1.35);
    // rivers: zero-crossings of another noise, narrowing toward the mountains
    if (u.rivers > 0.5) {
      let rn = abs(fbm(w * 1.3 + vec2f(40.0 + u.seed, 3.0), 5));
      let rw = 0.016 * (1.0 - smoothstep(0.0, 0.42, h)) + 0.002;
      let keep = smoothstep(-0.05, 0.1, fbm(w * 0.9 + vec2f(7.0, 70.0 + u.seed), 3));
      let riv = (1.0 - smoothstep(rw * 0.55, rw, rn)) * (1.0 - smoothstep(0.36, 0.42, h)) * keep;
      col = mix(col, vec3f(0.2, 0.48, 0.66), riv);
    }
    // trees: a jittered grid of round canopies in forest biomes (only when big enough to see)
    let ts = 0.02;
    let tpx = ts / u.view.z;
    if (tpx > 3.0) {
      let cell = floor(w / ts);
      let hsh = hash23(cell + vec2f(u.seed, 0.0));
      let tc = (cell + 0.25 + 0.5 * hsh.xy) * ts;
      let rr = ts * (0.3 + 0.15 * hsh.z);
      let fa = forestAmount(h, m);
      if (hsh.z < fa * 1.1) {
        let dsh = length(w - tc - vec2f(rr * 0.35, rr * 0.35)) / rr;
        col *= 1.0 - 0.35 * (1.0 - smoothstep(0.8, 1.05, dsh));
        let dt = length(w - tc) / rr;
        let can = mix(vec3f(0.12, 0.32, 0.14), vec3f(0.3, 0.55, 0.26), clamp(1.0 - length(w - tc + vec2f(rr * 0.35, rr * 0.35)) / rr, 0.0, 1.0));
        col = mix(col, can, (1.0 - smoothstep(0.85, 1.0, dt)) * smoothstep(3.0, 6.0, tpx));
      }
    }
  } else {
    // coast foam: animated bands parallel to the shoreline + a white line at the shore
    let dpt = -h;
    let waves = sin(dpt * 140.0 - t * 2.2 + fbm(w * 8.0, 2) * 2.0);
    let foam = smoothstep(0.75, 1.0, waves) * (1.0 - smoothstep(0.0, 0.05, dpt)) * 0.55;
    col = mix(col, vec3f(0.92, 0.97, 1.0), foam + (1.0 - smoothstep(0.0, 0.006, dpt)) * 0.8);
    col *= 0.94 + 0.06 * perlin(w * 30.0 + vec2f(t * 0.3, 0.0));
  }
  // biome legend inset (bottom-right): x = moisture, y = elevation; marker = point under the mouse
  let iw = floor(u.resolution.y * 0.3);
  let ih = floor(u.resolution.y * 0.22);
  let io = u.resolution - vec2f(iw + 12.0, ih + 12.0);
  let lq = (px - io) / vec2f(iw, ih);
  if (lq.x > -0.02 && lq.x < 1.02 && lq.y > -0.03 && lq.y < 1.03) {
    let inside = step(0.0, lq.x) * step(lq.x, 1.0) * step(0.0, lq.y) * step(lq.y, 1.0);
    let lh = mix(0.68, -0.08, lq.y);
    var lc = biome(lh, lq.x);
    lc = mix(lc, lc * 0.6, (1.0 - smoothstep(0.0, 1.5, abs(lh) / fwidth(lh))));
    col = mix(vec3f(0.02), lc, inside);
    if (u.mouse.w > 0.5) {
      let mw = toWorld(u.mouse.xy);
      let mh = islandHeight(mw, oc);
      let mm = islandMoist(mw);
      let mp = io + vec2f(mm, (0.68 - clamp(mh, -0.08, 0.68)) / 0.76) * vec2f(iw, ih);
      let dm = length(px - mp);
      col = mix(col, vec3f(0.0), 1.0 - smoothstep(5.0, 6.5, dm));
      col = mix(col, vec3f(1.0), 1.0 - smoothstep(3.0, 4.2, dm));
    }
  }
  return col;
}

// ================================================================== side-view caves
fn surfaceY(x: f32) -> f32 {
  return fbm(vec2f(x * 1.2 + u.seed * 9.1, 0.5), 5) * 0.24 + 0.05 * sin(x * 0.9 + u.seed);
}

// material of the block at world position c:
// 0 sky, 1 cave air, 2 grass, 3 dirt, 4 stone, 5 deep stone, 6 copper, 7 iron, 8 gold, 9 crystal, 10 lava, 11 sand
fn caveMat(c: vec2f, T: f32) -> f32 {
  let sy = surfaceY(c.x);
  let depth = c.y - sy;
  if (depth < 0.0) { return 0.0; }
  let k = vec2f(u.seed * 5.3, u.seed * 2.1);
  // caves: "spaghetti" tunnels where |noise| is small, plus big caverns where another noise is high
  let n1 = fbm(c * vec2f(2.6, 3.8) + k, 4);
  let n2 = fbm(c * 1.5 + k + vec2f(7.0, 3.0), 4);
  let deep = smoothstep(0.05, 0.6, depth);
  let tunnel = abs(n1) < u.caveAmt * 0.075 * (0.4 + 0.6 * deep);
  let cavern = n2 > 0.34 - u.caveAmt * 0.24 * deep;
  if (depth > T * 2.5 && (tunnel || cavern)) {
    if (depth > 1.35 && n2 > 0.42 - u.caveAmt * 0.22 && fract(c.y * 2.0) > 0.0 && n1 > 0.0) { return 10.0; }
    return 1.0;
  }
  let dirtDepth = 0.11 + 0.05 * fbm(vec2f(c.x * 5.0, 3.0), 3);
  var m = 4.0;
  if (depth < dirtDepth) {
    m = 3.0;
    if (depth < T && sy > 0.13) { m = 11.0; }
    else if (depth < T) { m = 2.0; }
  } else if (depth > 0.95 + 0.1 * n1) { m = 5.0; }
  // ores: small clumps from high-frequency value noise, depth-dependent
  let oa = u.ore * 0.16;
  let o1 = valueNoise(c * 22.0 + k + vec2f(13.0, 0.0));
  let o2 = valueNoise(c * 20.0 + k + vec2f(91.0, 7.0));
  let o3 = valueNoise(c * 26.0 + k + vec2f(3.0, 55.0));
  if (m >= 3.0 && m <= 5.0) {
    if (depth > 1.05 && o3 > 1.0 - oa * 0.8) { return 9.0; }
    if (depth > 0.55 && o2 > 1.0 - oa * 0.9) { return 8.0; }
    if (depth > 0.2 && depth < 1.3 && o1 > 1.0 - oa) { return 7.0; }
    if (depth < 0.7 && o3 > 1.0 - oa * 1.1) { return 6.0; }
  }
  return m;
}

fn matColor(m: f32, hsh: f32, sub: f32) -> vec3f {
  var c = vec3f(0.0);
  if (m < 2.5) { c = vec3f(0.36, 0.72, 0.25); }
  else if (m < 3.5) { c = vec3f(0.47, 0.32, 0.2); }
  else if (m < 4.5) { c = vec3f(0.45, 0.45, 0.48); }
  else if (m < 5.5) { c = vec3f(0.3, 0.28, 0.36); }
  else if (m < 6.5) { c = mix(vec3f(0.45, 0.45, 0.48), vec3f(0.85, 0.48, 0.22), step(0.45, sub)); }
  else if (m < 7.5) { c = mix(vec3f(0.45, 0.45, 0.48), vec3f(0.78, 0.72, 0.68), step(0.45, sub)); }
  else if (m < 8.5) { c = mix(vec3f(0.3, 0.28, 0.36), vec3f(1.0, 0.82, 0.25), step(0.5, sub)); }
  else if (m < 9.5) { c = mix(vec3f(0.3, 0.28, 0.36), vec3f(0.55, 0.9, 1.0), step(0.55, sub)); }
  else if (m < 10.5) { c = vec3f(1.0, 0.45, 0.1); }
  else { c = vec3f(0.86, 0.78, 0.52); }
  return c * (0.88 + 0.16 * hsh) * (0.92 + 0.12 * sub);
}

fn isSolid(m: f32) -> f32 { return step(1.5, m) * (1.0 - step(9.5, m) * step(m, 10.5)); }

fn exCaves(px: vec2f) -> vec3f {
  let w = toWorld(px);
  let T = 1.0 / u.blocks;
  let ti = floor(w / T);
  let tc = (ti + 0.5) * T;
  let f = fract(w / T);
  let m = caveMat(tc, T);
  let hsh = hash21(ti + vec2f(u.seed, 0.0));
  // 4x4 "pixels" inside every block for a pixel-art texture
  let sub = hash21(floor(f * 4.0) + ti * 4.0);
  let sy = surfaceY(tc.x);
  let depth = tc.y - sy;
  let mw = toWorld(u.mouse.xy);
  var col = vec3f(0.0);
  if (m < 0.5) {
    // sky with distant hills
    let sky = mix(vec3f(0.42, 0.68, 0.95), vec3f(0.78, 0.9, 1.0), smoothstep(-0.6, 0.1, w.y));
    let hill = fbm(vec2f(w.x * 0.8 + 30.0, 1.0), 3) * 0.2 - 0.05;
    col = mix(sky, vec3f(0.55, 0.72, 0.85), step(hill, w.y + 0.25) * 0.6);
    let cl = smoothstep(0.1, 0.4, fbm(w * vec2f(1.5, 4.0) + vec2f(u.time * 0.02, 0.0), 4)) * smoothstep(0.0, -0.3, w.y + 0.25);
    col = mix(col, vec3f(1.0), cl * 0.8);
  } else if (m < 1.5 || (m > 9.5 && m < 10.5)) {
    // cave air: darkened background wall of whatever is around
    var bgm = 4.0;
    if (depth < 0.13) { bgm = 3.0; }
    if (depth > 0.95) { bgm = 5.0; }
    col = matColor(bgm, hsh, sub) * 0.42;
    if (m > 9.5) {
      let flow = 0.5 + 0.5 * sin(w.x * 60.0 + u.time * 2.0 + sub * 3.0);
      col = mix(vec3f(0.95, 0.3, 0.05), vec3f(1.0, 0.75, 0.2), flow * sub);
    }
  } else {
    col = matColor(m, hsh, sub);
    if (m > 8.5 && m < 9.5) { col += vec3f(0.3, 0.6, 0.7) * step(0.9, sub) * (0.5 + 0.5 * sin(u.time * 4.0 + hsh * 20.0)); }
    // bevel: light top/left edges, shade bottom/right ones where the neighbour is open
    let ex = select(1.0, -1.0, f.x < 0.5);
    let ey = select(1.0, -1.0, f.y < 0.5);
    let edgeD = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    if (edgeD < 0.2) {
      let nx = caveMat(tc + vec2f(ex * T, 0.0), T);
      let ny = caveMat(tc + vec2f(0.0, ey * T), T);
      let ox = (1.0 - isSolid(nx)) * (1.0 - smoothstep(0.0, 0.18, min(f.x, 1.0 - f.x)));
      let oy = (1.0 - isSolid(ny)) * (1.0 - smoothstep(0.0, 0.18, min(f.y, 1.0 - f.y)));
      col *= 1.0 - 0.45 * max(ox * step(0.0, ex), oy * step(0.0, ey));
      col *= 1.0 + 0.35 * max(ox * step(ex, 0.0), oy * step(ey, 0.0));
    }
  }
  // lighting: daylight fades with depth; the mouse is your torch; lava glows
  if (m > 0.5) {
    let day = mix(1.0, 0.38, smoothstep(0.03, 0.45, depth));
    let dm = length(w - mw) / (0.16 / sqrt(u.view.w));
    let torch = 1.4 / (1.0 + dm * dm * 3.0) * u.mouse.w;
    let deepGlow = vec3f(1.0, 0.4, 0.1) * smoothstep(1.2, 1.6, depth) * 0.25;
    var light = vec3f(day) + vec3f(1.0, 0.78, 0.5) * torch + deepGlow;
    if (m > 9.5 && m < 10.5) { light = vec3f(1.0); }
    col *= light;
  }
  // block grid (subtle) when zoomed in
  let gpx = T / u.view.z;
  let gl = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * gpx;
  col *= 1.0 - 0.15 * (1.0 - smoothstep(0.0, 1.0, gl)) * smoothstep(8.0, 16.0, gpx) * step(0.5, m);
  return col;
}

// ================================================================== shaded relief
fn reliefHeight(w: vec2f) -> f32 {
  let oc = clamp(i32(u.detail + log2(max(u.view.w / 0.55, 1.0))), 2, 11);
  let q = w * 0.75 + vec2f(u.seed * 13.7, u.seed * 7.3);
  let r = ridged(q * 0.7, oc);
  let f = fbmEx(q * 1.1 + 4.0, oc, 2.0, 0.42);
  return r * 0.95 + f * 0.45 - 0.08;
}

fn hypso(h: f32) -> vec3f {
  var c = vec3f(0.42, 0.62, 0.38);
  c = mix(c, vec3f(0.66, 0.74, 0.46), smoothstep(0.15, 0.3, h));
  c = mix(c, vec3f(0.86, 0.79, 0.54), smoothstep(0.3, 0.45, h));
  c = mix(c, vec3f(0.72, 0.56, 0.4), smoothstep(0.45, 0.6, h));
  c = mix(c, vec3f(0.62, 0.58, 0.58), smoothstep(0.6, 0.72, h));
  c = mix(c, vec3f(0.98, 0.98, 1.0), smoothstep(0.74, 0.82, h));
  return c;
}

fn exRelief(px: vec2f) -> vec3f {
  let w = toWorld(px);
  let h = reliefHeight(w);
  let e = u.view.z * 1.5;
  let hx = reliefHeight(w + vec2f(e, 0.0));
  let hy = reliefHeight(w + vec2f(0.0, e));
  let lake = 0.1;
  // hillshade: Lambert lighting of the surface normal; azimuth measured clockwise from north
  let sc = u.exag * 0.12;
  let n = normalize(vec3f((h - hx) / e * sc, (h - hy) / e * sc, 1.0));
  let az = radians(u.azimuth);
  let alt = radians(45.0);
  let L = vec3f(sin(az) * cos(alt), -cos(az) * cos(alt), sin(alt));
  let hs = clamp(dot(n, L), 0.0, 1.0);
  var col = hypso(h);
  if (h < lake) {
    col = mix(vec3f(0.55, 0.75, 0.88), vec3f(0.35, 0.58, 0.78), smoothstep(lake, lake - 0.12, h));
  } else {
    col *= 0.3 + 0.95 * hs / max(L.z, 0.3) * 0.75;
  }
  // contours every interval, every 5th one thicker (an "index contour")
  let v = h / u.interval;
  let fw = max(fwidth(v), 0.0001);
  let line = 1.0 - smoothstep(0.5, 1.4, abs(fract(v + 0.5) - 0.5) / fw);
  let v5 = v / 5.0;
  let fw5 = max(fwidth(v5), 0.0001);
  let line5 = 1.0 - smoothstep(0.9, 1.9, abs(fract(v5 + 0.5) - 0.5) / fw5);
  let fade = 1.0 - smoothstep(0.25, 0.5, fw);
  var cc = vec3f(0.5, 0.32, 0.16);
  if (h < lake) { cc = vec3f(0.2, 0.42, 0.65); }
  col = mix(col, cc, max(line * 0.55 * fade, line5 * 0.85));
  // shoreline
  let sl = abs(h - lake) / max(fwidth(h), 0.00001);
  col = mix(col, vec3f(0.15, 0.35, 0.55), 1.0 - smoothstep(0.6, 1.6, sl));
  // map grid every 0.25 world units + paper vignette
  let g = abs(fract(w / 0.25 + 0.5) - 0.5) * 0.25 / u.view.z;
  col = mix(col, vec3f(0.25, 0.3, 0.4), (1.0 - smoothstep(0.3, 1.0, min(g.x, g.y))) * 0.25);
  let vq = px / u.resolution - 0.5;
  col *= 1.0 - 0.35 * dot(vq, vq);
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = exIsland(px); }
  else if (ex == 1) { col = exCaves(px); }
  else { col = exRelief(px); }
  col += (ign(px) - 0.5) / 255.0;
  return vec4f(col, 1.0);
}
`;

export default shaderScene({
  interaction: 'Drag to pan · mouse wheel to zoom.',
  wheel: true,
  examples: [
    {
      id: 'island',
      label: 'Island & biomes',
      kind: 'In a game',
      note: 'Height = fBm minus a radial falloff (so the edges sink into the sea). A second noise gives <b>moisture</b>. The biome is looked up from height × moisture (inset, bottom-right — the dot is the spot under your mouse). Add hillshading, coast foam, fake rivers and trees. Zoom in: more octaves are added as you zoom, so detail never runs out.',
    },
    {
      id: 'caves',
      label: 'Side-view caves',
      kind: 'In a game',
      note: 'Terraria-style: the surface is 1D noise along x; below it, blocks are dirt then stone. <b>Caves</b> are wherever 2D noise crosses a threshold (thin “spaghetti” tunnels where |noise| is small, caverns where another noise is high). Ores are small high-frequency noise blobs that depend on depth. Every block is evaluated at its center, so the world snaps to a grid. Your mouse is a torch.',
      params: { seed: 3 },
    },
    {
      id: 'relief',
      label: 'Shaded relief map',
      kind: 'Real life',
      note: 'How real topographic maps are drawn: <b>hypsometric tint</b> (color by elevation), <b>hillshading</b> (Lambert lighting of the slope, light from the north-west by convention) and <b>contour lines</b> (every interval, every 5th one bold), all from one height function — here ridged noise + fBm.',
      params: { detail: 5, interval: 0.035, exag: 4 },
    },
  ],
  controls: [
    { type: 'slider', key: 'seed', label: 'Seed', min: 0, max: 99, step: 1, value: 0, help: 'Same seed = same world. Games share seeds so players can visit the same map.' },
    { type: 'slider', key: 'sea', label: 'Sea level', min: -0.3, max: 0.3, step: 0.005, value: 0, showFor: ['island'], help: 'Raise to flood the lowlands into an archipelago.' },
    { type: 'slider', key: 'detail', label: 'Base octaves', min: 2, max: 9, step: 1, value: 6, showFor: ['island', 'relief'], help: 'Octaves at the default zoom; one more is added each time you zoom 2×.' },
    { type: 'toggle', key: 'rivers', label: 'Rivers', value: true, showFor: ['island'], help: 'Cheap fake: the zero-line of a separate noise, narrowing uphill. Real rivers need a flow simulation.' },
    { type: 'slider', key: 'caveAmt', label: 'Cave amount', min: 0, max: 1, step: 0.01, value: 0.55, showFor: ['caves'], help: 'Lowers the thresholds: wider tunnels, bigger caverns.' },
    { type: 'slider', key: 'ore', label: 'Ore abundance', min: 0, max: 1, step: 0.01, value: 0.5, showFor: ['caves'], help: 'Copper & iron near the top, gold deeper, crystals at the bottom.' },
    { type: 'slider', key: 'blocks', label: 'Blocks per world unit', min: 16, max: 160, step: 1, value: 64, showFor: ['caves'], help: 'Block resolution: low = chunky retro, high = detailed.' },
    { type: 'slider', key: 'azimuth', label: 'Light direction (°)', min: 0, max: 360, step: 1, value: 315, showFor: ['island', 'relief'], help: 'Where the sun comes from (315° = north-west, the map-making convention).' },
    { type: 'slider', key: 'exag', label: 'Vertical exaggeration', min: 0.5, max: 8, step: 0.1, value: 3, showFor: ['relief'], help: 'Multiply slopes before lighting so gentle hills read clearly.' },
    { type: 'slider', key: 'interval', label: 'Contour interval', min: 0.01, max: 0.1, step: 0.001, value: 0.025, showFor: ['relief'], format: (v) => `${Math.round(v * 4000)} m`, help: 'Height step between contour lines (pretend the peak is ~4 km).' },
    { type: 'button', key: 'resetView', label: 'Reset view' },
  ],
  uniforms: {
    seed: 'f32', sea: 'f32', detail: 'f32', rivers: 'f32', caveAmt: 'f32', ore: 'f32', blocks: 'f32', azimuth: 'f32', exag: 'f32', interval: 'f32',
    view: 'vec4f',
  },
  include: ['noise'],
  onAction(key, ctx) {
    if (key === 'resetView' || key === 'reset') views[ctx.example]?.reset();
  },
  bind(params, ctx) {
    const pz = views[ctx.example] || views.island;
    pz.update(ctx);
    if (ctx.example === 'island') {
      setLabels(ctx, [{ text: 'biome = f(moisture →, height ↑)', style: 'right:8px;bottom:calc(22% + 18px);font-size:11px;opacity:.85' }]);
    } else if (ctx.example === 'caves') {
      setLabels(ctx, [{ text: `zoom ${pz.zoom.toFixed(2)}×`, style: 'right:8px;bottom:8px;opacity:.8' }]);
    } else {
      setLabels(ctx, [{ text: `contours every ${Math.round(params.interval * 4000)} m · bold every ${Math.round(params.interval * 20000)} m`, style: 'right:8px;bottom:8px;opacity:.85' }]);
    }
    return { view: [pz.x, pz.y, 1 / (pz.zoom * ctx.height), pz.zoom] };
  },
  code: CODE,
  about: {
    summary: 'Procedural worlds are noise plus rules: a height function, a few thresholds and lookup tables turn random numbers into islands, biomes, caves and maps — infinite, seeded and generated on the fly.',
    what: `<p><b>Island & biomes</b>: an overhead map like a strategy game or roguelike overworld. <b>Side-view caves</b>: a Terraria/Starbound-style block world.
      <b>Shaded relief map</b>: the cartographic techniques (tint, hillshade, contours) that make terrain readable. All three are computed per pixel every frame,
      so you can pan and zoom forever.</p>`,
    how: `<ol>
      <li><b>Height</b>: fBm noise. For an island subtract a <i>falloff</i> that grows with distance from the center (<code>h = fbm(p) + 0.4 − 0.72·d²</code>), so the coast closes.</li>
      <li><b>Moisture</b>: a second, independent noise. The <b>biome</b> is a 2D lookup: (moisture, height) → desert, grassland, forest, rainforest, shrubland, taiga, rock, snow
        (a simplified Whittaker diagram — the inset in the corner).</li>
      <li><b>Hillshading</b>: estimate the slope by sampling the height one step right and one step down; the surface normal is
        <code>normalize(h − hx, h − hy, 1)</code>; brightness = <code>dot(normal, lightDir)</code>.</li>
      <li><b>Contours</b>: <code>fract(height / interval)</code> wraps every interval; drawing where it wraps (with <code>fwidth</code> for anti-aliasing) gives evenly thick iso-lines.</li>
      <li><b>Caves</b>: decide the material of each block from its center position: above the 1D surface curve = sky; below = dirt, stone, deep stone by depth;
        holes where 2D noise passes a threshold; ores where small, high-frequency noise is extreme. Quantising the position to a block grid gives the blocky look.</li>
      <li><b>Infinite detail</b>: the camera (center + zoom) is a uniform; the octave count grows with the zoom level so zoomed-in views get new detail instead of blur.</li>
    </ol>`,
    uses: [
      { title: 'Open worlds & survival', text: 'Minecraft, Terraria, Valheim, No Man’s Sky: seeded noise worlds generated in chunks as you explore.' },
      { title: 'Roguelike & strategy maps', text: 'Overworlds, islands and continents (Civilization-like map scripts, Dwarf Fortress), biome placement, resource distribution.' },
      { title: 'Readable terrain', text: 'Hillshading and contours make height maps legible in map screens, minimaps and editors.' },
      { title: 'Tooling', text: 'Generate a candidate world on the GPU in milliseconds, then let designers pick seeds and hand-edit.' },
    ],
    try: [
      'On <b>Island & biomes</b>, raise <i>Sea level</i> to 0.15: the island breaks into an archipelago. Hover the map and watch the dot move in the biome table.',
      'Zoom far into a coastline: new octaves keep appearing, so the coast never stops being detailed (a fractal!).',
      'On <b>Side-view caves</b>, set <i>Cave amount</i> to 1 and pan down with the mouse: caverns, gold, crystals and lava near the bottom.',
      'Lower <i>Blocks per world unit</i> to 24 for a chunky retro look.',
      'On <b>Shaded relief map</b>, sweep <i>Light direction</i>: with light from the south-east, valleys appear to pop out like ridges (the “relief inversion” illusion).',
    ],
    ask: [
      'procedural island map with biomes from height and moisture',
      'Terraria-style side-view terrain with caves and ores',
      'seeded world generation that is the same every time',
      'hillshaded height map with contour lines',
      'infinite pan & zoom map with level-of-detail octaves',
      'chunk-based world generation on the GPU',
    ],
    perf: `<p>The island evaluates ~3 fBm heights (for the slope) + moisture + rivers per pixel: about 35 noise lookups. Fine every frame, but a real game would
      <b>generate once into textures or chunk data</b> (on the GPU this takes milliseconds) and only redo it when the seed changes or a new chunk comes into view.
      The cave view samples up to 3 blocks per pixel (itself + 2 neighbours for edge shading) only near block borders.</p>`,
    api: `<p>Identical in WebGL2 and WebGPU for this per-pixel version. In WebGPU, the natural next step is a <b>compute shader</b> that writes the generated
      world into a storage buffer (block ids for a tilemap, or a height/biome texture) — which the game can then also read back for collisions and pathfinding.</p>`,
    code: [
      {
        title: 'Island: fBm minus a radial falloff, then a biome lookup',
        lang: 'wgsl',
        src: `fn islandHeight(w: vec2f, oc: i32) -> f32 {
  let n = fbm(w * 1.5 + seedOffset, oc);
  let d = length(w * vec2f(0.72, 1.0));
  return n * 0.95 + 0.4 - 0.72 * d * d - u.sea;   // below 0 = water
}
// hillshade from finite differences
let hx = islandHeight(w + vec2f(e, 0.0), oc);
let hy = islandHeight(w + vec2f(0.0, e), oc);
let n = normalize(vec3f((h - hx) / e * 0.35, (h - hy) / e * 0.35, 1.0));
col = biome(h, moisture) * clamp(0.45 + 0.75 * dot(n, L), 0.3, 1.35);`,
      },
      {
        title: 'Caves: one material per block, from thresholds on noise',
        lang: 'wgsl',
        src: `let depth = c.y - surfaceY(c.x);                    // 1D noise surface
if (depth < 0.0) { return SKY; }
let n1 = fbm(c * vec2f(2.6, 3.8) + k, 4);
let n2 = fbm(c * 1.5 + k + vec2f(7.0, 3.0), 4);
let tunnel = abs(n1) < u.caveAmt * 0.06;               // thin "spaghetti" caves
let cavern = n2 > 0.36 - u.caveAmt * 0.22 * deep;      // big open caverns
if (tunnel || cavern) { return CAVE_AIR; }
if (depth > 0.55 && valueNoise(c * 20.0 + k) > 1.0 - oa) { return GOLD; }`,
      },
      {
        title: 'Contour lines with fwidth (constant pixel width at any zoom)',
        lang: 'wgsl',
        src: `let v = h / u.interval;
let line = 1.0 - smoothstep(0.5, 1.4, abs(fract(v + 0.5) - 0.5) / fwidth(v));
let line5 = /* same with v / 5.0, a bit thicker */;
col = mix(col, contourColor, max(line * 0.55, line5 * 0.85));`,
      },
    ],
    links: [
      { title: 'Red Blob Games — Making maps with noise', url: 'https://www.redblobgames.com/maps/terrain-from-noise/', note: 'elevation, moisture, biomes, islands' },
      { title: 'Amit Patel — Polygonal map generation', url: 'http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/' },
      { title: 'Shaded relief (Tom Patterson)', url: 'https://www.shadedrelief.com/', note: 'cartographic hillshading techniques' },
    ],
  },
});

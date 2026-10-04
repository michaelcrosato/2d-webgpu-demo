import { shaderScene } from '../../core/shaderscene.js';
import { GAME_SCENE_CODE, GAME_SCENE_INCLUDES } from '../../core/gamescene.js';

// Low-poly & flat geometric styles.
// A triangular lattice (rows shifted by half a cell = equilateral triangles) whose vertices are jittered by a
// hash. For each pixel we find the triangle that contains it (check the 3x3 neighbouring lattice cells, 18
// triangles, barycentric test). The triangle is then filled with ONE flat color: sampled at its centroid
// (filter), chosen by what it represents (landscape) or by a gradient field (abstract), and shaded with a
// normal computed from per-vertex heights -> flat Lambert lighting, the signature "faceted" look.

const CODE = /* wgsl */ `
struct Tri { a: vec2f, b: vec2f, c: vec2f, id: vec3f };

// height of the lattice vertex in "scene" terms (used to decide which vertices move)
fn animMask(j: f32) -> f32 {
  if (i32(u.example) == 1) {
    let y = j * 0.866 * u.cell / u.resolution.y;
    return smoothstep(0.6, 0.66, y);                       // only the sea moves
  }
  return 1.0;
}

// a jittered vertex of the triangular lattice, in cell units
fn latticeV(i: f32, j: f32) -> vec2f {
  let base = vec2f(i + 0.5 * fmod(j, 2.0), j * 0.866);
  let h = hash22(vec2f(i, j) + vec2f(u.seed * 13.1, u.seed * 7.7));
  var off = (h - 0.5) * u.jitter * 0.8;
  let ph = h * TAU;
  off += vec2f(sin(u.time * u.speed + ph.x), cos(u.time * u.speed * 0.83 + ph.y)) * u.jitter * 0.13 * step(0.001, u.speed) * animMask(j);
  return base + off;
}

fn bary(a: vec2f, b: vec2f, c: vec2f, p: vec2f) -> vec3f {
  let e = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  let w0 = ((b.x - p.x) * (c.y - p.y) - (b.y - p.y) * (c.x - p.x)) / e;
  let w1 = ((c.x - p.x) * (a.y - p.y) - (c.y - p.y) * (a.x - p.x)) / e;
  return vec3f(w0, w1, 1.0 - w0 - w1);
}

// find the triangle containing p (cell units): test 3 rows x 3 columns x 2 triangles
fn findTri(p: vec2f) -> Tri {
  let j0 = floor(p.y / 0.866);
  var best = Tri(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec3f(0.0));
  var bestW = -1.0e9;
  for (var dj = -1; dj <= 1; dj++) {
    let j = j0 + f32(dj);
    let odd = fmod(j, 2.0);
    let i0 = floor(p.x - 0.5 * odd);
    for (var di = -1; di <= 1; di++) {
      let i = i0 + f32(di);
      let v00 = latticeV(i, j);
      let v10 = latticeV(i + 1.0, j);
      let v01 = latticeV(i, j + 1.0);
      let v11 = latticeV(i + 1.0, j + 1.0);
      // even rows: (00,10,01) + (10,11,01); odd rows: (00,11,01) + (00,10,11)
      var a1 = v00; var b1 = v10; var c1 = v01;
      var a2 = v10; var b2 = v11; var c2 = v01;
      if (odd > 0.5) { b1 = v11; a2 = v00; b2 = v10; c2 = v11; }
      let w1 = bary(a1, b1, c1, p);
      let m1 = min(w1.x, min(w1.y, w1.z));
      if (m1 > bestW) { bestW = m1; best = Tri(a1, b1, c1, vec3f(i, j, 0.0)); }
      let w2 = bary(a2, b2, c2, p);
      let m2 = min(w2.x, min(w2.y, w2.z));
      if (m2 > bestW) { bestW = m2; best = Tri(a2, b2, c2, vec3f(i, j, 1.0)); }
    }
  }
  return best;
}

fn edgeDist(t: Tri, p: vec2f) -> f32 {
  return min(sdSegment(p, t.a, t.b), min(sdSegment(p, t.b, t.c), sdSegment(p, t.c, t.a)));
}

// flat normal of a triangle whose corners have heights za, zb, zc (x, y in the same units)
fn faceNormal(a: vec2f, b: vec2f, c: vec2f, za: f32, zb: f32, zc: f32) -> vec3f {
  var n = normalize(cross(vec3f(b - a, zb - za), vec3f(c - a, zc - za)));
  if (n.z < 0.0) { n = -n; }
  return n;
}

fn lightDir() -> vec3f {
  let a = u.lightAngle * PI / 180.0;
  return normalize(vec3f(cos(a), sin(a), 0.9));
}

// ------------------------------------------------------------------ 1. filter over the game image
fn lowpolyFilter(px: vec2f) -> vec3f {
  let cs = u.cell;
  let tri = findTri(px / cs);
  let a = tri.a * cs;
  let b = tri.b * cs;
  let c = tri.c * cs;
  let cen = (a + b + c) / 3.0;
  // flat color: centroid sample + three samples half-way to the corners
  var col = TEX(game, cen / u.resolution).rgb * 2.0;
  col += TEX(game, mix(cen, a, 0.5) / u.resolution).rgb;
  col += TEX(game, mix(cen, b, 0.5) / u.resolution).rgb;
  col += TEX(game, mix(cen, c, 0.5) / u.resolution).rgb;
  col /= 5.0;
  // facets: treat image brightness at the corners as height -> one normal per triangle
  let za = luma(TEX(game, a / u.resolution).rgb) * cs * 1.5;
  let zb = luma(TEX(game, b / u.resolution).rgb) * cs * 1.5;
  let zc = luma(TEX(game, c / u.resolution).rgb) * cs * 1.5;
  let n = faceNormal(a, b, c, za, zb, zc);
  let lam = dot(n, lightDir());
  col *= mix(1.0, 0.55 + 0.6 * lam, u.facet);
  let d = edgeDist(tri, px / cs) * cs;
  col = mix(col, min(col * 1.4 + vec3f(0.08), vec3f(1.0)), (1.0 - smoothstep(0.4, 1.4, d)) * u.edges * u.edgeAlpha);
  // compare lens
  if (u.lens > 0.5 && u.mouse.w > 0.5) {
    let r = u.resolution.y * 0.14;
    let dm = length(px - u.mouse.xy);
    col = mix(col, TEX(game, px / u.resolution).rgb, 1.0 - smoothstep(r - 1.0, r, dm));
    col = mix(col, vec3f(1.0), (1.0 - smoothstep(1.0, 2.0, abs(dm - r))) * 0.9);
  }
  return col;
}

// ------------------------------------------------------------------ 2. faceted landscape
fn ridgeA(x: f32) -> f32 { return 0.56 - 0.26 * ridged(vec2f(x * 1.1 + 3.0, 1.0), 3); }   // far range (y of crest)
fn ridgeB(x: f32) -> f32 { return 0.62 - 0.2 * ridged(vec2f(x * 1.6 + 9.0, 4.0), 3); }    // near range
fn hillY(x: f32, aspect: f32) -> f32 { let k = (x - aspect * 0.9) / 0.5; return 1.08 - 0.4 * exp(-k * k) + 0.015 * sin(x * 7.0); }

// region of a point: 0 sky, 1 sun, 2 far mountains, 3 near mountains, 4 sea, 5 hill, 6 pine
fn regionOf(q: vec2f, aspect: f32) -> f32 {
  let hy = hillY(q.x, aspect);
  // pines on the hill: tall triangles
  for (var k = 0; k < 3; k++) {
    let tx = aspect * 0.76 + 0.13 * f32(k);
    let base = hillY(tx, aspect) + 0.03;
    let hgt = 0.26 + 0.06 * fract(f32(k) * 0.618 + 0.3);
    let rel = q - vec2f(tx, base);
    if (rel.y < 0.0 && rel.y > -hgt && abs(rel.x) < (rel.y + hgt) * 0.28) { return 6.0; }
  }
  if (q.y > hy) { return 5.0; }
  if (q.y > 0.64) { return 4.0; }
  if (q.y > ridgeB(q.x)) { return 3.0; }
  if (q.y > ridgeA(q.x)) { return 2.0; }
  let sp = vec2f(aspect * 0.3, 0.3);
  if (length(q - sp) < 0.075) { return 1.0; }
  if (length(q - sp) < 0.13) { return 0.5; }
  return 0.0;
}

fn heightAt(q: vec2f, region: f32) -> f32 {
  if (region > 1.5 && region < 2.5) { return (q.y - ridgeA(q.x)) * 0.8 + 0.06 * perlin(q * 9.0); }
  if (region > 2.5 && region < 3.5) { return (q.y - ridgeB(q.x)) * 0.9 + 0.07 * perlin(q * 11.0 + 3.0); }
  if (region > 3.5 && region < 4.5) { return 0.012 * sin(q.x * 40.0 + u.time * 1.3) + 0.01 * sin(q.x * 23.0 - q.y * 60.0 + u.time * 0.9); }
  if (region > 4.5) { return (q.y - hillY(q.x, u.resolution.x / u.resolution.y)) * 0.3 + 0.03 * perlin(q * 12.0); }
  return 0.006 * perlin(q * 14.0);
}

fn landscape(px: vec2f) -> vec3f {
  let aspect = u.resolution.x / u.resolution.y;
  let cs = u.cell;
  let tri = findTri(px / cs);
  let ap = tri.a * cs / u.resolution.y;
  let bp = tri.b * cs / u.resolution.y;
  let cp = tri.c * cs / u.resolution.y;
  let cen = (ap + bp + cp) / 3.0;
  let reg = regionOf(cen, aspect);
  let n = faceNormal(ap, bp, cp, heightAt(ap, reg), heightAt(bp, reg), heightAt(cp, reg));
  let L = lightDir();
  let lam = clamp(dot(n, L), 0.0, 1.0);
  let rnd = hash21(tri.id.xy + vec2f(tri.id.z * 0.37, 5.0)) - 0.5;
  var col = vec3f(0.0);
  if (reg < 0.75) {
    col = mix(vec3f(0.98, 0.72, 0.52), vec3f(0.36, 0.42, 0.72), 1.0 - smoothstep(0.0, 0.55, cen.y));
    col *= 1.0 + rnd * 0.06 * u.facet;
    if (reg > 0.25) { col = mix(col, vec3f(1.0, 0.86, 0.62), 0.45 + 0.1 * rnd); }   // halo ring around the sun
  } else if (reg < 1.5) {
    col = vec3f(1.0, 0.93, 0.7) * (1.0 + rnd * 0.05);
  } else if (reg < 2.5) {
    let base = mix(vec3f(0.55, 0.5, 0.72), vec3f(0.8, 0.66, 0.72), 0.35);
    col = base * mix(1.0, 0.62 + 0.55 * lam, u.facet);
    col = mix(col, vec3f(0.96, 0.95, 1.0) * mix(1.0, 0.75 + 0.3 * lam, u.facet), step(cen.y, ridgeA(cen.x) + 0.05) * step(cen.y, 0.42));
  } else if (reg < 3.5) {
    let base = vec3f(0.3, 0.3, 0.5);
    col = base * mix(1.0, 0.45 + 0.85 * lam, u.facet);
    col = mix(col, vec3f(0.92, 0.93, 1.0) * mix(1.0, 0.7 + 0.35 * lam, u.facet), step(cen.y, ridgeB(cen.x) + 0.04) * step(cen.y, 0.5));
  } else if (reg < 4.5) {
    let depth = smoothstep(0.64, 1.0, cen.y);
    col = mix(vec3f(0.38, 0.55, 0.8), vec3f(0.07, 0.18, 0.4), depth) * mix(1.0, 0.75 + 0.35 * lam, u.facet);
    // glints in the sun's column
    let hv = normalize(L + vec3f(0.0, 0.0, 1.0));
    let sx = aspect * 0.3;
    col += vec3f(1.0, 0.85, 0.6) * pow(max(dot(n, hv), 0.0), 60.0) * exp(-abs(cen.x - sx) * 7.0) * 0.5;
  } else if (reg < 5.5) {
    col = mix(vec3f(0.35, 0.62, 0.3), vec3f(0.85, 0.75, 0.45), 1.0 - smoothstep(0.62, 0.7, cen.y + 0.03 * rnd)) * mix(1.0, 0.55 + 0.6 * lam, u.facet);
  } else {
    // pines: dark green with a little per-triangle variation
    col = vec3f(0.1, 0.32, 0.22) * (1.0 + rnd * 0.25 * u.facet);
  }
  let d = edgeDist(tri, px / cs) * cs;
  col = mix(col, min(col * 1.3 + vec3f(0.06), vec3f(1.0)), (1.0 - smoothstep(0.4, 1.4, d)) * u.edges * u.edgeAlpha);
  return col;
}

// ------------------------------------------------------------------ 3. geometric abstract
fn pal(t: f32) -> vec3f {
  let k = i32(u.scheme);
  var c0 = vec3f(0.06, 0.1, 0.3); var c1 = vec3f(0.1, 0.65, 0.7); var c2 = vec3f(0.85, 0.35, 0.75);
  if (k == 1) { c0 = vec3f(0.15, 0.08, 0.3); c1 = vec3f(0.92, 0.3, 0.42); c2 = vec3f(1.0, 0.8, 0.35); }
  if (k == 2) { c0 = vec3f(0.62, 0.86, 0.8); c1 = vec3f(0.96, 0.9, 0.78); c2 = vec3f(0.98, 0.62, 0.62); }
  if (k == 3) { c0 = vec3f(0.08, 0.08, 0.1); c1 = vec3f(0.35, 0.36, 0.4); c2 = vec3f(0.85, 0.86, 0.9); }
  if (t < 0.5) { return mixOklab(c0, c1, t * 2.0); }
  return mixOklab(c1, c2, t * 2.0 - 1.0);
}

fn abstractArt(px: vec2f) -> vec3f {
  let cs = u.cell;
  let tri = findTri(px / cs);
  let ap = tri.a * cs / u.resolution.y;
  let bp = tri.b * cs / u.resolution.y;
  let cp = tri.c * cs / u.resolution.y;
  let cen = (ap + bp + cp) / 3.0;
  // a slowly flowing gradient field sampled once per triangle
  let t = u.time * 0.05;
  let f = 0.5 + 0.35 * sin(cen.x * 2.2 - cen.y * 1.4 + t * 3.0) + 0.25 * fbm(cen * 1.6 + vec2f(t, -t), 3);
  var col = pal(clamp(f, 0.0, 1.0));
  // facets from a smooth height field + per-triangle variation
  let ha = fbm(ap * 2.5 + vec2f(t), 2) * 0.12;
  let hb = fbm(bp * 2.5 + vec2f(t), 2) * 0.12;
  let hc = fbm(cp * 2.5 + vec2f(t), 2) * 0.12;
  let n = faceNormal(ap, bp, cp, ha, hb, hc);
  let lam = dot(n, lightDir());
  let rnd = hash21(tri.id.xy + vec2f(tri.id.z * 0.37, 9.0)) - 0.5;
  col *= mix(1.0, 0.75 + 0.45 * lam + rnd * 0.18, u.facet);
  // the mouse is a light that lifts nearby triangles
  let m = u.mouse.xy / u.resolution.y;
  col += pal(1.0) * 0.35 * exp(-length(cen - m) * 6.0) * u.mouse.w;
  let d = edgeDist(tri, px / cs) * cs;
  col = mix(col, vec3f(1.0), (1.0 - smoothstep(0.3, 1.3, d)) * u.edges * u.edgeAlpha * 0.6);
  return col;
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let ex = i32(u.example);
  var col = vec3f(0.0);
  if (ex == 0) { col = lowpolyFilter(px); }
  else if (ex == 1) { col = landscape(px); }
  else { col = abstractArt(px); }
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export default shaderScene({
  interaction: 'Hover: a lens shows the original (filter) or the mouse lights the triangles (abstract).',
  examples: [
    {
      id: 'filter',
      label: 'Low-poly filter',
      kind: 'In a game',
      note: 'The platformer rebuilt from flat triangles. Each triangle takes the average color around its centroid; the image brightness at its three corners acts as a height, giving every facet a tilt that catches the light.',
      params: { cell: 26, jitter: 0.6, edges: false, facet: 0.5, speed: 0, lightAngle: 225 },
    },
    {
      id: 'landscape',
      label: 'Faceted landscape',
      kind: 'Real life',
      note: 'Mountains, sea, a hill with pines — every element is just “which region is this triangle’s centroid in?”. Each region has its own height function; the flat normal of each triangle is lit by one directional light. Only the sea’s vertices move.',
      params: { cell: 34, jitter: 0.7, edges: false, facet: 0.85, speed: 0.8, lightAngle: 200 },
    },
    {
      id: 'abstract',
      label: 'Geometric abstract',
      kind: 'Abstract',
      note: 'Animated vertices + a flowing gradient sampled once per triangle = the “Trianglify” wallpaper look. Move the mouse to light it up; switch palettes and toggle the edges.',
      params: { cell: 70, jitter: 0.75, edges: true, edgeAlpha: 0.3, facet: 0.8, speed: 0.6, lightAngle: 235 },
    },
  ],
  controls: [
    { type: 'slider', key: 'cell', label: 'Triangle size (px)', min: 8, max: 160, step: 1, value: 26, help: 'Spacing of the lattice. Smaller = more detail, bigger = more abstract.' },
    { type: 'slider', key: 'jitter', label: 'Vertex jitter', min: 0, max: 0.9, step: 0.01, value: 0.6, help: '0 = perfect equilateral grid. Higher = irregular, Delaunay-like triangles.' },
    { type: 'slider', key: 'facet', label: 'Facet shading', min: 0, max: 1, step: 0.01, value: 0.5, help: 'How much each triangle’s flat normal is lit by the light. 0 = flat color only.' },
    { type: 'slider', key: 'lightAngle', label: 'Light direction', min: 0, max: 360, step: 1, value: 225, unit: '°' },
    { type: 'toggle', key: 'edges', label: 'Show triangle edges', value: false },
    { type: 'slider', key: 'edgeAlpha', label: 'Edge strength', min: 0, max: 1, step: 0.01, value: 0.35 },
    { type: 'slider', key: 'speed', label: 'Vertex motion', min: 0, max: 3, step: 0.01, value: 0, showFor: ['landscape', 'abstract'], help: 'Vertices wobble around their rest position (landscape: only the sea).' },
    {
      type: 'select', key: 'scheme', label: 'Palette', value: 'aurora', showFor: ['abstract'],
      options: [
        { value: 'aurora', label: 'Aurora' },
        { value: 'sunset', label: 'Sunset' },
        { value: 'pastel', label: 'Pastel' },
        { value: 'graphite', label: 'Graphite' },
      ],
    },
    { type: 'slider', key: 'seed', label: 'Random seed', min: 1, max: 40, step: 1, value: 1, help: 'A different hash = a different triangulation.' },
    { type: 'toggle', key: 'lens', label: 'Compare lens under the mouse', value: true, showFor: ['filter'] },
  ],
  uniforms: {
    cell: 'f32', jitter: 'f32', facet: 'f32', lightAngle: 'f32', edges: 'f32', edgeAlpha: 'f32',
    speed: 'f32', scheme: 'f32', seed: 'f32', lens: 'f32',
  },
  include: [...new Set(['hash', 'noise', 'sdf', 'color', 'math', ...GAME_SCENE_INCLUDES])],
  // WORKAROUND (core bug): input: 'game' binds the game target while rendering into it on WebGPU,
  // so the platformer image is rendered as an ordinary first pass named "game" instead.
  passes: [{ name: 'game', format: 'rgba8unorm', code: GAME_SCENE_CODE }],
  code: CODE,
  about: {
    summary: 'Flat-colored triangles and faceted lighting: the clean geometric look of low-poly art, done as a filter, a landscape and an abstract wallpaper.',
    what: `<p>Three uses of one idea: <b>cover the screen with triangles and give each one a single color</b>. The color comes from the game image (filter),
      from what the triangle represents (landscape) or from a gradient (abstract). A per-triangle normal adds the faceted, cut-gem lighting.</p>`,
    how: `<ol>
      <li><b>Lattice.</b> Vertices sit on a grid whose odd rows are shifted by half a cell and squashed by √3/2 — perfect equilateral triangles.</li>
      <li><b>Jitter.</b> Each vertex is moved by a hash of its grid index (the same random offset every frame, so triangles are stable). Optionally a <code>sin(time)</code> wobble animates them.</li>
      <li><b>Point location.</b> For each pixel, test the 18 triangles of the 3×3 neighbouring cells with barycentric weights and keep the one that contains it.</li>
      <li><b>One color per triangle.</b> Everything is evaluated at the <i>centroid</i> (average of the corners), so all pixels of a triangle get the same value.</li>
      <li><b>Facets.</b> Give the three corners a height (image brightness, mountain shape, waves…), take the cross product of two edges → a flat normal → Lambert lighting <code>dot(n, light)</code>.</li>
      <li><b>Edges.</b> The distance from the pixel to the triangle’s three sides draws optional anti-aliased outlines.</li>
    </ol>`,
    uses: [
      { title: 'Strategy & puzzle games', text: 'The Battle of Polytopia, Monument Valley and Lara Croft GO use flat-shaded geometric art that stays crisp at any resolution and reads clearly on small screens.' },
      { title: 'Backgrounds & menus', text: 'Trianglify-style animated gradients for title screens, loading screens and websites.' },
      { title: 'Stylised filters', text: 'Turn photos or gameplay into faceted art for transitions, “shattered” effects or a paused-game backdrop.' },
      { title: 'Indie 3D-in-2D', text: 'Alto’s Adventure and many mobile games mix flat geometric layers with simple gradients for a clean, cheap-to-render look.' },
    ],
    try: [
      'Set <i>Vertex jitter</i> to 0: a perfectly regular triangle grid. Raise it slowly to see it become organic.',
      'On the filter, push <i>Triangle size</i> to 100+: the game turns into a few dozen color facets — still recognisable!',
      'Rotate <i>Light direction</i> on the landscape: lit and shadowed sides of the mountains swap.',
      'Set <i>Facet shading</i> to 0 — flat colors only; the 3D feel disappears.',
      'On <b>Geometric abstract</b>, try the Sunset palette with edges on, then move the mouse across.',
    ],
    ask: [
      'a low-poly triangulated filter with flat-shaded facets',
      'Trianglify-style animated gradient background',
      'jittered triangle lattice with per-triangle flat colors',
      'flat-shaded low-poly landscape with a directional light',
      'shatter the screen into triangles for a transition',
    ],
    perf: `<p>Moderate and constant: per pixel ~36 vertex hashes and 18 barycentric tests, plus up to 8 texture reads. It does not depend on the triangle size.
      In a production game you would build a real triangle mesh once (CPU Delaunay) and draw it with a vertex buffer — then each pixel costs almost nothing.</p>`,
    api: `<p>Fragment-only, so it runs the same on WebGPU and WebGL2. On WebGPU you could instead generate the jittered mesh in a compute shader into a vertex buffer and
      rasterize real triangles — faster for huge resolutions and needed if you want each triangle to animate (shatter, fly away).</p>`,
    code: [
      {
        title: 'Find the triangle under a pixel',
        lang: 'wgsl',
        src: `let j0 = floor(p.y / 0.866);
for (var dj = -1; dj <= 1; dj++) {
  let j = j0 + f32(dj);
  let odd = fmod(j, 2.0);                  // odd rows are shifted half a cell
  let i0 = floor(p.x - 0.5 * odd);
  for (var di = -1; di <= 1; di++) {
    let i = i0 + f32(di);
    let v00 = latticeV(i, j);         let v10 = latticeV(i + 1.0, j);
    let v01 = latticeV(i, j + 1.0);   let v11 = latticeV(i + 1.0, j + 1.0);
    // two triangles per cell; keep the one where min(barycentric) is largest
    let w1 = bary(a1, b1, c1, p);
    if (min(w1.x, min(w1.y, w1.z)) > bestW) { … }
  }
}`,
      },
      {
        title: 'Flat (faceted) lighting',
        lang: 'wgsl',
        src: `fn faceNormal(a: vec2f, b: vec2f, c: vec2f, za: f32, zb: f32, zc: f32) -> vec3f {
  var n = normalize(cross(vec3f(b - a, zb - za), vec3f(c - a, zc - za)));
  if (n.z < 0.0) { n = -n; }      // always face the viewer
  return n;
}
let lam = dot(n, lightDir());
col *= mix(1.0, 0.55 + 0.6 * lam, u.facet);`,
      },
    ],
    links: [{ title: 'Delaunay triangulation (Wikipedia)', url: 'https://en.wikipedia.org/wiki/Delaunay_triangulation', note: 'the “best” way to triangulate random points' }],
  },
});

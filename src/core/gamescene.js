// A procedural, animated 2D platformer vista drawn entirely in one fragment shader.
// It is the default "input image" for post-processing & art-style scenes, so they have
// something game-like to work on: sky, sun, clouds, parallax mountains & hills, trees,
// a tiled ground, a brick platform with spinning coins, a hopping hero, torches and fireflies.
//
// Written in portable WGSL (runs on WebGPU and, translated, on WebGL2).
// Needs includes: hash, noise, sdf, math. Uses u.time and u.resolution.

export const GAME_SCENE_INCLUDES = ['hash', 'noise', 'sdf', 'math'];

export const GAME_SCENE_CODE = /* wgsl */ `
fn gs_aa(d: f32, px: f32) -> f32 { return clamp(0.5 - d / px, 0.0, 1.0); }

fn gs_mountain(x: f32) -> f32 { return 0.52 - 0.17 * ridged(vec2f(x * 1.1, 3.7), 4); }
fn gs_hill(x: f32) -> f32 { return 0.655 + 0.035 * sin(x * 2.3) + 0.025 * perlin(vec2f(x * 2.7, 1.3)); }

fn gs_tree(p: vec2f, base: vec2f, h: f32) -> f32 {
  // two stacked triangles (pine) + trunk; returns signed distance-ish
  let w: f32 = h * 0.42;
  let t1: f32 = sdTriangle(p, base + vec2f(-w, -h * 0.15), base + vec2f(w, -h * 0.15), base + vec2f(0.0, -h * 0.75));
  let t2: f32 = sdTriangle(p, base + vec2f(-w * 0.8, -h * 0.5), base + vec2f(w * 0.8, -h * 0.5), base + vec2f(0.0, -h));
  let trunk: f32 = sdBox(p - (base + vec2f(0.0, -h * 0.08)), vec2f(w * 0.12, h * 0.1));
  return min(min(t1, t2), trunk);
}

fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res: vec2f = u.resolution;
  let aspect: f32 = res.x / res.y;
  let p: vec2f = px / res.y;          // x in [0, aspect], y in [0, 1], y down
  let pw: f32 = 1.5 / res.y;          // ~1.5 pixel in p units (for anti-aliasing)
  let t: f32 = u.time;
  let scroll: f32 = t * 0.12;

  // ---- sky ----
  var col: vec3f = mix(vec3f(0.16, 0.34, 0.74), vec3f(0.98, 0.76, 0.56), smoothstep(0.0, 0.72, p.y));
  let sunPos: vec2f = vec2f(aspect * 0.78, 0.25);
  let sd: f32 = length(p - sunPos);
  col += vec3f(1.0, 0.82, 0.45) * exp(-sd * 8.0) * 0.6;
  col = mix(col, vec3f(1.0, 0.98, 0.88), gs_aa(sd - 0.055, pw));

  // ---- clouds ----
  let cq: vec2f = vec2f(p.x * 1.4 + t * 0.025, p.y * 3.2);
  let cn: f32 = fbm(cq * 2.2, 4) * 0.5 + 0.5;
  let band: f32 = smoothstep(0.04, 0.16, p.y) * smoothstep(0.44, 0.26, p.y);
  let cloud: f32 = smoothstep(0.5, 0.68, cn) * band;
  let cshade: f32 = smoothstep(0.5, 0.8, fbm(cq * 2.2 + vec2f(0.0, 0.25), 3) * 0.5 + 0.5);
  col = mix(col, mix(vec3f(1.0, 0.97, 0.94), vec3f(0.78, 0.74, 0.86), cshade * 0.7), cloud * 0.92);

  // ---- far mountains (parallax 0.15) ----
  let mx: f32 = p.x + scroll * 0.15;
  let mh: f32 = gs_mountain(mx);
  let haze: vec3f = vec3f(0.62, 0.62, 0.82);
  if (p.y > mh) {
    var mc: vec3f = mix(vec3f(0.36, 0.38, 0.62), vec3f(0.27, 0.27, 0.48), smoothstep(mh, mh + 0.25, p.y));
    let snow: f32 = smoothstep(0.43, 0.39, mh) * smoothstep(mh + 0.035, mh + 0.01, p.y);
    mc = mix(mc, vec3f(0.93, 0.94, 1.0), snow);
    col = mix(col, mix(mc, haze, 0.35), gs_aa(mh - p.y, pw));
  }

  // ---- hills with pine trees (parallax 0.4) ----
  let hx: f32 = p.x + scroll * 0.4;
  let hh: f32 = gs_hill(hx);
  var hillCol: vec3f = mix(vec3f(0.30, 0.58, 0.34), vec3f(0.16, 0.36, 0.25), smoothstep(hh, hh + 0.2, p.y));
  let hillMask: f32 = gs_aa(hh - p.y, pw);
  // trees: one candidate per cell, check neighbours so they can overlap cell borders
  var treeMask: f32 = 0.0;
  let cellW: f32 = 0.11;
  let cell: f32 = floor(hx / cellW);
  for (var k: i32 = -1; k <= 1; k++) {
    let c: f32 = cell + f32(k);
    let r: f32 = hash11(c * 1.37);
    if (r > 0.35) {
      let tx: f32 = (c + 0.2 + 0.6 * hash11(c * 7.1)) * cellW;
      let th: f32 = 0.06 + 0.07 * hash11(c * 3.3);
      let base: vec2f = vec2f(tx, gs_hill(tx) + 0.012);
      let d: f32 = gs_tree(vec2f(hx, p.y), base, th);
      treeMask = max(treeMask, gs_aa(d, pw));
    }
  }
  col = mix(col, mix(hillCol, haze, 0.12), hillMask);
  col = mix(col, vec3f(0.10, 0.27, 0.19), treeMask);

  // ---- fireflies ----
  for (var i: i32 = 0; i < 10; i++) {
    let fi: f32 = f32(i);
    let fp: vec2f = vec2f(fmod(hash11(fi * 4.1) * aspect * 1.3 - scroll * 0.4 + 10.0 * aspect, aspect * 1.2) - 0.05,
                          0.58 + 0.15 * hash11(fi * 9.7) + 0.02 * sin(t * 1.3 + fi));
    let fd: f32 = length(p - fp);
    let blink: f32 = 0.5 + 0.5 * sin(t * 3.0 + fi * 2.0);
    col += vec3f(0.9, 1.0, 0.45) * exp(-fd * 120.0) * blink * 0.9;
  }

  // ---- ground (parallax 1.0) ----
  let gx: f32 = p.x + scroll;
  let groundY: f32 = 0.8;
  let blade: f32 = 0.006 * hash11(floor(gx * 260.0)) + 0.004 * sin(gx * 90.0 + t * 2.0);
  if (p.y > groundY - blade - 0.01) {
    let tile: vec2f = vec2f(gx, p.y - groundY) / 0.05;
    let tid: vec2f = floor(tile);
    let tf: vec2f = fract(tile);
    var dirt: vec3f = mix(vec3f(0.52, 0.34, 0.2), vec3f(0.42, 0.26, 0.15), hash21(tid));
    let edge: f32 = min(min(tf.x, 1.0 - tf.x), min(tf.y, 1.0 - tf.y));
    dirt *= 0.75 + 0.25 * smoothstep(0.0, 0.08, edge);
    let peb: vec4f = voronoi(vec2f(gx, p.y) * 60.0);
    dirt = mix(dirt, vec3f(0.62, 0.55, 0.48), smoothstep(0.25, 0.18, peb.x) * step(0.7, hash21(peb.zw)));
    dirt *= mix(1.0, 0.55, smoothstep(groundY, 1.0, p.y));
    let grassD: f32 = groundY + 0.022 - p.y;
    var g: vec3f = mix(vec3f(0.36, 0.72, 0.28), vec3f(0.55, 0.85, 0.32), smoothstep(groundY + 0.02, groundY - 0.01, p.y));
    let gm: f32 = gs_aa(-grassD, pw);
    let topMask: f32 = gs_aa(groundY - blade - p.y, pw);
    col = mix(col, mix(dirt, g, gm), topMask);
  }

  // ---- floating brick platform + coins ----
  let seg: f32 = 1.1;
  let sx: f32 = fmod(gx, seg);
  let segId: f32 = floor(gx / seg);
  if (hash11(segId * 2.7) > 0.3) {
    let pb: vec2f = vec2f(sx - 0.55, p.y - 0.56);
    let pd: f32 = sdBox(pb, vec2f(0.2, 0.025));
    if (pd < pw * 2.0) {
      var bq: vec2f = vec2f(sx, p.y) / vec2f(0.05, 0.025);
      bq.x += 0.5 * floor(bq.y);
      let bf: vec2f = fract(bq);
      let mortar: f32 = smoothstep(0.0, 0.08, min(min(bf.x, 1.0 - bf.x), min(bf.y, 1.0 - bf.y)) * 1.5);
      let bc: vec3f = mix(vec3f(0.55, 0.22, 0.14), vec3f(0.72, 0.32, 0.2), hash21(floor(bq)));
      col = mix(col, mix(vec3f(0.85, 0.78, 0.66), bc, mortar), gs_aa(pd, pw));
    }
    for (var c: i32 = 0; c < 4; c++) {
      let cxp: f32 = 0.43 + f32(c) * 0.08;
      let spin: f32 = abs(cos(t * 3.0 + f32(c) * 0.6 + segId));
      let cp: vec2f = vec2f(sx - cxp, p.y - 0.49 - 0.006 * sin(t * 4.0 + f32(c)));
      let cdist: f32 = sdEllipseApprox(cp, vec2f(0.018 * max(spin, 0.12), 0.022));
      let shine: f32 = smoothstep(0.02, 0.0, length(cp - vec2f(-0.005 * spin, -0.008)));
      let coinCol: vec3f = mix(vec3f(0.95, 0.68, 0.12), vec3f(1.0, 0.95, 0.55), shine);
      col = mix(col, coinCol, gs_aa(cdist, pw));
      col += vec3f(1.0, 0.8, 0.3) * exp(-max(length(cp) - 0.02, 0.0) * 60.0) * 0.12;
    }
  }

  // ---- torches ----
  let tseg: f32 = 0.9;
  let tx: f32 = fmod(gx + 0.3, tseg) - 0.45;
  let post: f32 = sdBox(vec2f(tx, p.y - 0.75), vec2f(0.006, 0.05));
  col = mix(col, vec3f(0.25, 0.17, 0.12), gs_aa(post, pw));
  let flick: f32 = 0.85 + 0.15 * sin(t * 17.0 + floor((gx + 0.3) / tseg) * 3.0) * sin(t * 23.0);
  let fq: vec2f = vec2f(tx, (p.y - 0.688) * 0.7);
  let fd: f32 = length(fq) - 0.011 * flick;
  col = mix(col, vec3f(1.0, 0.85, 0.35), gs_aa(fd, pw));
  col += vec3f(1.0, 0.5, 0.12) * exp(-max(fd, 0.0) * 22.0) * 0.55 * flick;

  // ---- hero (screen-fixed, hopping) ----
  let hop: f32 = abs(sin(t * 2.6));
  let heroBase: vec2f = vec2f(0.45, groundY - 0.042 - hop * 0.13);
  let squash: f32 = 1.0 + 0.18 * (1.0 - smoothstep(0.0, 0.25, hop));
  let hp: vec2f = (p - heroBase) * vec2f(1.0 / squash, squash);
  // shadow
  let shadowW: f32 = 0.04 * (1.0 - 0.5 * hop);
  let sh: f32 = sdEllipseApprox(p - vec2f(0.45, groundY + 0.004), vec2f(shadowW, 0.008));
  col = mix(col, col * 0.55, gs_aa(sh, pw) * 0.8);
  // cape
  let cape: f32 = sdTriangle(hp, vec2f(-0.03, -0.03), vec2f(-0.03, 0.02), vec2f(-0.07 - 0.01 * sin(t * 8.0), 0.03));
  col = mix(col, vec3f(0.25, 0.3, 0.75), gs_aa(cape, pw));
  let body: f32 = sdRoundBox(hp, vec2f(0.034, 0.04), 0.016);
  col = mix(col, vec3f(0.08, 0.05, 0.06), gs_aa(body - 0.004, pw));
  col = mix(col, mix(vec3f(0.95, 0.32, 0.25), vec3f(1.0, 0.55, 0.4), smoothstep(0.02, -0.03, hp.y)), gs_aa(body, pw));
  let eyeL: f32 = length((hp - vec2f(0.004, -0.012)) * vec2f(1.0, 0.7)) - 0.009;
  let eyeR: f32 = length((hp - vec2f(0.024, -0.012)) * vec2f(1.0, 0.7)) - 0.009;
  col = mix(col, vec3f(1.0), gs_aa(min(eyeL, eyeR), pw));
  let pupil: f32 = min(length(hp - vec2f(0.007, -0.011)), length(hp - vec2f(0.027, -0.011))) - 0.0045;
  col = mix(col, vec3f(0.05), gs_aa(pupil, pw));

  // ---- foreground bushes (parallax 1.6, dark silhouettes) ----
  let fx: f32 = p.x + scroll * 1.6;
  let bushTop: f32 = 0.95 - 0.07 * smoothstep(0.35, 0.75, valueNoise(vec2f(fx * 3.0, 0.5))) - 0.015 * valueNoise(vec2f(fx * 25.0, 2.0));
  col = mix(col, vec3f(0.05, 0.13, 0.09), gs_aa(bushTop - p.y, pw));

  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

/**
 * For custom WebGPU scenes: render the game scene into a target each frame.
 *   const game = createGameScene(gpu);
 *   const tex = game.render(encoder, time, width, height); // returns a target {texture, view, ...}
 */
export function createGameScene(gpu) {
  const block = gpu.uniforms({ resolution: 'vec2f', time: 'f32' }, 'U');
  const fx = gpu.fullscreen({ label: 'game-scene', code: GAME_SCENE_CODE, uniforms: block, include: GAME_SCENE_INCLUDES, format: 'rgba8unorm' });
  let target = null;
  return {
    render(encoder, time, width, height) {
      if (!target || target.width !== width || target.height !== height) {
        target?.destroy();
        target = gpu.target(width, height, { label: 'game-scene' });
      }
      block.set('resolution', [width, height]).set('time', time);
      fx.draw(encoder, target);
      return target;
    },
    get target() {
      return target;
    },
  };
}

// Optional user-supplied image that replaces the procedural game scene as the input of
// post-processing / art-style scenes (shaderScene({ input: 'game' })).
// The app's "Input image" panel calls setInputImage(); scenes pick it up on the next frame.

let current = null; // { bitmap, name, version }
let version = 0;

export function setInputImage(bitmap, name = 'image') {
  current = bitmap ? { bitmap, name, version: ++version } : null;
}
export function getInputImage() {
  return current;
}

/** Portable-WGSL fullscreen pass that draws texture `img` "cover"-fitted (like CSS object-fit: cover). */
export const COVER_CODE = /* wgsl */ `
fn shade(uv: vec2f, px: vec2f) -> vec4f {
  let res = u.resolution;
  let isz = TEXSIZE(img);
  let s = max(res.x / isz.x, res.y / isz.y);
  let q = ((px - res * 0.5) / s + isz * 0.5) / isz;
  return vec4f(TEX(img, q).rgb, 1.0);
}`;

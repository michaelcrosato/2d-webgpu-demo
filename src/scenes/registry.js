// Every scene in the showcase. Modules are loaded lazily when selected.
// badges: 'gl2' = also runs on WebGL2, 'compute' = uses compute shaders, 'interactive' = mouse/keys.

export const CATEGORIES = [
  {
    id: 'start',
    title: 'Start Here',
    color: '#7dd3fc',
    blurb: 'How the GPU draws 2D, what shaders are, and why WebGPU compute changes the game.',
  },
  {
    id: 'shapes',
    title: 'Shapes, Lines & Color',
    color: '#f9a8d4',
    blurb: 'Crisp vector shapes, gradients, blending and text — the building blocks of every 2D frame.',
  },
  {
    id: 'procedural',
    title: 'Procedural Generation',
    color: '#86efac',
    blurb: 'Noise, cells, fractals and patterns: infinite worlds and textures from a few lines of math.',
  },
  {
    id: 'sprites',
    title: 'Sprites, Tiles & Cameras',
    color: '#fcd34d',
    blurb: 'Batching, animation, tilemaps, parallax and cameras — the core of a 2D game renderer.',
  },
  {
    id: 'lighting',
    title: 'Lighting & Shadows',
    color: '#fdba74',
    blurb: 'Dynamic lights, normal maps, shadows, bloom and global illumination in 2D.',
  },
  {
    id: 'vfx',
    title: 'Particles & VFX',
    color: '#c4b5fd',
    blurb: 'Fire, magic, weather, trails, lightning and explosions — the juice that makes games feel alive.',
  },
  {
    id: 'simulation',
    title: 'GPU Simulations',
    color: '#5eead4',
    blurb: 'Fluids, flocks, sand, slime and physics running massively parallel on the GPU.',
  },
  {
    id: 'post',
    title: 'Post-Processing',
    color: '#a5b4fc',
    blurb: 'Full-screen effects applied after the scene is drawn: blur, grading, distortion, glitches.',
  },
  {
    id: 'styles',
    title: 'Art Styles',
    color: '#f0abfc',
    blurb: 'Pixel art, CRT, 1-bit, comic, painterly, neon… popular looks and how to build them.',
  },
  {
    id: 'tricks',
    title: 'Pseudo-3D & Game Tricks',
    color: '#fda4af',
    blurb: 'Mode 7, raycasting, water, foliage, destructible terrain, portals and other classic tricks.',
  },
];

const S = (category, id, title, blurb, badges, load) => ({ category, id, title, blurb, badges, load });

export const SCENES = [
  // ---------------------------------------------------------------- start
  S('start', 'gpu-pipeline', 'How a GPU Draws 2D', 'Vertices, triangles, rasterization and pixels — the pipeline behind every sprite.', ['gl2', 'interactive'], () => import('./start/gpu-pipeline.js')),
  S('start', 'shader-basics', 'Shaders 101: UV, Time & Math', 'A fragment shader runs once per pixel. See how uv, time, step and sin paint images.', ['gl2'], () => import('./start/shader-basics.js')),
  S('start', 'compute-power', 'Compute Shaders: WebGPU’s Superpower', 'Move a million particles. Compare CPU JavaScript vs a GPU compute shader.', ['compute', 'interactive'], () => import('./start/compute-power.js')),

  // ---------------------------------------------------------------- shapes
  S('shapes', 'sdf-shapes', 'Signed Distance Fields (SDF)', 'Resolution-independent shapes from math: union, blend, outline, glow.', ['gl2', 'interactive'], () => import('./shapes/sdf-shapes.js')),
  S('shapes', 'gradients-color', 'Gradients & Color Spaces', 'Linear, radial and conic gradients; sRGB vs linear vs OKLab mixing; banding and dithering.', ['gl2'], () => import('./shapes/gradients-color.js')),
  S('shapes', 'lines-curves', 'Lines, Curves & Vector Graphics', 'Thick lines, joins, dashes, Bézier curves and glowing lasers on the GPU.', ['interactive'], () => import('./shapes/lines-curves.js')),
  S('shapes', 'blend-modes', 'Blend Modes & Transparency', 'Alpha, additive, multiply, screen, premultiplied alpha — and when to use each.', [], () => import('./shapes/blend-modes.js')),
  S('shapes', 'antialiasing', 'Anti-aliasing & Pixel Scaling', 'Jaggies, MSAA, SDF smoothing, nearest vs linear and crisp pixel-art scaling.', [], () => import('./shapes/antialiasing.js')),
  S('shapes', 'ui-rendering', 'Game UI: Bars, Panels & Cooldowns', 'Health bars, radial cooldowns, nine-slice panels and juicy UI, all on the GPU.', ['gl2', 'interactive'], () => import('./shapes/ui-rendering.js')),
  S('shapes', 'sdf-text', 'Text Rendering: Bitmap vs SDF', 'Crisp text at any size with outlines, glow and shadows via signed distance fields.', ['compute', 'interactive'], () => import('./shapes/sdf-text.js')),

  // ---------------------------------------------------------------- procedural
  S('procedural', 'noise', 'Noise: Value, Perlin, Simplex & Worley', 'The random-but-smooth functions behind clouds, terrain, fire and water.', ['gl2'], () => import('./procedural/noise.js')),
  S('procedural', 'fbm-domain-warp', 'fBm & Domain Warping', 'Layer noise for detail, then bend space with noise: clouds, marble, gas giants.', ['gl2'], () => import('./procedural/fbm-domain-warp.js')),
  S('procedural', 'voronoi', 'Voronoi & Cellular Patterns', 'Cells, cracks, stained glass, scales and territory maps.', ['gl2', 'interactive'], () => import('./procedural/voronoi.js')),
  S('procedural', 'terrain-gen', 'World & Terrain Generation', 'Islands, biomes, caves and heightmaps generated from noise.', ['gl2', 'interactive'], () => import('./procedural/terrain-gen.js')),
  S('procedural', 'patterns-tiling', 'Patterns, Tiling & Truchet', 'Repetition, hex grids, truchet tiles and kaleidoscopes.', ['gl2'], () => import('./procedural/patterns-tiling.js')),
  S('procedural', 'fractals', 'Fractals: Mandelbrot & Julia', 'Infinite detail from z = z² + c. Zoom in forever.', ['gl2', 'interactive'], () => import('./procedural/fractals.js')),

  // ---------------------------------------------------------------- sprites
  S('sprites', 'sprite-batching', 'Sprite Batching & Instancing', 'Draw 100,000+ sprites in one call: bunnymark, bullet hell, crowds.', ['compute', 'interactive'], () => import('./sprites/sprite-batching.js')),
  S('sprites', 'sprite-animation', 'Sprite Animation & Atlases', 'Flipbooks, texture atlases, squash & stretch and palette swaps.', ['interactive'], () => import('./sprites/sprite-animation.js')),
  S('sprites', 'tilemaps', 'GPU Tilemaps', 'Whole tile layers in one draw call, animated tiles, autotiling and huge maps.', ['compute', 'interactive'], () => import('./sprites/tilemaps.js')),
  S('sprites', 'parallax', 'Parallax Scrolling & Depth', 'Layers moving at different speeds plus atmospheric fog create depth.', ['gl2', 'interactive'], () => import('./sprites/parallax.js')),
  S('sprites', 'camera-2d', '2D Cameras: Follow, Shake & Zoom', 'Smooth follow, dead zones, look-ahead, trauma-based screen shake and zoom.', ['interactive'], () => import('./sprites/camera-2d.js')),
  S('sprites', 'sprite-effects', 'Sprite Shader Effects', 'Hit flash, outlines, dissolve, hologram, freeze and x-ray silhouettes.', ['interactive'], () => import('./sprites/sprite-effects.js')),
  S('sprites', 'isometric', 'Isometric Worlds & Depth Sorting', '2:1 isometric projection, height maps and correct draw ordering.', ['interactive'], () => import('./sprites/isometric.js')),

  // ---------------------------------------------------------------- lighting
  S('lighting', 'lights-2d', 'Dynamic 2D Lights & Normal Maps', 'Point lights, falloff, colored light and normal-mapped sprites.', ['compute', 'interactive'], () => import('./lighting/lights-2d.js')),
  S('lighting', 'shadows-2d', '2D Shadows & Line of Sight', 'Hard and soft shadows from occluders, visibility and stealth cones.', ['compute', 'interactive'], () => import('./lighting/shadows-2d.js')),
  S('lighting', 'global-illumination', '2D Global Illumination', 'Light that bounces: emissive surfaces light the world, via ray marching distance fields.', ['compute', 'interactive'], () => import('./lighting/global-illumination.js')),
  S('lighting', 'bloom', 'Bloom & HDR Glow', 'Bright things bleed light: threshold, blur pyramid and tone mapping.', ['interactive'], () => import('./lighting/bloom.js')),
  S('lighting', 'god-rays', 'God Rays & Light Shafts', 'Volumetric-looking beams through trees, windows and water.', ['gl2', 'interactive'], () => import('./lighting/god-rays.js')),
  S('lighting', 'fog-of-war', 'Fog of War & Visibility', 'Explored vs visible areas, soft reveal and memory, RTS & roguelike style.', ['compute', 'interactive'], () => import('./lighting/fog-of-war.js')),
  S('lighting', 'day-night', 'Day/Night Cycle & Ambient Light', 'Sky gradients, sun and moon, stars, and lights that switch on at dusk.', ['gl2'], () => import('./lighting/day-night.js')),

  // ---------------------------------------------------------------- vfx
  S('vfx', 'gpu-particles', 'GPU Particle Systems', 'Emitters, forces and lifetimes for a million particles: fireworks, magic, sparks.', ['compute', 'interactive'], () => import('./vfx/gpu-particles.js')),
  S('vfx', 'fire-effects', 'Three Ways to Make Fire', 'Noise shader fire, the classic Doom PSX fire, and particle fire compared.', ['gl2'], () => import('./vfx/fire-effects.js')),
  S('vfx', 'weather', 'Weather: Rain, Snow & Wind', 'Layered precipitation with splashes, wind gusts, fog and lightning flashes.', ['compute'], () => import('./vfx/weather.js')),
  S('vfx', 'trails-ribbons', 'Trails, Ribbons & Slashes', 'Mouse trails, sword slashes, comet tails and motion streaks.', ['interactive'], () => import('./vfx/trails-ribbons.js')),
  S('vfx', 'lightning', 'Lightning & Electric Arcs', 'Branching bolts, chain lightning and crackling arcs with glow.', ['interactive'], () => import('./vfx/lightning.js')),
  S('vfx', 'explosions', 'Game Feel: Hits & Explosions', 'Shockwaves, debris, flashes, hit-stop and screen shake — "juice".', ['interactive'], () => import('./vfx/explosions.js')),

  // ---------------------------------------------------------------- simulation
  S('simulation', 'boids', 'Flocking (Boids)', 'Separation, alignment, cohesion: birds, fish and swarms emerge from three rules.', ['compute', 'interactive'], () => import('./simulation/boids.js')),
  S('simulation', 'fluid-sim', 'Fluid Simulation', 'Real-time Navier–Stokes: ink, smoke and paint you can stir.', ['compute', 'interactive'], () => import('./simulation/fluid-sim.js')),
  S('simulation', 'reaction-diffusion', 'Reaction–Diffusion', 'Two chemicals, two rules: coral, zebra stripes, leopard spots.', ['gl2', 'interactive'], () => import('./simulation/reaction-diffusion.js')),
  S('simulation', 'game-of-life', 'Cellular Automata', 'Conway’s Life and friends — complex behaviour from tiny local rules.', ['gl2', 'interactive'], () => import('./simulation/game-of-life.js')),
  S('simulation', 'falling-sand', 'Falling Sand (Noita-style)', 'Every pixel is simulated: sand, water, fire, smoke, wood and more.', ['compute', 'interactive'], () => import('./simulation/falling-sand.js')),
  S('simulation', 'physarum', 'Slime Mold (Physarum)', 'Millions of agents following trails form organic transport networks.', ['compute', 'interactive'], () => import('./simulation/physarum.js')),
  S('simulation', 'water-ripples', 'Water Ripples', 'The wave equation on a grid: ponds, puddles, raindrops and refraction.', ['gl2', 'interactive'], () => import('./simulation/water-ripples.js')),
  S('simulation', 'verlet-physics', 'Verlet Physics: Ropes & Cloth', 'Points and constraints: rope bridges, flags, chains and jelly.', ['compute', 'interactive'], () => import('./simulation/verlet-physics.js')),
  S('simulation', 'nbody', 'N-Body Gravity & Galaxies', 'Every star pulls every other star: galaxy collisions on the GPU.', ['compute', 'interactive'], () => import('./simulation/nbody.js')),
  S('simulation', 'metaballs', 'Metaballs & Goo', 'Blobs that melt together: lava lamps, slime and gooey UI.', ['gl2', 'interactive'], () => import('./simulation/metaballs.js')),
  S('simulation', 'jump-flood', 'Jump Flooding: Distance Fields', 'Turn any shape into a distance field in log₂(n) passes: outlines, glows, Voronoi.', ['compute', 'interactive'], () => import('./simulation/jump-flood.js')),

  // ---------------------------------------------------------------- post
  S('post', 'blur', 'Blur: Gaussian, Kawase & Tilt-Shift', 'Separable blurs, fast dual-filter blur, radial zoom blur and miniature tilt-shift.', ['gl2', 'interactive'], () => import('./post/blur.js')),
  S('post', 'color-grading', 'Color Grading, LUTs & Film Look', 'Exposure, contrast, temperature, LUT presets, vignette, grain and letterbox.', ['gl2'], () => import('./post/color-grading.js')),
  S('post', 'distortion', 'Distortion: Shockwaves & Heat Haze', 'Shockwave rings, heat shimmer, swirl, lens and underwater warps.', ['gl2', 'interactive'], () => import('./post/distortion.js')),
  S('post', 'chromatic-glitch', 'Chromatic Aberration & Glitch', 'RGB split, block glitches, scan jitter and datamosh-style corruption.', ['gl2', 'interactive'], () => import('./post/chromatic-glitch.js')),
  S('post', 'edge-detection', 'Convolution & Edge Detection', 'Kernels for blur, sharpen, emboss; Sobel outlines and night vision.', ['gl2'], () => import('./post/edge-detection.js')),
  S('post', 'transitions', 'Screen Transitions', 'Fades, wipes, iris, pixelate, dissolve and more between two scenes.', ['gl2'], () => import('./post/transitions.js')),
  S('post', 'feedback-trails', 'Feedback & Motion Trails', 'Reuse the previous frame: ghosting, smears, infinite tunnels and dream effects.', ['gl2', 'interactive'], () => import('./post/feedback-trails.js')),

  // ---------------------------------------------------------------- styles
  S('styles', 'pixel-art', 'Pixel Art Pipeline', 'Low-res render targets, palette quantization: Game Boy, PICO-8, NES.', ['gl2', 'interactive'], () => import('./styles/pixel-art.js')),
  S('styles', 'dithering', 'Dithering & 1-bit', 'Bayer, blue noise and error-diffusion looks: Obra Dinn, Playdate, print.', ['gl2', 'interactive'], () => import('./styles/dithering.js')),
  S('styles', 'crt-retro', 'CRT, Arcade & VHS', 'Curvature, scanlines, shadow masks, bloom, tracking errors and tape noise.', ['gl2', 'interactive'], () => import('./styles/crt-retro.js')),
  S('styles', 'neon-synthwave', 'Neon, Synthwave & Vector Glow', 'Glowing lines, retro grids and Geometry-Wars-style vector arenas.', ['gl2', 'interactive'], () => import('./styles/neon-synthwave.js')),
  S('styles', 'toon-comic', 'Comic, Halftone & Cel Shading', 'Ink outlines, halftone dots, posterized cel shading and speed lines.', ['gl2', 'interactive'], () => import('./styles/toon-comic.js')),
  S('styles', 'painterly', 'Painterly: Oil, Watercolor & Sketch', 'Kuwahara oil paint, watercolor bleeding and pencil hatching filters.', ['gl2', 'interactive'], () => import('./styles/painterly.js')),
  S('styles', 'ascii-terminal', 'ASCII & Text-Mode', 'Render the world as characters: ASCII art, Matrix rain, roguelike terminals.', ['gl2', 'interactive'], () => import('./styles/ascii-terminal.js')),
  S('styles', 'silhouette', 'Silhouette & Atmosphere', 'Limbo/Inside-style layered silhouettes, fog, grain and rim light.', ['gl2'], () => import('./styles/silhouette.js')),
  S('styles', 'low-poly', 'Low-Poly & Flat Geometric', 'Triangulated landscapes, flat shading and geometric abstraction.', ['gl2', 'interactive'], () => import('./styles/low-poly.js')),
  S('styles', 'hand-drawn', 'Hand-Drawn: Line Boil & Paper', 'Wobbly animated linework, paper grain, crayon and chalkboard looks.', ['gl2'], () => import('./styles/hand-drawn.js')),
  S('styles', 'holo-foil', 'Holographic Foil & Shiny Cards', 'Tilt-reactive rainbow foil, glitter and metallic sheen (Balatro/Pokémon style).', ['gl2', 'interactive'], () => import('./styles/holo-foil.js')),

  // ---------------------------------------------------------------- tricks
  S('tricks', 'mode7', 'Mode 7 (SNES Pseudo-3D)', 'A tilted, rotating textured floor: kart racers and airship maps.', ['gl2', 'interactive'], () => import('./tricks/mode7.js')),
  S('tricks', 'raycaster', 'Raycasting (Wolfenstein-style)', 'Fake 3D from a 2D grid map, one ray per screen column.', ['gl2', 'interactive'], () => import('./tricks/raycaster.js')),
  S('tricks', 'pseudo-3d-road', 'Pseudo-3D Road (OutRun-style)', 'Curves, hills and scaling roadside sprites from a 1D road.', ['gl2', 'interactive'], () => import('./tricks/pseudo-3d-road.js')),
  S('tricks', 'water-2d', '2D Water: Reflection & Refraction', 'Spring-based surface waves, reflections, caustics and underwater tint.', ['interactive'], () => import('./tricks/water-2d.js')),
  S('tricks', 'grass-foliage', 'Wind, Grass & Foliage', 'Vertex-shader wind, interactive grass that bends around the player.', ['compute', 'interactive'], () => import('./tricks/grass-foliage.js')),
  S('tricks', 'destructible-terrain', 'Destructible Terrain', 'Worms-style terrain stored in a texture you can blast apart.', ['interactive'], () => import('./tricks/destructible-terrain.js')),
  S('tricks', 'render-to-texture', 'Render Targets: Minimaps & Portals', 'Render the world into a texture, then reuse it: minimaps, portals, mirrors.', ['interactive'], () => import('./tricks/render-to-texture.js')),
  S('tricks', 'masking-stencil', 'Masks & Stencils', 'Spotlight reveals, x-ray vision, scratch cards and shaped windows.', ['interactive'], () => import('./tricks/masking-stencil.js')),
];

export const sceneById = (id) => SCENES.find((s) => s.id === id);

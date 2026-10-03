// TEMPORARY core smoke test (will be replaced by the real scene)
import { ShapeBatch, SpriteBatch, Camera2D } from '../../core/batch.js';
import { getAtlas } from '../../core/assets.js';
import { createGameScene } from '../../core/gamescene.js';
export default {
  examples: [{ id: 'a', label: 'A' }, { id: 'b', label: 'B' }],
  controls: [],
  about: { summary: 'smoke' },
  async init(ctx) {
    const gpu = ctx.gpu;
    const atlas = await getAtlas();
    const tex = gpu.textureFromImage(atlas.canvas, { label: 'atlas' });
    const ntex = gpu.textureFromImage(atlas.normalCanvas, { label: 'atlasN' });
    const shapes = new ShapeBatch(gpu);
    const sprites = new SpriteBatch(gpu, { texture: tex.createView(), filter: 'nearest' });
    const cam = new Camera2D();
    const game = createGameScene(gpu);
    return {
      frame(ctx) {
        cam.setViewport(ctx.width, ctx.height);
        const enc = ctx.encoder;
        const canvas = { view: ctx.target, format: gpu.format };
        if (ctx.example === 'b') {
          const g = game.render(enc, ctx.time, ctx.width, ctx.height);
          gpu.blit(enc, g, canvas);
          return;
        }
        shapes.begin(); sprites.begin();
        shapes.circle(100, 100, 40, '#ff8800', { glow: 8 });
        shapes.line(50, 200, 300, 260, 8, [0.3, 0.8, 1, 1]);
        shapes.rect(320, 60, 120, 80, '#44ff88', { radius: 16, rotation: ctx.time });
        shapes.triangle(480, 200, 560, 60, 600, 220, '#ff44aa', { stroke: 4 });
        shapes.flush(enc, canvas, cam, { clear: [0.05, 0.05, 0.08, 1] });
        const names = atlas.names();
        names.forEach((n, i) => {
          const f = atlas.frames[n];
          sprites.draw(40 + (i % 16) * 36, 300 + Math.floor(i / 16) * 40, f.w * 2, f.h * 2, { uv: atlas.uv(n) });
        });
        const run = atlas.anim('hero_run');
        sprites.draw(ctx.width - 80, 80, 96, 96, { uv: atlas.uv(run[Math.floor(ctx.time * 10) % run.length]) });
        sprites.flush(enc, canvas, cam);
        shapes.circle(200, 120, 30, '#ffffff', { alpha: 0.5 });
        shapes.flush(enc, canvas, cam, { blend: 'additive' });
        sprites.draw(ctx.width - 200, 80, 128, 128, { uv: [0, 0, 1, 1] });
        sprites.flush(enc, canvas, cam, { texture: ntex.createView() });
      },
    };
  },
};

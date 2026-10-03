// Small helpers shared by the simulation scenes (boids, fluid-sim, reaction-diffusion,
// game-of-life, falling-sand, physarum).

/** Add a live readout chip to the scene overlay (kept clear of the hint chip / toolbar). */
export function readout(ctx, css = 'right:8px;bottom:8px') {
  const d = document.createElement('div');
  d.className = 'tag';
  d.style.cssText = css;
  ctx.overlay.append(d);
  return d;
}

/**
 * Record several compute dispatches into ONE compute pass (cheaper than a pass per dispatch).
 * In WebGPU every dispatch is its own synchronisation scope, so a dispatch always sees the
 * storage writes of the previous one, even inside the same pass.
 */
export function computeSeq(encoder, label = 'sim') {
  const pass = encoder.beginComputePass({ label });
  return {
    run(prog, entry, groups, resources) {
      pass.setPipeline(prog.computePipeline(entry));
      pass.setBindGroup(0, prog.bind(resources));
      const g = Array.isArray(groups) ? groups : [groups];
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(g[0])), Math.max(1, Math.ceil(g[1] || 1)), Math.max(1, Math.ceil(g[2] || 1)));
      return this;
    },
    end() {
      pass.end();
    },
  };
}

/**
 * Keep a slider in the control panel in sync after the scene changed its value itself
 * (e.g. a "preset" select that sets several sliders). The framework has no API for this yet,
 * so we update the DOM widget if it exists — harmless if it doesn't.
 */
export function syncSlider(controls, key, value) {
  const c = controls.find((x) => x.key === key);
  const input = typeof document !== 'undefined' ? document.getElementById(`ctl-${key}`) : null;
  if (!c || !input) return;
  const log = c.log && c.min > 0;
  const pos = log ? Math.log(value / c.min) / Math.log(c.max / c.min) : (value - c.min) / (c.max - c.min);
  input.value = String(Math.round(Math.min(1, Math.max(0, pos)) * 1000));
  input.style.setProperty('--fill', `${(input.value / 10).toFixed(1)}%`);
  const out = input.closest('.ctl')?.querySelector('output');
  if (out) {
    const d = c.step && c.step < 1 ? (String(c.step).split('.')[1] || '').length : 0;
    out.textContent = c.format ? c.format(value) : (+value).toFixed(Math.min(d, 4));
  }
}

/** Fixed-rate stepper: converts real time into a whole number of simulation steps. */
export function stepper() {
  let acc = 0;
  return {
    /** rate = steps per second; returns how many steps to run now (capped). */
    take(dt, rate, cap = 64) {
      acc += dt * rate;
      const n = Math.min(cap, Math.floor(acc));
      acc -= n;
      if (acc > cap) acc = 0;
      return n;
    },
    reset() {
      acc = 0;
    },
  };
}

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

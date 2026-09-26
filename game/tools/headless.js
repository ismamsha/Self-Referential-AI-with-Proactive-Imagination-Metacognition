// Headless check of the simulation: metacognition on vs off, with the world
// shifting every `shiftEvery` seconds.
// Usage: node game/tools/headless.js [seeds=40] [seconds=600] [shiftEvery=90]
const S = require('../sim.js');
const seeds = +(process.argv[2] || 40), T = +(process.argv[3] || 600), every = +(process.argv[4] || 90);

function run(metacog, seed) {
  const w = new S.World(seed), a = new S.Amoeba(w, { metacog });
  const dt = 1 / 30;
  let next = every, since = Infinity, deaths = 0, dmg = 0, dmgN = 0, repairs = 0;
  const log = a.log.bind(a);
  a.log = (kind, text) => { if (kind === 'repair') repairs++; log(kind, text); };
  for (let t = 0; t < T; t += dt) {
    if (t >= next) { w.shift(); next += every; since = 0; }
    w.step(dt);
    const alive = a.alive;
    a.step(dt);
    if (alive && !a.alive) deaths++;
    since += dt;
    if (since < 45 && a.alive) { dmg += a.damage; dmgN++; }
  }
  return { deaths, food: a.totals.food, bad: a.totals.toxin + a.totals.spoiled, dmg: dmgN ? dmg / dmgN : 0, repairs };
}

console.log(`${seeds} seeds, ${T} s each, world shifts every ${every} s`);
for (const m of [true, false]) {
  const sum = { deaths: 0, food: 0, bad: 0, dmg: 0, repairs: 0 };
  for (let s = 1; s <= seeds; s++) { const r = run(m, s); for (const k in sum) sum[k] += r[k]; }
  const f = k => (sum[k] / seeds).toFixed(2);
  console.log(`metacognition ${m ? 'on ' : 'off'}  deaths ${f('deaths')}  food ${f('food')}  bad bites ${f('bad')}  ` +
    `damage in 45 s after a shift ${f('dmg')}  repairs ${f('repairs')}`);
}

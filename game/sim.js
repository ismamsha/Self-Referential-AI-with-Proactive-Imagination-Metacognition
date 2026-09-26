/*
 * Amoeba — simulation core (no DOM, runs in the browser or in Node).
 *
 * A single-celled organism forages in a dish of particles. Each particle has
 * a visible pattern (colour x shape) and a hidden effect (food, toxin or
 * inert). The organism has to learn which patterns are which, and it does so
 * with the mechanisms from this repository:
 *
 *   Self-state (main.py SelfState)         energy r_t, damage d_t, doubt_t
 *   Proactive imagination (imagine_futures) before each move it predicts its
 *                                          own next state for every option and
 *                                          scores it with the composite utility
 *   Nonlinear anxiety (compute_effective_beta)
 *                                          beta rises near low energy, high
 *                                          damage, and with doubt
 *   Metacognition (update_metacognition)   doubt is a leaky integrator of the
 *                                          gap between imagined and real
 *                                          outcomes; high doubt -> caution mode
 *   Sparse slots + imprint growth (V4)     a new slot is imprinted the first
 *                                          time a novel pattern is tasted
 *   Slot doubt + plasticity gate + REPAIR  per-slot confident-error rate; it
 *   (V4.1)                                 raises the learning rate and resets
 *                                          slots that have gone wrong
 *
 * "Numb" switches interoception off: the organism perceives the outcome it
 * imagined instead of the real one, so its doubt falls to zero while its real
 * prediction error can be large (doubt without feedback goes blind).
 * "Metacognition off" is the `basic` ablation: no doubt signal reaches the
 * controller, so there is no caution, no plasticity gate and no repair.
 */
(function (root) {
  'use strict';

  var COLORS = [
    { name: 'rose', hue: 346 },
    { name: 'amber', hue: 38 },
    { name: 'moss', hue: 110 },
    { name: 'indigo', hue: 232 }
  ];
  var SHAPES = ['round', 'spiky', 'square'];
  var N_PATTERNS = COLORS.length * SHAPES.length;

  // Real effect of each kind on the organism.
  var EFFECTS = {
    food: { e: 0.22, d: -0.02 },
    toxin: { e: -0.06, d: 0.3 },
    inert: { e: 0.02, d: 0.0 }
  };
  var SPOILED = { e: -0.04, d: 0.08 }; // food that went off (feedback noise)

  // Prior for a pattern the organism has never tasted.
  var NOVEL_PRIOR = { e: 0.03, d: 0.04 };

  var P = {
    width: 900,
    height: 620,
    capacity: 16,        // brain slots
    particles: 30,
    speed: 150,          // px/s
    senseRadius: 420,
    drainIdle: 0.006,    // energy/s
    drainMove: 0.016,
    healRest: 0.006,     // damage/s while resting
    decideEvery: 0.35,   // s
    spoilProb: 0.08,
    baseBeta: 0.5,
    cautionOn: 0.3,      // doubt thresholds with hysteresis
    cautionOff: 0.12,
    repairAt: 0.2,       // slot doubt that triggers REPAIR
    exploration: 0.05
  };

  function patternName(p) {
    return COLORS[Math.floor(p / SHAPES.length)].name + ' ' + SHAPES[p % SHAPES.length];
  }
  function patternColor(p) { return COLORS[Math.floor(p / SHAPES.length)]; }
  function patternShape(p) { return SHAPES[p % SHAPES.length]; }

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  // Small seeded RNG so headless runs are reproducible.
  function makeRng(seed) {
    var s = (seed >>> 0) || 1;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >>> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  function shuffle(a, rng) {
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  // Normalised prediction error between an imagined and a real outcome.
  function outcomeError(pred, real) {
    return clamp((Math.abs(pred.e - real.e) / 0.3 + Math.abs(pred.d - real.d) / 0.3) / 2, 0, 1);
  }

  // ------------------------------------------------------------------ World
  function World(seed) {
    this.rng = makeRng(seed || 7);
    this.time = 0;
    this.shifts = 0;
    this.kinds = [];
    this.particles = [];
    this.nextId = 1;
    this.randomizeKinds();
    for (var i = 0; i < P.particles; i++) this.spawn();
  }

  World.prototype.randomizeKinds = function () {
    var kinds = [];
    for (var i = 0; i < N_PATTERNS; i++) kinds.push(i < 5 ? 'food' : i < 9 ? 'toxin' : 'inert');
    this.kinds = shuffle(kinds, this.rng);
  };

  // Reversal (V4 --reversal): two food patterns turn toxic and two toxins
  // become food. The rest of the world stays as it was.
  World.prototype.shift = function () {
    var food = [], toxin = [], i;
    for (i = 0; i < N_PATTERNS; i++) {
      if (this.kinds[i] === 'food') food.push(i);
      if (this.kinds[i] === 'toxin') toxin.push(i);
    }
    shuffle(food, this.rng); shuffle(toxin, this.rng);
    var changed = [];
    for (i = 0; i < 2; i++) {
      this.kinds[food[i]] = 'toxin';
      this.kinds[toxin[i]] = 'food';
      changed.push(food[i], toxin[i]);
    }
    this.shifts++;
    return changed;
  };

  World.prototype.spawn = function (x, y, pat) {
    var rng = this.rng;
    if (pat === undefined || pat === null) {
      // Keep the dish balanced: pick a kind first, then a pattern of that kind.
      var r = rng();
      var want = r < 0.5 ? 'food' : r < 0.85 ? 'toxin' : 'inert';
      var options = [];
      for (var i = 0; i < N_PATTERNS; i++) if (this.kinds[i] === want) options.push(i);
      pat = options[Math.floor(rng() * options.length)];
    }
    var p = {
      id: this.nextId++,
      pat: pat,
      x: x !== undefined ? x : 30 + rng() * (P.width - 60),
      y: y !== undefined ? y : 30 + rng() * (P.height - 60),
      vx: (rng() - 0.5) * 10,
      vy: (rng() - 0.5) * 10,
      spin: rng() * Math.PI * 2,
      r: 9,
      age: 0
    };
    this.particles.push(p);
    return p;
  };

  World.prototype.effectOf = function (pat) {
    var kind = this.kinds[pat];
    if (kind === 'food' && this.rng() < P.spoilProb) return { e: SPOILED.e, d: SPOILED.d, kind: 'spoiled' };
    var eff = EFFECTS[kind];
    return { e: eff.e, d: eff.d, kind: kind };
  };

  World.prototype.remove = function (p) {
    var i = this.particles.indexOf(p);
    if (i >= 0) this.particles.splice(i, 1);
  };

  World.prototype.step = function (dt) {
    this.time += dt;
    var rng = this.rng;
    for (var i = 0; i < this.particles.length; i++) {
      var p = this.particles[i];
      p.age += dt;
      p.vx += (rng() - 0.5) * 16 * dt;
      p.vy += (rng() - 0.5) * 16 * dt;
      p.vx *= 0.99; p.vy *= 0.99;
      p.x += p.vx * dt; p.y += p.vy * dt;
      p.spin += dt * 0.3;
      if (p.x < 14 || p.x > P.width - 14) { p.vx = -p.vx; p.x = clamp(p.x, 14, P.width - 14); }
      if (p.y < 14 || p.y > P.height - 14) { p.vy = -p.vy; p.y = clamp(p.y, 14, P.height - 14); }
    }
    while (this.particles.length < P.particles) this.spawn();
  };

  // ------------------------------------------------------------------ Agent
  function Amoeba(world, opts) {
    opts = opts || {};
    this.world = world;
    this.metacog = opts.metacog !== false;
    this.numb = false;
    this.events = [];
    this.life = 0;
    this.best = 0;
    this.lives = 0;
    this.totals = { food: 0, toxin: 0, inert: 0, spoiled: 0 };
    this.birth();
  }

  Amoeba.prototype.birth = function () {
    this.x = P.width / 2;
    this.y = P.height / 2;
    this.vx = 0; this.vy = 0;
    this.energy = 0.75;          // real r_t
    this.damage = 0.1;           // real d_t
    this.feltEnergy = this.energy; // what the organism believes (differs only when numb)
    this.feltDamage = this.damage;
    this.doubt = 0.0;
    this.beta = P.baseBeta;
    this.caution = false;
    this.slots = [];
    this.target = null;
    this.mode = 'rest';
    this.plan = [];
    this.decideTimer = 0;
    this.age = 0;
    this.alive = true;
    this.deadTimer = 0;
    this.realErr = 0;   // EMA of the true prediction error, felt or not
    this.eaten = { food: 0, toxin: 0, inert: 0, spoiled: 0 };
    this.lives++;
  };

  Amoeba.prototype.log = function (kind, text) {
    this.events.push({ t: this.world.time, kind: kind, text: text });
    if (this.events.length > 60) this.events.shift();
  };

  Amoeba.prototype.slotFor = function (pat) {
    for (var i = 0; i < this.slots.length; i++) if (this.slots[i].pat === pat) return this.slots[i];
    return null;
  };

  // Imprint growth: a new slot tuned to exactly this pattern.
  Amoeba.prototype.imprint = function (pat, outcome) {
    if (this.slots.length >= P.capacity) {
      // Capacity pressure: prune the least-used slot.
      var worst = 0;
      for (var i = 1; i < this.slots.length; i++) if (this.slots[i].use < this.slots[worst].use) worst = i;
      this.log('prune', 'Pruned the slot for ' + patternName(this.slots[worst].pat));
      this.slots.splice(worst, 1);
    }
    var slot = { pat: pat, e: outcome.e, d: outcome.d, n: 1, sd: 0, use: 1, flash: 1, flashKind: 'imprint', born: this.world.time };
    this.slots.push(slot);
    this.log('imprint', 'New slot imprinted for ' + patternName(pat));
    return slot;
  };

  // main.py compute_effective_beta (+ MetacognitiveSelfAgent doubt boost).
  Amoeba.prototype.effectiveBeta = function (r, d) {
    var b = P.baseBeta;
    if (r < 0.3) b += (0.3 - r) * 2.0 * 0.3;
    if (d > 0.6) b += (d - 0.6) * 2.0 * 0.2;
    if (this.metacog && this.doubt > 0.1) b += this.doubt * 3.0 * (this.caution ? 1.5 : 1.0);
    return clamp(b, 0.1, 2.5);
  };

  // Value of an imagined self-state (composite utility, main.py style).
  Amoeba.prototype.stateValue = function (r, d) {
    var v = r - 1.2 * d;
    if (r < 0.35) v -= (0.35 - r) * 2.0;   // resource penalty
    if (d > 0.5) v -= (d - 0.5) * 1.5;     // degradation penalty
    if (r <= 0 || d >= 1) v -= 5;          // imagined death
    return v;
  };

  // What the organism expects from a pattern, and how unsure it is.
  Amoeba.prototype.expect = function (pat) {
    var slot = this.slotFor(pat);
    if (!slot) return { e: NOVEL_PRIOR.e, d: NOVEL_PRIOR.d, unc: 1, novel: true, slot: null };
    var unc = 1 / (slot.n + 1);
    if (this.metacog) unc = clamp(unc + slot.sd * 0.7, 0, 1);
    return { e: slot.e, d: slot.d, unc: unc, novel: false, slot: slot };
  };

  // Proactive imagination: predict the next self-state for every option.
  Amoeba.prototype.imagine = function () {
    var r = this.feltEnergy, d = this.feltDamage;
    var beta = this.effectiveBeta(r, d);
    this.beta = beta;
    var now = this.stateValue(r, d);
    var options = [];
    var parts = this.world.particles;
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      var dx = p.x - this.x, dy = p.y - this.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > P.senseRadius) continue;
      var ex = this.expect(p.pat);
      var travel = (dist / P.speed) * P.drainMove;
      var r2 = r - travel + ex.e;
      var d2 = d + ex.d;
      var harm = Math.max(0.05, Math.max(0, -ex.e) + Math.max(0, ex.d) + (ex.novel ? 0.15 : 0));
      var util = this.stateValue(r2, d2) - now;
      // Anxiety: uncertain outcomes are discounted by beta x uncertainty.
      util -= beta * ex.unc * harm * 0.6;
      // Epistemic foraging: when calm and fed, novelty is worth something.
      if (ex.novel && r > 0.5 && d < 0.5) util += 0.08 * (1 - (this.metacog ? this.doubt : 0));
      options.push({ kind: 'eat', p: p, dist: dist, pe: ex.e, pd: ex.d, unc: ex.unc, novel: ex.novel, util: util, r2: r2, d2: d2, slot: ex.slot });
    }
    // Resting: imagined 2 s of recovery.
    var rr = r - P.drainIdle * 2, rd = Math.max(0, d - P.healRest * 2);
    // Resting finds no food, so hunger makes it look worse.
    var hunger = Math.max(0, 0.6 - r) * 0.3;
    options.push({ kind: 'rest', p: null, dist: 0, pe: rr - r, pd: rd - d, unc: 0, novel: false, util: this.stateValue(rr, rd) - now + 0.005 - hunger, r2: rr, d2: rd });
    return options;
  };

  Amoeba.prototype.decide = function () {
    var options = this.imagine();
    var allowed = options;
    if (this.caution) {
      // Caution mode: only trusted, clearly good slots, or rest. A starving
      // cell cannot afford caution, so hunger lets untried patterns back in.
      var starving = this.feltEnergy < 0.4;
      allowed = options.filter(function (o) {
        if (o.kind === 'rest') return true;
        if (o.novel) return starving;
        return o.slot.sd < 0.25 && o.pe > 0.05 && o.pd < 0.08;
      });
    }
    allowed.sort(function (a, b) { return b.util - a.util; });
    var choice = allowed[0];
    // Doubt-adjusted exploration (main.py: rate * (1 + 2 * doubt)).
    var rate = P.exploration * (1 + 2 * (this.metacog ? this.doubt : 0));
    var eatable = allowed.filter(function (o) { return o.kind === 'eat'; });
    if (eatable.length && this.world.rng() < rate && this.feltEnergy > 0.3) {
      choice = eatable[Math.floor(this.world.rng() * eatable.length)];
      choice.explore = true;
    }
    options.sort(function (a, b) { return b.util - a.util; });
    this.plan = options.slice(0, 4);
    if (this.plan.indexOf(choice) < 0) this.plan.push(choice);
    this.plan.forEach(function (o) { o.chosen = o === choice; o.blocked = allowed.indexOf(o) < 0; });
    this.target = choice.kind === 'eat' ? choice.p : null;
    this.expected = choice;
    this.mode = choice.kind === 'rest' ? 'rest' : choice.explore ? 'explore' : choice.novel ? 'curious' : 'forage';
  };

  Amoeba.prototype.eat = function (p) {
    var world = this.world;
    var real = world.effectOf(p.pat);
    var ex = this.expect(p.pat);
    var pred = { e: ex.e, d: ex.d };
    world.remove(p);
    this.target = null;

    // Reality.
    this.energy = clamp(this.energy + real.e, 0, 1);
    this.damage = clamp(this.damage + real.d, 0, 1);
    this.eaten[real.kind]++;
    this.totals[real.kind]++;

    var realErr = outcomeError(pred, real);
    this.realErr = 0.7 * this.realErr + 0.3 * realErr;

    // Perception: numb organisms feel what they imagined.
    var felt = this.numb ? pred : real;
    this.feltEnergy = this.numb ? clamp(this.feltEnergy + felt.e, 0, 1) : this.energy;
    this.feltDamage = this.numb ? clamp(this.feltDamage + felt.d, 0, 1) : this.damage;
    var err = outcomeError(pred, felt);

    this.lastBite = { x: p.x, y: p.y, pat: p.pat, real: real, pred: pred, err: realErr, felt: err, t: world.time };

    var slot = ex.slot;
    if (!slot) {
      this.imprint(p.pat, felt);
    } else {
      slot.use += 1;
      // Surprise = error weighted by how confident the slot was (V4.1).
      var confidence = slot.n / (slot.n + 1);
      var surprise = err * confidence;
      slot.sd = 0.7 * slot.sd + 0.3 * surprise;
      // Learning rate: running average (grows rigid with experience) ...
      var lr = 1 / (slot.n + 1);
      // ... opened by the plasticity gate when the slot is in doubt.
      if (this.metacog) lr = Math.min(1, lr * Math.exp(4 * slot.sd));
      slot.e += lr * (felt.e - slot.e);
      slot.d += lr * (felt.d - slot.d);
      slot.n += 1;
      if (err > 0.35) { slot.flash = 1; slot.flashKind = 'surprise'; this.log('surprise', 'Surprise: ' + patternName(p.pat) + ' was not what I imagined'); }
      // REPAIR: a confident slot that keeps failing is reset to relearn.
      if (this.metacog && slot.sd > P.repairAt && slot.n >= 3) {
        slot.e = felt.e; slot.d = felt.d; slot.n = 1; slot.sd = 0;
        slot.flash = 1; slot.flashKind = 'repair';
        this.log('repair', 'Repaired the slot for ' + patternName(p.pat));
      }
    }

    this.updateDoubt(err);
  };

  // main.py MetacognitiveSelfAgent.update_metacognition, rescaled.
  Amoeba.prototype.updateDoubt = function (err) {
    var old = this.doubt;
    var proposed = 0.7 * old + 0.3 * err;
    this.doubt = Math.min(old + 0.15, proposed);
    if (err < 0.08) this.doubt *= 0.85;
    this.doubt = clamp(this.doubt, 0, 1);
    if (!this.metacog) { this.caution = false; return; }
    if (!this.caution && this.doubt > P.cautionOn) {
      this.caution = true;
      this.log('caution', 'Doubt is high: entering caution mode');
    } else if (this.caution && this.doubt < P.cautionOff) {
      this.caution = false;
      this.log('calm', 'Predictions match again: leaving caution mode');
    }
  };

  Amoeba.prototype.setNumb = function (on) {
    if (this.numb === on) return;
    this.numb = on;
    if (on) {
      this.log('numb', 'Interoception off: I now feel what I imagine');
    } else {
      // Feeling returns all at once: the gap between belief and reality
      // is one big prediction error.
      var gap = { e: this.feltEnergy, d: this.feltDamage };
      var err = outcomeError(gap, { e: this.energy, d: this.damage });
      this.feltEnergy = this.energy;
      this.feltDamage = this.damage;
      this.log('numb', 'Interoception back: belief was off by ' + Math.round(err * 100) + '%');
      this.updateDoubt(err);
    }
  };

  Amoeba.prototype.step = function (dt) {
    for (var s = 0; s < this.slots.length; s++) this.slots[s].flash = Math.max(0, this.slots[s].flash - dt * 1.2);
    if (!this.alive) {
      this.deadTimer += dt;
      if (this.deadTimer > 2.0) this.birth();
      return;
    }
    this.age += dt;
    this.life = this.age;
    // Metabolism is predictable, so time without surprises calms doubt
    // (in main.py every accurately predicted step shrinks doubt).
    this.doubt *= Math.exp(-dt / 25);
    this.realErr *= Math.exp(-dt / 25);
    if (this.metacog && this.caution && this.doubt < P.cautionOff) {
      this.caution = false;
      this.log('calm', 'Doubt has settled: leaving caution mode');
    }
    if (this.age > this.best) this.best = this.age;

    // Keep the target only while it exists.
    if (this.target && this.world.particles.indexOf(this.target) < 0) { this.target = null; this.decideTimer = P.decideEvery; }

    this.decideTimer += dt;
    if (this.decideTimer >= P.decideEvery) { this.decideTimer = 0; this.decide(); }

    var moving = false;
    if (this.target) {
      var dx = this.target.x - this.x, dy = this.target.y - this.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 16 + this.target.r) {
        this.eat(this.target);
        this.decideTimer = P.decideEvery;
      } else {
        var sp = P.speed * (this.caution ? 0.8 : 1);
        this.vx += ((dx / dist) * sp - this.vx) * Math.min(1, dt * 4);
        this.vy += ((dy / dist) * sp - this.vy) * Math.min(1, dt * 4);
        moving = true;
      }
    }
    if (!moving) { this.vx *= 0.9; this.vy *= 0.9; }
    this.x = clamp(this.x + this.vx * dt, 20, P.width - 20);
    this.y = clamp(this.y + this.vy * dt, 20, P.height - 20);

    var drain = (moving ? P.drainMove : P.drainIdle) * dt;
    var heal = moving ? 0 : P.healRest * dt;
    this.energy = clamp(this.energy - drain, 0, 1);
    this.damage = clamp(this.damage - heal, 0, 1);
    // Metabolism is known, so felt state tracks it too.
    this.feltEnergy = this.numb ? clamp(this.feltEnergy - drain, 0, 1) : this.energy;
    this.feltDamage = this.numb ? clamp(this.feltDamage - heal, 0, 1) : this.damage;

    if (this.energy <= 0 || this.damage >= 1) {
      this.alive = false;
      this.deadTimer = 0;
      this.log('death', (this.energy <= 0 ? 'Starved' : 'Poisoned') + ' after ' + this.age.toFixed(0) + ' s. A new cell starts with an empty brain.');
    }
  };

  var api = {
    World: World,
    Amoeba: Amoeba,
    params: P,
    COLORS: COLORS,
    SHAPES: SHAPES,
    N_PATTERNS: N_PATTERNS,
    patternName: patternName,
    patternColor: patternColor,
    patternShape: patternShape
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AmoebaSim = api;
})(this);

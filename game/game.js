/* Amoeba Mind — rendering and controls. Simulation lives in sim.js. */
(function () {
  'use strict';
  var S = window.AmoebaSim;
  var P = S.params;

  var dish = document.getElementById('dish');
  var ctx = dish.getContext('2d');
  var brain = document.getElementById('brain');
  var bctx = brain.getContext('2d');
  var chart = document.getElementById('chart');
  var cctx = chart.getContext('2d');
  var $ = function (id) { return document.getElementById(id); };

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var state = {
    paused: false,
    speed: 1,
    showImagination: true,
    showTruth: false,
    dropPattern: null,
    seed: Math.floor(Math.random() * 1e9),
    history: [],
    floaters: [],
    histTimer: 0,
    uiTimer: 0,
    lastBite: null,
    lastEventCount: 0,
    t: 0
  };
  var world, cell;

  function newCulture() {
    state.seed = (state.seed * 1103515245 + 12345) >>> 0;
    world = new S.World(state.seed);
    var meta = cell ? cell.metacog : true;
    cell = new S.Amoeba(world, { metacog: meta });
    cell.log('start', 'A fresh culture: ' + S.N_PATTERNS + ' patterns, meanings unknown');
    state.history = [];
    state.floaters = [];
    state.lastBite = null;
    syncButtons();
  }

  // ---------------------------------------------------------------- theme
  var C = {};
  function readTokens() {
    var cs = getComputedStyle(document.documentElement);
    ['ground', 'panel', 'ink', 'muted', 'rule', 'hema', 'good', 'bad', 'warn', 'dish', 'dish-edge',
      'cell', 'membrane', 'nucleus', 'grid-line', 'particle-l'].forEach(function (k) {
      C[k] = cs.getPropertyValue('--' + k).trim();
    });
  }
  readTokens();
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', readTokens);
  }
  new MutationObserver(readTokens).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  function patFill(pat, alpha) {
    var hue = S.patternColor(pat).hue;
    return 'hsla(' + hue + ', 62%, ' + C['particle-l'] + ', ' + (alpha === undefined ? 1 : alpha) + ')';
  }

  // ---------------------------------------------------------------- shapes
  function shapePath(g, shape, x, y, r, spin) {
    g.beginPath();
    if (shape === 'round') {
      g.arc(x, y, r, 0, Math.PI * 2);
    } else if (shape === 'spiky') {
      var n = 7;
      for (var i = 0; i < n * 2; i++) {
        var a = spin + (i / (n * 2)) * Math.PI * 2;
        var rr = i % 2 ? r * 0.5 : r * 1.15;
        var px = x + Math.cos(a) * rr, py = y + Math.sin(a) * rr;
        if (i) g.lineTo(px, py); else g.moveTo(px, py);
      }
      g.closePath();
    } else {
      var s = r * 0.85, k = s * 0.35;
      g.save(); g.translate(x, y); g.rotate(spin);
      g.moveTo(-s + k, -s);
      g.arcTo(s, -s, s, s, k); g.arcTo(s, s, -s, s, k);
      g.arcTo(-s, s, -s, -s, k); g.arcTo(-s, -s, s, -s, k);
      g.closePath();
      g.restore();
    }
  }

  function swatchSvg(pat) {
    var hue = S.patternColor(pat).hue, shape = S.patternShape(pat);
    var fill = 'hsl(' + hue + ',62%,52%)';
    var body;
    if (shape === 'round') body = '<circle cx="10" cy="10" r="7" fill="' + fill + '"/>';
    else if (shape === 'square') body = '<rect x="3.5" y="3.5" width="13" height="13" rx="3.5" fill="' + fill + '"/>';
    else {
      var pts = [];
      for (var i = 0; i < 14; i++) {
        var a = -Math.PI / 2 + (i / 14) * Math.PI * 2, r = i % 2 ? 4 : 9;
        pts.push((10 + Math.cos(a) * r).toFixed(1) + ',' + (10 + Math.sin(a) * r).toFixed(1));
      }
      body = '<polygon points="' + pts.join(' ') + '" fill="' + fill + '"/>';
    }
    return '<svg viewBox="0 0 20 20" aria-hidden="true">' + body + '</svg>';
  }

  // ---------------------------------------------------------------- dropper
  var dropper = $('dropper');
  function buildDropper() {
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'swatch random'; btn.id = 'drop-random';
    btn.textContent = 'Any'; btn.setAttribute('aria-pressed', 'true');
    btn.addEventListener('click', function () { state.dropPattern = null; syncDropper(); });
    dropper.appendChild(btn);
    for (var p = 0; p < S.N_PATTERNS; p++) {
      (function (pat) {
        var b = document.createElement('button');
        b.type = 'button'; b.className = 'swatch'; b.id = 'drop-' + pat;
        b.innerHTML = swatchSvg(pat);
        b.title = S.patternName(pat);
        b.setAttribute('aria-label', 'Drop ' + S.patternName(pat));
        b.setAttribute('aria-pressed', 'false');
        b.addEventListener('click', function () { state.dropPattern = pat; syncDropper(); });
        dropper.appendChild(b);
      })(p);
    }
  }
  function syncDropper() {
    $('drop-random').setAttribute('aria-pressed', String(state.dropPattern === null));
    for (var p = 0; p < S.N_PATTERNS; p++) $('drop-' + p).setAttribute('aria-pressed', String(state.dropPattern === p));
  }

  dish.addEventListener('pointerdown', function (ev) {
    var rect = dish.getBoundingClientRect();
    var x = ((ev.clientX - rect.left) / rect.width) * P.width;
    var y = ((ev.clientY - rect.top) / rect.height) * P.height;
    world.spawn(x, y, state.dropPattern === null ? undefined : state.dropPattern);
    if (world.particles.length > 48) world.particles.shift();
    state.floaters.push({ x: x, y: y, text: 'dropped', color: C.muted, t: 0 });
  });

  // ---------------------------------------------------------------- controls
  function syncButtons() {
    $('btnPause').firstChild.nodeValue = state.paused ? 'Resume ' : 'Pause ';
    $('btnSpeed').firstChild.nodeValue = 'Speed ' + state.speed + '× ';
    $('btnMeta').setAttribute('aria-pressed', String(cell.metacog));
    $('btnNumb').setAttribute('aria-pressed', String(cell.numb));
    $('btnImagine').setAttribute('aria-pressed', String(state.showImagination));
    $('btnTruth').setAttribute('aria-pressed', String(state.showTruth));
  }
  function shiftWorld() {
    var changed = world.shift();
    cell.log('shift', 'The world shifted: ' + changed.map(S.patternName).join(', ') + ' changed meaning');
    state.floaters.push({ x: P.width / 2, y: 40, text: 'The world shifted', color: C.hema, t: 0, big: true });
  }
  var actions = {
    pause: function () { state.paused = !state.paused; },
    speed: function () { state.speed = state.speed === 1 ? 2 : state.speed === 2 ? 4 : 1; },
    shift: shiftWorld,
    meta: function () {
      cell.metacog = !cell.metacog;
      if (!cell.metacog) cell.caution = false;
      cell.log('meta', cell.metacog ? 'Metacognition on: doubt steers behaviour again' : 'Metacognition off: doubt no longer reaches the controller');
    },
    numb: function () { cell.setNumb(!cell.numb); },
    imagine: function () { state.showImagination = !state.showImagination; },
    truth: function () { state.showTruth = !state.showTruth; },
    reset: newCulture
  };
  function act(name) { actions[name](); syncButtons(); renderPanel(); }
  $('btnPause').addEventListener('click', function () { act('pause'); });
  $('btnSpeed').addEventListener('click', function () { act('speed'); });
  $('btnShift').addEventListener('click', function () { act('shift'); });
  $('btnMeta').addEventListener('click', function () { act('meta'); });
  $('btnNumb').addEventListener('click', function () { act('numb'); });
  $('btnImagine').addEventListener('click', function () { act('imagine'); });
  $('btnTruth').addEventListener('click', function () { act('truth'); });
  $('btnReset').addEventListener('click', function () { act('reset'); });
  document.addEventListener('keydown', function (ev) {
    if (ev.target && /INPUT|TEXTAREA|SELECT/.test(ev.target.tagName)) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    var map = { ' ': 'pause', f: 'speed', s: 'shift', m: 'meta', n: 'numb', i: 'imagine', t: 'truth' };
    var name = map[ev.key.toLowerCase()];
    if (!name) return;
    ev.preventDefault();
    act(name);
  });

  // ---------------------------------------------------------------- dish
  function drawDish(time) {
    var W = P.width, H = P.height;
    ctx.fillStyle = C.dish;
    ctx.fillRect(0, 0, W, H);
    // Faint counting grid, like a haemocytometer.
    ctx.strokeStyle = C['grid-line'];
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var x = 0; x <= W; x += 50) { ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); }
    for (var y = 0; y <= H; y += 50) { ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); }
    ctx.stroke();

    // Particles.
    var parts = world.particles;
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      var grow = Math.max(0, Math.min(1, p.age * 3));
      var r = p.r * grow;
      ctx.fillStyle = patFill(p.pat, 0.9);
      shapePath(ctx, S.patternShape(p.pat), p.x, p.y, r, p.spin);
      ctx.fill();
      if (state.showTruth) {
        var kind = world.kinds[p.pat];
        if (kind !== 'inert') {
          ctx.strokeStyle = kind === 'food' ? C.good : C.bad;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(p.x, p.y, r + 6, 0, Math.PI * 2); ctx.stroke();
        }
      }
    }

    if (state.showImagination && cell.alive) drawImagination();
    drawCell(time);
    drawFloaters();

    // Lens vignette.
    var g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.42, W / 2, H / 2, Math.max(W, H) * 0.72);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, 'rgba(0,0,0,0.22)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }

  function fmtDelta(v) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2); }

  function drawImagination() {
    var plan = cell.plan || [];
    ctx.save();
    ctx.font = '500 12px "IBM Plex Mono", ui-monospace, monospace';
    for (var i = 0; i < plan.length; i++) {
      var o = plan[i];
      if (o.kind !== 'eat' || world.particles.indexOf(o.p) < 0) continue;
      var good = o.util > 0;
      var col = o.blocked ? C.muted : good ? C.good : C.bad;
      ctx.globalAlpha = o.chosen ? 0.95 : o.blocked ? 0.35 : 0.6;
      ctx.strokeStyle = col;
      ctx.lineWidth = o.chosen ? 2 : 1.25;
      ctx.setLineDash(o.chosen ? [] : [5, 6]);
      ctx.beginPath(); ctx.moveTo(cell.x, cell.y); ctx.lineTo(o.p.x, o.p.y); ctx.stroke();
      // Ghost of the imagined self at the target.
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.arc(o.p.x, o.p.y, 20, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      var label = o.novel ? '? new' : fmtDelta(o.pe) + ' E';
      if (o.blocked) label += ' ✕';
      var tx = o.p.x + 24, ty = o.p.y - 14;
      if (tx > P.width - 80) tx = o.p.x - 84;
      var w = ctx.measureText(label).width + 10;
      ctx.globalAlpha = o.blocked ? 0.5 : 0.92;
      ctx.fillStyle = C.panel;
      ctx.fillRect(tx - 5, ty - 12, w, 18);
      ctx.fillStyle = col;
      ctx.fillText(label, tx, ty + 1);
    }
    ctx.restore();
  }

  function drawCell(time) {
    var x = cell.x, y = cell.y;
    var alpha = cell.alive ? 1 : Math.max(0, 1 - cell.deadTimer / 1.5);
    if (alpha <= 0) return;
    var speed = Math.sqrt(cell.vx * cell.vx + cell.vy * cell.vy);
    var heading = Math.atan2(cell.vy, cell.vx);
    var base = 17 + cell.energy * 7;
    var n = 32;
    var wob = reduceMotion ? 0 : 1;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    for (var i = 0; i <= n; i++) {
      var a = (i / n) * Math.PI * 2;
      var r = base
        + wob * 2.2 * Math.sin(a * 3 + time * 1.7)
        + wob * 1.6 * Math.sin(a * 5 - time * 2.3);
      // Pseudopod reaching toward where it is heading.
      var along = Math.cos(a - heading);
      if (speed > 10 && along > 0) r += Math.pow(along, 4) * Math.min(14, speed / 10);
      var px = x + Math.cos(a) * r, py = y + Math.sin(a) * r;
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = C.cell;
    ctx.fill();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = cell.numb ? C.muted : cell.alive ? C.membrane : C.bad;
    if (cell.numb) ctx.setLineDash([2, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Nucleus drifts opposite to motion; it reddens with damage.
    var nx = x - Math.cos(heading) * Math.min(5, speed / 25);
    var ny = y - Math.sin(heading) * Math.min(5, speed / 25);
    ctx.beginPath();
    ctx.arc(nx, ny, 6 + cell.damage * 3, 0, Math.PI * 2);
    ctx.fillStyle = cell.damage > 0.5 ? C.bad : C.nucleus;
    ctx.globalAlpha = alpha * 0.85;
    ctx.fill();

    // Caution halo.
    if (cell.caution && cell.alive) {
      var pulse = reduceMotion ? 0 : Math.sin(time * 5) * 2;
      ctx.globalAlpha = alpha * 0.8;
      ctx.strokeStyle = C.warn;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.arc(x, y, base + 12 + pulse, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }
    // Resting: slow concentric ripple.
    if (cell.alive && cell.mode === 'rest' && !reduceMotion) {
      var ph = (time * 0.6) % 1;
      ctx.globalAlpha = (1 - ph) * 0.35;
      ctx.strokeStyle = C.membrane;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, base + 6 + ph * 22, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
  }

  function drawFloaters() {
    ctx.save();
    ctx.textAlign = 'center';
    for (var i = state.floaters.length - 1; i >= 0; i--) {
      var f = state.floaters[i];
      var life = f.big ? 2.8 : 1.8;
      if (f.t > life) { state.floaters.splice(i, 1); continue; }
      ctx.globalAlpha = Math.max(0, 1 - f.t / life);
      ctx.font = f.big ? '700 22px "Bricolage Grotesque", system-ui, sans-serif' : '600 13px "IBM Plex Mono", ui-monospace, monospace';
      ctx.fillStyle = f.color;
      ctx.fillText(f.text, f.x, f.y - f.t * 18);
    }
    ctx.restore();
  }

  function noticeBite() {
    var b = cell.lastBite;
    if (!b || b === state.lastBite) return;
    state.lastBite = b;
    var kind = b.real.kind;
    var color = kind === 'food' ? C.good : kind === 'inert' ? C.muted : C.bad;
    var text = kind === 'food' ? '+energy' : kind === 'inert' ? 'bland' : kind === 'spoiled' ? 'spoiled!' : 'toxin!';
    state.floaters.push({ x: b.x, y: b.y - 12, text: text, color: color, t: 0 });
    if (b.err > 0.35) state.floaters.push({ x: b.x, y: b.y + 18, text: cell.numb ? '(not felt)' : 'surprise', color: C.warn, t: -0.2 });
  }

  // ---------------------------------------------------------------- brain
  function drawBrain() {
    var W = brain.width, H = brain.height;
    var cols = 4, gap = 8, size = (W - gap * (cols - 1)) / cols;
    bctx.clearRect(0, 0, W, H);
    bctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    for (var i = 0; i < P.capacity; i++) {
      var cx = (i % cols) * (size + gap), cy = Math.floor(i / cols) * (size + gap);
      var slot = cell.slots[i];
      roundRect(bctx, cx, cy, size, size, 10);
      if (!slot) {
        bctx.setLineDash([3, 4]);
        bctx.strokeStyle = C.rule; bctx.lineWidth = 1.5; bctx.stroke();
        bctx.setLineDash([]);
        continue;
      }
      var value = slot.e - 1.2 * slot.d; // what the slot expects, as in the utility
      var good = value >= 0;
      var strength = Math.min(1, Math.abs(value) / 0.35);
      bctx.fillStyle = good ? C.good : C.bad;
      bctx.globalAlpha = 0.12 + 0.28 * strength;
      bctx.fill();
      bctx.globalAlpha = 1;
      bctx.strokeStyle = C.rule; bctx.lineWidth = 1; bctx.stroke();

      // Pattern glyph.
      bctx.fillStyle = patFill(slot.pat);
      shapePath(bctx, S.patternShape(slot.pat), cx + size / 2, cy + size / 2 - 4, 11, 0.3);
      bctx.fill();

      // Slot doubt ring.
      var sd = Math.min(1, slot.sd / P.repairAt);
      bctx.strokeStyle = C.rule; bctx.lineWidth = 3;
      bctx.beginPath(); bctx.arc(cx + size / 2, cy + size / 2 - 4, 20, 0, Math.PI * 2); bctx.stroke();
      if (sd > 0.01) {
        bctx.strokeStyle = C.bad;
        bctx.beginPath(); bctx.arc(cx + size / 2, cy + size / 2 - 4, 20, -Math.PI / 2, -Math.PI / 2 + sd * Math.PI * 2); bctx.stroke();
      }

      // Expected energy and sample count.
      bctx.fillStyle = C.ink;
      bctx.textAlign = 'left';
      bctx.fillText(fmtDelta(slot.e), cx + 6, cy + size - 7);
      bctx.textAlign = 'right';
      bctx.fillStyle = C.muted;
      bctx.fillText('n' + slot.n, cx + size - 6, cy + size - 7);

      // Flash on imprint / surprise / repair.
      if (slot.flash > 0) {
        roundRect(bctx, cx + 1, cy + 1, size - 2, size - 2, 9);
        bctx.globalAlpha = slot.flash;
        bctx.lineWidth = 3;
        bctx.strokeStyle = slot.flashKind === 'repair' ? C.hema : slot.flashKind === 'surprise' ? C.warn : C.good;
        bctx.stroke();
        bctx.globalAlpha = 1;
      }
    }
    bctx.textAlign = 'left';
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  }

  // ---------------------------------------------------------------- chart
  function drawChart() {
    var W = chart.width, H = chart.height, padL = 24, padB = 14;
    var plotW = W - padL, plotH = H - padB - 6;
    cctx.clearRect(0, 0, W, H);
    cctx.font = '500 9px "IBM Plex Mono", ui-monospace, monospace';
    cctx.fillStyle = C.muted;
    cctx.strokeStyle = C.rule;
    cctx.lineWidth = 1;
    [0, 0.5, 1].forEach(function (v) {
      var y = 6 + plotH * (1 - v);
      cctx.beginPath(); cctx.moveTo(padL, y + 0.5); cctx.lineTo(W, y + 0.5); cctx.stroke();
      cctx.fillText(v.toFixed(1), 0, y + 3);
    });
    // Caution threshold.
    var ty = 6 + plotH * (1 - P.cautionOn);
    cctx.setLineDash([3, 3]);
    cctx.strokeStyle = C.warn;
    cctx.globalAlpha = 0.6;
    cctx.beginPath(); cctx.moveTo(padL, ty); cctx.lineTo(W, ty); cctx.stroke();
    cctx.setLineDash([]);
    cctx.globalAlpha = 1;
    cctx.fillText('now', W - 18, H - 2);
    cctx.fillText('−60 s', padL, H - 2);

    var hist = state.history;
    if (hist.length < 2) return;
    var t1 = hist[hist.length - 1].t, t0 = t1 - 60;
    function line(key, color, fill) {
      cctx.beginPath();
      var started = false, lastX = 0;
      for (var i = 0; i < hist.length; i++) {
        var h = hist[i];
        if (h.t < t0) continue;
        var x = padL + ((h.t - t0) / 60) * plotW;
        var y = 6 + plotH * (1 - Math.min(1, h[key]));
        if (started) cctx.lineTo(x, y); else { cctx.moveTo(x, y); started = true; }
        lastX = x;
      }
      cctx.strokeStyle = color; cctx.lineWidth = 2; cctx.stroke();
      if (fill) {
        cctx.lineTo(lastX, 6 + plotH);
        var firstX = padL + Math.max(0, (hist.find(function (h) { return h.t >= t0; }).t - t0) / 60) * plotW;
        cctx.lineTo(firstX, 6 + plotH);
        cctx.closePath();
        cctx.globalAlpha = 0.15; cctx.fillStyle = color; cctx.fill(); cctx.globalAlpha = 1;
      }
      var last = hist[hist.length - 1];
      cctx.beginPath();
      cctx.arc(padL + plotW, 6 + plotH * (1 - Math.min(1, last[key])), 3, 0, Math.PI * 2);
      cctx.fillStyle = color; cctx.fill();
    }
    line('real', C.bad, false);
    line('doubt', C.warn, true);
  }

  // ---------------------------------------------------------------- panel
  function setBar(id, v) { $(id).style.width = (Math.max(0, Math.min(1, v)) * 100).toFixed(1) + '%'; }

  function renderPanel() {
    setBar('mEnergy', cell.energy);
    $('vEnergy').textContent = cell.energy.toFixed(2);
    var felt = $('mFelt');
    felt.hidden = !cell.numb;
    if (cell.numb) felt.style.left = 'calc(' + (cell.feltEnergy * 100).toFixed(1) + '% - 1px)';
    setBar('mDamage', cell.damage);
    $('vDamage').textContent = cell.damage.toFixed(2);
    setBar('mDoubt', cell.doubt);
    $('vDoubt').textContent = cell.doubt.toFixed(2);
    $('tCaution').style.left = (P.cautionOn * 100) + '%';
    setBar('mBeta', (cell.beta - 0.1) / 2.4);
    $('vBeta').textContent = cell.beta.toFixed(2);
    $('ageText').textContent = cell.alive ? 'alive ' + cell.age.toFixed(0) + ' s' : 'dead';

    var note;
    if (!cell.metacog) note = 'Metacognition is off: doubt is measured but nothing acts on it.';
    else if (cell.numb) note = 'Numb: the bar marker shows the energy it believes it has.';
    else if (cell.caution) note = '<b>Caution mode.</b> It only eats slots it trusts, or rests.';
    else note = 'The tick on the doubt bar is the caution threshold.';
    $('stateNote').innerHTML = note;

    var chip = $('modeChip'), mode, text;
    if (!cell.alive) { mode = 'dead'; text = 'Dead · regrowing'; }
    else if (cell.numb) { mode = 'numb'; text = 'Numb'; }
    else if (cell.caution) { mode = 'caution'; text = 'Cautious'; }
    else {
      mode = 'ok';
      text = { rest: 'Resting', forage: 'Foraging', curious: 'Curious', explore: 'Exploring' }[cell.mode] || 'Foraging';
    }
    chip.setAttribute('data-mode', mode);
    $('modeText').textContent = text + (state.paused ? ' · paused' : '');

    $('slotCount').textContent = cell.slots.length + ' / ' + P.capacity;
    $('sBest').textContent = cell.best.toFixed(0);
    $('sLives').textContent = cell.lives;
    $('sShifts').textContent = world.shifts;
    $('sBad').textContent = cell.totals.toxin + cell.totals.spoiled;

    if (cell.events.length !== state.lastEventCount || cell.events[cell.events.length - 1] !== state.lastEvent) {
      state.lastEventCount = cell.events.length;
      state.lastEvent = cell.events[cell.events.length - 1];
      var html = '';
      for (var i = cell.events.length - 1; i >= Math.max(0, cell.events.length - 14); i--) {
        var e = cell.events[i];
        html += '<li data-kind="' + e.kind + '"><time>' + e.t.toFixed(0) + ' s</time><span>' + e.text + '</span></li>';
      }
      $('journal').innerHTML = html;
    }
    drawBrain();
    drawChart();
  }

  // ---------------------------------------------------------------- loop
  var last = performance.now();
  function frame(now) {
    var real = Math.max(0, Math.min(0.1, (now - last) / 1000));
    last = now;
    if (!state.paused) {
      var steps = state.speed * 2;
      var dt = real / 2;
      for (var s = 0; s < steps; s++) {
        world.step(dt);
        cell.step(dt);
        noticeBite();
        state.histTimer += dt;
        if (state.histTimer > 0.25) {
          state.histTimer = 0;
          state.history.push({ t: world.time, doubt: cell.doubt, real: cell.realErr });
          if (state.history.length > 400) state.history.shift();
        }
      }
      state.t += real * state.speed;
      for (var i = 0; i < state.floaters.length; i++) state.floaters[i].t += real;
    }
    drawDish(state.t);
    state.uiTimer += real;
    if (state.uiTimer > 0.1) { state.uiTimer = 0; renderPanel(); }
    requestAnimationFrame(frame);
  }

  buildDropper();
  newCulture();
  renderPanel();
  requestAnimationFrame(frame);

  // Exposed for screenshots and debugging.
  window.amoebaMind = { get world() { return world; }, get cell() { return cell; }, state: state, act: act };
})();

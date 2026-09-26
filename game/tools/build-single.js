// Inline style.css, sim.js and game.js into one body-only HTML fragment
// (used to publish a hosted copy). Usage: node game/tools/build-single.js out.html
const fs = require('fs'), path = require('path');
const dir = path.join(__dirname, '..');
const read = f => fs.readFileSync(path.join(dir, f), 'utf8');
const html = read('index.html');
const body = html.split('<!-- BODY START -->')[1].split('<!-- BODY END -->')[0];
const fonts = html.match(/<link rel="stylesheet" href="https:\/\/fonts[^>]+>/)[0];
const out = `<title>Amoeba Mind</title>\n${fonts}\n<style>\n${read('style.css')}</style>\n${body}\n<script>\n${read('sim.js')}</script>\n<script>\n${read('game.js')}</script>\n`;
fs.writeFileSync(process.argv[2] || path.join(dir, 'amoeba-mind.html'), out);

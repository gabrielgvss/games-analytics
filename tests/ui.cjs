const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {this.tag = tag; this.children = []; this.listeners = {}; this._value = ''; this.hidden = false; this.style = {}; this.textContent = ''; this.attributes = {};}
  get value() {return this._value;}
  set value(value) {this._value = String(value);}
  append(...items) {this.children.push(...items);}
  replaceChildren(...items) {this.children = [...items];}
  addEventListener(type, callback) {this.listeners[type] = callback;}
  setAttribute(key, value) {this.attributes[key] = value;}
  querySelector(selector) {return this.querySelectorAll(selector)[0] ?? null;}
  querySelectorAll(selector) {return this.children.filter(item => item.tag === selector);}
  click() {this.listeners.click?.({target:this});}
  focus() {document.activeElement = this;}
  blur() {document.activeElement = null;}
}

const elements = new Map();
const document = {
  activeElement: null,
  getElementById(id) {if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id);},
  createElement(tag) {return new Element(tag);},
  createTextNode(value) {const node = new Element('#text'); node.textContent = value; return node;},
};
const root = path.resolve(__dirname, '..');
const payload = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'results.json'), 'utf8'));
const source = fs.readFileSync(path.join(root, 'web', 'app.js'), 'utf8');
vm.runInNewContext(source, {PAYLOAD:payload, document, setTimeout:callback=>callback()});

assert.equal(elements.get('catalog-count').textContent, payload.report.catalog.length.toLocaleString('pt-BR'));
assert.match(elements.get('selected-title').textContent, /Need for Speed Carbon/);
assert.notEqual(elements.get('prediction').textContent, '—');
assert.equal(elements.get('comparison').hidden, false);
assert.equal(elements.get('quick-games').children.length, 3);

const search = elements.get('game-search');
search.value = 'Wii Sports';
search.listeners.input({target:search});
const results = elements.get('search-results');
assert.equal(results.hidden, false);
assert.ok(results.children.length >= 2);
results.children[0].click();
assert.match(elements.get('selected-title').textContent, /^Wii Sports · Wii$/);
assert.equal(results.hidden, true);
assert.equal(elements.get('comparison').hidden, false);

elements.get('Critic_Score').value = '100';
elements.get('game-form').listeners.input({target:elements.get('Critic_Score')});
assert.equal(elements.get('selected-title').textContent, 'Wii Sports · cenário alterado');
assert.equal(elements.get('comparison').hidden, false);
assert.ok(elements.get('drivers').children.length > 0);

search.value = 'jogo que nao existe xyz';
search.listeners.input({target:search});
assert.match(results.children[0].textContent, /Nenhum jogo/);
process.stdout.write('Busca, seleção e atualização dinâmica: OK\n');

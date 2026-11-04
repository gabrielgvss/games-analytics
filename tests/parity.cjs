const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'web', 'app.js'), 'utf8').split('function readForm')[0];
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {PAYLOAD: {report: {}, model: input.model}, document: {}};
vm.runInNewContext(`${source}\nglobalThis.portablePredict = predict;`, context);
process.stdout.write(JSON.stringify(input.records.map(record => context.portablePredict(record))));

"use strict";
const report = PAYLOAD.report;
const model = PAYLOAD.model;
const numericFields = Object.keys(model.numeric);
const categoryFields = Object.keys(model.categorical);
const fields = [...numericFields, ...categoryFields];
const pt = (value, digits = 2) => Number(value).toLocaleString("pt-BR", {minimumFractionDigits: digits, maximumFractionDigits: digits});
const byId = id => document.getElementById(id);
const labels = {Year_of_Release:"Ano",Critic_Score:"Nota da crítica",Critic_Count:"Quantidade de críticas",User_Score:"Nota do público",User_Count:"Quantidade de avaliações",Platform:"Plataforma",Genre:"Gênero",Rating:"Classificação",Publisher:"Empresa",Developer:"Estúdio"};

function predict(record) {
  const contributions = {};
  let raw = model.baseline;
  for (const field of numericFields) {
    const spec = model.numeric[field];
    const value = record[field];
    let effect;
    if (value === null || value === undefined || value === "") effect = spec.missing;
    else {
      const current = Number(value);
      effect = spec.slope * (spec.log ? Math.log1p(current) - Math.log1p(spec.reference) : current - spec.reference);
    }
    contributions[field] = effect;
    raw += effect;
  }
  for (const field of categoryFields) {
    const mapping = model.categorical[field];
    const value = record[field];
    const key = value === null || value === undefined || value === "" ? "__MISSING__" : String(value);
    const effect = Object.prototype.hasOwnProperty.call(mapping, key) ? mapping[key] : mapping.__UNKNOWN__;
    contributions[field] = effect;
    raw += effect;
  }
  return {raw, clipped: Math.max(0, raw), contributions};
}

function readForm() {
  const record = {};
  for (const field of fields) {
    const value = byId(field).value.trim();
    record[field] = value === "" ? null : numericFields.includes(field) ? Number(value) : value;
  }
  return record;
}

function fillForm(record) {
  for (const field of fields) byId(field).value = record[field] ?? "";
}

function validate(record) {
  for (const field of numericFields) {
    const value = record[field];
    if (value === null) continue;
    if (!Number.isFinite(value)) return `Confira o campo “${labels[field]}”.`;
    if (field.endsWith("_Count") && (!Number.isInteger(value) || value < 0)) return `${labels[field]} precisa ser uma contagem igual ou maior que zero.`;
  }
  if (record.Critic_Score !== null && (record.Critic_Score < 0 || record.Critic_Score > 100)) return "A nota da crítica vai de 0 a 100.";
  if (record.User_Score !== null && (record.User_Score < 0 || record.User_Score > 10)) return "A nota do público vai de 0 a 10.";
  const year = record.Year_of_Release;
  if (year !== null && (!Number.isInteger(year) || year < 1970 || year > 2035)) return "Informe um ano entre 1970 e 2035.";
  return "";
}

const catalog = report.catalog;
let activeCatalogIndex = null;
let sourceCatalogIndex = null;
let hasEditedScenario = false;
const normalized = text => String(text ?? "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("pt-BR");
const searchable = catalog.map(game => normalized(`${game.name} ${game.platform} ${game.year ?? ""}`));

function hideSearch() {
  byId("search-results").hidden = true;
  byId("game-search").setAttribute("aria-expanded", "false");
}

function showSearch(query) {
  const list = byId("search-results");
  list.replaceChildren();
  const phrase = normalized(query).trim();
  const words = phrase.split(/\s+/).filter(Boolean);
  if (!words.length) {hideSearch(); return;}
  const matches = [];
  for (let i = 0; i < catalog.length; i++) {
    if (!words.every(word => searchable[i].includes(word))) continue;
    const name = normalized(catalog[i].name);
    const rank = name === phrase ? 0 : name.startsWith(phrase) ? 1 : name.includes(phrase) ? 2 : 3;
    matches.push({index:i, rank, length:name.length});
  }
  matches.sort((a,b) => a.rank-b.rank || a.length-b.length || a.index-b.index);
  matches.length = Math.min(matches.length, 9);
  if (!matches.length) {
    const empty = document.createElement("p"); empty.className = "empty";
    empty.textContent = "Nenhum jogo encontrado. Tente outro título ou plataforma.";
    list.append(empty);
  }
  for (const {index} of matches) {
    const game = catalog[index];
    const button = document.createElement("button"); button.type = "button"; button.setAttribute("role", "option");
    const title = document.createElement("strong"); title.textContent = game.name;
    const detail = document.createElement("small"); detail.textContent = `${game.platform} · ${game.year ?? "ano não informado"}`;
    button.append(title, detail);
    button.addEventListener("click", () => selectGame(index));
    list.append(button);
  }
  list.hidden = false;
  byId("game-search").setAttribute("aria-expanded", "true");
}

function selectGame(index) {
  const game = catalog[index];
  activeCatalogIndex = index;
  sourceCatalogIndex = index;
  hasEditedScenario = false;
  fillForm(game.record);
  byId("game-search").value = `${game.name} · ${game.platform}`;
  hideSearch();
  update();
}

function showDrivers(effects, scenario = false) {
  const container = byId("drivers");
  container.replaceChildren();
  const biggest = Object.entries(effects).filter(([,value]) => Math.abs(value) >= .005).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0,3);
  if (!biggest.length) {
    const empty = document.createElement("p"); empty.textContent = scenario ? "Essa mudança quase não alterou a estimativa." : "Escolha um jogo para ver o que mais pesou na estimativa."; container.append(empty); return;
  }
  for (const [field, effect] of biggest) {
    const row = document.createElement("div"); row.className = "driver";
    const name = document.createElement("strong"); name.textContent = labels[field];
    const value = document.createElement("span"); value.className = effect >= 0 ? "up" : "down"; value.textContent = `${effect >= 0 ? "+" : "−"}${pt(Math.abs(effect))} mi`;
    const description = document.createElement("small"); description.textContent = scenario ? (effect >= 0 ? "A mudança elevou o palpite" : "A mudança reduziu o palpite") : (effect >= 0 ? "Elevou o palpite frente a um jogo típico" : "Reduziu o palpite frente a um jogo típico");
    row.append(name, value, description); container.append(row);
  }
}

function update() {
  const record = readForm();
  const game = activeCatalogIndex === null ? null : catalog[activeCatalogIndex];
  const sourceGame = hasEditedScenario && sourceCatalogIndex !== null ? catalog[sourceCatalogIndex] : null;
  const comparisonGame = game || sourceGame;
  const filled = fields.some(field => record[field] !== null);
  const warning = byId("result-warning");
  if (!filled) {
    byId("selected-title").textContent = "Escolha um jogo";
    byId("prediction").textContent = "—";
    byId("result-summary").textContent = "Busque um título acima ou abra os controles para criar seu cenário.";
    byId("comparison").hidden = true; warning.hidden = true; byId("scenario-story").hidden = true; byId("drivers").replaceChildren(); return;
  }
  const invalid = validate(record);
  if (invalid) {
    byId("prediction").textContent = "—";
    byId("result-summary").textContent = "Ajuste os dados para continuar.";
    byId("comparison").hidden = true; warning.textContent = invalid; warning.hidden = false; byId("scenario-story").hidden = true; byId("drivers").replaceChildren(); return;
  }
  const result = predict(record);
  byId("selected-title").textContent = game ? `${game.name} · ${game.platform}` : sourceGame ? `${sourceGame.name} · cenário alterado` : "Seu cenário";
  byId("prediction").textContent = pt(result.clipped);
  const scenarioStory = byId("scenario-story");
  if (sourceGame) {
    const original = predict(sourceGame.record);
    const changes = fields.filter(field => {
      const before = sourceGame.record[field] ?? null;
      const after = record[field] ?? null;
      return numericFields.includes(field) && before !== null && after !== null ? Number(before) !== Number(after) : before !== after;
    });
    const deltas = Object.fromEntries(fields.map(field => [field, result.contributions[field] - original.contributions[field]]));
    showDrivers(deltas, true);
    byId("explain-step").textContent = "02 / IMPACTO DA MUDANÇA";
    byId("explain-title").textContent = "O que mudou no palpite?";
    byId("explain-intro").textContent = "Cada linha mostra quanto uma alteração deslocou a estimativa, mantendo os outros dados do jogo.";
    const change = result.clipped - original.clipped;
    scenarioStory.textContent = changes.length
      ? `Você mudou ${changes.map(field => labels[field]).join(", ")}. O palpite passou de ${pt(original.clipped)} para ${pt(result.clipped)} milhões${change === 0 ? "." : ` (${change > 0 ? "+" : "−"}${pt(Math.abs(change))} milhões).`}`
      : "Os dados continuam iguais aos do jogo selecionado. Mude uma variável abaixo para testar uma hipótese.";
    scenarioStory.hidden = false;
  } else {
    byId("explain-step").textContent = "01 / POR TRÁS DO PALPITE";
    byId("explain-title").textContent = "O que mais pesou?";
    byId("explain-intro").textContent = "Veja quais informações mais afastam a estimativa de um jogo típico.";
    showDrivers(result.contributions);
    scenarioStory.hidden = true;
  }
  if (comparisonGame) {
    const delta = result.clipped - comparisonGame.actual;
    const direction = delta >= 0 ? "acima" : "abaixo";
    byId("result-summary").textContent = `Neste jogo, a estimativa ficou ${pt(Math.abs(delta))} milhão${Math.abs(delta) === 1 ? "" : "ões"} ${direction} do que aconteceu de verdade.`;
    const max = Math.max(comparisonGame.actual, result.clipped, .01);
    byId("pred-bar").style.width = `${Math.max(1, result.clipped / max * 100)}%`;
    byId("actual-bar").style.width = `${Math.max(1, comparisonGame.actual / max * 100)}%`;
    byId("pred-caption").textContent = sourceGame ? "NOVO PALPITE" : "ESTIMATIVA";
    byId("pred-label").textContent = `${pt(result.clipped)} mi`;
    byId("actual-label").textContent = `${pt(comparisonGame.actual)} mi`;
    byId("comparison").hidden = false;
  } else {
    byId("result-summary").textContent = "Uma referência para comparar cenários parecidos — não uma previsão segura de vendas futuras.";
    byId("comparison").hidden = true;
  }
  const outOfContext = [];
  const changedFields = fields.filter(field => {
    const before = sourceGame?.record[field] ?? null;
    const after = record[field] ?? null;
    return sourceGame ? (numericFields.includes(field) && before !== null && after !== null ? Number(before) !== Number(after) : before !== after) : after !== null;
  });
  for (const field of changedFields) {
    const value = record[field];
    if (numericFields.includes(field) && value !== null) {
      const range = model.context.numeric_ranges[field];
      if (value < range.min || value > range.max) outOfContext.push(`${labels[field]} (${value}) está fora dos valores vistos, de ${pt(range.min,0)} a ${pt(range.max,0)}.`);
    } else if (categoryFields.includes(field) && value !== null && !Object.prototype.hasOwnProperty.call(model.categorical[field], String(value))) {
      outOfContext.push(`${labels[field]} “${value}” não aparece nos jogos usados na comparação.`);
    }
  }
  warning.textContent = outOfContext.length ? `${outOfContext.join(" ")} Essa parte do resultado é uma extrapolação e merece cautela.` : "";
  warning.hidden = outOfContext.length === 0;
}

function init() {
  byId("catalog-count").textContent = catalog.length.toLocaleString("pt-BR");
  const small = report.deciles[0];
  byId("small-games-copy").textContent = `Entre os jogos que menos venderam, a média real foi de cerca de ${Math.round(small.real * 1000)} mil unidades. A estimativa média foi de ${Math.round(small.previsto * 1000)} mil. Uma diferença pequena em milhões, mas enorme para um jogo desse tamanho.`;
  byId("hit-copy").textContent = `Wii Sports vendeu ${pt(report.wii.actual)} milhões de unidades. A conta estimou ${pt(report.wii.predicted)} milhões. O modelo não enxergou o fenômeno que ele se tornou.`;
  byId("method-copy").textContent = `Nesse grupo guardado, a distância média entre a estimativa e a venda real foi de ${pt(report.final_raw.MAE)} milhão de unidades. Quando damos mais peso aos grandes erros, a medida sobe para ${pt(report.final_raw.RMSE)} milhão. O modelo captou cerca de ${Math.round(report.final_raw.R2 * 100)}% da variação das vendas; muita coisa continua sem explicação nos dados disponíveis.`;
  for (const field of categoryFields) {
    const list = byId(`list-${field}`);
    for (const value of model.options[field]) {const option = document.createElement("option"); option.value = value; list.append(option);}
  }
  const suggestions = [
    ["Need for Speed Carbon", "PS3", "Jogo comum"],
    ["Wii Sports", "Wii", "Grande sucesso"],
    ["2002 FIFA World Cup", "XB", "Vendas baixas"],
  ];
  for (const [name,platform,label] of suggestions) {
    const index = catalog.findIndex(game => game.name === name && game.platform === platform);
    if (index < 0) continue;
    const button = document.createElement("button"); button.type = "button";
    const bold = document.createElement("b"); bold.textContent = `${label}: `;
    button.append(bold, document.createTextNode(name));
    button.addEventListener("click", () => selectGame(index));
    byId("quick-games").append(button);
  }
  const first = catalog.findIndex(game => game.name === "Need for Speed Carbon" && game.platform === "PS3");
  if (first >= 0) selectGame(first);
}

byId("game-search").addEventListener("input", event => showSearch(event.target.value));
byId("game-search").addEventListener("keydown", event => {
  if (event.key === "Escape") {hideSearch(); byId("game-search").blur();}
  if (event.key === "Enter" && !byId("search-results").hidden) {const first = byId("search-results").querySelector("button"); if (first) {event.preventDefault(); first.click();}}
  if (event.key === "ArrowDown" && !byId("search-results").hidden) {const first = byId("search-results").querySelector("button"); if (first) {event.preventDefault(); first.focus();}}
});
byId("search-results").addEventListener("keydown", event => {
  if (event.key === "Escape") {hideSearch(); byId("game-search").focus();}
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    const buttons = [...byId("search-results").querySelectorAll("button")];
    const next = buttons.indexOf(document.activeElement) + (event.key === "ArrowDown" ? 1 : -1);
    if (buttons[next]) {event.preventDefault(); buttons[next].focus();}
  }
});
byId("game-form").addEventListener("input", () => {activeCatalogIndex = null; hasEditedScenario = true; update();});
byId("game-form").addEventListener("reset", () => {activeCatalogIndex = null; sourceCatalogIndex = null; hasEditedScenario = false; byId("game-search").value = ""; setTimeout(update, 0);});
init();

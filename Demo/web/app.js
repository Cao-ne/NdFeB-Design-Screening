const form = document.querySelector("#predictForm");
const formulaInput = document.querySelector("#formulaInput");
const predictButton = document.querySelector("#predictButton");
const statusBox = document.querySelector("#statusBox");
const statusText = document.querySelector("#statusText");
const formulaEcho = document.querySelector("#formulaEcho");
const domainWarning = document.querySelector("#domainWarning");
const modelMeta = document.querySelector("#modelMeta");
const compareBody = document.querySelector("#compareBody");
const compareNote = document.querySelector("#compareNote");
const buildVersion = document.querySelector("#buildVersion");
const exampleButtons = [...document.querySelectorAll(".example-chip")];
const cards = new Map(
  [...document.querySelectorAll(".result-card")].map((card) => [card.dataset.model, card])
);

const numberFormat = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 1,
});

function setBusy(isBusy) {
  predictButton.disabled = isBusy;
  predictButton.classList.toggle("is-busy", isBusy);
  predictButton.querySelector(".button-label").textContent = isBusy ? "计算中" : "开始预测";
  exampleButtons.forEach((button) => {
    button.disabled = isBusy;
  });
  cards.forEach((card) => {
    card.classList.toggle("is-loading", isBusy);
  });
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusBox.classList.toggle("is-error", isError);
}

function formatValue(value) {
  return numberFormat.format(value);
}

function updateBound(card, result) {
  const ha = card.querySelector('[data-field="ha"]');
  const me = card.querySelector('[data-field="me"]');
  const bounded = card.querySelector('[data-field="bounded"]');
  const note = card.querySelector('[data-field="meNote"]');

  bounded.textContent = formatValue(result.pred_bounded);
  if (result.ha_kA_m == null || result.me == null) {
    ha.textContent = "—";
    me.textContent = "—";
    note.textContent = "数据库中暂无该配方 Ha，未做上界校验。";
    note.classList.remove("is-out");
    return;
  }

  ha.textContent = formatValue(result.ha_kA_m);
  me.textContent = Number(result.me).toFixed(3);
  if (result.me_status === "violates_upper_bound") {
    note.textContent = "预测值超过 Ha 上界，已压到上界；ME > 1，需人工复核。";
    note.classList.add("is-out");
  } else if (result.me_status === "within_bound") {
    note.textContent = "Hcj 低于 Ha 上界，ME 落在 (0,1) 内。";
    note.classList.remove("is-out");
  } else {
    note.textContent = "ME 不在 (0,1) 内，需人工复核。";
    note.classList.add("is-out");
  }
}

function updateCard(card, result) {
  card.querySelector('[data-field="value"]').textContent = formatValue(result.value);
  card.querySelector('[data-field="unit"]').textContent = result.unit;
  card.querySelector('[data-field="interval"]').textContent =
    `${formatValue(result.lower_80)} – ${formatValue(result.upper_80)}`;

  const domain = card.querySelector('[data-field="domain"]');
  domain.textContent = result.in_domain ? "适用域内" : "适用域外";
  domain.classList.remove("is-pending", "is-in", "is-out");
  domain.classList.add(result.in_domain ? "is-in" : "is-out");

  const span = Math.max(result.upper_80 - result.lower_80, Number.EPSILON);
  const point = ((result.value - result.lower_80) / span) * 100;
  const pointPosition = Math.min(100, Math.max(0, point));
  card.querySelector('[data-field="marker"]').style.left = `${pointPosition}%`;
  card.setAttribute(
    "aria-label",
    `${result.name}预测为 ${formatValue(result.value)} ${result.unit}，` +
      `80% 区间 ${formatValue(result.lower_80)} 至 ${formatValue(result.upper_80)} ${result.unit}，` +
      `${result.in_domain ? "在适用域内" : "在适用域外"}`
  );

  if (result.key === "Hcj") {
    updateBound(card, result);
  }
}

function updateMetadata(results) {
  const versions = results
    .map((result) => `${result.key} ${result.data_version || "未知"}`)
    .join(" / ");
  const latestTime = results
    .map((result) => result.model_created_at)
    .filter(Boolean)
    .sort()
    .at(-1);
  const timeText = latestTime
    ? new Date(latestTime).toLocaleString("zh-CN", { hour12: false })
    : "未知";
  modelMeta.textContent = `模型生成 ${timeText} · 数据版本 ${versions}`;
}

function relativeError(predicted, actual) {
  if (!Number.isFinite(actual) || actual === 0) return null;
  return (Math.abs(predicted - actual) / actual) * 100;
}

function renderCompare(results) {
  const byKey = new Map(results.map((result) => [result.key, result]));
  const ordered = ["Ms", "Tc", "Hcj"].map((key) => byKey.get(key)).filter(Boolean);
  if (!ordered.length) return;

  const makeRow = (label, fill) => {
    const row = document.createElement("tr");
    const head = document.createElement("th");
    head.scope = "row";
    head.textContent = label;
    row.append(head);
    ordered.forEach((result) => {
      const cell = document.createElement("td");
      fill(cell, result);
      row.append(cell);
    });
    return row;
  };

  const rows = [
    makeRow("预测值", (cell, result) => {
      cell.textContent = formatValue(result.pred_bounded ?? result.value);
    }),
    makeRow("最近配方", (cell, result) => {
      if (!result.nearest) {
        cell.textContent = "—";
        return;
      }
      const name = document.createElement("span");
      name.textContent = result.nearest.formula;
      const meta = document.createElement("small");
      meta.textContent = result.nearest.exact
        ? `同一配方 · n=${result.nearest.n_records}`
        : `d=${Number(result.nearest.distance).toFixed(3)} · n=${result.nearest.n_records}`;
      cell.append(name, meta);
      cell.classList.add("formula-cell");
    }),
    makeRow("数据库实测", (cell, result) => {
      cell.textContent = result.nearest ? formatValue(result.nearest.value) : "—";
    }),
    makeRow("相对误差 / %", (cell, result) => {
      const nearest = result.nearest;
      const predicted = result.pred_bounded ?? result.value;
      const error = nearest ? relativeError(predicted, nearest.value) : null;
      if (error == null) {
        cell.textContent = "—";
        return;
      }
      cell.textContent = `${nearest.exact ? "" : "≈"}${formatValue(error)}`;
      if (!nearest.exact) {
        cell.classList.add("is-approx");
        cell.title = "最近配方与输入配方不同，误差包含成分差异，仅供参考";
      }
    }),
  ];

  compareBody.replaceChildren(...rows);

  const exact = ordered.filter((result) => result.nearest?.exact).length;
  const method = "实测取库中同配方中位数，相对误差 = |预测 − 实测| / 实测。";
  if (exact === ordered.length) {
    compareNote.textContent =
      "三项都命中同一配方，相对误差可直接反映模型偏差；Hcj 仍受晶界与工艺影响，误差通常最大。" + method;
  } else {
    compareNote.textContent =
      `${exact}/${ordered.length} 项命中同一配方；带 ≈ 的是近似配方（d>0），` +
      "误差包含成分差异，只作量级参照。Hcj 还受晶界与工艺影响。" +
      method;
  }
}

async function predict(formula) {
  const cleanFormula = formula.trim();
  if (!cleanFormula) {
    setStatus("请输入化学式。", true);
    formulaInput.focus();
    return;
  }

  setBusy(true);
  setStatus("正在计算三项性能…");
  domainWarning.hidden = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ formula: cleanFormula }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `请求失败（${response.status}）`);
    }

    formulaEcho.textContent = payload.formula;
    payload.results.forEach((result) => {
      const card = cards.get(result.key);
      if (card) updateCard(card, result);
    });
    updateMetadata(payload.results);
    renderCompare(payload.results);
    domainWarning.hidden = payload.all_in_domain;
    setStatus(
      payload.all_in_domain
        ? "预测完成；三项结果均在当前适用域内。"
        : "预测完成；存在适用域外结果，请谨慎解释。"
    );
  } catch (error) {
    setStatus(error.message || "无法完成预测，请检查本地服务。", true);
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict(formulaInput.value);
});

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    formulaInput.value = button.dataset.formula;
    predict(formulaInput.value);
  });
});

window.addEventListener("DOMContentLoaded", () => {
  loadBuildVersion();
  predict(formulaInput.value);
});

async function loadBuildVersion() {
  if (!buildVersion) return;
  try {
    const response = await fetch("/api/version");
    if (!response.ok) throw new Error("版本接口不可用");
    const info = await response.json();
    const dirtyMark = info.dirty ? " · 有未提交改动" : "";
    buildVersion.textContent = `版本 ${info.version}${dirtyMark}`;
    buildVersion.title = `提交 ${info.commit} · ${info.commit_date || "日期未知"}`;
  } catch {
    buildVersion.textContent = "版本信息不可用";
  }
}

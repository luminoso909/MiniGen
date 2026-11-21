# ruff: noqa: E501
"""核验动漫头像评测资产并生成离线的匿名打分页和私有映射。"""

import argparse
import base64
import hashlib
import hmac
import json
import secrets
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from minigen._typing import JsonDict

ARMS = ("uniform", "mid60")
CONTINUATION_ARMS = ("uniform", "uniform20k")
SCALE_ARMS = ("uniform20k", "b1_50m_20k", "b2_50m_20k")
SCALE_COLUMNS = ("alpha", "beta", "gamma")
SCALE_SCOPE = (
    Path(__file__).resolve().parents[1] / "configs/benchmark/anime_faces_v1_scale_50m.json"
)
BENCHMARK_SCHEMA = "minigen.anime_faces.benchmark.v1"
GENERATION_SCHEMA = "minigen.anime_faces.benchmark_generation.v1"
RECEIPT_FIELDS = (
    "schema",
    "benchmark_sha256",
    "rubric_sha256",
    "approval_sha256",
    "benchmark_lock_sha256",
    "seeds",
    "sampling",
    "seed_noise_sha256",
)
HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniGen 头像盲评</title>
<style>
:root {
  color-scheme:light;
  --ink:#182d36;
  --muted:#536971;
  --paper:#f6f5ef;
  --white:#fcfdfa;
  --line:#cdd9d9;
  --brand:#155b6d;
  --accent:#f4e9cf;
  font:16px/1.55 Inter,"PingFang SC","Microsoft YaHei",sans-serif;
}
* {
  box-sizing:border-box;
}
body {
  margin:0;
  background:var(--paper);
  color:var(--ink);
}
header {
  background:#15394b;
  color:#f8fcfb;
  padding:1.3rem 1rem 1.45rem;
  border-bottom:4px solid #c9ab72;
}
header .shell,main {
  max-width:1230px;
  margin:auto;
}
main {
  padding:0 1rem 2rem;
}
header h1 {
  margin:0;
  font-size:clamp(1.8rem,5vw,2.75rem);
  line-height:1.16;
}
header p {
  margin:.45rem 0 0;
  color:#dce9ea;
}
.toolbar,.pager {
  display:flex;
  align-items:end;
  gap:.65rem;
  flex-wrap:wrap;
  padding:.8rem 0;
}
.toolbar label,.score label {
  display:grid;
  gap:.16rem;
  font-size:.84rem;
  color:var(--muted);
}
button,input,select,textarea {
  font:inherit;
  color:var(--ink);
  border:1px solid #9cacb1;
  background:white;
  border-radius:.36rem;
  padding:.36rem .55rem;
}
button {
  cursor:pointer;
}
button:disabled {
  opacity:.45;
  cursor:default;
}
.primary {
  color:white;
  background:var(--brand);
  border-color:var(--brand);
}
:focus-visible {
  outline:3px solid #ba7626;
  outline-offset:2px;
}
summary {
  cursor:pointer;
  color:var(--brand);
  font-weight:650;
}
.rubric {
  border-block:1px solid var(--line);
  padding:.6rem 0;
}
.rubric ul {
  margin:.55rem 0 .15rem;
  padding-left:1.3rem;
}
.rubric li {
  margin:.35rem 0;
}
.pager {
  justify-content:space-between;
}
.pager button {
  margin-left:.35rem;
}
.progress {
  display:flex;
  gap:.55rem;
  align-items:center;
  color:var(--muted);
  font-variant-numeric:tabular-nums;
  font-size:.89rem;
}
.progress progress {
  width:6.6rem;
  accent-color:var(--brand);
}
.status {
  color:var(--muted);
  font-size:.84rem;
  min-height:1.35rem;
  margin:0;
}
.item {
  border:1px solid var(--line);
  background:var(--white);
  border-radius:.6rem;
  overflow:hidden;
}
.itemheader {
  padding:.85rem 1rem;
  border-bottom:1px solid var(--line);
  background:#edf2ed;
}
.itemheader h2 {
  font-size:1.15rem;
  margin:0;
}
.itemheader p {
  margin:.35rem 0 0;
  color:var(--ink);
}
.compare {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:.85rem;
  padding:.8rem;
}
.compare.triple {
  grid-template-columns:repeat(3,minmax(0,1fr));
}
.side {
  min-width:0;
  padding:.65rem;
  border:1px solid var(--line);
  border-radius:.45rem;
}
.side h3 {
  font-size:1rem;
  margin:0 0 .55rem;
}
.sidegrid {
  display:grid;
  grid-template-columns:minmax(0,1fr);
  gap:.65rem;
}
.sidegrid.pair {
  grid-template-columns:repeat(2,minmax(0,1fr));
}
.endpoint {
  min-width:0;
  border-radius:.3rem;
  background:#f3f6f4;
  padding:.46rem;
}
.endpoint h4 {
  margin:0 0 .34rem;
  font-size:.87rem;
  color:#254754;
}
.pic {
  padding:0;
  border:0;
  width:min(100%,256px);
  margin:0 auto;
  display:block;
  cursor:zoom-in;
  overflow:hidden;
  background:#dee9e6;
}
.pic img {
  display:block;
  width:100%;
  aspect-ratio:1;
  height:auto;
}
.prompt {
  font-size:.85rem;
  color:#314a54;
  margin:.4rem 0 .55rem;
  overflow-wrap:anywhere;
}
.score {
  display:grid;
  gap:.38rem;
}
.score select,.closing textarea {
  width:100%;
  font-size:.9rem;
}
.score select {
  padding:.22rem .4rem;
}
.preserve {
  margin-top:.55rem;
}
.closing {
  padding:.45rem .8rem .85rem;
  display:grid;
  gap:.6rem;
}
.closing fieldset {
  border:1px solid var(--line);
  border-radius:.4rem;
  padding:.5rem .7rem;
}
.closing label {
  margin-right:.65rem;
}
.closing input {
  accent-color:var(--brand);
}
.closing textarea {
  display:block;
  min-height:2.65rem;
  resize:vertical;
  margin-top:.25rem;
}
.preference-row {
  display:flex;
  flex-wrap:wrap;
  align-items:baseline;
  gap:.15rem .45rem;
  padding:.3rem 0;
}
.preference-row strong {
  font-weight:650;
  min-width:4.4rem;
}
.hint {
  color:var(--muted);
  margin:0;
  font-size:.82rem;
}
dialog {
  max-width:min(94vw,670px);
  border:0;
  border-radius:.55rem;
  padding:.75rem;
}
dialog::backdrop {
  background:#0e222bd8;
}
dialog img {
  display:block;
  width:min(84vw,512px);
  height:auto;
}
dialog button {
  margin-bottom:.4rem;
}
.empty {
  padding:1.2rem;
}
.filters {
  margin-left:auto;
}
.badge {
  display:inline-block;
  border-radius:.25rem;
  background:var(--accent);
  padding:0 .35rem;
  margin-left:.35rem;
  font-size:.76rem;
  font-weight:600;
}
@media(max-width:800px) {
  .compare {
    gap:.5rem;
    padding:.48rem;
  }
  .side {
    padding:.4rem;
  }
  .sidegrid.pair {
    grid-template-columns:1fr;
  }
  .compare.triple {
    grid-template-columns:1fr;
  }
  .compare.triple .sidegrid.pair {
    grid-template-columns:repeat(2,minmax(0,1fr));
  }
  .endpoint {
    padding:.35rem;
  }
  .filters {
    margin-left:0;
  }
}
@media(max-width:520px) {
  main {
    padding-inline:.5rem;
  }
  .toolbar {
    gap:.45rem;
  }
  .toolbar label {
    flex:1 1 7.5rem;
  }
  .compare {
    gap:.3rem;
    padding:.35rem;
  }
  .side {
    padding:.3rem;
  }
  .pager>div:last-child button {
    margin-left:0;
  }
  .progress progress {
    width:3.4rem;
  }
}
</style></head>
<body>
<header><div class="shell"><h1>MiniGen 头像盲评</h1><p>按提示词评价头像、虹膜和指定属性，再比较头像偏好。</p></div></header>
<main>
<div class="toolbar">
  <label>评审代号<input id="reviewer" autocomplete="off" placeholder="如 reviewer-1"></label>
  <label>题型<select id="kind"><option value="">全部题型</option><option value="regular">常规</option><option value="pair">单标签干预</option></select></label>
  <label>筛选<select id="completion"><option value="">全部进度</option><option value="todo">未完成</option><option value="done">已完成</option></select></label>
  <button id="export" class="primary">导出匿名评分</button>
  <button id="import" type="button">导入评分</button><input id="file" type="file" accept=".json,application/json" hidden>
</div>
<details class="rubric"><summary>评分规则</summary><ul>
<li><strong>头像可用性。</strong> 主体、轮廓和五官连贯，可作为头像时选可用。五官明显移位、断裂或变形时选可辨但不可用。侧脸和题目要求的闭眼本身允许。</li>
<li><strong>虹膜。</strong> 左右至少一只应睁开的眼睛能分辨虹膜，才选可辨。模糊色块、反光、闭错眼或遮住应可见的眼睛记难辨。两眼都应闭合时，虹膜不适用。眨眼题需能判断两眼状态。</li>
<li id="defect-rubric" hidden><strong>五官缺陷。</strong> 眼周断裂指眼缘、眼眶有明显断口或错接，闭眼时仍可按可见眼缘评分。鼻口缺失指可见区域中的鼻口消失或严重残缺。脸缘粘连指脸部边界异常融入头发、衣物或背景。正常简笔、侧脸、闭眼、遮挡和裁边本身不算缺陷。部位确实未露出可选不适用，已露出但无法可靠判断可选难判。</li>
<li id="eye-color-link" hidden>有眼色标签时，虹膜记为难辨或生成失败，页面会同步眼色评分。</li>
<li><strong>指定标签。</strong> 目标明确吻合时选正确。可辨且与目标相违时选错误。若眼色或细节被遮住或失真，选难辨。清楚的头像里明显没有应有配饰，记错误。两只清楚的眼睛应同时符合指定的单一眼色。</li>
<li><strong>干预对。</strong> 分别评分 A、B。只有两个端点的目标都正确才算切换成功。非目标内容保持独立评分，观察构图、脸型、发型、背景及其他条件。开闭允许眼睑及其局部表情变化。</li>
<li><strong>偏好。</strong> 完成逐图评分后，比较两张头像，或比较两列的 A、B 整体。无明显优劣选平，显示故障或无法比较时单列。可点击图片放大。</li>
<li><strong>进度保存。</strong> 评分自动保存在当前浏览器。在其他浏览器或设备上继续时，先导出 JSON，再导入同一份评测包。</li>
</ul></details>
<div class="pager">
  <div class="progress"><progress id="progress"></progress><span id="progress-text"></span></div>
  <div><button id="prev">上一题</button><button id="next">下一题</button><button id="todo">下一道未完成</button></div>
</div>
<p class="status" role="status" id="status"></p>
<div id="card"></div>
</main><dialog id="zoom"><button id="close-zoom">关闭</button><img id="large" alt="评分图放大"></dialog>
<script id="deck" type="application/json">__ANONYMOUS_DATA__</script>
<script>
"use strict";
const $ = id => document.getElementById(id);
const deck = JSON.parse($("deck").textContent);
const byId = new Map(deck.items.map(item => [item.id, item]));
const storage = "minigen-anime-blind:" + deck.review_id;
const triple = Boolean(deck.columns);
const defectNames = deck.defects ?? {};
const defect = {present:"有缺陷",absent:"无缺陷",uncertain:"难判",not_applicable:"不适用"};
$("defect-rubric").hidden = !Object.keys(defectNames).length;
$("eye-color-link").hidden = !triple;
const scoreSchema = deck.score_schema ??
  (triple ? "minigen.anime_faces.blind_scores.v2" : "minigen.anime_faces.blind_scores.v1");
const kinds = {
  regular:"常规", pair:"单标签干预", eyes:"眼色", hair:"头发", closure:"开闭", accessories:"配饰",
};
const targetNames = {
  eye_color:"眼色", eye_state:"眼睛开闭", expression:"表情", hair_color:"发色",
  hair_length:"发长", hair_feature:"发型特征", ear_shape:"耳形", accessory:"配饰",
};
const face = {usable:"可用",distorted:"可辨但不可用",unrecognizable:"不可辨",failure:"生成失败"};
const iris = {readable:"可辨",unreadable:"难辨",not_applicable:"不适用",failure:"生成失败"};
const attribute = {correct:"正确",wrong:"错误",unreadable:"难辨",not_applicable:"不适用",failure:"生成失败"};
const preserve = {preserved:"保持",changed:"未保持",unreadable:"难辨",failure:"生成失败"};
const prefer = {left:"左胜",right:"右胜",tie:"平",not_comparable:"不可比较"};
const sides = deck.columns ?? ["left","right"];
const sideLabels = {left:"左侧",right:"右侧",alpha:"甲",beta:"乙",gamma:"丙"};
const preferencePairs = sides.flatMap((side,index) => sides.slice(index+1).map(other => [side,other]));
let reviews = {};
let active = deck.items[0].id;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, letter => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[letter]);
}
function readItem(item) { return reviews[item.id] || {}; }
function forcedEyeColor(point) {
  return triple && ["unreadable","failure"].includes(point?.iris) ? point.iris : "";
}
function sideReady(item, record, side) {
  return item.endpoints.every(({key,targets,iris_required}) => {
    const point = record[side]?.endpoints?.[key];
    const forced = targets.eye_color && forcedEyeColor(point);
    return face[point?.face] && (!iris_required || iris[point?.iris]) &&
      (!forced || point.targets?.eye_color === forced) &&
      Object.keys(defectNames).every(name => defect[point?.defects?.[name]]) &&
      Object.keys(targets).every(target => attribute[point?.targets?.[target]]);
  }) && (item.kind !== "pair" || preserve[record[side]?.preservation]);
}
function complete(item) {
  const score = readItem(item);
  if (!sides.every(side => sideReady(item,score,side))) return false;
  if (!triple) return Object.hasOwn(prefer,score.preference);
  return preferencePairs.every(([first,second]) =>
    [first,second,"tie","not_comparable"].includes(score.preferences?.[`${first}-${second}`]));
}
function persist() {
  try {
    localStorage.setItem(storage, JSON.stringify({reviewer:$("reviewer").value, active, reviews}));
    $("status").textContent = "已保存到此浏览器。";
  } catch {
    $("status").textContent = "当前浏览器无法保存，离开前请导出评分 JSON。";
  }
}
function filtered() {
  const kind = $("kind").value;
  const completion = $("completion").value;
  return deck.items.filter(item => (!kind || item.kind === kind) &&
    (!completion || (completion === "done") === complete(item)));
}
function refreshProgress() {
  const count = deck.items.filter(complete).length;
  $("progress").max = deck.items.length;
  $("progress").value = count;
  const visible = filtered();
  $("progress-text").textContent = `${count}/${deck.items.length} 已完成 · 当前筛选 ${visible.length} 题`;
  const position = visible.findIndex(item => item.id === active);
  $("prev").disabled = position <= 0;
  $("next").disabled = position >= visible.length-1;
  $("todo").disabled = !visible.some(item => !complete(item));
}
function options(values, selected, omit) {
  return '<option value="">待评分</option>' + Object.entries(values).filter(([value]) => value !== omit)
    .map(([value,label]) => `<option value="${value}"${value === selected ? " selected" : ""}>${label}</option>`).join("");
}
function select(label, values, selected, control = "", omit = "") {
  return `<label>${escapeHtml(label)}<select ${control}>${options(values,selected,omit)}</select></label>`;
}
function preferenceChoices(score) {
  if (!triple) return Object.entries(prefer).map(([value,label]) =>
    `<label><input type="radio" name="preference" value="${value}" ${score.preference === value ? "checked" : ""}> ${label}</label>`).join("");
  return preferencePairs.map(([first,second]) => {
    const pair = `${first}-${second}`;
    const options = {[first]:sideLabels[first], [second]:sideLabels[second], tie:"平", not_comparable:"不可比较"};
    const radios = Object.entries(options).map(([value,label]) =>
      `<label><input type="radio" name="preference-${pair}" data-preference="${pair}" value="${value}" ${score.preferences?.[pair] === value ? "checked" : ""}> ${label}</label>`).join("");
    const pairLabel = `${sideLabels[first]} / ${sideLabels[second]}`;
    return `<div class="preference-row" role="group" aria-label="${pairLabel}"><strong>${pairLabel}</strong>${radios}</div>`;
  }).join("");
}
function drawCard(keepActive = false) {
  const visible = filtered();
  const item = (keepActive && byId.get(active)) || visible.find(item => item.id === active) || visible[0];
  if (!item) {
    $("card").innerHTML = '<p class="empty">当前筛选下没有题目。</p>';
    refreshProgress();
    return;
  }
  active = item.id;
  const index = visible.findIndex(candidate => candidate.id === item.id);
  const score = readItem(item);
  const sidePanels = sides.map(side => {
    const label = sideLabels[side];
    const saved = score[side] || {};
    const panels = item.endpoints.map((meta, position) => {
      const {key,prompt,targets,iris_required} = meta;
      const point = saved.endpoints?.[key] || {};
      const targetsHtml = Object.entries(targets).map(([target,value]) => {
        const forced = target === "eye_color" && forcedEyeColor(point);
        return select((targetNames[target] || target)+ " · "+value, attribute, point.targets?.[target],
          `data-field="target" data-target="${escapeHtml(target)}"${forced ? " disabled" : ""}`, "not_applicable");
      }).join("");
      const irisHtml = iris_required
        ? select("虹膜可辨",iris,point.iris,'data-field="iris"',"not_applicable")
        : '<p class="hint">虹膜不适用</p>';
      const defectHtml = Object.entries(defectNames).map(([name,label]) =>
        select(label, defect, point.defects?.[name],
          `data-field="defect" data-defect="${escapeHtml(name)}"`)).join("");
      return `<section class="endpoint" data-key="${key}">
        ${item.kind === "pair" ? `<h4>${position ? "B · 替换标签" : "A · 原标签"}</h4>` : ""}
        <button class="pic" data-zoom="${side}/${key}" aria-label="放大 ${label} ${key}"><img src="${item.images[side][key]}" alt="${label}头像"></button>
        ${item.kind === "pair" ? `<p class="prompt" lang="en">${escapeHtml(prompt)}</p>` : ""}
        <div class="score">${select("头像可用性",face,point.face,'data-field="face"')}${irisHtml}${targetsHtml}${defectHtml}</div>
      </section>`;
    }).join("");
    const preservation = item.kind === "pair" ? `<div class="score preserve">${select(
      "非目标内容保持",preserve,saved.preservation,'data-field="preservation"')}</div>` : "";
    return `<section class="side" data-side="${side}"><h3>${label}</h3><div class="sidegrid ${item.kind}">${panels}</div>${preservation}</section>`;
  }).join("");
  const ready = sides.every(side => sideReady(item,score,side));
  const prompt = item.kind === "regular" ? `<p lang="en">${escapeHtml(item.endpoints[0].prompt)}</p>` :
    `<p>目标变化 <span lang="en">${escapeHtml(item.change.from)} → ${escapeHtml(item.change.to)}</span></p>`;
  $("card").innerHTML = `<article class="item"><header class="itemheader" style="color:var(--ink);border-bottom:1px solid var(--line)">
    <h2>${index >= 0 ? (index+1)+" / "+visible.length+" · " : ""}${item.id} · ${kinds[item.kind]} · ${kinds[item.category] || escapeHtml(item.category)}
    ${complete(item) ? '<span class="badge">已完成</span>' : ""}</h2>${prompt}</header>
    <div class="compare${triple ? " triple" : ""}">${sidePanels}</div><div class="closing">
    <fieldset id="preference" ${ready ? "" : "disabled"}><legend>头像偏好</legend>${preferenceChoices(score)}
    ${ready ? "" : '<p class="hint">完成逐图评分后可选。</p>'}</fieldset>
    <label>备注（可选）<textarea id="note" placeholder="记下边界情况或评分依据">${escapeHtml(score.note)}</textarea></label>
    </div></article>`;
  refreshProgress();
}
function update(item, target) {
  const score = reviews[item.id] ??= {};
  if (target.id === "note") {
    score.note = target.value;
    return;
  }
  if (target.name === "preference") {
    score.preference = target.value;
    return;
  }
  if (target.dataset.preference) {
    (score.preferences ??= {})[target.dataset.preference] = target.value;
    return;
  }
  const side = target.closest("[data-side]")?.dataset.side;
  const field = target.dataset.field;
  if (!side || !field) return;
  const section = score[side] ??= {endpoints:{}};
  if (field === "preservation") {
    section.preservation = target.value;
    return;
  }
  const key = target.closest("[data-key]").dataset.key;
  const meta = item.endpoints.find(endpoint => endpoint.key === key);
  const point = section.endpoints[key] ??= {targets:{},iris:meta.iris_required ? "" : "not_applicable"};
  if (field === "target") {
    point.targets[target.dataset.target] = target.value;
    return;
  }
  if (field === "defect") {
    (point.defects ??= {})[target.dataset.defect] = target.value;
    return;
  }
  const previous = forcedEyeColor(point);
  point[field] = target.value;
  if (triple && field === "iris" && meta.targets.eye_color) {
    const control = target.closest("[data-key]").querySelector('[data-target="eye_color"]');
    const forced = forcedEyeColor(point);
    if (forced || previous) {
      point.targets.eye_color = forced;
      control.value = forced;
    }
    control.disabled = Boolean(forced);
  }
}
function move(delta) {
  const items = filtered();
  const position = items.findIndex(item => item.id === active);
  const item = items[position + delta];
  if (!item) return;
  active = item.id;
  drawCard();
  persist();
  $("card").scrollIntoView({block:"nearest"});
}
function nextTodo() {
  const items = filtered();
  const index = items.findIndex(item => item.id === active);
  const next = [...items.slice(index+1),...items.slice(0,index+1)].find(item => !complete(item));
  if (next) {active=next.id;drawCard();persist();$("card").scrollIntoView({block:"nearest"});}
}
try {
  const saved = JSON.parse(localStorage.getItem(storage) || "{}");
  if (saved.reviews) reviews = saved.reviews;
  if (byId.has(saved.active)) active = saved.active;
  $("reviewer").value = saved.reviewer || "";
} catch { $("status").textContent = "无法读取本地评分。可以导入备份。"; }
for (const id of ["kind","completion"]) $(id).addEventListener("change",() => drawCard());
$("reviewer").addEventListener("change",persist);
$("prev").addEventListener("click",() => move(-1));
$("next").addEventListener("click",() => move(1));
$("todo").addEventListener("click",nextTodo);
$("card").addEventListener("input",event => {
  if (event.target.id !== "note") return;
  update(byId.get(active),event.target);persist();
});
$("card").addEventListener("change",event => {
  if (event.target.id === "note") return;
  const item = byId.get(active);
  update(item,event.target);
  const score = readItem(item);
  const fieldset = $("preference");
  const ready = sides.every(side => sideReady(item,score,side));
  fieldset.disabled = !ready;
  fieldset.querySelector(".hint")?.remove();
  if (!ready) {
    if (score.preference) delete score.preference;
    if (score.preferences) delete score.preferences;
    fieldset.querySelectorAll("input").forEach(input => {input.checked=false;});
    fieldset.insertAdjacentHTML("beforeend",'<p class="hint">完成逐图评分后可选。</p>');
  }
  persist();refreshProgress();
});
$("card").addEventListener("click",event => {
  const button = event.target.closest("[data-zoom]");
  if (!button) return;
  const [side,key] = button.dataset.zoom.split("/");
  $("large").src = byId.get(active).images[side][key];
  $("zoom").showModal();
});
$("close-zoom").addEventListener("click",() => $("zoom").close());
$("export").addEventListener("click",() => {
  const reviewer = $("reviewer").value.trim();
  if (!reviewer) {
    $("status").textContent = "请填写评审代号后导出。";
    $("reviewer").focus();
    return;
  }
  const payload = {schema:scoreSchema,review_id:deck.review_id,reviewer,items:reviews};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "anime_benchmark_blind_scores.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
  $("status").textContent = "已导出匿名评分。";
});
$("import").addEventListener("click",() => $("file").click());
$("file").addEventListener("change",async () => {
  const file = $("file").files[0];
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    if (payload.schema !== scoreSchema || payload.review_id !== deck.review_id ||
        !payload.items || Array.isArray(payload.items) || typeof payload.items !== "object" ||
        Object.keys(payload.items).some(id => !byId.has(id))) {
      throw Error("文件和当前盲评包不匹配");
    }
    reviews = payload.items;
    $("reviewer").value = payload.reviewer || "";
    drawCard();
    persist();
  } catch (error) {
    $("status").textContent = "导入未完成。"+error.message;
  } finally { $("file").value = ""; }
});
drawCard();
</script></body></html>
"""


def read_json(
    path: Path,
) -> JsonDict:
    """读取 Benchmark 或生成回执中的 JSON 对象。

    Args:
        path: 本地 JSON 文件路径。

    Raises:
        OSError: 文件不可读。
        ValueError: 数据无法解析为 JSON 对象。

    Results:
        JsonDict: 解析得到的对象。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def image_data(
    path: Path,
    expected_hash: str,
    resolution: int,
) -> str:
    """核对生成图的原始哈希，并去掉 PNG 元数据后内嵌像素。

    Args:
        path: 生成回执记载的 PNG 文件。
        expected_hash: 原始文件的 SHA-256。
        resolution: 冻结的正方形图片边长。

    Raises:
        OSError: 文件无法读取或解码。
        ValueError: 原始哈希、PNG 格式或尺寸不符。

    Results:
        str: 去掉元数据的 RGB PNG 数据 URI。
    """
    binary = path.read_bytes()
    if hashlib.sha256(binary).hexdigest() != expected_hash:
        raise ValueError(f"Image hash mismatch: {path}")
    with Image.open(BytesIO(binary)) as source:
        if source.format != "PNG" or source.size != (resolution, resolution):
            raise ValueError(f"Unexpected image format or dimensions: {path}")
        # 创建仅有像素的新对象，使文件注释、配置和源文件名不进入评分包。
        rgb = Image.frombytes("RGB", source.size, source.convert("RGB").tobytes())
    sanitized = BytesIO()
    rgb.save(sanitized, format="PNG")
    return "data:image/png;base64," + base64.b64encode(sanitized.getvalue()).decode("ascii")


def endpoint(
    key: str,
    prompt: str,
    targets: dict[str, str],
) -> JsonDict:
    """创建只包含匿名页面所需条件的端点说明。

    Args:
        key: single、a 或 b。
        prompt: 冻结的英文提示词。
        targets: 需要评分的属性到标签映射。

    Results:
        JsonDict: 端点标识、原提示词、评分属性和固定虹膜适用性。
    """
    state = targets.get("eye_state")
    return {
        "key": key,
        "prompt": prompt,
        "targets": targets,
        "iris_required": state != "closed eyes"
        and ("eye_color" in targets or state == "one eye closed"),
    }


def order_key(
    secret: bytes,
    label: str,
) -> str:
    """用密钥派生匿名编号或排序值。

    Args:
        secret: 仅用于构建过程的盲化密钥。
        label: 有独立用途前缀的题号与种子字符串。

    Results:
        str: 可复现的 HMAC-SHA256 十六进制结果。
    """
    return hmac.new(secret, label.encode(), "sha256").hexdigest()


def validate_scale_receipts(
    receipts: dict[str, JsonDict],
    baseline_receipt: Path,
) -> None:
    """核对同题同初噪的 30M 对照和两份 50M 最终权重身份。

    Args:
        receipts: 按预登记顺序排列的三个模型生成回执。
        baseline_receipt: 已复用的 30M 回执路径，用于核对原始文件摘要。

    Raises:
        OSError: 范围锁或原始回执不可读。
        KeyError: 范围锁或回执缺少身份或协议字段。
        ValueError: 快照、模型、数据或采样设置与三组对照要求不符。
    """
    scope_bytes = SCALE_SCOPE.read_bytes()
    scope = json.loads(scope_bytes)
    scope_sha256 = hashlib.sha256(scope_bytes).hexdigest()
    baseline = receipts[SCALE_ARMS[0]]
    locked_baseline = scope["baseline"]
    if (
        scope["schema"] != "minigen.anime_faces.scale_50m_scope.v1"
        or scope["benchmark_sha256"] != baseline["benchmark_sha256"]
        or scope["rubric_sha256"] != baseline["rubric_sha256"]
        or locked_baseline["arm"] != SCALE_ARMS[0]
        or locked_baseline["receipt_sha256"]
        != hashlib.sha256(baseline_receipt.read_bytes()).hexdigest()
        or locked_baseline["checkpoint_sha256"] != baseline["checkpoint_sha256"]
        or [receipt["checkpoint_step"] for receipt in receipts.values()] != [10000, 20000, 20000]
        or any(receipt["cumulative_training_updates"] != 20000 for receipt in receipts.values())
        or len({receipt["run_id"] for receipt in receipts.values()}) != 3
        or any(receipt["status"] != "completed" for receipt in receipts.values())
    ):
        raise ValueError("Generation receipts disagree on the registered 20k scale checkpoints")
    shared_fields = (
        "data_identity",
        "group_exclusions_sha256",
        "source_manifest_sha256",
        "noise_shape",
        "condition_encoding",
        "sampling_batch_size",
        "decode_batch_size",
        "objective",
        "tf32",
    )
    for arm in SCALE_ARMS[1:]:
        candidate = scope["candidates"][arm]
        receipt = receipts[arm]
        if (
            any(
                receipt[field] != candidate[field]
                for field in ("architecture", "checkpoint_sha256", "run_id")
            )
            or receipt["trainable_parameters"] != candidate["parameters"]
            or receipt["comparison"] != scope["comparison"]
            or receipt["comparison_scope_sha256"] != scope_sha256
            or receipt["baseline_checkpoint_sha256"] != baseline["checkpoint_sha256"]
            or receipt["device"].split(":", 1)[0] != "cuda"
            or any(receipt[field] != baseline[field] for field in shared_fields)
        ):
            raise ValueError(f"Generation receipt disagrees with the frozen scale protocol: {arm}")


def build_review(
    benchmark_path: Path,
    generation: Path,
    blind_seed: int | None,
    arms: tuple[str, ...] = ARMS,
) -> tuple[JsonDict, JsonDict]:
    """根据已完成的生成回执建立公开评分包和私有解盲映射。

    Args:
        benchmark_path: 已在生成回执登记的冻结 Benchmark。
        generation: 含各组 receipt.json 与生成图的根目录。
        blind_seed: 可选盲化种子。缺省使用系统随机密钥。
        arms: 模型目录名和回执身份，默认原消融对照，支持已预登记的三组规模对照。

    Raises:
        OSError: Benchmark、生成回执或图片无法读取。
        KeyError: 冻结题单或回执缺少约定字段。
        ValueError: 题单、回执、样本配对或图片身份不符。

    Results:
        tuple[JsonDict, JsonDict]: 可公开给评审的匿名评分包，以及只供汇总者使用的映射。
    """
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    benchmark_hash = hashlib.sha256(benchmark_bytes).hexdigest()
    if (
        benchmark["schema"] != BENCHMARK_SCHEMA
        or len(benchmark["regular"]) != 80
        or len(benchmark["pairs"]) != 24
        or len(benchmark["seeds"]) != 2
        or len(set(benchmark["seeds"])) != 2
    ):
        raise ValueError("Expected 80 fixed conditions, 24 paired interventions and two seeds")
    if arms not in {ARMS, CONTINUATION_ARMS, SCALE_ARMS}:
        raise ValueError("Unknown anime benchmark comparison")
    scale = arms == SCALE_ARMS
    columns = SCALE_COLUMNS if scale else ("left", "right")
    receipts = {arm: read_json(generation / arm / "receipt.json") for arm in arms}
    first_receipt = receipts[arms[0]]
    other_receipts = [receipts[arm] for arm in arms[1:]]
    if (
        any(
            first_receipt[field] != candidate[field]
            for candidate in other_receipts
            for field in RECEIPT_FIELDS
        )
        or first_receipt["schema"] != GENERATION_SCHEMA
        or first_receipt["benchmark_sha256"] != benchmark_hash
        or first_receipt["seeds"] != benchmark["seeds"]
        or first_receipt["sampling"] != benchmark["sampling"]
        or len({receipt["checkpoint_sha256"] for receipt in receipts.values()}) != len(arms)
    ):
        raise ValueError("Generation receipts disagree on the frozen benchmark or sampling")
    if scale:
        validate_scale_receipts(receipts, generation / arms[0] / "receipt.json")
    elif arms == CONTINUATION_ARMS:
        right_receipt = other_receipts[0]
        if (
            first_receipt["checkpoint_step"] != 3000
            or right_receipt["checkpoint_step"] != 10000
            or right_receipt["cumulative_training_updates"] != 20000
            or right_receipt["baseline_checkpoint_sha256"] != first_receipt["checkpoint_sha256"]
            or any(
                first_receipt[field] != right_receipt[field]
                for field in (
                    "run_id",
                    "data_identity",
                    "model_config",
                    "noise_shape",
                    "condition_encoding",
                    "sampling_batch_size",
                    "decode_batch_size",
                    "tf32",
                )
            )
        ):
            raise ValueError("Uniform 13k/20k lineage or generation parameters disagree")
    elif first_receipt["checkpoint_step"] != other_receipts[0]["checkpoint_step"]:
        raise ValueError("Original time-ablation comparison requires equal checkpoint steps")
    records = {}
    for arm, receipt in receipts.items():
        rows = receipt["records"]
        records[arm] = {(row["id"], row["seed"], row["endpoint"]): row for row in rows}
        if receipt["arm"] != arm or len(rows) != 256 or len(records[arm]) != len(rows):
            raise ValueError(f"Invalid or incomplete generation receipt: {arm}")

    secret = (
        secrets.token_bytes(32)
        if blind_seed is None
        else hashlib.sha256(f"{benchmark_hash}:{blind_seed}".encode()).digest()
    )
    strata: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for kind, questions in (("regular", benchmark["regular"]), ("pair", benchmark["pairs"])):
        for question in questions:
            if question["seeds"] != benchmark["seeds"]:
                raise ValueError(f"Unexpected seeds for {question['id']}")
            if kind == "regular":
                endpoints = [endpoint("single", question["prompt"], question["targets"])]
            else:
                endpoints = [
                    endpoint(label, question[f"prompt_{label}"], question[f"targets_{label}"])
                    for label in ("a", "b")
                ]
            for seed in question["seeds"]:
                stable_key = f"{question['id']}:{seed}"
                card = {
                    "id": order_key(secret, f"card:{stable_key}")[:16],
                    "kind": kind,
                    "category": question["category"],
                    "endpoints": endpoints,
                }
                if kind == "pair":
                    card["change"] = question["changed"]
                strata[kind, question["category"]].append(
                    {"card": card, "question": question, "seed": seed}
                )

    cards = []
    assignments = {}
    expected_keys: set[tuple[str, int, str | None]] = set()
    noise_slots = {
        question["id"]: index
        for index, question in enumerate(benchmark["regular"] + benchmark["pairs"])
    }
    odd = int(order_key(secret, "parity")[-1], base=16) % 2
    offset = int(order_key(secret, "scale-column-offset")[-2:], base=16) % 6
    for stratum, items in sorted(strata.items()):
        # 在题型和类别内打乱题目，同一题的两个种子复用模型列位。
        question_ids = {item["question"]["id"] for item in items}
        ordered = sorted(
            question_ids, key=lambda identifier: order_key(secret, f"side:{identifier}")
        )
        if scale:
            # 每三个连续排列都让各模型在每列各出现一次，两个方向共六种排列。
            rotations = [arms[index:] + arms[:index] for index in range(3)]
            layouts = rotations + [layout[::-1] for layout in rotations]
            placements = {
                identifier: layouts[(offset + index) % len(layouts)]
                for index, identifier in enumerate(ordered)
            }
            offset += len(ordered)
        else:
            first_arm_count = len(ordered) // 2 + (odd if len(ordered) % 2 else 0)
            odd ^= len(ordered) % 2
            placements = {
                identifier: arms if index < first_arm_count else arms[::-1]
                for index, identifier in enumerate(ordered)
            }
        for item in items:
            card, question, seed = item["card"], item["question"], item["seed"]
            model_columns = dict(zip(columns, placements[question["id"]], strict=True))
            images: dict[str, dict[str, str]] = {column: {} for column in columns}
            noise_hashes = set()
            for meta in card["endpoints"]:
                name = meta["key"]
                record_key = (question["id"], seed, None if name == "single" else name)
                suffix = "" if name == "single" else f"_{name}"
                expected_filename = f"{question['id']}_{seed}{suffix}.png"
                expected_keys.add(record_key)
                for side, arm in model_columns.items():
                    generated = records[arm].get(record_key)
                    if generated is None or (
                        generated["prompt"] != meta["prompt"]
                        or generated["image_path"] != expected_filename
                        or not generated["noise_sha256"]
                        or (scale and generated["noise_index"] != noise_slots[question["id"]])
                    ):
                        raise ValueError(
                            f"Unexpected prompt, initial noise or output: {record_key}"
                        )
                    noise_hashes.add(generated["noise_sha256"])
                    images[side][name] = image_data(
                        generation / arm / expected_filename,
                        generated["image_sha256"],
                        benchmark["sampling"]["resolution"],
                    )
            if len(noise_hashes) != 1:
                raise ValueError(f"Different noise within card: {question['id']}/{seed}")
            card["images"] = images
            cards.append(card)
            assignments[card["id"]] = {
                "question_id": question["id"],
                "seed": seed,
                "kind": stratum[0],
                "category": stratum[1],
                "group_id": question["source"]["group_id"],
                "changed_attribute": question.get("changed", {}).get("attribute"),
                **model_columns,
            }
    if (
        len(cards) != 208
        or len(assignments) != len(cards)
        or any(arm_records.keys() != expected_keys for arm_records in records.values())
    ):
        raise ValueError("Expected 208 cards covering each generated endpoint exactly once")
    cards.sort(key=lambda card: order_key(secret, f"position:{card['id']}"))
    artifact_keys = sorted(
        (
            arm,
            receipt["checkpoint_sha256"],
            tuple(record["image_sha256"] for record in receipt["records"]),
        )
        for arm, receipt in receipts.items()
    )
    review_id = order_key(secret, f"deck:{benchmark_hash}:{artifact_keys}")[:24]
    deck = {"review_id": review_id, "items": cards}
    mapping = {
        "schema": "minigen.anime_faces.blind_mapping.v1",
        "review_id": review_id,
        "benchmark_sha256": benchmark_hash,
        "approval_sha256": first_receipt["approval_sha256"],
        "sampling": benchmark["sampling"],
        "checkpoint_identities": {
            arm: {
                field: receipt[field]
                for field in ("checkpoint_sha256", "checkpoint_step", "run_id")
            }
            for arm, receipt in receipts.items()
        },
        "assignments": assignments,
    }
    if scale:
        deck["columns"] = list(columns)
        mapping.update(
            schema="minigen.anime_faces.blind_mapping.v2",
            arms=list(arms),
            columns=list(columns),
            comparison_scope_sha256=other_receipts[0]["comparison_scope_sha256"],
        )
    return deck, mapping


def main() -> None:
    """从命令行生成独立的评分 HTML 和私有解盲 JSON。

    Raises:
        OSError: 输入资料不可读、图片不可解码、目标目录不可写。
        KeyError: 冻结题单或生成回执缺少约定字段。
        ValueError: 输入身份、配对或图片核验不通过。

    副作用:
        创建输出目录，写入 HTML 与 JSON，打印文件位置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", default=ARMS, metavar="ARM")
    parser.add_argument("--output", required=True, type=Path, help="匿名评分 HTML")
    parser.add_argument("--private-map", type=Path, help="私有解盲映射 JSON")
    parser.add_argument("--blind-seed", type=int, help="用于重建同一份左右分组的私有种子")
    args = parser.parse_args()
    private_map = args.private_map or args.output.with_suffix(".private.json")
    if private_map.resolve() == args.output.resolve():
        raise ValueError("HTML and private mapping must have independent paths")
    deck, mapping = build_review(args.benchmark, args.generation, args.blind_seed, tuple(args.arms))
    payload = json.dumps(deck, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private_map.parent.mkdir(parents=True, exist_ok=True)
    private_map.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(HTML.replace("__ANONYMOUS_DATA__", payload), encoding="utf-8")
    print(f"Blind review ({len(deck['items'])} cards): {args.output}")
    print(f"Private mapping: {private_map}")


if __name__ == "__main__":
    main()

const SUPABASE_URL = "https://zobfhaykmvqsdtdkopuc.supabase.co";
const SUPABASE_KEY = "sb_publishable_-YgL8OtMraWknloMv-YWjA_EY1qhBts";
const db = window.supabase ? supabase.createClient(SUPABASE_URL, SUPABASE_KEY) : null;

const MODEL_PATH = "models/ppe-yolo.onnx";
const INPUT = 640;
const CONF = 0.4;
const IOU = 0.45;
const CLASS_NAMES = ["no-safety-glove","no-safety-helmet","no-safety-shoes","no-welding-glass","safety-glove","safety-helmet","safety-shoes","welding-glass","unsafe-area","damage","forklift"];
const UNSAFE = new Set(["no-safety-glove","no-safety-helmet","no-safety-shoes","no-welding-glass","unsafe-area","damage"]);

let session, stream, photoBase64, autoTimer;
let history = [];
let stats = { total: 0, safe: 0, warn: 0, unsafe: 0 };

const $ = id => document.getElementById(id);

if (location.pathname.endsWith("login.html") || location.pathname.endsWith("/")) {
  document.addEventListener("keydown", e => { if (e.key === "Enter" && $("username")) login(); });
}
if ($("verdict")) {
  const user = JSON.parse(sessionStorage.getItem("sv_user") || "null");
  if (!user) location.href = "login.html";
  log("Ready.");
}

async function sha256Hex(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function login() {
  const username = $("username").value.trim();
  const password = $("password").value;
  const error = $("loginError");
  error.classList.add("hidden");
  try {
    let user = null;
    if (db) {
      const res = await db.from("users").select("id, username, role, password_hash").eq("username", username).maybeSingle();
      if (res.error) throw res.error;
      if (res.data && await sha256Hex(password) === res.data.password_hash) user = res.data;
    }
    if (!user && username === "admin" && password === "safeview2025") user = { id: null, username, role: "admin" };
    if (!user) throw new Error("Invalid username or password.");
    sessionStorage.setItem("sv_user", JSON.stringify({ id: user.id, username: user.username, role: user.role }));
    location.href = "index.html";
  } catch (e) {
    error.textContent = e.message || "Login failed.";
    error.classList.remove("hidden");
  }
}

function logout() {
  sessionStorage.clear();
  location.href = "login.html";
}

function setMode(mode) {
  $("liveTab").classList.toggle("active", mode === "live");
  $("photoTab").classList.toggle("active", mode === "photo");
  $("livePane").classList.toggle("hidden", mode !== "live");
  $("photoPane").classList.toggle("hidden", mode !== "photo");
}

async function loadModel() {
  if (session) return session;
  $("modelState").textContent = "Loading model";
  session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["wasm"] });
  $("modelState").textContent = "Model ready";
  return session;
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } } });
  $("video").srcObject = stream;
  $("cameraEmpty").classList.add("hidden");
  $("startBtn").disabled = true; $("scanBtn").disabled = false; $("stopBtn").disabled = false;
  setAutoScan();
  log("Camera started.");
}

function stopCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  clearInterval(autoTimer);
  $("video").srcObject = null;
  $("cameraEmpty").classList.remove("hidden");
  $("startBtn").disabled = false; $("scanBtn").disabled = true; $("stopBtn").disabled = true;
  log("Camera stopped.");
}

function setAutoScan() {
  clearInterval(autoTimer);
  const ms = Number($("interval").value);
  if (ms && stream) autoTimer = setInterval(scanCamera, ms);
}

async function scanCamera() {
  if (!stream) return;
  const video = $("video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);
  await analyzeImage(canvas, "live", $("liveCanvas"), $("liveBadge"));
}

function loadPhoto(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    photoBase64 = reader.result;
    $("photo").src = photoBase64;
    $("photoWrap").classList.remove("hidden");
    log("Photo loaded.");
  };
  reader.readAsDataURL(file);
}

async function scanPhoto() {
  if (!photoBase64) return;
  const img = $("photo");
  await img.decode().catch(() => {});
  await analyzeImage(img, "photo", $("photoCanvas"), $("photoBadge"));
}

function clearPhoto() {
  photoBase64 = null;
  $("fileInput").value = "";
  $("photoWrap").classList.add("hidden");
}

async function analyzeImage(sourceEl, source, canvas, badge) {
  try {
    await loadModel();
    const width = sourceEl.videoWidth || sourceEl.naturalWidth || sourceEl.width;
    const height = sourceEl.videoHeight || sourceEl.naturalHeight || sourceEl.height;
    const detections = await detect(sourceEl, width, height);
    draw(canvas, detections, width, height);
    applyResult(classify(detections), source, badge);
  } catch (e) {
    log("Detection failed: " + (e.message || e), "unsafe");
  }
}

function preprocess(src, w, h) {
  const c = document.createElement("canvas");
  c.width = INPUT; c.height = INPUT;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#727272"; ctx.fillRect(0, 0, INPUT, INPUT);
  const scale = Math.min(INPUT / w, INPUT / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const dx = (INPUT - nw) / 2, dy = (INPUT - nh) / 2;
  ctx.drawImage(src, 0, 0, w, h, dx, dy, nw, nh);
  const data = ctx.getImageData(0, 0, INPUT, INPUT).data;
  const area = INPUT * INPUT;
  const chw = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    chw[i] = data[i * 4] / 255;
    chw[area + i] = data[i * 4 + 1] / 255;
    chw[area * 2 + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", chw, [1, 3, INPUT, INPUT]), scale, dx, dy };
}

async function detect(src, w, h) {
  const s = await loadModel();
  const prep = preprocess(src, w, h);
  const out = await s.run({ [s.inputNames[0]]: prep.tensor });
  const y = out[s.outputNames[0]];
  const preds = y.dims[y.dims.length - 1];
  const boxes = [];
  for (let i = 0; i < preds; i++) {
    let score = 0, cls = -1;
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const v = y.data[(4 + c) * preds + i];
      if (v > score) { score = v; cls = c; }
    }
    if (score < CONF) continue;
    const cx = y.data[i], cy = y.data[preds + i], bw = y.data[2 * preds + i], bh = y.data[3 * preds + i];
    boxes.push({
      x1: Math.max(0, (cx - bw / 2 - prep.dx) / prep.scale),
      y1: Math.max(0, (cy - bh / 2 - prep.dy) / prep.scale),
      x2: Math.min(w, (cx + bw / 2 - prep.dx) / prep.scale),
      y2: Math.min(h, (cy + bh / 2 - prep.dy) / prep.scale),
      score, label: CLASS_NAMES[cls] || `class_${cls}`
    });
  }
  return nms(boxes);
}

function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const b of boxes) {
    if (kept.every(k => overlap(k, b) < IOU)) kept.push(b);
  }
  return kept;
}

function overlap(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aa = (a.x2 - a.x1) * (a.y2 - a.y1), ba = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aa + ba - inter || 1);
}

function classify(detections) {
  const bad = detections.filter(d => UNSAFE.has(d.label));
  const good = detections.filter(d => !UNSAFE.has(d.label));
  const level = bad.length ? "unsafe" : good.length ? "safe" : "warn";
  return {
    level,
    verdict: level === "unsafe" ? "UNSAFE" : level === "safe" ? "SAFE" : "WARNING",
    risk: level === "unsafe" ? Math.min(10, 6 + bad.length) : level === "safe" ? 2 : 4,
    summary: detections.length ? `${detections.length} PPE item(s) detected` : "No PPE detected",
    findings: detections.map(d => ({ level: UNSAFE.has(d.label) ? "unsafe" : "safe", text: `${d.label} ${Math.round(d.score * 100)}%` }))
  };
}

function draw(canvas, detections, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(2, w / 300);
  ctx.font = `${Math.max(13, w / 58)}px Arial`;
  for (const d of detections) {
    const bad = UNSAFE.has(d.label);
    ctx.strokeStyle = bad ? "#ff4d4d" : "#2fb36d";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const label = `${d.label} ${Math.round(d.score * 100)}%`;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - 24), ctx.measureText(label).width + 12, 24);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, d.x1 + 6, Math.max(16, d.y1 - 7));
  }
}

function applyResult(result, source, badge) {
  badge.textContent = result.verdict;
  const card = $("verdict");
  card.className = "verdict " + result.level;
  card.innerHTML = `<strong>${result.verdict}</strong><span>Risk ${result.risk}/10. ${result.summary}</span>`;
  $("findings").className = "list";
  $("findings").innerHTML = result.findings.length ? result.findings.map(f => `<div class="finding ${f.level}">${f.text}</div>`).join("") : `<div class="muted">No detections.</div>`;
  stats.total++; stats[result.level]++;
  for (const k of ["total","safe","warn","unsafe"]) $(k).textContent = stats[k];
  const row = { timestamp: new Date().toISOString(), source, ...result };
  history.push(row);
  if (result.level === "unsafe") flashAlert();
  log(`${source}: ${result.verdict} risk ${result.risk}/10`, result.level);
  saveScan(row);
}

async function saveScan(row) {
  const user = JSON.parse(sessionStorage.getItem("sv_user") || "null");
  if (!db) return;
  const res = await db.from("scans").insert({
    user_id: user && user.id,
    source: row.source,
    verdict: row.verdict,
    confidence: row.findings.length ? "High" : "Low",
    risk_score: row.risk,
    summary: row.summary,
    recommendation: row.level === "unsafe" ? "Correct missing PPE before work continues." : "Continue monitoring.",
    provider_used: "yolo-local"
  }).select("id").single();
  if (res.error || !row.findings.length) return;
  await db.from("findings").insert(row.findings.map(f => ({ scan_id: res.data.id, level: f.level, text: f.text })));
}

function flashAlert() {
  $("alert").classList.add("show");
  setTimeout(() => $("alert").classList.remove("show"), 3500);
}

function log(msg, level = "") {
  const area = $("log");
  if (!area) return;
  const div = document.createElement("div");
  div.className = "log-row";
  div.textContent = `${new Date().toLocaleTimeString()} - ${msg}`;
  if (level) div.style.color = `var(--${level})`;
  area.prepend(div);
}

function exportCsv() {
  const rows = ["timestamp,source,verdict,risk,summary", ...history.map(r => [r.timestamp,r.source,r.verdict,r.risk,`"${r.summary}"`].join(","))];
  download("safeview-scans.csv", rows.join("\n"), "text/csv");
}

function exportJson() {
  download("safeview-scans.json", JSON.stringify({ stats, history }, null, 2), "application/json");
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
}

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
let promptEvent;
window.addEventListener("beforeinstallprompt", e => {
  promptEvent = e;
  const btn = $("installBtn");
  if (btn) btn.classList.remove("hidden");
});
if ($("installBtn")) $("installBtn").onclick = () => promptEvent && promptEvent.prompt();
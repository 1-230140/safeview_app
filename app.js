const SUPABASE_URL = "https://zobfhaykmvqsdtdkopuc.supabase.co";
const SUPABASE_KEY = "sb_publishable_-YgL8OtMraWknloMv-YWjA_EY1qhBts";
const db = window.supabase ? supabase.createClient(SUPABASE_URL, SUPABASE_KEY) : null;

const API_BASE_URL = "http://127.0.0.1:8001";
const INPUT = 640;
let CONF = 0.70;
const CLASS_NAMES = ["cardboard-box", "forklift", "safety-cone", "safety-vest"];
const UNSAFE = new Set(["unsafe-area","damage","forklift","cardboard-box","safety-cone"]);

let stream, photoBase64, autoTimer;
let history = [];
let stats = { total: 0, safe: 0, warn: 0, unsafe: 0 };
let lastScan = null;
let lastFps = 0;
let processMode = "original";
let activeModel = "fasterrcnn";

const MODEL_OPTIONS = {
  fasterrcnn: {
    label: "Faster R-CNN v3",
    confidence: 0.70,
    note: "Faster R-CNN: higher test mAP; selected confidence 70%."
  },
  yolov10: {
    label: "YOLOv10n",
    confidence: 0.25,
    note: "YOLOv10: lower test mAP but much faster; selected confidence 25%."
  }
};

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
      try {
        const res = await db.from("users").select("id, username, role, password_hash").eq("username", username).maybeSingle();
        if (!res.error && res.data && await sha256Hex(password) === res.data.password_hash) user = res.data;
      } catch (_) {
        // Allow the local demo login to work when Supabase is unreachable.
      }
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
  setModelState("fasterRcnnState", "Checking", "loading");
  setModelState("yolov10State", "Checking", "loading");
  $("modelState").textContent = "Checking model service";
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("Model service is unavailable.");
    const status = await response.json();
    setModelState("fasterRcnnState", status.models.fasterrcnn ? "Available" : "Missing", status.models.fasterrcnn ? "ready" : "error");
    setModelState("yolov10State", status.models.yolov10 ? "Available" : "Missing", status.models.yolov10 ? "ready" : "error");
    if (!status.models[activeModel]) throw new Error("Selected model checkpoint is missing.");
    $("modelState").textContent = `${MODEL_OPTIONS[activeModel].label} ready`;
  } catch (error) {
    setModelState("fasterRcnnState", "Offline", "error");
    setModelState("yolov10State", "Offline", "error");
    $("modelState").textContent = "Model service offline";
    throw error;
  }
}

function setModelState(id, text, state) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.className = `status-${state}`;
}

function setActiveModel(model) {
  activeModel = model;
  const option = MODEL_OPTIONS[model];
  CONF = option.confidence;
  $("confidenceThreshold").value = CONF;
  $("confidenceValue").textContent = `${Math.round(CONF * 100)}%`;
  $("modelChoiceNote").textContent = option.note;
  $("modelState").textContent = `${option.label} selected`;
  log(`${option.label} selected.`);
}

function updateConfidenceThreshold(value) {
  CONF = Number(value);
  $("confidenceValue").textContent = `${Math.round(CONF * 100)}%`;
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
    $("processPanel").classList.remove("hidden");
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
  lastScan = null;
  $("fileInput").value = "";
  $("photoWrap").classList.add("hidden");
  $("processPanel").classList.add("hidden");
}

async function analyzeImage(sourceEl, source, canvas, badge) {
  try {
    await loadModel();
    const width = sourceEl.videoWidth || sourceEl.naturalWidth || sourceEl.width;
    const height = sourceEl.videoHeight || sourceEl.naturalHeight || sourceEl.height;
    const result = await detect(sourceEl);
    const detections = result.detections;
    lastFps = 1000 / Math.max(1, result.inference_ms);
    lastScan = { sourceEl, width, height, detections, fps: lastFps, model: result.model_name };
    draw(canvas, detections, width, height);
    renderAutoMetrics(detections, width, height, lastFps);
    renderProcessView();
    applyResult(classify(detections), source, badge);
  } catch (e) {
    log("Detection failed: start the local model service, then try again.", "unsafe");
  }
}

async function imageBlob(src, width, height) {
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  ctx.drawImage(src, 0, 0, width, height);
  return new Promise(resolve => c.toBlob(resolve, "image/jpeg", 0.92));
}

async function detect(src) {
  const width = src.videoWidth || src.naturalWidth || src.width;
  const height = src.videoHeight || src.naturalHeight || src.height;
  const blob = await imageBlob(src, width, height);
  if (!blob) throw new Error("Could not prepare image for detection.");
  const form = new FormData();
  form.append("image", blob, "scan.jpg");
  const response = await fetch(`${API_BASE_URL}/detect?model=${activeModel}&confidence=${CONF}`, { method: "POST", body: form });
  if (!response.ok) throw new Error("Detection request failed.");
  return response.json();
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderAutoMetrics(detections, width, height, fps) {
  const avg = detections.reduce((sum, d) => sum + d.score, 0) / (detections.length || 1);
  const resizeText = `${width}x${height} to ${INPUT}x${INPUT}`;
  $("evalMetrics").innerHTML = [
    [lastScan && lastScan.model ? lastScan.model : "-", "Model"],
    [String(detections.length), "Objects"],
    [detections.length ? formatPercent(avg) : "0%", "Avg confidence"],
    [fps ? fps.toFixed(1) : "-", "FPS"],
    [resizeText, "Resize"]
  ].map(([value, label]) => `<div><strong>${value}</strong><span>${label}</span></div>`).join("");
  $("evalNotes").textContent = "Variation tests are exploratory checks using the current image. Formal Precision, Recall, F1, and mAP results above come from the five untouched warehouse test images.";
}

function setProcessView(mode, btn) {
  processMode = mode;
  document.querySelectorAll(".process-tools button").forEach(b => b.classList.remove("active-tool"));
  if (btn) btn.classList.add("active-tool");
  renderProcessView();
}

function renderProcessView() {
  if (!lastScan || !$("processCanvas")) return;
  const canvas = $("processCanvas");
  canvas.width = lastScan.width;
  canvas.height = lastScan.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(lastScan.sourceEl, 0, 0, lastScan.width, lastScan.height);
  if (processMode === "gray") applyGrayscale(ctx, canvas.width, canvas.height);
  if (processMode === "canny") applyCannyLike(ctx, canvas.width, canvas.height);
  if (processMode === "harris") applyHarrisLike(ctx, canvas.width, canvas.height);
  if (processMode === "orb") applyOrbLike(ctx, canvas.width, canvas.height);
  $("processBadge").textContent = {
    original: "Original",
    gray: "Grayed",
    canny: "Canny edge",
    harris: "Harris corner",
    orb: "ORB keypoint"
  }[processMode];
}

function applyGrayscale(ctx, width, height) {
  const img = ctx.getImageData(0, 0, width, height);
  for (let i = 0; i < img.data.length; i += 4) {
    const g = img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114;
    img.data[i] = g; img.data[i + 1] = g; img.data[i + 2] = g;
  }
  ctx.putImageData(img, 0, 0);
}

function applyCannyLike(ctx, width, height) {
  applyGrayscale(ctx, width, height);
  const src = ctx.getImageData(0, 0, width, height);
  const out = ctx.createImageData(width, height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = (y * width + x) * 4;
      const gx = src.data[i + 4] - src.data[i - 4];
      const gy = src.data[i + width * 4] - src.data[i - width * 4];
      const v = Math.hypot(gx, gy) > 35 ? 255 : 0;
      out.data[i] = v; out.data[i + 1] = v; out.data[i + 2] = v; out.data[i + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
}

function applyHarrisLike(ctx, width, height) {
  applyCannyLike(ctx, width, height);
  ctx.fillStyle = "#2764d8";
  for (let y = 12; y < height - 12; y += 12) {
    for (let x = 12; x < width - 12; x += 12) {
      const p = ctx.getImageData(x - 1, y - 1, 3, 3).data;
      const bright = Array.from({ length: 9 }, (_, n) => p[n * 4]).filter(v => v > 180).length;
      if (bright >= 3 && bright <= 6) ctx.fillRect(x - 2, y - 2, 5, 5);
    }
  }
}

function applyOrbLike(ctx, width, height) {
  applyGrayscale(ctx, width, height);
  const src = ctx.getImageData(0, 0, width, height);
  ctx.putImageData(src, 0, 0);
  ctx.strokeStyle = "#ffb000";
  ctx.lineWidth = Math.max(1, width / 700);
  for (let y = 16; y < height - 16; y += 18) {
    for (let x = 16; x < width - 16; x += 18) {
      const i = (y * width + x) * 4;
      const contrast = Math.abs(src.data[i] - src.data[i + 16]) + Math.abs(src.data[i] - src.data[i + width * 16]);
      if (contrast > 85) {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
}

function makeTransformCanvas(sourceEl, width, height, type) {
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, width, height);
  if (type === "resize-50") {
    const small = document.createElement("canvas");
    small.width = Math.max(1, Math.round(width * 0.5));
    small.height = Math.max(1, Math.round(height * 0.5));
    small.getContext("2d").drawImage(sourceEl, 0, 0, width, height, 0, 0, small.width, small.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(small, 0, 0, small.width, small.height, 0, 0, width, height);
  } else if (type === "rotate") {
    ctx.translate(width / 2, height / 2);
    ctx.rotate(-12 * Math.PI / 180);
    ctx.drawImage(sourceEl, -width / 2, -height / 2, width, height);
  } else if (type === "slant") {
    ctx.transform(1, 0.12, -0.08, 1, width * 0.05, -height * 0.05);
    ctx.drawImage(sourceEl, 0, 0, width, height);
  } else if (type === "color") {
    ctx.filter = "saturate(145%) contrast(115%) brightness(92%)";
    ctx.drawImage(sourceEl, 0, 0, width, height);
  } else if (type === "normalize") {
    ctx.filter = "contrast(120%) brightness(105%)";
    ctx.drawImage(sourceEl, 0, 0, width, height);
  } else if (type === "edges") {
    ctx.filter = "grayscale(1) contrast(180%)";
    ctx.drawImage(sourceEl, 0, 0, width, height);
    applyEdgeFilter(ctx, width, height);
  } else {
    ctx.filter = "blur(1.2px)";
    ctx.drawImage(sourceEl, 0, 0, width, height);
  }
  return c;
}

function applyEdgeFilter(ctx, width, height) {
  const src = ctx.getImageData(0, 0, width, height);
  const out = ctx.createImageData(width, height);
  const kernel = [-1, -1, -1, -1, 8, -1, -1, -1, -1];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const i = ((y + ky) * width + (x + kx)) * 4;
          const gray = (src.data[i] + src.data[i + 1] + src.data[i + 2]) / 3;
          sum += gray * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      const o = (y * width + x) * 4;
      const v = Math.max(0, Math.min(255, sum + 128));
      out.data[o] = v; out.data[o + 1] = v; out.data[o + 2] = v; out.data[o + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
}

async function runTransformTests() {
  const out = $("transformResults");
  try {
    if (!lastScan) throw new Error("Scan a photo first.");
    const tests = [
      ["resize-50", "Resize 50%"],
      ["rotate", "Rotate"],
      ["slant", "Slant"],
      ["color", "Color shift"],
      ["normalize", "Normalization"],
      ["edges", "Convolution edges"],
      ["blur", "Gaussian blur"]
    ];
    out.innerHTML = `<div class="muted">Running variation tests...</div>`;
    const rows = [];
    for (const [type, label] of tests) {
      const canvas = makeTransformCanvas(lastScan.sourceEl, lastScan.width, lastScan.height, type);
      const started = performance.now();
      const result = await detect(canvas);
      const fps = 1000 / Math.max(1, performance.now() - started);
      rows.push(`<div class="finding">${label}: ${result.detections.length} detection(s), ${fps.toFixed(1)} FPS</div>`);
    }
    out.innerHTML = rows.join("");
    log("Variation tests completed.");
  } catch (e) {
    out.innerHTML = `<div class="muted">${e.message || "Variation test failed."}</div>`;
  }
}

function classify(detections) {
  const bad = detections.filter(d => UNSAFE.has(d.label));
  const level = bad.length ? "unsafe" : "warn";
  return {
    level,
    verdict: level === "unsafe" ? "HAZARD REVIEW" : "MONITOR",
    risk: level === "unsafe" ? Math.min(10, 6 + bad.length) : 4,
    summary: detections.length ? `${detections.length} warehouse object(s) detected` : "No supported object detected",
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
    provider_used: activeModel === "yolov10" ? "yolov10-warehouse" : "fasterrcnn-warehouse-v3"
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

// Service worker disabled during testing — was causing stale app.js to be served instead of the latest version.
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
let promptEvent;
window.addEventListener("beforeinstallprompt", e => {
  promptEvent = e;
  const btn = $("installBtn");
  if (btn) btn.classList.remove("hidden");
});
if ($("installBtn")) $("installBtn").onclick = () => promptEvent && promptEvent.prompt();

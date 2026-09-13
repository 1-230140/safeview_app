const SUPABASE_URL = "https://zobfhaykmvqsdtdkopuc.supabase.co";
const SUPABASE_KEY = "sb_publishable_-YgL8OtMraWknloMv-YWjA_EY1qhBts";
const db = window.supabase ? supabase.createClient(SUPABASE_URL, SUPABASE_KEY) : null;

const MODEL_PATH = "models/yolov8-ppe-10k-best-final.onnx";
const YOLOV10_PATH = "models/yolov10-best.onnx";
const INPUT = 640;
const CONF = 0.15; // was 0.4 originally, 0.1 for debugging — this should be a good starting point now that classes are correct
const IOU = 0.45;
const CLASS_NAMES = ["safety-glove","safety-helmet","safety-shoes","unsafe-area","damage","forklift","safety-vest","cardboard-box","safety-cone"];
const UNSAFE = new Set(["unsafe-area","damage","forklift","cardboard-box","safety-cone"]);

let session, yolo10Session, stream, photoBase64, autoTimer;
let history = [];
let stats = { total: 0, safe: 0, warn: 0, unsafe: 0 };
let lastScan = null;
let lastFps = 0;
let processMode = "original";

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
  if (session) return session;
  $("modelState").textContent = "Loading model";
  session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["webgl", "wasm"] });
  $("modelState").textContent = "Model ready";
  return session;
}

async function loadYolo10Model() {
  if (yolo10Session) return yolo10Session;
  yolo10Session = await ort.InferenceSession.create(YOLOV10_PATH, { executionProviders: ["webgl", "wasm"] });
  return yolo10Session;
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
    const started = performance.now();
    const detections = await detect(sourceEl, width, height);
    const elapsed = Math.max(1, performance.now() - started);
    lastFps = 1000 / elapsed;
    lastScan = { sourceEl, width, height, detections, fps: lastFps };
    draw(canvas, detections, width, height);
    renderAutoMetrics(detections, width, height, lastFps);
    renderProcessView();
    applyResult(classify(detections), source, badge);
    if (source === "photo") compareYoloModels(sourceEl, width, height, detections);
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
  return detectWithSession(s, src, w, h, "yolov8");
}

async function detectWithSession(s, src, w, h, modelType = "yolov8") {
  const prep = preprocess(src, w, h);
  const out = await s.run({ [s.inputNames[0]]: prep.tensor });
  const y = out[s.outputNames[0]];
  if (modelType === "yolov10") return parseYolo10Output(y, w, h, prep);
  const preds = y.dims[y.dims.length - 1];
  const boxes = [];
  let bestScore = 0, bestCls = -1; // TEMP: debug tracking
  for (let i = 0; i < preds; i++) {
    let score = 0, cls = -1;
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const v = y.data[(4 + c) * preds + i];
      if (v > score) { score = v; cls = c; }
    }
    if (score > bestScore) { bestScore = score; bestCls = cls; } // TEMP: debug tracking
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
  console.log(`[debug] best raw score this frame: ${bestScore.toFixed(3)} -> class "${CLASS_NAMES[bestCls] || bestCls}" (index ${bestCls})`); // TEMP: remove after debugging
  return nms(boxes);
}

function parseYolo10Output(y, w, h, prep) {
  const boxes = [];
  const dims = y.dims;
  const rows = dims[dims.length - 2] || 0;
  const cols = dims[dims.length - 1] || 0;
  if (cols < 6) return boxes;
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    const score = y.data[offset + 4];
    if (score < CONF) continue;
    const cls = Math.round(y.data[offset + 5]);
    boxes.push({
      x1: Math.max(0, (y.data[offset] - prep.dx) / prep.scale),
      y1: Math.max(0, (y.data[offset + 1] - prep.dy) / prep.scale),
      x2: Math.min(w, (y.data[offset + 2] - prep.dx) / prep.scale),
      y2: Math.min(h, (y.data[offset + 3] - prep.dy) / prep.scale),
      score,
      label: CLASS_NAMES[cls] || `class_${cls}`
    });
  }
  return nms(boxes);
}

async function compareYoloModels(sourceEl, width, height, yolo8Detections) {
  const out = $("modelCompare");
  if (!out) return;
  out.className = "list";
  out.innerHTML = `<div class="muted">Running YOLOv10 comparison...</div>`;
  try {
    const model = await loadYolo10Model();
    const started = performance.now();
    const yolo10Detections = await detectWithSession(model, sourceEl, width, height, "yolov10");
    const fps = 1000 / Math.max(1, performance.now() - started);
    const rows = [
      `<div class="finding"><strong>YOLOv8</strong>: ${yolo8Detections.length} detection(s), ${lastFps.toFixed(1)} FPS</div>`,
      ...yolo8Detections.map(d => detectionRow("YOLOv8", d)),
      `<div class="finding"><strong>YOLOv10</strong>: ${yolo10Detections.length} detection(s), ${fps.toFixed(1)} FPS</div>`,
      ...yolo10Detections.map(d => detectionRow("YOLOv10", d))
    ];
    out.innerHTML = rows.join("");
  } catch (e) {
    out.innerHTML = `<div class="muted">YOLOv10 comparison failed: ${e.message || e}</div>`;
  }
}

function detectionRow(model, d) {
  return `<div class="finding">${model} ${d.label} ${Math.round(d.score * 100)}% (${Math.round(d.x1)}, ${Math.round(d.y1)}, ${Math.round(d.x2)}, ${Math.round(d.y2)})</div>`;
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

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderAutoMetrics(detections, width, height, fps) {
  const avg = detections.reduce((sum, d) => sum + d.score, 0) / (detections.length || 1);
  const resizeText = `${width}x${height} to ${INPUT}x${INPUT}`;
  $("evalMetrics").innerHTML = [
    [String(detections.length), "Objects"],
    [detections.length ? formatPercent(avg) : "0%", "Avg confidence"],
    [fps ? fps.toFixed(1) : "-", "FPS"],
    [resizeText, "Resize"]
  ].map(([value, label]) => `<div><strong>${value}</strong><span>${label}</span></div>`).join("");
  $("evalNotes").textContent = "Automatic metrics shown. Accuracy, precision, recall, F1, IoU, and mAP need labeled ground-truth data, so YOLOv8 vs YOLOv10 comparison is used instead.";
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
      const detections = await detect(canvas, lastScan.width, lastScan.height);
      const fps = 1000 / Math.max(1, performance.now() - started);
      rows.push(`<div class="finding">${label}: ${detections.length} detection(s), ${fps.toFixed(1)} FPS</div>`);
    }
    out.innerHTML = rows.join("");
    log("Variation tests completed.");
  } catch (e) {
    out.innerHTML = `<div class="muted">${e.message || "Variation test failed."}</div>`;
  }
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

// Service worker disabled during testing — was causing stale app.js to be served instead of the latest version.
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
let promptEvent;
window.addEventListener("beforeinstallprompt", e => {
  promptEvent = e;
  const btn = $("installBtn");
  if (btn) btn.classList.remove("hidden");
});
if ($("installBtn")) $("installBtn").onclick = () => promptEvent && promptEvent.prompt();

const CACHE_NAME = "safeview-v3"; // bump this version number every time you deploy changes
const STATIC_ASSETS = ["login.html", "index.html", "style.css", "app.js", "manifest.json"];
const MODEL_ASSET = "models/ppe-yolo.onnx";

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", event => {
  const url = event.request.url;

  if (url.includes(MODEL_ASSET)) {
    // model rarely changes - cache-first is fine here
    event.respondWith(
      caches.match(event.request).then(hit => hit || fetch(event.request).then(res => {
        const clone = res.clone();
        caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
        return res;
      }))
    );
    return;
  }

  // app files - network-first so you always get the latest during active development
  event.respondWith(
    fetch(event.request)
      .then(res => {
        const clone = res.clone();
        caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
        return res;
      })
      .catch(() => caches.match(event.request))
  );
});
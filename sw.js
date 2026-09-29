const CACHE_NAME = "safeview-v4"; // bump this version number every time you deploy changes
const STATIC_ASSETS = ["login.html", "index.html", "style.css", "app.js", "manifest.json"];

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

  // never intercept non-GET requests (POST/PUT/DELETE) - Cache API can't store them,
  // and API calls like Supabase inserts should always go straight to network
  if (event.request.method !== "GET") {
    return; // let the browser handle it normally, don't touch it
  }

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

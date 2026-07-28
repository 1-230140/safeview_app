self.addEventListener("install", event => {
  event.waitUntil(caches.open("safeview-v1").then(cache => cache.addAll([
    "login.html", "index.html", "style.css", "app.js", "manifest.json"
  ])));
});
self.addEventListener("fetch", event => {
  event.respondWith(caches.match(event.request).then(hit => hit || fetch(event.request)));
});

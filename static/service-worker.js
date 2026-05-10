const CACHE_NAME = "mini-jarvis-v1";

const STATIC_ASSETS = [
    "/",
    "/static/manifest.json",
    "/static/icon-192.png",
    "/static/icon-512.png"
];

self.addEventListener("install", function (event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function (cache) {
            return cache.addAll(STATIC_ASSETS);
        })
    );

    self.skipWaiting();
});

self.addEventListener("activate", function (event) {
    event.waitUntil(
        caches.keys().then(function (cacheNames) {
            return Promise.all(
                cacheNames.map(function (cacheName) {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );

    self.clients.claim();
});

self.addEventListener("fetch", function (event) {
    const request = event.request;

    if (request.method !== "GET") {
        return;
    }

    event.respondWith(
        fetch(request)
            .then(function (response) {
                return response;
            })
            .catch(function () {
                return caches.match(request).then(function (cachedResponse) {
                    return cachedResponse || caches.match("/");
                });
            })
    );
});
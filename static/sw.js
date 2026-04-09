// PikoLab Service Worker
const CACHE_NAME = 'pikolab-v1';

// Ressources statiques à mettre en cache pour le démarrage
const STATIC_ASSETS = [
  '/app/static/manifest.json',
  '/app/static/icon.svg',
  '/app/static/icon-192.png',
  '/app/static/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      // On ignore les erreurs d'assets manquants (icônes PNG optionnelles)
      return Promise.allSettled(
        STATIC_ASSETS.map(url => cache.add(url).catch(() => {}))
      );
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Stratégie : Network First pour les requêtes dynamiques Streamlit,
// Cache First pour les assets statiques
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Assets statiques → Cache First
  if (url.pathname.startsWith('/app/static/')) {
    event.respondWith(
      caches.match(event.request).then(
        cached => cached || fetch(event.request).then(response => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          return response;
        })
      )
    );
    return;
  }

  // Tout le reste (app Streamlit) → Network Only
  // Streamlit nécessite le serveur Python, pas de cache applicatif
});

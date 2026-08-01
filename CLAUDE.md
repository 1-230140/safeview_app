# SafeView PPE Scanner

Client-side PWA that runs a YOLOv8 object-detection model in the browser to flag
missing/damaged PPE (helmets, vests, gloves, safety lines) from a live camera feed
or an uploaded photo. No backend server — everything runs in the browser, with
Supabase used only as a hosted Postgres + auth data store.

## Stack

- Plain HTML/CSS/JS — no bundler, no framework, no npm build step
- **onnxruntime-web** (CDN) — runs `models/ppe-yolo.onnx` (YOLOv8, exported ONNX, 640×640 input) on the `wasm` execution provider
- **@supabase/supabase-js** (CDN) — stores scans/findings, looks up users for login
- Service worker (`sw.js`) + `manifest.json` — installable PWA, caches the app shell only (not the model)

## Files

| File | Purpose |
|---|---|
| `login.html` | Sign-in screen |
| `index.html` | Main scanner UI (camera + photo tabs, findings panel) |
| `app.js` | All application logic — auth, camera, inference, drawing, Supabase I/O, CSV/JSON export |
| `style.css` | All styling, CSS custom properties in `:root` |
| `models/ppe-yolo.onnx` | The detection model (~12MB, binary — don't open/edit) |
| `supabase.sql` | DB schema: `users`, `scans`, `findings` tables + RLS policies |
| `sw.js` | Service worker, app-shell caching only |

There is no build/lint/test tooling in this repo — it's edit-and-refresh.
Serve locally with any static file server (camera access needs `https://` or `localhost`),
e.g. `python3 -m http.server`.

## Key conventions

- `CLASS_NAMES` and `UNSAFE` (the set of class names treated as a violation) live at the
  top of `app.js`. If the model is retrained, `CLASS_NAMES` **must** match the new
  `data.yaml` names order exactly, or detections will be mislabeled.
- Detection pipeline: `preprocess` (letterbox to 640×640) → `detect` (run ONNX session,
  decode boxes, filter by `CONF`, run `nms`) → `classify` (unsafe if any `UNSAFE` label
  present) → `draw` (canvas overlay) → `applyResult` (updates UI + stats) → `saveScan`
  (writes to Supabase).
- Auth is a flat `sessionStorage` flag (`sv_user`) checked at the top of `app.js` on
  every page load — there's no route guarding beyond that.
- `$(id)` is a shorthand for `document.getElementById(id)`, used everywhere instead of querySelector.

## Known issues / things to be careful about

- **`supabase.sql` grants `anon` SELECT on `public.users`**, including `password_hash`.
  Combined with the client-side login check, this means the publishable Supabase key
  can be used to read every user's password hash directly. Don't extend this pattern to
  other tables without adding real RLS scoping (e.g. `auth.uid() = user_id`) — Claude
  should flag this rather than replicate it if asked to add new tables.
- The `admin` / `safeview2025` credential is a hardcoded fallback in `app.js` (`login()`),
  independent of the Supabase `users` table. Treat it as a known default, not something
  to silently remove unless asked.
- `SUPABASE_URL` / `SUPABASE_KEY` in `app.js` are the public/anon key — expected to be
  client-visible, not a leak by itself.
- The ONNX model file is binary; don't attempt to read/diff/edit it as text.

## When making changes

- Keep everything framework-free unless explicitly asked to introduce a build step.
- If changing detection classes or thresholds (`CONF`, `IOU`), keep `CLASS_NAMES` in sync
  with the model actually in `models/ppe-yolo.onnx`.
- If touching `supabase.sql`, preserve RLS being *enabled*, and think about whether a
  policy should really be `anon`-readable before adding one.

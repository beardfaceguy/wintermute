# talkingHead — Playwright e2e

Hermetic end-to-end tests for the talkingHead web UI. The chat WebSocket
(`ws://localhost:8000/ws/chat`) and the TTS endpoints
(`/api/chat/speak{,/health}`) are stubbed inside the browser via
`installMocks(page, ...)` from [`helpers.ts`](./helpers.ts), so no backend,
vLLM, or network access is required.

## Running

```bash
# from talkingHead/frontend/
npm run test:e2e         # headless, default reporter
npm run test:e2e:ui      # interactive Playwright UI
npm run test:e2e -- --headed   # watch a real browser
```

Vite's dev server (`npm run dev` on `:5173`) is auto-started by Playwright
and reused if it's already up.

## Structure

- `helpers.ts` — `installMocks(page, { ws, tts })` injects a controllable
  `WebSocket` mock and routes the TTS HTTP endpoints. Always call before
  `page.goto("/")`.
- `chat.spec.ts` — text input, send button, Enter/Shift+Enter, multi-token
  streaming into a single bubble, sentinel never leaks into the UI.
- `tts.spec.ts` — speaker toggle visibility based on `/health`, mute
  persistence in `localStorage`, `/api/chat/speak` POST on assistant
  completion, mute suppression.

## Adding tests

```ts
import { expect, test } from "@playwright/test";
import { installMocks } from "./helpers";

test("…", async ({ page }) => {
  await installMocks(page, { ws: { autoReplyTokens: ["…"] } });
  await page.goto("/");
  // …
});
```

## Live-backend e2e (future)

For runs against a real backend (`uvicorn app.main:app` + Piper voice file
present), don't call `installMocks`. That mode is currently out of scope —
tracked under the Design C / sentence-streaming follow-up since it'll need
a richer WS protocol anyway.

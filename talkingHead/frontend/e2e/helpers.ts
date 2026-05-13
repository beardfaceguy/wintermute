import type { Page, Route } from "@playwright/test";

/**
 * Hermetic test harness for talkingHead.
 *
 * Vite serves the real frontend bundle on :5173, but the chat WebSocket
 * (ws://localhost:8000/ws/chat) and TTS health probe never exist in CI.
 * We replace the global `WebSocket` constructor with a controllable mock
 * that the test can drive via window.__mockWS, and we route() the
 * /api/chat/speak/* HTTP calls so no real backend is needed.
 *
 * Always call `installMocks(page, ...)` BEFORE `page.goto("/")`.
 */

export type MockWSConfig = {
  /** If true, the next assistant turn auto-streams these chunks back. */
  autoReplyTokens?: string[];
  /** Sentinel string the backend would emit on completion. */
  endOfStreamSentinel?: string;
  /** Delay between simulated tokens (ms) — keeps tests deterministic. */
  tokenIntervalMs?: number;
};

export type TtsConfig = {
  /** Whether /api/chat/speak/health reports the backend voice is available. */
  enabled?: boolean;
  /** If true, /api/chat/speak returns a tiny fake WAV. Otherwise 503. */
  speakOk?: boolean;
};

const DEFAULT_SENTINEL = "[[DONE]]";

/**
 * Inject the WS mock into the page before any app code runs and route the
 * TTS endpoints to in-memory fixtures.
 */
export async function installMocks(
  page: Page,
  opts: { ws?: MockWSConfig; tts?: TtsConfig } = {},
): Promise<void> {
  const wsCfg: Required<MockWSConfig> = {
    autoReplyTokens: opts.ws?.autoReplyTokens ?? ["Hello ", "there!"],
    endOfStreamSentinel: opts.ws?.endOfStreamSentinel ?? DEFAULT_SENTINEL,
    tokenIntervalMs: opts.ws?.tokenIntervalMs ?? 20,
  };

  await page.addInitScript((cfg: Required<MockWSConfig>) => {
    type Listener = (event: { data: string } | { type: string }) => void;

    type MockSocket = {
      readyState: number;
      onopen: ((e: Event) => void) | null;
      onmessage: ((e: { data: string }) => void) | null;
      onclose: ((e?: CloseEvent) => void) | null;
      onerror: ((e?: Event) => void) | null;
      send: (msg: string) => void;
      close: () => void;
      _listeners: Record<string, Listener[]>;
      addEventListener: (name: string, cb: Listener) => void;
      removeEventListener: (name: string, cb: Listener) => void;
    };

    const sentMessages: string[] = [];
    const sockets: MockSocket[] = [];

    function makeSocket(_url: string): MockSocket {
      const sock: MockSocket = {
        readyState: 0,
        onopen: null,
        onmessage: null,
        onclose: null,
        onerror: null,
        _listeners: {},
        addEventListener(name: string, cb: Listener) {
          (this._listeners[name] ??= []).push(cb);
        },
        removeEventListener(name: string, cb: Listener) {
          const arr = this._listeners[name];
          if (!arr) return;
          const i = arr.indexOf(cb);
          if (i >= 0) arr.splice(i, 1);
        },
        send(msg: string) {
          sentMessages.push(msg);
          const tokens = cfg.autoReplyTokens;
          if (!tokens || tokens.length === 0) return;
          let i = 0;
          const tick = () => {
            if (i >= tokens.length) {
              this.onmessage?.({ data: cfg.endOfStreamSentinel });
              return;
            }
            this.onmessage?.({ data: tokens[i] });
            i += 1;
            window.setTimeout(tick, cfg.tokenIntervalMs);
          };
          window.setTimeout(tick, cfg.tokenIntervalMs);
        },
        close() {
          this.readyState = 3;
          this.onclose?.();
        },
      };
      sockets.push(sock);
      // Open async so onopen handler attached after construction still fires.
      window.setTimeout(() => {
        sock.readyState = 1;
        sock.onopen?.(new Event("open"));
      }, 0);
      return sock;
    }

    const ctor = function (this: unknown, url: string) {
      return makeSocket(url);
    } as unknown as typeof WebSocket;
    (ctor as unknown as { CONNECTING: number }).CONNECTING = 0;
    (ctor as unknown as { OPEN: number }).OPEN = 1;
    (ctor as unknown as { CLOSING: number }).CLOSING = 2;
    (ctor as unknown as { CLOSED: number }).CLOSED = 3;
    (window as unknown as { WebSocket: typeof WebSocket }).WebSocket = ctor;

    // Expose a debug surface so tests can introspect what the app sent.
    (window as unknown as { __mockWS: unknown }).__mockWS = {
      sentMessages,
      sockets,
      pushToken: (text: string) => {
        sockets[sockets.length - 1]?.onmessage?.({ data: text });
      },
      complete: () => {
        sockets[sockets.length - 1]?.onmessage?.({
          data: cfg.endOfStreamSentinel,
        });
      },
    };
  }, wsCfg);

  const ttsCfg: Required<TtsConfig> = {
    enabled: opts.tts?.enabled ?? true,
    speakOk: opts.tts?.speakOk ?? true,
  };

  await page.route("**/api/chat/speak/health", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        enabled: ttsCfg.enabled,
        voice_path: "/fake/voice.onnx",
        length_scale: 1.0,
        noise_scale: 0.667,
        noise_w: 0.8,
        error: ttsCfg.enabled ? null : "disabled in test fixture",
      }),
    });
  });

  await page.route("**/api/chat/speak", async (route: Route) => {
    if (!ttsCfg.speakOk) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "TTS unavailable in test fixture" }),
      });
      return;
    }
    // Minimal but valid 44-byte RIFF/WAVE header + a few silent samples.
    const header = new Uint8Array([
      0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
      0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
      0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
      0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
    ]);
    await route.fulfill({
      status: 200,
      contentType: "audio/wav",
      body: Buffer.from(header),
    });
  });
}

/** Convenience: read the array of payloads the app sent over the mock WS. */
export async function getSentMessages(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const w = window as unknown as {
      __mockWS?: { sentMessages: string[] };
    };
    return w.__mockWS ? [...w.__mockWS.sentMessages] : [];
  });
}

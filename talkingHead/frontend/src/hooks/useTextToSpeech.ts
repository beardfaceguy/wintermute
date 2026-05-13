// src/hooks/useTextToSpeech.ts
//
// Design A (MVP): when an assistant turn finishes, POST the whole message
// text to the backend Piper endpoint, get back a WAV blob, and play it
// through a single reused <audio> element. Calling speak() again cancels
// any in-flight request and any currently playing audio so we never stack
// or overlap. Sentence-streaming (Design C) is tracked separately.

import { useCallback, useEffect, useRef, useState } from "react";
import rawconfig from "../../../../config/shared_api_config.json";
import { debugLog } from "../utils/debug";

const apiBase = (() => {
  const cfg = rawconfig as { web_interface: { scheme: string; host: string; port: number } };
  const host =
    cfg.web_interface.host === "localhost" &&
    typeof window !== "undefined" &&
    window.location.hostname !== "localhost"
      ? window.location.hostname
      : cfg.web_interface.host;
  return `${cfg.web_interface.scheme}://${host}:${cfg.web_interface.port}`;
})();

const speakUrl = `${apiBase}/api/chat/speak`;
const healthUrl = `${apiBase}/api/chat/speak/health`;

type UseTextToSpeechOptions = {
  enabled: boolean;
};

type UseTextToSpeechReturn = {
  available: boolean;
  speak: (text: string) => Promise<void>;
  stop: () => void;
};

export default function useTextToSpeech({
  enabled,
}: UseTextToSpeechOptions): UseTextToSpeechReturn {
  const [available, setAvailable] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const currentBlobUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // One-shot health probe. Failure is non-fatal; available stays false and
  // the speaker toggle button can hide itself.
  useEffect(() => {
    let cancelled = false;
    fetch(healthUrl)
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => {
        if (cancelled) return;
        setAvailable(Boolean(body && body.enabled));
      })
      .catch(() => {
        if (!cancelled) setAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Tear down audio + blob URL on unmount.
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
      }
      if (currentBlobUrlRef.current) {
        URL.revokeObjectURL(currentBlobUrlRef.current);
        currentBlobUrlRef.current = null;
      }
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
    };
  }, []);

  const stop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (currentBlobUrlRef.current) {
      URL.revokeObjectURL(currentBlobUrlRef.current);
      currentBlobUrlRef.current = null;
    }
  }, []);

  const speak = useCallback(
    async (text: string): Promise<void> => {
      const trimmed = text.trim();
      if (!trimmed || !enabled || !available) return;
      stop();

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      try {
        const resp = await fetch(speakUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: trimmed }),
          signal: ctrl.signal,
        });
        if (!resp.ok) {
          debugLog("useTextToSpeech: speak failed:", resp.status);
          return;
        }
        const blob = await resp.blob();
        if (ctrl.signal.aborted) return;
        const url = URL.createObjectURL(blob);
        currentBlobUrlRef.current = url;

        if (!audioRef.current) {
          audioRef.current = new Audio();
        }
        const audio = audioRef.current;
        audio.src = url;
        // Older browsers reject without a play() promise; ignore rejections
        // (autoplay restrictions etc) — we'll just not hear this one turn.
        const playPromise = audio.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch((err) => {
            debugLog("useTextToSpeech: audio.play rejected:", err);
          });
        }
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") return;
        debugLog("useTextToSpeech: error:", err);
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
      }
    },
    [available, enabled, stop],
  );

  return { available, speak, stop };
}

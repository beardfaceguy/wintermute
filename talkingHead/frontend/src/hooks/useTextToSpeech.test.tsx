import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import useTextToSpeech from "./useTextToSpeech";

class MockAudio {
  src = "";
  currentTime = 0;
  paused = true;
  pause = vi.fn(() => {
    this.paused = true;
  });
  play = vi.fn(() => {
    this.paused = false;
    return Promise.resolve();
  });
}

let lastAudio: MockAudio | null = null;

beforeEach(() => {
  lastAudio = null;
  vi.stubGlobal(
    "Audio",
    vi.fn(() => {
      lastAudio = new MockAudio();
      return lastAudio;
    }),
  );
  // Patch URL static methods in place rather than replacing the URL global —
  // the hook's cleanup effect runs after vitest tears down test fixtures, and
  // we want revokeObjectURL to keep working through that teardown.
  (URL as unknown as { createObjectURL: (b: Blob) => string }).createObjectURL =
    vi.fn(() => "blob:mock-url");
  (URL as unknown as { revokeObjectURL: (u: string) => void }).revokeObjectURL =
    vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function mockHealth(enabled: boolean) {
  const fetchMock = vi.fn(async (url: string) => {
    if (url.includes("/api/chat/speak/health")) {
      return {
        ok: true,
        json: async () => ({ enabled }),
      } as unknown as Response;
    }
    return {
      ok: true,
      blob: async () => new Blob(["wav-bytes"], { type: "audio/wav" }),
    } as unknown as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("useTextToSpeech", () => {
  it("probes the health endpoint and reports availability", async () => {
    mockHealth(true);
    const { result } = renderHook(() => useTextToSpeech({ enabled: true }));
    await waitFor(() => expect(result.current.available).toBe(true));
  });

  it("reports unavailable when backend says disabled", async () => {
    mockHealth(false);
    const { result } = renderHook(() => useTextToSpeech({ enabled: true }));
    await waitFor(() => expect(result.current.available).toBe(false));
  });

  it("speak() is a no-op when disabled by the user", async () => {
    const fetchMock = mockHealth(true);
    const { result } = renderHook(() => useTextToSpeech({ enabled: false }));
    await waitFor(() => expect(result.current.available).toBe(true));

    fetchMock.mockClear();
    await act(async () => {
      await result.current.speak("hello");
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("speak() POSTs the text and starts audio playback", async () => {
    const fetchMock = mockHealth(true);
    const { result } = renderHook(() => useTextToSpeech({ enabled: true }));
    await waitFor(() => expect(result.current.available).toBe(true));

    fetchMock.mockClear();
    await act(async () => {
      await result.current.speak("hello there");
    });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/chat/speak"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ text: "hello there" }),
      }),
    );
    expect(lastAudio).not.toBeNull();
    expect(lastAudio!.play).toHaveBeenCalled();
    expect(lastAudio!.src).toBe("blob:mock-url");
  });

  it("speak() ignores empty / whitespace input", async () => {
    const fetchMock = mockHealth(true);
    const { result } = renderHook(() => useTextToSpeech({ enabled: true }));
    await waitFor(() => expect(result.current.available).toBe(true));

    fetchMock.mockClear();
    await act(async () => {
      await result.current.speak("   ");
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("stop() pauses audio and revokes the blob URL", async () => {
    mockHealth(true);
    const { result } = renderHook(() => useTextToSpeech({ enabled: true }));
    await waitFor(() => expect(result.current.available).toBe(true));

    await act(async () => {
      await result.current.speak("hi");
    });
    act(() => {
      result.current.stop();
    });
    expect(lastAudio!.pause).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
  });
});

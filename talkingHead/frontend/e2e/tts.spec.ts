import { expect, test } from "@playwright/test";
import { installMocks } from "./helpers";

test.describe("TTS — speaker toggle (Design A)", () => {
  test("speaker toggle is rendered when the backend reports enabled", async ({
    page,
  }) => {
    await installMocks(page, { tts: { enabled: true, speakOk: true } });
    await page.goto("/");

    await expect(
      page.getByRole("button", { name: /Mute voice output/ }),
    ).toBeVisible();
  });

  test("speaker toggle is hidden when /api/chat/speak/health says disabled", async ({
    page,
  }) => {
    await installMocks(page, { tts: { enabled: false } });
    // Wait for the health probe to fire so we know the hook ran. Using
    // waitForResponse instead of networkidle because vite's HMR socket
    // keeps the page from ever going fully idle in dev mode.
    const healthCall = page.waitForResponse((r) =>
      r.url().includes("/api/chat/speak/health"),
    );
    await page.goto("/");
    await healthCall;

    await expect(
      page.getByRole("button", { name: /voice output/ }),
    ).toHaveCount(0);
  });

  test("clicking the toggle persists the muted preference to localStorage", async ({
    page,
  }) => {
    await installMocks(page, { tts: { enabled: true } });
    await page.goto("/");

    const toggle = page.getByRole("button", { name: /Mute voice output/ });
    await toggle.click();

    // Now muted: button label flips and aria-pressed is false.
    const muted = page.getByRole("button", { name: /Enable voice output/ });
    await expect(muted).toBeVisible();
    await expect(muted).toHaveAttribute("aria-pressed", "false");

    const stored = await page.evaluate(() =>
      window.localStorage.getItem("tts_enabled"),
    );
    expect(stored).toBe("false");
  });

  test("muted preference survives a page reload", async ({ page }) => {
    await installMocks(page, { tts: { enabled: true } });
    await page.goto("/");
    await page.evaluate(() =>
      window.localStorage.setItem("tts_enabled", "false"),
    );

    await page.reload();
    await expect(
      page.getByRole("button", { name: /Enable voice output/ }),
    ).toBeVisible();
  });

  test("speaker toggle posts to /api/chat/speak after an assistant turn (when enabled)", async ({
    page,
  }) => {
    await installMocks(page, {
      tts: { enabled: true, speakOk: true },
      ws: { autoReplyTokens: ["good ", "morning."], tokenIntervalMs: 5 },
    });
    const speakRequest = page.waitForRequest((req) =>
      req.url().endsWith("/api/chat/speak") && req.method() === "POST",
    );

    await page.goto("/");
    await page.getByPlaceholder("Type a message...").fill("hi");
    await page.getByRole("button", { name: "Send" }).click();

    const req = await speakRequest;
    expect(JSON.parse(req.postData() ?? "{}")).toEqual({
      text: "good morning.",
    });
  });

  test("muting before assistant completion suppresses the /api/chat/speak call", async ({
    page,
  }) => {
    await installMocks(page, {
      tts: { enabled: true, speakOk: true },
      ws: { autoReplyTokens: ["silence."], tokenIntervalMs: 5 },
    });
    let speakCalled = false;
    page.on("request", (req) => {
      if (req.url().endsWith("/api/chat/speak") && req.method() === "POST") {
        speakCalled = true;
      }
    });

    await page.goto("/");
    await page.getByRole("button", { name: /Mute voice output/ }).click();
    await page.getByPlaceholder("Type a message...").fill("hi");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.locator(".chat-assistant").last()).toHaveText("silence.");
    // Allow any latent fetch a moment to fire (it shouldn't).
    await page.waitForTimeout(200);
    expect(speakCalled).toBe(false);
  });
});

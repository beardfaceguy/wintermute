import { expect, test } from "@playwright/test";
import { getSentMessages, installMocks } from "./helpers";

test.describe("Chat — text flow", () => {
  test("renders the input, send button, and mic toggle on load", async ({
    page,
  }) => {
    await installMocks(page);
    await page.goto("/");

    await expect(page.getByPlaceholder("Type a message...")).toBeVisible();
    await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Start Recording|Stop Recording/ }),
    ).toBeVisible();
  });

  test("Send button posts the typed message over the WS", async ({ page }) => {
    await installMocks(page, {
      ws: { autoReplyTokens: ["pong"], tokenIntervalMs: 5 },
    });
    await page.goto("/");

    const input = page.getByPlaceholder("Type a message...");
    await input.fill("hello world");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.locator(".chat-user").last()).toHaveText("hello world");
    await expect(page.locator(".chat-assistant").last()).toHaveText("pong");
    await expect(input).toHaveValue("");

    const sent = await getSentMessages(page);
    expect(sent).toHaveLength(1);
    expect(JSON.parse(sent[0])).toEqual({ message: "hello world" });
  });

  test("Enter (without Shift) sends; Shift+Enter inserts a newline", async ({
    page,
  }) => {
    await installMocks(page, {
      ws: { autoReplyTokens: ["ack"], tokenIntervalMs: 5 },
    });
    await page.goto("/");

    const input = page.getByPlaceholder("Type a message...");
    await input.fill("line1");
    await input.press("Shift+Enter");
    await expect(input).toHaveValue(/^line1\n?$/);

    await input.fill("send me");
    await input.press("Enter");
    await expect(page.locator(".chat-user").last()).toHaveText("send me");
    await expect(input).toHaveValue("");
  });

  test("empty messages are not sent", async ({ page }) => {
    await installMocks(page);
    await page.goto("/");

    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.locator(".chat-user")).toHaveCount(0);
    expect(await getSentMessages(page)).toEqual([]);
  });

  test("multi-token streams accumulate into a single assistant bubble", async ({
    page,
  }) => {
    await installMocks(page, {
      ws: {
        autoReplyTokens: ["The ", "quick ", "brown ", "fox."],
        tokenIntervalMs: 5,
      },
    });
    await page.goto("/");

    await page.getByPlaceholder("Type a message...").fill("hi");
    await page.getByRole("button", { name: "Send" }).click();

    await expect(page.locator(".chat-assistant").last()).toHaveText(
      "The quick brown fox.",
    );
    // Only one assistant bubble — tokens shouldn't create extras.
    await expect(page.locator(".chat-assistant")).toHaveCount(1);
  });

  test("end-of-stream sentinel does not show up in the rendered text", async ({
    page,
  }) => {
    await installMocks(page, {
      ws: { autoReplyTokens: ["done."], tokenIntervalMs: 5 },
    });
    await page.goto("/");

    await page.getByPlaceholder("Type a message...").fill("ping");
    await page.getByRole("button", { name: "Send" }).click();

    const last = page.locator(".chat-assistant").last();
    await expect(last).toHaveText("done.");
    await expect(last).not.toContainText("[[DONE]]");
  });
});

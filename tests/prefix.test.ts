import { describe, expect, it } from "vitest";

import { prefixPrompt } from "@/lib/prefix";

describe("prefixPrompt", () => {
  it("adds the hopper prefix to a raw prompt", () => {
    expect(prefixPrompt("a diner at dawn")).toBe("hopper style a diner at dawn");
  });

  it("does not duplicate an existing hopper prefix", () => {
    expect(prefixPrompt("hopper style, empty theater seats")).toBe(
      "hopper style empty theater seats",
    );
  });
});

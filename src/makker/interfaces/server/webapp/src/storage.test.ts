import { afterEach, describe, expect, it, vi } from "vitest";

import { stageSize } from "./canvas/stageSize";
import { readStored, writeStored } from "./storage";

describe("storage", () => {
  afterEach(() => vi.restoreAllMocks());

  it("is read as empty when the browser refuses it", () => {
    // A cross-site frame with storage blocked throws on any access.
    const denied = () => {
      throw new DOMException("denied", "SecurityError");
    };
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(denied);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(denied);

    expect(readStored("local", "ja_sidebar")).toBeNull();
    expect(() => writeStored("session", "ja_updated", "1")).not.toThrow();
    expect(() => stageSize()).not.toThrow();
  });
});

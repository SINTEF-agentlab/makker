import { describe, expect, it } from "vitest";

import { argPreview, asObject, listingNote, shortPath, summarizeArgs, toolNote, unitNote } from "./toolPolicy";

describe("argPreview", () => {
  it("prefers a description, then known keys, then the first scalar value", () => {
    expect(argPreview({ code: "line1\nline2", description: "Run the base case" })).toBe("Run the base case");
    expect(argPreview({ code: "line1\nline2" })).toBe("line1");
    expect(argPreview({ command: "ls -la" })).toBe("ls -la");
    expect(argPreview({ pattern: "setup_well", path: "/a/b/c" })).toBe("setup_well");
    expect(argPreview({ anything: "value" })).toBe("value");
  });

  it("shows a path as its last two segments", () => {
    expect(argPreview({ file_path: "/Users/me/pkg/examples/wells_intro.jl" })).toBe("examples/wells_intro.jl");
    expect(shortPath("a.jl")).toBe("a.jl");
  });

  it("skips object-valued first args and empty args", () => {
    expect(argPreview({ nested: { a: 1 } })).toBe("");
    expect(argPreview(null)).toBe("");
  });
});

describe("notes", () => {
  it("counts non-blank lines", () => {
    expect(unitNote("a\nb\nc", "line")).toBe("3 lines");
    expect(unitNote("only", "match", "matches")).toBe("1 match");
  });
  it("counts quoted entries in an ls repr", () => {
    expect(listingNote("['a/', 'b/', 'c.jl']")).toBe("3 entries");
    expect(listingNote("['solo/']")).toBe("1 entry");
    expect(listingNote("[]")).toBe("0 entries");
    // A filename with an apostrophe is repr'd with double quotes; the count must
    // not be thrown off by the apostrophe inside it.
    expect(listingNote("[\"it's.jl\", 'b/']")).toBe("2 entries");
  });
  it("says what each tool's result amounts to", () => {
    expect(toolNote("read_file", {}, "@@ lines 1-74 of 74 @@\na")).toBe("74 lines");
    expect(toolNote("read_file", {}, "@@ lines 101-200 of 222 | next offset 200 @@\na")).toBe("lines 101–200");
    // A file read as an image has no text to count.
    expect(toolNote("read_file", { file_path: "plot.png" }, "")).toBe("");
    expect(toolNote("grep", {}, "/a.jl:\n  12: x\n  40: y")).toBe("2 matches");
    expect(toolNote("grep", {}, "/a.jl\n/b.jl")).toBe("2 files");
    expect(toolNote("grep", {}, "No matches found")).toBe("no matches");
    expect(toolNote("glob", {}, "['/a.jl', '/b.jl']")).toBe("2 files");
    expect(toolNote("glob", {}, "No files found")).toBe("0 files");
    expect(toolNote("write_file", { content: "a\nb\nc" }, "Updated file")).toBe("3 lines");
    expect(toolNote("edit_file", { old_string: "a", new_string: "b\nc" }, "ok")).toBe("+2 −1");
    expect(toolNote("record_attempt", {}, "attempt #2 (parent #1) · rmse=0.1 · id=abc")).toBe("attempt #2 (parent #1) · rmse=0.1");
    expect(toolNote("run_julia", { code: "1" }, "2")).toBe("");
  });
});

describe("asObject", () => {
  it("accepts objects and JSON strings, rejects arrays and junk", () => {
    expect(asObject({ a: 1 })).toEqual({ a: 1 });
    expect(asObject('{"b":2}')).toEqual({ b: 2 });
    expect(asObject("[1,2]")).toBeNull();
    expect(asObject("nope")).toBeNull();
    expect(asObject(42)).toBeNull();
  });
});

describe("summarizeArgs", () => {
  it("renders key: value lines", () => {
    expect(summarizeArgs({ a: "x", b: 2 })).toBe("a: x\nb: 2");
  });
});

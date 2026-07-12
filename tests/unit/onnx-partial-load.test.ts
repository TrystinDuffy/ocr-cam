import { describe, it, expect, vi } from "vitest";
import { OcrError } from "../../src/types/errors";

const detectorSession = vi.hoisted(() => ({ inputNames: ["in"], outputNames: ["out"] })) as any;

const releaseSession = vi.hoisted(() => vi.fn(async () => {}));
const loadOrt = vi.hoisted(() => vi.fn(async () => ({} as any)));
const loadSession = vi.hoisted(() =>
  vi.fn(async (url: string) => {
    if (url === "/det") return detectorSession;
    throw new OcrError("MODEL_LOAD_FAILED", "recognizer failed");
  })
);

vi.mock("../../src/engine/modelLoader.js", () => ({
  loadOrt,
  loadSession,
  releaseSession,
}));

describe("OnnxOcrEngine load", () => {
  it("releases the successfully loaded detector session when recognizer load fails", async () => {
    vi.resetModules();

    releaseSession.mockClear();
    loadOrt.mockClear();
    loadSession.mockClear();

    const { createOnnxOcrEngine } = await import("../../src/engine/OnnxOcrEngine");

    const engine = createOnnxOcrEngine({
      detectorModelUrl: "/det",
      recognizerModelUrl: "/rec",
      wasmPath: "/wasm/",
      alphabet: "ab",
    });

    const phases: string[] = [];
    await expect(engine.load(undefined, (p) => phases.push(p.phase))).rejects.toMatchObject({
      code: "MODEL_LOAD_FAILED",
    });

    expect(phases).toEqual(["ort", "detector"]);

    expect(releaseSession).toHaveBeenCalled();
    expect(releaseSession).toHaveBeenCalledWith(detectorSession);

    await expect(engine.recognize({} as any, {} as any)).rejects.toMatchObject({
      code: "MODEL_LOAD_FAILED",
    });
  });
});

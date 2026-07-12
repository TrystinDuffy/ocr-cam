import { describe, it, expect, vi } from "vitest";

const createOnnxOcrEngineMock = vi.fn();
const createOcrControllerMock = vi.fn();

vi.mock("../../src/engine/OnnxOcrEngine.js", () => ({
  createOnnxOcrEngine: (...args: any[]) => createOnnxOcrEngineMock(...args),
}));

vi.mock("../../src/controller/OcrController.js", () => ({
  createOcrController: (...args: any[]) => createOcrControllerMock(...args),
}));

describe("createBrowserOcr default engine", () => {
  beforeEach(() => {
    createOnnxOcrEngineMock.mockClear();
    createOcrControllerMock.mockClear();
  });

  it("constructs the default ONNX engine when engine is not provided", async () => {
    vi.resetModules();

    const { createBrowserOcr } = await import("../../src/index");

    const engineObj = { engine: true };
    const ctrlObj = { controller: true };

    createOnnxOcrEngineMock.mockReturnValue(engineObj);
    createOcrControllerMock.mockReturnValue(ctrlObj);

    const config: any = {
      engine: {
        detectorModelUrl: "/models/detector.onnx",
        recognizerModelUrl: "/models/recognizer.onnx",
        wasmPath: "/onnx/",
      },
    };

    const ctrl = createBrowserOcr(config);

    expect(createOnnxOcrEngineMock).toHaveBeenCalledWith(config.engine);
    expect(createOcrControllerMock).toHaveBeenCalledWith(config, engineObj);
    expect(ctrl).toBe(ctrlObj);
  });

  it("uses the provided engine when passed", async () => {
    vi.resetModules();

    const { createBrowserOcr } = await import("../../src/index");

    createOnnxOcrEngineMock.mockReturnValue({});
    const ctrlObj = { controller: true };
    createOcrControllerMock.mockReturnValue(ctrlObj);

    const config: any = {
      engine: {
        detectorModelUrl: "/models/detector.onnx",
        recognizerModelUrl: "/models/recognizer.onnx",
        wasmPath: "/onnx/",
      },
    };
    const providedEngine = { custom: true };

    const ctrl = createBrowserOcr(config, providedEngine as any);

    expect(createOnnxOcrEngineMock).not.toHaveBeenCalled();
    expect(createOcrControllerMock).toHaveBeenCalledWith(config, providedEngine);
    expect(ctrl).toBe(ctrlObj);
  });
});

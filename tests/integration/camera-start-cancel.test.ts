import { describe, it, expect, vi } from "vitest";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
};

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

let currentStartDeferred: Deferred<any> | null = null;

const cameraCtrlMock = {
  start: vi.fn(() => {
    if (!currentStartDeferred) throw new Error("start called without deferred");
    return currentStartDeferred.promise;
  }),
  stop: vi.fn(),
  getStream: vi.fn(() => null),
  getVideoElement: vi.fn(() => null),
  isRunning: vi.fn(() => false),
};

vi.mock("../../src/camera/CameraController.js", () => ({
  createCameraController: () => cameraCtrlMock,
}));

function makeConfig(): any {
  return {
    engine: {
      detectorModelUrl: "/models/detector.onnx",
      recognizerModelUrl: "/models/recognizer.onnx",
      wasmPath: "/onnx/",
    },
  };
}

describe("Camera start cancellation", () => {
  it("stopCamera cancels pending startCamera() (rejects AbortError)", async () => {
    vi.resetModules();

    const { createOcrController } = await import("../../src/controller/OcrController");
    const { createStubEngine } = await import("../../src/engine/StubEngine");

    currentStartDeferred = createDeferred<any>();
    cameraCtrlMock.start.mockClear();
    cameraCtrlMock.stop.mockClear();

    const ctrl = createOcrController(makeConfig(), createStubEngine());
    await ctrl.load();
    expect(ctrl.getState()).toBe("ready");

    const startPromise = ctrl.startCamera();
    expect(ctrl.getState()).toBe("starting-camera");

    await ctrl.stopCamera();
    expect(ctrl.getState()).toBe("ready");

    currentStartDeferred.resolve({ stream: {} as any, width: 640, height: 480 });

    await expect(startPromise).rejects.toMatchObject({ name: "AbortError" });
    expect(cameraCtrlMock.stop).toHaveBeenCalled();
  });

  it("destroy cancels pending startCamera() (rejects AbortError)", async () => {
    vi.resetModules();

    const { createOcrController } = await import("../../src/controller/OcrController");
    const { createStubEngine } = await import("../../src/engine/StubEngine");

    currentStartDeferred = createDeferred<any>();
    cameraCtrlMock.start.mockClear();
    cameraCtrlMock.stop.mockClear();

    const ctrl = createOcrController(makeConfig(), createStubEngine());
    await ctrl.load();
    expect(ctrl.getState()).toBe("ready");

    const startPromise = ctrl.startCamera();
    const destroyPromise = ctrl.destroy();

    currentStartDeferred.resolve({ stream: {} as any, width: 640, height: 480 });

    await expect(startPromise).rejects.toMatchObject({ name: "AbortError" });
    await destroyPromise;

    expect(ctrl.getState()).toBe("destroyed");
    expect(cameraCtrlMock.stop).toHaveBeenCalled();
  });
});

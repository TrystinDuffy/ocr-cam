import { describe, it, expect, vi } from "vitest";

describe("CameraController cleanup", () => {
  it("cleans up tracks and clears refs when video.play() rejects", async () => {
    vi.resetModules();
    const { createCameraController } = await import("../../src/camera/CameraController");

    try {
      const trackStop = vi.fn();
      const track = {
        stop: trackStop,
        readyState: "live",
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      } as any;

      const stream = {
        getTracks: () => [track],
        getVideoTracks: () => [track],
      } as any;

      const getUserMedia = vi.fn(async () => stream);
      vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } } as any);

      const playError = new Error("video.play failed");
      const video = {
        playsInline: false,
        muted: false,
        srcObject: null,
        videoWidth: 0,
        videoHeight: 0,
        play: vi.fn(() => Promise.reject(playError)),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      } as any;

      vi.stubGlobal("document", {
        createElement: vi.fn(() => video),
      } as any);

      const cameraCtrl = createCameraController();

      await expect(cameraCtrl.start()).rejects.toThrow("video.play failed");
      expect(trackStop).toHaveBeenCalledTimes(1);
      expect(cameraCtrl.getStream()).toBeNull();
      expect(cameraCtrl.getVideoElement()).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

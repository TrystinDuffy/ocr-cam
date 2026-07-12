import { OcrError } from "../types/errors.js";
import { createStateMachine } from "./state.js";
import { resolveCrop, cropRegionToPixelRect } from "../crop/crop.js";
import { translateDetectionFromCrop } from "../crop/coordinates.js";
import { TypedEventTarget } from "../utilities/events.js";
import { createCameraController, } from "../camera/CameraController.js";
import { createFrameScheduler, } from "../camera/frameScheduler.js";
import { createOcrView } from "../view/OcrView.js";
let nextSessionId = 1;
export function createOcrController(config, engine) {
    const state = createStateMachine();
    const events = new TypedEventTarget();
    const cameraCtrl = createCameraController({
        onTrackEnded: handleTrackEnded,
    });
    const frameScheduler = createFrameScheduler(config.inference?.maxFps ?? 8);
    let currentEngine = engine ?? null;
    let loadPromise = null;
    let sessionId = 0;
    let frameId = 0;
    let inferenceBusy = false;
    let currentCropConfig = config.crop ?? null;
    let resolvedCrop = null;
    let destroyed = false;
    let currentView = null;
    // Used to prevent startCamera() from “winning” after stopCamera() is called
    // during an in-flight getUserMedia/video startup.
    let cameraStartAttemptId = 0;
    function createAbortError() {
        if (typeof DOMException !== "undefined") {
            return new DOMException("Aborted", "AbortError");
        }
        const err = new Error("Aborted");
        err.name = "AbortError";
        return err;
    }
    const temporalCfg = config.inference?.temporal;
    let temporalTick = 0;
    let temporalBuffer = [];
    let lastTemporalResult = null;
    let prevInputForSmoothing = null;
    function resetTemporal() {
        temporalTick = 0;
        temporalBuffer = [];
        lastTemporalResult = null;
        prevInputForSmoothing = null;
    }
    function rectIoU(a, b) {
        const ax1 = a.x;
        const ay1 = a.y;
        const ax2 = a.x + a.width;
        const ay2 = a.y + a.height;
        const bx1 = b.x;
        const by1 = b.y;
        const bx2 = b.x + b.width;
        const by2 = b.y + b.height;
        const x1 = Math.max(ax1, bx1);
        const y1 = Math.max(ay1, by1);
        const x2 = Math.min(ax2, bx2);
        const y2 = Math.min(ay2, by2);
        const interW = Math.max(0, x2 - x1);
        const interH = Math.max(0, y2 - y1);
        const inter = interW * interH;
        const union = a.width * a.height + b.width * b.height - inter;
        return union > 0 ? inter / union : 0;
    }
    function weightedAvgBox(dets) {
        let sumW = 0;
        let x = 0;
        let y = 0;
        let w = 0;
        let h = 0;
        for (const d of dets) {
            const wgt = Math.max(0, d.confidence);
            sumW += wgt;
            x += d.box.x * wgt;
            y += d.box.y * wgt;
            w += d.box.width * wgt;
            h += d.box.height * wgt;
        }
        if (sumW <= 0) {
            const n = dets.length || 1;
            const avgX = dets.reduce((acc, d) => acc + d.box.x, 0) / n;
            const avgY = dets.reduce((acc, d) => acc + d.box.y, 0) / n;
            const avgW = dets.reduce((acc, d) => acc + d.box.width, 0) / n;
            const avgH = dets.reduce((acc, d) => acc + d.box.height, 0) / n;
            return { x: avgX, y: avgY, width: avgW, height: avgH };
        }
        return { x: x / sumW, y: y / sumW, width: w / sumW, height: h / sumW };
    }
    function smoothDetectionsFromBuffer(buffer, boxIouThreshold) {
        if (buffer.length === 0)
            return [];
        // Anchor smoothing to the *latest* OCR result.
        // This prevents older bad detections from pulling the averaged output away
        // over time (the drift you were seeing).
        const latest = buffer[buffer.length - 1];
        const anchors = latest.detections;
        const smoothed = anchors
            .map((anchor) => {
            const members = [];
            for (const r of buffer) {
                for (const d of r.detections) {
                    if (rectIoU(anchor.box, d.box) >= boxIouThreshold) {
                        members.push(d);
                    }
                }
            }
            // `members` must include `anchor` itself, but keep a defensive fallback.
            if (members.length === 0)
                members.push(anchor);
            const box = weightedAvgBox(members);
            const confidence = members.reduce((acc, d) => acc + d.confidence, 0) / Math.max(1, members.length);
            return {
                // Text comes from the newest frame.
                text: anchor.text,
                confidence,
                box: {
                    x: box.x,
                    y: box.y,
                    width: box.width,
                    height: box.height,
                },
            };
        })
            .sort((a, b) => (a.box.y - b.box.y) || (a.box.x - b.box.x));
        return smoothed;
    }
    function applyInputTemporalSmoothing(imageData, alpha) {
        // EMA: smoothed = alpha*current + (1-alpha)*previous
        if (!prevInputForSmoothing) {
            prevInputForSmoothing = imageData;
            return;
        }
        if (prevInputForSmoothing.width !== imageData.width ||
            prevInputForSmoothing.height !== imageData.height) {
            prevInputForSmoothing = imageData;
            return;
        }
        const cur = imageData.data;
        const prev = prevInputForSmoothing.data;
        const wPrev = 1 - alpha;
        for (let i = 0; i < cur.length; i++) {
            cur[i] = Math.round(cur[i] * alpha + prev[i] * wPrev);
        }
        // Store the newest smoothed frame for next iteration.
        prevInputForSmoothing = imageData;
    }
    function applyContrastStretch(imageData) {
        // Lightweight luma contrast stretch based on subsampled min/max.
        // Helps with lighting flicker across frames.
        const { data, width: w, height: h } = imageData;
        // Sample on a coarse grid to estimate luma min/max.
        const grid = 24;
        let minL = 255;
        let maxL = 0;
        const denomY = Math.max(1, grid - 1);
        const denomX = Math.max(1, grid - 1);
        for (let gy = 0; gy < grid; gy++) {
            const y = Math.floor((gy / denomY) * (h - 1));
            for (let gx = 0; gx < grid; gx++) {
                const x = Math.floor((gx / denomX) * (w - 1));
                const idx = (y * w + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const l = (r + g + b) / 3;
                if (l < minL)
                    minL = l;
                if (l > maxL)
                    maxL = l;
            }
        }
        const range = maxL - minL;
        if (!(range > 10))
            return; // avoid extreme noise amplification
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const l = (r + g + b) / 3;
                const lNew = Math.min(1, Math.max(0, (l - minL) / range));
                const lNorm = l / 255;
                if (lNorm <= 1e-6) {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                    continue;
                }
                const scale = lNew / lNorm;
                const nr = Math.max(0, Math.min(255, r * scale));
                const ng = Math.max(0, Math.min(255, g * scale));
                const nb = Math.max(0, Math.min(255, b * scale));
                data[idx] = Math.round(nr);
                data[idx + 1] = Math.round(ng);
                data[idx + 2] = Math.round(nb);
            }
        }
    }
    // Reusable canvases to avoid per-frame allocation
    let reuseCanvas = null;
    let reuseCtx = null;
    function assertNotDestroyed() {
        if (destroyed) {
            throw new OcrError("DESTROYED", "This controller has been destroyed");
        }
    }
    function emitState() {
        events.emit("statechange", state.getState());
        emitActions();
    }
    function computeActions() {
        const s = state.getState();
        const canLoad = s === "idle" || s === "error";
        const canStartCamera = s === "ready";
        const canStopCamera = s === "running" || s === "starting-camera";
        const canToggleDrawing = s === "running" && !!currentView;
        const canClearDrawing = s === "running" && !!currentView;
        return {
            canLoad,
            canStartCamera,
            canStopCamera,
            canToggleDrawing,
            canClearDrawing,
        };
    }
    function emitActions() {
        events.emit("actionschange", computeActions());
    }
    function handleTrackEnded() {
        if (state.getState() === "running") {
            frameScheduler.stop();
            cameraCtrl.stop();
            inferenceBusy = false;
            state.transition("error");
            emitState();
            const error = new OcrError("CAMERA_START_FAILED", "Camera tracks ended unexpectedly");
            events.emit("error", error);
            config.onError?.(error);
        }
    }
    function updateCropForSource(width, height) {
        resolvedCrop = currentCropConfig
            ? resolveCrop(currentCropConfig, width, height)
            : null;
    }
    function ensureCanvas(w, h) {
        if (reuseCanvas && reuseCanvas.width === w && reuseCanvas.height === h && reuseCtx) {
            return reuseCtx;
        }
        reuseCanvas = new OffscreenCanvas(w, h);
        reuseCtx = reuseCanvas.getContext("2d");
        if (!reuseCtx)
            throw new Error("Failed to create OffscreenCanvas 2D context");
        return reuseCtx;
    }
    async function processFrame(timestamp) {
        if (state.getState() !== "running")
            return;
        if (inferenceBusy)
            return;
        if (!currentEngine)
            return;
        const video = cameraCtrl.getVideoElement();
        if (!video || video.readyState < 2)
            return;
        if (temporalCfg) {
            temporalTick++;
            const stride = temporalCfg.stride ?? 1;
            if (stride > 1 && temporalTick % stride !== 0) {
                if (currentView && lastTemporalResult) {
                    currentView.updateResult(lastTemporalResult);
                }
                return;
            }
        }
        inferenceBusy = true;
        const currentFrameId = ++frameId;
        const sourceWidth = video.videoWidth;
        const sourceHeight = video.videoHeight;
        try {
            const crop = resolvedCrop;
            let frameData;
            if (crop) {
                const ctx = ensureCanvas(crop.width, crop.height);
                ctx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
                const imageData = ctx.getImageData(0, 0, crop.width, crop.height);
                frameData = { imageData, width: crop.width, height: crop.height };
            }
            else {
                const ctx = ensureCanvas(sourceWidth, sourceHeight);
                ctx.drawImage(video, 0, 0, sourceWidth, sourceHeight);
                const imageData = ctx.getImageData(0, 0, sourceWidth, sourceHeight);
                frameData = { imageData, width: sourceWidth, height: sourceHeight };
            }
            if (temporalCfg) {
                const alpha = temporalCfg.inputSmoothingAlpha;
                if (typeof alpha === "number" && alpha > 0 && alpha < 1) {
                    applyInputTemporalSmoothing(frameData.imageData, alpha);
                }
                if (temporalCfg.contrastStretch) {
                    applyContrastStretch(frameData.imageData);
                }
            }
            const context = {
                sessionId,
                frameId: currentFrameId,
                timestamp,
                crop: crop ? { ...crop } : null,
                sourceSize: { width: sourceWidth, height: sourceHeight },
            };
            const engineResult = await currentEngine.recognize(frameData, context);
            // Reject stale results
            if (state.getState() !== "running")
                return;
            const detections = crop
                ? engineResult.detections.map((det) => translateDetectionFromCrop(det, crop))
                : engineResult.detections;
            const combinedText = detections.map((det) => det.text).join("\n");
            const result = {
                sessionId,
                frameId: currentFrameId,
                timestamp,
                sourceSize: { width: sourceWidth, height: sourceHeight },
                crop: crop ? cropRegionToPixelRect(crop) : null,
                detections,
                text: combinedText,
                inferenceDurationMs: engineResult.inferenceDurationMs,
            };
            let output = result;
            if (temporalCfg) {
                const windowSize = Math.max(1, temporalCfg.windowSize ?? 3);
                if (windowSize > 1) {
                    temporalBuffer.push(result);
                    while (temporalBuffer.length > windowSize)
                        temporalBuffer.shift();
                    const boxIouThreshold = temporalCfg.boxIouThreshold ?? 0.5;
                    const smoothed = smoothDetectionsFromBuffer(temporalBuffer, boxIouThreshold);
                    const smoothedText = smoothed.map((d) => d.text).join("\n");
                    output = { ...result, detections: smoothed, text: smoothedText };
                }
                lastTemporalResult = output;
            }
            events.emit("result", output);
            config.onResult?.(output);
            if (currentView) {
                currentView.updateResult(output);
            }
        }
        catch (err) {
            if (state.getState() === "running") {
                const error = err instanceof OcrError
                    ? err
                    : new OcrError("INFERENCE_FAILED", err instanceof Error ? err.message : "Unknown inference error", { cause: err });
                events.emit("error", error);
                config.onError?.(error);
            }
        }
        finally {
            inferenceBusy = false;
        }
    }
    const controller = {
        load() {
            assertNotDestroyed();
            if (state.getState() === "ready" || state.getState() === "running") {
                return;
            }
            if (loadPromise)
                return loadPromise;
            state.transition("loading");
            emitState();
            loadPromise = (async () => {
                try {
                    if (!currentEngine) {
                        throw new OcrError("MODEL_LOAD_FAILED", "No OCR engine provided – pass an engine to createBrowserOcr()");
                    }
                    await currentEngine.load(undefined, (progress) => {
                        events.emit("loadprogress", progress);
                        config.onLoadProgress?.(progress);
                    });
                    loadPromise = null;
                    state.transition("ready");
                    emitState();
                }
                catch (err) {
                    state.transition("error");
                    emitState();
                    loadPromise = null;
                    throw err;
                }
            })();
            return loadPromise;
        },
        async startCamera(options) {
            assertNotDestroyed();
            // Already running – return current session info
            if (state.getState() === "running") {
                const video = cameraCtrl.getVideoElement();
                return {
                    sessionId,
                    width: video?.videoWidth ?? 0,
                    height: video?.videoHeight ?? 0,
                    facingMode: undefined,
                };
            }
            // Require loaded engine
            if (state.getState() !== "ready") {
                throw new OcrError("INVALID_STATE", `Cannot start camera in state "${state.getState()}" – call load() first`);
            }
            state.transition("starting-camera");
            emitState();
            const myAttemptId = ++cameraStartAttemptId;
            try {
                let constraints;
                if (options?.constraints) {
                    constraints = options.constraints;
                }
                else if (options?.facingMode) {
                    constraints = { audio: false, video: { facingMode: { ideal: options.facingMode } } };
                }
                else if (config.camera?.constraints) {
                    constraints = config.camera.constraints;
                }
                else if (config.camera?.facingMode) {
                    constraints = { audio: false, video: { facingMode: { ideal: config.camera.facingMode } } };
                }
                const info = await cameraCtrl.start(constraints);
                // If stopCamera() was called while we were starting, don't transition
                // into the running state.
                if (myAttemptId !== cameraStartAttemptId || state.getState() !== "starting-camera") {
                    cameraCtrl.stop();
                    throw createAbortError();
                }
                sessionId = nextSessionId++;
                frameId = 0;
                updateCropForSource(info.width, info.height);
                resetTemporal();
                state.transition("running");
                emitState();
                const sessionInfo = {
                    sessionId,
                    width: info.width,
                    height: info.height,
                    facingMode: undefined,
                };
                events.emit("camerastart", sessionInfo);
                const video = cameraCtrl.getVideoElement();
                if (video) {
                    frameScheduler.start(video, { onFrame: processFrame });
                }
                return sessionInfo;
            }
            catch (err) {
                const cancelled = myAttemptId !== cameraStartAttemptId || state.getState() !== "starting-camera";
                if (cancelled) {
                    cameraCtrl.stop();
                    if (err instanceof Error && err.name === "AbortError")
                        throw err;
                    throw createAbortError();
                }
                // Camera error must not corrupt loaded engine – go to error, which can
                // recover to idle (load is still valid).
                state.transition("error");
                emitState();
                throw err;
            }
        },
        async stopCamera() {
            assertNotDestroyed();
            const currentState = state.getState();
            if (currentState !== "running" && currentState !== "starting-camera") {
                return;
            }
            // Invalidate any in-flight startCamera() attempt.
            cameraStartAttemptId++;
            state.transition("stopping-camera");
            emitState();
            frameScheduler.stop();
            cameraCtrl.stop();
            inferenceBusy = false;
            reuseCanvas = null;
            reuseCtx = null;
            resetTemporal();
            // Detach view before transitioning (view destruction must happen on camera stop)
            controller.detachView();
            // Transition back to ready (engine is still loaded)
            state.transition("ready");
            emitState();
            events.emit("camerastop");
        },
        attachView(container, options) {
            assertNotDestroyed();
            if (state.getState() !== "running") {
                throw new OcrError("INVALID_STATE", `Cannot attach view in state "${state.getState()}" – camera must be running`);
            }
            // Destroy existing view if any
            controller.detachView();
            const video = cameraCtrl.getVideoElement();
            if (!video) {
                throw new OcrError("INVALID_STATE", "No video element available");
            }
            currentView = createOcrView(container, video, resolvedCrop, {
                outsideCropOpacity: options?.outsideCropOpacity,
                showCropBorder: options?.showCropBorder,
                showBoundingBoxes: options?.showBoundingBoxes,
                showRecognizedText: options?.showRecognizedText,
                drawingEnabled: options?.drawingEnabled,
            });
            emitActions();
            return {
                detach() {
                    controller.detachView();
                },
            };
        },
        detachView() {
            if (currentView) {
                currentView.destroy();
                currentView = null;
            }
            emitActions();
        },
        setCrop(crop) {
            assertNotDestroyed();
            currentCropConfig = crop;
            resetTemporal();
            if (state.getState() === "running") {
                const video = cameraCtrl.getVideoElement();
                if (video) {
                    updateCropForSource(video.videoWidth, video.videoHeight);
                }
            }
            else {
                resolvedCrop = null;
            }
            if (currentView && state.getState() === "running") {
                currentView.updateCrop(resolvedCrop);
            }
            events.emit("cropchange", resolvedCrop);
        },
        getCrop() {
            return resolvedCrop;
        },
        setDrawingEnabled(enabled) {
            assertNotDestroyed();
            if (currentView) {
                currentView.setDrawingEnabled(enabled);
            }
        },
        clearDrawing() {
            assertNotDestroyed();
            if (currentView) {
                currentView.clearDrawing();
            }
        },
        getState() {
            return state.getState();
        },
        on(event, listener) {
            assertNotDestroyed();
            return events.on(event, listener);
        },
        async destroy() {
            if (destroyed)
                return;
            destroyed = true;
            // Invalidate any in-flight startCamera() attempt so it won't transition
            // into running after teardown.
            cameraStartAttemptId++;
            frameScheduler.stop();
            cameraCtrl.stop();
            controller.detachView();
            events.removeAllListeners();
            reuseCanvas = null;
            reuseCtx = null;
            if (currentEngine) {
                await currentEngine.dispose();
                currentEngine = null;
            }
            state.transition("destroyed");
            loadPromise = null;
        },
    };
    // Emit initial action state so UIs can set button disabled/enabled states
    // immediately without waiting for the first state transition.
    emitActions();
    return controller;
}
//# sourceMappingURL=OcrController.js.map
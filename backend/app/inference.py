import json
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ONNX_PATH = ARTIFACTS_DIR / "quality_model.onnx"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"


class QualityPredictor:
    def __init__(self):
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)

        self.img_size = self.config["input_resolution"]
        self.norm_mean = np.array(self.config["normalization"]["mean"], dtype=np.float32)
        self.norm_std = np.array(self.config["normalization"]["std"], dtype=np.float32)
        self.thresholds = self.config["issue_thresholds"]

        self.session = ort.InferenceSession(str(ONNX_PATH))
        print(f"Model loaded — inputs: {[i.name for i in self.session.get_inputs()]}")

    # ---- classical CV features (must match training exactly) ----
    @staticmethod
    def _laplacian_variance(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _mean_luminance(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    @staticmethod
    def _noise_std(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return float((gray - blurred).std())

    def _extract_cv_features(self, img_bgr):
        return np.array([
            self._laplacian_variance(img_bgr),
            self._mean_luminance(img_bgr),
            self._noise_std(img_bgr),
        ], dtype=np.float32)

    def _severity(self, value, low, high, inverted=False):
        if inverted:
            if value < low * 0.5:
                return "high", min(1.0, (low - value) / low)
            elif value < low:
                return "medium", 0.6
            return "low", 0.3
        span = high - low
        if value < low or value > high:
            dist = min(abs(value - low), abs(value - high))
            sev = "high" if dist > span * 0.3 else "medium"
            return sev, min(1.0, dist / span)
        return "low", 0.2

    def _build_issues(self, img_bgr):
        lap_var = self._laplacian_variance(img_bgr)
        luminance = self._mean_luminance(img_bgr)
        noise_std = self._noise_std(img_bgr)
        t = self.thresholds

        issues = {}
        sev, conf = self._severity(lap_var, t["blur_low"], None, inverted=True)
        issues["blur"] = {"detected": lap_var < t["blur_low"], "severity": sev,
                           "confidence": round(conf, 3), "raw_value": round(lap_var, 2)}

        if luminance < t["underexposure"]:
            sev, conf = self._severity(luminance, t["underexposure"], t["overexposure"])
            issues["underexposure"] = {"detected": True, "severity": sev,
                                        "confidence": round(conf, 3), "raw_value": round(luminance, 2)}
        if luminance > t["overexposure"]:
            sev, conf = self._severity(luminance, t["underexposure"], t["overexposure"])
            issues["overexposure"] = {"detected": True, "severity": sev,
                                       "confidence": round(conf, 3), "raw_value": round(luminance, 2)}

        sev, conf = self._severity(noise_std, 0, t["noise_high"])
        issues["noise"] = {"detected": noise_std > t["noise_high"], "severity": sev,
                            "confidence": round(conf, 3), "raw_value": round(noise_std, 2)}
        return issues

    def _preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - self.norm_mean) / self.norm_std
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0).astype(np.float32)

    def predict(self, img_bytes: bytes) -> dict:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image — invalid or corrupted file")

        img_batch = self._preprocess(img_bgr)
        cv_batch = np.expand_dims(self._extract_cv_features(img_bgr), axis=0)

        outputs = self.session.run(None, {"image": img_batch, "cv_features": cv_batch})
        score = float(outputs[0][0])

        return {
            "quality_score": round(score, 2),
            "issues": self._build_issues(img_bgr),
        }


# Singleton — loaded once at server startup, reused across requests
predictor = QualityPredictor()
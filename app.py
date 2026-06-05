import json
import time
import traceback
from pathlib import Path
from functools import wraps

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.mlProject.logging import logger

app = Flask(__name__)
CORS(app)   # izinkan cross-origin request (untuk konsumsi API eksternal)

# ── Lazy-load pipeline ────────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.mlProject.pipeline.prediction import PredictionPipeline
        _pipeline = PredictionPipeline.get_instance()
    return _pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────
def success(data: dict, status: int = 200):
    return jsonify({"status": "success", **data}), status

def error(message: str, status: int = 400):
    logger.info(f"[{status}] {message}")
    return jsonify({"status": "error", "message": message}), status

def require_model(f):
    """Decorator — pastikan model sudah di-load sebelum endpoint dijalankan."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            pipeline = get_pipeline()
            if not pipeline.is_ready():
                return error("Model belum siap. Jalankan training pipeline dulu.", 503)
        except FileNotFoundError as e:
            return error(str(e), 503)
        except Exception as e:
            return error(f"Gagal load model: {str(e)}", 503)
        return f(*args, **kwargs)
    return wrapper


# ── Routes — UI ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Serve halaman utama."""
    return render_template("index.html")


# ── Routes — API ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Response:
        {
          "status": "success",
          "model_loaded": true,
          "artifacts": { ... }
        }
    """
    try:
        pipeline = get_pipeline()
        loaded   = pipeline.is_ready()
    except Exception:
        loaded = False

    artifacts = {
        "model":         Path("artifacts/model_trainer/model.pkl").exists(),
        "bins":          Path("artifacts/data_preprocessing/woe_bins.pkl").exists(),
        "scorecard":     Path("artifacts/model_trainer/scorecard.pkl").exists(),
        "feature_names": Path("artifacts/model_trainer/feature_names.json").exists(),
        "metrics":       Path("artifacts/model_evaluation/metrics.json").exists(),
    }

    return success({
        "model_loaded": loaded,
        "artifacts":    artifacts,
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """
    Return model evaluation metrics.

    Response:
        {
          "status": "success",
          "metrics": { "auc_test": ..., "ks_test": ..., ... }
        }
    """
    eval_path  = Path("artifacts/model_evaluation/metrics.json")
    train_path = Path("artifacts/model_trainer/train_metrics.json")

    result = {}

    if eval_path.exists():
        with open(eval_path) as f:
            result["evaluation"] = json.load(f)

    if train_path.exists():
        with open(train_path) as f:
            result["training"] = json.load(f)

    if not result:
        return error("Metrics belum tersedia. Jalankan pipeline terlebih dahulu.", 404)

    return success({"metrics": result})


@app.route("/api/predict", methods=["POST"])
@require_model
def predict_api():
    """
    Prediksi credit score dari raw input.

    Request body (JSON):
        {
          "Age": 32,
          "Annual_Income": 60000,
          "Monthly_Inhand_Salary": 4500,
          ... (field lainnya)
        }

    Response:
        {
          "status": "success",
          "prediction": 0,
          "label": "Good / Standard",
          "risk_level": "Low Risk",
          "probability": { "bad": 0.21, "good": 0.79 },
          "credit_score": 624,
          "latency_ms": 45
        }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return error("Request body kosong atau bukan JSON yang valid.")

    if not isinstance(data, dict):
        return error("Format input harus berupa JSON object.")

    t0 = time.time()

    try:
        pipeline = get_pipeline()
        result   = pipeline.predict(data)
        latency  = round((time.time() - t0) * 1000, 1)

        logger.info(
            f"Predict — label={result['label']} | "
            f"prob_bad={result['probability']['bad']:.3f} | "
            f"score={result['credit_score']} | {latency}ms"
        )

        return success({**result, "latency_ms": latency})

    except Exception as e:
        logger.info(f"Predict error: {traceback.format_exc()}")
        return error(f"Prediksi gagal: {str(e)}", 500)


# Alias /predict → /api/predict (kompatibel dengan app.py sebelumnya)
@app.route("/predict", methods=["POST"])
@require_model
def predict_alias():
    return predict_api()


@app.route("/api/predict/batch", methods=["POST"])
@require_model
def predict_batch():
    """
    Prediksi batch — terima list of records.

    Request body (JSON):
        [
          {"Age": 32, "Annual_Income": 60000, ...},
          {"Age": 45, "Annual_Income": 80000, ...}
        ]

    Response:
        {
          "status": "success",
          "count": 2,
          "results": [ {...}, {...} ],
          "latency_ms": 120
        }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return error("Request body kosong.")

    if not isinstance(data, list):
        return error("Batch predict membutuhkan array JSON.")

    if len(data) > 500:
        return error("Maksimal 500 records per batch request.", 413)

    t0       = time.time()
    results  = []
    errors   = []

    pipeline = get_pipeline()

    for i, record in enumerate(data):
        try:
            results.append({"index": i, **pipeline.predict(record)})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    latency = round((time.time() - t0) * 1000, 1)
    logger.info(f"Batch predict — {len(results)} success, {len(errors)} error, {latency}ms")

    return success({
        "count":      len(data),
        "success":    len(results),
        "failed":     len(errors),
        "results":    results,
        "errors":     errors,
        "latency_ms": latency,
    })


@app.route("/train", methods=["GET"])
def train():
    """
    Trigger full training pipeline.
    Gunakan dengan hati-hati — proses ini memakan waktu beberapa menit.
    """
    try:
        logger.info("Training pipeline triggered via /train")

        from src.mlProject.pipeline.stage_01_data_ingestion    import DataIngestionTrainingPipeline
        from src.mlProject.pipeline.stage_02_data_validation   import DataValidationTrainingPipeline
        from src.mlProject.pipeline.stage_03_data_preprocessing import DataPreprocessingTrainingPipeline
        from src.mlProject.pipeline.stage_04_model_trainer      import ModelTrainerTrainingPipeline
        from src.mlProject.pipeline.stage_05_model_evaluation   import ModelEvaluationPipeline

        DataIngestionTrainingPipeline().main()
        DataValidationTrainingPipeline().main()
        DataPreprocessingTrainingPipeline().main()
        metrics = ModelTrainerTrainingPipeline().main()
        eval_metrics = ModelEvaluationPipeline().main()

        # Reset singleton agar artifacts terbaru di-load
        global _pipeline
        _pipeline = None

        return success({
            "message":      "Training pipeline selesai",
            "train_metrics": metrics,
            "eval_metrics":  eval_metrics,
        })

    except Exception as e:
        logger.info(f"Training error: {traceback.format_exc()}")
        return error(f"Training gagal: {str(e)}", 500)


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return error("Endpoint tidak ditemukan.", 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return error("Method tidak diizinkan.", 405)

@app.errorhandler(500)
def internal_error(e):
    return error("Internal server error.", 500)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting CreditScore AI server on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
# Keras Distributed Training Starter (TensorFlow + Node + React)

A lightweight starter project for a **real-world scenario**: training an image classifier for **factory defect detection** with Keras best practices and TensorFlow distributed strategy, then exposing model training status in a small Node/React UI.

## Why this project

This follows Keras guidance for distributed training:
- `tf.distribute.MirroredStrategy` for multi-GPU single-machine training
- `tf.data` pipeline with `map`, `shuffle`, `prefetch`
- Transfer learning (MobileNetV2) for practical speed/accuracy
- Production callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`)

Reference:
- Keras: https://keras.io/
- Distributed training guide: https://keras.io/guides/distributed_training_with_tensorflow/

---

## Project structure

- `src/train_distributed.py` — TensorFlow/Keras distributed training script
- `src/server.js` — Node API to read training report
- `src/frontend/App.jsx` — React UI to display training outcome
- `artifacts/training_report.json` — generated after training
- `artifacts/*.keras` — saved model files

---

## Real-time use case guidance

### Scenario
You receive new labeled factory images every day (e.g., `ok/`, `defect/` folders).  
You retrain daily and deploy updated model if validation accuracy is acceptable.

### Recommended flow
1. Collect and validate fresh labeled images.
2. Trigger `train_distributed.py` on a GPU machine.
3. Script writes `artifacts/training_report.json`.
4. Node API reads report and frontend shows status.
5. CI/CD can promote model automatically if metric threshold passes.

### How this resolves the problem
- Distributed training reduces retraining time.
- Best-practice data pipeline avoids training bottlenecks.
- Monitoring endpoint/UI gives quick operational visibility.

---

## Setup

## 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install tensorflow
```

Prepare data:

```text
data/images/
  ok/
    img1.jpg
    ...
  defect/
    img2.jpg
    ...
```

Run training:

```bash
python src/train_distributed.py
```

---

## 2) Node API

```bash
npm install
npm run api
```

API endpoints:
- `GET /api/health`
- `GET /api/training-status`

---

## 3) React frontend

```bash
cd src/frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`).

---

## Notes / best practices

- Keep model creation and compile **inside** `strategy.scope()`.
- Use global batch size = per-replica batch * number of replicas.
- Start with transfer learning for faster convergence.
- Log metrics and artifacts for reproducibility.
- For large scale, move from MirroredStrategy to MultiWorker + orchestration.

---

## Next improvements (optional)

- Add model evaluation gates (e.g., min precision/recall).
- Add drift monitoring and auto-retraining trigger.
- Store reports in a DB/object store instead of local file.
- Add prediction API endpoint for online inference.
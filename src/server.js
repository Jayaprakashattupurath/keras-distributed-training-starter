/**
 * Lightweight Node API to expose training status to frontend.
 * In real deployment, this would read from a job queue or ML pipeline metadata store.
 */
const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(express.json());

const REPORT_PATH = process.env.REPORT_PATH || path.join(__dirname, "..", "artifacts", "training_report.json");

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.get("/api/training-status", (_req, res) => {
  try {
    if (!fs.existsSync(REPORT_PATH)) {
      return res.json({
        trained: false,
        message: "No training report yet. Run `python src/train_distributed.py` first."
      });
    }

    const report = JSON.parse(fs.readFileSync(REPORT_PATH, "utf-8"));
    return res.json({ trained: true, report });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

const port = process.env.PORT || 4000;
app.listen(port, () => {
  console.log(`API listening on http://localhost:${port}`);
});
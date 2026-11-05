import React, { useEffect, useState } from "react";

export default function App() {
  const [data, setData] = useState({ loading: true });

  useEffect(() => {
    fetch("http://localhost:4000/api/training-status")
      .then((r) => r.json())
      .then((json) => setData({ loading: false, ...json }))
      .catch((e) => setData({ loading: false, error: e.message }));
  }, []);

  if (data.loading) return <p>Loading training status...</p>;
  if (data.error) return <p>Error: {data.error}</p>;

  return (
    <main style={{ fontFamily: "Arial, sans-serif", padding: 24 }}>
      <h1>Defect Detection Training Monitor</h1>
      {!data.trained ? (
        <p>{data.message}</p>
      ) : (
        <section>
          <p><strong>Replicas:</strong> {data.report.replicas_in_sync}</p>
          <p><strong>Global batch size:</strong> {data.report.global_batch_size}</p>
          <p><strong>Validation accuracy:</strong> {(data.report.final_val_accuracy * 100).toFixed(2)}%</p>
          <p><strong>Model path:</strong> {data.report.model_path}</p>
          <p><strong>Classes:</strong> {data.report.class_names.join(", ")}</p>
        </section>
      )}
    </main>
  );
}
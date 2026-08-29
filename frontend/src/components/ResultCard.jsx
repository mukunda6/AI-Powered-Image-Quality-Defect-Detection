import IssueBadge from "./IssueBadge";

export default function ResultCard({ result }) {
  const scoreColor =
    result.quality_score >= 70 ? "#22c55e" :
    result.quality_score >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div className="result-card">
      <h3>{result.filename}</h3>

      <div className="score-circle" style={{ borderColor: scoreColor }}>
        <span className="score-value" style={{ color: scoreColor }}>
          {result.quality_score.toFixed(1)}
        </span>
        <span className="score-label">/ 100</span>
      </div>

      <div className="issues-grid">
        {Object.entries(result.issues).map(([name, issue]) => (
          <IssueBadge key={name} name={name} issue={issue} />
        ))}
      </div>
    </div>
  );
}
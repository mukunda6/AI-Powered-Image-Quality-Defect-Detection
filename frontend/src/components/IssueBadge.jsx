export default function IssueBadge({ name, issue }) {
  const severityColors = {
    low: "#22c55e",
    medium: "#f59e0b",
    high: "#ef4444",
  };

  const color = severityColors[issue.severity] || "#94a3b8";

  return (
    <div className="issue-badge" style={{ borderColor: color }}>
      <div className="issue-header">
        <span className="issue-name">{name}</span>
        <span className="issue-status" style={{ color }}>
          {issue.detected ? "Detected" : "OK"}
        </span>
      </div>
      <div className="issue-details">
        <span>Severity: <strong>{issue.severity}</strong></span>
        <span>Confidence: <strong>{(issue.confidence * 100).toFixed(0)}%</strong></span>
      </div>
    </div>
  );
}
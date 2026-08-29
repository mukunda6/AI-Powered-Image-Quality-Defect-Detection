from pydantic import BaseModel
from typing import Dict


class IssueDetail(BaseModel):
    detected: bool
    severity: str
    confidence: float
    raw_value: float


class QualityResponse(BaseModel):
    filename: str
    quality_score: float
    issues: Dict[str, IssueDetail]
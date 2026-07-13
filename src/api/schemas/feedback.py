from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    predicted_class: str
    true_class: str
    confidence_score: float

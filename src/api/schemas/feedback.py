from pydantic import BaseModel


class FeedBackRequest(BaseModel):
    predicted_class: str
    true_class: str
    confidence_score: float

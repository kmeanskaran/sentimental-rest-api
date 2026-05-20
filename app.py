from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_api.service import SimpleSentimentModel


model = SimpleSentimentModel()


class EvaluateRequest(BaseModel):
    sentence: str = Field(..., min_length=1, description="Input tweet or sentence to classify.")


class EvaluateResponse(BaseModel):
    text: str
    clean_text: str
    predicted_label: int
    predicted_sentiment: str
    probability_positive: float
    probability_negative: float


class EvaluateBatchRequest(BaseModel):
    sentences: list[str] = Field(
        ...,
        min_length=1,
        description="List of input tweets or sentences to classify.",
    )


class EvaluateBatchResponse(BaseModel):
    predictions: list[EvaluateResponse]


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="Twitter Sentiment API",
    version="1.0.0",
    description="Binary sentiment inference API for the Twitter sentiment assignment.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    sentence = payload.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=422, detail="`sentence` must not be empty.")
    result = model.predict(sentence)
    return EvaluateResponse(**result)


@app.post("/evaluate/batch", response_model=EvaluateBatchResponse)
def evaluate_batch(payload: EvaluateBatchRequest) -> EvaluateBatchResponse:
    sentences = [sentence.strip() for sentence in payload.sentences]
    if any(not sentence for sentence in sentences):
        raise HTTPException(status_code=422, detail="`sentences` must not contain empty strings.")

    predictions = model.predict_batch(sentences)
    return EvaluateBatchResponse(
        predictions=[EvaluateResponse(**prediction) for prediction in predictions]
    )

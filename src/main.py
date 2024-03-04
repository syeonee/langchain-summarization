from fastapi import FastAPI
from summarization import get_summarization
from schema import Summary

app = FastAPI()


@app.post("/summary")
def summary_news(request: Summary):
    return {"summaryNews": get_summarization(request)}

@app.get("/")
def hello():
    return {"hello": "world"}

# 提供Agent所有能力接口
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses  import StreamingResponse
from agent.langchain_assistant import assistant_query

app = FastAPI()

class ChatRequest(BaseModel):
    query:str=Field(description="用户输入的问题")


@app.post("/chat")
def chat_endpoint(request:ChatRequest):
    # 以流式的方式返回结果
    return StreamingResponse(
        assistant_query(query=request.query),
        media_type="text/event-stream"
    )

@app.post("/clear_history")
def clear_history():
    pass


@app.get("/get_history")
def get_history():
    pass

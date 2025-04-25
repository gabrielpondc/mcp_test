import asyncio
import time
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def fake_openai(request: Request):
    print("请求参数:", request.headers, request.query_params, await request.json())
    data = await request.json()
    stream = data.get("stream", False)
    if stream:
        async def event_generator():
            # 模拟流式返回数据
            for part in ["```", "json", "{\n    \"key\": \"value\"\n}\n", "```"]:
                yield json.dumps({
                    "choices": [{
                        "delta": {"content": part},
                        "index": 0,
                        "finish_reason": None
                    }]
                }) + "\n"
                await asyncio.sleep(1)
            yield json.dumps({
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }) + "\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        resp = {
            "id": "fake-id-123",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "fake-model",
            "choices": [{
                "text": """```json\n{\n    \"key\": \"value\"\n}\n```""",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        return JSONResponse(resp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, Header, Request
from pydantic import BaseModel
from typing import Optional
import json
app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None

@app.post("/{full_path:path}")
async def test_route(request: Request = None):
    print(request)
    #获得请求参数
    full_path = request.path_params["full_path"]
    #获得请求头
    headers = request.headers
    #获得请求体
    item = await request.json()
    # Return the full path, item, and headers
    print(json.dumps(item,ensure_ascii=False))
    return 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

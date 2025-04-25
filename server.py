from mcp.server.fastmcp import FastMCP, Context
import datetime
import sys
from fastapi import Body
# 创建 FastMCP 服务器实例 - 使用更常用的端口 8000
mcp = FastMCP(name="测试服务器", description="这是一个测试服务器", version="0.1.0", port=8080)

# 使用装饰器注册工具
@mcp.tool(
    name="查询天气", 
    description=f"可以查询历史和未来的天气，返回天气信息"
)
async def query_weather(location: str=Body('',description='查询地址'), time: str = Body(datetime.datetime.now().strftime("%Y-%m-%d"),description='查询时间，格式为YYYY-MM-DD')):
    """
    查询指定城市在指定时间的天气情况
    
    Args:
        location: 查询天气的城市
        time: 查询天气的时间,格式为YYYY-MM-DD
    
    Returns:
        天气信息
    """
    
    return {"content": f"{time}的{location}的天气是超级大暴雨，狂风黄色预警，气温25度。"}

# 简化运行方式
if __name__ == "__main__":
    print(f"启动 MCP 服务器在 http://0.0.0.0:8000...", file=sys.stderr)
    print(f"要通过代理访问，请使用命令: mcp-proxy --sse-host=0.0.0.0 --sse-port=8080 --target-url=http://localhost:8000", file=sys.stderr)
    mcp.run('sse')      # 改为 8000 端口)
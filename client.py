import json
import asyncio
import re
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI

def format_tools_for_llm(tool) -> str:
    """对tool进行格式化
    Returns:
        格式化之后的tool描述
    """
    args_desc = []
    if "properties" in tool.inputSchema:
        for param_name, param_info in tool.inputSchema["properties"].items():
            arg_desc = (
                f"- {param_name}: {param_info.get('description', 'No description')}"
            )
            if param_name in tool.inputSchema.get("required", []):
                arg_desc += " (required)"
            args_desc.append(arg_desc)

    return f"Tool: {tool.name}\nDescription: {tool.description}\nArguments:\n{chr(10).join(args_desc)}"


class Client:
    def __init__(self):
        self._exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()  # 防止并发连接/断开问题
        self.is_connected = False
        self.client = OpenAI(
            base_url="http://10.1.51.23:9998/v1",
            api_key="YOUR-KEY-HERE",
        )
        self.model = "melhornes/gemma3-27b-tools:latest"
        self.messages = []

    async def connect_server(self, server_url):
        async with self._lock:  # 防止并发调用 connect
            print(f"尝试连接到: {server_url}")
            self._exit_stack = AsyncExitStack()
            # 1. 进入 SSE 上下文，但不退出
            sse_cm = sse_client(server_url)
            # 手动调用 __aenter__ 获取流，并存储上下文管理器以便后续退出
            streams = await self._exit_stack.enter_async_context(sse_cm)
            print("SSE 流已获取。")

            # 2. 进入 Session 上下文，但不退出
            session_cm = ClientSession(streams[0], streams[1])
            # 手动调用 __aenter__ 获取 session
            self.session = await self._exit_stack.enter_async_context(session_cm)
            print("ClientSession 已创建。")

            # 3. 初始化 Session
            await self.session.initialize()
            print("Session 已初始化。")

            # 4. 获取并存储工具列表
            response = await self.session.list_tools()
            self.tools = {tool.name: tool for tool in response.tools}
            print(f"成功获取 {len(self.tools)} 个工具:")
            for name, tool in self.tools.items():
                print(f"  - {name}: {tool.description}")

            print("连接成功并准备就绪。")

        # 修改系统提示
        tools_description = "\n".join([format_tools_for_llm(tool) for tool in response.tools])
        system_prompt = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            '"```json" is not allowed'
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        self.messages.append({"role": "system", "content": system_prompt})

    async def disconnect(self):
        """关闭 Session 和连接。"""
        async with self._lock:
            if self._exit_stack:
                await self._exit_stack.aclose()
                print("连接已关闭")

    async def chat(self, prompt, role="user"):
        """与LLM进行交互"""
        self.messages.append({"role": role, "content": prompt})

        # 初始化 LLM API 调用
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False,
        )
        llm_response = response.choices[0].message.content
        return llm_response

    async def execute_tool(self, llm_response: str):
        """Process the LLM response and execute tools if needed.
        Args:
            llm_response: The response from the LLM.
        Returns:
            The result of tool execution or the original response.
        """
        try:
            pattern = r"```json\n(.*?)\n?```"
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                llm_response = match.group(1)
            tool_call = json.loads(llm_response) 
            if "tool" in tool_call and "arguments" in tool_call:
                if tool_call["tool"] in self.tools:
                    try:
                        print(f"[提示]：正在调用工具 {tool_call['tool']}")
                        result = await self.session.call_tool(
                            tool_call["tool"], tool_call["arguments"]
                        )
                        print(f"[执行结果]: {result}")
                        return f"Tool execution result: {result}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        print(error_msg)
                        return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("MCP 客户端启动")
        print("输入 /bye 退出")

        while True:
            prompt = input(">>> ").strip()
            if "/bye" in prompt.lower():
                break

            response = await self.chat(prompt)
            self.messages.append({"role": "assistant", "content": response})
            print(f"[LLM响应]: {response}")

            result = await self.execute_tool(response)
            while result != response:
                response = await self.chat(result, "system")
                self.messages.append(
                    {"role": "assistant", "content": response}
                )
                print(f"[结果处理后的响应]: {response}")
                result = await self.execute_tool(response)
            
            if result == response:
                print(response)


async def main():
    try:
        client = Client()
        # 连接到 MCP SSE 服务端
        await client.connect_server("http://127.0.0.1:8080/sse")
        await client.chat_loop()
    except Exception as e:
        print(f"主程序发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 无论如何，最后都要尝试断开连接并清理资源
        print("\n正在关闭客户端...")
        await client.disconnect()
        print("客户端已关闭。")


if __name__ == '__main__':
    asyncio.run(main())
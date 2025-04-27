import json
import asyncio
import re
import sys
from typing import Optional, List, Dict, Any
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
        self.workflow_messages = []  # 用于工作流规划的消息历史

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

    async def plan_workflow(self, user_query: str) -> List[Dict[str, Any]]:
        """规划工具调用的工作流
        
        Args:
            user_query: 用户的问题
            
        Returns:
            计划的工作流步骤列表，每个步骤是一个工具调用
        """
        # 创建规划工作流的系统提示
        tools_description = "\n".join([format_tools_for_llm(tool) for tool in self.tools.values()])
        planning_prompt = (
            "你是一个工作流规划助手。你需要根据用户的问题规划一系列工具调用来解决问题。\n\n"
            f"可用工具:\n{tools_description}\n\n"
            "请分析用户问题，确定需要调用哪些工具以及调用顺序。"
            "如果问题可以直接回答而不需要工具，请返回空列表。"
            "返回JSON格式的工作流计划，格式如下:\n"
            "[\n"
            "  {\n"
            '    "tool": "工具名称",\n'
            '    "arguments": {\n'
            '      "参数名": "参数值"\n'
            "    },\n"
            '    "reason": "为什么需要调用这个工具"\n'
            "  },\n"
            "  ...\n"
            "]\n"
        )
        
        # 构建消息
        workflow_messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": f"为以下问题规划工作流: {user_query}"}
        ]
        
        # 调用LLM获取工作流计划
        response = self.client.chat.completions.create(
            model=self.model,
            messages=workflow_messages,
            stream=False,
        )
        plan_response = response.choices[0].message.content
        
        try:
            # 尝试从回复中提取JSON
            pattern = r"```json\n(.*?)\n?```"
            match = re.search(pattern, plan_response, re.DOTALL)
            if match:
                plan_response = match.group(1)
            
            workflow = json.loads(plan_response)
            if not isinstance(workflow, list):
                return []
                
            print(f"[工作流规划]: 计划执行 {len(workflow)} 个工具调用")
            for i, step in enumerate(workflow):
                print(f"  步骤 {i+1}: {step.get('tool')} - {step.get('reason', '无原因说明')}")
                
            return workflow
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[警告] 无法解析工作流计划: {e}")
            return []

    async def execute_workflow(self, workflow: List[Dict[str, Any]], user_query: str) -> str:
        """按顺序执行工作流中的工具
        
        Args:
            workflow: 工作流步骤列表
            user_query: 原始用户问题
            
        Returns:
            执行结果汇总
        """
        if not workflow:
            return "无需执行工具，直接回答问题。"
            
        results = []
        for i, step in enumerate(workflow):
            tool_name = step.get("tool")
            arguments = step.get("arguments", {})
            
            if tool_name not in self.tools:
                results.append(f"[错误] 步骤 {i+1}: 找不到工具 '{tool_name}'")
                continue
                
            try:
                print(f"[执行] 步骤 {i+1}: 调用工具 {tool_name}")
                result = await self.session.call_tool(tool_name, arguments)
                result_str = f"步骤 {i+1} ({tool_name}) 结果: {result}"
                print(f"[结果] {result_str}")
                results.append(result_str)
            except Exception as e:
                error_msg = f"步骤 {i+1} ({tool_name}) 执行失败: {str(e)}"
                print(f"[错误] {error_msg}")
                results.append(error_msg)
        
        # 将所有结果组合成一个字符串
        return "\n\n".join(results)

    async def evaluate_tools_results(self, tools_results: str, user_query: str) -> Dict[str, Any]:
        """评估工具执行结果是否足够回答用户问题
        
        Args:
            tools_results: 工具执行的结果
            user_query: 用户的原始问题
            
        Returns:
            评估结果，包括是否满足要求和可能需要的额外工具
        """
        eval_prompt = (
            "你是一个工具结果评估专家。请评估提供的工具执行结果是否足够回答用户的问题。"
            "只根据用户的原始问题进行判断，不推测用户可能的隐含意图或扩展需求。"
            "如果不足够，请指出还需要使用哪些工具来获取缺少的信息。\n\n"
            f"可用工具:\n{', '.join(self.tools.keys())}\n\n"
            "请返回JSON格式的评估结果，格式如下:\n"
            "{\n"
            '  "results_sufficient": true/false,\n'
            '  "missing_information": "描述缺少的信息",\n'
            '  "suggested_tools": [\n'
            '    {\n'
            '      "tool": "工具名称",\n'
            '      "arguments": {"参数名": "参数值"},\n'
            '      "reason": "为什么需要调用这个工具"\n'
            '    }\n'
            "  ]\n"
            "}\n"
        )
        
        eval_messages = [
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": f"用户问题: {user_query}\n\n工具执行结果: {tools_results}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=eval_messages,
            stream=False,
        )
        eval_response = response.choices[0].message.content
        
        try:
            pattern = r"```json\n(.*?)\n?```"
            match = re.search(pattern, eval_response, re.DOTALL)
            if match:
                eval_response = match.group(1)
                
            evaluation = json.loads(eval_response)
            print(f"[评估结果]: 工具结果{'' if evaluation.get('results_sufficient', True) else '不'}足够回答问题")
            if not evaluation.get('results_sufficient', True):
                print(f"[缺少信息]: {evaluation.get('missing_information', '未指定')}")
            return evaluation
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[警告] 无法解析评估结果: {e}")
            return {
                "results_sufficient": True,
                "missing_information": "无法确定",
                "suggested_tools": []
            }

    async def generate_final_answer(self, user_query: str, workflow_results: str) -> str:
        """基于工作流执行结果生成最终回答
        
        Args:
            user_query: 用户的原始问题
            workflow_results: 工作流执行的结果
            
        Returns:
            最终回答
        """
        answer_prompt = (
            "请基于执行的工具结果，为用户问题生成一个全面、准确的回答。"
            "回答应该直接针对用户的问题，并综合所有工具执行的结果。"
            "保持回答简洁但信息丰富，重点关注最相关的信息。"
        )
        
        answer_messages = [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": f"用户问题: {user_query}\n\n工具执行结果:\n{workflow_results}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=answer_messages,
            stream=False,
        )
        return response.choices[0].message.content

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("MCP 客户端启动")
        print("输入 /bye 退出")

        while True:
            user_query = input(">>> ").strip()
            if "/bye" in user_query.lower():
                break

            # 记录用户问题到消息历史
            self.messages.append({"role": "user", "content": user_query})
            
            # 1. 规划工作流
            print("[系统] 正在规划工作流...")
            workflow = await self.plan_workflow(user_query)
            
            # 2. 如果有工作流，按顺序执行工具
            if workflow:
                print("[系统] 开始执行工作流...")
                workflow_results = await self.execute_workflow(workflow, user_query)
            else:
                workflow_results = "无需执行工具，直接回答问题。"
                
            # 3. 评估工具执行结果是否足够
            max_iterations = 3  # 限制最大迭代次数，防止无限循环
            current_results = workflow_results
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # 评估工具结果
                print(f"[系统] 评估工具执行结果 (迭代 {iteration}/{max_iterations})...")
                evaluation = await self.evaluate_tools_results(current_results, user_query)
                
                # 如果结果足够或没有建议的工具，跳出循环
                if evaluation.get("results_sufficient", True) or not evaluation.get("suggested_tools"):
                    break
                    
                # 执行建议的额外工具
                print(f"[系统] 工具结果不足，正在执行额外工具...")
                additional_workflow = evaluation.get("suggested_tools", [])
                
                if additional_workflow:
                    additional_results = await self.execute_workflow(additional_workflow, user_query)
                    # 合并之前的结果
                    current_results += f"\n\n迭代 {iteration} 额外工具执行结果:\n{additional_results}"
                else:
                    break
            
            # 4. 根据最终工具结果生成最终回答
            print("[系统] 基于工具结果生成最终回答...")
            final_answer = await self.generate_final_answer(user_query, current_results)
            
            # 5. 添加最终答案到消息历史并输出
            self.messages.append({"role": "assistant", "content": final_answer})
            print(f"\n[回答]:\n{final_answer}\n")


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
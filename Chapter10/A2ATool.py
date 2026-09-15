from ..Chapter7.Tool import Tool, ToolParameter
from A2A_official import A2AClientWrapper
from typing import Dict, Any, List
import asyncio


class A2ATool(Tool):
    def __init__(self, url: str):
        super().__init__(
            name="a2a_tool",
            description="A2A 工具，用于与 A2A 服务器交互，实现智能体之间的通信"
        )
        self.url = url

    def run(self, parameters: Dict[str, Any]):
        """同步入口：将异步逻辑封装在 asyncio.run() 中，确保 Agent 框架能正常调用。

        Tool 基类的 run() 是同步方法，但 A2A 通信需要使用 async/await。
        这里用 asyncio.run() 桥接同步和异步。
        """
        return asyncio.run(self._run(parameters))

    async def _run(self, parameters: Dict[str, Any]):
        required_parameters = [p.name for p in self.get_parameters() if p.required]
        if not all(param in parameters for param in required_parameters):
            return f"缺少必填参数: {', '.join(required_parameters)}"

        action = parameters.get("action")
        if not action:
            return "缺少必填参数: action"

        client = A2AClientWrapper(self.url)
        try:
            await client.connect()

            if action == "task":
                question = parameters.get("question")
                if not question:
                    return "缺少必填参数: question"
                result = await client.send_message(question)
                return result
            elif action == "get_info":
                result = await client.get_info()
                return str(result)
            else:
                return f"未知操作: {action}"
        except Exception as e:
            return f"执行失败: {str(e)}"
        finally:
            await client.disconnect()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：task(执行任务), get_info(获取信息)",
                required=True
            ),
            ToolParameter(
                name="question",
                type="string",
                description="要执行的任务或问题(仅在 action 为 task 时必填)",
                required=False
            ),
        ]
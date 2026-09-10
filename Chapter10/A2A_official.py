"""
基于官方 a2a-sdk 的 A2A 协议实现

使用官方 a2a-sdk 库实现完整的 Agent-to-Agent Protocol 功能。
官方仓库: https://github.com/a2aproject/a2a-python
安装: pip install a2a-sdk[server]

本实现包含:
- AgentCard: 标准化的 Agent 能力描述名片 (/.well-known/agent-card.json)
- AgentExecutor: Agent 核心执行逻辑
- JSON-RPC 2.0: 标准通信协议 (tasks/send, tasks/get, tasks/cancel)
- Task 模型: 带状态机的任务生命周期管理
- 结构化消息: Message + Part (TextPart/DataPart) 格式
- 客户端: 基于官方 ClientFactory 的 A2A 客户端
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import uuid

from google.protobuf import struct_pb2

from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue, InMemoryQueueManager
from a2a.server.request_handlers import (
    LegacyRequestHandler,
    SimpleRequestContextBuilder,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.routes.common import (
    DefaultServerCallContextBuilder,
    ServerCallContextBuilder,
)
from a2a.helpers.proto_helpers import get_message_text, new_text_message
from a2a.helpers.agent_card import display_agent_card
from a2a.client.client import Client, ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.types import SendMessageRequest, SendMessageConfiguration, GetTaskRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


# ============================================================================
# Agent Card 构建
# ============================================================================

def build_agent_card(
    name: str = "A2A Example Agent",
    description: str = "An example agent demonstrating A2A protocol compliance",
    version: str = "1.0.0",
    url: str = "http://localhost:8000",
) -> AgentCard:
    """构建符合 A2A 协议规范的 AgentCard。

    AgentCard 是 A2A 协议的核心发现机制，每个 Agent 通过
    /.well-known/agent-card.json 端点暴露此卡片，供其他 Agent 发现和调用。

    Args:
        name: Agent 名称
        description: Agent 描述
        version: Agent 版本
        url: Agent 服务基础 URL

    Returns:
        完整的 AgentCard 对象
    """
    return AgentCard(
        name=name,
        description=description,
        version=version,
        provider=AgentProvider(
            organization="A2A Demo Organization",
            url=url,
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            extended_agent_card=False,
        ),
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding="jsonrpc",
                protocol_version="1.0",
            ),
        ],
        skills=[
            AgentSkill(
                id="calculate",
                name="calculate",
                description="计算数学表达式，支持加减乘除和括号",
                tags=["math", "calculation", "arithmetic"],
                examples=["calculate 2 + 3 * 4", "calculate (10 + 5) / 3"],
                input_modes=["text", "text/plain"],
                output_modes=["text", "text/plain"],
            ),
            AgentSkill(
                id="greet",
                name="greet",
                description="生成友好的问候语",
                tags=["greeting", "social"],
                examples=["hello", "hi there", "greet me"],
                input_modes=["text", "text/plain"],
                output_modes=["text", "text/plain"],
            ),
            AgentSkill(
                id="summarize",
                name="summarize",
                description="对输入文本进行摘要",
                tags=["text", "nlp", "summarization"],
                examples=["summarize: 这是一段很长的文本..."],
                input_modes=["text", "text/plain"],
                output_modes=["text", "text/plain"],
            ),
        ],
    )


# ============================================================================
# Agent Executor - 核心执行逻辑
# ============================================================================

class ExampleAgentExecutor(AgentExecutor):
    """A2A Agent 执行器，包含 Agent 的核心业务逻辑。

    实现 AgentExecutor 接口的 execute() 和 cancel() 方法。
    execute() 是 Agent 的主要入口，接收用户消息并产生响应。
    """

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """执行 Agent 核心逻辑。

        读取 RequestContext 中的用户消息，根据消息内容匹配合适的技能，
        并将响应通过 EventQueue 发布。

        Args:
            context: 请求上下文，包含用户消息、任务ID等信息
            event_queue: 事件队列，用于发布 Task/Message 等事件
        """
        user_text = self._extract_user_text(context)

        if not user_text:
            await event_queue.enqueue_event(
                new_text_message(
                    text="请提供您的问题或指令。",
                    context_id=context.context_id,
                    task_id=context.task_id,
                )
            )
            return

        task_id = context.task_id
        context_id = context.context_id

        # 发布 Task 对象，表示开始处理任务（异步/长时间运行模式）
        task = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.TASK_STATE_SUBMITTED,
                timestamp=None,
            ),
        )
        await event_queue.enqueue_event(task)

        # 发布状态更新：工作中
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_WORKING,
                    message=new_text_message(
                        text="正在处理您的问题...",
                        context_id=context_id,
                        task_id=task_id,
                    ),
                ),
            )
        )

        # 技能匹配与执行
        response_text = self._process_skill(user_text, context_id, task_id)

        # 发布最终响应消息
        await event_queue.enqueue_event(
            new_text_message(
                text=response_text,
                context_id=context_id,
                task_id=task_id,
            )
        )

        # 发布状态更新：完成
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_COMPLETED,
                    message=new_text_message(
                        text="任务处理完成。",
                        context_id=context_id,
                        task_id=task_id,
                    ),
                ),
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """取消正在执行的任务。

        Args:
            context: 请求上下文
            event_queue: 事件队列
        """
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_CANCELED,
                    message=new_text_message(
                        text="任务已被取消。",
                        context_id=context.context_id,
                        task_id=context.task_id,
                    ),
                ),
            )
        )

    def _extract_user_text(self, context: RequestContext) -> str:
        """从请求上下文中提取用户输入的文本。

        Args:
            context: 请求上下文

        Returns:
            用户输入的文本字符串
        """
        if context.params and context.params.message:
            return get_message_text(context.params.message)
        return ""

    def _process_skill(
        self, text: str, context_id: str | None, task_id: str | None
    ) -> str:
        """根据输入文本匹配合适的技能并执行。

        Args:
            text: 用户输入文本
            context_id: 上下文ID
            task_id: 任务ID

        Returns:
            技能执行结果文本
        """
        text_lower = text.lower().strip()

        # 匹配计算技能
        if any(kw in text_lower for kw in ("calculate", "计算", "calc", "compute")):
            return self._calculate(text)

        # 匹配问候技能
        if any(kw in text_lower for kw in ("hello", "hi", "你好", "greet", "问候")):
            return self._greet(text)

        # 匹配摘要技能
        if any(kw in text_lower for kw in ("summarize", "summary", "摘要", "总结")):
            return self._summarize(text)

        # 默认响应
        return (
            "您好！我是 A2A 协议兼容的 Agent。\n"
            "我可以帮您做以下事情：\n"
            "  - 计算数学表达式 (例如: calculate 2 + 3 * 4)\n"
            "  - 问候语 (例如: hello)\n"
            "  - 文本摘要 (例如: summarize: 一段文字)\n\n"
            "请告诉我您需要什么帮助？"
        )

    def _calculate(self, text: str) -> str:
        """安全地计算数学表达式。

        Args:
            text: 包含数学表达式的文本

        Returns:
            计算结果
        """
        match = re.search(r"(?:calculate|计算|calc|compute)\s+(.+)", text, re.IGNORECASE)
        if match:
            expression = match.group(1).strip()
            allowed_chars = set("0123456789+-*/() .^%")
            if not all(c in allowed_chars for c in expression):
                return f"错误: 表达式包含不允许的字符。仅支持数字和运算符 + - * / ( ) ^ %"
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return f"计算结果: {expression} = {result}"
            except Exception as e:
                return f"计算错误: {str(e)}"
        return "请提供要计算的表达式，例如: calculate 2 + 3"

    def _greet(self, text: str) -> str:
        """生成问候语。

        Args:
            text: 用户输入

        Returns:
            问候语
        """
        return (
            "Hello! 👋\n"
            "我是基于 A2A 协议构建的智能 Agent。\n"
            "我可以帮您进行计算、文本摘要等任务。\n"
            "有什么我可以帮您的吗？"
        )

    def _summarize(self, text: str) -> str:
        """对文本进行简单摘要。

        Args:
            text: 用户输入文本（包含 "summarize:" 前缀后的内容）

        Returns:
            摘要结果
        """
        match = re.search(
            r"(?:summarize|summary|摘要|总结)\s*[:：]?\s*(.+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            content = match.group(1).strip()
            if len(content) < 50:
                return f"摘要: 文本较短，原文即摘要:\n{content}"
            # 简单摘要：取前100个字符 + "..."
            summary = content[:100] + "..." if len(content) > 100 else content
            return (
                f"摘要 (原文本 {len(content)} 字符):\n"
                f"{summary}\n\n"
                f"提示: 这是简单的截断摘要，实际场景可接入 LLM 进行智能摘要。"
            )
        return "请提供要摘要的文本，例如: summarize: 这是一段很长的文本..."


# ============================================================================
# A2A Server - 服务器搭建
# ============================================================================

class A2AServerBuilder:
    """A2A 服务器构建器，封装服务器所需的全部组件。

    使用官方 a2a-sdk 的组件组装完整的 A2A 服务器：
    - AgentCard: 能力描述
    - AgentExecutor: 执行逻辑
    - TaskStore: 任务存储
    - QueueManager: 事件队列管理
    - RequestHandler: 请求处理
    - Starlette 路由: HTTP 端点
    """

    def __init__(
        self,
        agent_card: AgentCard | None = None,
        agent_executor: AgentExecutor | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        Args:
            agent_card: AgentCard 实例
            agent_executor: AgentExecutor 实例
            host: 服务器主机地址
            port: 服务器端口
        """
        self.agent_card = agent_card or build_agent_card()
        self.agent_executor = agent_executor or ExampleAgentExecutor()
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"

        # 更新 AgentCard 中的 URL
        self.agent_card.supported_interfaces[0].url = self.url

        # 初始化核心组件
        self.task_store = InMemoryTaskStore()
        self.queue_manager = InMemoryQueueManager()

        self.request_handler = LegacyRequestHandler(
            agent_executor=self.agent_executor,
            task_store=self.task_store,
            agent_card=self.agent_card,
            queue_manager=self.queue_manager,
            request_context_builder=SimpleRequestContextBuilder(
                task_store=self.task_store,
            ),
        )

    def build_app(self):
        """构建 Starlette 应用，包含所有 A2A 协议路由。

        Returns:
            Starlette Application 实例
        """
        try:
            from starlette.applications import Starlette
        except ImportError:
            raise ImportError(
                "A2A server requires starlette. "
                "Install with: pip install a2a-sdk[server]"
            )

        # 创建 routes
        routes = []

        # 1. Agent Card 路由 (/.well-known/agent-card.json)
        routes.extend(
            create_agent_card_routes(
                agent_card=self.agent_card,
                card_url=AGENT_CARD_WELL_KNOWN_PATH,
            )
        )

        # 2. JSON-RPC 路由 (处理 tasks/send, tasks/get, tasks/cancel 等)
        routes.extend(
            create_jsonrpc_routes(
                request_handler=self.request_handler,
                rpc_url="/",
                context_builder=DefaultServerCallContextBuilder(),
            )
        )

        # 3. 健康检查路由
        async def health_check(request):
            from starlette.responses import JSONResponse
            return JSONResponse({
                "status": "healthy",
                "agent": self.agent_card.name,
                "protocol": "A2A",
                "version": self.agent_card.version,
            })

        routes.append(
            type(
                "Route",
                (),
                {"path": "/health", "endpoint": health_check, "methods": ["GET"]},
            )
        )

        app = Starlette(routes=routes)
        return app

    def run(self) -> None:
        """启动 A2A 服务器。"""
        import uvicorn

        app = self.build_app()

        # 打印服务器信息
        print("=" * 60)
        print("  A2A Protocol Server")
        print("=" * 60)
        print()
        display_agent_card(self.agent_card)
        print()
        print("🚀 A2A 服务器启动中...")
        print(f"📍 URL: {self.url}")
        print(f"📇 Agent Card: {self.url}{AGENT_CARD_WELL_KNOWN_PATH}")
        print(f"📡 JSON-RPC: {self.url}/")
        print(f"❤️  Health: {self.url}/health")
        print()

        uvicorn.run(app, host=self.host, port=self.port, log_level="warning")


# ============================================================================
# A2A Client - 客户端
# ============================================================================

class A2AClientWrapper:
    """A2A 客户端封装，基于官方 ClientFactory。

    使用官方 a2a-sdk 的 ClientFactory 自动发现 AgentCard 并建立连接。
    支持 JSON-RPC 协议进行标准化的 A2A 通信。
    """

    def __init__(self, server_url: str):
        """
        Args:
            server_url: A2A 服务器 URL
        """
        self.server_url = server_url.rstrip("/")
        self._client: Client | None = None
        self._config = ClientConfig(
            streaming=True,
            polling=False,
            accepted_output_modes=["text", "text/plain"],
        )

    async def connect(self) -> None:
        """连接到 A2A 服务器，自动发现 AgentCard 并建立客户端。"""
        factory = ClientFactory(self._config)
        self._client = await factory.create_from_url(self.server_url)
        print(f"✅ 已连接到 A2A Agent: {self.server_url}")

    async def disconnect(self) -> None:
        """断开与 A2A 服务器的连接。"""
        if self._client:
            await self._client.close()
            self._client = None
            print("🔌 已断开连接")

    async def send_message(self, text: str, task_id: str | None = None) -> str:
        """向 A2A Agent 发送消息并获取响应。

        使用标准的 A2A 协议：构建 SendMessageRequest，
        通过 JSON-RPC tasks/send 方法发送。

        Args:
            text: 消息文本
            task_id: 可选的任务ID（用于恢复已有任务）

        Returns:
            Agent 的响应文本
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        request = SendMessageRequest(
            message=Message(
                message_id=str(uuid.uuid4()),
                role=Role.ROLE_USER,
                parts=[Part(text=text)],
                task_id=task_id or "",
            ),
            configuration=SendMessageConfiguration(
                accepted_output_modes=["text", "text/plain"],
                return_immediately=False,
            ),
        )

        responses: list[str] = []
        async for response in self._client.send_message(request):
            if response.HasField("message"):
                responses.append(get_message_text(response.message))
            elif response.HasField("task"):
                task = response.task
                if task.status.state == TaskState.TASK_STATE_COMPLETED:
                    if task.artifacts:
                        for artifact in task.artifacts:
                            responses.append(
                                f"[Artifact: {artifact.name}] {get_message_text_from_parts(artifact.parts)}"
                            )
                elif task.status.state == TaskState.TASK_STATE_FAILED:
                    responses.append(
                        f"任务失败: {get_message_text(task.status.message)}"
                    )
                elif task.status.state == TaskState.TASK_STATE_CANCELED:
                    responses.append("任务已被取消。")
            elif response.HasField("status_update"):
                pass  # 中间状态更新，可忽略或记录
            elif response.HasField("artifact_update"):
                artifact = response.artifact_update.artifact
                responses.append(
                    f"[Artifact: {artifact.name}] {get_message_text_from_parts(artifact.parts)}"
                )

        return "\n".join(responses) if responses else "No response received."

    async def get_task(self, task_id: str) -> str:
        """查询任务状态。

        使用标准的 A2A 协议 tasks/get 方法。

        Args:
            task_id: 任务ID

        Returns:
            任务状态描述
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        task = await self._client.get_task(GetTaskRequest(id=task_id))
        state_name = TaskState.Name(task.status.state)
        return f"Task {task_id}: {state_name}"

    async def get_info(self) -> dict:
        """获取 AgentCard 信息。"""
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        # 通过 ClientFactory 的 card_resolver 获取 AgentCard
        from a2a.client.card_resolver import A2ACardResolver
        resolver = A2ACardResolver()
        card = await resolver.resolve(self.server_url)
        return {
            "name": card.name,
            "description": card.description,
            "version": card.version,
            "skills": [{"id": s.id, "name": s.name, "description": s.description} for s in card.skills],
            "capabilities": {
                "streaming": card.capabilities.streaming,
                "push_notifications": card.capabilities.push_notifications,
            },
        }


def get_message_text_from_parts(parts) -> str:
    """从 Part 列表中提取文本。

    Args:
        parts: Part 列表

    Returns:
        拼接后的文本
    """
    texts = []
    for part in parts:
        if part.text:
            texts.append(part.text)
    return "\n".join(texts)


# ============================================================================
# 示例：客户端交互
# ============================================================================

async def run_client_example(server_url: str = "http://localhost:8000"):
    """运行客户端示例，演示完整的 A2A 协议交互流程。

    Args:
        server_url: A2A 服务器 URL
    """
    client = A2AClientWrapper(server_url)
    try:
        await client.connect()

        # 获取 Agent 信息
        print("\n" + "=" * 60)
        print("  获取 Agent 信息")
        print("=" * 60)
        info = await client.get_info()
        print(f"名称: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"版本: {info['version']}")
        print(f"技能: {[s['name'] for s in info['skills']]}")

        # 测试计算技能
        print("\n" + "=" * 60)
        print("  测试: 计算技能")
        print("=" * 60)
        result = await client.send_message("calculate 2 + 3 * 4")
        print(f"问题: calculate 2 + 3 * 4")
        print(f"回答: {result}")

        # 测试问候技能
        print("\n" + "=" * 60)
        print("  测试: 问候技能")
        print("=" * 60)
        result = await client.send_message("Hello, how are you?")
        print(f"问题: Hello, how are you?")
        print(f"回答: {result}")

        # 测试摘要技能
        print("\n" + "=" * 60)
        print("  测试: 摘要技能")
        print("=" * 60)
        long_text = (
            "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
            "这些任务包括学习、推理、问题解决、感知和语言理解。近年来，随着深度学习技术的突破，"
            "AI 在图像识别、自然语言处理和自动驾驶等领域取得了显著进展。"
        )
        result = await client.send_message(f"summarize: {long_text}")
        print(f"问题: summarize: {long_text}")
        print(f"回答: {result}")

    finally:
        await client.disconnect()


# ============================================================================
# 主入口
# ============================================================================

def main():
    """主入口函数，支持服务器模式和客户端模式。

    服务器模式:
        python A2A_official.py --mode server --port 8000

    客户端模式:
        python A2A_official.py --mode client --url http://localhost:8000
    """
    parser = argparse.ArgumentParser(
        description="A2A Protocol - 基于官方 a2a-sdk 的实现"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "client"],
        default="server",
        help="运行模式: server 或 client (默认: server)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="客户端连接的服务器 URL (默认: http://localhost:8000)",
    )

    args = parser.parse_args()

    if args.mode == "server":
        builder = A2AServerBuilder(
            host=args.host,
            port=args.port,
        )
        builder.run()
    elif args.mode == "client":
        asyncio.run(run_client_example(args.url))


if __name__ == "__main__":
    main()
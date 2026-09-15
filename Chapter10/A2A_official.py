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
from dataclasses import dataclass, field
from typing import Any, Callable

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
from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
    SimpleRequestContextBuilder,
)
from a2a.server.events import EventQueue, InMemoryQueueManager
from a2a.server.request_handlers import LegacyRequestHandler
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
# Skill 注册中心 - 支持装饰器动态注册
# ============================================================================


@dataclass
class SkillDefinition:
    """Skills 的定义信息，用于生成 AgentCard 中的 AgentSkill。"""
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text", "text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text", "text/plain"])


class SkillRegistry:
    """Skill 注册中心，支持装饰器和普通方法两种方式注册 skill。

    装饰器用法::

        registry = SkillRegistry()

        @registry.skill("calculate", description="计算数学表达式", keywords=["calc", "计算"])
        def calculate(text, context_id, task_id):
            return "计算结果: 42"

        @registry.skill  # 自动使用函数名作为 skill_id
        def greet(text, context_id, task_id):
            return "Hello!"

    普通方法用法::

        registry.register("translate", translate_func,
                           description="翻译功能", keywords=["translate", "翻译"])
    """

    def __init__(self):
        self._skills: dict[str, tuple[SkillDefinition, Callable]] = {}
        self._keywords: dict[str, list[str]] = {}

    # ======== 装饰器入口 ========
    def skill(
        self,
        skill_id_or_func: str | Callable,
        *,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        keywords: list[str] | None = None,
    ):
        """装饰器：将函数注册为一个 skill。

        支持两种用法::

            # 用法1: 带参数的装饰器
            @registry.skill("greet", description="生成问候语", keywords=["hello", "hi"])
            def greet(text, ctx_id, task_id):
                return "Hello!"

            # 用法2: 不带参数的装饰器（以函数名作为 skill_id）
            @registry.skill
            def calculate(text, ctx_id, task_id):
                return "结果"

        Args:
            skill_id_or_func: skill ID 字符串，或直接传入函数（不带括号调用时）
            name: skill 显示名称，默认与 skill_id 相同
            description: skill 描述
            tags: 标签列表
            examples: 示例列表
            keywords: 触发关键词列表
        """
        if callable(skill_id_or_func):
            func = skill_id_or_func
            return self._register(
                func=func,
                skill_id=func.__name__,
                name=func.__name__,
                description=func.__doc__ or "",
                tags=[],
                examples=[],
                keywords=[],
            )

        def decorator(func):
            return self._register(
                func=func,
                skill_id=skill_id_or_func,
                name=name or skill_id_or_func,
                description=description,
                tags=tags or [],
                examples=examples or [],
                keywords=keywords or [],
            )
        return decorator

    # ======== 普通方法注册入口 ========
    def register(
        self,
        skill_id: str,
        func: Callable,
        *,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> Callable:
        """以普通方法的方式注册一个 skill 处理函数。

        Args:
            skill_id: skill ID
            func: 处理函数
            name: skill 显示名称
            description: skill 描述
            tags: 标签列表
            examples: 示例列表
            keywords: 触发关键词列表

        Returns:
            原函数（不改变其行为）
        """
        return self._register(
            func=func,
            skill_id=skill_id,
            name=name or skill_id,
            description=description,
            tags=tags or [],
            examples=examples or [],
            keywords=keywords or [],
        )

    def _register(
        self,
        func: Callable,
        skill_id: str,
        name: str,
        description: str,
        tags: list[str],
        examples: list[str],
        keywords: list[str],
    ):
        """内部注册逻辑。"""
        definition = SkillDefinition(
            id=skill_id,
            name=name,
            description=description,
            tags=tags,
            examples=examples,
        )
        self._skills[skill_id] = (definition, func)
        self._keywords[skill_id] = keywords
        print(f"📝 Skill 已注册: [{skill_id}] {name}")
        return func

    # ======== 查询方法 ========
    def find_handler(self, text: str) -> tuple[SkillDefinition, Callable] | None:
        """根据输入文本的关键词匹配找到对应的 skill handler。

        多个关键词同时匹配时，选择在文本中出现位置最早的那个，
        避免"计算机"误匹配"计算"这类子串问题。

        Args:
            text: 用户输入文本

        Returns:
            (SkillDefinition, handler_func) 或 None
        """
        text_lower = text.lower()
        best_match = None  # (position, skill_id)
        for skill_id, kw_list in self._keywords.items():
            if not kw_list:
                continue
            for kw in kw_list:
                pos = text_lower.find(kw)
                if pos != -1:
                    if best_match is None or pos < best_match[0]:
                        best_match = (pos, skill_id)
        if best_match:
            return self._skills[best_match[1]]
        return None

    def get_agent_skills(self) -> list[AgentSkill]:
        """生成 AgentCard 所需的 skills 列表。"""
        return [
            AgentSkill(
                id=defn.id,
                name=defn.name,
                description=defn.description,
                tags=defn.tags,
                examples=defn.examples,
                input_modes=defn.input_modes,
                output_modes=defn.output_modes,
            )
            for defn, _func in self._skills.values()
        ]

    @property
    def skill_count(self) -> int:
        """已注册的 skill 数量。"""
        return len(self._skills)


# ============================================================================
# Agent Card 构建
# ============================================================================

def build_agent_card(
    name: str = "A2A Example Agent",
    description: str = "An example agent demonstrating A2A protocol compliance",
    version: str = "1.0.0",
    url: str = "http://localhost:8000",
    skills: list[AgentSkill] | None = None,
) -> AgentCard:
    """构建符合 A2A 协议规范的 AgentCard。

    AgentCard 是 A2A 协议的核心发现机制，每个 Agent 通过
    /.well-known/agent-card.json 端点暴露此卡片，供其他 Agent 发现和调用。

    Args:
        name: Agent 名称
        description: Agent 描述
        version: Agent 版本
        url: Agent 服务基础 URL
        skills: 可选的 skills 列表。若为 None，则使用默认的硬编码 skills。

    Returns:
        完整的 AgentCard 对象
    """
    if skills is None:
        skills = [
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
        ]

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
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ],
        skills=skills,
    )


# ============================================================================
# Agent Executor - 核心执行逻辑
# ============================================================================

# 默认兜底响应（当没有任何 skill 匹配时使用）
_DEFAULT_FALLBACK_RESPONSE = (
    "您好！我是 A2A 协议兼容的 Agent。\n"
    "我可以帮您做以下事情：\n"
    "  - 计算数学表达式 (例如: calculate 2 + 3 * 4)\n"
    "  - 问候语 (例如: hello)\n"
    "  - 文本摘要 (例如: summarize: 一段文字)\n\n"
    "请告诉我您需要什么帮助？"
)


class ExampleAgentExecutor(AgentExecutor):
    """A2A Agent 执行器，基于 SkillRegistry 进行技能匹配与执行。

    实现 AgentExecutor 接口的 execute() 和 cancel() 方法。
    execute() 是 Agent 的主要入口，接收用户消息并产生响应。

    技能匹配通过 SkillRegistry 完成，支持通过装饰器动态注册技能。
    """

    def __init__(self, registry: SkillRegistry | None = None):
        """
        Args:
            registry: SkillRegistry 实例。若为 None，将使用空 registry
                      且仅返回默认兜底响应。
        """
        self._registry = registry or SkillRegistry()

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """执行 Agent 核心逻辑。

        读取 RequestContext 中的用户消息，通过 SkillRegistry 匹配合适的技能，
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

        # 通过 SkillRegistry 进行技能匹配与执行
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
        return context.get_user_input()

    def _process_skill(
        self, text: str, context_id: str | None, task_id: str | None
    ) -> str:
        """通过 SkillRegistry 匹配并执行技能。

        优先通过关键词匹配已注册的 skill handler；
        若没有匹配到任何 skill，则返回默认兜底响应。

        Args:
            text: 用户输入文本
            context_id: 上下文ID
            task_id: 任务ID

        Returns:
            技能执行结果文本
        """
        result = self._registry.find_handler(text)
        if result:
            _definition, handler_func = result
            return handler_func(text, context_id, task_id)

        return _DEFAULT_FALLBACK_RESPONSE


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

    支持通过装饰器动态注册 skill::

        server = A2AServerBuilder()

        @server.skill("calculate", description="计算数学表达式", keywords=["calc", "计算"])
        def calculate(text, ctx_id, task_id):
            return "计算结果: 42"
    """

    def __init__(
        self,
        agent_card: AgentCard | None = None,
        agent_executor: AgentExecutor | None = None,
        skill_registry: SkillRegistry | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        Args:
            agent_card: AgentCard 实例
            agent_executor: AgentExecutor 实例
            skill_registry: SkillRegistry 实例（与 agent_executor 配合使用）
            host: 服务器绑定地址（0.0.0.0 表示监听所有网络接口）
            port: 服务器端口
        """
        self._registry = skill_registry or SkillRegistry()

        # 如果传入了 agent_card 但不指定 skills，则从 registry 获取
        self.agent_card = agent_card or build_agent_card(
            skills=self._registry.get_agent_skills()
        )
        self.agent_executor = agent_executor or ExampleAgentExecutor(
            registry=self._registry
        )
        self.host = host
        self.port = port
        # 对外公布的 URL：0.0.0.0 只是绑定地址，客户端需要用 localhost 或实际 IP 访问
        advertised_host = "localhost" if host in ("0.0.0.0", "::") else host
        self.url = f"http://{advertised_host}:{port}"

        # 更新 AgentCard 中的 URL
        if self.agent_card.supported_interfaces:
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

    # ======== 装饰器入口：支持 @server.skill(...) ========
    def skill(self, skill_id=None, **kwargs):
        """装饰器：将函数注册为一个 skill 并同步更新 AgentCard。

        支持两种用法::

            # 用法1: 带参数的装饰器
            @server.skill("greet", description="生成问候语", keywords=["hello", "hi"])
            def greet(text, ctx_id, task_id):
                return "Hello!"

            # 用法2: 不带参数的装饰器（以函数名作为 skill_id）
            @server.skill
            def calculate(text, ctx_id, task_id):
                return "结果"

        Args:
            skill_id: skill ID（可选，不带括号调用时传入函数）
            **kwargs: 传递给 SkillRegistry.skill() 的其他参数
        """
        # 同步更新 AgentCard 中的 skills 列表
        original_skill = self._registry.skill
        registry = self._registry

        if callable(skill_id):
            # @server.skill 不带括号的用法
            result = original_skill(skill_id, **kwargs)
            self._sync_agent_card_skills()
            return result

        # @server.skill("id", ...) 带参数的用法
        def decorator(func):
            result = original_skill(skill_id, **kwargs)(func)
            self._sync_agent_card_skills()
            return result
        return decorator

    def _sync_agent_card_skills(self):
        """从 registry 同步 skills 到 AgentCard。"""
        new_skills = self._registry.get_agent_skills()
        self.agent_card.ClearField("skills")
        self.agent_card.skills.extend(new_skills)

    def build_app(self):
        """构建 Starlette 应用，包含所有 A2A 协议路由。

        在构建前会从 SkillRegistry 同步最新的 skills 到 AgentCard。

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

        # 在构建路由前同步 skills
        self._sync_agent_card_skills()

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

        card = self._client._card
        return {
            "name": card.name,
            "description": card.description,
            "version": card.version,
            "skills": [
                {"id": s.id, "name": s.name, "description": s.description}
                for s in card.skills
            ],
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

        # ================================================================
        # 通过装饰器动态注册 Skills
        # 每个 skill 是一个独立的函数，通过 @builder.skill(...) 注册
        # ================================================================

        @builder.skill(
            "calculate",
            name="calculate",
            description="计算数学表达式，支持加减乘除和括号",
            tags=["math", "calculation", "arithmetic"],
            examples=["calculate 2 + 3 * 4", "calculate (10 + 5) / 3"],
            keywords=["calculate", "计算", "calc", "compute"],
        )
        def calculate(text, context_id, task_id):
            match = re.search(
                r"(?:calculate|计算|calc|compute)\s+(.+)", text, re.IGNORECASE
            )
            if match:
                expression = match.group(1).strip()
                allowed_chars = set("0123456789+-*/() .^%")
                if not all(c in allowed_chars for c in expression):
                    return (
                        "错误: 表达式包含不允许的字符。"
                        "仅支持数字和运算符 + - * / ( ) ^ %"
                    )
                try:
                    result = eval(expression, {"__builtins__": {}}, {})
                    return f"计算结果: {expression} = {result}"
                except Exception as e:
                    return f"计算错误: {str(e)}"
            return "请提供要计算的表达式，例如: calculate 2 + 3"

        @builder.skill(
            "greet",
            name="greet",
            description="生成友好的问候语",
            tags=["greeting", "social"],
            examples=["hello", "hi there", "greet me"],
            keywords=["hello", "hi", "你好", "greet", "问候"],
        )
        def greet(text, context_id, task_id):
            return (
                "Hello! 👋\n"
                "我是基于 A2A 协议构建的智能 Agent。\n"
                "我可以帮您进行计算、文本摘要等任务。\n"
                "有什么我可以帮您的吗？"
            )

        @builder.skill(
            "summarize",
            name="summarize",
            description="对输入文本进行摘要",
            tags=["text", "nlp", "summarization"],
            examples=["summarize: 这是一段很长的文本..."],
            keywords=["summarize", "summary", "摘要", "总结"],
        )
        def summarize(text, context_id, task_id):
            match = re.search(
                r"(?:summarize|summary|摘要|总结)\s*[:：]?\s*(.+)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                content = match.group(1).strip()
                if len(content) < 50:
                    return f"摘要: 文本较短，原文即摘要:\n{content}"
                summary = content[:100] + "..." if len(content) > 100 else content
                return (
                    f"摘要 (原文本 {len(content)} 字符):\n"
                    f"{summary}\n\n"
                    f"提示: 这是简单的截断摘要，实际场景可接入 LLM 进行智能摘要。"
                )
            return "请提供要摘要的文本，例如: summarize: 这是一段很长的文本..."

        builder.run()
    elif args.mode == "client":
        asyncio.run(run_client_example(args.url))


if __name__ == "__main__":
    main()
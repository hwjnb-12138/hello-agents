from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
import math

from ..Chapter7.Message import Message
from ..Chapter8.MemoryTool import MemoryTool


@dataclass
class ContextPacket:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token: int = 0
    relevance: float = 0.0

    def __post_init__(self):
        if self.token == 0:
            self.token = count_tokens(self.content)


@dataclass
class ContextConfig:
    max_tokens: int = 8000
    reserve_ratio: float = 0.15  # 生成余量（10-20%）
    min_relevance: float = 0.3  # 最小相关性阈值
    enable_mmr: bool = True  # 启用最大边际相关性（多样性）
    mmr_lambda: float = 0.7  # MMR平衡参数（0=纯多样性, 1=纯相关性）
    system_prompt_template: str = ""  # 系统提示模板
    enable_compression: bool = True  # 启用压缩
    
    def get_available_tokens(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.config = config or ContextConfig()
        self.memory_tool = memory_tool
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def build(
        self,
        query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instruction: Optional[str] = None,
        additional_packet: Optional[List[ContextPacket]] = None
    ):
        packets = self._gather(query, conversation_history, system_instruction, additional_packet)
        
    def _gather(
        self,
        query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instruction: Optional[str] = None,
        additional_packet: Optional[List[ContextPacket]] = None
    ) -> List[ContextPacket]:
        packets = []

        if system_instruction:
            packets.append(ContextPacket(content=system_instruction, metadata={"type": "system_instruction"}))

        if self.memory_tool:
            try:
                state_results = self.memory_tool._search_memories(
                    query = "任务状态 OR 目标 OR 结论",
                    limit = 5,
                    threshold = 0.7
                )
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(content=state_results, metadata={"type": "task_state", "importance": "high"}))

                related_results = self.memory_tool._search_memories(
                    query = query,
                    limit = 5
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(content=related_results, metadata={"type": "related_memory"}))
            except Exception as e:
                print(f"记忆检索失败: {e}")

        if conversation_history:
            history = conversation_history[-5:]
            text = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
            packets.append(ContextPacket(content=text, metadata={"type": "conversation_history"}))

        packets.extend(additional_packet)

        return packets

    def _select(self, query: str, packets: List[ContextPacket]) -> List[ContextPacket]:
        scored_packets = List[Tuple[float, ContextPacket]] = []
        query_tokens = set(query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                packet.relevance = len(query_tokens & content_tokens) / len(query_tokens)
            else:
                packet.relevance = 0.0

            delta = max((datetime.now() - packet.timestamp).total_seconds(), 0)
            recency_score = math.exp(-delta / 3600)

            score = 0.7 * packet.relevance + 0.3 * recency_score
            scored_packets.append((score, packet))

        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "system_instruction"]
        remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
                    if p.metadata.get("type") != "system_instruction"]
        filtered_packets = [p for p in remaining if p.relevance >= self.config.min_relevance]

        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0
        for p in system_packets:
            if used_tokens + p.token <= available_tokens:
                selected.append(p)
                used_tokens += p.token
        for p in filtered_packets:
            if used_tokens + p.token > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token
        
        return selected

    def _structure(self, query: str, selected_packets: List[ContextPacket]) -> str:
        sections = []

        # [Role & Policies] - 系统指令
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "system_instruction"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)
        
        # [Task] - 当前任务
        sections.append(f"[Task]\n用户问题：{query}")
        
        # [State] - 任务状态
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题：\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)
        
        # [Evidence] - 事实证据
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事实与引用：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)
        
        # [Context] - 辅助材料（历史等）
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "conversation_history"]
        if p3_packets:
            context_section = "[Context]\n对话历史与背景：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)
        
        # [Output] - 输出约束
        output_section = """[Output]
                            请按以下格式回答：
                            1. 结论（简洁明确）
                            2. 依据（列出支撑证据及来源）
                            3. 风险与假设（如有）
                            4. 下一步行动建议（如适用）"""
        sections.append(output_section)
        
        return "\n\n".join(sections)

    def _compress(self, context: str) -> str:
        if not self.config.enable_compression:
            return context

        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        if current_tokens <= available_tokens:
            return context

        # 简单截断策略（保留前N个token）
        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens})，执行截断")
        
        # 按段落截断，保留结构
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)

def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        return len(text) // 4

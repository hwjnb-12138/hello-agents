from typing import List, Dict, Any
from datetime import datetime

try:
    from ..Chapter7.Tool import Tool, ToolParameter
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Chapter7'))
    from Chapter7.Tool import Tool, ToolParameter

from MemoryManager import MemoryManager
from Memory import MemoryConfig

class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        config: MemoryConfig = None,
        memory_types: List[str] = None,
    ):
        super().__init__(name="memory_tool", description="记忆工具 - 可以存储和检索对话历史、知识和经验")
        self.memory_config = config or MemoryConfig()
        self.memory_types = memory_types or ["working", "semantic"]

        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
        )

        self.current_session_id = None
        self.conversation_count = 0
    
    def run(self, parameters: Dict[str, Any]) -> str:
        required_parameters = [p.name for p in self.get_parameters() if p.required]
        if not all(param in parameters for param in required_parameters):
            return f"缺少必填参数: {', '.join(required_parameters)}"
        
        action = parameters.get("action")
        if action == "add":
            return self._add_memory(
                content=parameters.get("content"),
                memory_type=parameters.get("memory_type", "working"),
                importance=parameters.get("importance", 0.5),
            )
        elif action == "search":
            return self._search_memories(
                query=parameters.get("query"),
                limit=parameters.get("limit", 5),
                memory_types=parameters.get("memory_type"),
                threshold=parameters.get("threshold", 0.5),
            )
        elif action == "forget":
            return self._forget_memories(
                strategy=parameters.get("strategy", "importance_based"),
                threshold=parameters.get("threshold", 0.1),
                max_days=parameters.get("max_days", 30),
            )
        elif action == "consolidate":
            return self._consolidate(
                from_type=parameters.get("from_type", "working"),
                to_type=parameters.get("to_type", "semantic"),
                importance_threshold=parameters.get("importance_threshold", 0.7),
            )
        elif action == "update":
            return self._update_memory(
                memory_id=parameters.get("memory_id"),
                content=parameters.get("content"),
                importance=parameters.get("importance")
            )
        elif action == "stats":
            return self._get_stats()
        elif action == "remove":
            return self._remove_memory(
                memory_id=parameters.get("memory_id"),
            )
        elif action == "clear":
            return self._clear_memories()
        else:
            return f"未知操作: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "要执行的操作:"
                    "add(添加记忆), search(搜索记忆), forget(遗忘记忆), consolidate(整合记忆),"
                    "update(更新记忆), stats(获取记忆统计), remove(删除记忆), clear(清空所有记忆)"
                ),
                required=True,
            ),
            ToolParameter(name="content", type="string", description="记忆内容（add/update时可用）", required=False),
            ToolParameter(name="query", type="string", description="搜索查询（search时可用）", required=False),
            ToolParameter(name="memory_type", type="string", description="记忆类型：working, semantic（默认：working）", required=False, default="working"),
            ToolParameter(name="importance", type="number", description="重要性分数，0.0-1.0（add/update时可用）", required=False),
            ToolParameter(name="limit", type="integer", description="搜索结果数量限制（默认：5）", required=False, default=5),
            ToolParameter(name="memory_id", type="string", description="目标记忆ID（update/remove时必需）", required=False),
            ToolParameter(name="strategy", type="string", description="遗忘策略：importance_based/time_based/capacity_based（forget时可用）", required=False, default="importance_based"),
            ToolParameter(name="threshold", type="number", description="遗忘阈值（forget时可用，默认0.1）", required=False, default=0.1),
            ToolParameter(name="max_days", type="integer", description="最大保留天数（forget策略为time_based时可用）", required=False, default=30),
            ToolParameter(name="from_type", type="string", description="整合来源类型（consolidate时可用，默认working）", required=False, default="working"),
            ToolParameter(name="to_type", type="string", description="整合目标类型（consolidate时可用，默认semantic）", required=False, default="semantic"),
            ToolParameter(name="importance_threshold", type="number", description="整合重要性阈值（默认0.7）", required=False, default=0.7),
        ]
        

    def _add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
    ):
        if not content:
            return "添加记忆失败：记忆内容不能为空"
        metadata = {}
        try:
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            metadata["session_id"] = self.current_session_id
            metadata["timestamp"] = datetime.now().isoformat()
            
            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
            )

            return f"记忆已添加，ID：{memory_id}"
        except ValueError as e:
            return f"添加记忆失败：{str(e)}"
    
    def _search_memories(
        self,
        query: str,
        memory_types: List[str] = None,
        limit: int = 5,
        threshold: float = 0.5,
    ):
        try:
            memory_types = memory_types or self.memory_types
            if isinstance(memory_types, str):
                memory_types = [memory_types]

            memories = self.memory_manager.retrieve_memories(
                query=query,
                memory_types=memory_types,
                limit=limit,
                threshold=threshold,
            )

            if not memories:
                return "未找到相关记忆"
            
            results = []
            for i in range(len(memories)):
                content = memories[i].content[:100] + "..." if len(memories[i].content) > 100 else memories[i].content
                results.append(f"{i+1}. [{memories[i].memory_type}] {content} ({memories[i].importance:.2f})")
            
            return "\n".join(results)
        except Exception as e:
            return f"检索记忆失败：{str(e)}"
    
    def _get_stats(self):
        try:
            stats = self.memory_manager.get_memory_stats()

            stats_info = [
                f"📈 记忆系统统计",
                f"总记忆数: {stats['total_memories']}",
                f"启用的记忆类型: {', '.join(stats['enabled_types'])}",
                f"会话ID: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}"
            ]

            return "\n".join(stats_info)
        except Exception as e:
            return f"获取记忆统计失败：{str(e)}"
    
    def auto_record_conversation(self, user_input: str, agent_response: str):
        """自动记录对话

        这个方法可以被Agent调用来自动记录对话历史
        """
        self.conversation_count += 1
        # 记录用户输入
        self._add_memory(
            content=f"用户: {user_input}",
            memory_type="working",
            importance=0.6,
        )

        # 记录Agent响应
        self._add_memory(
            content=f"助手: {agent_response}",
            memory_type="working",
            importance=0.7,
        )

        # 如果是重要对话，记录为语义记忆
        if len(agent_response) > 100 or "重要" in user_input or "记住" in user_input:
            interaction_content = f"对话 - 用户: {user_input}\n助手: {agent_response}"
            self._add_memory(
                content=interaction_content,
                memory_type="semantic",
                importance=0.8,
            )
    
    def _update_memory(self, memory_id: str, content: str, importance: float = None):
            metadata = {}
            try:
                metadata["session_id"] = self.current_session_id
                metadata["timestamp"] = datetime.now().isoformat()

                res = self.memory_manager.update_memory(
                    memory_id=memory_id,
                    content=content,
                    importance=importance,
                    metadata=metadata,
                )
                return f"记忆已更新，ID：{memory_id}" if res else "记忆不存在"
            except ValueError as e:
                return f"更新记忆失败：{str(e)}"
    
    def _remove_memory(self, memory_id: str):
        try:
            res = self.memory_manager.remove_memory(memory_id)
            return f"记忆已删除，ID：{memory_id}" if res else "记忆不存在"
        except ValueError as e:
            return f"删除记忆失败：{str(e)}"

    def _clear_memories(self):
        try:
            self.memory_manager.clear_memories()
            return "所有记忆已清除"
        except Exception as e:
            return f"清除记忆失败：{str(e)}"
    
    def _forget_memories(self, strategy: str = "importance_based", threshold: float = 0.1, max_days: int = 30):
        try:
            forget_num = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_days=max_days,
            )
            return f"遗忘 {forget_num} 条记忆"
        except Exception as e:
            return f"遗忘记忆失败：{str(e)}"
    
    def _consolidate(self, from_type: str = "working", to_type: str = "semantic", importance_threshold: float = 0.7):
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
            )
            return f"🔄 已整合 {count} 条记忆为长期记忆（{from_type} → {to_type}，阈值={importance_threshold}）"
        except Exception as e:
            return f"❌ 整合记忆失败: {str(e)}"
    
    def clear_session(self):
        """清除当前会话"""
        self.current_session_id = None
        self.conversation_count = 0

        # 清理工作记忆
        wm = self.memory_manager.memory_types.get('working') if hasattr(self.memory_manager, 'memory_types') else None
        if wm:
            wm.clear()
    
    def forget_old_memories(self, max_days: int = 30):
        return self.memory_manager.forget_memories(
            strategy="age_based",
            max_days=max_days
        )
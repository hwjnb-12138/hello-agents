import logging
import uuid
from datetime import datetime
from Memory import MemoryConfig, MemoryItem
from WorkingMemory import WorkingMemory
from SemanticMemory import SemanticMemory
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_semantic: bool = True
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        self.memory_types = {}
        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)
        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)

        logger.info(f"MemoryManager初始化完成，用户：{self.user_id}，启用记忆类型：{self.memory_types.keys()}")

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        importance = self._calculate_importance(importance, content, metadata)

        memory_item = MemoryItem(
            id = str(uuid.uuid4()),
            content = content,
            memory_type = memory_type,
            user_id = self.user_id,
            timestamp = datetime.now(),
            importance = importance,
            metadata = metadata or {}
        )

        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            logger.debug(f"添加记忆到 {memory_type}，ID：{memory_id}")
            return memory_id
        else:
            return ValueError(f"不支持的记忆类型：{memory_type}")
        
    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[list[str]] = None,
        limit: int = 5,
        threshold: float = 0.1,
    ) -> List[MemoryItem]:
        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        results = []
        type_limit = max(1, limit // len(memory_types))
        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]
                try:
                    type_result = memory_instance.retrieve(
                        query = query,
                        limit = type_limit,
                        threshold = threshold,
                        user_id = self.user_id
                    )
                    results.extend(type_result)
                except Exception as e:
                    logger.error(f"检索 {memory_type} 记忆时出错：{e}")
                    continue
        
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:limit]
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        for memory_type, memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)
        
        logger.warning(f"未找到记忆ID为 {memory_id} 的记忆")
        return False

    def remove_memory(self, memory_id: str,) -> bool:
        for memory_type, memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)
        
        logger.warning(f"未找到记忆ID为 {memory_id} 的记忆")
        return False

    def _calculate_importance(self, importance: float, content: str, metadata: Optional[Dict[str, Any]]):
        if len(content) > 100:
            importance += 0.1
        
        keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in keywords):
            importance += 0.2
        
        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2

        
        return min(importance, 1.0)
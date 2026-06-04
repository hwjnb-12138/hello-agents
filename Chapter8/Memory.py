from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

class MemoryItem(BaseModel):
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any] = {}

class MemoryConfig(BaseModel):

    storage_path: str = "./memory_data"
    max_capacity: int = 1000
    importance_threshold: float = 0.5
    decay_factor: float = 0.95

    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

class BaseMemory(ABC):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem):
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    def _generate_id(self):
        import uuid
        return str(uuid.uuid4())
    
    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        importance = base_importance

        if len(content) > 100:
            importance += 0.1
        
        keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in keywords):
            importance += 0.2

        return min(importance, 1.0)
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
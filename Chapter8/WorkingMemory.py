from typing import Dict, Any, List
from datetime import datetime, timedelta
from typing import List
from Memory import BaseMemory, MemoryConfig, MemoryItem

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        self.memories: List[MemoryItem] = []

    def add(self, memory_item: MemoryItem):
        self._expire_old_memories()
        self.memories.append(memory_item)
        self.current_tokens += len(memory_item.content.split())
        self._enforce_capacity_limits()
        
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> List[MemoryItem]:
        self._expire_old_memories()
        if not self.memories:
            return []
        active_memories = [memory for memory in self.memories if not memory.metadata.get("forgotten", False)]

        filtered_memories = active_memories
        if user_id:
            filtered_memories = [memory for memory in active_memories if memory.user_id == user_id]

        if not filtered_memories:
            return []
        
        vector_scores = {}
        if SKLEARN_AVAILABLE:
            try:
                documents = [query] + [m.content for m in filtered_memories]
                print(f"documents: {documents}")
                vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
                tfidf_matrix = vectorizer.fit_transform(documents)
                query_vector = tfidf_matrix[0:1]
                doc_vectors = tfidf_matrix[1:]
                similarities = cosine_similarity(query_vector, doc_vectors).flatten()
                for i, memory in enumerate(filtered_memories):
                    vector_scores[memory.id] = similarities[i]
            except Exception:
                pass

        query_lower = query.lower()
        scored_memories = []
        for memory in filtered_memories:
            content_lower = memory.content.lower()
            # 获取向量分数（如果有）
            vector_score = vector_scores.get(memory.id, 0.0)
            
            # 关键词匹配分数
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                # 分词匹配
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8

            # 混合分数：向量检索 + 关键词匹配
            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score
            
            # 时间衰减
            decay_factor = self._calculate_time_decay(memory.timestamp)
            base_relevance *= decay_factor

            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))
            
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        for memory in self.memories:
            if memory.id == memory_id:
                if content:
                    old_tokens = len(memory.content.split())
                    new_tokens = len(content.split())
                    memory.content = content
                    self.current_tokens += new_tokens - old_tokens
                if importance is not None:
                    memory.importance = importance
                if metadata:
                    memory.metadata.update(metadata)
                return True
        return False
    
    def remove(self, memory_id: str) -> bool:
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                removed_memory = self.memories.pop(i)
                self.current_tokens -= len(removed_memory.content.split())
                return True
        return False

    def has_memory(self, memory_id: str) -> bool:
        return any(memory.id == memory_id for memory in self.memories)
    
    def clear(self):
        self.memories.clear()
        self.current_tokens = 0

    def get_stats(self):
        self._expire_old_memories()
        active_memories = self.memories

        return {
            "count": len(active_memories),  # 活跃记忆数量
            "forgotten_count": 0,  # 工作记忆中已遗忘的记忆会被直接删除
            "total_count": len(self.memories),  # 总记忆数量
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working"
        }
    
    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """获取最近的记忆"""
        sorted_memories = sorted(
            self.memories, 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        return sorted_memories[:limit]
    
    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        """获取重要记忆"""
        sorted_memories = sorted(
            self.memories,
            key=lambda x: x.importance,
            reverse=True
        )
        return sorted_memories[:limit]

    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆"""
        return self.memories.copy()
    
    def get_context_summary(self, max_length: int = 500) -> str:
        if not self.memories:
            return "暂无工作记忆"
        
        sorted_memories = sorted(
            self.memories,
            key=lambda x: (x.importance, x.timestamp),
            reverse=True
        )

        summary_parts = []
        current_length = 0
        for memory in sorted_memories:
            if current_length + len(memory.content) <= max_length:
                summary_parts.append(memory.content)
                current_length += len(memory.content)
            else:
                remain = max_length - current_length
                if remain > 50:
                    summary_parts.append(memory.content[:remain] + "...")
                break
        
        return "Working Memory Context:\n" + "\n".join(summary_parts)

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_days: int = 1) -> int:
        forgotten_count = 0
        current_time = datetime.now()

        removed = []
        cutoff_ttl = current_time - timedelta(minutes = self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                removed.append(memory.id)
        
        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    removed.append(memory.id)
        elif strategy == "time_based":
            cutoff_time = current_time - timedelta(days=max_days)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    removed.append(memory.id)
        elif strategy == "capacity_based":
            if len(self.memories) > self.max_capacity:
                sorted_memories = sorted(
                    self.memories,
                    key=lambda x: self._calculate_priority(x),
                    reverse=True
                )
                removed_memories = sorted_memories[self.max_capacity:]
                for memory in removed_memories:
                    removed.append(memory.id)
        
        for memory_id in removed:
            if self.remove(memory_id):
                forgotten_count += 1
        
        return forgotten_count

    def _calculate_priority(self, memory: MemoryItem) -> float:
        priority = memory.importance
        decay_factor = self._calculate_time_decay(memory.timestamp)
        priority *= decay_factor
        return priority
    
    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """计算时间衰减因子"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        
        # 指数衰减（工作记忆衰减更快）
        decay_factor = self.config.decay_factor ** (hours_passed / 6)  # 每6小时衰减
        return max(0.1, decay_factor)  # 最小保持10%的权重
    
    def _expire_old_memories(self):
        if not self.memories:
            return
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept: List[MemoryItem] = []
        removed_tokens = 0
        for memory in self.memories:
            if memory.timestamp >= cutoff_time:
                kept.append(memory)
            else:
                removed_tokens += len(memory.content.split())
        if len(kept) == len(self.memories):
            return
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_tokens)

    def _enforce_capacity_limits(self):
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()
        
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()


    def _remove_lowest_priority_memory(self):
        if not self.memories:
            return
        
        lowest_memory = min(self.memories, key=lambda m: self._calculate_priority(m))
        self.remove(lowest_memory.id)
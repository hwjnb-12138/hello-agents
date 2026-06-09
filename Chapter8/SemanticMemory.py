from typing import Dict, Any, List
from datetime import datetime, timedelta
from Memory import BaseMemory, MemoryConfig, MemoryItem

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SemanticMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        self.max_capacity = self.config.max_capacity
        self.memories: List[MemoryItem] = []

    def add(self, memory_item: MemoryItem):
        self.memories.append(memory_item)
        self._enforce_capacity_limits()
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, threshold: float = 0.1, user_id: str = None, **kwargs) -> List[MemoryItem]:
        if not self.memories:
            return []

        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]

        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        vector_scores = {}
        if SKLEARN_AVAILABLE:
            try:
                documents = [query] + [m.content for m in filtered_memories]
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
            vector_score = vector_scores.get(memory.id, 0.0)

            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8

            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score

            decay_factor = self._calculate_time_decay(memory.timestamp)
            base_relevance *= decay_factor

            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score >= threshold:
                scored_memories.append((final_score, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        for memory in self.memories:
            if memory.id == memory_id:
                if content:
                    memory.content = content
                if importance is not None:
                    memory.importance = importance
                if metadata:
                    memory.metadata.update(metadata)
                return True
        return False

    def remove(self, memory_id: str) -> bool:
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                self.memories.pop(i)
                return True
        return False

    def has_memory(self, memory_id: str) -> bool:
        return any(memory.id == memory_id for memory in self.memories)

    def clear(self):
        self.memories.clear()

    def get_stats(self):
        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]
        return {
            "count": len(active_memories),
            "forgotten_count": len(self.memories) - len(active_memories),
            "total_count": len(self.memories),
            "max_capacity": self.max_capacity,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "memory_type": "semantic"
        }

    def get_all(self) -> List[MemoryItem]:
        return self.memories.copy()

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_days: int = 30) -> int:
        forgotten_count = 0
        current_time = datetime.now()

        removed = []
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
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = self.config.decay_factor ** (hours_passed / 24)
        return max(0.3, decay_factor)

    def _enforce_capacity_limits(self):
        if len(self.memories) > self.max_capacity:
            sorted_memories = sorted(
                self.memories,
                key=lambda x: (x.importance, x.timestamp),
                reverse=True
            )
            self.memories = sorted_memories[:self.max_capacity]
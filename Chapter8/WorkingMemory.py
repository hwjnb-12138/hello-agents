from Memory import BaseMemory, MemoryConfig

class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

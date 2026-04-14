from typing import Optional, Dict, Literal, Any
from datetime import datetime
from pydantic import BsaeModel

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BsaeModel):
    role: MessageRole
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, role: MessageRole, content: str, **kwargs):
        super().__init__(
            role = role, 
            content = content, 
            timestamp = kwargs.get('timestamp', datetime.now()),
            metadata = kwargs.get('metadata', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"""{self.role}: {self.content}"""
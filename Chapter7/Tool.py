from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]):
        """运行工具"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数"""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling Schema格式"""
        parameters = self.get_parameters()

        properities = {}
        required = []

        for parameter in parameters:
            prop = {
                "type": parameter.type,
                "description": parameter.description
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if parameter.default is not None:
                prop["description"] = f"{parameter.description} (default: {parameter.default})"
            
            # 如果是数组类型，添加 items 字段
            if parameter.type == "array":
                prop["items"] = {"type": "string"}
            
            properities[parameter.name] = prop

            if parameter.required:
                required.append(parameter.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properities,
                    "required": required
                }
            }
        }



class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """注册工具"""
        if tool.name in self._tools:
            print(f"Tool with name {tool.name} already registered, will overwrite")
        self._tools[tool.name] = tool
        print(f"Tool: {tool.name} registered")

    def register_function(self, name: str, description: str, function: callable):
        """注册函数"""
        if name in self._functions:
            print(f"Function with name {name} already regitered, will overwrite")
        self._functions[name] = {
            "description": description,
            "function": function
        }
        print(f"Function: {name} registered")

    def get_tools_descriptions(self) -> str:
        """获取所有工具的描述"""
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")
        
        return "\n".join(descriptions) if descriptions else "No tools registered"
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def get_function(self, name: str) -> Optional[callable]:
        """获取函数"""
        function_info = self._functions.get(name)
        return function_info["function"] if function_info else None
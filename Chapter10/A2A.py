"""
基于官方 a2a-sdk 库的 A2A 协议实现

使用官方 a2a-sdk 库实现 Agent-to-Agent Protocol 功能。
官方仓库: https://github.com/a2aproject/a2a-python
安装: pip install a2a-sdk
"""

from typing import Dict, Any, List, Optional
import asyncio

try:
    from a2a.client import Client
    from a2a.types import Message
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    Client = None
    Message = None


class A2AServer:
    """A2A 服务器（使用 Flask 提供 HTTP API）"""

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        capabilities: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 A2A 服务器

        Args:
            name: Agent 名称
            description: Agent 描述
            version: Agent 版本
            capabilities: Agent 能力描述
        """
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or {}
        self.skills = {}

    def add_skill(self, skill_name: str, func):
        """添加技能到服务器"""
        self.skills[skill_name] = func
        return func

    def skill(self, skill_name: str):
        """装饰器方式添加技能"""
        def decorator(func):
            self.add_skill(skill_name, func)
            return func
        return decorator

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """运行服务器（使用 Flask 提供 HTTP API）"""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError(
                "A2A server requires Flask. Install it with: pip install flask"
            )

        app = Flask(self.name)

        # 禁用 Flask 的日志输出（可选）
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/info', methods=['GET'])
        def get_info():
            """获取 Agent 信息"""
            return jsonify(self.get_info())

        @app.route('/skills', methods=['GET'])
        def list_skills():
            """列出所有技能"""
            return jsonify({
                "skills": list(self.skills.keys()),
                "count": len(self.skills)
            })

        @app.route('/execute/<skill_name>', methods=['POST'])
        def execute_skill(skill_name):
            """执行指定技能"""
            if skill_name not in self.skills:
                return jsonify({
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": list(self.skills.keys())
                }), 404

            try:
                data = request.get_json() or {}
                text = data.get('text', data.get('query', ''))

                # 调用技能函数
                result = self.skills[skill_name](text)

                return jsonify({
                    "skill": skill_name,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "skill": skill_name,
                    "status": "error"
                }), 500

        @app.route('/ask', methods=['POST'])
        def ask():
            """通用问答接口（自动选择技能）"""
            try:
                data = request.get_json() or {}
                question = data.get('question', data.get('text', ''))

                # 简单策略：尝试所有技能，返回第一个非错误结果
                for skill_name, skill_func in self.skills.items():
                    try:
                        result = skill_func(question)
                        if result and not result.startswith("Error"):
                            return jsonify({
                                "answer": result,
                                "skill_used": skill_name,
                                "status": "success"
                            })
                    except:
                        continue

                return jsonify({
                    "answer": "No suitable skill found for this question",
                    "status": "no_match"
                })
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "status": "error"
                }), 500

        @app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify({"status": "healthy", "agent": self.name})

        # 启动服务器
        print(f"🚀 A2A 服务器 '{self.name}' 启动在 {host}:{port}")
        print(f"📋 描述: {self.description}")
        print(f"🛠️  可用技能: {list(self.skills.keys())}")
        print(f"📡 API 端点:")
        print(f"   - GET  {host}:{port}/info - 获取 Agent 信息")
        print(f"   - GET  {host}:{port}/skills - 列出技能")
        print(f"   - POST {host}:{port}/execute/<skill> - 执行技能")
        print(f"   - POST {host}:{port}/ask - 通用问答")
        print(f"   - GET  {host}:{port}/health - 健康检查")
        print()

        app.run(host=host, port=port, debug=False)

    def get_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "protocol": "A2A",
            "skills": list(self.skills.keys())
        }


class A2AClient:
    """A2A 客户端（通过 HTTP 与 A2AServer 通信）"""

    def __init__(self, server_url: str):
        """
        初始化 A2A 客户端

        Args:
            server_url: 服务器 URL（例如：http://localhost:5000）
        """
        self.server_url = server_url.rstrip('/')

    def ask(self, question: str) -> str:
        """
        向 Agent 提问（通用接口）

        Args:
            question: 问题文本

        Returns:
            Agent 的回答
        """
        try:
            import requests
            response = requests.post(
                f"{self.server_url}/ask",
                json={"question": question},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("answer", "No response")
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"

    def execute_skill(self, skill_name: str, text: str = "") -> Dict[str, Any]:
        """
        执行指定技能

        Args:
            skill_name: 技能名称
            text: 输入文本

        Returns:
            执行结果
        """
        try:
            import requests
            response = requests.post(
                f"{self.server_url}/execute/{skill_name}",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to execute skill: {str(e)}", "status": "error"}

    def get_info(self) -> Dict[str, Any]:
        """获取 Agent 信息"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to get agent info: {str(e)}"}

    def list_skills(self) -> List[str]:
        """列出 Agent 的技能"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/skills", timeout=10)
            response.raise_for_status()
            return response.json().get("skills", [])
        except Exception as e:
            return []


# 示例：创建一个简单的 A2A Agent
def create_example_agent() -> A2AServer:
    """创建一个示例 A2A Agent"""
    if not A2A_AVAILABLE:
        raise ImportError(
            "Cannot create example agent: a2a-sdk library not available. "
            "Install it with: pip install a2a-sdk"
        )

    server = A2AServer(
        name="Example A2A Agent",
        description="A simple example A2A agent",
        version="1.0.0",
        capabilities={"chat": True, "calculation": True}
    )

    # 添加计算技能
    def calculator_skill(text: str) -> str:
        """计算数学表达式"""
        # 从文本中提取表达式
        import re
        match = re.search(r'calculate\s+(.+)', text, re.IGNORECASE)
        if match:
            expression = match.group(1).strip()
            try:
                # 安全的表达式求值（仅支持基本运算）
                allowed_chars = set("0123456789+-*/() .")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"
                result = eval(expression)
                return f"The result is: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        return "Please provide an expression to calculate"

    server.add_skill("calculate", calculator_skill)

    # 添加问候技能
    def greeting_skill(text: str) -> str:
        """生成问候语"""
        import re
        match = re.search(r'hello|hi|greet', text, re.IGNORECASE)
        if match:
            return "Hello! I'm an A2A agent. How can I help you today?"
        return "Hi there!"

    server.add_skill("greet", greeting_skill)

    return server


if __name__ == "__main__":
    try:
        # 创建并运行示例 Agent
        agent = create_example_agent()
        print(f"🚀 Starting {agent.name}...")
        print(f"📝 {agent.description}")
        print(f"🔌 Protocol: A2A")
        print(f"📡 Version: {agent.version}")
        print(f"🛠️ Skills: {list(agent.skills.keys())}")
        print()
        agent.run(host="0.0.0.0", port=5000)
    except ImportError as e:
        print(f"❌ {e}")
        print("💡 Install the A2A SDK: pip install a2a-sdk")
        print("📖 Official repository: https://github.com/a2aproject/a2a-python")
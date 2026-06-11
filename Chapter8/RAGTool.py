from typing import List, Dict, Any

try:
    from ..Chapter7.Tool import Tool, ToolParameter
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Chapter7'))
    from Chapter7.Tool import Tool, ToolParameter

from Embedding import Embedding


class RAGTool(Tool):
    def __init__(self, embedding_model: str = "BAAI/bge-m3"):
        super().__init__(
            name="rag_tool",
            description="RAG工具 - 将文档存入向量数据库，并支持基于语义相似度的文档检索"
        )
        self.embedding = Embedding(model=embedding_model)

    def run(self, parameters: Dict[str, Any]) -> str:
        required_parameters = [p.name for p in self.get_parameters() if p.required]
        if not all(param in parameters for param in required_parameters):
            return f"缺少必填参数: {', '.join(required_parameters)}"

        action = parameters.get("action")
        if action == "add":
            return self._add_document(
                document=parameters.get("document"),
            )
        elif action == "search":
            return self._search_documents(
                query=parameters.get("query"),
                top_k=parameters.get("top_k", 3),
            )
        elif action == "clear":
            return self._clear_documents()
        elif action == "stats":
            return self._get_stats()
        else:
            return f"未知操作: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "要执行的操作: "
                    "add(添加文档到向量库), search(根据语义搜索文档), "
                    "clear(清空所有文档), stats(获取向量库统计信息)"
                ),
                required=True,
            ),
            ToolParameter(
                name="document",
                type="string",
                description="要添加的文档内容（add时必需）",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询语句（search时必需）",
                required=False,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回最相似文档的数量（默认：3）",
                required=False,
                default=3,
            ),
        ]

    def _add_document(self, document: str) -> str:
        if not document:
            return "添加文档失败：文档内容不能为空"

        try:
            self.embedding.get_document_embedding(document)
            doc_count = len(self.embedding.vectorstore.vectorstore)
            return f"文档已添加，当前向量库共有 {doc_count} 条文档"
        except Exception as e:
            return f"添加文档失败：{str(e)}"

    def _search_documents(self, query: str, top_k: int = 3) -> str:
        if not query:
            return "搜索失败：查询内容不能为空"

        try:
            results = self.embedding.search(query, top_k=top_k)

            if not results:
                return "未找到相关文档"

            output_lines = []
            for i, doc in enumerate(results):
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                output_lines.append(f"{i + 1}. {preview}")

            return "\n".join(output_lines)
        except Exception as e:
            return f"搜索文档失败：{str(e)}"

    def _clear_documents(self) -> str:
        try:
            count = len(self.embedding.vectorstore.vectorstore)
            self.embedding.vectorstore.vectorstore.clear()
            return f"已清空所有文档，共删除 {count} 条"
        except Exception as e:
            return f"清空文档失败：{str(e)}"

    def _get_stats(self) -> str:
        try:
            count = len(self.embedding.vectorstore.vectorstore)
            return f"向量库中共有 {count} 条文档"
        except Exception as e:
            return f"获取统计信息失败：{str(e)}"
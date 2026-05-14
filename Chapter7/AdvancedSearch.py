import os
from Tool import Tool, ToolParameter
from tavily import TavilyClient
from serpapi import SerpApiClient
from typing import List, Dict, Any

class AdvancedSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name = "advanced_search",
            description = "智能搜索工具，支持多个搜索源，如Tavily、SerpApi等，自动整合搜索结果"
        )
        self.search_sources = []
        self._set_search_sources()

    def _set_search_sources(self):
        if os.getenv("Tavily_API_KEY"):
            self.tavily_client = TavilyClient(api_key = os.getenv("Tavily_API_KEY"))
            self.search_sources.append("Tavily")
        
        if os.getenv("SERPAPI_API_KEY"):
            self.search_sources.append("SerpApi")
        
        if self.search_sources:
            print(f"已配置搜索源: {self.search_sources}")
        else:
            print("未配置任何搜索源")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name = "query",
                type = "string",
                description = "用户输入的需要查询的内容",
                required = True
            ),
            ToolParameter(
                name = "search_sources",
                type = "array",
                description = "用户指定的搜索源，默认使用所有配置的搜索源",
                required = False,
                default = self.search_sources
            )
        ]
    
    def run(self, parameters: Dict[str, Any]):
        query = parameters["query"]
        search_sources = parameters.get("search_sources", self.search_sources)
        
        print(f"开始进行智能搜索，查询内容: {query}")
        for source in search_sources:
            if source == "Tavily":
                response = self.tavily_client.search(query = query, include_answer = "advanced")
                if response.get("answer"):
                    return response["answer"]
            
                results = []
                for result in response.get("results", []):
                    results.append(f"- {result['title']}: {result['content']}")
                
                if not results:
                    return "抱歉，没有搜索到相关结果。"

                return "根据搜索，为您找到以下信息:\n" + "\n".join(results)
            elif source == "SerpApi":
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": os.getenv("SERPAPI_API_KEY"),
                    "gl": "cn",  # 国家代码
                    "hl": "zh-cn", # 语言代码
                }
            
                client = SerpApiClient(params)
                results = client.get_dict()

                # 智能解析:优先寻找最直接的答案
                if "answer_box_list" in results:
                    return "\n".join(results["answer_box_list"])
                if "answer_box" in results and "answer" in results["answer_box"]:
                    return results["answer_box"]["answer"]
                if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                    return results["knowledge_graph"]["description"]
                if "organic_results" in results and results["organic_results"]:
                    # 如果没有直接答案，则返回前三个有机结果的摘要
                    snippets = [
                        f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                        for i, res in enumerate(results["organic_results"][:3])
                    ]
                    return "\n\n".join(snippets)
                
                return f"对不起，没有找到关于 '{query}' 的信息。"
        
        return "抱歉，没有搜索到相关结果。"
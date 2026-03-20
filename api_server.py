import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 模块 1：系统代理配置
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 实例化 FastAPI 应用
app = FastAPI(title="FDCA 学术智能体核心 API", description="前后端分离的后端大脑")

# ==========================================
# 模块 2：初始化引擎 (服务器启动时只加载一次，极大提升速度)
# ==========================================
llm = ChatOpenAI(
    api_key="your_api_key_here",
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.5
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vector_db = None

# ==========================================
# 模块 3：定义工具库
# ==========================================
@tool
def search_local_papers(query: str) -> str:
    """当用户询问特定的电化学机理、Ni-O-Co催化剂、HMF氧化等专业文献时调用。"""
    if not vector_db:
        return "本地数据库未初始化"
    docs = vector_db.similarity_search(query, k=5)
    res = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get('page', 0) + 1
        source = os.path.basename(doc.metadata.get('source', '未知'))
        res += f"【片段 {i+1}】(来源: {source}, 第 {page} 页)\n{doc.page_content}\n\n"
    return res if res else "未找到相关本地文献。"

@tool
def search_internet(query: str) -> str:
    """当本地文献不足或询问最新进展时调用。自动将查询转化为英文学术关键词进行检索。"""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query[:50])
    except Exception as e:
        return f"网络搜索暂时不可用(原因：{str(e)}),请导师基于现有知识和本地文献回答。"
tools = [search_local_papers, search_internet]

# ==========================================
# 模块 4：定义 API 数据接口格式 (Schema)
# ==========================================
class ChatRequest(BaseModel):
    query: str
    mode: str = "文献综述模式"

# ==========================================
# 模块 5：核心 API 路由 (接收请求 -> 处理 -> 返回)
# ==========================================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if request.mode == "文献综述模式":
        sys_msg = "你是一个拥有全网搜索能力的电化学导师。优先使用 search_local_papers 工具，不足时果断使用 search_internet。"
    elif request.mode =="疯狂科学家":
        sys_msg = "你是一个极其严苛的疯狂科学家。提供降维打击的创新点和极其硬核的表征策略（如 CV, LSV, 原位 FTIR）。"
    elif request.mode =="催化剂参数制表机":
        sys_msg = """你是一个极度严谨的电化学数据提取专家。
        请优先调用 search_local_papers 工具，或者 search_internet 工具，搜索文献中提及的各种催化剂（尤其是 Ni-O-Co 等双金属/过渡金属催化剂）在 HMF 电催化氧化中的核心参数。
        
        【🔥 强制输出格式】：
        你必须且只能输出一个 Markdown 格式的表格，方便前端直接渲染。
        表格必须包含以下列，请严格遵守：
        | 催化剂名称 | HMF转化率(%) | FDCA产率/法拉第效率(%) | 电解液条件 | 核心优势简述 | 来源文献 |
        
        如果某个具体数据在文献中找不到，请填 "-"。
        除了这个表格和表格下方一句简短的总结外，绝对不要输出任何多余的废话！"""
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        response = agent_executor.invoke({"input": request.query})
        return {"status": "success", "answer": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"智能体运行失败: {str(e)}")
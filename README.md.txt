# 🔬 智元学术 (Scholar Agent V4.0)

基于大语言模型与多工具路由 (Tool-Calling) 构建的电化学专属 AI 学术智能体。采用 FastAPI + Streamlit 前后端分离微服务架构，专为 FDCA、HMF 阳极氧化等电化学前沿研究打造的自动化文献分析与数据提取引擎。

## ✨ 核心特性

- **🧠 多重宇宙人格**：无缝切换“严谨文献综述”与“疯狂科学家”模式，提供从理论梳理到硬核电化学表征（CV, LSV, 原位 FTIR）的降维打击指导。
- **⚙️ 结构化制表机**：自动从海量长篇文献中提取催化剂性能参数（HMF转化率、法拉第效率等），一键生成高颜值 Markdown 对比表格，加速论文写作。
- **🛠️ 自主外挂调用 (Agentic RAG)**：AI 拥有自主判断力。针对特定机理优先调用 FAISS 本地向量库检索私有 PDF 文献；当本地信息不足或查询最新进展时，自动将中文转化为学术英文关键词，调用 DuckDuckGo 侦察兵进行全网检索。
- **🚀 工业级微服务架构**：
  - **后端 (FastAPI)**：静默承载 LLM 推理、大容量向量搜索与 Agent 调度，保障高并发下的极速响应。
  - **前端 (Streamlit)**：轻量级 UI 渲染，只负责展示与 API 通信。

## 📦 技术栈

- **核心框架**: LangChain, FastAPI, Streamlit
- **大模型引擎**: GPT-4o-mini (通过 OpenRouter 接入)
- **向量数据库**: FAISS, HuggingFaceEmbeddings (`all-MiniLM-L6-v2`)
- **网络侦察**: DuckDuckGo Search API

## 🚀 快速启动

### 1. 环境准备
克隆本项目后，创建并激活虚拟环境：
```bash
python -m venv rag_env
# Windows 激活
.\rag_env\Scripts\activate
# 安装依赖
pip install fastapi uvicorn pydantic streamlit langchain langchain-openai langchain-community faiss-cpu sentence-transformers duckduckgo-search
2. 启动核心大脑 (后端 API)
在终端运行 FastAPI 服务器：

Bash
uvicorn api_server:app --reload
API 文档将运行在 http://127.0.0.1:8000/docs

3. 启动交互界面 (前端 Web)
开启一个新的终端标签页，激活虚拟环境后运行：

Bash
streamlit run web_app.py

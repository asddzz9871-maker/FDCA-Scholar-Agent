
#设置国内镜像源

import os
import glob

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from multiprocessing import context

DB_PATH = "faiss_index"
LEDGER_FILE = "processed_papers.txt"

llm = ChatOpenAI(
    api_key="your-api",
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.1
)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def get_processed_files():
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE,"r",encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()
    
def mark_as_processed(filepath):
    with open(LEDGER_FILE,"a",encoding="utf-8")as f:
        f.write(filepath+"\n")

def process_new_papers(directory_path):
    pdf_files = glob.glob(f"{directory_path}/*.pdf")
    processed_files = get_processed_files()
    new_files = [f for f in pdf_files if f not in processed_files]
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, #每个数据块最多包含500个字符
        chunk_overlap=50, #相邻两个数据块之间保留50个字符的重叠
        separators=["\n\n","\n","。",".",",","，"] #告诉机器优先按段落切，再按句子切
    )

    for file in new_files:
        print(f"正在处理{os.path.basename(file)}")

    #模块1：数据加载
    #初始化加载器，将PDF里的文字按页提取出来
        loader =PyPDFLoader(file)
    #load()方法执行实际的读取动作，返回一个列表，列表里每一项是一页的内容
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        mark_as_processed(file)

    print(f"[切分完成] 共新获得了{len(all_chunks)}个数据块。")
    return all_chunks

#提取metadata,在pypfdloader里加载时记下了页码
def format_academic_docs(docs):
    formatted_text = ""
    for i,doc in enumerate(docs):
        #从字典里提取page，如果没有默认为0，真实的页码要+1
        page_num = doc.metadata.get('page',0)+1
        source_file = os.path.basename(doc.metadata.get('source','未知文献'))
        formatted_text += f"【文献片段{i+1}】(来源:{source_file},第{page_num}页)\n{doc.page_content}\n\n"
    return formatted_text

#主程序执行区
if __name__ == "__main__":
    print("\n"+"="*50)
    print("---启动RAG检索引擎---")
    print("="*50)

    paper_dir = "fdca_papers"

    new_chunks = process_new_papers(paper_dir)

    if os.path.exists(DB_PATH):
        print("加载本地安全文件")
        vector_db = FAISS.load_local(DB_PATH,embeddings_model,allow_dangerous_deserialization=True)
        if new_chunks:
            print("将新向量追加到历史数据库中")
            vector_db.add_documents(new_chunks)
            vector_db.save_local(DB_PATH)
            print("数据库更新完毕")
    else:
        if new_chunks:
            print("未检测到历史数据库，正在从零构建全新数据库")
            vector_db = FAISS.from_documents(documents=new_chunks,embedding=embeddings_model)
            vector_db.save_local(DB_PATH)
            print("全新数据库建立")
        else:
            print("没有检测到历史数据库")
            exit()

   

print("\n"+"="*50)
print("输入q或quit退出")
print("="*50)


#编写提示词
academic_template = """你是一个顶尖的电化学与催化材料领域学术导师。请阅读以下文献片段，并结合你的专业知识回答问题。

【核心指令】：
1. 优先使用文献片段中的具体数据和事实来支撑你的论点，并严格在句末标注引用（如：“...提高了法拉第效率 [文献A.pdf, 第5页]”）。
2. 深度推理许可：如果提供的文献片段不能完全回答问题，或者问题需要更高维度的总结与机理推演，请**大胆调用你自身的电化学、物理化学底座知识进行深入分析**。
3. 必须明确区分信息来源：对于来自文献的内容，必须加方括号引用；对于你基于理论延伸的推理分析，请使用“基于电化学理论推测...”或“从更宏观的机理来看...”作为引导语。
4. 回答要具有启发性，像导师一样引导学生思考。

【检索到的文献原文】：
{context}

【导师的提问】：{question}

【你的学术回答】："""

prompt_template = PromptTemplate.from_template(academic_template)

while True:
    user_query = input("\n 输入要查询的问题：")
    if user_query.lower() in ['q','quit','exit']:
        print("已退出")
        break
    if not user_query.strip():
        continue
    print(f"检索最相关的5个片段")
    matched_docs = vector_db.similarity_search(user_query, k=3)
    context_str = format_academic_docs(matched_docs)

    final_prompt = prompt_template.format(context=context_str,question=user_query)

    response = llm.invoke(final_prompt)

    print("\n"+"-"*60)
    print("[AI导师的最终回答]")
    print("-"*60)
    print(response.content)
    print("-"*60)

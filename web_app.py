
import streamlit as st
import requests


# ==========================================
# 模块 1：系统配置与初始化
# ==========================================

st.set_page_config(page_title="FDCA 学术智库", page_icon="🔬", layout="wide")


with st.sidebar:
    st.title("导师控制台")
    st.markdown("---")
    selected_mode = st.radio(
        "选择导师人格与专区：",
        ("文献综述模式", "疯狂科学家 (创新与实验)", "催化剂参数制表机"), # <--- 新增了制表机频道
        help="切换频道会自动隔离聊天记录"
    )
    st.markdown("---")
    st.caption("提示：前后端分离架构已启用，前端只负责显示，思考过程在后端完成。")
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = selected_mode
    elif selected_mode != st.session_state.current_mode:
        st.session_state.current_mode = selected_mode
        st.rerun()

session_key = f"messages_{st.session_state.current_mode}"
if session_key not in st.session_state:
    st.session_state[session_key]=[]
st.title(f"{st.session_state.current_mode}")


# ==========================================
# 模块 3：Web 聊天渲染循环
# ==========================================
# 把历史聊天记录一条条画在网页上
for msg in st.session_state[session_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 网页底部的聊天输入框
if prompt := st.chat_input("向我提问，例如：Ni-O-Co 桥接结构如何破除线性比例关系？"):
    
    # 1. 把用户的问题显示在界面上
    st.session_state[session_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 开始思考并展示加载动画
    with st.chat_message("assistant"):
        with st.spinner("导师在后端服务器进行深度思考与全面检索"):
            try:
            #前端向后端发请求
                api_url = "http://127.0.0.1:8000/api/chat"
#把用户的问题和当前选择的模式打包为后端需要的格式
                payload = {
                    "query":prompt,
                    "mode":st.session_state.current_mode
                }
                #发送post请求给Fastapi后端
                response = requests.post(api_url,json=payload)
                #如果后端成功返回数据（状态码200）
                if response.status_code ==200:
                    answer = response.json().get("answer","未获取到答案")
                    st.markdown(answer)
                    st.session_state[session_key].append({"role":"assistant","content":answer})
                else:
                    st.error(f"后端服务器报错:{response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"智能体运行出错：确认是否运行`uvicorn api_server:app --reload`")
        
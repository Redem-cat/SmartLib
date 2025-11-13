import streamlit as st
from pathlib import Path
from ragSystem import RagSystem

# =========================
# 🔹 页面样式定义
# =========================
st.set_page_config(page_title="智能图书检索系统", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');

.main-header {
    font-size: 3rem;
    background: linear-gradient(135deg, #8B4513, #CD853F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    font-family: 'Noto Serif SC', serif;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 4px solid #2196f3;
    margin-left: 2rem;
    padding: 0.8rem;
    border-radius: 0.8rem;
}

.assistant-message {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 4px solid #8B4513;
    margin-right: 2rem;
    padding: 0.8rem;
    border-radius: 0.8rem;
}

.source-info {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    padding: 0.8rem;
    border-radius: 0.8rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    border: 1px solid #ce93d8;
}

.status-success { color: #2e7d32; font-weight: bold; }
.status-error { color: #d32f2f; font-weight: bold; }
.status-warning { color: #f57c00; font-weight: bold; }

.example-button {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border: 1px solid #ffb74d;
    border-radius: 0.5rem;
    padding: 0.5rem;
    margin: 0.2rem 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.example-button:hover {
    background: linear-gradient(135deg, #ffe0b2, #ffcc80);
    transform: translateX(5px);
}

.metric-card {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    padding: 1rem;
    border-radius: 0.8rem;
    text-align: center;
    margin: 0.5rem 0;
    border: 1px solid #81c784;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 🔹 工具函数
# =========================
def getEnvInfo(key):
    """从 .env 文件读取变量"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.startswith(key + "="):
                    return line.split("=")[1].strip()
    return None


def display_chat_message(role, content, sources=None):
    """显示用户和助手消息"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>🧑 您:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>📖 智能助手:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

    if sources:
        with st.expander(f"📄 参考文档片段 ({len(sources)}个)", expanded=False):
            for i, source in enumerate(sources, 1):
                similarity_color = "#4caf50" if source['similarity'] > 0.5 else "#ff9800"
                st.markdown(f"""
                <div class="source-info">
                    <strong>📄 片段 {i}: {source['source']}</strong>
                    <span style="background:{similarity_color};color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;">
                        相似度: {source['similarity']:.3f}
                    </span>
                    <br><em>📝 内容预览:</em><br>{source.get('content_preview', source.get('content', '')[:120] + '...')}
                </div>
                """, unsafe_allow_html=True)


# =========================
# 🔹 初始化系统状态
# =========================
def init_session_state():
    if 'system_init' not in st.session_state:
        st.session_state.system_init = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = getEnvInfo('DEEPSEEK_API_KEY')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_top_k' not in st.session_state:
        st.session_state.search_top_k = 10
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.5

    if not st.session_state.system_init:
        with st.spinner("🔧 智能图书检索系统初始化中..."):
            rag = RagSystem(st.session_state.api_key)
            rag.initialize()
            st.session_state.rag_system = rag
            st.session_state.system_init = True


# =========================
# 🔹 主界面
# =========================
def main():
    st.markdown('<h1 class="main-header">📚 智能图书检索系统</h1>', unsafe_allow_html=True)

    init_session_state()

    # ========== Sidebar ==========
    with st.sidebar:
        st.header("⚙️ 系统配置")
        api_key_input = st.text_input("DeepSeek API 密钥", type="password",
                                      value=st.session_state.api_key or "",
                                      help="从 .env 读取或手动输入")
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input

        if st.button("💾 保存配置"):
            with open(".env", "w", encoding="utf-8") as f:
                f.write(f"DEEPSEEK_API_KEY={st.session_state.api_key}")
            st.success("✅ API 密钥已保存！")

        st.divider()
        st.header("📊 系统状态")
        if st.session_state.system_init:
            st.markdown('<span class="status-success">✅ 系统已就绪</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">⚠️ 系统未初始化</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <strong>📄 文档数量:</strong> {len(st.session_state.rag_system.documents) if st.session_state.system_init else 0}
        </div>
        <div class="metric-card">
            <strong>🧩 文档分块数:</strong> {len(st.session_state.rag_system.doc_chunks) if st.session_state.system_init else 0}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔁 重新初始化系统"):
            st.session_state.system_init = False
            st.rerun()

        st.divider()
        st.header("🔧 搜索参数设置")
        st.session_state.search_top_k = st.slider("最大返回文档数", 3, 20, st.session_state.search_top_k)
        st.session_state.similarity_threshold = st.slider("相似度阈值", 0.1, 1.0, st.session_state.similarity_threshold)


    # ========== 主体内容 ==========
    st.header("💬 智能对话助手")

    # 快捷问题
    st.subheader("⚡ 快捷提问")
    cols = st.columns(4)
    questions = {
        "人物关系": "红楼梦中主要人物之间的关系是怎样的？",
        "情节梗概": "请简述红楼梦第一回的主要情节。",
        "文学手法": "红楼梦中有哪些主要的文学手法？",
        "象征意义": "红楼梦中‘金玉良缘’的象征意义是什么？"
    }

    for i, (label, question) in enumerate(questions.items()):
        if cols[i % 4].button(f"💡 {label}", use_container_width=True):
            st.session_state.user_input = question

    # 对话输入
    user_input = st.text_input("请输入您的问题：", value=st.session_state.get("user_input", ""))
    col_send, col_clear = st.columns([1, 1])
    with col_send:
        send_clicked = st.button("🚀 发送", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("🧹 清空", use_container_width=True)

    if clear_clicked:
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.success("✅ 对话已清空")
        st.stop()

    if send_clicked and user_input.strip():
        if not st.session_state.system_init:
            st.error("⚠️ 系统尚未初始化，请检查配置。")
        else:
            with st.spinner("🤔 正在检索与生成回答..."):
                rag = st.session_state.rag_system
                result = rag.ask(user_input)

                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", result['answer'], result['source']))
                st.session_state.user_input = ""

    # 显示聊天历史
    for msg in st.session_state.chat_history:
        if len(msg) == 2:
            display_chat_message(msg[0], msg[1])
        else:
            display_chat_message(msg[0], msg[1], msg[2])


if __name__ == "__main__":
    main()

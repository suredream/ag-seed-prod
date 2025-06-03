import streamlit as st
import requests
from datetime import datetime
import json
import os

# 页面配置
st.set_page_config(
    page_title="AI助手多页应用",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
def initialize_session_state():
    """初始化会话状态变量"""
    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = ""
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "anthropic/claude-3-sonnet"
    
    # 为每个页面初始化聊天历史
    pages = ["通用助手", "代码助手", "写作助手", "分析助手"]
    for page in pages:
        if f"chat_history_{page}" not in st.session_state:
            st.session_state[f"chat_history_{page}"] = []

def setup_openrouter_client():
    """检查OpenRouter API密钥是否设置"""
    if st.session_state.openrouter_api_key:
        return True
    return False

def get_ai_response(messages, context=""):
    """调用OpenRouter API获取回复"""
    try:
        # 如果有上下文，添加到系统消息中
        system_message = f"你是一个有用的AI助手。{context}"
        
        headers = {
            "Authorization": f"Bearer {st.session_state.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",  # Streamlit默认端口
            "X-Title": "Streamlit多页聊天应用"
        }
        
        data = {
            "model": st.session_state.selected_model,
            "messages": [
                {"role": "system", "content": system_message}
            ] + messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API错误: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"错误: {str(e)}"

def create_chatbox(page_name, context=""):
    """创建聊天框组件"""
    st.subheader(f"💬 {page_name}")
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state[f"chat_history_{page_name}"]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # 用户输入
    user_input = st.chat_input(f"在{page_name}中输入您的问题...")
    
    if user_input:
        # 添加用户消息到历史
        st.session_state[f"chat_history_{page_name}"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 显示用户消息
        st.chat_message("user").write(user_input)
        
        # 获取AI回复
        if setup_openrouter_client():
            with st.spinner("正在思考..."):
                # 准备消息历史（只保留最近10条消息以控制token使用）
                recent_messages = []
                for msg in st.session_state[f"chat_history_{page_name}"][-10:]:
                    recent_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                ai_response = get_ai_response(recent_messages[:-1], context)
                
                # 添加AI回复到历史
                st.session_state[f"chat_history_{page_name}"].append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # 显示AI回复
                st.chat_message("assistant").write(ai_response)
        else:
            st.error("请先在侧边栏设置OpenRouter API密钥")
    
    # 清除聊天历史按钮
    if st.button(f"🗑️ 清除{page_name}聊天记录", key=f"clear_{page_name}"):
        st.session_state[f"chat_history_{page_name}"] = []
        st.rerun()

def sidebar_config():
    """侧边栏配置"""
    with st.sidebar:
        st.title("🤖 AI助手配置")
        
        # API密钥设置
        st.subheader("API设置")
        api_key = st.text_input(
            "OpenRouter API密钥",
            type="password",
            value=st.session_state.openrouter_api_key,
            help="请输入您的OpenRouter API密钥"
        )
        
        if api_key != st.session_state.openrouter_api_key:
            st.session_state.openrouter_api_key = api_key
        
        # 模型选择
        models = {
            "Claude 3 Sonnet": "anthropic/claude-3-sonnet",
            "Claude 3 Haiku": "anthropic/claude-3-haiku",
            "GPT-4 Turbo": "openai/gpt-4-turbo",
            "GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
            "Gemini Pro": "google/gemini-pro",
            "Llama 2 70B": "meta-llama/llama-2-70b-chat",
            "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct"
        }
        
        selected_model_name = st.selectbox(
            "选择模型",
            list(models.keys()),
            index=0,
            help="选择要使用的AI模型"
        )
        st.session_state.selected_model = models[selected_model_name]
        
        # 状态显示
        if st.session_state.openrouter_api_key:
            st.success("✅ API密钥已设置")
            st.info(f"🤖 当前模型: {selected_model_name}")
        else:
            st.warning("⚠️ 请设置API密钥")
        
        st.divider()
        
        # 页面导航
        st.subheader("📄 页面导航")
        page_names = ["通用助手", "代码助手", "写作助手", "分析助手"]
        selected_page = st.selectbox("选择页面", page_names, key="page_selector")
        
        st.divider()
        
        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            **如何使用：**
            1. 在API设置中输入您的OpenRouter API密钥
            2. 选择要使用的AI模型
            3. 选择不同的助手页面
            4. 在聊天框中输入问题
            5. 每个页面都有独立的聊天历史
            
            **页面说明：**
            - **通用助手**: 回答各类通用问题
            - **代码助手**: 专注编程和技术问题
            - **写作助手**: 协助写作和文案创作
            - **分析助手**: 数据分析和逻辑推理
            
            **获取API密钥：**
            1. 访问 https://openrouter.ai
            2. 注册账户并获取API密钥
            3. 在上方输入框中粘贴密钥
            """)
        
        return selected_page

def main():
    """主函数"""
    initialize_session_state()
    
    # 侧边栏配置
    selected_page = sidebar_config()
    
    # 主标题
    st.title("🤖 AI助手多页应用")
    st.markdown("---")
    
    # 根据选择的页面显示相应的聊天界面
    if selected_page == "通用助手":
        st.markdown("## 🌟 通用AI助手")
        st.markdown("我可以帮助您解答各种问题，从日常咨询到知识问答。")
        create_chatbox(
            "通用助手",
            "你是一个通用AI助手，能够回答各种类型的问题，包括但不限于：常识问答、生活建议、学习指导等。请用友好、专业的语气回复。"
        )
    
    elif selected_page == "代码助手":
        st.markdown("## 💻 代码AI助手")
        st.markdown("我专门帮助您解决编程问题，包括代码编写、调试、优化等。")
        create_chatbox(
            "代码助手",
            "你是一个专业的编程助手，精通多种编程语言和开发技术。请提供准确的代码示例，并解释代码的工作原理。注重代码的可读性和最佳实践。"
        )
    
    elif selected_page == "写作助手":
        st.markdown("## ✍️ 写作AI助手")
        st.markdown("我可以协助您进行各种写作任务，包括文章创作、文案优化、语言润色等。")
        create_chatbox(
            "写作助手",
            "你是一个专业的写作助手，擅长各种文体的写作，包括技术文档、创意写作、商业文案等。请注重语言的准确性、流畅性和表达效果。"
        )
    
    elif selected_page == "分析助手":
        st.markdown("## 📊 分析AI助手")
        st.markdown("我专注于数据分析、逻辑推理和深度思考，帮助您解决复杂问题。")
        create_chatbox(
            "分析助手",
            "你是一个专业的分析助手，擅长数据分析、逻辑推理、问题解决和深度思考。请提供结构化的分析，包含清晰的逻辑链条和实用的建议。"
        )
    
    # 页面底部信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前页面", selected_page)
    with col2:
        total_messages = sum(len(st.session_state[f"chat_history_{page}"]) 
                           for page in ["通用助手", "代码助手", "写作助手", "分析助手"])
        st.metric("总消息数", total_messages)
    with col3:
        current_messages = len(st.session_state[f"chat_history_{selected_page}"])
        st.metric("当前页消息数", current_messages)

if __name__ == "__main__":
    main()
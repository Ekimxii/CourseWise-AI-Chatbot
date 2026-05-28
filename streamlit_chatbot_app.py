"""
CourseWise RAG Chatbot - Streamlit Application
Production-ready chatbot interface matching CourseWise branding
"""

# ============================================================================
# ORIGINAL VERSION (commented out - preserved for reference)
# ============================================================================
# import streamlit as st
# from coursewise_chatbot import initialize_chatbot
#
# # Page config
# st.set_page_config(
#     page_title="CourseWise Assistant",
#     page_icon="",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
#
# # Custom CSS
# st.markdown("""
# <style>
#     .main {
#         background: linear-gradient(180deg, #f5faf8 0%, #ffffff 100%);
#     }
#     .stApp {
#         background: #f5faf8;
#     }
#     .chat-header {
#         text-align: center;
#         padding: 30px 20px;
#         background: white;
#         border-radius: 12px 12px 0 0;
#         margin-bottom: 0;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#     }
#     .chat-header h1 {
#         color: #2d5a3d;
#         font-size: 32px;
#         font-weight: 600;
#         margin: 0;
#         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#     }
#     .chat-header p {
#         color: #666;
#         font-size: 15px;
#         margin: 8px 0 0 0;
#     }
#     .message-user {
#         background: #e8f5e9;
#         padding: 14px 18px;
#         border-radius: 20px 20px 4px 20px;
#         margin: 12px 0 12px 25%;
#         text-align: left;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#     }
#     .message-bot {
#         background: #ffffff;
#         padding: 14px 18px;
#         border-radius: 20px 20px 20px 4px;
#         margin: 12px 25% 12px 0;
#         border: 1px solid #e0e0e0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#     }
#     .chat-container {
#         background: white;
#         border-radius: 0 0 12px 12px;
#         padding: 20px;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         max-height: 600px;
#         overflow-y: auto;
#     }
# </style>
# """, unsafe_allow_html=True)
#
# # Initialize session state
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'chatbot' not in st.session_state:
#     try:
#         with st.spinner("Initializing chatbot..."):
#             st.session_state.chatbot = initialize_chatbot()
#             st.session_state.chatbot_ready = True
#     except Exception as e:
#         st.error(f"Failed to initialize chatbot: {str(e)}")
#         st.session_state.chatbot_ready = False
#         st.session_state.chatbot_error = str(e)
#         st.stop()
#
# # Header
# st.markdown("""
# <div class="chat-header">
#     <h1>CourseWise Assistant</h1>
#     <p>AI-powered transfer & articulation support</p>
# </div>
# """, unsafe_allow_html=True)
#
# # Chat container
# st.markdown('<div class="chat-container">', unsafe_allow_html=True)
#
# # Display messages
# for message in st.session_state.messages:
#     if message['role'] == 'user':
#         st.markdown(f'<div class="message-user"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<div class="message-bot"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
#
# st.markdown('</div>', unsafe_allow_html=True)
#
# # Quick suggestions
# st.markdown("### Quick Questions")
# col1, col2, col3 = st.columns(3)
# suggestions = [
#     "How do I use ATAIN?",
#     "What is credit articulation?",
#     "How does transfer pathway work?"
# ]
# for i, col in enumerate([col1, col2, col3]):
#     with col:
#         if st.button(suggestions[i], key=f"suggestion_{i}", use_container_width=True):
#             if 'user_input' not in st.session_state:
#                 st.session_state.user_input = ""
#             st.session_state.user_input = suggestions[i]
#             st.rerun()
#
# # User input
# user_input = st.text_input(
#     "Ask about transfer pathways, credit articulation, or platform help...",
#     key="user_input",
#     label_visibility="collapsed",
#     placeholder="Type your question here..."
# )
#
# if user_input and user_input.strip():
#     # Check if chatbot is ready
#     if not st.session_state.get('chatbot_ready', False):
#         st.error("Chatbot is not ready. Please refresh the page.")
#         st.stop()
#
#     # Add user message
#     st.session_state.messages.append({'role': 'user', 'content': user_input})
#
#     # Get bot response
#     try:
#         with st.spinner("Thinking..."):
#             result = st.session_state.chatbot.chat(user_input)
#             bot_response = result['answer']
#
#         # Add bot message
#         st.session_state.messages.append({'role': 'assistant', 'content': bot_response})
#     except Exception as e:
#         error_msg = f"I encountered an error: {str(e)}. Please try again."
#         st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
#         st.error(f"Error: {str(e)}")
#
#     # Clear input and rerun
#     if 'user_input' in st.session_state:
#         del st.session_state.user_input
#     st.rerun()

# ============================================================================
# V2 — REDESIGNED UI (CourseWise brand-aligned, presentation-ready)
# ============================================================================

import streamlit as st
import time
from datetime import datetime
from coursewise_chatbot import initialize_chatbot

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CourseWise Assistant",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Global styles — CourseWise brand
# ----------------------------------------------------------------------------
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap" rel="stylesheet">

<style>
    :root {
        --cw-bg: #f0f7f4;
        --cw-bg-alt: #e8f5f0;
        --cw-text: #1a3a2e;
        --cw-muted: #6b9e85;
        --cw-accent: #2db87a;
        --cw-accent-dark: #239a63;
        --cw-border: #d4ede2;
        --cw-card: #ffffff;
        --cw-error-bg: #fdecea;
        --cw-error-text: #8a2a20;
    }

    /* Force light theme + subtle minimal background */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stMain"],
    section.main, .main {
        color: var(--cw-text) !important;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 80% 60% at 50% 0%, #e8f5f0 0%, rgba(232,245,240,0) 70%),
            radial-gradient(ellipse 60% 50% at 100% 100%, #e3f2ea 0%, rgba(227,242,234,0) 65%),
            linear-gradient(180deg, #f2f9f5 0%, #edf6f1 100%) !important;
        background-attachment: fixed !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, header[data-testid="stHeader"], footer,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {
        visibility: hidden !important;
        display: none !important;
        height: 0 !important;
    }

    html, body, [class*="css"], .stApp, .stMarkdown,
    p, span, div, input, textarea, button {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Shell: the Streamlit block-container IS our centered column */
    .block-container {
        max-width: 820px !important;
        padding: 0 28px 140px 28px !important;
        margin: 0 auto !important;
    }

    /* Header */
    .cw-header {
        position: sticky;
        top: 0;
        z-index: 100;
        background: rgba(240, 247, 244, 0.88);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-bottom: 1px solid var(--cw-border);
        padding: 20px 4px 18px 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0;
    }
    .cw-brand { display: flex; align-items: center; gap: 14px; }
    .cw-logo {
        width: 44px; height: 44px;
        border-radius: 13px;
        background: linear-gradient(135deg, var(--cw-accent) 0%, #3dba7e 100%);
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 22px;
        box-shadow: 0 6px 18px rgba(45, 184, 122, 0.28);
    }
    .cw-title {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        font-weight: 400;
        color: var(--cw-text);
        line-height: 1;
        letter-spacing: -0.4px;
    }
    .cw-status {
        display: flex; align-items: center; gap: 9px;
        font-size: 14px;
        color: var(--cw-muted);
        font-weight: 500;
    }
    .cw-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: var(--cw-accent);
        animation: cw-pulse 1.8s ease-out infinite;
    }
    @keyframes cw-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(45, 184, 122, 0.55); }
        70%  { box-shadow: 0 0 0 10px rgba(45, 184, 122, 0);  }
        100% { box-shadow: 0 0 0 0 rgba(45, 184, 122, 0);     }
    }

    /* Message rows */
    .cw-row {
        display: flex;
        margin: 16px 0;
        animation: cw-slide-in 0.35s cubic-bezier(.22,.9,.3,1.2);
    }
    .cw-row-user { justify-content: flex-end; }
    .cw-row-bot  { justify-content: flex-start; align-items: flex-start; gap: 12px; }
    @keyframes cw-slide-in {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .cw-avatar {
        flex-shrink: 0;
        width: 34px; height: 34px;
        border-radius: 10px;
        background: var(--cw-bg-alt);
        border: 1px solid var(--cw-border);
        display: flex; align-items: center; justify-content: center;
        color: var(--cw-accent);
        font-size: 16px;
        margin-top: 2px;
    }
    .cw-bubble-user {
        background: var(--cw-accent);
        color: #ffffff !important;
        padding: 13px 18px;
        border-radius: 20px 20px 4px 20px;
        max-width: 78%;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0 4px 14px rgba(45, 184, 122, 0.22);
        word-wrap: break-word;
    }
    .cw-bubble-bot {
        background: var(--cw-card);
        color: var(--cw-text) !important;
        padding: 14px 18px;
        border-radius: 20px 20px 20px 4px;
        border-left: 3px solid var(--cw-accent);
        max-width: 78%;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 12px rgba(26, 58, 46, 0.07);
        word-wrap: break-word;
    }
    .cw-meta {
        font-size: 12px;
        color: var(--cw-muted);
        margin-top: 6px;
        margin-left: 4px;
        letter-spacing: 0.2px;
    }
    .cw-bubble-error {
        background: var(--cw-error-bg);
        color: var(--cw-error-text) !important;
        border-left: 3px solid #d6594b;
        padding: 14px 18px;
        border-radius: 20px 20px 20px 4px;
        max-width: 78%;
        font-size: 15px;
        line-height: 1.5;
    }
    .cw-typing {
        display: inline-flex;
        gap: 6px;
        padding: 16px 20px;
        background: var(--cw-card);
        border-radius: 20px 20px 20px 4px;
        border-left: 3px solid var(--cw-accent);
        box-shadow: 0 2px 12px rgba(26, 58, 46, 0.07);
        align-items: center;
    }
    .cw-typing span {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--cw-accent);
        opacity: 0.5;
        animation: cw-bounce 1.3s infinite ease-in-out;
    }
    .cw-typing span:nth-child(2) { animation-delay: 0.15s; }
    .cw-typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes cw-bounce {
        0%, 80%, 100% { transform: translateY(0);    opacity: 0.4; }
        40%           { transform: translateY(-6px); opacity: 1; }
    }

    /* Empty state — sized so heading + subtext + 2x2 chips fit above input */
    .cw-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px 20px 24px 20px;
        animation: cw-fade 0.6s ease-out;
    }
    @keyframes cw-fade { from {opacity:0;} to {opacity:1;} }
    .cw-empty h2 {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 46px;
        color: var(--cw-text);
        margin: 0 0 14px 0;
        font-weight: 700;
        letter-spacing: -1.4px;
        line-height: 1.08;
    }
    .cw-empty p {
        color: var(--cw-muted);
        font-size: 18px;
        max-width: 540px;
        margin: 0 auto 28px auto;
        line-height: 1.5;
    }

    /* Suggestion chips — override dark-theme button backgrounds */
    .stButton > button,
    .stButton > button:focus,
    .stButton > button:active {
        width: 100% !important;
        background-color: var(--cw-card) !important;
        background: var(--cw-card) !important;
        color: #145c3d !important;
        border: 1.5px solid var(--cw-border) !important;
        border-radius: 14px !important;
        padding: 13px 18px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: all 0.22s ease !important;
        box-shadow: 0 2px 8px rgba(26, 58, 46, 0.04) !important;
        box-sizing: border-box !important;
    }
    .stButton > button > div,
    .stButton > button p {
        color: #145c3d !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: var(--cw-bg-alt) !important;
        background: var(--cw-bg-alt) !important;
        border-color: var(--cw-accent) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(45, 184, 122, 0.18) !important;
    }
    .stButton > button:hover > div,
    .stButton > button:hover p {
        color: #0f4730 !important;
    }

    /* Chat input bar — ONLY the pill floats; strip around it is transparent */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: 820px !important;
        margin: 0 auto !important;
        padding: 12px 28px 20px 28px !important;
    }

    /* The single white pill — no nested borders */
    [data-testid="stChatInput"] > div:first-child {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1.5px solid var(--cw-border) !important;
        border-radius: 16px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(26, 58, 46, 0.08) !important;
    }
    [data-testid="stChatInput"] > div:first-child:focus-within {
        border-color: var(--cw-accent) !important;
        box-shadow: 0 4px 24px rgba(45, 184, 122, 0.18) !important;
    }
    /* Kill Streamlit's inner textarea border/background that was causing the double-outline */
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"] > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 16px !important;
        color: var(--cw-text) !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        caret-color: var(--cw-accent) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--cw-muted) !important;
        opacity: 0.75;
    }
    [data-testid="stChatInput"] button {
        background-color: #1a7a52 !important;
        background: #1a7a52 !important;
        border-radius: 10px !important;
        border: none !important;
        color: white !important;
        transition: background 0.2s ease !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #145c3d !important;
        background: #145c3d !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: white !important;
        color: white !important;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--cw-border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cw-muted); }

    [data-testid="stSpinner"] { color: var(--cw-accent) !important; }
</style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "chatbot" not in st.session_state:
    # Full-screen branded loading overlay while the chatbot initializes
    loader = st.empty()
    loader.markdown(
        """
<div style="
    position: fixed; inset: 0; z-index: 9999;
    background: linear-gradient(180deg, #f2f9f5 0%, #edf6f1 100%);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 20px;
    animation: cw-fade 0.4s ease-out;
">
    <div style="
        width: 64px; height: 64px; border-radius: 18px;
        background: linear-gradient(135deg, #2db87a 0%, #3dba7e 100%);
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 30px;
        box-shadow: 0 10px 28px rgba(45, 184, 122, 0.3);
        animation: cw-bounce-in 1.6s ease-in-out infinite;
    ">✦</div>
    <div style="
        font-family: 'DM Serif Display', serif;
        font-size: 24px; color: #1a3a2e; letter-spacing: -0.3px;
    ">CourseWise Assistant</div>
    <div style="
        font-family: 'DM Sans', sans-serif;
        font-size: 14px; color: #6b9e85;
    ">Warming up the model…</div>
</div>
<style>
@keyframes cw-bounce-in {
    0%, 100% { transform: translateY(0) scale(1);    }
    50%      { transform: translateY(-8px) scale(1.04); }
}
</style>
        """,
        unsafe_allow_html=True,
    )
    try:
        st.session_state.chatbot = initialize_chatbot()
        st.session_state.chatbot_ready = True
        loader.empty()
        st.rerun()
    except Exception as e:
        loader.empty()
        st.session_state.chatbot_ready = False
        st.session_state.chatbot_error = str(e)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="cw-header">
    <div class="cw-brand">
        <div class="cw-logo">✦</div>
        <div class="cw-title">CourseWise Assistant</div>
    </div>
    <div class="cw-status">
        <span class="cw-dot"></span>
        <span>Online</span>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def render_user(content: str) -> str:
    safe = content.replace("<", "&lt;").replace(">", "&gt;")
    return f'<div class="cw-row cw-row-user"><div class="cw-bubble-user">{safe}</div></div>'

def render_bot(content: str, meta: str | None = None, error: bool = False) -> str:
    safe = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    bubble_class = "cw-bubble-error" if error else "cw-bubble-bot"
    meta_html = f'<div class="cw-meta">{meta}</div>' if meta else ""
    return (
        '<div class="cw-row cw-row-bot">'
        '<div class="cw-avatar">✦</div>'
        f'<div style="display:flex;flex-direction:column;">'
        f'<div class="{bubble_class}">{safe}</div>'
        f'{meta_html}'
        '</div>'
        '</div>'
    )

# ----------------------------------------------------------------------------
# Body — empty state OR message list
# ----------------------------------------------------------------------------
if not st.session_state.messages and st.session_state.pending_input is None:
    st.markdown(
        """
<div class="cw-empty">
    <h2>How can I help you today?</h2>
    <p>Ask about transfer pathways, ATAIN, the CourseWise platform, or credit articulation.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "How do I access ATAIN?",
        "What is credit articulation?",
        "How do I upload data?",
        "What is the Data Use Agreement?",
    ]
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if st.button(suggestions[0], key="sug_0", use_container_width=True):
            st.session_state.pending_input = suggestions[0]
            st.rerun()
        if st.button(suggestions[2], key="sug_2", use_container_width=True):
            st.session_state.pending_input = suggestions[2]
            st.rerun()
    with c2:
        if st.button(suggestions[1], key="sug_1", use_container_width=True):
            st.session_state.pending_input = suggestions[1]
            st.rerun()
        if st.button(suggestions[3], key="sug_3", use_container_width=True):
            st.session_state.pending_input = suggestions[3]
            st.rerun()
else:
    # Render full history
    history_html = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            history_html.append(render_user(m["content"]))
        elif m.get("error"):
            history_html.append(render_bot(m["content"], error=True))
        else:
            history_html.append(render_bot(m["content"], meta=m.get("meta")))
    st.markdown("".join(history_html), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input bar (Streamlit pins chat_input to bottom automatically)
# ----------------------------------------------------------------------------
typed = st.chat_input("Ask a question...")

# ----------------------------------------------------------------------------
# Handle input — either typed or suggestion-chip
# ----------------------------------------------------------------------------
incoming = typed or st.session_state.pending_input
if incoming:
    st.session_state.pending_input = None

    if not st.session_state.get("chatbot_ready", False):
        st.session_state.messages.append({"role": "user", "content": incoming})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I couldn't initialize. {st.session_state.get('chatbot_error', 'Please refresh the page.')}",
            "error": True,
        })
        st.rerun()

    # Append user message and render immediately
    st.session_state.messages.append({"role": "user", "content": incoming})

    # Re-render user bubble now (so it appears before typing dots)
    # We use a placeholder to show typing dots, then replace with streamed text.
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        '<div class="cw-row cw-row-bot">'
        '<div class="cw-avatar">✦</div>'
        '<div class="cw-typing"><span></span><span></span><span></span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Query the chatbot
    try:
        result = st.session_state.chatbot.chat(incoming)
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        confidence = None
        if isinstance(result, dict):
            confidence = result.get("confidence") or result.get("score")

        # Stream char-by-char
        streamed = ""
        for ch in answer:
            streamed += ch
            safe = streamed.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            typing_placeholder.markdown(
                '<div class="cw-row cw-row-bot">'
                '<div class="cw-avatar">✦</div>'
                f'<div class="cw-bubble-bot">{safe}<span style="opacity:0.5;">▍</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.015)

        # Finalize (no cursor)
        meta_parts = [datetime.now().strftime("%I:%M %p").lstrip("0")]
        if confidence is not None:
            try:
                meta_parts.append(f"confidence {float(confidence)*100:.0f}%")
            except Exception:
                pass
        meta_text = " · ".join(meta_parts)
        typing_placeholder.markdown(render_bot(answer, meta=meta_text), unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": meta_text,
        })
    except Exception as e:
        err_text = f"Something went wrong: {str(e)}. Please try again."
        typing_placeholder.markdown(render_bot(err_text, error=True), unsafe_allow_html=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": err_text,
            "error": True,
        })

    st.rerun()

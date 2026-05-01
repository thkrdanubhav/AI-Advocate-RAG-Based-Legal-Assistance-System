import streamlit as st
from rag_pipeline import get_response

st.set_page_config(page_title="AI Lawyer", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("About AI Lawyer")
    st.markdown("""
    **AI Lawyer** is your legal assistant trained on documents like:
    - 📜 The Constitution of India  
    - ⚖️ Indian Penal Code  
      
    **Ask legal questions** and receive intelligent, document-grounded answers instantly.

    ---
    **Powered by:**
    - LangChain
    - FAISS
    - DeepSeek / HuggingFace Embeddings
    - Streamlit
    """)

# --- Header Section ---
st.markdown(
    """
    <div style="padding: 10px 20px; background-color: #004d99; color: white; border-radius: 8px;">
        <h2>AI Lawyer: Your Legal Assistant</h2>
        <p>Instant, document-based legal help grounded in Indian law.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- Chat Section ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear"):
        st.session_state.messages = []

# Input field
user_input = st.text_input("Enter your legal question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("AI Lawyer is analyzing your query..."):
        try:
            response = get_response(user_input)
            st.session_state.messages.append({"role": "ai", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "ai", "content": f"Error: {str(e)}"})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="background-color:#e6f0ff; color:#000; padding:12px 15px; border-radius:8px; margin-bottom:10px;">
                <strong>You:</strong><br>{msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#f4f9f2; color:#111; border-left:5px solid #2e7d32; padding:12px 15px; border-radius:8px; margin-bottom:15px;">
                <strong>AI Lawyer's Response:</strong><br>{msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Footer Note ---
st.markdown("---")
st.markdown(
    "<center><small>⚠️ This tool provides information for educational and assistance purposes only. Always consult a qualified lawyer for official legal advice.</small></center>",
    unsafe_allow_html=True
)



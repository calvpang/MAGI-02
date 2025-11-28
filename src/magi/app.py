"""MAGI-02 Streamlit Frontend.

A web interface for interacting with the MAGI multi-agent council.
"""

import streamlit as st

from magi.agents.council import BALTHASAR, CASPER, MELCHIOR, MAGICouncil
from magi.config import LM_STUDIO_BASE_URL, LM_STUDIO_MODEL
from magi.tools.rag import clear_collection, get_or_create_collection, ingest_documents

# Page configuration
st.set_page_config(
    page_title="MAGI-02 Council",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for MAGI-themed styling
st.markdown(
    """
<style>
    .magi-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .magi-title {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .magi-subtitle {
        color: #0f3460;
        font-size: 1rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .melchior {
        border-left: 4px solid #3498db;
        background-color: rgba(52, 152, 219, 0.1);
    }
    .balthasar {
        border-left: 4px solid #e74c3c;
        background-color: rgba(231, 76, 60, 0.1);
    }
    .casper {
        border-left: 4px solid #2ecc71;
        background-color: rgba(46, 204, 113, 0.1);
    }
    .consensus-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #9b59b6;
        background-color: rgba(155, 89, 182, 0.1);
    }
    .stButton > button {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "council" not in st.session_state:
        st.session_state.council = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "connected" not in st.session_state:
        st.session_state.connected = False


def render_header():
    """Render the MAGI header."""
    st.markdown(
        """
    <div class="magi-header">
        <div class="magi-title">🔮 MAGI-02 COUNCIL 🔮</div>
        <div class="magi-subtitle">Multi-Agent Governance Intelligence System</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with settings and RAG management."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LM Studio connection settings
        st.subheader("LM Studio Connection")
        base_url = st.text_input("Base URL", value=LM_STUDIO_BASE_URL)
        model = st.text_input("Model", value=LM_STUDIO_MODEL)

        if st.button("Connect to LM Studio", type="primary"):
            try:
                st.session_state.council = MAGICouncil(base_url=base_url, model=model)
                st.session_state.connected = True
                st.success("✅ Connected to LM Studio!")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                st.session_state.connected = False

        # Connection status
        if st.session_state.connected:
            st.success("🟢 Connected")
        else:
            st.warning("🟡 Not Connected")

        st.divider()

        # RAG Management
        st.subheader("📚 RAG Management")

        # Show document count
        try:
            collection = get_or_create_collection()
            doc_count = collection.count()
            st.info(f"📄 {doc_count} document chunks indexed")
        except Exception:
            st.info("📄 No documents indexed")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Ingest Docs"):
                with st.spinner("Ingesting documents..."):
                    result = ingest_documents()
                    if "error" in str(result.get("status", "")):
                        st.error(result.get("message", "Ingestion failed"))
                    else:
                        st.success(
                            f"✅ Ingested {result.get('files_processed', 0)} files, "
                            f"{result.get('chunks_added', 0)} chunks"
                        )

        with col2:
            if st.button("🗑️ Clear Docs"):
                result = clear_collection()
                if result.get("status") == "success":
                    st.success("✅ Documents cleared")
                else:
                    st.error(result.get("message", "Clear failed"))

        st.divider()

        # Agent information
        st.subheader("🤖 Council Members")

        with st.expander("MELCHIOR-1 (Scientist)"):
            st.write(MELCHIOR.description)

        with st.expander("BALTHASAR-2 (Mother)"):
            st.write(BALTHASAR.description)

        with st.expander("CASPER-3 (Woman)"):
            st.write(CASPER.description)

        st.divider()

        # Clear chat
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def render_agent_response(agent_name: str, response: str, vote: str):
    """Render an individual agent's response."""
    agent_class = agent_name.split("-")[0].lower()
    emoji_map = {
        "melchior": "🔬",
        "balthasar": "❤️",
        "casper": "✨",
    }
    emoji = emoji_map.get(agent_class, "🤖")

    st.markdown(
        f"""
    <div class="agent-card {agent_class}">
        <strong>{emoji} {agent_name}</strong> [{vote}]
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(response)


def render_consensus(consensus: str):
    """Render the council's consensus response."""
    st.markdown(
        """
    <div class="consensus-card">
        <strong>🔮 COUNCIL CONSENSUS</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(consensus)


def render_chat():
    """Render the chat interface."""
    # Mode selection
    mode = st.radio(
        "Query Mode",
        ["Full Council", "Single Agent"],
        horizontal=True,
        help="Full Council: All three agents deliberate. Single Agent: Query one agent directly.",
    )

    if mode == "Single Agent":
        selected_agent = st.selectbox(
            "Select Agent",
            ["MELCHIOR-1", "BALTHASAR-2", "CASPER-3"],
        )
    else:
        selected_agent = None

    # Show existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                if "responses" in message:
                    # Full council response
                    tabs = st.tabs(["Consensus", "MELCHIOR-1", "BALTHASAR-2", "CASPER-3"])

                    with tabs[0]:
                        render_consensus(message["content"])

                    for i, (name, response) in enumerate(message["responses"].items()):
                        with tabs[i + 1]:
                            vote = message.get("votes", {}).get(name, "N/A")
                            render_agent_response(name, response, vote)
                else:
                    # Single agent response
                    st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask the MAGI Council..."):
        if not st.session_state.connected or st.session_state.council is None:
            st.error("Please connect to LM Studio first!")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("The Council is deliberating..."):
                try:
                    if mode == "Full Council":
                        result = st.session_state.council.deliberate(prompt)

                        # Store full result in message
                        message = {
                            "role": "assistant",
                            "content": result["consensus"],
                            "responses": result["responses"],
                            "votes": result["votes"],
                        }
                        st.session_state.messages.append(message)

                        # Display result
                        tabs = st.tabs(["Consensus", "MELCHIOR-1", "BALTHASAR-2", "CASPER-3"])

                        with tabs[0]:
                            render_consensus(result["consensus"])

                        for i, (name, response) in enumerate(result["responses"].items()):
                            with tabs[i + 1]:
                                vote = result["votes"].get(name, "N/A")
                                render_agent_response(name, response, vote)
                    else:
                        # Single agent query
                        response = st.session_state.council.quick_query(prompt, selected_agent)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        st.write(response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

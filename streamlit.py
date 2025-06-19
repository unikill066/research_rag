# streamlit.py
import streamlit as st
from utils.agents import MultiAgentSystem
from utils.rag import RAG
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_multi_agent_system():
    """
    Initializes and caches the RAG and MultiAgentSystem instances.
    Using st.cache_resource to avoid re-initialization on every rerun.
    """
    try:
        # You might need to configure RAG properly based on your actual RAG implementation
        # For demonstration, assuming a basic RAG initialization.
        # If your RAG needs paths to data, embeddings, etc., pass them here.
        rag_manager = RAG()
        agent_system = MultiAgentSystem(rag_manager=rag_manager, llm_model="gpt-3.5-turbo")
        logger.info("MultiAgentSystem initialized successfully.")
        return agent_system
    except Exception as e:
        logger.error(f"Error initializing MultiAgentSystem: {e}")
        st.error(f"Failed to initialize the AI system. Please check the backend configuration. Error: {e}")
        return None

agent_system = initialize_multi_agent_system()

st.set_page_config(page_title="Multi-Agent Chatbot", page_icon="🤖")

st.title("🤖 Multi-Agent Chatbot")
st.markdown("Ask me anything! I can use RAG and internet search to answer your questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if agent_system is None:
        with st.chat_message("assistant"):
            st.markdown("The AI system is not initialized. Please try again later or check logs.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                initial_state = {
                    "query": prompt,
                    "messages": [HumanMessage(content=prompt)],
                    "rag_results": [],
                    "internet_results": [],
                    "agent_decisions": {},
                    "synthesis_result": {},
                    "final_answer": "",
                    "metadata": {"timestamp": datetime.now().isoformat()}
                }
                full_response = ""
                message_placeholder = st.empty()

                # Iterate through the LangGraph events (if you want to show intermediate steps)
                # For a simpler chatbot, you might just run it to completion and get the final state.
                # Here, we'll run it to completion and then display the final answer.
                
                # The 'stream' method is generally preferred for live updates in LangGraph
                # However, the current _build_agent_graph structure returns a compiled graph
                # which is typically run with .invoke() or .stream()
                
                # Let's assume agent_system.graph is the compiled graph from _build_agent_graph()
                # You'll need to make agent_system.graph accessible or add a run method.
                # For now, let's assume agent_system has a method to run the graph.
                
                # Modify your MultiAgentSystem class to expose the compiled graph
                # or provide a run method for the graph.
                # Example:
                # In MultiAgentSystem __init__: self.graph = self._build_agent_graph()
                # In streamlit.py: agent_system.graph.invoke(initial_state)

                # Assuming you add `self.graph = self._build_agent_graph()` to your MultiAgentSystem's __init__
                # and make `_build_agent_graph` return the compiled graph.
                
                final_state = None
                try:
                    for s in agent_system.graph.stream(initial_state):
                        if END in s:
                            final_state = s[END]
                            break
                        else:
                            pass 
                    
                    if final_state and "final_answer" in final_state:
                        full_response = final_state["final_answer"]
                    else:
                        full_response = "I could not generate a complete answer."

                except Exception as e:
                    logger.error(f"Error running agent system: {e}")
                    full_response = f"An error occurred while processing your request: {e}"

                message_placeholder.markdown(full_response)
                
            except Exception as e:
                logger.error(f"Streamlit chat input processing error: {e}")
                full_response = f"An unexpected error occurred: {e}"
                st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
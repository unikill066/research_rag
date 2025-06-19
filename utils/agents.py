"""
Author: Nikhil Nageshwar Inturi (GitHub: unikill066, email: inturinikhilnageshwar@gmail.com)

Multi-Agent System Module
Implements the LangGraph-based multi-agent system for RAG + Internet search
"""

# imports
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import logging, json
from rag import RAG
from query import QueryEngine

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    messages: List[BaseMessage]
    rag_results: List[Dict[str, Any]]
    internet_results: List[Dict[str, Any]]
    agent_decisions: Dict[str, Any]
    synthesis_result: Dict[str, Any]
    final_answer: str
    metadata: Dict[str, Any]

class MultiAgentSystem:
    """
    LangGraph-based multi-agent system combining RAG and internet search
    """
    def __init__(self, rag_manager: RAG, llm_model: str = "gpt-3.5-turbo", search_tool: str = "duckduckgo"):
        """
        Initialize the multi-agent system
        """
        self.rag_manager = rag_manager
        self.query_engine = QueryEngine(rag_manager)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

        if search_tool == "duckduckgo":
            self.search_tool = DuckDuckGoSearchRun()
        else:
            raise ValueError(f"Unsupported search tool: {search_tool}")

        self.graph = self._build_agent_graph()

    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        workflow.add_node("supervisor", self._supervisor_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("internet_agent", self._internet_agent)
        workflow.add_node("synthesis_agent", self._synthesis_agent)
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                "rag_only": "rag_agent",
                "internet_only": "internet_agent",
                "both_parallel": "rag_agent",
                "synthesis": "synthesis_agent"
            }
        )
        workflow.add_conditional_edges("rag_agent",self._route_from_rag,{"internet": "internet_agent", "synthesis": "synthesis_agent"})
        workflow.add_conditional_edges("internet_agent",self._route_from_internet,{"synthesis": "synthesis_agent"})
        workflow.add_conditional_edges("synthesis_agent",self._route_from_synthesis,{"loop": "supervisor", "end": END})
        return workflow.compile()

    def _supervisor_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        prompt = f"""
        You are a supervisor agent that decides how to best answer a user query.
        Query: \"{query}\"

        Decide one approach:
        - rag_only
        - internet_only
        - both_parallel
        - synthesis
        Respond in JSON with keys: decision, reasoning.
        """
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(resp.content)
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            data = {"decision":"both_parallel","reasoning":"fallback"}

        state["agent_decisions"]["supervisor"] = data
        state["messages"].append(AIMessage(content=f"Supervisor: {data}"))
        return state

    def _rag_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        try:
            search_resp = self.query_engine.search(query=query, k=5, use_query_expansion=True, rerank_results=True)
            eval_prompt = f"Evaluate relevance/completeness of RAG results for query: {query}."
            eval_resp = self.llm.invoke([HumanMessage(content=eval_prompt)])
            evaluation = json.loads(eval_resp.content)
        except Exception as e:
            logger.error(f"RAG error: {e}")
            search_resp, evaluation = {"results":[]}, {"need_internet": True}

        state["rag_results"] = [{"search_response":search_resp, "evaluation":evaluation}]
        state["agent_decisions"]["rag_evaluation"] = evaluation
        return state

    def _internet_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        try:
            opt_prompt = f"Optimize this query for web search: {query}" 
            opt_resp = self.llm.invoke([HumanMessage(content=opt_prompt)])
            opt_query = opt_resp.content.strip()
            results = self.search_tool.run(opt_query)

            eval_prompt = f"Evaluate web results for query: {query}" 
            eval_resp = self.llm.invoke([HumanMessage(content=eval_prompt)])
            evaluation = json.loads(eval_resp.content)
        except Exception as e:
            logger.error(f"Internet error: {e}")
            results, evaluation = [], {"completeness_score":0}

        state["internet_results"] = [{"search_query":opt_query, "results":results, "evaluation":evaluation}]
        state["agent_decisions"]["internet_evaluation"] = evaluation
        return state

    def _synthesis_agent(self, state: AgentState) -> AgentState:
        query = state["query"]
        rag = state.get("rag_results", [])
        net = state.get("internet_results", [])

        rag_context = ("" if not rag else "\n".join(
            f"Doc {i+1}: {d['search_response']['results'][i].page_content[:200]}..."
            for i,d in enumerate(rag)
        ))
        net_context = ("" if not net else str(net[0]["results"])[:1500])

        prompt = f"""
        Synthesize answer for: {query}
        Internal: {rag_context}
        Internet: {net_context}
        Include: main answer, sources, confidence, notes.
        """
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            answer = resp.content
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            answer = "Error synthesizing."

        rag_q = (sum(rag[0]["evaluation"].get(k,0) for k in ["relevance_score","completeness_score","quality_score"]) / 3
                 if rag else 0)
        net_q = (sum(net[0]["evaluation"].get(k,0) for k in ["relevance_score","completeness_score","reliability_score"]) / 3
                 if net else 0)
        conf = max(rag_q, net_q) / 5.0

        state["synthesis_result"] = {"answer":answer, "confidence":conf, "sources_used":{"rag":bool(rag),"internet":bool(net)}}
        state["final_answer"] = answer
        state["agent_decisions"]["synthesis"] = {"decision":"loop" if conf<0.7 else "end"}
        return state

    # Routing functions
    def _route_supervisor(self, state: AgentState) -> str:
        return state["agent_decisions"]["supervisor"]["decision"]

    def _route_from_rag(self, state: AgentState) -> str:
        rag_eval = state["agent_decisions"].get("rag_evaluation", {})
        sup = state["agent_decisions"]["supervisor"]["decision"]
        if sup == "both_parallel" or rag_eval.get("need_internet"):
            return "internet_agent"
        return "synthesis_agent"

    def _route_from_internet(self, state: AgentState) -> str:
        ie = state["agent_decisions"].get("internet_evaluation", {})
        sup = state["agent_decisions"]["supervisor"]["decision"]
        return "synthesis_agent"

    def _route_from_synthesis(self, state: AgentState) -> str:
        dec = state["agent_decisions"]["synthesis"]["decision"]
        return "supervisor" if dec == "loop" else END




# rag = RAG()
# agent_system = MultiAgentSystem(rag)

# graph = agent_system.graph.get_graph()
# png_bytes = graph.draw_mermaid_png()

# with open("./agent_workflow.png", "wb") as f:
#     f.write(png_bytes)
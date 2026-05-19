"""
GatewayAgentBuilder — concrete LangGraph workflow builder.

Workflow: START → agent_node → tools_condition → tool_node → agent_node → … → END

Graph is compiled once at startup. Model is resolved per-request from
state["model"] via middleware — see docs/architecture.md.
"""
import logging

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.agent.base_agent_builder import BaseAgentBuilder
from app.agent.graph_builder import GraphFlowState
from app.agent.middleware.agent_middleware import GatewayAgentMiddleware
from app.agent.llm_registry import LLMRegistry
from app.config.logger import get_trace_id
from app.tools.assistant_tools import ALL_TOOLS

logger = logging.getLogger(__name__)


class GatewayAgentBuilder(BaseAgentBuilder):
    """
    Builds and compiles the gateway's LangGraph workflow.

    Args:
        llm_registry:  The populated LLMRegistry (all providers registered).
        middleware:    GatewayAgentMiddleware instance for model resolution
                       and tool call interception.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        middleware: GatewayAgentMiddleware,
    ) -> None:
        self._registry = llm_registry
        self._middleware = middleware
        self._tools = ALL_TOOLS
        logger.info(
            "GatewayAgentBuilder: initialising | tools=[%s]",
            ", ".join(t.name for t in self._tools),
        )
        super().__init__()  # calls _create_workflow() → sets self.workflow

    # ── Workflow construction ─────────────────────────────────────────────

    def _create_workflow(self):
        """
        Build the StateGraph with middleware-injected nodes.

        agent_node:  Resolves the model from state["model"] via middleware,
                     binds tools, invokes the LLM.
        tool_node:   Executes each tool call from the last AI message,
                     wrapping each invocation with middleware pre/post hooks.
        """
        middleware = self._middleware
        tools = self._tools
        tool_map = {t.name: t for t in tools}

        # ── agent node ────────────────────────────────────────────────────
        async def agent_node(state: GraphFlowState) -> dict:
            # Middleware resolves the actual LLM for this request
            llm = middleware.resolve_model(state)
            llm_with_tools = llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        # ── tool node ─────────────────────────────────────────────────────
        async def tool_node(state: GraphFlowState) -> dict:
            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", [])
            context = state.get("request_context")
            results: list[ToolMessage] = []

            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                tool_id = call["id"]
                tool = tool_map.get(tool_name)

                if tool is None:
                    logger.warning(
                        "[%s] Unknown tool requested: %s", get_trace_id(), tool_name
                    )
                    result_content = f"Tool '{tool_name}' is not available."
                else:
                    start = middleware.pre_tool_call(tool_name, tool_args, context)
                    try:
                        result_content = await tool.ainvoke(tool_args)
                        middleware.post_tool_call(tool_name, result_content, start, context)
                    except Exception as exc:
                        logger.error(
                            "[%s] Tool '%s' raised: %s", get_trace_id(), tool_name, exc
                        )
                        result_content = f"Error executing '{tool_name}': {exc}"

                results.append(
                    ToolMessage(
                        content=str(result_content),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

            return {"messages": results}

        # ── graph assembly ────────────────────────────────────────────────
        graph = StateGraph(GraphFlowState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        compiled = graph.compile()
        logger.info("GatewayAgentBuilder: workflow compiled successfully")
        return compiled

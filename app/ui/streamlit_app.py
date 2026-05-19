"""
Resilient AI — Streamlit Dashboard

3 Tabs:
  1. Chat          — real-time streaming REST chat with tool call cards + failover badges
  2. A2A Inspector — fire JSON-RPC 2.0 tasks and inspect artifacts
  3. Monitoring    — metrics: chat requests, tools, providers, latency, errors/min

Logs are written to rotating files (not shown in the UI):
  logs/YYYY-MM-DD/gateway_SYSTEM_{date}_{hour}.log
  logs/YYYY-MM-DD/gateway_ERROR_{date}_{hour}.log
  logs/YYYY-MM-DD/gateway_REQ_RESP_{date}_{hour}.log
  logs/YYYY-MM-DD/gateway_UI_{date}_{hour}.log
"""
import sys
import pathlib

# Ensure project root is on sys.path so `app` is importable whether Streamlit
# is launched via start.sh (editable install) or directly from any working dir.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import time
import json
import uuid

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

# Separate connect vs read timeouts: connect fails fast if API is down,
# read gives the API time to respond while handling concurrent LLM work.
_HEALTH_TIMEOUT = httpx.Timeout(connect=1.5, read=8.0, write=5.0, pool=5.0)
_HEALTH_RATE_LIMIT_S = 30  # re-check at most once every 30 s per browser session

st.set_page_config(
    page_title="Resilient AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _api(path: str) -> str:
    return f"{API_BASE}{path}"


def _check_health() -> tuple[bool, str]:
    """Rate-limited health check — re-checks at most once every 30 s per session.

    Uses st.session_state instead of @st.cache_data so:
      - the result is per-browser-session (not shared across users)
      - we control exactly when it re-fires (not tied to Streamlit's rerun clock)
      - a ReadTimeout during a long LLM response doesn't flip the status to Offline
    """
    now = time.monotonic()
    last_ts = st.session_state.get("_health_ts", 0.0)

    if now - last_ts < _HEALTH_RATE_LIMIT_S:
        # Return the last known result — no network call
        return st.session_state.get("_health_result", (False, "Checking…"))

    try:
        r = httpx.get(_api("/health"), timeout=_HEALTH_TIMEOUT)
        result: tuple[bool, str] = (
            (True, "") if r.status_code == 200 else (False, f"HTTP {r.status_code}")
        )
    except httpx.ConnectError:
        result = (False, "Connection refused — is the API running?")
    except httpx.ReadTimeout:
        result = (False, "ReadTimeout — API is busy, retrying soon")
    except Exception as exc:
        result = (False, type(exc).__name__)

    st.session_state["_health_result"] = result
    st.session_state["_health_ts"] = now
    return result


@st.cache_data(ttl=10)
def _get_metrics() -> dict:
    """Cached for 10 s — reduces API hammering when the monitoring tab rerenders."""
    try:
        r = httpx.get(_api("/metrics"), timeout=5)
        return r.json()
    except Exception:
        return {}


def _init_session(system_prompt: str) -> str | None:
    try:
        r = httpx.post(
            _api("/v1/session/init"),
            json={"system_prompt": system_prompt},
            timeout=5,
        )
        return r.json().get("session_id")
    except Exception as e:
        st.error(f"Failed to create session: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Resilient AI")
    st.caption("Damco Engineering Showcase")

    health_ok, health_err = _check_health()
    health_label = "🟢 Online" if health_ok else f"🔴 Offline{' — ' + health_err if health_err else ''}"
    st.markdown("**API Status:** " + health_label)

    st.divider()
    st.subheader("Configuration")

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant. Use available tools when relevant.",
        height=80,
    )

    from app.agent.agent_util import llm_provider_map
    model_options = list(llm_provider_map.keys())
    selected_model = st.selectbox("Primary LLM Provider", model_options, index=0)

    processor_options = ["LANGGRAPH", "GOOGLE_ADK"]
    selected_processor = st.selectbox("Processor Backend", processor_options, index=0)

    if selected_processor != "LANGGRAPH":
        st.info(
            f"**{selected_processor}** is an extension point stub. "
            "It will show wiring instructions — not execute a real LLM. "
            "See the source in `app/agent/processors/` to activate."
        )

    st.divider()
    st.caption("Available tools: calculate · search_knowledge_base · get_weather · summarize_text")


# ── Session State Init ─────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    sid = _init_session(system_prompt)
    st.session_state["session_id"] = sid
    st.session_state["messages"] = []
    st.session_state["tool_calls"] = []


# ── Tabs ───────────────────────────────────────────────────────────────────

tab_chat, tab_a2a, tab_monitor = st.tabs([
    "💬 Chat", "🔗 A2A Inspector", "📊 Monitoring"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_tools = st.columns([2, 1])

    with col_chat:
        st.subheader("Chat")

        # Session controls
        ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
        with ctrl1:
            sid_display = st.session_state.get("session_id", "None")
            st.caption(f"Session: `{sid_display[:8] if sid_display else 'None'}…`")
        with ctrl2:
            if st.button("🔄 New Session"):
                sid = _init_session(system_prompt)
                st.session_state["session_id"] = sid
                st.session_state["messages"] = []
                st.session_state["tool_calls"] = []
                st.rerun()
        with ctrl3:
            stream_mode = st.toggle("Stream", value=True)

        # Chat history container
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(f"**You:** {content}")
                    st.divider()
                elif role == "failover":
                    st.warning(f"⚡ Failover: {content}")
                elif role == "assistant":
                    st.markdown(f"**Assistant:** {content}")
                    st.divider()

        # Real-time streaming placeholders (visible only while streaming)
        stream_placeholder = st.empty()
        failover_placeholder = st.empty()

        # Input
        user_input = st.chat_input("Ask something… try 'calculate 12 * 8' or 'weather in Tokyo'")

        if user_input and st.session_state.get("session_id"):
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["tool_calls"] = []

            collected_text: list[str] = []
            tool_cards: list[dict] = []

            if stream_mode:
                try:
                    with httpx.Client(timeout=60) as client:
                        with client.stream(
                            "POST",
                            _api("/v1/stream"),
                            json={
                                "session_id": st.session_state["session_id"],
                                "message": user_input,
                                "model": selected_model,
                                "processor": selected_processor,
                                "system_prompt": system_prompt,
                            },
                        ) as resp:
                            for line in resp.iter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                raw = line[5:].strip()
                                try:
                                    chunk = json.loads(raw)
                                except Exception:
                                    continue

                                ctype = chunk.get("type")
                                ccontent = chunk.get("content", "")

                                if ctype == "TEXT":
                                    collected_text.append(ccontent)
                                    # Update placeholder in real time — user sees text appear
                                    stream_placeholder.markdown(
                                        "**Assistant:** " + "".join(collected_text) + " ▌"
                                    )
                                elif ctype == "FAILOVER":
                                    failover_placeholder.warning(f"⚡ Failover: {ccontent}")
                                    st.session_state["messages"].append(
                                        {"role": "failover", "content": ccontent}
                                    )
                                elif ctype == "TOOL_RESULT":
                                    meta = chunk.get("metadata", {})
                                    tool_cards.append({
                                        "tool": meta.get("tool", "tool"),
                                        "result": ccontent,
                                    })
                                elif ctype == "ERROR":
                                    collected_text.append(f"⚠️ Error: {ccontent}")
                                    stream_placeholder.markdown(
                                        "**Assistant:** " + "".join(collected_text)
                                    )
                except Exception as e:
                    collected_text.append(f"⚠️ Connection error: {e}")

                # Clear streaming placeholders — final message goes into chat_container on rerun
                stream_placeholder.empty()
                failover_placeholder.empty()

            else:
                # Non-streaming /chat endpoint
                with st.spinner("Thinking…"):
                    try:
                        r = httpx.post(
                            _api("/v1/chat"),
                            json={
                                "session_id": st.session_state["session_id"],
                                "message": user_input,
                                "model": selected_model,
                                "processor": selected_processor,
                                "system_prompt": system_prompt,
                            },
                            timeout=60,
                        )
                        reply = r.json().get("reply", "")
                        collected_text.append(reply if reply else "⚠️ No response received.")
                    except Exception as e:
                        collected_text.append(f"⚠️ Error: {e}")

            full_reply = "".join(collected_text).strip()
            st.session_state["messages"].append({"role": "assistant", "content": full_reply})
            st.session_state["tool_calls"] = tool_cards
            st.rerun()

    with col_tools:
        st.subheader("🔧 Tool Calls")
        tool_calls = st.session_state.get("tool_calls", [])
        if not tool_calls:
            st.caption("No tool calls yet. Try asking to calculate something or check the weather.")
        for tc in tool_calls:
            with st.expander(f"🛠 {tc['tool']}", expanded=True):
                st.markdown(f"**Result:**\n{tc['result']}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — A2A INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_a2a:
    st.subheader("🔗 A2A Protocol Inspector")
    st.caption(
        "Send Agent-to-Agent JSON-RPC 2.0 tasks directly. "
        "This demonstrates the same agent logic serving a machine-to-machine protocol."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Send Task**")
        a2a_context = st.text_input("Context ID (leave blank for new)", value="")
        a2a_message = st.text_input("Message", value="What is 15 squared?")
        st.caption("Model is selected automatically by the gateway (with failover).")

        if st.button("🚀 Send A2A Task"):
            context_id = a2a_context.strip() or str(uuid.uuid4())
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tasks/send",
                "params": {
                    "contextId": context_id,
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": a2a_message}],
                    },
                },
            }

            st.markdown("**Request JSON:**")
            st.code(json.dumps(payload, indent=2), language="json")

            try:
                resp = httpx.post(_api("/agent/a2a/tasks/send"), json=payload, timeout=30)
                result = resp.json()
                st.session_state["last_a2a_result"] = result
                st.session_state["last_a2a_context"] = context_id
            except Exception as e:
                st.error(f"A2A request failed: {e}")
                result = {}

            st.markdown("**Response:**")
            st.json(result)

    with col_b:
        st.markdown("**Task Artifacts**")
        last_result = st.session_state.get("last_a2a_result", {})
        task_id = last_result.get("result", {}).get("id") if last_result else None

        if task_id:
            st.caption(f"Task ID: `{task_id}`")
            st.caption(f"Context ID: `{st.session_state.get('last_a2a_context', '-')}`")

            try:
                poll_resp = httpx.get(_api(f"/agent/a2a/tasks/{task_id}"), timeout=5)
                task_data = poll_resp.json()
            except Exception as e:
                task_data = {"error": str(e)}

            state = task_data.get("status", {}).get("state", "unknown")
            state_color = {"completed": "🟢", "working": "🟡", "failed": "🔴"}.get(state, "⚪")
            st.markdown(f"**Status:** {state_color} {state.upper()}")

            artifacts = task_data.get("artifacts", [])
            if artifacts:
                st.markdown("**Artifacts:**")
                full_text = " ".join(a.get("text", "") for a in artifacts)
                st.markdown(f"> {full_text}")
            else:
                st.caption("No artifacts yet.")

            if st.button("🔄 Refresh"):
                st.rerun()
        else:
            st.caption("Send a task to see artifacts here.")

        st.divider()
        st.markdown("**Agent Card**")
        if st.button("Fetch Agent Card"):
            try:
                card_resp = httpx.get(_api("/agent/a2a/agent-card"), timeout=5)
                st.json(card_resp.json())
            except Exception as e:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MONITORING
# ═══════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.subheader("📊 Monitoring")

    if st.button("🔄 Refresh Stats"):
        st.rerun()

    m = _get_metrics()

    if m:
        # ── KPI Cards ─────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Chat Requests", m.get("requests_total", 0))
        c2.metric("A2A Tasks", m.get("a2a_tasks_total", 0))
        c3.metric("Cache Hit Rate", f"{m.get('cache_hit_rate_pct', 0)}%")
        c4.metric("Failovers", sum(m.get("llm_failovers", {}).values()))
        c5.metric("Avg Latency", f"{m.get('avg_latency_ms', 0):.0f}ms")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            # Tool call bar chart
            tool_data = m.get("tool_calls", {})
            if tool_data:
                st.markdown("**Tool Calls by Tool**")
                st.bar_chart(tool_data)
            else:
                st.caption("No tool calls recorded yet.")

            # Latency history line chart
            latency_hist = m.get("latency_history", [])
            if latency_hist:
                st.markdown("**Request Latency (ms) — last 100**")
                st.line_chart(latency_hist)

            # Errors per minute bar chart
            error_data = m.get("errors_per_minute", {})
            if any(v > 0 for v in error_data.values()):
                st.markdown("**Errors per Minute — last 10 min**")
                st.bar_chart(error_data)
            else:
                st.caption("No errors in the last 10 minutes. ✅")

        with col_right:
            # LLM provider usage
            llm_data = m.get("llm_calls", {})
            if llm_data:
                st.markdown("**LLM Calls by Provider**")
                st.bar_chart(llm_data)
            else:
                st.caption("No LLM calls recorded yet.")

            # Failover counts
            failover_data = m.get("llm_failovers", {})
            if failover_data:
                st.markdown("**Failovers by Provider**")
                st.bar_chart(failover_data)
            else:
                st.caption("No failovers recorded.")
    else:
        st.warning("Could not fetch metrics — is the API running?")

    # ── Log file reference ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📁 Log Files")
    st.caption(
        "Logs are written to rotating hourly files — not streamed here. "
        "Check the `logs/` folder in the project root."
    )
    from datetime import datetime as _dt
    _today = _dt.now().strftime("%Y-%m-%d")
    _hour  = _dt.now().strftime("%H")
    _base  = f"logs/{_today}/"
    st.code(
        f"{_base}gateway_SYSTEM_{_today}_{_hour}.log    ← app lifecycle, LLM events\n"
        f"{_base}gateway_ERROR_{_today}_{_hour}.log     ← WARNING and above (all loggers)\n"
        f"{_base}gateway_REQ_RESP_{_today}_{_hour}.log  ← every HTTP request/response\n"
        f"{_base}gateway_UI_{_today}_{_hour}.log        ← UI / Streamlit layer",
        language="text",
    )

import streamlit as st
import requests
import time

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="Supervisor Console",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Supervisor Chat Console")
st.caption("Function-based incident & safety query interface")

backend_url = "http://127.0.0.1:8000/chat"

# ------------------------
# SESSION STATE
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# DISPLAY HISTORY
# ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------
# USER INPUT
# ------------------------
if prompt := st.chat_input("Ask about incidents, reports, or stats..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            start = time.time()

            res = requests.post(
                backend_url,
                json={"query": prompt},
                timeout=60,
            )
            res.raise_for_status()
            data = res.json()
            latency = round(time.time() - start, 2)

            result = data.get("result", {})
            tool_called = data.get("tool_called", "")
            assistant_text = ""

            # ----------------------------
            # Route rendering by tool type
            # ----------------------------
            if isinstance(result, dict) and result.get("success"):
                tool = result.get("tool", tool_called)

                # ---- get_incidents ----
                if tool == "get_incidents":
                    incidents = result.get("incidents", [])
                    assistant_text = f"### 📋 Recent Incidents ({len(incidents)})\n\n"

                    if incidents:
                        for inc in incidents:
                            sid = inc.get("id", "?")
                            sev = inc.get("severity", "?")
                            dec = inc.get("final_decision", "?")
                            score = inc.get("compliance_score", "?")
                            name = inc.get("full_name", "Unknown")
                            missing = inc.get("missing_ppe", "")
                            created = inc.get("created_at", "")

                            assistant_text += (
                                f"**#{sid}** — {sev} | {dec} | "
                                f"Score: {score} | "
                                f"Worker: {name}\n"
                            )
                            if missing:
                                assistant_text += f"  Missing PPE: _{missing}_\n"
                            if created:
                                assistant_text += f"  _{created}_\n"
                            assistant_text += "\n"
                    else:
                        assistant_text += "_No incidents found._\n"

                # ---- get_report ----
                elif tool == "get_report":
                    inc = result.get("incident", {})
                    esc = result.get("escalation_actions", [])

                    assistant_text = f"### 📄 Incident Report #{inc.get('id', '?')}\n\n"
                    assistant_text += f"**Severity:** {inc.get('severity', '?')}\n\n"
                    assistant_text += f"**Decision:** {inc.get('final_decision', '?')}\n\n"
                    assistant_text += f"**Compliance Score:** {inc.get('compliance_score', '?')}\n\n"
                    assistant_text += f"**Detected PPE:** {inc.get('detected_ppe', 'None')}\n\n"
                    assistant_text += f"**Missing PPE:** {inc.get('missing_ppe', 'None')}\n\n"
                    assistant_text += f"**Confidence:** {inc.get('confidence', '?')}\n\n"
                    assistant_text += f"**OSHA Codes:** {inc.get('osha_codes', 'None')}\n\n"
                    assistant_text += f"**Escalation:** {inc.get('escalation', 'None')}\n\n"

                    if inc.get("override_applied"):
                        assistant_text += f"**Override Reason:** {inc.get('override_reason', '')}\n\n"

                    report_text = inc.get("report", "No report available")
                    assistant_text += f"---\n**Report:**\n\n{report_text}\n\n"

                    if esc:
                        assistant_text += "#### Escalation Actions\n"
                        for action in esc:
                            assistant_text += f"- {action.get('action', '?')}: {action.get('details', '')}\n"
                        assistant_text += "\n"

                # ---- get_stats ----
                elif tool == "get_stats":
                    assistant_text = "### 📊 Safety Dashboard\n\n"
                    assistant_text += f"**Total Incidents:** {result.get('total_incidents', 0)}\n\n"
                    assistant_text += f"**Average Compliance Score:** {result.get('avg_compliance_score', 'N/A')}\n\n"

                    if result.get("by_severity"):
                        assistant_text += "#### 🚨 Severity Breakdown\n"
                        for sev, count in result["by_severity"].items():
                            assistant_text += f"- **{sev}**: {count}\n"
                        assistant_text += "\n"

                    if result.get("by_decision"):
                        assistant_text += "#### ✅ Decision Breakdown\n"
                        for dec, count in result["by_decision"].items():
                            assistant_text += f"- **{dec}**: {count}\n"
                        assistant_text += "\n"

                    if result.get("most_common_missing_ppe"):
                        assistant_text += "#### 🦺 Most Common Missing PPE\n"
                        for item, count in result["most_common_missing_ppe"].items():
                            assistant_text += f"- {item}: {count}\n"
                        assistant_text += "\n"

                    if result.get("total_escalation_actions", 0) > 0:
                        assistant_text += f"**Total Escalation Actions:** {result['total_escalation_actions']}\n\n"
                        if result.get("escalations_by_type"):
                            assistant_text += "#### Escalations by Type\n"
                            for action, count in result["escalations_by_type"].items():
                                assistant_text += f"- {action}: {count}\n"
                            assistant_text += "\n"

                # ---- unknown tool ----
                else:
                    assistant_text = f"✅ Result from `{tool}`:\n```json\n{result}\n```\n"

            else:
                error_msg = ""
                if isinstance(result, dict):
                    error_msg = result.get("error", "")
                assistant_text = f"⚠️ Request failed.{f' {error_msg}' if error_msg else ''}"

            assistant_text += f"\n\n⏱ _Response time: {latency}s_ · 🔧 _Tool: {tool_called}_"

            st.markdown(assistant_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text,
            })

        except requests.exceptions.ConnectionError:
            error_text = "⚠️ Cannot reach backend. Is the server running on port 8000?"
            st.markdown(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})

        except Exception as e:
            error_text = f"⚠️ Error: {str(e)}"
            st.markdown(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})

# streamlit run supervisor_chat.py --server.port 8502
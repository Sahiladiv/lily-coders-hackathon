import streamlit as st
import requests
import time

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="Supervisor Console",
    page_icon="🛡️",
    layout="wide"
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

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            start = time.time()

            res = requests.post(
                backend_url,
                json={"query": prompt},
                timeout=60
            )

            res.raise_for_status()
            data = res.json()
            latency = round(time.time() - start, 2)

            result = data.get("result", {})
            tool_called = data.get("tool_called", "")

            assistant_text = ""

            # ----------------------------
            # Convert Result → Chat Format
            # ----------------------------

            if isinstance(result, dict) and result.get("success"):

                total = result.get("total_incidents", 0)
                avg_score = result.get("avg_compliance_score", "N/A")

                assistant_text += f"### 📊 Incident Summary\n\n"
                assistant_text += f"**Total Incidents:** {total}\n\n"
                assistant_text += f"**Average Compliance Score:** {avg_score}\n\n"

                # Severity
                if "by_severity" in result:
                    assistant_text += "#### 🚨 Severity Breakdown\n"
                    for sev, count in result["by_severity"].items():
                        assistant_text += f"- **{sev.title()}**: {count}\n"
                    assistant_text += "\n"

                # Decision
                if "by_decision" in result:
                    assistant_text += "#### ✅ Decision Breakdown\n"
                    for dec, count in result["by_decision"].items():
                        assistant_text += f"- **{dec.title()}**: {count}\n"
                    assistant_text += "\n"

                # PPE
                if "most_common_missing_ppe" in result:
                    assistant_text += "#### 🦺 Most Common Missing PPE\n"
                    for item, count in result["most_common_missing_ppe"].items():
                        assistant_text += f"- {item}: {count}\n"
                    assistant_text += "\n"

            else:
                assistant_text = "⚠️ I couldn’t retrieve the requested data."

            assistant_text += f"\n⏱ _Response time: {latency}s_"

            # Display nicely in chat bubble
            st.markdown(assistant_text)

            # Save full response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text
            })

        except Exception as e:
            error_text = f"⚠️ Error connecting to backend: {str(e)}"
            st.markdown(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_text
            })
            
    # Store assistant summary in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Tool: {tool_called}"
    })

    # streamlit run app.py --server.port 8502
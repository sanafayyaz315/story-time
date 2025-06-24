import streamlit as st
import httpx
import time

API_URL = "http://localhost:8000/generate"

# ——— Page setup ———
st.set_page_config(page_title="StoryTime Chat", layout="wide")
st.title("📖 Story Generator")

# ——— Session state ———
if "stop_streaming" not in st.session_state:
    st.session_state.stop_streaming = False

if "history" not in st.session_state:
    # history is a list of tuples: (role, content)
    st.session_state.history = []

# ——— 1) Render chat history up top ———
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Add a spacer so the input always feels “pushed” to the bottom
st.markdown("<br><br><br>", unsafe_allow_html=True)

# ——— 2) Input + Stop in a bottom row ———
col_input, col_stop = st.columns([6, 1], gap="small")
with col_input:
    # Single-arg only: this is your placeholder
    user_input = st.chat_input("Once upon a time…")
with col_stop:
    if st.button("⛔ Stop"):
        st.session_state.stop_streaming = True

# ——— 3) On user submit ———
if user_input:
    # Clear stop flag
    st.session_state.stop_streaming = False

    # Show & store user message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare assistant bubble
    full_output = ""
    st.session_state.history.append(("assistant", ""))  # placeholder
    assistant_idx = len(st.session_state.history) - 1

    with st.chat_message("assistant"):
        output_container = st.empty()

        # Stream from FastAPI
        try:
            with httpx.stream("POST", API_URL, json={"input_text": user_input}, timeout=None) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_text():
                    if st.session_state.stop_streaming:
                        output_container.markdown(full_output + "\n\n**⛔ Stopped by user.**")
                        break
                    full_output += chunk
                    output_container.markdown(full_output)
                    time.sleep(0.01)

        except Exception as e:
            output_container.error(f"⚠️ Error: {e}")

    # Save the assistant’s real response back into history
    st.session_state.history[assistant_idx] = ("assistant", full_output)

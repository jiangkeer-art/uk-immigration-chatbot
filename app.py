import streamlit as st
from rag_module import rag_answer

st.set_page_config(page_title="Immigration Assistant", page_icon="🇬🇧")
st.title("🇬🇧 Immigration Q&A Assistant")
st.caption("Answer questions regarding immigration, visas, and passports based on official UK government documents.")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption("source: " + ", ".join(msg["sources"]))

if prompt := st.chat_input("Please enter your question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag_answer(prompt)
            st.markdown(answer)
            if sources:
                st.caption("source: " + ", ".join(sources))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
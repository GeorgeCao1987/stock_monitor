from __future__ import annotations

import streamlit as st

from bot import build_default_bot

st.set_page_config(page_title="WolfBot", page_icon="🐺", layout="wide")

if "wolfbot" not in st.session_state:
    st.session_state.wolfbot = build_default_bot()
if "messages" not in st.session_state:
    st.session_state.messages = []

bot = st.session_state.wolfbot

st.title("🐺 WolfBot · 阿狼交易体系研究助手")
st.caption("基于《自立自强》+ NGA 阿狼发言 + 行情数据。它不是阿狼本人，也不替你下单。")

with st.sidebar:
    st.subheader("知识库")
    stats = bot.kb.stats()
    st.write(f"PDF 片段：{stats.get('pdf', 0)}")
    st.write(f"NGA 发言：{stats.get('nga', 0)}")
    st.caption("新增 NGA 发言后，重启应用即可重新建索引。")
    if st.button("重新加载知识库"):
        st.session_state.wolfbot = build_default_bot()
        st.rerun()

    st.divider()
    st.subheader("建议问法")
    st.markdown("- `002916 深南电路现在按阿狼体系怎么看？`\n- `什么叫反抽转反弹？`\n- `缩量跌破为什么不能直接算破位？`\n- `调整期应该怎么控制仓位？`\n- `今天适不适合做T，要看什么？`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("输入股票代码 + 问题，或直接问交易体系……")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("检索阿狼语料并分析……"):
            answer = bot.answer(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

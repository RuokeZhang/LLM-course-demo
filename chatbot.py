import os
from agents import Head_Agent


class Chatbot:
    """
    Wrapper around Head_Agent to provide a simple chat() interface
    for Streamlit or other front-end applications.
    """

    def __init__(self):
        # 从环境变量读取，部署时在 Streamlit Cloud 的 Secrets 里配置
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        pinecone_key = os.environ.get("PINECONE_API_KEY", "")
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "machine-learning-index")

        if not openai_key or not pinecone_key:
            raise ValueError(
                "请设置环境变量 OPENAI_API_KEY 和 PINECONE_API_KEY。"
                "本地：在终端 export 或建 .env；部署：在 Streamlit Cloud → App → Settings → Secrets 里配置。"
            )

        self.head_agent = Head_Agent(
            openai_key=openai_key,
            pinecone_key=pinecone_key,
            pinecone_index_name=pinecone_index_name
        )

        self.head_agent.setup_sub_agents()

    def chat(self, user_input: str) -> str:

        user_input = user_input.strip()

        if not user_input:
            return "Please enter a valid question."

        # 1️⃣ Obnoxious check
        is_obnoxious = self.head_agent.obnoxious_agent.check_query(user_input)

        if is_obnoxious:
            return "Your query is inappropriate. Please ask a respectful question."

        # 2️⃣ Rewrite query
        rewritten_query = self.head_agent.context_rewriter.rephrase(
            self.head_agent.history,
            user_input
        )

        # 3️⃣ Query Pinecone
        docs = self.head_agent.query_agent.query_vector_store(rewritten_query)

        matches = docs.get("matches", [])
        doc_texts = []

        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                doc_texts.append(match["metadata"]["text"])

        context = "\n\n".join(doc_texts)

        # 4️⃣ Relevance check
        relevance_input = f"""
User Query:
{rewritten_query}

Retrieved Documents:
{context}
"""

        relevance = self.head_agent.relevant_agent.get_relevance(relevance_input)


        if not docs.get("matches"):
            response = "I don't know the answer based on the available documents."
        else:
            response = self.head_agent.answering_agent.generate_response(
                rewritten_query,
                docs,
                self.head_agent.history
            )

        # 5️⃣ Update history
        self.head_agent.history.append(f"User: {user_input}")
        self.head_agent.history.append(f"Assistant: {response}")

        return response
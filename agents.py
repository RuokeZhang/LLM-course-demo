from openai import OpenAI
from pinecone import Pinecone
import json
from typing import List, Dict, Any


class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = (
            "You are a moderation agent. Your job is to decide if the user query is obnoxious, offensive, or inappropriate (e.g. insults, hate speech, harassment). "
            "You must reply with exactly one word: Yes or No. "
            "Yes = the query is obnoxious or inappropriate. No = the query is acceptable."
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        if not response:
            return False
        text = response.strip().lower()
        first = (text.split() or [""])[0].rstrip(".,")
        if first in ("yes", "y"):
            return True
        if first in ("no", "n"):
            return False
        # 兜底：整句就是 yes/no
        if text in ("yes", "no", "yes.", "no."):
            return text.startswith("y")
        return False

    def check_query(self, query):
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content
        return self.extract_action(answer)


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        self.client = openai_client
        self.prompt = (
            "You are a context rewriter agent. "
            "Rewrite the latest user query so it is self-contained and unambiguous. "
            "Return only the rewritten query."
        )

    def rephrase(self, user_history, latest_query):
        history_text = "\n".join(user_history) if user_history else ""

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"Conversation history:\n{history_text}\n\nLatest query:\n{latest_query}"
                },
            ],
        )

        return response.choices[0].message.content


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.index = pinecone_index
        self.client = openai_client

    def query_vector_store(self, query, k=5):
        embedding_response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        query_vector = embedding_response.data[0].embedding

        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )

        return results


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are an answering agent. "
            "Use the provided documents to answer the user's query. "
            "If the answer is not contained in the documents, say you don't know."
        )

    def generate_response(self, query, docs, conv_history, k=5):

        matches = docs.get("matches", [])[:k]

        doc_texts = []
        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                doc_texts.append(match["metadata"]["text"])

        context = "\n\n".join(doc_texts)
        history_text = "\n".join(conv_history) if conv_history else ""

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"""
Conversation history:
{history_text}

Relevant documents:
{context}

User question:
{query}
"""
                },
            ],
        )

        return response.choices[0].message.content


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "Determine if retrieved documents are relevant to the user query. "
            "Answer strictly with 'Yes' or 'No'."
        )

    def get_relevance(self, conversation) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": conversation},
            ],
        )

        return response.choices[0].message.content.strip()


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:

        self.client = OpenAI(api_key=openai_key)

        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(pinecone_index_name)

        self.history = []

        self.obnoxious_agent = None
        self.context_rewriter = None
        self.query_agent = None
        self.answering_agent = None
        self.relevant_agent = None

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.client, None)
        self.answering_agent = Answering_Agent(self.client)
        self.relevant_agent = Relevant_Documents_Agent(self.client)


class TestDatasetGenerator:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.dataset = {
            "obnoxious": [],
            "irrelevant": [],
            "relevant": [],
            "small_talk": [],
            "hybrid": [],
            "multi_turn": []
        }

    def generate_synthetic_prompts(self, category: str, count: int) -> List[Dict]:

        prompt = f"""
Generate {count} JSON test cases for category: {category}.
Return only valid JSON list.
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You generate structured JSON test data."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except:
            return []

    def build_full_dataset(self):
        self.dataset["obnoxious"] = self.generate_synthetic_prompts("obnoxious", 10)
        self.dataset["irrelevant"] = self.generate_synthetic_prompts("irrelevant", 10)
        self.dataset["relevant"] = self.generate_synthetic_prompts("relevant", 10)
        self.dataset["small_talk"] = self.generate_synthetic_prompts("small_talk", 5)
        self.dataset["hybrid"] = self.generate_synthetic_prompts("hybrid", 8)
        self.dataset["multi_turn"] = self.generate_synthetic_prompts("multi_turn", 7)
        return self.dataset

    def save_dataset(self, filepath: str = "test_set.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=4)

    def load_dataset(self, filepath: str = "test_set.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        return self.dataset
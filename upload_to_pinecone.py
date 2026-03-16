import os
import uuid
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

# ========== 1. 从环境变量读取 API keys（本地可 export 或建 .env）==========
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "machine-learning-index")

if not OPENAI_KEY or not PINECONE_KEY:
    print("错误：请先设置环境变量 OPENAI_API_KEY 和 PINECONE_API_KEY")
    print("示例：export OPENAI_API_KEY='your-key' && export PINECONE_API_KEY='your-key'")
    exit(1)

# ========== 2. Initialize OpenAI and Pinecone ==========
client = OpenAI(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# ========== 3. Read the PDF file ==========
reader = PdfReader("machine-learning.pdf")
full_text = ""

for page in reader.pages:
    full_text += page.extract_text() + "\n"

# ========== 4. Split text into chunks ==========
chunk_size = 800
chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i+chunk_size]
    chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")

# ========== 5. Generate embeddings and upload to Pinecone ==========
for chunk in chunks:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    vector = response.data[0].embedding

    index.upsert(
        vectors=[{
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {"text": chunk}
        }]
    )

print("Upload complete!")
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

loader = DirectoryLoader("documents", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"    

for doc in documents:
    if start_marker in doc.page_content:
        doc.page_content = doc.page_content.split(start_marker)[1]
    if end_marker in doc.page_content:
        doc.page_content = doc.page_content.split(end_marker)[0]


print(f"Loaded {len(documents)} documents")
print(f"Document length after cleanup: {len(documents[0].page_content)} characters")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print(chunks[0])

#embeddings:
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

test_vector = embeddings.embed_query("What does the monster want?")
print(f"Vector length: {len(test_vector)}")
print(f"First 5 numbers: {test_vector[:5]}")
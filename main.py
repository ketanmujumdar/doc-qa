from langchain.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


ollama = Ollama(base_url='http://localhost:11434', model="llama2")
loader = PyPDFLoader("PRD.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_split = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_split, embedding=GPT4AllEmbeddings())

qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())


question = "Is there a deadline for the project?"

print(qachain({"query":question}))




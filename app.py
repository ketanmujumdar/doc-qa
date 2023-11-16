import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from htmlTemplates import css, bot_template, user_template

def main():
    st.set_page_config(page_title='Document Listner', page_icon=':shark:', layout='wide')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('Document Listner :shark')
    st.subheader('This is a simple app to listen to your document and can ask questions about it.')

    user_question = st.text_input('Enter your question here')

    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload your document here', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing your documents'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation(vectorstore)




def get_pdf_text(pdf_docs):
    text = ''
    for doc in pdf_docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=GPT4AllEmbeddings())
    return vectorstore


def get_conversation(vectorstore):
    # ollama = Ollama(base_url='http://localhost:11434', model="falcon")
    ollama = Ollama(base_url='http://localhost:11434', model="llama2")
    # memory = ConversationBufferMemory(memory_key="document-listner", return_message=True)
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    return qachain


def handle_user_input(user_question):
    st.write(user_template.replace("{{MSG}}",user_question), unsafe_allow_html=True)
    bot_answer = st.session_state.conversation({"query":user_question})
    st.write(bot_template.replace("{{MSG}}",bot_answer["result"]), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração da página do Streamlit
st.set_page_config(page_title="RAG Inteligente (Gemini)", layout="centered")
st.title("📚 Chat com seus Documentos (Gemini AI)")

# Sidebar para inserir a chave da API do Google
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if api_key:
  os.environ["GOOGLE_API_KEY"] = api_key

  # Upload do arquivo PDF
  uploaded_file = st.file_uploader(
      "Envie um documento PDF para consulta", type="pdf"
  )

  if uploaded_file is not None:
    # Salvar temporariamente o PDF
    with open("temp.pdf", "wb") as f:
      f.write(uploaded_file.getbuffer())

    with st.spinner("Processando e vetorizando o documento com Google AI..."):
      # 1. Carregar o documento
      loader = PyPDFLoader("temp.pdf")
      docs = loader.load()

      # 2. Dividir em Chunks
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000, chunk_overlap=200
      )
      splits = text_splitter.split_documents(docs)

      # 3. Criar Embeddings e Banco Vetorial (Usando Google)
      embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
      vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
      retriever = vectorstore.as_retriever(
          search_kwargs={"k": 3}
      )

      # 4. Configurar o LLM e o Prompt (Usando Gemini 1.5 Flash)
      llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

      template = """Você é um assiststrente prestativo para responder a perguntas.
      Use os seguintes trechos de contexto recuperados para responder à pergunta. 
      Se você não sabe a resposta, diga que não sabe.

      Contexto:
      {context}

      Pergunta: {question}
      """
      prompt = ChatPromptTemplate.from_template(template)

      def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

      # Cadeia moderna do LangChain
      rag_chain = (
          {"context": retriever | format_docs, "question": RunnablePassthrough()}
          | prompt
          | llm
          | StrOutputParser()
      )

      st.session_state["rag_chain"] = rag_chain
      st.session_state["retriever"] = retriever

    st.success("Documento processado com sucesso! Pode fazer perguntas abaixo.")

  # Se o documento já foi processado, exibe a caixa de texto para perguntas
  if "rag_chain" in st.session_state:
    user_query = st.text_input("O que você quer saber sobre o documento?")

    if user_query:
      with st.spinner("Buscando resposta no Gemini..."):
        # Executa o RAG
        answer = st.session_state["rag_chain"].invoke(user_query)
        st.write("### Resposta:")
        st.write(answer)

        # Buscar fontes utilizadas para mostrar as referências
        source_docs = st.session_state["retriever"].invoke(user_query)
        with st.expander("Ver trechos de referência (Fontes)"):
          for i, doc in enumerate(source_docs):
            st.markdown(
                f"**Trecho {i+1} (Página"
                f" {doc.metadata.get('page', 0) + 1}):**"
            )
            st.text(doc.page_content[:300] + "...")
else:
  st.warning("Insira sua Google Gemini API Key na barra lateral para começar.")
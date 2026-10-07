# SmartRAG Profissional: Chat Inteligente com Documentos

Aplicação de Inteligência Artificial baseada em **RAG (Retrieval-Augmented Generation)** desenvolvida em Python para permitir consultas interativas e precisas em documentos PDF. O projeto utiliza busca semântica em banco de dados vetorial para fornecer respostas fundamentadas com indicação de fontes, eliminando alucinações de LLMs.

## Arquitetura do Sistema
1. **Ingestão:** Leitura e fragmentação (*Chunking*) de arquivos PDF utilizando divisores baseados em caracteres com sobreposição de contexto (*overlap*).
2. **Indexação:** Transformação dos trechos em vetores semânticos (*Embeddings*) armazenados localmente no **ChromaDB**.
3. **Recuperação e Geração:** Busca por similaridade dos trechos mais relevantes combinados dinamicamente com o modelo da OpenAI através da linguagem de expressão moderna do **LangChain (LCEL)**.

## Tecnologias Utilizadas
- **Python** (Linguagem principal)
- **Streamlit** (Interface gráfica web interativa)
- **LangChain** (Orquestração de correntes e manipulação de prompts)
- **ChromaDB** (Banco de dados vetorial local)
- **OpenAI API** (GPT-4o-mini e modelos de Embedding)

## Como Executar o Projeto Localmente

1. Clone o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/smart-rag.git](https://github.com/seu-usuario/smart-rag.git)
   cd smart-rag

2. Crie e ative o ambiente virtual:
   ```bash
    python3 -m venv venv
    source venv/bin/activate

3. Instale as dependências necessárias:
   ```bash
    pip install langchain langchain-core langchain-community langchain-openai chromadb pypdf streamlit

4. Execute a instalação Streamlit:
   ```bash
    streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.enableCORS=false --server.enableXsrfProtection=false

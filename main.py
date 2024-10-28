import fitz
import yfinance as yf
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import gradio as gr

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client()
books_collection = client.get_or_create_collection("livros_investimentos")
finance_collection = client.get_or_create_collection("financeiro")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def process_and_store_book_data(pdf_path, title):
    text = extract_text_from_pdf(pdf_path)
    paragraphs = text.split("\n\n")

    for i, paragraph in enumerate(paragraphs):
        embedding = embedding_model.encode(paragraph).tolist()

        books_collection.add(
            documents=[paragraph],
            metadatas=[{"title": title, "paragraph_id": i}],
            ids=[f"{title}_{i}"],
            embeddings=[embedding]
        )

pdf_files = {
    "O Investidor Inteligente": "livros/o-investidor-inteligente.pdf",
    "O Mais Importante para o Investidor": "livros/o-mais-importante-para-o-investidor.pdf",
    "Pai Rico Pai Pobre": "livros/pai-rico-pai-pobre.pdf",
    "Solidity para Iniciantes": "livros/solidity-para-iniciantes.pdf"
}

for title, path in pdf_files.items():
    process_and_store_book_data(path, title)
    print(f"Processado: {title}")

def fetch_investment_data(ticker):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period="1y")
    info = stock.info
    last_close = historical_data['Close'].iloc[-1]
    data_text = (
        f"Empresa: {info.get('shortName', 'N/A')}\n"
        f"Setor: {info.get('sector', 'N/A')}\n"
        f"Indústria: {info.get('industry', 'N/A')}\n"
        f"Resumo: {info.get('longBusinessSummary', 'N/A')}\n"
        f"Último preço de fechamento: {last_close}\n"
        f"Máximo (1 ano): {historical_data['Close'].max()}\n"
        f"Mínimo (1 ano): {historical_data['Close'].min()}\n"
        f"Variação (1 ano): {((last_close / historical_data['Close'].iloc[0]) - 1) * 100:.2f}%\n"
        f"Capitalização de mercado: {info.get('marketCap', 'N/A')}"
    )
    embedding = embedding_model.encode(data_text).tolist()
    finance_collection.add(
        documents=[data_text],
        metadatas=[{"ticker": ticker}],
        ids=[ticker],
        embeddings=[embedding]
    )


tickers = ["AAPL", "TSLA", "GOOGL", "VOO", "NVDA", "AMZN"]
for ticker in tickers:
    fetch_investment_data(ticker)
    print(f"Dados financeiros processados para: {ticker}")


def retrieve_documents_from_books(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = books_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = [doc for sublist in results['documents'] for doc in sublist]
    return docs


def retrieve_documents_from_finance(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = finance_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = [doc for sublist in results['documents'] for doc in sublist]
    return docs


def generate_response_with_context(query):
    book_docs = retrieve_documents_from_books(query)
    finance_docs = retrieve_documents_from_finance(query)

    context = "\n\n".join(book_docs + finance_docs)


    print("Contexto combinado:", context)


    messages = [
        {"role": "system", "content": "Você é um assistente especializado em investimentos."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
    ]


    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"messages": messages, "max_tokens": 300}
    )


    if response.status_code == 200:
        generated_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        print("Resposta gerada:", generated_text)
        return generated_text
    else:
        print("Erro na geração da resposta:", response.status_code, response.text)
        return "Erro na geração da resposta."

def process_query(query):
    response = generate_response_with_context(query)
    return response


interface = gr.Interface(
    fn=process_query,
    inputs="text",
    outputs="text",
    title="Consultor de Investimentos e Conhecimento",
    description="Consulte sobre estratégias de investimento, dados de mercado e fundamentos."
)

interface.launch()

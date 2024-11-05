import pika
import json
import os
import concurrent.futures
from pathlib import Path

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import (
    TextConverter,
    PDFToTextConverter,
    DocxToTextConverter,
    PreProcessor,
    EmbeddingRetriever,
    BM25Retriever,
    JoinDocuments,
)
import openai

# Set your OpenAI API key
openai.api_key = ''

# Initialize the document store
document_store = InMemoryDocumentStore(use_bm25=True)

def write_documents(file_path):
    """Convert and write the documents to the document store."""
    if file_path.endswith(".docx"):
        converter = DocxToTextConverter()
    elif file_path.endswith(".txt"):
        converter = TextConverter()
    else:
        converter = PDFToTextConverter()

    doc = converter.convert(file_path, meta=None)[0]
    preprocessor = PreProcessor(split_by='word', split_length=350)
    docs = preprocessor.process([doc])

    document_store.write_documents(docs)

    # Update BM25 index
    retriever = BM25Retriever(document_store=document_store)
    document_store.update_embeddings(retriever)

def chunk_documents(file_path):
    """Chunk the documents for summarization."""
    if file_path.endswith(".docx"):
        converter = DocxToTextConverter()
    elif file_path.endswith(".txt"):
        converter = TextConverter()
    else:
        converter = PDFToTextConverter()

    doc = converter.convert(file_path, meta=None)[0]
    preprocessor = PreProcessor(split_by='word', split_length=3000)
    docs = preprocessor.process([doc])
    return [d.content for d in docs]

def query_pipeline(query):
    """Query the pipeline for context using hybrid retrieval and reciprocal rank fusion."""
    retriever_emb = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=False,
    )
    retriever_bm25 = BM25Retriever(document_store=document_store)
    join_docs = JoinDocuments(join_mode="reciprocal_rank_fusion", top_k=4)

    retrieved_docs_emb = retriever_emb.retrieve(query=query, top_k=4)
    retrieved_docs_bm25 = retriever_bm25.retrieve(query=query, top_k=4)
    documents = join_docs.run(documents=[retrieved_docs_emb, retrieved_docs_bm25])['documents']
    return documents

def query_router(query, conversation_history=[]):
    """Route the query to the appropriate choice based on the system response."""
    messages = [
        {"role": "system", "content": """You are a professional decision making query router bot for a chatbot system that decides whether a user's query requires a summary, 
a retrieval of extra information from a vector database, or a simple greeting/gratitude/salutation response. If the query
requires a summary, you will reply with only "(1)". If the query requires extra information, you will reply with only "(2)".
If the query requires a simple greeting/gratitude/salutation/ or an answer to a follow up question based on conversation history 
response, you will reply with only "(3)"."""},
        {"role": "user", "content": f"""You are given a user's query in the <query> field. You are responsible for routing the query to the appropriate
choice as described in the system response. <query>{query}</query> You are also given the history of the conversation in the <history>{conversation_history}</history> field."""}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def map_summarizer(query, chunk):
    """Summarize each chunk of text based on a user's query."""
    messages = [
        {"role": "system", "content": """You are a professional corpus summarizer for a chatbot system. 
You are responsible for summarizing a chunk of text according to a user's query."""},
        {"role": "user", "content": f"""You are given a user's query in the <query> field. Respond appropriately to the user's input
using the provided chunk in the <chunk> tags: <query>{query}</query>\n <chunk>{chunk}</chunk>"""}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def reduce_summarizer(query, analyses):
    """Summarize the list of summaries into a final summary based on a user's query."""
    analyses_text = "\n".join(analyses)
    messages = [
        {"role": "system", "content": """You are a professional corpus summarizer for a chatbot system. 
You are responsible for summarizing a list of summaries according to a user's query."""},
        {"role": "user", "content": f"""You are given a user's query in the <query> field. Respond appropriately to the user's input
using the provided list of summaries in the <chunk> tags: <query>{query}</query>\n <chunk>{analyses_text}</chunk>"""}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def simple_responder(query, conversation_history=[]):
    """Respond to a user's query based on a simple follow-up response."""
    messages = [
        {"role": "system", "content": """You are a professional greeting/gratitude/salutation/ follow-up responder for a chatbot system. 
You are responsible for responding politely to a user's query."""},
    ]
    for msg in conversation_history:
        messages.append(msg)
    messages.append({"role": "user", "content": query})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def summary_tool(query, file_path):
    """Summarize the document based on a user's query."""
    chunks = chunk_documents(file_path)
    analyses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(map_summarizer, query, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            analyses.append(future.result())
    final_summary = reduce_summarizer(query, analyses)
    return final_summary

def context_tool(query):
    """Retrieve context based on a user's query."""
    context_docs = query_pipeline(query)
    context = "\n".join([doc.content for doc in context_docs])
    messages = [
        {"role": "system", "content": """You are a professional Q/A responder for a chatbot system. 
You are responsible for responding to a user query using ONLY the context provided within the <context> tags."""},
        {"role": "user", "content": f"""You are given a user's query in the <query> field. Respond appropriately to the user's input using only the context
in the <context> field:\n <query>{query}</query>\n <context>{context}</context>"""}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def process_query(query, file_path, conversation_history):
    # First, route the query
    intent = query_router(query, conversation_history).strip()

    if intent == "(1)":
        if file_path:
            if not os.path.exists(file_path):
                return "Error: File not found."
            write_documents(file_path)
            response = summary_tool(query, file_path)
        else:
            return "Error: No file provided for summarization."
    elif intent == "(2)":
        response = context_tool(query)
    elif intent == "(3)":
        response = simple_responder(query, conversation_history)
    else:
        response = "Error: Could not determine intent."
    return response

# Set up RabbitMQ connection and channel
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='input_queue')

def on_request(ch, method, props, body):
    message = json.loads(body)
    # Extract data from the message
    session_id = message.get('session_id')
    query = message.get('query')
    file_path = message.get('file_path')
    conversation_history = message.get('conversation_history', [])

    # Process the query
    response = process_query(query, file_path, conversation_history)

    # Send the response back to the reply_to queue
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id
        ),
        body=response
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='input_queue', on_message_callback=on_request)
print(" [x] Awaiting RPC requests")
channel.start_consuming()

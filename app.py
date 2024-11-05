import streamlit as st
import openai
import uuid
import pika
import json
import os

# Initialize OpenAI API key (if needed)
openai.api_key = ''  # Optional: You can set this in the worker instead

# Define the RAGAgent class with RabbitMQ integration
class RAGAgent:
    def __init__(self, host='localhost'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        # Declare a callback queue for receiving responses
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        # Set up a consumer for the callback queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
        self.response = None
        self.correlation_id = None

    def on_response(self, ch, method, props, body):
        if self.correlation_id == props.correlation_id:
            self.response = body

    def invoke_agent(self, query: str, file) -> str:
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Prepare the message to send to the worker
        message = {
            'session_id': st.session_state.session_id,
            'query': query,
            'file_path': os.path.join("uploads", file.name) if file else None,
            'conversation_history': st.session_state.messages
        }

        self.response = None
        self.correlation_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='input_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.correlation_id,
            ),
            body=json.dumps(message)
        )

        # Wait for the response from the worker
        while self.response is None:
            self.connection.process_data_events()
        return self.response.decode()

    def __del__(self):
        self.connection.close()

def clear_convo():
    st.session_state["messages"] = []

def init():
    st.set_page_config(page_title="Agentic RAG", page_icon=":robot_face:")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

if __name__ == "__main__":
    init()

    agent = RAGAgent()

    # Streamlit components
    st.title("Agentic RAG :robot_face:")

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    st.sidebar.markdown(
        """This app demonstrates agentic Retrieval Augmented Generation (RAG). It is capable of routing a user query to the appropriate choice 
        of either summarizing a document, providing extra information from a vector database, or providing a simple follow-up response.
        The agent itself does not depend on any orchestrator (e.g., LangChain) and uses Haystack only to index and retrieve documents."""
    )
    openai.api_key = api_key  # Optional: Remove if set in the worker
    clear_button = st.sidebar.button(
        "Clear Conversation", key="clear", on_click=clear_convo
    )

    file = st.file_uploader("Choose a file to index...", type=["docx", "pdf", "txt"])
    clicked = st.button("Upload File", key="Upload")
    if file and clicked:
        with st.spinner("Uploading file..."):
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            file_path = os.path.join("uploads", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success("File uploaded!")

    user_input = st.chat_input("Say something")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Processing your request..."):
            res = agent.invoke_agent(user_input, file)
            st.session_state.messages.append({"role": "assistant", "content": res})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

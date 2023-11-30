from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import FastEmbedEmbeddings
from langchain.chains import ConversationalRetrievalChain


class ChatPDF:
    def __init__(self):
        self.embedding = None
        self.vector_store = None
        self.retriever = None
        self.chain = None

        self.model = ChatOllama(model="mistral")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.init_embedding()
        self.prompt = ChatPromptTemplate.from_template(
            "Answer questions about a PDF document based the retrieved documents below:\n{context}\n"
            "Question: {question}",
        )

    def init_embedding(self):
        self.embedding = FastEmbedEmbeddings()

    def ingest(self, pdf_file_path: str):
        loader = UnstructuredPDFLoader(
            pdf_file_path,
            mode="elements",
            strategy="fast"
        )
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)

        vector_store = Qdrant.from_documents(
            chunks,
            self.embedding,
            location=":memory:",
            collection_name="pdf"
        )
        self.retriever = vector_store.as_retriever()

        self.chain = ConversationalRetrievalChain.from_llm(self.model, self.retriever)

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain({"question": query, "chat_history": ""})


if __name__ == "__main__":
    chat = ChatPDF()
    chat.ingest("cidr2021_paper17.pdf")
    print(chat.ask("Who is the authors?"))

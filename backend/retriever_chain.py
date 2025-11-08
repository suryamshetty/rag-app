# backend/retriever_chain.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

class QARetriever:
    """
    Modern retrieval + generation using Google Gemini (no deprecated classes)
    """

    def __init__(self, vectorstore, model_name: str = "models/gemini-2.5-flash", api_key: str = None):
        # ✅ Load Gemini API key (from env or directly passed)
        self.api_key = "AIzaSyBAOHhv4kWHwevhKhfQHopQBcnAFoHog4U"

        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found. Set it in your environment or pass explicitly.")

        # ✅ Initialize Gemini chat model
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.3,
            max_output_tokens=512,
        )

        # ✅ Convert FAISS store to retriever
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ✅ Prompt Template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant.\n"
                "Use the following context to answer the question accurately.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer concisely and clearly:"
            ),
        )

    def query(self, question: str):
        print(f"🔍 Querying: {question}")

        # Retrieve top documents
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format the prompt
        final_prompt = self.prompt.format(context=context, question=question)

        # Generate answer
        response = self.llm.invoke(final_prompt)

        return {
            "answer": response.content,
            "sources": [doc.metadata for doc in docs],
        }


def build_qa_chain(vectorstore):
    # ✅ Choose Gemini model
    MODEL_NAME = "models/gemini-2.5-flash"  # or "gemini-1.5-pro" for deeper reasoning
    return QARetriever(vectorstore=vectorstore, model_name=MODEL_NAME)

from chunker import DocumentChunker
from retriever import Retriever
from llm_interface import LLMInterface
from dotenv import load_dotenv
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import gradio as gr

class RAGSystem:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.retriever = Retriever()
        self.llm = LLMInterface()
        
    def load_documents(self, text: str):
        chunks = self.chunker.chunk_text(text)
        self.retriever.add_documents(chunks)
        
    def answer_question(self, 
                       query: str, 
                       api_key: str,
                       use_bm25: bool = True, 
                       use_semantic: bool = True) -> str:
        relevant_chunks = self.retriever.retrieve(
            query, 
            use_bm25=use_bm25, 
            use_semantic=use_semantic
        )
        
        return self.llm.generate_response(query, relevant_chunks, api_key)

    def run(self, text, api_key):
        return self.answer_question(text, api_key)
# Приклад використання
if __name__ == "__main__":
    rag = RAGSystem()
    
    # text = "\n".join(str(gutenberg.raw(file_id)) for file_id in gutenberg.fileids())
    text = str(gutenberg.raw('bible-kjv.txt'))
    rag.load_documents(text)
    
    demo = gr.Interface(
        fn=rag.run,
        inputs=["text", "text"],
        outputs=["text"],
        title="RAG System with Groq LLM, trained on Bible",
    )
    demo.launch()

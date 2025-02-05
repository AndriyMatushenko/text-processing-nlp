import spacy
import re
import numpy as np
import gradio as gr
from transformers import pipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# Ініціалізація моделей
nlp = spacy.load('uk_core_news_sm')
bert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
generator = pipeline('text2text-generation', model='facebook/bart-large-cnn')

# Попередня обробка тексту
def preprocess_text_spacy(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])

# Клас для пошуку
class TextRetriever:
    def __init__(self, corpus):
        self.original_corpus = corpus
        self.processed_corpus = [preprocess_text_spacy(doc) for doc in corpus]
        self.bm25 = BM25Okapi([doc.split() for doc in self.processed_corpus])
        self.faiss_index = self._build_faiss_index()

    def _build_faiss_index(self):
        embeddings = np.array(bert_model.encode(self.processed_corpus))
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index

    def search(self, query, top_n=3):
        processed_query = preprocess_text_spacy(query)

        # Пошук BM25
        bm25_docs = self.bm25.get_top_n(processed_query.split(), self.processed_corpus, n=top_n)

        # Пошук FAISS
        query_vector = bert_model.encode(query)
        _, indices = self.faiss_index.search(np.array([query_vector]).astype('float32'), top_n)
        faiss_docs = [self.processed_corpus[i] for i in indices[0] if i < len(self.processed_corpus)]

        # Об'єднання результатів
        combined = list({doc: None for doc in bm25_docs + faiss_docs}.keys())[:top_n]
        return [self.original_corpus[self.processed_corpus.index(doc)] for doc in combined]

# Генератор відповідей
class ResponseGenerator:
    def generate_response(self, query, retrieved_docs):
        context = ' '.join(retrieved_docs)
        prompt = f'Контекст: {context} \nПитання: {query} \nВідповідь:'
        result = generator(prompt, max_length=150, num_beams=4, early_stopping=True)
        return result[0]['generated_text'].replace('Відповідь:', '').strip()

# Ініціалізація системи
corpus = [
    "Машинне навчання використовує алгоритми для аналізу та прогнозування даних.",
    "Штучний інтелект має широкий спектр застосувань у науці та техніці.",
    "Технології обробки природної мови сприяють автоматизації багатьох завдань у різних сферах.",
    "У світі зростає потреба в системах для обробки великих обсягів даних.",
    "Використання нейронних мереж дозволяє значно покращити точність моделей у різних галузях."
]

retriever = TextRetriever(corpus)
response_generator = ResponseGenerator()

def process_query(query):
    retrieved_docs = retriever.search(query, top_n=3)
    response = response_generator.generate_response(query, retrieved_docs)
    return response, "\n\n".join(f"📄 {doc}" for doc in retrieved_docs)

# Створення інтерфейсу Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="Привіт!") as demo:
    gr.Markdown("# 🦉 Інформаційна Технологія Обробки Неструктурованих Текстів")
    gr.Markdown("Система для пошуку та аналізу інформації")

    with gr.Row():
        query_input = gr.Textbox(
            label="Введіть ваш запит:",
            placeholder="Наприклад: Що таке цивільний кодекс?",
            scale=5
        )
        submit_btn = gr.Button("Пошук", variant="primary", scale=1)

    with gr.Row():
        response_output = gr.Textbox(label="Відповідь", interactive=False)
        docs_output = gr.Textbox(label="Знайдені документи", interactive=False)

    examples = gr.Examples(
        examples=[
            ["Що таке машинне навчання?"],
            ["Як ШІ допомагає в науці?"],
            ["Що таке обробка природної мови?"]
        ],
        inputs=[query_input]
    )

    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=[response_output, docs_output]
    )

if __name__ == "__main__":
    demo.launch()

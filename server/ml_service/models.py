import os
import requests
import gc
import numpy as np
import torch
import transformers
import faiss

from .preprocessing import clean_html
from config import URL_GEN, URL_TRANSL, \
                   XRapidAPI_Key_GEN, XRapidAPI_Host_GEN, XRapidAPI_Key_TRANSL, XRapidAPI_Host_TRANSL


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
LLM_DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
LLM_DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
LLM_DEFAULT_SYSTEM_PROMPT = "Ты — русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
EMBEDDER_MODEL_NAME = "DeepPavlov/rubert-base-cased"


# Класс RAG-системы
class RAG:
    def __average_pool(self, last_hidden_states, attention_mask):
        """
        Функция, возвращает эмбеддинги с последнего слоя
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __init_embedder(self):
        """
        Инициализация LLM-модели для получения векторных представлений текстов
        """
        self.embedder_model = transformers.AutoModel.from_pretrained(EMBEDDER_MODEL_NAME)
        self.embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(EMBEDDER_MODEL_NAME)
        self.embedder_model.to(self.device)
        self.emb_dim = 768

    def __init__(self, knowledge_base, qa_pairs, device):
        """
        Конструктор класса RAG
        """
        self.device = device

        # инициализация моделей
        self.__init_embedder()

        # настройки API-доступа
        self.URL_GEN = URL_GEN
        self.URL_TRANSL = URL_TRANSL

        self.headers_gen = {
            "content-type": "application/json",
            "X-RapidAPI-Key": XRapidAPI_Key_GEN,
            "X-RapidAPI-Host": XRapidAPI_Host_GEN
        }

        self.headers_transl = {
            "content-type": "application/json",
            "X-RapidAPI-Key": XRapidAPI_Key_TRANSL,
            "X-RapidAPI-Host": XRapidAPI_Host_TRANSL,
        }

        self.payload_gen = {
            "model": "mistral-7b",
            "messages": [],
            "temperature": 0.5,
            "top_p": 0.95,
            "max_tokens": 1000,
            "use_cache": False,
            "stream": False
        }

        self.payload_transl = {
            "texts": [],
            "tl": "ru",
            "sl": "en"
        }

        # сохранение данных
        self.knw_base = knowledge_base
        self.qa_base = qa_pairs

        # векторизация базы знаний и создание индекса
        chunk_embs = self.vectorize(self.knw_base['chunk'])
        self.chunk_index = faiss.IndexFlatIP(self.emb_dim)
        self.chunk_index.add(chunk_embs)

        # векторизация вопросов и создание индекса
        ques_embs = self.vectorize(self.qa_base['question'])
        self.ques_index = faiss.IndexFlatIP(self.emb_dim)
        self.ques_index.add(ques_embs)

    @torch.no_grad()
    def vectorize(self, data):
        """
        Получение векторных представлений текстов
        """
        data_vectorized = np.zeros((len(data), self.emb_dim))
        for index, sentence in enumerate(data):
            batch_dict = self.embedder_tokenizer(sentence,
                                                 max_length=512,
                                                 padding=True,
                                                 truncation=True,
                                                 return_tensors='pt').to(self.device)
            outputs    = self.embedder_model(**batch_dict)
            embeddings = self.__average_pool(outputs.last_hidden_state,
                                             batch_dict['attention_mask'])
            data_vectorized[index] = embeddings.cpu().detach().numpy()
        return data_vectorized

    def generate(self, input):
        """
        Функция, генерирует ответ на запрос пользователя
        """
        self.payload_gen["messages"].append({"role": "user", "content": input})
        response_gen = requests.post(self.URL_GEN,
                                     json=self.payload_gen,
                                     headers=self.headers_gen).json()['choices'][0]['message']['content']

        self.payload_transl["texts"] = [response_gen]
        response_transl = requests.post(self.URL_TRANSL,
                                        json=self.payload_transl,
                                        headers=self.headers_transl).json()["texts"]

        self.payload_gen["messages"].append({"role": "assistant", "content": response_transl})
        return response_transl

    def ask(self, user_request, k_neighbours=(15, 15)):
        """
        Функция поиска чанков и генерации ответа на запрос пользователя
        """
        user_request = clean_html(user_request)
        user_request_vector = self.vectorize([user_request])

        # поиск чанков по запросу
        _, chunk_indexes_by_req = self.chunk_index.search(user_request_vector, k_neighbours[1])

        # поиск чанков по эталону
        _, ques_indexes_by_req = self.ques_index.search(user_request_vector, k_neighbours[0])
        standard_answer        = self.qa_base['answer'].iloc[ques_indexes_by_req[0][0]]
        standard_answer_vector = self.vectorize([standard_answer])
        _, chunk_indexes_by_st = self.chunk_index.search(standard_answer_vector, k_neighbours[1])

        # генерация ответа
        answer_by_req = self.knw_base['chunk'].iloc[chunk_indexes_by_req[0][0]]
        answer_by_st  = self.knw_base['chunk'].iloc[chunk_indexes_by_st[0][0]]

        input = f"""
            вопрос: {user_request}
            промпты: {standard_answer}, {answer_by_req}, {answer_by_st}
        """
        output = self.generate(input)

        return output

    def clean_history(self):
        """
        Функция для очистки истории сообщений
        """
        self.payload_gen["messages"] = []

    def __del__(self):
        """
        Деструктор, удаляет LLM-модели из памяти GPU
        """
        del self.embedder_model
        gc.collect()
        torch.cuda.empty_cache()
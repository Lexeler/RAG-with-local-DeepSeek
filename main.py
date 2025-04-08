import os
import faiss
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from prompts import SYSTEM_PROMPT


# === Настройки и пути к файлам ===
VECTOR_DB_DIR = 'vector_db'
INDEX_FILE_PATH = f'{VECTOR_DB_DIR}/index.faiss'


def database_exists_and_is_up_to_date(knowledge_base_path, index_file_path):
    """
    Проверяет, существует ли индекс и не изменился ли исходный файл базы знаний
    после создания индекса.
    
    :param knowledge_base_path: Путь к файлу базы знаний.
    :param index_file_path: Путь к файлу индекса.
    :return: True, если индекс существует и он актуален, иначе False.
    """
    if not os.path.exists(index_file_path):
        return False
    
    file_modified_time = os.path.getmtime(knowledge_base_path)
    last_db_update_time = os.path.getmtime(index_file_path)
    
    return file_modified_time <= last_db_update_time


def create_or_load_database(knowledge_base_path, index_file_path):
    """
    Создаёт или загружает базу данных Faiss. Если индекс существует и актуален,
    он загружается. В противном случае индекс пересоздаётся.
    
    :param knowledge_base_path: Путь к файлу базы знаний.
    :param index_file_path: Путь к файлу индекса.
    :return: Кортеж (db, chunks), где
             db — объект Faiss-индекса,
             chunks — список текстовых блоков из базы знаний.
    """
    if database_exists_and_is_up_to_date(knowledge_base_path, index_file_path):
        # Загружаем существующую базу данных
        db = faiss.read_index(index_file_path)
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            knowledge_data = f.read()
        chunks = knowledge_data.split('##')
    else:
        # Создаём новую базу данных
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            knowledge_data = f.read()
        chunks = knowledge_data.split('##')
        
        # Создаём эмбеддинги для каждого блока
        embeddings_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        embeddings = embeddings_model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')

        # Создаём индекс
        db = faiss.IndexFlatL2(embeddings.shape[1])
        db.add(embeddings)
        
        # Сохраняем индекс на диск
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        faiss.write_index(db, index_file_path)

    return db, chunks


def search(query, db, chunks):
    """
    Ищет наиболее близкий (похожий) блок в базе знаний по заданному запросу.
    
    :param query: Текстовый запрос.
    :param db: Объект Faiss-индекса.
    :param chunks: Список текстовых блоков (строк).
    :return: Наиболее похожий текстовый блок из базы.
    """
    embeddings_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    query_embedding = embeddings_model.encode([query]).astype('float32')
    
    # Ищем среди 1 ближайшего соседа
    k = 1
    distances, indices = db.search(query_embedding, k)

    # Находим индекс блока с минимальной дистанцией
    min_distance_idx = np.argmin(distances[0])
    
    return chunks[indices[0][min_distance_idx]].strip()


def call_api(content):
    """
    Вызывает API для получения ответа на вопрос, используя предварительно 
    сформированный текст сообщения (content).
    
    :param content: Текст (prompt), который будет отправлен в тело запроса.
    :return: Словарь (JSON), полученный в ответ от API.
    """
    url = "http://localhost:8000/v3/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "temperature": 0,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def main():
    """
    Основная функция: запрос пути к базе данных, проверка/создание индекса, ввод
    вопроса, поиск похожего блока в базе и отправка запроса к API для 
    получения итогового ответа.
    """
    # Запрос пути к базе знаний от пользователя
    knowledge_base_path = input('Введите путь до базы данных: ').strip()
    if not knowledge_base_path:
        knowledge_base_path = 'knowledge_base.txt'
        print('Путь до базы данных не указан, используется значение по умолчанию: knowledge_base.txt')
    
    print('Загрузка базы данных...')
    db, chunks = create_or_load_database(knowledge_base_path, INDEX_FILE_PATH)
    print('База данных загружена.')
    
    # Ввод вопроса
    question = input('Введите вопрос: ').strip()
    print('Поиск ближайшего блока из базы данных...')
    base_text = search(question, db, chunks)
    print('Ближайший блок найден:', base_text)
    
    # Формируем контент для отправки в API
    content = (
        f'Ответьте на вопрос на основе базы знаний: {base_text}\n\n'
        f'Вопрос: {question}'
    )
    
    print("Выполняю запрос...")
    result = call_api(content)
    
    # Обрабатываем ответ
    full_answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
    short_answer = full_answer.split('\n</think>\n\n')[-1].strip()
    print("\n\nОтвет:", short_answer)


if __name__ == "__main__":
    main()
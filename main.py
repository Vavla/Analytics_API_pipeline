import os
import csv
import json
import pandas as pd
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

chunk_size = 10
INPUT_CSV = "input.csv"
OUTPUT_JSON = "result.json"

PROMPT = """
Ты — профессиональная система извлечения данных из текстовых описаний товаров.

Твоя задача: проанализировать входной файл и вернуть JSON массив: [{"id": "", "price": "", "brand": "", "category": ""}]

Примеры категории товара:
 - "Электроника" (если это ноутбук, телефон и т.п.)
 - "Обувь" (если товар - кроссовки, сандалии и т.п)
 - "Одежда"
 - "Бытовая техника"(если товар - холодильник, микроволновка и другая бытовая электроника)
 - "Канцелярские товары"
 - "Косметика"
 - "Туризм" (например, средства от клещей)

Используй НЕ ТОЛЬКО перечисленные категории.

Цена товара - число без указания валюты. Название бренда указывать так, как написано в описании товара (тот же регистр и тому подобное)
Если информация отсутствует — ставь прочерк, не придумывай.

Формат ответа: Только JSON, без пояснений"""

def read_csv_data(filepath: str):
    chunks = pd.read_csv(filepath, encoding='utf-8', chunksize=chunk_size, on_bad_lines='skip')
    return chunks

def main():
    client = OpenAI(
        base_url=os.getenv('OPENROUTER_BASE_URL'),
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Файл {INPUT_CSV} не найден.")
        return
        
    reader = read_csv_data(INPUT_CSV)
    
    all_results = []
    chunk_number = 0

    for chunk in reader:
        chunk_number += 1
        attempt = 0
        
        if chunk.empty:
            continue

        while attempt < 3:
            try:
                attempt += 1
                print(f"Обработка чанка {chunk_number}, попытка {attempt}...")
                
                response = client.chat.completions.create(
                    model=os.getenv('MODEL_NAME'),
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": chunk.to_string(index=False, header=True)}
                    ]
                )

                if not response.choices[0].message.content:
                    print(f"Пустой ответ от API для чанка {chunk_number}")
                    continue
                    
                content = response.choices[0].message.content.strip()
                
                # Очистка от маркеров кода
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # Парсим JSON
                parsed_data = json.loads(content)
                
                if isinstance(parsed_data, dict):
                    all_results.append(parsed_data)
                elif isinstance(parsed_data, list):
                    all_results.extend(parsed_data)
                else:
                    print(f"Неожиданный формат данных: {type(parsed_data)}")
                    
                print(f"Чанк {chunk_number} успешно обработан")
                break
                
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON для чанка {chunk_number}, попытка {attempt}: {e}")
                    
            except Exception as e:
                print(f"Ошибка для чанка {chunk_number}, попытка {attempt}: {e}")

    if all_results:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в {OUTPUT_JSON}")
    else:
        print("Нет данных для сохранения. Проверьте входной файл и API.")

if __name__ == "__main__":
    main()
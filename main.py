import os
import csv
import json
import pandas as pd
from typing import List, Dict, Any
from openai import OpenAI  # OpenRouter использует совместимый с OpenAI API
from dotenv import load_dotenv

load_dotenv()

chunk_size = 20
INPUT_CSV = "input.csv"
OUTPUT_JSON = "result.json"

def read_csv_data(filepath: str) -> List[Dict[str, str]]:
    reader = pd.read_csv(filepath, encoding='utf-8', chunksize=chunk_size)
    return reader



def main():

    client = OpenAI(
        base_url=os.getenv('OPENROUTER_BASE_URL'),
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    if not os.path.exists(INPUT_CSV):
            print(f"ERROR: Файл {INPUT_CSV} не найден.")
            return
            
    reader = read_csv_data(INPUT_CSV)
    print([i for i in reader])

    # with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # print(f"Результат анализа сохранен в {OUTPUT_JSON}")

if __name__ == "__main__":
    # Устанавливаем переменную окружения для предотвращения ошибок SSL (иногда требуется)
    #os.environ['SSL_CERT_FILE'] = os.getenv('SSL_CERT_FILE', '')
    main()
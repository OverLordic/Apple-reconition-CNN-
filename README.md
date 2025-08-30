Проект: Классификация изображений apple / not_apple

Структура:
- src/ — код (apple_cnn.py)
- data/ — локальные данные (train/test не хранятся в репо)
- models/, artifacts/ — модели и артефакты (игнорируются)
- image/ — примеры картинок (опционально)

Установка:
1) Python 3.10+ 
2) python -m venv .venv
3) .venv\Scripts\activate   (Windows)
4) pip install -r requirements.txt

Данные:
Создать папки:
data/train/apple, data/train/not_apple
data/test/apple,  data/test/not_apple
Положить свои изображения.

Запуск:
python src/apple_cnn.py
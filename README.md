# Лабораторная 10. Поиск похожих изоражений
## Установка зависимостей
Выполнить команду
```
pip install -r requirements.txt
```
В папках `val2017` и `voc2012` разместить изображения из датасетов **COCO 2017 Val images \[5K/1GB]** и **VOC2012** соответственно.
## Команды
Выполняются в приведённом порядке, можно пропустить первые две.

Обучить k-средних:
```
python train_model.py
```
Проиндексировать изображения:
```
python gen_db.py
```
Запустить приложение:
```
streamlit run main.py
```
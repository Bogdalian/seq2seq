# Описание конвейера обработки данных

В этом документе описывается конвейер обработки данных, реализованный в предоставленных скриптах Python: `feature_dataset_manager.py` и `model_data_preparator.py`. Конвейер подготавливает данные для моделей машинного обучения, выполняя конструирование признаков и этапы предварительной обработки с использованием Apache Spark.

## 1. Управление набором признаков (`feature_dataset_manager.py`)

Этот скрипт фокусируется на конструировании признаков и создании наборов данных. Он считывает необработанные данные, фильтрует и преобразует их, а также генерирует два ключевых набора данных: `filtered_df` и `classifier_data`.

### Входные данные

Скрипт ожидает два CSV-файла в качестве входных данных:

-   `actions.csv`: Содержит информацию о действиях пользователя.
    -   Схема: `guid` (String), `date` (Timestamp), `result` (Integer)
-   `triggers.csv`: Содержит информацию о триггерах, связанных с действиями пользователя.
    -   Схема: `guid` (String), `date` (Timestamp), `trigger` (Integer), `type` (Integer)

Эти пути настраиваются с помощью аргументов скрипта.

### Этапы обработки

1.  **Загрузка и фильтрация данных**:
    *   Считывает `actions.csv` и `triggers.csv` в Spark DataFrames.
    *   Переименовывает и фильтрует `actions` DataFrame, чтобы создать `offers` DataFrame, добавляя столбец `next_offer_date` с использованием оконной функции, чтобы определить дату следующего предложения для каждого пользователя.
    *   Переименовывает и фильтрует `triggers` DataFrame, чтобы создать `actions` DataFrame.

2.  **Фильтрация действий перед предложениями**:
    *   Объединяет `actions` и `offers` DataFrames по столбцу `guid`.
    *   Фильтрует объединенный DataFrame, чтобы включить только действия, которые произошли до соответствующего предложения, но после предыдущего предложения (или начала временного периода, если это первое предложение).
    *   Результирующий DataFrame - `filtered_df`, который содержит действия пользователя, которые произошли в соответствующем временном интервале перед каждым предложением.

3.  **Генерация данных для классификатора**:
    *   Фильтрует `filtered_df`, чтобы включить только действия, которые произошли в течение указанного временного окна (`time_window`, по умолчанию 7 дней) до даты предложения.
    *   Группирует отфильтрованные данные по `guid` и `offer_date` и подсчитывает количество действий для каждой группы.
    *   Фильтрует эти группы, чтобы включить только те, у которых минимальное количество действий (`min_actions`, по умолчанию 30).
    *   Объединяет отобранные группы с исходным `filtered_df`, чтобы создать `classifier_data` DataFrame.

### Выходные данные

Скрипт создает два файла Parquet:

-   `filtered_df.parquet`: Содержит отфильтрованные данные о действиях.
-   `classifier_data.parquet`: Содержит данные, подготовленные для задач классификации, с минимальным количеством действий в течение указанного временного окна перед каждым предложением.

### Параметры

Скрипт принимает следующие параметры:

-   `actions_path`: Путь к файлу `actions.csv`.
-   `triggers_path`: Путь к файлу `triggers.csv`.
-   `output_dir`: Каталог, в котором будут сохранены выходные файлы Parquet.
-   `force_recompute`: Если `True`, принудительно пересчитывает наборы данных, даже если они уже существуют.
-   `min_actions`: Минимальное количество действий, необходимое для включения группы в набор данных `classifier_data`.
-   `time_window`: Временное окно (в днях) для учета действий перед предложением.

## 2. Подготовка данных модели (`model_data_preparator.py`)

Этот скрипт фокусируется на предварительной обработке наборов данных `filtered_df` и `classifier_data`, сгенерированных `feature_dataset_manager.py`, чтобы сделать их пригодными для моделей машинного обучения. Он использует библиотеку `ptls` для предварительной обработки.

### Входные данные

Скрипт считывает два файла Parquet в качестве входных данных:

-   `filtered_df.parquet`: Выходные данные из `feature_dataset_manager.py`.
-   `classifier_data.parquet`: Выходные данные из `feature_dataset_manager.py`.

### Этапы обработки

1.  **Загрузка данных**:
    *   Считывает `filtered_df.parquet` и `classifier_data.parquet` в Spark DataFrames.

2.  **Предварительная обработка данных**:
    *   Использует `PysparkDataPreprocessor` из библиотеки `ptls` для предварительной обработки DataFrames.
    *   Преобразователь преобразует столбец `action_date` в метку времени и кодирует столбец `action` как категориальные признаки.

3.  **Сохранение данных**:
    *   Сохраняет предварительно обработанные DataFrames в новые файлы Parquet в указанном выходном каталоге.

### Выходные данные

Скрипт создает два предварительно обработанных файла Parquet:

-   `preprocessed_filtered_df.parquet`: Предварительно обработанная версия `filtered_df`.
-   `preprocessed_classifier_data.parquet`: Предварительно обработанная версия `classifier_data`.

### Параметры

Скрипт принимает следующие параметры:

-   `filtered_df_path`: Путь к файлу `filtered_df.parquet`.
-   `classifier_data_path`: Путь к файлу `classifier_data.parquet`.
-   `output_dir`: Каталог, в котором будут сохранены выходные файлы Parquet.

## 3. Выполнение

Оба скрипта предназначены для выполнения с использованием Apache Spark. Пример использования представлен в блоке `if __name__ == '__main__':` каждого скрипта. 

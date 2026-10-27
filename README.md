# Рекомендательные системы — ДЗ 1

**Автор:** Гудков Денис Андреевич.

**Датасет:** Amazon Reviews 2018 — Video Games, 5-core.
[Первоисточник](https://nijianmo.github.io/amazon/index.html),
[отзывы](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz),
[метаданные](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz).

## Запуск

Команды выполняются из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab experiments.ipynb
```

Выполните ячейки ноутбука по порядку. При первом запуске отзывы и метаданные
скачаются в `data/`. Данные и материалы курса исключены из Git.

## Воспроизводимость

Запускайте ноутбук с чистого ядра, выполняя все ячейки по порядку.
Версии библиотек зафиксированы в `requirements.txt`. Разбиение по времени
детерминировано; ALS, FM и timeSVD++ используют `SEED = 42`.
Сравнивайте метрики, выбранные параметры и рекомендации: время выполнения меняется.

SHA-256 исходных архивов:

```text
a0bd63e90ad3bd577778678672bb2e7e88b8c085683c86aca2f18c1ab543cb84  Video_Games_5.json.gz
4069e17c7efa33e7574c2e77f1fe0b0c6f2a42a7ac29c4422696d788d8a8e0a6  meta_Video_Games.json.gz
```

## Файлы

- [experiments.ipynb](experiments.ipynb) — подготовка данных, подбор параметров, сравнение на test и выводы.
- [models/](models/) — реализации моделей и общие функции оценки.
- [hw1/hw1.md](hw1/hw1.md) — условие задания.
- [requirements.txt](requirements.txt) — зависимости с проверенными версиями.


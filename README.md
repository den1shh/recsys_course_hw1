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

## Файлы

- [experiments.ipynb](experiments.ipynb) — подготовка данных, подбор параметров, сравнение на test и выводы.
- [models/](models/) — реализации моделей и общие функции оценки.
- [hw1/hw1.md](hw1/hw1.md) — условие задания.
- [requirements.txt](requirements.txt) — зависимости с проверенными версиями.


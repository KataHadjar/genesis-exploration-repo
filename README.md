# Genesis exploration

Набор демонстрационных экспериментов с физическим движком [Genesis World](https://genesis-world.readthedocs.io/). Скрипты находятся в `scripts/`, готовые записи запусков — в `videos/`.

## Состав

| Скрипт | Что показывает | Запуск |
| --- | --- | --- |
| `scripts/materials_on_slope.py` | Движение тел с разными материалами по наклонной плоскости | `python scripts/materials_on_slope.py` |
| `scripts/mouse_control_with_cloth.py` | Интерактивное управление тканью мышью | `python scripts/mouse_control_with_cloth.py` |
| `scripts/two_robot_cloth_handover.py` | Передача и растяжение ткани двумя роботами | `python scripts/two_robot_cloth_handover.py` |

Записи результатов:

- `videos/materials_on_slope.mp4`
- `videos/mouse_control_with_cloth.mp4`
- `videos/two_robots_cloth_handover.mp4`

## Окружение

Нужен Python **3.10–3.13**. Genesis требует PyTorch; подходящий вариант PyTorch зависит от ОС, GPU и версии CUDA/ROCm. Установите его по [официальной инструкции PyTorch](https://pytorch.org/get-started/locally/), затем установите проект и Genesis:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch      # Для GPU выберите команду с сайта PyTorch
python -m pip install -e .
```

В `pyproject.toml` указаны поддерживаемая версия Python и зависимость `genesis-world`. PyTorch устанавливается отдельно, чтобы выбрать сборку под конкретное оборудование. Для CPU запустите `materials_on_slope.py` как есть или используйте флаг `--cpu` у двух остальных сценариев. Скрипты с графическим окном требуют рабочего графического окружения; `materials_on_slope.py` также поддерживает `--headless`.

Проверьте интерпретатор и доступность основных библиотек:

```bash
python scripts/check_environment.py
```

## Запуск сценариев

Команды запускайте из корня репозитория. Основные параметры:

```bash
python scripts/materials_on_slope.py --headless --steps 300
python scripts/mouse_control_with_cloth.py --cpu
python scripts/two_robot_cloth_handover.py --cpu
```

Чтобы увидеть все параметры конкретного сценария, добавьте `--help`. Сценарии с мышью и двумя роботами по умолчанию выбирают GPU; на системах без поддерживаемого GPU укажите `--cpu`.

## Структура

```text
.
├── scripts/    # эксперименты и проверка окружения
├── videos/     # видео готовых запусков
├── README.md
└── pyproject.toml
```

Локальное виртуальное окружение, кэши Python и сгенерированные численные артефакты не следует добавлять в Git.

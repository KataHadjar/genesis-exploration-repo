# Genesis exploration

Набор демонстрационных экспериментов с физическим движком [Genesis World](https://genesis-world.readthedocs.io/). Скрипты находятся в `scripts/`, готовые записи запусков — в `videos/`.

## Состав

| Скрипт | Что показывает | Запуск |
| --- | --- | --- |
| `scripts/materials_on_slope.py` | Движение тел с разными материалами по наклонной плоскости | `python scripts/materials_on_slope.py` |
| `scripts/mouse_control_with_cloth.py` | Интерактивное управление тканью мышью | `python scripts/mouse_control_with_cloth.py` |
| `scripts/two_robot_cloth_handover.py` | Передача и растяжение ткани двумя роботами | `python scripts/two_robot_cloth_handover.py` |
| `scripts/pick_water_cup_place_and_fall.py` | Робот Franka поднимает и наклоняет чашку с SPH-жидкостью, затем отпускает её | `python scripts/pick_water_cup_place_and_fall.py` |

Записи результатов:

- `videos/materials_on_slope.mp4`
- `videos/mouse_control_with_cloth.mp4`
- `videos/two_robots_cloth_handover.mp4`
- `videos/pick_water_cup_place_and_fall.mp4`

Для визуализации чашки скрипт использует `scripts/water_cup.obj`. Этот OBJ-меш нужен только для отображения: физическая форма чашки задана в скрипте простыми коллизионными элементами.

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
python scripts/pick_water_cup_place_and_fall.py --cpu
```

В сценарии с чашкой также доступен флаг `--headless` для запуска без окна просмотра; он не меняет backend. Чтобы увидеть параметры конкретного сценария, добавьте `--help`. Сценарии с мышью, роботами и чашкой по умолчанию выбирают GPU; на системах без поддерживаемого GPU укажите `--cpu`.

## Структура

```text
.
├── scripts/    # эксперименты, проверка окружения и water_cup.obj
├── videos/     # видео готовых запусков
├── README.md
└── pyproject.toml
```

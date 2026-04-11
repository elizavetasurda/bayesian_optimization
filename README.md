# Bayesian Optimization with Constraints

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Байесовская оптимизация с ограничениями, реализующая четыре метода обработки ограничений:

- **CEI** (Constrained Expected Improvement) - ожидаемое улучшение с учётом вероятности выполнения ограничений
- **Penalty** - метод штрафных функций с адаптивным коэффициентом
- **Lagrange** - метод модифицированной функции Лагранжа с обновлением множителей
- **Barrier** - метод внутренней точки с логарифмическим барьером

##  Требования

- Python 3.14 или выше
- uv (рекомендуемый менеджер пакетов)

##  Установка

### Используя uv (рекомендуется)

```bash
git clone https://github.com/elizavetasurda/bayesian_optimization.git
cd bayesian_optimization
uv venv --python 3.14
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
uv lock
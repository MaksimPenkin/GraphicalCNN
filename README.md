# GraphicalCNN

Python==3.8.7

Реализовать удобный графический пользовательский интерфейс для обучения нейронных моделей
классификации и сегментации изображений с использованием фреймворка глубокого
обучения PyTorch.

Пользователю даётся возможность задавать желаемые функции потерь, конфигурацию архитектуры
и прочие гиперпараметры требуемые для обучения свёрточных моделей.
Графический интерфейс должен предоставлять возможность контролировать ход обучения, например
через графическое отображение кривых выбранных функций потерь.

## Установка PyTorch
1. https://pytorch.org/get-started/previous-versions/
2. В проекте используется версия 1.7.1+cu110
3. Если на Вашем компьютере установлена GPU и CUDA 11.0.3_451.82_win10, то
* pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
4. Если на Вашем компьютере только CPU или не установлена CUDA, то
* pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

## Запуск приложения
1. Создание тренировочного датасета: python utility/generate_toy_dataset.py --num_images=1000 --path_out data/train
2. Создание валидационного датасета: python utility/generate_toy_dataset.py --num_images=50 --path_out data/val
3. python run.py
4. Для просмотра кривых обучения:
* cd EXPERIMENT_PATH
* tensorboard --logdir "logs" --host localhost

## Проверка на flake8 и pydocstyle
Run in CMD: flake8
Run in CMD: pydocstyle

## Создание документации
1. cd docs
2. sphinx-apidoc -o . ..
3. make html
4. Документация будет в "docs/_build/html/index.html"


## Интерфейсная модель
![alt text](imgs/img1.png)
![alt text](imgs/img2.png)
![alt text](imgs/img3.png)
![alt text](imgs/img4.png)

## Программные возможности
* выбор базового блока UNet архитектуры: 'ResBlock', 'ConvBlock';
* выбор целевой функции потерь, например: 'BCE', L2';
* выбор метода оптимизации модели, например; ADAM, SGD;
* число эпох обучения, размер подаваемого в сеть пакета изображений (англ. batch size)
и прочие гиперпараметры для обучения свёрточных моделей.

## Фреймворки
* Графический интерфейс: TkInter;
* Обучение глубоких нейронных сетей: PyTorch;
* Работа с изображениями: OpenCV, NumPy;
* Анализаторы качества кода: flake8, pydocstyle;
* Создание документации: sphinx.

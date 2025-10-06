
##### RU

#### Главное назначение
Данная программа создана для решения лабораторной работы 1 из курса "Соверменные нейросетевые технологии". Необходимо было написать собственный прецептрон без помощи специальных пакетов для создания нейросетей. Для решения поставленной задачи, были использованы следующие пакеты Python:
- `kagglehub` - для скачивания датасета, который будет содержать в себе тестовые и тренировочные случаи;
- `os` - для работы с директориями и файлами;
- `random` - рандомизация;
- `enum` - для создания удобных перечислений в коде;
- `matplotlib` - преобразование картинки в массив чисел;
- `numpy` - использование массивов numpy и сохранение/экспорт массивов в файл/из файла.
 
#### Как пользоваться
Скачиваете все необходимые пакеты для Python и запускаете `main.py`, выставив необходимые настройки в `utility.py`. Да, нет никакого дружелюбного интерфейса, но задача была не в этом, уж простите. Модель может получить веса с файла или научится на тренировочной выборке. Можно потестировать на тестовой выборке или нарисовать файлик размером 224х224 (да, разрешение "забито гвоздями") и скормить коду. Удачи!

##### EN

#### Primary Purpose 
This program was developed to complete Lab Assignment 1 from the course "Modern Neural Network Technologies". The task required implementing a custom perceptron without using specialized neural network libraries. To accomplish this, the following Python packages were utilized:   
- `kagglehub` – for downloading the dataset containing training and test samples;  
- `os` – for handling directories and files;  
- `random` – for randomization;  
- `enum` – for creating convenient enumerations in the code;  
- `matplotlib` – for converting images into numerical arrays;  
- `numpy` – for working with NumPy arrays and saving/loading arrays to/from files.
     

#### How to Use 
Install all required Python packages and run `main.py`, adjusting the necessary settings in `utility.py`. Yes, there's no user-friendly interface (sorry about that), but that wasn't the goal of this assignment. The model can either load pre-trained weights from a file or train from scratch on the provided training set. You can evaluate its performance on the test set or create your own 224×224 image file (yes, the resolution is hardcoded) and feed it to the program. Good luck! 
 

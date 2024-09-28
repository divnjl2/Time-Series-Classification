import numpy as np
import pandas as pd

# Параметры синусоиды
amplitude = 50
offset = 150  # Сдвиг для получения значений от 100 до 200
num_periods = 5
period_length = 100  # Длина периода в секундах
sampling_interval = 1  # Интервал дискретизации в секундах
total_time = num_periods * period_length  # Общее время

# Создаем временной вектор
t = np.arange(0, total_time, sampling_interval)

# Генерируем синусоиду
Price = offset + amplitude * np.sin(2 * np.pi * t / period_length)

# Вычисляем первую производную (численно)
First_Derivative = np.gradient(Price, sampling_interval)

# Вычисляем вторую производную (численно)
Second_Derivative = np.gradient(First_Derivative, sampling_interval)

# Инициализируем метки классов
Label = np.zeros_like(Price, dtype=int)

# Функция для определения метки для каждой точки
def assign_label(index, Price):
    current_price = Price[index]
    idx = index
    reached_plus90 = False
    reached_minus90 = False
    while idx < len(Price):
        price_diff = Price[idx] - current_price
        if price_diff >= 90:
            reached_plus90 = True
            time_to_plus90 = idx - index
            break
        elif price_diff <= -90:
            reached_minus90 = True
            time_to_minus90 = idx - index
            break
        idx += 1
    if reached_plus90:
        return +1
    elif reached_minus90:
        return -1
    else:
        return 0

# Присваиваем метки классов
for i in range(len(Price)):
    Label[i] = assign_label(i, Price)

# Создаем DataFrame
df = pd.DataFrame({
    'Price': Price,
    'First_Derivative': First_Derivative,
    'Second_Derivative': Second_Derivative,
    'Label': Label
})

# Сохраняем в CSV
df.to_csv('sine_wave_dataset.csv', index=False)

# Вывод первых нескольких строк для проверки
print(df.head())

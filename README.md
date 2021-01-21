# Решение второй задачи соревнования по анализу данных Alfa Battle 2.0

https://alfabattle.ru/2

https://boosters.pro/championship/alfabattle2/rating

Результат без сеток (только LGBM) - 0.776 на привате

БОльший упор ставился на feature engineering:

1_Data_to_spark.ipynb - засовываем данные в спарк

2_Features_main.ipynb - сбор признаков, используя спарк (очень много различных признаков)

3_Features_smooth.ipynb - Кодирование таргетом на транзакциях (написано на спарке). Дополнительные признаки на основе кодирования таргетом категориальных признаков (и их комбинации) с различными финальными агрегациями (mean, max, std).

4_Pipeline_LGBM_blend_NN.ipynb - Blending LGBM моделей с разными random_state. Blending результатов LGBM с advanced baseline (RNN)
# Решение второй задачи соревнования по анализу данных Alfa Battle 2.0

https://alfabattle.ru/2

https://boosters.pro/championship/alfabattle2/rating

Результат без сеток (только LGBM) - 0.776 на привате

Результат с блендингом сеток (0.6 * LGBM + 0.4 * RNN) - 0.7812 на привате (9 место)

БОльший упор ставился на feature engineering:

1_Data_to_spark.ipynb - засовываем данные в спарк

2_Features_main.ipynb - сбор признаков, используя спарк (очень много различных признаков)

3_Features_smooth.ipynb - Кодирование таргетом на транзакциях (написано на спарке). Дополнительные признаки на основе кодирования таргетом категориальных признаков (и их комбинации) с различными финальными агрегациями (mean, max, std).

4_1_Fasttext_mcc.ipynb - Фичи (на основе стекинга) supervised fasttext на mcc

4_2_Fasttext_card_type.ipynb - Фичи (на основе стекинга) supervised fasttext на card_type

5_Pipeline_LGBM_blend_NN.ipynb - Blending LGBM моделей с разными random_state. Blending результатов LGBM с advanced baseline (RNN)
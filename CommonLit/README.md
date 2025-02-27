## CommonLit - Evaluate Student Summaries
##### Решение (600е место из 2100) соревнования от образовательной организации Commonlit: https://kaggle.com/competitions/commonlit-evaluate-student-summaries/overview

## 1. [Цель](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview)
> Оценить качество пересказов (summaries), написанных учащимися. Задача соревнования состоит в том, чтобы построить модель, позволяющую оценить, насколько хорошо учащиеся объясняют основную идею и детали исходного текста, а также ясность (clarity), точность(precision) и беглость(fluency) языка.  
Таким образом, модель должна предсказывать два числовых параметра резюме: содержание (content) (идея, детали источника) и формулировки (wording ) (ясность, точность и беглость языка), опираясь на коллекцию реальных резюме учащихся для обучения модели. 

## 2.	Основные действия

> Финальная мета модель обучена над сгенерированными признаками - предсказаниями трансформеров (Bertav3Base, Bertav3Large) и кастомными фичами: (кол-во ошибок в предложений, пересечени n грамм токеном между промптом и эссе, кол-во предложений, начинающихся с большой буквы, кол-во слов/ токенов/ стоп слов и так далее). В качестве мета модели выбрана LightGBM.

> ### Предсказания трансформеров
> Основные фичи мета модели – предсказания трансформеров (Bertav3Base, Bertav3Large). Для получения предсказаний требуется адаптированная архитектура Bert. Чтобы адаптировать и расширить архитектуру Bert, необходимо использовать кастомную архитектуру модели над трансформерами, свой собственный пулинг и код обучения/инференса (модель определяется через класс cfg, все необходимые параметры: путь до весов, тип пулинга, длину последовательность и так далее). За основу взят ноутбук [[1](https://www.kaggle.com/code/bulivington/transformers-predictions-base)]. В коде обучается трансформер Bertav3Base и сформированы out of fold предсказания для теста. 
Трансформер Bertav3Large уже обучен в коде [[2](https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3)]
В коде определены функции для инференса теста уже обученной Bertav3Large.

> ### Кастомные фичи
> Для данного соревнования можно придумать множество фичей и без обучения трансформеров: статистики по кол-ву слов/токенов, уровень сложности читаемости текста, сравнение промпта и эссе, кол-во ошибок в предложений, пересечени н грамм токеном между промптом и эссе, кол-во предложений начинающихся с большой буквы, кол-во стоп слов и так далее.
 С помощью датасета CommonLit Texts (c фичами title, author, description, grade, genre, lexile, is_prose) добавлены новые фичи ('author', 'description', 'grade', 'genre', 'lexile', 'is_prose', 'author_type', 'author_frequency' и тд).
Проведен Feature Engineering с применением spell autocorrect. На основании ноутбука [[3](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect)] получены признаки, позволяющие понять структуру и связность текста: (prompt/summary_length, length_ratio, splling_err_num, word_overlap_count, comma_count, avg_word_length, bigram/trigram_overlap_count ets)

> ### Мета модель LightGBM
> Далее для обучения финальной модели - объединяем стэк предсказания трансформеров и сгенерированные признаки. Финальная мета модель над всеми сгенерированными признаками - LightGBM с гиперпараметрами, подобранными с помощью Optuna.

## 3. Личный объем участия
> 1.	Доработала кастомную архитектуру модели Berta, добавив:
> - варианты пулинга для выхода трансформеров (Mean, Max, Cls, MeanMax, GemText)
> - варианты предварительной обработки текста перед токенизацией (комбинации признаков)
 
> 2.	Обучала кастомные модели с различными параметрами над трансформерами Berta (Roberta, Deberta base, small, large) для выбора лучших
> 3.	Обучила мета модель LightGBM над предсказаниями Bertav3Base, Bertav3Large и кастомными фичами с подборанными с помощью Optuna гипепараметрами

## 4. Ссылки

1.	[https://www.kaggle.com/code/bulivington/transformers-predictions-base](https://www.kaggle.com/code/bulivington/transformers-predictions-base)
2.	[https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3](https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3)
3.	[https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect)




# ISIC 2024 - Skin Cancer Detection with 3D-TBP

**Language**: [🇺🇸 English](#english) | [🇷🇺 Русский](#русский)

---

## English

### Overview

The project develops AI algorithms for analyzing 3D Total Body Photography (TBP) skin lesion images to identify histologically confirmed skin cancer cases. The binary classification algorithm distinguishes malignant from benign skin lesions using lower quality images (similar to smartphone captures). The development aims to improve early skin cancer diagnosis and assist in conditions with limited access to specialized medical equipment, such as dermatoscopes.

- **Main objective**: Create binary classification to improve patient triage efficiency
- **Goal**: Distinguish benign cases from malignant ones. Each image (isic_id) is assigned a probability (target) in range [0, 1] indicating malignancy likelihood
- **Evaluation Metric**: Partial Area Under ROC Curve (pAUC) above 80% True Positive Rate (TPR) for binary classification of malignant examples. Scores range within [0.0, 0.2]

### Dataset Description

**Data Format**

The dataset consists of diagnostically labeled images (JPEG) with additional metadata. The .csv file contains:
- Binary diagnostic labels (target)
- Potential input variables (e.g., age_approx, sex, anatom_site_general, etc.)
- Additional attributes (e.g., image source and precise diagnosis)

The dataset contains all lesions from a subset of thousands of patients observed from 2015 to 2024 across nine institutions on three continents.

To simulate non-dermoscopic images, standardized cropped lesion images obtained through **3D Total Body Photography (TBP)** (Vectra WB360) are used. After obtaining the total body image (TBP), AI-based software identifies individual lesion foci on the given 3D image.

### Solution Approach

The code represents an **ensemble of three boosting algorithms** (LightGBM, CatBoost, XGBoost) with data sampling and feature engineering for malignant skin lesion classification, achieving **partial AUC >80% TPR**.

### Main Code Components (isic_2024_final_blending.ipynb)

1. **Data Processing**
2. **Image Probability Predictions** - Using fine-tuned EfficientNet models to extract malignancy probabilities
3. **Feature Engineering** - Created 50+ new features
4. **Patient-level Normalization** (critical as one patient may have multiple lesions)
5. **Two-stage Sampling** (Oversampling, Undersampling) for class imbalance handling
6. **Ensemble of Three Boosting Models** (LightGBM, CatBoost, XGBoost)

### Key Solution Features

1. **Data Processing**: 
   - Numerical features like `age_approx` are converted to appropriate data types
   - Missing values filled with median

2. **New Feature Categories**:
   - **Geometric lesion characteristics**
   - **Color characteristics**
   - **3D spatial features**
   - **Composite indices**

3. **Sampling Strategy**:
   - **Oversampling**: Increases malignant cases proportion to 0.3%
   - **Undersampling**: Reduces overall size to 1% of original

4. **Validation Strategy**: 
   - `StratifiedGroupKFold (n_splits=5)` for data splitting considering groups (`patient_id`) to prevent data leakage between folds
   - Stratification by target to account for class imbalance

5. **Ensemble**: 
   - Predictions from LightGBM, XGBoost and CatBoost combined via `VotingClassifier`
   - Boosting hyperparameters optimized with Optuna
   - Least significant features removed before predictions (determined in `feature_importance.ipynb` notebook for 13 new features)

### Repository Scripts

#### 1. feature_importance.ipynb

Optuna optimization script for three gradient boosting algorithms with feature importance analysis.
Adds **13 new medically significant features** to the original set.

**XGBoost Optimization Features**:
- **Dual regularization**: L1 (alpha) and L2 (lambda) simultaneously
- **Multiple column sampling**: `colsample_bytree` + `colsample_bynode` for flexible control
- **Histogram method**: `tree_method='hist'` - fast algorithm for large datasets
- **Categorical support**: automatic via OHE preprocessing

**CatBoost Optimization Features**:
- **Categorical handling**: `cat_features=cat_cols` - native categorical feature processing
- **Bootstrap strategy**: 'Bernoulli' - CatBoost-specific sampling method
- **Regularization**: `l2_leaf_reg` instead of `lambda_l2` (CatBoost terminology)
- **Column sampling**: `colsample_bylevel` - tree level sampling
- **Range tuning**: conservative ranges (4-8 depth vs 15-250 leaves in LGB)

**LightGBM Optimization Features**:
- **Tree Parzen Estimator (TPE)**: adaptive hyperparameter sampling
- **Log-scale sampling**: for regularization parameters (avoids local minima)
- **Class balancing**: automatic weight selection for class imbalance

#### 2. pl_submission_(efficientnet_b0).ipynb

Executes probability predictions recorded in `target_3` feature using pre-trained model ensemble.

1. **Model Loading**: Multiple EfficientNet-B0 checkpoints from `/kaggle/input/isic-2024-fails/epoch-2*` (trained on different data folds)

2. **Binary Classification**: Each model adapted for skin lesion classification (0=benign, 1=malignant)

3. **Advanced Augmentations** from SIIM-ISIC Melanoma Classification winners:
   - **Geometric transformations**: Transpose, VerticalFlip, HorizontalFlip, ShiftScaleRotate, CoarseDropout
   - **Color changes**: ColorJitter, HueSaturationValue, CLAHE
   - **Blurs**: MotionBlur, MedianBlur, GaussianBlur
   - **Noise**: GaussNoise
   - **Distortions**: OpticalDistortion, GridDistortion, ElasticTransform
   - **Special**: Microscope augmentation

4. **GeM (Generalized Mean) Pooling**

5. **Higher quality through averaging, more stable predictions**

#### 3. main.ipynb and isic_pytorch_training_baseline_image_only.ipynb

`main.ipynb` executes probability predictions recorded in `target_effnetv1b0` feature (training code - `isic_pytorch_training_baseline_image_only.ipynb`).

Model optimized for low-quality medical images (simulating smartphone captures) targeting AUROC metric maximization for reliable malignant lesion classification.

In the modified EfficientNet architecture (`ISICModel` class), the standard multi-class classifier is replaced with a binary classifier with probabilistic output (skin lesion malignancy probability: 0=benign, 1=malignant).

**Solution Features**:
1. **Class imbalance handling** through specialized Dataset and sampling
2. **Advanced architecture** with improved pooling (GeM) for better feature extraction
3. **Strong augmentations** simulating real smartphone capture defects for model robustness
4. **Pre-trained EfficientNet** with Noisy Student

---

## Русский

### Обзор

Требуется разработать алгоритмы ИИ для анализа 3D-фотографий кожных образований всего тела (TBP) с целью выявления гистологически подтвержденных случаев рака кожи. Алгоритм бинарной классификации должен отличать злокачественные поражения кожи от доброкачественных, используя изображения более низкого качества (аналогичных снимкам со смартфона). Разработки направлены на улучшение ранней диагностики рака кожи и помощь в условиях ограниченного доступа к специализированной медицинской помощи, например к дерматоскопу.

- **Основная задача**: создание бинарной классификации для повышения эффективности сортировки пациентов
- **Цель**: Отличить доброкачественные случаи от злокачественных. Для каждого изображения (isic_id) назначается вероятность (target) в диапазоне [0, 1], что случай является злокачественным
- **Метрика оценки**: Частичная площадь под ROC-кривой (pAUC), превышающая 80% истинно положительных результатов (TPR) для бинарной классификации злокачественных примеров. Таким образом, оценки варьируются в пределах [0.0; 0.2]

### Описание набора данных

**Формат данных**

Набор данных состоит из диагностически маркированных изображений (JPEG) с дополнительными метаданными. Файл .csv содержит:
- Бинарную диагностическую метку (target)
- Потенциальные входные переменные (например, age_approx, sex, anatom_site_general и т. д.)
- Дополнительные атрибуты (например, image source и precise diagnosis)

Набор данных содержит все поражения из подмножества тысяч пациентов, наблюдавшихся в период с 2015 по 2024 год в девяти учреждениях на трех континентах.

Для имитации недермоскопических изображений используются стандартизированные обрезанные изображения поражений, полученные с помощью **3D-фотографии всего тела (TBP)** (Vectra WB360). После получения изображения всего тела (TBP) программное обеспечение на основе ИИ идентифицирует отдельные очаги поражения на данном 3D-снимке.

### Подход к решению

Код представляет **ансамбль из трех бустингов** (LightGBM, CatBoost, XGBoost) с сэмплингом данных и feature engineering для классификации злокачественных поражений кожи, достигая **частичной AUC >80% TPR**.

### Из чего состоит основной код (isic_2024_final_blending.ipynb)

1. **Обработка данных**
2. **Получение предсказаний вероятностей злокачественности изображений** с помощью дообученных моделей типа EfficientNet
3. **Feature Engineering** – создано более 50 новых признаков
4. **Нормализация по пациентам** (это критически важно, так как у одного пациента может быть несколько поражений)
5. **Двухэтапный сэмплинг** (Oversampling, Undersampling) для обработки дисбаланса классов
6. **Ансамбль трех бустингов** (LightGBM, CatBoost, XGBoost)

### Ключевые особенности решения

1. **Обработка данных**: 
   - Числовые признаки, такие как `age_approx`, преобразуются в нужный тип данных
   - Пропущенные значения заполняются медианой

2. **Категории созданных новых признаков**:
   - **Геометрические характеристики поражения**
   - **Цветовые характеристики**
   - **3D пространственные признаки**
   - **Составные индексы**

3. **Стратегия сэмплинга**:
   - **Oversampling**: увеличивает долю злокачественных случаев до 0.3%
   - **Undersampling**: снижает общий размер до 1% от исходного

4. **Стратегия валидации**: 
   - Используется `StratifiedGroupKFold (n_splits=5)` для разделения данных с учетом групп (`patient_id`), что предотвращает утечку данных между фолдами
   - Стратификация по target, чтобы учесть несбалансированность классов

5. **Ансамблирование**: 
   - Предсказания от LightGBM, XGBoost и CatBoost объединяются через `VotingClassifier`
   - Гиперпараметры бустингов оптимизированы Optuna
   - Наиболее незначительные фичи удалены перед предсказаниями (незначительные фичи определяются в ноутбуке `feature_importance.ipynb` для 13 новых признаков)

### Скрипты в репозитории

#### 1. feature_importance.ipynb

Скрипт Optuna-оптимизации трех градиентных бустингов с feature importance.
Добавляет **13 новых медицински значимых признаков** к исходному набору.

**Особенности XGBoost optimization**:
- **Dual regularization**: и L1 (alpha) и L2 (lambda) одновременно
- **Multiple column sampling**: `colsample_bytree` + `colsample_bynode` для более гибкого контроля
- **Histogram method**: `tree_method='hist'` - быстрый алгоритм для больших датасетов
- **Categorical support**: автоматически через OHE preprocessing

**Особенности CatBoost optimization**:
- **Categorical handling**: `cat_features=cat_cols` - нативная обработка категориальных признаков
- **Bootstrap strategy**: 'Bernoulli' - специфичный для CatBoost метод сэмплинга
- **Regularization**: `l2_leaf_reg` вместо `lambda_l2` (CatBoost terminology)
- **Column sampling**: `colsample_bylevel` - сэмплинг по уровням дерева
- **Range tuning**: более консервативные диапазоны (4-8 глубина vs 15-250 листьев в LGB)

**Особенности LightGBM optimization**:
- **Tree Parzen Estimator (TPE)**: адаптивный сэмплинг гиперпараметров
- **Log-scale sampling**: для параметров регуляризации (избегает локальных минимумов)
- **Class balancing**: автоматический подбор весов для дисбаланса классов

#### 2. pl_submission_(efficientnet_b0).ipynb

Выполняет предсказания вероятностей, записанных в фичу `target_3`, используя ансамбль предобученных моделей.

1. **Загрузка моделей**: Несколько чекпоинтов (EfficientNet-B0) из директории `/kaggle/input/isic-2024-fails/epoch-2*` (обучены на разных фолдах данных)

2. **Бинарная классификация**: Каждая модель адаптирована для бинарной классификации поражений кожи (0 - доброкачественное, 1 - злокачественное)

3. **Продвинутые аугментации** из решения победителей SIIM-ISIC Melanoma Classification:
   - **Геометрические трансформации**: Transpose, VerticalFlip, HorizontalFlip, ShiftScaleRotate, CoarseDropout
   - **Цветовые изменения**: ColorJitter, HueSaturationValue, CLAHE
   - **Размытия**: MotionBlur, MedianBlur, GaussianBlur
   - **Шум**: GaussNoise
   - **Искажения**: OpticalDistortion, GridDistortion, ElasticTransform
   - **Специальная**: Microscope аугментация

4. **GeM (Generalized Mean) пулинг**

5. **Выше качество за счет усреднения, более стабильные предсказания**

#### 3. main.ipynb и isic_pytorch_training_baseline_image_only.ipynb

`main.ipynb` выполняет предсказания вероятностей, записанных в фичу `target_effnetv1b0` (код обучения - `isic_pytorch_training_baseline_image_only.ipynb`).

Модель оптимизирована для работы с медицинскими изображениями низкого качества (симулирующими снимки со смартфона) и нацелена на максимизацию AUROC метрики для надежной классификации злокачественных поражений кожи.

В модифицированной архитектуре EfficientNet (class `ISICModel`) стандартный многоклассовый классификатор заменяется на бинарный классификатор с вероятностным выходом (что поражение кожи является злокачественным, где 0 = доброкачественное, 1 = злокачественное).

**Особенности решения**:
1. **Борьба с дисбалансом классов** через специальный Dataset и сэмплирование
2. **Более сложная архитектура** за счет продвинутого pooling (GeM) для лучшего извлечения признаков
3. **Сильные аугментации**, имитирующие реальные дефекты смартфонной съемки, делая модель устойчивой к ним
4. **Предобученная модель EfficientNet** с Noisy Student

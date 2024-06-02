## Open Problems – Single-Cell Perturbations
##### Это решение (заняло 43е место из 1098, серебряная медаль) соревнования: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations

## 1. Формализация [проекта] (https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview)
> Цель: моделировать, как малые молекулы лекарств изменяют дифференциальную экспрессию генов (differential expression (DE)), в различных типах клеток. Модель должна оценивать влияние экспериментального возмущения (chemical perturbations) лекарств на клетки, и, соответственно, уровень экспрессии каждого из 18211 генов в транскрипции.

## 2.	Основные действия

> Финальная  модель представляет бленд двух моделей:
> 1. weight_1: 0.5 Pyboost model
> 2. weight_2: 0.5 NN model

> ### Предсказания трансформеров
> 1. Pyboost features
Our main improvement from public pyboost implementation was that we explored and added as a feature categories for each drug:
> rug_cls = {
"Antifungal": ["Clotrimazole", "Ketoconazole"],
"Corticosteroid": ["Mometasone Furoate"],
"Kinase Inhibitors": ["Idelalisib", "Vandetanib", "Bosutinib", "Ceritinib", "Crizotinib",
"Cabozantinib", "Dasatinib", "Selumetinib", "Trametinib", "Lapatinib",
"Canertinib", "Palbociclib", "Dabrafenib", "Ricolinostat","Tamatinib", "Tivozanib",
"Quizartinib","Sunitinib","Foretinib","Imatinib","R428","BMS-387032","CGP 60474",
"TIE2 Kinase Inhibitor","Masitinib","Saracatinib","CC-401","RN-486","GO-6976",
"HMN-214","BMS-777607","Tivantinib","CEP-37440","TPCA-1","AZ628","PF-03814735",
"PRT-062607","AT 7867", "BI-D1870", "Mubritinib", "GLPG0634","Ruxolitinib", "ABT-199 (GDC-0199)",
"Nilotinib"],
"Antiviral": ["Lamivudine", "AMD-070 (hydrochloride)", "BMS-265246"],
"Sunscreen agent" : ["Oxybenzone"],
"Antineoplastic": ["Vorinostat", "Flutamide", "Ixabepilone", "Topotecan", "CEP-18770 (Delanzomib)",
"Resminostat", "Decitabine", "MGCD-265", "GSK-1070916","BAY 61-3606","Navitoclax", "Porcn Inhibitor III","GW843682X","Prednisolone","Tosedostat",
"Scriptaid", "AZD-8330", "Belinostat","BMS-536924","Pomalidomide","Methotrexate","HYDROXYUREA",
"PD-0325901","SB525334","AVL-292","AZD4547","OSI-930","AZD3514","MLN 2238","Dovitinib","K-02288",
"Midostaurin","I-BET151","FK 866","Tipifarnib","BX 912","SCH-58261","BAY 87-2243",
"YK 4-279","Ganetespib (STA-9090)","Oprozomib (ONX 0912)","AT13387","Tipifarnib","Flutamide","Perhexiline","Sgc-cbp30","IMD-0354",
"IKK Inhibitor VII", "UNII-BXU45ZH6LI","ABT737","Dactolisib", "CGM-097", "TGX 221","Azacitidine","Defactinib",
"PF-04691502", "5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine"],
"Selective Estrogen Receptor Modulator (SERM)": ["Raloxifene"],
"Antidiabetic (DPP-4 Inhibitor)": ["Linagliptin","Alogliptin"],
"Antidepressant": ["Buspirone", "Clomipramine", "Protriptyline", "Nefazodone","RG7090"],
"Antibiotic": ["Isoniazid","Doxorubicin"],
"Antipsychotic": ["Penfluridol"],
"Antiarrhythmic": ["Amiodarone","Proscillaridin A"],
"Alkaloid": ["Colchicine"],
"Antiviral (HIV)": ["Tenofovir","Efavirenz"],
"Allergy": ["Desloratadine","Chlorpheniramine","Clemastine","GSK256066","SLx-2119", "TR-14035", "Tacrolimus"],
"Anticoagulant": ["Rivaroxaban"],
"Alcohol deterrent":["Disulfiram"],
"Cocaine addiction":["Vanoxerine"],
"Erectile dysfunction":["Vardenafil"],
"Calcium channel blocker":["TL_HRAS26"],
"Anti-endotoxemic":["CGP 60474"],
"Acne treatment":["O-Demethylated Adapalene"],
"Stroke":["Pitavastatin Calcium","Atorvastatin"],
"Stem cell work":["CHIR-99021"],
"Hypertension":["Riociguat"],
"Heart failure":["Proscillaridin A;Proscillaridin-A", "Colforsin"],
"Regenerative":["LDN 193189"],
"Psoriasis":["Tacalcitol"],
"Unknown_1": ["STK219801"],
"Unknown_2": ["IN1451"]
Another features were 'cell_type' and 'sm_name' encoded with QuantileEncoder(quantile =.8)
TruncatedSVD(n_components=50) was applied to target.

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

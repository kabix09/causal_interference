# 📊 Model Selection Context — Dobór Modeli ML dla Agenta CLI

> **Cel tego pliku:** Szczegółowy przewodnik dla agenta CLI do automatycznego doboru modeli ML/regresji na podstawie charakterystyki zbioru danych. Agent powinien analizować dane wejściowe i dobierać metodę zgodnie z poniższymi heurystykami i wymaganiami.

---

## 🧭 Jak używać tego pliku (dla agenta)

1. **Zidentyfikuj typy zmiennych** w zbiorze danych → Sekcja I
2. **Sprawdź charakterystykę rozkładów** → Sekcja II
3. **Oceń cel modelowania** (klasyfikacja / regresja / causal inference) → Sekcja III
4. **Wybierz model** zgodnie z macierzą decyzyjną → Sekcja IV
5. **Zweryfikuj założenia modelu** wobec danych → Sekcja V
6. **Dobierz metrykę ewaluacyjną** → Sekcja VI
7. **Sprawdź czerwone flagi** przed finalnym wyborem → Sekcja VII

---

## I. Taksonomia zmiennych i ich wpływ na dobór modelu

### 1.1 Zmienne binarne (0/1, tak/nie)

**Charakterystyka:**
- Przyjmują dokładnie 2 wartości
- Np. `housing`, `loan`, `default`, `deposit`, `has_email`

**Wpływ na modele:**
- Modele liniowe (OLS, logit) **zakładają addytywny, liniowy wpływ** każdej binarnej zmiennej — co jest rzadko prawdą
- Interakcje między zmiennymi binarnymi (np. `loan=1 AND housing=1`) są niewidoczne dla regresji bez ręcznego tworzenia interaction terms
- Drzewa decyzyjne i boosting **naturalnie wykrywają interakcje** bez dodatkowego feature engineeringu

**Kiedy binarnych zmiennych jest dużo (≥ 5 binarnych cech):**
- Unikaj czystej regresji liniowej / logistycznej bez interaction terms
- Preferuj: Decision Tree, Random Forest, XGBoost, LightGBM
- Alternatywnie: logit + ręczne interakcje (ale kosztowne i podatne na błędy)

**Sygnał dla agenta:**
```
if count(binary_features) >= 5:
    deprioritize: [LinearRegression, LogisticRegression]
    prefer: [DecisionTree, RandomForest, XGBoost, LightGBM]
```

---

### 1.2 Zmienne kategoryczne (nominalne, wielowartościowe)

**Charakterystyka:**
- Wiele wartości bez porządku (np. `job`, `education`, `contact`, `month`, `marital`)
- Wysoka kardynalność = wiele unikalnych wartości (np. `zip_code` z tysiącami wartości)

**Strategie kodowania i ich wpływ:**

| Metoda kodowania | Kiedy używać | Problemy |
|---|---|---|
| One-Hot Encoding (OHE) | niska kardynalność (< 15 kategorii), modele liniowe | curse of dimensionality przy wysokiej kardynalności |
| Label Encoding | drzewa decyzyjne, boosting | NIE używać z modelami liniowymi (zakłada porządek) |
| Target Encoding | wysoka kardynalność + drzewa/boosting | ryzyko data leakage — zawsze używać z cross-val |
| Embedding (np. Entity Embedding) | sieci neuronowe, bardzo wysoka kardynalność | wymaga dużo danych |
| Frequency Encoding | gdy liczba wystąpień kategorii ma znaczenie | traci informację o tożsamości kategorii |

**Sygnał dla agenta:**
```
if any(feature.cardinality > 15 for feature in categorical_features):
    warn: "Wysoka kardynalność — rozważ Target Encoding lub Embedding zamiast OHE"
    avoid: OHE for these features
if using LinearRegression or LogisticRegression:
    require: OHE (never Label Encoding)
if using DecisionTree or XGBoost:
    prefer: Label Encoding or Target Encoding
```

---

### 1.3 Zmienne ciągłe (numeryczne)

**Charakterystyka:**
- Przyjmują wartości rzeczywiste (np. `age`, `balance`, `campaign`, `pdays`, `duration`)

**Kluczowe własności do sprawdzenia:**

| Własność | Test | Implikacja dla modelu |
|---|---|---|
| Liniowość z targetem | Scatter plot, Pearson r | Tylko modele liniowe jeśli r jest wysoki i monotoniczny |
| Nieliniowość | LOWESS, Spearman vs Pearson | Potrzebne drzewa / GAM / transformacja |
| Wartości zerowe / ujemne | min(feature) | Wyklucza transformację log, wpływa na normalizację |
| Outliers (wartości odstające) | IQR, z-score | Modele liniowe i SVM wrażliwe; drzewa odporne |
| Rozkład silnie skośny | skewness > 1 lub < -1 | Rozważ log(x+1) lub sqrt(x) przed liniową regresją |
| Efekty progowe (threshold) | np. age > 60 zmienia zachowanie | Drzewa automatycznie wykrywają; OLS nie |
| Efekty saturacji | np. więcej kampanii nie pomaga | Log-transformacja lub GAM |
| Bimodalne/multimodalne | histogram | Może sugerować ukryte podgrupy — rozważ clustering |

**Sygnał dla agenta:**
```
for each continuous_feature:
    if skewness(feature) > 2: suggest log1p transform for linear models
    if has_threshold_effect(feature): prefer tree-based models
    if correlation(feature, target) < 0.05 AND nonlinear_test = significant:
        warn: "Zmienna wydaje się nieliniowo powiązana z targetem"
        suggest: polynomial features or tree-based model
```

---

### 1.4 Zmienne porządkowe (ordinalne)

**Charakterystyka:**
- Kategorie z naturalnym porządkiem: `low < medium < high`, `young < middle < senior`

**Kodowanie:**
- Label Encoding **jest tu właściwe** (w przeciwieństwie do nominalnych)
- Modele liniowe mogą działać jeśli odstępy między kategoriami są równe
- Drzewa i boosting działają dobrze bez dodatkowych założeń

---

### 1.5 Zmienne czasowe / sekwencyjne

**Charakterystyka:**
- Daty, godziny, identyfikatory sesji, kolejność zdarzeń

**Implikacje:**
- **Nigdy nie używaj Random K-Fold CV** — powoduje data leakage
- Zawsze używaj **TimeSeriesSplit** lub walk-forward validation
- Opóźnienia (lags), rolling averages mogą być wartościowymi feature'ami
- Modele: XGBoost z lag features, LSTM, Prophet, ARIMA (zależy od celu)

---

### 1.6 Zmienne rzadkie (sparse features)

**Charakterystyka:**
- Większość wartości = 0 (np. TF-IDF, one-hot dla wysokiej kardynalności)

**Implikacje:**
- Przechowuj jako sparse matrix (scipy.sparse)
- Random Forest: słabo radzi sobie ze sparse data → preferuj XGBoost, LightGBM, Linear SVM
- L1 regularization (Lasso, SVM z L1) naturalnie zeruje nieistotne współczynniki

---

## II. Charakterystyka zbioru danych — co sprawdzić przed wyborem modelu

### 2.1 Checklist diagnostyczny

```
□ Ile jest rekordów? (< 1000 / 1k–100k / > 100k)
□ Ile jest zmiennych? (< 10 / 10–100 / > 100)
□ Jaka jest proporcja klas (dla klasyfikacji)?
□ Czy zmienne są liniowo powiązane z targetem?
□ Czy są silne interakcje między zmiennymi?
□ Czy są outliers w zmiennych ciągłych?
□ Czy są zmienne z dużą liczbą brakujących wartości (> 20%)?
□ Czy zmienne kategoryczne mają wysoką kardynalność?
□ Czy dane są balansowane / nierównoważne klas?
□ Czy potrzebna jest interpretowalność (np. regulacje, causal inference)?
```

### 2.2 Rozmiar danych a wybór modelu

| Rozmiar zbioru | Preferowane modele | Czego unikać |
|---|---|---|
| < 500 rekordów | LogisticRegression, LinearSVM, Ridge/Lasso, prostsze DT | Głębokie sieci, bardzo głębokie drzewa, XGBoost bez cross-val |
| 500 – 5 000 | RandomForest, XGBoost, GBM, SVM (RBF kernel), LogReg z regularyzacją | Sieci neuronowe (ryzyko overfittingu) |
| 5 000 – 100 000 | XGBoost, LightGBM, RandomForest, LogReg, LinearSVM | — |
| > 100 000 | LightGBM, XGBoost, sieci neuronowe, SGD-based modele | RandomForest (wolny na bardzo dużych zbiorach) |
| > 1 000 000 | LightGBM, SGD Logistic, sieci neuronowe | XGBoost (pamięć!), RandomForest |

### 2.3 Nierównowaga klas (class imbalance)

**Kiedy jest problemem:** proporcja klas < 1:5 (np. 90%/10%) lub bardziej

**Techniki mitygacji:**

| Technika | Kiedy | Jak |
|---|---|---|
| class_weight='balanced' | zawsze jako baseline | parametr modelu |
| SMOTE | umiarkowana nierównowaga (1:10) | oversample minority class |
| ADASYN | gdy minority class jest niejednorodna | adaptywny SMOTE |
| Threshold tuning | po treningu modelu | dobierz próg probability zamiast 0.5 |
| Ensemble z BalancedBagging | silna nierównowaga | BalancedBaggingClassifier ze sklearn |

**Metryki przy nierównowadze klas:**
- **NIE używaj Accuracy** — wprowadza w błąd
- Używaj: F1-score, Precision-Recall AUC, MCC (Matthews Correlation Coefficient), ROC-AUC

---

## III. Cel modelowania i jego wpływ na dobór

### 3.1 Klasyfikacja binarna

**Cel:** przewidzieć czy zdarzenie zajdzie (0/1)

**Modele rankingowane (od najlepszego do gorszego dla złożonych danych):**
1. XGBoost / LightGBM — **najlepszy ogólny wybór** dla tabelarycznych danych z nieliniowościami
2. Random Forest — dobry baseline, odporny na outliers
3. Gradient Boosting (sklearn) — wolniejszy niż XGBoost, ale stabilny
4. Logistic Regression — **tylko gdy** dane są liniowo separowalne LUB jako baseline
5. SVM (RBF kernel) — dobry przy małych zbiorach z nieliniowościami, wolny na dużych
6. Decision Tree — interpretowalny, ale przeucza się; używaj z ograniczeniem głębokości
7. Neural Network — gdy dużo danych i zasobów obliczeniowych

**Kiedy logistic regression MOŻE działać:**
- Zmienna zależna i niezależne są liniowo separowalne w przestrzeni logit
- Wszystkie zmienne ciągłe, bez silnych interakcji
- Dobre OHE dla kategorycznych
- Dane są dobrze znormalizowane
- Sprawdzono liniowość za pomocą: `LogisticRegression` vs `DecisionTreeClassifier(max_depth=3)` porównując AUC

---

### 3.2 Regresja (zmienna ciągła)

**Modele:**

| Model | Idealne dane | Nie działa gdy |
|---|---|---|
| OLS (Linear Regression) | liniowe zależności, ciągłe cechy, brak outlierów, brak heteroskedastyczności | nieliniowość, outliers, kolinearność |
| Ridge / Lasso / ElasticNet | jak OLS + dużo skorelowanych zmiennych | silna nieliniowość |
| Polynomial Regression | umiarkowana nieliniowość, mało zmiennych | curse of dimensionality przy wielu zmiennych |
| Regresja logarytmiczna | target lub cechy mają charakter saturacji / wykładniczy wzrost | wartości ≤ 0, brak kształtu logarytmicznego |
| Decision Tree Regressor | nieliniowość, interakcje, mieszane typy | mało danych, potrzeba precyzji interpolacji |
| Random Forest Regressor | jw. + szum w danych | bardzo duże zbiory (wolny) |
| XGBoost / LightGBM | jw. + bardzo złożone relacje | mało danych bez cross-val |
| GAM (Generalized Additive Models) | nieliniowość cech ale NIE interakcje; interpretowalność | złożone interakcje między cechami |
| SVR (Support Vector Regression) | mały zbiór, nieliniowość (RBF kernel) | bardzo duże zbiory |

**Kiedy regresja logarytmiczna jest właściwa (ważne!):**
```
Wymagania dla modelu log-transformacji:
✅ target > 0 dla wszystkich rekordów (lub target + epsilon)
✅ target ma rozkład silnie prawostronnie skośny (skewness > 1.5)
✅ relacja między cechą a targetem jest multiplicatywna, nie addytywna
✅ efekty malejących przyrostów są widoczne na scatterplot
✅ po transformacji log(target) rozkład jest bliski normalnemu

Sprawdź: if np.log1p(target).skewness() < original_target.skewness() / 2 → log transform helps
```

---

### 3.3 Causal Inference (wnioskowanie przyczynowe)

**Cel:** zrozumieć nie tylko "co" ale "dlaczego" i "jaki byłby efekt interwencji"

**Kluczowe zasady dla agenta:**

```
CZERWONE FLAGI — nigdy nie naruszaj:
❌ NIE używaj zmiennych post-treatment jako cech (np. 'duration' w bank marketing)
❌ NIE interpretuj współczynników regresji jako efektów przyczynowych bez kontroli confounders
❌ NIE używaj feature importance z ML jako dowodu przyczynowości
```

**Podejścia do causal inference:**

| Metoda | Kiedy | Wymagania |
|---|---|---|
| Logistic Regression z kontrolą confounders | RCT lub quasi-experiment, dużo danych | liniowość, brak interakcji confounders |
| Propensity Score Matching (PSM) | dane obserwacyjne, brak RCT | dobra specyfikacja modelu propensity |
| Double ML (DoubleML) | dane obserwacyjne, złożone confounders | > 1000 rekordów, biblioteka `doubleml` |
| CausalForest (GRF) | heterogeniczne efekty leczenia, wiele podgrup | > 1000 rekordów, `econml` lub `grf` (R) |
| DiD (Difference in Differences) | dane panelowe, naturalny eksperyment | równoległe trendy przed interwencją |
| IV (Instrumental Variables) | endogenność treatment | dobry instrument (rzadki w praktyce) |

**Dla bank marketing dataset:**
- Treatment = kontakt telefoniczny (campaign)
- Outcome = `deposit`
- Confounders = `age`, `balance`, `job`, `education`, `housing`, `loan`
- Zalecane: Double ML lub CausalForest — kontrolują nieliniowe confounders

---

## IV. Macierz decyzyjna — szybki dobór modelu

```
WEJŚCIE: charakterystyka danych
WYJŚCIE: ranking modeli

Krok 1: Sprawdź rozmiar danych
  < 500 rekordów → ogranicz się do: LogReg, LinearSVM, DT(max_depth≤4), Ridge/Lasso
  500–100k → pełen wybór
  > 100k → preferuj LightGBM, SGD, sieci neuronowe

Krok 2: Sprawdź typy zmiennych
  dużo binarnych (≥5) AND dużo kategorycznych → odrzuć czyste modele liniowe
  tylko ciągłe, liniowe z targetem → OLS/LogReg może wystarczyć
  mieszane typy → boosting > drzewa > liniowe

Krok 3: Sprawdź nieliniowość
  LOWESS/scatter plot sugeruje nieliniowość → odrzuć OLS/LogReg
  efekty progowe widoczne → drzewa i boosting
  saturacja (malejące przyrosty) → log-transform OR GAM OR boosting

Krok 4: Sprawdź cel
  klasyfikacja + interpretowalność + compliance → LogReg (jeśli możliwe) lub DT z ograniczeniem głębokości
  klasyfikacja + maksymalna dokładność → XGBoost/LightGBM
  regresja + interpretacja współczynników → OLS (po weryfikacji założeń) lub Ridge/Lasso
  regresja + nieliniowość → XGBoost/RF/GAM
  causal inference → Double ML lub CausalForest (NIE zwykły ML)

Krok 5: Sprawdź nierównowagę klas
  imbalance > 1:5 → dodaj class_weight='balanced' + zmień metrykę na F1/PR-AUC
  imbalance > 1:10 → rozważ SMOTE + ensemble methods

WYNIK: lista 2-3 rekomendowanych modeli z uzasadnieniem
```

---

## V. Wymagania założeń modelu — szczegółowa weryfikacja

### 5.1 OLS (Ordinary Least Squares) — regresja liniowa

**Kompletna lista założeń do weryfikacji:**

| Założenie | Test | Jak naprawić przy naruszeniu |
|---|---|---|
| Liniowość | scatter plot, RESET test, partial regression plots | transformacja zmiennych, polynomial features |
| Normalność reszt | Q-Q plot, Shapiro-Wilk (dla małych n), Jarque-Bera | transformacja targetu, robust regression |
| Homoskedastyczność | Breusch-Pagan test, Goldfeld-Quandt | WLS (weighted least squares), log transform |
| Brak autokorelacji | Durbin-Watson test (wartość ~2 = OK) | ARIMA, GLS, dodać lag feature |
| Brak multikolinearności | VIF (Variance Inflation Factor) < 10 | Ridge/Lasso, PCA, usunąć skorelowane cechy |
| Brak wpływowych obserwacji | Cook's distance, leverage plot | usunąć outliers, robust regression |
| Egzogeniczność (causal) | wiedza domenowa | IV, Double ML |

**Kiedy OLS MOŻE działać na mieszanych danych:**
- Po kodowaniu OHE zmiennych kategorycznych
- Po usunięciu outlierów (lub robust regression)
- Po weryfikacji liniowości każdej cechy z targetem
- Przy niskim VIF (< 5 dla wszystkich cech)

---

### 5.2 Logistic Regression — szczegółowe wymagania

**Czego brakuje w podstawowych tutorialach:**

```
Pełna lista wymagań dla poprawnej regresji logistycznej:

1. LINIOWOŚĆ W PRZESTRZENI LOGIT
   - Test Box-Tidwell: dla każdej zmiennej x sprawdź czy x*log(x) jest istotne
   - Jeśli tak → zmienna ma nieliniowy efekt na log-odds → transformuj LUB użyj drzewa
   
2. BRAK COMPLETE/PERFECT SEPARATION
   - Jeśli jakaś kombinacja cech perfekcyjnie rozdziela klasy → współczynniki → ∞
   - Objaw: bardzo duże SE współczynników, ostrzeżenia modelu
   - Rozwiązanie: regularyzacja (L1/L2), więcej danych, usunięcie cechy
   
3. BRAK MULTIKOLINEARNOŚCI
   - Podobnie jak OLS: VIF < 10
   
4. WYSTARCZAJĄCA LICZBA ZDARZEŃ
   - Reguła: min 10-20 zdarzeń (klasa=1) na każdą zmienną w modelu
   - Przy 50 zmiennych: min 500-1000 przypadków klasy=1
   
5. BRAK SILNYCH INTERAKCJI
   - Regresja logistyczna nie łapie A*B interakcji bez ręcznego dodania
   - Sprawdź: czy DT(max_depth=2) znacznie bije LogReg? Jeśli tak → są interakcje
   
6. POPRAWNE KODOWANIE KATEGORYCZNYCH
   - OHE z drop_first=True (uniknij multikolinearności)
   - Nigdy Label Encoding dla nominalnych
```

**Diagnoza niskiej jakości modelu logistycznego (jak w banku: AUC=0.5, accuracy=11.4%):**
```
Krok 1: Sprawdź baseline accuracy (accuracy przy przewidywaniu majorty class)
  Jeśli model_accuracy ≈ baseline_accuracy → model nic nie nauczył
  
Krok 2: Sprawdź feature importance
  if all |coefficients| < 0.1 → cechy nie mają liniowego wpływu na log-odds
  
Krok 3: Test nieliniowości
  Porównaj LogReg(C=1) vs DecisionTree(max_depth=3)
  if AUC(DT) >> AUC(LogReg) → dane są nieliniowe → zmień model
  
Krok 4: Sprawdź kodowanie
  Czy zmienne kategoryczne są poprawnie zakodowane (OHE)?
  Czy zmienne ciągłe są znormalizowane? (StandardScaler lub MinMaxScaler)
  
Krok 5: Sprawdź czy nie ma data leakage
  Czy target zmienna nie jest wśród cech?
  Czy zmienne post-treatment są wykluczone?
```

---

### 5.3 Decision Tree — kiedy naprawdę działa

**Idealne warunki:**
- Nieliniowe zależności z efektami progowymi (age > 60, balance < 0)
- Interakcje między zmiennymi (np. student + brak kredytu = wysoka skłonność do depozytu)
- Mieszane typy zmiennych (bin + continuous + categorical razem)
- Potrzebna interpretowalność (max_depth ≤ 5)

**Wymagania do unikania overfittingu:**
```python
# Minimalne parametry kontroli overfittingu
DecisionTreeClassifier(
    max_depth=5,           # ogranicz głębokość
    min_samples_leaf=20,   # min 20 rekordów w liściu
    min_samples_split=50,  # min 50 rekordów do podziału
    class_weight='balanced' # przy nierównowadze klas
)
```

**Kiedy DT jest zbyt niestabilny:**
- Małe zmiany w danych → drastycznie inne drzewo
- Rozwiązanie: Random Forest (ensembling) lub przycinanie (pruning via ccp_alpha)

---

### 5.4 XGBoost — wymagania i pułapki

**Idealne warunki:**
- Tabelaryczne dane (structured data)
- Mieszane typy zmiennych
- Nieliniowości i interakcje
- Dane 1k–10M rekordów

**Minimalne hiperparametry do strojenia (kolejność ważności):**
```python
XGBClassifier(
    n_estimators=300,      # 1. liczba drzew (z early stopping)
    max_depth=4,           # 2. głębokość drzewa (3-6 zazwyczaj)
    learning_rate=0.05,    # 3. mniejszy = lepiej, ale wolniej
    subsample=0.8,         # 4. losowanie wierszy
    colsample_bytree=0.8,  # 5. losowanie kolumn
    min_child_weight=5,    # 6. min waga w liściu (regularyzacja)
    scale_pos_weight=N,    # 7. dla nierównowagi: N = count(neg)/count(pos)
    eval_metric='auc',     # 8. metryka (auc, logloss, etc.)
    early_stopping_rounds=50  # 9. zatrzymaj gdy brak poprawy
)
```

**Najczęstsze błędy:**
- Brak early stopping → overfitting
- max_depth zbyt duży (> 8) → overfitting
- Brak scale_pos_weight przy nierównowadze klas → model ignoruje minority class
- Brak cross-validation przy małych zbiorach

---

### 5.5 GAM (Generalized Additive Models) — niedoceniany model

**Kiedy GAM jest najlepszym wyborem:**
- Nieliniowe relacje INDYWIDUALNYCH cech z targetem
- Brak (lub słabe) interakcje między cechami
- Wymagana interpretowalność (każdy efekt cząstkowy jest widoczny na wykresie)
- Dane > 500 rekordów
- Regulowane środowisko (finanse, medycyna, prawo) gdzie "dlaczego" jest ważne

**Biblioteki:**
- Python: `pygam` (GAM z regularyzacją)
- Python: `interpret` (ExplainableBoostingMachine — GAM + interakcje)
- R: `mgcv` (najbardziej zaawansowany)

**GAM kontra inne modele:**
```
GAM: interpretowalne nieliniowe efekty GŁÓWNYCH CECH, słaby przy interakcjach
XGBoost: wszystko — nieliniowość I interakcje, ale czarna skrzynka
LogReg: tylko liniowe efekty, ale w pełni interpretowalny
```

---

### 5.6 Neural Networks — kiedy naprawdę warto

**Nie używaj sieci neuronowych gdy:**
- < 5000 rekordów tabelarycznych (boosting wygra)
- Potrzebna interpretowalność
- Krótki czas na eksperymentowanie
- Nie masz GPU

**Używaj gdy:**
- > 50 000 rekordów tabelarycznych i XGBoost nie wystarczy
- Dane obrazkowe, tekstowe, dźwiękowe
- Złożone interakcje cech wyższego rzędu
- Masz czas na hyperparameter tuning

**Dla tabelarycznych danych:** TabNet, FT-Transformer (Revisiting Deep Learning for Tabular Data) mogą bić XGBoost, ale rzadko i przy dużych zbiorach.

---

## VI. Dobór metryki ewaluacyjnej

### 6.1 Klasyfikacja binarna

**Macierz wyboru metryki:**

| Sytuacja | Metryka | Dlaczego |
|---|---|---|
| Balansowane klasy, ważne oba błędy | F1-score (macro) | harmonic mean precision/recall |
| Bardzo nierównowaga, koszt FN >> FP | Recall (Sensitivity) | np. wykrywanie chorób, fraudów |
| Bardzo nierównowaga, koszt FP >> FN | Precision | np. spam filter, reklamy |
| Ranking/scoring (nie próg decyzyjny) | ROC-AUC | area under ROC curve |
| Nierównowaga + ranking | PR-AUC (Precision-Recall AUC) | lepszy niż ROC-AUC przy silnej nierównowadze |
| Ogólna ocena przy nierównowadze | MCC (Matthews Correlation Coefficient) | uwzględnia wszystkie 4 wartości macierzy pomyłek |
| Kalibracja probabilistyczna | Brier Score, Log-Loss | gdy probability matters, nie tylko klasa |

**Dla bank marketing dataset:**
```
Nierównowaga klas (zazwyczaj ~12% yes vs ~88% no)
→ Nie używaj Accuracy (misleading)
→ Główna metryka: PR-AUC lub F1 dla klasy "yes"
→ Pomocnicza: ROC-AUC
→ Przy causal inference: skup się na Calibration (Brier Score)
```

---

### 6.2 Regresja

| Metryka | Kiedy | Wrażliwość na outliers |
|---|---|---|
| MSE (Mean Squared Error) | gdy duże błędy są nieproporcjonalnie kosztowne | wysoka |
| RMSE | jak MSE, ale w oryginalnych jednostkach | wysoka |
| MAE (Mean Absolute Error) | gdy outliers są spodziewane lub mniej ważne | niska |
| MAPE | gdy ważna jest względna wielkość błędu | bardzo wysoka (problemy przy y≈0) |
| SMAPE | jak MAPE ale symetryczna | umiarkowana |
| R² | ocena dopasowania modelu ogółem | umiarkowana |
| Adjusted R² | R² + kara za liczbę zmiennych | umiarkowana |
| Huber Loss | kompromis MSE/MAE | niska-umiarkowana |

---

### 6.3 Walidacja krzyżowa — dobór strategii

| Sytuacja | Strategia CV | Dlaczego |
|---|---|---|
| Dane standardowe (IID) | StratifiedKFold(k=5) | zachowanie proporcji klas |
| Dane czasowe | TimeSeriesSplit(n_splits=5) | brak data leakage z przyszłości |
| Małe zbiory (< 500) | LeaveOneOut lub KFold(k=10) | maksymalne użycie danych |
| Bardzo nierównowaga klas | StratifiedKFold obowiązkowo | bez tego fold może nie mieć klasy minority |
| Grupy/hierarchia (np. klienci w oddziałach) | GroupKFold | unikaj leakage między grupami |

---

## VII. Czerwone flagi — nie rób tego

### 7.1 Ogólne pułapki

```
❌ Używanie Accuracy jako metryki przy nierównowadze klas
❌ Brak walidacji krzyżowej przy małych zbiorach
❌ Feature engineering na całym zbiorze (przed train/test split) → data leakage
❌ Normalizacja/standaryzacja fit na całym zbiorze zamiast tylko train → data leakage
❌ Porównywanie modeli na zbiorze treningowym (nie testowym)
❌ Używanie zmiennych post-treatment w modelach przyczynowych
❌ Label Encoding dla nominalnych zmiennych kategorycznych w modelach liniowych
❌ Ignorowanie nierównowagi klas
❌ XGBoost bez early stopping i CV → gwarantowany overfitting
```

### 7.2 Specyficzne dla bank marketing

```
❌ 'duration' — zmienna post-treatment (długość rozmowy nie jest znana PRZED decyzją o kampanii)
❌ Używanie LogisticRegression bez testu liniowości w logit space
❌ Porównywanie modeli bez uwzględnienia class imbalance w metryce
❌ Ignorowanie 'pdays' = 999 (oznacza "nigdy nie kontaktowano") — to nie jest 999 dni, to brakująca wartość
```

### 7.3 Typowe błędy feature engineeringu

```
❌ Tworzenie ratio features bez sprawdzenia czy mianownik może być 0
❌ Binning zmiennych ciągłych arbitralnie (zamiast na podstawie danych lub domenowej wiedzy)
❌ Usuwanie zmiennych z niską korelacją liniową bez sprawdzenia korelacji nieliniowej
❌ Traktowanie 'pdays=999' jako wartości numerycznej
```

---

## VIII. Szybkie komendy diagnostyczne (Python)

```python
import pandas as pd
import numpy as np
from scipy import stats

def diagnose_dataset(df, target_col):
    """Szybka diagnostyka zbioru danych dla doboru modelu"""
    
    report = {}
    
    # Rozmiar
    report['n_rows'] = len(df)
    report['n_features'] = len(df.columns) - 1
    
    # Typy zmiennych
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    categorical_cols = [c for c in df.select_dtypes('object').columns]
    continuous_cols = [c for c in df.select_dtypes('number').columns if c != target_col]
    
    report['binary_features'] = binary_cols
    report['categorical_high_cardinality'] = [c for c in categorical_cols if df[c].nunique() > 15]
    report['continuous_features'] = continuous_cols
    
    # Nierównowaga klas
    class_ratio = df[target_col].value_counts(normalize=True)
    report['class_balance'] = class_ratio.to_dict()
    report['is_imbalanced'] = class_ratio.min() < 0.2
    
    # Skośność zmiennych ciągłych
    skewness = {c: df[c].skew() for c in continuous_cols}
    report['highly_skewed'] = {c: s for c, s in skewness.items() if abs(s) > 2}
    
    # Rekomendacja
    if len(binary_cols) >= 5 or len(categorical_cols) >= 3:
        report['linear_model_warning'] = True
        report['recommendation'] = ['XGBoost', 'LightGBM', 'RandomForest']
    else:
        report['linear_model_warning'] = False
        report['recommendation'] = ['LogisticRegression', 'XGBoost', 'RandomForest']
    
    return report
```

---

## IX. Podsumowanie — matryca decyzyjna (skrót dla agenta)

| Cecha danych | Implikacja | Preferowany model |
|---|---|---|
| Dużo binarnych (≥5) | nieliniowe interakcje | XGBoost, RF, LightGBM |
| Kategoryczne z kardynalnością > 15 | OHE nie wystarczy | XGBoost + Target Encoding |
| Ciągłe z progami (thresholds) | nieliniowość | Drzewa, XGBoost |
| Ciągłe z saturacją | log-transform lub GAM | GAM, log-OLS, XGBoost |
| < 500 rekordów | overfitting risk | LogReg, Ridge, DT(max_depth≤4) |
| Nierównowaga klas > 1:5 | zniekształcone metryki | balansowanie + F1/PR-AUC |
| Potrzeba interpretowalności | regulacje / causal | LogReg, GAM, DT(shallow) |
| Causal inference | nie używaj ML wprost | Double ML, CausalForest, PSM |
| Dane czasowe | data leakage z K-Fold | TimeSeriesSplit, LSTM, XGB+lags |
| Wynik logit AUC ≈ 0.5–0.55 | brak liniowej separacji | XGBoost, RF — zmień model |

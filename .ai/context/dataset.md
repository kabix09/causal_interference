# Bank Marketing Dataset — Kontekst danych wejściowych

Źródło danych:  
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data

---

## Opis zmiennych

### Dane demograficzne (cechy klienta)
- **age (wiek)**  
  Wiek klienta w latach. Wpływa na skłonność do oszczędzania oraz sposób targetowania oferty.

- **job (zawód)**  
  Rodzaj wykonywanej pracy (np. admin., technician, management). Wskaźnik statusu społeczno-ekonomicznego.

- **marital (stan cywilny)**  
  Status związku (np. married, single, divorced). Wpływa na stabilność finansową i potrzeby oszczędnościowe.

- **education (wykształcenie)**  
  Poziom edukacji (np. secondary, tertiary). Może korelować z wiedzą finansową.

- **default (niewypłacalność)**  
  Informacja binarna (yes/no), czy klient ma kredyt niespłacany terminowo.

---

### Dane finansowe
- **balance (saldo)**  
  Średnie roczne saldo konta. Kluczowy wskaźnik zasobności klienta.

- **housing (kredyt hipoteczny)**  
  Czy klient posiada kredyt mieszkaniowy (yes/no).

- **loan (pożyczka osobista)**  
  Czy klient posiada inne pożyczki osobiste (yes/no).

---

### Dane operacyjne
- **contact (metoda kontaktu)**  
  Forma komunikacji (np. cellular, telephone, unknown).

- **day (dzień)**  
  Dzień miesiąca ostatniego kontaktu.

- **month (miesiąc)**  
  Miesiąc kontaktu (np. may, aug). Może odzwierciedlać sezonowość.

- **campaign (liczba kontaktów)**  
  Liczba interakcji z klientem w ramach bieżącej kampanii.

---

### Dane historyczne
- **pdays (dni od poprzedniego kontaktu)**  
  Liczba dni od ostatniego kontaktu w poprzedniej kampanii  
  (`-1` oznacza brak wcześniejszego kontaktu).

- **previous (poprzednia liczba kontaktów)**  
  Liczba kontaktów przed obecną kampanią.

- **poutcome (wynik poprzedniej kampanii)**  
  Rezultat poprzedniej kampanii (np. success, failure, unknown).

---

### Inne
- **duration (czas trwania rozmowy)**  
  Czas trwania ostatniej rozmowy (w sekundach).  
  Jest to zmienna wynikająca z interwencji (post-treatment), a nie jej przyczyna.

- **deposit (cel / zmienna Y)**  
  Zmienna binarna (yes/no) — czy klient otworzył depozyt.

---

## Założenia analizy przyczynowej

Do badania wpływu (causal inference) należy używać tylko zmiennych, którymi bank może manipulować.

---

## Role zmiennych

### Outcome (wynik)
- **Y:** `deposit`  
  Zmienna docelowa — końcowy efekt procesu.

---

### Treatment (interwencja)
- **X:** `contact`, `campaign`, `day`, `month`  
  Zmienne kontrolowane przez bank (operator `do(x)`).

- Cel: oszacowanie wpływu interwencji na wynik (np. **ATE — Average Treatment Effect**)

---

### Confounders (zakłócacze)
- **Z:** `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, `default`  

  Zmienne wpływające zarówno na interwencję, jak i wynik.  
  Należy je kontrolować (back-door adjustment), aby odizolować rzeczywisty wpływ działań banku.

---

### Zmienne wykluczone
- **duration**

  Jest to zmienna post-treatment (potomek interwencji).  
  Jej użycie jako zakłócacza prowadziłoby do błędnych wniosków (bias).

---
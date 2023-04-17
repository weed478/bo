# Pakowanie Dostawczaka

## Problem

**Ładowanie paczek do samochodu dostawczego.**

Samochód dostawczy ma pokonać trasę składającą się z $N$ stopów.
Na każdy z nich ma trafić 0+ konkretnych paczek z magazynu.
Mamy daną listę paczek w magazynie, ich rozmiary i wagi.
Musimy załadować pewne paczki do samochodu według poniższych założeń.

Samochód ma bagażniki o wymiarach $x \times y \times z$. Każda paczka ma przypisany punkt gdzie powinna zostać dostarczona (przyjmujemy to jako numer punktu dostawy i zakładamy że jest ustalony). Paczek w magazynie mamy potencjalnie więcej niż jesteśmy w stanie załadować. Chcemy zmaksymalizować wypełnienie samochodu przy jednoczesnym zminimalizowaniu ilości pracy jaką trzeba będzie wykonać wyładowując paczki w kolejnych punktach. Dodatkowo paczki z krótszym terminem dostawy powinny mieć większy priorytet umieszczenia w samochodzie.

Na każdym stopie kierowca musi wyjąć paczki z ładowni przez drzwi (musi przedostać się przez pozostałe paczki). Umiejętne układanie paczek w ładowni zmniejsza czas potrzebny na ich wyjmowanie.

## Model

### Dane

* $N$ - liczba stopów
* $M$ - liczba ładunków w magazynie:
  * $a_i$ - długosć ładunku
  * $b_i$ - szerokość ładunku
  * $c_i$ - wysokość ładunku
  * $n_i$ - numer przystanku dla ładunku
  * $d_i$ - deadline na dostawę ładunku

### Rozwiązanie

Ładownię dzielimy na grid 3D (2D uproszczenie), każde pole ma współrzędne $(x, y, z)$.

$\pi_{x,y,z}$  - numer ładunku $1 \dots M$ na pozycji $(x,y,z)$ w ładowni, jeśli pole jest puste to $0$

### Funkcja kosztu

$$
f(\pi) = \sum_{i=1}^M \left(
    \left(t_i + g(d_i)\right)
    \cdot
    1_i
\right) + h
$$

gdzie:

* $t_i$ - czas na wyjęcie paczki na odpowiednim przystanku wyrażony jako masa ładunków między paczką a drzwiami

* $1_i$ - 1 jeśli bierzemy paczkę 1, 0 jeśli nie

* $g$ - jakaś funkcja która ma dużą wartość, gdy jest blisko deadline-u i ma małe znaczenie gdy jest więcej niż kilka dni

* $h$ - procent załadowania samochodu

Ewentualne sensowne rozbudowanie:\
liczba paczek które trzeba wyjąć aby wyjąć zadaną paczkę

Możliwe dodatkowe ograniczenia:\
minimalna/maksymalna sumaryczna waga
rozłożenie symetrycznie względem osi

## Instalacja

```bash
pip install -r requirements.txt
```

## Użycie

nie ma

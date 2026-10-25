"""
Parabol üzerinde PSO ile MAKSİMUM bulma - el hesabının kodu

Problem
-------
f(x) = -x^2 + 5x + 20 fonksiyonunun [-10, 10] aralığındaki maksimumunu bul.
(Fonksiyon, el hesabındaki fitness değerlerinden çıkarıldı: f(2.3) = 26.21,
 f(0.6) = 22.64, f(8.3) = -7.39 ... Analitik çözüm: x* = -b/(2a) = 2.5,
 f(x*) = 26.25)

El hesabındaki ayarlar
----------------------
- 9 parçacık, tek bilinmeyen (x) -> her parçacık tek bir sayı
- başlangıçta v = 0 ve pbest = parçacığın kendi konumu (henüz hareket etmedik)
- w = c1 = c2 = 1
- 1. iterasyonda r1 = 0.213, r2 = 0.876 ; 2. iterasyonda r1 = 0.113, r2 = 0.706
- Her iterasyonun sırası: hız -> konum -> fitness -> pbest -> gbest

Formüller
---------
v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
x(t+1) = x(t) + v(t+1)
Maksimizasyon:  f(x_yeni) > f(pbest)  ->  pbest = x_yeni
"""

import numpy as np

ALT, UST = -10.0, 10.0  # arama aralığı


def f(x):
    """Maksimize edilecek parabol: f(x) = -x^2 + 5x + 20."""
    return -(x**2) + 5 * x + 20


def kes(a, basamak):
    """Sayıyı YUVARLAMADAN 'basamak' ondalık haneye keser (elle hesaptaki gibi).

    basamak=None ise sayıyı tam hassasiyetle bırakır.
    """
    if basamak is None:
        return a
    kat = 10**basamak
    # round(…, 6): 26.21 gibi sayıların 26.2099999… diye kesilmesini engeller
    return np.trunc(np.round(np.asarray(a, dtype=float) * kat, 6)) / kat


def tablo_yazdir(baslik, x, v, fit, pbest_x, x_ham=None, disari=None):
    print(baslik)
    print(" i |        v |        x |     f(x) |    pbest")
    for i in range(len(x)):
        uyari = ""
        if disari is not None and disari[i]:
            uyari = f"   <- {x_ham[i]:.4f} aralık dışı, sınıra çekildi, v=0"
        print(
            f"{i + 1:2d} | {v[i]:8.4f} | {x[i]:8.4f} | {fit[i]:8.4f} | {pbest_x[i]:8.4f}{uyari}"
        )


def pso_maksimize(
    x0,
    rassal,
    w=1.0,
    c1=1.0,
    c2=1.0,
    n_iter=2,
    basamak=None,
    seed=0,
    verbose=True,
):
    """f(x)'i maksimize eder.

    x0      : başlangıç konumları (her parçacık tek sayı)
    rassal  : [(r1, r2), ...] verilen iterasyonlar için rassal sayı çiftleri;
              daha fazla iterasyon istenirse kalanı rastgele üretilir
    basamak : None -> tam hassasiyet, 2 -> elle hesaptaki gibi 2 haneye kes
    """
    rng = np.random.default_rng(seed)

    # --- 0. iterasyon: başlangıç ---
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)  # ilk hızlar 0
    fit = kes(f(x), basamak)
    pbest_x, pbest_f = x.copy(), fit.copy()  # henüz hareket etmedik: pbest = x
    g = int(np.argmax(pbest_f))  # amaç maksimize etmek -> en büyük fitness
    gbest_x, gbest_f = pbest_x[g], pbest_f[g]
    gecmis = [gbest_f]

    if verbose:
        tablo_yazdir("0. iterasyon (başlangıç: v = 0, pbest = x)", x, v, fit, pbest_x)
        print(f"gbest = x{g + 1} = {gbest_x:.4f}  (fitness {gbest_f:.4f})\n")

    for t in range(1, n_iter + 1):
        # Rassal sayılar her iterasyonda değişir (o iterasyonda tüm parçacıklar için aynı)
        if t <= len(rassal):
            r1, r2 = rassal[t - 1]
        else:
            r1, r2 = rng.random(), rng.random()

        # Hız: v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        v = kes(w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x), basamak)

        # Konum: x(t+1) = x(t) + v(t+1)
        x_ham = kes(x + v, basamak)

        # Aralık kontrolü: [-10, 10] dışına çıkan parçacık sınıra çekilir, hızı sıfırlanır
        disari = (x_ham < ALT) | (x_ham > UST)
        x = np.clip(x_ham, ALT, UST)
        v = np.where(disari, 0.0, v)

        # Yeni konumların fitness'ı
        fit = kes(f(x), basamak)

        # pbest güncelle (maksimizasyon: f(x_yeni) > f(pbest))
        iyi = fit > pbest_f
        pbest_x = np.where(iyi, x, pbest_x)
        pbest_f = np.where(iyi, fit, pbest_f)

        # gbest: pbest'ler içindeki en iyi
        g = int(np.argmax(pbest_f))
        gbest_x, gbest_f = pbest_x[g], pbest_f[g]
        gecmis.append(gbest_f)

        if verbose:
            tablo_yazdir(
                f"{t}. iterasyon  (r1 = {r1:.3f}, r2 = {r2:.3f})",
                x, v, fit, pbest_x, x_ham, disari,
            )
            print(f"gbest = x{g + 1} = {gbest_x:.4f}  (fitness {gbest_f:.4f})\n")

    return gbest_x, gbest_f, gecmis


if __name__ == "__main__":
    X0 = [-9.6, -6, -2.6, -1.1, 0.6, 2.3, 2.8, 8.3, 10]
    RASSAL = [(0.213, 0.876), (0.113, 0.706)]  # el hesabındaki (r1, r2) çiftleri

    print("=" * 66)
    print("A) EL HESABININ TEKRARI  (w = c1 = c2 = 1, sayılar 2 haneye kesilir)")
    print("=" * 66)
    pso_maksimize(X0, RASSAL, n_iter=2, basamak=2)

    print("=" * 66)
    print("B) Daha uzun çalıştırma: 100 iterasyon (3. iterasyondan sonra r1, r2 rastgele)")
    print("=" * 66)
    print("Gerçek maksimum: x = 2.5, f(x) = 26.25")
    for w in (1.0, 0.7):
        bx, bf, gecmis = pso_maksimize(X0, RASSAL, w=w, n_iter=100, seed=1, verbose=False)
        print(f"w = {w}: gbest x = {bx:.6f}, f(x) = {bf:.6f}")

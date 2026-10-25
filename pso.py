"""
Particle Swarm Optimization (PSO) - Parçacık Sürü Optimizasyonu

Bir fonksiyonu minimize eder. Her parçacık; kendi bulduğu en iyi konumu (pbest)
ve komşularının bulduğu en iyi konumu (sosyal en iyi) hesaba katarak hareket eder:

    v = w*v + c1*r1*(pbest - x) + c2*r2*(sosyal_en_iyi - x)
    x = x + v

İki topoloji desteklenir:
    "global": sosyal en iyi = tüm sürünün en iyisi (gbest). Hızlı yakınsar ama
              çok yerel minimumlu fonksiyonlarda erken takılabilir.
    "ring"  : sosyal en iyi = halka üzerindeki en yakın 2*n_neighbors komşunun
              (ve kendisinin) en iyisi (lbest). Bilgi sürüye yavaş yayılır, bu
              yüzden keşif uzun sürer ve yerel minimuma takılma azalır.
"""

import numpy as np


class PSO:
    def __init__(
        self,
        func,
        bounds,
        n_particles=30,
        max_iter=200,
        w_start=0.9,      # başlangıç atalet ağırlığı
        w_end=0.4,        # bitiş atalet ağırlığı (lineer azalır)
        c1=2.0,           # bilişsel katsayı (kendi en iyisine çekim)
        c2=2.0,           # sosyal katsayı (sürünün en iyisine çekim)
        v_max_ratio=0.2,  # hız sınırı: arama aralığının oranı
        tol=1e-12,        # erken durdurma eşiği
        patience=50,      # kaç iterasyon iyileşme olmazsa dur
        topology="global",  # "global" (gbest) veya "ring" (lbest)
        n_neighbors=2,    # ring topolojisinde her yandaki komşu sayısı
        seed=None,
    ):
        if topology not in ("global", "ring"):
            raise ValueError('topology "global" veya "ring" olmalı')
        self.topology = topology
        self.n_neighbors = n_neighbors
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)  # şekil: (boyut, 2)
        self.dim = self.bounds.shape[0]
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.tol = tol
        self.patience = patience
        self.rng = np.random.default_rng(seed)

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.v_max = v_max_ratio * (self.ub - self.lb)

    def optimize(self, verbose=False):
        n, d = self.n_particles, self.dim

        # Başlangıç konumları ve hızları
        x = self.rng.uniform(self.lb, self.ub, size=(n, d))
        v = self.rng.uniform(-self.v_max, self.v_max, size=(n, d))

        # Kişisel en iyiler
        fitness = np.apply_along_axis(self.func, 1, x)
        pbest_x = x.copy()
        pbest_f = fitness.copy()

        # Global en iyi
        g_idx = np.argmin(pbest_f)
        gbest_x = pbest_x[g_idx].copy()
        gbest_f = pbest_f[g_idx]

        history = [gbest_f]
        stall = 0

        # Ring topolojisi: her parçacığın komşu indeksleri (kendisi dahil)
        if self.topology == "ring":
            offsets = np.arange(-self.n_neighbors, self.n_neighbors + 1)
            nbr_idx = (np.arange(n)[:, None] + offsets[None, :]) % n  # (n, 2k+1)

        for it in range(self.max_iter):
            # Atalet ağırlığı lineer olarak azalır: önce keşif, sonra sömürü
            w = self.w_start - (self.w_start - self.w_end) * it / max(self.max_iter - 1, 1)

            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))

            # Sosyal en iyi: global ise gbest, ring ise komşuluğun en iyisi
            if self.topology == "ring":
                best_local = np.argmin(pbest_f[nbr_idx], axis=1)
                social_x = pbest_x[nbr_idx[np.arange(n), best_local]]
            else:
                social_x = gbest_x

            # Hız güncelleme
            v = (
                w * v
                + self.c1 * r1 * (pbest_x - x)
                + self.c2 * r2 * (social_x - x)
            )
            v = np.clip(v, -self.v_max, self.v_max)

            # Konum güncelleme
            x = x + v

            # Sınır kontrolü: sınıra yapıştır ve o yöndeki hızı sıfırla
            below, above = x < self.lb, x > self.ub
            x = np.clip(x, self.lb, self.ub)
            v[below | above] = 0.0

            # Değerlendirme
            fitness = np.apply_along_axis(self.func, 1, x)

            # pbest güncelle
            improved = fitness < pbest_f
            pbest_x[improved] = x[improved]
            pbest_f[improved] = fitness[improved]

            # gbest güncelle
            g_idx = np.argmin(pbest_f)
            if pbest_f[g_idx] < gbest_f - self.tol:
                stall = 0
            else:
                stall += 1
            if pbest_f[g_idx] < gbest_f:
                gbest_f = pbest_f[g_idx]
                gbest_x = pbest_x[g_idx].copy()

            history.append(gbest_f)

            if verbose and (it % 20 == 0 or it == self.max_iter - 1):
                print(f"İterasyon {it:4d} | en iyi değer = {gbest_f:.6e}")

            if stall >= self.patience:
                if verbose:
                    print(f"{it}. iterasyonda erken durdu (iyileşme yok).")
                break

        return gbest_x, gbest_f, history


# ---------------------------------------------------------------------------
# Test fonksiyonları
# ---------------------------------------------------------------------------
def sphere(x):
    return np.sum(x**2)


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


if __name__ == "__main__":
    dim = 10
    problems = [
        ("Sphere", sphere, [(-5.12, 5.12)] * dim),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * dim),
        ("Rosenbrock", rosenbrock, [(-2.048, 2.048)] * dim),
    ]

    # Her problem ve topoloji için 10 farklı tohumla çalıştır, ortalamayı yazdır
    n_runs = 10
    for name, f, bounds in problems:
        print(f"\n=== {name} ({dim} boyut, {n_runs} çalıştırma) ===")
        for topo in ("global", "ring"):
            finals, iters = [], []
            for seed in range(n_runs):
                pso = PSO(
                    f, bounds, n_particles=40, max_iter=1000,
                    patience=200, topology=topo, seed=seed,
                )
                _, best_f, hist = pso.optimize()
                finals.append(best_f)
                iters.append(len(hist) - 1)
            finals = np.array(finals)
            print(
                f"{topo:6s} | ortalama: {finals.mean():.3e} | "
                f"en iyi: {finals.min():.3e} | en kötü: {finals.max():.3e} | "
                f"ort. iterasyon: {np.mean(iters):.0f}"
            )

"""
PSO görselleştirmeleri. Grafikleri gorseller/ klasörüne PNG olarak kaydeder.

    python gorsel.py

1) parabol_iterasyonlar.png : el hesabının 0., 1. ve 2. iterasyonunda parçacıklar
2) yakinsama_parabol.png    : parabolde gbest hatasının iterasyonlara göre değişimi
3) yakinsama_pso.png        : genel PSO'da (pso.py) test fonksiyonlarının yakınsaması
"""

import os

import matplotlib

matplotlib.use("Agg")  # ekran açmadan doğrudan dosyaya kaydet
import matplotlib.pyplot as plt
import numpy as np

from pso import PSO, rastrigin, rosenbrock, sphere
from pso_parabol import ALT, RASSAL, UST, X0, f, pso_maksimize

OPT_X, OPT_F = 2.5, 26.25  # parabolün analitik maksimumu
KLASOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gorseller")


def parabol_iterasyonlar(kayit, dosya):
    """Parabolün üzerinde her iterasyondaki parçacıkları ve gbest'i çizer."""
    xs = np.linspace(ALT, UST, 400)
    fig, eksenler = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for t, ax in enumerate(eksenler):
        k = kayit[t]
        ax.plot(xs, f(xs), color="0.65", lw=2, zorder=1)
        ax.axvline(
            OPT_X, color="tab:green", ls="--", lw=1.2, zorder=1,
            label=f"gerçek maksimum (x = {OPT_X})",
        )
        ax.scatter(k["x"], k["fit"], s=50, color="tab:blue", zorder=3, label="parçacık")
        if t != 1:  # 1. iterasyonda numaralar yakınlaştırma kutusunda gösterilir
            for i, (xi, fi) in enumerate(zip(k["x"], k["fit"]), start=1):
                ax.annotate(
                    str(i), (xi, fi), textcoords="offset points", xytext=(5, 6),
                    fontsize=8, color="tab:blue",
                )

        gx = k["gbest_x"]
        ax.scatter(
            [gx], [f(gx)], marker="*", s=260, color="tab:orange",
            edgecolor="black", linewidth=0.6, zorder=2, label="gbest",
        )

        # 1. iterasyonda parçacıklar zirvede çok sıkışık: yakınlaştırılmış kutu ekle
        if t == 1:
            ins = ax.inset_axes([0.55, 0.08, 0.42, 0.42])
            ins.plot(xs, f(xs), color="0.65", lw=2)
            ins.axvline(OPT_X, color="tab:green", ls="--", lw=1.2)
            ins.scatter(
                [gx], [f(gx)], marker="*", s=130, color="tab:orange",
                edgecolor="black", linewidth=0.5, zorder=2,
            )
            ins.scatter(k["x"], k["fit"], s=28, color="tab:blue", zorder=3)
            for i, (xi, fi) in enumerate(zip(k["x"], k["fit"]), start=1):
                kayma = (-3, -11) if i == 6 else (3, 4)  # 6 ve 7 çok yakın
                ins.annotate(
                    str(i), (xi, fi), textcoords="offset points", xytext=kayma,
                    fontsize=7, color="tab:blue",
                )
            ins.set_xlim(0.5, 3.6)
            ins.set_ylim(23, 26.6)
            ins.tick_params(labelsize=7)
            ins.grid(alpha=0.25)
            ax.indicate_inset_zoom(ins, edgecolor="0.4")

        # Aralık dışına çıkıp sınıra çekilen parçacıkları işaretle
        for i in np.flatnonzero(k["disari"]):
            ax.annotate(
                f"{i + 1}. parçacık: {k['x_ham'][i]:.2f} → {k['x'][i]:.0f}'a çekildi",
                xy=(k["x"][i], k["fit"][i]), xytext=(-4, -85),
                arrowprops=dict(arrowstyle="->", color="tab:red"),
                fontsize=8, color="tab:red",
            )

        baslik = "0. iterasyon (başlangıç)" if t == 0 else f"{t}. iterasyon"
        ax.set_title(f"{baslik}\ngbest: x = {gx:.2f}, f = {f(gx):.2f}")
        ax.set_xlabel("x")
        ax.set_xlim(ALT - 0.5, UST + 1.5)
        ax.set_ylim(-130, 38)
        ax.grid(alpha=0.25)

    eksenler[0].set_ylabel("f(x) = −x² + 5x + 20")
    handles, labels = eksenler[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("PSO el hesabı: parçacıklar parabolün üzerinde (w = c1 = c2 = 1)")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(dosya, dpi=150)
    plt.close(fig)


def yakinsama_parabol(dosya):
    """Parabolde gbest hatasının (26.25 - f(gbest)) iterasyonlara göre değişimi."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for w, renk in ((1.0, "tab:blue"), (0.7, "tab:orange")):
        _, _, gecmis, _ = pso_maksimize(
            X0, RASSAL, w=w, n_iter=100, seed=1, verbose=False
        )
        # 0 değeri log eksende çizilemez: makine hassasiyetine (1e-16) sabitle
        hata = np.maximum(OPT_F - np.array(gecmis), 1e-16)
        ax.semilogy(hata, color=renk, lw=2, label=f"w = {w}")

    ax.set_xlabel("iterasyon")
    ax.set_ylabel("hata = 26.25 − f(gbest)")
    ax.set_title("Parabolde yakınsama (100 iterasyon, tek çalıştırma, seed = 1)")
    ax.text(
        0.99, 0.09, "1e-16: makine hassasiyeti (hata ≈ 0)",
        transform=ax.transAxes, ha="right", fontsize=8, color="0.35",
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dosya, dpi=150)
    plt.close(fig)


def yakinsama_pso(dosya):
    """Genel PSO'da (global topoloji) test fonksiyonlarının yakınsama eğrisi."""
    dim = 10
    problemler = [
        ("Sphere", sphere, [(-5.12, 5.12)] * dim),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * dim),
        ("Rosenbrock", rosenbrock, [(-2.048, 2.048)] * dim),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ad, fonksiyon, sinirlar in problemler:
        _, _, hist = PSO(
            fonksiyon, sinirlar, n_particles=40, max_iter=1000, patience=200, seed=0
        ).optimize()
        ax.semilogy(hist, lw=2, label=ad)

    ax.set_xlabel("iterasyon")
    ax.set_ylabel("en iyi değer (gbest)")
    ax.set_title("Genel PSO'da yakınsama (10 boyut, global topoloji, seed = 0)")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dosya, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(KLASOR, exist_ok=True)

    # El hesabıyla aynı ayarlar (2 haneye kesme), sessiz çalıştır
    _, _, _, kayit = pso_maksimize(X0, RASSAL, n_iter=2, basamak=2, verbose=False)

    parabol_iterasyonlar(kayit, os.path.join(KLASOR, "parabol_iterasyonlar.png"))
    yakinsama_parabol(os.path.join(KLASOR, "yakinsama_parabol.png"))
    yakinsama_pso(os.path.join(KLASOR, "yakinsama_pso.png"))
    print("Grafikler kaydedildi:", KLASOR)

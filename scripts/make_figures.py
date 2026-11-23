"""Regenerate the README figures in docs/img/.

    python scripts/make_figures.py

Uses the package's own embedding, AMI and FNN code. Reads the Sleep-EDF demo
night with a tiny EDF reader so MNE is not required. Needs matplotlib.
"""
from pathlib import Path
import subprocess, sys, tempfile, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from awakening.features import delay_embedding  # noqa: E402
from awakening.diagnostics import ami_delay, fnn_curve  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BG, FG, MUTED, LINE = "#0e0f13", "#e6e6e6", "#9aa0ad", "#262a33"
ACCENT = "#b48ead"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": MUTED, "axes.edgecolor": LINE,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans Mono",
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": FG, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
})


def read_edf(path, wanted, start_s, stop_s):
    """Minimal EDF reader (physical units, uV for these channels)."""
    raw = Path(path).read_bytes()
    ns = int(raw[252:256])
    h = 256
    field = lambda off, size: [raw[h + off + i*size: h + off + (i+1)*size].decode().strip() for i in range(ns)]
    labels = field(0, 16)
    o = 16*ns + 80*ns + 8*ns
    pmin = np.array(list(map(float, field(o, 8)))); o += 8*ns
    pmax = np.array(list(map(float, field(o, 8)))); o += 8*ns
    dmin = np.array(list(map(float, field(o, 8)))); o += 8*ns
    dmax = np.array(list(map(float, field(o, 8)))); o += 8*ns + 80*ns
    nsamp = np.array(list(map(int, field(o, 8))))
    header_bytes = int(raw[184:192])
    record_s = float(raw[244:252])
    per_record = int(nsamp.sum())
    data = np.frombuffer(raw[header_bytes:], dtype="<i2")
    data = data[: (len(data)//per_record)*per_record].reshape(-1, per_record)
    out = []
    for name in wanted:
        i = labels.index(name)
        a = int(nsamp[:i].sum())
        x = data[:, a:a+nsamp[i]].ravel().astype(float)
        x = (x - dmin[i]) * (pmax[i]-pmin[i]) / (dmax[i]-dmin[i]) + pmin[i]
        fs = nsamp[i] / record_s
        out.append(x[int(start_s*fs):int(stop_s*fs)])
    return np.column_stack(out), fs


def pca3(E):
    E = E - E.mean(axis=0)
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    return E @ vt[:3].T


def lorenz_figure():
    sol = solve_ivp(lambda t, s: [10*(s[1]-s[0]), s[0]*(28-s[2])-s[1], s[0]*s[1]-8/3*s[2]],
                    (0, 60), [1, 1, 1], t_eval=np.arange(0, 60, 0.01), rtol=1e-9, atol=1e-9)
    X = sol.y.T[1000:]
    x = X[:, 0]
    tau = 9
    E = delay_embedding(x, tau, 3)
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(np.arange(1500)*0.01, x[:1500], color=FG, lw=0.9)
    ax0.set_title("1 · one measured signal  x(t)")
    ax0.set_xlabel("t")
    for k, (P, title) in enumerate([(X, "2 · the true system (x, y, z)"),
                                    (E[:, ::-1], f"3 · rebuilt from x alone, tau = {tau}")]):
        ax = fig.add_subplot(1, 3, k+2, projection="3d")
        ax.plot(*P.T, lw=0.35, color=ACCENT if k else FG, alpha=0.9)
        ax.set_title(title)
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0, 0, 0, 0)); a.line.set_color((0, 0, 0, 0))
        ax.view_init(18, -60)
    fig.tight_layout()
    fig.savefig(OUT / "takens_lorenz.png", dpi=150)
    plt.close(fig)


def eeg_figures():
    data, fs = read_edf(ROOT/"demo"/"SC4001E0-PSG.edf", ["EEG Fpz-Cz", "EEG Pz-Oz"], 20, 80)
    fs = int(fs)
    t = np.arange(len(data))/fs
    z = (data - data.mean(0)) / data.std(0)
    tau_s, m = 5, 5  # same as docs/VALIDATION.md: tau = 5 samples, m = 5 per channel
    E = delay_embedding(z, tau_s, m)
    P = pca3(E)

    # hero: signal -> geometry
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.4, wspace=0.02)
    for row, name in enumerate(["EEG Fpz-Cz", "EEG Pz-Oz"]):
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t[:10*fs], data[:10*fs, row], color=FG, lw=0.7)
        ax.set_title(f"{name} · 10 s of {len(t)//fs} s", fontsize=10)
        ax.set_ylabel("uV")
        if row == 1:
            ax.set_xlabel("time (s)")
    ax = fig.add_subplot(gs[:, 1], projection="3d")
    c = np.linspace(0, 1, len(P))
    for i in range(0, len(P)-1, 4):
        ax.plot(*P[i:i+5].T, color=plt.cm.magma(0.3 + 0.65*c[i]), lw=0.3, alpha=0.75)
    ax.set_title(f"joint embedding · 2 ch x m={m}, tau={tau_s} samples\n{E.shape[0]} x {E.shape[1]}, shown as PCA 3D (colour = time)", fontsize=9.5)
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0)); a.line.set_color((0, 0, 0, 0))
    ax.view_init(22, -50)
    ax.set_box_aspect((1, 1, 0.8), zoom=1.35)
    fig.suptitle("Sleep-EDF SC4001E0 (demo/), 20–80 s", x=0.01, ha="left", color=MUTED, fontsize=10)
    fig.savefig(OUT / "eeg_embedding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # diagnostics: AMI and FNN on Fpz-Cz
    x = z[:, 0]
    best, ami = ami_delay(x, max_lag=40)
    fnn = fnn_curve(x, best, max_dimension=8, theiler=fs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    lags = np.arange(1, len(ami)+1)
    a1.plot(lags, ami, color=FG, lw=1.2, marker="o", ms=3)
    a1.axvline(best, color=ACCENT, lw=1.2, ls="--")
    a1.annotate(f"first local minimum\ntau = {best} samples ({1000*best/fs:.0f} ms)", (best, ami[best-1]),
                xytext=(-170, 70), textcoords="offset points", color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    a1.set_title("average mutual information · choosing tau")
    a1.set_xlabel("lag (samples)"); a1.set_ylabel("AMI (bits)")
    dims = [r["dimension"] for r in fnn]
    frac = [100*r["false_fraction"] for r in fnn]
    a2.plot(dims, frac, color=FG, lw=1.2, marker="o", ms=4)
    a2.set_title(f"false nearest neighbours · choosing m")
    a2.set_xlabel("embedding dimension m"); a2.set_ylabel("false neighbours (%)")
    a2.set_ylim(0, max(frac)*1.1 + 1)
    fig.suptitle("EEG Fpz-Cz, Sleep-EDF SC4001E0, 20–80 s · exploratory, not a proof of chaos",
                 x=0.01, ha="left", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "diagnostics.png", dpi=150)
    plt.close(fig)
    return best, frac


def forecast_figure():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = lambda *a: subprocess.run([sys.executable, "-m", "awakening", *a], cwd=ROOT, check=True,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run("demo", "--output", str(tmp/"d"))
        run("train", "--manifest", str(tmp/"d"/"manifest.json"), "--output", str(tmp/"r"))
        preds = {k: pd.read_csv(tmp/"r"/f"{k}_test_predictions.csv") for k in ("spectral", "joint")}
        events = pd.read_csv(tmp/"r"/"events.csv")
        thresholds = {k: json.loads((tmp/"r"/"report.json").read_text()) for k in ("joint",)}
    joint = preds["joint"]
    rec_col = next(c for c in joint.columns if c in ("recording", "recording_id", "id"))
    rec = joint[rec_col].iloc[0]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for k, color, lw in (("spectral", MUTED, 1.0), ("joint", ACCENT, 1.6)):
        p = preds[k]; p = p[p[rec_col] == rec]
        tcol = next(c for c in p.columns if c in ("time", "t", "time_s"))
        pcol = next(c for c in p.columns if "prob" in c)
        ax.plot(p[tcol], p[pcol], color=color, lw=lw, label=k)
    ev = events[events[next(c for c in events.columns if c in ("recording", "recording_id", "id"))] == rec]
    ocol = next(c for c in ev.columns if "onset" in c or c in ("time", "t"))
    for i, o in enumerate(ev[ocol]):
        ax.axvline(o, color=FG, lw=1, ls=":")
        ax.text(o, 1.02, "awakening", color=FG, fontsize=9, ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("P(awakening in [t+30, t+150) s)")
    ax.set_title(f"held-out test subject {rec} · SYNTHETIC data with injected pre-wake changes", pad=18)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "forecast_synthetic.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    lorenz_figure()
    print("EEG tau, FNN %:", eeg_figures())
    forecast_figure()
    print("figures in", OUT)

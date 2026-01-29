# gui.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_gen import generate_two_class_data
from mlp import MLP, LRScheduleCosine
from mnist_loader import load_mnist_via_torch
from utils import as_int, as_float


class MLP_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ai del rey - MLP")
        self.geometry("1180x720")

        # --- datasets ---
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.mode = tk.StringVar(value="toy")  # "toy" or "mnist"

        # default model: 2D -> hidden -> 2 outputs
        self.model = MLP(layer_sizes=[2, 16, 16, 2], lr=0.05, beta=2.0, seed=0)

        self._apply_rose_theme()
        self._build_plot()
        self._build_controls()
        self._redraw()

    def _apply_rose_theme(self):
        self.rose_bg = "#fde7ee"
        self.rose_panel = "#f8c9d6"
        self.rose_accent = "#c2185b"
        self.rose_text = "#3a1020"
        self.rose_white = "#ffffff"
        self.plot_bg = "#fff2f6"

        self.configure(bg=self.rose_bg)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Panel.TFrame", background=self.rose_panel)
        style.configure("Rose.TLabel", background=self.rose_panel, foreground=self.rose_text, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.rose_panel, foreground=self.rose_text, font=("Segoe UI", 12, "bold"))
        style.configure("Section.TLabel", background=self.rose_panel, foreground=self.rose_text, font=("Segoe UI", 11, "bold"))

        style.configure("Rose.TButton", background=self.rose_accent, foreground=self.rose_white,
                        padding=8, font=("Segoe UI", 10, "bold"))
        style.map("Rose.TButton", background=[("active", "#d81b60"), ("pressed", "#ad1457")])

        style.configure("Rose.TEntry", fieldbackground=self.rose_white, foreground=self.rose_text)
        style.configure("Rose.TCombobox", fieldbackground=self.rose_white, foreground=self.rose_text)
        style.configure("Rose.TRadiobutton", background=self.rose_panel, foreground=self.rose_text)

    def _build_controls(self):
        panel = ttk.Frame(self, padding=10, style="Panel.TFrame")
        panel.pack(side=tk.LEFT, fill=tk.Y)

        def lab(text: str, row: int, col: int = 0, colspan: int = 1, style: str = "Rose.TLabel", pady=(3, 3)):
            ttk.Label(panel, text=text, style=style).grid(row=row, column=col, columnspan=colspan, sticky="w", pady=pady)

        def entry(row: int, label: str, default):
            lab(label, row, 0)
            var = tk.StringVar(value=str(default))
            ttk.Entry(panel, textvariable=var, width=14, style="Rose.TEntry").grid(row=row, column=1, sticky="w", padx=6, pady=3)
            return var

        r = 0
        lab("ai del rey", r, colspan=2, style="Header.TLabel", pady=(0, 10)); r += 1

        lab("Dataset", r, colspan=2, style="Section.TLabel", pady=(0, 6)); r += 1
        ttk.Radiobutton(panel, text="Toy (2D modes)", variable=self.mode, value="toy", style="Rose.TRadiobutton",
                        command=self._on_mode_change).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Radiobutton(panel, text="MNIST (flattened)", variable=self.mode, value="mnist", style="Rose.TRadiobutton",
                        command=self._on_mode_change).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1

        lab("Toy data generation", r, colspan=2, style="Section.TLabel", pady=(0, 8)); r += 1
        self.var_modes0 = entry(r, "Modes (class 0):", 2); r += 1
        self.var_modes1 = entry(r, "Modes (class 1):", 2); r += 1
        self.var_spm    = entry(r, "Samples per mode:", 80); r += 1
        self.var_mean_lo = entry(r, "Mean min:", -1.0); r += 1
        self.var_mean_hi = entry(r, "Mean max:",  1.0); r += 1
        self.var_std_lo  = entry(r, "Std min:",  0.10); r += 1
        self.var_std_hi  = entry(r, "Std max:",  0.45); r += 1
        self.var_seed    = entry(r, "Random seed:", 0); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1

        lab("Network", r, colspan=2, style="Section.TLabel", pady=(0, 8)); r += 1
        self.var_arch   = entry(r, "Layer sizes:", "2,16,16,2"); r += 1
        self.var_lr     = entry(r, "LR η:", 0.05); r += 1
        self.var_epochs = entry(r, "Epochs:", 20); r += 1
        self.var_beta   = entry(r, "Sigmoid β:", 2.0); r += 1
        self.var_bs     = entry(r, "Batch size:", 64); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        self.var_use_sched = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Variable LR (cosine)",
                        variable=self.var_use_sched).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.var_eta_min = entry(r, "η_min:", 0.005); r += 1
        self.var_eta_max = entry(r, "η_max:", 0.05); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1

        ttk.Button(panel, text="Load / Generate data", command=self.on_load_data, style="Rose.TButton")\
            .grid(row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Button(panel, text="Train network", command=self.on_train, style="Rose.TButton")\
            .grid(row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Button(panel, text="Reset weights", command=self.on_reset, style="Rose.TButton")\
            .grid(row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1

        ttk.Separator(panel).grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1
        self.status = tk.StringVar(value="Choose dataset, then Load/Generate.")
        ttk.Label(panel, textvariable=self.status, style="Rose.TLabel", wraplength=300)\
            .grid(row=r, column=0, columnspan=2, sticky="w")

        self._on_mode_change()

    def _build_plot(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8.2, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(self.rose_bg)
        self.ax.set_facecolor(self.plot_bg)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _parse_arch(self) -> list[int]:
        raw = self.var_arch.get().strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        sizes = [int(float(p)) for p in parts]

        if len(sizes) < 3:
            raise ValueError("Need ≥3 layers, e.g. 2,16,2 or 784,128,10")
        if len(sizes) > 5:
            raise ValueError("Max 5 layers total.")
        return sizes

    def _on_mode_change(self):
        # set a sensible default architecture per mode
        if self.mode.get() == "toy":
            self.var_arch.set("2,16,16,2")
            self.status.set("Toy mode: decision boundary will be shown.")
        else:
            self.var_arch.set("784,128,64,10")
            self.status.set("MNIST mode: plot shows loss text only (no boundary).")
        self._redraw()

    # ---------- plotting ----------
    def _plot_limits(self):
        if self.X is None or self.X.size == 0:
            return (-2, 2, -2, 2)
        x_min, x_max = float(self.X[:, 0].min()), float(self.X[:, 0].max())
        y_min, y_max = float(self.X[:, 1].min()), float(self.X[:, 1].max())
        pad_x = 0.25 * max(1e-6, x_max - x_min)
        pad_y = 0.25 * max(1e-6, y_max - y_min)
        return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

    def _draw_background(self):
        if self.mode.get() != "toy":
            return
        if self.X is None or self.X.size == 0:
            return
        x0, x1, y0, y1 = self._plot_limits()
        xs = np.linspace(x0, x1, 260)
        ys = np.linspace(y0, y1, 260)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.c_[XX.ravel(), YY.ravel()]

        pred = self.model.predict(grid).reshape(XX.shape)
        self.ax.contourf(XX, YY, pred, levels=[-0.5, 0.5, 1.5], alpha=0.25)

    def _draw_points(self):
        if self.mode.get() != "toy":
            return
        if self.X is None or self.y is None or self.X.size == 0:
            return
        X0 = self.X[self.y == 0]
        X1 = self.X[self.y == 1]
        if X0.size:
            self.ax.scatter(X0[:, 0], X0[:, 1], s=18, label="class 0")
        if X1.size:
            self.ax.scatter(X1[:, 0], X1[:, 1], s=18, label="class 1")
        self.ax.legend(loc="upper right")

    def _redraw(self):
        self.ax.clear()
        self.ax.set_facecolor(self.plot_bg)

        if self.mode.get() == "toy":
            self.ax.set_title("Toy samples + decision boundary (MLP)")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")

            if self.X is not None and self.X.size:
                self._draw_background()
                self._draw_points()
                x0, x1, y0, y1 = self._plot_limits()
                self.ax.set_xlim(x0, x1)
                self.ax.set_ylim(y0, y1)
            self.ax.grid(True, alpha=0.25)
        else:
            self.ax.set_title("MNIST training (no boundary plot)")
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")
            self.ax.text(0.02, 0.92, self.status.get(), transform=self.ax.transAxes)

        self.canvas.draw()

    # ---------- actions ----------
    def on_load_data(self):
        try:
            seed = as_int(self.var_seed.get(), "Random seed", 0)
            arch = self._parse_arch()
            lr = as_float(self.var_lr.get(), "LR η")
            beta = as_float(self.var_beta.get(), "Sigmoid β")

            if self.mode.get() == "toy":
                modes0 = as_int(self.var_modes0.get(), "Modes (class 0)", 0)
                modes1 = as_int(self.var_modes1.get(), "Modes (class 1)", 0)
                spm = as_int(self.var_spm.get(), "Samples per mode", 1)
                mean_lo = as_float(self.var_mean_lo.get(), "Mean min")
                mean_hi = as_float(self.var_mean_hi.get(), "Mean max")
                std_lo = as_float(self.var_std_lo.get(), "Std min")
                std_hi = as_float(self.var_std_hi.get(), "Std max")

                self.X, y01 = generate_two_class_data(
                    modes0=modes0, modes1=modes1, samples_per_mode=spm,
                    mean_low=mean_lo, mean_high=mean_hi,
                    std_low=std_lo, std_high=std_hi,
                    seed=seed
                )
                self.y = y01.astype(int)

                if arch[0] != 2 or arch[-1] != 2:
                    raise ValueError("Toy mode needs arch starting with 2 and ending with 2, e.g. 2,16,2")

                self.model = MLP(layer_sizes=arch, lr=lr, beta=beta, seed=seed)
                self.status.set(f"Toy data: {self.X.shape[0]} samples. Model reset.")
            else:
                Xtr, ytr = load_mnist_via_torch(train=True)
                self.X, self.y = Xtr, ytr

                if arch[0] != 784 or arch[-1] != 10:
                    raise ValueError("MNIST needs arch starting with 784 and ending with 10, e.g. 784,128,10")

                self.model = MLP(layer_sizes=arch, lr=lr, beta=beta, seed=seed)
                self.status.set(f"MNIST train loaded: {self.X.shape[0]} samples. Model reset.")

            self._redraw()
        except Exception as e:
            messagebox.showerror("Load/Generate error", str(e))

    def on_train(self):
        if self.X is None or self.y is None or self.X.size == 0:
            messagebox.showinfo("No data", "Load/Generate data first.")
            return

        try:
            lr = as_float(self.var_lr.get(), "LR η")
            epochs = as_int(self.var_epochs.get(), "Epochs", 1)
            beta = as_float(self.var_beta.get(), "Sigmoid β")
            bs = as_int(self.var_bs.get(), "Batch size", 1)

            if lr <= 0:
                raise ValueError("LR η must be > 0")
            if beta <= 0:
                raise ValueError("Sigmoid β must be > 0")

            self.model.set_params(lr=lr, beta=beta)

            sched = None
            if self.var_use_sched.get():
                eta_min = as_float(self.var_eta_min.get(), "η_min")
                eta_max = as_float(self.var_eta_max.get(), "η_max")
                if eta_min <= 0 or eta_max <= 0 or eta_max < eta_min:
                    raise ValueError("Need 0 < η_min ≤ η_max")
                sched = LRScheduleCosine(eta_min=eta_min, eta_max=eta_max)

            losses = self.model.train_minibatch(
                self.X, self.y,
                epochs=epochs,
                batch_size=bs,
                shuffle=True,
                lr_schedule=sched
            )
            acc = self.model.accuracy(self.X, self.y)
            self.status.set(f"Trained {epochs} epochs | CE={losses[-1]:.4f} | Acc={acc*100:.2f}%")
            self._redraw()

        except Exception as e:
            messagebox.showerror("Train error", str(e))

    def on_reset(self):
        try:
            seed = as_int(self.var_seed.get(), "Random seed", 0)
        except Exception:
            seed = 0
        self.model.reset(seed=seed)
        self.status.set("Weights reset.")
        self._redraw()

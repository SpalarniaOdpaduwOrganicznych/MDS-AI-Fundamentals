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
from neuron import SingleNeuron, LRSchedule
from utils import as_int, as_float


EVAL_ACTIVATIONS = [
    "heaviside",
    "logistic",
    "tanh",        # grade 4 (method B)
    "sin",         # grade 4 (method B)
    "sign",        # grade 5 eval
    "relu",        # grade 5 eval
    "leaky_relu",  # grade 5 eval
]

TRAINABLE = {"heaviside", "logistic", "tanh", "sin"}

# ox OD im fcking dying jk
class SingleNeuronGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌸AI task 3🌸")
        self.geometry("1120x700")

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

        self.model = SingleNeuron(lr=0.05, activation="heaviside", beta=2.0, leaky_alpha=0.01, seed=0)

        # Theme first, then build plot (so redraw won't crash), then controls
        self._apply_pink_theme()
        self._build_plot()
        self._build_controls()
        self._redraw()

    # ---------------- Theme ----------------
    def _apply_pink_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self._pink_bg = "#ffe4f1"
        self._pink_panel = "#ffd1e8"
        self._pink_accent = "#ff5aa5"
        self._text = "#3a1b2c"
        self._white = "#ffffff"

        self.configure(bg=self._pink_bg)

        # Frames / Labels
        style.configure("PinkBg.TFrame", background=self._pink_bg)
        style.configure("Panel.TFrame", background=self._pink_panel)

        style.configure("Pink.TLabel", background=self._pink_panel, foreground=self._text, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self._pink_panel, foreground=self._text, font=("Segoe UI", 13, "bold"))
        style.configure("Section.TLabel", background=self._pink_panel, foreground=self._text, font=("Segoe UI", 11, "bold"))

        # Buttons
        style.configure("Pink.TButton", background=self._pink_accent, foreground=self._white,
                        padding=8, font=("Segoe UI", 10, "bold"))
        style.map("Pink.TButton",
                  background=[("active", "#ff78b8"), ("pressed", "#ff3f9a")],
                  foreground=[("disabled", "#f3f3f3")])

        # Inputs
        style.configure("Pink.TEntry", fieldbackground=self._white, foreground=self._text)
        style.configure("Pink.TCombobox", fieldbackground=self._white, foreground=self._text)

        # Checkbutton
        style.configure("Pink.TCheckbutton", background=self._pink_panel, foreground=self._text)

        # Separator
        style.configure("Pink.TSeparator", background=self._pink_panel)

    # ---------------- UI ----------------
    def _build_controls(self):
        panel = ttk.Frame(self, padding=10, style="Panel.TFrame")
        panel.pack(side=tk.LEFT, fill=tk.Y)

        def label(text: str):
            return ttk.Label(panel, text=text, style="Pink.TLabel")

        def section(text: str, row: int):
            ttk.Label(panel, text=text, style="Section.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))

        def entry(row: int, label_text: str, default):
            label(label_text).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(default))
            ttk.Entry(panel, textvariable=var, width=12, style="Pink.TEntry").grid(row=row, column=1, sticky="w", padx=6, pady=3)
            return var

        r = 0
        ttk.Label(panel, text="🌸 Lana del rey task 3🌸", style="Header.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        r += 1

        section("Data generation", r); r += 1
        self.var_modes0 = entry(r, "Modes (class 0):", 2); r += 1
        self.var_modes1 = entry(r, "Modes (class 1):", 2); r += 1
        self.var_spm    = entry(r, "Samples per mode:", 80); r += 1

        ttk.Separator(panel, orient="horizontal", style="Pink.TSeparator").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

        self.var_mean_lo = entry(r, "Mean min:", -1.0); r += 1
        self.var_mean_hi = entry(r, "Mean max:",  1.0); r += 1
        self.var_std_lo  = entry(r, "Std min:",  0.10); r += 1
        self.var_std_hi  = entry(r, "Std max:",  0.45); r += 1
        self.var_seed    = entry(r, "Random seed:", 0); r += 1

        ttk.Separator(panel, orient="horizontal", style="Pink.TSeparator").grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1

        section("Neuron (evaluation/training)", r); r += 1

        label("Activation:").grid(row=r, column=0, sticky="w", pady=3)
        self.var_act = tk.StringVar(value="heaviside")
        cb = ttk.Combobox(panel, textvariable=self.var_act, width=12, state="readonly",
                          values=EVAL_ACTIVATIONS, style="Pink.TCombobox")
        cb.grid(row=r, column=1, sticky="w", padx=6, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_activation_change())
        r += 1

        self.var_lr     = entry(r, "Base LR η:", 0.05); r += 1
        self.var_epochs = entry(r, "Epochs:", 30); r += 1
        self.var_beta   = entry(r, "Sigmoid β:", 2.0); r += 1
        self.var_lalpha = entry(r, "Leaky α:", 0.01); r += 1

        ttk.Separator(panel, orient="horizontal", style="Pink.TSeparator").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

        # Grade 4 Method A: variable LR (cosine)
        self.var_use_sched = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            panel,
            text="Use variable LR (cosine)  🌷",
            variable=self.var_use_sched,
            style="Pink.TCheckbutton",
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        r += 1

        self.var_eta_min = entry(r, "η_min:", 0.005); r += 1
        self.var_eta_max = entry(r, "η_max:", 0.05); r += 1

        ttk.Separator(panel, orient="horizontal", style="Pink.TSeparator").grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1

        ttk.Button(panel, text="Generate data 🌸", command=self.on_generate, style="Pink.TButton").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4
        )
        r += 1
        ttk.Button(panel, text="Train neuron ✨", command=self.on_train, style="Pink.TButton").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4
        )
        r += 1
        ttk.Button(panel, text="Reset weights 💮", command=self.on_reset_weights, style="Pink.TButton").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4
        )
        r += 1

        ttk.Separator(panel, orient="horizontal", style="Pink.TSeparator").grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1
        self.status = tk.StringVar(value="Generate data to begin 🌼")
        ttk.Label(panel, textvariable=self.status, style="Pink.TLabel", wraplength=300).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )

        self._on_activation_change()

    def _build_plot(self):
        frame = ttk.Frame(self, padding=10, style="PinkBg.TFrame")
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7.9, 5.7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(self._pink_bg)
        self.ax.set_facecolor("#fff0f7")

        self.ax.set_title("Samples")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_activation_change(self):
        act = self.var_act.get().lower()
        try:
            lr = as_float(self.var_lr.get(), "Base LR η")
            beta = as_float(self.var_beta.get(), "Sigmoid β")
            lalpha = as_float(self.var_lalpha.get(), "Leaky α")
            self.model.set_params(lr=lr, activation=act, beta=beta, leaky_alpha=lalpha)
        except Exception:
            pass

        if act not in TRAINABLE:
            self.status.set(f"Activation '{act}' is evaluation-only 🌷 (Grade 5). Training will warn.")
        else:
            self.status.set("Ready 🌸 Generate data, then train.")
        self._redraw()

    # ---------------- Plot helpers ----------------
    def _plot_limits(self):
        if self.X is None or self.X.size == 0:
            return (-2, 2, -2, 2)
        x_min, x_max = float(self.X[:, 0].min()), float(self.X[:, 0].max())
        y_min, y_max = float(self.X[:, 1].min()), float(self.X[:, 1].max())
        pad_x = 0.25 * max(1e-6, x_max - x_min)
        pad_y = 0.25 * max(1e-6, y_max - y_min)
        return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

    def _draw_background(self):
        if self.X is None or self.X.size == 0:
            return

        x0, x1, y0, y1 = self._plot_limits()
        xs = np.linspace(x0, x1, 260)
        ys = np.linspace(y0, y1, 260)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.c_[XX.ravel(), YY.ravel()]

        zz = self.model.predict_label(grid).reshape(XX.shape)
        self.ax.contourf(XX, YY, zz, levels=[-0.5, 0.5, 1.5], alpha=0.25)

    def _draw_points(self):
        if self.X is None or self.y is None or self.X.size == 0:
            return
        X0 = self.X[self.y == 0]
        X1 = self.X[self.y == 1]
        if X0.size:
            self.ax.scatter(X0[:, 0], X0[:, 1], s=18, label="class 0")
        if X1.size:
            self.ax.scatter(X1[:, 0], X1[:, 1], s=18, label="class 1")
        self.ax.legend(loc="upper right")

    def _draw_boundary_line(self):
        if self.X is None or self.X.size == 0:
            return
        w1, w2, b = self.model.w
        x0, x1, y0, y1 = self._plot_limits()

        if abs(w2) < 1e-9:
            if abs(w1) < 1e-9:
                return
            x_line = -b / w1
            self.ax.plot([x_line, x_line], [y0, y1], linewidth=2)
        else:
            xs = np.array([x0, x1])
            ys = -(w1 * xs + b) / w2
            self.ax.plot(xs, ys, linewidth=2)

    def _add_flower_corners(self):
        if self.X is None or self.X.size == 0:
            return
        x0, x1, y0, y1 = self._plot_limits()
        self.ax.text(x0, y1, "🌸", fontsize=16, va="top", ha="left")
        self.ax.text(x1, y1, "🌷", fontsize=16, va="top", ha="right")
        self.ax.text(x0, y0, "🌼", fontsize=16, va="bottom", ha="left")
        self.ax.text(x1, y0, "💮", fontsize=16, va="bottom", ha="right")

    def _redraw(self):
        # Safety: if called before plot is built
        if not hasattr(self, "ax"):
            return

        self.ax.clear()
        self.ax.set_facecolor("#fff0f7")
        self.ax.set_title("Samples")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        if self.X is not None and self.X.size:
            self._draw_background()
            self._draw_points()
            self._draw_boundary_line()
            self._add_flower_corners()
            x0, x1, y0, y1 = self._plot_limits()
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)

        self.ax.grid(True, alpha=0.25)
        self.canvas.draw()

    # ---------------- Callbacks ----------------
    def on_generate(self):
        try:
            modes0 = as_int(self.var_modes0.get(), "Modes (class 0)", 0)
            modes1 = as_int(self.var_modes1.get(), "Modes (class 1)", 0)
            spm = as_int(self.var_spm.get(), "Samples per mode", 1)

            mean_lo = as_float(self.var_mean_lo.get(), "Mean min")
            mean_hi = as_float(self.var_mean_hi.get(), "Mean max")
            std_lo = as_float(self.var_std_lo.get(), "Std min")
            std_hi = as_float(self.var_std_hi.get(), "Std max")
            seed = as_int(self.var_seed.get(), "Random seed", 0)

            self.X, self.y = generate_two_class_data(
                modes0=modes0, modes1=modes1, samples_per_mode=spm,
                mean_low=mean_lo, mean_high=mean_hi,
                std_low=std_lo, std_high=std_hi,
                seed=seed
            )

            self.model.reset(seed=seed)
            self.status.set(f"Generated {self.X.shape[0]} samples 🌸 Weights reset.")
            self._redraw()
        except Exception as e:
            messagebox.showerror("Generate error", str(e))

    def on_train(self):
        if self.X is None or self.y is None or self.X.size == 0:
            messagebox.showinfo("No data", "Generate data first 🌼")
            return

        act = self.var_act.get().lower()
        if act not in TRAINABLE:
            messagebox.showwarning(
                "Evaluation-only activation",
                f"Activation '{act}' is required only for Grade 5 evaluation.\n"
                f"Training is not implemented for it (rubric doesn’t require it)."
            )
            return

        try:
            lr = as_float(self.var_lr.get(), "Base LR η")
            epochs = as_int(self.var_epochs.get(), "Epochs", 1)
            beta = as_float(self.var_beta.get(), "Sigmoid β")
            lalpha = as_float(self.var_lalpha.get(), "Leaky α")

            if lr <= 0:
                raise ValueError("Base LR η must be > 0")
            if beta <= 0:
                raise ValueError("Sigmoid β must be > 0")
            if lalpha <= 0:
                raise ValueError("Leaky α must be > 0")

            self.model.set_params(lr=lr, activation=act, beta=beta, leaky_alpha=lalpha)

            sched = None
            if self.var_use_sched.get():
                eta_min = as_float(self.var_eta_min.get(), "η_min")
                eta_max = as_float(self.var_eta_max.get(), "η_max")
                if eta_min <= 0 or eta_max <= 0 or eta_max < eta_min:
                    raise ValueError("Need 0 < η_min ≤ η_max")
                sched = LRSchedule(eta_min=eta_min, eta_max=eta_max)

            losses = self.model.train_sgd(self.X, self.y, epochs=epochs, shuffle=True, lr_schedule=sched)
            acc = self.model.accuracy(self.X, self.y)

            extra = " 🌷 (variable LR)" if sched is not None else ""
            self.status.set(f"Trained {epochs} epochs{extra}. MSE={losses[-1]:.4f}, Acc={acc*100:.1f}% ✨")
            self._redraw()

        except Exception as e:
            messagebox.showerror("Train error", str(e))

    def on_reset_weights(self):
        try:
            seed = as_int(self.var_seed.get(), "Random seed", 0)
        except Exception:
            seed = 0
        self.model.reset(seed=seed)
        self.status.set("Weights reset 💮")
        self._redraw()

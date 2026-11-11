"""focus_stack — стекинг изображений с графическим интерфейсом.

Вкладка «Ландшафт» — фокус-стекинг с пресетами золотого/синего часа.
Вкладка «Астрофото» — стекинг ночного неба в духе Sequator.

Запуск: python main.py
"""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from stacking.astro import MODE_LABELS_RU, stack_astro
from stacking.common import save_image
from stacking.landscape import PRESETS, focus_stack

PRESET_LABELS_RU = {
    "none": "Без обработки",
    "golden_hour": "Золотой час",
    "blue_hour": "Синий час",
}


class BaseStackTab(ttk.Frame):
    """Общая раскладка: список кадров, опции, запуск, прогресс, журнал."""

    tab_title = ""

    def __init__(self, master):
        super().__init__(master, padding=10)
        self._queue = queue.Queue()
        self._build_file_list()
        self.options = ttk.LabelFrame(self, text="Настройки", padding=8)
        self.options.pack(fill="x", pady=(8, 0))
        self._build_options()
        self._build_run_row()
        self._build_log()

    def _build_file_list(self):
        frame = ttk.LabelFrame(self, text="Кадры для стекинга", padding=8)
        frame.pack(fill="both", expand=True)
        self.file_list = tk.Listbox(frame, selectmode=tk.EXTENDED, height=8)
        scroll = ttk.Scrollbar(frame, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scroll.set)
        self.file_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        buttons = ttk.Frame(frame)
        buttons.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(buttons, text="Добавить…", command=self._add_files).pack(fill="x")
        ttk.Button(buttons, text="Удалить выбранные", command=self._remove_selected).pack(
            fill="x", pady=(4, 0)
        )
        ttk.Button(buttons, text="Очистить", command=lambda: self.file_list.delete(0, tk.END)).pack(
            fill="x", pady=(4, 0)
        )

    def _build_run_row(self):
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Стекинг ▶", command=self._run).pack(side="left")
        self.progress = ttk.Progressbar(row, mode="determinate", length=240)
        self.progress.pack(side="left", padx=8, fill="x", expand=True)
        self.status = ttk.Label(row, text="")
        self.status.pack(side="left")

    def _build_log(self):
        frame = ttk.LabelFrame(self, text="Журнал", padding=4)
        frame.pack(fill="both", pady=(8, 0))
        self.log = tk.Text(frame, height=6, state=tk.DISABLED, wrap="word")
        self.log.pack(fill="both", expand=True)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Выберите кадры",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("Все файлы", "*.*")],
        )
        for f in files:
            self.file_list.insert(tk.END, f)

    def _remove_selected(self):
        for index in reversed(self.file_list.curselection()):
            self.file_list.delete(index)

    def _paths(self):
        return list(self.file_list.get(0, tk.END))

    def _log(self, message):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _run(self):
        try:
            params = self._collect()
        except ValueError as exc:
            messagebox.showwarning(self.tab_title, str(exc), parent=self)
            return
        self.progress.configure(value=0)
        self._log("=== Запуск ===")
        threading.Thread(target=self._worker, args=(params,), daemon=True).start()
        self.after(100, self._poll_queue)

    def _worker(self, params):
        paths, kwargs, output = params
        try:
            result = self.run_task(paths, kwargs)
            save_image(output, result)
            self._queue.put(("done", f"Сохранено: {output}"))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, message = self._queue.get_nowait()
                if kind == "progress":
                    frac, msg = message
                    self.progress.configure(value=frac * 100)
                    self.status.configure(text=msg)
                    self._log(msg)
                elif kind == "done":
                    self.progress.configure(value=100)
                    self.status.configure(text="Готово")
                    self._log(message)
                    messagebox.showinfo(self.tab_title, message, parent=self)
                    return
                elif kind == "error":
                    self.status.configure(text="Ошибка")
                    self._log(f"ОШИБКА: {message}")
                    messagebox.showerror(self.tab_title, message, parent=self)
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _build_options(self):
        raise NotImplementedError

    def _collect(self):
        raise NotImplementedError

    def run_task(self, paths, kwargs):
        raise NotImplementedError

    def _ask_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 16 бит", "*.png"), ("TIFF 16 бит", "*.tif"), ("JPEG", "*.jpg")],
        )
        if path:
            self.output_var.set(path)

    def _progress_cb(self, frac, msg):
        self._queue.put(("progress", (frac, msg)))


class LandscapeTab(BaseStackTab):
    tab_title = "Ландшафт"

    def __init__(self, master):
        super().__init__(master)
        self.file_list.configure(selectmode=tk.EXTENDED)

    def _build_options(self):
        row1 = ttk.Frame(self.options)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Пресет:").pack(side="left")
        self.preset_var = tk.StringVar(value=PRESET_LABELS_RU["none"])
        ttk.Combobox(
            row1,
            textvariable=self.preset_var,
            values=[PRESET_LABELS_RU[p] for p in PRESETS],
            state="readonly",
            width=16,
        ).pack(side="left", padx=(4, 16))

        ttk.Label(row1, text="Детализация:").pack(side="left")
        self.detail_var = tk.DoubleVar(value=1.0)
        ttk.Scale(row1, from_=0.5, to=3.0, variable=self.detail_var, length=180).pack(
            side="left", padx=4
        )
        self.detail_label = ttk.Label(row1, text="1.0", width=4)
        self.detail_label.pack(side="left")
        self.detail_var.trace_add("write", lambda *_: self.detail_label.configure(
            text=f"{self.detail_var.get():.1f}"
        ))

        row2 = ttk.Frame(self.options)
        row2.pack(fill="x", pady=(6, 2))
        ttk.Label(row2, text="Результат:").pack(side="left")
        self.output_var = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="Сохранить как…", command=self._ask_output).pack(side="left")

    def _collect(self):
        paths = self._paths()
        if len(paths) < 2:
            raise ValueError("Добавьте минимум 2 кадра для фокус-стекинга")
        output = self.output_var.get().strip()
        if not output:
            raise ValueError("Укажите файл результата")
        label_to_key = {v: k for k, v in PRESET_LABELS_RU.items()}
        return paths, {
            "detail": float(self.detail_var.get()),
            "preset": label_to_key[self.preset_var.get()],
            "progress": self._progress_cb,
        }, output

    def run_task(self, paths, kwargs):
        return focus_stack(paths, **kwargs)


class AstroTab(BaseStackTab):
    tab_title = "Астрофото"

    def _build_options(self):
        row0 = ttk.Frame(self.options)
        row0.pack(fill="x", pady=2)
        ttk.Label(row0, text="Дарк-кадры:").pack(side="left")
        self.darks_var = tk.StringVar(value="не заданы")
        ttk.Label(row0, textvariable=self.darks_var, foreground="#555").pack(side="left", padx=4)
        ttk.Button(row0, text="Выбрать дарки…", command=self._pick_darks).pack(side="left")
        self.darks = []

        row1 = ttk.Frame(self.options)
        row1.pack(fill="x", pady=2)
        self.align_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Выравнивание по звёздам", variable=self.align_var).pack(side="left")
        ttk.Label(row1, text="Линия горизонта (доля высоты):").pack(side="left", padx=(16, 4))
        self.horizon_var = tk.DoubleVar(value=0.55)
        ttk.Scale(row1, from_=0.0, to=1.0, variable=self.horizon_var, length=160).pack(side="left")

        row2 = ttk.Frame(self.options)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Режим:").pack(side="left")
        self.mode_var = tk.StringVar(value=MODE_LABELS_RU["sigma_clip"])
        ttk.Combobox(
            row2,
            textvariable=self.mode_var,
            values=list(MODE_LABELS_RU.values()),
            state="readonly",
            width=22,
        ).pack(side="left", padx=4)
        ttk.Label(row2, text="Сигма:").pack(side="left", padx=(12, 2))
        self.sigma_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(row2, from_=1.0, to=6.0, increment=0.5, textvariable=self.sigma_var, width=5).pack(
            side="left"
        )

        row3 = ttk.Frame(self.options)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Яркость:").pack(side="left")
        self.exposure_var = tk.DoubleVar(value=1.0)
        ttk.Scale(row3, from_=0.5, to=8.0, variable=self.exposure_var, length=140).pack(
            side="left", padx=4
        )
        ttk.Label(row3, text="Насыщенность:").pack(side="left", padx=(12, 2))
        self.saturation_var = tk.DoubleVar(value=1.15)
        ttk.Scale(row3, from_=1.0, to=3.0, variable=self.saturation_var, length=140).pack(
            side="left", padx=4
        )
        self.chroma_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Гасить цветной шум", variable=self.chroma_var).pack(
            side="left", padx=(16, 0)
        )

        row4 = ttk.Frame(self.options)
        row4.pack(fill="x", pady=(6, 2))
        ttk.Label(row4, text="Результат:").pack(side="left")
        self.output_var = tk.StringVar(value="")
        ttk.Entry(row4, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row4, text="Сохранить как…", command=self._ask_output).pack(side="left")

    def _pick_darks(self):
        files = filedialog.askopenfilenames(
            title="Выберите дарк-кадры (с закрытым затвором)",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.tiff"), ("Все файлы", "*.*")],
        )
        self.darks = list(files)
        self.darks_var.set(f"{len(self.darks)} шт." if self.darks else "не заданы")

    def _collect(self):
        paths = self._paths()
        if len(paths) < 2:
            raise ValueError("Добавьте минимум 2 светлых кадра")
        output = self.output_var.get().strip()
        if not output:
            raise ValueError("Укажите файл результата")
        label_to_key = {v: k for k, v in MODE_LABELS_RU.items()}
        return paths, {
            "dark_paths": self.darks or None,
            "align": bool(self.align_var.get()),
            "horizon": float(self.horizon_var.get()),
            "mode": label_to_key[self.mode_var.get()],
            "sigma_k": float(self.sigma_var.get()),
            "exposure": float(self.exposure_var.get()),
            "saturation": float(self.saturation_var.get()),
            "chroma_nr": bool(self.chroma_var.get()),
            "progress": self._progress_cb,
        }, output

    def run_task(self, paths, kwargs):
        return stack_astro(paths, **kwargs)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("focus_stack — стекинг для пейзажа и астрофото")
        self.geometry("780x640")
        self.minsize(680, 560)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        notebook.add(LandscapeTab(notebook), text="  Ландшафт (фокус-стекинг)  ")
        notebook.add(AstroTab(notebook), text="  Астрофото (Sequator)  ")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()

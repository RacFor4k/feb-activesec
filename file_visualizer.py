import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class FileImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Binary File → Stretched Image")
        self.root.geometry("800x600")
        self.root.configure(bg="black")

        self.data = None
        self.mode = "grayscale"  # или "rgb"
        self.photo = None

        # Кнопки
        btn_frame = tk.Frame(root, bg="black")
        btn_frame.pack(pady=5)

        self.load_btn = tk.Button(btn_frame, text="Открыть файл", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.mode_btn = tk.Button(btn_frame, text="Режим: Ч/Б", command=self.toggle_mode)
        self.mode_btn.pack(side=tk.LEFT, padx=5)

        # Холст
        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Configure>", self.on_resize)

    def load_file(self):
        path = filedialog.askopenfilename(title="Выберите любой файл")
        if not path:
            return
        with open(path, "rb") as f:
            self.data = f.read()
        self.redraw()

    def toggle_mode(self):
        self.mode = "rgb" if self.mode == "grayscale" else "grayscale"
        self.mode_btn.config(text=f"Режим: {'RGB' if self.mode == 'rgb' else 'Ч/Б'}")
        self.redraw()

    def redraw(self, event=None):
        if not self.data:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        if self.mode == "grayscale":
            num_pixels = len(self.data)
            if num_pixels == 0:
                return
            side = int(len(self.data) ** 0.5)
            if side == 0:
                side = 1
            # Обрезаем или дополняем до side*side
            needed = side * side
            raw = self.data[:needed] if len(self.data) >= needed else self.data.ljust(needed, b'\x00')
            img = Image.frombuffer("L", (side, side), raw, "raw", "L", 0, 1)
        else:  # rgb
            num_bytes = len(self.data)
            num_pixels = num_bytes // 3
            if num_pixels == 0:
                return
            side = int(num_pixels ** 0.5)
            if side == 0:
                side = 1
            needed = side * side * 3
            raw = self.data[:needed] if len(self.data) >= needed else self.data.ljust(needed, b'\x00')
            img = Image.frombuffer("RGB", (side, side), raw, "raw", "RGB", 0, 1)

        # Растягиваем под размер холста с билинейной интерполяцией
        img = img.resize((w, h), Image.Resampling.BILINEAR)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_resize(self, event):
        if event.widget == self.root:
            self.redraw()

def main():
    root = tk.Tk()
    app = FileImageViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
import tkinter as tk

class Overlay:
    def __init__(self):
        self.root = tk.Tk()

        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.85)

        bg = "#000001"
        self.root.configure(bg=bg)
        self.root.wm_attributes("-transparentcolor", bg)

        w, h = 260, 120
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{w}x{h}+{sw-w-20}+20")

        font_main = ("Segoe UI", 11, "bold")

        self.lbl_blinks = tk.Label(self.root, fg="#00ff00", bg=bg, font=font_main)
        self.lbl_bpm = tk.Label(self.root, fg="#00ffff", bg=bg, font=font_main)
        self.lbl_warn = tk.Label(self.root, fg="#ff3333", bg=bg, font=font_main)
        self.lbl_status = tk.Label(self.root, fg="#ffaa00", bg=bg, font=font_main)

        self.lbl_blinks.pack(anchor="w")
        self.lbl_bpm.pack(anchor="w")
        self.lbl_warn.pack(anchor="w")
        self.lbl_status.pack(anchor="w")

    def update(self, blinks, bpm, warn, status):
        self.lbl_blinks.config(text=f"Blinks: {blinks}")
        self.lbl_bpm.config(text=f"BPM: {bpm}")
        self.lbl_warn.config(text="BLINK MORE!" if warn else "")
        self.lbl_status.config(text=status if status else "")
        self.root.update_idletasks()
        self.root.update()

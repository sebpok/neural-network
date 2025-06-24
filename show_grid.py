import tkinter as tk
import numpy as np

PIXEL_SIZE = 15
GRID_SIZE = 28
PEN_WIDTH = 2  # 3 pixels wide
PEN_HEIGHT = 2  # 3 pixels high
PEN_INC = 254 / 6

class DrawGrid(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, width=PIXEL_SIZE * GRID_SIZE, height=PIXEL_SIZE * GRID_SIZE, bg='black')
        self.pack()
        self.data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.drawing = False
        self.cur_x = None
        self.cur_y = None
        self.after_id = None

        self.bind("<Button-1>", self.paint_start)
        self.bind("<B1-Motion>", self.paint_move)
        self.bind("<ButtonRelease-1>", self.paint_end)

    def paint_start(self, event):
        self.drawing = True
        self.cur_x, self.cur_y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE
        self._paint()
        self._start_timer()

    def paint_move(self, event):
        self.cur_x, self.cur_y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE
        self._paint()

    def paint_end(self, event):
        self.drawing = False
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

    def _paint(self):
        changed = False
        # Draw a cross: horizontal and vertical lines intersecting at (cur_x, cur_y)
        half_w = PEN_WIDTH // 2
        half_h = PEN_HEIGHT // 2
        for dx in range(-half_w, half_w + 1):
            nx, ny = self.cur_x + dx, self.cur_y
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                prev = self.data[ny, nx]
                if prev < 254:
                    new_val = min(254, prev + PEN_INC)
                    if new_val != prev:
                        self.data[ny, nx] = new_val
                        self.draw_pixel(nx, ny)
                        changed = True
        for dy in range(-half_h, half_h + 1):
            nx, ny = self.cur_x, self.cur_y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                prev = self.data[ny, nx]
                if prev < 254:
                    new_val = min(254, prev + PEN_INC)
                    if new_val != prev:
                        self.data[ny, nx] = new_val
                        self.draw_pixel(nx, ny)
                        changed = True

    def _start_timer(self):
        if self.drawing:
            self._paint()
            self.after_id = self.after(30, self._start_timer)

    def draw_pixel(self, x, y):
        value = self.data[y, x]
        color = f"#{value:02x}{value:02x}{value:02x}"
        self.create_rectangle(
            x * PIXEL_SIZE, y * PIXEL_SIZE, (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
            fill=color, outline=color
        )

    def clear(self):
        self.data.fill(0)
        self.delete("all")


def main():
    root = tk.Tk()
    root.title("Rysowanie Paint 28x28 (ciągłe rozjaśnianie, z printem macierzy)")
    grid = DrawGrid(root)
    print(grid.data)
    clear_btn = tk.Button(root, text="Wyczyść", command=grid.clear)
    clear_btn.pack()

    show_data_btn = tk.Button(root, text="Pokaż macierz", command=lambda: print(grid.data))
    show_data_btn.pack()
    root.mainloop()

if __name__ == "__main__":
    main()

# Generated with Tkinter Designer; do not hand-edit.
# Upstream: https://github.com/ParthJadhav/Tkinter-Designer
# Generator commit: 61ce0665ce68e255fd8bead3c14aac6c877bfa05
# BSD-3-Clause notice: third_party/Tkinter-Designer-LICENSE.txt
# Source: design/desktop.json; regenerate with tools/generate_desktop_design.py.
from tkinter import Canvas, Frame


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius, **kwargs):
    radius = max(0, min(radius, abs(x2-x1)/2, abs(y2-y1)/2))
    points = [x1+radius,y1,x2-radius,y1,x2,y1,x2,y1+radius,
              x2,y2-radius,x2,y2,x2-radius,y2,x1+radius,y2,
              x1,y2,x1,y2-radius,x1,y1+radius,x1,y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class DesignerBrand(Frame):
    """Generated decoration, not the application controller."""
    def __init__(self, master):
        super().__init__(master, bg="#142035", width=216, height=112)
        self.pack_propagate(False)
        self.canvas = Canvas(self, bg="#142035", width=216,
                             height=112, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        canvas = self.canvas
        window = self


        create_rounded_rectangle(
            canvas,
            24,
            25,
            60,
            61,
            9.0,
            fill="#397CF6",
            outline="")



        create_rounded_rectangle(
            canvas,
            33,
            34,
            51,
            52,
            3.0,
            fill="#FFFFFF",
            outline="")



        create_rounded_rectangle(
            canvas,
            36,
            37,
            48,
            49,
            1.0,
            fill="#397CF6",
            outline="")



        canvas.create_text(
            73,
            29,
            anchor="nw",
            text="imageresizer",
            fill="#FFFFFF",
            font=("Segoe UI", 19 * -1, "bold", "roman")
        )



        canvas.create_text(
            24,
            80,
            anchor="nw",
            text="ПАКЕТНАЯ ОБРАБОТКА",
            fill="#AAB7CF",
            font=("Segoe UI", 10 * -1, "bold", "roman")
        )



class DesignerDiagram(Frame):
    """Generated decoration, not the application controller."""
    def __init__(self, master):
        super().__init__(master, bg="#FFFFFF", width=232, height=116)
        self.pack_propagate(False)
        self.canvas = Canvas(self, bg="#FFFFFF", width=232,
                             height=116, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        canvas = self.canvas
        window = self


        create_rounded_rectangle(
            canvas,
            0,
            0,
            232,
            116,
            12.0,
            fill="#F2F6FF",
            outline="")



        create_rounded_rectangle(
            canvas,
            35,
            18,
            197,
            99,
            8.0,
            fill="#D8E6FF",
            outline="")



        create_rounded_rectangle(
            canvas,
            57,
            28,
            175,
            88,
            5.0,
            fill="#FFFFFF",
            outline="")



        create_rounded_rectangle(
            canvas,
            66,
            34,
            166,
            81,
            2.0,
            fill="#9ABFFA",
            outline="")



        canvas.create_text(
            93,
            40,
            anchor="nw",
            text="↗",
            fill="#174A9C",
            font=("Segoe UI", 28 * -1, "bold", "roman")
        )

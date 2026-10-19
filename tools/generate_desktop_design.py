"""Regenerate the two composable Tkinter Designer panels; no Figma API/token.

Use a reviewed checkout of Tkinter-Designer at the pinned commit below:
python tools/generate_desktop_design.py --designer-repo /path/to/Tkinter-Designer
Only batchresize/ui_generated.py is overwritten, never application logic.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PIN = '61ce0665ce68e255fd8bead3c14aac6c877bfa05'
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = '''
class {{ class_name }}(Frame):
    """Generated decoration, not the application controller."""
    def __init__(self, master):
        super().__init__(master, bg="{{ window.bg_color }}", width={{ window.width }}, height={{ window.height }})
        self.pack_propagate(False)
        self.canvas = Canvas(self, bg="{{ window.bg_color }}", width={{ window.width }},
                             height={{ window.height }}, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        canvas = self.canvas
        window = self
        {% for element in elements %}
{{ element.to_code() | indent(8, true) }}
        {% endfor %}
'''
HEADER = '''# Generated with Tkinter Designer; do not hand-edit.
# Upstream: https://github.com/ParthJadhav/Tkinter-Designer
# Generator commit: ''' + PIN + '''
# BSD-3-Clause notice: third_party/Tkinter-Designer-LICENSE.txt
# Source: design/desktop.json; regenerate with tools/generate_desktop_design.py.
from tkinter import Canvas, Frame


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius, **kwargs):
    radius = max(0, min(radius, abs(x2-x1)/2, abs(y2-y1)/2))
    points = [x1+radius,y1,x2-radius,y1,x2,y1,x2,y1+radius,
              x2,y2-radius,x2,y2,x2-radius,y2,x1+radius,y2,
              x1,y2,x1,y2-radius,x1,y1+radius,x1,y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)

'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--designer-repo', required=True, type=Path)
    args = p.parse_args()
    actual = subprocess.check_output(['git','-C',str(args.designer_repo),'rev-parse','HEAD'], text=True).strip()
    if actual != PIN:
        raise SystemExit(f'Use the reviewed generator commit {PIN}, got {actual}')
    sys.path.insert(0, str(args.designer_repo.resolve()))
    from tkdesigner.figma.frame import Frame
    from jinja2 import Template
    document = json.loads((ROOT/'design/desktop.json').read_text(encoding='utf8'))
    code = HEADER
    # Frame creates only empty asset directories for these vector/text-only nodes.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='imageresizer-designer-') as work:
        for i, component in enumerate(document['components']):
            frame = Frame(component['node'], object(), Path(work), i)
            code += Template(TEMPLATE).render(class_name=component['class_name'],
                window=frame, elements=frame.elements) + '\n'
    code='\n'.join(line.rstrip() for line in code.splitlines()).rstrip()+'\n'
    compile(code,'ui_generated.py','exec')
    target = ROOT/'batchresize/ui_generated.py'
    target.write_text(code, encoding='utf8')
    print(f'Generated {target.name}: {len(document["components"])} panels, no API requests')

if __name__ == '__main__':
    main()

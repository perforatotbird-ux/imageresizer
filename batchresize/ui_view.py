"""Adaptive desktop view. Generated Designer components have no controller logic.

Tk/ttk only: no Pillow, NumPy, CUDA, Figma token, or network at GUI import time.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from .ui_generated import DesignerBrand
from .source_quality import QUALITY_SOURCE

BG = '#F4F6FA'
INK = '#17243B'
MUTED = '#596981'
BLUE = '#2361D1'
BORDER = '#DFE5EF'
SIDE = '#142035'


def _state_images(app, style):
    def checkbox(selected=False, disabled=False):
        image=tk.PhotoImage(master=app,width=27,height=20)
        image.put('white',to=(0,0,27,20))
        border='#7B8DA7' if not disabled else '#B4BECC'
        image.put(border,to=(1,1,19,19))
        image.put('#2361D1' if selected and not disabled else '#E4E9F1' if disabled else 'white',to=(2,2,18,18))
        if selected:
            color='white' if not disabled else '#52627B'
            for x,y in [(5,9),(6,10),(7,11),(8,12),(9,11),(10,10),(11,9),(12,8),(13,7),(14,6)]:
                image.put(color,to=(x,y,x+2,y+2))
        return image
    def knob(disabled=False):
        image=tk.PhotoImage(master=app,width=20,height=20)
        for y in range(20):
            for x in range(20):
                distance=((x-9.5)**2+(y-9.5)**2)**.5
                if distance<9:
                    image.put('#2361D1' if not disabled else '#98A6BC',(x,y))
        return image
    plain,checked,disabled,disabled_checked=checkbox(),checkbox(True),checkbox(False,True),checkbox(True,True)
    ball,muted_ball=knob(),knob(True)
    app._state_images=[plain,checked,disabled,disabled_checked,ball,muted_ball]
    style.element_create('Modern.Check.indicator','image',plain,
        ('disabled','selected',disabled_checked),('disabled',disabled),('selected',checked))
    style.layout('TCheckbutton',[('Checkbutton.padding',{'sticky':'nswe','children':[
        ('Modern.Check.indicator',{'side':'left','sticky':'w'}),
        ('Checkbutton.focus',{'side':'left','sticky':'w','children':[
            ('Checkbutton.label',{'sticky':'nswe'})]})]})])
    style.element_create('Modern.Scale.slider','image',ball,('disabled',muted_ball))
    style.layout('Horizontal.TScale',[('Horizontal.Scale.trough',{'sticky':'we','children':[
        ('Modern.Scale.slider',{'side':'left','sticky':''})]})])


def configure_style(app):
    style = ttk.Style(app)
    style.theme_use('clam')
    app.option_add('*TEntry.font', ('Segoe UI', 10))
    app.option_add('*TCombobox.font', ('Segoe UI', 10))
    app.option_add('*TCombobox*Listbox.font', ('Segoe UI', 10))
    style.configure('.', font=('Segoe UI', 10), background='white', foreground=INK)
    style.configure('Shell.TFrame', background=BG)
    style.configure('Card.TFrame', background='white')
    style.configure('TLabel', background='white', foreground=INK)
    style.configure('Muted.TLabel', foreground=MUTED)
    style.configure('Title.TLabel', font=('Segoe UI', 22, 'bold'), background=BG)
    style.configure('Subtitle.TLabel', foreground=MUTED, background=BG)
    style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'))
    style.configure('Eyebrow.TLabel', foreground=MUTED, font=('Segoe UI', 9, 'bold'))
    style.configure('Metric.TLabel', font=('Segoe UI', 17, 'bold'))
    style.configure('TEntry', fieldbackground='#FFFFFF', bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, padding=(11, 10))
    style.map('TEntry', bordercolor=[('focus', BLUE)], fieldbackground=[('disabled','#F3F5F8')])
    style.configure('TCombobox', fieldbackground='#FFFFFF', background='#F7F9FC',
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    arrowcolor=MUTED, arrowsize=14, padding=(10, 9))
    style.map('TCombobox', fieldbackground=[('readonly','white'),('disabled','#F3F5F8')],
              foreground=[('readonly',INK),('disabled',MUTED)])
    style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=(14, 10),
                    background='#F4F7FC', foreground=INK, bordercolor=BORDER,
                    lightcolor='#F4F7FC', darkcolor='#F4F7FC', focusthickness=2, focuscolor=BLUE)
    style.map('TButton', background=[('pressed','#DFE8F7'),('active','#E9F0FC')],
              foreground=[('disabled','#748097')])
    style.configure('Primary.TButton', background=BLUE, foreground='white',
                    bordercolor=BLUE, lightcolor=BLUE, darkcolor=BLUE, padding=(22, 12),
                    focuscolor='#A1C1FF')
    style.map('Primary.TButton', background=[('disabled','#6B84AE'),('pressed','#174BBA'),('active','#194FAF')],
              foreground=[('disabled','white'),('active','white')])
    style.configure('TCheckbutton', background='white', foreground=INK, padding=(0,5),
                    indicatorcolor='white', indicatorbackground='white', indicatorsize=17, wraplength=360)
    style.map('TCheckbutton', background=[('active','white')], indicatorbackground=[('selected',BLUE)])
    style.configure('TRadiobutton', background='white', padding=(0,5), indicatorsize=17, wraplength=360)
    style.map('TRadiobutton', background=[('active','white')])
    style.configure('Horizontal.TProgressbar', troughcolor='#E9EDF5', background=BLUE,
                    bordercolor='#E9EDF5', lightcolor=BLUE, darkcolor=BLUE, thickness=7)
    style.configure('Horizontal.TScale', background='white', troughcolor='#E6ECF6')
    _state_images(app, style)
    style.configure('Vertical.TScrollbar', background='#D5DDEA', troughcolor='white',
                    bordercolor='white', arrowcolor=MUTED, arrowsize=11)


class ScrollPage(ttk.Frame):
    """Per-page vertical scrolling, including keyboard/trackpad-friendly bindings."""
    def __init__(self, master):
        super().__init__(master, style='Card.TFrame')
        self.canvas = tk.Canvas(self, background='white', highlightthickness=0, bd=0)
        self.scroll = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas, padding=(24,22,18,24))
        self.window_id = self.canvas.create_window(0,0,window=self.body,anchor='nw')
        self.canvas.bind('<Configure>', self._resize)
        self.body.bind('<Configure>', lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.body.columnconfigure(0,weight=1)

    def _resize(self, event):
        self.canvas.itemconfigure(self.window_id,width=event.width)


class ModernView:
    PAGES = [('Папки', 'Источник и назначение'), ('Размер', 'Геометрия изображения'),
             ('JPEG', 'Качество и сохранение'), ('Движок', 'Ресурсы и надёжность')]

    def __init__(self, app, sizing_modes, quality_choices, subsampling_choices, size_presets):
        self.app = app
        self.sizing_modes, self.quality_choices = sizing_modes, quality_choices
        self.subsampling_choices, self.size_presets = subsampling_choices, size_presets
        self.controls = []
        self._summary_pending = False
        configure_style(app)
        app.configure(bg=BG)
        app.rowconfigure(0,weight=1)
        app.columnconfigure(1,weight=1)
        side = tk.Frame(app,bg=SIDE,width=216)
        side.grid(row=0,column=0,sticky='ns')
        side.grid_propagate(False)
        side.columnconfigure(0,weight=1)
        side.rowconfigure(6,weight=1)
        DesignerBrand(side).grid(row=0,column=0,sticky='ew',pady=(6,24))
        self.nav = []
        for i,(title,subtitle) in enumerate(self.PAGES):
            b=tk.Button(side,text=f'{i+1:02d}   {title}',anchor='w',bg=SIDE,fg='#B7C4DB',
                activebackground='#263B5E',activeforeground='white',relief='flat',bd=0,
                highlightthickness=2,highlightbackground=SIDE,highlightcolor='#8FBAFF',
                font=('Segoe UI',11,'bold'),padx=20,pady=14,cursor='hand2',
                command=lambda j=i:self.show_page(j),takefocus=True)
            b.grid(row=i+1,column=0,sticky='ew',padx=14,pady=3)
            self.nav.append(b)
        self.side_note=tk.Label(side,text='ЛОКАЛЬНО, БЕЗ ОБЛАКА\n\nФайлы остаются\nна вашем компьютере.',bg=SIDE,fg='#AAB7CF',justify='left',
                                anchor='w',font=('Segoe UI',9),wraplength=168,padx=24,pady=20)
        self.side_note.grid(row=7,column=0,sticky='ew')
        tk.Label(side,text='DESKTOP  /  UI 01',bg=SIDE,fg='#AAB7CF',anchor='w',
                 font=('Segoe UI',9,'bold'),padx=24,pady=20).grid(row=8,column=0,sticky='ew')

        main=ttk.Frame(app,padding=(28,24,28,20),style='Shell.TFrame')
        main.grid(row=0,column=1,sticky='nsew')
        main.columnconfigure(0,weight=1)
        main.rowconfigure(1,weight=1)
        header=ttk.Frame(main,style='Shell.TFrame')
        header.grid(row=0,column=0,sticky='ew',pady=(0,22))
        ttk.Label(header,text='Пакетная обработка',style='Title.TLabel').pack(anchor='w')
        ttk.Label(header,text='Меньше размер. Больше возможностей.',style='Subtitle.TLabel').pack(anchor='w',pady=(6,0))
        body=ttk.Frame(main,style='Shell.TFrame')
        body.grid(row=1,column=0,sticky='nsew')
        body.rowconfigure(0,weight=1)
        body.columnconfigure(0,weight=1)
        self.settings=tk.Frame(body,bg='white',highlightbackground=BORDER,highlightthickness=1)
        self.settings.grid(row=0,column=0,sticky='nsew',padx=(0,18))
        self.settings.rowconfigure(0,weight=1);self.settings.columnconfigure(0,weight=1)
        self.pages=[]
        for i in range(4):
            p=ScrollPage(self.settings);p.grid(row=0,column=0,sticky='nsew');self.pages.append(p)
        self._input(self.pages[0].body)
        self._resize_page(self.pages[1].body)
        self._jpeg(self.pages[2].body)
        self._engine(self.pages[3].body)
        self._summary(body)
        self._footer(main)
        self.current_page=0
        self.show_page(0)
        for var in [app.width_var,app.height_var,app.sizing_var,app.sizing_value_var,
                    app.mode_var,app.allow_upscale_var,app.jpeg_quality_var,app.backend_var,
                    app.filter_var,app.bg_color_var,app.overwrite_var,app.skip_existing_var,
                    app.max_file_size_var,app.min_quality_var,app.quality_var]:
            var.trace_add('write',self._schedule_summary)
        app.bind('<MouseWheel>',self._wheel,add='+')
        app.bind('<Button-4>',lambda e:self._wheel(e,-3),add='+')
        app.bind('<Button-5>',lambda e:self._wheel(e,3),add='+')
        app.bind('<Control-Return>',lambda e:app._start(),add='+')
        self.update_summary()
        self.set_running(False)

    def show_page(self,i):
        self.current_page=i
        self.pages[i].tkraise()
        for j,b in enumerate(self.nav):
            b.configure(bg='#263B5E' if i==j else SIDE,fg='white' if i==j else '#B7C4DB')

    def _wheel(self,event,units=None):
        # Scroll only the active settings page, never combobox popups/log/summary.
        if isinstance(event.widget,(ttk.Combobox,tk.Text)):
            return
        page=(self.summary_page if str(event.widget).startswith(str(self.summary_page))
              else self.pages[self.current_page])
        try:
            if not str(event.widget).startswith(str(page)):
                return
        except tk.TclError:
            return
        if units is None:
            if not event.delta:return
            units=-1 if event.delta>0 else 1
            if abs(event.delta)>=120:units=-int(event.delta/120)*3
        if page.body.winfo_height()>page.canvas.winfo_height():
            page.canvas.yview_scroll(units,'units')

    def heading(self,p,number,title,subtitle):
        ttk.Label(p,text=f'{number}  /  НАСТРОЙКИ',style='Eyebrow.TLabel').grid(row=0,column=0,sticky='w')
        ttk.Label(p,text=title,style='Heading.TLabel').grid(row=1,column=0,sticky='w',pady=(8,6))
        ttk.Label(p,text=subtitle,style='Muted.TLabel',wraplength=390).grid(row=2,column=0,sticky='w',pady=(0,10))

    def label(self,p,row,text):
        ttk.Label(p,text=text,font=('Segoe UI',10,'bold')).grid(row=row,column=0,sticky='w',pady=(8,7))

    def control(self,w):
        self.controls.append(w);return w

    def entry(self,p,row,var,width=None):
        w=self.control(ttk.Entry(p,textvariable=var,width=width or 16))
        w.grid(row=row,column=0,sticky='ew');return w

    def combo(self,p,row,values,variable=None):
        w=self.control(ttk.Combobox(p,state='readonly',values=values,textvariable=variable,width=12))
        w.grid(row=row,column=0,sticky='ew');return w

    def check(self,p,row,text,var):
        w=self.control(ttk.Checkbutton(p,text=text,variable=var));w.grid(row=row,column=0,sticky='w',pady=2);return w

    def _input(self,p):
        a=self.app
        self.heading(p,'01','Начнём с ваших файлов','Выберите профиль, исходную папку и место для готовых JPEG.')
        self.label(p,3,'Профиль задачи')
        a.profile_combo=self.control(ttk.Combobox(
            p,state='readonly',values=a._profile_labels(),textvariable=a.profile_var))
        a.profile_combo.grid(row=4,column=0,sticky='ew')
        a.profile_combo.bind('<<ComboboxSelected>>',a._on_profile_selected)
        profile_buttons=ttk.Frame(p)
        profile_buttons.grid(row=5,column=0,sticky='ew',pady=(8,4))
        profile_buttons.columnconfigure((0,1),weight=1)
        for index,(text,command) in enumerate([
                ('Сохранить',a._save_profile),('Дублировать',a._duplicate_profile),
                ('Переименовать',a._rename_profile),('Удалить',a._delete_profile),
                ('Импорт…',a._import_profiles),('Экспорт…',a._export_profile)]):
            self.control(ttk.Button(profile_buttons,text=text,command=command)).grid(
                row=index//2,column=index%2,sticky='ew',padx=(0,6),pady=2)
        ttk.Label(p,textvariable=a.profile_status_var,style='Muted.TLabel',
                  wraplength=380).grid(row=6,column=0,sticky='w',pady=(2,8))
        for row,label,var in [(7,'Исходная папка',a.input_var),(9,'Папка результатов',a.output_var)]:
            self.label(p,row,label)
            line=ttk.Frame(p);line.grid(row=row+1,column=0,sticky='ew');line.columnconfigure(0,weight=1)
            self.control(ttk.Entry(line,textvariable=var,width=12)).grid(row=0,column=0,sticky='ew',padx=(0,8))
            self.control(ttk.Button(line,text='Обзор…',command=lambda v=var:a._browse(v))).grid(row=0,column=1)
        info=tk.Frame(p,bg='#EEF4FF',padx=14,pady=12)
        info.grid(row=11,column=0,sticky='ew',pady=(12,8))
        tk.Label(info,text='Оригиналы — отдельно',font=('Segoe UI',10,'bold'),fg='#234A87',bg='#EEF4FF',anchor='w').pack(anchor='w')
        tk.Label(info,text='Подпапки сканируются автоматически.\nСтруктура каталогов сохраняется в результате.',font=('Segoe UI',9),fg='#335687',bg='#EEF4FF',justify='left').pack(anchor='w',pady=(5,0))
        self.label(p,12,'Лимит файлов')
        line=ttk.Frame(p);line.grid(row=13,column=0,sticky='ew')
        self.control(ttk.Entry(line,textvariable=a.limit_var,width=9)).pack(side='left')
        ttk.Label(line,text='Пусто — обработать все',style='Muted.TLabel').pack(side='left',padx=12)
        self.control(ttk.Button(p,text='Посчитать изображения',command=a._count_files)).grid(row=14,column=0,sticky='w',pady=(12,8))
        ttk.Label(p,textvariable=a.recursive_note,style='Muted.TLabel',wraplength=380).grid(row=15,column=0,sticky='w')
        nav_box = ttk.Frame(p); nav_box.grid(row=16, column=0, sticky='ew', pady=(18,4))
        ttk.Button(nav_box, text='Далее: Размер и геометрия  →', style='Primary.TButton', command=lambda: self.show_page(1)).pack(side='right')

    def _resize_page(self,p):
        a=self.app
        self.heading(p,'02','Размер и геометрия','Настройте способ масштабирования, границы и способ вписывания.')
        self.label(p,3,'Способ изменения размера')
        a.sizing_combo=self.combo(p,4,[x[1] for x in self.sizing_modes]);a.sizing_combo.current(0)
        a.sizing_combo.bind('<<ComboboxSelected>>',a._on_sizing_change)
        self.sizing_mode_hint=ttk.Label(p,text='',style='Muted.TLabel',wraplength=380,font=('Segoe UI',9,'bold'))
        self.sizing_mode_hint.grid(row=5,column=0,sticky='w',pady=(2,6))
        self.preset_label=ttk.Label(p,text='Пресет размера (для режима по границам)',font=('Segoe UI',10,'bold'))
        self.preset_label.grid(row=6,column=0,sticky='w',pady=(4,2))
        a.preset_combo=self.combo(p,7,[*self.size_presets,'custom']);a.preset_combo.set('web_1024')
        a.preset_combo.bind('<<ComboboxSelected>>',a._on_preset_change)
        dims=ttk.Frame(p);dims.grid(row=8,column=0,sticky='ew',pady=(8,0));dims.columnconfigure((0,2),weight=1)
        ttk.Label(dims,text='Ширина, px',style='Muted.TLabel').grid(row=0,column=0,sticky='w',pady=(0,4))
        self.width_entry=self.control(ttk.Entry(dims,textvariable=a.width_var,width=10))
        self.width_entry.grid(row=1,column=0,sticky='ew')
        ttk.Label(dims,text='×').grid(row=1,column=1,padx=10)
        ttk.Label(dims,text='Высота, px',style='Muted.TLabel').grid(row=0,column=2,sticky='w',pady=(0,4))
        self.height_entry=self.control(ttk.Entry(dims,textvariable=a.height_var,width=10))
        self.height_entry.grid(row=1,column=2,sticky='ew')
        self.sizing_value_label=ttk.Label(p,text='Значение масштаба',font=('Segoe UI',10,'bold'))
        self.sizing_value_label.grid(row=9,column=0,sticky='w',pady=(10,2))
        self.sizing_value_entry=self.entry(p,10,a.sizing_value_var)
        self.label(p,11,'Способ вписывания в рамку')
        fit_modes=[
            (12,'Contain — вписать целиком без обрезки (пропорции сохраняются)','contain'),
            (13,'Cover — заполнить рамку и обрезать излишки (под точный размер)','cover'),
            (14,'Pad — вписать целиком и дополнить канву полями до размера','pad')
        ]
        for row,text,val in fit_modes:
            self.control(ttk.Radiobutton(p,text=text,variable=a.mode_var,value=val,command=a._sync_fit_mode)).grid(row=row,column=0,sticky='w')
        anchor_frame=ttk.Frame(p);anchor_frame.grid(row=15,column=0,sticky='ew',pady=(6,2))
        ttk.Label(anchor_frame,text='Точка привязки кадрирования / полей:',style='Muted.TLabel').pack(side='left',padx=(0,8))
        a.crop_anchor_combo=self.control(ttk.Combobox(
            anchor_frame,state='readonly',width=14,textvariable=a.crop_anchor_var,
            values=['center','top','bottom','left','right','top-left','top-right','bottom-left','bottom-right']))
        a.crop_anchor_combo.pack(side='left')
        self.crop_anchor_hint=ttk.Label(p,text='',style='Muted.TLabel',wraplength=380)
        self.crop_anchor_hint.grid(row=16,column=0,sticky='w')
        self.check(p,17,'Разрешить увеличение маленьких фото (upscale)',a.allow_upscale_var)
        sharpen_frame=ttk.Frame(p);sharpen_frame.grid(row=18,column=0,sticky='ew',pady=(8,2))
        ttk.Label(sharpen_frame,text='Резкость контуров (Unsharp Mask):',style='Muted.TLabel').pack(side='left',padx=(0,8))
        a.sharpen_combo=self.control(ttk.Combobox(
            sharpen_frame,state='readonly',width=12,textvariable=a.sharpen_preset_var,
            values=['off','weak','standard']))
        a.sharpen_combo.pack(side='left')
        self.label(p,19,'Фильтр ресайза (алгоритм интерполяции)')
        self.combo(p,20,['lanczos3','mitchell','area','bilinear','nearest','box'],a.filter_var)
        ttk.Label(p,text='lanczos3 — наивысшая резкость фото; mitchell — мягкий без ореолов; area — чистое уменьшение.',style='Muted.TLabel',wraplength=380).grid(row=21,column=0,sticky='w',pady=(4,8))
        nav_box=ttk.Frame(p);nav_box.grid(row=22,column=0,sticky='ew',pady=(14,4))
        ttk.Button(nav_box,text='← Назад: Папки',command=lambda:self.show_page(0)).pack(side='left')
        ttk.Button(nav_box,text='Далее: Качество JPEG  →',style='Primary.TButton',command=lambda:self.show_page(2)).pack(side='right')

    def _jpeg(self,p):
        a=self.app
        self.heading(p,'03','Качество и сохранение','Выходной формат — JPEG. Баланс качества, размера и метаданных.')
        self.label(p,3,'Пресет качества')
        a.quality_combo=self.combo(p,4,[x[1] for x in self.quality_choices])
        default_index=next((i for i,(key,_) in enumerate(self.quality_choices)
                            if key=='balanced'),2)
        a.quality_combo.current(default_index)
        a.quality_combo.bind('<<ComboboxSelected>>',a._on_quality_change)
        self.label(p,5,'Шкала качества JPEG (1–100)')
        row=ttk.Frame(p);row.grid(row=6,column=0,sticky='ew');row.columnconfigure(0,weight=1)
        a.quality_scale=self.control(ttk.Scale(row,from_=1,to=100,orient='horizontal',command=a._on_quality_scale))
        a.quality_scale.grid(row=0,column=0,sticky='ew',padx=(0,16));a.quality_scale.set(80)
        self.quality_metric_label=ttk.Label(row,textvariable=a.jpeg_quality_var,style='Metric.TLabel',width=5)
        self.quality_metric_label.grid(row=0,column=1)
        self.label(p,7,'Цветовая субдискретизация (Chroma Subsampling)')
        a.subsampling_combo=self.combo(p,8,[x[1] for x in self.subsampling_choices]);a.subsampling_combo.current(0)
        a.subsampling_combo.bind('<<ComboboxSelected>>',a._on_subsampling_change)
        self.label(p,9,'Ограничение размера файла (Target Size Limit)')
        limit_row=ttk.Frame(p);limit_row.grid(row=10,column=0,sticky='ew')
        self.max_file_size_entry=self.control(ttk.Entry(limit_row,textvariable=a.max_file_size_var,width=12))
        self.max_file_size_entry.pack(side='left')
        ttk.Label(limit_row,text='Пусто — без ограничения (например, 500KB, 1.5MB)',style='Muted.TLabel').pack(side='left',padx=12)
        minq_frame=ttk.Frame(p);minq_frame.grid(row=11,column=0,sticky='ew',pady=(6,2))
        self.min_quality_label=ttk.Label(minq_frame,text='Минимальное качество для подбора:')
        self.min_quality_label.pack(side='left',padx=(0,8))
        self.min_quality_entry=self.control(ttk.Entry(minq_frame,textvariable=a.min_quality_var,width=6))
        self.min_quality_entry.pack(side='left')
        self.min_quality_hint=ttk.Label(p,text='',style='Muted.TLabel',wraplength=380)
        self.min_quality_hint.grid(row=12,column=0,sticky='w')
        self.allow_downscale_check=self.check(p,13,'Разрешить уменьшение разрешения при недостижимости лимита',a.allow_downscale_var)
        self.label(p,14,'Цвет канвы и фона (#RRGGBB)')
        self.entry(p,15,a.bg_color_var)
        ttk.Label(p,text='Используется для прозрачных PNG/WebP и для полей канвы в режиме Pad.',style='Muted.TLabel',wraplength=380).grid(row=16,column=0,sticky='w')
        self.label(p,17,'Метаданные и цветовой профиль')
        self.check(p,18,'Сохранять EXIF с нормализацией ориентации',a.exif_var)
        self.check(p,19,'Сохранять цветовой профиль ICC',a.icc_var)
        self.check(p,20,'Очищать персональные данные (GPS, серийные номера, авторство)',a.strip_personal_var)
        self.check(p,21,'Преобразовывать цвета в Web sRGB (для интернет-публикаций)',a.web_srgb_var)
        self.check(p,22,'Линейный цвет (Linear-light resampling & blending)',a.linear_light_var)
        self.label(p,23,'Существующие результаты в целевой папке')
        for row,text,val in [
            (24,'Перезаписывать существующие файлы при повторном запуске','overwrite'),
            (25,'Пропускать уже готовые файлы (с проверкой хеша и структуры)','skip')]:
            self.control(ttk.Radiobutton(p,text=text,variable=a.write_mode_var,value=val,command=a._sync_write_mode)).grid(row=row,column=0,sticky='w')
        nav_box=ttk.Frame(p);nav_box.grid(row=26,column=0,sticky='ew',pady=(14,4))
        ttk.Button(nav_box,text='← Назад: Размер',command=lambda:self.show_page(1)).pack(side='left')
        ttk.Button(nav_box,text='Далее: Движок и ресурсы  →',style='Primary.TButton',command=lambda:self.show_page(3)).pack(side='right')

    def _engine(self,p):
        import os
        a=self.app
        cpu_cores = os.cpu_count() or 8
        self.heading(p,'04','Движок и ресурсы','Выбор маршрута обработки, потоков и аппаратных ресурсов.')
        self.label(p,3,'Backend (движок выполнения)')
        self.combo(p,4,['auto','gpu','hybrid','cpu'],a.backend_var)
        ttk.Label(p,text='auto — автовыбор GPU при наличии (CUDA), иначе CPU. hybrid — декод CPU, ресайз GPU.',style='Muted.TLabel',wraplength=380).grid(row=5,column=0,sticky='w',pady=(4,0))
        self.label(p,6,'Многопоточность и очереди конвейера')
        rows=ttk.Frame(p);rows.grid(row=7,column=0,sticky='ew',pady=(8,0));rows.columnconfigure(1,weight=1)
        grid_items = [
            ('Потоки обработки (Processors / CPU ядра)', getattr(a, 'processors_var', a.workers_var), f'Пусто — Авто (CPU: {min(8, cpu_cores)} воркеров, GPU: 1 координатор). В системе: {cpu_cores} ядер'),
            ('Потоки чтения с диска (Readers)', getattr(a, 'readers_var', a.workers_var), 'Чтение исходных файлов с накопителя в память'),
            ('Потоки записи на диск (Writers)', getattr(a, 'writers_var', a.workers_var), 'Атомарная запись готовых результатов на диск'),
            ('Размер GPU batch (параллелизм GPU)', a.batch_var, 'Сколько фото видеокарта обрабатывает одновременно на CUDA'),
            ('Глубина очереди между стадиями', a.queue_var, 'Буфер задач между чтением, обработкой и записью'),
            ('Оценочный VRAM бюджет, GiB', a.vram_var, 'Мягкий ориентир памяти GPU для формирования пачек'),
        ]
        for i,(label,var,hint) in enumerate(grid_items):
            ttk.Label(rows,text=label).grid(row=i*2,column=0,sticky='w',padx=(0,12),pady=(4,1))
            self.control(ttk.Entry(rows,textvariable=var,width=8)).grid(row=i*2,column=1,sticky='ew',pady=(4,1))
            ttk.Label(rows,text=hint,style='Muted.TLabel',wraplength=380).grid(row=i*2+1,column=0,columnspan=2,sticky='w',pady=(0,5))
        note_frame = ttk.Frame(p)
        note_frame.grid(row=8, column=0, sticky='ew', pady=(8, 8))
        ttk.Label(note_frame, text=f'ℹ Как работает многопоточность:\n'
                                   f'• В режиме CPU обработка масштабируется по ядрам процессора (доступно ядер: {cpu_cores}).\n'
                                   '• В режиме GPU 1 поток-координатор отправляет пачки в CUDA, где ядра видеокарты работают параллельно.',
                  style='Muted.TLabel', wraplength=380).pack(anchor='w')
        self.label(p,9,'Надёжность')
        self.check(p,10,'CPU fallback при отказе GPU',a.fallback_var)
        self.check(p,11,'Пытаться читать усечённые JPEG',a.truncated_var)
        self.check(p,12,'Оптимизировать таблицы Huffman',a.huffman_var)
        self.check(p,13,'fsync перед публикацией результата',a.fsync_var)
        ttk.Label(p,text=f'Python рабочего процесса:\n{a.worker}',style='Muted.TLabel',wraplength=390).grid(row=14,column=0,sticky='w',pady=(8,0))
        nav_box=ttk.Frame(p);nav_box.grid(row=15,column=0,sticky='ew',pady=(14,4))
        ttk.Button(nav_box,text='← Назад: Качество JPEG',command=lambda:self.show_page(2)).pack(side='left')

    def _summary(self,parent):
        a=self.app
        card=tk.Frame(parent,bg='white',highlightbackground=BORDER,highlightthickness=1,width=322)
        card.grid(row=0,column=1,sticky='ns');card.grid_propagate(False)
        card.columnconfigure(0,weight=1);card.rowconfigure(0,weight=1)
        self.summary_page=ScrollPage(card);self.summary_page.grid(row=0,column=0,sticky='nsew')
        body=self.summary_page.body;body.configure(padding=(16,18,16,16))
        ttk.Label(body,text='ПРЕВЬЮ',style='Eyebrow.TLabel').pack(anchor='w')
        title=ttk.Frame(body);title.pack(fill='x',pady=(8,10))
        ttk.Label(title,text='Настоящий JPEG',font=('Segoe UI',12,'bold')).pack(side='left')
        self._preview_data=None;self._preview_images={}
        self.preview_canvas=tk.Canvas(body,height=190,background='#101827',
            highlightthickness=1,highlightbackground='#26324A',bd=0)
        self.preview_canvas.pack(fill='x',pady=(0,8))
        self.preview_canvas.bind('<Configure>',self._preview_resized)
        self.preview_canvas.bind('<ButtonPress-1>',lambda e:self.preview_canvas.scan_mark(e.x,e.y))
        self.preview_canvas.bind('<B1-Motion>',lambda e:self.preview_canvas.scan_dragto(e.x,e.y,gain=1))
        self.clear_preview()
        self.preview_choose=self.control(ttk.Button(body,text='Выбрать изображение…',command=a._choose_preview_file))
        self.preview_choose.pack(fill='x',pady=(0,8))
        mode_row=ttk.Frame(body);mode_row.pack(fill='x',pady=(0,6))
        ttk.Label(mode_row,text='Показ',style='Muted.TLabel').pack(side='left')
        self.preview_mode_combo=self.control(ttk.Combobox(mode_row,state='readonly',
            values=['Результат','Оригинал','До / после'],textvariable=a.preview_mode_var,width=13))
        self.preview_mode_combo.pack(side='right')
        view_row=ttk.Frame(body);view_row.pack(fill='x',pady=(0,8))
        self.preview_fit_button=self.control(ttk.Button(view_row,text='Вписать',command=lambda:a._set_preview_view('fit')))
        self.preview_fit_button.pack(side='left')
        self.preview_actual_button=self.control(ttk.Button(view_row,text='100%',command=lambda:a._set_preview_view('actual')))
        self.preview_actual_button.pack(side='left',padx=(6,0))
        ttk.Label(body,textvariable=a.preview_status_var,style='Muted.TLabel',
            font=('Segoe UI',9),wraplength=276).pack(anchor='w',pady=(0,4))
        ttk.Label(body,textvariable=a.preview_info_var,font=('Segoe UI',9,'bold'),
            wraplength=276).pack(anchor='w',pady=(0,2))
        ttk.Label(body,textvariable=a.preview_meta_var,style='Muted.TLabel',
            font=('Segoe UI',9),wraplength=276).pack(anchor='w',pady=(0,8))
        self.set_preview_view('fit')
        self.summary_size=tk.StringVar();self.summary_quality=tk.StringVar();self.summary_mode=tk.StringVar();self.summary_backend=tk.StringVar();self.summary_write=tk.StringVar()
        for label,var in [('ГРАНИЦЫ / МАСШТАБ',self.summary_size),('КАЧЕСТВО',self.summary_quality),('ГЕОМЕТРИЯ',self.summary_mode),('ДВИЖОК',self.summary_backend)]:
            ttk.Separator(body).pack(fill='x',pady=(0,6))
            ttk.Label(body,text=label,style='Eyebrow.TLabel').pack(anchor='w')
            ttk.Label(body,textvariable=var,font=('Segoe UI',11,'bold'),wraplength=276).pack(anchor='w',pady=(5,6))
        ttk.Label(body,textvariable=self.summary_write,style='Muted.TLabel',wraplength=276).pack(anchor='w',pady=(4,0))

    # -- real preview panel --------------------------------------------------
    def clear_preview(self,text='Выберите изображение для превью'):
        self._preview_data=None;self._preview_images={}
        canvas=self.preview_canvas;canvas.delete('all');canvas.configure(scrollregion='')
        width=max(canvas.winfo_width(),240)
        canvas.create_text(width/2,95,text=text,fill='#93A4C3',width=250,
            justify='center',font=('Segoe UI',9))

    def _preview_resized(self,event=None):
        if self._preview_data is None:
            self.clear_preview()
        else:
            self._draw_preview()
        self.app._on_preview_resize(event)

    def show_preview(self,data):
        images={}
        for key in ('original','result'):
            path=(data.get(key) or {}).get('path')
            if not path:
                return False
            try:
                images[key]=tk.PhotoImage(master=self.app,file=path)
            except tk.TclError:
                return False
        self._preview_images=images;self._preview_data=data
        self._draw_preview()
        return True

    def _draw_preview(self):
        canvas=self.preview_canvas;data=self._preview_data
        canvas.delete('all')
        if not data:
            self.clear_preview();return
        cw=max(canvas.winfo_width(),120);ch=max(canvas.winfo_height(),120)
        mode=self.app.preview_mode_var.get()
        if mode=='До / после':
            half=cw/2
            for key,label,x0 in (('original','До',0),('result','После',half)):
                image=self._preview_images.get(key)
                if image is None:continue
                x=x0+(half-image.width())/2;y=(ch-image.height())/2
                canvas.create_image(x,y,image=image,anchor='nw')
                canvas.create_text(x0+8,8,text=label,fill='#E7EEFB',anchor='nw',
                    font=('Segoe UI',9,'bold'))
            canvas.create_line(half,10,half,ch-10,fill='#3A4B6C')
            canvas.configure(scrollregion=(0,0,cw,ch))
            return
        key='original' if mode=='Оригинал' else 'result'
        image=self._preview_images.get(key)
        if image is None:return
        if self.app.preview_view_var.get()=='actual' and (image.width()>cw or image.height()>ch):
            canvas.configure(scrollregion=(0,0,max(cw,image.width()),max(ch,image.height())))
            canvas.create_image(0,0,image=image,anchor='nw')
            canvas.xview_moveto(0);canvas.yview_moveto(0)
        else:
            canvas.configure(scrollregion=(0,0,cw,ch))
            canvas.create_image(cw/2,ch/2,image=image,anchor='center')

    def set_preview_view(self,view):
        before=self.app.preview_mode_var.get()=='До / после'
        if before:view='fit'
        self.app.preview_view_var.set(view)
        self.preview_fit_button.configure(text='Вписать ✓' if view=='fit' else 'Вписать')
        self.preview_actual_button.configure(text='100% ✓' if view=='actual' else '100%')
        self.preview_actual_button.state(['disabled'] if before else ['!disabled'])
        if self._preview_data is not None:
            self._draw_preview()

    def _footer(self,parent):
        a=self.app
        panel=tk.Frame(parent,bg='white',highlightbackground=BORDER,highlightthickness=1)
        panel.grid(row=2,column=0,sticky='ew',pady=(18,0))
        panel.columnconfigure(0,weight=1)
        body=ttk.Frame(panel,padding=(20,16));body.grid(row=0,column=0,sticky='ew');body.columnconfigure(0,weight=1)
        left=ttk.Frame(body);left.grid(row=0,column=0,sticky='ew',padx=(0,16));left.columnconfigure(0,weight=1)
        self.status_label=ttk.Label(left,textvariable=a.status_var,font=('Segoe UI',10,'bold'),wraplength=450)
        self.status_label.grid(row=0,column=0,sticky='w')
        a.progress=ttk.Progressbar(left,mode='determinate',maximum=100)
        a.progress.grid(row=1,column=0,sticky='ew',pady=(10,8))
        self.metrics=tk.StringVar(value='0 файлов  ·  Готово к запуску')
        self.metrics_label=ttk.Label(left,textvariable=self.metrics,style='Muted.TLabel',font=('Segoe UI',9))
        self.metrics_label.grid(row=2,column=0,sticky='w')
        actions=ttk.Frame(body);actions.grid(row=0,column=1,sticky='e')
        self.cancel_button=ttk.Button(actions,text='Остановить',command=a._cancel)
        self.cancel_button.pack(side='left',padx=(0,8))
        self.start_button=ttk.Button(actions,text='Начать обработку  →',style='Primary.TButton',command=a._start)
        self.start_button.pack(side='left')
        tools=ttk.Frame(parent,style='Shell.TFrame');tools.grid(row=3,column=0,sticky='ew',pady=(10,0))
        self.log_button=ttk.Button(tools,text='Журнал событий  +',command=self.toggle_log)
        self.log_button.grid(row=0,column=0,sticky='w')
        self.output_button=ttk.Button(tools,text='Открыть результаты ↗',command=a._open_output)
        self.output_button.grid(row=0,column=1,sticky='w',padx=8)
        self.doctor_button=ttk.Button(tools,text='Проверить окружение',command=a._doctor)
        self.doctor_button.grid(row=0,column=2,sticky='e')
        tools.columnconfigure(2,weight=1)
        self._footer_stack = None
        self._toolbar_stack = None
        body.bind('<Configure>',lambda e:self._reflow_footer(e, left, actions),add='+')
        tools.bind('<Configure>',lambda e:self._reflow_tools(e),add='+')
        self.log_panel=tk.Frame(parent,bg='white',highlightbackground=BORDER,highlightthickness=1)
        self.log_panel.columnconfigure(0,weight=1);self.log_panel.rowconfigure(1,weight=1)
        log_header = tk.Frame(self.log_panel, bg='white')
        log_header.grid(row=0, column=0, columnspan=2, sticky='ew', padx=8, pady=(4, 4))
        log_header.columnconfigure(0, weight=1)
        ttk.Label(log_header, textvariable=a.threads_var, style='Muted.TLabel').grid(row=0, column=0, sticky='w')
        log_btn_box = ttk.Frame(log_header)
        log_btn_box.grid(row=0, column=1, sticky='e')
        self.copy_log_button = ttk.Button(log_btn_box, text='Копировать лог', command=a._copy_log)
        self.copy_log_button.pack(side='left', padx=(0, 4))
        self.save_log_button = ttk.Button(log_btn_box, text='Сохранить в файл…', command=a._save_log)
        self.save_log_button.pack(side='left', padx=(0, 4))
        self.clear_log_button = ttk.Button(log_btn_box, text='Очистить', command=a._clear_log)
        self.clear_log_button.pack(side='left')
        a.log=tk.Text(self.log_panel,height=7,wrap='word',state='disabled',bg='#17243B',fg='#E0E8F7',
                       insertbackground='white',selectbackground='#365889',bd=0,padx=12,pady=10,font=('Consolas',9))
        a.log.grid(row=1,column=0,sticky='nsew')
        scroll=ttk.Scrollbar(self.log_panel,command=a.log.yview);scroll.grid(row=1,column=1,sticky='ns');a.log.configure(yscrollcommand=scroll.set)
        self.log_visible=False

    def _reflow_footer(self,event,left,actions):
        scale=max(1.,float(self.app.tk.call('tk','scaling'))/1.3333)
        stacked=event.width < actions.winfo_reqwidth()+260*scale+16
        if stacked != self._footer_stack:
            self._footer_stack=stacked
            left.grid_configure(row=0,column=0,columnspan=2 if stacked else 1,padx=(0,0 if stacked else 16))
            actions.grid_configure(row=1 if stacked else 0,column=0 if stacked else 1,
                columnspan=2 if stacked else 1,sticky='e',pady=(12,0) if stacked else 0)
        wrap=max(180,event.width-(0 if stacked else actions.winfo_reqwidth()+16))
        self.status_label.configure(wraplength=wrap)
        self.metrics_label.configure(wraplength=wrap)

    def _reflow_tools(self,event):
        required=sum(b.winfo_reqwidth() for b in [self.log_button,self.output_button,self.doctor_button])+16
        stacked=required>event.width
        if stacked!=self._toolbar_stack:
            self._toolbar_stack=stacked
            self.doctor_button.grid_configure(row=1 if stacked else 0,column=0 if stacked else 2,
                columnspan=3 if stacked else 1,sticky='ew' if stacked else 'e',pady=(8,0) if stacked else 0)

    def toggle_log(self):
        self.log_visible=not self.log_visible
        if self.log_visible:
            self.log_panel.grid(row=4,column=0,sticky='ew',pady=(10,0));self.log_button.configure(text='Журнал событий  −')
        else:
            self.log_panel.grid_remove();self.log_button.configure(text='Журнал событий  +')

    def set_running(self,running):
        for control in self.controls:
            control.state(['disabled'] if running else ['!disabled'])
        self.start_button.state(['disabled'] if running else ['!disabled'])
        self.doctor_button.state(['disabled'] if running else ['!disabled'])
        self.cancel_button.state(['!disabled'] if running else ['disabled'])
        if not running:
            self.set_preview_view(self.app.preview_view_var.get())

    def _schedule_summary(self,*args):
        if not self._summary_pending:
            self._summary_pending=True
            self.app.after_idle(self.update_summary)

    def update_summary(self):
        self._summary_pending=False;a=self.app
        name = a.preset_combo.get()
        if name in self.size_presets:
            try:
                if (int(a.width_var.get()), int(a.height_var.get())) != self.size_presets[name]:
                    a.preset_combo.set('custom')
            except ValueError:
                a.preset_combo.set('custom')
        mode=a.sizing_var.get()
        if mode=='resolution':text=f'{a.width_var.get() or "—"} × {a.height_var.get() or "—"} px'
        else:text=f'{a.sizing_value_var.get() or "—"} '+{'percent':'%','megapixels':'МП','longest':'px · длинная сторона','shortest':'px · короткая сторона'}.get(mode,'')
        self.summary_size.set(text)
        if a.quality_var.get() == QUALITY_SOURCE:
            quality_text=f'Как у исходника  ·  {a.filter_var.get()}'
        else:
            quality_text=f'{a.jpeg_quality_var.get()} / 100  ·  {a.filter_var.get()}'
        sharpen_val = getattr(a, 'sharpen_preset_var', None)
        if sharpen_val and sharpen_val.get() in ('weak', 'standard'):
            quality_text += f'  ·  резкость: {sharpen_val.get()}'
        limit=a.max_file_size_var.get().strip()
        if limit:
            try:
                min_quality=min(int(a.min_quality_var.get()),int(a.jpeg_quality_var.get()))
            except ValueError:
                min_quality=int(a.min_quality_var.get() or 30)
            quality_text+=f'\nПредел {limit}  ·  мин. качество {min_quality}'
        self.summary_quality.set(quality_text)
        mode_val = a.mode_var.get()
        anchor = getattr(a, 'crop_anchor_var', None)
        anchor_text = f' ({anchor.get()})' if anchor and mode_val in ('cover', 'pad') else ''
        fit_map = {'contain':'Целиком, без обрезки','cover':'Заполнение + обрезка','pad':'Вписывание + поля'}
        fit_title = fit_map.get(mode_val, mode_val) + anchor_text
        if a.allow_upscale_var.get():
            fit_title += '\nУвеличение разрешено'
        else:
            fit_title += '\nБез увеличения'
        self.summary_mode.set(fit_title)
        self.summary_backend.set({'auto':'Автовыбор','gpu':'CUDA + automatic codec','hybrid':'CPU decode + CUDA','cpu':'Только CPU'}.get(a.backend_var.get(),'—'))
        write_text = 'Пропуск: с проверкой manifest' if a.skip_existing_var.get() or not a.overwrite_var.get() else 'Перезапись результатов'
        if getattr(a, 'strip_personal_var', None) and a.strip_personal_var.get():
            write_text += '  ·  Очистка GPS'
        if getattr(a, 'web_srgb_var', None) and a.web_srgb_var.get():
            write_text += '  ·  Web sRGB'
        self.summary_write.set(write_text)

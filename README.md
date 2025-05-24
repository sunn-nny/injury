# 受伤风险预警系统  

项目详细支持
微信联系：s18636284651

系统相关code

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib import font_manager

matplotlib.use("TkAgg")
plt.style.use('ggplot')

font_path = None
try:

    if os.name == 'nt':
        font_path = "C:/Windows/Fonts/simhei.ttf"
    elif os.name == 'posix':
        font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    else:
        font_path = '/System/Library/Fonts/PingFang.ttc'

    if font_path and os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    else:

        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                break
        else:
            print("警告: 未找到中文字体，图表文字可能无法正确显示")
            plt.rcParams['font.family'] = 'sans-serif'

except Exception as e:
    print(f"字体加载失败: {str(e)}")
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False


class InjuryPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("消防员受伤风险预测系统")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f4f8")
        self.root.minsize(1000, 600)

        self.base_width = 1200
        self.base_height = 800

        self.font_sizes = {
            'title': 14,
            'label': 12,
            'entry': 11,
            'button': 14,
            'result': 16,
            'footer': 10,
            'chart_title': 12,
            'chart_label': 10
        }

        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale = 1.0

        try:
            self.root.iconbitmap('fire_icon.ico')
        except:
            pass

        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('features.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
                expected_features = ['Player_Age', 'Player_Weight', 'Player_Height',
                                     'Previous_Injuries', 'Training_Intensity', 'Recovery_Time']
                if sorted(self.feature_names) != sorted(expected_features):
                    print(f"警告: 模型特征名称不匹配。期望: {expected_features}，实际: {self.feature_names}")
        except FileNotFoundError:
            messagebox.showerror("错误", "未找到模型文件")
            self.root.destroy()
            return

        self.root.bind("<Configure>", self.on_resize)

        self.create_widgets()
        self.setup_layout()
        self.adjust_fonts()

    def create_widgets(self):
        self.main_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.input_frame = tk.LabelFrame(self.main_frame, text="消防员数据输入",
                                         font=("SimHei", self.font_sizes['title'], "bold"),
                                         bg="#ffffff", bd=2, relief=tk.RIDGE,
                                         padx=15, pady=15)
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_frame = tk.LabelFrame(self.main_frame, text="受伤风险评估",
                                          font=("SimHei", self.font_sizes['title'], "bold"),
                                          bg="#ffffff", bd=2, relief=tk.RIDGE,
                                          padx=15, pady=15)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.entries = {}
        labels = {
            'Player_Age': '年龄 (岁):',
            'Player_Weight': '体重 (kg):',
            'Player_Height': '身高 (cm):',
            'Previous_Injuries': '既往受伤次数:',
            'Training_Intensity': '训练强度 (0-1):',
            'Recovery_Time': '恢复时间 (天):'
        }

        input_container = tk.Frame(self.input_frame, bg="#ffffff")
        input_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.label_widgets = []
        self.entry_widgets = []

        for row, (feature, label_text) in enumerate(labels.items()):
            row_frame = tk.Frame(input_container, bg="#ffffff")
            row_frame.grid(row=row, column=0, sticky="ew", pady=5)

            label = tk.Label(row_frame,
                             text=label_text,
                             font=("SimHei", self.font_sizes['label']),
                             bg="#ffffff",
                             width=15,
                             anchor="w")
            label.pack(side=tk.LEFT, padx=5)
            self.label_widgets.append(label)

            entry = ttk.Entry(row_frame,
                              font=("SimHei", self.font_sizes['entry']),
                              width=20)
            entry.pack(side=tk.RIGHT,
                       fill=tk.X,
                       expand=True,
                       padx=5)
            self.entries[feature] = entry
            self.entry_widgets.append(entry)

            input_container.rowconfigure(row, weight=1)

        input_container.columnconfigure(0, weight=1)

        button_frame = tk.Frame(self.input_frame, bg="#ffffff")
        button_frame.pack(fill=tk.X, pady=20)

        self.style = ttk.Style()
        self.style.configure('Accent.TButton',
                             font=("SimHei", self.font_sizes['button'], "bold"),
                             foreground="#ffffff",
                             background="#4a90e2",
                             padding=10)

        self.predict_btn = ttk.Button(button_frame,
                                      text="预测受伤风险",
                                      command=self.predict,
                                      style='Accent.TButton')
        self.predict_btn.pack(fill=tk.X, expand=True)

        self.result_label = tk.Label(self.result_frame,
                                     text="请输入数据并点击预测",
                                     font=("SimHei", self.font_sizes['result'], "bold"),
                                     bg="#ffffff",
                                     fg="#333333",
                                     wraplength=350,
                                     justify=tk.LEFT)
        self.result_label.pack(pady=20, fill=tk.X, padx=15)

        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.create_footer()

    def setup_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def create_footer(self):
        footer_frame = tk.Frame(self.root,
                                bg="#e6f2ff",
                                height=30,
                                relief=tk.SUNKEN)
        footer_frame.pack(side=tk.BOTTOM,
                          fill=tk.X,
                          padx=20,
                          pady=(5, 0))

        self.footer_label = tk.Label(footer_frame,
                                     text="消防员安全预警系统 v1.0 | 技术支持：XXX团队",
                                     font=("SimHei", self.font_sizes['footer']),
                                     bg="#e6f2ff",
                                     fg="#666666")
        self.footer_label.pack(pady=5)

    def on_resize(self, event):
        if event.widget == self.root:
            width = event.width
            height = event.height

            self.scale_x = width / self.base_width
            self.scale_y = height / self.base_height
            self.scale = (self.scale_x + self.scale_y) / 2

            self.adjust_fonts()

    def adjust_fonts(self):
        current_font = self.input_frame["font"]
        font_family = current_font[0] if isinstance(current_font, tuple) else "SimHei"
        self.input_frame["font"] = (font_family, int(self.font_sizes['title'] * self.scale), "bold")
        self.result_frame["font"] = (font_family, int(self.font_sizes['title'] * self.scale), "bold")

        for label in self.label_widgets:
            label["font"] = (font_family, int(self.font_sizes['label'] * self.scale))

        for entry in self.entry_widgets:
            entry["font"] = (font_family, int(self.font_sizes['entry'] * self.scale))

        self.style.configure('Accent.TButton',
                             font=(font_family, int(self.font_sizes['button'] * self.scale), "bold"))

        self.result_label["font"] = (font_family, int(self.font_sizes['result'] * self.scale), "bold")

        self.footer_label["font"] = (font_family, int(self.font_sizes['footer'] * self.scale))

        if hasattr(self, 'ax') and self.ax:
            self.ax.set_title("受伤风险概率分布", fontsize=int(self.font_sizes['chart_title'] * self.scale))
            self.ax.set_ylabel("概率", fontsize=int(self.font_sizes['chart_label'] * self.scale))
            self.figure.canvas.draw_idle()

    def predict(self):
        input_data = []
        for feature in self.feature_names:
            if feature not in self.entries:
                messagebox.showerror("系统错误", f"模型特征 '{feature}' 与界面输入不匹配")
                return
            value = self.entries[feature].get()
            if not value:
                messagebox.showerror("输入错误",
                                     f"{self.entries[feature].master.winfo_children()[0].cget('text').strip(':')} 不能为空")
                return
            try:
                input_data.append(float(value))
            except ValueError:
                messagebox.showerror("输入错误",
                                     f"{self.entries[feature].master.winfo_children()[0].cget('text').strip(':')} 必须是数字")
                return

        input_array = np.array([input_data])
        input_scaled = self.scaler.transform(input_array)

        prediction = self.model.predict(input_scaled)[0]
        proba = self.model.predict_proba(input_scaled)[0]

        result_config = {
            0: ("受伤风险: 低", "#38b000", "#f0f4f8"),
            1: ("受伤风险: 高", "#e63946", "#ffebee")
        }
        base_text, color, bg_color = result_config[prediction]
        result_text = f"{base_text}\n受伤概率: {proba[1]:.2%}\n未受伤概率: {proba[0]:.2%}"

        self.result_label.config(text=result_text, fg=color)

        self.root.configure(bg=bg_color)
        self.main_frame.configure(bg=bg_color)

        self.ax.clear()
        bars = self.ax.bar(["未受伤概率", "受伤概率"],
                           [proba[0], proba[1]],
                           color=["#4cc9f0", "#f72585"])
        self.ax.set_ylim(0, 1)
        self.ax.set_title("受伤风险概率分布", fontsize=int(self.font_sizes['chart_title'] * self.scale))
        self.ax.set_ylabel("概率", fontsize=int(self.font_sizes['chart_label'] * self.scale))

        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2., height + 0.03,
                         f"{height:.2%}",
                         ha='center',
                         va='bottom',
                         fontsize=int(self.font_sizes['chart_label'] * self.scale))

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = InjuryPredictionApp(root)
    root.mainloop()

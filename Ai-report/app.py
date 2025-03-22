from ultralytics import YOLO
import cv2
from collections import defaultdict
from tqdm import tqdm
import os
from openai import OpenAI  # 导入 OpenAI 库
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import markdown
from tkinter import ttk
from datetime import datetime

# 加载训练好的模型
model = YOLO('best.pt')# 模型地址

# 定义不同类别的颜色
colors = {
    'Excavator': (0, 255, 0, 0.5),
    'Gloves': (255, 0, 0, 0.5),
    'Hardhat': (0, 0, 255, 0.5),
    'Ladder': (255, 255, 0, 0.5),
    'Mask': (0, 255, 255, 0.5),
    'NO-Hardhat': (255, 0, 255, 0.5),
    'NO-Mask': (128, 0, 0, 0.5),
    'NO-Safety Vest': (0, 128, 0, 0.5),
    'Person': (0, 0, 128, 0.5),
    'SUV': (128, 128, 0, 0.5),
    'Safety Cone': (0, 128, 128, 0.5),
    'Safety Vest': (128, 0, 128, 0.5),
    'bus': (255, 128, 0, 0.5),
    'dump truck': (0, 255, 128, 0.5),
    'fire hydrant': (128, 255, 0, 0.5),
    'machinery': (0, 128, 255, 0.5),
    'mini-van': (128, 0, 255, 0.5),
    'sedan': (255, 0, 128, 0.5),
    'semi': (128, 255, 128, 0.5),
    'trailer': (255, 128, 128, 0.5),
    'truck and trailer': (128, 128, 255, 0.5),
    'truck': (255, 128, 255, 0.5),
    'van': (128, 255, 255, 0.5),
    'vehicle': (255, 255, 128, 0.5),
    'wheel loader': (128, 128, 128, 0.5)
}

# 处理结果
report = []
class_count = defaultdict(int)
scene_class_counts = []

def process_images(img_folder_path, output_folder_path):
    global report, class_count, scene_class_counts
    report = []
    class_count = defaultdict(int)
    scene_class_counts = []

    # 获取所有图片文件
    img_files = [os.path.join(img_folder_path, f) for f in os.listdir(img_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 使用 tqdm 显示进度条
    for idx, img_path in enumerate(tqdm(img_files, desc="Processing Images")):
        # 读取图像
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # 进行预测
        results = model.predict(source=img_path, save=False, save_txt=False)

        img_report = []
        scene_class_count = defaultdict(int)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 获取颜色
                color = colors.get(class_name, (0, 255, 0, 0.5))
                b, g, r, a = color

                # 绘制填充颜色的矩形
                overlay = img.copy()
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (b, g, r), -1)
                img = cv2.addWeighted(overlay, a, img, 1 - a, 0)

                # 绘制边界框
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (b, g, r), 2)

                # 绘制标签
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # 记录报告信息
                img_report.append({
                    'class': class_name,
                    'confidence': confidence,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })

                # 统计类别数量
                class_count[class_name] += 1
                scene_class_count[class_name] += 1

        # 保存处理后的图像
        output_img_path = os.path.join(output_folder_path, os.path.basename(img_path))
        cv2.imwrite(output_img_path, img)

        # 将每张图片的报告添加到总报告中，并在前面加上场景编号
        report.append(f"\nScene {idx + 1} ({os.path.basename(img_path)}):")
        report.append(f"Image Size: {width}x{height}")
        for item in img_report:
            report.append(
                f"Class: {item['class']}, Confidence: {item['confidence']:.2f}, Position: ({item['x1']:.2f}, {item['y1']:.2f}) - ({item['x2']:.2f}, {item['y2']:.2f})")

        # 添加场景的类别统计信息
        report.append("Class Statistics for this scene:")
        for class_name, count in scene_class_count.items():
            report.append(f"Class: {class_name}, Count: {count}")

        # 保存场景的类别统计信息
        scene_class_counts.append(scene_class_count)

        # 更新进度
        progress_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - Processed {idx + 1}/{len(img_files)} images\n")
        progress_text.see(tk.END)  # 自动滚动到最新插入的文本
        progress_text.update_idletasks()

    # 输出总报告
    total_summary = "\nTotal Summary:\n"
    total_summary += f"Processed {len(img_files)} images.\n"
    for line in report:
        total_summary += line + "\n"

    total_summary += "\nOverall Class Statistics:\n"
    for class_name, count in class_count.items():
        total_summary += f"Class: {class_name}, Count: {count}\n"

    return total_summary

def generate_ai_summary(total_summary):
    # 读取提示内容
    with open('prompt_en.txt', 'r', encoding='utf-8') as file:
        prompt_content = file.read()

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key="",  # 替换为你的 API 密钥
        base_url=""  # 指定 API 的 base URL
    )

    # 构建消息列表
    messages = [{'role': 'user', 'content': f"{prompt_content}\n\n{total_summary}"}]

    # 调用流式 API 进行总结
    ai_summary = ""
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用 GPT-3.5 模型
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:  # 检查 choices 是否存在
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:  # 检查 content 是否存在
                    ai_summary += delta.content
                    ai_report_text.insert(tk.END, delta.content)
                    ai_report_text.see(tk.END)  # 自动滚动到最新插入的文本
                    ai_report_text.update_idletasks()
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        ai_summary = "Failed to generate AI summary."

    return ai_summary

def save_ai_summary(ai_summary, output_md_path):
    # 将 AI 生成的回复保存为 Markdown 文件
    with open(output_md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(f"# 工地安全报告\n\n{ai_summary}")

    # 生成 HTML
    html_content = markdown.markdown(ai_summary)

def start_processing():
    img_folder_path = img_folder_entry.get()
    output_folder_path = output_folder_entry.get()
    output_md_path = md_file_entry.get()

    if not img_folder_path or not output_folder_path or not output_md_path:
        messagebox.showerror("错误", "请选择所有路径")
        return

    def run_processing():
        progress_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - 图像处理中...\n")
        progress_text.update_idletasks()
        total_summary = process_images(img_folder_path, output_folder_path)

        progress_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - 报告生成中...\n")
        progress_text.update_idletasks()
        ai_summary = generate_ai_summary(total_summary)

        progress_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - AI 总结中...\n")
        progress_text.update_idletasks()
        save_ai_summary(ai_summary, output_md_path)

        progress_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - 处理完成\n")
        progress_text.update_idletasks()
        root.after(0, lambda: messagebox.showinfo("完成", "处理完成"))

    threading.Thread(target=run_processing).start()

def select_img_folder():
    folder_path = filedialog.askdirectory()
    img_folder_entry.delete(0, tk.END)
    img_folder_entry.insert(0, folder_path)

def select_output_folder():
    folder_path = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, folder_path)

def select_md_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown files", "*.md")])
    md_file_entry.delete(0, tk.END)
    md_file_entry.insert(0, file_path)

def clear_ai_report():
    ai_report_text.delete(1.0, tk.END)

# 创建主窗口
root = tk.Tk()
root.title("工地安全报告生成器")

# 定义字体
FONT_FAMILY = "Arial"  # 字体名称
FONT_SIZE = 12  # 字体大小
FONT_STYLE = "normal"  # 字体样式（normal, bold, italic, underline, overstrike）

# 使用 ttk 样式美化界面
style = ttk.Style()
style.theme_use('clam')

# 自定义样式
style.configure('TButton', font=(FONT_FAMILY, FONT_SIZE, FONT_STYLE), padding=6, background='#4CAF50', foreground='white', borderwidth=0, relief='flat')
style.map('TButton',
          background=[('active', 'orange')],  # 鼠标悬停时背景变为橙色
          foreground=[('active', 'white')])   # 鼠标悬停时文字变为白色

style.configure('TLabel', font=(FONT_FAMILY, FONT_SIZE, FONT_STYLE), padding=6, foreground='black')
style.configure('TEntry', font=(FONT_FAMILY, FONT_SIZE, FONT_STYLE), padding=6, foreground='#333333', borderwidth=0, relief='flat')
style.configure('TFrame', background='#F0F0F0')
style.configure('TProgressbar', thickness=10, troughcolor='#E0E0E0', background='#4CAF50', borderwidth=0, relief='flat')

# 创建布局
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# 图片文件夹路径选择
img_folder_label = ttk.Label(frame, text="选择图片文件夹:")
img_folder_label.grid(row=0, column=0, sticky=tk.W)
img_folder_entry = ttk.Entry(frame, width=50)
img_folder_entry.grid(row=0, column=1)
img_folder_button = ttk.Button(frame, text="浏览", command=select_img_folder)
img_folder_button.grid(row=0, column=2)

# 识别图片保存路径选择
output_folder_label = ttk.Label(frame, text="选择识别图片保存路径:")
output_folder_label.grid(row=1, column=0, sticky=tk.W)
output_folder_entry = ttk.Entry(frame, width=50)
output_folder_entry.grid(row=1, column=1)
output_folder_button = ttk.Button(frame, text="浏览", command=select_output_folder)
output_folder_button.grid(row=1, column=2)

# Markdown 文件保存路径选择
md_file_label = ttk.Label(frame, text="选择 Markdown 文件保存路径:")
md_file_label.grid(row=2, column=0, sticky=tk.W)
md_file_entry = ttk.Entry(frame, width=50)
md_file_entry.grid(row=2, column=1)
md_file_button = ttk.Button(frame, text="浏览", command=select_md_file)
md_file_button.grid(row=2, column=2)

# 开始按钮
start_button = ttk.Button(frame, text="开始处理", command=start_processing)
start_button.grid(row=3, column=0, columnspan=3, pady=10)

# 清空按钮
clear_button = ttk.Button(frame, text="清空", command=clear_ai_report)
clear_button.grid(row=3, column=1, columnspan=3, pady=10)

# 图片显示框
image_frame = ttk.Frame(frame)
image_frame.grid(row=4, column=0, columnspan=3, pady=10)

# AI 报告生成框
ai_report_label = ttk.Label(frame, text="AI 报告生成:")
ai_report_label.grid(row=5, column=0, sticky=tk.W)
ai_report_text = tk.Text(frame, width=102, height=20, wrap=tk.WORD, bg='#F0F0F0', fg='#333333', font=(FONT_FAMILY, FONT_SIZE, FONT_STYLE))
ai_report_text.grid(row=6, column=0, columnspan=3, pady=10)

# 运行程序进度框
progress_label = ttk.Label(frame, text="程序进度:")
progress_label.grid(row=7, column=0, sticky=tk.W)
progress_text = tk.Text(frame, width=102, height=8, wrap=tk.WORD, bg='#F0F0F0', fg='#333333', font=(FONT_FAMILY, FONT_SIZE, FONT_STYLE))
progress_text.grid(row=8, column=0, columnspan=3, pady=10)

# 运行主循环
root.mainloop()

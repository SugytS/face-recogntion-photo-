import cv2
import face_recognition
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os


class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title("Распознавание лиц на фото")
        try:
            wanted_image = face_recognition.load_image_file('YOUR_FILE_NAME.png')
            self.wanted_encoding = face_recognition.face_encodings(wanted_image)[0]
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить эталонное изображение: {str(e)}")
            self.window.destroy()
            return
        self.create_widgets()
        self.current_image = None
        self.image_path = ""
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        self.control_panel = ttk.Frame(self.window)
        self.control_panel.pack(fill=tk.X, padx=10, pady=5)
        self.open_btn = ttk.Button(self.control_panel, text="Открыть фото", command=self.open_image)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn = ttk.Button(self.control_panel, text="Обработать", command=self.process_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.state(['disabled'])
        self.file_label = ttk.Label(self.control_panel, text="Файл не выбран")
        self.file_label.pack(side=tk.LEFT, padx=5, expand=True)
        self.exit_btn = ttk.Button(self.control_panel, text="Выход", command=self.on_close)
        self.exit_btn.pack(side=tk.RIGHT, padx=5)
        self.image_panel = ttk.Label(self.window)
        self.image_panel.pack(padx=10, pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            try:
                image = Image.open(file_path)
                self.current_image = image.copy()
                max_size = (800, 600)
                image.thumbnail(max_size, Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(image)
                self.image_panel.img_tk = img_tk
                self.image_panel.config(image=img_tk)
                self.process_btn.state(['!disabled'])
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def process_image(self):
        if not self.current_image or not self.image_path:
            return
        try:
            opencv_image = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
            processed_image = self.detect_faces(opencv_image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(processed_image)
            max_size = (800, 600)
            image.thumbnail(max_size, Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image)
            self.image_panel.img_tk = img_tk
            self.image_panel.config(image=img_tk)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки изображения: {str(e)}")

    def detect_faces(self, image):
        small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_image, model="hog")
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 2;
                right *= 2;
                bottom *= 2;
                left *= 2
                matches = face_recognition.compare_faces([self.wanted_encoding], face_encoding, tolerance=0.6)
                color = (0, 0, 255) if matches[0] else (0, 255, 0)
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        return image

    def on_close(self):
        if hasattr(self, 'current_image'):
            del self.current_image
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root, "Распознавание лиц на фото")
    root.mainloop()

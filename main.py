# Modernized Face Recognition Attendance System UI with Light/Dark Mode Toggle (LBPH Only, Final Polished Version)
import tkinter as tk
from tkinter import ttk, messagebox
import time
import datetime
import os
import csv
import cv2
import pandas as pd
import numpy as np
from PIL import Image

class StyleManager:
    def __init__(self, root):
        self.style = ttk.Style(root)
        self.root = root
        self.theme = 'light'  # default
        self.apply_light_theme()

    def apply_light_theme(self):
        self.root.configure(bg='#f0f0f0')
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', foreground='black', font=('Segoe UI', 12))
        self.style.configure('TButton', background='#0078D7', foreground='white', font=('Segoe UI', 11), padding=6)
        self.style.configure('Treeview', background='white', foreground='black', fieldbackground='white')
        self.style.configure('TEntry', fieldbackground='white', foreground='black')
        self.theme = 'light'

    def apply_dark_theme(self):
        self.root.configure(bg='#2e2e2e')
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2e2e2e')
        self.style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Segoe UI', 12))
        self.style.configure('TButton', background='#1a73e8', foreground='white', font=('Segoe UI', 11), padding=6)
        self.style.configure('Treeview', background='#3b3b3b', foreground='white', fieldbackground='#3b3b3b')
        self.style.configure('TEntry', fieldbackground='#1e1e1e', foreground='white')
        self.theme = 'dark'

    def toggle_theme(self):
        if self.theme == 'light':
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x600")

        self.style_manager = StyleManager(self.root)

        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=5)
        title = ttk.Label(header_frame, text="Face Recognition Attendance System", font=('Segoe UI', 20, 'bold'))
        title.pack(side='left')

        self.theme_var = tk.BooleanVar()
        theme_toggle = ttk.Checkbutton(header_frame, text="Dark Mode", variable=self.theme_var, command=self.toggle_theme)
        theme_toggle.pack(side='right')

        self.clock_label = ttk.Label(self.root, font=('Segoe UI', 14), anchor='center')
        self.clock_label.pack(fill='x')
        self.update_clock()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        ttk.Label(right_frame, text="Register New Student", font=('Segoe UI', 14, 'bold')).pack(pady=(0, 10))

        ttk.Label(right_frame, text="ID:").pack(anchor='w')
        self.id_entry = ttk.Entry(right_frame)
        self.id_entry.pack(fill='x', pady=2)

        ttk.Label(right_frame, text="Name:").pack(anchor='w')
        self.name_entry = ttk.Entry(right_frame)
        self.name_entry.pack(fill='x', pady=2)

        ttk.Button(right_frame, text="Take Images", command=self.take_images).pack(fill='x', pady=(10, 2))
        ttk.Button(right_frame, text="Save Profile", command=self.save_profile).pack(fill='x', pady=2)

        ttk.Label(left_frame, text="Attendance", font=('Segoe UI', 14, 'bold')).pack(pady=(0, 10))
        self.tree = ttk.Treeview(left_frame, columns=('id', 'name', 'date', 'time'), show='headings')
        self.tree.heading('id', text='ID')
        self.tree.heading('name', text='Name')
        self.tree.heading('date', text='Date')
        self.tree.heading('time', text='Time')
        self.tree.pack(fill='both', expand=True)

        ttk.Button(left_frame, text="Take Attendance", command=self.take_attendance).pack(fill='x', pady=(10, 2))
        ttk.Button(left_frame, text="Export to Excel", command=self.export_to_excel).pack(fill='x', pady=2)
        ttk.Button(left_frame, text="Quit", command=self.root.quit).pack(fill='x', pady=2)

    def take_images(self):
        id_value = self.id_entry.get()
        name_value = self.name_entry.get()

        if not id_value.isdigit() or not name_value.isalpha():
            messagebox.showerror("Invalid Input", "ID must be numeric and Name must be alphabetic.")
            return

        os.makedirs("TrainingImage", exist_ok=True)
        os.makedirs("StudentDetails", exist_ok=True)
        serial = 1

        if os.path.exists("StudentDetails/StudentDetails.csv"):
            with open("StudentDetails/StudentDetails.csv", 'r') as f:
                serial = sum(1 for row in csv.reader(f)) // 2 + 1

        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        sample_num = 0

        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sample_num += 1
                cv2.imwrite(f"TrainingImage/{name_value}.{serial}.{id_value}.{sample_num}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Taking Images - Press Q to Quit', img)
            if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 10:
                break

        cam.release()
        cv2.destroyAllWindows()

        with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
            writer.writerow([serial, '', id_value, '', name_value])

        messagebox.showinfo("Success", f"Images taken for ID: {id_value}")

    def take_attendance(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists("TrainingImageLabel/Trainner.yml"):
            messagebox.showerror("Model Missing", "Please train a profile first.")
            return
        recognizer.read("TrainingImageLabel/Trainner.yml")

        if not os.path.exists("StudentDetails/StudentDetails.csv"):
            messagebox.showerror("Data Missing", "Student details file not found.")
            return

        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        attendance = []
        threshold = 55

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (200, 200))
                serial, conf = recognizer.predict(face_resized)
                label_text = f"Unknown ({100-int(conf)}%)"
                if conf < threshold:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    name_row = df.loc[df['SERIAL NO.'] == serial]
                    if not name_row.empty:
                        student_id = name_row['ID'].values[0]
                        student_name = name_row['NAME'].values[0]
                        attendance.append((student_id, student_name, date, timeStamp))
                        label_text = f"{student_name} ({100-int(conf)}%)"
                cv2.putText(img, label_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (225, 0, 0), 2)

            cv2.imshow("Taking Attendance - Press Q to Quit", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        date_file = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y')
        os.makedirs("Attendance", exist_ok=True)
        filename = f"Attendance/Attendance_{date_file}.csv"

        with open(filename, 'a+', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['ID', 'Name', 'Date', 'Time'])
            for record in attendance:
                writer.writerow(record)

        for row in self.tree.get_children():
            self.tree.delete(row)
        for record in attendance:
            self.tree.insert('', 0, values=record)

        messagebox.showinfo("Attendance Complete", f"Attendance saved to {filename}\nRecognized {len(attendance)} student(s).")

    def save_profile(self):
        os.makedirs("TrainingImageLabel", exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            face_samples = []
            ids = []
            for image_path in image_paths:
                pil_img = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_img, 'uint8')
                id_ = int(os.path.split(image_path)[-1].split(".")[2])
                face_samples.append(img_numpy)
                ids.append(id_)
            return face_samples, ids

        try:
            faces, ids = get_images_and_labels("TrainingImage")
            recognizer.train(faces, np.array(ids))
            recognizer.save("TrainingImageLabel/Trainner.yml")
            messagebox.showinfo("Success", "Profile trained and saved successfully.")
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train profile: {str(e)}")

    def export_to_excel(self):
        date_file = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y')
        csv_path = f"Attendance/Attendance_{date_file}.csv"
        xlsx_path = f"Attendance/Attendance_{date_file}.xlsx"

        if not os.path.exists(csv_path):
            messagebox.showerror("Export Error", f"No CSV file found for today: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            df.to_excel(xlsx_path, index=False)
            messagebox.showinfo("Export Complete", f"Excel file saved to:\n{xlsx_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def toggle_theme(self):
        self.style_manager.toggle_theme()

    def update_clock(self):
        now = datetime.datetime.now()
        date_string = now.strftime("%d-%B-%Y")
        time_string = now.strftime("%H:%M:%S")
        self.clock_label.config(text=f"{date_string}  |  {time_string}")
        self.root.after(1000, self.update_clock)

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()





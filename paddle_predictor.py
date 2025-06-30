import os
import cv2
import csv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt

# Format roll number using batch year (e.g., 2023PECAI394)
def format_roll_number(text, batch_year):
    if text.isdigit() and len(text) == 3:
        return f"{batch_year}PECAI{text}"
    return None

# Preprocessing the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    preprocessed_path = "preprocessed.jpg"
    cv2.imwrite(preprocessed_path, sharpen)
    return preprocessed_path

# Visualization of OCR output
def visualize_results(image_path, result):
    image = cv2.imread(image_path)
    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    image = draw_ocr(image, boxes, texts, scores, font_path='C:/Windows/Fonts/arial.ttf')
    annotated_path = "annotated_output.jpg"
    cv2.imwrite(annotated_path, image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Roll Numbers")
    plt.axis("off")
    plt.show()

# Save to new CSV file with timestamp
def save_to_csv(roll_numbers, batch_year):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"roll_numbers_{batch_year}_{timestamp}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for rn in roll_numbers:
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rn])
    print(f"\n📁 Roll numbers saved to: {filename}")

# Core prediction logic
def predict_roll_number(image_path, batch_year):
    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return

    clean_image = preprocess_image(image_path)

    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='en',
        det=True,
        rec_algorithm='CRNN',
        rec_model_dir='./output/rec_digit/student_rec_infer',
        use_gpu=False,
        show_log=False
    )

    result = ocr.ocr(clean_image, cls=False)

    found = False
    roll_numbers = []
    corrected_result = []

    print("\n🔍 Review Detected Results:")
    for line in result:
        corrected_line = []
        for rec in line:
            box = rec[0]
            text, confidence = rec[1][0], rec[1][1]

            print(f"Detected: {text} | Confidence: {confidence:.4f}")
            if not text.isdigit() or len(text) != 3:
                corrected = input(f"⚠ Edit misread value '{text}' to correct 3-digit number (or press Enter to skip): ").strip()
                if corrected.isdigit() and len(corrected) == 3:
                    text = corrected

            formatted = format_roll_number(text, batch_year)
            if formatted:
                roll_numbers.append(formatted)
                found = True

            corrected_line.append([box, [text, confidence]])
        corrected_result.append(corrected_line)

    visualize_results(clean_image, corrected_result)

    if found:
        print("\n✅ Final Roll Numbers:")
        for rn in roll_numbers:
            print("-", rn)
        save_to_csv(roll_numbers, batch_year)
    else:
        print("⚠ No valid 3-digit number found.")

# GUI logic
def select_image_and_process():
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        messagebox.showwarning("No file selected", "Please select an image file.")
        return

    batch_year = simpledialog.askstring("Batch Year", "Enter the Batch Year (e.g., 2023):")
    if not batch_year or not batch_year.isdigit():
        messagebox.showerror("Invalid Batch", "Please enter a valid numeric batch year.")
        return

    predict_roll_number(image_path, batch_year)

if __name__ == "__main__":
    select_image_and_process()

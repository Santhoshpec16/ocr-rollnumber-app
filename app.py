import streamlit as st
import tempfile
import os
from datetime import datetime
from paddleocr import PaddleOCR, draw_ocr
import cv2
from paddle_predictor import preprocess_image, format_roll_number
import io
import csv

# -------------------- Page Setup --------------------
st.set_page_config(page_title="OCR Roll Number Extractor", layout="centered")
st.title("📸 Student Roll Number Extractor (OCR)")
st.markdown(
    "Upload an image and enter the batch year (e.g., `2024`). "
    "Then click 'Start Processing' to extract roll numbers using OCR and make edits if needed."
)

# -------------------- Session State Init --------------------
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None
    st.session_state.image_path = None
    st.session_state.started = False
    st.session_state.roll_numbers = []

# -------------------- OCR + Streamlit UI Logic --------------------
def streamlit_predict_roll_numbers(result, image_path, batch_year):
    roll_numbers = []
    corrected_result = []

    st.subheader("🔍 Review Detected Results")
    for line_num, line in enumerate(result):
        corrected_line = []
        for i, rec in enumerate(line):
            box = rec[0]
            text, confidence = rec[1][0], rec[1][1]

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**Detected:** `{text}` | **Confidence:** `{confidence:.4f}`")
                user_correction = st.text_input(
                    f"✏️ Edit (optional):", value=text, key=f"{line_num}_{i}_{text}"
                )
            with col2:
                st.markdown("")

            if user_correction.isdigit() and len(user_correction) == 3:
                text = user_correction

            formatted = format_roll_number(text, batch_year)
            if formatted:
                roll_numbers.append(formatted)

            corrected_line.append([box, [text, confidence]])
        corrected_result.append(corrected_line)

    # 🔍 Draw annotations
    image = cv2.imread(image_path)

    if corrected_result and corrected_result[0]:
        boxes = [line[0] for line in corrected_result[0]]
        texts = [line[1][0] for line in corrected_result[0]]
        scores = [line[1][1] for line in corrected_result[0]]

        annotated_img = draw_ocr(image, boxes, texts, scores, font_path='C:/Windows/Fonts/arial.ttf')
        annotated_path = "annotated_output.jpg"
        cv2.imwrite(annotated_path, annotated_img)

        st.image(annotated_path, caption="🖼️ Annotated OCR Output", use_column_width=True)
    else:
        st.warning("⚠️ Could not visualize OCR results.")

    if roll_numbers:
        st.success("✅ Final Roll Numbers:")
        st.write(roll_numbers)

        # ✅ CSV generation in memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Roll Number"])
        for rn in roll_numbers:
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rn])
        csv_data = output.getvalue().encode("utf-8")

        # ✅ Download button
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"roll_numbers_{batch_year}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No valid 3-digit roll numbers found.")

# -------------------- Upload and Process Section --------------------
batch_year = st.text_input("Enter Batch Year (e.g., 2024)", max_chars=4)
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file and batch_year:
    if not batch_year.isdigit() or len(batch_year) != 4:
        st.error("🚫 Please enter a valid 4-digit batch year.")
    else:
        if st.button("▶️ Start Processing"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_image_path = tmp_file.name

            # Preprocess and run OCR ONCE
            preprocessed_path = preprocess_image(temp_image_path)
            st.session_state.image_path = preprocessed_path

            ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                det=True,
                rec_algorithm='CRNN',
                rec_model_dir='./output/rec_digit/student_rec_infer',
                use_gpu=False,
                show_log=False
            )
            st.session_state.ocr_result = ocr.ocr(preprocessed_path, cls=False)
            st.session_state.started = True

# ✅ Show only after OCR is done
if st.session_state.started and st.session_state.ocr_result:
    streamlit_predict_roll_numbers(
        st.session_state.ocr_result,
        st.session_state.image_path,
        batch_year
    )
elif not uploaded_file:
    st.info("📝 Please upload an image and enter batch year to begin.")

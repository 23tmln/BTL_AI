import streamlit as st
from PIL import Image
import cv2
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


# Cấu hình giao diện trang Streamlit
st.set_page_config(
    page_title="Vietnamese Handwriting OCR",
    layout="centered"
)


# Cache model để tránh load lại model mỗi lần Streamlit refresh
@st.cache_resource
def load_model():
    # Load cấu hình mặc định của VietOCR với kiến trúc vgg_transformer
    config = Cfg.load_config_from_name("vgg_transformer")

    # Chạy model trên CPU
    config["device"] = "cpu"

    # Đường dẫn tới file model đã fine-tune
    config["weights"] = "handwriting_vietocr.pth"

    # Bật beam search để cải thiện chất lượng dự đoán
    config["predictor"]["beamsearch"] = True

    # Khởi tạo và trả về bộ nhận diện OCR
    return Predictor(config)


# Hàm phát hiện và tách các dòng chữ trong ảnh
def detect_lines(pil_img):
    # Chuyển ảnh PIL sang mảng NumPy dạng RGB
    img = np.array(pil_img.convert("RGB"))

    # Chuyển ảnh RGB sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Làm mờ nhẹ ảnh để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Nhị phân hóa ảnh bằng adaptive threshold
    # Chữ sẽ thành màu trắng, nền thành màu đen
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    # Tính tổng số pixel chữ trên từng hàng ngang
    # Nếu một hàng có nhiều pixel trắng thì khả năng cao hàng đó chứa chữ
    projection = np.sum(th > 0, axis=1)

    # Đặt ngưỡng để xác định hàng nào có chữ
    threshold = np.max(projection) * 0.08

    lines = []
    in_line = False
    start = 0

    # Duyệt từng hàng của ảnh để tìm vùng bắt đầu và kết thúc của dòng chữ
    for i, value in enumerate(projection):
        # Nếu số pixel chữ vượt ngưỡng và chưa ở trong dòng nào
        # thì đánh dấu vị trí bắt đầu dòng
        if value > threshold and not in_line:
            start = i
            in_line = True

        # Nếu số pixel chữ nhỏ hơn ngưỡng và đang ở trong một dòng
        # thì đánh dấu vị trí kết thúc dòng
        if value <= threshold and in_line:
            end = i
            in_line = False

            # Chỉ lấy những vùng đủ cao để tránh nhiễu nhỏ
            if end - start > 12:
                lines.append((start, end))

    # Nếu ảnh kết thúc nhưng vẫn đang nằm trong một dòng chữ
    # thì thêm dòng đó vào danh sách
    if in_line:
        lines.append((start, len(projection) - 1))

    merged = []

    # Gộp các dòng nằm quá gần nhau
    # Việc này giúp tránh trường hợp một dòng bị tách thành nhiều phần nhỏ
    for y1, y2 in lines:
        if not merged:
            merged.append([y1, y2])
        else:
            last = merged[-1]

            # Nếu khoảng cách giữa 2 dòng nhỏ hơn 10 pixel thì gộp lại
            if y1 - last[1] < 10:
                last[1] = y2
            else:
                merged.append([y1, y2])

    crops = []
    h, w = gray.shape

    # Cắt ảnh theo từng dòng đã phát hiện
    for y1, y2 in merged:
        pad = 10

        # Thêm khoảng đệm phía trên và dưới dòng chữ
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)

        # Cắt ảnh từ đầu đến cuối chiều ngang, chỉ giới hạn theo chiều dọc
        crop = pil_img.crop((0, y1, w, y2)).convert("RGB")

        # Chỉ lấy ảnh dòng có chiều cao hợp lệ
        if crop.height > 15:
            crops.append(crop)

    # Trả về danh sách ảnh từng dòng và vị trí các dòng
    return crops, merged


# Load model OCR
detector = load_model()


# Tiêu đề ứng dụng
st.title("Vietnamese Handwriting OCR")


# Cho phép người dùng upload ảnh
uploaded_file = st.file_uploader(
    "Upload ảnh chữ viết tay",
    type=["jpg", "jpeg", "png"]
)


# Cho người dùng chọn chế độ nhận diện
mode = st.radio(
    "Chế độ nhận diện",
    ["Tự động tách dòng", "OCR trực tiếp 1 dòng"]
)


# Nếu người dùng đã upload ảnh
if uploaded_file is not None:
    # Mở ảnh và chuyển sang RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Hiển thị ảnh đã upload
    st.subheader("Ảnh đã upload")
    st.image(image, use_container_width=True)

    # Khi người dùng bấm nút nhận diện
    if st.button("Nhận diện"):
        with st.spinner("Đang nhận diện..."):

            # Trường hợp ảnh chỉ có một dòng chữ
            if mode == "OCR trực tiếp 1 dòng":
                # Đưa toàn bộ ảnh vào model OCR
                text = detector.predict(image)

                # Hiển thị kết quả OCR
                st.subheader("Kết quả OCR")
                st.success(text)

            # Trường hợp ảnh có nhiều dòng và cần tách dòng trước
            else:
                # Tách ảnh thành nhiều dòng chữ
                lines, boxes = detect_lines(image)

                # Nếu không phát hiện được dòng nào
                # thì OCR trực tiếp toàn bộ ảnh
                if len(lines) == 0:
                    text = detector.predict(image)

                    st.subheader("Kết quả OCR")
                    st.success(text)

                else:
                    results = []

                    st.subheader("Các dòng đã tách")

                    # Duyệt qua từng dòng đã tách
                    for i, line_img in enumerate(lines):
                        # OCR từng dòng
                        text = detector.predict(line_img)

                        # Lưu kết quả từng dòng
                        results.append(text)

                        # Hiển thị ảnh dòng đã tách
                        st.image(
                            line_img,
                            caption=f"Dòng {i + 1}",
                            use_container_width=True
                        )

                        # Hiển thị kết quả OCR của dòng đó
                        st.write(text)

                    # Ghép kết quả các dòng lại thành văn bản hoàn chỉnh
                    final_text = "\n".join(results)

                    # Hiển thị kết quả cuối cùng
                    st.subheader("Kết quả cuối cùng")
                    st.text_area(
                        "Văn bản OCR",
                        final_text,
                        height=250
                    )

                    # Cho phép tải kết quả OCR về file .txt
                    st.download_button(
                        "Tải kết quả .txt",
                        final_text,
                        file_name="ket_qua_ocr.txt",
                        mime="text/plain"
                    )
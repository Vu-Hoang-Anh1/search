import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import pickle
import numpy as np
import streamlit as st
from scipy.spatial.distance import cdist

# Hàm tạo model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Hàm tiền xử lý, chuyển đổi hình ảnh thành tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Hàm trích xuất vector từ hình ảnh
def extract_vector(model, image_path):
    print("Xử lý:", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Trích xuất đặc trưng
    vector = model.predict(img_tensor)[0]
    # Chuẩn hóa vector = chia chia L2 norm (tự Google search)
    vector = vector / np.linalg.norm(vector)
    return vector

# Tạo giao diện Streamlit
st.title("Search for Similar Images")

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Kiểm tra xem ảnh đã được tải lên chưa
if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Lưu ảnh vào thư mục tạm thời
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Trích xuất vector đặc trưng từ ảnh tải lên
    model = get_extract_model()
    search_vector = extract_vector(model, temp_image_path)

    # Tải model và đường dẫn ảnh đã lưu
    vectors = pickle.load(open("vectors.pkl", "rb"))
    paths = pickle.load(open("paths.pkl", "rb"))

    # Tính khoảng cách giữa vector tìm kiếm và các vector trong tập dữ liệu
    distance = np.linalg.norm(vectors - search_vector, axis=1)

    # Chọn ra K ảnh có khoảng cách ngắn nhất
    K = 3
    ids = np.argsort(distance)[:K]
    nearest_images = [(paths[id], distance[id]) for id in ids]

   # Tạo giao diện Streamlit
    st.title("Top 3 gần giống nhất")

    # Hiển thị ảnh tìm kiếm
    st.write("Ảnh gốc:")
    st.image(temp_image_path)

    # Tạo layout cho các ảnh tương tự
    col1, col2, col3 = st.columns(3)

    # Hiển thị top 3 ảnh tương tự
    for idx, (path, dist) in enumerate(nearest_images):
        if idx == 0:
            with col1:
                st.write(f"Ảnh {idx+1}:")
                st.image(Image.open(path))
                st.write(f"Khoảng cách: {dist:.2f}")
        elif idx == 1:
            with col2:
                st.write(f"Ảnh {idx+1}:")
                st.image(Image.open(path))
                st.write(f"Khoảng cách: {dist:.2f}")
        else:
            with col3:
                st.write(f"Ảnh {idx+1}:")
                st.image(Image.open(path))
                st.write(f"Khoảng cách: {dist:.2f}")


    # Xóa ảnh tạm thời sau khi sử dụng
    os.unlink(temp_image_path)


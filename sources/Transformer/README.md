# Transformer - Dự Báo Giá Xe Ô Tô Cũ Tại TP. HCM

---

## 📌 Ứng dụng

Dự báo giá bán xe ô tô cũ tại thị trường **TP. Hồ Chí Minh** dựa trên mô tả văn bản tiếng Việt của người dùng. Người dùng chỉ cần nhập mô tả chiếc xe (hãng, năm, km, màu sắc, động cơ, tình trạng…), mô hình Transformer sẽ tự động đưa ra mức giá ước tính cùng khoảng tin cậy.

## 👥 Đối tượng sử dụng

- **Người dân TP.HCM** có nhu cầu mua/bán xe ô tô cũ trên các sàn giao dịch (Chợ Tốt, Bonbanh, Oto.com.vn).
- **Chủ salon xe cũ tại TP.HCM** cần tham khảo nhanh giá thị trường để định giá nhập/xuất xe.
- **Người mới mua xe lần đầu** muốn kiểm tra xem giá bán trên mạng có hợp lý hay không.

## 🔬 Phương pháp

- **Mô hình chính:** Fine-tune mô hình **DistilBERT đa ngôn ngữ** (`distilbert-base-multilingual-cased`, 66 triệu tham số) cho bài toán hồi quy (regression) trên cột `Description` (mô tả tiếng Việt) của dataset xe cũ.
- **Baseline so sánh:** Linear Regression + TF-IDF (đã có ở folder `LinearReg`).
- **Lý do chọn DistilBERT:** nhẹ hơn BERT ~40%, train nhanh, đã được huấn luyện trên 104 ngôn ngữ bao gồm tiếng Việt nên hiểu tốt ngữ cảnh mô tả xe.

## 📥 Input / 📤 Output

- **Input:** Văn bản mô tả tiếng Việt về chiếc xe. Ví dụ:
  > "Toyota Vios 2020, đi 30000 km, màu trắng, máy xăng 1.5L, số tự động, 5 chỗ, chính chủ"
- **Output:** Giá bán dự báo (đơn vị: triệu VNĐ) + khoảng giá ước lượng (±15%).

## 📊 Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| **RMSE** (triệu VNĐ) | Sai số bình phương trung bình - càng thấp càng tốt |
| **MAE** (triệu VNĐ) | Sai số tuyệt đối trung bình - dễ diễn giải |
| **R²** | Hệ số xác định (0 → 1) - càng gần 1.0 càng tốt |

Sau khi train, mô hình sẽ tự động vẽ 4 biểu đồ:
1. Training loss qua các epoch
2. So sánh RMSE giữa Transformer và Baseline
3. So sánh R² giữa Transformer và Baseline
4. Scatter plot giá thực tế vs giá dự đoán

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài đặt thư viện

```bash
py -m pip install --upgrade pip
py -m pip install torch transformers scikit-learn pandas numpy matplotlib streamlit
```

> Lưu ý: Nếu máy có GPU NVIDIA, cài `torch` bản CUDA để train nhanh hơn (~10-20 lần). Nếu chỉ có CPU thì vẫn chạy được nhưng chậm hơn.

### 2. Train mô hình

```bash
cd sources/Transformer
py transformer_train.py
```

Quá trình train sẽ:
- Tải dataset và lọc xe tại **TP. Hồ Chí Minh**.
- Train baseline Linear Regression + TF-IDF.
- Fine-tune DistilBERT trong 3 epoch.
- In kết quả RMSE/MAE/R² và vẽ biểu đồ so sánh.
- Lưu model vào thư mục `saved_model/` để dùng cho app.

> Thời gian train ước tính: ~5-15 phút trên CPU, ~1-3 phút trên GPU (tuỳ cấu hình máy).

### 3. Chạy giao diện web (Streamlit)

```bash
py -m streamlit run app.py
```

App sẽ chạy tại: **http://localhost:8501**

---

## 📁 Cấu trúc thư mục

```
sources/Transformer/
├── README.md                  # File này
├── transformer_train.py       # Script train model + đánh giá + vẽ biểu đồ
├── app.py                     # Giao diện Streamlit cho người dùng
├── transformer_result.png     # Biểu đồ kết quả (tự động tạo sau khi train)
└── saved_model/               # Model đã train (tự động tạo sau khi train)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── vocab.txt
```

---

## 🖥️ Giao diện App (Streamlit)

App gồm các phần chính:

1. **Sidebar:** Thông tin đề tài (ứng dụng, đối tượng, phương pháp, input/output).
2. **Phần giới thiệu:** Mô tả bài toán, số liệu dataset TP.HCM.
3. **Phần dự báo:** Ô nhập mô tả + 3 nút ví dụ mẫu + nút "Dự Báo Giá".
4. **Kết quả:** Giá dự báo + khoảng tin cậy + ghi chú.
5. **Lịch sử:** Các lần dự báo trong phiên làm việc.
6. **Thống kê thị trường:** Phân bố giá xe TP.HCM trong dataset.

---

## 🌐 Triển khai (Deployment) - Hosting cho user dùng

Sau khi train xong, có thể deploy app lên các nền tảng miễn phí:

### Cách 1: Streamlit Community Cloud (miễn phí, khuyến nghị)

1. Đẩy code lên GitHub (bao gồm folder `Transformer/` và `Dataset/`).
2. Truy cập [share.streamlit.io](https://share.streamlit.io), đăng nhập bằng GitHub.
3. Tạo app mới, trỏ tới `sources/Transformer/app.py`.
4. App sẽ chạy công khai, user truy cập qua URL được cấp.

### Cách 2: Hugging Face Spaces

1. Tạo Space mới trên [huggingface.co/spaces](https://huggingface.co/spaces), chọn SDK là Streamlit.
2. Upload code và model.
3. User truy cập URL của Space.

### Cách 3: VPS riêng (nâng cao)

```bash
# Chạy trên server, mở port 8501
nohup py -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

---

## 🛠️ Khắc phục lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-----------|-----------|
| `Không tìm thấy car-price.csv` | Sai đường dẫn dataset | Đảm bảo cấu trúc `sources/Dataset/car-price.csv` |
| `Chưa có model` | Chưa chạy `transformer_train.py` | Chạy file train trước |
| `Out of memory` khi train | RAM/GPU không đủ | Giảm `BATCH_SIZE` xuống 8 hoặc 4 trong file `transformer_train.py` |
| Train chậm | Chỉ có CPU | Bình thường, kiên nhẫn đợi 10-20 phút. Nếu có GPU NVIDIA thì cài PyTorch bản CUDA |
| `ConnectionError` khi tải DistilBERT | Mất mạng | Kiểm tra mạng, lần đầu chạy cần tải model ~250MB |

---

## 📚 Tài liệu tham khảo

- [DistilBERT paper](https://arxiv.org/abs/1910.01108) - Sanh et al., 2019
- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)
- Dataset: [car-price.csv trên Kaggle](https://www.kaggle.com/datasets/qucvinhdng/automobile-dataset)

---

## 👨‍🎓 Thông tin đồ án môn học

- **Môn học:** PPR501 - Phân tích & Trực quan hoá Dữ liệu (hoặc tương đương)
- **Phương pháp:** Transformer (Deep Learning)
- **Bài toán:** Dự báo (Prediction) - Hồi quy (Regression)
- **Khu vực:** TP. Hồ Chí Minh
- **Đối tượng:** Người dân TP.HCM, salon xe cũ

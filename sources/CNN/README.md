# Car-Classification

---

### Tổng quát :

* <p align = "justify">Xây dựng một model có khả năng phân loại xe thành bảy nhóm: Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Maruti Suzuki Swift, Tata Safari và Toyota Innova.</p>

* <p align = "justify">Người dùng cung cấp ảnh xe làm đầu vào; model sẽ phân loại ảnh vào một trong bảy nhóm trên và hiển thị kết quả.</p>

---

### Phương thức :

* <p align = "justify">Hệ thống sử dụng model Convolutional Neural Network (CNN) để phân loại xe trong ảnh đầu vào. Model được huấn luyện qua nhiều epoch để học các đặc trưng của từng nhóm xe và cải thiện độ chính xác.</p>

* <p align = "justify">ImageDataGenerator tạo các biến thể ảnh bằng cách áp dụng những phép biến đổi nhẹ trong quá trình huấn luyện. Việc này giúp model tăng khả năng tổng quát hóa và giảm nguy cơ học thuộc các nhiễu hoặc vị trí đặc trưng cố định trong dữ liệu.</p>

* <p align = "justify">Model được huấn luyện và đánh giá trên tập ảnh của bảy nhóm xe. Sau khi huấn luyện, model có thể dự đoán nhóm xe cho những ảnh mới chưa xuất hiện trong dữ liệu huấn luyện do người dùng cung cấp.</p>

---

### Dataset :

https://www.kaggle.com/code/kshitij192/car-images-classification-using-cnn/data

---

### Hướng dẫn chạy notebook `CNN/main.ipynb`

Các bước dưới đây dành cho Windows.

#### 1. Kiểm tra dữ liệu

Giữ nguyên notebook và thư mục dữ liệu theo cấu trúc sau. Mỗi thư mục con trong `Train` và `Test` là một nhãn xe:

```text
sources/
├── CNN/main.ipynb
└── Dataset/
    ├── Train/
    │   ├── Audi/
    │   └── ...
    └── Test/
        ├── Audi/
        └── ...
```

Notebook đọc dữ liệu bằng các đường dẫn tương đối `Dataset/Train` và `Dataset/Test`, vì vậy hãy chạy notebook với thư mục làm việc là `sources\CNN`.

#### 2. Tạo môi trường Python

Python 3.10 hoặc 3.11 được khuyến nghị. Mở PowerShell tại thư mục gốc của repository và chạy:

```powershell
cd .\sources\CNN
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 3. Cài các thư viện cần thiết

```powershell
py -m pip install --upgrade pip
py -m pip install tensorflow numpy matplotlib gradio pillow ipykernel jupyterlab
py -m ipykernel install --user --name car-analysis --display-name "Python (car-analysis)"
```

#### 4. Mở notebook

Khởi chạy JupyterLab hoặc VS Code tại thư mục `sources\CNN`:

```powershell
jupyter lab main.ipynb
```

Nếu dùng Visual Studio Code, mở thư mục `sources\CNN`, mở `main.ipynb`, rồi chọn kernel **Python (car-analysis)**. Đảm bảo thư mục làm việc của notebook là `sources\CNN` để các đường dẫn dataset ở bước 1 hoạt động.

#### 5. Chạy notebook từ đầu đến cuối

Chọn **Run All** hoặc chạy lần lượt từng cell từ trên xuống dưới, không bỏ qua các cell tạo dữ liệu và model. Cell đầu notebook cài Gradio; có thể đợi cell này hoàn tất trước khi chạy tiếp. Notebook sẽ nạp ảnh train/test, xây dựng CNN và training trong **20 epoch**. Thời gian chạy tùy cấu hình máy; không đóng kernel trong lúc training.

Khi chạy xong, các cell tiếp theo đánh giá model, hiển thị biểu đồ và khởi chạy giao diện dự đoán ảnh.

#### 6. Dùng giao diện dự đoán

Ở cell cuối, chờ Gradio khởi động rồi mở URL được in ra trong output để tải ảnh xe lên và xem kết quả phân loại. Notebook hiện gọi Gradio với `share=True`, vì vậy nó có thể tạo một URL công khai tạm thời; không tải lên ảnh riêng tư hoặc nhạy cảm. Dừng server bằng cách ngắt cell/kernel đang chạy.

#### Lỗi thường gặp

* **Không tìm thấy `Dataset/Train` hoặc `Dataset/Test`:** chạy notebook với thư mục làm việc là `sources\CNN` và kiểm tra tên thư mục `Dataset`, `Train`, `Test`.
* **Không tìm thấy TensorFlow hoặc Gradio:** chọn đúng kernel **Python (car-analysis)**; cài lại thư viện trong chính môi trường `.venv`.
* **PowerShell không cho kích hoạt `.venv`:** dùng trực tiếp `.\.venv\Scripts\python.exe -m pip install ...` và chọn interpreter `sources\.venv\Scripts\python.exe` làm kernel.

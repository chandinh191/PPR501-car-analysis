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

### Hướng dẫn chạy

Hướng dẫn cài môi trường dùng chung, kiểm tra dữ liệu và chạy notebook CNN
được gom tại repository-level [SETUP.md](../../SETUP.md). Cài các thư viện
qua `requirements.txt` ở thư mục gốc trước khi mở notebook.

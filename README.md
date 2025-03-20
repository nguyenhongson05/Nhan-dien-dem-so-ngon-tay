
# 🖐 NHẬN DIỆN & ĐẾM SỐ NGÓN TAY

![dainam](https://github.com/user-attachments/assets/bc536edc-1836-49d0-b8c3-1f139d34276f)

---

## 📌 GIỚI THIỆU DỰ ÁN
Dự án này sử dụng **YOLO** để nhận diện và đếm số ngón tay trong hình ảnh hoặc video theo thời gian thực.  
Mục tiêu chính:
- Phát hiện bàn tay và đếm số ngón tay đang giơ lên.
- Ứng dụng trong giáo dục, giao tiếp phi ngôn ngữ và hỗ trợ người khiếm thính.

---

## 👥 THÀNH VIÊN NHÓM
| Mã sinh viên   | Họ và tên                 | Phân chia công việc          |
|---------------|---------------------------|------------------------------|
| 1771020605    | Nguyễn Hồng Sơn            | Xây dựng mô hình YOLO, xử lý dữ liệu |
| 1771020517    | Nguyễn Trọng Đức Nguyên    | Viết thuật toán đếm số ngón tay |
| 1771020299    | Đinh Văn Hoàng             | Thiết kế giao diện, tối ưu hiệu suất |

---

## 🛠️ CÔNG NGHỆ SỬ DỤNG
- **Ngôn ngữ**: Python
- **Thư viện**: OpenCV, YOLOv5, NumPy, TensorFlow
- **Công cụ hỗ trợ**: Google Colab, Jupyter Notebook

---
🎯 Tính năng
- ✋ Phát hiện bàn tay và nhận diện cử chỉ trong hình ảnh hoặc video.
- 🔤 Nhận dạng ký tự viết tay dựa trên dữ liệu huấn luyện.
- ⚡ Hỗ trợ GPU để tối ưu hiệu suất nhận diện.
- 🔗 Tích hợp dễ dàng với các ứng dụng xử lý ảnh và AI khác.
---
## Các bước thực hiện
- 1️⃣Thu thập dữ liệu
  - Thu thập dữ liệu từ camera mở trực tiếp hoặc từ ảnh có sẵn
- 2️⃣Gán nhãn dữ liệu
  -  Sử dụng công cụ như LabelImg để gán nhãn vị trí bàn tay/ngón tay.
- 3️⃣Huấn luyện mô hình
  - Chỉnh sửa các file cấu hình (.yaml) cho dữ liệu và mô hình.
  - Sử dụng train.py để huấn luyện với tập dữ liệu.
  - Sau khi huấn luyện, mô hình tạo ra file best.pt (trọng số tối ưu nhất).
- 4️⃣Chạy mô hình
  - Dùng mô hình đã huấn luyện để nhận diện.
- 5️⃣Xem kết quả
  - Khi chạy chương trình có thể nhận diện trực tiếp trên camera hoặc có thể lưu ảnh vào thư mục.
![captured_5](https://github.com/user-attachments/assets/f6207e23-bc62-48ec-801a-5891631bdedc)

© 2025 NHÓM 3, NHẬN DIỆN KÝ TỰ TAY SỬ DỤNG KỸ THUẬT HỌC SÂU YOLO VÀ MEDIAPIPE, TRÍ TUỆ NHÂN TẠO, TRƯỜNG ĐẠI HỌC Đại NAM

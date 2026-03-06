# Network Anomaly Detection using Machine Learning (UNSW-NB15)

Dự án này xây dựng hệ thống **phát hiện bất thường trong lưu lượng mạng** bằng các mô hình **Machine Learning** trên bộ dữ liệu **UNSW-NB15**.

Hệ thống bao gồm các bước:

- Phân tích dữ liệu khám phá (EDA)
- Tiền xử lý dữ liệu
- Phân tích độ quan trọng của đặc trưng
- Huấn luyện nhiều mô hình Machine Learning
- Đánh giá mô hình
- Triển khai ứng dụng dự đoán bằng **Streamlit**

---

# 1. Giới thiệu bài toán

Trong các hệ thống mạng hiện đại, việc phát hiện các hành vi bất thường hoặc tấn công mạng là rất quan trọng.

Dự án này giải quyết bài toán **phân loại nhị phân**:

| Label | Ý nghĩa |
|------|------|
| 0 | Normal traffic |
| 1 | Attack traffic |

### Mục tiêu

- Phát hiện được nhiều **Attack** nhất có thể  
- Giảm **False Positive Rate (FPR)**  
- So sánh hiệu quả nhiều mô hình Machine Learning  
- Triển khai mô hình tốt nhất thành **web app**

---

# 2. Bộ dữ liệu

Dự án sử dụng bộ dữ liệu:

**UNSW-NB15 Dataset**

Files sử dụng:
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv


### Thông tin tập training

| Thuộc tính | Giá trị |
|------|------|
| Số mẫu | 175341 |
| Số cột | 45 |

### Phân bố nhãn

| Class | Count |
|------|------|
| Normal | 56000 |
| Attack | 119341 |

---

# 3. Pipeline của dự án

## 3.1 Phân tích dữ liệu (EDA)

Các bước:

- Kiểm tra missing values
- Phân bố class
- Ma trận tương quan
- Phân tích đặc trưng

Ví dụ:

- Class distribution
- Correlation heatmap

---

## 3.2 Tiền xử lý dữ liệu

### Loại bỏ các cột không cần thiếtattack_cat
id
response_body_len
ct_flw_http_mthd
trans_depth
dwin
ct_ftp_cmd
is_ftp_login


---

### Loại bỏ các feature tương quan cao

Sử dụng **Spearman correlation**

Nếu:
|corr| > 0.95


→ giữ feature liên quan label hơn.

---

### Xử lý outlier (clamping)

Nếu:
max(feature) > 10 * median


→ clamp về **95th percentile**

---

### Feature Engineering

Gom nhóm giá trị hiếm:
state
service
proto


---

### One-hot encoding

Sau khi encode:
47 features


---

# 4. Phân tích Feature Importance

Sử dụng **Random Forest**

Các phương pháp:

- Train
- Train 10-fold
- Combined
- Combined 10-fold

### Top feature quan trọng nhất
sttl
ct_state_ttl
dload
sload
dur
synack
dinpkt
sbytes
ct_srv_dst
sinpkt


---

# 5. Huấn luyện mô hình

Các mô hình được thử nghiệm:

- KNN
- MLP
- SVC
- Random Forest
- XGBoost

---

# 6. Chiến lược chọn threshold

Không sử dụng threshold mặc định **0.5**

Thay vào đó chọn threshold sao cho:
Recall_Attack >= 0.90
FPR <= 0.05


Điều này giúp:

- phát hiện được nhiều **attack**
- hạn chế **false alarm**

---

# 7. Kết quả mô hình

| Model | Accuracy | Recall_Attack | FPR | AUC |
|------|------|------|------|------|
| KNN | 0.9328 | 0.9247 | 0.0498 | 0.9871 |
| MLP | 0.9370 | 0.9310 | 0.0502 | 0.9904 |
| SVC | 0.9265 | 0.9151 | 0.0494 | 0.9860 |
| RF | **0.9561** | **0.9579** | **0.0477** | 0.9935 |
| XGB | 0.9541 | 0.9560 | 0.0501 | **0.9937** |

---

# 8. Mô hình tốt nhất

Mô hình được chọn:

**Random Forest**

### Threshold tối ưu
0.641094


### Kết quả

| Metric | Value |
|------|------|
| Accuracy | 0.9561 |
| Recall_Attack | 0.9579 |
| FPR | 0.0477 |
| FAR | 0.0439 |
| AUC | 0.9935 |

---

# 9. Cấu trúc project
machine-learning-network-anomaly-detection
│
├── data
│ ├── raw
│ └── preprocess
│
├── results
│
├── preprocess.ipynb
├── importance.ipynb
├── train.ipynb
│
├── unsw_streamlit_best_model_app
│ ├── app.py
│ ├── requirements.txt
│ ├── models
│ │ └── best_model.joblib
│ └── utils
│
└── README.md


---

# 10. Chạy ứng dụng Streamlit

## Tạo virtual environment (Python >= 3.9)
python -m venv venv

Kích hoạt môi trường
Linux / MacOS
  source venv/bin/activate
Windows (PowerShell)
  venv\Scripts\Activate.ps1

Cài đặt thư viện
  pip install -r requirements.txt

Chạy ứng dụng
  streamlit run app.py

Sử dụng

Upload file CSV thô của UNSW-NB15:

UNSW_NB15_testing-set.csv

Ứng dụng sẽ:

tiền xử lý dữ liệu

chạy model Random Forest

dự đoán Normal / Attack

# 11. Công nghệ sử dụng

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Joblib

Streamlit

# 12. Kết luận

Dự án đã xây dựng hệ thống phát hiện bất thường mạng dựa trên Machine Learning sử dụng bộ dữ liệu UNSW-NB15.

Kết quả cho thấy:

Random Forest là mô hình hiệu quả nhất

đạt Recall_Attack cao

giữ FPR dưới ngưỡng yêu cầu

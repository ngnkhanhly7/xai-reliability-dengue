# Decision log thực nghiệm

Ngày bắt đầu: 06/10/2026 (Asia/Bangkok, cùng UTC+7 với Việt Nam).

## D01 — Chuẩn protocol, trước huấn luyện và trước mở test

Đã đọc báo cáo research và kế hoạch thực nghiệm. Khi cấu hình khác nhau, dùng `ke-hoach-thuc-nghiem-lime-shap-dengue.md` làm protocol vận hành: split seed42 theo test20% rồi validation25% phần còn lại; RF seed11, 200 cây, depth5, leaf3, max_features=sqrt; LR C1; mẫu XAI seed101, stability102, backgrounds201–203, LIME301–305, SHAP yêu cầu32.

Khác với đề xuất sơ bộ trong báo cáo research: không dùng seeds2026/2027, không grid chọn bằng AP, không RF300/max_features=None, không backgrounds4101–4103. Các thay đổi chỉ là thống nhất hai tài liệu trước khi có đầu ra mô hình; không có test prediction được quan sát khi quyết định.

## D02 — Không tuning; máy và pilot

Kế hoạch cho phép giữ baseline khởi đầu nếu chạy bình thường. Khóa cấu hình LR/RF trong `configs/main.yaml`, không mở grid. RF n_jobs=1 và BLAS/OpenMP một thread để runtime có điều kiện rõ. Python3.13 trong `.venv` riêng; dependencies pin trong requirements; log toàn bộ versions và source hash thực sự cài.

Kế hoạch không đặt seed pilot validation: chọn seed103, 3 ca không nặng + 2 ca nặng. Mọi row_id là vị trí dòng CSV bắt đầu từ0, không phải patient ID. Sampling của mẫu XAI dùng NumPy RandomState, chọn trong từng lớp, rồi sort row_id; backgrounds dùng sklearn stratified sampling. Ghi IDs trước model prediction.

## D03 — Phần phụ và báo cáo

Chạy E0–E4 đầy đủ trước; thêm E4c và E5 khi core đã lưu và exact check đạt. RF seeds22/33 chỉ kiểm tra prediction, không thay checkpoint XAI. Bootstrap1000 phân tầng cho AUROC/AP dùng cùng resample indices cho LR/RF; CI có điều kiện trên split/model đã fit.

E5 theo kế hoạch mới: SHAP reference là10 clinical train rows × hai marker values =20 rows, cùng reference cho C/R/B; giải thích bản s=1 trên10 ca stability; test gồm q0.95/0.50/0.05 tổng hợp. Không dùng reference40 rows hoặc gate0.10/0.05 từ research. Xác nhận dependence bằng số đo liên tục trên validation; C phải đạt tolerance1e-12. Không dùng ngưỡng phát hiện marker phổ quát.

## D04 — Diagnostic và ties

Epsilon1e-6 như kế hoạch thực nghiệm (thay đề xuất1e-8 ở research). Giữ coefficient/attribution thô; chỉ dùng epsilon để báo gần0, đổi dấu và near ties. Top3 tie-breaking bằng feature_id; nếu toàn bộ giá trị gần0 ghi top3 undefined thay vì tạo agreement giả. Rank marker vẫn được báo kèm near-zero flag, không ép hệ số C về0.

Weighted R² LIME là in-sample. Bổ sung weighted RMSE/cỡ mẫu hiệu dụng bằng instrumenting LimeBase, không thay objective hoặc sampling. Không loại ca R² thấp. E4c nếu warning singular/rank deficiency sẽ ghi nhận, dừng budget đó theo protocol và không che warning.

## D05 — Hoàn thiện sau khi thực nghiệm đã chạy (06/10/2026)

E0–E5 và các phần bổ sung đã chạy. E4c budget 8 gặp warning thiếu hạng ngay lượt đầu nên dừng budget đó; budgets 16/32 giữ đủ 5 ca × 3 seeds. Không thay cấu hình để ép budget 8 đạt. KernelSHAP chính khớp exact trên 5 pilot với max error 3,33e-16; exact checks E5 cũng đạt.

Sau khi quan sát kết quả, chỉ bổ sung bảng phân bố reference, diễn giải Results/Discussion, kiểm tra hash của ID và schema kết quả, bảo toàn environment snapshot khi resume, và dựng lại report sau bước verification để trạng thái trong report đúng. Không đổi split, seed, checkpoint, ngưỡng, background hoặc attributions. Các edits này thuộc báo cáo/kiểm tra artifact, không phải chọn lại protocol từ test.

Cài dependencies trực tiếp gặp timeout ở CDN PyPI. Đã tải distributions từ PyPI bằng HTTP ranges, kiểm tra SHA-256 với metadata chính thức, rồi cài offline từ cache. pip check đạt; requirements-lock.txt ghi toàn bộ versions thực sự chạy. Cơ chế tải không thay nội dung package.

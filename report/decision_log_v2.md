# Decision log bổ sung v2

Ngày tổng hợp 06/10/2026, sau lượt tính v2. Các quyết định thiết kế được ghi trước lượt tính trong [protocol](supplementary_protocol.md), config và lock; nhật ký này tổng hợp kết quả thực hiện, không tự nhận có timestamp sớm hơn file thực tế.

## V2-01 — Giữ v1 và đánh dấu hậu nghiệm

Chụp v1 gồm code/config/README/report/results/models/figures trước khi sửa code cho v2; archive v1 có 426 files, CRC, SHA-256 và Markdown links đạt. Không ghi lại report hoặc CSV/JSON/figures/model v1. Phần bổ sung dùng thư mục run `dengue_20261006_supp_v2` riêng. Bảng [protocol amendments](protocol_amendments.md) phân biệt đề xuất research với v1 đã chạy, và giới hạn timestamp của D01–D04.

V2 lock lúc 11:41:55.954167 UTC+7, sau main test v1 10:47:36.941869. Phân tích bổ sung là exploratory, không đăng ký trước quan sát test. Chọn 10 validation cases bằng seed 1101 trước prediction/diagnostic mới; không fit lại RF hoặc chọn ngưỡng mới.

## V2-02 — Width và evaluation chung

Chọn 3 fit seeds 301–303, 3 widths, Nfit=2000, Neval=5000. Cùng perturbations giữa widths; evaluation dùng seed+1000 và bỏ dòng ép thành x. Báo native evaluation theo width riêng và evaluation với weights w0 cố định trên cùng điểm. Điều này được quyết định trước kết quả v2, nhằm tránh gọi việc đổi loss weighting là fidelity improvement. Ridge alpha giữ 1; chưa tách interaction/regularization.

## V2-03 — Biểu diễn tuổi và case 97

Giữ thêm ablation continuous-age theo default Gaussian sampler. Đánh giá cả hai representations trên cùng hai raw evaluation pools, weights chung trong mỗi pool. Ghi rõ sampling khi fit cũng đổi, không gán mọi cải thiện cho encoding thuần túy. Dùng sampler private của pinned LIME 0.2.0.1; kiểm tra encoder và 60 fits w0 với public API, max coefficient error 4,996e-16.

Probe tuổi cho 10 ca validation; row 97 test được nêu trước trong phản hồi, chỉ là minh họa hậu nghiệm. Không fit surrogate mới hoặc chọn width trên row 97. Probe giữ bốn binary inputs, cho RF chạy tuổi nguyên trong cùng bin và dùng surrogate quartile v1 đã lưu. RF range 0,301 nhưng indicators/output surrogate không đổi. Không quy toàn bộ residual v1 cho mất thông tin tuổi.

## V2-04 — SHAP reference size

Giữ 5 seeds 201–205, B20 giống policy v1; B100 bao gồm B20 và thêm đủ lớp để tổng có 17 severe. Full train 478 rows là reference empirical chung. Dùng validation cases và full enumeration; không đổi marginal game. 110 explanations và 6 exact checks đạt, max error 7,693e-16. Không mở rộng C/R/B.

## V2-05 — Diễn giải sau kết quả và đóng gói

Sau chạy, bổ sung diễn giải paired effects, output surrogate ngoài [0,1], việc sign flips tập trung vào nhóm hệ số yếu hơn, và sửa cách ghi khoảng tuổi **train 1–95** (toàn CSV 1–97). Đây là sửa report/summary từ dữ liệu đã lưu, không thay raw neighborhoods, coefficient, model, split hoặc config. Code report trước và sau hoàn thiện được phân biệt bằng hashes trong lock và manifest cuối.

Verification đã rerun fresh một ca cho cả hai encodings, coefficients khớp tuyệt đối; 383 files thuộc dữ liệu/report/config/artefacts v1 giữ nguyên hash. Bundle cuối chứa snapshot code v1 và code/results v2; không đóng gói `.venv`/wheels. Cài môi trường bằng requirements-lock.txt. Kiểm tra archive bằng CRC, manifest SHA-256 và local links; số đo runtime luôn có điều kiện trên máy/package versions đã log.

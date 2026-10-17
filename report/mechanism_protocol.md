# Protocol kiểm tra nguyên nhân độ khớp LIME — v3

Ngày phân tích theo client: **07/10/2026**. Protocol này được viết trước phép fit v3, **sau khi đã biết kết quả v1/v2**, kể cả kết quả trên các evaluation pools được tái sử dụng. Đây là phân tích thăm dò có đối chứng; lock trước thực thi không biến nó thành nghiên cứu đăng ký trước hay đánh giá trên một tập chưa từng xem.

## Câu hỏi và phạm vi cố định

V2 cho thấy cấu hình tuổi liên tục khớp RF tốt hơn, nhưng đồng thời đổi encoding, sampler và distances/kernel weights. Cần kiểm tra: giữ nguyên raw fit points, RF probabilities và weights thì thay encoding tuổi còn cải thiện không? Regularization alpha=1 và giới hạn additive của surrogate đóng góp thế nào vào sai số?

Giữ RF v1, split và 10 validation IDs của v2: 287, 369, 401, 492, 494, 519, 545, 639, 707, 754. Tái sử dụng 30 **quartile** fit neighborhoods (2000 điểm, seeds 301–303), không sinh thêm perturbations hay gọi RF trên test. Các target là probabilities của RF đã lưu, không phải nhãn Severity. Trọng số fit luôn exp(-d_quartile²/(2w0²)), w0=0.75sqrt(5); không tính lại distance từ representation mới.

## Năm arms đã chọn trước lượt tính

| Arm | Biểu diễn và surrogate | Mục đích |
| --- | --- | --- |
| quartile_ridge | 5 same-bin/value indicators, linear Ridge alpha=1 | Tái tạo baseline v2 w0, đối chiếu checkpoint/CSV |
| quartile_ols | Cùng 5 indicators, weighted least squares không penalty | Kiểm tra bỏ regularization; giải độc lập bằng numpy.linalg.lstsq với intercept |
| continuous_same_samples | Tuổi chuẩn hóa mean/SD train; 4 binary indicators; Ridge alpha=1 | **Chỉ đổi encoding**, cùng points/targets/weights và 5 hệ số |
| quartile_pairwise | 5 indicators và đủ 10 tích đôi, Ridge alpha=1 | Kiểm tra additive capacity; 15 hệ số, không gọi đây là LIME chuẩn 5 biến |
| quartile_cell_mean | Weighted mean từ fit cho từng pattern trong 32 patterns indicator | Diagnostic capacity của representation; pattern chưa thấy dùng weighted mean toàn fit |

Continuous_same_samples là surrogate có đối chứng, **không phải** chạy lại default continuous-age LIME: sampler và weights cố ý giữ quartile. Pairwise/cell mean là công cụ chẩn đoán, không đề xuất một phương pháp XAI mới hoặc phân bổ interaction coefficients thành SHAP.

## Evaluation và phép phân tích

- Primary: 5000 random points độc lập RNG của v2 quartile pool, cùng raw points, RF targets và quartile w0 weights cho cả năm arms. Không có dòng ép thành x trong evaluation.
- Secondary: cùng v2 Gaussian evaluation pool, cùng native Gaussian w0 weights cho mọi arm. Báo tuổi ngoài khoảng train và RMSE sau loại chúng như sensitivity; không clip tuổi hoặc probability.
- Đo fit/held-out weighted RMSE, R², point error, output ngoài [0,1], số pattern và coverage của cell mean. Point error chỉ kiểm tra ca validation gốc.
- Tổng hợp trung vị ba seeds trong từng ca, sau đó trung vị 10 ca. Paired effects tính chênh trong từng ca; không coi 150 fits hoặc 5000 perturbations là người bệnh độc lập. Chỉ báo mô tả, không thêm p-value/CI với giả định độc lập không kiểm chứng.
- Với evaluation quartile, mọi hàm chỉ nhận indicator pattern z phải có output hằng trong từng cell. Tính weighted within-cell variance của RF trên **evaluation sample**, và phần sai số giữa predicted cell value với weighted evaluation cell mean. Kiểm chứng identity MSE = within-cell variance + between-cell squared error cho bốn arms quartile. Đây là diagnostic/cận dưới **trên sample hữu hạn đã quan sát**, không phải ước lượng vô thiên hay chân lý của population. Evaluation cell means không được dùng để fit hoặc chọn arm.

## Bảo toàn và kiểm tra

Trước fit: ghi config/protocol/code/inputs hashes, model hash, IDs, versions, timestamp của máy và ngày client; đối chiếu inventory ZIP v2. Sau fit: xác minh các file v1/v2 có cùng hash trước lượt tính. Baseline quartile_ridge phải khớp saved v2 coefficients và held-out RMSE; OLS đối chiếu sklearn LinearRegression; pairwise design đối chiếu PolynomialFeatures(interaction_only=True). Một ca/seed được refit để kiểm tra deterministic results và reconstruction từ saved model.

## Quy tắc dừng và cách ra quyết định

Chạy đúng năm arms trên 30 neighborhoods rồi báo mọi kết quả. Không tăng seeds/budget, không thử thêm alpha/width, không chọn mô hình/threshold bằng test. Nếu kiểm tra matched encoding trả lời được khoảng trống của v2, dừng mở rộng thực nghiệm và ưu tiên trình bày contribution, giới hạn, bằng chứng tái lập cho hồ sơ lab. Các hướng profile-group evaluation, calibration, conditional SHAP, dataset/model mới được đánh giá riêng theo câu hỏi và giá trị ứng tuyển; không tự chạy chỉ vì có thể.

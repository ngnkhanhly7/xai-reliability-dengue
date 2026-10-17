# Protocol bổ sung v2 — khóa trước các lượt tính mới

Ngày 06/10/2026. Đây là phân tích **phát sinh sau khi đã xem test v1**. Không thay thế kết quả v1, không huấn luyện RF mới, không chọn lại threshold. Giữ đề tài, CSV, split và checkpoint RF seed 11. V1 được chụp byte-for-byte trong `release/dengue_v1_original.zip` trước sửa code/README để làm phần bổ sung.

## S1 — Chọn ca và khóa đầu vào

Chọn 10 validation rows, 5 mỗi lớp, NumPy RandomState seed 1101; lưu IDs trước tính prediction/diagnostic mới. Chỉ fit thống kê LIME từ 478 train rows của v1. Lưu hash RF, raw CSV, config, IDs, nguồn code và thời điểm khóa. Không dùng các kết quả này để chọn một cấu hình thay thế v1. Các bản CSV/JSON/hình v1 được kiểm tra hash với archive cuối lượt bổ sung.

## S2 — LIME: width và đánh giá ngoài neighborhood fit (A/C)

- 10 ca × 3 seeds (301–303) × 3 widths (0,5w0; w0; 2w0), w0=0,75sqrt(5), quartile train cố định, Ridge alpha=1, tất cả features. Tổng 90 fits.
- Mỗi ca/seed sinh **một** neighborhood fit bằng sampler chính thức LIME 0.2.0.1, 2000 dòng gồm x. Ba widths dùng đúng cùng inputs, indicators, distances và RF outputs; chỉ đổi weights. Không gọi sampler riêng cho mỗi width.
- Evaluation: sampler cùng thống kê/bins, seed=fit_seed+1000, 5000 dòng độc lập; sinh 5001 rồi bỏ dòng đầu bị LIME ép thành x. Không fit lại surrogate trên evaluation.
- Lưu in-sample weighted R²/RMSE, point error, held-out weighted R²/RMSE, effective N, coefficients và 3 cặp seed từng ca. Giữ R² âm và ca khớp kém.
- Báo hai cách đánh giá held-out: weights theo width của fit (mô tả neighborhood riêng), và **weights cố định w0 trên cùng evaluation points** (so sánh width trên cùng tiêu chí). Không tuyên bố cải thiện chỉ từ đổi trọng số đánh giá. Alpha giữ 1; thay tổng weights cũng thay strength regularization tương đối, chưa tách riêng yếu tố này.
- Kiểm tra tự dựng fit w0 khớp API LIME chính thức trên cùng seed/ca tới 1e-10; đối chiếu transform với encoder của package. Dùng private sampler đã ghi source hash; không giả định tương thích với phiên bản khác.

## S3 — Biểu diễn tuổi và probe cùng khoảng (B)

Thêm 30 fits tuổi liên tục, w0, cùng RF/binary features/2000 samples/3 seeds. Sampler LIME mặc định Gaussian theo train (sample_around_instance=False); ghi tỷ lệ tuổi nằm ngoài khoảng train, không clip âm thầm. Numeric age của surrogate chuẩn hóa bằng mean/SD train, categorical là indicator bằng giá trị ca.

Mỗi surrogate quartile/continuous được đánh giá trên **cả hai cùng raw evaluation pools**: pool quartile, weights quartile w0; pool Gaussian, weights continuous w0. Encoder của mỗi surrogate biến đổi raw inputs theo biểu diễn riêng. So RMSE/R² trong cùng pool/weights; không xếp hạng bằng R² native trên hai distribution khác nhau. Đây là thay đổi đồng thời representation và sampling khi fit, chưa phải tác động riêng của encoding. Không trộn trị tuyệt đối coefficient tuổi liên tục với coefficient indicator để gọi là cùng attribution units.

Probe tuổi cho 10 validation cases và row 97 đã được phản hồi nêu đích danh: giữ sex/NS1/IgM/IgG, chạy tuổi nguyên trong cùng quartile train. Lưu f_RF(age), indicator và g_quartile(age). Đo range f; nếu range>0 trong một biểu diễn không đổi, có bằng chứng mất thông tin. range/2 là lower bound minimax residual của mọi hàm hằng trên **grid này**, không phải phần trăm toàn bộ LIME error do chia khoảng. Row 97 thuộc test v1, probe này là minh họa hậu nghiệm, không dùng để chọn cấu hình hoặc làm bằng chứng xác nhận độc lập.

## S4 — KernelSHAP: kích thước background

Dùng cùng 10 validation cases. Mỗi size 20/100 có 5 seeds 201–205; full 478 train rows dùng một reference. Tổng 110 explanations. Background 20 dùng quy tắc v1; B100 giữ nguyên B20 và thêm đủ từng lớp từ phần train còn lại (seed+10000), để tổng 100 có số severe gần natural prevalence. Lưu ID và nested policy; seeds là reference replicates, không là bệnh nhân độc lập.

KernelExplainer scalar P(Severity=1), identity, l1_reg=0, nsamples=32 và kiểm tra 30 masks thực khi M=5. Full train chỉ là empirical reference chung. Lưu baseline shift, L1/L∞ attribution error tương đối full train, top-3 Jaccard, sign flips (epsilon 1e-6), init/explain runtime và RF rows. Independent exact enumeration trên 2 ca đầu tiên × sizes 20/100/full, seed 201 khi có sampling; tolerance 1e-6. Tổng hợp metric theo ca trước, rồi qua 10 ca; baseline variation tính trên 5 references, không nhân lên theo ca.

## S5 — Diễn giải đổi dấu v1 và đóng gói

Phân tích lại cache v1, không chạy lại test. Với từng ca/budget, xếp features theo |median coefficient| qua 5 seeds: top-2 và 3 còn lại. Ghi feature, median/SD, số cặp đổi dấu và độ lớn ở các cặp đổi dấu. Đây là phân nhóm tương đối cho mô tả, không là ngưỡng ý nghĩa lâm sàng.

Thêm protocol amendments, report v2, hình PNG/PDF, code/config và dữ liệu thô vào ZIP độc lập. Kiểm tra checksum, CRC và link Markdown sau đóng gói. README phải chỉ rõ v1 nguyên bản, v2 hậu nghiệm và các lệnh dựng lại report/kiểm tra. Không mở rộng marker C/R/B, không đổi dataset, không tuning model hoặc threshold trong v2.

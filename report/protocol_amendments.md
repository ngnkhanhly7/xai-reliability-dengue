# Phụ lục — Protocol amendments

Ngày tổng hợp: 06/10/2026. Báo cáo research là đề xuất trước thực nghiệm; kế hoạch thực nghiệm và config frozen xác định những gì v1 thực sự chạy. Không sửa hai tài liệu lịch sử để làm chúng trông như đã có cùng cấu hình từ đầu.

| Thành phần | Đề xuất trong research | V1 thực sự chạy | Lý do/căn cứ | Thời điểm, đã xem test? |
| --- | --- | --- | --- | --- |
| Split | seeds 2026/2027 | seed 42 cho hai bước phân tầng, 478/160/160 | Thống nhất với kế hoạch thực nghiệm, D01 | D01 mô tả quyết định trước predictions; config cuối khóa 10:47:30 UTC+7, chưa mở test |
| RF | 300 cây, max_features=None, grid | 200 cây, depth=5, leaf=3, max_features=sqrt, seed=11 | Giữ baseline kế hoạch, giảm scope tuning; D01/D02 | Trước model/test theo log; config khóa trước test |
| Chọn cấu hình | AP validation, grid | Không tuning; RF chọn trước cho XAI | Kế hoạch cho phép không tuning; kiểm tra XAI là trọng tâm | D02; chưa xem test |
| Reference seeds | 4101–4103 | 201–203, 20 dòng mỗi B | Theo kế hoạch thực nghiệm, D01 | Trước predictions theo log; IDs hash khóa trước test |
| Marker SHAP background | 20 clinical rows × s0/s1 = 40 | 10 clinical rows × s0/s1 = 20 | Scope nâng cao trong kế hoạch, D03 | D03 mô tả thiết kế trước main test; E5 thực thi sau E0–E4, khi main test đã mở |
| Marker explanations | Cả s=0 và s=1 | Chỉ giải thích s=1 trên 10 ca; vẫn đo paired p0/p1 | Theo kế hoạch; 60 explanations và kiểm chứng reliance riêng | Thiết kế D03 trước main test theo log; marker validation checkpoint trước marker test |
| Marker test environments | Một marker Bernoulli 0,5 độc lập ở test chính; không dùng test labels để tạo feature chính | Ba môi trường q=0,95/0,50/0,05 tạo marker bằng nhãn có chủ đích | Stress-test tương quan tổng hợp theo kế hoạch mới; tách hoàn toàn khỏi baseline năm biến | D03 mô tả thiết kế trước main test; môi trường synthetic thực thi sau core, khi nhãn test đã có |
| Marker gate | Các ngưỡng mean delta 0,10/0,05 | Báo dependence liên tục; chỉ C có tolerance 1e-12 | Không áp gate có tính tùy chọn của research vào kế hoạch mới | D03; không tune B bằng test |
| Epsilon signs/ties | 1e-8 | 1e-6; mass denominator 1e-12 | Theo kế hoạch, D04; thang số học, không lâm sàng | Trước các diagnostic theo log |
| Phần bổ sung v2 | Chưa có kết quả để đặt hypothesis | Width/held-out/continuous-age; age probe; B20/B100/full; sign-flip magnitude | Phản hồi dựa trên điểm yếu đã quan sát | **Sau khi đã xem test v1**; config/IDs khóa trước lượt tính v2, không chọn lại model v1 |

## Bằng chứng thời gian và giới hạn của log

- V1 bắt đầu 2026-10-06T10:47:29.288879+07:00.
- [Checkpoint khóa v1](../artifacts/dengue_20261006_v1/configuration_locked_before_test.json): 10:47:30.261025, gồm config/model/case/background hashes.
- [Run manifest v1](../artifacts/dengue_20261006_v1/run_manifest.json): test_first_opened_at=10:47:36.941869.
- [Decision log](decision_log.md) mô tả các quyết định D01–D04 trước predictions. **Không có timestamp riêng cho từng mục D01–D04**, nên không suy ra giờ/phút chính xác cho từng quyết định từ timestamp chỉnh sửa chung của file. Bằng chứng máy xác nhận cấu hình cuối đã được khóa trước lượt test; log bằng văn bản mô tả lý do và thứ tự.
- [Protocol bổ sung](supplementary_protocol.md) ghi rõ v2 hậu nghiệm. V2 ghi thời điểm khóa và IDs trong artifacts riêng trước tính diagnostics mới. Không gọi v2 là nghiên cứu confirmatory được đăng ký trước khi quan sát test.

Methods v1 nằm trong [báo cáo thực nghiệm nguyên bản](../bao-cao-thuc-nghiem-lime-shap-dengue.md). Báo cáo v2 trình bày Methods bổ sung và liên kết bộ bằng chứng, không thay kết quả baseline hoặc marker v1.

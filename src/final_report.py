"""Assemble the executed v1 and post-test v2 into one standalone review document."""
import re

from .common import ROOT, now


def section(document, start, end):
    return document[document.index(start):document.index(end)].strip()


def body_with_prefix(document, prefix):
    # Demote child headings first, then label the original numbered sections.
    document = re.sub(r"^### (.*)$", r"#### \1", document, flags=re.MULTILINE)
    document = re.sub(r"^## (\d+)\. (.*)$", lambda m: f"### {prefix}.{m[1]} — {m[2]}", document, flags=re.MULTILINE)
    return document


def build_final_report(store):
    v1 = (ROOT / "bao-cao-thuc-nghiem-lime-shap-dengue.md").read_text(encoding="utf-8")
    v2 = (ROOT / "bao-cao-bo-sung-lime-shap-dengue-v2.md").read_text(encoding="utf-8")
    main_v1 = section(v1, "## 1. Kết quả chính", "## 2. Protocol đã dùng").split("\n", 1)[1].strip()
    main_v2 = section(v2, "## 1. Kết quả chính", "## 2. Thiết kế").split("\n", 1)[1].strip()
    core = body_with_prefix(section(v1, "## 2. Protocol đã dùng", "## 10. Tái lập"), "I")
    supplementary = body_with_prefix(section(v2, "## 2. Thiết kế", "## 8. Protocol amendments"), "II")
    tail = body_with_prefix(v2[v2.index("## 8. Protocol amendments"):].strip(), "III")
    text = f"""# Báo cáo tổng hợp: độ tin cậy của LIME và Kernel SHAP trong phân loại dengue nặng

Bản đọc/nộp thống nhất, tạo lúc {now()} (UTC+7). Thực nghiệm gồm v1 `{store.cfg['supplementary']['parent_run']}` và phần bổ sung `{store.run_id}`. Methods dưới đây mô tả **cấu hình thực sự chạy**; các đề xuất khác trong research được giữ như lịch sử và đối chiếu ở phụ lục amendments.

V1 đã khóa trước lượt test. **V2 phát sinh sau khi xem test v1**, dùng validation cho ablation; probe riêng ca test 97 là minh họa hậu nghiệm đã nêu trước. Kết quả v1 được bảo toàn, v2 không dùng để chọn lại checkpoint, split hoặc threshold. [V1 nguyên bản](bao-cao-thuc-nghiem-lime-shap-dengue.md), [report v2 riêng](bao-cao-bo-sung-lime-shap-dengue-v2.md), [code/bộ bằng chứng v1 nguyên gốc](release/dengue_v1_original.zip).

## Kết quả nổi bật

{main_v1}

{main_v2}

## I. Thực nghiệm chính v1 — Methods và kết quả đã khóa

{core}

## II. Phân tích bổ sung v2 — Methods và các phép kiểm tra có đối chứng

{supplementary}

## III. Amendments, phạm vi kết luận và tái lập

{tail}
"""
    path = ROOT / "bao-cao-tong-hop-lime-shap-dengue.md"
    path.write_text(text, encoding="utf-8")
    store.stage("final_report", "complete", {"path": path.name, "v1_source_unchanged": True,
                "v2_is_post_test": True, "same_relative_artifact_links": True})


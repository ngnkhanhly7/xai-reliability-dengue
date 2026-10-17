import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FEATURES, ROOT, now, read_json, write_json

COLORS = {"C": "#475569", "R": "#d97706", "B": "#2563eb"}


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if value != 0 and abs(value) < .5 * 10**(-digits):
            return f"{value:.2g}"
        return f"{value:.{digits}f}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(df):
    columns = list(df.columns)
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n" + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))


def save_figure(store, fig, stem):
    fig.savefig(store.figures / (stem + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(store.figures / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def lookup(records, rid, method, background="201", context="main", seed=301, budget=None, arm="RF_seed11"):
    options = [r for r in records if r["row_id"] == rid and r["explainer"] == method and r["split"] == "test" and r["context"] == context and r["seed"] == seed and r["model_id"] == arm and (method == "LIME" or r["background_id"] == background) and (budget is None or r["budget_requested"] == budget)]
    assert len(options) == 1, (rid, method, len(options))
    return options[0]


def build_report(store):
    store.check_stage("E4")
    results = store.results
    read = lambda filename: pd.read_csv(results / filename)
    records = [read_json(p) for p in sorted((results / "explanation_cache").glob("*.json"))]
    t1, t2, t3, t4 = map(read, ["T1_split_counts.csv", "T2_test_metrics.csv", "T3_exact_verification.csv", "T4_main_comparison.csv"])
    stable = read("stability_lime.csv")
    bg = read("sensitivity_shap_background.csv")
    environment = read_json(store.artifacts / "environment.json")
    audit = read_json(store.artifacts / "dataset_audit.json")
    selected = read_json(store.artifacts / "splits/explanation_ids.json")
    roles = {}
    for group in ["TP", "FN", "FP"]:
        subset = t4[t4.confusion_group.eq(group)]
        if len(subset):
            rid = int(subset.row_id.min())
            roles.setdefault(rid, []).append(group)
    worst = stable[stable.budget.eq(2000)].sort_values(["jaccard_mean", "row_id"]).iloc[0]
    roles.setdefault(int(worst.row_id), []).append("lowest stability at N=2000")
    write_json(results / "case_study_ids.json", {"roles": roles, "selection_rule": "smallest row_id in TP/FN/FP within main20, plus lowest mean seed-pair Jaccard at N2000; ties row_id; repeated case combines roles"})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(roles), 2, figsize=(13, 3.8 * len(roles)), squeeze=False)
    case_sections = []
    for i, (rid, labels) in enumerate(roles.items()):
        row = t4[t4.row_id.eq(rid)].iloc[0]
        lr = lookup(records, rid, "LIME", budget=2000)
        sr = lookup(records, rid, "KernelSHAP", budget=32)
        for col, record in enumerate([lr, sr]):
            a = np.asarray(record["attributions"])
            names = record["conditions"] if col == 0 else [f"{name}={v:g}" for name, v in zip(FEATURES, record["input"])]
            ax = axes[i, col]
            ax.barh(np.arange(5), a, color=["#2563eb" if v >= 0 else "#dc2626" for v in a])
            ax.set_yticks(np.arange(5), names)
            ax.invert_yaxis()
            ax.axvline(0, color="#64748b", lw=.8)
            diagnostic = f"in-sample R²={fmt(record['lime_r2'])}; |g-f|={fmt(record['point_error'])}" if col == 0 else f"baseline={fmt(record['baseline'])}; additivity={record['additivity_error']:.1e}"
            ax.set_title(f"Row {rid}: {record['explainer']}\ny={int(row.y_true)}, p={row.predicted_probability:.3f}; {diagnostic}")
            ax.set_xlabel("Coefficient of displayed condition" if col == 0 else "Attribution relative to reference (probability units)")
        values = pd.DataFrame({"feature": FEATURES, "input": np.asarray(lr["input"], dtype=int), "LIME_condition": lr["conditions"], "LIME_coefficient": lr["attributions"], "Kernel_SHAP": sr["attributions"]})
        ltop = FEATURES[int(np.argmax(np.abs(lr["attributions"])))]
        stop = FEATURES[int(np.argmax(np.abs(sr["attributions"])))]
        case_sections.append(f"### Ca row_id={rid}: {'; '.join(labels)}\n\nNhãn thật={int(row.y_true)}, p={row.predicted_probability:.3f}, dự đoán={int(row.predicted_label)}; nhóm {row.confusion_group}. Top-3 Jaccard={row.top3_jaccard:.3f}. Biến có độ lớn lớn nhất: LIME {ltop}, SHAP {stop}. LIME g(x)={fmt(lr['local_pred'])}, R²={fmt(lr['lime_r2'])}, residual tại ca={fmt(lr['point_error'])}; SHAP baseline={fmt(sr['baseline'])}. Dấu LIME gắn với điều kiện ở bảng, dấu SHAP gắn với reference; không đọc chúng thành causal effect.\n\n{table(values)}\n\n")
    fig.tight_layout(pad=2.0)
    save_figure(store, fig, "F1_case_studies")
    budgets = sorted(stable.budget.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, field, title in zip(axes, ["jaccard_mean", "mean_feature_sd"], ["LIME seed stability per case", "Coefficient variation per case"]):
        values = [stable[stable.budget.eq(b)][field].to_numpy() for b in budgets]
        ax.boxplot(values, tick_labels=[str(b) for b in budgets])
        for rid, group in stable.groupby("row_id"):
            series = group.set_index("budget").loc[budgets, field]
            ax.plot(range(1, len(budgets) + 1), series, color="#2563eb", alpha=.25, lw=.8, marker=".")
        ax.set_xlabel("Perturbation budget")
        ax.set_title(title + " (10 cases)")
        ax.set_ylabel("Mean of 10 seed-pair Jaccards" if field == "jaccard_mean" else "Mean SD across 5 coefficients")
    axes[0].axhline(.46, color="#d97706", ls="--", label="uniform random top-3 expectation")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(store, fig, "F2_lime_stability")
    references = {key: lookup(records, selected["stability"][0], "KernelSHAP", background=key, budget=32)["baseline"] for key in ["201", "202", "203"]}
    background_description = pd.DataFrame([
        {"background": key, "n": values["n"], "severe": values["severe"],
         "age_mean": values["age_mean"], **values["binary_frequencies"],
         "baseline": references[key]}
        for key, values in audit["background_summaries"].items()
    ])
    background_description.to_csv(results / "background_summary.csv", index=False)
    pair_names = [("201", "202"), ("201", "203"), ("202", "203")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(list(references), list(references.values()), color="#2563eb")
    axes[0].set_title("Expected model output on B")
    axes[0].set_ylabel("Baseline probability")
    for ax, field, title in zip(axes[1:], ["attribution_l1_shift", "jaccard"], ["Attribution L1 shift", "Top-3 Jaccard"]):
        data = [bg[bg.background_a.astype(str).eq(a) & bg.background_b.astype(str).eq(b)][field].to_numpy() for a, b in pair_names]
        ax.boxplot(data, tick_labels=[a + "/" + b for a, b in pair_names])
        ax.set_title(title + " (10 cases)")
        ax.set_xlabel("Background pair")
    axes[2].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(store, fig, "F3_shap_background")
    budget_summary = stable.groupby("budget").agg(n_cases=("row_id", "size"), median_jaccard=("jaccard_mean", "median"), min_jaccard=("jaccard_mean", "min"), median_mean_feature_sd=("mean_feature_sd", "median"), median_r2=("median_r2", "median"), median_point_error=("median_point_error", "median"), cases_with_sign_flip=("sign_flip_feature_pairs", lambda v: int((v > 0).sum())), near_tie_runs=("near_tie_runs", "sum"))
    budget_summary.to_csv(results / "T4_lime_budget_summary.csv")
    bg_summary = bg.groupby(["background_a", "background_b"]).agg(n_cases=("row_id", "size"), median_jaccard=("jaccard", "median"), median_l1_shift=("attribution_l1_shift", "median"), max_l1_shift=("attribution_l1_shift", "max"), baseline_shift=("baseline_shift_absolute", "first")).reset_index()
    bg_summary.to_csv(results / "T4_shap_background_summary.csv", index=False)
    core_records = [r for r in records if r["context"] == "main" and r["split"] == "test" and r["row_id"] in selected["main"] and r["seed"] == 301 and ((r["explainer"] == "LIME" and r["budget_requested"] == 2000) or (r["explainer"] == "KernelSHAP" and r["budget_requested"] == 32 and r["background_id"] == "201"))]
    runtime = pd.DataFrame(core_records).groupby("explainer").agg(n=("row_id", "size"), median_explain_seconds=("runtime_seconds", "median"), median_initialization_seconds=("initialization_seconds", "median"), median_predict_batches=("call_count", "median"), median_predict_rows=("predicted_row_count", "median"), median_initialization_rows=("initialization_row_count", "median")).reset_index()
    runtime.to_csv(results / "runtime_main_summary.csv", index=False)
    root_results = f"results/{store.run_id}"
    root_artifacts = f"artifacts/{store.run_id}"
    root_figures = f"figures/{store.run_id}"
    rf, lr = t2[t2.model_id.eq("RF_seed11")].iloc[0], t2[t2.model_id.eq("LR")].iloc[0]
    intervals = read("bootstrap_intervals.csv")
    small = read("shap_small_budget.csv")
    tiny = small.groupby("budget").agg(runs=("row_id", "size"), median_exact_error=("max_absolute_error", "median"), max_exact_error=("max_absolute_error", "max"), rank_warnings=("rank_warning", "sum")).reset_index()
    tiny["median_exact_error"] = tiny.median_exact_error.map(lambda x: f"{x:.3g}")
    tiny["max_exact_error"] = tiny.max_exact_error.map(lambda x: f"{x:.3g}")
    t3out = t3[["row_id", "y_true", "effective_features", "mask_count", "max_absolute_error", "additivity_error", "lime_r2", "lime_point_error"]].copy()
    for column in ["max_absolute_error", "additivity_error"]:
        t3out[column] = t3out[column].map(lambda x: f"{x:.3g}")
    completed_scope = "E0–E5" if store.manifest["stages"].get("E5", {}).get("status") == "complete" else "E0–E4"
    sd100 = budget_summary.loc[100, "median_mean_feature_sd"]
    sd2000 = budget_summary.loc[2000, "median_mean_feature_sd"]
    worst_id = int(worst.row_id)
    worst_lime = lookup(records, worst_id, "LIME", budget=2000)
    worst_main = t4[t4.row_id.eq(worst_id)].iloc[0]
    fn_ids = [rid for rid, labels in roles.items() if "FN" in labels]
    case_interpretation = ""
    if fn_ids:
        fn_id = fn_ids[0]
        fn_shap = lookup(records, fn_id, "KernelSHAP", budget=32)
        positive = [FEATURES[j] for j, phi in enumerate(fn_shap["attributions"]) if phi > 1e-6]
        negative = [FEATURES[j] for j, phi in enumerate(fn_shap["attributions"]) if phi < -1e-6]
        case_interpretation += f"Ở FN row {fn_id}, các phi dương ({', '.join(positive) or 'không có'}) và âm ({', '.join(negative) or 'không có'}) cộng với baseline tạo ra xác suất dưới ngưỡng. Điều này mô tả RF, không giải thích vì sao người bệnh thực sự nặng. "
    case_interpretation += f"Ở ca có độ ổn định thấp nhất row {worst_id}, LIME g(x)≈{worst_lime['local_pred']:.3f} so với f(x)≈{worst_lime['predicted_probability']:.3f}, chênh khoảng {100 * worst_lime['point_error']:.1f} điểm phần trăm. Jaccard giữa hai phương pháp ở seed chính là {worst_main.top3_jaccard:.3f}, trong khi trung bình Jaccard giữa seed LIME là {worst.jaccard_mean:.3f}. Agreement giữa phương pháp vì thế không thay thế kiểm tra fidelity hoặc ổn định theo seed."
    age_bins = "; ".join(worst_lime["age_bins"])
    sign_flip_description = "; ".join(f"N={b}: {int(budget_summary.loc[b, 'cases_with_sign_flip'])}/10 ca" for b in budgets)
    near_tie_count = int(budget_summary.near_tie_runs.sum())
    marker_overview = ""
    if completed_scope == "E0–E5":
        marker_validation = read("T5_marker_validation_dependence.csv").set_index("arm")
        marker_environments = read("T5_marker_environment_metrics.csv")
        b_env = marker_environments[marker_environments.arm.eq("B")].set_index("environment")
        marker_overview = f"\nĐối chứng E5 xác nhận mô hình B phụ thuộc mạnh vào marker tổng hợp: mean |Δp| trên validation **{marker_validation.loc['B', 'mean_abs_delta']:.3f}**. Khi tương quan marker–nhãn đảo chiều ở test, AUROC của B đổi từ **{b_env.loc['q095', 'auroc']:.3f}** xuống **{b_env.loc['q005', 'auroc']:.3f}**. Đây là phép thử shortcut có kiểm soát trên dữ liệu synthetic.\n"
    text = f"""# Báo cáo thực nghiệm LIME và Kernel SHAP trên dengue

Ngày chạy: 06/10/2026, UTC+7. Run ID: `{store.run_id}`. Báo cáo tạo lúc {now()} từ CSV/JSON kết quả thực; cấu hình khóa tại [config_frozen.json]({root_artifacts}/config_frozen.json).

## 1. Kết quả chính

Đã chạy **{completed_scope}** theo protocol, gồm audit/split, LR/RF, pilot LIME/Kernel SHAP, kiểm chứng Shapley chính xác, giải thích 20 ca và kiểm tra độ nhạy trên 10 ca. RF test AUROC **{rf.auroc:.3f}**, AP **{rf.ap:.3f}**; LR AUROC **{lr.auroc:.3f}**, AP **{lr.ap:.3f}**. RF ở ngưỡng 0,5 bỏ sót **{int(rf.FN)}/{int(rf.severe)}** ca nặng trên toàn test. Đây là kết quả trên một split cố định, không phải đánh giá triển khai lâm sàng.

Kernel SHAP khi liệt kê đủ mask khớp phép tính Shapley độc lập trên 5 ca validation: sai số lớn nhất **{t3.max_absolute_error.max():.3g}**, nhỏ hơn tolerance 1e-6. LIME trên 20 ca có R² in-sample trung vị **{t4.lime_r2.median():.3f}**, residual tại ca lớn nhất **{t4.lime_point_error.max():.3f}**. Jaccard top-3 giữa hai phương pháp có trung vị **{t4.top3_jaccard.median():.3f}**; đây là mức nhất trí, không phải độ chính xác của lời giải thích.

Tăng budget LIME từ 100 lên 2000 làm trung vị độ lệch chuẩn hệ số giảm từ **{sd100:.3f}** xuống **{sd2000:.3f}**, nhưng trung vị Jaccard giữa seed không tăng. Với SHAP, thay background làm cả baseline và attribution thay đổi dù phép tính từng game đã chính xác. Vì vậy, độ đúng số học, độ khớp surrogate và độ ổn định phải được đánh giá riêng.
{marker_overview}

## 2. Protocol đã dùng

Ba câu hỏi thực nghiệm: hai phương pháp giải thích cùng dự đoán ra sao (RQ1); lời giải thích đổi thế nào khi đổi seed, budget hoặc background (RQ2); và chúng có phản ánh sự phụ thuộc đã xác nhận vào marker tổng hợp hay không (RQ3).

Áp dụng [kế hoạch thực nghiệm](ke-hoach-thuc-nghiem-lime-shap-dengue.md); các khác biệt so với đề xuất seed/grid ban đầu trong [research](bao-cao-research-lime-shap-dengue.md) được ghi **trước predictions** trong [decision log](report/decision_log.md). Split seed 42: test 20%, rồi validation 25% phần còn lại. RF: 200 cây, max_depth=5, min_samples_leaf=3, max_features=sqrt, seed=11, n_jobs=1. LR: C=1, solver=lbfgs, max_iter=2000, chuẩn hóa riêng tuổi bằng train. Cả hai dùng class_weight=None. Không tuning hoặc refit train+validation. Ngưỡng 0,5; model XAI chính là RF seed 11. Mẫu 20 ca gồm 10 mỗi nhãn, seed 101; stability 10 ca, seed 102. Background 20 dòng phân tầng từ train, seeds 201–203; LIME seeds 301–305. Pilot lấy 3 ca không nặng và 2 ca nặng ở validation, seed 103. E2 khóa checkpoint/config trước E3/test.

LIME chia tuổi theo quartile train, bốn biến còn lại là categorical; dùng đủ 5 features, Ridge alpha=1, kernel_width=0.75sqrt(5)≈1,677. Các khoảng tuổi học từ train: {age_bins}. Hệ số gắn với indicator “cùng giá trị/khoảng với ca đang giải thích”, không trực tiếp là hệ số trên tuổi liên tục hoặc mã xét nghiệm gốc. Tại ca gốc mọi indicator bằng 1, nên g(x)=intercept+tổng hệ số. R² được tính với trọng số kernel trên chính neighborhood đã fit; residual tại ca là |g(x)−f(x)|. Đây là diagnostic in-sample. [LIME API](https://lime-ml.readthedocs.io/en/latest/lime.html).

Kernel SHAP nhận hàm scalar f(x)=P_model(Severity=1), identity link, l1_reg=0.0 và nsamples=32. Game đã chọn là v_x(S)=mean_b∈B f(x_S,b_notS): giữ cột trong S của x, lấy các cột còn lại từ **cùng một dòng** b, rồi trung bình trên B. Không lấy độc lập mỗi cột từ một người khác. Exact enumeration tính đủ 32 coalition và tổng marginal contributions với trọng số Shapley; mã này độc lập bộ giải SHAP. Baseline phi0=mean_B f(b), tổng phi_j=f(x)−phi0. Có 30 mask không tầm thường khi M=5; hai endpoint được xử lý riêng. [Kernel SHAP API](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).

## 3. E0 — Dữ liệu/split (T1)

{table(t1)}

CSV có 798 dòng, 660 ca không nặng và 138 ca nặng; tuổi 1–97, không ô thiếu. Đầu vào theo thứ tự age/sex/NS1/IgM/IgG; RNA bỏ ngoài phạm vi. sex=0 là nữ, sex=1 là nam; xét nghiệm 0/1 là âm/dương tính; y=Severity=1 là ca nặng theo nhãn phát hành. Bài toán là phân loại mức độ nặng hồi cứu, không đặt một thời hạn dự báo chưa có trong dữ liệu.

SHA-256 CSV `{audit['sha256']}`. Row_id bắt đầu từ 0 chỉ là vị trí dòng. Có **{audit['test_rows_whose_profile_is_seen_in_train']}** dòng test có cùng profile 5 biến với ít nhất một dòng train; có {audit['profile_overlap']['train_test']} profile unique chung giữa train/test. Không xóa profile vì không xác nhận được người bệnh trùng. Thiếu ID/timestamp nên không bảo đảm tính độc lập theo bệnh nhân/thời gian; không gọi đây là external validation.

Nguồn cohort/endpoint/mã biến: [Huang et al. (2020)](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008960). [Audit]({root_artifacts}/dataset_audit.json), [split IDs]({root_artifacts}/splits/split_ids.json), [case IDs]({root_artifacts}/splits/explanation_ids.json), [background IDs]({root_artifacts}/splits/background_ids.json).

## 4. E1 — Predictive baselines (T2)

{table(t2[['model_id','auroc','ap','brier','balanced_accuracy','sensitivity','specificity','TP','FN','FP','TN']])}

Prevalence test={rf.prevalence:.3f}. AP theo average_precision_score, không thay bằng PR-AUC hình thang. Probability chưa được calibration riêng. Predictive performance và explanation fidelity là các câu hỏi khác nhau.

CI 95% từ 1000 bootstrap phân tầng theo nhãn, seed 701; cùng indices resample cho LR/RF. CI chỉ có điều kiện trên các checkpoint/split này, không bao gồm bất định do chọn train set hoặc quan hệ giữa người bệnh chưa biết.

{table(intervals[['model_id','metric','lower95','upper95']])}

[Validation metrics]({root_results}/T2_validation_metrics.csv), [test predictions]({root_results}/predictions_test.csv), [bootstrap samples]({root_results}/bootstrap_samples.csv), [RF seeds 22/33 bổ sung]({root_results}/additional_rf_seed_metrics.csv). Các seed bổ sung không thay checkpoint XAI.

LR có AUROC/AP test cao hơn RF trong split này, nhưng CI của chênh lệch RF−LR đều chứa 0. Chưa có cơ sở khẳng định mô hình nào vượt trội trên quần thể. RF vẫn được giữ làm model XAI vì đã chọn trước test. Ở ngưỡng 0,5, sensitivity thấp của cả hai cho thấy nhiều ca nặng bị bỏ sót; một lời giải thích có thể đúng với hành vi mô hình ngay cả khi dự đoán sai.

## 5. E2 — Numerical verification và pilot (T3)

{table(t3out)}

Các phi full-mask cộng về xác suất và khớp exact trên cùng reference. R² thấp hay residual lớn của LIME vẫn được giữ; không loại ca hoặc đổi kernel để cải thiện hình. Exact chỉ đúng với empirical game này, không phải ground truth sinh học. [Sai số từng feature]({root_results}/exact_validation_features.csv); 5 pilot neighborhoods lưu NPZ trong artifacts để debug.

## 6. E3 — So sánh 20 ca và case studies (T4/F1)

Trong tập cân bằng 20 ca, các nhóm dự đoán: {t4.confusion_group.value_counts().to_dict()}. Tỷ lệ của nhóm này không phải prevalence quần thể. R² LIME khoảng {t4.lime_r2.min():.3f}–{t4.lime_r2.max():.3f}; point residual khoảng {t4.lime_point_error.min():.3f}–{t4.lime_point_error.max():.3f}. SHAP additivity residual lớn nhất {t4.shap_additivity_error.max():.3g}. Không trừ trực tiếp coefficients LIME và phi SHAP để gọi là sai số vì các tham chiếu khác nhau.

Case study: row_id nhỏ nhất trong TP/FN/FP nếu nhóm tồn tại; thêm ca có độ ổn định thấp nhất ở N=2000 theo trung bình 10 cặp seed, ties theo row_id. Nếu một ca có hai vai trò thì ghi cả hai, không chọn ca khác để ép đủ 4 hình.

![F1: Cùng ca và đầu ra, hai phương pháp]({root_figures}/F1_case_studies.png)

{''.join(case_sections)}

{case_interpretation}

**Diễn giải cần kiểm chứng tiếp:** chia tuổi theo quartile làm nhiều giá trị tuổi có cùng indicator, trong khi RF vẫn nhận tuổi gốc và có thể đổi dự đoán trong cùng khoảng. Sự khác nhau về biểu diễn và reference có thể góp phần vào các bất đồng ở trên; thực nghiệm hiện tại chưa tách riêng ảnh hưởng từng yếu tố, nên không quy mọi residual cho discretization.

[Dữ liệu 20 ca]({root_results}/T4_main_comparison.csv), [attributions/conditions dài]({root_results}/explanations_main.csv), [metadata/diagnostic]({root_results}/explanation_diagnostics.csv). Mỗi explanation truy được về run/config/row/seed/budget/reference.

## 7. E4 — Stability và background sensitivity (F2/F3)

### LIME sampling

Thiết kế 10 ca × 3 budgets × 5 seeds = 150 explanations, trong đó 10 lượt N=2000/seed=301 được tái sử dụng từ E3. Jaccard tính trên 10 cặp seed trong từng ca, sau đó tổng hợp 10 ca; không coi các cặp là người bệnh độc lập. Top-3 dựa trên trị tuyệt đối; tie bằng nhau được phá theo feature_id. Epsilon 1e-6 dùng để đánh dấu gần 0 và near-tie ở ranh giới hạng 3/4; nếu tất cả hệ số gần 0, top-3/Jaccard ghi NA.

{table(budget_summary.reset_index())}

![F2: Distribution theo 10 ca, không gộp các cặp seed]({root_figures}/F2_lime_stability.png)

Top-3 của 5 biến có expected Jaccard 0,46 nếu chọn ngẫu nhiên đều/độc lập; đó chỉ là mốc toán học. Tăng budget không được giả định cải thiện từng ca. Coefficient SD, IQR và đổi dấu cũng đã lưu trong [feature stability]({root_results}/stability_lime_features.csv); [case stability]({root_results}/stability_lime.csv) và [seed pairs]({root_results}/stability_lime_pairs.csv).

`median_jaccard` là trung vị qua 10 ca của **trung bình 10 cặp seed** từng ca; `min_jaccard` là mức thấp nhất của các trung bình từng ca. Số ca có ít nhất một feature đổi dấu giữa seed: {sign_flip_description}. Có {near_tie_count} lượt near-tie ở ranh giới hạng 3/4 trên toàn thiết kế này. Các hệ số yếu có thể đổi dấu khi SD giảm; cần đọc trị tuyệt đối và IQR từng feature, không coi mọi đổi dấu là thay đổi có ý nghĩa y học. Median SD giảm khoảng {100 * (1 - sd2000 / sd100):.0f}% từ N=100 đến N=2000, trong khi median Jaccard lần lượt là {', '.join(fmt(budget_summary.loc[b, 'median_jaccard']) for b in budgets)}. Tăng số mẫu giảm biến động hệ số nhưng chưa tạo ra thứ hạng top-3 nhất quán hơn trong thiết kế này.

### Kernel SHAP reference

10 ca × 3 backgrounds = 30 explanations full-mask; 10 lượt B201 tái dùng E3. Baseline của ba B: {', '.join(k+'='+fmt(v) for k,v in references.items())}. Attribution đổi khi reference đổi là sensitivity của game; không tự gọi là numerical error. Các B đều theo cùng stratified sampling policy từ train, không đổi model hoặc conditional/marginal definition.

{table(background_description)}

Các cột sex/NS1/IgM/IgG là tỷ lệ giá trị 1 trong B. Mỗi B đều có 3/20 ca nặng, nhưng tuổi và tần suất xét nghiệm khác nhau. B202 có tuổi trung bình thấp hơn và baseline thấp hơn; đây là mô tả đồng thời, chưa phải phân tích nhân quả của chênh lệch baseline. [Phân bố reference]({root_results}/background_summary.csv).

{table(bg_summary)}

![F3: Baseline, attribution shift và agreement theo reference]({root_figures}/F3_shap_background.png)

Nguồn lý do phân biệt marginal/conditional: [Aas et al.](https://arxiv.org/html/1903.10464). Ba reference 20 dòng là sensitivity nhỏ; chưa chứng minh robustness trên mọi reference. [Từng background pair]({root_results}/sensitivity_shap_background.csv), [từng feature]({root_results}/sensitivity_shap_features.csv).

### SHAP budget nhỏ trên validation (E4c)

{table(tiny)}

Tất cả warning được lưu. Nếu rank-deficient/singular, dừng budget đó theo protocol. Full-mask không có sampling variance là hợp lệ; additivity tốt ở budget nhỏ chưa bảo đảm phi từng biến đúng. [Chi tiết budget/seeds/warnings]({root_results}/shap_small_budget.csv).

### Chi phí thực tế của lượt chính

{table(runtime)}

Tách initialization khỏi explain time; SHAP baseline initialization cần đánh giá reference. Số predicted rows gồm các đầu vào synthetic thực sự qua wrapper, không chỉ mask count. Runtime có điều kiện trên máy/versions/n_jobs đã log; hai budget không được gọi là cùng chi phí hoặc cùng estimand.
"""
    if store.manifest["stages"].get("E5", {}).get("status") == "complete":
        dependence = read("T5_marker_validation_dependence.csv")
        mark = read("marker_explanations.csv")
        env = read("T5_marker_environment_metrics.csv")
        summary = mark.groupby(["arm", "explainer"]).agg(n_cases=("row_id", "size"), median_abs_delta=("abs_delta", "median"), median_marker_mass=("marker_mass", "median"), median_marker_rank=("marker_rank", "median"), top3_fraction=("marker_in_top3", "mean"), max_abs_marker=("marker_attribution", lambda v: v.abs().max())).reset_index()
        summary.to_csv(results / "T5_marker_attribution_summary.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
        for ax, method in zip(axes[:2], ["LIME", "KernelSHAP"]):
            for arm in ["C", "R", "B"]:
                g = mark[mark.arm.eq(arm) & mark.explainer.eq(method)]
                ax.scatter(g.abs_delta, g.marker_mass, label=arm, color=COLORS[arm], alpha=.8)
            ax.set_title(method + ": known synthetic marker")
            ax.set_xlabel("|p(s=1)-p(s=0)|")
            ax.set_ylabel("Marker mass within method")
            ax.set_ylim(-.02, 1.02)
            ax.legend()
        for arm in ["C", "R", "B"]:
            g = env[env.arm.eq(arm)].sort_values("q_requested", ascending=False)
            axes[2].plot(g.q_requested.astype(str), g.auroc, label=arm, color=COLORS[arm], marker="o")
        axes[2].set_ylabel("AUROC on the same test rows")
        axes[2].set_xlabel("Requested marker-label match q")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Synthetic correlation shift")
        axes[2].legend()
        fig.tight_layout()
        save_figure(store, fig, "F4_marker_benchmark")
        text += f"""

## 8. E5 — Đối chứng marker tổng hợp (T5/F4)

Core E0–E4 hoàn thành trước khi chạy E5. C bỏ marker bằng thiết kế; R train marker random; B train marker khớp nhãn với q=0,95. C/R/B cùng hyperparameters/reference; clinical reference 10 train rows lặp s=0/s=1 thành 20 dòng, LIME dùng toàn train balanced marker pool. Giải thích bản s=1 của 10 ca stability cho cả ba arms. Không tune để ép B học marker. Dependence được đo trên 160 validation rows trước marker test evaluation.

{table(dependence[['arm','mean_abs_delta','median_abs_delta','max_abs_delta','prediction_flip_fraction']])}

[Validation checkpoint trước marker test]({root_artifacts}/marker_validation_checkpoint.json). Các mean |Δp| là mức dependence quan sát, không phải ngưỡng phát hiện phổ quát. C có Δp bằng 0 theo wrapper; R có thể học noise trên finite sample. [Tỷ lệ marker matching thực tế từng lớp/seed]({root_artifacts}/splits/marker_background_ids.json).

{table(summary)}

Mass được tính trong từng explainer; không đồng nhất coefficient mass LIME với attribution mass SHAP. Marker gần 0/undefined denominator được ghi riêng; không ép coefficient C bằng 0. Kernel SHAP đã được exact-check trên 5 ca validation mỗi arm, bao gồm dummy marker của C. [Marker exact errors]({root_results}/marker_exact_verification.csv).

Trong arm B, marker đứng hạng nhất ở cả 10 ca với cả hai phương pháp, phù hợp mức dependence đã xác nhận. Ở C, exact Shapley của marker bằng 0 theo thiết kế; các hệ số nhỏ của LIME thể hiện ảnh hưởng fit/sampling, và marker không lọt top-3. Ở R, cơ chế tạo marker độc lập với nhãn nhưng RF vẫn học được dependence trên mẫu hữu hạn. Vì vậy, attribution khác 0 ở R không tự động là false alarm: phải đối chiếu với |Δp| của **mô hình đã fit**, thay vì chỉ với cơ chế sinh feature. Chưa đặt một ngưỡng phân loại “phát hiện shortcut” phổ quát từ 10 ca.

![F4: Dependence, attribution marker và synthetic shift]({root_figures}/F4_marker_benchmark.png)

{table(env[['arm','environment','q_requested','matching_actual','matching_class0','matching_class1','auroc','ap','balanced_accuracy']])}

q=0,95/0,50/0,05 là ba bản synthetic của **cùng 160 test rows**, được tạo từ nhãn có chủ đích để đánh giá shortcut; không phải 3 cohort độc lập hay feature triển khai. Không trộn các số này vào baseline năm biến. [Environment predictions]({root_results}/marker_environment_predictions.csv), [paired dependence]({root_results}/marker_dependence.csv), [marker explanation từng ca]({root_results}/marker_explanations.csv). Logic xác nhận reliance/false alarm lấy từ [Adebayo et al.](https://arxiv.org/html/2212.04629); benchmark này chỉ kiểm tra marker đã biết.
"""
    verification = read_json(results / "verification.json") if (results / "verification.json").exists() else None
    verification_link = f"[Verification JSON]({root_results}/verification.json)." if verification else ""
    text += f"""

## 9. Giới hạn và phạm vi kết luận

RQ1: hai phương pháp nhất trí một phần về top-3 nhưng có ca ưu tiên feature và hướng đóng góp khác nhau; các diagnostic LIME cho thấy surrogate chưa khớp tốt ở nhiều ca. RQ2: hệ số LIME ít biến động hơn khi tăng budget, còn thứ hạng vẫn đổi theo seed; SHAP tính chính xác cho từng game vẫn nhạy với background. RQ3: trong đối chứng tổng hợp, cả hai phản ánh marker mà B phụ thuộc mạnh, và đối chứng C/R giúp phân biệt nhiễu fit với dependence mô hình thực sự học. Những kết quả này hỗ trợ đánh giá từng khía cạnh độ tin cậy, không tạo ra một thứ hạng thắng–thua chung giữa LIME và SHAP.

Kết quả là nghiên cứu hồi cứu một split, cohort DENV-2 Đài Loan, test nhỏ và lớp nặng ít; thiếu ID/timestamp, nhiều profile đầu vào giống nhau. Không patient-wise/temporal/external/prospective validation. Reference20 dòng và perturbations chưa được thẩm định clinical plausibility. Exact chỉ kiểm tra game đã chọn; surrogate fidelity/seed stability/agreement không chứng minh causal effect hoặc an toàn điều trị. Một RF chính không đại diện mọi mô hình. Bất đồng và ca khớp kém được giữ trong bảng, không chọn phương pháp thắng dựa trên một diagnostic.

CI bootstrap theo dòng còn giả định độc lập đơn vị dòng; thiếu patient ID nên không xác thực được giả định đó. Dữ liệu synthetic marker không chứng minh phát hiện shortcut chưa biết trong lâm sàng. Phân tích bổ sung RF seeds 22/33 và E4c được ghi riêng, không dùng để thay model/config sau khi xem test.

## 10. Tái lập và artifact

Máy: {environment['processor']}; {environment['physical_cpus']} physical/{environment['logical_cpus']} logical CPUs; RAM {environment['ram_gib']:.1f} GiB; RF n_jobs=1, BLAS/OpenMP 1 thread. Python: {environment['python'].split()[0]}. Package versions: {', '.join(k+'='+v for k,v in environment['packages'].items())}. Runtime không tách khỏi cấu hình máy này.

[Environment/source hashes]({root_artifacts}/environment.json), [run manifest]({root_artifacts}/run_manifest.json), [README](README.md), [requirements](requirements.txt), [requirements lock](requirements-lock.txt), [decision log](report/decision_log.md).

```powershell
.\\.venv\\Scripts\\python.exe -m src.run_experiments --run-id {store.run_id} --stage all
```

Lượt chạy lại cùng run_id đọc checkpoint/cache; dùng run_id mới để fit/chạy toàn bộ độc lập. Các JSON cache lưu ngay sau từng ca để resume. CSV/figures được dựng từ cache/result tables. F1–F4 có PNG và PDF trong figures của run. Không cần notebook state hoặc GPU. Cache có {len(records)} explanation records độc nhất; explanations_main.csv có 200 dòng feature (20 ca × 2 phương pháp × 5 features). [Toàn bộ attributions]({root_results}/explanations_all.csv) và [diagnostics]({root_results}/explanation_diagnostics.csv) gồm cả pilot, sensitivity, budget nhỏ và marker; các lượt dùng lại chỉ lưu một bản.

Trạng thái kiểm tra tái lập: {'đạt; rerun một ca cho mỗi explainer khớp với kết quả đã lưu, max difference LIME='+str(verification['lime_rerun_max_absolute_difference'])+', SHAP='+str(verification['shap_rerun_max_absolute_difference']) if verification else 'chưa chạy bước verification cuối; xem manifest khi pipeline hoàn tất'}. {verification_link}
"""
    (ROOT / "bao-cao-thuc-nghiem-lime-shap-dengue.md").write_text(text, encoding="utf-8")
    store.stage("report", "complete", {"path": "bao-cao-thuc-nghiem-lime-shap-dengue.md", "case_ids": list(roles)})

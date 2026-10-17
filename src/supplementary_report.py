import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, now, read_json
from .reporting import fmt, save_figure, table


def paired_boxplot(ax, frame, group_field, value_field, groups, labels, title, ylabel):
    values = [frame[frame[group_field].eq(g)][value_field].dropna().to_numpy() for g in groups]
    ax.boxplot(values, tick_labels=labels)
    for _, case in frame.groupby("row_id"):
        y = case.set_index(group_field).reindex(groups)[value_field]
        ax.plot(np.arange(1, len(groups) + 1), y, color="#2563eb", alpha=.25, marker=".", lw=.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def build_report(store):
    for stage in ["S2_S3_LIME", "S3_age_probe", "S4_SHAP", "S5_sign_flips", "verification"]:
        store.check_stage(stage)
    result_dir = f"results/{store.run_id}"
    artifact_dir = f"artifacts/{store.run_id}"
    figure_dir = f"figures/{store.run_id}"
    read = lambda name: pd.read_csv(store.results / name)
    fit, stable = read("lime_ablation.csv"), read("lime_stability.csv")
    metric_columns = ["point_error", "fit_r2", "fit_rmse", "heldout_native_r2", "heldout_native_rmse",
                      "heldout_common_quartile_r2", "heldout_common_quartile_rmse",
                      "heldout_common_continuous_r2", "heldout_common_continuous_rmse", "fit_effective_n"]
    per_case = fit.groupby(["representation", "width_factor", "row_id"])[metric_columns].median().reset_index()
    per_case = per_case.merge(stable, on=["representation", "width_factor", "row_id"], validate="one_to_one")
    summary = per_case.groupby(["representation", "width_factor"])[metric_columns + ["mean_pair_jaccard", "mean_feature_sd"]].median().reset_index()
    summary.insert(2, "n_cases", 10)
    per_case.to_csv(store.results / "lime_per_case_summary.csv", index=False)
    summary.to_csv(store.results / "lime_summary.csv", index=False)
    quartile = per_case[per_case.representation.eq("quartile")]
    reference = quartile[quartile.width_factor.eq(1)].set_index("row_id")
    paired = []
    for factor in [.5, 2.]:
        candidate = quartile[quartile.width_factor.eq(factor)].set_index("row_id")
        for field in ["point_error", "heldout_common_quartile_rmse", "heldout_common_continuous_rmse"]:
            delta = candidate[field] - reference[field]
            paired.append({"width_factor": factor, "metric": field, "n_cases": len(delta),
                           "median_paired_delta_vs_w0": delta.median(), "cases_lower_error": int((delta < -1e-12).sum()),
                           "cases_higher_error": int((delta > 1e-12).sum())})
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(store.results / "lime_paired_width_effects.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, field, title, ylabel in zip(axes,
            ["point_error", "heldout_common_quartile_rmse", "mean_pair_jaccard"],
            ["Point error", "Independent evaluation, fixed weights", "Seed stability"],
            ["|g(x)-f(x)|", "Weighted RMSE on shared quartile pool", "Mean Jaccard over 3 seed pairs"]):
        paired_boxplot(ax, quartile, "width_factor", field, [.5, 1., 2.], ["0.5w0", "w0", "2w0"], title + " (10 validation cases)", ylabel)
        ax.set_xlabel("Fit kernel width")
    axes[2].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(store, fig, "F5_lime_width_heldout")
    probe, probe_summary = read("age_within_bin_probe.csv"), read("age_within_bin_summary.csv")
    row97 = probe_summary[probe_summary.row_id.eq(97)].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    single = probe[probe.row_id.eq(97)]
    axes[0].plot(single.age, single.rf_probability, marker="o", label="RF, same four binary inputs")
    axes[0].plot(single.age, single.quartile_surrogate_probability, color="#dc2626", ls="--", label="LIME v1, identical indicators")
    axes[0].scatter([row97.original_age], [row97.original_probability], color="#111827", zorder=5, label="Original case")
    axes[0].set_title("Row 97: post-hoc age probe in one quartile")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Probability / surrogate output")
    axes[0].legend(fontsize=8)
    validation_probe = probe_summary[probe_summary.split.eq("validation")].sort_values("rf_range")
    axes[1].barh(validation_probe.row_id.astype(str), validation_probe.rf_range, color="#2563eb")
    axes[1].set_title("Age variation hidden by one indicator (validation)")
    axes[1].set_ylabel("Row ID")
    axes[1].set_xlabel("Max RF probability - min RF probability on integer grid")
    fig.tight_layout()
    save_figure(store, fig, "F6_age_within_bin")
    bg, shp = read("shap_background_descriptions.csv"), read("shap_vs_full_train.csv")
    bg_summary = bg.groupby("size").agg(n_references=("background_id", "size"), baseline_mean=("baseline", "mean"),
        baseline_sd=("baseline", "std"), baseline_min=("baseline", "min"), baseline_max=("baseline", "max"),
        median_baseline_abs_error_vs_full=("baseline_abs_error_vs_full", "median")).reset_index()
    shp_per_case = shp.groupby(["size", "row_id"])[["attribution_l1_vs_full", "attribution_linf_vs_full", "top3_jaccard_vs_full", "sign_flips_vs_full"]].mean().reset_index()
    shp_summary = shp_per_case.groupby("size").agg(n_cases=("row_id", "size"), median_case_mean_l1=("attribution_l1_vs_full", "median"),
        median_case_mean_linf=("attribution_linf_vs_full", "median"), median_case_mean_jaccard=("top3_jaccard_vs_full", "median"),
        mean_case_sign_flips=("sign_flips_vs_full", "mean")).reset_index()
    runtime = shp.groupby("size").agg(n_explanations=("row_id", "size"), median_init_seconds=("initialization_seconds", "median"),
        median_explain_seconds=("explain_seconds", "median"), median_init_rows=("initialization_rows", "median"),
        median_explain_rows=("predict_rows", "median")).reset_index()
    bg_summary.to_csv(store.results / "shap_baseline_size_summary.csv", index=False)
    shp_per_case.to_csv(store.results / "shap_background_per_case.csv", index=False)
    shp_summary.to_csv(store.results / "shap_background_size_summary.csv", index=False)
    runtime.to_csv(store.results / "shap_background_runtime.csv", index=False)
    full_baseline = float(bg[bg["size"].eq(478)].baseline.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for seed in [201, 202, 203, 204, 205]:
        pair = bg[bg.seed.eq(seed)].sort_values("size")
        axes[0].plot([1, 2], pair.baseline, color="#2563eb", alpha=.6, marker="o", lw=1)
    axes[0].axhline(full_baseline, color="#d97706", ls="--", label="Full train reference")
    axes[0].set_xticks([1, 2], ["20", "100"])
    axes[0].set_xlabel("Background size (5 nested references)")
    axes[0].set_ylabel("Baseline probability")
    axes[0].set_title("Variation in baseline")
    axes[0].legend(fontsize=8)
    for ax, field, title in zip(axes[1:], ["attribution_l1_vs_full", "top3_jaccard_vs_full"], ["L1 distance to full train", "Top-3 agreement with full train"]):
        paired_boxplot(ax, shp_per_case[shp_per_case["size"].isin([20, 100])], "size", field, [20, 100], ["20", "100"], title + " (10 cases)", "Mean across 5 references per case")
        ax.set_xlabel("Background size")
    axes[2].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(store, fig, "F7_shap_background_size")
    sign_features, sign_pairs = read("v1_sign_flip_features.csv"), read("v1_sign_flip_pairs.csv")
    signs = sign_features.groupby(["budget", "feature_group"]).agg(n_case_features=("feature_name", "size"),
        opposite_sign_pairs=("opposite_sign_pairs", "sum"), eligible_pairs=("n_pairs", "sum"),
        median_abs_coefficient=("median_coefficient", lambda v: v.abs().median())).reset_index()
    signs["opposite_pair_fraction"] = signs.opposite_sign_pairs / signs.eligible_pairs
    magnitudes = sign_pairs.groupby(["budget", "feature_group"]).agg(median_larger_abs_at_flip=("larger_absolute_coefficient", "median"),
        max_larger_abs_at_flip=("larger_absolute_coefficient", "max")).reset_index()
    signs = signs.merge(magnitudes, on=["budget", "feature_group"], how="left")
    signs.to_csv(store.results / "v1_sign_flip_summary.csv", index=False)
    parent_cache = ROOT / "results" / store.cfg["supplementary"]["parent_run"] / "explanation_cache"
    old_records = [read_json(p) for p in parent_cache.glob("*.json")]
    retention = []
    for (row_id, budget), case in sign_features.groupby(["row_id", "budget"]):
        target = set(case[case.feature_group.eq("top2_by_abs_median")].feature_name)
        case_records = [r for r in old_records if r["context"] == "main" and r["split"] == "test"
                        and r["explainer"] == "LIME" and r["row_id"] == row_id and r["budget_requested"] == budget]
        for r in case_records:
            ordering = sorted(range(5), key=lambda j: (-abs(r["attributions"][j]), j))[:2]
            retention.append({"row_id": row_id, "budget": budget, "seed": r["seed"],
                              "same_top2_as_median": set(r["feature_names"][j] for j in ordering) == target})
    rank_retention = pd.DataFrame(retention)
    rank_retention.to_csv(store.results / "v1_top2_rank_retention.csv", index=False)
    rank_summary = rank_retention.groupby("budget").agg(n_runs=("seed", "size"),
        top2_same_as_median=("same_top2_as_median", "sum")).reset_index()
    rank_summary.to_csv(store.results / "v1_top2_rank_summary.csv", index=False)
    validation_ids = read_json(store.artifacts / "splits/validation_ids.json")
    locked = read_json(store.artifacts / "supplementary_locked_before_execution.json")
    checked = read_json(store.results / "verification.json")
    parent = store.cfg["supplementary"]["parent_run"]
    train_ids = read_json(ROOT / "artifacts" / parent / "splits/split_ids.json")["train"]
    training_ages = pd.read_csv(ROOT / store.cfg["data"]["path"]).age.iloc[train_ids]
    outside_surrogate = fit.assign(local_pred_outside_01=fit.local_pred.lt(0) | fit.local_pred.gt(1)).groupby(
        ["representation", "width_factor"]
    ).agg(n_fits=("row_id", "size"), local_pred_outside_01=("local_pred_outside_01", "sum")).reset_index()
    outside_surrogate.to_csv(store.results / "lime_surrogate_bounds.csv", index=False)
    a_table = summary[summary.representation.eq("quartile")][["width_factor", "n_cases", "point_error", "fit_r2", "fit_rmse", "heldout_native_r2", "heldout_native_rmse", "heldout_common_quartile_r2", "heldout_common_quartile_rmse", "mean_pair_jaccard", "mean_feature_sd"]]
    b_table = summary[summary.width_factor.eq(1)][["representation", "n_cases", "point_error", "heldout_common_quartile_rmse", "heldout_common_quartile_r2", "heldout_common_continuous_rmse", "heldout_common_continuous_r2"]]
    native = fit.groupby("representation").agg(fit_outside_age_fraction=("fit_age_outside_train_fraction", "median"), eval_outside_age_fraction=("eval_age_outside_train_fraction", "median")).reset_index()
    q = summary[summary.representation.eq("quartile")].set_index("width_factor")
    wider = q.loc[2., "heldout_common_quartile_rmse"] - q.loc[1., "heldout_common_quartile_rmse"]
    narrower = q.loc[.5, "heldout_common_quartile_rmse"] - q.loc[1., "heldout_common_quartile_rmse"]
    continuous = summary[summary.representation.eq("continuous")].iloc[0]
    signs_main = signs[signs.budget.eq(2000)].set_index("feature_group")
    narrow_point = paired_df[paired_df.width_factor.eq(.5) & paired_df.metric.eq("point_error")].iloc[0]
    narrow_common = paired_df[paired_df.width_factor.eq(.5) & paired_df.metric.eq("heldout_common_quartile_rmse")].iloc[0]
    bg20, bg100 = bg_summary[bg_summary["size"].eq(20)].iloc[0], bg_summary[bg_summary["size"].eq(100)].iloc[0]
    error20, error100 = shp_summary[shp_summary["size"].eq(20)].iloc[0], shp_summary[shp_summary["size"].eq(100)].iloc[0]
    delta_sd = 100 * (1 - bg100.baseline_sd / bg20.baseline_sd)
    age_cases = int((validation_probe.rf_range > 1e-6).sum())
    amendments = (ROOT / "report/protocol_amendments.md").read_text(encoding="utf-8")
    amendments_table = amendments[amendments.index("| Thành phần"):amendments.index("## Bằng chứng")].strip()
    text = f"""# Báo cáo bổ sung v2: nguyên nhân LIME khớp kém và kích thước background SHAP

Ngày thực nghiệm 06/10/2026 (UTC+7), run `{store.run_id}`. Tạo report lúc {now()}. **Phân tích phát sinh sau khi đã xem test v1**; không thay thế kết quả [v1](bao-cao-thuc-nghiem-lime-shap-dengue.md). Đề tài, CSV, split 478/160/160, RF seed 11 và ngưỡng 0,5 giữ nguyên. Báo cáo v1, hình/kết quả/checkpoint nguyên gốc được kiểm tra hash và lưu trong [archive v1](release/dengue_v1_original.zip).

## 1. Kết quả chính

Đã chạy 120 fits LIME trên **10 validation cases**, gồm 90 quartile/width fits và 30 continuous-age fits; mỗi fit được đánh giá trên 5000 điểm độc lập với neighborhood fit. Các widths dùng chung evaluation points; 120 fits không là 120 ca hoặc 120 tập đánh giá độc lập. Có 60 đối chiếu với public LIME API, max coefficient error **{checked['official_api_max_error']:.2g}**. Đã chạy 110 KernelSHAP explanations với B20/B100/full train, kiểm chứng exact trên 6 lượt, max error **{checked['shap_exact_max_error']:.2g}**.

Width 0,5w0 giảm point error ở **{int(narrow_point.cases_lower_error)}/10** ca nhưng không cải thiện RMSE trên pool quartile/weights chung: cả {int(narrow_common.cases_higher_error)}/10 ca có RMSE tăng nhẹ, paired delta trung vị **{narrow_common.median_paired_delta_vs_w0:+.2g}**. Cấu hình tuổi liên tục giảm median held-out RMSE trên cùng pool quartile từ **{q.loc[1., 'heldout_common_quartile_rmse']:.3f}** xuống **{continuous.heldout_common_quartile_rmse:.3f}**, và trên pool Gaussian từ **{q.loc[1., 'heldout_common_continuous_rmse']:.3f}** xuống **{continuous.heldout_common_continuous_rmse:.3f}**. Đây là đối chiếu trên cùng evaluation inputs/weights, nhưng continuous fit cũng dùng sampler khác.

Probe ca test 97 xác nhận RF đổi xác suất từ **{row97.rf_min:.3f}** đến **{row97.rf_max:.3f}** khi tuổi đổi từ {int(row97.age_min)} đến {int(row97.age_max)} với bốn biến còn lại giữ nguyên. Quartile LIME v1 vẫn cho cùng giá trị **{row97.surrogate_constant:.3f}** vì indicators không đổi. Đây là bằng chứng trực tiếp về thông tin tuổi RF sử dụng mà biểu diễn đó làm mất; chưa quy toàn bộ residual cho nguyên nhân này.

Tăng background từ 20 lên 100 làm SD baseline qua 5 seeds đổi từ **{bg20.baseline_sd:.3f}** sang **{bg100.baseline_sd:.3f}** ({delta_sd:.0f}% giảm trong 5 references đã thử). Trung vị qua ca của mean L1 distance tới full train đổi từ **{error20.median_case_mean_l1:.3f}** sang **{error100.median_case_mean_l1:.3f}**. Full train có baseline **{full_baseline:.3f}**; đây là reference empirical, không phải chân lý lâm sàng.

## 2. Thiết kế, đối chứng và thời điểm khóa

[Protocol bổ sung](report/supplementary_protocol.md) và [config](configs/supplementary.yaml) được ghi trước lượt tính v2. Lock lúc **{locked['locked_at']}**, sau thời điểm mở test v1 **{locked['v1_test_first_opened_at']}**. Chọn validation seed 1101, 5 ca mỗi nhãn: {validation_ids['row_ids']}. Không chọn ca theo dự đoán/diagnostic mới, không fit lại RF, không chọn một width để thay v1. [Lock/config/model/ID hashes]({artifact_dir}/supplementary_locked_before_execution.json), [validation IDs]({artifact_dir}/splits/validation_ids.json).

Sampling/statistics/bins LIME fit bằng train; Ridge alpha=1, feature_selection=none, Nfit=2000, seeds 301–303. Mỗi ca/seed sinh một fit neighborhood, dùng chung inputs, indicators, distances và RF outputs cho ba widths. Evaluation seed=fit_seed+1000; sinh 5001 rồi bỏ dòng đầu bị sampler ép thành ca x để còn 5000 random points. Surrogate đã fit không cập nhật từ evaluation. Weighted RMSE/R² được đối chiếu riêng với sklearn metrics. [Raw neighborhoods + outputs]({artifact_dir}/neighborhoods/), [surrogate checkpoints]({artifact_dir}/surrogates/).

Kernel mặc định k(d)=exp(−d²/(2w²)). Với w0=0,75sqrt(5), mẫu khác cả 5 indicators có d²=5 và weight **{np.exp(-5 / (2 * (.75 * np.sqrt(5))**2)):.3f}**; tại 0,5w0 weight **{np.exp(-5 / (2 * (.5 * .75 * np.sqrt(5))**2)):.3f}**; tại 2w0 weight **{np.exp(-5 / (2 * (2 * .75 * np.sqrt(5))**2)):.3f}**. Đây là suy luận từ kernel thực sự cài, không phải kết quả đo clinical locality. [LIME API](https://lime-ml.readthedocs.io/en/latest/lime.html); installed source hash trong [environment]({artifact_dir}/environment.json).

## 3. A/C — Kernel width và fidelity trên điểm độc lập

Mọi số trong bảng là trung vị qua 10 ca; mỗi ca lấy trung vị metric qua 3 seeds. Jaccard/SD được tính trong từng ca trước tổng hợp. `fit_*` dùng neighborhood fit. `heldout_native_*` dùng evaluation points độc lập nhưng weights theo width riêng. `heldout_common_quartile_*` dùng **cùng raw evaluation points và cùng weights w0** cho tất cả fits, để giảm nhiễu khi so width. Native R² không tự là cùng mục tiêu đánh giá nếu weights đổi.

{table(a_table)}

![F5: Paired width ablation và held-out fidelity]({figure_dir}/F5_lime_width_heldout.png)

So với w0, chênh giữa các trung vị RMSE trên pool quartile/weights cố định là **{narrower:+.2g}** ở 0,5w0 và **{wider:+.2g}** ở 2w0. Bảng paired bên dưới tính chênh trong từng ca rồi tổng hợp, không trừ các trung vị để coi là paired effect.

{table(paired_df)}

Thu hẹp kernel làm surrogate ưu tiên vùng gần x hơn và giảm point error ở nhiều ca; RMSE chung vẫn xấp xỉ 0,149 ở cả ba widths. R² fit và held-out native gần nhau cho thấy độ khớp thấp còn xuất hiện ngoài perturbation đã fit, không chỉ là một diagnostic thuận lợi/bất lợi trên dữ liệu fit. Trong thiết kế này, đổi width chưa giải quyết được độ khớp surrogate trên cùng miền đánh giá.

Width cũng đổi tổng sample weights, nên với alpha cố định, strength Ridge tương đối đổi; chưa tách regularization khỏi locality. Chưa làm ablation alpha hoặc surrogate có interaction. Không dùng một R² tốt hơn trên weights khác để tuyên bố LIME được sửa toàn diện. Ridge surrogate không bị ràng buộc trong [0,1]; các output ngoài khoảng được giữ nguyên khi tính loss, không clip để làm point error đẹp hơn:

{table(outside_surrogate)}

[120 fit diagnostics]({result_dir}/lime_ablation.csv), [600 coefficients]({result_dir}/lime_coefficients.csv), [40 case/config stability rows]({result_dir}/lime_stability.csv), [per-case aggregates]({result_dir}/lime_per_case_summary.csv).

## 4. B — Quartile và tuổi liên tục trên cùng evaluation pools

Tuổi liên tục dùng Gaussian theo mean/SD train, sample_around_instance=False; bốn binary indicators giữ quy tắc LIME. Coefficient tuổi liên tục gắn với tuổi chuẩn hóa, coefficient quartile gắn với một điều kiện bằng bin: không cùng đơn vị để so attribution magnitude ngang hàng. **Thay representation cũng đổi sampler khi fit**; kết quả không tách được hiệu ứng encoding thuần túy.

Cả hai surrogate được chạy trên cùng pool quartile với weights quartile w0, và trên cùng pool Gaussian với weights Gaussian w0. So sánh trong từng cặp cột, không so native R² của hai distributions như cùng một bài toán.

{table(b_table)}

Continuous-age RMSE trên pool quartile={continuous.heldout_common_quartile_rmse:.3f}, trên pool Gaussian={continuous.heldout_common_continuous_rmse:.3f}: đều thấp hơn quartile tại w0 trên **cùng miền đánh giá tương ứng**. Kết quả hỗ trợ giả thuyết cấu hình có biểu diễn tuổi liên tục mô tả RF tốt hơn trong các ca đã thử, nhưng chưa xác định bao nhiêu cải thiện do encoding và bao nhiêu do sampling khi fit. Không chọn cấu hình theo test hoặc thay v1 bằng một cấu hình có số đẹp hơn.

{table(native)}

Các tỷ lệ trên là trung vị qua ca/seed. Sampler Gaussian có thể tạo tuổi ngoài khoảng train {int(training_ages.min())}–{int(training_ages.max())} (CSV toàn bộ có tuổi 1–97); giữ nguyên để mô tả đúng default API và ghi rõ hạn chế clinical plausibility. Không clip tuổi âm thầm. RF vẫn trả xác suất hữu hạn trên các inputs này; điều đó không xác nhận chúng có ý nghĩa lâm sàng.

## 5. Probe mất thông tin trong cùng khoảng tuổi

![F6: Age probe tại row 97 và 10 validation cases]({figure_dir}/F6_age_within_bin.png)

Ca 97 có grid tuổi {int(row97.age_min)}–{int(row97.age_max)}, range RF={row97.rf_range:.3f}. Mọi surrogate chỉ nhận cùng bộ quartile indicators phải cho cùng output trên grid này. Với mọi output hằng c, max_age |f(age)−c| ít nhất bằng range/2=**{row97.minimax_grid_error_lower_bound:.3f}**. Đây là lower bound minimax trên grid cố định, không nói bao nhiêu phần trăm point error ban đầu do discretization.

g(x)={row97.surrogate_constant:.3f} còn thấp hơn cả min f={row97.rf_min:.3f} trên grid. Mất thông tin tuổi chứng minh một hạn chế của biểu diễn, nhưng chưa giải thích được vì sao fit additive/Ridge đã cho hằng số cụ thể này. Neighborhood rộng, tương tác và regularization còn là các yếu tố cần tách; không gán toàn bộ residual v1 cho quartile.

Có **{age_cases}/10** ca validation có range RF lớn hơn 1e-6 trong cùng khoảng. [Từng tuổi/đầu ra]({result_dir}/age_within_bin_probe.csv), [summary 11 probes]({result_dir}/age_within_bin_summary.csv). Ca 97 thuộc test v1, được nêu trước trong phản hồi; probe hậu nghiệm này không dùng để chọn width/model hoặc làm confirmatory validation. Các probe thay tuổi không là can thiệp nhân quả trên bệnh nhân.

## 6. SHAP — Background 20, 100 và toàn train

Sử dụng cùng 10 validation cases, RF và empirical marginal replacement game. B20 seeds 201–205, phân tầng như v1. B100 **chứa B20 tương ứng**, thêm từng lớp từ train còn lại bằng seed+10000; mỗi B100 có 17 ca nặng, B20 có 3. Full reference là 478 train rows, 82 ca nặng. Các seeds là 5 nested reference replicates, không là 5 cohort.

{table(bg_summary)}

SD của full train ghi NA vì chỉ có một reference, không diễn giải NA thành một ước lượng SD bằng 0. Baseline được tổng hợp trên references, không lặp 10 lần theo ca. Attribution metrics được mean qua 5 references **trong từng ca** rồi tổng hợp qua 10 ca:

{table(shp_summary)}

![F7: Background size và khoảng cách tới full train]({figure_dir}/F7_shap_background_size.png)

{table(runtime)}

`initialization` gồm prediction trên B; `explain` gồm input gốc và các hybrid rows cho 30 nontrivial masks. Full reference cần nhiều model rows hơn; không chỉ đếm 32 coalition như 32 prediction calls. Tất cả 110 explanations dùng full enumeration và additivity đạt tolerance. Independent exact-check trên 2 ca × 3 sizes đạt max error {checked['shap_exact_max_error']:.2g}. [Exact errors]({result_dir}/shap_exact_verification.csv), [ID/composition từng B]({result_dir}/shap_background_descriptions.csv), [case/reference metrics]({result_dir}/shap_vs_full_train.csv), [feature/sign errors]({result_dir}/shap_feature_errors.csv).

Reference lớn hơn giảm biến thiên trong các samples đã thử nhưng không chứng minh mọi reference policy tương đương, hoặc giải quyết dependence giữa biến. Full train là đích so sánh finite-sample; game vẫn ghép x với b theo cùng quy tắc marginal. [Aas et al.](https://arxiv.org/abs/1903.10464).

Top-3 agreement trung vị cải thiện từ {error20.median_case_mean_jaccard:.3f} lên {error100.median_case_mean_jaccard:.3f}, trong khi số sign flips trung bình mỗi ca/reference vẫn là {error100.mean_case_sign_flips:.2f} ở cả hai sizes. Giảm baseline/L1 variation do đó chưa làm mọi diagnostic cải thiện. Từng sign flip và độ lớn feature được giữ trong CSV.

## 7. Đổi dấu v1: độ lớn và nhóm feature

Dùng cache v1, không chạy lại test. Trong từng ca/budget, xếp features theo |median coefficient| qua 5 seeds; top-2 và 3 còn lại là nhóm tương đối, không là ngưỡng lâm sàng. Tie phá theo feature_id; đổi dấu chỉ đếm khi hai hệ số trái dấu và đều có độ lớn >1e-6. `eligible_pairs` gồm cả cặp gần 0 để báo rõ mẫu số; số cặp không là bệnh nhân độc lập.

{table(signs)}

Vì các features không có cùng độ lớn, chỉ số “10/10 ca có đổi dấu” v1 cần đọc cùng tỷ lệ theo nhóm và độ lớn ở mỗi lần đổi dấu. [Feature-level statistics]({result_dir}/v1_sign_flip_features.csv), [từng cặp đổi dấu + magnitude]({result_dir}/v1_sign_flip_pairs.csv).

Ở N=2000, top-2 theo |median| có **{int(signs_main.loc['top2_by_abs_median', 'opposite_sign_pairs'])}/{int(signs_main.loc['top2_by_abs_median', 'eligible_pairs'])}** cặp trái dấu; nhóm 3 features còn lại có **{int(signs_main.loc['remaining3', 'opposite_sign_pairs'])}/{int(signs_main.loc['remaining3', 'eligible_pairs'])}**. Ở các cặp đổi dấu trong nhóm còn lại, median của hệ số có độ lớn lớn hơn là {signs_main.loc['remaining3', 'median_larger_abs_at_flip']:.3f}, max={signs_main.loc['remaining3', 'max_larger_abs_at_flip']:.3f}. Các đổi dấu quan sát tập trung ở nhóm hệ số yếu hơn theo phân loại tương đối này, giúp giải thích vì sao top-3 thay đổi dù hai feature lớn có dấu ổn định. Không suy ra các hệ số nhỏ là vô nghĩa về y học.

{table(rank_summary)}

Ở N=500/2000, bộ hai feature lớn nhất giữ nguyên trong mọi seed của cả 10 ca. Vì thế, biến động top-3 ở các budgets này nằm ở việc chọn feature thứ ba trong 3 feature còn lại. **Mốc toán học bổ sung:** nếu hai feature luôn cố định và feature thứ ba chọn đều/độc lập từ 3 candidates, expected Jaccard là (1/3)×1+(2/3)×(2/4)=2/3≈0,667. Đây là một giả định minh họa, không kết luận feature thứ ba thực sự ngẫu nhiên. Mốc 0,46 của v1 vẫn đúng cho hai top-3 hoàn toàn ngẫu nhiên từ 5 features, nhưng chưa mô tả điều kiện có hai feature mạnh ổn định. [Từng lượt giữ top-2]({result_dir}/v1_top2_rank_retention.csv).

## 8. Protocol amendments

{amendments_table}

Config v1 khóa lúc 10:47:30.261025 UTC+7 trước test_first_opened_at=10:47:36.941869. D01–D04 không có timestamp riêng, nên không suy ra giờ/phút chính xác từng quyết định từ mtime chung của decision log. [Phụ lục và bằng chứng thời gian](report/protocol_amendments.md). V2 được thiết kế sau test v1, có protocol/IDs/lock riêng trước lượt tính bổ sung.

## 9. Kết luận, giới hạn và câu hỏi còn mở

Bằng chứng age probe xác nhận quartile indicators có thể làm mất biến thiên RF sử dụng. Ablation width và held-out evaluation đo được riêng point fidelity và khả năng mô tả probabilities ngoài perturbation fit; tăng budget hoặc giảm SD coefficient không tự khắc phục mất thông tin biểu diễn. Continuous-age ablation có cùng evaluation domains để đối chiếu, nhưng đổi cả sampler khi fit; không đồng nhất hiệu ứng đó với encoding đơn thuần. Regularization và interaction chưa được tách riêng.

Background 100 gần full-train game hơn và có baseline ít biến thiên hơn trong 5 nested replicates đã quan sát. Không biến finite reference thành clinical truth hoặc coi việc tăng B là lời giải cho conditional-vs-marginal question. Sign-flip analysis đưa thêm độ lớn vào cách đọc instability v1.

Tất cả bổ sung đều exploratory/post-test; 10 validation cases và 3 fit seeds/5 backgrounds chưa đại diện mọi ca/cấu hình. Evaluation perturbations độc lập chỉ là độc lập RNG, không là cohort ngoài. Chưa chọn model/threshold bằng các kết quả này. V1 có một cohort DENV-2, thiếu patient ID/timestamp và profile overlap; chưa external/patient-wise/prospective validation. Không suy ra causal effect, cơ chế bệnh hoặc quyết định điều trị từ plots.

## 10. Bộ bằng chứng và tái lập

[README](README.md), [requirements lock](requirements-lock.txt), [decision log v2](report/decision_log_v2.md), [run manifest]({artifact_dir}/run_manifest.json), [verification]({result_dir}/verification.json). Verification đạt: {checked['protected_v1_files']} file v1 giữ nguyên hash; width ablation dùng cùng perturbations; evaluation seeds độc lập; 60 API checks đạt; fresh rerun quartile/continuous max coefficient difference={checked['fresh_rerun_max_coefficient_error']}; all SHAP enumeration và exact checks đạt. PNG/PDF F5–F7 lưu cùng run. Mỗi fit có diagnostics JSON, coefficients, raw neighborhoods/evaluation outputs và surrogate checkpoint.

```powershell
.\\.venv\\Scripts\\python.exe -m src.run_supplementary --stage all
.\\.venv\\Scripts\\python.exe -m src.run_supplementary --stage report
.\\.venv\\Scripts\\python.exe scripts\\package_evidence.py --supplementary
```

Cùng run_id đọc cache; dùng run_id mới để sinh/chạy lại toàn bộ cùng protocol. Snapshot ZIP v1 chứa code/config/README chính xác trước sửa bổ sung. Bundle v2 chứa cả snapshot đó và toàn bộ project/results mới, giữ relative links khi giải nén. Kiểm tra SHA-256/CRC/Markdown links của archive được ghi trong file verification cạnh ZIP.
"""
    path = ROOT / "bao-cao-bo-sung-lime-shap-dengue-v2.md"
    path.write_text(text, encoding="utf-8")
    store.stage("report", "complete", {"path": path.name, "figures": ["F5", "F6", "F7"]})
    from .final_report import build_final_report
    build_final_report(store)

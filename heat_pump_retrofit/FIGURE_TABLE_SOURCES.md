# منبع فیگرها و جداول (Figure & Table Sources)

## 📊 فیگرهای اصلی (Main Figures)

تمام فیگرهای اصلی از اسکریپت زیر تولید می‌شوند:

### `src/generate_revised_figures.py`

| فیگر | نام تابع | توضیحات |
|------|----------|---------|
| **Figure 1** | `fig01_workflow()` | نمودار گردش کار مطالعه (Study Workflow) |
| **Figure 2** | `fig02_climate_envelope()` | توزیع اقلیم و پوشش ساختمان (Climate & Envelope) |
| **Figure 3** | `fig03_thermal_intensity()` | توزیع شدت حرارتی (Thermal Intensity Distribution) |
| **Figure 4** | `fig04_validation()` | اعتبارسنجی با جداول رسمی RECS |
| **Figure 5** | `fig05_predicted_observed()` | پیش‌بینی vs مشاهده (Predicted vs Observed) |
| **Figure 6** | `fig06_shap_importance()` | اهمیت ویژگی‌ها با SHAP |
| **Figure 7** | `fig07_shap_dependence()` | نمودارهای وابستگی SHAP |
| **Figure 8** | `fig08_pareto()` | جبهه‌های پارتو (Pareto Fronts) |
| **Figure 9** | `fig09_viability_heatmaps()` | نقشه‌های حرارتی قابلیت اجرا HP |
| **Figure 10** | `fig10_us_map()` | نقشه آمریکا - قابلیت اجرا بر اساس بخش |
| **Figure 11** | `fig11_sensitivity()` | تحلیل حساسیت |
| **Figure 12** | `fig12_viability_contours()` | کانتورهای قابلیت اجرا |
| **Figure 13** | `fig13_interactions()` | اثرات تعاملی |
| **Figure 14** | `fig14_cop_limitation()` | محدودیت COP |
| **Figure 15** | `fig15_aggregation_bias()` | بایاس تجمیع HDD |
| **Figure 16** | `fig16_monte_carlo()` | توزیع عدم قطعیت NPV (Monte Carlo) |
| **Figure 17** | `fig17_sobol()` | شاخص‌های حساسیت Sobol |
| **Figure 18** | `fig18_viability_final()` | کانتورهای نهایی قابلیت اجرا |

---

## 📊 فیگرهای تکمیلی (Supplementary Figures)

تمام فیگرهای تکمیلی از اسکریپت زیر تولید می‌شوند:

### `src/generate_supplementary_figures.py`

| فیگر | نام تابع | توضیحات |
|------|----------|---------|
| **Figure S1** | `fig_s1_vif_correlation()` | تحلیل VIF و ماتریس همبستگی |
| **Figure S2** | `fig_s2_model_comparison()` | مقایسه مدل‌ها (XGBoost vs RF vs OLS) |
| **Figure S3** | `fig_s3_monte_carlo_distributions()` | توزیع‌های ورودی Monte Carlo |
| **Figure S4** | `fig_s4_economic_metrics()` | معیارهای اقتصادی (Payback, IRR) |
| **Figure S5** | `fig_s5_methane_sensitivity()` | حساسیت به نشت متان |
| **Figure S6** | `fig_s6_spatial_bias_quantification()` | کمی‌سازی بایاس مکانی |
| **Figure S7** | `fig_s7_descriptive_statistics()` | جدول آمار توصیفی |
| **Figure S8** | `fig_s8_viability_validation()` | اعتبارسنجی شاخص V |

---

## 📋 جداول (Tables)

جداول از اسکریپت زیر تولید می‌شوند:

### `src/generate_all_outputs.py`

| جدول | نام تابع | توضیحات |
|------|----------|---------|
| **Table 1** | `generate_table1_variables()` | تعریف متغیرها |
| **Table 2** | `generate_table2_sample_characteristics()` | مشخصات نمونه |
| **Table 3** | `generate_table3_model_performance()` | عملکرد مدل XGBoost |
| **Table 4** | `generate_table4_shap_importance()` | اهمیت ویژگی‌ها SHAP |
| **Table 5a** | `generate_table5_assumptions()` | فرضیات بازسازی |
| **Table 5b** | `generate_table5_assumptions()` | فرضیات پمپ حرارتی |
| **Table 5c** | `generate_table5_assumptions()` | قیمت‌های انرژی |
| **Table 6** | `generate_table6_nsga2_config()` | پیکربندی NSGA-II |
| **Table 7** | `generate_table7_tipping_points()` | خلاصه نقاط شکست |

---

## 📁 محل ذخیره فایل‌ها

- **فیگرهای نهایی**: `figures_revised/`
- **جداول**: `output/tables/`
- **خروجی داده**: `output/`

---

## 🔧 اسکریپت‌های اصلی پایپلاین

| اسکریپت | توضیحات |
|---------|---------|
| `01_data_prep.py` | آماده‌سازی داده‌ها |
| `02_descriptive_validation.py` | آمار توصیفی و اعتبارسنجی |
| `03_xgboost_model.py` | مدل XGBoost |
| `04_shap_analysis.py` | تحلیل SHAP |
| `05_retrofit_scenarios.py` | سناریوهای بازسازی |
| `06_nsga2_optimization.py` | بهینه‌سازی NSGA-II |
| `07_tipping_point_maps.py` | نقشه‌های نقطه شکست |

---

## 📊 نحوه تولید همه خروجی‌ها

```bash
# تولید همه فیگرهای اصلی
python src/generate_revised_figures.py

# تولید فیگرهای تکمیلی
python src/generate_supplementary_figures.py

# تولید جداول
python src/generate_all_outputs.py
```

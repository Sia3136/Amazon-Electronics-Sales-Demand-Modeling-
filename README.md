# Amazon-Electronics-Sales-Demand-Modeling

Project summary
---------------
This repository contains an end-to-end analysis and baseline modeling pipeline that predicts product demand (monthly purchases) for Amazon electronics products using a scraped/merged dataset of 42K+ items (2025). The pipeline performs data cleaning and feature engineering on the raw/uncleaned file (`amazon_products_sales_data_uncleaned.csv`), creates exploratory visualizations and simple business insights, selects features, trains baseline regression models (Linear, Ridge, Lasso) on a log-transformed demand target, and writes a cleaned dataset `dstask-3.csv`.

Dataset
-------
- Name: Amazon Electronics Products Sales Dataset (42K+ items) — 2025
- Source: Scraped/compiled dataset (provided to you). The repository contains an "uncleaned" CSV which this script ingests and preprocesses.
- Files:
  - `amazon_products_sales_data_uncleaned.csv` — input raw/uncleaned data (expected in project root).
  - `dstask-3.csv` — cleaned output written by the script.

Key results (from the run included in the code)
- Models trained to predict log(1 + bought_in_last_month)
  - Linear Regression
    - Mean Squared Error (MSE): 3.008377681747854
    - R² Score: 0.35392865285388486
  - Ridge Regression (GridSearchCV)
    - Best alpha: 0.1
    - MSE: 3.008369954229035
    - R² Score: 0.3539303123956784
  - Lasso Regression (GridSearchCV)
    - Best alpha: 0.001
    - MSE: 3.008114079818107
    - R² Score: 0.3539852633170242

Columns / example fields
------------------------
The raw file contains fields similar to:
- title, rating, number_of_reviews, bought_in_last_month, current/discounted_price, price_on_variant, listed_price, is_best_seller, is_sponsored, is_couponed, buy_box_availability, delivery_details, sustainability_badges, image_url, product_url, collected_at

Targets and features
--------------------
- Target (y): log1p(bought_in_last_month) — natural log of (1 + monthly purchases), used to stabilize variance and improve modeling.
- Features used (X) before selection / scaling:
  - rating (float)
  - number_of_reviews (float)
  - current/discounted_price (float)
  - price_on_variant (float)
  - listed_price (float)
  - is_sponsored (0/1)
  - is_couponed (0/1)
  - buy_box_availability (0/1)
  - has_sustainability_badge (0/1)
  - delivery_days (numeric)
  - discount (listed_price - current/discounted_price)
  - review_score (number_of_reviews * rating)
  - has_coupon_or_discount (0/1)

Cleaning & feature engineering (what the script does)
----------------------------------------------------
1. Type coercion and basic parsing:
   - Convert `title` to string.
   - Parse `rating` (keeps first 3 chars and converts to float), fill missing with column mean.
   - Clean `number_of_reviews` by removing commas and converting to float; missing → 0.
   - Parse `current/discounted_price`, `price_on_variant`, and `listed_price` into numeric floats (strip currency characters).
   - Parse `collected_at` to datetime (coerce errors).
2. Boolean / categorical mappings:
   - `is_sponsored`: 'Sponsored' → 1, 'Organic' → 0
   - `buy_box_availability`: 'Add to cart' → 1, missing → 0
   - `is_couponed`: 'No Coupon' → 0 else 1 (case-insensitive)
   - `has_sustainability_badge`: presence of any badge → 1, missing/none → 0 (then drop `sustainability_badges`)
3. Parse `bought_in_last_month` text like "300+ bought in past month", removing non-numeric tokens and converting to numeric (handles "K" → "000" in a basic way); missing → 0.
4. Extract `delivery_days` from `delivery_details` (first integer found); missing → 0 and then drop `delivery_details`.
5. Create derived features:
   - `discount` = listed_price - current/discounted_price
   - `review_score` = number_of_reviews * rating
   - `has_coupon_or_discount` = (is_couponed == 1 OR discount > 0)
6. Drop `is_best_seller` (unused) and other intermediary columns where appropriate.
7. Outlier removal: For every numeric column the script computes IQR-based bounds (Q1 − 1.5*IQR, Q3 + 1.5*IQR) and removes rows outside those bounds. Note: this is aggressive and can remove many rows—check remaining sample size after running.
8. Save cleaned DataFrame to `dstask-3.csv`.

Exploratory visualizations and business insights (from script)
--------------------------------------------------------------
- Couponed vs Bought in Last Month (boxplot)
  - Insight: Coupons drive sales. Use couponing strategically to boost sales for low-selling SKUs.
- Discount vs Bought in Last Month (scatter)
  - Insight: Discounts increase sales; focus discounting on mid-range discounts for best uplift.
- Listed Price vs Price on Variant (scatter)
  - Insight: Variant pricing is proportional to base price — ensure consistent variant pricing and fix zero/incorrect listed prices.
- Sustainability Badge vs Rating (boxplot)
  - Insight: A sustainability badge slightly correlates with higher ratings — highlight sustainability in marketing.
- Buy Box Availability vs Couponed (countplot)
  - Insight: Items in the buy box are slightly more likely to have coupons — sellers use coupons to improve buy box chances.
- Rating bins vs Bought in Last Month
  - Insight: Higher-rated products (4.5+) consistently sell more — prioritize marketing & inventory for top-rated products.

Modeling pipeline
-----------------
1. Prepare X and log-transformed y: y = np.log1p(df['bought_in_last_month'])
2. Scale features with StandardScaler
3. Feature selection: RFE (LinearRegression) used to select n_features_to_select=9 — selected features are printed during the run.
4. Train/test split: train_test_split(..., test_size=0.3, random_state=41)
5. Models trained and evaluated on the test set:
   - LinearRegression
   - Ridge (GridSearchCV over alpha = [0.1, 1, 10, 50, 100], 5-fold CV)
   - Lasso (GridSearchCV over alpha = [0.001, 0.01, 0.1, 1, 10], 5-fold CV)
6. Evaluation metrics: Mean Squared Error (MSE) and R²

How to run
----------
1. Place `amazon_products_sales_data_uncleaned.csv` in the project root (or update the path in the script).
2. Install dependencies:
   - Python 3.8+
   - pip install pandas numpy matplotlib seaborn scikit-learn
3. Save the provided code into a file (e.g., `analysis_amazon.py`) or run as a notebook.
4. Run:
```
python analysis_amazon.py
```
(or run notebook cells interactively)
5. Outputs:
   - Visualizations displayed (if run interactively)
   - Cleaned CSV: `dstask-3.csv`
   - Model performance printed to console

Reproducibility notes & caveats
------------------------------
- The script suppresses warnings (`warnings.filterwarnings('ignore')`) — remove that during development to catch issues.
- Rating parsing takes the first three characters of the rating string (e.g., `"4.6 out of 5 stars"` → `"4.6"`). Verify this logic on edge cases.
- `bought_in_last_month` parsing currently handles patterns like "300+ bought in past month" and converts "K" to "000" in a simple way — this may need refinement (e.g., "6K" → 6000).
- Outlier removal is done across all numeric columns with the same IQR rule — this can remove many valid observations (use with caution or consider column-wise or robust alternatives).
- RFE is used to inspect important features; the final split and model training use scaled features. Confirm whether you want to train only on selected features or on the full set.
- The model target is log1p transformed; interpret metric values in transformed space (inverse transform predictions with expm1 for business-facing metrics).
- Missing values are imputed with simple strategies (means or zeros); consider more robust imputation if needed.

Suggested next steps / improvements
----------------------------------
- Improve parsing of textual fields (robust K/+, ranges, and currency symbols).
- Use cross-validation for final performance reporting and consider time-based splits if data contains time dependencies.
- Try tree-based and ensemble models (RandomForest, XGBoost, LightGBM) which often perform better on tabular e-commerce data.
- Build a separate pipeline that persists trained models (joblib) and exposes an API for inference.
- Add SHAP or permutation importance to explain model predictions and support business decisions.
- Revisit outlier removal: consider winsorizing, robust scaling, or per-column thresholds.
- Expand feature engineering: category hierarchy, brand effects, text features from title (TF-IDF / embeddings), temporal seasonality from `collected_at`.



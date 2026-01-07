# Ball_Landing_Position_Prediction

Project summary
---------------
This repository contains an end-to-end experiment that predicts where a passed ball will land (x and y coordinates after the pass) using tracking and player metadata. The script reads a merged dataset (`merged.csv`), performs data cleaning and feature engineering, visualizes relationships, removes outliers, trains several regression models (Linear, Ridge, Lasso) and reports performance (MSE and R²). The cleaned dataset is saved as `dstask-2.csv`.

Dataset
-------
The input data comes from the 2025 Big Data Bowl and was prepared by the NFL Next Gen Stats team. The dataset used in this repository is expected to be a cleaned/merged CSV derived from that competition data. Please review the Big Data Bowl competition rules and the NFL Next Gen Stats terms of use before publishing or redistributing derivatives.

Key results (from the run included in the code)
- Linear Regression
  - Mean Squared Error (MSE): 18.898472360256285
  - R² Score: 0.9263678171657588
- Ridge Regression
  - Best alpha: 0.1
  - MSE: 18.898447738549784
  - R² Score: 0.9263678498796282
- Lasso Regression
  - Best alpha: 0.001
  - MSE: 18.89842923744751
  - R² Score: 0.9263667012983716

The three models perform almost identically on this dataset.

Requirements
------------
- Python 3.8+
- The following Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

Install with pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
(or use a requirements file and `pip install -r requirements.txt`)

Files
-----
- `merged.csv` — Input dataset (required). This should be the merged CSV derived from the 2025 Big Data Bowl / NFL Next Gen Stats dataset (place it in the project root or update the path in the script).
- `analysis.py` (or your Jupyter notebook) — The provided code performs all steps: cleaning, visualization, modeling and saving the cleaned CSV.
- `dstask-2.csv` — Output cleaned dataset written by the script.

Data description
----------------
The raw (merged) data contains tracking and player metadata rows similar to:

example fields:
game_id, play_id, player_to_predict, nfl_id, frame_id, play_direction, absolute_yardline_number, player_name, player_height, player_weight, ..., x_x, y_x, x_y, y_y, ball_land_x, ball_land_y, speed, acceleration, direction, orientation, num_frames_output

Target variables:
- `x_y` and `y_y` — the x and y coordinates after the pass (the model predicts these).

Features used for training (X)
- frame_id
- play_direction (encoded: right=1, left=0)
- absolute_yardline_number
- player_height (converted to inches)
- player_weight
- player_age (derived from `player_birth_date`)
- x_x, y_x — player position before the pass
- speed
- acceleration
- direction
- orientation
- num_frames_output
- ball_land_x, ball_land_y
- player_position (label-encoded)
- player_side (Defense=1, Offense=0)
- player_role (Defensive Coverage=1, Targeted Receiver=0)

Target (y)
- x_y, y_y

Processing steps implemented
----------------------------
1. Read CSV: `pd.read_csv('/merged.csv')`.
2. Basic inspection: `.info()`, `.describe()`, `.tail()`.
3. Rename ambiguous columns:
   - `s` → `speed`
   - `a` → `acceleration`
   - `dir` → `direction`
   - `o` → `orientation`
4. Drop unused or identifying columns:
   - `game_id`, `play_id`, `nfl_id`, `player_name`, `player_to_predict`
5. Encode categorical variables:
   - `player_position` with `LabelEncoder`
   - `play_direction`: `right` → 1, `left` → 0
   - `player_side`: `Defense` → 1, `Offense` → 0
   - `player_role`: `Defensive Coverage` → 1, `Targeted Receiver` → 0
6. Convert `player_height` from "feet-inches" text (e.g. `6-1`) to integer inches.
7. Convert `player_birth_date` to datetime and compute `player_age` in years; drop the original birth date.
8. Visualizations:
   - Correlation heatmap for movement features.
   - Scatterplots to inspect relationships between pre- and post-pass coordinates and ball landing coordinates.
9. Outlier removal:
   - For every numeric column, remove rows outside the IQR-based bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
   - Note: This is an aggressive procedure and may drop many rows; monitor remaining sample size.
10. Standardize features using `StandardScaler`.
11. Train/test split: `train_test_split(test_size=0.3, random_state=41)`.
12. Train models:
   - LinearRegression
   - Ridge with GridSearchCV over alpha in [0.1, 1, 10, 50, 100], 5-fold CV
   - Lasso with GridSearchCV over alpha in [0.001, 0.01, 0.1, 1, 10], 5-fold CV
13. Evaluate with MSE and R².
14. Save cleaned DataFrame to `dstask-2.csv`.

How to run
----------
1. Place `merged.csv` (derived from the 2025 Big Data Bowl data) into the project root (or update the path in the script).
2. Save the provided code into a file, e.g. `analysis.py` (or run it in a notebook cell).
3. Run:
```
python analysis.py
```
or run the notebook interactively.

Notes and caveats
-----------------
- The code suppresses warnings (`warnings.filterwarnings('ignore')`), so some preprocessing warnings may be hidden. Remove that suppression during development if you want to see warnings.
- The IQR-based outlier removal is applied to all numeric columns iteratively. This can remove a large portion of rows if there are skewed distributions. Consider more conservative thresholds or column-specific handling.
- `LabelEncoder` for `player_position` encodes labels into integers but does not preserve any ordinal meaning—consider one-hot encoding for non-ordinal categorical variables.
- Age is computed using the current date at runtime (`pd.Timestamp.today()`), which makes results time-sensitive. For reproducibility, consider using a fixed reference date (e.g., season start date).
- Model evaluation uses a single train/test split. Consider cross-validation for more robust performance estimates and to inspect variance in metrics.
- The models predict two targets simultaneously (multi-output regression). Scikit-learn's linear models support multi-output out of the box but consider building separate models or using multi-output specific algorithms if appropriate.
- Check Big Data Bowl / NFL Next Gen Stats licensing and competition rules before sharing or publishing outputs derived from the dataset.

Possible improvements
---------------------
- Use feature selection or regularization tuning with a wider grid.
- Try tree-based models (RandomForest, XGBoost) which can capture non-linear relationships.
- Add polynomial features or interactions for more expressive linear models.
- Visualize residuals and predicted vs actual scatter plots to spot systematic errors.
- Persist trained models with joblib / pickle and provide an inference API.
- If dataset is large or imbalanced across plays/players, consider stratified sampling or grouping by play to avoid data leakage.




import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import logging
import time
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ========== Configuration ==========
EXCEL_PATH = "LTF Challenge data with dictionary.xlsx"
TRAIN_SHEET = "TrainData"
TEST_SHEET = "TestData"
TARGET_COL = "Target_Variable/Total Income"
ID_COL = "FarmerID"

FAST_MODE = True

PROJECT_ROOT = "."
CHAPTER_ID = "training_farmer_income_models"
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images", CHAPTER_ID)
os.makedirs(IMAGES_DIR, exist_ok=True)

LOG_FILE = os.path.join(PROJECT_ROOT, "training_log.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ========== Utility Functions ==========

def save_fig(name, tight_layout=True, extension="png", dpi=300):
    path = os.path.join(IMAGES_DIR, f"{name}.{extension}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=extension, dpi=dpi)
    plt.close()
    logging.info(f"📊 Saved plot: {path}")

def clean_column_names(df):
    df.columns = (
        df.columns.str.replace(r"[^\w]", "_", regex=True)
                  .str.replace(r"__+", "_", regex=True)
                  .str.strip("_")
    )
    return df

def encode_categorical_columns(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col].astype(str))[0]
    return df

def load_data():
    train = pd.read_excel(EXCEL_PATH, sheet_name=TRAIN_SHEET)
    test = pd.read_excel(EXCEL_PATH, sheet_name=TEST_SHEET)
    logging.info("📥 Data loaded successfully.")
    return train, test

# ========== Evaluation Plotting ==========

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def plot_and_save_all(X, y, model):
    train_preds = model.predict(X)

    # Histogram of target
    plt.figure(figsize=(8, 6))
    sns.histplot(y, bins=50, kde=True, color='skyblue')
    plt.title("Target Income Distribution")
    plt.xlabel("Income")
    save_fig("target_distribution")

    # Feature Importance
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    top_feat = feat_imp.sort_values(by="Importance", ascending=False).head(30)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=top_feat)
    plt.title("Top 30 Feature Importances")
    save_fig("feature_importance_top30")

    # SHAP Plot
    if not FAST_MODE:
        sample_X = X.sample(n=1000, random_state=42)
        shap_values = model.predict(sample_X, pred_contrib=True)
        shap.summary_plot(shap_values[:, :-1], sample_X, show=False)
        fig = plt.gcf()
        fig.savefig(os.path.join(IMAGES_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Correlation Heatmap
        # Final Clean Correlation Heatmap (Top 10 Features + Target)
    top_corr_features = X.corrwith(y).abs().sort_values(ascending=False).head(10).index.tolist()
    top_corr_features.append(TARGET_COL)

    corr_matrix = X[top_corr_features[:-1]].copy()
    corr_matrix[TARGET_COL] = y
    corr = corr_matrix.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",  # <-- Cool to warm gradient
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='white',
        square=True,
        cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.title("Correlation Matrix (Top Features)", fontsize=14)
    plt.xticks(rotation=90, ha="center", fontsize=9)   # <-- No slant
    plt.yticks(rotation=0, fontsize=9)
    sns.despine()
    save_fig("correlation_heatmap")



    # Prediction Error Histogram
    errors = y - train_preds
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=40, kde=True, color="salmon")
    plt.title("Prediction Error Distribution")
    save_fig("prediction_error_distribution")

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y, train_preds, alpha=0.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel("Actual Income")
    plt.ylabel("Predicted Income")
    plt.title("Actual vs Predicted Income")
    save_fig("actual_vs_predicted")

    # Pairplot (Top 5 Features)
    top5 = top_feat['Feature'].head(5).tolist()
    pair_df = X[top5].copy()
    pair_df['Target'] = y
    if FAST_MODE:
        pair_df = pair_df.sample(n=2000, random_state=42)
    sns.pairplot(pair_df)
    save_fig("pairplot_top5_features")

    # Learning Curve
       # Learning Curve using MAPE (Compact & Styled)
    from sklearn.utils import shuffle

    X_shuf, y_shuf = shuffle(X, y, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)
    train_mape, val_mape = [], []

    for frac in train_sizes:
        n = int(frac * len(X_shuf))
        X_sub, y_sub = X_shuf[:n], y_shuf[:n]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31)
        model.fit(X_sub, y_sub)
        train_mape.append(mape(y_sub, model.predict(X_sub)))
        val_mape.append(mape(y, model.predict(X)))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * len(X), train_mape, label="Training MAPE", color="blue", marker='o')
    plt.plot(train_sizes * len(X), val_mape, label="Validation MAPE", color="orange", marker='o')
    plt.fill_between(train_sizes * len(X), np.array(train_mape)*0.95, np.array(train_mape)*1.05, alpha=0.2, color='blue')
    plt.fill_between(train_sizes * len(X), np.array(val_mape)*0.95, np.array(val_mape)*1.05, alpha=0.2, color='orange')
    plt.title("Learning Curve (MAPE)", fontsize=14)
    plt.xlabel("Training Set Size")
    plt.ylabel("MAPE (%)")
    plt.legend()
    save_fig("learning_curve")

    # Residual Plot
    residuals = y - train_preds
    plt.figure(figsize=(8, 6))
    plt.scatter(train_preds, residuals, alpha=0.4, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted Income")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    save_fig("residual_plot")

    # MAPE by Top Features
        # MAPE by Feature Bin – subplot version (Top 5 features)
    top5 = top_feat['Feature'].head(6).tolist()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(top5):
        try:
            bins = pd.qcut(X[col], q=5, duplicates='drop')
            df_temp = pd.DataFrame({'y_true': y, 'y_pred': train_preds, 'bin': bins})
            mape_by_bin = df_temp.groupby('bin').apply(lambda d: mape(d['y_true'], d['y_pred']))
            sns.barplot(x=mape_by_bin.index.astype(str), y=mape_by_bin.values, ax=axes[i], palette="viridis")
            axes[i].set_title(f"MAPE by {col} Bin")
            axes[i].set_ylabel("MAPE (%)")
            axes[i].set_xlabel("Binned Range")
            axes[i].tick_params(axis='x', rotation=45)
        except Exception as e:
            axes[i].set_visible(False)
            logging.warning(f"Could not generate MAPE subplot for {col}: {e}")

    # Title
    fig.suptitle("MAPE by Feature Bin (Top 6 Features)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig("mape_by_features")


    # Top 15 Correlations
    top_corr = X.corrwith(y).sort_values(key=abs, ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')
    plt.title("Top 15 Feature Correlations with Target")
    plt.xlabel("Correlation")
    save_fig("top15_correlations")

   



# ========== Main ==========

def main():
    start_time = time.time()
    train_df, test_df = load_data()

    y = train_df[TARGET_COL]
    X = train_df.drop(columns=[TARGET_COL, ID_COL])
    X_test = test_df[X.columns]

    X = clean_column_names(X)
    X_test = clean_column_names(X_test)
    X = encode_categorical_columns(X)
    X_test = encode_categorical_columns(X_test)

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X_test)
    test_df[TARGET_COL] = y_pred

    submission = test_df[[ID_COL, TARGET_COL]].copy()
    submission.columns = ['FarmerID', 'Predicted_Income']
    submission['FarmerID'] = submission['FarmerID'].apply(lambda x: f'="{x}"')
    csv_path = "Predictions.csv"
    submission.to_csv(csv_path, index=False)
    logging.info(f"📤 Predictions saved to: {csv_path}")

    plot_and_save_all(X, y, model)

    print(f"✅ Script finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

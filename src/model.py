import logging
import warnings
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 

# Paths & logging
DATA_PATH  = Path("D:/Full projet/EcoFlight_Delay_Predictor/data/process.csv")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
 
 
# Load data
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = df.drop("is_delayed", axis=1)
    y = df["is_delayed"]
    logger.info(
        f"Dataset: {len(df)} rows | "
        f"Delayed: {y.sum()} ({y.mean()*100:.1f}%)"
    )
    return X, y
 
 
# Model definitions
# (SMOTE is applied inside the pipeline so it only sees training folds)
def build_pipelines() -> dict:
    smote = SMOTE(random_state=42)
    return {
        "Logistic Regression": ImbPipeline([
            ("smote",      smote),
            ("scaler",     StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear", random_state=42
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("smote",      smote),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": ImbPipeline([
            ("smote",      smote),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200, random_state=42
            )),
        ]),
    }
 
 
# Evaluation helper
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_prob)
    pr_auc      = average_precision_score(y_test, y_prob)
 
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["On-Time (0)", "Delayed (1)"]
    ))
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}  ← better metric for imbalanced data")
 
    return {"name": name, "model": model, "roc_auc": roc_auc, "pr_auc": pr_auc}
 
 
# Feature importance (tree models only)
def print_feature_importance(model, feature_names: list[str]) -> None:
    clf = model.named_steps.get("classifier")
    if not hasattr(clf, "feature_importances_"):
        return
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print("\nFeature Importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<28} {imp:.4f}  {bar}")
 
 
# Main training loop
def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
 
    pipelines = build_pipelines()
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results   = []
 
    # ── Cross-validation comparison ──────────────────────────────────────
    print("\n" + "="*55)
    print("  CROSS-VALIDATION COMPARISON (5-Fold, F1-macro)")
    print("="*55)
    for name, pipeline in pipelines.items():
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        print(f"  {name:<28}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
 
    # ── Full train + evaluation on hold-out ──────────────────────────────
    print("\n\n" + "="*55)
    print("  HOLD-OUT TEST SET EVALUATION")
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        result = evaluate_model(name, pipeline, X_test, y_test)
        results.append(result)
 
    # ── Feature importances ──────────────────────────────────────────────
    feature_names = X.columns.tolist()
    for res in results:
        if res["name"] in ("Random Forest", "Gradient Boosting"):
            print(f"\n[{res['name']}]", end="")
            print_feature_importance(res["model"], feature_names)
 
    # ── Select & save best model ─────────────────────────────────────────
    best = max(results, key=lambda r: r["pr_auc"])
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best["model"], model_path)
 
    print(f"\n{'='*55}")
    print(f"  Best model : {best['name']}")
    print(f"  PR-AUC     : {best['pr_auc']:.4f}")
    print(f"  Saved to   : {model_path}")
    print(f"{'='*55}\n")
 
 
if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib


# Feature descriptions
feauture_desc = {
    "X1": "My order was delivered on time ",
    "X2": "Contents of my order were as expected",
    "X3": "I ordered everything I wanted to order",
    "X4": "I paid a good price for my order",
    "X5": "I am satisfied with my courier",
    "X6": "The app makes ordering easy for me",
    "Y" : "Customer Sentiment (1=Happy, 0=Unhappy)"
}

# Get descriptive name for a feature
def get_feature_description(feature):
    if feature in feauture_desc:
        return f"{feature}: {feauture_desc[feature]}"
    

# Load the dataset and split into train/test
def load_data(file_path, target_col="Y", test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print("Full dataset distribution:")
    print(y.value_counts(normalize=True))
    print("\nTrain distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTest distribution:")
    print(y_test.value_counts(normalize=True))

    return df, X_train, X_test, y_train, y_test


# Explore categorical features
def plot_counts(df, features):
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    
    for i, f in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.countplot(x=f, data=df)
        
        desc = get_feature_description(f)
        plt.title(f"{desc}", fontsize=12, fontweight='bold')
        plt.xlabel("Rating (1-5)")
        plt.ylabel("Count")

        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().annotate(int(height), (p.get_x() + p.get_width() / 2, height),
                                   ha='center', va='bottom', fontsize=9)
                
    plt.suptitle("Distribution of Customer Satisfaction Ratings", fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    fig_path = "results/figures/feature_counts.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")

    plt.show()

# Heatmap of correlations
def plot_corr(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")

    footnote_text = "Feature Descriptions:\n"
    for i in range(1, 7):
        feature_code = f"X{i}"
        if feature_code in feauture_desc:
            footnote_text += f"{feature_code}: {feauture_desc[feature_code]}\n"
    if "Y" in feauture_desc:
        footnote_text += f"Y: {feauture_desc['Y']}"
    
    plt.figtext(0.02, 0.02, footnote_text, fontsize=7, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                verticalalignment='bottom', horizontalalignment='left')

    fig_path = "results/figures/correlation_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")

    plt.show()

# Feature vs target plots
def feature_vs_target(df, features, target="Y"):
    plt.figure(figsize=(14, 8))

    for i, f in enumerate(features, 1):
        plt.subplot(2, 3, i)
        cross = pd.crosstab(df[f], df[target], normalize='index') * 100
        cross.plot(kind='bar', ax=plt.gca())
        plt.title(f"{f} vs {target}")
        plt.ylabel("Percent")
        plt.xticks(rotation=0)
    plt.tight_layout()

    fig_path = "results/figures/feature_vs_target.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")

    plt.show()


# Simple feature importance using RandomForest
def get_feature_ranking(X, y):
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
        )
    
    model.fit(X, y)

    imp_df = pd.DataFrame({
        "feature": X.columns, 
        "importance": model.feature_importances_
        })
    
    imp_df = imp_df.sort_values("importance", ascending=False)

    print("\nRandom Forest Feature Importance:")
    print(imp_df)
    return imp_df["feature"].tolist()


# Evaluate a model on test set and cross-validation
def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auc = None
    
    cv = cross_val_score(
        model, X_train, y_train, cv=5
        )

    return {
        "accuracy": acc, 
        "auc": auc, 
        "cv_mean": cv.mean(), 
        "cv_std": cv.std()
        }


# Train different models with top features
def test_models(X_train, X_test, y_train, y_test, features, models):
    results = []

    for k in range(1, len(features)+1):
        top_k = features[:k]
        X_tr_sub = X_train[top_k]
        X_te_sub = X_test[top_k]

        print(f"\nTop {k} features: {top_k}")

        for name, model in models.items():
            metrics = evaluate(
                model, X_tr_sub, X_te_sub, y_train, y_test
                )
            
            print(f"{name}: Test Acc={metrics['accuracy']:.3f}")
            
            results.append({
                "model": name,
                "num_features": k,
                "features": ", ".join(top_k),
                "test_acc": metrics["accuracy"],
                "test_auc": metrics["auc"],
                "cv_mean": metrics["cv_mean"],
                "cv_std": metrics["cv_std"]
            })

    return pd.DataFrame(results)


# Save the best model and related info
def save_best(results_df, X_train, y_train, models, 
              model_file="results/best_model.pkl",
              importance_file="results/summary/feature_importance.csv",
              results_file="results/summary/model_results.csv"):
    
    best = results_df.loc[results_df["test_acc"].idxmax()]
    
    print("\nBest model configuration:")
    print(best)
    
    best_model = models[best["model"]]
    final_features = best["features"].split(", ")
    best_model.fit(X_train[final_features], y_train)
    
    joblib.dump(best_model, model_file)
    
    # Save feature importance
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        imp = abs(best_model.coef_[0])
    else:
        imp = [1/len(final_features)]*len(final_features)
    
    final_importance = pd.DataFrame({"feature": final_features, "importance": imp})
    final_importance.to_csv(importance_file, index=False)
    
    results_df.to_csv(results_file, index=False)
    
    print("Saved model, results, and feature importance.")


# Main workflow
def main():
    file_path = "data/ACME-HappinessSurvey2020.csv"
    df, X_train, X_test, y_train, y_test = load_data(file_path)
    
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    plot_counts(df, features)
    plot_corr(df)
    feature_vs_target(df, features)
    
    top_features = get_feature_ranking(X_train, y_train)
    
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, 
            class_weight="balanced",
            random_state=42
            ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            class_weight="balanced",
            random_state=42
            ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
            ),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
    }
    
    results = test_models(X_train, X_test, y_train, y_test, top_features, models)

    save_best(results, X_train, y_train, models)


if __name__ == "__main__":
    main()
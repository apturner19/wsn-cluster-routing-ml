####### Machine Learning Model for Cluster-Based Routing in Wireless Sensor Networks #######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load & clean dataset
df = pd.read_csv('WSN-DS.csv')
df.columns = df.columns.str.strip()

# Drop irrelevant / leakage-prone features
initial_drop = ['id', 'Time', 'who CH']
leak_cols = [
    'ADV_S','ADV_R','JOIN_S','JOIN_R',
    'SCH_S','SCH_R','DATA_S','DATA_R',
    'Data_Sent_To_BS','Expaned Energy'
]
to_drop = [c for c in initial_drop + leak_cols if c in df.columns]
df = df.drop(columns=to_drop)

# Define X and y
target = 'Is_CH'
features = [c for c in df.columns if c != target]
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing pipelines
numeric_feats     = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_feats = X_train.select_dtypes(include=['object']).columns.tolist()

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler())
])
cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_feats),
    ('cat', cat_pipeline, categorical_feats)
])

# Precompute processed features
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Develop custom neural network model
input_dim = X_train_proc.shape[1]
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = nn_model.fit(
    X_train_proc, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate neural network
y_pred_nn = (nn_model.predict(X_test_proc).ravel() >= 0.5).astype(int)

metrics = {
    'Neural Network': {
        'Accuracy': accuracy_score(y_test, y_pred_nn),
        'Precision': precision_score(y_test, y_pred_nn),
        'Recall': recall_score(y_test, y_pred_nn),
        'F1': f1_score(y_test, y_pred_nn)
    }
}

print("\n** Neural Network Results **")
nn_eval = nn_model.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test Loss:      {nn_eval[0]:.4f}")
print(f"Test Accuracy:  {nn_eval[1]:.4f}")
print(f"Test Precision: {nn_eval[2]:.4f}")
print(f"Test Recall:    {nn_eval[3]:.4f}")
print(classification_report(y_test, y_pred_nn))

# Baseline Models for comparison
models = {
    'Logistic Regression': LogisticRegression(solver='saga', max_iter=500, n_jobs=-1),
    'Random Forest':       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'SVM (RBF kernel)':    SVC(kernel='rbf', probability=True, tol=1e-2, max_iter=5000, random_state=42)
}

for name, clf in models.items():
    # Train SVM on 30% of the dataset for improved speed
    if name == 'SVM (RBF kernel)':
        X_tr, _, y_tr, _ = train_test_split(
            X_train_proc, y_train, train_size=0.3, stratify=y_train, random_state=42
        )
    else:
        X_tr, y_tr = X_train_proc, y_train
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_test_proc)
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    }
    print(f"\n** {name} **")
    print(classification_report(y_test, y_pred))

# Plot bar chart to compare performance
metrics_df = pd.DataFrame(metrics)
fig, ax = plt.subplots()
x = np.arange(len(metrics_df.index))
width = 0.2
for i, model_name in enumerate(metrics_df.columns):
    ax.bar(x + i*width, metrics_df[model_name].values, width, label=model_name)
ax.set_xticks(x + width*(len(metrics_df.columns)-1)/2)
ax.set_xticklabels(metrics_df.index)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

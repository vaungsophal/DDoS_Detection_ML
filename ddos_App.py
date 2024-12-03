from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels

# Streamlit UI
st.title("DDoS Detection using Machine Learning")

# Subtitle with Styling
st.markdown("""
#### Detect, Analyze, and Prevent Distributed Denial of Service Attacks  
**Empowered by cutting-edge machine learning models.**
""")

st.sidebar.header("Upload Your Dataset")

# File uploader for CSV
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  # Remove any leading/trailing spaces from column names

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Encoding the 'Label' column to numeric values
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # Define the label mapping explicitly
    label_mapping = {0: 'BENIGN', 1: 'DDoS'} 
    # st.write(f"Label Mapping: {label_mapping}

    # Selecting necessary features
    necessary_features = [
        'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Bwd Packet Length Max', 
        'Bwd Packet Length Std', 'Packet Length Mean', 'Average Packet Size', 
        'Packet Length Std', 'Max Packet Length', 'Packet Length Variance', 
        'PSH Flag Count', 'Flow IAT Std', 'Flow IAT Mean', 'Fwd IAT Max', 'Flow IAT Max', 
        'Fwd IAT Std', 'ACK Flag Count', 'Idle Max', 'Idle Mean', 'Idle Std', 'Idle Min',
        'Subflow Bwd Bytes', 'Total Length of Bwd Packets', 'Fwd IAT Total', 'Active Min', 
        'Flow Duration', 'Active Mean', 'Fwd IAT Mean'
    ]

    # Keep only relevant features and the label
    data = data[necessary_features + ['Label']]

    # Convert all features to numeric, coercing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with mean of each column
    data.fillna(data.mean(), inplace=True)

    # Splitting dataset into features (X) and label (y)
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Explicitly convert all columns to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Remove non-numeric columns if any
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        st.warning(f"Removing non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)

    # Replace infinite values and handle NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if np.any(np.isnan(X)):
        st.warning("NaN values detected. Replacing with column mean using imputer.")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.write("Dataset preprocessing completed successfully.")

    # Model Selection
    st.sidebar.header("Choose Model")
    model_choice = st.sidebar.selectbox("Select a Model", ["Random Forest", "Decision Tree", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"])

    model_filename = "model.joblib"  # File name to save the model

    # Load the model if it exists
    try:
        model = joblib.load(model_filename)
    except FileNotFoundError:
        model = None

    if st.sidebar.button("Run Model"):
        try:
            models = {}
            # Select model(s) based on user choice
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
                models["Random Forest"] = model
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=0)
                models["Decision Tree"] = model
            elif model_choice == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier(n_neighbors=5)
                models["K-Nearest Neighbors (KNN)"] = model
            elif model_choice == "Support Vector Machine (SVM)":
                model = SVC(kernel='linear', random_state=0, probability=True)
                models["Support Vector Machine (SVM)"] = model

            for model_name, model in models.items():
                st.write(f"### {model_name}")
                # Train the model
                model.fit(X_train, y_train)

                # Predict with the model
                y_pred = model.predict(X_test)

                # Map numeric predictions to string labels
                y_pred_mapped = [label_mapping[label] if label in label_mapping else 'OTHER ATTACK' for label in y_pred]

                # Display the predictions with string labels
                st.write("##### Predictions with String Labels:")
                st.dataframe(pd.DataFrame({'Predicted Labels (Numeric)': y_pred, 'Predicted Labels (String)': y_pred_mapped}))

                # Calculate percentage of each class
                label_counts = pd.Series(y_pred).value_counts(normalize=True) * 100
                class_names = [label_mapping.get(idx, 'OTHER ATTACK') for idx in label_counts.index]
                # Calculate percentages
                total_predictions = len(y_pred)
                other_attack_count = sum(1 for label in y_pred if label > 1)
                other_attack_percentage = (other_attack_count / total_predictions) * 100

                # Display results
                st.write(f"##### {model_name} Class Distribution (Percentage):")
                class_distribution_df = pd.DataFrame({
                    "Class": class_names,
                    "Percentage (%)": label_counts.values
                })
                st.dataframe(class_distribution_df)
                
                # Visualization of percentage
                plt.figure(figsize=(8, 6))
                label_counts.plot(kind='bar', color=['blue', 'green'], alpha=0.7)
                plt.title(f"{model_name} Predictions (%)")
                plt.ylabel("Percentage")
                plt.xticks(rotation=0)  # Keep labels horizontal
                plt.ylim(0, 100)  # Set the y-axis limit to 100%
                plt.xlabel("Class")
                st.pyplot(plt)

                # Get unique labels present in `y_test`
                present_labels = unique_labels(y_test, y_pred)
                present_target_names = [label_mapping.get(label, 'OTHER ATTACK') for label in present_labels]

                # Update classification report
                st.write(f"##### {model_name} Classification Report:")
                cr = classification_report(
                    y_test,
                    y_pred,
                    target_names=present_target_names,
                    labels=present_labels,
                    output_dict=True
                )
                st.write(pd.DataFrame(cr).transpose())

                # Display metrics
                st.write("##### Accuracy: ", accuracy_score(y_test, y_pred))
                st.write("##### F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
                st.write("##### Precision: ", precision_score(y_test, y_pred, average='micro'))

                # Update confusion matrix
                st.write(f"##### {model_name} Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred, labels=present_labels)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=present_target_names,
                            yticklabels=present_target_names)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt)

            # Feature Importance Visualization (for models that support it)
            if hasattr(model, "feature_importances_"):
                st.write(f"##### {model_name} Feature Importance Visualization")
                feature_importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)

                # Plot the top 10 important features
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x='Importance', 
                    y='Feature', 
                    data=feature_importance_df.head(10),
                    palette="viridis"
                )
                plt.title("Top 10 Feature Importances")
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                st.pyplot(plt)

                # Display entire feature importance table
                st.write("##### Full Feature Importance Table:")
                st.dataframe(feature_importance_df)

            # Save the last trained model
            joblib.dump(model, model_filename)
            # st.success(f"Model saved to {model_filename}")

        except Exception as e:
            st.error(f"Error during model training or evaluation: {e}")


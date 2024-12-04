# DDoS Detection Using Machine Learning

This project implements a machine learning-based system for detecting and classifying Distributed Denial of Service (DDoS) attacks. The system features an interactive Streamlit interface for real-time analysis of network traffic data, making it simple to differentiate between benign and malicious patterns.

## Features
- **User-Friendly Interface:** Built with Streamlit for easy dataset upload, model selection, and performance visualization.
- **Machine Learning Models:** Supports Random Forest, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) classifiers.
- **Real-Time Analysis:** Interactive analysis with confusion matrices, classification reports, and feature importance plots.
- **Scalable Design:** Efficient preprocessing and high-accuracy detection workflows.
- **Model Persistence:** Trained models are saved using `joblib` for reuse.

## Technologies Used
- **Programming Language:** Python  
- **Libraries:**
  - `Streamlit` - User interface
  - `Scikit-learn` - Machine learning models and metrics
  - `Pandas`, `NumPy` - Data manipulation
  - `Matplotlib`, `Seaborn` - Visualization
  - `Joblib` - Model saving/loading

## Workflow
1. **Preprocessing:**
   - Missing values are filled using column means.
   - Labels are encoded: `0` for benign traffic and `1` for DDoS attacks.
   - Features are scaled using `StandardScaler`.
2. **Model Training:**
   - Dataset split into 70% training and 30% testing subsets.
   - Models evaluated with accuracy, precision, F1 score, and confusion matrix.
3. **Results & Visualization:**
   - Includes performance metrics, class distribution, confusion matrices, and feature importance plots.

## Sample Results
Below is an example of the application output, showcasing evaluation metrics and visualizations:

`[Go to see in the report and slide]`

## How to Use
1. Clone this repository:
   ```bash
   https://github.com/vaungsophal/DDoS_Detection_ML.git

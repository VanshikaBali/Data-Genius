# Data Genius

## 📌 Overview
**Data Genius** is a comprehensive data analysis and visualization platform built using **Streamlit**. It allows users to clean, analyze, visualize, and extract insights from structured datasets with ease.

## 🎯 Features
### 🔹 User Authentication
- Secure login/signup system using **bcrypt** for password hashing.
- SQLite database for storing user credentials.

### 🔹 Data Cleaning
- Handle missing values by filling with mean, median, mode, or a constant.
- Remove duplicates.
- Detect and remove outliers (Z-score & IQR method).
- Convert data types (int, float, string, datetime).
- Remove unnecessary columns.

### 🔹 Data Analysis
- Basic statistical summaries (mean, median, standard deviation, etc.).
- Correlation matrix for feature relationships.
- Missing and unique values summary.
- Grouping and sorting functionality.

### 🔹 Data Visualization
- Bar Chart, Line Chart, Scatter Plot, Histogram, Box Plot, Violin Plot.
- Pie Chart, Heatmap, Pair Plot, and Area Chart.
- Interactive plotting using **Matplotlib** & **Seaborn**.

### 🔹 Advanced Analysis
- **Descriptive Analysis**: Summarizing numerical and categorical data.
- **Exploratory Data Analysis (EDA)**: Detect patterns and relationships.
- **Inferential Analysis**: Statistical tests (T-Test, ANOVA, Regression).
- **Predictive Analysis**: Machine Learning models (Random Forest, Linear Regression, etc.).
- **Text Analysis**: WordCloud for text-based insights.
- **Time Series Analysis**: Identifying trends and seasonality.

### 🔹 Insights & Reporting
- Generates key insights from data.
- Allows users to save insights and visualizations.
- Supports exporting insights to a CSV file.

## 🛠 Installation
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-repo/data-genius.git
cd data-genius
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Application**
```bash
streamlit run app.py
```

## 📦 Dependencies (requirements.txt)
```
streamlit
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
wordcloud
sqlite3
bcrypt
```

## 📂 Project Structure
```
Data-Genius/
│-- app.py                   # Main Streamlit app file
│-- requirements.txt          # Required dependencies
│-- data_genius.db            # SQLite database
│-- README.md                 # Documentation
```

## 🚀 Future Enhancements
- **Real-time Data Processing**
- **Integration with Cloud Storage**
- **Advanced Machine Learning Models**
- **User Dashboards & Reports**

## 📧 Contact
For any queries, feel free to reach out at vanshikabali2004@gmail.com


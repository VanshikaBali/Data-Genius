import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import io
import sqlite3
import base64
from datetime import datetime
import bcrypt  
import os
import matplotlib.pyplot as plt
import seaborn as sns


def set_custom_style():
    st.markdown(
        """
        <style>
/* Background Color - Light Mud Brown */
body, .stApp {
    background-color: #C0A98B !important; /* Light Mud Brown */
    color: #5D4037 !important; /* Dark Brown Font */
}

/* Dark Brown Text Everywhere */
.stApp, p, h1, h2, h3, h4, h5, h6, 
.stTextInput > label, .stSelectbox > label, 
.stMultiSelect > label, .stSlider > label, 
.stDataFrame, .stMarkdown, .stTable, 
.stButton, .stRadio, .stCheckbox {
    color: #5D4037 !important; /* Dark Brown */
}

/* Input Fields */
.stTextInput > div > div > input, 
.stSelectbox > div > div > div, 
.stMultiSelect > div > div > div {
    color: #5D4037 !important; /* Dark Brown Text */
    background-color: white !important; /* White Background */
    border: 2px solid #5D4037 !important; /* Dark Brown Border */
    border-radius: 6px !important;
    padding: 8px !important;
}

/* Input Fields on Focus (When Clicked) */
.stTextInput > div > div > input:focus, 
.stSelectbox > div > div > div:focus, 
.stMultiSelect > div > div > div:focus {
    border: 2px solid #8D6E63 !important; /* Medium Brown Border */
    box-shadow: 0 0 6px rgba(141, 110, 99, 0.5) !important;
}


/* Buttons */
.stButton > button {
    background-color: white !important; /* White Background */
    color: #5D4037 !important; /* Dark Brown Text */
    border: 2px solid #5D4037 !important; /* Dark Brown Border */
    border-radius: 8px !important;
    padding: 10px 18px !important;
    transition: all 0.3s ease !important;
}

/* Button Hover Effect */
.stButton > button:hover {
    background-color: #D7B899 !important; /* Light Brown Hover */
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    border: 2px solid #5D4037 !important; /* Dark Brown Border */
} 


/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: rgba(180, 140, 110, 0.9) !important; /* Softer Brown Sidebar */
    border-right: 2px solid rgba(100, 80, 60, 0.5) !important; /* Dark Brown Border */
}

/* Data Table Styling */
.stDataFrame table, .stTable table {
    color: #5D4037 !important; /* Dark Brown */
    background-color: rgba(210, 180, 150, 0.8) !important; /* Soft Brown */
}

.stDataFrame th, .stTable th {
    background-color: rgba(190, 160, 130, 0.9) !important; /* Slightly Darker Brown */
    color: #5D4037 !important; /* Dark Brown */
}

.stDataFrame td, .stTable td {
    background-color: rgba(200, 170, 140, 0.7) !important; /* Medium Brown */
    color: #5D4037 !important; /* Dark Brown */
}
</style>
        """,
        unsafe_allow_html=True
    )
set_custom_style()


import bcrypt
from datetime import datetime

def save_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()  # ✅ Hash password
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect('data_genius.db') as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, registration_date) VALUES (?, ?, ?)", 
                      (username, hashed_password, current_date))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Username already exists
        
def check_user(username, password):
    with sqlite3.connect('data_genius.db') as conn:
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        
        if user:
            stored_password = user[0]
            return bcrypt.checkpw(password.encode(), stored_password.encode())
        return False


def get_current_data():
    """Returns the most up-to-date dataset to use across all pages"""
    if 'cleaned_data' in st.session_state and not st.session_state['cleaned_data'].empty:
        return st.session_state['cleaned_data']
    elif 'df' in st.session_state:
        return st.session_state['df']
    return None

# Login Page
def login_page():
    st.title("Data Genius - User Authentication")

    # Create two columns for login and signup
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['is_admin'] = True
                st.success("Admin Login Successful!")
                st.rerun()
            elif check_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['is_admin'] = False
                st.success("Login Successful!")
                st.rerun()
            else:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['is_admin'] = False
                save_user(username, password)
                st.success("New user created and logged in!")
                st.rerun()

    with col2:
        st.subheader("Sign Up")
        new_username = st.text_input("Create Username", key="signup_username")
        new_password = st.text_input("Create Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif save_user(new_username, new_password):
                st.success("Account created successfully! Please login.")
            else:
                st.error("Username already exists. Please choose a different one.")


def home_page():
    st.title('Data Genius - Home')
    st.subheader('Welcome to Data Genius - Your Advanced Data Analysis Platform!')
    
    st.markdown("""
    ## About Data Genius

    **Data Genius** is a comprehensive, all-in-one platform designed to transform raw data into actionable insights. Our advanced suite of tools empowers users of all skill levels to unlock the full potential of their data.

    ### What You Can Do With Data Genius:

    #### 🧹 Data Cleaning and Preparation
    - Remove or fill missing values with just a few clicks
    - Identify and handle outliers that may skew your analysis
    - Convert data types to ensure compatibility with advanced analysis
    - Remove duplicates and unnecessary columns
    - Standardize and normalize your data for optimal analysis

    #### 📊 Data Visualization
    - Generate beautiful, publication-ready charts and graphs
    - Customize visualizations to highlight key insights
    - Export high-resolution images for presentations and reports
    - Explore relationships between variables through multiple visualization types
    - Share visualizations with stakeholders to communicate insights effectively

    #### 🔍 Advanced Analysis Tools
    - Perform statistical tests to validate hypotheses
    - Build predictive models to forecast future trends
    - Conduct time series analysis to identify patterns over time
    - Cluster analysis to segment your data into meaningful groups
    - Text analysis for extracting insights from unstructured data

    ### Perfect For:
    - Business analysts looking to make data-driven decisions
    - Researchers analyzing experimental results
    - Students learning data science and analytics
    - Marketers analyzing campaign performance
    - Financial analysts tracking market trends
    - Healthcare professionals analyzing patient data
    - Any professional working with structured data

    Start your data analysis journey today by uploading a CSV file below!
    """)

    # File upload with increased size limit (900MB = 900 * 1024 * 1024 bytes)
    st.markdown("### Upload Your Data")
    uploaded_file = st.file_uploader("Upload your CSV file here (up to 900MB):", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write('**Data Preview:**')
            st.write(df.head())
            
            # Store the DataFrame in session state
            st.session_state['df'] = df
            st.success('File uploaded successfully! Proceed to Data Cleaning.')
        except Exception as e:
            st.error(f"Error uploading file: {e}")

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

def data_cleaning_page():
    st.title('🧼 Data Genius - Data Cleaning')

    # Ensure data is available
    if 'cleaned_data' not in st.session_state:
        if 'df' in st.session_state:
            st.session_state['cleaned_data'] = st.session_state['df'].copy()
        else:
            st.warning("⚠ Please upload and clean your data first.")
            return

    df_cleaned = st.session_state['cleaned_data']


    # Display the current cleaned dataset
    st.write("### 🔍 Cleaned Data Preview")
    st.dataframe(df_cleaned)

    # --- RESET OPTION ---
    if st.button("🔄 Reset to Original Data"):
        st.session_state['cleaned_data'] = st.session_state['df'].copy()
        st.success("✅ Data has been reset to original!")

    # --- MISSING VALUES HANDLING ---
    st.subheader("🔍 Missing Values Handling")
    missing_values_option = st.radio(
        "How would you like to handle missing values?",
        ["Keep as is", "Remove Rows with Missing Values", "Fill Missing Values"]
    )

    if missing_values_option == "Remove Rows with Missing Values":
        missing_before = df_cleaned.isnull().sum().sum()
        df_cleaned.dropna(inplace=True)
        missing_removed = missing_before - df_cleaned.isnull().sum().sum()
        st.success(f'✅ Missing values removed: {missing_removed}')

    elif missing_values_option == "Fill Missing Values":
        fill_method = st.radio("Fill missing values with:", ["Constant Value", "Mean", "Median", "Mode"])

        if fill_method == "Constant Value":
            fill_value = st.text_input('Enter value to fill missing cells:')
            if fill_value:
                df_cleaned.fillna(fill_value, inplace=True)
                st.success(f'✅ Missing values filled with: {fill_value}')

        elif fill_method == "Mean":
            df_cleaned.fillna(df_cleaned.mean(numeric_only=True), inplace=True)
            st.success('✅ Missing values filled with column mean.')

        elif fill_method == "Median":
            df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)
            st.success('✅ Missing values filled with column median.')

        elif fill_method == "Mode":
            for col in df_cleaned.columns:
                if not df_cleaned[col].mode().empty:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            st.success('✅ Missing values filled with column mode.')

    # --- DUPLICATES HANDLING ---
    st.subheader("🗑 Remove Duplicates")
    if st.checkbox('Remove Duplicates'):
        rows_before = len(df_cleaned)
        df_cleaned.drop_duplicates(inplace=True)
        duplicates_removed = rows_before - len(df_cleaned)
        st.success(f'✅ Duplicates removed: {duplicates_removed}')

    # --- OUTLIER DETECTION ---
    st.subheader("📊 Outlier Detection & Removal")
    if st.checkbox('Detect and Remove Outliers'):
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox('Select column for outlier detection:', numeric_cols)
            outlier_method = st.radio("Choose method:", ["Z-Score", "IQR"])

            if outlier_method == "Z-Score":
                threshold = st.slider('Z-Score threshold:', 2.0, 5.0, 3.0, 0.1)
                z_scores = np.abs(stats.zscore(df_cleaned[selected_col].dropna()))
                outliers_indices = df_cleaned[z_scores > threshold].index
                st.write(f"🔍 Detected {len(outliers_indices)} outliers using Z-Score.")

                if st.button('🗑 Remove Outliers'):
                    df_cleaned.drop(outliers_indices, inplace=True)
                    st.success(f"✅ Removed {len(outliers_indices)} outliers!")

            elif outlier_method == "IQR":
                Q1 = df_cleaned[selected_col].quantile(0.25)
                Q3 = df_cleaned[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = df_cleaned[(df_cleaned[selected_col] < lower_bound) | (df_cleaned[selected_col] > upper_bound)]
                st.write(f"🔍 Detected {len(outliers)} outliers using IQR.")

                if st.button('🗑 Remove Outliers'):
                    df_cleaned = df_cleaned[(df_cleaned[selected_col] >= lower_bound) & (df_cleaned[selected_col] <= upper_bound)]
                    st.success(f"✅ Removed {len(outliers)} outliers!")

    # --- DATA TYPE CONVERSION ---
    st.subheader("🔄 Data Type Conversion")
    if st.checkbox('Convert Data Type'):
        column = st.selectbox('Select column to convert:', df_cleaned.columns)
        dtype = st.selectbox('Convert to:', ['int', 'float', 'string', 'datetime'])

        if st.button('✅ Convert'):
            try:
                if dtype == 'int':
                    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce').fillna(0).astype(int)
                elif dtype == 'float':
                    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce').fillna(0).astype(float)
                elif dtype == 'string':
                    df_cleaned[column] = df_cleaned[column].astype(str)
                elif dtype == 'datetime':
                    df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')

                st.success(f'✅ Column {column} converted to {dtype} successfully!')
            except Exception as e:
                st.error(f'❌ Error: {e}')

    # --- REMOVE UNNECESSARY COLUMNS ---
    st.subheader("🛑 Remove Unnecessary Columns")
    if st.checkbox('Remove Columns'):
        columns_to_remove = st.multiselect('Select columns to remove:', df_cleaned.columns)
        if st.button('🗑 Remove Selected Columns'):
            if columns_to_remove:
                df_cleaned.drop(columns=columns_to_remove, inplace=True)
                st.success(f'✅ Removed columns: {", ".join(columns_to_remove)}')
            else:
                st.warning('⚠ No columns selected for removal.')

    # --- STORE CLEANED DATA ---
    if st.button("✅ Save Cleaned Data"):
        st.session_state['cleaned_data'] = df_cleaned
        st.session_state['df'] = df_cleaned
        st.success("✅ Cleaned Data saved successfully!")

        st.write("### 📊 Final Cleaned Data Preview")
        st.dataframe(df_cleaned)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_analysis_page():
    st.title("📊 Data Analysis")

    # Use cleaned data if available
    if 'cleaned_data' not in st.session_state:
        if 'df' in st.session_state:
            st.session_state['cleaned_data'] = st.session_state['df'].copy()
        else:
            st.warning("⚠ Please upload and clean your data first.")
            return

    df = st.session_state['cleaned_data']

    st.markdown("### 🔍 Dataset Preview")
    st.write(df.head())

    # --- COLUMN SELECTION ---
    st.markdown("### 🎯 Select Columns for Analysis")
    all_columns = df.columns.tolist()  # ✅ Fixed
    selected_columns = st.multiselect("Select columns", options=all_columns)

    # ✅ Stop execution if no columns are selected
    if not selected_columns:
        st.warning("⚠ Please select at least one column for analysis.")
        return

    sub_df = df[selected_columns]  # ✅ Fixed

    # --- SELECTED DATA PREVIEW ---
    st.markdown("### 📊 Selected Data Preview")
    st.write(sub_df.head())

    # --- DATASET DIMENSIONS ---
    st.markdown("### 📏 Dataset Dimensions")
    st.write(f"Rows: {sub_df.shape[0]}, Columns: {sub_df.shape[1]}")

    # --- STATISTICAL SUMMARY ---
    st.markdown("### 📊 Statistical Summary")
    st.write(sub_df.describe(include='all'))

    # --- CORRELATION MATRIX ---
    numeric_cols = sub_df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 1:
        st.markdown("### 🔥 Correlation Analysis")
        corr_columns = st.multiselect("Select numeric columns for correlation analysis:", numeric_cols, 
                                      default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)

        if len(corr_columns) >= 2:
            correlation = sub_df[corr_columns].corr()
            st.write(correlation)

            # Heatmap
            st.markdown("### 🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠ Please select at least two numeric columns for correlation.")

    # --- MISSING & UNIQUE VALUES ANALYSIS ---
    st.markdown("### 📌 Missing & Unique Values")
    missing_values = sub_df.isnull().sum()
    unique_values = sub_df.nunique()

    info_df = pd.DataFrame({
        'Column': sub_df.columns,
        'Missing Values': missing_values,
        'Unique Values': unique_values
    })
    st.write(info_df)

    # --- STANDARD DEVIATION ANALYSIS ---
    numeric_sub_df = sub_df.select_dtypes(include=['number'])
    if not numeric_sub_df.empty:
        st.markdown("### 📉 Standard Deviation Analysis")
        std_dev_df = numeric_sub_df.std().reset_index()
        std_dev_df.columns = ['Column', 'Standard Deviation']
        st.write(std_dev_df)

    # --- SORT DATA ---
    st.markdown("### 🔃 Sort Data")
    sort_column = st.selectbox("Select a column to sort by:", sub_df.columns)
    sort_order = st.radio("Sort order:", ["Ascending", "Descending"])

    if sort_order == "Ascending":
        sorted_df = sub_df.sort_values(by=sort_column, ascending=True)
    else:
        sorted_df = sub_df.sort_values(by=sort_column, ascending=False)

    st.markdown("### 🔃 Sorted Data")
    st.write(sorted_df)

    # --- GROUPBY ANALYSIS ---
    st.markdown("### 🔄 Group By Analysis")
    group_by_column = st.selectbox("Select column to group by:", sub_df.columns)
    aggregation_column = st.selectbox("Select column for aggregation:", sub_df.columns)
    aggregation_function = st.selectbox("Select aggregation function:", ['sum', 'mean', 'max', 'min', 'count'])

    if st.button("Apply GroupBy"):
        try:
            if aggregation_function != 'count':
                sub_df[aggregation_column] = pd.to_numeric(sub_df[aggregation_column], errors='coerce')

            grouped_data = sub_df.groupby(group_by_column)[aggregation_column].agg(aggregation_function).reset_index()

            st.markdown("### 🔄 Grouped Data Results")
            st.write(grouped_data)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            
def data_visualization_page():
    st.title("📊 Data Visualization")

    # Use cleaned data if available
    if 'cleaned_data' in st.session_state:
        df = st.session_state['cleaned_data']
        st.success("✅ Using Cleaned Data for Visualization")
    elif 'df' in st.session_state:
        df = st.session_state['df']
        st.info("ℹ Using original data. For better results, clean your data first.")
    else:
        st.warning("⚠ Please upload a CSV file from the Home page first.")
        return

    # --- GRAPH SELECTION ---
    graph_type = st.selectbox("📈 Select Graph Type", [
        "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", 
        "Pie Chart", "Pair Plot", "Violin Plot", "Area Chart", "Heatmap"
    ], key="graph_type")

    try:
        # --- PAIR PLOT ---
        if graph_type == "Pair Plot":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect(
                    "Select columns for Pair Plot", options=numeric_cols, default=numeric_cols[:3], key="pair_plot_cols"
                )
                if len(selected_cols) >= 2 and st.button("📊 Generate Pair Plot", key="pair_plot_btn"):
                    pair_fig = sns.pairplot(df[selected_cols])
                    st.pyplot(pair_fig)
            else:
                st.warning("⚠ Not enough numeric columns for a Pair Plot.")

        # --- HEATMAP ---
        elif graph_type == "Heatmap":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect(
                    "Select columns for Heatmap", options=numeric_cols, default=numeric_cols[:5], key="heatmap_cols"
                )
                if len(selected_cols) >= 2 and st.button("🔥 Generate Heatmap", key="heatmap_btn"):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation = df[selected_cols].corr()
                    sns.heatmap(correlation, annot=True, cmap='viridis', ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠ Not enough numeric columns for a Heatmap.")

        # --- OTHER CHARTS ---
        elif graph_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Violin Plot", "Area Chart"]:
            x_axis = st.selectbox("📌 Select X-axis", df.columns, key="x_axis")
            y_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            # Ensure Y-axis selection is only for numeric data
            if graph_type == "Histogram":
                y_axis = st.selectbox("📌 Select Numeric Column for Histogram", y_numeric_cols, key="hist_y_axis")
            else:
                y_axis = st.selectbox("📌 Select Y-axis", y_numeric_cols, key="y_axis")

            if st.button(f"📊 Generate {graph_type}", key=f"{graph_type}_btn"):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Conditional logic based on chart type
                if graph_type == "Bar Chart":
                    sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
                elif graph_type == "Histogram":
                    sns.histplot(df[y_axis], kde=True, ax=ax)  # ✅ Fixed histogram
                elif graph_type == "Line Chart":
                    sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
                elif graph_type == "Scatter Plot":
                    sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
                elif graph_type == "Box Plot":
                    sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)
                elif graph_type == "Violin Plot":
                    sns.violinplot(x=df[x_axis], y=df[y_axis], ax=ax)
                elif graph_type == "Area Chart":
                    sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    ax.fill_between(df[x_axis], df[y_axis], alpha=0.3)  # ✅ Fixed fill_between

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

        # --- PIE CHART ---
        elif graph_type == "Pie Chart":
            x_axis = st.selectbox("📌 Select X-axis for Pie Chart", df.columns, key="pie_chart_x")
            
            if st.button("📊 Generate Pie Chart", key="pie_chart_btn"):
                value_counts = df[x_axis].value_counts()

                if value_counts.empty:
                    st.warning("⚠ Not enough data to create a Pie Chart.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

@st.cache_data
def get_cleaned_data():
    return st.session_state.get('cleaned_data', None)

def advanced_analysis_page():
    st.markdown("<h2 style='color: #FF5733;'><em>Advanced Analysis</em></h2>", unsafe_allow_html=True)

    # Use cleaned data if available
    if 'cleaned_data' in st.session_state:
        data = st.session_state['cleaned_data']
    else:
        st.warning("⚠ Please upload and clean data first.")
        return

    selected_analysis = st.selectbox("🔍 Select Type of Analysis", [
        "Descriptive Analysis", "Exploratory Data Analysis", "Inferential Analysis", 
        "Predictive Analysis", 
        "Text Analysis", "Time Series Analysis"
    ])

    # --- DESCRIPTIVE ANALYSIS ---
    if selected_analysis == "Descriptive Analysis":
        st.subheader("📊 Descriptive Analysis")
        st.write(data.describe())

    # --- EXPLORATORY DATA ANALYSIS ---
    elif selected_analysis == "Exploratory Data Analysis":
        st.subheader("📉 Exploratory Data Analysis (EDA)")
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.empty:
            st.error("⚠ No numerical columns found for correlation analysis.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # --- INFERENTIAL ANALYSIS ---
    elif selected_analysis == "Inferential Analysis":
        st.subheader("📏 Inferential Analysis")
        test_type = st.selectbox("Select Statistical Test", [
            "T-Test", "ANOVA", "Chi-Square Test", "Regression Analysis"
        ])

        if test_type == "T-Test":
            col1, col2 = st.selectbox("Select first numerical column", data.select_dtypes(include=['number']).columns), \
                         st.selectbox("Select second numerical column", data.select_dtypes(include=['number']).columns)
            if col1 and col2 and st.button("Run T-Test"):
                t_stat, p_value = stats.ttest_ind(data[col1].dropna(), data[col2].dropna(), nan_policy='omit')
                st.write(f"T-Test Results:\n- **T-statistic**: {t_stat:.4f}\n- **P-value**: {p_value:.4f}")

        elif test_type == "ANOVA":
            groupby_col, value_col = st.selectbox("Select grouping column (categorical)", data.select_dtypes(include=['object']).columns), \
                                     st.selectbox("Select value column (numerical)", data.select_dtypes(include=['number']).columns)
            if groupby_col and value_col and st.button("Run ANOVA"):
                groups = [data[data[groupby_col] == name][value_col].dropna() for name in data[groupby_col].unique()]
                f_stat, p_value = stats.f_oneway(*groups)
                st.write(f"ANOVA Results:\n- **F-statistic**: {f_stat:.4f}\n- **P-value**: {p_value:.4f}")

    # --- PREDICTIVE ANALYSIS (Your Provided Code) ---
    elif selected_analysis == "Predictive Analysis":
        st.subheader("🤖 Predictive Analysis")
        target = st.selectbox("🎯 Select Target Column", data.columns)
        is_regression = data[target].dtype in ['int64', 'float64'] and len(data[target].unique()) > 10
        model_type = st.radio("Select Model Type", ["Automatic", "Classification", "Regression"])

        if model_type == "Automatic":
            is_classification = not is_regression
        else:
            is_classification = model_type == "Classification"

        model_name = st.selectbox("Select Model", 
                                  ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"] if is_classification 
                                  else ["Linear Regression", "Random Forest Regressor", "Ridge", "Lasso"])

        feature_cols = st.multiselect("Select Feature Columns", 
                                      [col for col in data.columns if col != target],
                                      default=[col for col in data.columns if col != target][:5])

        X = data[feature_cols].copy()
        y = data[target].copy()

        if not np.issubdtype(y.dtype, np.number):
            st.warning("⚠ Please select a numerical target column for Predictive Analysis.")
            return  
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X.fillna(X.mean(numeric_only=True), inplace=True)

        if np.issubdtype(y.dtype, np.number):
            y.fillna(y.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if is_classification:
            model = RandomForestClassifier() if model_name == "Random Forest" else LogisticRegression()
        else:
            model = LinearRegression() if model_name == "Linear Regression" else RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if is_classification:
           st.write(f"🎯 **Accuracy Score:** {model.score(X_test, y_test):.4f}")
           st.write("📌 **Accuracy Score**: This tells how well the model correctly classifies the test data. A score of 1 means perfect predictions, while 0 means completely incorrect predictions.")
           st.write("📊 **Classification Report:**")
           st.text(classification_report(y_test, y_pred))
        else:
           st.write(f"📈 **R² Score:** {r2_score(y_test, y_pred):.4f}")
           st.write("📌 **R² Score**: This measures how well the regression model fits the data. A score close to 1 indicates a good fit, while a score near 0 or negative means the model is not performing well.")
           st.write(f"🔍 **Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.4f}")
           st.write("📌 **MSE**: This shows how far the predicted values are from the actual values. A lower MSE means better predictions, while a higher value indicates poor model performance.")
        
    # --- TEXT ANALYSIS ---
    elif selected_analysis == "Text Analysis":
        st.subheader("📜 Text Analysis")
        text_column = st.selectbox("Select text column", data.select_dtypes(include=['object']).columns)
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(data[text_column].astype(str)))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # --- TIME SERIES ANALYSIS ---
    elif selected_analysis == "Time Series Analysis":
        st.subheader("📅 Time Series Analysis")
        date_col, value_col = st.selectbox("Select Date Column", data.columns), \
                              st.selectbox("Select Numeric Column", data.select_dtypes(include=['number']).columns)
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data = data.dropna(subset=[date_col, value_col]).set_index(date_col)
        st.line_chart(data[value_col])

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import io
import os

import sqlite3
import pandas as pd

def get_all_data():
    with sqlite3.connect('data_genius.db') as conn:
        cursor = conn.cursor()

        # Fetch users data
        users_df = pd.read_sql_query("SELECT * FROM users", conn)

        # Check if insights table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='insights';")
        if cursor.fetchone():
            insights_df = pd.read_sql_query("SELECT * FROM insights", conn)
        else:
            insights_df = pd.DataFrame(columns=["id", "username", "insights_text", "date_created"])

    return users_df, insights_df 

def fetch_user_data():
    users_df, insights_df, *_ = get_all_data()
    with sqlite3.connect('data_genius.db') as conn:
        cursor = conn.cursor()

        # Check if 'users' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        if not cursor.fetchone():
            return pd.DataFrame(columns=["id", "username", "registration_date"]), pd.DataFrame(columns=["id", "username", "insights_text", "date_created"])

        # Fetch user data
        users_query = "SELECT id, username, registration_date FROM users"
        users_df = pd.read_sql_query(users_query, conn)

        # Check if 'insights' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='insights';")
        if not cursor.fetchone():
            return users_df, pd.DataFrame(columns=["id", "username", "insights_text", "date_created"])

        # Fetch insights data
        insights_query = "SELECT id, username, insights_text, date_created FROM insights"
        insights_df = pd.read_sql_query(insights_query, conn)

    return users_df, insights_df  # ✅ Now inside a function

# Call the function where needed
users_df, insights_df = fetch_user_data()


def save_insights(username, insights_text):
    # Ensure insights database table exists
    init_db()
    
    # Save to database
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        c.execute("INSERT INTO insights (username, insights_text, date_created) VALUES (?, ?, ?)", 
                (username, insights_text, current_date))
        conn.commit()
        conn.close()
        
        # Optional: Save to CSV as backup
        CSV_FILE = "saved_insights.csv"
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
        else:
            df = pd.DataFrame(columns=["username", "insights_text", "date_created"])

        new_data = pd.DataFrame([[username, insights_text, current_date]], 
                                columns=["username", "insights_text", "date_created"])
        df = pd.concat([df, new_data], ignore_index=True)
        
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving insights: {e}")
        return False

def init_db():
    """Make sure database tables exist"""
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()

    # Ensure insights table exists
    c.execute('''
    CREATE TABLE IF NOT EXISTS insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        insights_text TEXT,
        date_created TEXT,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')

    # Ensure visualizations table exists
    c.execute('''
    CREATE TABLE IF NOT EXISTS visualizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        image BLOB,
        date_created TEXT,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')

    conn.commit()
    conn.close()

def generate_textual_insights(df):
    """Generate comprehensive text insights from dataframe"""
    insights = []

    # Overall Dataset Summary
    insights.append(f"📌 **Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.**")
    insights.append(f"🔹 **Total Missing Values:** {df.isnull().sum().sum()}")
    insights.append(f"🔹 **Total Duplicates:** {df.duplicated().sum()}")

    # Column-wise Analysis
    for col in df.columns:
        insights.append(f"\n### 🔍 Column: **{col}**")
        
        # If Numeric Column
        if pd.api.types.is_numeric_dtype(df[col]):
            insights.append(f"➡ **Mean:** {df[col].mean():.2f} | **Median:** {df[col].median():.2f}")
            insights.append(f"➡ **Min Value:** {df[col].min()} | **Max Value:** {df[col].max()}")
            insights.append(f"➡ **Standard Deviation:** {df[col].std():.2f}")

        # If Categorical Column
        elif df[col].dtype == 'object':
            if not df[col].empty and df[col].notna().any():
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                
                # Only get least common if there are at least 2 unique values
                value_counts = df[col].value_counts()
                if len(value_counts) >= 2:
                    least_common = value_counts.index[-1]
                    insights.append(f"✅ **Most Common Value:** {most_common} ({value_counts.iloc[0]} occurrences)")
                    insights.append(f"⚠ **Least Common Value:** {least_common} ({value_counts.iloc[-1]} occurrences)")
                else:
                    insights.append(f"✅ **Most Common Value:** {most_common} ({value_counts.iloc[0] if not value_counts.empty else 0} occurrences)")

        # Missing Values Check
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            insights.append(f"⚠ **Missing Values:** {missing_count} ({(missing_count / df.shape[0]) * 100:.2f}%)")

    return "\n".join(insights)

def get_user_insights(username):
    """Retrieve saved insights for a specific user"""
    with sqlite3.connect('data_genius.db') as conn:
        query = "SELECT insights_text, date_created FROM insights WHERE username = ? ORDER BY date_created DESC"
        insights_df = pd.read_sql_query(query, conn, params=(username,))
    return insights_df

def insights_page():
    st.title("🔍 Data Intelligence Report")

    # Fetch data from session state
    if 'cleaned_data' in st.session_state and not st.session_state['cleaned_data'].empty:
        df = st.session_state['cleaned_data']
    elif 'df' in st.session_state and not st.session_state['df'].empty:
        df = st.session_state['df']
    else:
        st.warning("⚠ No data available. Please upload and clean your data first.")
        return

    # Create tabs for current insights and saved insights
    tab1, tab2 = st.tabs(["Current Insights", "Saved Insights"])
    
    with tab1:
        st.markdown("## 📊 Dataset Overview")
        overview_text = f"- **Total Records:** {df.shape[0]:,}\n"
        overview_text += f"- **Total Variables:** {df.shape[1]:,}\n"
        overview_text += f"- **Missing Values:** {df.isnull().sum().sum():,}\n"
        overview_text += f"- **Duplicated Rows:** {df.duplicated().sum():,}\n"
        st.markdown(overview_text)

        # --- DESCRIPTIVE ANALYSIS ---
        st.markdown("## 📊 Understanding Your Data")
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if not numeric_cols.empty:
            desc_stats = df[numeric_cols].describe()
            st.dataframe(desc_stats)
            
            insights_text = "### 📊 Descriptive Analysis Results\n"
            for col in numeric_cols:
                insights_text += f"\n🔹 **{col}:**\n"
                insights_text += f"   - Average Value: {df[col].mean():.2f}\n"
                insights_text += f"   - Range: {df[col].min()} to {df[col].max()}\n"
        else:
            insights_text = "No numeric columns available for descriptive analysis.\n"
            st.warning("⚠ No numeric columns available for descriptive analysis.")

        # --- CORRELATION INSIGHTS ---
        st.markdown("## 🔗 Business Insights from Correlation")
        num_df = df.select_dtypes(include=['number'])

        if num_df.shape[1] < 2:
            st.warning("⚠ Not enough numerical columns for correlation analysis.")
            insights_text += "⚠ Not enough numerical columns for correlation analysis.\n"
        else:
            correlation_matrix = num_df.corr()
            
            # Create a heatmap of the correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Find strongest correlations
            strong_correlations = correlation_matrix.unstack().sort_values(ascending=False)
            strong_correlations = strong_correlations[strong_correlations < 0.99]  # Exclude perfect correlations
            high_corr_pairs = strong_correlations[:5]  # Top 5 correlations

            st.markdown("### 🔥 Key Correlations Found:")
            insights_text += "### 🔥 Key Correlations Found:\n"
            if not high_corr_pairs.empty:
                for idx, correlation in enumerate(high_corr_pairs.items()):
                    (feature_x, feature_y), corr_value = correlation
                    insight = f"🔗 **{feature_x} & {feature_y}** have a correlation of **{corr_value:.2f}**."
                    insights_text += insight + "\n"
                    st.write(insight)
            else:
                insights_text += "No significant correlations found.\n"
                st.write("No significant correlations found.")

        # Generate additional insights
        st.markdown("## 🧠 Advanced Insights")
        with st.spinner("Generating comprehensive insights..."):
            full_insights = generate_textual_insights(df)
            st.markdown(full_insights)
            insights_text += "\n\n### Comprehensive Analysis\n" + full_insights

        # Save Insights Button
        if 'username' in st.session_state:
            if st.button("💾 Save Insights"):
                success = save_insights(st.session_state['username'], insights_text)
                if success:
                    st.success("✅ Insights saved successfully!")
                else:
                    st.error("Failed to save insights. Please try again.")
        else:
            st.warning("Please log in to save insights.")

    with tab2:
        if 'username' in st.session_state:
            st.markdown("## 📚 Your Saved Insights")
            insights_df = get_user_insights(st.session_state['username'])
            
            if not insights_df.empty:
                for idx, row in insights_df.iterrows():
                    with st.expander(f"Insights from {row['date_created']}"):
                        st.markdown(row['insights_text'])
            else:
                st.info("You don't have any saved insights yet. Analyze your data and save insights!")
        else:
            st.warning("Please log in to view your saved insights.")


def save_insights(username, insights_text):
    # Ensure insights database table exists
    init_db()
    
    
    # Save to database
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
         c.execute("INSERT INTO insights (username, insights_text, date_created) VALUES (?, ?, ?)", 
                (username, insights_text, current_date))
        conn.commit()
        
        # Get the ID of the newly inserted insight
        insight_id = c.lastrowid
        conn.close()
        print("✅ Insights saved successfully!") 
        
        # Get the ID of the newly inserted insight
        insight_id = c.lastrowid
        conn.close()
        
        # Always save to CSV as primary storage
        CSV_FILE = "insights_data.csv"
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
        else:
            df = pd.DataFrame(columns=["id", "username", "insights_text", "date_created"])

        new_data = pd.DataFrame([[insight_id, username, insights_text, current_date]], 
                                columns=["id", "username", "insights_text", "date_created"])
        df = pd.concat([df, new_data], ignore_index=True)
        
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving insights: {e}")
        return False


def get_insights_from_csv():
    """Retrieve insights from CSV file instead of database"""
    CSV_FILE = "insights_data.csv"
    if os.path.exists(CSV_FILE):
        insights_df = pd.read_csv(CSV_FILE)
        return insights_df
    else:
        return pd.DataFrame(columns=["id", "username", "insights_text", "date_created"])

def is_admin():
    return st.session_state.get("username") == "admin"

def database_page():
    if not is_admin():
        st.error("⛔ Access Denied: Admin Only!")
        return

    st.title("📂 Admin Database View")

    users_df, insights_df = get_all_data()  

    combined_df = pd.concat([
        users_df.assign(Type="User"),
        insights_df.assign(Type="Insight")
    ], ignore_index=True)

    st.subheader("📊 Combined Data View")
    st.write(combined_df)
    
    # Add CSV export functionality
    st.subheader("📥 Export Insights as CSV")
    
    if st.button("Download Insights as CSV"):
        csv = insights_df.to_csv(index=False)
        
        # Create a download button
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="insights_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ CSV file ready for download!")
    
    # Add filter options
    st.subheader("🔍 Filter Insights")
    
    # Filter by username
    if not insights_df.empty:
        usernames = ["All"] + list(insights_df["username"].unique())
        selected_username = st.selectbox("Select Username:", usernames)
        
        if selected_username != "All":
            filtered_insights = insights_df[insights_df["username"] == selected_username]
        else:
            filtered_insights = insights_df
            
        st.write(filtered_insights)
        
        # Export filtered insights
        if st.button("Export Filtered Insights as CSV"):
            csv = filtered_insights.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_insights.csv">Download Filtered CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Filtered CSV ready for download!")

            
# Main Function to Handle Page Navigation
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login_page()
    else:
        st.sidebar.title("Navigation")
        page_choice = st.sidebar.selectbox(
            'Select Page:', 
            ['Home', 'Data Cleaning', 'Data Analysis', 'Data Visualization', 'Advanced Analysis', 'Insights','Admin Database', 'Logout']
        )

        if page_choice == 'Home':
            home_page()
        elif page_choice == 'Data Cleaning':
            data_cleaning_page()
        elif page_choice == 'Data Analysis':
            data_analysis_page()
        elif page_choice == 'Data Visualization':
            data_visualization_page()
        elif page_choice == 'Advanced Analysis':
            advanced_analysis_page()
        elif page_choice == 'Insights':
            insights_page()
        elif page_choice == 'Admin Database':
            database_page()
        elif page_choice == 'Logout':
            st.session_state.clear()
            st.success("You have been logged out.")
            st.rerun()


if __name__ == '__main__':
    main()



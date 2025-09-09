import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Home Credit Risk Dashboard")

# Use st.cache_data to load and process data only once
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the application_train.csv data.
    """
    df = pd.read_csv(file_path)

    # 1. Convert ages: DAYS_BIRTH is negative
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25

    # 2. Employment tenure: DAYS_EMPLOYED is negative
    # Replace the outlier value (365243) with NaN so it can be imputed later
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25

    # 3. Create financial ratios
    df['DTI'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['LOAN_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # 4. Handle missing values (simple strategy for this dashboard)
    # For key numeric columns used in the dashboard, we'll impute with the median
    numeric_cols_to_impute = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS', 'EMPLOYMENT_YEARS', 'DTI', 'LOAN_TO_INCOME', 'ANNUITY_TO_CREDIT']
    for col in numeric_cols_to_impute:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # For a key categorical column, impute with the mode
    if 'OCCUPATION_TYPE' in df.columns:
        df['OCCUPATION_TYPE'].fillna(df['OCCUPATION_TYPE'].mode()[0], inplace=True)

    # 7. Define income brackets (using quantiles)
    df['INCOME_BRACKET'] = pd.qcut(df['AMT_INCOME_TOTAL'], q=[0, 0.25, 0.75, 1], labels=['Low', 'Mid', 'High'])
    
    return df

# Load the data
df = load_and_preprocess_data(r'D:\Home Credit EDA\data\application_train.csv')

st.title("📘 Home Credit Default Risk Dashboard")

# --- Main App Structure will go here ---

# --- SIDEBAR ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Page", ["Overview & Data Quality", "Target & Risk Segmentation", "Demographics & Household Profile", "Financial Health & Affordability", "Correlations & Drivers"])

st.sidebar.header("Global Filters")

# Gender filter
gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['CODE_GENDER'].unique(),
    default=df['CODE_GENDER'].unique()
)

# Age range filter
age_min, age_max = st.sidebar.slider(
    "Select Age Range",
    min_value=int(df['AGE_YEARS'].min()),
    max_value=int(df['AGE_YEARS'].max()),
    value=(int(df['AGE_YEARS'].min()), int(df['AGE_YEARS'].max()))
)

# Income bracket filter
income_bracket = st.sidebar.multiselect(
    "Select Income Bracket",
    options=df['INCOME_BRACKET'].unique(),
    default=df['INCOME_BRACKET'].unique()
)

# Apply filters to the dataframe
filtered_df = df[
    (df['CODE_GENDER'].isin(gender)) &
    (df['AGE_YEARS'].between(age_min, age_max)) &
    (df['INCOME_BRACKET'].isin(income_bracket))
]

# --- PAGE ROUTING ---
# We will add functions for each page below

def page_overview(df):
    st.header("Page 1: Overview & Data Quality")

    # KPIs
    st.subheader("Key Performance Indicators")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Applicants", f"{df['SK_ID_CURR'].nunique():,}")
    kpi_cols[1].metric("Default Rate", f"{df['TARGET'].mean():.2%}")
    kpi_cols[2].metric("Repaid Rate", f"{(1 - df['TARGET'].mean()):.2%}")
    kpi_cols[3].metric("Median Age", f"{df['AGE_YEARS'].median():.1f} Years")
    kpi_cols[4].metric("Median Annual Income", f"${df['AMT_INCOME_TOTAL'].median():,.0f}")
    
    st.subheader("Data Distributions and Quality")
    graph_cols = st.columns(2)

    # Graph 1: Target Distribution
    target_counts = df['TARGET'].value_counts().reset_index()
    target_counts.columns = ['TARGET', 'count']
    target_counts['TARGET'] = target_counts['TARGET'].map({0: 'Repaid', 1: 'Default'})
    fig_target = px.pie(target_counts, names='TARGET', values='count', title='Loan Repayment Status (0: Repaid, 1: Default)', hole=0.3)
    graph_cols[0].plotly_chart(fig_target, use_container_width=True)

    # Graph 2: Age Distribution
    fig_age = px.histogram(df, x='AGE_YEARS', nbins=50, title='Age Distribution of Applicants')
    graph_cols[1].plotly_chart(fig_age, use_container_width=True)

    # Graph 3: Income Distribution
    fig_income = px.histogram(df, x='AMT_INCOME_TOTAL', nbins=100, title='Total Income Distribution')
    fig_income.update_xaxes(range=[0, df['AMT_INCOME_TOTAL'].quantile(0.95)]) # Trim outliers for better view
    graph_cols[0].plotly_chart(fig_income, use_container_width=True)

    # Graph 4: Credit Amount Distribution
    fig_credit = px.histogram(df, x='AMT_CREDIT', nbins=100, title='Credit Amount Distribution')
    fig_credit.update_xaxes(range=[0, df['AMT_CREDIT'].quantile(0.95)]) # Trim outliers
    graph_cols[1].plotly_chart(fig_credit, use_container_width=True)

    # Narrative
    st.markdown("""
    ### Narrative Insights
    - The overall default rate is a key indicator of portfolio risk. In this dataset, approximately **8%** of applicants have difficulty with repayment.
    - The applicant base is primarily of working age, with a concentration between **25 and 50 years old**.
    - Both income and credit amount distributions are **right-skewed**, indicating that a majority of applicants have lower-to-mid-range incomes and loan amounts, with a few high-value cases.
    """)


# Page 2: Target & Risk Segmentation

def page_risk_segmentation(df):
    st.header("Page 2: Target & Risk Segmentation")
    st.markdown("Understanding how default rates vary across key segments.")

    # KPIs
    st.subheader("Default Rates by Segment")
    kpi_cols = st.columns(4)
    defaults = df[df['TARGET'] == 1]
    kpi_cols[0].metric("Total Defaults", f"{len(defaults):,}")
    kpi_cols[1].metric("Avg Income (Defaulters)", f"${defaults['AMT_INCOME_TOTAL'].mean():,.0f}")
    kpi_cols[2].metric("Avg Credit (Defaulters)", f"${defaults['AMT_CREDIT'].mean():,.0f}")
    kpi_cols[3].metric("Avg Employment (Defaulters)", f"{defaults['EMPLOYMENT_YEARS'].mean():.1f} Years")

    graph_cols = st.columns(2)

    # Default rate by Gender
    gender_default = df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
    fig_gender = px.bar(gender_default, x='CODE_GENDER', y='TARGET', title='Default Rate by Gender', labels={'TARGET': 'Default Rate (%)'})
    fig_gender.update_yaxes(tickformat=".2%")
    graph_cols[0].plotly_chart(fig_gender, use_container_width=True)
    
    # Default rate by Education
    edu_default = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
    fig_edu = px.bar(edu_default, x='NAME_EDUCATION_TYPE', y='TARGET', title='Default Rate by Education Level')
    fig_edu.update_yaxes(tickformat=".2%")
    graph_cols[1].plotly_chart(fig_edu, use_container_width=True)

    # Income by Target
    fig_income_target = px.box(df, x='TARGET', y='AMT_INCOME_TOTAL', title='Income Distribution by Repayment Status')
    fig_income_target.update_yaxes(range=[0, df['AMT_INCOME_TOTAL'].quantile(0.9)]) # Zoom in
    graph_cols[0].plotly_chart(fig_income_target, use_container_width=True)

    # Age by Target
    fig_age_target = px.violin(df, x='TARGET', y='AGE_YEARS', title='Age Distribution by Repayment Status', box=True)
    graph_cols[1].plotly_chart(fig_age_target, use_container_width=True)
    
    st.markdown("""
    ### Narrative Insights
    - **Demographic Risk:** Default rates appear higher for males and individuals with lower levels of education (e.g., Secondary).
    - **Financial Profile:** Applicants who default tend to have slightly lower incomes, although there is significant overlap. Younger applicants also show a slightly higher tendency to default.
    """)

# Page 3: Demographics & Household Profile    

def page_demographics(df):
    st.header("👨‍👩‍👧‍👦 Demographics & Household Profile")
    st.markdown("Who are the applicants? Exploring their household structure and personal attributes.")

    st.subheader("Applicant Profile")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Avg Family Size", f"{df['CNT_FAM_MEMBERS'].mean():.2f}")
    kpi_cols[1].metric("% With Children", f"{(df['CNT_CHILDREN'] > 0).mean():.1%}")
    kpi_cols[2].metric("% Higher Education", f"{(df['NAME_EDUCATION_TYPE'] == 'Higher education').mean():.1%}")
    kpi_cols[3].metric("% Home Owners", f"{(df['FLAG_OWN_REALTY'] == 'Y').mean():.1%}")

    st.subheader("Distributions")
    chart_cols = st.columns(2)

    # THIS SECTION IS FIXED
    occ_dist = df['OCCUPATION_TYPE'].value_counts().nlargest(10).reset_index()
    occ_dist.columns = ['Occupation Type', 'Count'] # Explicitly name columns
    fig_occ = px.bar(occ_dist, x='Occupation Type', y='Count', title='Top 10 Occupation Types')
    chart_cols[0].plotly_chart(fig_occ, use_container_width=True, key="demo_occ_bar")
    
    housing_dist = df['NAME_HOUSING_TYPE'].value_counts().reset_index()
    housing_dist.columns = ['Housing Type', 'Count'] # Explicitly name columns
    fig_housing = px.pie(housing_dist, names='Housing Type', values='Count', title='Housing Type Distribution', hole=0.4)
    chart_cols[1].plotly_chart(fig_housing, use_container_width=True, key="demo_housing_pie")
    # END OF FIX

    fig_age_violin = px.violin(df, x='TARGET', y='AGE_YEARS', box=True, title='Age Distribution by Target')
    st.plotly_chart(fig_age_violin, use_container_width=True, key="demo_age_violin")

    st.markdown("""
    ### Narrative
    - **Life Stage:** The typical applicant is a homeowner, often without children, and works in common labor-intensive jobs.
    - **Age and Risk:** The violin plot reinforces that the distribution of defaulters is skewed towards a younger age compared to non-defaulters.
    """)



# Page 4: Financial Health & Affordability
def page_financial_health(df):
    st.header("💰 Financial Health & Affordability")
    st.markdown("Analyzing applicants' ability to repay based on financial ratios and stress indicators.")

    st.subheader("Key Financial Ratios")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Avg Debt-to-Income (DTI)", f"{df['DTI'].mean():.2%}")
    kpi_cols[1].metric("Avg Loan-to-Income (LTI)", f"{df['LOAN_TO_INCOME'].mean():.2f}")
    non_defaulters = df[df['TARGET'] == 0]['AMT_INCOME_TOTAL'].mean()
    defaulters = df[df['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()
    kpi_cols[2].metric("Income Gap (Non-Def - Def)", f"${non_defaulters - defaulters:,.0f}")
    kpi_cols[3].metric("% High Credit (>1M)", f"{(df['AMT_CREDIT'] > 1_000_000).mean():.1%}")

    st.subheader("Financial Relationships")
    
    income_bracket_risk = df.groupby('INCOME_BRACKET', observed=False)['TARGET'].mean().reset_index()
    fig_income_risk = px.bar(income_bracket_risk, x='INCOME_BRACKET', y='TARGET', title='Default Rate by Income Bracket')
    fig_income_risk.update_yaxes(title_text='Default Rate', tickformat=".2%")
    st.plotly_chart(fig_income_risk, use_container_width=True, key="fin_income_risk_bar")

    st.markdown("A sample of 5,000 points is used for the scatter plot for performance.")
    
    # --- THIS SECTION IS FIXED ---
    # 1. Create the sample DataFrame first to avoid shape errors.
    # Check if there are enough rows to sample
    if len(df) >= 5000:
        sample_df = df.sample(5000, random_state=42)
    else:
        sample_df = df.copy() # Use all data if less than 5000

    # 2. Use the 'sample_df' for ALL arguments: data, x, y, and color.
    fig_scatter = px.scatter(
        sample_df,
        x='AMT_INCOME_TOTAL', # Switched to Income vs Credit as per original assignment
        y='AMT_CREDIT',
        color=sample_df['TARGET'].map({0: 'Repaid', 1: 'Default'}),
        title='Income vs. Credit Amount (Sampled)',
        opacity=0.6,
        labels={'color': 'Loan Status', 'AMT_INCOME_TOTAL': 'Total Income', 'AMT_CREDIT': 'Credit Amount'}
    )
    # Adjust axis ranges based on the sample for better visualization
    fig_scatter.update_xaxes(range=[0, sample_df['AMT_INCOME_TOTAL'].quantile(0.99)])
    fig_scatter.update_yaxes(range=[0, sample_df['AMT_CREDIT'].quantile(0.99)])
    st.plotly_chart(fig_scatter, use_container_width=True, key="fin_scatter")
    # --- END OF FIX ---

    st.markdown("""
    ### Narrative
    - **Affordability Risk:** Applicants in the 'Low' income bracket have a noticeably higher default rate, suggesting affordability is a key risk factor.
    - **Ratio Analysis:** The scatter plot shows that while defaults occur across the spectrum, there is a concentration of defaults at higher Loan-to-Income (LTI) ratios, indicating that loans which are large relative to income are riskier.
    """)

# Page 5: Correlations & Drivers
def page_correlations(df):
    st.header("Page 5: Correlations, Drivers & Interactive Analysis")
    st.markdown("Identifying the key features correlated with default risk.")

    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=np.number)
    
    # Calculate correlations with TARGET
    corr_matrix = numeric_df.corr()
    target_corr = corr_matrix['TARGET'].sort_values(ascending=False)

    # KPIs
    st.subheader("Top Correlated Features with Default")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Highest Positive Corr.", f"{target_corr.index[1]} ({target_corr.iloc[1]:.3f})")
    kpi_cols[1].metric("Highest Negative Corr.", f"{target_corr.index[-1]} ({target_corr.iloc[-1]:.3f})")
    kpi_cols[2].metric("Corr(Age, TARGET)", f"{df['AGE_YEARS'].corr(df['TARGET']):.3f}")
    kpi_cols[3].metric("Corr(Employment, TARGET)", f"{df['EMPLOYMENT_YEARS'].corr(df['TARGET']):.3f}")

    # Bar chart of top correlations
    st.subheader("Top 20 Features Correlated with TARGET")
    top_corr = pd.concat([target_corr.head(10), target_corr.tail(10)]).sort_values(ascending=False)
    fig_corr = px.bar(top_corr, x=top_corr.values, y=top_corr.index, orientation='h',title='Features Correlated with Loan Default',labels={'x': 'Correlation Coefficient', 'y': 'Feature'})
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Heatmap of key financial variables
    st.subheader("Correlation Heatmap of Key Variables")
    heatmap_cols = ['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'DTI', 'LOAN_TO_INCOME']
    heatmap_corr = df[heatmap_cols].corr()
    
    fig_heatmap, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
    st.pyplot(fig_heatmap)

    st.markdown("""
    ### Narrative Insights & Policy Ideas
    - **Key Drivers:** Features like `AGE_YEARS` and `EMPLOYMENT_YEARS` are negatively correlated with default, meaning older, more experienced applicants are less risky. Conversely, factors related to external credit scores (if we were to include them) are often the strongest predictors.
    - **Policy Rule Candidates:**
        - **Age & Employment:** The strong negative correlation suggests that policies could be adjusted for younger applicants with short employment histories, perhaps requiring a co-signer or a lower initial credit limit.
        - **Affordability Ratios:** Even with weak correlation, `DTI` and `LTI` are logical policy levers. We could set stricter caps on these ratios for segments identified as high-risk on Page 2.
    """)
# Update the routing logic at the end of app.py
# --- PAGE ROUTING ---
if page == "Overview & Data Quality":
    page_overview(filtered_df)
elif page == "Target & Risk Segmentation":
    page_risk_segmentation(filtered_df)
elif page == "Demographics & Household Profile":
    page_demographics(filtered_df)
elif page == "Financial Health & Affordability":
    page_financial_health(filtered_df)
elif page == "Correlations & Drivers":
    page_correlations(filtered_df)
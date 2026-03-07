# churn dashboard
# run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_preprocessing_artifacts, prepare_single_customer
from src.model import ChurnModel
from src.analysis import ChurnAnalyzer

st.set_page_config(page_title="Churn Analytics", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    # load the trained model and scaler
    try:
        model = ChurnModel.load('models/churn_model.pkl')
        scaler, feature_names = load_preprocessing_artifacts('models/')
        return model, scaler, feature_names
    except Exception as e:
        print(f"couldnt load model: {e}")
        return None, None, None


@st.cache_data
def load_data():
    # try predictions file first, fall back to raw data
    try:
        if os.path.exists('reports/customer_predictions.csv'):
            return pd.read_csv('reports/customer_predictions.csv')
        return pd.read_csv('customer_churn_data.csv')
    except:
        return None


def main():
    st.title("📊 Customer Churn Analytics Dashboard")
    
    model, scaler, feature_names = load_model()
    df = load_data()
    
    if model is None:
        st.error("⚠️ Model not found! Run `python train_model.py` first.")
        return
    
    page = st.sidebar.radio("Navigation", ["📈 Overview", "🔍 Predict Churn", "📊 Analytics"])
    
    if page == "📈 Overview":
        st.header("Business Overview")
        
        if 'churn_probability' in df.columns:
            analyzer = ChurnAnalyzer(df)
            metrics = analyzer.calculate_business_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", f"{metrics.total_customers:,}")
            col2.metric("Churn Rate", f"{metrics.churn_rate:.1%}")
            col3.metric("High Risk", f"{metrics.high_risk_customers:,}")
            col4.metric("Revenue at Risk", f"${metrics.revenue_at_risk:,.0f}/mo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Distribution")
                risk_counts = df['risk_category'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                           color=risk_counts.index,
                           color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'})
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Probability Distribution")
                fig = px.histogram(df, x='churn_probability', nbins=30)
                fig.add_vline(x=0.40, line_dash="dash", line_color="orange")
                fig.add_vline(x=0.75, line_dash="dash", line_color="red")
                st.plotly_chart(fig)
    
    elif page == "🔍 Predict Churn":
        st.header("Predict Customer Churn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
            contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "None"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        
        with col2:
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                customer = {
                    'Age': age, 'Gender': gender, 'Tenure': tenure,
                    'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
                    'ContractType': contract_type, 'InternetService': internet_service,
                    'TechSupport': tech_support
                }
                
                try:
                    X = prepare_single_customer(customer, scaler, feature_names)
                    prob = model.predict_proba(X)[0]
                    risk = model.classify_risk(X)[0]
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob * 100,
                        title={'text': "Churn Probability"},
                        number={'suffix': '%'},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': 'red' if prob > 0.75 else 'orange' if prob > 0.4 else 'green'},
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(0,255,0,0.2)'},
                                {'range': [40, 75], 'color': 'rgba(255,165,0,0.2)'},
                                {'range': [75, 100], 'color': 'rgba(255,0,0,0.2)'}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    if risk == "High Risk":
                        st.error(f"⚠️ {risk} - Immediate attention needed!")
                    elif risk == "Medium Risk":
                        st.warning(f"⚡ {risk} - Monitor closely")
                    else:
                        st.success(f"✅ {risk} - Customer is stable")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif page == "📊 Analytics":
        st.header("Churn Analytics")
        
        if 'Churn' in df.columns:
            df_plot = df.copy()
            df_plot['churned'] = (df_plot['Churn'] == 'Yes').astype(int)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Churn by Contract Type")
                churn_by_contract = df_plot.groupby('ContractType')['churned'].mean() * 100
                fig = px.bar(x=churn_by_contract.index, y=churn_by_contract.values,
                           labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'})
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Churn by Tenure")
                df_plot['tenure_seg'] = pd.cut(df_plot['Tenure'], bins=[0,6,12,24,48,100],
                                               labels=['0-6mo','6-12mo','1-2yr','2-4yr','4+yr'])
                churn_by_tenure = df_plot.groupby('tenure_seg')['churned'].mean() * 100
                fig = px.bar(x=churn_by_tenure.index.astype(str), y=churn_by_tenure.values,
                           labels={'x': 'Tenure', 'y': 'Churn Rate (%)'})
                st.plotly_chart(fig)
        
        st.subheader("Customer Data")
        st.dataframe(df)


if __name__ == '__main__':
    main()
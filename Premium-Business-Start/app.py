import streamlit as st
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import re
import base64
from io import BytesIO
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Premium Business Planner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Base */
    body, .main {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #E2E8F0;  /* light text for dark bg */
    }

    .stApp {
        background-color: #1A202C; /* dark muted background */
    }

    /* Header */
    .main-header {
        background: #2D3748; /* dark card background */
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
        border: 1px solid #4A5568;
        text-align: center;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        color: #F7FAFC;
    }
    .main-header p {
        font-size: 0.95rem;
        color: #CBD5E0;
        margin-top: 0.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #2D3748; /* darker card */
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #4A5568;
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.7);
    }
    .metric-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #CBD5E0;
        margin-bottom: 0.25rem;
    }
    .metric-card p {
        font-size: 1.4rem;
        font-weight: 700;
        color: #63B3ED; /* muted blue */
        margin: 0;
    }

    /* Section Titles */
    .section-header {
        color: #E2E8F0;
        font-weight: 700;
        font-size: 1.4rem;
        margin: 2rem 0 1rem;
        border-bottom: 2px solid #4A5568;
        padding-bottom: 0.25rem;
    }

    /* Buttons */
    .premium-button {
        background-color: #2B6CB0;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .premium-button:hover {
        background-color: #1E429F;
    }

    /* Alerts */
    .success-message {
        background-color: #38A169; 
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
    }
    .warning-message {
        background-color: #DD6B20;
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: #2D3748;
        border-right: 1px solid #4A5568;
    }

    /* Tables */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    table thead th {
        background-color: #4A5568;
        color: #E2E8F0;
        font-weight: 600;
        padding: 0.75rem;
        border-bottom: 1px solid #718096;
    }
    table tbody td {
        padding: 0.75rem;
        border-bottom: 1px solid #718096;
        font-size: 0.95rem;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)





# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BusinessData:
    """Data class to store all business information"""
    business_type: str = ""
    business_name: str = ""
    slogan: str = ""
    target_location: str = ""
    budget: float = 0.0
    target_audience: str = ""
    business_model: str = ""
    timeline: str = ""

class PremiumBusinessPlanner:
    """Premium Business Planner with Streamlit UI"""
    
    def __init__(self):
        self.setup_gemini()
        self.business_data = BusinessData()
        
    def setup_gemini(self):
        """Setup Gemini API"""
        api_key =  "AIzaSyCkEZYIIQdYBBLfo3G-UJe1FoPNsa9eDAg"
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        self.gemini_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=generation_config
        )
    
    def call_gemini(self, prompt: str) -> str:
        """Call Gemini API with error handling"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return f"Error: {str(e)}"
    
    def create_financial_chart(self, business_type: str, budget: float):
        """Create financial projections chart"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Generate realistic financial projections based on business type and budget
        startup_costs = budget * 0.7
        monthly_expenses = budget * 0.1
        
        # Revenue growth simulation
        initial_revenue = monthly_expenses * 0.5
        revenue = []
        expenses = []
        cumulative_profit = []
        running_total = -startup_costs
        
        for i in range(12):
            # Revenue grows exponentially but levels off
            month_revenue = initial_revenue * (1 + (i * 0.3)) * (1 + np.random.normal(0, 0.1))
            month_expenses = monthly_expenses * (1 + (i * 0.02)) * (1 + np.random.normal(0, 0.05))
            
            revenue.append(max(0, month_revenue))
            expenses.append(month_expenses)
            
            running_total += month_revenue - month_expenses
            cumulative_profit.append(running_total)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Revenue vs Expenses', 'Cumulative Profit/Loss', 'Revenue Growth Trend', 'Expense Breakdown'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Revenue vs Expenses
        fig.add_trace(
            go.Bar(x=months, y=revenue, name="Revenue", marker_color='#48bb78'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=months, y=expenses, name="Expenses", marker_color='#f56565'),
            row=1, col=1
        )
        
        # Cumulative Profit/Loss
        fig.add_trace(
            go.Scatter(x=months, y=cumulative_profit, mode='lines+markers', 
                      name="Cumulative P&L", line=dict(color='#667eea', width=3)),
            row=1, col=2
        )
        
        # Revenue Growth Trend
        fig.add_trace(
            go.Scatter(x=months, y=revenue, mode='lines+markers', 
                      name="Revenue Trend", line=dict(color='#38a169', width=3)),
            row=2, col=1
        )
        
        # Expense Breakdown
        expense_categories = ['Marketing', 'Operations', 'Rent', 'Utilities', 'Salaries', 'Other']
        expense_values = [25, 30, 20, 5, 15, 5]
        
        fig.add_trace(
            go.Pie(labels=expense_categories, values=expense_values, name="Expenses"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Financial Projections Dashboard - {business_type.title()}",
            title_x=0.5
        )
        
        return fig
    
    def create_market_analysis_chart(self):
        """Create market analysis visualization"""
        # Simulate market data
        market_segments = ['Target Market', 'Secondary Market', 'Tertiary Market', 'Niche Market']
        market_size = [45, 30, 15, 10]
        growth_rate = [12, 8, 5, 15]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Market Share Distribution', 'Growth Opportunities'),
            specs=[[{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # Market Share Pie Chart
        colors = ['#667eea', '#764ba2', '#48bb78', '#ed8936']
        fig.add_trace(
            go.Pie(labels=market_segments, values=market_size, 
                  marker_colors=colors, name="Market Share"),
            row=1, col=1
        )
        
        # Growth Rate Bar Chart
        fig.add_trace(
            go.Bar(x=market_segments, y=growth_rate, 
                  marker_color=['#667eea', '#764ba2', '#48bb78', '#ed8936'],
                  name="Growth Rate %"),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text="Market Analysis Overview",
            title_x=0.5
        )
        
        return fig
    
    def create_timeline_chart(self, timeline: str):
        """Create implementation timeline chart"""
        # Parse timeline and create phases
        phases = [
            'Business Planning', 'Legal Setup', 'Funding & Investment', 
            'Location & Setup', 'Marketing Launch', 'Operations Start', 
            'Growth & Scaling', 'Performance Review'
        ]
        
        # Calculate timeline based on user input
        try:
            timeline_months = int(re.findall(r'\d+', timeline)[0]) if re.findall(r'\d+', timeline) else 6
        except:
            timeline_months = 6
            
        start_date = datetime.now()
        duration_per_phase = timeline_months / len(phases)
        
        fig = go.Figure()
        
        colors = ['#667eea', '#764ba2', '#48bb78', '#ed8936', '#38a169', '#9f7aea', '#f093fb', '#f5576c']
        
        for i, phase in enumerate(phases):
            phase_start = start_date + timedelta(days=i * duration_per_phase * 30)
            phase_end = phase_start + timedelta(days=duration_per_phase * 30)
            
            fig.add_trace(go.Scatter(
                x=[phase_start, phase_end],
                y=[phase, phase],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=10),
                marker=dict(size=10),
                name=phase,
                hovertemplate=f"<b>{phase}</b><br>Start: %{{x}}<br>Duration: {duration_per_phase:.1f} months<extra></extra>"
            ))
        
        fig.update_layout(
            title="Implementation Timeline",
            xaxis_title="Timeline",
            yaxis_title="Project Phases",
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def create_risk_matrix(self):
        """Create risk assessment matrix"""
        risks = [
            {'name': 'Market Competition', 'probability': 0.8, 'impact': 0.7},
            {'name': 'Economic Downturn', 'probability': 0.4, 'impact': 0.9},
            {'name': 'Regulatory Changes', 'probability': 0.3, 'impact': 0.6},
            {'name': 'Supply Chain Issues', 'probability': 0.5, 'impact': 0.5},
            {'name': 'Technology Disruption', 'probability': 0.6, 'impact': 0.8},
            {'name': 'Talent Shortage', 'probability': 0.7, 'impact': 0.4},
            {'name': 'Funding Challenges', 'probability': 0.4, 'impact': 0.8},
            {'name': 'Customer Acquisition', 'probability': 0.9, 'impact': 0.6}
        ]
        
        fig = go.Figure()
        
        for risk in risks:
            color = 'red' if risk['probability'] > 0.7 and risk['impact'] > 0.7 else \
                   'orange' if risk['probability'] > 0.5 and risk['impact'] > 0.5 else 'green'
            
            fig.add_trace(go.Scatter(
                x=[risk['probability']],
                y=[risk['impact']],
                mode='markers+text',
                marker=dict(size=20, color=color, opacity=0.7),
                text=risk['name'],
                textposition="top center",
                name=risk['name'],
                hovertemplate=f"<b>{risk['name']}</b><br>Probability: {risk['probability']:.0%}<br>Impact: {risk['impact']:.0%}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Risk Assessment Matrix",
            xaxis_title="Probability",
            yaxis_title="Impact",
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            height=600,
            showlegend=False
        )
        
        # Add quadrant lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        
        return fig

def main():
    planner = PremiumBusinessPlanner()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: #white; margin-bottom: 0.5rem;">🚀 Premium Business Planner</h1>
        <p style="text-align: center; color: #718096; font-size: 1.2rem; margin: 0;">
            Professional-grade business planning for discerning entrepreneurs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("### 📋 Business Information")
        
        business_type = st.text_input("Business Type", placeholder="e.g., Tech Startup, Restaurant, Consulting")
        target_location = st.text_input("Target Location", placeholder="e.g., New York, NY")
        budget = st.number_input("Startup Budget ($)", min_value=0.0, value=50000.0, step=1000.0)
        target_audience = st.text_input("Target Audience", placeholder="e.g., Young professionals, Small businesses")
        business_model = st.selectbox("Business Model", ["B2B", "B2C", "B2B2C", "Subscription", "Marketplace", "Freemium"])
        timeline = st.text_input("Launch Timeline", placeholder="e.g., 6 months", value="6 months")
        
        st.markdown("---")
        generate_plan = st.button("🎯 Generate Business Plan", type="primary", use_container_width=True)
    
    # Main content area
    if generate_plan and business_type:
        # Store business data
        planner.business_data.business_type = business_type
        planner.business_data.target_location = target_location
        planner.business_data.budget = budget
        planner.business_data.target_audience = target_audience
        planner.business_data.business_model = business_model
        planner.business_data.timeline = timeline
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        

        status_text.text("🏷️ Generating business names...")
        progress_bar.progress(10)

        names_prompt = f"""
        Generate 5 creative and memorable business names for a {business_type} business.
        Consider the target audience: {target_audience}
        Location: {target_location}

        Return only the names, one per line.
        """

        business_names_text = planner.call_gemini(names_prompt)
        business_names = [name.strip() for name in business_names_text.split('\n') if name.strip()][:5]

        # Automatically use the first generated name
        selected_name = business_names[0] if business_names else f"{business_type} Business"
        planner.business_data.business_name = selected_name

        # Display all suggested names for reference
        st.markdown("### 🏷️ Suggested Business Names")
        st.markdown(f"**Selected:** {selected_name}")
        if len(business_names) > 1:
            st.markdown(f"**Other suggestions:** {', '.join(business_names[1:])}")

        # Generate slogan
        status_text.text("🎨 Generating slogan...")
        progress_bar.progress(20)

        slogan_prompt = f"""
        Create a powerful, memorable slogan for "{selected_name}", 
        a {business_type} business targeting {target_audience}.
        Return only the slogan.
        """

        slogan = planner.call_gemini(slogan_prompt)
        planner.business_data.slogan = slogan

        # Continue with the rest of the code (business overview display, etc.)
        
        # Display business overview
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #white; margin-bottom: 1rem;">🏢 {selected_name}</h2>
            <p style="font-style: italic; font-size: 1.1rem; color: #4A5568; margin-bottom: 1rem;">"{slogan}"</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div><strong>Type:</strong> {business_type}</div>
                <div><strong>Location:</strong> {target_location}</div>
                <div><strong>Budget:</strong> ${budget:,.2f}</div>
                <div><strong>Model:</strong> {business_model}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Financial projections with visualization
        status_text.text("💰 Generating financial projections...")
        progress_bar.progress(40)
        
        st.markdown("### 📊 Financial Dashboard")
        financial_chart = planner.create_financial_chart(business_type, budget)
        st.plotly_chart(financial_chart, use_container_width=True)
        
        # Market analysis
        status_text.text("📈 Analyzing market...")
        progress_bar.progress(60)
        
        st.markdown("### 🎯 Market Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            market_chart = planner.create_market_analysis_chart()
            st.plotly_chart(market_chart, use_container_width=True)
        
        with col2:
            # Key metrics
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Market Metrics</h4>
                <div style="display: grid; gap: 1rem; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Total Market Size:</span>
                        <span><strong>$2.4B</strong></span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Growth Rate:</span>
                        <span><strong>12.5%</strong></span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Market Penetration:</span>
                        <span><strong>0.8%</strong></span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Competitive Density:</span>
                        <span><strong>Medium</strong></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation timeline
        status_text.text("⏰ Creating timeline...")
        progress_bar.progress(80)
        
        st.markdown("### 📅 Implementation Timeline")
        timeline_chart = planner.create_timeline_chart(timeline)
        st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Risk analysis
        st.markdown("### ⚠️ Risk Assessment Matrix")
        risk_chart = planner.create_risk_matrix()
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Comprehensive analysis sections
        status_text.text("📋 Generating comprehensive analysis...")
        progress_bar.progress(90)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Marketing Strategy", "⚖️ Legal Requirements", "🔧 Operations", "💡 Recommendations"])
        
        with tab1:
            marketing_prompt = f"""
            Create a detailed marketing strategy for "{selected_name}", 
            targeting {target_audience} in {target_location}.
            Budget: ${budget}. Business model: {business_model}.
            
            Include digital marketing, traditional approaches, and KPIs.
            """
            marketing_strategy = planner.call_gemini(marketing_prompt)
            st.markdown(f"""
            <div class="metric-card">
                {marketing_strategy}
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            legal_prompt = f"""
            Outline legal requirements for starting a {business_type} business 
            in {target_location}. Include licenses, permits, business structure, 
            taxes, and compliance requirements.
            """
            legal_requirements = planner.call_gemini(legal_prompt)
            st.markdown(f"""
            <div class="metric-card">
                {legal_requirements}
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            operations_prompt = f"""
            Develop a comprehensive operational plan for "{selected_name}".
            Include staffing, technology, processes, and scalability.
            Timeline: {timeline}, Budget: ${budget}
            """
            operational_plan = planner.call_gemini(operations_prompt)
            st.markdown(f"""
            <div class="metric-card">
                {operational_plan}
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            recommendations_prompt = f"""
            Provide strategic recommendations and next steps for "{selected_name}".
            Focus on critical success factors, potential pitfalls to avoid, 
            and key milestones. Be specific and actionable.
            """
            recommendations = planner.call_gemini(recommendations_prompt)
            st.markdown(f"""
            <div class="metric-card">
                {recommendations}
            </div>
            """, unsafe_allow_html=True)
        
        # Completion
        progress_bar.progress(100)
        status_text.text("✅ Business plan generated successfully!")
        
        st.markdown("""
        <div class="success-message">
            🎉 Your premium business plan has been generated successfully! 
            Review all sections and download the complete report.
        </div>
        """, unsafe_allow_html=True)
        
        # Download button (placeholder)
        if st.button("📥 Download Complete Business Plan", type="primary", use_container_width=True):
            st.balloons()
            st.success("Business plan download initiated! (Feature coming soon)")

    else:
        # Welcome screen with features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3>📊 Advanced Analytics</h3>
                <p>Comprehensive financial projections, market analysis, and risk assessments with interactive visualizations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3>🤖 AI-Powered Insights</h3>
                <p>Leverage advanced AI to generate personalized business strategies and actionable recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3>📈 Professional Reports</h3>
                <p>Export publication-ready business plans suitable for investors, lenders, and stakeholders.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <h3 style="color: #2D3748;">Ready to transform your business idea into reality?</h3>
            <p style="color: #718096; font-size: 1.1rem;">Fill in your business details in the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
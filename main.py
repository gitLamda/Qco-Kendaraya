import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Custom styling and page configuration
st.set_page_config(
    page_title="QCO කේන්දරය",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Remove default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Background image */
    .stApp {
        background-image: url("https://cdn.pixabay.com/photo/2013/03/24/11/10/horoscope-96309_1280.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Semi-transparent overlay to improve text readability */
    .main, .block-container {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.7), rgba(59, 130, 246, 0.7)) !important, 
                url('https://www.transparenttextures.com/patterns/stardust.png');
    background-blend-mode: overlay;
    background-color: rgba(30, 58, 138, 0.7); /* More transparency */
    border-radius: 20px;
    padding: 20px;
    margin-top: 10px;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.4); /* Soft glow */
}


    .chart-title {
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
        color: white !important;  # Changed from #2e7031 to white
    }

    /* Main headline styling */
    .main-headline {
        text-align: center;
        font-size: 3rem;
        font-style: italic;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #333;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
    }

    /* Form label styling */
    .bold-italic-label {
    color: white !important;
    font-weight: bold !important;
    font-style: italic !important;
    font-size: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}

    /* Company logo styling */
    .company-logo {
        position: absolute;
        top: 20px;
        left: 20px;
        border-radius: 15px;
        max-width: 150px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    /* File uploader styling */
    .stFileUploader>div>div {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }

    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }

    /* Container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Make headers and text stand out against background */
    h2, h3, p, label {
        text-shadow: 0px 0px 3px rgba(255, 255, 255, 0.8);
    }

    /* Card styling for predictions */
    .prediction-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Target the specific checkbox labels */
    div[data-testid="stHorizontalBlock"] label p {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }

    /* Hover effects */
    div[data-testid="stHorizontalBlock"] label:hover p {
        text-shadow: 0 0 8px rgba(255,255,255,0.8);
    }

    /* Model selection styling */
    .model-selector {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Scrollable container for results */
    .scrollable-results {
        max-height: 800px;
        overflow-y: auto;
        padding-right: 10px;
    }

    /* Enhance visualization containers */
    .visualization-container {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    /* Section dividers */
    .section-divider {
        border-top: 2px solid rgba(76, 175, 80, 0.3);
        margin: 25px 0;
    }

    /* Chart title styling */
    .chart-title {
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
        color: #2e7031;
    }
</style>
""", unsafe_allow_html=True)


# Define custom class if needed for unpickling
class GroupRareSilhouette:
    def __init__(self):
        pass

    def transform(self, X):
        return X


# Cached model loading function
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Load models
try:
    model1 = load_model("new_rf_qco.pkl")  # Historia - Classification model
    model2 = load_model("lr_qco.pkl")  # Critical Path - Regression model
    model3 = load_model("qco_predictor.pkl")  # Talento - Classification
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please ensure model files are in the same directory.")
    st.stop()

# Features for models
model1_features = ['Priority', 'Tier', 'Module Repeatability', 'Efficiency', 'Module Achievement']
model2_features = [
    'Do-ability  Sample complete by Technician',
    'Focus training Plan with 4Ms,Focus Training 70% TM Count',
    'Team Member Allocation for Layout', 'Floater allocation for QCO',
    'Critical /M/C Pre-setup(done by Mech, check by GL, check by QC)',
    'M/C, Layout and space allocation on time for QCO',
    'Feeding Plan Ready', 'STW Sheet Handover On time',
    'Standard video sharing',
    'Cut-Kit received by 4.30 pm/Checked cut Panels with  Patterns',
    'Mechanic on time attend  - 7.30am,M/C Setting start on time - 7.30am,GL attend on time - 7.30am',
    'Sample Done By GL on plan time',
    'TM Training Start on plan time,TM training complete with mockups',
    "All operation's  mockups Verify by QC",
    'All Work Place Arranged & defined', 'One Hour Production',
    '1st 10 PCS Review ', 'Yamazumi Done by IE ',
    "TM's 70% potential Efficiency Availability", 'Changeover Quality FTT',
    'Module Machine movement on time'
]
model3_features = [
    'Priority',
    'Skill'
]


# Create enhanced gauge chart
def create_gauge_chart(value, title, color="darkblue", height=300):
    # Define color gradient based on value
    if value < 50:
        bar_color = "firebrick"
    elif value < 75:
        bar_color = "orange"
    else:
        bar_color = "forestgreen"

    # Create the gauge chart with white bold percentage
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'valueformat': '.1f',
            'suffix': '%',
            'font': {
                'size': 28,  # Increased size
                'color': 'white',  # White color
                'family': "Arial Black"  # Bold font
            }
        },
        title={
            'text': title,
            'font': {
                'size': 16,
                'color': '#333',
                'family': 'Arial'
            }
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 25], 'color': "#ffcccc"},
                {'range': [25, 50], 'color': "#ffebcc"},
                {'range': [50, 75], 'color': "#e6ffcc"},
                {'range': [75, 100], 'color': "#ccffcc"}
            ],
        }
    ))

    # Layout adjustments for better contrast
    fig.update_layout(
        height=height,
        margin=dict(t=50, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')  # Ensures all text is white by default
    )

    return fig


# Create comparison chart
def create_comparison_chart(values, titles, colors):
    fig = go.Figure()

    for i, (value, title, color) in enumerate(zip(values, titles, colors)):
        fig.add_trace(go.Bar(
            x=[title],
            y=[value],
            name=title,
            marker_color=color,
            text=f"{value:.1f}%",
            textposition='auto',
            textfont=dict(color='white', size=14)  # White text on bars
        ))

    fig.update_layout(
        title={
            'text': "Model Comparison",
            'font': {
                'size': 18,
                'color': 'white'  # White title
            }
        },
        height=400,
        yaxis=dict(
            range=[0, 100],
            title=dict(
                text="Hit Rate (%)",
                font=dict(color='white')
            ),
            tickfont=dict(color='white')   # <— y‑axis tick labels
    ),
    xaxis=dict(
        tickfont=dict(color='white')   # <— x‑axis tick labels
    ),
        legend=dict(
            font=dict(color='white')  # White legend text
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# Radar chart for model impact
def create_radar_chart(values, titles):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=titles,
        fill='toself',
        fillcolor='rgba(76, 175, 80, 0.3)',
        line=dict(color='rgb(76, 175, 80)', width=2),
        name="Model Impact"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            )
        ),
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_comparison_bar(weighted_value, unweighted_value):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Weighted Prediction'],
        y=[weighted_value],
        name='Weighted Prediction',
        marker_color='#673ab7',
        text=[f'{weighted_value:.1f}%'],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=['Theoretical Maximum'],
        y=[unweighted_value],
        name='Potential Maximum',
        marker_color='#9575cd',
        text=[f'{unweighted_value:.1f}%'],
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(
            text="Weighted vs Theoretical Maximum Comparison",
            font=dict(color='white', size=16)
    ),
    yaxis=dict(
        range=[0, 100],
        title=dict(
            text="Prediction Percentage",
            font=dict(color='white')
        ),
        tickfont=dict(color='white')
    ),
    xaxis=dict(
        tickfont=dict(color='white')
    ),
    legend=dict(
        font=dict(color='white')
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400
)
return fig


# Main app
def main():
    # Main heading
    st.markdown(
        '''
        <h1 class="main-headline">
        <span class="lion-icon" style="
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
            width: 100px;
            height: 100px;
            background-image: url('https://cdn.pixabay.com/photo/2016/09/05/17/40/zodiac-1647168_1280.jpg');
            background-size: contain;
            background-repeat: no-repeat;
            filter: drop-shadow(0 0 5px gold) brightness(1.1);
            border-radius: 50%;
        "></span>
            <span style="color: white; font-weight: bold; font-size: 1.8em; margin-right: 8px;">QCO</span> 
            <span class="sinhala-glow" style="
                font-family: 'Arial', sans-serif;
                color: #ff9d00;
                font-size: 1.2em;
                display: inline-block;
                transform: rotate(-5deg);
                position: relative;
            ">
                කේන්දරය
                <div class="coin-shower">
                    <span class="coin">💰</span>
                    <span class="coin">💵</span>
                    <span class="coin">🪙</span>
                    <span class="coin">💰</span>
                    <span class="coin">💵</span>
                </div>
            </span>
        </h1>

        <style>
            @keyframes glow {
                0% {
                    text-shadow: 0 0 10px #ff6600;
                }
                50% {
                    text-shadow: 0 0 20px #ff3300, 0 0 30px #ff0066;
                }
                100% {
                    text-shadow: 0 0 15px #ffcc00, 0 0 25px #ff9933;
                }
            }

            @keyframes shower {
                0% {
                    transform: translateY(-100%) rotate(0deg);
                    opacity: 0;
                }
                50% {
                    opacity: 1;
                }
                100% {
                    transform: translateY(100%) rotate(360deg);
                    opacity: 0;
                }
            }

            .coin-shower {
                position: absolute;
                top: -50px;
                left: -20px;
                right: -20px;
                bottom: -50px;
                pointer-events: none;
            }

            .coin {
                position: absolute;
                font-size: 0.8em;
                animation: shower 3s linear infinite;
                color: gold;
                opacity: 0;
            }

            .coin:nth-child(1) { left: 10%; animation-delay: 0s; }
            .coin:nth-child(2) { left: 30%; animation-delay: 0.5s; }
            .coin:nth-child(3) { left: 50%; animation-delay: 1s; }
            .coin:nth-child(4) { left: 70%; animation-delay: 1.5s; }
            .coin:nth-child(5) { left: 90%; animation-delay: 2s; }

            .sinhala-glow {
                animation: glow 2s ease-in-out infinite alternate;
            }

            .main-headline {
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: visible;
            }

            .main-headline:hover .sinhala-glow {
                transform: rotate(-5deg) scale(1.05);
                transition: all 0.3s ease;
            }

            .main-headline:hover span:first-child {
                transform: rotate(10deg) scale(1.2);
                transition: all 0.3s ease;
            }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Input section
    input_container = st.container()
    with input_container:
        st.markdown('<p class="bold-italic-label">Upload Excel File</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Upload Excel File",
            type=["xlsx", "xls"],
            label_visibility="collapsed",
            help="Please upload an Excel file containing your QCO data"
        )

        # Input columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="bold-italic-label">Module Number</p>', unsafe_allow_html=True)

            # Initialize session state for storing uploaded data
            if 'uploaded_data' not in st.session_state:
                st.session_state.uploaded_data = None
                st.session_state.modules = []
                st.session_state.styles = []
                st.session_state.module_style_map = {}

            # Process uploaded file when changed
            if uploaded_file and st.session_state.uploaded_data != uploaded_file:
                df = pd.read_excel(uploaded_file)
                df['Module Number'] = df['Module Number'].astype(str).str.strip().str.lower()
                df['Style Number'] = df['Style Number'].astype(str).str.strip().str.lower()

                # Create module-style mapping
                st.session_state.module_style_map = df.groupby('Module Number')['Style Number'] \
                    .unique().apply(list).to_dict()

                st.session_state.modules = list(st.session_state.module_style_map.keys())
                st.session_state.styles = []
                st.session_state.uploaded_data = uploaded_file

            # Module dropdown
            module_input = st.selectbox(
                "Module Number",
                options=st.session_state.modules,
                index=0 if st.session_state.modules else None,
                format_func=lambda x: x.upper(),
                help="Select module number from uploaded file",
                label_visibility="collapsed"
            )

        with col2:
            st.markdown('<p class="bold-italic-label">Style Number</p>', unsafe_allow_html=True)

            # Update styles based on selected module
            if module_input and st.session_state.module_style_map:
                style_options = st.session_state.module_style_map.get(module_input, [])
            else:
                style_options = []

            style_input = st.selectbox(
                "Style Number",
                options=style_options,
                index=0 if style_options else None,
                format_func=lambda x: x.upper(),
                help="Select style number for chosen module",
                label_visibility="collapsed"
            )

        # Model selection with improved UI
        st.markdown('<p class="bold-italic-label">Select Prediction Model(s)</p>', unsafe_allow_html=True)
        model_container = st.container()
        with model_container:
            col_model1, col_model2, col_model3 = st.columns(3)

            with col_model1:
                st.markdown('<div class="model-checkbox">', unsafe_allow_html=True)
                use_model1 = st.checkbox("Historia (30%)", value=True,  # Updated weight
                                         help="Classification model based on historical data")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_model2:
                st.markdown('<div class="model-checkbox">', unsafe_allow_html=True)
                use_model2 = st.checkbox("Critical Path (60%)", value=True,  # Updated weight
                                         help="Regression model focused on process factors")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_model3:
                st.markdown('<div class="model-checkbox">', unsafe_allow_html=True)
                use_model3 = st.checkbox("Talento (10%)", value=True,
                                         help="Classification model for talent factors")
                st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        predict_button = st.button("🔮Predict!", use_container_width=True)

    # Divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Results container with scrolling
    results_container = st.container()

    # Prediction logic
    if predict_button:
        # Validate inputs
        if uploaded_file is None:
            st.error("⚠️ Please upload an Excel file!")
            return
        if not module_input or not style_input:
            st.error("⚠️ Please enter both Module and Style numbers!")
            return
        if not (use_model1 or use_model2 or use_model3):
            st.error("⚠️ Please select at least one model!")
            return

        # Read and process data
        try:
            df = pd.read_excel(uploaded_file)

            # Clean up the columns
            df['Module Number'] = df['Module Number'].astype(str).str.strip().str.lower()
            df['Style Number'] = df['Style Number'].astype(str).str.strip().str.lower()

            module_input = str(module_input).strip().lower()
            style_input = str(style_input).strip().lower()

            # Now match like a champ
            row = df[(df['Module Number'] == module_input) & (df['Style Number'] == style_input)]

            if row.empty:
                st.error("⚠️ No matching record found for the provided Module and Style numbers.")
                return

            row = row.iloc[0]

            # Store prediction results
            results = {}
            model_colors = {"Historia": "royalblue", "Critical Path": "green", "Talento": "darkred"}

            # Success message
            with results_container:
                st.markdown(f"""
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0;">✅ Prediction for Module {module_input}, Style {style_input}</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create tabs for different view options
                tab1, tab2 = st.tabs(["📊 Individual Models", "🌟 Combined Analysis"])

                with tab1:
                    # Scrollable area for individual model results
                    st.markdown('<div class="scrollable-results">', unsafe_allow_html=True)

                    # Model 1 Prediction (Historia)
                    if use_model1:
                        st.markdown("""
                        <div class="visualization-container">
                            <h3 style="color: royalblue; margin-top: 0;">🔍 Historia (Classification Model)</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        input_data1 = pd.DataFrame({feat: [row[feat]] for feat in model1_features})
                        proba = model1.predict_proba(input_data1)[0]
                        hit_prob = proba[1] * 100
                        results["Historia"] = hit_prob

                        col1, col2 = st.columns([3, 2])

                        with col1:
                            fig1 = create_gauge_chart(hit_prob, "", "royalblue")
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            # Feature importance display
                            st.markdown('<div class="chart-title">Key Input Parameters</div>', unsafe_allow_html=True)
                            feature_data = pd.DataFrame({
                                'Feature': model1_features,
                                'Value': [str(row[feat]) for feat in model1_features]  # Convert to string
                            })
                            st.dataframe(feature_data.astype(str), hide_index=True, use_container_width=True)

                            # Overall impact
                            impact = hit_prob * 0.3
                            st.markdown(f"""
                            <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <p style="margin: 0; font-weight: bold;">Impact on Final Score: {impact:.1f}%</p>
                                <p style="margin: 0; font-size: 0.9rem; color: #666;">
                                    (30% of total weight)
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Add divider if multiple models are selected
                    if use_model1 and (use_model2 or use_model3):
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # Model 2 Prediction (Critical Path)
                    if use_model2:
                        st.markdown("""
                        <div class="visualization-container">
                            <h3 style="color: green; margin-top: 0;">📊 Critical Path (Regression Model)</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        input_data2 = pd.DataFrame({feat: [row[feat]] for feat in model2_features})
                        hit_rate = model2.predict(input_data2)[0]
                        results["Critical Path"] = hit_rate

                        col1, col2 = st.columns([3, 2])

                        with col1:
                            fig2 = create_gauge_chart(hit_rate, "", "green")
                            st.plotly_chart(fig2, use_container_width=True)

                        with col2:
                            st.markdown('<div class="chart-title">Incomplete Critical Activities</div>',
                                        unsafe_allow_html=True)

                            # Create a dataframe for model2 features without converting to string right away
                            critical_features = pd.DataFrame({
                                'Factor': model2_features,
                                'Value': [row[feat] for feat in model2_features]
                            })

                            # Filter to find activities that are not completed (value == 0)
                            incomplete_activities = critical_features[critical_features['Value'] == 0]

                            # Check if there are any incomplete activities
                            if incomplete_activities.empty:
                                st.markdown(
                                    "<p style='color: green; font-weight: bold;'>All activities are completed!</p>",
                                    unsafe_allow_html=True)
                            else:
                                st.dataframe(incomplete_activities, hide_index=True, use_container_width=True)

                            # Overall impact
                            impact = hit_rate * 0.6
                            st.markdown(f"""
                            <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <p style="margin: 0; font-weight: bold;">Impact on Final Score: {impact:.1f}%</p>
                                <p style="margin: 0; font-size: 0.9rem; color: #666;">
                                    (60% of total weight)
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Add divider if model 3 is also selected
                    if (use_model1 or use_model2) and use_model3:
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # Model 3 Prediction (Talento)
                    if use_model3:
                        st.markdown("""
                        <div class="visualization-container">
                            <h3 style="color: darkred; margin-top: 0;">👥 Talento (Classification Model)</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        input_data3 = pd.DataFrame({feat: [row[feat]] for feat in model3_features})
                        proba3 = model3.predict_proba(input_data3)[0]
                        talent_prob = proba3[1] * 100
                        results["Talento"] = talent_prob

                        col1, col2 = st.columns([3, 2])

                        with col1:
                            fig3 = create_gauge_chart(talent_prob, "", "darkred")
                            st.plotly_chart(fig3, use_container_width=True)

                        with col2:
                            # Talent factors
                            st.markdown('<div class="chart-title">Team Factors</div>', unsafe_allow_html=True)

                            talent_data = pd.DataFrame({
                                'Factor': model3_features,
                                'Value': [str(row[feat]) for feat in model3_features]  # Convert to string
                            })
                            st.dataframe(talent_data.astype(str), hide_index=True, use_container_width=True)

                            # Overall impact
                            impact = talent_prob * 0.1
                            st.markdown(f"""
                            <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <p style="margin: 0; font-weight: bold;">Impact on Final Score: {impact:.1f}%</p>
                                <p style="margin: 0; font-size: 0.9rem; color: #666;">
                                    (10% of total weight)
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)  # Close scrollable div

                with tab2:
                    # Combined Analysis tab
                    models_used = []
                    values = []
                    colors = []
                    impacts = []
                    impact_titles = []

                    final_prediction = 0

                    if use_model1:
                        models_used.append("Historia")
                        values.append(results["Historia"])
                        colors.append("royalblue")
                        impact = results["Historia"] * 0.3
                        impacts.append(impact)
                        impact_titles.append("Historia Impact")
                        final_prediction += impact

                    if use_model2:
                        models_used.append("Critical Path")
                        values.append(results["Critical Path"])
                        colors.append("green")
                        impact = results["Critical Path"] * 0.6
                        impacts.append(impact)
                        impact_titles.append("Critical Path Impact")
                        final_prediction += impact

                    if use_model3:
                        models_used.append("Talento")
                        values.append(results["Talento"])
                        colors.append("darkred")
                        impact = results["Talento"] * 0.1
                        impacts.append(impact)
                        impact_titles.append("Talento Impact")
                        final_prediction += impact

                        # Calculate unweighted average
                        unweighted_average = sum(results.values()) / len(results) if results else 0

                    # Display combined results
                    st.markdown("""
                        <div class="visualization-container">
                            <h3 style="color: purple; text-align: center; margin-top: 0;">🌟 Combined Analysis</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    # Final prediction gauge
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        final_fig = create_gauge_chart(
                            final_prediction,
                            "",
                            "purple",
                            height=400
                        )
                        st.plotly_chart(final_fig, use_container_width=True)

                        # Decision guidance
                        if final_prediction >= 85:
                            recommendation = "✅ High probability of success - Proceed with confidence"
                            color = "#28a745"
                        elif final_prediction >= 70:
                            recommendation = "🟡 Good probability - Monitor key factors"
                            color = "#ffc107"
                        elif final_prediction >= 50:
                            recommendation = "⚠️ Moderate risk - Address critical gaps"
                            color = "#fd7e14"
                        else:
                            recommendation = "❌ High risk - Focus more on the gaps"
                            color = "#dc3545"

                        st.markdown(f"""
                        <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px; margin-top: 20px; text-align: center;">
                            <h3 style="margin: 0;">Recommendation</h3>
                            <p style="margin: 10px 0 0 0; font-size: 1.2rem;">{recommendation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        comparison_bar = create_comparison_bar(final_prediction, unweighted_average)
                        st.plotly_chart(comparison_bar, use_container_width=True)

                        # Model comparison chart
                        if len(models_used) > 1:
                            comparison_fig = create_comparison_chart(values, models_used, colors)
                            st.plotly_chart(comparison_fig, use_container_width=True)

                            # Impact radar chart
                            radar_fig = create_radar_chart(impacts, impact_titles)
                            st.markdown('<div class="chart-title">Model Impact Analysis</div>', unsafe_allow_html=True)
                            st.plotly_chart(radar_fig, use_container_width=True)

                    # Detailed breakdown
                    st.markdown('<div class="chart-title">Detailed Score Breakdown</div>', unsafe_allow_html=True)

                    breakdown_data = []
                    for model, value in results.items():
                        weight = 0.3 if model == "Historia" else 0.6 if model == "Critical Path" else 0.1
                        impact = value * weight
                        breakdown_data.append({
                            "Model": model,
                            "Raw Score": f"{value:.1f}%",
                            "Weight": f"{weight * 100:.0f}%",
                            "Weighted Impact": f"{impact:.1f}%",
                            "Contribution": f"{(impact / final_prediction * 100):.1f}%"
                        })

                    breakdown_df = pd.DataFrame(breakdown_data)
                    new_row = pd.DataFrame({
                        "Model": "FINAL SCORE",
                        "Raw Score": "-",
                        "Weight": "100%",
                        "Weighted Impact": f"{final_prediction:.1f}%",
                        "Contribution": "100.0%"
                    }, index=[0])
                    breakdown_df = pd.concat([breakdown_df, new_row], ignore_index=True)

                    st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            import traceback
            st.exception(traceback.format_exc())


if __name__ == "__main__":
    main()

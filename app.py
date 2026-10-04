import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime

# Load model and features
model = joblib.load("recommendation_model.pkl")
model_features = joblib.load("model_features.pkl")

# Purpose mapping
purpose_map = {
    'Work': 'Commute', 'Part-time Work': 'Commute', 'University': 'Commute', 'Gym': 'Commute',
    'School': 'Education',
    'Grocery Store': 'Errand', 'Shopping Centre': 'Errand', 'Library': 'Errand', 'Pharmacy': 'Errand',
    'Hospital': 'Health',
    'Cricket Ground': 'Leisure', 'Museum': 'Leisure', 'Park': 'Leisure',
    "Friend's Place": 'Leisure', 'Stadium': 'Leisure', 'Cinema': 'Leisure',
    'Cafe': 'Leisure', 'Beach': 'Leisure',
    'Train Station': 'Transit', 'Airport': 'Travel',
    "Grandparents' House": 'Family'
}

co2_per_km = {
    "Walk": 0, "Bicycle": 0,
    "Car": 192, "Bus": 105,
    "Train": 41, "Tram": 60,
    "Flight": 150
}

# UI Settings
st.set_page_config(page_title="CarbonWise: Personal Environmental Impact Tracker", layout="wide")
st.title("CarbonWise: Personal Environmental Impact Tracker")

st.markdown("""
Built to better understand the environmental impact of everyday travel, this project started as a way to reflect on how much carbon gets emitted just by getting from place to place. I originally planned to use my Google Maps Takeout data to analyse this—but turns out my location history was off the whole time :( So I came up with the idea of simulating my entire travel history instead. I mapped out 23 years of my life, from childhood (mostly home and family outings), to school years (hello daily bus rides), uni life, part-time work, and everything in between.

The dashboard shows total CO₂ emissions, total distance travelled, breaks down emissions by transport mode (with flights shown separately), even tracks your monthly emissions against a personal goal, and points out greener travel alternatives with estimated carbon savings. Behind the scenes, there’s a simple decision tree model that figures out when a more sustainable mode (like walking, cycling, or public transport) makes sense—based not just on distance, but also on the purpose of the trip and whether switching would actually reduce emissions in a meaningful way.

There's also a custom input section where anyone can add their own trips and instantly get the same kind of insights and suggestions based on their data. Basically, it’s a tool to help reflect on how we move through the world—and how we could do it more sustainably.
""")

st.markdown("Feel free to download a sample travel log below to see how the dashboard works with real-ish data.")

# Sample data download button moved here, just after intro paragraph
try:
    df_sample = pd.read_csv("travel_data.csv").sample(n=50)
    st.download_button("Download Sample Data", df_sample.to_csv(index=False), "sample_travel_data.csv")
except Exception as e:
    st.warning("Sample data not available for download.")

# Function to enrich and predict
def enrich_data(df):
    df['Purpose'] = df['To'].map(purpose_map).fillna("Other")
    df['Weekday'] = pd.to_datetime(df['Date']).dt.day_name()
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M').astype(str)

    input_data = pd.get_dummies(df[['Distance_km', 'Duration_min', 'Purpose']])
    for col in model_features:
        if col not in input_data:
            input_data[col] = 0
    input_data = input_data[model_features]

    df['Predicted_Mode'] = model.predict(input_data)
    df['Predicted_Emission_g'] = df.apply(
        lambda row: round(row['Distance_km'] * co2_per_km.get(row['Predicted_Mode'], 100), 1), axis=1)
    df['CO2_Saved_g'] = df['CO2_emission_g'] - df['Predicted_Emission_g']
    df['CO2_Saved_g'] = df['CO2_Saved_g'].apply(lambda x: max(x, 0))
    return df

# Dashboard renderer
def render_dashboard(df, co2_goal):
    df = enrich_data(df)

    st.header("Environmental Impact Overview")
    st.caption("Here’s a quick summary of the travel distance, how much CO₂ that added up to, and how many greener swaps were possible.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total CO₂ Emissions", f"{int(df['CO2_emission_g'].sum()):,} g")
    col2.metric("Total Distance Travelled", f"{df['Distance_km'].sum():.1f} km")
    col3.metric("Greener Alternatives Available", f"{(df['Mode'] != df['Predicted_Mode']).sum()} trips")

    st.markdown("### Emissions by Transport Mode")
    st.caption("This breaks down how much each mode contributed to the total emissions (excluding flights).")
    non_flight = df[df['Mode'] != 'Flight']
    fig = px.bar(non_flight.groupby('Mode')['CO2_emission_g'].sum().reset_index(),
                 x='Mode', y='CO2_emission_g',
                 labels={'CO2_emission_g': 'Carbon emitted (grams)'},
                 color='Mode', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Flight Emissions")
    st.caption("Flying packs a bigger punch when it comes to carbon—here’s how flights stacked up.")
    flights = df[df['Mode'] == 'Flight']
    if not flights.empty:
        fig = px.bar(flights.groupby('To')['CO2_emission_g'].sum().reset_index(),
                     x='To', y='CO2_emission_g', color='To',
                     labels={'CO2_emission_g': 'Carbon emitted (grams)'},
                     color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Looks like you didn’t fly anywhere during this time frame.")

    st.markdown("### Monthly Emissions vs Carbon Emissions Goal (Excluding Flights)")
    st.caption("See how the emissions change month by month, and whether they stayed under the goal set.")
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M').astype(str)
    monthly = df[df['Mode'] != 'Flight'].groupby('Month').agg({'CO2_emission_g': 'sum'}).reset_index()
    monthly['Goal'] = co2_goal
    monthly['Exceeded'] = monthly['CO2_emission_g'] > monthly['Goal']
    monthly['Opacity'] = monthly['CO2_emission_g'].apply(lambda x: 0.4 if x < co2_goal * 0.5 else 1.0)
    fig = px.bar(monthly, x='Month', y='CO2_emission_g', color='Exceeded',
                 labels={'CO2_emission_g': 'Carbon emitted (grams)'},
                 color_discrete_map={True: 'crimson', False: 'seagreen'},
                 title="Monthly CO₂ Emissions vs Goal",
                 opacity=monthly['Opacity'])
    fig.add_scatter(x=monthly['Month'], y=monthly['Goal'], name='Goal', mode='lines+markers',
                    line=dict(dash='dash', width=3, color='white'))
    fig.update_layout(yaxis=dict(range=[0, monthly['CO2_emission_g'].max() * 1.1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Greener Mode Recommendations")
    recs = df[df['Mode'] != df['Predicted_Mode']]
    if not recs.empty:
        counts = recs['Predicted_Mode'].value_counts().reset_index()
        counts.columns = ['Predicted_Mode', 'Count']
        counts['Label'] = counts['Predicted_Mode'] + ' (' + counts['Count'].astype(str) + ' trips)'
        fig = px.pie(counts, names='Label', values='Count',
                     title="What the model thinks I should’ve done instead",
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alternative mode suggestions.")

    st.markdown("### Carbon Savings Estimate")
    st.success(f"Potential Carbon savings: **{int(df['CO2_Saved_g'].sum()):,} g**")

    st.markdown("### Some interesting patterns I found")
    peak_days = df['Weekday'].value_counts().head(2).index.tolist()
    top_places = df.groupby('To')['CO2_emission_g'].sum().sort_values(ascending=False).head(2).index.tolist()
    st.markdown(f"•**Most frequent travel days**: {', '.join(peak_days)}")
    st.markdown(f"•**Top emission-heavy destinations**: {', '.join(top_places)}")

# Sidebar Goal
st.sidebar.header("Set a monthly carbon goal you'd like to aim for")
st.sidebar.caption("This is just a reference point—it helps track whether we're staying within a sustainable range each month.")
co2_goal = st.sidebar.slider("grams", 5000, 100000, 25000, 5000)

# Mode
mode = st.radio("What do you want to explore?", ["Look through my travel history", "Try it with your own trips"])

if mode == "Look through my travel history":
    df = pd.read_csv("travel_data.csv")
    render_dashboard(df, co2_goal)

else:
    st.subheader("Enter Your Trips")
    n = st.number_input("How many trips?", 1, 20, step=1)
    inputs = []
    for i in range(n):
        st.markdown(f"##### Trip {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input(f"Date", key=f"date_{i}")
        with col2:
            from_loc = st.text_input("From", key=f"from_{i}")
        with col3:
            to_loc = st.text_input("To", key=f"to_{i}")
        col4, col5, col6 = st.columns(3)
        with col4:
            mode_ = st.selectbox("Mode", list(co2_per_km.keys()), key=f"mode_{i}")
        with col5:
            dist = st.number_input("Distance (km)", 0.0, 10000.0, step=0.1, key=f"dist_{i}")
        with col6:
            dur = st.number_input("Duration (min)", 0.0, 10000.0, step=0.1, key=f"dur_{i}")
        co2 = round(dist * co2_per_km.get(mode_, 100), 1)
        inputs.append({
            "Date": date,
            "From": from_loc,
            "To": to_loc,
            "Mode": mode_,
            "Distance_km": dist,
            "Duration_min": dur,
            "CO2_emission_g": co2,
            "Trip_Type": "Local"
        })

    if st.button("Get insights from your trips"):
        user_df = pd.DataFrame(inputs)
        render_dashboard(user_df, co2_goal)

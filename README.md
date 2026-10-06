## CarbonWise: A Personal Environmental Impact Tracker
I built CarbonWise to explore the environmental impact of my everyday travel habits and see where I could realistically make more sustainable choices. I originally planned to use my Google Maps Takeout data, but unfortunately my location history had been turned off the whole time. So instead, I created synthetic travel data representing 23 years of my life, from childhood and school through university, part-time work, and everything in between. I then used this data to train a Decision Tree model that suggests greener alternatives based on things like distance, trip duration, and purpose. The project brings everything together in an interactive Streamlit dashboard where I can explore my travel history, track emissions against a personal monthly goal, see potential carbon savings, and even enter my own trips to get personalised recommendations.

### Running it locally

Clone the repository and navigate into the project:

git clone https://github.com/your-username/carbonwise.git

cd carbonwise

Install the required Python packages:

pip install streamlit pandas plotly scikit-learn joblib

Train the recommendation model:

python train_model.py

This will generate the model files:

recommendation_model.pkl, model_features.pkl

Then start the Streamlit dashboard:

streamlit run app.py

The app should open automatically in your browser.

Make sure travel_data.csv is in the project directory before running the dashboard.

### Why I built it

I wanted to build something that connected data, machine learning, and a real-world problem rather than just training a model for the sake of it.

I was particularly interested in the idea that sustainable travel isn't always as simple as saying "take public transport instead." A 1 km trip to the shops is very different from a 20 km commute, and the purpose of a trip can change what alternatives are actually realistic. That led me to build a recommendation system that considers the context of each trip rather than just looking at emissions in isolation.

The project is also quite personal. Since I couldn't access my actual location history, I ended up creating a simulated version of my own travel history and using it to ask a question I think is more interesting than simply calculating a carbon footprint:

Where could I realistically have made different choices?

### Next steps

There are a few things I'd like to explore next:

- Replace the synthetic data with real travel data if I can access it in the future.
- Let users upload their own CSV travel history instead of entering trips manually.
- Improve the emissions estimates by accounting for factors like vehicle type, occupancy and different public transport systems.
- Add maps and geographic visualisations to show travel patterns.
- Experiment with more advanced recommendation models and compare them with the current Decision Tree approach.
- Track changes in travel behaviour over time and make the recommendations more personalised.

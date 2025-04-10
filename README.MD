# Climate Change Impact Predictor

![Project Banner](app/assets/banner.png)

A machine learning application that predicts climate change impacts using satellite data from Google Earth Engine.

## Features

- 🌍 Interactive exploration of climate data from satellites
- 🤖 Machine learning models (Random Forest and Neural Networks) for impact prediction
- 📊 Dashboard for visualizing results and trends
- 🛰️ Direct integration with Google Earth Engine API

## Project Structure

The project is organized as follows:
- data/: Contains raw and processed datasets.
- models/: Saved machine learning models.
- src/: Source code for data processing, modeling and visualizations.
- app/: Streamlit application code.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/climate-change-impact-predictor.git
   cd climate-change-impact-predictor
2. Set up a virtual environment (recommended)
python -m venv venv
venv\scripts\activate
3. Install dependencies
   pip install -r requirements.txt
4. Set up Google Earth Engine authentication
- Create a Google Earth Engine account.
- run earthengine authenticate in your terminal and follow the instructions.
5. Run the application
   streamlit run app/main.py

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

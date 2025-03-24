# Penguin Classification Project

## Overview
This project implements a machine learning system to classify penguins into three different species (Adelie, Chinstrap, and Gentoo) based on their physical measurements. The system fetches new penguin data daily from an API, predicts the species using a trained Random Forest model, and stores the results in a SQLite database.
## Output for the project
https://vitwip.github.io/Penguins_Clasification_Adelie/

## Project Components

### 1. Data Collection
- Initial dataset loaded from the seaborn library's penguins dataset
- New penguin data fetched daily at 7:30 AM from the API endpoint
- Data includes measurements: bill length, bill depth, flipper length, and body mass

### 2. Database Structure
The project uses SQLite with two main tables:
- `penguins`: Stores penguin measurements and predicted species
  - `id`: Unique identifier
  - `species`: Predicted penguin species (Adelie, Chinstrap, or Gentoo)
  - `bill_length_mm`: Length of the penguin's bill in millimeters
  - `bill_depth_mm`: Depth of the penguin's bill in millimeters
  - `flipper_length_mm`: Length of the penguin's flipper in millimeters
  - `body_mass_g`: Mass of the penguin in grams
  - `is_original`: Flag to distinguish between initial dataset (0) and API-fetched data (1)
- `penguin_dates`: Stores timestamps for each penguin entry
  - `id`: References the penguin ID
  - `date`: Timestamp when the penguin data was collected

### 3. Model Training
The project implements and compares four different feature selection methods:
1. **Filter Method (Mutual Information)**: Evaluates features independently based on their relationship with the target variable
2. **Wrapper Method (Recursive Feature Elimination)**: Iteratively removes features and evaluates model performance
3. **Embedded Method (Random Forest Feature Importance)**: Uses the model's internal feature importance metrics
4. **Permutation Importance**: Measures feature importance by randomly shuffling feature values

The best-performing model is saved as `best_penguin_model.joblib` for future predictions.

### 4. Prediction System
The system provides two main prediction capabilities:
- **Automated predictions** for new penguin data fetched from the API
- **Interactive predictions** through a command-line interface where users can input measurements
- **Example predictions** to demonstrate the model's performance on representative samples

### 5. Web Interface
The project includes a GitHub Pages website that displays:
- The latest penguin prediction
- Historical prediction data
- Statistics about the predictions and model performance

## Workflow

### Initialization Process
1. Sets up logging with date-based file output in the `logs` directory
2. Checks if the database exists; if not, creates and populates it with initial data
3. Verifies the trained model exists; if not, trains a new model using the training module
4. Updates the database with new penguin data from the API

### Daily Automated Workflow
A GitHub Action is configured to run daily at 7:30 AM to:
1. Fetch new penguin data from the API
2. Predict the species using the trained model
3. Store the prediction in the database
4. Update the GitHub Pages website with the new prediction

## Technical Details

### Dependencies
The project requires the following Python packages:
- pandas, numpy: For data manipulation
- scikit-learn: For machine learning algorithms and evaluation
- matplotlib: For visualization
- seaborn: For the initial dataset
- sqlite3: For database operations
- requests: For API communication
- joblib: For model serialization

### File Structure
- `main.py`: Entry point that orchestrates the workflow
- `create_database.py`: Creates and populates the SQLite database
- `penguins_training_model.py`: Implements feature selection and model training
- `predict_penguin.py`: Provides prediction functionality
- `update_penguins.py`: Fetches new data from the API and updates the database
- `generate_site_data.py`: Creates data files for the GitHub Pages website
- `.github/workflows/`: Contains GitHub Actions configuration files
- `docs/`: Contains the GitHub Pages website files

### Running the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`
3. For interactive predictions: `python predict_penguin.py`
4. For example predictions: `python predict_penguin.py --examples`

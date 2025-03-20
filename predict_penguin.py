import sys
import pandas as pd
import numpy as np
import joblib

def predict_species(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    # Load the saved model
    loaded_model_data = joblib.load('best_penguin_model.joblib')
    model = loaded_model_data['model']
    scaler = loaded_model_data['scaler']
    features = loaded_model_data['features']
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Select only the features used by the model
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
    feature_mask = np.zeros(len(input_data.columns), dtype=bool)
    for feature in features:
        feature_idx = list(input_data.columns).index(feature)
        feature_mask[feature_idx] = True
    input_selected = input_scaled_df.iloc[:, feature_mask]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(input_selected)[0]
    prediction = model.predict(input_selected)[0]
    confidence = float(probabilities[model.classes_.tolist().index(prediction)])
    
    return prediction, confidence

def main():
    print("\n===== Penguin Species Predictor =====\n")
    print("This program predicts the species of a penguin based on its physical measurements.")
    print("Please enter the following measurements:\n")
    
    try:
        # Get user input for penguin measurements
        bill_length = float(input("Bill length (mm): "))
        bill_depth = float(input("Bill depth (mm): "))
        flipper_length = float(input("Flipper length (mm): "))
        body_mass = float(input("Body mass (g): "))
        
        # Make prediction
        species, confidence = predict_species(bill_length, bill_depth, flipper_length, body_mass)
        
        # Display result with confidence
        print(f"\nPredicted penguin species: {species} (Confidence: {confidence:.2%})")
        
        # Provide some context about the species
        if species == "Adelie":
            print("\nAdelie penguins are known for their small size and distinctive white eye ring.")
        elif species == "Chinstrap":
            print("\nChinstrap penguins have a distinctive black line under their chin, resembling a helmet strap.")
        elif species == "Gentoo":
            print("\nGentoo penguins are the third largest penguin species and have a bright orange-red bill.")
            
    except ValueError:
        print("\nError: Please enter valid numeric values for all measurements.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

def run_examples():
    print("\n===== Example Predictions =====\n")
    
    examples = [
        # Adelie example
        {"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181.0, "body_mass_g": 3750.0},
        # Chinstrap example
        {"bill_length_mm": 49.5, "bill_depth_mm": 19.0, "flipper_length_mm": 200.0, "body_mass_g": 3800.0},
        # Gentoo example
        {"bill_length_mm": 46.2, "bill_depth_mm": 14.5, "flipper_length_mm": 217.0, "body_mass_g": 4725.0},
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"  Bill length: {example['bill_length_mm']} mm")
        print(f"  Bill depth: {example['bill_depth_mm']} mm")
        print(f"  Flipper length: {example['flipper_length_mm']} mm")
        print(f"  Body mass: {example['body_mass_g']} g")
        
        species, confidence = predict_species(
            example['bill_length_mm'],
            example['bill_depth_mm'],
            example['flipper_length_mm'],
            example['body_mass_g']
        )
        
        print(f"  Predicted species: {species} (Confidence: {confidence:.2%})\n")

if __name__ == "__main__":
    # Check if user wants to run examples
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        run_examples()
    else:
        main()
        print("\nTip: Run with '--examples' to see example predictions.")
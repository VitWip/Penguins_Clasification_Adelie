#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Penguin Species Prediction Module.

This module provides functionality to predict penguin species based on physical measurements.
It uses a pre-trained machine learning model to classify penguins into their respective species
(Adelie, Chinstrap or Gentoo) using measurements like bill length, bill depth, flipper length,
and body mass.

The module includes functions for both single predictions and batch example predictions,
making it suitable for both interactive use and automated processing.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging

def predict_species(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """Predict the species of a penguin based on its physical measurements.
    
    Args:
        bill_length_mm (float): Length of the penguin's bill in millimeters
        bill_depth_mm (float): Depth of the penguin's bill in millimeters
        flipper_length_mm (float): Length of the penguin's flipper in millimeters
        body_mass_g (float): Mass of the penguin in grams
    
    Returns:
        tuple: A tuple containing:
            - str: Predicted species name (Adelie, Chinstrap, or Gentoo)
            - float: Confidence score of the prediction (0-1)
    """
    # Load the saved model
    logging.info("Loading penguin classification model...")
    model_path = os.path.join(os.path.dirname(__file__), 'best_penguin_model.joblib')
    loaded_model_data = joblib.load(model_path)
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
    logging.info("Making prediction...")
    probabilities = model.predict_proba(input_selected)[0]
    prediction = model.predict(input_selected)[0]
    confidence = float(probabilities[model.classes_.tolist().index(prediction)])
    
    logging.info(f"Prediction complete: {prediction} with {confidence:.2%} confidence")
    return prediction, confidence

def main():
    """Interactive command-line interface for penguin species prediction.
    
    Prompts the user for penguin measurements and displays the predicted species
    along with interesting facts about the predicted species.
    
    Handles invalid inputs and unexpected errors gracefully with appropriate error messages.
    """
    logging.info("Starting Penguin Species Predictor")
    print("\n===== Penguin Species Predictor =====\n")
    print("This program predicts the species of a penguin based on its physical measurements.")
    print("Please enter the following measurements:\n")
    
    try:
        # Get user input for penguin measurements
        bill_length = float(input("Bill length (mm): "))
        bill_depth = float(input("Bill depth (mm): "))
        flipper_length = float(input("Flipper length (mm): "))
        body_mass = float(input("Body mass (g): "))
        
        logging.info(f"Received measurements - Bill length: {bill_length}mm, Bill depth: {bill_depth}mm, "  
                     f"Flipper length: {flipper_length}mm, Body mass: {body_mass}g")
        
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
        logging.error("Invalid input: Please enter valid numeric values for all measurements")
        print("\nError: Please enter valid numeric values for all measurements.")
    except Exception as e:
        logging.error(f"An error occurred during prediction: {str(e)}")
        print(f"\nAn error occurred: {e}")

def run_examples():
    """Run example predictions using pre-defined measurement sets.
    
    This function demonstrates the prediction capability using representative examples
    for each penguin species (Adelie, Chinstrap, and Gentoo).
    
    Useful for testing the model's performance and understanding typical measurements
    for each species.
    """
    logging.info("Running example predictions")
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
        logging.info(f"Running example {i}")
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
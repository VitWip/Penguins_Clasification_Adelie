#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main Application Module for Penguin Classification Project.

This module serves as the entry point for the penguin classification application.
It handles the initialization of necessary components and orchestrates the workflow:
1. Sets up logging with date-based file output
2. Ensures database exists and is populated with initial data
3. Verifies the trained model exists, training if necessary
4. Updates the database with new penguin data from the API

The application uses a comprehensive logging system to track operations
and potential errors across all components.
"""

import logging
import os
from datetime import datetime

# Import modules from their new locations
from src.model.predict_penguin import predict_species
from src.data.create_database import create_penguin_database
from src.api.update_penguins import update_database_with_new_penguin
from src.web.generate_site_data import generate_site_data

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging with date-based file
log_filename = os.path.join('logs', f'penguin_classifier_{datetime.now().strftime("%Y-%m-%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def main():
    """Main application function that orchestrates the workflow.
    
    This function performs the following operations in sequence:
    1. Checks for and creates the database if it doesn't exist
    2. Verifies the existence of the trained model, importing training module if needed
    3. Updates the database with new penguin data from the API
    4. Generates site data for the GitHub Pages website
    
    The function includes error handling and logging for all major operations.
    
    Raises:
        Exception: If any critical operation fails during execution
    """
    try:
        # Check if database exists, if not create it
        db_path = os.path.join(os.path.dirname(__file__), 'src', 'data', 'penguins.db')
        if not os.path.exists(db_path):
            logging.info("Database not found. Creating new database...")
            create_penguin_database()
            logging.info("Database created successfully!\n")
        
        # Check if model exists
        model_path = os.path.join(os.path.dirname(__file__), 'src', 'model', 'best_penguin_model.joblib')
        if not os.path.exists(model_path):
            logging.info("Model not found. Training new model...")
            from src.model import penguins_training_model
            logging.info("Model training completed!\n")
        
        # Update database with new penguin data
        update_database_with_new_penguin()

        # Generate site data
        generate_site_data()

    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program terminated with error: {str(e)}")
        exit(1)
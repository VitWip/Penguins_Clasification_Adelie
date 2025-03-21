#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Penguin Database Update Module.

This module handles fetching new penguin data from an external API and updating the local database.
It includes functionality to:
1. Fetch new penguin measurements from a remote API
2. Check for duplicate entries using timestamps
3. Predict the penguin species using the trained model
4. Store the new penguin data in the database

The module ensures data integrity by checking for existing entries and handles various
error cases that might occur during the API request or database operations.
"""

import requests
import sqlite3
import os
from datetime import datetime
from src.model.predict_penguin import predict_species
import logging


def update_database_with_new_penguin():
    """Fetch new penguin data from API and update the database.
    
    This function performs the following operations:
    1. Fetches new penguin measurement data from the specified API endpoint
    2. Checks if the penguin data already exists in the database using its timestamp
    3. If new data, predicts the penguin species using the trained model
    4. Stores the new penguin data and its timestamp in the database
    
    The function includes comprehensive error handling and logging for:
    - API request failures
    - Database operation errors
    - Duplicate entry detection
    - General unexpected errors
    
    Returns:
        None. Results are logged and printed to console.
    """
    
    # API endpoint
    api_url = 'http://130.225.39.127:8000/new_penguin/'
    
    try:
        # Fetch data from API
        logging.info("Fetching new penguin data from API...")
        response = requests.get(api_url)
        response.raise_for_status()
        penguin_data = response.json()
        
        # Connect to SQLite database to check for existing entry
        conn = sqlite3.connect('penguins.db')
        cursor = conn.cursor()

        # Check if datetime already exists
        cursor.execute('SELECT COUNT(*) FROM penguin_dates WHERE date = ?', (penguin_data['datetime'],))
        exists = cursor.fetchone()[0] > 0

        if exists:
            logging.info("Skipping: A penguin with this datetime already exists in the database.")
            conn.close()
            return

        # Extract measurements
        bill_length = penguin_data['bill_length_mm']
        bill_depth = penguin_data['bill_depth_mm']
        flipper_length = penguin_data['flipper_length_mm']
        body_mass = penguin_data['body_mass_g']
        
        logging.info(f"Processing new penguin measurements - Bill length: {bill_length}mm, Bill depth: {bill_depth}mm, "
                     f"Flipper length: {flipper_length}mm, Body mass: {body_mass}g")
        
        # Predict species using the model
        species, confidence = predict_species(bill_length, bill_depth, flipper_length, body_mass)
        
        # Database connection already established above
        logging.info("Inserting new penguin data into database...")
        
        # Insert new penguin data
        cursor.execute('''
        INSERT INTO penguins (species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, is_original)
        VALUES (?, ?, ?, ?, ?, 1)
        ''', (species, bill_length, bill_depth, flipper_length, body_mass))
        
        # Get the ID of the newly inserted penguin
        penguin_id = cursor.lastrowid
        
        # Insert the datetime into penguin_dates table
        cursor.execute('''
        INSERT INTO penguin_dates (id, date)
        VALUES (?, ?)
        ''', (penguin_id, penguin_data['datetime']))
        
        # Commit changes and close connection
        conn.commit()
        
        # Log success message with details
        logging.info("=== New Penguin Added to Database ===")
        logging.info(f"Measurements: Bill Length: {bill_length}mm, Bill Depth: {bill_depth}mm, "
                    f"Flipper Length: {flipper_length}mm, Body Mass: {body_mass}g")
        logging.info(f"Predicted Species: {species} (Confidence: {confidence:.2%})")
        logging.info("Database Status: Successfully added with is_original = 1")
        
        # Print user-friendly message
        print("\n=== New Penguin Added to Database ===")
        print(f"Measurements:")
        print(f"  Bill Length: {bill_length} mm")
        print(f"  Bill Depth: {bill_depth} mm")
        print(f"  Flipper Length: {flipper_length} mm")
        print(f"  Body Mass: {body_mass} g")
        print(f"Predicted Species: {species} (Confidence: {confidence:.2%})")
        print(f"Database Status: Successfully added with is_original = 1")
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching data from API: {e}"
        logging.error(error_msg)
        print(error_msg)
    except sqlite3.Error as e:
        error_msg = f"Database error: {e}"
        logging.error(error_msg)
        print(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logging.error(error_msg)
        print(error_msg)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    update_database_with_new_penguin()
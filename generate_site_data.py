#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data Generation Script for GitHub Pages.

This script generates the necessary data files for the Jekyll-based GitHub Pages website.
It extracts information from the SQLite database and creates JSON files containing:
1. Latest prediction data
2. Historical predictions
3. Summary statistics

The generated files are placed in the assets directory for Jekyll to use.
"""

import sqlite3
import json
import os
from datetime import datetime

def create_data_directory():
    """Create assets directory if it doesn't exist."""
    data_dir = os.path.join('docs', 'assets')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def get_latest_prediction(cursor):
    """Fetch the most recent prediction from the database."""
    cursor.execute("""
        SELECT p.species, p.bill_length_mm, p.bill_depth_mm, p.flipper_length_mm, 
               p.body_mass_g, pd.date
        FROM penguins p
        JOIN penguin_dates pd ON p.id = pd.id
        WHERE p.is_original = 1
        ORDER BY pd.date DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    
    if result:
        return {
            'species': result[0],
            'bill_length_mm': result[1],
            'bill_depth_mm': result[2],
            'flipper_length_mm': result[3],
            'body_mass_g': result[4],
            'datetime': result[5]
        }
    return None

def get_historical_predictions(cursor, limit=50):
    """Fetch historical predictions from the database."""
    cursor.execute("""
        SELECT p.species, p.bill_length_mm, p.bill_depth_mm, p.flipper_length_mm, 
               p.body_mass_g, pd.date
        FROM penguins p
        JOIN penguin_dates pd ON p.id = pd.id
        WHERE p.is_original = 1
        ORDER BY pd.date DESC
        LIMIT ?
    """, (limit,))
    
    predictions = []
    for row in cursor.fetchall():
        predictions.append({
            'species': row[0],
            'bill_length_mm': row[1],
            'bill_depth_mm': row[2],
            'flipper_length_mm': row[3],
            'body_mass_g': row[4],
            'datetime': row[5]
        })
    return predictions

def calculate_statistics(cursor):
    """Calculate summary statistics from the database."""
    # Get total predictions
    cursor.execute("SELECT COUNT(*) FROM penguins WHERE is_original = 1")
    total_predictions = cursor.fetchone()[0]
    
    # Get species distribution
    cursor.execute("""
        SELECT species, COUNT(*) * 100.0 / COUNT(*) OVER()
        FROM penguins
        WHERE is_original = 1
        GROUP BY species
    """)
    species_dist = {row[0]: round(row[1], 1) for row in cursor.fetchall()}
    
    return {
        'total_predictions': total_predictions,
        'species_distribution': {
            'adelie': species_dist.get('Adelie', 0),
            'chinstrap': species_dist.get('Chinstrap', 0),
            'gentoo': species_dist.get('Gentoo', 0)
        }
    }

def generate_site_data():
    """Generate all necessary data files for the Jekyll site."""
    data_dir = create_data_directory()
    
    try:
        # Connect to the database
        conn = sqlite3.connect('penguins.db')
        cursor = conn.cursor()
        
        # Generate latest prediction data
        latest_prediction = get_latest_prediction(cursor)
        with open(os.path.join(data_dir, 'latest_prediction.json'), 'w') as f:
            json.dump(latest_prediction, f, indent=2)
        
        # Generate historical predictions
        predictions = get_historical_predictions(cursor)
        with open(os.path.join(data_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Generate statistics
        statistics = calculate_statistics(cursor)
        with open(os.path.join(data_dir, 'statistics.json'), 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print("Site data generated successfully!")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    generate_site_data()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Database Creation Module for Penguin Classification Project.

This module handles the initial setup of the SQLite database for storing penguin data.
It loads the penguin dataset from seaborn, processes it, and creates the necessary database
tables with appropriate schema.

Tables Created:
    - penguins: Stores penguin measurements and species information
    - penguin_dates: Stores timestamps for penguin records

Note: The is_original flag in the penguins table is used to distinguish between
original dataset entries (0) and new entries from the API (1).
"""

import sqlite3
import seaborn as sns
import pandas as pd
import logging

def create_penguin_database():
    """Create and populate the SQLite database with penguin data.
    
    This function performs the following operations:
    1. Loads the penguins dataset from seaborn
    2. Preprocesses the data by removing unnecessary columns and null values
    3. Creates the database schema with required tables
    4. Populates the tables with the processed data
    
    The function uses logging to track the progress and any potential errors.
    
    Raises:
        sqlite3.Error: If there's an issue with database operations
    """
    # Load the dataset
    logging.info("Loading penguin dataset...")
    penguins = sns.load_dataset("penguins").dropna()
    data = penguins.drop(columns=['island', 'sex'])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create SQLite database
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src', 'data', 'penguins.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        logging.info("Creating database tables...")
        # Create penguins table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS penguins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species TEXT NOT NULL,
            bill_length_mm REAL,
            bill_depth_mm REAL,
            flipper_length_mm REAL,
            body_mass_g REAL,
            confidence TEXT DEFAULT '100.00%',
            is_original INTEGER DEFAULT 1
        )
        ''')
        
        # Create penguin_dates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS penguin_dates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TIMESTAMP NOT NULL
        )''')
        
        # Insert data
        logging.info("Inserting data into database...")
        for _, row in data.iterrows():
            cursor.execute('''
            INSERT INTO penguins (species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, confidence, is_original)
            VALUES (?, ?, ?, ?, ?, '100.00%', 0)
            ''', (
                row['species'],
                row['bill_length_mm'],
                row['bill_depth_mm'],
                row['flipper_length_mm'],
                row['body_mass_g']
            ))
        
        # Commit changes and close connection
        conn.commit()
        logging.info("Database created successfully!")
        logging.info(f"Total records inserted: {len(data)}")
        
        # Verify data
        cursor.execute('SELECT COUNT(*) FROM penguins')
        count = cursor.fetchone()[0]
        logging.info(f"Records in database: {count}")
        
    except sqlite3.Error as e:
        logging.error(f"Database error occurred: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_penguin_database()
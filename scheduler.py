import schedule
import time
import subprocess
import os

def run_prediction():
    # Run the main script which handles predictions and updates
    subprocess.run(["python", "/app/main.py"])

# Schedule the task to run every day at specific times
schedule.every().day.at("07:30").do(run_prediction)
schedule.every().day.at("18:20").do(run_prediction)

print("Scheduler started. Running predictions at 07:30 and 18:20 daily.")

while True:
    schedule.run_pending()
    time.sleep(1)

import pandas as pd
import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

CSV_FILE = "car_data.csv"

def initialize_csv():
    """Create the CSV file if it does not exist."""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Brand", "Name", "Last_PMS_KM", "Last_PMS_Date"])
        df.to_csv(CSV_FILE, index=False)

def save_car_data(brand, name, kms, date):
    """Save new car data into the CSV file."""
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        initialize_csv()
        df = pd.read_csv(CSV_FILE)

    new_entry = pd.DataFrame({
        "Brand": [brand],
        "Name": [name],
        "Last_PMS_KM": [kms],
        "Last_PMS_Date": [date]
    })

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def read_car_data():
    """Read and return the car data from CSV."""
    try:
        return pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        initialize_csv()
        return pd.read_csv(CSV_FILE)

def predict_next_pms(last_kms, last_date_str):
    """Predict the next preventive maintenance schedule (PMS)."""
    KM_INTERVAL = 5000
    TIME_INTERVAL_MONTHS = 6

    try:
        if isinstance(last_date_str, str):
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        else:
            # If it's already a datetime or date object
            last_date = last_date_str
    except (ValueError, TypeError):
        last_date = datetime.today()

    next_km = last_kms + KM_INTERVAL
    next_date = last_date + timedelta(days=30 * TIME_INTERVAL_MONTHS)

    return next_km, next_date

def get_maintenance_tips(brand):
    """Get maintenance tips based on the car brand."""
    tips = {
        "Toyota": [
            "Always use Toyota-approved engine oil.",
            "Check brake pads every 10,000 km.",
            "Inspect air filter every 20,000 km.",
            "Replace spark plugs every 40,000 km."
        ],
        "Honda": [
            "Use genuine Honda parts for replacement.",
            "Check CVT fluid every 20,000 km.",
            "Replace engine air filter every 30,000 km.",
            "Inspect brake fluid every 3 years."
        ],
        "Ford": [
            "Use fully synthetic oil for best performance.",
            "Watch for battery health in rainy season.",
            "Check power steering fluid regularly.",
            "Inspect suspension components every 40,000 km."
        ],
        "Mitsubishi": [
            "Flush your radiator every 40,000 km.",
            "Inspect transmission fluid regularly.",
            "Check timing belt condition at 60,000 km.",
            "Replace fuel filter every 30,000 km."
        ],
        "Kia": [
            "Follow the Kia maintenance schedule strictly.",
            "Use Kia-approved parts and fluids.",
            "Inspect braking system every 10,000 km.",
            "Check suspension components annually."
        ],
        "Nissan": [
            "Replace CVT fluid as recommended by Nissan.",
            "Check cooling system every 30,000 km.",
            "Inspect brake pads and rotors regularly.",
            "Replace air filter every 15,000-20,000 km."
        ],
        "Hyundai": [
            "Follow Hyundai's recommended service schedule.",
            "Use Hyundai genuine parts for best performance.",
            "Check transmission fluid every 40,000 km.",
            "Inspect timing belt at 60,000 km for older models."
        ],
        "Jeep": [
            "Check 4x4 system regularly if used off-road.",
            "Inspect transfer case fluid every 30,000 km.",
            "Use Mopar parts for best compatibility.",
            "Check undercarriage for damage after off-road driving."
        ],
        "Suzuki": [
            "Change oil every 5,000 km for optimal performance.",
            "Check valve clearance at 20,000 km intervals.",
            "Inspect cooling system regularly.",
            "Check brake fluid every 2 years."
        ],
        "Isuzu": [
            "For diesel engines, use proper diesel engine oil.",
            "Check fuel/water separator regularly.",
            "Inspect turbocharger for leaks and damage.",
            "Replace fuel filter every 20,000 km."
        ]
    }
    return tips.get(brand, ["Follow general car maintenance guidelines."])

def get_dealer_locations(brand):
    """Get nearby dealer locations based on the car brand."""
    dealers = {
        "Toyota": [
            "Toyota San Fernando - McArthur Hwy, San Fernando, Pampanga",
            "Toyota Angeles - Angeles-Magalang Rd, Angeles, Pampanga"
        ],
        "Honda": [
            "Honda Cars Pampanga - San Agustin, San Fernando, Pampanga",
            "Honda Cars Angeles-Clark - M.A. Roxas Highway, Clark Freeport Zone, Angeles City"
        ],
        "Mitsubishi": [
            "Mitsubishi Motors Pampanga - Jose Abad Santos Avenue, San Fernando City",
            
            "Mitsubishi Motors Angeles City - Balibago Highway, Angeles, Pampanga"
        ],
        "Ford": [
            "Ford Pampanga - Jose Abad Santos Avenue, San Fernando",
            "Ford Clark - Clark Auto Zone, M.A. Roxas Highway, Clark, Pampanga"
        ],
        "Kia": [
            "Kia Pampanga - Jose Abad Santos Ave., San Fernando City",
            "Kia Clark - LGC Automotive Services Building, M.A. Roxas Highway, Clark Freeport Zone"
        ],
        "Nissan": [
            "Nissan Pampanga - McArthur Highway, Dolores, San Fernando",
            "Nissan Clark - LGC Automotive Services Building, M.A. Roxas Highway, Clark Freeport Zone"
        ],
        "Hyundai": [
            "Hyundai Pampanga - Jose Abad Santos Ave., Dolores, San Fernando City",
            "Hyundai Clark Pampanga - LGC Automotive Service Building, M.A. Roxas Avenue, Clark Freeport Zone, Angeles City"
        ],
        "Jeep": [
            "Jeep Pampanga by Auto Nation - San Fernando, Pampanga"
        ],
        "Suzuki": [
            "Suzuki Auto San Fernando - San Fernando, Pampanga",
            "Suzuki Auto Angeles - 113 MacArthur Highway, Angeles, Pampanga"
        ],
        "Isuzu": [
            "Isuzu Pampanga - Olongapo-Gapan Road, Dolores, San Fernando",
            "Isuzu Clark - Clark Freeport Zone, Pampanga"
        ]
    }
    return dealers.get(brand, ["No known nearby dealers."])


def predict_part_failure(brand, model, current_km):
    """
    Predict when specific car parts may fail based on mileage and car model.
    Returns a list of predictions for parts like brake pads, timing belts, etc.
    """
    predictions = []
    
    # Part lifespans in km (adjust based on brand/model if needed)
    part_lifespans = {
        "Brake Pads": 42000,
        "Timing Belt": 100000,
        "Tires": 60000,
        "Air Filter": 20000,
        "Spark Plugs": 40000
    }
    
    # Adjust lifespans for specific brands/models (example)
    if brand.lower() == "toyota" and "vios" in model.lower():
        part_lifespans["Brake Pads"] = 45000  # Example adjustment
    
    for part, lifespan in part_lifespans.items():
        km_remaining = lifespan - current_km
        if km_remaining <= 5000:  # Warn if within 5,000 km
            prediction = f"{part} may need replacement in ~{km_remaining:,} km. Average replacement at {lifespan:,} km."
            predictions.append(prediction)
    
    if not predictions:
        predictions.append("No immediate part replacements needed based on current mileage.")
    
    return predictions
import streamlit as st
from utils import initialize_csv, save_car_data, read_car_data, get_dealer_locations, get_maintenance_tips, predict_part_failure, predict_next_pms
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import os
import pandas as pd
import speech_recognition as sr
import feedparser
from bs4 import BeautifulSoup
import requests
import urllib3
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import google.generativeai as genai


# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY =("your api")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize
initialize_csv()
try:
    chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.3)
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {e}")
    st.stop()

# Voice Input Function
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand.")
        return ""
    except Exception as e:
        st.error(f"Voice recognition error: {e}")
        return ""

# Cache chatbot responses
@st.cache_data
def get_chat_response(_chat, prompt):
    try:
        response = _chat([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logging.error(f"Chat API error: {e}")
        return None

# Set page config
st.set_page_config(
    page_title="🚗 AutoBuddy",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'next_pms_date' not in st.session_state:
    st.session_state.next_pms_date = None
if 'next_pms_vehicle' not in st.session_state:
    st.session_state.next_pms_vehicle = ""

# Apply clean light theme CSS
st.markdown("""
<style>
    :root {
        --background: #f8f9fa;
        --card-bg: #ffffff;
        --text-color: #212529;
        --primary: #0d6efd;
        --secondary: #6c757d;
        --border: #dee2e6;
    }
    .stApp {
        background-color: var(--background);
        color: var(--text-color);
    }
    .stCard {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    p, div, span, label {
        color: var(--text-color);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
        font-weight: 600;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border: 1px solid var(--border);
        border-radius: 4px;
    }
    .stSelectbox > div > div {
        border: 1px solid var(--border);
        border-radius: 4px;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="value"] span {
        color: var(--text-color) !important;
    }
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.375rem 0.75rem;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: var(--card-bg);
        border-right: 1px solid var(--border);
    }
    .stTextInput label, 
    .stNumberInput label, 
    .stSelectbox label, 
    .stSlider label {
        color: var(--text-color);
        font-weight: 500;
    }
    .chat-message {
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 4px;
    }
    .user-message {
        background-color: #e9ecef;
        text-align: right;
    }
    .bot-message {
        background-color: #f1f3f5;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Create custom card function
def card(title, content):
    st.markdown(f"""
    <div class="stCard">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# App Title with branding
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🚙 AutoBuddy")
    st.markdown("##### *Your car maintenance companion*")

st.markdown("---")

# Navigation using tabs
tabs = st.tabs(["💬 Chat", "🔧 Update My PMS", "📊 View PMS History", "🛠️ Parts Finder", "📰 Car Industry News", "🚘 Car Recommendation"])

# ====== CHAT TAB ======
with tabs[0]:
    st.subheader("Talk to your Car Assistant!")
    st.markdown("Describe any car issues (e.g., 'My car makes a clicking noise when I turn the key') or ask about maintenance.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div><strong>You:</strong> {message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div><strong>🤖 AutoBuddy:</strong> {message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message...", 
            placeholder="e.g., 'My car is making a clicking noise when I turn the key'...",
            key="user_input"
        )
        col1, col2 = st.columns([5, 1])
        with col1:
            submit_button = st.form_submit_button("Send")
            voice_input_clicked = st.form_submit_button("🎤 Voice input")
    
    # Handle text input
    if submit_button and user_input:
        if len(user_input.strip()) < 5 or user_input.lower() in ["toyota", "honda", "mitsubishi", "ford", "kia", "nissan", "hyundai", "jeep", "suzuki", "isuzu"]:
            st.warning("Please describe a specific car issue (e.g., 'My Toyota makes a clicking noise') or ask a maintenance question (e.g., 'How often should I change my Toyota's oil?').")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner('AutoBuddy is analyzing...'):
                prompt = f"""You are AutoBuddy, a car maintenance expert. Respond directly to the user's input without introductory phrases like 'I'm ready to put on my mechanic's hat.' 
                - If the user describes a car issue (e.g., '{user_input}'), provide a possible diagnosis and next steps.
                - If the user asks a maintenance question, answer concisely.
                - If the user provides only a car brand (e.g., 'Toyota'), give general maintenance tips for that brand and ask for more details.
                Example responses:
                - Issue: 'My Toyota Vios makes a clicking noise when I turn the key' → 'A clicking noise could indicate a weak battery, faulty starter motor, or loose connections. Next steps: 1. Check battery voltage (~12.6V when off). 2. Inspect terminals for corrosion. 3. Test the starter motor if battery is fine.'
                - Question: 'How often should I change my Toyota's oil?' → 'Change the oil every 5,000-10,000 km using Toyota-approved oil.'
                - Brand: 'Toyota' → 'For Toyota vehicles, change oil every 5,000-10,000 km, check brake pads every 10,000 km, and replace spark plugs every 40,000 km. Could you describe a specific issue or ask a question?'
                Respond to: '{user_input}'"""
                logging.info(f"Sending prompt: {prompt}")
                response_text = get_chat_response(chat, prompt)
                if not response_text or "mechanic's hat" in response_text.lower() or len(response_text.strip()) < 20:
                    response_text = "Sorry, I didn't understand your request. Please describe a specific car issue (e.g., 'My car makes a clicking noise') or ask a maintenance question (e.g., 'How often should I change my oil?')."
                logging.info(f"API Response: {response_text}")
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()
    
    # Handle voice input
    if voice_input_clicked:
        voice_text = voice_input()
        if voice_text:
            if len(voice_text.strip()) < 5 or voice_text.lower() in ["toyota", "honda", "mitsubishi", "ford", "kia", "nissan", "hyundai", "jeep", "suzuki", "isuzu"]:
                st.warning("Please describe a specific car issue (e.g., 'My Toyota makes a clicking noise') or ask a maintenance question (e.g., 'How often should I change my Toyota's oil?').")
            else:
                st.session_state.messages.append({"role": "user", "content": voice_text})
                with st.spinner('AutoBuddy is analyzing...'):
                    prompt = f"""You are AutoBuddy, a car maintenance expert. Respond directly to the user's input without introductory phrases like 'I'm ready to put on my mechanic's hat.' 
                    - If the user describes a car issue (e.g., '{voice_text}'), provide a possible diagnosis and next steps.
                    - If the user asks a maintenance question, answer concisely.
                    - If the user provides only a car brand (e.g., 'Toyota'), give general maintenance tips for that brand and ask for more details.
                    Example responses:
                    - Issue: 'My Toyota Vios makes a clicking noise when I turn the key' → 'A clicking noise could indicate a weak battery, faulty starter motor, or loose connections. Next steps: 1. Check battery voltage (~12.6V when off). 2. Inspect terminals for corrosion. 3. Test the starter motor if battery is fine.'
                    - Question: 'How often should I change my Toyota's oil?' → 'Change the oil every 5,000-10,000 km using Toyota-approved oil.'
                    - Brand: 'Toyota' → 'For Toyota vehicles, change oil every 5,000-10,000 km, check brake pads every 10,000 km, and replace spark plugs every 40,000 km. Could you describe a specific issue or ask a question?'
                    Respond to: '{voice_text}'"""
                    logging.info(f"Sending prompt: {prompt}")
                    response_text = get_chat_response(chat, prompt)
                    if not response_text or "mechanic's hat" in response_text.lower() or len(response_text.strip()) < 20:
                        response_text = "Sorry, I didn't understand your request. Please describe a specific car issue (e.g., 'My car makes a clicking noise') or ask a maintenance question (e.g., 'How often should I change my oil?')."
                    logging.info(f"API Response: {response_text}")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ====== OTHER TABS (UNCHANGED) ======
with tabs[1]:
    st.subheader("Update Your PMS Record")
    col1, col2 = st.columns(2)
    with col1:
        card("Car Details", """
            Select your car brand and enter the model.
        """)
        brand = st.selectbox("Select Car Brand", 
                            ["Toyota", "Honda", "Mitsubishi", "Ford", "Kia", "Nissan", 
                             "Hyundai", "Jeep", "Suzuki", "Isuzu"])
        model = st.text_input("Car Model", placeholder="e.g. Civic, Vios, etc.")
    with col2:
        card("Maintenance Details", """
            Enter your last PMS reading and location.
        """)
        last_pms_km = st.number_input("Last PMS Odometer Reading (km)", min_value=0, step=100)
        last_pms_date = st.date_input("Last PMS Date")
        location = st.text_input("Your Location", placeholder="e.g. Angeles City")
    save_button = st.button("💾 Save PMS Record")
    if save_button:
        if model and last_pms_km > 0:
            save_car_data(brand, model, last_pms_km, last_pms_date.strftime("%Y-%m-%d"))
            st.success(f"✅ PMS record saved for your {brand} {model}!")
            st.balloons()
        else:
            st.error("Please fill in all required fields")


estimated_cost_by_segment = {
    "Subcompact": {  # e.g., Toyota Vios, Honda Brio, Suzuki Dzire
        "10,000": (2500, 3500),
        "20,000": (3000, 4000),
        "30,000": (3500, 5000),
        "40,000": (4000, 5500),
        "50,000": (4500, 6000),
        "60,000": (5000, 6500),
        "70,000": (5500, 7000),
        "80,000": (6000, 7500),
        "90,000": (6500, 8000),
    },
    "Compact": {  # e.g., Honda Civic, Toyota Altis
        "10,000": (3000, 4000),
        "20,000": (4000, 5500),
        "30,000": (5000, 7000),
        "40,000": (5500, 7500),
        "50,000": (6000, 8000),
        "60,000": (6500, 8500),
        "70,000": (7000, 9000),
        "80,000": (7500, 9500),
        "90,000": (8000, 10000),
    },
    "MPV": {  # e.g., Toyota Innova, Mitsubishi Xpander
        "10,000": (3500, 5000),
        "20,000": (4500, 6000),
        "30,000": (5500, 8000),
        "40,000": (6000, 8500),
        "50,000": (6500, 9000),
        "60,000": (7000, 9500),
        "70,000": (7500, 10000),
        "80,000": (8000, 11000),
        "90,000": (8500, 12000),
    },
    "SUV": {  # e.g., Fortuner, Montero, Everest
        "10,000": (4000, 6000),
        "20,000": (6000, 8000),
        "30,000": (7000, 10000),
        "40,000": (7500, 10500),
        "50,000": (8000, 11000),
        "60,000": (8500, 11500),
        "70,000": (9000, 12000),
        "80,000": (9500, 13000),
        "90,000": (10000, 14000),
    },
    "Pickup": {  # e.g., Toyota Hilux, Ford Raptor
        "10,000": (4500, 6500),
        "20,000": (6000, 8500),
        "30,000": (7500, 10500),
        "40,000": (8000, 11000),
        "50,000": (8500, 11500),
        "60,000": (9000, 12000),
        "70,000": (9500, 13000),
        "80,000": (10000, 14000),
        "90,000": (10500, 15000),
    },
    "Luxury MPV/SUV": {  # e.g., Alphard, Land Cruiser
        "10,000": (6000, 9000),
        "20,000": (9000, 12000),
        "30,000": (10000, 14000),
        "40,000": (11000, 15000),
        "50,000": (12000, 16000),
        "60,000": (13000, 17000),
        "70,000": (14000, 18000),
        "80,000": (15000, 19000),
        "90,000": (16000, 20000),
    },
}

car_model_to_segment = {
    "Toyota Vios": "Subcompact",
    "Honda Brio": "Subcompact",
    "Suzuki Dzire": "Subcompact",
    "Toyota Altis": "Compact",
    "Honda Civic": "Compact",
    "Toyota Innova": "MPV",
    "Mitsubishi Xpander": "MPV",
    "Toyota Fortuner": "SUV",
    "Mitsubishi Montero Sport": "SUV",
    "Ford Everest": "SUV",
    "Toyota Alphard": "Luxury MPV/SUV",
    "Toyota Land Cruiser": "Luxury MPV/SUV",
    "Toyota Hilux": "Pickup",
    "Ford Raptor": "Pickup",
}


# --- Tab 2: View PMS History & Next Maintenance Prediction ---

with tabs[2]:
    if st.session_state.get("next_pms_date"):
        today = datetime.today().date()
        reminder_day = st.session_state.next_pms_date.date() - timedelta(days=1)
        if today == reminder_day:
            st.warning(
                f"📢 Reminder: PMS for your {st.session_state.next_pms_vehicle} is scheduled for tomorrow ({st.session_state.next_pms_date.strftime('%B %d, %Y')})."
            )

    df = read_car_data()
    if df.empty:
        st.warning("🔍 No PMS records found. Please add your first PMS record in the Update tab!")
    else:
        st.markdown("### Your PMS Records")
        st.dataframe(df, use_container_width=True)
        st.markdown("---")
        st.subheader("Possible Your Next Maintenance")

        col1, col2 = st.columns(2)

        with col1:
                car_options = [
                    f"{i}. {row['Brand']} {row['Name']}"
                    for i, (_, row) in enumerate(df.iterrows())
                    if pd.notna(row['Last_PMS_KM'])
                ]

                selected_car = st.selectbox("Select a vehicle", car_options)

                # Extract the correct index from the numbered string (e.g., "1. Toyota Vios" → 0)
                selected_idx = int(selected_car.split(".")[0]) 


        with col2:
            predict_button = st.button("🔮 Calculate My Next PMS")

        if predict_button and not df.empty:

            row = df.iloc[selected_idx]
            full_name = f"{row['Brand']} {row['Name']}"

            # Predict next PMS km and date (assuming your predict_next_pms function exists)
            next_km, next_date = predict_next_pms(row['Last_PMS_KM'], row['Last_PMS_Date'])

            # Determine car segment
            car_segment = car_model_to_segment.get(full_name, "Compact")

            # Format milestone string WITH commas to match dictionary keys
            milestone_value = round(next_km / 10000) * 10000
            milestone = f"{milestone_value:,}"  # e.g., "10,000"

            # Get estimated cost range from dictionary, fallback to (3500, 5000)
            cost_range = estimated_cost_by_segment.get(car_segment, {}).get(milestone, (3500, 5000))
            estimated_cost = f"₱{cost_range[0]:,} - ₱{cost_range[1]:,}"

            # Save to session state for reminder use
            st.session_state.next_pms_date = next_date
            st.session_state.next_pms_vehicle = full_name

            # Display next PMS details and dealer info
            col1, col2 = st.columns(2)

            with col1:
                card("Next PMS Details", f"""
                🚘 Vehicle: {full_name}
                🔄 Next PMS at: {next_km:,} km
                📅 Date: {next_date.strftime('%B %d, %Y')}
                💸 Estimated Cost: {estimated_cost}
                """)

            with col2:
                dealer = get_dealer_locations(row['Brand'])
                dealer_html = ""
                if isinstance(dealer, list):
                    for idx, d in enumerate(dealer, start=1):
                        dealer_html += f"{idx}. {d}  "
                else:
                    dealer_html = f"1. {dealer}"

                card("Authorized Service Centers", f"""
                🏢 Nearest {row['Brand']} Dealers in Pampanga:<br>
                {dealer_html}
                """)

            # Prepare downloadable note in the requested format
            dealers_list = []
            if isinstance(dealer, list):
                dealers_list = dealer
            else:
                dealers_list = [dealer]

            # Format dealer list with numbering and new lines
            dealer_note_lines = [f"{idx}. {d}" for idx, d in enumerate(dealers_list, start=1)]
            dealer_note_str = "\n".join(dealer_note_lines)

            note = f"""Next PMS Reminder:
Vehicle: {full_name}
Next PMS Date: {next_date.strftime('%B %d, %Y')}
Target KM: {next_km:,} km
Estimated Cost: {estimated_cost}
Nearest Dealer(s):
{dealer_note_str}"""

            st.download_button(
                label="📝 Download PMS Note",
                data=note,
                file_name=f"{next_date.strftime('%Y-%m-%d')}_PMS_Note.txt",
                mime="text/plain"
            )

            st.markdown("---")
            st.subheader("Replacement Part Alerts")

            parts_predictions = predict_part_failure(row['Brand'], row['Name'], row['Last_PMS_KM'])

            col1, col2 = st.columns(2)

            with col1:
                card("Parts Likely to Need Replacement", "<br>".join(parts_predictions))

            with col2:
                tips = get_maintenance_tips(row['Brand'])
                tips_html = "<ol>"
                for tip in tips[:3]:
                    tips_html += f"<li>{tip}</li>"
                tips_html += "</ol>"
                card(f"Maintenance Tips for {row['Brand']}", tips_html)

with tabs[3]:
    st.subheader("Find Car Parts Online")
    st.markdown("""
    Browse high-quality parts and accessories for your car brand and model.

    - 🧩 Engine Parts
    - 🔋 Batteries
    - 🛞 Tires and Wheels
    - 🚗 Interior and Exterior Accessories

    > 💡 Tip: Always use manufacturer-recommended parts for best performance and safety.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌐 Official Brand Websites (Philippines)")
        brand_links = {
            "Toyota": "https://www.toyota.com.ph",
            "Honda": "https://www.hondaphil.com",
            "Mitsubishi": "https://www.mitsubishi-motors.com.ph",
            "Ford": "https://www.ford.com.ph",
            "Kia": "https://www.kia.com/ph/main.html",
            "Nissan": "https://www.nissan.ph",
            "Hyundai": "https://www.hyundai.ph",
            "Suzuki": "https://auto.suzuki.com.ph",
            "Isuzu": "https://www.isuzuphil.com",
            "Jeep": "https://www.jeep.com.ph",
            "Autodoc (Global Parts Shop)": "https://www.autodoc.co.uk"
        }
        for brand, url in brand_links.items():
            st.markdown(f"- [{brand} Parts Website]({url})")
        st.markdown("### 🌍 Off-Road Parts Suppliers (Philippines)")
        off_road_links = {
            "PartsPro.PH": "https://partspro.ph",
            "Overland Kings": "https://overlandkings.ph",
            "Dubshop": "https://dubshop.ph",
            "Premium Overland": "https://premiumoverland.ph",
            "MD Juan Enterprises": "https://mdjuan.com.ph",
            "199 Offroad House": "https://199offroadhouse.com",
            "Ride Offroad": "https://rideoffroad.ph"
        }
        for brand, url in off_road_links.items():
            st.markdown(f"- [{brand}]({url})")
    with col2:
        st.markdown("### 🇯🇵 JDM Car Parts")
        st.markdown("""
        - [Nengun Performance](https://www.nengun.com/)
        - [RHDJapan](https://www.rhdjapan.com/)
        - [JDM Parts Ru](https://jdmparts.ru/)
        - [Up Garage](https://www.upgarage.com/)
        """)
        st.markdown("### 🇺🇸 Muscle Car Parts")
        st.markdown("""
        - [Summit Racing](https://www.summitracing.com/)
        - [JEGS](https://www.jegs.com/)
        - [Classic Industries](https://www.classicindustries.com/)
        - [LMC Truck](https://www.lmctruck.com/)
        """)
        st.markdown("### 🇩🇪 Euro Sports Car Parts")
        st.markdown("""
        - [ECS Tuning](https://www.ecstuning.com/)
        - [FCP Euro](https://www.fcpeuro.com/)
        - [Turner Motorsport (BMW)](https://www.turnermotorsport.com/)
        - [Pelican Parts (Porsche, BMW, Mercedes)](https://www.pelicanparts.com/)
        """)
    st.info("You can also ask AutoBuddy in the chat tab for part suggestions!")

with tabs[4]:
    st.subheader("📰 Latest Car Industry News")
    rss_feeds = {
        "CarGuide PH": "https://feeds.feedburner.com/Carguideph",
        "Motor1": "https://www.motor1.com/rss/",
        "Autoblog": "https://www.autoblog.com/rss/feed"
    }
    selected_source = st.selectbox("Select News Source", list(rss_feeds.keys()))
    feed_url = rss_feeds[selected_source]
    with st.spinner("Fetching news..."):
        feed = feedparser.parse(feed_url)
    if feed.bozo:
        st.error(f"❌ Error parsing feed: {feed.bozo_exception}")
    elif not feed.entries:
        st.warning("⚠️ No articles found. The feed might be empty or temporarily down.")
    else:
        try:
            for entry in feed.entries[:5]:
                st.markdown(f"### 🔗 [{entry.title}]({entry.link})")
                st.caption(entry.published if "published" in entry else "No date")
                soup = BeautifulSoup(entry.summary, "html.parser")
                img_tag = soup.find("img")
                if img_tag and img_tag.get("src"):
                    try:
                        st.image(img_tag["src"], width=400)
                    except:
                        st.write("(No image available)")
                summary_text = soup.get_text()
                st.write(summary_text.strip().split("Read More")[0])
                st.markdown("---")
        except Exception as e:
            st.error(f"Error displaying article: {e}")

with tabs[5]:
    st.subheader("🚘 Smart Car Recommendation")
    st.markdown("Tell us your preferences and AutoBuddy will find your perfect match!")

    # --- FORM-BASED CAR RECOMMENDATION ---
    with st.form("car_recommendation_form"):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.selectbox("💰 Budget Range", ["Any", "< ₱800k", "₱800k – ₱1.2M", "₱1.2M – ₱1.8M", "₱1.8M - 2.7M", "3M+"])
            fuel_type = st.selectbox("⛽ Preferred Fuel Type", ["Any", "Gasoline", "Diesel", "Hybrid", "Electric"])
            transmission = st.selectbox("🔁 Transmission", ["Any", "Automatic", "Manual"])
            seating = st.selectbox("🪑 Seating Capacity", ["Any", "2 Seaters", "4 - 5 Seaters", " 6 - 7 Seaters", "8+ Seaters or More"])
        with col2:
            brand_preference = st.selectbox("🚗 Brand Preference", 
                                            ["Any", "Toyota", "Honda", "Ford", "Hyundai", "Mitsubishi", "Kia", "Nissan", "Isuzu", "Mazda", "Subaru", "Volkswagen", "BMW", "Mercedez-Benz", "Audi", "Geely"])
            car_type = st.selectbox("🚙 Car Type", ["Any", "Sedan", "Hatchback", "Pickup", "Van", "SUV"])
            usage = st.selectbox("🛣️ Select Performance Preference", ["Economy", "Sporty", "Off-Roading", "Balanced"])
            engine = st.selectbox("Select Engine Size", ["Any", "1.0L - 1.5L", "1.5L - 2.0L", "2.0L - 2.5", "2.5L - 3.0L", "Above 3.0L"])

        submit_reco = st.form_submit_button("🔍 Recommend a Car")

    if submit_reco:


        prompt = f"""
        Act as an automotive expert in the Philippine market. Based on the following user preferences, recommend as many real car models (2023–2025) available in the Philippines that match or closely match the criteria:

        - Budget: {budget}
        - Fuel Type: {fuel_type}
        - Transmission: {transmission}
        - Seating Capacity: {seating}
        - Preferred Brand: {brand_preference}
        - Car Type: {car_type}
        - Intended Use: {usage}
        - Engine Size: {engine}

        For each car, provide:
        - Full Model Name
        - Estimated Price (in PHP)
        - Key Features (fuel economy, safety, tech)
        - Description of the car
        - Why it's a good match for the user's input
        - Pros and cons of the car

        Make sure the car models are sold in the Philippines as of 2024–2025. Do not recommend concept cars or unavailable models. Format the output in markdown with bold headers and bullet points for clarity.
        """

        with st.spinner("🤖 AutoBuddy is thinking..."):
            recommendation = get_chat_response(chat, prompt)

        if recommendation:
            card("🔎 Your Car Match", recommendation)
        else:
            st.error("❌ AutoBuddy couldn't find a match. Try changing your preferences.")

        # --- CONVERSATIONAL CHATBOT ---
    st.markdown("---")
    st.subheader("💬 Chat with AutoBuddy")
    st.markdown("Ask me anything about car specs, recommendations, or comparisons!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are AutoBuddy, an automotive expert specializing in the Philippine car market (2023–2025). Provide helpful, concise, and friendly answers to user questions about cars. Politely decline any questions about mechanical problems, repairs, or diagnostics."}
        ]

    # Delete button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are AutoBuddy, an automotive expert specializing in the Philippine car market (2023–2025). Provide helpful, concise, and friendly answers to user questions about cars. Politely decline any questions about mechanical problems, repairs, or diagnostics."}
        ]
        st.success("Chat history cleared!")

    # Display existing messages
    for message in st.session_state.chat_history[1:]:  # Skip system message
        if message["role"] == "user":
            st.markdown(f"**🧑 You:** {message['content']}")
        else:
            st.markdown(f"**🤖 AutoBuddy:** {message['content']}")

    # Chat input at the bottom
    user_query = st.chat_input("Ask AutoBuddy about cars...")

    # Basic problem-related keyword filtering
    problem_keywords = [
        "won't start", "not starting", "engine noise", "check engine", "broken",
        "problem", "issue", "overheat", "vibration", "malfunction", "diagnose",
        "repair", "fix", "stalling", "squeaking", "smoke", "trouble", "oil leak"
    ]

    def is_problem_related(query):
        return any(word in query.lower() for word in problem_keywords)

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        if is_problem_related(user_query):
            auto_reply = (
                "I'm here to help with car comparisons, specs, and recommendations! 🚗\n\n"
                "But for car problems or repairs, it's best to consult a certified mechanic or service center. 🔧"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": auto_reply})
            st.rerun()

        else:
            with st.spinner("AutoBuddy is typing..."):
                try:
                    response = chat.invoke(st.session_state.chat_history)
                    reply = response.content if hasattr(response, "content") else response
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ AutoBuddy encountered an error: {e}")



# Footer
st.markdown("---")
st.markdown("### 💡 Quick Tips")
tips_container = st.expander("Show car maintenance quick tips", expanded=False)
with tips_container:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Oil Changes**")
        st.markdown("• Change oil every 5,000-10,000 km")
        st.markdown("• Use manufacturer recommended grade")
        st.markdown("• Use Legit Parts")
    with col2:
        st.markdown("**Tire Care**")
        st.markdown("• Check tire pressure weekly")
        st.markdown("• Rotate tires every 10,000 km")
        st.markdown("• Tires expire after 6 years")
    with col3:
        st.markdown("**Battery Life**")
        st.markdown("• Clean terminals regularly")
        st.markdown("• Replace every 3-5 years")
        st.markdown("• Turn off lights & electronics")

st.markdown("<div style='text-align: center; color: #666;'>© 2025 AutoBuddy | Your Smart Car Companion</div>", unsafe_allow_html=True)

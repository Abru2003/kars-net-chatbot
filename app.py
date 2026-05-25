import streamlit as st
import pandas as pd
import google.generativeai as genai
import json

# --- UI Setup ---
st.set_page_config(page_title="KARS-Net AMR Assistant", page_icon="🩺", layout="centered")
st.title("🩺 KARS-Net Clinical Assistant")
st.caption("Powered by the 2023 Kerala Antimicrobial Resistance Surveillance Network Data")

# --- Load Data ---
# @st.cache_data prevents the app from reloading the CSV every time the user types a message
@st.cache_data
def load_data():
    return pd.read_csv('amr_data.csv')

df = load_data()

# --- The Backend Tool ---
def get_best_antibiotic(pathogen: str, specimen: str, location: str) -> str:
    """Fetches the best antibiotic with the lowest resistance for a given pathogen, specimen, and location."""
    
    filtered = df[
        (df['Pathogen'].str.contains(pathogen, case=False, na=False)) & 
        (df['Specimen'].str.contains(specimen, case=False, na=False))
    ]
    
    if specimen.lower() == 'blood':
        filtered = filtered[filtered['Location'].str.contains(location, case=False, na=False)]
    else:
        filtered = filtered[filtered['Location'] == 'All']

    if filtered.empty:
        return json.dumps({"error": "No data found for this specific combination in the AMR report."})

    best_options = filtered.sort_values(by='Resistance_Percentage', ascending=True).head(3)
    return best_options.to_json(orient="records")

def get_all_pathogens() -> str:
    """Returns a list of all unique pathogens tracked in the AMR database."""
    pathogens = df['Pathogen'].unique().tolist()
    return json.dumps({"available_pathogens": pathogens})

def get_general_report(pathogen: str) -> str:
    """Fetches all antibiotic resistance data for a specific pathogen across all specimens and locations. Use this when a user asks for a general report on a bacteria."""
    filtered = df[df['Pathogen'].str.contains(pathogen, case=False, na=False)]
    
    if filtered.empty:
        return json.dumps({"error": f"No data found for {pathogen}."})
    
    # Return all rows for this pathogen
    return filtered.to_json(orient="records")

def get_all_antibiotics() -> str:
    """Returns a list of all unique antibiotics tracked in the AMR database."""
    antibiotics = df['Antibiotic'].unique().tolist()
    # Sort them alphabetically so it looks clean in the chat
    antibiotics.sort()
    return json.dumps({"available_antibiotics": antibiotics})

def calculate_average_resistance(pathogen: str, specimen: str, antibiotic: str) -> str:
    """Calculates the mathematical average (mean) resistance percentage of a specific antibiotic 
    against a pathogen for a given specimen across all reported locations (ICU, IPD, OPD)."""
    
    # Filter for the pathogen, specimen, and antibiotic
    filtered = df[
        (df['Pathogen'].str.contains(pathogen, case=False, na=False)) & 
        (df['Specimen'].str.contains(specimen, case=False, na=False)) &
        (df['Antibiotic'].str.contains(antibiotic, case=False, na=False))
    ]
    
    # Drop rows where data is N/A or missing before doing math
    filtered = filtered.dropna(subset=['Resistance_Percentage'])
    
    if filtered.empty:
        return json.dumps({"error": f"No valid numerical data found to compute an average for {pathogen} against {antibiotic}."})
    
    # Calculate the average resistance
    avg_resistance = filtered['Resistance_Percentage'].mean()
    
    # Collect individual values to show the breakdown in the response
    breakdown = filtered[['Location', 'Resistance_Percentage']].to_dict(orient="records")
    
    return json.dumps({
        "pathogen": pathogen,
        "specimen": specimen,
        "antibiotic": antibiotic,
        "average_resistance_percentage": round(avg_resistance, 2),
        "breakdown": breakdown
    })

# --- API Setup ---
# We use Streamlit Secrets to safely store your API key so your instructor/public can't steal it
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# --- Initialize Chat Session ---
if "chat_session" not in st.session_state:
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        # Include all 5 tools in the array now
        tools=[get_best_antibiotic, get_all_pathogens, get_general_report, get_all_antibiotics, calculate_average_resistance], 
        system_instruction=(
            "You are a highly capable clinical assistant analyzing the 2024 Kerala AMR Surveillance Network Data (2023 data cycle). "
            "You have five tools at your disposal: "
            "1. Use 'get_all_pathogens' if the user wants a list of bacteria tracked in the database. "
            "2. Use 'get_all_antibiotics' if the user asks what antibiotics are present or tracked. "
            "3. Use 'get_general_report' if the user asks for a general overview or full report of a specific pathogen without specifying a location/specimen. "
            "4. Use 'calculate_average_resistance' if the user asks mathematical questions about the average, mean, or location breakdown of a specific drug against a specific bacteria. "
            "5. Use 'get_best_antibiotic' ONLY if the user asks for the best treatment or lowest resistance profile and provides a pathogen, specimen, and location. "
            "If the user asks an analytical or mathematical question, explain the math by showing the individual location breakdown alongside the calculated average. "
            "Always be professional, concise, and clearly state resistance percentages."
        )
    )
    st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)
    st.session_state.messages = []

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a clinical question (e.g., Best treatment for E. coli in ICU blood?)"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing AMR data..."):
            response = st.session_state.chat_session.send_message(prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
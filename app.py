import streamlit as st
import pandas as pd
import google.generativeai as genai
import json

# --- UI Setup ---
st.set_page_config(page_title="KARS-Net AMR Assistant", page_icon="🩺", layout="centered")
st.title("🩺 KARS-Net Clinical Assistant")
st.caption("Powered by 2021-2024 Kerala Antimicrobial Resistance Surveillance Network Data")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('merged_amr_data.csv')
    # Treat Year as a string so it filters perfectly
    df['Year'] = df['Year'].astype(str)
    return df

df = load_data()

# --- The Backend Tools ---
def get_best_antibiotic(pathogen: str, specimen: str, location: str, year: str = "2024") -> str:
    """Fetches the best antibiotic with the lowest resistance for a given pathogen, specimen, location, and year."""
    filtered = df[
        (df['Year'] == str(year)) &
        (df['Pathogen'].str.contains(pathogen, case=False, na=False)) &
        (df['Specimen'].str.contains(specimen, case=False, na=False)) &
        (df['Location'].str.contains(location, case=False, na=False))
    ]
    
    # Fallback: if they ask for ICU but the older year only has 'All' or 'Overall'
    if filtered.empty:
        filtered = df[
            (df['Year'] == str(year)) &
            (df['Pathogen'].str.contains(pathogen, case=False, na=False)) &
            (df['Specimen'].str.contains(specimen, case=False, na=False))
        ]

    if filtered.empty:
        return json.dumps({"error": f"No data found for {pathogen} in {specimen} for the year {year}."})

    best_options = filtered.dropna(subset=['Resistance_Percentage']).sort_values(by='Resistance_Percentage', ascending=True).head(3)
    return best_options.to_json(orient="records")

def get_all_pathogens() -> str:
    """Returns a list of all unique pathogens tracked in the AMR database."""
    pathogens = df['Pathogen'].unique().tolist()
    return json.dumps({"available_pathogens": pathogens})

def get_all_antibiotics() -> str:
    """Returns a list of all unique antibiotics tracked in the AMR database."""
    antibiotics = df['Antibiotic'].dropna().unique().tolist()
    antibiotics.sort()
    return json.dumps({"available_antibiotics": antibiotics})

def get_general_report(pathogen: str, year: str = "2024") -> str:
    """Fetches all antibiotic resistance data for a specific pathogen across all specimens and locations for a specific year."""
    filtered = df[
        (df['Year'] == str(year)) &
        (df['Pathogen'].str.contains(pathogen, case=False, na=False))
    ]
    if filtered.empty:
        return json.dumps({"error": f"No data found for {pathogen} in {year}."})
    return filtered.to_json(orient="records")

def calculate_average_resistance(pathogen: str, specimen: str, antibiotic: str, year: str = "2024") -> str:
    """Calculates the average resistance of an antibiotic against a pathogen for a specimen across all locations."""
    filtered = df[
        (df['Year'] == str(year)) &
        (df['Pathogen'].str.contains(pathogen, case=False, na=False)) & 
        (df['Specimen'].str.contains(specimen, case=False, na=False)) &
        (df['Antibiotic'].str.contains(antibiotic, case=False, na=False))
    ]
    filtered = filtered.dropna(subset=['Resistance_Percentage'])
    if filtered.empty:
        return json.dumps({"error": f"No numerical data to compute average for {pathogen} against {antibiotic} in {year}."})
    
    avg_resistance = filtered['Resistance_Percentage'].mean()
    breakdown = filtered[['Location', 'Resistance_Percentage']].to_dict(orient="records")
    return json.dumps({
        "year": year,
        "pathogen": pathogen,
        "specimen": specimen,
        "antibiotic": antibiotic,
        "average_resistance_percentage": round(avg_resistance, 2),
        "breakdown": breakdown
    })

def track_resistance_trend(pathogen: str, specimen: str, location: str, antibiotic: str) -> str:
    """Tracks resistance percentage over multiple years (2021-2024) to show chronological trends."""
    filtered = df[
        (df['Pathogen'].str.contains(pathogen, case=False, na=False)) & 
        (df['Specimen'].str.contains(specimen, case=False, na=False)) &
        (df['Antibiotic'].str.contains(antibiotic, case=False, na=False))
    ]
    
    # Try to filter by location, fallback to all locations if exact match missing across all 4 years
    loc_filtered = filtered[filtered['Location'].str.contains(location, case=False, na=False)]
    if not loc_filtered.empty:
        filtered = loc_filtered
        
    if filtered.empty:
        return json.dumps({"error": "No trend data available for this selection."})
        
    trend = filtered.sort_values(by='Year', ascending=True)[['Year', 'Location', 'Resistance_Percentage']]
    return trend.to_json(orient="records")

# --- API Setup ---
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# --- Initialize Chat Session ---
if "chat_session" not in st.session_state:
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        tools=[get_best_antibiotic, get_all_pathogens, get_general_report, get_all_antibiotics, calculate_average_resistance, track_resistance_trend], 
        system_instruction=(
            "You are a clinical assistant analyzing Kerala AMR Surveillance Data from 2021 to 2024. "
            "When extracting arguments for tools, if the user mentions a specific year (2021, 2022, 2023, or 2024), pass it to the tool. If they don't specify a year, ALWAYS default to 2024. "
            "Use the 'track_resistance_trend' tool if a user asks how resistance has changed over time, if it is increasing/decreasing, or requests historical trends. "
            "If the user asks for a recommendation, use 'get_best_antibiotic'. "
            "For location, if it's not ICU, IPD, or OPD, try 'All' or 'Overall'. "
            "Always present trend timelines chronologically using bullet points or clean markdown tables. Be professional and accurate."
        )
    )
    st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)
    st.session_state.messages = []

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a clinical question (e.g., How has E. coli resistance to Meropenem changed over time?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing AMR data..."):
            response = st.session_state.chat_session.send_message(prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
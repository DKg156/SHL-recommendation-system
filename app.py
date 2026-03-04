import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SHL Recommender", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL")

st.title("SHL Assessment Recommendation Engine")

mode = st.radio(
    "Choose Input Type:",
    ["Text Query", "JD URL"],
    horizontal=True
)

query = ""
jd_url = ""

if mode == "Text Query":
    query = st.text_area("Enter job description or query", height=200)
else:
    jd_url = st.text_input("Enter JD URL")

if st.button("Recommend Assessments"):

    if mode == "Text Query":
        if not query.strip():
            st.warning("Please enter a job description or hiring requirement.")
            st.stop()
        payload = {"query": query.strip()}

    else:
        if not jd_url.strip():
            st.warning("Please enter a JD URL.")
            st.stop()
        payload = {"query": jd_url.strip()}

    try:
        with st.spinner("Finding best assessments..."):
            response = requests.post(
                f"{BACKEND_URL}/recommend",
                json=payload,
                timeout=60
            )

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
            st.code(response.text)
        else:
            data = response.json()
            recs = data.get("recommended_assessments", [])

            if not recs:
                st.info("No recommendations found.")
            else:
                st.subheader("Top Recommendations")

                rows = []
                for x in recs:
                    rows.append({
                        "Name": x.get("name", ""),
                        "URL": x.get("url", ""),
                        "Duration (mins)": x.get("duration", 0),
                        "Remote": x.get("remote_support", ""),
                        "Adaptive": x.get("adaptive_support", ""),
                        "Type": ", ".join(x.get("test_type", [])),
                    })

                df = pd.DataFrame(rows)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "URL": st.column_config.LinkColumn("URL"),
                        "Duration (mins)": st.column_config.NumberColumn(
                            "Duration (mins)",
                            help="Approximate completion time"
                        ),
                    },
                )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

import streamlit as st
import time

st.set_page_config(page_title="Domain Verifier", layout="centered")
st.title("Domain Verifier")

# Description/instructions
st.markdown(
    """
    <div style='margin-bottom: 20px; font-size: 16px;'>
    <b>Instructions:</b><br>
    1. Upload your domain list as a CSV file with a header named Domain<br>
    2. Indicate how many times you want the domain list to be processed.<br>
    3. Click <b>Start</b> to begin processing. You can stop anytime with the <b>Stop</b> button.<br>
    4. When processing is complete, click <b>Download Result</b> to download your processed file.
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload section
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Number input for processing times
num_times = st.number_input("How many times to process the file?", min_value=1, max_value=100, value=4)

# Start button (full width, green)
start = st.button(
    "Start",
    key="start_btn",
    help="Start processing",
    use_container_width=True,
)


# Row with Stop and Download Result buttons (50% each)
col_stop, col_download = st.columns(2)
with col_stop:
    stop = st.button(
        "Stop",
        key="stop_btn",
        help="Stop processing",
        use_container_width=True,
    )

with col_download:
    st.download_button(
        label="Download Result",
        data="domain,verified\nexample.com,True\nexample2.com,False",  # Dummy CSV data
        file_name="verified_domains.csv",
        mime="text/csv",
        help="Download the processed result after completion.",
        use_container_width=True,
    )

# Dummy progress data
progress = 0.65  # 65% done
current = 65
max_domains = 100

# Progress bar section
progress_bar = st.progress(progress)

# Progress bar indicators
col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown(f"<span style='font-size: 16px;'>{int(progress*100)}%</span>", unsafe_allow_html=True)
with col_right:
    st.markdown(f"<span style='font-size: 16px; float: right;'>{current}/{max_domains}</span>", unsafe_allow_html=True)

# Optionally, show a message for demo
st.info("This is a demo. Progress and numbers are dummy data.")
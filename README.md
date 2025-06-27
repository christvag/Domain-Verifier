# Domain Verifier Pro

A sophisticated, user-friendly web application for bulk verifying the reachability of website domains from a CSV file. Built with Streamlit, this tool provides real-time progress updates, detailed logging, and a persistent history of all past verification runs.

## ✨ Key Features

- **Intuitive Web Interface:** A simple, clean UI powered by Streamlit for easy file uploads and interaction.
- **Two-Pass Verification System:**
  - **Pass 1 (High-Speed HEAD Scan):** Quickly identifies easily reachable domains using a highly concurrent `HEAD` request scan.
  - **Pass 2 (Robust GET Scan):** Performs a more intensive `GET` request scan with retries for any domains that failed the first pass, ensuring maximum accuracy.
- **Real-Time Progress:** Watch the verification happen live with an auto-updating progress bar and activity log.
- **Persistent Run History:** All verification runs are automatically saved to a local SQLite database.
- **Downloadable Results:** Download a CSV of reachable domains or a full report for both the current run and any past run directly from the history table.
- **Asynchronous Backend:** Utilizes `asyncio` and `httpx` for high-performance, non-blocking domain checking.

## 🚀 Getting Started

Follow these instructions to get the Domain Verifier running on your local machine.

### Prerequisites

- Python 3.8 or newer
- `pip` (Python package installer)

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/Domain-Verifier.git
cd Domain-Verifier
```

### 2. Set Up a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to keep dependencies isolated.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Launch the Streamlit application with the following command:

```bash
streamlit run domainVerifier.py
```

Your web browser should automatically open with the application running. If not, navigate to the "Local URL" provided in your terminal (usually `http://localhost:8501`).

## 📖 How to Use

1.  **Upload CSV:** Click the "Browse files" button to upload your CSV file.
    - **Requirement:** Your CSV file must contain a column named `website` that lists the domains you want to verify.
2.  **Start Verification:** Click the **"Start Verification"** button. The button will become disabled, and the progress bar and activity log will appear.
3.  **Monitor Progress:** Watch the live updates to see the verification status.
4.  **View Results:** Once the process is complete, a "Verification complete!" message will appear.
    - The main "Download Result" button will provide the list of reachable domains from the most recent run.
    - The **"Past Runs"** table at the bottom will now include your latest run.
5.  **Download Past Results:** In the "Past Runs" table, you can click the download icon (⬇️) in the "Reachable File" or "Full Report" columns to download the results from any historical run.
6.  **Refresh History:** Click the "🔄 Refresh History" button at any time to manually reload the list of past runs.

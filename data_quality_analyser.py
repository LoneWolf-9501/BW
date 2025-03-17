import pandas as pd
import numpy as np
import os
from utils.df_utils import ODR
from utils.logger_tool import setup_logger
from datetime import datetime

LOG = setup_logger('data_quality')

TIME_COL = 'Timestamp'
INDEX_COL = 'Sensor_1'

REPORT = []  
REPORT_FILE = f'/home/lonewolf/coding/BRAINWIRED/Vetto/Init/reports/DATA_check_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def quality_checks(df: pd.DataFrame, df_name: str):
    """Perform quality checks on the given DataFrame and generate a REPORT."""
    global REPORT

    REPORT.append({"Check": "Filename", "Status": "ℹ️ Info", "Details": df_name})

    # Check for missing values
    nan_summary = df.isna().sum()
    total_nans = nan_summary.sum()

    if total_nans > 0:
        REPORT.append({
            "Check": "Missing Values",
            "Status": "⚠️ Issues Found",
            "Details": f"{total_nans} missing values in {nan_summary[nan_summary > 0].count()} columns"
        })
    else:
        REPORT.append({"Check": "Missing Values", "Status": "✅ OK", "Details": "No missing values detected"})

    # Convert timestamp column
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')

    # Log unparsed timestamps
    if df[TIME_COL].isna().any():
        missing_times = df[TIME_COL].isna().sum()
        REPORT.append({
            "Check": "Timestamp Parsing",
            "Status": "⚠️ Issues Found",
            "Details": f"{missing_times} timestamps could not be parsed"
        })

    # Sort by time and index column
    df = df.sort_values([TIME_COL, INDEX_COL]).reset_index(drop=True)

    # Check if index column is numeric
    if not pd.api.types.is_numeric_dtype(df[INDEX_COL]):
        df[INDEX_COL] = pd.to_numeric(df[INDEX_COL], errors='coerce')
        REPORT.append({
            "Check": "Index Column Type",
            "Status": "⚠️ Converted",
            "Details": f"'{INDEX_COL}' was non-numeric and has been converted"
        })

    # Compute index differences
    df["diff"] = df[INDEX_COL].diff()
    df.loc[0, "diff"] = 1.0

    # **🔹 Packet Loss Check**
    packet_loss_df = df[df["diff"] != 1]

    if not packet_loss_df.empty:
        packet_loss_indices = packet_loss_df.index.tolist()  # Get index positions
        REPORT.append({
            "Check": "Packet Loss",
            "Status": "❌ ERROR",
            "Details": f"Packet loss detected at indices: {packet_loss_indices}"
        })
    else:
        REPORT.append({"Check": "Packet Loss", "Status": "✅ OK", "Details": "No packet loss detected"})

    
    df["time"] = df[TIME_COL].dt.floor("s")
    odrs = ODR(df, sort_col="time")  

    unique_odrs = set(odrs['odr'].unique())  

    if unique_odrs == {24, 25, 26}:
        REPORT.append({"Check": "ODR", "Status": "✅ OK", "Details": f"Valid ODR values found:\n{odrs}\n"})
    else:
        REPORT.append({"Check": "ODR", "Status": "❌ ERROR", "Details": f"Invalid ODR values detected:\n{odrs}\n"})


    REPORT.append({"Check": "-"*16, "Status": "-"*10, "Details": "-" * 80})  

    LOG.info(f"✅ Quality checks completed for: {df_name}")



def folder_walker(file_dir):
    buffer = []
    """Walk through a folder and apply quality checks on each CSV file."""
    for root, _, files in os.walk(file_dir, topdown=True):
        for name in files:
            file = os.path.join(root, name)
            if os.path.isfile(file) and file.endswith(".csv"):
                try:
                    meta_df = pd.read_csv(file)
                    quality_checks(meta_df, name)
                    buffer.append(meta_df)
                except Exception as e:
                    LOG.error(f"❌ Error processing {file}: {e}")

    df = pd.concat(buffer, ignore_index=True)
    quality_checks(df, 'After merging')

def main():
    global REPORT

    folder_walker(file_dir='/home/lonewolf/coding/BRAINWIRED/Vetto/Init/Data/data_togo/')

    # Convert REPORT list to a DataFrame
    df_report = pd.DataFrame(REPORT)

    # Save report in multiple formats
    df_report.to_html(REPORT_FILE + ".html", index=False)
    df_report.to_markdown(REPORT_FILE + ".md", index=False)

    LOG.info(f"📊 Data Quality Report saved at {REPORT_FILE}.[html|md]")


if __name__ == "__main__":
    main()

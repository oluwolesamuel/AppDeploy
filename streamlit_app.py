import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF

# Title of the app
st.title("Portfolio Change Calculator")

manager_story = st.text_input("Tell us what changes you are making, cient name, porfolios you will use and peer for comparison")

text_results = []

tables = []

# Function to generate a PDF
def generate_pdf(text_results2, dataframes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add text results
    for text in text_results2:
        pdf.multi_cell(0, 10, text)
        pdf.ln(5)  # Add spacing

    # Add DataFrames
    for title, df in dataframes:
        pdf.ln(5)  # Add spacing before the table
        pdf.set_font("Arial", size=8)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", size=8)

        # Set custom width for the first column and distribute remaining width to other columns
        first_col_width = 45  # Make the first column wider (adjust as needed)
        remaining_width = pdf.w - 20 - first_col_width  # Subtract the margins and first column width
        col_width = remaining_width / (len(df.columns) - 1)  # Distribute remaining width evenly for other columns

        row_height = pdf.font_size + 2

        # Write header row
        pdf.cell(first_col_width, row_height, str(df.columns[0]), border=1)  # First column header
        for col in df.columns[1:]:
            pdf.cell(col_width, row_height, str(col), border=1)  # Other column headers
        pdf.ln(row_height)

        # Write data rows
        for row in df.itertuples(index=False):
            pdf.cell(first_col_width, row_height, str(row[0]), border=1)  # First column value
            for value in row[1:]:
                pdf.cell(col_width, row_height, str(value), border=1)  # Other column values
            pdf.ln(row_height)

    return pdf.output(dest='S').encode('latin1')

text_results.append(manager_story)

# Upload the Excel file

def dataload():
    url = "https://github.com/oluwolesamuel/AppDeploy/raw/refs/heads/main/appdata2.parquet"
    return pd.read_parquet(url)





#uploaded_file = st.file_uploader("Upload your Excel file", type=["parquet"])

uploaded_file = dataload()

if uploaded_file is not None:
    # Load the Excel file
    #df = pd.read_parquet(uploaded_file)

    df = uploaded_file

    df[['PortfolioName', 'EntityID', 'DataSource', 'AssetClass', 'PortfolioManager', 'Type', 'ReturnType']] = df[
        ['PortfolioName', 'EntityID', 'DataSource', 'AssetClass', 'PortfolioManager', 'Type', 'ReturnType']
    ].astype("string")
    df['Return'] = df['Return'].fillna(0)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Step 1: User input for fund selection      
    portfolio_names = df['PortfolioName'].unique()

    records = df['PortfolioName'].value_counts()

    record_counts = [f"{name}|{count}" for name, count in records.items()]

    recent_date = df['ReturnDate'].max()

    st.write(f"The newest date in the dataset is {recent_date}.")

    
    # Search input box
    search_query = st.text_input("Search for portfolio names:")

    # Filter portfolio names based on the search query
    if search_query:
        filtered_portfolios = [name for name in record_counts if name.lower().startswith(search_query.lower())]
    else:
        filtered_portfolios = record_counts  # Show all if no search query

# Display filtered options in a multiselect box for the user to select
    selected_funds = st.multiselect("Select funds from the list:", filtered_portfolios)

    selected_funds2 = [fund.split("|")[0] for fund in selected_funds]

# Show selected funds and filter the dataframe
    if selected_funds2:
        
        funds_df = df[df['PortfolioName'].isin(selected_funds2)]
        

        fund_counts = funds_df['PortfolioName'].value_counts().reset_index()

        fund_counts.rename(columns={'index': 'PortfolioName', 'count': 'History'}, inplace=True)


        Months_of_history = fund_counts
              
       
        # Step 2: Confirm if the user is satisfied with the selection
        satisfied = st.radio("Are you satisfied with these results?", options=["Yes", "No"])

        if satisfied == "Yes":
            # Allow user to proceed
            st.success("Great! Let's proceed.")

            Months_of_history['RM_Weight'] = None
            Months_of_history['CW_Weight'] = None
            Months_of_history['PW_Weight'] = None
            Months_of_history["Fund_Score"] = None

            # Step 3: Input weights for RMs, CWs, and PWs
            st.write("Input weights for RMs, CWs, and PWs (values must sum to 100 for each column):")

            updated_data = st.data_editor (
                Months_of_history,
                column_config = {
                    "RM_Weight": st.column_config.NumberColumn(label="RM Weight"),
                    "CW_Weight": st.column_config.NumberColumn(label="CW Weight"),
                    "PW_Weight": st.column_config.NumberColumn(label="PW Weight"),
                    "Fund_Score": st.column_config.NumberColumn(label="Fund Score"),
                },
                key="portfolio_weights_editor"
            )    

            if updated_data.isnull().values.any():
                st.warning("Please fill in all values")  

            else:
                st.success("All data has been entered.")    
                updated_data['ABS_CWVSRM'] = updated_data['CW_Weight']-updated_data['RM_Weight']
                updated_data['ABS_PWVSRM'] = updated_data['PW_Weight']-updated_data['RM_Weight']

            st.write(updated_data)

            tables.append(("Input Table", updated_data))

                        
            
            rm_weights = updated_data['RM_Weight'] #st.text_input("Enter RM weights as a comma-separated list:")
            cw_weights = updated_data['CW_Weight'] #st.text_input("Enter CW weights as a comma-separated list:")
            pw_weights = updated_data['PW_Weight'] #st.text_input("Enter PW weights as a comma-separated list:")
            

            vol_month = st.text_input("Enter the historical volatility periods you'd like to compute as a comma-seperated list:")

            
            if (rm_weights>0).all() and (cw_weights>0).all() and (pw_weights>0).all():

                
                
                try:
                    rm_weights = updated_data['RM_Weight'].values / 100 #np.array([float(w) for w in rm_weights.split(",")]) / 100
                    cw_weights = updated_data['CW_Weight'].values / 100 #np.array([float(w) for w in cw_weights.split(",")]) / 100
                    pw_weights = updated_data['PW_Weight'].values / 100 #np.array([float(w) for w in pw_weights.split(",")]) / 100

                    # Pivot the data
                    funds_pivot = funds_df.pivot(index="ReturnDate", columns="PortfolioName", values="Return").reset_index()
                    funds_pivot.columns.name = None
                    funds_pivot = funds_pivot.sort_values(by="ReturnDate", ascending=False)
                    fund_returns = funds_pivot.iloc[:, 1:].to_numpy()

                    # Perform calculations
                    funds_pivot['RM'] = np.dot(fund_returns, rm_weights)
                    funds_pivot['CW'] = np.dot(fund_returns, cw_weights)
                    funds_pivot['PW'] = np.dot(fund_returns, pw_weights)

                    

                    # Benchmark fund input
                                       
                    benchmark = st.selectbox(
                        "Search and select your peer fund:",
                        options=df['PortfolioName'].unique(),
                        format_func=lambda x: x if isinstance(x, str) else ""
                    )
                    
                    
                    if benchmark:
                        bm_df = df[df['PortfolioName'].str.lower() == benchmark.lower()]
                        if not bm_df.empty:
                            bm_pivot = bm_df.pivot(index='ReturnDate', columns='PortfolioName', values='Return').reset_index()
                            bm_pivot.columns.name = None
                            bm_pivot = bm_pivot.sort_values(by="ReturnDate", ascending=False)

                            # Merge tables
                            full_table = pd.merge(funds_pivot, bm_pivot, how='inner', on='ReturnDate')
                            bm_name = bm_pivot.columns[1]

                            # Calculate active values
                            full_table['RM_active'] = full_table['RM'] - full_table[bm_name]
                            full_table['CW_active'] = full_table['CW'] - full_table[bm_name]
                            full_table['PW_active'] = full_table['PW'] - full_table[bm_name]
                            full_table['CWvRM'] = full_table['CW_active'] - full_table['RM_active']
                            full_table['PWvRM'] = full_table['PW_active'] - full_table['RM_active']

                            
                            # Volatility calculation
                            def std_dev(df, value_col, num_months):
                                nearest_date = df['ReturnDate'].max()
                                date_start = nearest_date - pd.DateOffset(months=num_months)
                                filtered_df = df[(df['ReturnDate'] >= date_start) & (df['ReturnDate'] <= nearest_date)]
                                return round(filtered_df[value_col].std() * (12**0.5) * 100,4) if not filtered_df.empty else None

                            
                            vol_periods = np.array([int(w) for w in vol_month.split(",")])
                            vol_table = pd.DataFrame({
                                "VolatilityPeriod": [f"{m}m" for m in vol_periods],
                                "RM": [std_dev(full_table, 'RM', m) for m in vol_periods],
                                "CW": [std_dev(full_table, 'CW', m) for m in vol_periods],
                                "PW": [std_dev(full_table, 'PW', m) for m in vol_periods],
                                "RM_active": [std_dev(full_table, 'RM_active', m) for m in vol_periods],
                                "CW_active": [std_dev(full_table, 'CW_active', m) for m in vol_periods],
                                "PW_active": [std_dev(full_table, 'PW_active', m) for m in vol_periods],
                                "CWvRM": [std_dev(full_table, 'CWvRM', m) for m in vol_periods],
                                "PWvRM": [std_dev(full_table, 'PWvRM', m) for m in vol_periods],
                            })

                            st.write("Volatility Table")

                            st.dataframe(vol_table)

                            tables.append(("Volatility Table",vol_table))

                        else:
                            st.error("Benchmark fund not found.")                    

                except Exception as e:
                    st.error(f"Error: {e}")
        
        
        else:
            st.warning("Please reselect your funds and try again.")

    if st.button("Download Results"):
            pdf_data = generate_pdf(text_results, tables)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="results.pdf",
                mime="application/pdf"
                             )
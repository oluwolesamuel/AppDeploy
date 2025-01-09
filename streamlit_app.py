import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF

# Title of the app
st.title("Portfolio Change Calculator")

doc_title = st.text_input("Enter the document title.")

model_name = st.text_input("Enter your model name.")

ref_model = st.text_input("What is your Reference Model?")

peer_group = st.text_input("What peer group are you using?")

manager_name = st.text_input("Enter manager name.")

analysis_date = st.text_input("What is the analyis date?")

model_exp = st.text_input("Explain the model.")

text_results = []

tables = []

# Function to generate a PDF
def generate_pdf(pdf_title, text_results2, dataframes):
    """
    Generates a PDF with a title, text results (each with a title), and dataframes.

    Parameters:
    - pdf_title: Title of the PDF document.
    - text_results: A list of tuples (title, text), where each tuple contains a title for the text.
    - dataframes: A list of tuples (title, dataframe), where each tuple contains a title for the DataFrame.

    Returns:
    - PDF content as a bytes object.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Add PDF title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, pdf_title, ln=True, align="C")
    pdf.ln(10)  # Add spacing after the title

    # Add text results with titles
    for title, text in text_results2:
        pdf.set_font("Arial", style="B", size=12)  # Bold font for the title
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(1)  # Add some space after the title

        pdf.set_font("Arial", size=10)  # Normal font for the text
        pdf.multi_cell(0, 10, text)
        pdf.ln(5)  # Add spacing after each text block

    # Add DataFrames with titles
    for title, df in dataframes:
        pdf.ln(5)  # Add spacing before the table
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", size=8)

        # Set custom width for the first column and distribute remaining width to other columns
        first_col_width = 45  # Adjust as needed
        remaining_width = pdf.w - 20 - first_col_width
        col_width = remaining_width / (len(df.columns) - 1)

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

    # Return the PDF as a byte object
    return pdf.output(dest="S").encode("latin1")

text_results.append(("Model Name",model_name))
text_results.append(("Reference Model",ref_model))
text_results.append(("Peer Group",peer_group))
text_results.append(("Manager Name", manager_name))
text_results.append(("Date of Analysis",analysis_date))
text_results.append(("Model Explanation", model_exp))


def dataload():
    url = "https://github.com/oluwolesamuel/AppDeploy/raw/refs/heads/main/appdata2.parquet"
    return pd.read_parquet(url)



uploaded_file = dataload()

if uploaded_file is not None:
    
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

            Months_of_history['RM_Weight'] = 0.00
            Months_of_history['CW_Weight'] = 0.00
            Months_of_history['PW_Weight'] = 0.00
            Months_of_history["Fund_Score"] = 0.00

            # Step 3: Input weights for RMs, CWs, and PWs
            st.write("Input weights for RMs, CWs, and PWs (values must sum to 100 for each column):")

            ## st.session to ensure too many reruns are prevented

            #if "updated_data" not in st.session_state:
            st.session_state["updated_data"] = Months_of_history.copy()

            st.session_state["updated_data"] = st.data_editor(
                st.session_state["updated_data"],
                column_config = {
                    "RM_Weight": st.column_config.NumberColumn(label="RM Weight", format="%.2f"),
                    "CW_Weight": st.column_config.NumberColumn(label="CW Weight", format="%.2f"),
                    "PW_Weight": st.column_config.NumberColumn(label="PW Weight", format="%.2f"),
                    "Fund_Score": st.column_config.NumberColumn(label="Fund Score", format="%.2f"),
                },
                key="portfolio_weights_editor"

            )     

            ## input validation button

            if st.button("Validate input"):
                updated_data = st.session_state["updated_data"]
                if ( (updated_data["RM_Weight"].sum() == 100) and (updated_data["CW_Weight"].sum() == 100) and (updated_data["PW_Weight"].sum() == 100) and (updated_data["Fund_Score"] > 0).all() ):
                    
                    st.success("All data has been entered correctly.")
                    updated_data["ABS_CWVSRM"] = abs(updated_data["CW_Weight"] - updated_data["RM_Weight"])
                    updated_data["ABS_PWVSRM"] = abs(updated_data["PW_Weight"] - updated_data["RM_Weight"])
                    st.write(updated_data)
                else:
                    st.warning("Some values have not been entered correctly. Please check your input.")   


            updated_data = st.session_state["updated_data"]

        
            #

            measures_table = pd.DataFrame({
                'Measure' : ['SA Equity(Hist/Curr)', 'Global Equity(Hist/Curr)', 'Total Global(Hist/Curr)',
                'RR EXP excl TAA Alpha', 'TAA Alpha', 'EXP Return Net Fees'],
                'RM' : [0,0,0,0,0,0],
                'Current' : [0,0,0,0,0,0],
                'Proposed' : [0,0,0,0,0,0]

                })

                       
            measures_table['RM'] = 0.00
            measures_table['Current'] = 0.00
            measures_table['Proposed'] = 0.00

            measures_table2 = st.data_editor( measures_table,
            column_config = {
                'RM' : st.column_config.NumberColumn(label="RM", format = "%.2f"),
                'Current' : st.column_config.NumberColumn(label="Current", format = "%.2f"),
                'Proposed' : st.column_config.NumberColumn(label="Proposed", format = "%.2f"),
            },
            key="weights_editor"
            )         

            sum_row = measures_table2.loc[3, ['RM', 'Current', 'Proposed']] + measures_table2.loc[4, ['RM', 'Current', 'Proposed']]

            new_row = pd.DataFrame([{'Measure':'Total RR EXP', 'RM':sum_row['RM'], 
            'Current':sum_row['Current'], 'Proposed':sum_row['Proposed']}])

            measures_table2 = pd.concat([measures_table2,new_row], ignore_index = True)


            weighted = pd.DataFrame({
                'Measure' : ['WeightedAVGScore'],
                'RM' : [0],
                'Current' : [0],
                'Proposed' : [0]
            })

            #pd.DataFrame(weighted)

            

            weighted['RM'] = round((np.dot((updated_data['RM_Weight'].values/100),(updated_data['Fund_Score'].values/100)))*100,4)
            weighted['Current'] = round((np.dot((updated_data['CW_Weight'].values/100),(updated_data['Fund_Score'].values/100)))*100,4)
            weighted['Proposed'] = round((np.dot((updated_data['PW_Weight'].values/100),(updated_data['Fund_Score'].values/100)))*100,4)

            measures_table3 = pd.concat([measures_table2,weighted], ignore_index = True)

            measures_table3['P-RM'] = round((measures_table3['Proposed'] - measures_table3['RM']),4)

            measures_table3['P-CW'] = round((measures_table3['Proposed'] - measures_table3['Current']),4)

            st.write(measures_table3)

            tables.append(("Input Table", updated_data))

            tables.append(("Other Measures", measures_table3))                                   
            
            rm_weights = updated_data['RM_Weight'] #st.text_input("Enter RM weights as a comma-separated list:")
            cw_weights = updated_data['CW_Weight'] #st.text_input("Enter CW weights as a comma-separated list:")
            pw_weights = updated_data['PW_Weight'] #st.text_input("Enter PW weights as a comma-separated list:")
            

            vol_month = st.text_input("Enter the historical volatility periods you'd like to compute as a comma-seperated list:")

            benchmark = st.selectbox(
                        "Search and select your peer fund:",
                        options=df['PortfolioName'].unique(),
                        format_func=lambda x: x if isinstance(x, str) else ""
                    )           
            

            if st.button("Proceed with Calculations"):
                              
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

                    vol_table = pd.DataFrame()

                    # Benchmark fund input                 
                                       
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
            pdf_data = generate_pdf(doc_title,text_results, tables)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="results.pdf",
                mime="application/pdf"
                             )

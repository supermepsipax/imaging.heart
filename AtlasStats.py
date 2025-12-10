import pandas as pd
import os
# -- no intermediate artery --
table5_data = [
    ["LEFT MAIN DIAM., MM", 3.5, 0.8, 3.2, 3.6, 4.1],
    ["LAD DIAM., MM",       3.2, 0.7, 2.8, 3.2, 3.6],
    ["LCX DIAM., MM",       3.0, 0.7, 2.5, 3.1, 3.5],
    ["ANGLE A, °",          126.7, 21.4, 115.8, 129.8, 141.6],
    ["ANGLE B, °",          75.2, 23.3, 61.4, 73.1, 88.9],
    ["ANGLE C, °",          138.6, 20.0, 132.0, 140.2, 150.2],
    ["INFLOW ANGLE, °",     8.3, 24.6, -8.1, 10.4, 25.8],
    ["LEFT MAIN LENGTH, MM",10.2, 5.6, 7.0, 10.0, 13.0],
]
columns = ["Measurement", "Mean", "SD", "Q1", "Q2", "Q3"]
df5 = pd.DataFrame(table5_data, columns=columns)

# -- with intermediate artery (trifurcation) --
table6_data = [
    ["LEFT MAIN DIAM., MM", 3.5, 0.8, 3.0, 3.6, 4.1],
    ["LAD DIAM., MM",       3.1, 0.9, 2.4, 3.3, 3.8],
    ["LCX DIAM., MM",       2.9, 0.8, 2.4, 2.9, 3.4],
    ["INT. DIAM., MM",      2.6, 0.5, 2.1, 2.5, 2.9],
    ["ANGLE A, °",          120.0, 18.5, 112.6, 118.1, 130.9],
    ["ANGLE B1, °",         58.4, 19.4, 41.4, 60.0, 72.8],
    ["ANGLE B2, °",         46.1, 14.5, 35.4, 45.4, 56.7],
    ["ANGLE B, °",          88.6, 21.1, 77.4, 89.0, 100.6],
    ["ANGLE C, °",          136.6, 14.1, 127.7, 137.6, 147.6],
    ["ANGLE D, °",          141.8, 22.4, 130.7, 142.6, 158.5],
    ["INFLOW ANGLE, °",     14.6, 23.8, -3.9, 20.0, 34.5],
    ["LEFT MAIN LENGTH, MM",11.3, 4.4, 8.0, 10.0, 14.0],
]
df6 = pd.DataFrame(table6_data, columns=columns)


def save_reference_tables(output_folder="results", filename="reference_coronary.xlsx"):
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)

    # Add titles for clarity
    df5_with_title = pd.concat([
        pd.DataFrame([["--- BIFURCATION ---"] + [""] * (len(columns)-1)], columns=columns),
        df5
    ], ignore_index=True)

    df6_with_title = pd.concat([
        pd.DataFrame([["--- TRIFURCATION ---"] + [""] * (len(columns)-1)], columns=columns),
        df6
    ], ignore_index=True)

    # Blank row between tables
    blank_row = pd.DataFrame([[""] * len(columns)], columns=columns)

    # Combine into one sheet
    combined = pd.concat([df5_with_title, blank_row, df6_with_title], ignore_index=True)

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        combined.to_excel(writer, sheet_name="ReferenceTables", index=False)

    print(f"Saved reference tables to {filepath}")


if __name__ == "__main__":
    save_reference_tables()

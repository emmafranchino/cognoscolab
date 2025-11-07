import os

def export_df(df, export_path, filename: str, overwrite=False, extension='csv'):
    '''Export the DataFrame to a file in the specified export path.

    Parameters:
    - df: pandas DataFrame to be exported.
    - export_path: str, path to the directory where the file will be saved.
    - overwrite: bool, whether to overwrite existing files (default is False).
    - filename: str, name of the output file (default is None).
    - extension: str, file extension (default is 'csv').
    '''
    # ensure the output folder exists else create it
    os.makedirs(export_path, exist_ok=True)

    # check if the file already exists
    if filename and os.path.exists(os.path.join(export_path, f"{filename}.{extension}")):
        if not overwrite:
            print(f"File {filename}.{extension} already exists in {export_path}. Use overwrite=True to replace it.")
            return
        else:
            print(f"Overwriting existing file {filename}.{extension} in {export_path}.")

    # save the df to a file in the output folder
    output_file = os.path.join(export_path, f"{filename}.{extension}")
    if extension == 'csv':
        df.to_csv(output_file, index=False)
    elif extension == 'xlsx':
        df.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported file extension. Use 'csv' or 'xlsx'.")
    print(f"Data saved to {output_file}\n")
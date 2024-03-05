import csv

# Specify the input and output file paths
input_file_path = '/Users/jackmorse/Documents/University/Year 4/FYP/Physics-FYP/Data Files/21-Feb-2024/HCF_120cm_16of40_61510um_Jack_21022024_1019.txt'
output_file_path = '/Users/jackmorse/Documents/University/Year 4/FYP/Physics-FYP/Data Files/21-Feb-2024/csv/HCF_120cm_16of40_61510um_Jack_21022024_1019.csv'

# Read the data from the text file
with open(input_file_path, 'r') as infile:
    # Assuming the data is space-separated, you can use space as the delimiter
    data = [line.split() for line in infile]

# Write the data to a CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print(f"CSV file '{output_file_path}' has been created from '{input_file_path}'.")
import os
if not os.path.exists(output_file_path):
    print("Doesnt actually exist")
if os.path.exists(output_file_path):
    print(f"Exists as {output_file_path}")

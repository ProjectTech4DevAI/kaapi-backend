Upload a CSV or Excel dataset for assessment workflows.

The file is stored in object storage and indexed as an assessment dataset
for the current organization and project.

Excel files are cleaned on read: blank rows are dropped, then any column with no
header or no values. `total_items`, the preview and the run all see the cleaned
sheet, so a 1000-row sheet with 100 filled rows reports `total_items: 100`.

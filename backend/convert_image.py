
import json, csv

with open('./app/inquilab/summary_output_imagev1/google/gemini-2.5-flash/gemini-2.5-flash.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tsv_path = './app/inquilab/summary_output_imagev1/google/gemini-2.5-flash/gemini-2.5-flash.tsv'

with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['CID', 'Problem', 'Solution', 'Image URL', 'Image Summary', 'Idea Summary'])
    
    for entry in data:
        image_surmmary_value = entry.get('Image Summary', '')
        idea_summary_value = entry.get('Idea Summary', '')
        writer.writerow([
            entry.get('CID', ''),
            entry.get('Problem', ''),
            entry.get('Solution', ''),
            entry.get('Image URL', ''),
            entry.get('Image Summary', ''),
            entry.get('Idea Summary', ''),
        ])

print(f'Done. Wrote {len(data)} rows with prettified JSON.')
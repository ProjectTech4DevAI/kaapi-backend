
import json, csv

with open('./app/inquilab/output_image/google/gemini-3.1-flash-lite-preview/gemini-3.1-flash-lite-preview.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tsv_path = './app/inquilab/output_image/google/gemini-3.1-flash-lite-preview/gemini-3.1-flash-lite-preview.tsv'

with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['CID', 'Problem', 'Solution', 'Image URL', 'Image Summary'])
    
    for entry in data:
        image_surmmary = entry.get('Image Summary', {})
        if image_surmmary:
            image_surmmary_value = image_surmmary.get("image_summary")
        
        writer.writerow([
            entry.get('CID', ''),
            entry.get('Problem', ''),
            entry.get('Solution', ''),
            entry.get('Image URL', ''),
            image_surmmary_value
        ])

print(f'Done. Wrote {len(data)} rows with prettified JSON.')
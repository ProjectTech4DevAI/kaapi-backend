
import json, csv

with open('./app/inquilab/output_multiple_imagesv1/google/gemini-2.5-flash/gemini-2.5-flash.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tsv_path = './app/inquilab/output_multiple_imagesv1/google/gemini-2.5-flash/gemini-2.5-flash.tsv'

with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['CID', 'Problem', 'Solution', 'Image URL', 'scores_with_images'])
    
    for entry in data:
        scores = entry.get('scores_with_images') or {}
        scores_obj = {}
        for key in ['Novelty', 'Usefulness', 'Feasibility', 'Scalability', 'Sustainability', 'Image_contains_summary']:
            if key in scores:
                scores_obj[key] = scores[key]
        
        writer.writerow([
            entry.get('CID', ''),
            entry.get('Problem', ''),
            entry.get('Solution', ''),
            json.dumps(entry.get('Image URL', []), ensure_ascii=False),
            json.dumps(scores_obj, indent=4, ensure_ascii=False)
        ])

print(f'Done. Wrote {len(data)} rows with prettified JSON.')
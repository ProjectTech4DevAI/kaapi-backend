
import json, csv

with open('./app/inquilab/output_multi/openai/gpt-4o-mini-wo-images/gpt-4o-mini-wo-images.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tsv_path = './app/inquilab/output_multi/openai/gpt-4o-mini-wo-images/gpt-4o-mini-wo-images.tsv'

with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['scores_without_images'])
    
    for entry in data:
        scores = entry.get('scores_without_images', {})
        scores_obj = {}
        for key in ['Novelty', 'Usefulness', 'Feasibility', 'Scalability', 'Sustainability']:
            if key in scores:
                scores_obj[key] = scores[key]
        
        writer.writerow([
            json.dumps(scores_obj, indent=4, ensure_ascii=False)
        ])

print(f'Done. Wrote {len(data)} rows with prettified JSON.')
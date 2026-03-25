
import json, csv, sys, glob

if len(sys.argv) < 2:
    print("Usage: python convert_feedback.py <path_to_json>")
    sys.exit(1)

json_path = sys.argv[1]

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

tsv_path = json_path.replace('.json', '.tsv')

criteria = ['Novelty', 'Usefulness', 'Feasibility', 'Scalability', 'Sustainability']

with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    header = []
    for c in criteria:
        header.extend([c, 'Reason'])
    header.extend(['Attachment Summary', 'Idea Feedback'])
    writer.writerow(['CID'] + header)

    for entry in data:
        row = [entry.get('CID', '')]
        for c in criteria:
            obj = entry.get(c, {})
            if isinstance(obj, dict):
                row.append(obj.get('score', ''))
                row.append(obj.get('reason', ''))
            else:
                row.extend(['', ''])
        row.append(entry.get('Attachment Summary', ''))
        row.append(entry.get('Idea Feedback', ''))
        writer.writerow(row)

print(f'Done. Wrote {len(data)} rows to {tsv_path}')

"""
Combines with-attachments and without-attachments JSON results into a single TSV
with the structure:
  Idea Details | Without Attachments | With Attachments

Usage:
  python convert_feedback_combined.py <with_images_json> <wo_images_json>

Output TSV is written next to the with-images JSON file.
"""

import json, csv, sys

if len(sys.argv) < 3:
    print("Usage: python convert_feedback_combined.py <with_images_json> <wo_images_json>")
    sys.exit(1)

with_path = sys.argv[1]
wo_path = sys.argv[2]

with open(with_path, "r", encoding="utf-8") as f:
    with_data = json.load(f)

with open(wo_path, "r", encoding="utf-8") as f:
    wo_data = json.load(f)

# Index wo_data by CID
wo_by_cid = {str(entry.get("CID", "")): entry for entry in wo_data}

tsv_path = with_path.replace(".json", ".tsv")

criteria = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]

score_headers = []
for c in criteria:
    score_headers.extend([c, "Reason"])
score_headers.extend(["Attachment Summary", "Idea Feedback"])

with open(tsv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")

    # Row 1: group headers
    idea_details_cols = 4  # CID, Problem, Solution, Attachments url
    score_cols = len(score_headers)
    group_row = ["Idea Details"] + [""] * (idea_details_cols - 1)
    group_row += ["Without Attachments"] + [""] * (score_cols - 1)
    group_row += ["With Attachments"] + [""] * (score_cols - 1)
    writer.writerow(group_row)

    # Row 2: column headers
    header = ["CID", "Problem", "Solution", "Attachments url"]
    header += score_headers  # without attachments
    header += score_headers  # with attachments
    writer.writerow(header)

    for entry in with_data:
        cid = str(entry.get("CID", ""))
        wo_entry = wo_by_cid.get(cid, {})

        # Idea details
        row = [
            cid,
            entry.get("Problem", ""),
            entry.get("Solution", ""),
            entry.get("Image URL", ""),
        ]

        # Without attachments scores
        for c in criteria:
            obj = wo_entry.get(c, {})
            if isinstance(obj, dict):
                row.append(obj.get("score", ""))
                row.append(obj.get("reason", ""))
            else:
                row.extend(["", ""])
        row.append(wo_entry.get("Attachment Summary", ""))
        row.append(wo_entry.get("Idea Feedback", ""))

        # With attachments scores
        for c in criteria:
            obj = entry.get(c, {})
            if isinstance(obj, dict):
                row.append(obj.get("score", ""))
                row.append(obj.get("reason", ""))
            else:
                row.extend(["", ""])
        row.append(entry.get("Attachment Summary", ""))
        row.append(entry.get("Idea Feedback", ""))

        writer.writerow(row)

print(f"Done. Wrote {len(with_data)} rows to {tsv_path}")

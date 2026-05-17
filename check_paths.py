import json
from pathlib import Path

# Load the groups data
with open('data/scans/ViberPhotos_20260508_074902/groups.json', 'r') as f:
    groups = json.load(f)

print('First few groups and their image paths:')
for i, group in enumerate(groups[:3]):
    print(f'Group {i}: {group["group_id"]}')
    for img in group['images'][:2]:  # Show first 2 images per group
        print(f'  {img}')
    print()
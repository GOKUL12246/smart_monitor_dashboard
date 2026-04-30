import json
import os

with open('missing_persons_faces/metadata.json') as f:
    data = json.load(f)
    
print('Database Contents:')
for person in data:
    print(f"Name: {person.get('name')}")
    print(f"ID: {person.get('person_id')}")
    print(f"Has DeepFace: {person.get('deepface_embedding') is not None}")
    print()

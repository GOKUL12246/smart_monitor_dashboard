#!/usr/bin/env python
"""Check if any faces are in the database"""

from missing_person_database import MissingPersonDatabase
import os

print("\n" + "=" * 60)
print("DATABASE STATUS CHECK")
print("=" * 60)

db = MissingPersonDatabase()
persons = db.get_all_persons()

print(f"\nTotal persons in database: {len(persons)}")

if len(persons) == 0:
    print("\n⚠️  DATABASE IS EMPTY!")
    print("\nYou need to upload your photo:")
    print("1. Go to http://localhost:5000")
    print("2. Click 'Missing Person' tab")
    print("3. Click 'Upload Missing Person Photo'")
    print("4. Select a clear photo of your face")
    print("5. Enter your name")
    print("6. Click Upload")
else:
    print("\n[OK] People found in database:\n")
    for i, person in enumerate(persons, 1):
        name = person.get('name', 'Unknown')
        person_id = person.get('person_id', 'unknown')
        features = person.get('features_count', 0)
        has_deepface = person.get('has_deepface', False)
        
        print(f"{i}. {name}")
        print(f"   ID: {person_id}")
        print(f"   Features: {features}")
        print(f"   DeepFace: {'YES' if has_deepface else 'NO'}")

# Check files
print("\n" + "=" * 60)
print("FILE STATUS")
print("=" * 60)

base_dir = "missing_persons_faces"
files_to_check = [
    "metadata.json",
    "embeddings.npy",
    "deepface_embeddings.npy",
    "detection_history.json"
]

for filename in files_to_check:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"[OK] {filename}: {size} bytes")
    else:
        print(f"[MISSING] {filename}: NOT FOUND")

print("\n" + "=" * 60)

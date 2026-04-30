#!/usr/bin/env python
import json
import os
import numpy as np

db_dir = 'missing_persons_faces'
metadata_file = os.path.join(db_dir, 'metadata.json')
embeddings_file = os.path.join(db_dir, 'embeddings.npy')
deepface_file = os.path.join(db_dir, 'deepface_embeddings.npy')

print('=' * 60)
print('DATABASE DIAGNOSTIC')
print('=' * 60)
print(f'Database dir exists: {os.path.exists(db_dir)}')
print(f'Metadata file exists: {os.path.exists(metadata_file)}')
print(f'ORB embeddings file exists: {os.path.exists(embeddings_file)}')
print(f'DeepFace embeddings file exists: {os.path.exists(deepface_file)}')

if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        data = json.load(f)
        persons = data.get('persons', [])
        print(f'\n✅ PERSONS IN DATABASE: {len(persons)}')
        if len(persons) == 0:
            print('   ❌ NO PERSONS! Need to upload photo to missing_person page!')
        else:
            for p in persons:
                print(f'   - {p.get("name")}: {p.get("face_size")} face, {p.get("features_count")} features')
else:
    print('\n❌ No metadata file found! Database is empty!')

if os.path.exists(embeddings_file):
    emb = np.load(embeddings_file, allow_pickle=True)
    print(f'\n✅ ORB embeddings: {len(emb)} persons')
    for i, e in enumerate(emb):
        status = 'OK' if e is not None else 'MISSING'
        print(f'   Person {i}: {status}')
else:
    print('\n❌ No ORB embeddings file!')

print('\n' + '=' * 60)
print('ACTION: Go to http://localhost:5000/missing_person')
print('Upload your clear photo -> it should appear in this database!')
print('=' * 60)

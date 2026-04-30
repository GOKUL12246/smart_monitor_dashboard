#!/usr/bin/env python3
"""
Check what was uploaded to the database
"""
import json
import os
import numpy as np

print("="*70)
print("CHECKING DATABASE FOR YOUR UPLOAD")
print("="*70)

# Check metadata
if os.path.exists('missing_persons_faces/metadata.json'):
    with open('missing_persons_faces/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\nTotal persons in database: {len(metadata)}")
    for i, person in enumerate(metadata, 1):
        print(f"\n{i}. {person.get('name', 'Unknown')}")
        print(f"   ID: {person.get('person_id', 'N/A')}")
        print(f"   Has DeepFace embedding: {person.get('deepface_embedding') is not None}")
        if person.get('deepface_embedding'):
            emb = person.get('deepface_embedding')
            if isinstance(emb, list):
                print(f"   Embedding shape: {len(emb)}D")
            else:
                print(f"   Embedding type: {type(emb)}")
else:
    print("\nERROR: No database found!")

# Check embeddings file
print("\n" + "="*70)
print("EMBEDDINGS FILE STATUS")
print("="*70)

if os.path.exists('missing_persons_faces/embeddings.npy'):
    try:
        emb_data = np.load('missing_persons_faces/embeddings.npy', allow_pickle=True)
        print(f"ORB Embeddings loaded: {len(emb_data)} persons")
        for i, item in enumerate(emb_data):
            if item is not None:
                print(f"  Person {i}: {len(item)} features" if hasattr(item, '__len__') else f"  Person {i}: {type(item)}")
    except Exception as e:
        print(f"ERROR loading embeddings: {e}")
else:
    print("ERROR: embeddings.npy not found!")

if os.path.exists('missing_persons_faces/deepface_embeddings.npy'):
    try:
        deep_data = np.load('missing_persons_faces/deepface_embeddings.npy', allow_pickle=True)
        print(f"\nDeepFace Embeddings loaded: {len(deep_data)} persons")
        for i, item in enumerate(deep_data):
            if item is not None:
                if isinstance(item, np.ndarray):
                    print(f"  Person {i}: {item.shape} deepface vector")
                else:
                    print(f"  Person {i}: {type(item)}")
            else:
                print(f"  Person {i}: None (not available)")
    except Exception as e:
        print(f"ERROR loading deepface embeddings: {e}")
else:
    print("\nERROR: deepface_embeddings.npy not found!")

print("\n" + "="*70)

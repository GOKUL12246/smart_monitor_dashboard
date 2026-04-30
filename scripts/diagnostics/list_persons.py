import json

d = json.load(open('missing_persons_faces/metadata.json'))
print('Uploaded persons:')
for p in d.get('persons', []):
    name = p.get('name')
    has_deep = p.get('deepface_embedding') is not None
    print(f'  - {name}: DeepFace={has_deep}')

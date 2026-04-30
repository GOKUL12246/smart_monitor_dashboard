import requests
import time

time.sleep(1)

session = requests.Session()
session.post('http://localhost:5000/login', data={'username': 'admin', 'password': 'temple2026'}, allow_redirects=False)
mp_response = session.get('http://localhost:5000/missing_person')

print(f'Status: {mp_response.status_code}')
print(f'Contains "Upload Photo": {"Upload Photo" in mp_response.text}')
print(f'Contains uploadForm: {"uploadForm" in mp_response.text}')
print(f'Contains photoInput: {"photoInput" in mp_response.text}')
print(f'Page length: {len(mp_response.text)} chars')

if 'Upload Photo' in mp_response.text:
    print('\n✅ Upload form is visible on the page!')
    # Find and show the section
    idx = mp_response.text.find('Upload Photo')
    section = mp_response.text[max(0, idx-200):min(len(mp_response.text), idx+500)]
    print('\n--- Upload Section ---')
    print(section)
else:
    print('\n❌ Upload form is NOT visible')

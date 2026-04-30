import requests
import time
import os

time.sleep(1)

session = requests.Session()

# Check what credentials are expected
print("Testing login with different credentials...")

# Try login
response = session.post('http://localhost:5000/login', data={'username': 'admin', 'password': 'password'})
print(f'Login response status: {response.status_code}')
print(f'Login response URL: {response.url}')
print(f'Session cookies: {session.cookies.get_dict()}')

# Try accessing dashboard
dashboard_response = session.get('http://localhost:5000/dashboard')
print(f'\nDashboard response status: {dashboard_response.status_code}')
print(f'Dashboard response length: {len(dashboard_response.text)}')
print(f'Is redirected to login: {"login" in dashboard_response.text}')

# Try missing_person
mp_response = session.get('http://localhost:5000/missing_person')
print(f'\nMissing person response status: {mp_response.status_code}')
print(f'Missing person response length: {len(mp_response.text)}')
print(f'Is redirected to login: {"login" in mp_response.text}')

# Check what error we get if we try login again
print("\n--- Checking default users in users.json ---")
try:
    import json
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    users_path = os.path.join(root_dir, 'config', 'users.json')
    with open(users_path, 'r') as f:
        users = json.load(f)
        print(f"Users: {users}")
except Exception as e:
    print(f"users.json error: {e}")

import requests
import time

time.sleep(1)

# Test login and inspect response headers
session = requests.Session()
response = session.post('http://localhost:5000/login', data={'username': 'admin', 'password': 'password'}, allow_redirects=False)

print(f'LOGIN RESPONSE')
print(f'Status: {response.status_code}')

# Check for Set-Cookie
if 'Set-Cookie' in response.headers:
    print(f'Set-Cookie: {response.headers["Set-Cookie"]}')
else:
    print('NO Set-Cookie header!')

print(f'\nAll response headers:')
for key, value in response.headers.items():
    print(f'  {key}: {value}')

print(f'\nSession cookies after login:')
print(f'  {session.cookies.get_dict()}')

# Check the response body
print(f'\nResponse status is redirect? {response.status_code in [301, 302, 303, 307, 308]}')
print(f'Response has Location header? {"Location" in response.headers}')
if "Location" in response.headers:
    print(f'  Location: {response.headers["Location"]}')

import requests

# Test 1: Successful code
print("Test 1: Successful code")
response = requests.post(
    'http://localhost:8004/code-interpreter',
    json={'code': 'x = 5\ny = 10\nprint(x + y)'}
)
print(response.json())
print()

# Test 2: Code with error
print("Test 2: Code with error")
response = requests.post(
    'http://localhost:8004/code-interpreter',
    json={'code': 'x = 10\ny = 0\nresult = x / y'}
)
print(response.json())

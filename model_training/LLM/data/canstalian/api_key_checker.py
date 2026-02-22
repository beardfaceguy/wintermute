import requests

class APIKeyChecker:
    def __init__(self, api_key: str, endpoint: str):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        if not endpoint:
            raise ValueError("Endpoint cannot be empty.")
        self.api_key = api_key
        self.endpoint = endpoint

    def check_api_key(self) -> bool:
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        try:
            response = requests.get(self.endpoint, headers=headers)
            if response.status_code == 200:
                return True
            else:
                print(f"Invalid API key: {response.reason}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while checking the API key: {e}")
            return False
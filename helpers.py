import requests
import logging
from config import FLASK_SERVER_URL, ADDITIONAL_SERVER_URL

def send_to_backend(data):
    """Send detection data to multiple backends"""
    servers = [
        FLASK_SERVER_URL,
        ADDITIONAL_SERVER_URL
    ]
    
    for url in servers:
        try:
            response = requests.post(
                url,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            logging.info(f"Data sent successfully to {url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send to {url}: {str(e)}")
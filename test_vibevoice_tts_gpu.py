"""Test VibeVoice TTS on Pod via local API (port 7860)"""
import sys, time, shutil
sys.stdout.reconfigure(encoding='utf-8')
from gradio_client import Client, handle_file

# Connect to local Pod tunnel
client = Client('http://127.0.0.1:7860/')
print('Available endpoints:')
print(client.view_api(return_format='str')[:1000])

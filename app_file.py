import os

# Ensure data directory exists
os.makedirs('/app/data', exist_ok=True)

# Export database path
DATABASE_NAME = '/app/data/health_chatbot_enhanced.db'

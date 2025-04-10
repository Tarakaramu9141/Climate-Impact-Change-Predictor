import ee
from dotenv import load_dotenv
import os

load_dotenv()

# Debug info
print("\n=== Debug Information ===")
print(f"Current directory: {os.getcwd()}")
print(f"Files present: {os.listdir()}")

# Get credentials
service_account = os.getenv('GEE_SERVICE_ACCOUNT')
key_path = os.getenv('GEE_SERVICE_KEY_PATH')

print(f"\nService Account: {service_account}")
print(f"Key Path: {key_path}")

# Verify file exists
if not os.path.exists(key_path):
    print(f"\n❌ ERROR: Key file not found!")
else:
    print(f"\n✅ Key file found")
    
    # Initialize GEE
    try:
        credentials = ee.ServiceAccountCredentials(service_account, key_path)
        ee.Initialize(credentials)
        print("\n🌎 Success! Google Earth Engine initialized")
        
        # Test API functionality
        print("\nRunning test computation...")
        test_image = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        elev_value = test_image.reduceRegion(
            ee.Reducer.first(), 
            ee.Geometry.Point([78.4, 17.4]), 
            30
        ).getInfo()
        print(f"Test elevation value: {elev_value}")
        print("✅ All systems operational!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if "Cannot read" in str(e):
            print("Please check your service account has Earth Engine access")
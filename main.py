import cv2
from inference_sdk import InferenceHTTPClient

# 1. Configuration - Use the credentials from your RAPID screenshot
# Use the "Private API Key" from your Roboflow Settings
# API_KEY = "e5X9QkH4oVr3n1rMzpT3" 
WORKSPACE = "nerfhacks"
WORKFLOW_ID = "find-targets-2" # Updated based on your latest terminal output

# 2. Initialize Client
# IMPORTANT: Use the standard detect endpoint if serverless is failing
# Change the URL to match your RAPID screenshot
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com", 
    api_key="e5X9QkH4oVr3n1rMzpT3" 
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    try:
        # 3. Use run_workflow correctly
        # The images dict key must match the input name in your Roboflow Workflow
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": frame} 
        )

        # 4. Parse RAPID Output
        # Workflows return a list of results. We look for the predictions inside.
        if result and len(result) > 0:
            # Adjust the key 'predictions' if you renamed the output in Roboflow
            predictions = result[0].get("output", {}).get("predictions", [])
            
            for pred in predictions:
                x, y = int(pred['x']), int(pred['y'])
                w, h = int(pred['width']), int(pred['height'])
                
                # Draw the target lock
                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                print(f"LOCKED: X={x} Y={y}")

    except Exception as e:
        print(f"Inference Error: {e}")

    cv2.imshow("Turret RAPID Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
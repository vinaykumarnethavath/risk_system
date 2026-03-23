import subprocess
import os

print("Running pipeline...")
result = subprocess.run([r".venv\Scripts\python.exe", "main.py", "--ticker", "TSLA", "--json"], capture_output=True, text=True)

with open("output_capture.txt", "w", encoding="utf-8") as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\n--- ERRORS ---\n")
        f.write(result.stderr)

if result.returncode != 0:
    print(f"Pipeline failed with code {result.returncode}")
else:
    print("Pipeline ran successfully.")
    
    # Now generate and print report text
    try:
        import json
        with open("output_capture.txt", "r", encoding="utf-8") as f:
            content = f.read()
            # Find JSON start
            json_start = content.find("{")
            if json_start != -1:
                assessment_json = content[json_start:]
                assessment = json.loads(assessment_json)
                
                # We need to load assessment properly or call the function directly
                # It's easier to call the function directly in this script!
                pass
    except Exception as e:
        print(f"Failed to parse JSON: {e}")

print("\n--- Narrative Report Verification ---")
# Let's create a separate script to call the function directly with current modules

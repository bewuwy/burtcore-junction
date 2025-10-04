import os
import requests

def test_upload_video():
    # Define the path to the test video file
    test_file_path = "backend/tests/non_hate_video_18.mp4"

    # Ensure the test file exists
    assert os.path.exists(test_file_path), "Test video file does not exist."

    # Define the URL for the upload endpoint
    url = "http://127.0.0.1:8000/upload-video/"

    # Open the file in binary mode and send it to the upload endpoint
    with open(test_file_path, "rb") as test_file:
        response = requests.post(url, files={"file": ("non_hate_video_18.mp4", test_file, "video/mp4")})

    # Assert the response status code and content
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    response_data = response.json()
    print("Response data:", response_data)
    assert "file_path" in response_data, "Response does not contain 'file_path'."

    # Verify the uploaded file exists in the temp_uploads directory
    uploaded_file_path = response_data["file_path"]
    assert os.path.exists(uploaded_file_path), "Uploaded file does not exist in the expected directory."

test_upload_video()
print("Test completed successfully.")
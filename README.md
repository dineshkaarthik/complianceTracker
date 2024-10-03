# complianceCheck
 
To run the application on your local setup, please follow these steps:

Clone the repository:
Open a terminal and run:
git clone <repository_url>
(Replace <repository_url> with the actual URL of the project repository)

Navigate to the project directory:
cd CompCheckPDFExtractor

Create a virtual environment (optional but recommended):
python -m venv venv
Activate the virtual environment:

On Windows: venv\Scripts\activate
On macOS and Linux: source venv/bin/activate
Install the required dependencies:
pip install -r requirements.txt

Set up the environment variables:
Create a .env file in the project root directory and add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Run the Flask application:
python app.py

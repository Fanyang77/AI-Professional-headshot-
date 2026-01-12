📸 Selfie → Professional Headshot

Turn any casual selfie into a clean, studio-quality professional headshot using AI.
This app automatically crops, enhances, and transforms user-uploaded selfies into realistic business-ready portraits with professional lighting, backgrounds, and attire.

Built with Streamlit + OpenAI Image Editing API.

✨ Features

📤 Upload any selfie (JPG / PNG)

🤖 AI-powered studio headshot generation

✂️ Automatic face detection & head-and-shoulders cropping

🎨 Professional lighting, background & outfit styling

🧠 Identity-preserving (keeps your face realistic)

⬇️ Download polished or AI-generated images

🔐 Secure API key handling via .env

🖼 How it works

User uploads a selfie

App automatically:

Detects the face

Crops to a professional headshot framing

Applies color, contrast, and sharpness polish

The processed image is sent to OpenAI’s Image Edit API

AI transforms it into a studio-style professional headshot

The user downloads the final result

🧰 Tech Stack

Frontend: Streamlit

Image Processing: Pillow, OpenCV

AI: OpenAI gpt-image-1 (Image Editing API)

Environment: Python + dotenv

🚀 Local Setup
1. Clone the repo
git clone https://github.com/yourname/headshot-app.git
cd headshot-app

2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate   (Mac/Linux)

3. Install dependencies
pip install -r requirements.txt

4. Add your OpenAI API key

Create a .env file in the project root:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx


⚠️ Do not commit .env — keep it private.

5. Run the app
streamlit run app.py


Open in browser:

http://localhost:8501

🧪 Usage

Upload a selfie

Choose:

Background

Outfit style

Click Generate professional headshot

Download your AI-generated headshot

🛡 Privacy

Images are processed in memory

No files are saved to disk

API keys are never exposed to the frontend

For production, use HTTPS and server-side processing.

🧠 Example Prompts

The AI receives instructions like:

“Turn this selfie into a professional studio headshot. Clean lighting, realistic skin texture, business-casual attire, neutral background, identity preserved.”

This ensures natural, professional results — not cartoonish or over-filtered images.

🏗 Future Improvements

Multiple style presets (LinkedIn, Actor, Corporate, Startup, etc.)

Batch processing

Face retouching toggle

Background upload

Headshot history

🧑‍💻 Author

Built by Fan Yang
AI Engineer & Product Builder

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import requests
import secrets
from flask import Flask, request, jsonify, send_file
from google.cloud import speech, texttospeech_v1
from google.oauth2 import service_account
from googletrans import Translator
import os
import io
from flask_cors import CORS
from pydub import AudioSegment
import subprocess
from flask import Flask, send_from_directory
import textwrap
from dotenv import load_dotenv

# server running on : https://ayurbotserver.vercel.app/

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure secret key for session management
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://ayur-bot.vercel.app"]}}, supports_credentials=True)

# Load user details from Excel

file_path = "member_details.xlsx"
df = pd.read_excel(file_path)

# Global variable for Aadhaar number
global_aadhaar_number = None  # Initialize global Aadhaar number variable

# Groq API setup


load_dotenv()  # Load environment variables

API_KEY = os.getenv("API_KEY") # Replace with your actual API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

#initialize global variable
language_code = 'en'

# Function to select language

def select_language(choice):
    global language_code  # Declare it as global to modify the outer scope variable
    language_map = {
        '1': 'te',  # Telugu
        '2': 'hi',  # Hindi
        '3': 'en',  # English
        '4': 'ta'   # Tamil
    }
    language_code = language_map.get(choice, 'en')  # Default to English if invalid choice
    print(f"Language set to: {language_code}")
    return language_code


def convert_webm_to_wav(input_file, output_file="audio.wav"):
    try:
        # Delete the output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # FFmpeg command to convert WebM to WAV
        command = [
            "ffmpeg", "-i", input_file,
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            output_file
        ]
        
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error in conversion: {e}")
        raise Exception("Error in audio conversion")


# Main function for audio to text
def audio_to_text(client_file, audio_file, language_code):
    print('Audio to text function is running')

    try:
        # Check if the audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file {audio_file} not found.")
        
        # Check file extension and convert if WebM
        file_extension = os.path.splitext(audio_file)[1].lower()
        if file_extension == '.webm':
            output_file_path = "uploads/audio.wav"
            convert_webm_to_wav(audio_file, output_file_path)
            audio_file = output_file_path  # Use the newly converted WAV file

        # Detect sample rate using pydub
        audio_segment = AudioSegment.from_file(audio_file, format="wav")  # Specify format as wav
        sample_rate_hertz = audio_segment.frame_rate
        print(f"Detected sample rate: {sample_rate_hertz}")

        # Set encoding to LINEAR16 for WAV format
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(client_file)
        client = speech.SpeechClient(credentials=credentials)

        # Read audio file content
        with io.open(audio_file, 'rb') as f:
            content = f.read()

        # Set up Google Speech API configuration
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate_hertz,
            language_code=language_code
        )

        # Recognize speech
        response = client.recognize(config=config, audio=audio)
        if response.results:
            transcript = [result.alternatives[0].transcript for result in response.results]
            # Optional translation function
            send_data()
            print(f"Transcript: {transcript}")
            text = translator_english(transcript)
            print(f"Transcript: {text}")
            return transcript, text  # Return translated text if needed, or just the transcript
        else:
            return "No transcription results."

    except FileNotFoundError as fnf_error:
        return f"File error: {fnf_error}"
    except Exception as e:
        return f"Error: {e}"

    
# Function for translation to English
def translator_english(transcript, target_language='en'):
    translator = Translator()
    translations = []
    for text in transcript:
        translation= translator.translate(text, dest=target_language)
        translations.append(translation.text)
    return translations

def translator_language(transcript, language_code):
    translator = Translator()
    translations = []
    
    # Ensure `transcript` is iterable
    if isinstance(transcript, str):
        transcript = [transcript]  # Convert string to a list
    
    try:
        for text in transcript:
            translation = translator.translate(text, dest=language_code)
            translations.append(translation.text)
        translated_text = ' '.join(translations)
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def text_to_audio(text, language_code, output_file="uploads/output_audio.mp3"):
    # Ensure that the 'uploads' folder exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    # Set the path to the Google Cloud credentials (ensure this file exists)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sa_speechtotext.json"

    client = texttospeech_v1.TextToSpeechClient()

    # Prepare the text for conversion to speech
    synthesis_input = texttospeech_v1.SynthesisInput(text=text)

    # Define voice selection parameters
    voice = texttospeech_v1.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech_v1.SsmlVoiceGender.FEMALE
    )

    # Define audio configuration (MP3 encoding)
    audio_config = texttospeech_v1.AudioConfig(
        audio_encoding=texttospeech_v1.AudioEncoding.MP3
    )

    # Request the speech synthesis
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Save the generated audio to the specified file path in the uploads folder
    with open(output_file, "wb") as out:
        out.write(response.audio_content)

    return output_file  # Return the path to the saved audio file


# Greeting messages in different languages
greetings = {
    "Hindi": "नमस्ते! आप कैसे हैं?",
    "Telugu": "నమస్కారం! మీరు ఎలా ఉన్నారు?",
    "Tamil": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "Bengali": "নমস্কার! আপনি কেমন আছেন?",
    "Marathi": "नमस्कार! तुम्ही कसे आहात?",
    "Gujarati": "નમસ્તે! તમે કેમ છો?",
    "Kannada": "ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿದ್ದೀರಾ?",
    "Malayalam": "നമസ്കാരം! നിങ്ങൾക്ക് സുഖമാണോ?",
    "Odia": "ନମସ୍କାର! ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"
}

# Aadhaar Authentication API
@app.route("/auth", methods=["POST"])
def authenticate():
    global global_aadhaar_number  # Access the global variable

    print("Received request to authenticate Aadhaar")

    data = request.json
    aadhaar_number = str(data.get("aadhaar")).strip()

    print("Received Aadhaar number:", aadhaar_number)

    if len(aadhaar_number) != 12 or not aadhaar_number.isdigit():
        return jsonify({"message": "Invalid Aadhaar number"}), 400

    # Check if Aadhaar exists in the database
    user = df[df["Aadhar Number"] == int(aadhaar_number)]
    if user.empty:
        return jsonify({"message": "Aadhaar not found"}), 404

    session["aadhaar"] = aadhaar_number  # Store Aadhaar in session
    global_aadhaar_number = aadhaar_number  # Set the global Aadhaar number
    print("Aadhaar authentication successful!")
    print("Aadhaar number:", aadhaar_number)
    return jsonify({"message": "Aadhaar authentication successful!"}), 200
@app.route('/get-data', methods=['GET'])
def send_data():
    print('Send data function is running')
    data = {"message": "Hello from Flask!", "transcript": "This is a sample transcript."}
    return jsonify(data)


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        # Get the uploaded audio file from the request
        audio_file = request.files['audio']
        

        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Define the file path
        audio_file_path = os.path.join('uploads', audio_file.filename)

        # Save file with overwrite logic
        save_file_with_overwrite(audio_file, audio_file_path)

        print(language_code)
        # Process the audio file and get transcript
        # Get the credentials file path from the environment variable
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Now, use credentials_path instead of hardcoding 'sa_speechtotext.json'
        transcript, text = audio_to_text(credentials_path, audio_file_path, language_code)

        # Return the transcript
        return jsonify({
            "message": "Audio uploaded and processed successfully",
            "transcript": transcript,
            "text": text
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process-text', methods=['POST'])
def process_text():
    try:
        # Ensure JSON data is received
        if not request.is_json:
            return jsonify({"error": "Invalid content type. Expected application/json"}), 400

        # Parse JSON data
        data = request.json
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        user_input = data.get('userInput', '')
        if not user_input:
            return jsonify({"error": "Missing 'userInput' field in JSON"}), 400

        print(f"Received text reply: {user_input}")
        print(f"Language code: {language_code}")
        translated_text = translator_language(user_input, language_code)
        # Process the input and generate a response
        print(f"Translated text: {translated_text}")
        print(f"Language code: {language_code}")
        text_to_audio(translated_text, language_code)
        # Define the folder where audio files are stored
        UPLOAD_FOLDER = 'uploads'
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        get_audio()
        bot_reply = translated_text

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print(f"Error processing text: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500


def save_file_with_overwrite(file, path):
    # Check if file already exists and delete it
    if os.path.exists(path):
        os.remove(path)
    # Save the new file
    file.save(path)


@app.route('/get-audio', methods=['GET'])
def get_audio():
    try:
        # Path to the audio file
        audio_filename = 'output_audio.mp3'

        # Ensure the file exists
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404

        # Serve the audio file
        return send_from_directory(app.config['UPLOAD_FOLDER'], audio_filename, as_attachment=True)
    
    except Exception as e:
        print(f"Error serving audio file: {e}")
        return jsonify({"error": "Failed to retrieve the audio file"}), 500
    


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        print("Processing audio...")
        audio_file = request.files['audio']
        language_code = request.form.get('language_code', 'en')

        # Save the uploaded file
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        audio_file_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_file_path)
        print('process_audio function is running')
        print(language_code)
        # Transcribe the audio to text
        transcript = audio_to_text('texttospeech.json', audio_file_path, language_code)

        # Translate the transcript and convert to speech
        audio_output = translator_language(transcript, language_code)

        return send_file(audio_output, mimetype='audio/mp3')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/set-language', methods=['POST'])
def set_language():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        language = data.get('language')

        if language:
            # Store the selected language or perform additional processing
            # Example: save language to a session or database
            print(f"Language set to {language}")
            if language == 'telugu':
                language_code = select_language('1')
            elif language == 'hindi':
                language_code = select_language('2')
            elif language == 'english':
                language_code = select_language('3')
            elif language == 'tamil':
                language_code = select_language('4')

            print(language_code)
            # Send a success response
            return jsonify({"message": "Language set successfully"}), 200
        else:
            return jsonify({"error": "No language provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Store conversation history
user_conversations = {}
@app.route("/chat", methods=["POST"])
def chat():
    global global_aadhaar_number  # Access the global variable
    aadhaar_number = global_aadhaar_number  # Retrieve Aadhaar number

    if not aadhaar_number:
        return jsonify({"message": "User not authenticated"}), 403

    user = df[df["Aadhar Number"] == int(aadhaar_number)]
    if user.empty:
        return jsonify({"message": "User not found"}), 404

    user_details = user.iloc[0].to_dict()
    user_language = user_details.get("Local Language", "English")
    greeting = f"Namaste! How can I assist you today in {user_language}?"
    
    # Retrieve conversation history for the user
    if aadhaar_number not in user_conversations:
        user_conversations[aadhaar_number] = [
            {"role": "system", "content": "You are an Ayurvedic health advisor who provides context-aware responses."},
            {"role": "user", "content": f"User Details: {user_details}"},
        ]

    data = request.json
    messages = data.get("messages", [])

    if messages:
        user_input = messages[-1]["content"]
        user_conversations[aadhaar_number].append({"role": "user", "content": user_input})

    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": user_conversations[aadhaar_number], "temperature": 0.7},
    )

    if response.status_code == 200:
        bot_reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
        user_conversations[aadhaar_number].append({"role": "assistant", "content": bot_reply})  # Store bot response
        print("Bot reply:", bot_reply)
        # Generate a summary before the full response

        summary = summarize_text(bot_reply)
        print("Summary:", summary)
        
        # conversation.append({"role": "assistant", "content": bot_reply})  # Store full bot response
        # user_conversations[aadhaar_number] = conversation  # Update conversation history
        
        return jsonify({"response": bot_reply}), 200
        # return {"summary": summary, "messages": split_messages}
    else:
        return jsonify({"error": f"Error {response.status_code}: {response.text}"}), 500
    




def summarize_text(text):
    payload = {
        "model": "llama3-8b-8192", 
        "messages": [
            {"role": "system", "content": "Summarize the following text."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.5
    }

    response = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"












@app.before_request
def log_request():
    print(f"Incoming request: {request.method} {request.url}")

@app.route('/api/sloka/<int:chapter>/<int:verse>', methods=['GET'])
def get_sloka(chapter, verse):
    try:
        print(f"Fetching sloka for Chapter {chapter}, Verse {verse}")
        api_url = f"https://gita-api.vercel.app/tel/verse/{chapter}/{verse}"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch sloka"}), 500

        print("Sloka fetched successfully")
        return jsonify(response.json())
    except Exception as e:
        print(f"Error fetching sloka: {e}")
        return jsonify({"error": "Error fetching sloka"}), 500

@app.route('/')
def home():
    return "Welcome to Bhagavad Gita API Server"



# def summarize_response(response_text):
#     """Summarizes key points from a chatbot response."""
#     if len(response_text) < 150:
#         return response_text  # No need to summarize short responses
    
#     # Basic summary by extracting key information (you can improve this using NLP)
#     key_phrases = ["In summary,", "To summarize,", "Key points:"]
#     for phrase in key_phrases:
#         if phrase in response_text:
#             return response_text.split(phrase, 1)[1].strip()  # Extract summary part
    
#     # If no summary found, return the full response
#     print(textwrap.shorten(response_text, width=200, placeholder="...") )

#     return textwrap.shorten(response_text, width=200, placeholder="...")  # Simple shortening




# # Chatbot API
# Chatbot API
# @app.route("/chat", methods=["POST"])
# def chat():
#     global global_aadhaar_number  # Access the global variable

#     aadhaar_number = global_aadhaar_number  # Retrieve the Aadhaar number from the global variable
#     print("aadhaar_number_from_global:", aadhaar_number)
#     if not aadhaar_number:
#         return jsonify({"message": "User not authenticated"}), 403

#     user = df[df["Aadhar Number"] == int(aadhaar_number)]
#     if user.empty:
#         return jsonify({"message": "User not found"}), 404

#     user_details = user.iloc[0].to_dict()
#     print("User Details:", user_details)
#     user_language = user_details.get("Local Language", "English")
#     greeting = greetings.get(user_language, "Hello! How are you?")
#     initial_question = "How are you feeling today? Are you experiencing any health concerns?"
#     follow_up_question = "What brings you here today? How can I assist you?"

#     data = request.json
#     print("Received JSON data:", data)

#     messages = data.get("messages", [])
#     if not messages:
#         # If no previous messages, send the initial greeting and questions
#         initial_messages = [
#             {"role": "system","content": "You are an Ayurvedic health advisor who speaks multiple languages. Start by greeting the user and asking how they are feeling today. Then, ask why they are here and what health concerns they have. Do not ask too many questions at once—keep the conversation natural and focused. Provide Ayurvedic remedies, lifestyle advice, and herbal medicine recommendations based on their responses. Keep their medical history in mind but do not mention it unless necessary. Your responses should be concise, empathetic, and easy to understand."},
#             {"role": "user", "content": f"User Details: {user_details}"},
#             {"role": "assistant", "content": f"{greeting} {initial_question}"},
#         ]
#         return jsonify({"response": f"{greeting} {initial_question} {follow_up_question}"})

#     # Get the last user message
#     user_input = None
#     for msg in reversed(messages):
#         if msg["role"] == "user" and msg.get("content"):
#             user_input = msg["content"]
#             break

#     print("User Input:", user_input)
#     if not user_input:
#         return jsonify({"message": "No input provided"}), 400

#     messages = [
#         {"role": "system", "content": "You are an Ayurvedic health advisor who speaks multiple languages. Respond based on the user's language and medical history."},
#         {"role": "user", "content": f"User Details: {user_details}"},
#         {"role": "assistant", "content": greeting},
#         {"role": "user", "content": user_input},
#     ]

#     response = requests.post(
#         GROQ_API_URL,
#         headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
#         json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.7},
#     )

#     print("Response Status Code:", response.status_code)
#     print("Response Text:", response.text)

#     if response.status_code == 200:
#         choices = response.json().get("choices", [])
#         if choices:
#             bot_reply = choices[0].get("message", {}).get("content", "No response received.")
#             return jsonify({"response": bot_reply})
#         else:
#             return jsonify({"message": "No valid choices found in response."}), 500
#     else:
#         return jsonify({"error": f"Error {response.status_code}: {response.text}"}), 500

# @app.route("/chat", methods=["POST"])
# def chat():
#     global global_aadhaar_number  # Access the global variable

#     aadhaar_number = global_aadhaar_number  # Retrieve the Aadhaar number from the global variable
#     print("aadhaar_number_from_global:", aadhaar_number)
#     if not aadhaar_number:
#         return jsonify({"message": "User not authenticated"}), 403

#     user = df[df["Aadhar Number"] == int(aadhaar_number)]
#     if user.empty:
#         return jsonify({"message": "User not found"}), 404

#     user_details = user.iloc[0].to_dict()
#     print("User Details:", user_details)
#     user_language = user_details.get("Local Language", "English")
#     greeting = greetings.get(user_language, "Hello! How are you?")
#     print("Greeting:", greeting)

#     data = request.json
#     print("Received JSON data:", data)
#     # Extract the latest user message from the 'messages' list
#     messages = data.get("messages", [])
#     if not messages:
#         return jsonify({"message": "No messages provided"}), 400

#     # Get the last user message
#     user_input = None
#     for msg in reversed(messages):
#         if msg["role"] == "user" and msg.get("content"):
#             user_input = msg["content"]
#             break

#     print("User Input:", user_input)

#     if not user_input:
#         return jsonify({"message": "No input provided"}), 400

#     messages = [
#         {"role": "system", "content": "You are an Ayurvedic health advisor who speaks multiple languages. Respond based on the user's language and medical history."},
#         {"role": "user", "content": f"User Details: {user_details}"},
#         {"role": "assistant", "content": greeting},
#         {"role": "user", "content": user_input},
#     ]
#     print("Messages sent to the API:", messages)

#     response = requests.post(
#         GROQ_API_URL,
#         headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
#         json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.7},
#     )
#     print("Response Status Code:", response.status_code)
#     print("Response Text:", response.text)

#     if response.status_code == 200:
#         # Ensure the response structure is correct
#         choices = response.json().get("choices", [])
#         if choices:
#             bot_reply = choices[0].get("message", {}).get("content", "No response received.")
#             return jsonify({"response": bot_reply})
#         else:
#             return jsonify({"message": "No valid choices found in response."}), 500
#     else:
#         return jsonify({"error": f"Error {response.status_code}: {response.text}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

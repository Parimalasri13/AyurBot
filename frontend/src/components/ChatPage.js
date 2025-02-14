// src/components/ChatPage.js
import React, { useState, useRef } from 'react';
import './styles/ChatPage.css'; // Ensure you move the styles to a separate CSS file

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const chatBoxRef = useRef(null);

  // Function to add a message to the chatbox
  const addMessage = (sender, message) => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender, message },
    ]);
    // Scroll to the bottom of the chatbox
    chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
  };

  // Function to handle sending text messages
  const sendMessage = () => {
    if (userInput.trim()) {
      addMessage("user", userInput);
      sendToRasa(userInput);
      setUserInput('');
    }
  };

  // Function to send a message to Rasa (bot)
  const sendToRasa = (message) => {
    fetch('http://localhost:5005/webhooks/rest/webhook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    })
    .then((response) => response.json())
    .then((data) => {
      if (data[0] && data[0].text) {
        sendTextReplyToServer(data[0].text);
        
      }
    });
  };

  function playAudioFromServer() {
    // Fetch the audio file from the server
    fetch('https://ayurbotserver.vercel.app/get-audio')
        .then(response => response.blob())  // Convert the response to a Blob (binary data)
        .then(audioBlob => {
            // Create an audio element
            const audio = new Audio(URL.createObjectURL(audioBlob));
            
            // Play the audio
            audio.play();
        })
        .catch(error => {
            console.error('Error fetching the audio:', error);
        });
}
  // Function to handle text reply from the server
  const sendTextReplyToServer = (text) => {
    fetch('https://ayurbotserver.vercel.app/process-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userInput: text }),
    })
    .then((response) => response.json())
    .then((data) => {
      if (data.reply) {
        addMessage('bot', data.reply);
        playAudioFromServer();
      }
    });
  };

  // Function to start audio listening (Web Speech API or similar)
  const startListening = () => {
    console.log("Listening...");
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks);
          sendAudioToServer(audioBlob);
        };

        mediaRecorder.start();

        setTimeout(() => {
          mediaRecorder.stop();
          stream.getTracks().forEach(track => track.stop());
        }, 5000);
      });
  };

  // Function to send audio to the server for transcription
  const sendAudioToServer = (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.webm');

    fetch('https://ayurbotserver.vercel.app/upload-audio', {
      method: 'POST',
      body: formData,
    })
    .then((response) => response.json())
    .then((data) => {
      if (Array.isArray(data.transcript) && data.transcript.length > 0) {
        addMessage('user', data.transcript[0]);
        sendToRasa(data.text[0]);
      }
    });
  };

  return (
    <div>

      <main>
        <div className="chat-section">
          <img src="./assets/robo.png" alt="Robot" />
        </div>
        <div className="character-section">
          <img src="./assets/oldman.png" alt="Man" />
        </div>

        <div className="chatbox">
          <h2>AyurBot Chat</h2>
          <div className="messages" ref={chatBoxRef}>
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}-message`}>
                {msg.message}
              </div>
            ))}
          </div>
          <div className="input-container">
            <button onClick={startListening}>🎤</button>
            <input
              type="text"
              id="user-input"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Type your message..."
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ChatPage;

import React from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate hook
import './styles/LanguageSelection.css'; // Import the CSS file for styling

const LanguageSelection = () => {
  const navigate = useNavigate(); // Initialize the navigate function

  // Function to handle language selection
  const selectLanguage = async (language) => {
    try {
      // Store the selected language in localStorage for later use
      localStorage.setItem('language', language);

      // Send the language code to Flask backend using fetch API
      const response = await fetch('http://127.0.0.1:5000/set-language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language }),
      });

      // Check if the response is ok (status code 200-299)
      if (!response.ok) {
        throw new Error('Failed to set language');
      }

      const data = await response.json(); // Get the response data

      // Successfully set the language, now navigate to the chat page
      navigate('/chat');  // Replace '/chat' with the actual route to your chat page
    } catch (error) {
      console.error('Error setting language:', error);
      alert('An error occurred while setting the language. Please try again.');
    }
  };

  return (
    <div>
      <main>
        <div className="content">
          <div className="illustration">
            <img src="./assets/man1.png" alt="Man Illustration" />
          </div>
          <div className="languages">
            <button onClick={() => selectLanguage('telugu')}>Telugu</button>
            <button onClick={() => selectLanguage('english')}>English</button>
            <button onClick={() => selectLanguage('hindi')}>Hindi</button>
            <button onClick={() => selectLanguage('kanada')}>Kanada</button>
            <button onClick={() => selectLanguage('malayalam')}>Malayalam</button>
            <button onClick={() => selectLanguage('tamil')}>Tamil</button>
          </div>
          <div className="illustration">
            <img src="./assets/woman.png" alt="Woman Illustration" />
          </div>
        </div>
      </main>
    </div>
  );
};

export default LanguageSelection;

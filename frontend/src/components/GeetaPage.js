import React, { useState } from 'react';
import './styles/GeethaPage.css'; // Make sure to create a CSS file to style the components accordingly

const GeetaPage = () => {
  const [chapter, setChapter] = useState('');
  const [sloka, setSloka] = useState('');
  const [slokaText, setSlokaText] = useState('Enter a valid Chapter and Sloka.');
  const [bhavamText, setBhavamText] = useState('Bhavam will appear here.');
  const [audioSrc, setAudioSrc] = useState('');

  const fetchSloka = async () => {
    try {
      const response = await fetch(`https://spiritual-path.onrender.com/api/sloka/${chapter}/${sloka}`);
      const data = await response.json();

      setSlokaText(data.verse || "Sloka not available.");
      setBhavamText(data.translation || "Bhavam not available.");
      setAudioSrc(data.audio_link || "");
    } catch (error) {
      console.error("Error fetching sloka:", error);
      setSlokaText("Error fetching sloka.");
      setBhavamText("Error fetching bhavam.");
    }
  };

  const playAudio = () => {
    if (audioSrc) {
      const audioPlayer = new Audio(audioSrc);
      audioPlayer.volume = 1.0;
      audioPlayer.play();
    } else {
      alert("Audio not available.");
    }
  };

  return (
    <div className="app">
      {/* Navbar */}
     

      {/* Main Content */}
      <main className="content">
        {/* Krishna Image */}
        <div className="image-container">
          <img
            src="./assets/krish.jpg"
            alt="Lord Krishna"
            className="krishna-image"
          />
        </div>

        {/* Sloka and Bhavam */}
        <div className="sloka-container">
          <div className="input-row">
            <div className="input-group">
              <label htmlFor="chapter">Chapter Number</label>
              <input
                type="number"
                id="chapter"
                min="1"
                placeholder="Enter Chapter number"
                value={chapter}
                onChange={(e) => setChapter(e.target.value)}
              />
            </div>
            <div className="input-group">
              <label htmlFor="sloka">Sloka Number</label>
              <input
                type="number"
                id="sloka"
                min="1"
                placeholder="Enter Sloka number"
                value={sloka}
                onChange={(e) => setSloka(e.target.value)}
              />
            </div>
            <button className="submit-button" onClick={fetchSloka}>Submit</button>
          </div>
        </div>
      </main>

      {/* Sloka and Bhavam Content */}
      <div className="content_container">
        <div className="sloka-content">
          <div className="sloka">
            <h3><bold>Sloka:</bold></h3>
            <p>{slokaText}</p>
          </div>
          <div className="bhavam">
            <h3>Bhavam:</h3>
            <p>{bhavamText}</p>
          </div>
        </div>
        <div className="action-icons">
          <button className="sound-icon" onClick={playAudio}>
            <img src="./assets/speaker.png" alt="Play Audio" className="speaker-icon" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default GeetaPage;

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../components/styles/Slokas.css';
import countries from "../components/data";


export default function Geetha() {
    const [chapter, setChapter] = useState('');
    const [verse, setVerse] = useState('');
    const [sloka, setSloka] = useState('');
    const [bhavam, setBhavam] = useState('');
    const [error, setError] = useState('');
    const [audioSrc, setAudioSrc] = useState('');
    const speaker = process.env.PUBLIC_URL + "/assets/speaker.png";



    

    const playAudio = () => {
        const audioPlayer = new Audio(audioSrc);
        audioPlayer.volume = 1.0;
        audioPlayer.play();
      };

    const fetchSloka = async (e) => {
        e.preventDefault();
        setError('');
        try {
            const response = await axios.get(`https://spiritual-path.onrender.com/api/sloka/${chapter}/${verse}`);
            const data = response.data;
            setSloka(data.verse);
            setBhavam(data.translation);
            setAudioSrc(data.audio_link);
        } catch (error) {
            console.error('Error fetching sloka:', error);
            setError('Error fetching sloka. Please try again.');
        }
    };

    useEffect(() => {
        const fromText = document.querySelector(".myths-from-text");
        if (fromText) {
            fromText.value = `${sloka}\n\n\n${bhavam}`;
        }
    }, [sloka, bhavam]);

    useEffect(() => {
        const audio = new Audio(audioSrc);
        if (audioSrc) {
            audio.play();
        }
    }, [audioSrc]);
    
    

    useEffect(() => {
        const fromText = document.querySelector(".myths-from-text");
        if (fromText) {
            fromText.value = `${sloka}\n\n\n${bhavam}`;
        }
    }, [sloka, bhavam]);

    useEffect(() => {
        const toText = document.querySelector(".myths-to-text");
        const selectTag = document.querySelectorAll("select");
        const icons = document.querySelectorAll(".myths-row i");
        const translateBtn = document.querySelector(".myths-button1");

        selectTag.forEach((tag, id) => {
            for (let country_code in countries) {
                let selected =
                    id === 0
                        ? country_code === "te-IN"
                            ? "selected"
                            : ""
                        : country_code === "hi-IN"
                            ? "selected"
                            : "";
                let option = `<option ${selected} value="${country_code}">${countries[country_code]}</option>`;
                tag.insertAdjacentHTML("beforeend", option);
            }
        });

        function insertNewLineAfterNumber(translatedText) {
            // Use a regular expression to find a number in the text
            return translatedText.replace(/(\d+)/, '$1\n\n\n\n');
        }

        const fromText = document.querySelector(".myths-from-text");
        fromText.addEventListener("keyup", () => {
            if (!fromText.value) {
                toText.value = "";
            }
        });

        translateBtn.addEventListener("click", () => {
            let text = fromText.value.trim();
            let translateFrom = selectTag[0].value;
            let translateTo = selectTag[1].value;
            if (!text) return;
            toText.setAttribute("placeholder", "Translating...");
            let apiUrl = `https://api.mymemory.translated.net/get?q=${text}&langpair=${translateFrom}|${translateTo}`;
            fetch(apiUrl)
                .then((res) => res.json())
                .then((data) => {
                    // Assuming data.responseData.translatedText contains the original text
                    console.log(data)
                    let translatedText = data.responseData.translatedText;

                    // Update the text with a new line after the number
                    let updatedText = insertNewLineAfterNumber(translatedText);

                    // Set the updated text to the input field
                    toText.value = updatedText;
                    
                    toText.setAttribute("placeholder", "Translation");
                });
        });

        
    }, []);

    return (
        <>
            <div>
                <div className="myths-box">
                    <div className="myths-imge">
                        <img className="myths-img" src="./assets/Krish.jpg" alt="Krishna" />
                    </div>
                    <div className="myths-form">
                        <p className='myths-start'>Enter the Chapter and Slokha Number to get respective Slokha And Its Meaning</p>
                        <form onSubmit={fetchSloka}>
                            <input
                                type="number"
                                name="chapter"
                                placeholder="Enter Chapter Number"
                                value={chapter}
                                onChange={e => setChapter(e.target.value)}
                                min="1"
                                max="30"
                                className="myths-input"
                            />
                            <br />
                            <input
                                type="number"
                                name="slokha"
                                placeholder="Enter Slokha Number"
                                value={verse}
                                onChange={e => setVerse(e.target.value)}
                                min="1"
                                max="50"
                                className="myths-input"
                            />
                            <br />
                            <input type="submit" value="Submit" className="myths-input" />
                        </form>
                       
                    </div>
                </div>
                <div className="myths-bhavam">
                    <div className="myths-para">
                        {error && <p className="myths-error">{error}</p>}
                        {sloka && <p style={{ cursor: 'pointer' }}>slokam: {sloka}</p>}
                        {bhavam && <p style={{ cursor: 'pointer' }}>Bhavam: {bhavam}</p>}
                        <img src= {speaker}  alt="Speak" onClick={playAudio} style={{ cursor: 'pointer', width: '30px', height: '30px', marginLeft: '500px' }} />

                    </div>
                </div>

                <div>
                    <div className="myths-container">
                        <div className="myths-wrapper">
                            <div className="myths-text-input">
                                <textarea
                                    spellCheck="false"
                                    className="myths-from-text"
                                    placeholder="Enter text"
                                ></textarea>
                                <textarea
                                    spellCheck="false"
                                    readOnly
                                    disabled
                                    className="myths-to-text"
                                    placeholder="Translation"
                                ></textarea>
                            </div>
                            <ul className="myths-controls">
                                <li className="myths-row myths-from">
                                    <div className="myths-icons">
                                        <i id="from" className="fas fa-volume-up"></i>
                                        <i id="from" className="fas fa-copy"></i>
                                    </div>
                                    <select></select>
                                </li>
                                <li className="myths-exchange">
                                    <i className="fas fa-exchange-alt"></i>
                                </li>
                                <li className="myths-row myths-to">
                                    <select></select>
                                    <div className="myths-icons">
                                        <i id="to" className="fas fa-volume-up"></i>
                                        <i id="to" className="fas fa-copy"></i>
                                    </div>
                                </li>
                            </ul>
                        </div>
                        <button className='myths-button1'>Translate Text</button>
                    </div>
                </div>
            </div>
        </>
    );
}









// // src/pages/Home.js
// import React from 'react';
// import Content from '../components/content' // Import the Content component
// import BotIcon from '../components/BotIcon'; 
// import Geetha from '../components/GeethaPage'; // Import the BotIcon component

// const Home = () => {
//     return (
//         <div>
 //               <Geetha />
//         </div>
//     );
// };

// export default Home;

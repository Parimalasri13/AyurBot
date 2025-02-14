

// import { useState, useRef } from "react";
// import { useLocation } from "react-router-dom";
// import './styles/Communication.css'; 

// export default function Communication() {
//   const location = useLocation();
//   const aadhaar = location.state?.aadhaar || "";
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState("");
//   const [loading, setLoading] = useState(false);
//   const chatBoxRef = useRef(null);

//   const handleSend = async () => {
//     if (!input.trim()) return;

//     const newMessages = [...messages, { role: "user", content: input }];
//     setMessages(newMessages);
//     setInput("");
//     setLoading(true);

//     try {
//       const response = await fetch("https://ayurbotserver.vercel.app/chat", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ messages: newMessages })
//       });
//       const data = await response.json();
//       setMessages([...newMessages, { role: "bot", content: data.response }]);
//     } catch (error) {
//       alert("Error connecting to chat");
//     }
//     setLoading(false);
//   };

//   return (
//     <div className="communication-container">
//       <main className="communication-main">
//         <div className="communication-chat-section">
//           <h2 className="communication-title">AyurBot Chat</h2>
//           <div className="communication-messages" ref={chatBoxRef}>
//             {messages.map((msg, index) => (
//               <div key={index} className={`communication-message communication-${msg.role}-message`}>
//                 {msg.content}
//               </div>
//             ))}
//           </div>
//           <div className="communication-input-container">
//             <input
//               type="text"
//               value={input}
//               onChange={(e) => setInput(e.target.value)}
//               placeholder="Type your message..."
//               className="communication-input"
//             />
//             <button 
//               onClick={handleSend} 
//               disabled={loading} 
//               className="communication-send-button"
//             >
//               Send
//             </button>
//           </div>
//         </div>
//       </main>
//     </div>
//   );
// }


// import { useState, useRef } from "react";
// import { useLocation } from "react-router-dom";
// import { Volume2 } from "lucide-react"; // Using Lucide React for the speaker icon
// import './styles/Communication.css'; 
// import {  useEffect } from "react";

// export default function Communication() {
//   const location = useLocation();
//   const aadhaar = location.state?.aadhaar || "";
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState("");
//   const [loading, setLoading] = useState(false);
//   const chatBoxRef = useRef(null);

  

//   const formatBotResponse = (content) => {
//     const lines = content.split("\n").map((line) => line.trim()).filter(Boolean); // Remove empty lines
//     return lines.map((line, index) => 
//       line.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
//     );
//   };
  
//   const TypingResponse = ({ content }) => {
//     const formattedLines = formatBotResponse(content);
//     const [displayedLines, setDisplayedLines] = useState([]);
  
//     useEffect(() => {
//       let index = 0;
  
//       const interval = setInterval(() => {
//         if (index < formattedLines.length) {
//           setDisplayedLines((prevLines) => [...prevLines, formattedLines[index]]);
//           index++;
//         } else {
//           clearInterval(interval);
//         }
//       }, 500); // Adjust speed as needed
  
//       return () => clearInterval(interval);
//     }, [content]);
  
//     return (
//       <div>
//         {displayedLines.map((line, index) => (
//           <p key={index} dangerouslySetInnerHTML={{ __html: line }} />
//         ))}
//       </div>
//     );
//   };
  
  


//   const handleSend = async () => {
//     if (!input.trim()) return;

//     const newMessages = [...messages, { role: "user", content: input }];
//     setMessages(newMessages);
//     setInput("");
//     setLoading(true);

//     try {
//       const response = await fetch("https://ayurbotserver.vercel.app/chat", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ messages: newMessages })
//       });
//       const data = await response.json();
//       setMessages([...newMessages, { role: "bot", content: formatBotResponse(data.response) }]);
//     } catch (error) {
//       alert("Error connecting to chat");
//     }
//     setLoading(false);
//   };

//   const speakText = (text) => {
//     const utterance = new SpeechSynthesisUtterance(text);
//     utterance.lang = "en-US"; // You can change the language as needed
//     speechSynthesis.speak(utterance);
//   };

//   return (
//     <div className="communication-container">
//       <main className="communication-main">
//         <div className="communication-chat-section">
//           <h2 className="communication-title">AyurBot Chat</h2>
//           <div className="communication-messages" ref={chatBoxRef}>
//             {messages.map((msg, index) => (
//               <div key={index} className={`communication-message communication-${msg.role}-message`}>
//                 {msg.content}
//                 {msg.role === "bot" && (
//                   <button 
//                     className="communication-speaker-button" 
//                     onClick={() => speakText(msg.content)}
//                   >
//                     <Volume2 size={20} />
//                   </button>
//                 )}
//               </div>
//             ))}
//           </div>
//           <div className="communication-input-container">
//             <input
//               type="text"
//               value={input}
//               onChange={(e) => setInput(e.target.value)}
//               placeholder="Type your message..."
//               className="communication-input"
//             />
//             <button 
//               onClick={handleSend} 
//               disabled={loading} 
//               className="communication-send-button"
//             >
//               Send
//             </button>
//           </div>
//         </div>
//       </main>
//     </div>
//   );
// }



import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { Volume2 } from "lucide-react"; // Using Lucide React for the speaker icon
import "./styles/Communication.css";

export default function Communication() {
  const location = useLocation();
  const aadhaar = location.state?.aadhaar || "";
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatBoxRef = useRef(null);

  // Function to format bot response for bold text
  const formatBotResponse = (content) => {
    const lines = content.split("\n").map((line) => line.trim()).filter(Boolean); // Remove empty lines
    return lines.map((line) =>
      line.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
    );
  };

  // Component to show bot's response as a typing effect
  const TypingResponse = ({ content }) => {
    const formattedLines = formatBotResponse(content);
    const [displayedLines, setDisplayedLines] = useState([]);

    useEffect(() => {
      let index = 0;

      const interval = setInterval(() => {
        if (index < formattedLines.length) {
          setDisplayedLines((prevLines) => [...prevLines, formattedLines[index]]);
          index++;
        } else {
          clearInterval(interval);
        }
      }, 500); // Adjust speed as needed

      return () => clearInterval(interval);
    }, [content]);

    return (
      <div>
        {displayedLines.map((line, index) => (
          <p key={index} dangerouslySetInnerHTML={{ __html: line }} />
        ))}
      </div>
    );
  };

  // Handle sending message
  const handleSend = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("https://ayurbotserver.vercel.app/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages }),
      });
      const data = await response.json();
      
      // Add bot response with formatted content
      setMessages([...newMessages, { role: "bot", content: data.response }]);
    } catch (error) {
      alert("Error connecting to chat");
    }
    setLoading(false);
  };

  // Function to speak the text aloud
  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US"; // You can change the language as needed
    speechSynthesis.speak(utterance);
  };

  return (
    <div className="communication-container">
      <main className="communication-main">
        <div className="communication-chat-section">
          <h2 className="communication-title">AyurBot Chat</h2>
          <div className="communication-messages" ref={chatBoxRef}>
            {messages.map((msg, index) => (
              <div key={index} className={`communication-message communication-${msg.role}-message`}>
                {msg.role === "bot" ? <TypingResponse content={msg.content} /> : msg.content}
                {msg.role === "bot" && (
                  <button 
                    className="communication-speaker-button" 
                    onClick={() => speakText(msg.content)}
                  >
                    <Volume2 size={20} />
                  </button>
                )}
              </div>
            ))}
          </div>
          <div className="communication-input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="communication-input"
            />
            <button 
              onClick={handleSend} 
              disabled={loading} 
              className="communication-send-button"
            >
              Send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

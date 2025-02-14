import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import Home from './pages/Home'; // You should create corresponding page components
import Myths from './pages/Myths';
import Geetha from './pages/Geetha';
import About from './pages/About';
import Footer from './components/Footer';
import LanguageSelection from './components/LanguageSelection';
import ChatPage from './components/ChatPage';
import Chat from './pages/Chat';
import CommunicationPage from './components/CommunicationPage';

function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        {/* Default route for Home page */}
        <Route path="/" element={<Home />} />
        <Route path="/home" element={<Home />} />
        <Route path="/myths" element={<Myths />} />
        <Route path="/geetha" element={<Geetha />} />
        <Route path="/about" element={<About />} />
        <Route path="/language-selection" element={<LanguageSelection />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat_dynamic" element={<Chat />} />
        <Route path="/communication" element={<CommunicationPage />} />
      </Routes>
      <Footer />
    </Router>

  );
}

export default App;

// src/components/BotIcon.js
import React from 'react';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom
import './styles/BotIcon.css';

const BotIcon = () => {
  return (
    <div className="bot-icon">
      {/* Use Link instead of anchor tag */}
      <Link to="/language-selection">
        <img src="./assets/logo.jpg" alt="Bot Icon" />
      </Link>
    </div>
  );
};

export default BotIcon;

// src/pages/Home.js
import React from 'react';
import Content from '../components/content' // Import the Content component
import BotIcon from '../components/BotIcon';  // Import the BotIcon component

const Home = () => {
    return (
        <div>
            <Content />  {/* Render the Content component */}
            <BotIcon />  {/* Render the BotIcon component */}
        </div>
    );
};

export default Home;


import React from 'react';
import './styles/content.css';


const Content = () => {
    return (
        <div className="content1">
            <div className="left-section">
                <img src="./assets/logo1.png" alt="Tree Illustration" className="tree" />
                <div className="images-circle">
                    <img src="./assets/image1.png" alt="Image 1" />
                    <img src="./assets/image2.png" alt="Image 2" />
                    <img src="./assets/image3.png" alt="Image 3" />
                    <img src="./assets/image4.png" alt="Image 4" />
                    <img src="./assets/image5.png" alt="Image 5" />
                    <img src="./assets/image6.png" alt="Image 6" />
                </div>
            </div>
            <div className="right-section">
                <h1>AyurBot</h1>
                <p className="description">
                    Healthcare with Advanced Ayurvedic Solutions
                    
                </p>
                <p className="features">
                <ul>
                        <li>Multilingual and Culturally Relevant Interaction</li>
                        <li>Holistic Health Recommendations</li>
                        <li>Integration of Bhagavad Gita Teachings</li>
                        <li>Myth-Busting and Fact-Based Education</li>
                    </ul>
                </p>
            </div>
        </div>
    );
};

export default Content;

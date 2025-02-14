// // src/pages/About.js
// import React from 'react';
// import '../components/styles/About.css';

// const About = () => {
//   return (
//     <div>

//       <div className="container1">
//         <section className="section">
//           <img src="./assets/vision-image.jpg" alt="Vision" />
//           <div>
//             <h2>Our Vision</h2>
//             <p>
//               To make Ayurvedic knowledge accessible to everyone, empowering people to lead healthier and more balanced lives through natural and holistic methods.
//             </p>
//           </div>
//         </section>

//         <section className="section">
//           <img src="./assets/ayurvedic.jpg" alt="What We Offer" />
//           <div>
//             <h2>What We Offer</h2>
//             <h3>Ayurvedic Medicine</h3>
//             <p>
//               Ayurvedic medicines promote holistic healing by addressing the root cause of illnesses rather than just symptoms...
//             </p>
//           </div>
//         </section>

//         <section className="section">
//           <img src="./assets/geetha.jpg" alt="Bhagavad Geetha" />
//           <div>
//             <h3>Bhagavad Geeta</h3>
//             <p>
//               The Bhagavad Gita offers profound guidance for calming the mind and reducing stress...
//             </p>
//           </div>
//         </section>

//         <section className="section">
//           <img src="./assets/myths.jpg" alt="Myths" />
//           <div>
//             <h3>Myths</h3>
//             <p>
//               Myths often serve as a bridge between ancient beliefs and scientific reasoning...
//             </p>
//           </div>
//         </section>

//         <section className="section">
//           <img src="./assets/howitworks.jpg" alt="How It Works" />
//           <div>
//             <h2>How It Works</h2>
//             <p>
//               Users can interact with our intelligent bot to discuss their health concerns...
//             </p>
//           </div>
//         </section>

//         <h2>Meet the Team</h2>
//         <div className="team">
//           <div className="team-member">
//             <img src="./assets/pari.jpg" alt="Team Member 1" />
//             <h3>S.Parimala Sri</h3>
//             <p>Backend Developer</p>
//           </div>
//           <div className="team-member">
//             <img src="./assets/dev.jpg" alt="Team Member 2" />
//             <h3>T.Devika</h3>
//             <p>Backend Developer</p>
//           </div>
//           <div className="team-member">
//             <img src=",/assets/team-member3.jpg" alt="Team Member 3" />
//             <h3>T.Chaitanya</h3>
//             <p>Frontend Designer</p>
//           </div>
//           <div className="team-member">
//             <img src="./assets/vamsi.jpg" alt="Team Member 4" />
//             <h3>V.Sri Krishna Vamsi</h3>
//             <p>Frontend Designer</p>
//           </div>
//         </div>
//       </div>

     
//     </div>
//   );
// };

// export default About;



import React from 'react';
import '../components/styles/About.css';

const About = () => {
  return (
    <div className="about-page">
      <div className="container1">
        

        <section className="section">
          <img src="./assets/image.png" alt="Vision" />
          <div>
            <h2>Our Vision</h2>
            <p>
              Our mission is to make Ayurvedic knowledge accessible to everyone, empowering individuals to lead healthier, more balanced lives using natural and holistic methods.
            </p>
          </div>
        </section>

        <section className="section">
          <img src="./assets/ayurvedic.jpg" alt="What We Offer" />
          <div>
            <h2>What We Offer</h2>
            <h3>Ayurvedic Medicine</h3>
            <p>
              Ayurvedic medicine promotes holistic healing by addressing the root causes of illnesses rather than just alleviating symptoms. Our platform provides personalized recommendations based on your unique constitution.
            </p>
            <h3>Diet & Nutrition</h3>
            <p>
              Proper nutrition is at the heart of Ayurveda. We provide guidance on Ayurvedic diets tailored to your body type and health concerns.
            </p>
          </div>
        </section>

        <section className="section">
          <img src="./assets/geetha.jpg" alt="Bhagavad Geeta" />
          <div>
            <h2>Spiritual Well-Being</h2>
            <h3>Bhagavad Gita</h3>
            <p>
              The Bhagavad Gita offers profound guidance for calming the mind, reducing stress, and leading a meaningful life. Our platform integrates these teachings into wellness practices.
            </p>
          </div>
        </section>

        <section className="section">
          <img src="./assets/myths.jpg" alt="Myths" />
          <div>
            <h2>Busting Myths</h2>
            <p>
              Many misconceptions exist about Ayurveda. We provide evidence-based insights to help you distinguish between myths and facts, ensuring a well-informed approach to health.
            </p>
          </div>
        </section>

        <section className="section">
          <img src="./assets/howitworks.jpg" alt="How It Works" />
          <div>
            <h2>How It Works</h2>
            <p>
              Users can interact with our intelligent chatbot to discuss their health concerns. Our AI-powered system provides personalized Ayurvedic solutions, diet plans, and wellness strategies based on individual health conditions.
            </p>
            <p>
              The chatbot also offers recommendations on yoga, meditation, and mindfulness practices to support holistic well-being.
            </p>
          </div>
        </section>

        <h2 className="team-heading">Meet the Team</h2>
        <p className="team-intro">A passionate team dedicated to bringing Ayurveda and technology together.</p>
        <div className="team">
          <div className="team-member">
            <img src="./assets/pari.jpg" alt="S.Parimala Sri" />
            <h3>S.Parimala Sri</h3>
            <p>Backend Developer</p>
          </div>
          <div className="team-member">
            <img src="./assets/dev.jpg" alt="T.Devika" />
            <h3>T.Devika</h3>
            <p>Backend Developer</p>
          </div>
          <div className="team-member">
            <img src="./assets/team-member3.jpg" alt="T.Chaitanya" />
            <h3>T.Chaitanya</h3>
            <p>Frontend Designer</p>
          </div>
          <div className="team-member">
            <img src="./assets/vamsi.jpg" alt="V.Sri Krishna Vamsi" />
            <h3>V.Sri Krishna Vamsi</h3>
            <p>Frontend Designer</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;


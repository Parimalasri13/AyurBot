// import { useState } from "react";

// import './styles/authentication.css';

// export default function AadhaarAuth() {
//   const [aadhaar, setAadhaar] = useState("");
//   const [error, setError] = useState("");

//   const handleChange = (e) => {
//     const value = e.target.value.replace(/\D/g, ""); // Allow only digits
//     setAadhaar(value);
//     setError(value.length === 12 ? "" : "Aadhaar number must be 12 digits");
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (aadhaar.length !== 12) {
//       setError("Aadhaar number must be 12 digits");
//       return;
//     }
//     try {
//       const response = await fetch("https://ayurbotserver.vercel.app/auth", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ aadhaar }),
//       });
//       const data = await response.json();
//       alert(data.message);
//     } catch (error) {
//       alert("Error connecting to server");
//     }
//   };

//   return (
//     <div className="auth-container">
//       <div className="auth-box">
//         {/* Image Section */}
//         <div className="auth-image">
//           <img src="./assets/aadhar.jpg" alt="Aadhaar" />
//         </div>
//         {/* Form Section */}
//         <div className="auth-form">
//           <h2>Aadhaar Authentication</h2>
//           <form onSubmit={handleSubmit}>
//             <label>Enter Aadhaar Number</label>
//             <input
//               type="text"
//               maxLength="12"
//               value={aadhaar}
//               onChange={handleChange}
//             />
//             {error && <p className="error">{error}</p>}
//             <button type="submit">Submit</button>
//           </form>
//         </div>
//       </div>
//     </div>
//   );
// }




import { useState } from "react";
import { useNavigate } from "react-router-dom";  // Import useNavigate
import "./styles/authentication.css";

export default function AadhaarAuth() {
  const [aadhaar, setAadhaar] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();  // Initialize useNavigate

  const handleChange = (e) => {
    const value = e.target.value.replace(/\D/g, ""); // Allow only digits
    setAadhaar(value);
    setError(value.length === 12 ? "" : "Aadhaar number must be 12 digits");
  };

  const handleSubmit = async (e) => {
    
    e.preventDefault();
    if (aadhaar.length !== 12) {
        setError("Aadhaar number must be 12 digits");
        return;
      } 
      
    try {
        
        const response = await fetch("https://ayurbotserver.vercel.app/auth", {
            credentials: "include",
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ aadhaar }),
          
      });
      const data = await response.json();
      
      if (response.ok) {
        alert(data.message);
        navigate("/communication"); // Redirect to communication page
      } else {
        alert("Authentication failed");
      }
    } catch (error) {
      alert("Error connecting to server");
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        {/* Image Section */}
        <div className="auth-image">
          <img src="./assets/aadhar.jpg" alt="Aadhaar" />
        </div>
        {/* Form Section */}
        <div className="auth-form">
          <h2>Aadhaar Authentication</h2>
          <form onSubmit={handleSubmit}>
            <label>Enter Aadhaar Number</label>

            <input
              type="text"
              maxLength="12"
              value={aadhaar}
              onChange={handleChange}
            />
            {error && <p className="error">{error}</p>}
            <h6>If you Dont have account use - 1000000000000</h6>
            <button type="submit">Submit</button>
          </form>
        </div>
      </div>
    </div>
  );
}

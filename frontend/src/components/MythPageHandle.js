import "./styles/MythPageHandle.css";
import myths from "./myths";
import MythPage from "./MythsPage";
import { useState } from "react";
const MythPageHandle = () => {
  const [sform , setSform] = useState(false);
  return (
    <>
      <div className="fullpage">
        {/* <h1 className="heading">Superstitions and their Reasons</h1> */}
        <div className="container">
          {myths.map((m, index) => (
            <MythPage
              key={index}
              image={m.img}
              question={m.ques}
              reason={m.resn}
            />
          ))}
        </div>
        <div className="parent">
          <div className="contribute">
            <p >
              Want To Contribute Any Superstions ?{" "}
            <button  style={{ color: 'rgb(35, 165, 165)' }} onClick={()=>setSform(!sform) }>Click Here</button>
            </p>
          </div>
        </div>
        { sform && <div className="contribute-form">
        <form action="/">
            <input type="text" placeholder="Enter Superstition"  /> <br />
            <input type="text" placeholder="Enter Reason"  /> <br />
            <button className='cbutton' style={{ color: 'rgb(35, 165, 165)' }}>Submit</button>

        </form>
    </div>}
      </div>
    </>
  );
};
export default MythPageHandle;

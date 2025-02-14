import './styles/MythsPage.css'

const MythPage = (props)=>{  
  
    return ( 
      <>
        <div className="card">
          <div className="card-inner">
            <div className="front">
                <img src={props.image} alt="templebell" />
                <p>{props.question}</p>  
            </div>
            <div className="back">
                <p >{props.reason}</p>
            </div>
          </div>
        </div>
      </>
    );
} 
export default MythPage
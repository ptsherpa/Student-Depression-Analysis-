import React, { useState } from "react";

const App = () => {
  const [inputs, setInputs] = useState({
    gender: "",
    age: "",
    academicPressure: "",
    cgpa: "",
    studySatisfaction: "",
    sleepDuration: "",
    workStudyHours: "",
    financialStress: "",
    suicidalThoughts: "",
    familyHistory: "",
    dietaryHabits: "",
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSelectChange = (name, value) => {
    setInputs({ ...inputs, [name]: value });
  };

  const handleInputChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      // Validate Inputs
      for (const key in inputs) {
        if (!inputs[key]) {
          setError(`Please provide a valid value for ${key.replace(/([A-Z])/g, " $1")}`);
          return;
        }
      }

      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputs),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data.prediction);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("An error occurred. Please try again.");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Depression Prediction</h1>

      <div style={{ marginBottom: "20px" }}>
        <label>Gender:</label>
        <select
          name="gender"
          value={inputs.gender}
          onChange={(e) => handleSelectChange("gender", e.target.value)}
        >
          <option value="">Select Gender</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Age:</label>
        <input
          type="number"
          name="age"
          placeholder="Enter Age"
          value={inputs.age}
          onChange={handleInputChange}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Academic Pressure (1-5):</label>
        <select
          name="academicPressure"
          value={inputs.academicPressure}
          onChange={(e) => handleSelectChange("academicPressure", e.target.value)}
        >
          <option value="">Select Academic Pressure</option>
          {[1, 2, 3, 4, 5].map((val) => (
            <option key={val} value={val}>
              {val}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>CGPA:</label>
        <input
          type="number"
          name="cgpa"
          step="0.1"
          placeholder="Enter CGPA"
          value={inputs.cgpa}
          onChange={handleInputChange}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Study Satisfaction (1-5):</label>
        <select
          name="studySatisfaction"
          value={inputs.studySatisfaction}
          onChange={(e) => handleSelectChange("studySatisfaction", e.target.value)}
        >
          <option value="">Select Study Satisfaction</option>
          {[1, 2, 3, 4, 5].map((val) => (
            <option key={val} value={val}>
              {val}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Sleep Duration (hours):</label>
        <select
          name="sleepDuration"
          value={inputs.sleepDuration}
          onChange={(e) => handleSelectChange("sleepDuration", e.target.value)}
        >
          <option value="">Select Sleep Duration</option>
          {[5, 6, 7, 8, 9].map((val) => (
            <option key={val} value={val}>
              {val} hours
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Work/Study Hours:</label>
        <input
          type="number"
          name="workStudyHours"
          placeholder="Enter Work/Study Hours"
          value={inputs.workStudyHours}
          onChange={handleInputChange}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Financial Stress (1-5):</label>
        <select
          name="financialStress"
          value={inputs.financialStress}
          onChange={(e) => handleSelectChange("financialStress", e.target.value)}
        >
          <option value="">Select Financial Stress</option>
          {[1, 2, 3, 4, 5].map((val) => (
            <option key={val} value={val}>
              {val}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Suicidal Thoughts:</label>
        <select
          name="suicidalThoughts"
          value={inputs.suicidalThoughts}
          onChange={(e) => handleSelectChange("suicidalThoughts", e.target.value)}
        >
          <option value="">Select Suicidal Thoughts</option>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Family History of Mental Illness:</label>
        <select
          name="familyHistory"
          value={inputs.familyHistory}
          onChange={(e) => handleSelectChange("familyHistory", e.target.value)}
        >
          <option value="">Select Family History</option>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Dietary Habits:</label>
        <select
          name="dietaryHabits"
          value={inputs.dietaryHabits}
          onChange={(e) => handleSelectChange("dietaryHabits", e.target.value)}
        >
          <option value="">Select Dietary Habits</option>
          <option value="Healthy">Healthy</option>
          <option value="Moderate">Moderate</option>
          <option value="Unhealthy">Unhealthy</option>
        </select>
      </div>

      <button onClick={handleSubmit}>Predict</button>

      {result !== null && <h2>Prediction: {result === 1 ? "Depression" : "No Depression"}</h2>}
      {error && <h2 style={{ color: "red" }}>{error}</h2>}
    </div>
  );
};

export default App;

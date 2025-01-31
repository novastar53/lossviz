import { useState } from 'react';

function FileUpload() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.onnx')) {
      setSelectedFile(file);
    } else {
      alert('Please select a valid ONNX model file.');
      event.target.value = '';
    }
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    
    if (!selectedFile) {
      alert('Please select an ONNX model file!');
      return;
    }

    // Here we would process the ONNX model and generate the loss landscape
    console.log('Processing model:', selectedFile.name);
    alert(`Analyzing loss landscape for ${selectedFile.name}`);
  };

  return (
    <div className="file-upload">
      <h2>Model Upload</h2>
      <form onSubmit={handleUpload}>
        <div className="upload-container">
          <div className="config-section">
            <label>ONNX Model File</label>
            <input
              type="file"
              accept=".onnx"
              onChange={handleFileSelect}
              className="file-input"
            />
            <small className="file-hint">
              Upload your neural network in ONNX format
            </small>
          </div>

          <button type="submit" className="upload-button">
            Analyze Loss Landscape
          </button>
        </div>
      </form>
      
      <div className="preview-container">
        {selectedFile && (
          <div>
            <h3>Selected Model:</h3>
            <ul className="file-list">
              <li>
                {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default FileUpload;
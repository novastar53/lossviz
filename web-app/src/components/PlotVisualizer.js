import { useState } from 'react';
import Plot from 'react-plotly.js';
import './PlotVisualizer.css';

function PlotVisualizer() {
  const [plotType, setPlotType] = useState('2d');
  
  // Placeholder data for empty plot
  const emptyPlotData = {
    x: [],
    y: [],
    z: [],
    type: plotType === '2d' ? 'contour' : 'surface',
    colorscale: [
      [0, '#1a1a1a'],
      [0.5, '#2196f3'],
      [1, '#ffffff']
    ]
  };

  const plotLayout = {
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    font: {
      color: 'black',
      size: 12
    },
    title: {
      text: 'Loss Landscape Visualization',
      font: {
        color: '#2196f3',
        size: 16
      }
    },
    xaxis: {
      title: 'Direction 1',
      gridcolor: '#e1e1e1',
      zerolinecolor: '#999999',
      showgrid: true,
      zeroline: true,
      showline: true,
      mirror: true,
      linecolor: '#999999'
    },
    yaxis: {
      title: 'Direction 2',
      gridcolor: '#e1e1e1',
      zerolinecolor: '#999999',
      showgrid: true,
      zeroline: true,
      showline: true,
      mirror: true,
      linecolor: '#999999'
    },
    scene: {
      xaxis: { 
        title: 'Direction 1',
        gridcolor: '#e1e1e1',
        zerolinecolor: '#999999',
        showgrid: true,
        zeroline: true,
        showline: true
      },
      yaxis: { 
        title: 'Direction 2',
        gridcolor: '#e1e1e1',
        zerolinecolor: '#999999',
        showgrid: true,
        zeroline: true,
        showline: true
      },
      zaxis: { 
        title: 'Loss',
        gridcolor: '#e1e1e1',
        zerolinecolor: '#999999',
        showgrid: true,
        zeroline: true,
        showline: true
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 }
      },
      bgcolor: 'white'
    },
    width: 800,
    height: 500,
    margin: { l: 60, r: 30, t: 50, b: 50 },
    showlegend: false
  };

  return (
    <div className="plot-visualizer">
      <div className="plot-controls">
        <div className="plot-type-controls">
          <button
            className={`plot-button ${plotType === '2d' ? 'active' : ''}`}
            onClick={() => setPlotType('2d')}
          >
            2D Contour Plot
          </button>
          <button
            className={`plot-button ${plotType === '3d' ? 'active' : ''}`}
            onClick={() => setPlotType('3d')}
          >
            3D Surface Plot
          </button>
        </div>
      </div>
      
      <div className="plot-container">
        <Plot
          data={[emptyPlotData]}
          layout={plotLayout}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
              format: 'png',
              filename: 'loss_landscape',
              height: 500,
              width: 800,
              scale: 2
            }
          }}
          style={{
            width: '100%',
            height: '100%',
            minHeight: '500px'
          }}
        />
      </div>
    </div>
  );
}

export default PlotVisualizer;

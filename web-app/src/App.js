import './App.css';
import FileUpload from './components/FileUpload';
import PlotVisualizer from './components/PlotVisualizer';
import Footer from './components/Footer';
import './components/FileUpload.css';
import './components/PlotVisualizer.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Loss Landscape Visualizer</h1>
        <p>Upload and visualize neural network loss landscapes with interactive 2D and 3D plots</p>
      </header>
      <main>
        <div className="content-grid">
          <aside>
            <FileUpload />
          </aside>
          <section>
            <PlotVisualizer />
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );
}

export default App;

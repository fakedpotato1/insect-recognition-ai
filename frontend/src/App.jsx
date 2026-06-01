import { useState } from "react";
import UploadBox from "./components/UploadBox";
import ResultModal from "./components/ResultModal";
import logo from "../assets/logo.png";
import { detectInsect } from "./api/DetectApi";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const handleDetect = async (file) => {
    setLoading(true);
    setResult(null);
    setError(null);
    setShowModal(false);

    try {
      const data = await detectInsect(file);
      setResult(data);
      setShowModal(true);
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setResult(null);
  };

  const handleReset = () => {
    setError(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-cyan-50 text-slate-900 flex flex-col">

      {/* NAVBAR */}
      <header className="w-full flex items-center px-4 sm:px-8 py-3 sm:py-4 border-b border-cyan-200 bg-cyan-800 sm:bg-white">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-cyan-100 flex items-center justify-center overflow-hidden flex-shrink-0">
            <img src={logo} alt="Logo" className="h-full w-full object-contain" />
          </div>
          <div className="text-base sm:text-lg font-bold leading-tight text-white sm:text-slate-900">
            Insect Recognition AI
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="flex-1 flex items-start sm:items-center justify-center px-0 sm:px-6 py-0 sm:py-10">
        <div className="w-full max-w-3xl">

          {/* CARD */}
          <div className="bg-white sm:border sm:border-slate-200 sm:shadow-2xl sm:rounded-2xl p-6 sm:p-10">

            {/* TITLE */}
            <div className="text-center mb-6 sm:mb-8">
              <h1 className="text-2xl sm:text-4xl font-bold tracking-wide">
                Insect Recognition AI
              </h1>
              <p className="text-slate-400 mt-2 text-sm sm:text-base">
                Upload an image for detection
              </p>
            </div>

            {/* UPLOAD BOX */}
            <div className="mt-4 sm:mt-6">
              <UploadBox onDetect={handleDetect} onReset={handleReset} />
            </div>

            {/* LOADING */}
            {loading && (
              <div className="mt-6 flex items-center justify-center gap-3 text-cyan-700">
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                <span className="text-sm font-medium">Analysing image...</span>
              </div>
            )}

            {/* ERROR */}
            {error && (
              <div className="mt-6 px-4 py-3 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm text-center">
                {error}
              </div>
            )}

          </div>
        </div>
      </main>

      {/* RESULT MODAL */}
      {showModal && (
        <ResultModal result={result} onClose={handleCloseModal} />
      )}

    </div>
  );
}

export default App;
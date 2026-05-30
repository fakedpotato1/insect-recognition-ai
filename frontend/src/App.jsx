import UploadBox from "./components/UploadBox";
import logo from "../assets/logo.png";

function App() {

  const handleDetect = (file) => {
    // file = the image passed up from UploadBox
    // call your detection/AI code here
    console.log("Detecting:", file);
  };

  return (
    <div className="min-h-screen bg-cyan-50 text-slate-900 flex flex-col">

      {/* NAVBAR */}
      <header className="w-full flex items-center px-4 sm:px-8 py-3 sm:py-4 border-b border-cyan-200 bg-cyan-800 sm:bg-white">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-cyan-100 flex items-center justify-center overflow-hidden flex-shrink-0">
            <img
              src={logo}
              alt="Logo"
              className="h-full w-full object-contain"
            />
          </div>

          <div className="text-base sm:text-lg font-bold leading-tight text-white sm:text-slate-900">
            Insect Recognition AI
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="flex-1 flex items-start sm:items-center justify-center px-0 sm:px-6 py-0 sm:py-10">
        <div className="w-full max-w-3xl">

          {/* CARD — flat on mobile, raised on desktop */}
          <div className="bg-white sm:border sm:border-slate-200 sm:shadow-2xl sm:rounded-2xl p-6 sm:p-10">

            {/* TITLE AREA */}
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
              <UploadBox onDetect={handleDetect} />
            </div>

          </div>

        </div>
      </main>

    </div>
  );
}

export default App;
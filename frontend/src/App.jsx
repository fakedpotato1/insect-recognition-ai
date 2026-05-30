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
      <header className="w-full flex items-center justify-between px-8 py-4 border-b border-cyan-200 bg-white">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-cyan-100 flex items-center justify-center overflow-hidden">
            <img
              src={logo}
              alt="Logo"
              className="h-full w-full object-contain"
            />
          </div>

          <div className="text-lg font-bold">
            Insect Recognition AI
          </div>
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-3xl">

          {/* CARD */}
          <div className="bg-white border border-slate-200 shadow-2xl rounded-2xl p-10">

            {/* TITLE AREA */}
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold tracking-wide">
                Insect Recognition AI
              </h1>

              <p className="text-slate-400 mt-2">
                Upload an image for detection
              </p>
            </div>

            {/* UPLOAD BOX AREA */}
            <div className="mt-6">
              <UploadBox onDetect={handleDetect} />
            </div>

          </div>

        </div>
      </main>

    </div>
  );
}

export default App;
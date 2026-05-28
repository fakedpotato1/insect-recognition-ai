import UploadBox from "./components/UploadBox";
import logo from "../assets/logo.png";

function App() {
  const handleFileSelect = (file) => {
    console.log("Selected file:", file);
  };

  return (
    <div className="min-h-screen bg-cyan-50 text-slate-900 flex flex-col">

      {/* NAVBAR */}
      <header className="w-full flex items-center justify-between px-8 py-4 border-b border-slate-200 bg-white">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center overflow-hidden">
            <img
              src={logo}
              alt="Logo"
              className="h-full w-full object-contain"
            />
          </div>

          <div className="text-lg font-semibold">
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
              <UploadBox />
            </div>

          </div>

        </div>
      </main>

    </div>
  );
}

export default App;
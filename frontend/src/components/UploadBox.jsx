import { useRef, useState, useEffect } from "react";
import { X, Upload, Camera, RefreshCw, ScanSearch } from "lucide-react";

function UploadBox({ onFileSelect, onDetect }) {
  const inputRef = useRef(null);
  const cameraRef = useRef(null);

  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFile = (file) => {
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview(imageUrl);
    setSelectedFile(file);

    if (onFileSelect) {
      onFileSelect(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const openFilePicker = () => {
    inputRef.current?.click();
  };

  const openCamera = () => {
    cameraRef.current?.click();
  };

  const resetInput = (input) => {
    if (input.current) {
      input.current.value = "";
    }
  };

  const removeImage = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview(null);
    setSelectedFile(null);

    inputRef.current && (inputRef.current.value = "");
    cameraRef.current && (cameraRef.current.value = "");

    if (onFileSelect) {
      onFileSelect(null);
    }
  };

  const handleDetect = () => {
    if (onDetect && selectedFile) {
      onDetect(selectedFile);
    }
  };

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  return (
    <div className="w-full">
      {/* Gallery Upload Input */}
      <input
        type="file"
        accept="image/*"
        ref={inputRef}
        className="hidden"
        onChange={(e) => {
          handleFile(e.target.files[0]);
          resetInput(inputRef);
        }}
      />

      {/* Camera Input (Mobile) */}
      <input
        type="file"
        accept="image/*"
        capture="environment"
        ref={cameraRef}
        className="hidden"
        onChange={(e) => {
          handleFile(e.target.files[0]);
          resetInput(cameraRef);
        }}
      />

      {/* Upload Area */}
      <div
        onClick={!preview ? openFilePicker : undefined}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`
          h-72 rounded-xl border-2 border-dashed
          flex items-center justify-center
          transition
          ${
            isDragging
              ? "border-cyan-400 bg-cyan-100"
              : "border-slate-700 hover:border-slate-500"
          }
        `}
      >
        {!preview ? (
          <div className="text-center px-4">
            <p className="text-slate-300 text-lg font-medium">
              Drag & Drop Image Here
            </p>

            <p className="text-slate-500 text-sm mt-2">
              or choose an option below
            </p>

            <div className="flex flex-col sm:flex-row gap-3 justify-center mt-5">
              {/* Primary — solid */}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  openFilePicker();
                }}
                className="flex items-center justify-center gap-2 px-5 py-2.5 bg-cyan-700 hover:bg-cyan-800 text-cyan-50 font-medium rounded-lg transition-colors"
              >
                <Upload size={16} />
                Upload Photo
              </button>

              {/* Secondary — outline */}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  openCamera();
                }}
                className="flex items-center justify-center gap-2 px-5 py-2.5 border-2 border-cyan-700 text-cyan-600 hover:bg-cyan-700 hover:text-cyan-50 font-medium rounded-lg transition-colors"
              >
                <Camera size={16} />
                Take Photo
              </button>
            </div>
          </div>
        ) : (
          <div className="relative h-full w-full">
            <img
              src={preview}
              alt="preview"
              className="h-full w-full object-contain rounded-xl"
            />

            {/* X button to clear image */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                removeImage();
              }}
              className="
                absolute top-2 right-2
                h-8 w-8
                flex items-center justify-center
                rounded-full
                bg-black/60 hover:bg-black/80
                text-white
                transition
              "
            >
              <X size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Actions after image selected */}
      {preview && (
        <div className="flex gap-3 mt-4">
          {/* Change Photo — outline, secondary */}
          <button
            type="button"
            onClick={openFilePicker}
            className="flex items-center justify-center gap-2 px-4 py-2.5 border-2 border-cyan-700 text-cyan-600 hover:bg-cyan-700 hover:text-cyan-50 font-medium rounded-lg transition-colors"
          >
            <RefreshCw size={16} />
            Change Photo
          </button>

          {/* Detect! — primary, calls onDetect prop back to App.jsx */}
          <button
            type="button"
            onClick={handleDetect}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-cyan-700 hover:bg-cyan-800 text-cyan-50 font-medium rounded-lg transition-colors"
          >
            <ScanSearch size={16} />
            Detect!
          </button>
        </div>
      )}
    </div>
  );
}

export default UploadBox;
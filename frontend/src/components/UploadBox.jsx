import { useRef, useState } from "react";

function UploadBox({ onFileSelect }) {
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Handle file selection
  const handleFile = (file) => {
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    setPreview(imageUrl);

    if (onFileSelect) {
      onFileSelect(file);
    }
  };

  // Drag events
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  return (
    <div className="w-full">

      {/* Hidden file input */}
      <input
        type="file"
        accept="image/*"
        ref={inputRef}
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />

      {/* Upload Area */}
      <div
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`
          h-72 rounded-xl border-2 border-dashed flex items-center justify-center cursor-pointer transition
          ${isDragging
            ? "border-emerald-400 bg-slate-800"
            : "border-slate-700 hover:border-slate-500"
          }
        `}
      >
        {!preview ? (
          <div className="text-center">
            <p className="text-slate-300 text-lg font-medium">
              Drag & Drop Image Here
            </p>
            <p className="text-slate-500 text-sm mt-2">
              or click to upload
            </p>
          </div>
        ) : (
          <img
            src={preview}
            alt="preview"
            className="h-full w-full object-contain rounded-xl"
          />
        )}
      </div>
    </div>
  );
}

export default UploadBox;
import { useState, useRef } from "react";
import { Mic, MicOff } from "lucide-react";

export function VoiceToggleButton() {
  const [listening, setListening] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const toggleMic = async () => {
    if (listening) {
      mediaRecorderRef.current?.stop();
      setListening(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          const audioFile = new File([audioBlob], "voice_input.webm", { type: "audio/webm" });

          const formData = new FormData();
          formData.append("file", audioFile);

          try {
            const response = await fetch("http://localhost:8000/api/chat/voice", {
              method: "POST",
              body: formData,
            });

            const result = await response.json();
            console.log("Transcript:", result.transcript);
          } catch (err) {
            console.error("Failed to send audio", err);
          }
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start();
        setListening(true);
      } catch (err) {
        console.error("Microphone access failed", err);
      }
    }
  };

  return (
    <button
      onClick={toggleMic}
      className={`px-4 py-1 rounded-md h-full transition-all ${listening ? "bg-red-100" : "bg-slate-100"}`}
      title="Toggle microphone"
    >
      {listening ? <MicOff className="text-red-500" /> : <Mic className="text-slate-600" />}
    </button>
  );
}

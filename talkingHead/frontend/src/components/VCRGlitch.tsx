// src/components/VCRGlitch.tsx
import { useEffect, useState } from "react";

export default function VCRGlitch() {
  const [glitch, setGlitch] = useState(false);

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout>;

    const triggerGlitch = () => {
      setGlitch(true);
      timeoutId = setTimeout(() => {
        setGlitch(false);
        const delay = Math.random() * 3000 + 1500;
        timeoutId = setTimeout(triggerGlitch, delay);
      }, 150);
    };

    triggerGlitch();
    return () => clearTimeout(timeoutId);
  }, []);

  return (
    <div
      className={`pointer-events-none fixed top-0 left-0 w-screen h-screen z-[9990] ${
        glitch ? "animate-vcrGlitch" : ""
      }`}
        style={{
        background:
            "repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(255, 0, 0, 0.05) 1px, rgba(0, 255, 255, 0.05) 2px)",
        mixBlendMode: "difference",  // Try 'difference' or 'hard-light'
        opacity: glitch ? 1 : 0,
        transition: "opacity 0.05s ease-in-out",
        }}
    />
  );
}

// src/components/StaticBackground.tsx
import { useEffect, useState } from "react";

const NUM_FRAMES = 200;

export default function StaticBackground() {
  const [frame, setFrame] = useState(0);

    useEffect(() => {
    let animationFrameId: number;
    let lastTimestamp = 0;
    const FRAME_INTERVAL = 1000 / 24; // target 15 FPS (adjust to taste)

    const render = (timestamp: number) => {
        if (timestamp - lastTimestamp >= FRAME_INTERVAL) {
        const randomFrame = Math.floor(Math.random() * NUM_FRAMES);
        setFrame(randomFrame);
        lastTimestamp = timestamp;
        }
        animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => cancelAnimationFrame(animationFrameId);
    }, []);

  return (
    <>
      {/* Solid black background */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          backgroundColor: "black",
          zIndex: -3,
        }}
      />

      {/* Static noise */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          backgroundImage: `url("/static_frames/static_${String(frame).padStart(3, "0")}.png")`,
          backgroundSize: "cover",
          opacity: 0.07,
          zIndex: -2,
          imageRendering: "pixelated",
          mixBlendMode: "screen" as const,
          pointerEvents: "none",
        }}
      />

        {/* Single scrolling scanline */}
        <div
        style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            background: `linear-gradient(
            to bottom,
            transparent 0%,
            rgba(40, 90, 0, 0.3) 50%,
            transparent 100%
            )`,
            backgroundRepeat: "no-repeat",
            backgroundSize: "100% 200%",
            animation: "scanlineSlide .001s linear infinite",
            zIndex: -1,
            pointerEvents: "none",
            mixBlendMode: "screen" as const,
        }}
        />

    </>
  );
}

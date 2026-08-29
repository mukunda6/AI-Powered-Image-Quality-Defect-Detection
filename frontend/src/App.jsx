import { useState, useEffect, useRef } from "react";
import UploadCard from "./components/UploadCard";
import ResultCard from "./components/ResultCard";
import "./App.css";

function App() {
  const [result, setResult] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationId;
    let width, height;
    let particles = [];

    const PARTICLE_COUNT = 140;
    const CONNECT_DISTANCE = 130;
    const COLOR = "56, 224, 224";

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    function createParticles() {
      particles = [];
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.4,
          vy: (Math.random() - 0.5) * 0.4,
          radius: Math.random() * 1.8 + 0.6,
          pulseOffset: Math.random() * Math.PI * 2,
        });
      }
    }

    function draw(time) {
      ctx.fillStyle = "#050914";
      ctx.fillRect(0, 0, width, height);

      for (let p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;

        const pulse = Math.sin(time / 800 + p.pulseOffset) * 0.5 + 0.5;
        const glow = 0.4 + pulse * 0.6;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${COLOR}, ${glow})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = `rgba(${COLOR}, 0.8)`;
        ctx.fill();
      }
      ctx.shadowBlur = 0;

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECT_DISTANCE) {
            const opacity = (1 - dist / CONNECT_DISTANCE) * 0.35;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(${COLOR}, ${opacity})`;
            ctx.lineWidth = 0.6;
            ctx.stroke();
          }
        }
      }

      animationId = requestAnimationFrame(draw);
    }

    resize();
    createParticles();
    animationId = requestAnimationFrame(draw);

    const handleResize = () => {
      resize();
      createParticles();
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <>
      <canvas ref={canvasRef} className="particle-canvas" />

      <div className="app">
        <header>
          <h1>AI-Powered Image Quality & Defect Detection</h1>
          <p>Upload an image to get a quality score and detect issues</p>
        </header>

        <main>
          <UploadCard onResult={setResult} />
          {result && <ResultCard result={result} />}
        </main>
      </div>
    </>
  );
}

export default App;
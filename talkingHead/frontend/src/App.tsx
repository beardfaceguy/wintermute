// src/App.tsx
import Chat from './components/Chat'
import StaticBackground from './components/StaticBackground'
import VCRGlitch from './components/VCRGlitch'
function App() {
  return (
    <div className="h-screen">
    <div className="hidden animate-vcrGlitch" />
    <StaticBackground />
    <VCRGlitch />
      <Chat />
      
    </div>
    
  )
}

export default App

'use client';

import { useEffect, useState } from 'react';

export default function HeroSection() {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      setIsVisible(scrollPosition < window.innerHeight);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const smoothScrollTo = (targetId: string) => {
    const target = document.getElementById(targetId);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="relative h-screen flex flex-col items-center justify-center overflow-hidden">
      {/* Sticky Header */}
      <header className="fixed top-0 left-0 w-full bg-white shadow-md z-50">
        <div className="max-w-8xl mx-auto px-9 py-3 flex justify-between items-center">
          <h1 className="text-lg font-bold text-gray-800">Cataract Detection AI</h1>
          <nav>
            <ul className="flex space-x-6">
              <li>
                <button
                  onClick={() => smoothScrollTo('hero-section')}
                  className="text-gray-800 hover:text-blue-500"
                >
                  Home
                </button>
              </li>
              <li>
                <button
                  onClick={() => smoothScrollTo('upload-section')}
                  className="text-gray-800 hover:text-blue-500"
                >
                  Upload
                </button>
              </li>
              <li>
                <button
                  onClick={() => smoothScrollTo('results-section')}
                  className="text-gray-800 hover:text-blue-500"
                >
                  Results
                </button>
              </li>
              <li>
                <button
                  onClick={() => smoothScrollTo('about-section')}
                  className="text-gray-800 hover:text-blue-500"
                >
                  About
                </button>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Hero Content */}
      <div
        id="hero-section"
        className={`relative z-10 text-center px-4 transition-opacity duration-500 ${
          isVisible ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <h1 className="text-3xl md:text-5xl font-bold text-gray-800 mb-4">Cataract Detection AI</h1>
        <p className="text-xl md:text-2xl text-gray-600">AI-powered precision for early cataract detection</p>
        <button
          onClick={() => smoothScrollTo('upload-section')}
          className="mt-8 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors text-lg"
        >
          Get Started
        </button>
      </div>

      {/* Background Visuals */}
      <div className="absolute inset-0 z-0 opacity-20">
        <svg className="w-full h-full" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="40" stroke="#4299e1" strokeWidth="0.5" fill="none" />
          <path d="M50 10 A40 40 0 0 1 90 50" stroke="#48bb78" strokeWidth="0.5" fill="none" />
          <path d="M50 90 A40 40 0 0 1 10 50" stroke="#48bb78" strokeWidth="0.5" fill="none" />
        </svg>
      </div>
    </section>
  );
}

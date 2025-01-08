'use client';

import { useEffect, useState } from 'react';

export default function AboutSection() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      setIsVisible(scrollPosition >= window.innerHeight * 2.5);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <section
      id="about-section"
      className={`py-16 px-4 bg-gray-50 transition-opacity duration-500 ${
        isVisible ? 'opacity-100' : 'opacity-0'
      }`}
    >
      <div className="max-w-3xl mx-auto text-center">
        <h2 className="text-3xl font-bold mb-8">About Our Technology</h2>
        <p className="text-lg text-gray-700 leading-relaxed">
          Our app uses state-of-the-art AI models trained on thousands of images to identify cataracts with exceptional accuracy. Upload an image, and our AI provides instant predictions to support early diagnosis and care.
        </p>
      </div>
    </section>
  );
}

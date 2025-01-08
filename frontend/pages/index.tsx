import { useState } from 'react';
import HeroSection from '@/components/HeroSection';
import UploadSection from '@/components/UploadSection';
import ResultsSection from '@/components/ResultsSection';
import AboutSection from '@/components/AboutSection';

export default function Home() {
    const [prediction, setPrediction] = useState<string | null>(null);

    return (
        <main className="min-h-screen bg-gray-50">
            <HeroSection />
            <UploadSection setPrediction={setPrediction} />
            <ResultsSection prediction={prediction} />
            <AboutSection/>
        </main>
    );
}

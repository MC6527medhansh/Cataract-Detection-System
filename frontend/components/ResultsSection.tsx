'use client';

export default function ResultsSection({ prediction }: { prediction: string | null }) {
    return (
        <section id="results-section" className="py-16 px-4 bg-white">
            <div className="max-w-3xl mx-auto text-center">
                <h2 className="text-3xl font-bold mb-8">Prediction Results</h2>
                {prediction ? (
                    <div className="bg-blue-50 rounded-lg p-8">
                        <p className="text-2xl font-semibold text-blue-800">
                            Prediction: {prediction}
                        </p>
                    </div>
                ) : (
                    <p className="text-gray-500">Upload an image to see the prediction results.</p>
                )}
            </div>
        </section>
    );
}

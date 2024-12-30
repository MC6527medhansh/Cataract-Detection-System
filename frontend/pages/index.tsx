import React, { useState } from 'react';

export default function Home() {
    const [file, setFile] = useState<File | null>(null); // File or null
    const [prediction, setPrediction] = useState<number | null>(null); // Prediction is a number or null
    const [loading, setLoading] = useState<boolean>(false); // Boolean for loading state

    // Add type for event parameter
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]); // Set the selected file
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;
    
        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
    
        try {
            console.log('Sending file to backend...');
            const response = await fetch('http://127.0.0.1:5001/predict', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            console.log('Backend response:', data);
            setPrediction(data.prediction);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };
    

    return (
        <div>
            <h1>Cataract Detection</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit" disabled={loading}>
                    {loading ? 'Predicting...' : 'Upload & Predict'}
                </button>
            </form>
            {prediction !== null && <p>Prediction: {prediction}</p>}
        </div>
    );
}

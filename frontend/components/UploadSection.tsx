'use client'

import { useState, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'

export default function UploadSection({ setPrediction }: { setPrediction: (prediction: string | null) => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [isVisible, setIsVisible] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY
      setIsVisible(scrollPosition >= window.innerHeight * 0.5)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const onDrop = (acceptedFiles: File[]) => {
    setFile(acceptedFiles[0])
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false,
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      setIsUploading(true)
      const response = await fetch('http://127.0.0.1:5001/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to fetch prediction from backend')
      }

      const result = await response.json()
      setPrediction(result.prediction.toString())

      const resultsSection = document.getElementById('results-section')
      if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' })
      }
    } catch (error) {
      console.error('Error uploading file:', error)
      setPrediction(null)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <section
      id="upload-section"
      className={`py-16 px-4 transition-opacity duration-500 ${isVisible ? 'opacity-100' : 'opacity-0'}`}
    >
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-8">Upload an Eye Image</h2>
        <form onSubmit={handleSubmit}>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
            }`}
          >
            <input {...getInputProps()} />
            {file ? (
              <p className="text-gray-700">Selected file: {file.name}</p>
            ) : isDragActive ? (
              <p className="text-blue-500">Drop the image here</p>
            ) : (
              <p className="text-gray-500">Drag and drop an eye image here, or click to select a file</p>
            )}
          </div>
          <div className="mt-6 text-center">
            <button
              type="submit"
              disabled={!file || isUploading}
              className={`bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-lg`}
            >
              {isUploading ? 'Predicting...' : 'Predict Cataracts'}
            </button>
          </div>
        </form>
      </div>
    </section>
  )
}

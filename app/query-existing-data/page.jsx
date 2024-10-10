'use client'

import { useState } from 'react';
import Link from 'next/link';

const modelOptions = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4o-2024-05-13", "gpt-4o-mini"];

const QueryExistingData = () => {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(modelOptions[0]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, model: selectedModel }),
      });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
      } else {
        setResult({ error: data.error });
      }
    } catch (error) {
      setResult({ error: error.message });
    }
    setIsLoading(false);
  };

  return (
    <div className='flex flex-col items-center justify-center min-h-screen p-4'>
      <h1 className='text-4xl font-bold mb-8 text-black'>Query Existing Data</h1>
      <form onSubmit={handleSubmit} className='w-full max-w-md'>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your question"
          className='input w-full mb-4 text-black'
        />
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className='input w-full mb-4 text-black'
        >
          {modelOptions.map((model) => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
        <button type="submit" className='button w-full' disabled={isLoading}>
          {isLoading ? 'Processing...' : 'ASK YOUR QUERY'}
        </button>
      </form>
      {result && (
        <div className='mt-8 p-4 bg-white rounded shadow w-full max-w-md'>
          <h2 className='text-2xl font-bold mb-4 text-black'>Result:</h2>
          {result.error ? (
            <p className='text-red-500'>{result.error}</p>
          ) : (
            <>
              <h3 className='text-xl font-semibold mb-2 text-black'>SQL Query:</h3>
              <pre className='text-black whitespace-pre-wrap bg-gray-100 p-2 rounded'>{result.sqlQuery}</pre>
              <h3 className='text-xl font-semibold mt-4 mb-2 text-black'>Query Result:</h3>
              <pre className='text-black whitespace-pre-wrap bg-gray-100 p-2 rounded'>{JSON.stringify(result.queryResult, null, 2)}</pre>
            </>
          )}
        </div>
      )}
      <Link href="/" className='button mt-8'>
        Back to Home
      </Link>
    </div>
  )
}

export default QueryExistingData;
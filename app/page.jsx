'use client';

import Link from 'next/link';
import { useSession } from "next-auth/react";
import SignInButton from "@components/SignInButton";

const Home = () => {
  const { data: session, status } = useSession();

  console.log("Session:", session); // Debugging: Log the session
  console.log("Status:", status); // Debugging: Log the status

  return (
    <div className='flex flex-col items-center justify-center min-h-screen'>
      <h1 className='text-4xl font-bold mb-8 text-primary'>IRCC 🍁 TICASUK</h1>
      <SignInButton />
      
      {/* Session status with improved contrast */}
      <div className='mt-4 p-2 bg-gray-800 text-white rounded'>
        <p className='text-lg font-semibold'>
          Session status: <span className='font-bold'>{status}</span>
        </p>
      </div>

      {status === "authenticated" ? (
        <div className='space-y-4 mt-8'>
          <Link href="/welcome" className='button block text-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded'>
            WELCOME
          </Link>
          <Link href="/query-existing-data" className='button block text-center bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded'>
            Query existing Data
          </Link>
        </div>
      ) : (
        <div className='mt-6 p-4 bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700'>
          <p className='text-lg font-semibold'>Please sign in to access the main content.</p>
        </div>
      )}
    </div>
  )
}

export default Home;
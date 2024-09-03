'use client';

import { signIn, signOut, useSession } from "next-auth/react";

export default function SignInButton() {
  const { data: session, status } = useSession();

  console.log("Session in SignInButton:", session); // Debugging: Log the session
  console.log("Auth status:", status); // Debugging: Log the authentication status

  if (session && session.user) {
    return (
      <div className="flex flex-col items-center space-y-2">
        <p className="text-black font-semibold">Signed in as {session.user.email}</p>
        <button 
          onClick={() => signOut()} 
          className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded shadow-md transition duration-300 ease-in-out"
        >
          Sign out
        </button>
      </div>
    );
  }
  return (
    <button 
      onClick={() => signIn('google')} 
      className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded shadow-md transition duration-300 ease-in-out"
    >
      Sign in with Google
    </button>
  );
}
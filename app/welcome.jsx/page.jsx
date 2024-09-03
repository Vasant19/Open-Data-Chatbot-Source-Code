import Link from 'next/link';

const Welcome = () => {
  return (
    <div className='flex flex-col items-center justify-center min-h-screen'>
      <h1 className='text-4xl font-bold mb-8'>Welcome to TICASUK</h1>
      <p className='text-lg mb-8'>
        TICASUK: Where the four winds gather their treasures from all parts of the world, the greatest of which is knowledge.
      </p>
      <Link href="/" className='button'>
        Back to Home
      </Link>
    </div>
  )
}

export default Welcome;
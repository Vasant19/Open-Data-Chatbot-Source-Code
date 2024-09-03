import '@styles/globals.css'
import Image from 'next/image'
import SessionProvider from '@components/SessionProvider';
import { getServerSession } from "next-auth/next"

export const metadata = {
    title: 'IRCC 🍁 TICASUK',
    description: 'Interactive Data to Understand Data'
}

export default async function RootLayout({ children }) {
  const session = await getServerSession();

  return (
    <html lang="en">
      <body className='bg-flag'>
        <SessionProvider session={session}>
          <div className='main'>
            <div className='gradient' />
          </div>
          <header className='w-full flex justify-between items-center p-4'>
            <Image src="/cf3.png" alt="IRCC Logo" width={150} height={50} />
            <a href="https://www.canada.ca/en/immigration-refugees-citizenship.html" target="_blank" rel="noopener noreferrer">
              IRCC Website
            </a>
          </header>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
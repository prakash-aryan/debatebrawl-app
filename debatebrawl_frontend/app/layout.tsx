import { Metadata } from 'next';
import { Providers } from './providers';
import Navbar from '@/components/ui/Navbar';
import Footer from '@/components/ui/Footer';
import '../styles/main.css';

const title = 'DebateBrawl - AI-Powered Debate Platform';
const description = 'Engage in thrilling debates with AI assistance and advanced strategies.';

export const metadata: Metadata = {
  title,
  description,
  openGraph: {
    title,
    description
  }
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <Navbar />
          <main className="min-h-screen pt-16">
            {children}
          </main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}
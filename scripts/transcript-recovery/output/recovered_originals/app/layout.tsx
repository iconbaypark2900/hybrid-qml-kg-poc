import type { Metadata } from 'next';
import './globals.css';
import { Sidebar } from '@/components/sidebar/Sidebar';

export const metadata: Metadata = {
  title: 'Hetionet · QML',
  description: 'Quantum-classical drug repurposing dashboard on Hetionet v1.0',
  icons: { icon: '/favicon.ico' }
};

// Apply the sidebar-collapsed class before paint to prevent a flash of expanded sidebar.
const SIDEBAR_INIT_SCRIPT = `
(function(){
  try {
    if (window.localStorage.getItem('hetqml.sidebar-collapsed') === 'true') {
      document.documentElement.classList.add('preload-sidebar-collapsed');
    }
  } catch(e){}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: SIDEBAR_INIT_SCRIPT }} />
        <style dangerouslySetInnerHTML={{
          __html: `
            html.preload-sidebar-collapsed body { /* applied at React mount */ }
          `
        }} />
      </head>
      <body>
        <Sidebar />
        <main className="main">{children}</main>
      </body>
    </html>
  );
}

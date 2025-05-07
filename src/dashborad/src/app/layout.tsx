import type { Metadata } from "next";
import { MantineProvider, ColorSchemeScript } from '@mantine/core';
import '@mantine/core/styles.css';
import '@mantine/charts/styles.css';

export const metadata: Metadata = {
  title: "Model Performance Dashboard",
  description: "Dashboard to visualize model performance metrics",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <ColorSchemeScript />
        <link
          rel="stylesheet"
          as="style"
          crossOrigin="anonymous"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"
        />
      </head>
      <body style={{ fontFamily: 'Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, \'Helvetica Neue\', \'Segoe UI\', \'Apple SD Gothic Neo\', \'Noto Sans KR\', \'Malgun Gothic\', \'Apple Color Emoji\', \'Segoe UI Emoji\', \'Segoe UI Symbol\', sans-serif' }}>
        <MantineProvider defaultColorScheme="light">
          {children}
        </MantineProvider>
      </body>
    </html>
  );
}

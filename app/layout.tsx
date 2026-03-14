import type { Metadata } from "next";
import { Cormorant_Garamond, IBM_Plex_Sans } from "next/font/google";

import "@/app/globals.css";

const display = Cormorant_Garamond({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600"],
});

const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Hopper Studio",
  description: "Generate and publish Hopper-inspired scenes.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${sans.variable}`}>{children}</body>
    </html>
  );
}

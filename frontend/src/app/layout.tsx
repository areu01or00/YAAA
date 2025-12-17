import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PaperMap",
  description: "Semantic paper explorer for arxiv",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EcoText",
  description: "Analyze sustainability claims and learn how to identify greenwashing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Makeup Transfer — makeup as a portable RGBA layer",
  description:
    "Extract makeup from a photo as a translucent RGBA layer in 3D UV space; re-apply it skin-tone-aware to any face.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

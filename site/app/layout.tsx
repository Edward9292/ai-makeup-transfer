import type { Metadata } from "next";
import { Jost, Karla } from "next/font/google";
import "./globals.css";

/* Jost — geometric, cosmetics-packaging voice, for headings and labels.
   Karla — humanist body face, quiet at small sizes.
   Deliberately no monospace: this is a product, not a terminal. */
const display = Jost({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["300", "400", "500", "600"],
});

const body = Karla({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Makeup Transfer — one look, any face",
  description:
    "Lift a makeup look off a photo, save it as a transparent layer, and wear it on any face — matched to that person's own skin.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>{children}</body>
    </html>
  );
}

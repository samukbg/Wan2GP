export interface SocialPostProps {
  headline?: string;
  subheadline?: string;
  cta?: string;
  backgroundImageUrl: string;
  brandName?: string;
  logoUrl?: string;
  layout?: "split-bottom" | "top-title" | "center-punch" | "lower-third" | "slide-01" | "slide-02" | "slide-03" | "slide-04" | "slide-05" | "slide-06" | "slide-07" | "slide-08" | "slide-09" | string;
  accentColor?: string;
  secondaryColor?: string;
  overlayOpacity?: number;
  fontFamily?: string;
  badges?: { text: string }[];
  highlights?: { stat: string; label: string }[];
  tagline?: string;
  language?: string;
}


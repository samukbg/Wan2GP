import React from 'react';
import ReactDOM from 'react-dom/client';

// Import templates directly from the remotion codebase!
import { SocialPost } from './templates/SocialPost';
import { AIImageOnGrid } from './templates/AIImageOnGrid';
import { AnnotatedScreenshot } from './templates/AnnotatedScreenshot';
import { HookTitle } from './templates/HookTitle';
import { ImageCard } from './templates/ImageCard';
import { NotificationToast } from './templates/NotificationToast';
import { PortraitBurst } from './templates/PortraitBurst';
import { SplitReveal } from './templates/SplitReveal';
import { ToolLogoBurst } from './templates/ToolLogoBurst';

const templates: Record<string, React.FC<any>> = {
  SocialPost,
  AIImageOnGrid,
  AnnotatedScreenshot,
  HookTitle,
  ImageCard,
  NotificationToast,
  PortraitBurst,
  SplitReveal,
  ToolLogoBurst,
};

// Expose render function for Playwright
window.renderTemplate = (templateName: string, props: any) => {
  const Template = templates[templateName];
  if (!Template) {
    document.getElementById('root')!.innerHTML = `<h1 style="color:white">Template not found: ${templateName}</h1>`;
    return;
  }
  
  const root = ReactDOM.createRoot(document.getElementById('root')!);
  root.render(<Template {...props} />);
};

// Indicate to Playwright that the page is ready to accept commands
window.reactReady = true;

import React from 'react';

export const AbsoluteFill = ({ children, style, ...props }) => (
  <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, ...style }} {...props}>
    {children}
  </div>
);

export const useVideoConfig = () => ({
  width: 1080,
  height: 1350,
  fps: 30,
  durationInFrames: 30,
});

export const useCurrentFrame = () => 30; // Final frame so animations are at their end state

export const interpolate = (frame, inputRange, outputRange, options) => {
  // Simplistic mock for final frame: just return the last output range value
  return outputRange[outputRange.length - 1];
};

export const spring = ({ frame, fps, config }) => {
  // Always return 1 (finished state) for still images
  return 1;
};

export const Sequence = ({ children, from, durationInFrames, style, ...props }) => {
  // Just render the children for a still image
  return <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, ...style }} {...props}>{children}</div>;
};

export const Easing = {
  bezier: () => (t) => t,
  in: () => (t) => t,
  out: () => (t) => t,
  inOut: () => (t) => t,
  linear: (t) => t,
  ease: (t) => t,
  sin: (t) => t,
  quad: (t) => t,
  cubic: (t) => t,
  poly: () => (t) => t,
  circle: (t) => t,
  exp: (t) => t,
  elastic: () => (t) => t,
  back: () => (t) => t,
  bounce: (t) => t,
  step0: (t) => t,
  step1: (t) => t,
};

export const Img = (props) => (
  <img {...props} onError={(e) => { e.currentTarget.style.display = 'none'; }} />
);

export const staticFile = (path) => path;

// Mock continueRender/delayRender so SafeImg doesn't crash
export const continueRender = () => {};
export const delayRender = () => 0;

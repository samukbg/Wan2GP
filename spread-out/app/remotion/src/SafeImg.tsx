import React, { useState } from 'react';
import { continueRender, delayRender } from 'remotion';

export const SafeImg: React.FC<React.ImgHTMLAttributes<HTMLImageElement>> = (props) => {
  const [handle] = useState(() => delayRender("Loading SafeImg: " + props.src));

  return (
    <img
      {...props}
      onLoad={(e) => {
        continueRender(handle);
        if (props.onLoad) props.onLoad(e);
      }}
      onError={(e) => {
        e.currentTarget.style.display = 'none';
        continueRender(handle);
        if (props.onError) props.onError(e);
      }}
    />
  );
};

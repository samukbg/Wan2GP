import { AbsoluteFill, Composition, Still, staticFile } from "remotion";
import { EditedVideo, type EditedVideoProps } from "./EditedVideo";
import { StyleShowcase } from "./StyleShowcase";
import { DarkGridBg, LightGridBg } from "./templates/Backgrounds";
import { SocialPost, type SocialPostProps } from "./templates/SocialPost";
import props from "./props.json";

const DarkGridFrame = () => <AbsoluteFill><DarkGridBg /></AbsoluteFill>;
const LightGridFrame = () => <AbsoluteFill><LightGridBg /></AbsoluteFill>;

const typed = props as unknown as EditedVideoProps & {
  fps: number;
  width: number;
  height: number;
  durationInFrames: number;
};

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="EditedVideo"
        component={EditedVideo}
        durationInFrames={typed.durationInFrames}
        fps={typed.fps}
        width={typed.width}
        height={typed.height}
        defaultProps={typed}
      />
      <Composition
        id="StyleShowcase"
        component={StyleShowcase}
        durationInFrames={30 * 125}
        fps={30}
        width={1080}
        height={1920}
      />
      {/* 16:9 variant for verifying landscape templates side-by-side. */}
      <Composition
        id="StyleShowcaseLandscape"
        component={StyleShowcase}
        durationInFrames={30 * 125}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="DarkGridFrame"
        component={DarkGridFrame}
        durationInFrames={1}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="LightGridFrame"
        component={LightGridFrame}
        durationInFrames={1}
        fps={30}
        width={1080}
        height={1920}
      />
      {/* ─── Still compositions (social post images) ─────────────────────── */}
      <Still
        id="SocialPost"
        component={SocialPost}
        width={1080}
        height={1920}
        defaultProps={{
          headline: "Your Headline Here",
          subheadline: "Supporting copy that drives action.",
          cta: "Learn More",
          backgroundImageUrl: "https://images.pexels.com/photos/3183150/pexels-photo-3183150.jpeg",
          brandName: "SPREAD OUT",
          layout: "split-bottom",
          accentColor: "#CFFF05",
          badges: [{ text: "NEW" }],
          highlights: [],
        } satisfies SocialPostProps}
      />
      <Still
        id="SocialPostSquare"
        component={SocialPost}
        width={1080}
        height={1080}
        defaultProps={{
          headline: "Your Headline Here",
          backgroundImageUrl: "https://images.pexels.com/photos/3183150/pexels-photo-3183150.jpeg",
          layout: "center-punch",
          accentColor: "#CFFF05",
        } satisfies SocialPostProps}
      />
    </>
  );
};

// Re-exported so Remotion picks it up
export { staticFile };

import {
  AbsoluteFill,
  Img,
  interpolate,
  spring,
  Still,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// ─── Types ─────────────────────────────────────────────────────────────────────

export type SocialPostBadge = {
  text: string;
  color?: string;
  bgColor?: string;
};

export type SocialPostHighlight = {
  stat: string;
  label: string;
  icon?: string; // SVG path or emoji
};

export type SocialPostProps = {
  /** The main headline / hook text */
  headline: string;
  /** Supporting subheadline or body copy */
  subheadline?: string;
  /** Call-to-action text */
  cta?: string;
  /** URL of the Pexels background image */
  backgroundImageUrl: string;
  /** Brand / product name */
  brandName?: string;
  /** Layout variant */
  layout?: "split-bottom" | "top-title" | "center-punch" | "lower-third";
  /** Theme accent color (hex). Defaults to Neo Lime. */
  accentColor?: string;
  /** Secondary accent color */
  secondaryColor?: string;
  /** Background overlay opacity 0..1 */
  overlayOpacity?: number;
  /** Small badges/tags */
  badges?: SocialPostBadge[];
  /** Up to 3 highlight stats */
  highlights?: SocialPostHighlight[];
  /** Tagline below CTA */
  tagline?: string;
  /** Font style */
  fontFamily?: string;
};

// ─── Helpers ───────────────────────────────────────────────────────────────────

const RAISIN_BLACK = "#0F121A";
const NEO_LIME = "#CFFF05";
const WHITE = "#FFFFFF";

const escapeText = (s: string) => s ?? "";

// ─── SVG Decorative Elements ───────────────────────────────────────────────────

const DiagonalLines: React.FC<{ color: string; opacity?: number }> = ({
  color,
  opacity = 0.08,
}) => (
  <svg
    style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    viewBox="0 0 1080 1920"
    preserveAspectRatio="xMidYMid slice"
  >
    {Array.from({ length: 20 }).map((_, i) => (
      <line
        key={i}
        x1={i * 120 - 600}
        y1={0}
        x2={i * 120 + 400}
        y2={1920}
        stroke={color}
        strokeWidth={1}
        opacity={opacity}
      />
    ))}
  </svg>
);

const CornerAccent: React.FC<{ color: string; position: "tl" | "tr" | "bl" | "br" }> = ({
  color,
  position,
}) => {
  const transforms: Record<string, string> = {
    tl: "translate(0, 0)",
    tr: "translate(1080, 0) scale(-1, 1)",
    bl: "translate(0, 1920) scale(1, -1)",
    br: "translate(1080, 1920) scale(-1, -1)",
  };
  return (
    <svg
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      viewBox="0 0 1080 1920"
      preserveAspectRatio="xMidYMid slice"
    >
      <g transform={transforms[position]}>
        <path d={`M0,0 L200,0 L200,8 L8,8 L8,200 L0,200 Z`} fill={color} opacity={0.9} />
        <path d={`M0,0 L100,0 L100,4 L4,4 L4,100 L0,100 Z`} fill={color} opacity={0.5} transform="translate(20,20)" />
      </g>
    </svg>
  );
};

const PulseRing: React.FC<{ x: number; y: number; color: string; size?: number }> = ({
  x,
  y,
  color,
  size = 60,
}) => (
  <svg
    style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    viewBox="0 0 1080 1920"
    preserveAspectRatio="xMidYMid slice"
  >
    <circle cx={x} cy={y} r={size} fill="none" stroke={color} strokeWidth={2} opacity={0.3} />
    <circle cx={x} cy={y} r={size * 0.6} fill="none" stroke={color} strokeWidth={1.5} opacity={0.5} />
    <circle cx={x} cy={y} r={size * 0.25} fill={color} opacity={0.8} />
  </svg>
);

// ─── Layout Variants ───────────────────────────────────────────────────────────

const LayoutSplitBottom: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props, frame, fps }) => {
  const accent = props.accentColor || NEO_LIME;
  const secondary = props.secondaryColor || "#00F0FF";
  const font = props.fontFamily || "'Montserrat', sans-serif";

  return (
    <>
      {/* Gradient overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `linear-gradient(
            180deg,
            rgba(15,18,26,0.2) 0%,
            rgba(15,18,26,0.1) 30%,
            rgba(15,18,26,0.7) 55%,
            rgba(15,18,26,0.97) 80%,
            ${RAISIN_BLACK} 100%
          )`,
        }}
      />

      {/* SVG decorative layer */}
      <DiagonalLines color={accent} opacity={0.04} />
      <CornerAccent color={accent} position="tl" />
      <CornerAccent color={accent} position="br" />

      {/* Brand badge top */}
      {props.brandName && (
        <div
          style={{
            position: "absolute",
            top: 80,
            left: 0,
            right: 0,
            display: "flex",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              background: `${accent}22`,
              border: `1.5px solid ${accent}66`,
              borderRadius: 100,
              padding: "10px 28px",
              color: accent,
              fontFamily: font,
              fontWeight: 700,
              fontSize: 28,
              letterSpacing: 4,
              textTransform: "uppercase",
            }}
          >
            {escapeText(props.brandName)}
          </div>
        </div>
      )}

      {/* Bottom content panel */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "60px 64px 100px",
        }}
      >
        {/* Badges */}
        {props.badges && props.badges.length > 0 && (
          <div style={{ display: "flex", gap: 12, marginBottom: 28, flexWrap: "wrap" }}>
            {props.badges.map((b, i) => (
              <div
                key={i}
                style={{
                  background: b.bgColor || `${accent}25`,
                  border: `1px solid ${b.color || accent}`,
                  borderRadius: 8,
                  padding: "8px 20px",
                  color: b.color || accent,
                  fontFamily: font,
                  fontWeight: 700,
                  fontSize: 22,
                  letterSpacing: 2,
                  textTransform: "uppercase",
                }}
              >
                {b.text}
              </div>
            ))}
          </div>
        )}

        {/* Headline */}
        <div
          style={{
            fontFamily: font,
            fontWeight: 900,
            fontSize: 76,
            lineHeight: 1.05,
            color: WHITE,
            textTransform: "uppercase",
            letterSpacing: -1,
            marginBottom: 24,
            WebkitTextStroke: "1px rgba(0,0,0,0.3)",
          }}
        >
          {escapeText(props.headline)}
        </div>

        {/* Accent rule */}
        <div
          style={{
            width: 80,
            height: 4,
            background: accent,
            borderRadius: 2,
            marginBottom: 24,
          }}
        />

        {/* Subheadline */}
        {props.subheadline && (
          <div
            style={{
              fontFamily: font,
              fontWeight: 500,
              fontSize: 34,
              lineHeight: 1.4,
              color: "rgba(255,255,255,0.8)",
              marginBottom: 40,
            }}
          >
            {escapeText(props.subheadline)}
          </div>
        )}

        {/* Highlights row */}
        {props.highlights && props.highlights.length > 0 && (
          <div style={{ display: "flex", gap: 40, marginBottom: 40 }}>
            {props.highlights.slice(0, 3).map((h, i) => (
              <div key={i} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div
                  style={{
                    fontFamily: font,
                    fontWeight: 900,
                    fontSize: 52,
                    color: accent,
                    lineHeight: 1,
                  }}
                >
                  {h.stat}
                </div>
                <div
                  style={{
                    fontFamily: font,
                    fontWeight: 500,
                    fontSize: 24,
                    color: "rgba(255,255,255,0.6)",
                    textTransform: "uppercase",
                    letterSpacing: 2,
                  }}
                >
                  {h.label}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* CTA */}
        {props.cta && (
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 16,
              background: accent,
              borderRadius: 16,
              padding: "22px 48px",
              color: RAISIN_BLACK,
              fontFamily: font,
              fontWeight: 900,
              fontSize: 32,
              letterSpacing: 1,
              textTransform: "uppercase",
            }}
          >
            {escapeText(props.cta)}
            <svg width={24} height={24} viewBox="0 0 24 24" fill="none">
              <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke={RAISIN_BLACK} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        )}

        {/* Tagline */}
        {props.tagline && (
          <div
            style={{
              fontFamily: font,
              fontWeight: 400,
              fontSize: 22,
              color: "rgba(255,255,255,0.4)",
              marginTop: 24,
              letterSpacing: 1,
            }}
          >
            {escapeText(props.tagline)}
          </div>
        )}
      </div>
    </>
  );
};

const LayoutCenterPunch: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props, frame, fps }) => {
  const accent = props.accentColor || NEO_LIME;
  const font = props.fontFamily || "'Montserrat', sans-serif";

  return (
    <>
      {/* Dark vignette */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(ellipse at center, rgba(15,18,26,0.4) 0%, rgba(15,18,26,0.9) 80%, ${RAISIN_BLACK} 100%)`,
        }}
      />

      <DiagonalLines color={accent} opacity={0.05} />
      <PulseRing x={540} y={960} color={accent} size={360} />
      <CornerAccent color={accent} position="tl" />
      <CornerAccent color={accent} position="tr" />

      {/* Center content */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "80px 80px",
          textAlign: "center",
        }}
      >
        {/* Accent top line */}
        <div style={{ width: 60, height: 4, background: accent, borderRadius: 2, marginBottom: 40 }} />

        {props.brandName && (
          <div
            style={{
              color: accent,
              fontFamily: font,
              fontWeight: 700,
              fontSize: 26,
              letterSpacing: 6,
              textTransform: "uppercase",
              marginBottom: 40,
            }}
          >
            {escapeText(props.brandName)}
          </div>
        )}

        <div
          style={{
            fontFamily: font,
            fontWeight: 900,
            fontSize: 90,
            lineHeight: 1.0,
            color: WHITE,
            textTransform: "uppercase",
            letterSpacing: -2,
            marginBottom: 36,
            textShadow: `0 0 60px ${accent}40`,
          }}
        >
          {escapeText(props.headline)}
        </div>

        <div style={{ width: 80, height: 4, background: accent, borderRadius: 2, margin: "0 auto 36px" }} />

        {props.subheadline && (
          <div
            style={{
              fontFamily: font,
              fontWeight: 500,
              fontSize: 36,
              lineHeight: 1.4,
              color: "rgba(255,255,255,0.75)",
              marginBottom: 60,
              maxWidth: 800,
            }}
          >
            {escapeText(props.subheadline)}
          </div>
        )}

        {props.highlights && props.highlights.length > 0 && (
          <div style={{ display: "flex", gap: 60, marginBottom: 60, justifyContent: "center" }}>
            {props.highlights.slice(0, 3).map((h, i) => (
              <div key={i} style={{ textAlign: "center" }}>
                <div style={{ fontFamily: font, fontWeight: 900, fontSize: 64, color: accent, lineHeight: 1 }}>
                  {h.stat}
                </div>
                <div
                  style={{
                    fontFamily: font,
                    fontWeight: 500,
                    fontSize: 22,
                    color: "rgba(255,255,255,0.55)",
                    letterSpacing: 2,
                    textTransform: "uppercase",
                    marginTop: 8,
                  }}
                >
                  {h.label}
                </div>
              </div>
            ))}
          </div>
        )}

        {props.cta && (
          <div
            style={{
              background: accent,
              borderRadius: 100,
              padding: "24px 56px",
              color: RAISIN_BLACK,
              fontFamily: font,
              fontWeight: 900,
              fontSize: 30,
              letterSpacing: 2,
              textTransform: "uppercase",
              display: "inline-flex",
              alignItems: "center",
              gap: 16,
            }}
          >
            {escapeText(props.cta)}
          </div>
        )}
      </div>
    </>
  );
};

const LayoutTopTitle: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props, frame, fps }) => {
  const accent = props.accentColor || NEO_LIME;
  const font = props.fontFamily || "'Montserrat', sans-serif";

  return (
    <>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `linear-gradient(
            180deg,
            rgba(15,18,26,0.92) 0%,
            rgba(15,18,26,0.6) 45%,
            rgba(15,18,26,0.15) 65%,
            rgba(15,18,26,0.85) 100%
          )`,
        }}
      />
      <DiagonalLines color={accent} opacity={0.04} />
      <CornerAccent color={accent} position="tl" />
      <CornerAccent color={accent} position="br" />

      {/* Top content block */}
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, padding: "80px 64px 0" }}>
        {props.brandName && (
          <div
            style={{
              color: accent,
              fontFamily: font,
              fontWeight: 700,
              fontSize: 26,
              letterSpacing: 5,
              textTransform: "uppercase",
              marginBottom: 32,
            }}
          >
            {escapeText(props.brandName)}
          </div>
        )}

        {/* Thin rule */}
        <div style={{ width: 56, height: 4, background: accent, borderRadius: 2, marginBottom: 32 }} />

        <div
          style={{
            fontFamily: font,
            fontWeight: 900,
            fontSize: 82,
            lineHeight: 1.02,
            color: WHITE,
            textTransform: "uppercase",
            letterSpacing: -1,
            marginBottom: 28,
          }}
        >
          {escapeText(props.headline)}
        </div>

        {props.subheadline && (
          <div
            style={{
              fontFamily: font,
              fontWeight: 500,
              fontSize: 32,
              lineHeight: 1.5,
              color: "rgba(255,255,255,0.75)",
            }}
          >
            {escapeText(props.subheadline)}
          </div>
        )}
      </div>

      {/* Bottom CTA */}
      {(props.cta || props.tagline) && (
        <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "0 64px 100px" }}>
          {props.highlights && props.highlights.length > 0 && (
            <div style={{ display: "flex", gap: 40, marginBottom: 40 }}>
              {props.highlights.slice(0, 3).map((h, i) => (
                <div key={i}>
                  <div style={{ fontFamily: font, fontWeight: 900, fontSize: 54, color: accent, lineHeight: 1 }}>
                    {h.stat}
                  </div>
                  <div
                    style={{
                      fontFamily: font,
                      fontSize: 22,
                      color: "rgba(255,255,255,0.55)",
                      letterSpacing: 2,
                      textTransform: "uppercase",
                    }}
                  >
                    {h.label}
                  </div>
                </div>
              ))}
            </div>
          )}
          {props.cta && (
            <div
              style={{
                background: accent,
                display: "inline-flex",
                alignItems: "center",
                gap: 14,
                borderRadius: 14,
                padding: "20px 44px",
                color: RAISIN_BLACK,
                fontFamily: font,
                fontWeight: 900,
                fontSize: 30,
                textTransform: "uppercase",
              }}
            >
              {escapeText(props.cta)}
              <svg width={22} height={22} viewBox="0 0 24 24" fill="none">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke={RAISIN_BLACK} strokeWidth={2.5} strokeLinecap="round" />
              </svg>
            </div>
          )}
          {props.tagline && (
            <div
              style={{
                fontFamily: font,
                fontSize: 22,
                color: "rgba(255,255,255,0.4)",
                marginTop: 20,
              }}
            >
              {escapeText(props.tagline)}
            </div>
          )}
        </div>
      )}
    </>
  );
};

// ─── Main Component ────────────────────────────────────────────────────────────

export const SocialPost: React.FC<SocialPostProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const layout = props.layout || "split-bottom";

  const renderLayout = () => {
    switch (layout) {
      case "center-punch":
        return <LayoutCenterPunch props={props} frame={frame} fps={fps} />;
      case "top-title":
        return <LayoutTopTitle props={props} frame={frame} fps={fps} />;
      case "split-bottom":
      case "lower-third":
      default:
        return <LayoutSplitBottom props={props} frame={frame} fps={fps} />;
    }
  };

  return (
    <AbsoluteFill style={{ background: RAISIN_BLACK }}>
      {/* Background image (Pexels) */}
      <Img
        src={props.backgroundImageUrl}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          objectPosition: "center",
        }}
      />

      {/* Layout-specific overlays */}
      {renderLayout()}
    </AbsoluteFill>
  );
};

import {
  AbsoluteFill,
  Img,
  interpolate,
  spring,
  Still,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import {
  LayoutSlide01,
  LayoutSlide02,
  LayoutSlide03,
  LayoutSlide04,
  LayoutSlide05,
  LayoutSlide06,
  LayoutSlide07,
  LayoutSlide08,
  LayoutSlide09
} from "./CarouselSlides";
import { SocialPostProps } from "./SocialPostTypes";

// ─── Types ─────────────────────────────────────────────────────────────────────

export type SocialPostBadge = {
  text: string;
  color?: string;
  bgColor?: string;
};

export type SocialPostHighlight = {
  stat: string;
  label: string;
  icon?: string;
};


// ─── Helpers ───────────────────────────────────────────────────────────────────

const RAISIN_BLACK = "#0F121A";
const NEO_LIME = "#CFFF05";
const WHITE = "#FFFFFF";

const escapeText = (s: string) => s ?? "";

const hexToRgb = (hex: string) => {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
};

// Luminance check — returns true if color is "light"
const isLight = (hex: string) => {
  const { r, g, b } = hexToRgb(hex || "#000000");
  return (r * 299 + g * 587 + b * 114) / 1000 > 128;
};

// ─── Layout 1: SPLIT-BOTTOM ────────────────────────────────────────────────────
// Archetype: Bold magazine editorial. Content anchored to the bottom third.
// Dark gradient rises from the bottom. Image breathes in the top two-thirds.
// Hallmark: Full-width headline, accent left border rule, badge pills top-right.

const LayoutSplitBottom: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props }) => {
  const accent = props.accentColor || NEO_LIME;
  const font = props.fontFamily || "'Montserrat', sans-serif";
  const onAccent = isLight(accent) ? RAISIN_BLACK : WHITE;

  return (
    <>
      {/* Editorial bottom gradient — image shows fully at top */}
      <div style={{
        position: "absolute", inset: 0,
        background: `linear-gradient(180deg, transparent 0%, transparent 35%, rgba(0,0,0,0.65) 60%, rgba(0,0,0,0.93) 80%, #000 100%)`,
      }} />

      {/* Accent bar — left edge running the full height of the content panel */}
      <div style={{
        position: "absolute", left: 0, bottom: 0,
        width: 8, height: "52%",
        background: `linear-gradient(180deg, transparent 0%, ${accent} 40%, ${accent} 100%)`,
      }} />

      {/* Badges — top-right floating pills */}
      {props.badges && props.badges.length > 0 && (
        <div style={{ position: "absolute", top: 72, right: 56, display: "flex", flexDirection: "column", gap: 12, alignItems: "flex-end" }}>
          {props.badges.map((b, i) => (
            <div key={i} style={{
              background: b.bgColor || accent,
              borderRadius: 6, padding: "10px 22px",
              color: b.color || onAccent,
              fontFamily: font, fontWeight: 800,
              fontSize: 22, letterSpacing: 3,
              textTransform: "uppercase",
            }}>{b.text}</div>
          ))}
        </div>
      )}

      {/* Content panel */}
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "0 56px 80px 72px" }}>
        {/* Brand label */}
        {props.brandName && (
          <div style={{
            color: accent, fontFamily: font, fontWeight: 700,
            fontSize: 24, letterSpacing: 5, textTransform: "uppercase",
            marginBottom: 20, opacity: 0.9,
          }}>{escapeText(props.brandName)}</div>
        )}

        {/* Headline — huge, left-aligned, white */}
        <div style={{
          fontFamily: font, fontWeight: 900, fontSize: 88,
          lineHeight: 0.97, color: WHITE,
          textTransform: "uppercase", letterSpacing: -2,
          marginBottom: 28,
        }}>{escapeText(props.headline)}</div>

        {/* Subheadline */}
        {props.subheadline && (
          <div style={{
            fontFamily: font, fontWeight: 400, fontSize: 32,
            lineHeight: 1.45, color: "rgba(255,255,255,0.72)",
            marginBottom: 36, maxWidth: 880,
          }}>{escapeText(props.subheadline)}</div>
        )}

        {/* Stats row */}
        {Array.isArray(props.highlights) && props.highlights.length > 0 && (
          <div style={{ display: "flex", gap: 48, marginBottom: 36 }}>
            {props.highlights.slice(0, 3).map((h, i) => (
              <div key={i}>
                <div style={{ fontFamily: font, fontWeight: 900, fontSize: 56, color: accent, lineHeight: 1 }}>{h.stat}</div>
                <div style={{ fontFamily: font, fontSize: 22, color: "rgba(255,255,255,0.5)", letterSpacing: 2, textTransform: "uppercase", marginTop: 4 }}>{h.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* CTA */}
        {props.cta && (
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 14,
            color: accent, fontFamily: font,
            fontWeight: 900, fontSize: 30,
            letterSpacing: 1, textTransform: "uppercase",
          }}>
            {escapeText(props.cta)}
            <svg width={22} height={22} viewBox="0 0 24 24" fill="none">
              <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke={accent} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        )}

        {/* Tagline */}
        {props.tagline && (
          <div style={{ fontFamily: font, fontSize: 22, color: "rgba(255,255,255,0.38)", marginTop: 20, letterSpacing: 1 }}>
            {escapeText(props.tagline)}
          </div>
        )}
      </div>
    </>
  );
};

// ─── Layout 2: CENTER-PUNCH ────────────────────────────────────────────────────
// Archetype: Full-bleed cinematic impact. Heavy overlay, centered composition.
// Hallmark: Massive headline with glow, NO corner brackets, radial vignette,
// optional floating stat chips in a horizontal row below the headline.

const LayoutCenterPunch: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props }) => {
  const accent = props.accentColor || NEO_LIME;
  const secondary = props.secondaryColor || "#FF6B35";
  const font = props.fontFamily || "'Montserrat', sans-serif";
  const onAccent = isLight(accent) ? RAISIN_BLACK : WHITE;

  return (
    <>
      {/* Radial cinematic vignette — no linear gradient */}
      <div style={{
        position: "absolute", inset: 0,
        background: `radial-gradient(ellipse 90% 85% at 50% 50%, rgba(0,0,0,0.45) 0%, rgba(0,0,0,0.82) 70%, #000 100%)`,
      }} />

      {/* Horizontal accent bar top */}
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 6, background: `linear-gradient(90deg, ${accent}, ${secondary}, ${accent})` }} />

      {/* Brand — centered at top */}
      {props.brandName && (
        <div style={{
          position: "absolute", top: 72, left: 0, right: 0,
          display: "flex", justifyContent: "center",
        }}>
          <div style={{
            background: "rgba(0,0,0,0.5)", border: `1.5px solid ${accent}55`,
            borderRadius: 100, padding: "10px 32px",
            color: accent, fontFamily: font,
            fontWeight: 700, fontSize: 26,
            letterSpacing: 5, textTransform: "uppercase",
          }}>{escapeText(props.brandName)}</div>
        </div>
      )}

      {/* Center-stage content */}
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "120px 80px", textAlign: "center",
      }}>
        {/* Badges */}
        {Array.isArray(props.badges) && props.badges.length > 0 && (
          <div style={{ display: "flex", gap: 12, marginBottom: 40, justifyContent: "center" }}>
            {props.badges.map((b, i) => (
              <div key={i} style={{
                background: b.bgColor || `${accent}25`,
                border: `1px solid ${b.color || accent}`,
                borderRadius: 6, padding: "8px 20px",
                color: b.color || accent,
                fontFamily: font, fontWeight: 700,
                fontSize: 22, letterSpacing: 2, textTransform: "uppercase",
              }}>{b.text}</div>
            ))}
          </div>
        )}

        {/* Massive headline with text glow */}
        <div style={{
          fontFamily: font, fontWeight: 900, fontSize: 100,
          lineHeight: 0.95, color: WHITE,
          textTransform: "uppercase", letterSpacing: -3,
          marginBottom: 40,
          textShadow: `0 0 80px ${accent}60, 0 0 160px ${accent}30`,
        }}>{escapeText(props.headline)}</div>

        {/* Accent separator dot line */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 40 }}>
          <div style={{ height: 2, width: 80, background: `${accent}80` }} />
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: accent }} />
          <div style={{ height: 2, width: 80, background: `${accent}80` }} />
        </div>

        {/* Subheadline */}
        {props.subheadline && (
          <div style={{
            fontFamily: font, fontWeight: 400, fontSize: 36,
            lineHeight: 1.4, color: "rgba(255,255,255,0.75)",
            marginBottom: 56, maxWidth: 820,
          }}>{escapeText(props.subheadline)}</div>
        )}

        {/* Stat chips — horizontal floating boxes */}
        {Array.isArray(props.highlights) && props.highlights.length > 0 && (
          <div style={{ display: "flex", gap: 24, marginBottom: 56, justifyContent: "center", flexWrap: "wrap" }}>
            {props.highlights.slice(0, 3).map((h, i) => (
              <div key={i} style={{
                background: "rgba(255,255,255,0.08)",
                border: `1px solid rgba(255,255,255,0.15)`,
                borderRadius: 16, padding: "20px 32px", textAlign: "center",
                backdropFilter: "blur(8px)",
              }}>
                <div style={{ fontFamily: font, fontWeight: 900, fontSize: 64, color: accent, lineHeight: 1 }}>{h.stat}</div>
                <div style={{ fontFamily: font, fontSize: 22, color: "rgba(255,255,255,0.5)", letterSpacing: 2, textTransform: "uppercase", marginTop: 6 }}>{h.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* CTA — floating text */}
        {props.cta && (
          <div style={{
            color: accent, fontFamily: font,
            fontWeight: 900, fontSize: 32,
            letterSpacing: 2, textTransform: "uppercase",
          }}>{escapeText(props.cta)}</div>
        )}
      </div>
    </>
  );
};

// ─── Layout 3: TOP-TITLE ───────────────────────────────────────────────────────
// Archetype: Editorial magazine with image as hero. Content floats at the top.
// Hallmark: Semi-transparent frosted card anchored top-left, image fills bottom,
// CTA is a ghost/outline button at the bottom. Clean two-zone design.

const LayoutTopTitle: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props }) => {
  const accent = props.accentColor || NEO_LIME;
  const secondary = props.secondaryColor || "#FFFFFF";
  const font = props.fontFamily || "'Montserrat', sans-serif";
  const onAccent = isLight(accent) ? RAISIN_BLACK : WHITE;

  return (
    <>
      {/* Light top vignette so image shows in the bottom half */}
      <div style={{
        position: "absolute", inset: 0,
        background: `linear-gradient(180deg, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.70) 38%, rgba(0,0,0,0.1) 58%, rgba(0,0,0,0.7) 88%, rgba(0,0,0,0.92) 100%)`,
      }} />

      {/* Frosted card panel — top area */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0,
        padding: "80px 64px 56px",
        background: "rgba(0,0,0,0.15)",
        borderBottom: `3px solid ${accent}33`,
      }}>
        {/* Accent left bracket */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: 28 }}>
          <div style={{ width: 6, height: "100%", minHeight: 160, background: accent, borderRadius: 3, flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            {/* Brand */}
            {props.brandName && (
              <div style={{
                color: accent, fontFamily: font, fontWeight: 700,
                fontSize: 24, letterSpacing: 5, textTransform: "uppercase",
                marginBottom: 20,
              }}>{escapeText(props.brandName)}</div>
            )}

            {/* Badges inline */}
            {Array.isArray(props.badges) && props.badges.length > 0 && (
              <div style={{ display: "flex", gap: 10, marginBottom: 20, flexWrap: "wrap" }}>
                {props.badges.map((b, i) => (
                  <div key={i} style={{
                    background: b.bgColor || accent,
                    borderRadius: 4, padding: "6px 16px",
                    color: b.color || onAccent,
                    fontFamily: font, fontWeight: 800,
                    fontSize: 20, letterSpacing: 2, textTransform: "uppercase",
                  }}>{b.text}</div>
                ))}
              </div>
            )}

            {/* Headline */}
            <div style={{
              fontFamily: font, fontWeight: 900, fontSize: 78,
              lineHeight: 1.0, color: WHITE,
              textTransform: "uppercase", letterSpacing: -2,
              marginBottom: 24,
            }}>{escapeText(props.headline)}</div>

            {/* Subheadline */}
            {props.subheadline && (
              <div style={{
                fontFamily: font, fontWeight: 400, fontSize: 30,
                lineHeight: 1.5, color: "rgba(255,255,255,0.75)",
              }}>{escapeText(props.subheadline)}</div>
            )}
          </div>
        </div>
      </div>

      {/* Stats — mid-screen floating row */}
      {Array.isArray(props.highlights) && props.highlights.length > 0 && (
        <div style={{
          position: "absolute",
          bottom: props.cta ? 220 : 140,
          left: 0, right: 0,
          display: "flex", justifyContent: "center", gap: 0,
          padding: "0 40px",
        }}>
          {props.highlights.slice(0, 3).map((h, i) => (
            <div key={i} style={{
              flex: 1, textAlign: "center",
              padding: "24px 16px",
              borderRight: i < 2 ? `1px solid rgba(255,255,255,0.15)` : "none",
              background: "rgba(0,0,0,0.4)",
              backdropFilter: "blur(12px)",
            }}>
              <div style={{ fontFamily: font, fontWeight: 900, fontSize: 60, color: accent, lineHeight: 1 }}>{h.stat}</div>
              <div style={{ fontFamily: font, fontSize: 22, color: "rgba(255,255,255,0.55)", letterSpacing: 2, textTransform: "uppercase", marginTop: 8 }}>{h.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Bottom zone — CTA + tagline */}
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "0 64px 80px" }}>
        {props.cta && (
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 16,
            color: accent, fontFamily: font,
            fontWeight: 900, fontSize: 30,
            letterSpacing: 1, textTransform: "uppercase",
          }}>
            {escapeText(props.cta)}
            <svg width={22} height={22} viewBox="0 0 24 24" fill="none">
              <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke={accent} strokeWidth={2.5} strokeLinecap="round" />
            </svg>
          </div>
        )}
        {props.tagline && (
          <div style={{ fontFamily: font, fontSize: 22, color: "rgba(255,255,255,0.38)", marginTop: 20 }}>{escapeText(props.tagline)}</div>
        )}
      </div>
    </>
  );
};

// ─── Layout 4: LOWER-THIRD ─────────────────────────────────────────────────────
// Archetype: Broadcast chyron / data dashboard. Image dominates the full frame.
// A glassy ticker/chyron bar at the bottom carries brand + headline.
// Hallmark: Minimal text, huge white space, bold typography in a tight band.
// Stats appear as a horizontal strip above the chyron.

const LayoutLowerThird: React.FC<{
  props: SocialPostProps;
  frame: number;
  fps: number;
}> = ({ props }) => {
  const accent = props.accentColor || NEO_LIME;
  const secondary = props.secondaryColor || "#FFFFFF";
  const font = props.fontFamily || "'Montserrat', sans-serif";
  const onAccent = isLight(accent) ? RAISIN_BLACK : WHITE;

  return (
    <>
      {/* Very subtle top vignette — image is the hero */}
      <div style={{
        position: "absolute", inset: 0,
        background: `linear-gradient(180deg, rgba(0,0,0,0.3) 0%, transparent 30%, transparent 55%, rgba(0,0,0,0.75) 80%, rgba(0,0,0,0.97) 100%)`,
      }} />

      {/* Badges top-left */}
      {props.badges && props.badges.length > 0 && (
        <div style={{ position: "absolute", top: 72, left: 56, display: "flex", gap: 12 }}>
          {props.badges.map((b, i) => (
            <div key={i} style={{
              background: b.bgColor || accent,
              borderRadius: 6, padding: "10px 22px",
              color: b.color || onAccent,
              fontFamily: font, fontWeight: 800,
              fontSize: 24, letterSpacing: 3, textTransform: "uppercase",
            }}>{b.text}</div>
          ))}
        </div>
      )}

      {/* Stats strip — floating above the chyron */}
      {props.highlights && props.highlights.length > 0 && (
        <div style={{
          position: "absolute",
          bottom: 300,
          left: 0, right: 0,
          display: "flex",
        }}>
          {props.highlights.slice(0, 3).map((h, i) => (
            <div key={i} style={{
              flex: 1, textAlign: "center",
              padding: "28px 24px",
              background: i % 2 === 0 ? `${accent}E6` : "rgba(0,0,0,0.75)",
              backdropFilter: "blur(10px)",
            }}>
              <div style={{
                fontFamily: font, fontWeight: 900, fontSize: 72,
                color: i % 2 === 0 ? onAccent : accent, lineHeight: 1,
              }}>{h.stat}</div>
              <div style={{
                fontFamily: font, fontSize: 24,
                color: i % 2 === 0 ? `${onAccent}BB` : "rgba(255,255,255,0.55)",
                letterSpacing: 2, textTransform: "uppercase", marginTop: 8,
              }}>{h.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Chyron bar — the signature of this layout */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0,
        background: `${accent}`,
        padding: "32px 56px 56px",
        borderTop: `6px solid ${isLight(accent) ? "rgba(0,0,0,0.2)" : "rgba(255,255,255,0.15)"}`,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 24 }}>
          <div style={{ flex: 1 }}>
            {props.brandName && (
              <div style={{
                fontFamily: font, fontWeight: 700, fontSize: 22,
                letterSpacing: 5, textTransform: "uppercase",
                color: `${onAccent}88`, marginBottom: 10,
              }}>{escapeText(props.brandName)}</div>
            )}
            <div style={{
              fontFamily: font, fontWeight: 900, fontSize: 60,
              lineHeight: 1.0, color: onAccent,
              textTransform: "uppercase", letterSpacing: -1,
            }}>{escapeText(props.headline)}</div>
            {props.subheadline && (
              <div style={{
                fontFamily: font, fontWeight: 500, fontSize: 26,
                color: `${onAccent}CC`, marginTop: 12, lineHeight: 1.35,
              }}>{escapeText(props.subheadline)}</div>
            )}
          </div>

          {/* CTA text */}
          {props.cta && (
            <div style={{
              color: accent, fontFamily: font,
              fontWeight: 900, fontSize: 24,
              letterSpacing: 1, textTransform: "uppercase",
              textAlign: "center", minWidth: 200, flexShrink: 0,
            }}>
              {escapeText(props.cta)}
              <div style={{ marginTop: 8 }}>
                <svg width={28} height={28} viewBox="0 0 24 24" fill="none" style={{ margin: "0 auto", display: "block" }}>
                  <path d="M12 5v14M5 12l7 7 7-7" stroke={accent} strokeWidth={2.5} strokeLinecap="round" />
                </svg>
              </div>
            </div>
          )}
        </div>

        {props.tagline && (
          <div style={{ fontFamily: font, fontSize: 20, color: `${onAccent}70`, marginTop: 16 }}>{escapeText(props.tagline)}</div>
        )}
      </div>
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
      case "slide-01": return <LayoutSlide01 props={props} />;
      case "slide-02": return <LayoutSlide02 props={props} />;
      case "slide-03": return <LayoutSlide03 props={props} />;
      case "slide-04": return <LayoutSlide04 props={props} />;
      case "slide-05": return <LayoutSlide05 props={props} />;
      case "slide-06": return <LayoutSlide06 props={props} />;
      case "slide-07": return <LayoutSlide07 props={props} />;
      case "slide-08": return <LayoutSlide08 props={props} />;
      case "slide-09": return <LayoutSlide09 props={props} />;
      case "center-punch":
        return <LayoutCenterPunch props={props} frame={frame} fps={fps} />;
      case "top-title":
        return <LayoutTopTitle props={props} frame={frame} fps={fps} />;
      case "lower-third":
        return <LayoutLowerThird props={props} frame={frame} fps={fps} />;
      case "split-bottom":
      default:
        return <LayoutSplitBottom props={props} frame={frame} fps={fps} />;
    }
  };

  return (
    <AbsoluteFill style={{ background: RAISIN_BLACK }}>
      {/* 4:5 Cropped Background Image Container */}
      <div style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: 1080,
        height: 1350, // Force 4:5 aspect ratio
        overflow: "hidden"
      }}>
        <Img
          src={props.backgroundImageUrl}
          style={{
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            objectFit: "cover", objectPosition: "center",
          }}
        />
      </div>
      {renderLayout()}
    </AbsoluteFill>
  );
};

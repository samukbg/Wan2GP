import React from 'react';
import { SafeImg } from "../SafeImg";
import { SocialPostProps } from './SocialPost';
import { Img } from 'remotion';

// Helpers
const escapeText = (s?: string) => s ?? "";

const css = `
.slide {
  position:relative;
  width:1080px;
  height:1350px;
  overflow:hidden;
  display:block;
}
.ov-bottom {
  position:absolute;inset:0;
  background:linear-gradient(to bottom, rgba(7,17,31,0.10) 0%, rgba(7,17,31,0.28) 32%, rgba(7,17,31,0.75) 58%, rgba(7,17,31,0.94) 75%, rgba(7,17,31,0.98) 100%);
}
.ov-dual {
  position:absolute;inset:0;
  background:linear-gradient(to bottom, rgba(7,17,31,0.88) 0%, rgba(7,17,31,0.42) 28%, rgba(7,17,31,0.22) 50%, rgba(7,17,31,0.60) 72%, rgba(7,17,31,0.96) 100%);
}
.glass {
  background:rgba(7,17,31,0.76);
  backdrop-filter:blur(28px);
  border:1px solid var(--accent);
  box-shadow: 0 0 0 1px rgba(0,255,128,0.06), inset 0 1px 0 rgba(255,255,255,0.04), 0 24px 64px rgba(0,0,0,0.6);
}
.eyebrow {
  font-size:21px;letter-spacing:0.20em;
  text-transform:uppercase;color:var(--accent);
  font-weight:700;margin-bottom:20px;
}
.h1 {
  font-family:Georgia,serif; font-size:98px;line-height:1.03; letter-spacing:-0.025em; color:white;font-weight:700;
}
.h2 {
  font-family:Georgia,serif; font-size:76px;line-height:1.08; letter-spacing:-0.02em; color:white;font-weight:700;
}
.g {
  color:var(--accent);
}
.body {
  font-size:36px;line-height:1.55; color:rgba(255,255,255,0.72);
}
.rule {
  height:1px; background:linear-gradient(to right,var(--accent) 0%,rgba(0,255,128,0.12) 55%,transparent 100%); margin:30px 0;
}
.badge {
  display:inline-block; padding:10px 24px; border:1.5px solid var(--accent); background:rgba(0,255,128,0.1);
  font-size:21px;letter-spacing:0.16em; text-transform:uppercase; color:var(--accent);font-weight:700; border-radius:4px;
}
.logo-top-white { position:absolute;top:64px;left:64px;max-width:180px;height:auto;z-index:10; }
.vignette { position:absolute;inset:0;z-index:2;pointer-events:none; background:radial-gradient(ellipse 88% 88% at 50% 36%, transparent 36%, rgba(7,17,31,0.28) 62%, rgba(7,17,31,0.60) 100%); }
`;

export const SlideBase: React.FC<{props: SocialPostProps, children: React.ReactNode}> = ({props, children}) => {
  const accent = props.accentColor || '#CFFF05';
  const [logoFailed, setLogoFailed] = React.useState(false);

  const logo = (props.logoUrl && !logoFailed) ? (
    <img
      className="logo-top-white" 
      src={props.logoUrl} 
      style={{objectFit: 'contain', maxHeight: 80}}
      onError={() => setLogoFailed(true)}
    />
  ) : (
    <div className="logo-top-white" style={{color: 'white', fontSize: 32, fontWeight: 800}}>{props.brandName}</div>
  );
  return (
    <div className="slide" style={{'--accent': accent} as any}>
      <style>{css}</style>
      {logo}
      {children}
    </div>
  );
};

// 1. Hook
export const LayoutSlide01: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-bottom" style={{background: 'linear-gradient(to bottom, rgba(7,17,31,0.10) 0%, rgba(7,17,31,0.28) 32%, rgba(7,17,31,0.75) 58%, rgba(7,17,31,0.94) 75%, rgba(7,17,31,0.98) 100%)'}} />
    <div className="vignette" />
    <div className="cnt" style={{position:'absolute', bottom:190, left:64, right:64, zIndex:10}}>
      {props.badges && props.badges.length > 0 && <p className="eyebrow" style={{textShadow: '0 0 24px rgba(0,255,128,0.5), 0 0 52px rgba(0,255,128,0.16)', marginBottom: 10}}>{props.badges[0].text}</p>}
      <h1 className="h1" style={{fontSize: 84, letterSpacing: '-0.032em', lineHeight: 1.06}}>{props.headline}</h1>
      <div className="rule" />
      <p className="body" style={{color: 'white', fontSize: 38, fontWeight: 300}}>{props.subheadline}</p>
      {props.cta && <p className="swipe" style={{color: 'white', textAlign: 'center', fontSize: 26, marginTop: 32}}>{props.cta}</p>}
    </div>
  </SlideBase>
);

// 2. Notification
export const LayoutSlide02: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-bottom" style={{background: 'linear-gradient(to bottom, rgba(7,17,31,0.04) 0%, rgba(7,17,31,0.16) 28%, rgba(7,17,31,0.60) 52%, rgba(7,17,31,0.88) 70%, rgba(7,17,31,0.97) 100%)'}} />
    <div className="vignette" />
    <div className="top" style={{position:'absolute', top:120, left:64, right:64, zIndex:10}} />
    <div className="notif glass" style={{position:'absolute', bottom:180, left:52, right:52, zIndex:10, padding: '26px 28px 24px', borderRadius: 18}}>
      <div style={{display:'flex', alignItems:'center', gap:14, marginBottom:22}}>
        <div style={{width:44, height:44, background:'rgba(0,255,128,0.14)', border:'1px solid var(--accent)', borderRadius:8, display:'flex', alignItems:'center', justifyContent:'center', color:'var(--accent)', fontWeight:800}}>{props.brandName?.[0] || 'V'}</div>
        <span style={{fontSize:19, letterSpacing:'0.12em', color:'white', fontWeight:700, textTransform:'uppercase'}}>{props.brandName}</span>
        <span style={{fontSize:19, color:'rgba(255,255,255,0.7)', marginLeft:'auto'}}>{props.badges?.[0]?.text || 'agora'}</span>
      </div>
      <div style={{fontFamily:'Georgia,serif', fontSize:66, letterSpacing:'-0.01em', color:'white', fontWeight:700, lineHeight:1, marginBottom:10}}>{props.headline}</div>
      <div style={{fontFamily:'Georgia,serif', fontSize:110, color:'var(--accent)', fontWeight:700, textShadow:'0 0 60px rgba(0,255,128,0.55)', lineHeight:0.92, marginBottom:20}}>{props.highlights?.[0]?.stat || '-23%'}</div>
      <div style={{display:'flex', alignItems:'center', justifyContent:'space-between', borderTop:'1px solid rgba(0,255,128,0.20)', paddingTop:20}}>
        <span style={{fontSize:26, color:'white'}}>{props.subheadline}</span>
        <span style={{fontSize:26, color:'var(--accent)', fontWeight:700}}>{props.cta || 'Garantir agora →'}</span>
      </div>
    </div>
  </SlideBase>
);

// 3. Big Number
export const LayoutSlide03: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-bottom" style={{background: 'linear-gradient(to bottom, rgba(7,17,31,0.04) 0%, rgba(7,17,31,0.16) 28%, rgba(7,17,31,0.60) 52%, rgba(7,17,31,0.88) 70%, rgba(7,17,31,0.97) 100%)'}} />
    <div className="vignette" />
    <div className="cnt" style={{position:'absolute', bottom:258, left:64, right:64, zIndex:10}}>
      <div style={{fontFamily:'Georgia,serif', fontSize:200, lineHeight:0.88, letterSpacing:'-0.04em', color:'var(--accent)', fontWeight:700, textShadow:'0 0 100px rgba(0,255,128,0.55)'}}>{props.highlights?.[0]?.stat}</div>
      <h2 className="h2" style={{marginTop: 8}}>{props.headline}</h2>
      <div className="rule" />
      <p className="body" style={{color:'white', fontSize:38, fontWeight:300}}>{props.subheadline}</p>
    </div>
  </SlideBase>
);

// 4. Chart (Simulação de preço)
export const LayoutSlide04: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-dual" />
    <div className="top" style={{position:'absolute', top:220, left:64, right:64, zIndex:10}}>
      {props.badges?.[0] && <p className="eyebrow">◆  {props.badges[0].text}</p>}
      <h1 className="h1">{props.headline}</h1>
      <p className="body" style={{marginTop:24, fontSize:32, color:'rgba(255,255,255,0.85)'}}>{props.subheadline}</p>
    </div>
    <div className="chart-box glass" style={{position:'absolute', bottom:100, left:64, right:64, padding:'44px 44px 34px', borderRadius:20}}>
      <p className="eyebrow" style={{marginBottom:14, fontSize:19}}>{props.highlights?.[0]?.label || 'SIMULAÇÃO'}</p>
      <div style={{width:'100%', height:160, position:'relative'}}>
        <svg viewBox="0 0 952 160" preserveAspectRatio="none">
          <line x1="0" y1="40" x2="952" y2="40" stroke="rgba(255,255,255,0.07)" strokeWidth="1"/>
          <line x1="0" y1="80" x2="952" y2="80" stroke="rgba(255,255,255,0.07)" strokeWidth="1"/>
          <line x1="0" y1="120" x2="952" y2="120" stroke="rgba(255,255,255,0.07)" strokeWidth="1"/>
          <polyline points="0,50 140,60 280,44 380,80 460,118 600,156 780,148 952,110" fill="none" stroke="rgba(180,180,200,0.55)" strokeWidth="2.5" strokeLinejoin="round"/>
          <line x1="460" y1="0" x2="460" y2="160" stroke="var(--accent)" strokeWidth="2" strokeDasharray="6 4"/>
          <circle cx="460" cy="118" r="8" fill="var(--accent)"/>
          <text x="472" y="22" fontSize="18" fill="var(--accent)" fontFamily="Segoe UI,sans-serif">sua compra</text>
          <text x="600" y="128" fontSize="18" fill="var(--accent)" fontFamily="Segoe UI,sans-serif">crédito → você</text>
        </svg>
      </div>
      <div style={{display:'flex', justifyContent:'space-between', marginTop:16, fontSize:21, color:'rgba(255,255,255,0.6)', fontWeight:600}}>
        <span>Antes</span>
        <span>Depois</span>
      </div>
    </div>
  </SlideBase>
);

// 5. Bullet List
export const LayoutSlide05: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-bottom" />
    <div className="vignette" />
    <div className="cnt" style={{position:'absolute', bottom:160, left:64, right:64, zIndex:10}}>
      {props.badges?.[0] && <p className="eyebrow">◆  {props.badges[0].text}</p>}
      <h1 className="h1">{props.headline}</h1>
      <div className="rule" />
      <div style={{display:'flex', flexDirection:'column', gap:28}}>
        {props.highlights?.map((h, i) => (
          <div key={i} style={{display:'flex', alignItems:'center'}}>
            <div style={{width:14, height:14, borderRadius:'50%', background:'var(--accent)', marginRight:28, boxShadow:'0 0 12px var(--accent)'}}></div>
            <span style={{fontSize:38, color:'white', fontWeight:500, letterSpacing:'0.01em'}}>{h.stat || h.label}</span>
          </div>
        ))}
      </div>
    </div>
  </SlideBase>
);

// 6. Table (Before/After)
export const LayoutSlide06: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-dual" />
    <div className="cnt" style={{position:'absolute', top:220, left:64, right:64, zIndex:10}}>
      {props.badges?.[0] && <p className="eyebrow" style={{color:'white', textShadow:'0 0 10px rgba(0,0,0,0.8)'}}>◆ {props.badges[0].text}</p>}
      <h1 className="h1">{props.headline}</h1>
      
      <div className="glass" style={{marginTop: 64, borderRadius: 20, overflow: 'hidden'}}>
        <div style={{display:'grid', gridTemplateColumns:'1fr 150px 150px', padding:'24px 32px', background:'rgba(255,255,255,0.06)', borderBottom:'1px solid rgba(255,255,255,0.1)'}}>
          <div></div>
          <div style={{fontSize:20, color:'rgba(255,255,255,0.5)', fontWeight:700, textTransform:'uppercase', textAlign:'center'}}>Comum</div>
          <div style={{fontSize:20, color:'var(--accent)', fontWeight:700, textTransform:'uppercase', textAlign:'center'}}>{props.brandName}</div>
        </div>
        {props.highlights?.map((h, i) => (
          <div key={i} style={{display:'grid', gridTemplateColumns:'1fr 150px 150px', padding:'32px', borderBottom: i === (props.highlights!.length - 1) ? 'none' : '1px solid rgba(255,255,255,0.06)', alignItems:'center'}}>
            <div style={{fontSize:28, color:'white', fontWeight:500, paddingRight:20}}>{h.label}</div>
            <div style={{fontSize:26, color:'rgba(255,255,255,0.5)', textAlign:'center'}}>{h.stat.split('|')[0] || 'X'}</div>
            <div style={{fontSize:26, color:'var(--accent)', fontWeight:700, textAlign:'center'}}>{h.stat.split('|')[1] || '✓'}</div>
          </div>
        ))}
      </div>
    </div>
  </SlideBase>
);

// 7. Info/Badges
export const LayoutSlide07: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-dual" />
    <div className="cnt" style={{position:'absolute', top:260, left:64, right:64, zIndex:10}}>
      <div style={{display:'flex', gap:16, flexWrap:'wrap', marginBottom:36}}>
        {props.badges?.map((b, i) => <div key={i} className="badge">{b.text}</div>)}
      </div>
      <h1 className="h1">{props.headline}</h1>
      <p className="body" style={{marginTop:30, fontSize:38, color:'white', fontWeight:300}}>{props.subheadline}</p>
    </div>
  </SlideBase>
);

// 8. Stats
export const LayoutSlide08: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div className="ov-bottom" style={{background: 'linear-gradient(to bottom, rgba(7,17,31,0.04) 0%, rgba(7,17,31,0.2) 20%, rgba(7,17,31,0.85) 60%, rgba(7,17,31,0.98) 100%)'}} />
    <div className="cnt" style={{position:'absolute', bottom:100, left:64, right:64, zIndex:10}}>
      {props.highlights?.[0] && (
        <div style={{marginBottom: 60}}>
          <div style={{fontFamily:'Georgia,serif', fontSize:180, color:'var(--accent)', fontWeight:700, lineHeight:0.9, letterSpacing:'-0.03em', textShadow:'0 0 100px rgba(0,255,128,0.4)'}}>{props.highlights[0].stat}</div>
          <div style={{fontSize:38, color:'white', fontWeight:500, marginTop:16}}>{props.highlights[0].label}</div>
        </div>
      )}
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:32}}>
        {props.highlights?.slice(1).map((h, i) => (
          <div key={i} className="glass" style={{padding:'36px 32px', borderRadius:20}}>
            <div style={{fontFamily:'Georgia,serif', fontSize:72, color:'white', fontWeight:700, lineHeight:1, marginBottom:12}}>{h.stat}</div>
            <div style={{fontSize:24, color:'var(--accent)', fontWeight:600, letterSpacing:'0.04em'}}>{h.label}</div>
          </div>
        ))}
      </div>
    </div>
  </SlideBase>
);

// 9. CTA
export const LayoutSlide09: React.FC<{props: SocialPostProps}> = ({props}) => (
  <SlideBase props={props}>
    <div style={{position:'absolute', inset:0, background:'rgba(7,17,31,0.6)', backdropFilter:'blur(20px)'}} />
    <div style={{position:'absolute', top:360, left:64, right:64, textAlign:'center', zIndex:10}}>
      <div style={{fontFamily:'Georgia,serif', fontSize:130, color:'white', fontWeight:700, lineHeight:0.95, letterSpacing:'-0.02em', textTransform:'uppercase'}}>{props.headline}</div>
      <div style={{fontSize:38, color:'var(--accent)', fontWeight:500, marginTop:32, letterSpacing:'0.02em'}}>{props.subheadline}</div>
      
      <div style={{marginTop:100}}>
        <div style={{background:'var(--accent)', color:'rgba(7,17,31,1)', fontSize:36, fontWeight:800, padding:'32px 64px', borderRadius:80, display:'inline-block', textTransform:'uppercase', letterSpacing:'0.04em', boxShadow:'0 0 40px rgba(0,255,128,0.4), inset 0 -4px 0 rgba(0,0,0,0.1)'}}>
          {props.cta || 'Experimente Grátis'}
        </div>
      </div>
    </div>
  </SlideBase>
);

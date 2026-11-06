import Link from "next/link"
import { BgCanvas } from "@/app/components/bg-canvas"
import { Wordmark } from "@/app/components/wordmark"
import { Telemetry } from "@/app/components/telemetry"
import { ScrollProgress } from "@/app/components/scroll-progress"
import { HeroSection } from "@/app/components/hero-section"
import { ChapterShape } from "@/app/components/chapter-shape"
import { ChapterSpace } from "@/app/components/chapter-space"
import { ExplorerSection } from "@/app/components/explorer-section"
import { StrategySection } from "@/app/components/strategy-section"
import { FilmGrain } from "@/app/components/film-grain"
import { SplitText } from "@/app/components/split-text"

export default function LandingPage() {
  return (
    <>
      <BgCanvas />
      <FilmGrain />
      <Wordmark />
      <Telemetry />
      <ScrollProgress />

      <HeroSection />
      <ChapterShape />
      <ChapterSpace />
      <ExplorerSection />
      <StrategySection />

      <section className="query-section">
        <div className="query-inner reveal-group">
          <div className="query-label reveal-child">R1VER | Begin</div>
          <SplitText as="h2" className="query-title" delay={0.15}>
            Ask the <span className="em">river.</span>
          </SplitText>
          <p className="section-sub reveal-child" style={{ textAlign: "center", margin: "0 auto 48px" }}>
            Full Explorer and Strategy viewer at production scale.
          </p>
          <div className="query-cta-group reveal-child">
            <Link href="/explorer" className="query-cta">Explore Hands</Link>
            <Link href="/strategy" className="query-cta">Query Strategy</Link>
          </div>
        </div>
      </section>

      <footer className="site-footer reveal">
        <div className="footer-row">
          <span className="footer-wordmark">R<span className="one">1</span>VER</span>
          <span>Thomas Ou · MMXXVI</span>
          <span className="footer-links">
            <a href="https://thomasou.com" target="_blank" rel="noopener noreferrer">thomasou.com</a>
            <a href="https://github.com/Smokeybear10" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="https://www.linkedin.com/in/thomasou0/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          </span>
        </div>
      </footer>
    </>
  )
}

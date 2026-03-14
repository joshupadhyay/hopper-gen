const sampleWorks = [
  {
    title: "Sunday Matinee",
    attribution: "anonymous",
    date: "March 14, 2026",
    prompt: "A corner cinema at dusk with two figures waiting under amber light",
  },
  {
    title: "Rooms for Let",
    attribution: "night clerk",
    date: "March 11, 2026",
    prompt: "A roadside motel office at blue hour with a single lamp and empty parking lot",
  },
  {
    title: "After the Rain",
    attribution: "window seat",
    date: "March 8, 2026",
    prompt: "A diner window streaked with rain and a cashier looking out onto the avenue",
  },
];

export default function PaperPage() {
  return (
    <main className="paper-shell">
      <header className="paper-header">
        <p className="eyebrow">Paper Explorations</p>
        <h1>Three frontend directions for Hopper Studio.</h1>
        <p className="lede">
          These mock frames are meant to help pick a visual direction before we
          push more polish into the production routes.
        </p>
      </header>

      <section className="paper-grid">
        <article className="paper-card paper-card-wall">
          <div className="paper-card-head">
            <p className="eyebrow">Option A</p>
            <h2>Moving Background Wall</h2>
          </div>

          <div className="paper-frame paper-frame-home">
            <div className="wall-atmosphere">
              {sampleWorks.concat(sampleWorks).map((work, index) => (
                <div className="wall-piece" key={`${work.title}-${index}`}>
                  <div className="wall-piece-image" />
                  <span>{work.title}</span>
                </div>
              ))}
            </div>

            <div className="paper-overlay-panel">
              <p className="eyebrow">Homepage / Generator</p>
              <h3>Quiet paintings from a single line of text.</h3>
              <div className="paper-textarea">
                A diner at dawn with rain on the windows and one cashier waiting
              </div>
              <div className="paper-actions">
                <button className="primary-button" type="button">
                  Generate
                </button>
                <span>Atmospheric, immersive, and gallery-like.</span>
              </div>
            </div>
          </div>
        </article>

        <article className="paper-card paper-card-carousel">
          <div className="paper-card-head">
            <p className="eyebrow">Option B</p>
            <h2>Featured Carousel</h2>
          </div>

          <div className="paper-frame paper-frame-result">
            <div className="paper-mock-stage">
              <div className="paper-mock-art-frame">
                <div className="paper-mock-art" />
              </div>
              <div className="paper-plaque">
                <strong>{sampleWorks[0].title}</strong>
                <span>{sampleWorks[0].date}</span>
              </div>
            </div>

            <aside className="paper-sidebar">
              <p className="eyebrow">Result / Publish</p>
              <h3>Review the image, then publish if it belongs on the wall.</h3>
              <div className="paper-field">
                <label>Title</label>
                <div className="paper-input">Sunday Matinee</div>
              </div>
              <div className="paper-field">
                <label>Attribution</label>
                <div className="paper-input">anonymous</div>
              </div>
              <div className="paper-actions stack">
                <button className="primary-button" type="button">
                  Publish your artwork
                </button>
                <a className="secondary-link" href="#paper-detail">
                  Generate again
                </a>
              </div>
            </aside>
          </div>
        </article>

        <article className="paper-card paper-card-reading" id="paper-detail">
          <div className="paper-card-head">
            <p className="eyebrow">Option C</p>
            <h2>Reading Room</h2>
          </div>

          <div className="paper-frame paper-frame-detail">
            <div className="reading-room-meta">
              <p className="eyebrow">Artwork Detail / Permalink</p>
              <h3>{sampleWorks[1].title}</h3>
              <dl className="reading-room-list">
                <div>
                  <dt>Attribution</dt>
                  <dd>{sampleWorks[1].attribution}</dd>
                </div>
                <div>
                  <dt>Date</dt>
                  <dd>{sampleWorks[1].date}</dd>
                </div>
                <div>
                  <dt>Prompt</dt>
                  <dd>{sampleWorks[1].prompt}</dd>
                </div>
                <div>
                  <dt>Permalink</dt>
                  <dd>hopper.studio/artwork/rooms-for-let</dd>
                </div>
              </dl>
            </div>

            <div className="reading-room-art">
              <div className="catalog-frame">
                <div className="catalog-art" />
              </div>
              <div className="catalog-caption">
                Exhibition-style detail page with more formal typography and
                metadata rhythm.
              </div>
            </div>
          </div>
        </article>
      </section>
    </main>
  );
}
